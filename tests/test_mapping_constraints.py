# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tests for GEMM tensor mapping constraints."""

import ast
import json
import textwrap
from typing import Optional, Tuple

import pytest
from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.elements import Axis, Buffer
from mercury.ir.nodes import AxisDef, BufferMatch, Program
from mercury.search.mapping_constraints import (
    ExactTensorLayoutConstraints,
    LogicalTensorLayoutConstraints,
    derive_logical_local_shape,
    exact_layout_signature_from_buffer,
    logical_layout_signature_from_buffer,
    load_operator_tensor_mapping_constraints,
    load_tensor_mapping_constraints,
    logical_layout_signature_equal,
    program_satisfies_logical_layout_constraints,
    program_satisfies_exact_layout_constraints,
    program_satisfies_tensor_mapping_constraints,
)
from mercury.search.search import search
from utils.gemm_dsl import format_gemm_template


def _build_gemm_program(m: int = 128, n: int = 128, k: int = 128):
    source = format_gemm_template(m, n, k)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return builder.visit(node)
    raise ValueError("Could not find function definition in GEMM template")


def _write_config(tmp_path, payload, filename: str = "mapping.json") -> str:
    path = tmp_path / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _search_gemm_programs(
    mesh_shape: Tuple[int, ...],
    config_path: Optional[str] = None,
):
    program = _build_gemm_program()
    world_size = 1
    for dim in mesh_shape:
        world_size *= dim
    mesh = DeviceMesh(list(range(world_size)), mesh_shape)

    constraints = None
    if config_path is not None:
        constraints = load_tensor_mapping_constraints(config_path)

    programs = list(
        search(
            program,
            mesh,
            ["I", "J", "K"],
            tensor_mapping_constraints=constraints,
        )
    )
    programs.sort(key=generate_pytorch_code)
    return programs


def _build_mock_program(
    name: str,
    mesh_shape: Tuple[int, ...],
    a_spec,
    b_spec,
    c_spec,
    k_dim: int = 8,
    n_dim: int = 16,
) -> Program:
    world_size = 1
    for dim_size in mesh_shape:
        world_size *= int(dim_size)
    mesh = DeviceMesh(list(range(world_size)), mesh_shape)

    axis_i = Axis("I", 16, 16)
    axis_k = Axis("K", int(k_dim), int(k_dim))
    axis_j = Axis("J", int(n_dim), int(n_dim))

    buffer_a = Buffer(
        tensor="a",
        shape=[16, int(k_dim)],
        bound_axes=[[axis_i], [axis_k]],
        axes_factor=[[1], [1]],
        shard_spec=ShardingSpec(mesh, a_spec),
        read=True,
        write=False,
    )
    buffer_b = Buffer(
        tensor="b",
        shape=[int(k_dim), int(n_dim)],
        bound_axes=[[axis_k], [axis_j]],
        axes_factor=[[1], [1]],
        shard_spec=ShardingSpec(mesh, b_spec),
        read=True,
        write=False,
    )
    buffer_c = Buffer(
        tensor="c",
        shape=[16, int(n_dim)],
        bound_axes=[[axis_i], [axis_j]],
        axes_factor=[[1], [1]],
        shard_spec=ShardingSpec(mesh, c_spec),
        read=True,
        write=True,
    )

    return Program(
        name=name,
        inputs=[],
        defaults=[],
        outputs=[],
        body=[
            AxisDef(axis_i),
            AxisDef(axis_k),
            AxisDef(axis_j),
            BufferMatch(buffer=buffer_a, tensor_name="a"),
            BufferMatch(buffer=buffer_b, tensor_name="b"),
            BufferMatch(buffer=buffer_c, tensor_name="c"),
        ],
        mesh=mesh,
    )


def test_load_tensor_mapping_constraints_default_template():
    constraints = load_tensor_mapping_constraints("config/gemm_tensor_mapping_flexible.json")
    assert constraints.summary_by_matrix() == {
        "A": "flexible",
        "B": "flexible",
        "C": "flexible",
    }


def test_load_operator_tensor_mapping_constraints_default_template():
    constraints = load_operator_tensor_mapping_constraints(
        "config/ffn_tensor_mapping.json",
        ["gate", "up", "down"],
    )
    assert constraints.summary_by_operator() == {
        "gate": {"A": "flexible", "B": "flexible", "C": "flexible"},
        "up": {"A": "flexible", "B": "flexible", "C": "flexible"},
        "down": {"A": "flexible", "B": "flexible", "C": "flexible"},
    }


def test_load_operator_tensor_mapping_constraints_reject_fixed_activation(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "operators": {
                "gate": {
                    "matrices": {
                        "A": {
                            "mode": "fixed",
                            "mapping": ["R", {"shard": ["device"]}],
                        }
                    }
                }
            },
        },
    )

    with pytest.raises(ValueError, match="matrix A must be flexible"):
        load_operator_tensor_mapping_constraints(config_path, ["gate", "up", "down"])


def test_load_operator_tensor_mapping_constraints_fill_defaults(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "operators": {
                "gate": {
                    "matrices": {
                        "B": {
                            "mode": "fixed",
                            "mapping": [
                                {"shard": ["device"]},
                                "R",
                            ],
                        }
                    }
                }
            },
        },
    )
    constraints = load_operator_tensor_mapping_constraints(
        config_path,
        ["gate", "up", "down"],
    )
    summary = constraints.summary_by_operator()

    assert summary["gate"]["A"] == "flexible"
    assert summary["gate"]["B"] == "fixed [S(device), R]"
    assert summary["gate"]["C"] == "flexible"

    assert summary["up"] == {"A": "flexible", "B": "flexible", "C": "flexible"}
    assert summary["down"] == {"A": "flexible", "B": "flexible", "C": "flexible"}


def test_load_tensor_mapping_constraints_missing_mapping(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "A": {"mode": "fixed"},
            },
        },
    )

    with pytest.raises(ValueError, match="must define a mapping"):
        load_tensor_mapping_constraints(config_path)


def test_load_tensor_mapping_constraints_flexible_with_mapping(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "B": {"mode": "flexible", "mapping": ["R", "R"]},
            },
        },
    )

    with pytest.raises(ValueError, match="cannot define a mapping"):
        load_tensor_mapping_constraints(config_path)


def test_load_tensor_mapping_constraints_unknown_matrix_name(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "D": {"mode": "flexible"},
            },
        },
    )

    with pytest.raises(ValueError, match="Unsupported matrices"):
        load_tensor_mapping_constraints(config_path)


def test_load_tensor_mapping_constraints_unknown_topology_token(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "A": {
                    "mode": "fixed",
                    "mapping": [
                        "R",
                        {"shard": ["socket"]},
                    ],
                },
            },
        },
    )

    with pytest.raises(ValueError, match="Unsupported topology token"):
        load_tensor_mapping_constraints(config_path)


def test_load_tensor_mapping_constraints_mapping_rank_mismatch(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "C": {
                    "mode": "fixed",
                    "mapping": ["R"],
                },
            },
        },
    )

    with pytest.raises(ValueError, match="length 2"):
        load_tensor_mapping_constraints(config_path)


def test_search_without_constraints_matches_flexible_template():
    baseline_programs = _search_gemm_programs((2, 2))
    flexible_programs = _search_gemm_programs(
        (2, 2),
        "config/gemm_tensor_mapping_flexible.json",
    )

    assert len(baseline_programs) == len(flexible_programs)
    assert [
        generate_pytorch_code(program) for program in baseline_programs
    ] == [
        generate_pytorch_code(program) for program in flexible_programs
    ]


def test_search_single_fixed_matrix_prunes_candidates(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "B": {
                    "mode": "fixed",
                    "mapping": [
                        "R",
                        {"shard": ["device"]},
                    ],
                },
            },
        },
    )

    baseline_programs = _search_gemm_programs((2, 2))
    constrained_programs = _search_gemm_programs((2, 2), config_path)
    constraints = load_tensor_mapping_constraints(config_path)

    assert 0 < len(constrained_programs) < len(baseline_programs)
    assert all(
        program_satisfies_tensor_mapping_constraints(program, constraints)
        for program in constrained_programs
    )




def test_exact_layout_constraints_match_same_layout():
    mesh_shape = (2, 2)
    a_spec = [(ShardType.SHARD, [0]), ShardType.REPLICATE]
    b_spec = [ShardType.REPLICATE, (ShardType.SHARD, [1])]
    c_spec = [ShardType.REPLICATE, (ShardType.SHARD, [0])]
    program = _build_mock_program("exact_match", mesh_shape, a_spec, b_spec, c_spec)

    constraints = ExactTensorLayoutConstraints(
        matrices={
            "A": exact_layout_signature_from_buffer(program.body[3].buffer),
            "B": exact_layout_signature_from_buffer(program.body[4].buffer),
            "C": exact_layout_signature_from_buffer(program.body[5].buffer),
        }
    )

    assert program_satisfies_exact_layout_constraints(program, constraints)


def test_exact_layout_constraints_reject_mismatched_mesh_shape():
    program = _build_mock_program(
        "mesh_a",
        (2, 2),
        [(ShardType.SHARD, [0]), ShardType.REPLICATE],
        [ShardType.REPLICATE, ShardType.REPLICATE],
        [ShardType.REPLICATE, (ShardType.SHARD, [1])],
    )
    other = _build_mock_program(
        "mesh_b",
        (4,),
        [(ShardType.SHARD, [0]), ShardType.REPLICATE],
        [ShardType.REPLICATE, ShardType.REPLICATE],
        [ShardType.REPLICATE, (ShardType.SHARD, [0])],
    )

    constraints = ExactTensorLayoutConstraints(
        matrices={
            "A": exact_layout_signature_from_buffer(other.body[3].buffer),
            "B": exact_layout_signature_from_buffer(other.body[4].buffer),
            "C": exact_layout_signature_from_buffer(other.body[5].buffer),
        }
    )

    assert not program_satisfies_exact_layout_constraints(program, constraints)


def test_exact_layout_constraints_reject_mismatched_shard_dims():
    base = _build_mock_program(
        "base",
        (2, 2),
        [(ShardType.SHARD, [0]), ShardType.REPLICATE],
        [ShardType.REPLICATE, (ShardType.SHARD, [1])],
        [ShardType.REPLICATE, (ShardType.SHARD, [0])],
    )
    other = _build_mock_program(
        "other",
        (2, 2),
        [(ShardType.SHARD, [1]), ShardType.REPLICATE],
        [ShardType.REPLICATE, (ShardType.SHARD, [0, 1])],
        [ShardType.REPLICATE, (ShardType.SHARD, [1])],
    )

    constraints = ExactTensorLayoutConstraints(
        matrices={
            "A": exact_layout_signature_from_buffer(other.body[3].buffer),
            "B": exact_layout_signature_from_buffer(other.body[4].buffer),
            "C": exact_layout_signature_from_buffer(other.body[5].buffer),
        }
    )

    assert not program_satisfies_exact_layout_constraints(base, constraints)


def test_logical_layout_signature_uses_global_shape_not_execution_local_shape():
    mesh_shape = (2, 2)
    program = _build_mock_program(
        "logical_sig",
        mesh_shape,
        [(ShardType.SHARD, [0]), ShardType.REPLICATE],
        [ShardType.REPLICATE, (ShardType.SHARD, [1])],
        [(ShardType.SHARD, [0]), (ShardType.SHARD, [1])],
    )
    buffer_a = program.body[3].buffer
    signature = logical_layout_signature_from_buffer(buffer_a)

    assert signature.global_shape == (16, 8)
    assert signature.mesh_shape == mesh_shape
    assert signature.shard_specs == (("S", (0,)), ("R", ()))


def test_derive_logical_local_shape_roundtrip():
    signature = logical_layout_signature_from_buffer(
        _build_mock_program(
            "logical_roundtrip",
            (2, 2),
            [(ShardType.SHARD, [0]), ShardType.REPLICATE],
            [ShardType.REPLICATE, ShardType.REPLICATE],
            [ShardType.REPLICATE, ShardType.REPLICATE],
        ).body[3].buffer
    )
    local_shape = derive_logical_local_shape(
        global_shape=signature.global_shape,
        mesh_shape=signature.mesh_shape,
        shard_specs=signature.shard_specs,
    )
    assert local_shape == (8, 8)


def test_program_satisfies_logical_layout_constraints():
    base = _build_mock_program(
        "logical_base",
        (2, 2),
        [(ShardType.SHARD, [0]), ShardType.REPLICATE],
        [ShardType.REPLICATE, (ShardType.SHARD, [1])],
        [(ShardType.SHARD, [0]), (ShardType.SHARD, [1])],
    )
    other = _build_mock_program(
        "logical_other",
        (2, 2),
        [(ShardType.SHARD, [1]), ShardType.REPLICATE],
        [ShardType.REPLICATE, (ShardType.SHARD, [0])],
        [(ShardType.SHARD, [1]), (ShardType.SHARD, [0])],
    )

    constraints = LogicalTensorLayoutConstraints(
        matrices={
            "A": logical_layout_signature_from_buffer(base.body[3].buffer),
            "B": logical_layout_signature_from_buffer(base.body[4].buffer),
            "C": logical_layout_signature_from_buffer(base.body[5].buffer),
        }
    )

    assert program_satisfies_logical_layout_constraints(base, constraints)
    assert not program_satisfies_logical_layout_constraints(other, constraints)
    assert logical_layout_signature_equal(
        logical_layout_signature_from_buffer(base.body[3].buffer),
        constraints.matrices["A"],
    )


def test_parse_shard_factor():
    """Verify shard_factor field parses correctly."""
    constraints = load_tensor_mapping_constraints(
        "config/gemm_tensor_mapping_fixed_b_device.json"
    )
    summary = constraints.summary_by_matrix()
    assert summary["A"] == "flexible"
    assert summary["B"] == "fixed [S(device, factor=16), R]"
    assert summary["C"] == "flexible"


def test_parse_shard_factor_rejects_non_integer(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "B": {
                    "mode": "fixed",
                    "mapping": [
                        {"shard": ["device"], "shard_factor": "four"},
                        "R",
                    ],
                },
            },
        },
    )
    with pytest.raises(ValueError, match="shard_factor must be an integer >= 2"):
        load_tensor_mapping_constraints(config_path)


def test_parse_shard_factor_rejects_value_one(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "B": {
                    "mode": "fixed",
                    "mapping": [
                        {"shard": ["device"], "shard_factor": 1},
                        "R",
                    ],
                },
            },
        },
    )
    with pytest.raises(ValueError, match="shard_factor must be an integer >= 2"):
        load_tensor_mapping_constraints(config_path)


def test_parse_shard_factor_rejects_without_shard(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "B": {
                    "mode": "fixed",
                    "mapping": [
                        {"shard_factor": 4},
                        "R",
                    ],
                },
            },
        },
    )
    with pytest.raises(ValueError, match="must contain the 'shard' field"):
        load_tensor_mapping_constraints(config_path)


def test_shard_factor_match_accepts_valid_subset():
    """shard_factor match: actual_dims subset of expected, product == factor."""
    mesh_shape = (4, 4)
    b_spec = [(ShardType.SHARD, [0]), (ShardType.SHARD, [1])]
    program = _build_mock_program(
        "sf_accept",
        mesh_shape,
        [ShardType.REPLICATE, ShardType.REPLICATE],
        b_spec,
        [ShardType.REPLICATE, ShardType.REPLICATE],
        k_dim=16,
        n_dim=16,
    )
    program = Program(
        name=program.name,
        inputs=program.inputs,
        defaults=program.defaults,
        outputs=program.outputs,
        body=program.body,
        mesh=program.mesh,
        topology_metadata={"device_dims": [0, 1]},
    )

    constraints = load_tensor_mapping_constraints(
        "config/gemm_tensor_mapping_fixed_b_device.json"
    )
    assert program_satisfies_tensor_mapping_constraints(program, constraints)


def test_shard_factor_match_rejects_wrong_product():
    """shard_factor match: actual mesh product != factor -> reject."""
    mesh_shape = (2, 8)
    b_spec = [(ShardType.SHARD, [0]), (ShardType.SHARD, [1])]
    program = _build_mock_program(
        "sf_reject",
        mesh_shape,
        [ShardType.REPLICATE, ShardType.REPLICATE],
        b_spec,
        [ShardType.REPLICATE, ShardType.REPLICATE],
        k_dim=16,
        n_dim=16,
    )
    program = Program(
        name=program.name,
        inputs=program.inputs,
        defaults=program.defaults,
        outputs=program.outputs,
        body=program.body,
        mesh=program.mesh,
        topology_metadata={"device_dims": [0, 1]},
    )

    constraints = load_tensor_mapping_constraints(
        "config/gemm_tensor_mapping_fixed_b_device.json"
    )
    assert not program_satisfies_tensor_mapping_constraints(program, constraints)


def test_shard_factor_match_logical_factors_consistent():
    """Verify shard_factor constraint matching is consistent with logical factor computation."""
    from mercury.search.topology_policy import compute_program_logical_shard_factors

    mesh_shape = (4, 4)
    b_spec = [(ShardType.SHARD, [0]), (ShardType.SHARD, [1])]
    program = _build_mock_program(
        "sf_logical_consistent",
        mesh_shape,
        [ShardType.REPLICATE, ShardType.REPLICATE],
        b_spec,
        [ShardType.REPLICATE, ShardType.REPLICATE],
        k_dim=16,
        n_dim=16,
    )
    program = Program(
        name=program.name,
        inputs=program.inputs,
        defaults=program.defaults,
        outputs=program.outputs,
        body=program.body,
        mesh=program.mesh,
        topology_metadata={"device_dims": [0, 1]},
    )

    # This program satisfies the fixed_b_device constraint (factor=16 total)
    constraints = load_tensor_mapping_constraints(
        "config/gemm_tensor_mapping_fixed_b_device.json"
    )
    assert program_satisfies_tensor_mapping_constraints(program, constraints)

    # Logical factors should show device=(4, 4) for B
    logical_factors = compute_program_logical_shard_factors(program)
    b_factors = logical_factors.get("B")
    assert b_factors is not None
    assert b_factors.domain_factors.get("device") == (4, 4)
    assert b_factors.total_factor("device") == 16


def test_logical_factor_constraint_matching():
    """Test the program_satisfies_logical_factor_constraints() API."""
    from mercury.search.mapping_constraints import (
        program_satisfies_logical_factor_constraints,
    )

    mesh_shape = (4, 2)
    a_spec = [(ShardType.SHARD, [0]), ShardType.REPLICATE]
    b_spec = [ShardType.REPLICATE, (ShardType.SHARD, [1])]
    c_spec = [(ShardType.SHARD, [0]), (ShardType.SHARD, [1])]
    program = _build_mock_program(
        "logical_factor_test",
        mesh_shape,
        a_spec,
        b_spec,
        c_spec,
    )
    program = Program(
        name=program.name,
        inputs=program.inputs,
        defaults=program.defaults,
        outputs=program.outputs,
        body=program.body,
        mesh=program.mesh,
        topology_metadata={"device_dims": [0, 1]},
    )

    # A is sharded on dim 0 via device (factor 4)
    assert program_satisfies_logical_factor_constraints(
        program, {"A": {"device": (4,)}}
    )

    # B is sharded on dim 1 via device (factor 2)
    assert program_satisfies_logical_factor_constraints(
        program, {"B": {"device": (2,)}}
    )

    # C is sharded on both dims via device
    assert program_satisfies_logical_factor_constraints(
        program, {"C": {"device": (4, 2)}}
    )

    # Wrong factor should fail
    assert not program_satisfies_logical_factor_constraints(
        program, {"A": {"device": (2,)}}
    )
