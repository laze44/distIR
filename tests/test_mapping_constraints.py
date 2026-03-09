# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tests for GEMM tensor mapping constraints."""

import ast
import json
import textwrap
from typing import Optional, Tuple

import pytest
from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.search.mapping_constraints import (
    load_operator_tensor_mapping_constraints,
    load_tensor_mapping_constraints,
    program_satisfies_tensor_mapping_constraints,
)
from mercury.search.search import search
from utils.gemm_dsl import gemm_manage_reduction


def _build_gemm_program(m: int = 128, n: int = 128, k: int = 128):
    source = gemm_manage_reduction.format(M_LEN=m, N_LEN=n, K_LEN=k)
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


def test_load_tensor_mapping_constraints_default_template():
    constraints = load_tensor_mapping_constraints("config/gemm_tensor_mapping.json")
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
                            "mapping": ["R", {"shard": ["intra_node"]}],
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
                                {"shard": ["intra_node"]},
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
    assert summary["gate"]["B"] == "fixed [S(intra_node), R]"
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
        "config/gemm_tensor_mapping.json",
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
                        {"shard": ["intra_node"]},
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


def test_search_multiple_fixed_matrices_apply_intersection(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "B": {
                    "mode": "fixed",
                    "mapping": [
                        "R",
                        {"shard": ["intra_node"]},
                    ],
                },
                "C": {
                    "mode": "fixed",
                    "mapping": [
                        {"shard": ["inter_node"]},
                        {"shard": ["intra_node"]},
                    ],
                },
            },
        },
    )

    single_programs = _search_gemm_programs(
        (2, 2),
        _write_config(
            tmp_path,
            {
                "version": 1,
                "matrices": {
                    "B": {
                        "mode": "fixed",
                        "mapping": [
                            "R",
                            {"shard": ["intra_node"]},
                        ],
                    },
                },
            },
            "single_b_mapping.json",
        ),
    )
    constrained_programs = _search_gemm_programs((2, 2), config_path)
    constraints = load_tensor_mapping_constraints(config_path)

    assert 0 < len(constrained_programs) <= len(single_programs)
    assert all(
        program_satisfies_tensor_mapping_constraints(program, constraints)
        for program in constrained_programs
    )


def test_search_mixed_constraint_matches_flattened_candidates(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "version": 1,
            "matrices": {
                "C": {
                    "mode": "fixed",
                    "mapping": [
                        "R",
                        {"shard": ["mixed"]},
                    ],
                },
            },
        },
    )

    constrained_programs = _search_gemm_programs((2, 2), config_path)
    constraints = load_tensor_mapping_constraints(config_path)

    assert len(constrained_programs) > 0
    assert all(program.mesh.shape == (4,) for program in constrained_programs)
    assert all(
        program_satisfies_tensor_mapping_constraints(program, constraints)
        for program in constrained_programs
    )
