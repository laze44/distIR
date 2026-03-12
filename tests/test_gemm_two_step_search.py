# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tests for GEMM two-step search."""

import json

from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.elements import Axis, Buffer
from mercury.ir.nodes import AxisDef, BufferMatch, Program
from mercury.search.estimate import EstimateResult, load_hardware_config
from mercury.search.gemm_two_step_search import (
    GEMMLayoutPlan,
    enumerate_gemm_step1_layout_plans,
    search_gemm_two_step,
)
from mercury.search.mapping_constraints import (
    TensorMappingConstraints,
    load_tensor_mapping_constraints,
    logical_layout_signature_from_buffer,
)


def _build_mock_program(
    name: str,
    mesh_shape,
    a_spec,
    b_spec,
    c_spec,
    m_dim: int = 16,
    k_dim: int = 8,
    n_dim: int = 16,
) -> Program:
    world_size = 1
    for dim_size in mesh_shape:
        world_size *= int(dim_size)
    mesh = DeviceMesh(list(range(world_size)), tuple(mesh_shape))

    axis_i = Axis("I", int(m_dim), int(m_dim))
    axis_k = Axis("K", int(k_dim), int(k_dim))
    axis_j = Axis("J", int(n_dim), int(n_dim))

    buffer_a = Buffer(
        tensor="a",
        shape=[int(m_dim), int(k_dim)],
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
        shape=[int(m_dim), int(n_dim)],
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


def _matrix_layouts(program: Program):
    return {
        "A": logical_layout_signature_from_buffer(program.body[3].buffer),
        "B": logical_layout_signature_from_buffer(program.body[4].buffer),
        "C": logical_layout_signature_from_buffer(program.body[5].buffer),
    }


def test_enumerate_gemm_step1_layout_plans_respects_fixed_mapping(tmp_path):
    config_path = tmp_path / "gemm_mapping.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "matrices": {
                    "A": {
                        "mode": "fixed",
                        "mapping": [
                            {"shard": ["intra_node"]},
                            "R",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    constraints = load_tensor_mapping_constraints(str(config_path))
    hw_config = load_hardware_config("config/h100.json")
    mesh = DeviceMesh([0, 1], (1, 2))

    ranked_plans = enumerate_gemm_step1_layout_plans(
        problem_shape=(16, 16, 8),
        origin_mesh=mesh,
        hw_config=hw_config,
        tensor_mapping_constraints=constraints,
        layout_top_k=8,
    )

    assert len(ranked_plans) > 0
    assert all(
        plan.boundary_layouts["A"].shard_specs[0] == ("S", (1,))
        and plan.boundary_layouts["A"].shard_specs[1] == ("R", ())
        for plan in ranked_plans
    )


def test_search_gemm_two_step_step2_filters_by_logical_boundary(monkeypatch):
    from mercury.search import gemm_two_step_search as gemm_module

    mesh_shape = (2, 2)
    i0 = [(ShardType.SHARD, [0]), ShardType.REPLICATE]
    i1 = [(ShardType.SHARD, [1]), ShardType.REPLICATE]
    b0 = [ShardType.REPLICATE, (ShardType.SHARD, [1])]
    c0 = [(ShardType.SHARD, [0]), (ShardType.SHARD, [1])]
    c1 = [(ShardType.SHARD, [1]), (ShardType.SHARD, [0])]

    seed_program = _build_mock_program("seed", mesh_shape, i0, b0, c0)
    wrong_fast = _build_mock_program("wrong_fast", mesh_shape, i1, b0, c1)
    good_slow = _build_mock_program("good_slow", mesh_shape, i0, b0, c0)
    good_fast = _build_mock_program("good_fast", mesh_shape, i0, b0, c0)

    selected_layouts = _matrix_layouts(good_fast)
    impossible_layouts = dict(selected_layouts)
    impossible_layouts["C"] = selected_layouts["C"].__class__(
        mesh_shape=selected_layouts["C"].mesh_shape,
        global_shape=selected_layouts["C"].global_shape,
        shard_specs=(("R", ()), ("R", ())),
    )

    ranked_plans = [
        GEMMLayoutPlan(
            problem_shape=(16, 16, 8),
            topology_shape=mesh_shape,
            boundary_layouts=selected_layouts,
            step1_obligations_bytes={},
            step1_cost_terms_ms={},
            step1_total_time_ms=1.0,
        ),
        GEMMLayoutPlan(
            problem_shape=(16, 16, 8),
            topology_shape=mesh_shape,
            boundary_layouts=impossible_layouts,
            step1_obligations_bytes={},
            step1_cost_terms_ms={},
            step1_total_time_ms=2.0,
        ),
    ]

    def fake_enumerate(
        problem_shape,
        origin_mesh,
        hw_config,
        tensor_mapping_constraints,
        layout_top_k,
    ):
        del problem_shape, origin_mesh, hw_config, tensor_mapping_constraints, layout_top_k
        return ranked_plans

    def fake_estimate_program(program, hw_config):
        del hw_config
        costs = {
            "wrong_fast": 0.1,
            "good_slow": 5.0,
            "good_fast": 2.0,
        }
        total = costs[program.name]
        return EstimateResult(total, 0.0, total)

    monkeypatch.setattr(gemm_module, "enumerate_gemm_step1_layout_plans", fake_enumerate)
    monkeypatch.setattr(gemm_module, "estimate_program", fake_estimate_program)

    hw_config = load_hardware_config("config/h100.json")
    origin_mesh = DeviceMesh(list(range(4)), mesh_shape)
    result = search_gemm_two_step(
        input_program=seed_program,
        origin_mesh=origin_mesh,
        split_axis_names=["I", "J", "K"],
        hw_config=hw_config,
        tensor_mapping_constraints=TensorMappingConstraints(matrices={}),
        layout_top_k=2,
        candidate_programs=[wrong_fast, good_slow, good_fast],
        show_progress=False,
    )

    assert result.selected_program.name == "good_fast"
    assert result.selected_step2_total_time_ms == 2.0
    assert result.unsupported_plan_count == 1
