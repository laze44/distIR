# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tests for FFN two-step search with boundary layouts and edge reshard transitions."""

import ast
import textwrap

from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.elements import Axis, Buffer
from mercury.ir.nodes import AxisDef, BufferMatch, Program
from mercury.search.estimate import EstimateResult, load_hardware_config
from mercury.search.ffn_two_step_search import search_ffn_two_step
from mercury.search.mapping_constraints import (
    load_operator_tensor_mapping_constraints,
    logical_layout_signature_from_buffer,
)
from mercury.search.search import search
from utils.ffn_dsl import (
    ffn_down_gemm_manage_reduction,
    ffn_gate_gemm_manage_reduction,
    ffn_up_gemm_manage_reduction,
)


def _build_program_from_source(source: str):
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return builder.visit(node)
    raise ValueError("Could not find function definition")


def _build_mock_program(
    name: str,
    mesh_shape,
    a_spec,
    b_spec,
    c_spec,
    k_dim: int,
    n_dim: int,
):
    world_size = 1
    for dim_size in mesh_shape:
        world_size *= int(dim_size)
    mesh = DeviceMesh(list(range(world_size)), tuple(mesh_shape))

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


def test_search_ffn_two_step_returns_ranked_plans_with_real_candidates():
    m_len = 64
    d_model = 64
    d_ffn = 64

    operator_sources = {
        "gate": ffn_gate_gemm_manage_reduction.format(
            M_LEN=m_len,
            DM_LEN=d_model,
            DFFN_LEN=d_ffn,
        ),
        "up": ffn_up_gemm_manage_reduction.format(
            M_LEN=m_len,
            DM_LEN=d_model,
            DFFN_LEN=d_ffn,
        ),
        "down": ffn_down_gemm_manage_reduction.format(
            M_LEN=m_len,
            DM_LEN=d_model,
            DFFN_LEN=d_ffn,
        ),
    }
    programs = {
        operator_name: _build_program_from_source(source)
        for operator_name, source in operator_sources.items()
    }

    mesh = DeviceMesh([0, 1], (1, 2))
    constraints = load_operator_tensor_mapping_constraints(
        "config/ffn_tensor_mapping.json",
        ["gate", "up", "down"],
    )

    candidate_programs = {}
    for operator_name in ("gate", "up", "down"):
        operator_candidates = list(
            search(
                programs[operator_name],
                mesh,
                ["I", "J", "K"],
                tensor_mapping_constraints=constraints.get(operator_name),
            )
        )
        assert len(operator_candidates) > 0
        candidate_programs[operator_name] = operator_candidates

    hw_config = load_hardware_config("config/h100.json")
    result = search_ffn_two_step(
        operator_programs=programs,
        origin_mesh=mesh,
        split_axis_names=["I", "J", "K"],
        hw_config=hw_config,
        tensor_mapping_constraints=constraints,
        layout_top_k=2,
        candidate_programs=candidate_programs,
        show_progress=False,
    )

    assert set(result.selected_programs.keys()) == {"gate", "up", "down"}
    assert set(result.selected_plan.activation_layouts.keys()) == {"L_in", "L_mid", "L_out"}
    assert set(result.selected_plan.weight_layouts.keys()) == {"W_gate", "W_up", "W_down"}
    assert 1 <= len(result.ranked_plans) <= 2
    assert result.ranked_plans[0].step1_total_time_ms <= result.ranked_plans[-1].step1_total_time_ms
    assert result.total_time_ms >= 0.0

    assert (
        logical_layout_signature_from_buffer(result.selected_programs["gate"].body[3].buffer)
        == result.selected_plan.operator_layouts["gate"]["A"]
    )
    assert (
        logical_layout_signature_from_buffer(result.selected_programs["down"].body[5].buffer)
        == result.selected_plan.operator_layouts["down"]["C"]
    )


def test_search_ffn_two_step_step2_uses_logical_layout_filter(monkeypatch):
    from mercury.search import ffn_two_step_search as two_step_module

    mesh_shape = (2, 2)
    i0 = [(ShardType.SHARD, [0]), ShardType.REPLICATE]
    i1 = [(ShardType.SHARD, [1]), ShardType.REPLICATE]
    m0 = [ShardType.REPLICATE, (ShardType.SHARD, [0])]
    o0 = [(ShardType.SHARD, [0]), ShardType.REPLICATE]
    b0 = [ShardType.REPLICATE, ShardType.REPLICATE]

    step1_candidates = {
        "gate": [_build_mock_program("gate_plan", mesh_shape, i0, b0, m0, k_dim=8, n_dim=16)],
        "up": [_build_mock_program("up_plan", mesh_shape, i0, b0, m0, k_dim=8, n_dim=16)],
        "down": [_build_mock_program("down_plan", mesh_shape, m0, b0, o0, k_dim=16, n_dim=8)],
    }

    rerun_pools = {
        "gate_seed": [
            _build_mock_program("gate_wrong_fast", mesh_shape, i1, b0, m0, k_dim=8, n_dim=16),
            _build_mock_program("gate_exact_slow", mesh_shape, i0, b0, m0, k_dim=8, n_dim=16),
        ],
        "up_seed": [
            _build_mock_program("up_wrong_fast", mesh_shape, i1, b0, m0, k_dim=8, n_dim=16),
            _build_mock_program("up_exact_slow", mesh_shape, i0, b0, m0, k_dim=8, n_dim=16),
        ],
        "down_seed": [
            _build_mock_program("down_wrong_fast", mesh_shape, i1, b0, o0, k_dim=16, n_dim=8),
            _build_mock_program("down_exact_slow", mesh_shape, m0, b0, o0, k_dim=16, n_dim=8),
        ],
    }

    exec_costs = {
        "gate_plan": 1.0,
        "up_plan": 1.0,
        "down_plan": 1.0,
        "gate_wrong_fast": 0.1,
        "gate_exact_slow": 3.0,
        "up_wrong_fast": 0.1,
        "up_exact_slow": 3.0,
        "down_wrong_fast": 0.1,
        "down_exact_slow": 3.0,
    }

    def fake_estimate_program(program, hw_config):
        del hw_config
        exec_ms = exec_costs[program.name]
        return EstimateResult(exec_ms, 0.0, exec_ms)

    def fake_search_with_progress(
        input_program,
        origin_mesh,
        split_axis_names,
        tensor_mapping_constraints=None,
        program_filter=None,
        progress_desc=None,
        show_progress=True,
        miniters=32,
        mininterval=0.5,
    ):
        del origin_mesh, split_axis_names, tensor_mapping_constraints, progress_desc
        del show_progress, miniters, mininterval
        for program in rerun_pools[input_program.name]:
            if program_filter is None or program_filter(program):
                yield program

    monkeypatch.setattr(two_step_module, "estimate_program", fake_estimate_program)
    monkeypatch.setattr(two_step_module, "search_with_progress", fake_search_with_progress)

    operator_programs = {
        "gate": Program(name="gate_seed", inputs=[], defaults=[], outputs=[], body=[], mesh=None),
        "up": Program(name="up_seed", inputs=[], defaults=[], outputs=[], body=[], mesh=None),
        "down": Program(name="down_seed", inputs=[], defaults=[], outputs=[], body=[], mesh=None),
    }

    hw_config = load_hardware_config("config/h100.json")
    origin_mesh = DeviceMesh(list(range(4)), mesh_shape)
    result = search_ffn_two_step(
        operator_programs=operator_programs,
        origin_mesh=origin_mesh,
        split_axis_names=["I", "J", "K"],
        hw_config=hw_config,
        layout_top_k=1,
        candidate_programs=step1_candidates,
        show_progress=False,
    )

    assert result.selected_programs["gate"].name == "gate_exact_slow"
    assert result.selected_programs["up"].name == "up_exact_slow"
    assert result.selected_programs["down"].name == "down_exact_slow"


def test_step1_allows_mismatch_with_explicit_edge_transitions(monkeypatch):
    from mercury.search import ffn_two_step_search as two_step_module

    mesh_shape = (2, 2)
    i0 = [(ShardType.SHARD, [0]), ShardType.REPLICATE]
    i1 = [(ShardType.SHARD, [1]), ShardType.REPLICATE]
    m0 = [ShardType.REPLICATE, (ShardType.SHARD, [0])]
    m1 = [ShardType.REPLICATE, (ShardType.SHARD, [1])]
    o0 = [(ShardType.SHARD, [0]), ShardType.REPLICATE]
    b0 = [ShardType.REPLICATE, ShardType.REPLICATE]

    candidates = {
        "gate": [_build_mock_program("gate0", mesh_shape, i0, b0, m0, k_dim=8, n_dim=16)],
        "up": [_build_mock_program("up0", mesh_shape, i1, b0, m1, k_dim=8, n_dim=16)],
        "down": [_build_mock_program("down0", mesh_shape, m0, b0, o0, k_dim=16, n_dim=8)],
    }

    def fake_estimate_program(program, hw_config):
        del hw_config
        return EstimateResult(1.0, 0.0, 1.0)

    def fake_estimate_reshard(src_layout, dst_layout, hw_config, origin_mesh, dtype=None):
        del hw_config, origin_mesh, dtype
        return 0.0 if src_layout == dst_layout else 5.0

    monkeypatch.setattr(two_step_module, "estimate_program", fake_estimate_program)
    monkeypatch.setattr(
        two_step_module,
        "estimate_reshard_time_from_logical_layout",
        fake_estimate_reshard,
    )

    hw_config = load_hardware_config("config/h100.json")
    origin_mesh = DeviceMesh(list(range(4)), mesh_shape)
    ranked_plans, candidate_counts = two_step_module._top_plans_from_candidates(
        candidate_programs=candidates,
        hw_config=hw_config,
        origin_mesh=origin_mesh,
        layout_top_k=1,
    )

    assert candidate_counts == {"gate": 1, "up": 1, "down": 1}
    assert len(ranked_plans) == 1
    plan = ranked_plans[0]
    assert len(plan.edge_transitions) > 0
    assert sum(plan.step1_edge_costs_ms.values()) > 0.0
    assert plan.step1_total_time_ms == (
        sum(plan.step1_operator_costs_ms.values()) + sum(plan.step1_edge_costs_ms.values())
    )
