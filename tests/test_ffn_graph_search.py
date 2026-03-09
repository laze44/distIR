# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tests for FFN graph-level joint search."""

import ast
import textwrap

from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.elements import Axis, Buffer
from mercury.ir.nodes import AxisDef, BufferMatch, Program
from mercury.search.estimate import EstimateResult, load_hardware_config
from mercury.search.ffn_graph_search import search_ffn
from mercury.search.mapping_constraints import load_operator_tensor_mapping_constraints
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
        shard_spec=ShardingSpec(mesh, [ShardType.REPLICATE, ShardType.REPLICATE]),
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


def test_search_ffn_returns_non_empty_result_with_real_candidates():
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

    candidates = {}
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
        candidates[operator_name] = operator_candidates

    hw_config = load_hardware_config("config/h100.json")
    result = search_ffn(candidates, hw_config, mesh)

    assert set(result.selected_programs.keys()) == {"gate", "up", "down"}
    assert "L_in" in result.selected_layouts
    assert "L_mid" in result.selected_layouts
    assert result.total_time_ms >= 0.0


def test_search_ffn_joint_optimum_not_equal_independent_exec_sum(monkeypatch):
    from mercury.search import ffn_graph_search as ffn_module

    mesh_shape = (2, 2)
    i0 = [(ShardType.SHARD, [0]), ShardType.REPLICATE]
    i1 = [(ShardType.SHARD, [1]), ShardType.REPLICATE]
    m0 = [ShardType.REPLICATE, (ShardType.SHARD, [0])]
    m1 = [ShardType.REPLICATE, (ShardType.SHARD, [1])]

    candidates = {
        "gate": [
            _build_mock_program("gate_fast", mesh_shape, i0, m0, k_dim=8, n_dim=16),
            _build_mock_program("gate_slow", mesh_shape, i0, m1, k_dim=8, n_dim=16),
        ],
        "up": [
            _build_mock_program("up_fast", mesh_shape, i1, m1, k_dim=8, n_dim=16),
            _build_mock_program("up_slow", mesh_shape, i0, m1, k_dim=8, n_dim=16),
        ],
        "down": [
            _build_mock_program("down_fast", mesh_shape, m0, i0, k_dim=16, n_dim=8),
            _build_mock_program("down_slow", mesh_shape, m1, i0, k_dim=16, n_dim=8),
        ],
    }

    exec_costs = {
        "gate_fast": 1.0,
        "gate_slow": 3.0,
        "up_fast": 1.0,
        "up_slow": 3.0,
        "down_fast": 1.0,
        "down_slow": 3.0,
    }

    def fake_estimate_program(program, hw_config):
        exec_ms = exec_costs[program.name]
        return EstimateResult(exec_ms, 0.0, exec_ms)

    def fake_estimate_reshard(src_buffer, dst_buffer, hw_config, origin_mesh):
        src_sig = ffn_module._layout_signature(src_buffer)
        dst_sig = ffn_module._layout_signature(dst_buffer)
        return 0.0 if src_sig == dst_sig else 5.0

    monkeypatch.setattr(ffn_module, "estimate_program", fake_estimate_program)
    monkeypatch.setattr(ffn_module, "estimate_reshard_time", fake_estimate_reshard)

    hw_config = load_hardware_config("config/h100.json")
    origin_mesh = DeviceMesh(list(range(4)), mesh_shape)
    result = search_ffn(candidates, hw_config, origin_mesh)

    independent_exec_sum = (
        min(exec_costs["gate_fast"], exec_costs["gate_slow"])
        + min(exec_costs["up_fast"], exec_costs["up_slow"])
        + min(exec_costs["down_fast"], exec_costs["down_slow"])
    )

    assert result.total_time_ms > independent_exec_sum


def test_search_ffn_both_l_in_and_l_mid_affect_selection(monkeypatch):
    from mercury.search import ffn_graph_search as ffn_module

    mesh_shape = (2, 2)
    i0 = [(ShardType.SHARD, [0]), ShardType.REPLICATE]
    i1 = [(ShardType.SHARD, [1]), ShardType.REPLICATE]
    m0 = [ShardType.REPLICATE, (ShardType.SHARD, [0])]
    m1 = [ShardType.REPLICATE, (ShardType.SHARD, [1])]

    candidates = {
        "gate": [
            _build_mock_program("gate0", mesh_shape, i0, m0, k_dim=8, n_dim=16),
            _build_mock_program("gate1", mesh_shape, i1, m1, k_dim=8, n_dim=16),
        ],
        "up": [
            _build_mock_program("up0", mesh_shape, i0, m1, k_dim=8, n_dim=16),
            _build_mock_program("up1", mesh_shape, i1, m0, k_dim=8, n_dim=16),
        ],
        "down": [
            _build_mock_program("down0", mesh_shape, m0, i0, k_dim=16, n_dim=8),
            _build_mock_program("down1", mesh_shape, m1, i0, k_dim=16, n_dim=8),
        ],
    }

    def fake_estimate_program(program, hw_config):
        return EstimateResult(1.0, 0.0, 1.0)

    def full_cost(src_buffer, dst_buffer, hw_config, origin_mesh):
        src_sig = ffn_module._layout_signature(src_buffer)
        dst_sig = ffn_module._layout_signature(dst_buffer)
        return 0.0 if src_sig == dst_sig else 10.0

    def ignore_l_in_cost(src_buffer, dst_buffer, hw_config, origin_mesh):
        if src_buffer.get_shape()[1] == 8 and dst_buffer.get_shape()[1] == 8:
            return 0.0
        return full_cost(src_buffer, dst_buffer, hw_config, origin_mesh)

    def ignore_l_mid_cost(src_buffer, dst_buffer, hw_config, origin_mesh):
        if src_buffer.get_shape()[1] == 16 and dst_buffer.get_shape()[1] == 16:
            return 0.0
        return full_cost(src_buffer, dst_buffer, hw_config, origin_mesh)

    monkeypatch.setattr(ffn_module, "estimate_program", fake_estimate_program)

    hw_config = load_hardware_config("config/h100.json")
    origin_mesh = DeviceMesh(list(range(4)), mesh_shape)

    monkeypatch.setattr(ffn_module, "estimate_reshard_time", full_cost)
    baseline = search_ffn(candidates, hw_config, origin_mesh)

    monkeypatch.setattr(ffn_module, "estimate_reshard_time", ignore_l_in_cost)
    ignore_l_in = search_ffn(candidates, hw_config, origin_mesh)

    monkeypatch.setattr(ffn_module, "estimate_reshard_time", ignore_l_mid_cost)
    ignore_l_mid = search_ffn(candidates, hw_config, origin_mesh)

    assert baseline.selected_indices != ignore_l_in.selected_indices
    assert baseline.total_time_ms != ignore_l_mid.total_time_ms
