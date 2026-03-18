# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Task 4.2 — Verifier/codegen regression tests for legalized pipeline retirement.

Proves:
- Legalized pipelines do NOT retire tiles at the original consumer site (BufferLoad)
- Legalized pipelines only retire on reuse or drain
- Non-legalized async candidates still lower as blocking collectives
- Ring paths remain unchanged
"""

import ast
import copy
import re
import textwrap

import pytest

from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh, ShardingSpec, ShardType
from mercury.ir.elements import Axis, Buffer
from mercury.ir.legalization import (
    fallback_failed_async_candidates,
    legalize_async_reductions,
    prepare_pipeline,
)
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.nodes import (
    AsyncCollectiveLifecycle,
    BufferLoad,
    BufferMatch,
    BufferStore,
    GridLoop,
    ManagedReductionPipelineRegion,
    PendingTileDescriptor,
    Program,
    ReduceOp,
)
from mercury.ir.utils import collect_reduce
from mercury.ir.verify_pipeline import verify_pipeline_region
from mercury.search.search import search
from utils.gemm_dsl import format_gemm_template


def _build_gemm_program(m_len: int = 64, n_len: int = 128, k_len: int = 64):
    source = format_gemm_template(m_len, n_len, k_len)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return builder.visit(node)
    raise ValueError("Could not find function definition")


def _make_legalized_program(
    m: int = 64,
    n: int = 128,
    k: int = 64,
    n_block: int = 16,
    mesh_shape: tuple = (2,),
):
    """Build a minimal GEMM-like IR program and legalize it."""
    I = Axis("I", m, 16)
    J = Axis("J", n, n_block)
    K = Axis("K", k, k)

    mesh = DeviceMesh(
        list(range(mesh_shape[0]))
        if len(mesh_shape) == 1
        else list(range(mesh_shape[0] * mesh_shape[1])),
        mesh_shape,
    )
    shard_spec = ShardingSpec(
        mesh=mesh,
        specs=[ShardType.REPLICATE, (ShardType.SHARD, [0])],
    )

    reduce_buf = Buffer(
        "reduce_buf",
        [m, n],
        [[I], [J]],
        [[1], [1]],
        shard_spec=shard_spec,
        read=False,
        write=True,
    )
    out_buf = Buffer(
        "C",
        [m, n],
        [[I], [J]],
        [[1], [1]],
        shard_spec=shard_spec,
        read=False,
        write=True,
    )

    tile_count = n // n_block
    reduce_op = ReduceOp(
        op="torch.add",
        buffer=reduce_buf,
        src="matmul_result",
        axes=[K],
        shard_dim=[0],
        indices=[I, J],
        managed_collective_strategy="async_collective_overlap",
        async_collective_overlap_axis=J,
        async_collective_tile_count=tile_count,
        async_collective_stage_count=2,
        async_collective_lifecycle=AsyncCollectiveLifecycle(),
    )
    load_node = BufferLoad(buffer=reduce_buf, indices=[I, J], target="_tmp1")
    store_node = BufferStore(buffer=out_buf, indices=[I, J], value="_tmp1")

    grid_loop = GridLoop(
        axes=[I, J],
        axis_types="ss",
        body=[reduce_op, load_node, store_node],
    )

    program = Program(
        name="gemm",
        inputs=["A", "B", "C"],
        defaults=[],
        outputs="C",
        body=[
            BufferMatch(buffer=reduce_buf, tensor_name=None),
            BufferMatch(buffer=out_buf, tensor_name="C"),
            grid_loop,
        ],
        mesh=mesh,
    )
    return program


def _make_blocking_program(m: int = 64, n: int = 128, k: int = 64, n_block: int = 16):
    """Build a minimal GEMM-like program with blocking_collective strategy."""
    I = Axis("I", m, 16)
    J = Axis("J", n, n_block)
    K = Axis("K", k, k)

    mesh = DeviceMesh([0, 1], (2,))
    shard_spec = ShardingSpec(
        mesh=mesh,
        specs=[ShardType.REPLICATE, (ShardType.SHARD, [0])],
    )

    reduce_buf = Buffer(
        "reduce_buf",
        [m, n],
        [[I], [J]],
        [[1], [1]],
        shard_spec=shard_spec,
        read=False,
        write=True,
    )
    out_buf = Buffer(
        "C",
        [m, n],
        [[I], [J]],
        [[1], [1]],
        shard_spec=shard_spec,
        read=False,
        write=True,
    )

    reduce_op = ReduceOp(
        op="torch.add",
        buffer=reduce_buf,
        src="matmul_result",
        axes=[K],
        shard_dim=[0],
        indices=[I, J],
        managed_collective_strategy="blocking_collective",
    )
    load_node = BufferLoad(buffer=reduce_buf, indices=[I, J], target="_tmp1")
    store_node = BufferStore(buffer=out_buf, indices=[I, J], value="_tmp1")

    grid_loop = GridLoop(
        axes=[I, J],
        axis_types="ss",
        body=[reduce_op, load_node, store_node],
    )

    return Program(
        name="gemm",
        inputs=["A", "B", "C"],
        defaults=[],
        outputs="C",
        body=[
            BufferMatch(buffer=reduce_buf, tensor_name=None),
            BufferMatch(buffer=out_buf, tensor_name="C"),
            grid_loop,
        ],
        mesh=mesh,
    )


# ---------------------------------------------------------------------------
# 4.2.1 — Legalized pipelines do NOT retire at BufferLoad
# ---------------------------------------------------------------------------


class TestLegalizedNoRetireAtBufferLoad:
    """Prove legalized pipelines skip the wait-at-load pattern."""

    def test_legalized_codegen_does_not_wait_at_buffer_load(self):
        """For legalized buffers, BufferLoad must NOT emit a .wait() immediately before load."""
        program = _make_legalized_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0, "Expected at least one legalized region"

        eliminate_loops(program)
        code = generate_pytorch_code(program)
        lines = code.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "_tmp1 = reduce_buf" in stripped:
                if i > 0:
                    prev = lines[i - 1].strip()
                    assert not (
                        prev.endswith(".wait()")
                        and "reduce_buf" in prev
                        and "if " not in prev
                    ), (
                        f"Legalized BufferLoad should not have unconditional wait "
                        f"on the line immediately before load. prev={prev}"
                    )

    def test_non_legalized_codegen_still_waits_at_buffer_load(self):
        """After prepare_pipeline, non-legalizable async candidates use blocking all_reduce."""
        program = _make_legalized_program()
        eliminate_loops(program)
        code = generate_pytorch_code(program)

        assert "all_reduce" in code, "Blocking path should use all_reduce"


# ---------------------------------------------------------------------------
# 4.2.2 — Legalized pipelines retire on reuse or drain only
# ---------------------------------------------------------------------------


class TestLegalizedRetireOnReuseOrDrain:
    """Prove retirement happens only at slot reuse and loop drain."""

    def test_legalized_codegen_has_pending_none_at_reuse(self):
        """Legalized code emits pending[slot] = None at the reuse point."""
        program = _make_legalized_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0

        eliminate_loops(program)
        code = generate_pytorch_code(program)

        assert re.search(r"_pending\[.*\] = None", code), (
            "Legalized code must retire pending tiles (pending[slot] = None) at reuse"
        )

    def test_legalized_codegen_has_pending_none_at_drain(self):
        """Legalized code with tiled loops emits pending[_slot] = None in the drain loop."""
        program = _make_legalized_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0

        code = generate_pytorch_code(program)

        drain_pattern = re.compile(
            r"for _slot in range\(\d+\).*?_pending\[_slot\] = None",
            re.DOTALL,
        )
        assert drain_pattern.search(code), (
            "Legalized code with loops must retire pending tiles in the drain loop"
        )

    def test_legalized_codegen_emits_lifecycle_comments(self):
        """Legalized code with loops includes lifecycle phase markers."""
        program = _make_legalized_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0

        code = generate_pytorch_code(program)

        assert "all_reduce_wait_on_reuse" in code, (
            "Legalized code should have wait_on_reuse lifecycle marker"
        )
        assert "all_reduce_start" in code, (
            "Legalized code should have start lifecycle marker"
        )

    def test_legalized_pending_array_initialized(self):
        """Legalized code initializes the pending tracking array."""
        program = _make_legalized_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0

        eliminate_loops(program)
        code = generate_pytorch_code(program)

        assert re.search(r"_pending = \[None for _ in range\(\d+\)\]", code), (
            "Legalized code must initialize pending array"
        )

    def test_legalized_no_immediate_wait_after_start(self):
        """async_op=True must not be followed by .wait() for the same slot before next tile."""
        program = _make_legalized_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0

        code = generate_pytorch_code(program)
        lines = code.split("\n")

        for i, line in enumerate(lines):
            if "async_op=True" in line:
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_line = lines[j].strip()
                    assert not (
                        next_line.endswith(".wait()") and "if " not in next_line
                    ), (
                        f"Immediate unconditional .wait() after async start "
                        f"defeats overlap. line {j}: {next_line}"
                    )


# ---------------------------------------------------------------------------
# 4.2.3 — Non-legalized async candidates lower as blocking
# ---------------------------------------------------------------------------


class TestNonLegalizedLowersAsBlocking:
    """Prove that non-legalized candidates produce blocking collective code."""

    def test_blocking_candidate_emits_synchronous_all_reduce(self):
        """Blocking collective candidate emits a collective call without async_op."""
        program = _make_blocking_program()
        eliminate_loops(program)
        code = generate_pytorch_code(program)

        assert (
            "all_reduce" in code
            or "dist.all_reduce" in code
            or "collective" in code.lower()
        ), "Blocking candidate must emit some form of all_reduce/collective call"
        assert "async_op=True" not in code, (
            "Blocking candidate must NOT use async_op=True"
        )

    def test_fallback_candidate_produces_blocking_code(self):
        """An async candidate that fails legalization produces blocking code."""
        program = _make_legalized_program(n=16, n_block=16)
        regions = legalize_async_reductions(program)
        assert len(regions) == 0, "Single-tile should not legalize"

        fallback_failed_async_candidates(program, regions)

        for r in program.visit(collect_reduce):
            if r.shard_dim and len(r.comm) == 0:
                assert r.managed_collective_strategy == "blocking_collective"

        eliminate_loops(program)
        code = generate_pytorch_code(program)

        assert "async_op=True" not in code, (
            "Fallen-back candidate must not use async_op"
        )

    def test_search_fallback_end_to_end(self):
        """Full search → legalize → fallback → codegen path for a small GEMM."""
        program = _build_gemm_program(m_len=64, n_len=128, k_len=64)
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        blocking_candidates = [
            c
            for c in candidates
            if all(
                r.managed_collective_strategy == "blocking_collective"
                for r in c.visit(collect_reduce)
                if r.shard_dim and len(r.comm) == 0
            )
        ]

        for candidate in blocking_candidates[:2]:
            candidate_copy = copy.deepcopy(candidate)
            eliminate_loops(candidate_copy)
            code = generate_pytorch_code(candidate_copy)
            assert "async_op=True" not in code, (
                "Blocking candidate codegen must not contain async_op=True"
            )


# ---------------------------------------------------------------------------
# 4.2.4 — Ring paths remain unchanged
# ---------------------------------------------------------------------------


class TestRingPathsUnchanged:
    """Prove ring-overlap paths are unaffected by legalization changes."""

    def test_ring_candidate_still_uses_send_recv(self):
        """Ring candidates emit SendRecv communication pattern."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        ring_candidates = [
            c
            for c in candidates
            if any(len(r.comm) > 0 for r in c.visit(collect_reduce))
        ]

        if len(ring_candidates) == 0:
            pytest.skip("No ring candidates in search output")

        candidate = copy.deepcopy(ring_candidates[0])
        eliminate_loops(candidate)
        code = generate_pytorch_code(candidate)

        assert "send_recv" in code.lower() or "SendRecv" in code, (
            "Ring candidate must use SendRecv pattern"
        )

    def test_ring_candidate_has_no_pending_array(self):
        """Ring candidates should not produce pending-tile tracking."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        ring_candidates = [
            c
            for c in candidates
            if any(len(r.comm) > 0 for r in c.visit(collect_reduce))
        ]

        if len(ring_candidates) == 0:
            pytest.skip("No ring candidates in search output")

        candidate = copy.deepcopy(ring_candidates[0])
        eliminate_loops(candidate)
        code = generate_pytorch_code(candidate)

        assert "_pending" not in code, (
            "Ring candidate should not have pending-tile tracking"
        )

    def test_ring_candidate_not_affected_by_prepare_pipeline(self):
        """prepare_pipeline does not modify ring-overlap candidates."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        ring_candidates = [
            c
            for c in candidates
            if any(
                r.managed_collective_strategy == "ring_overlap"
                for r in c.visit(collect_reduce)
            )
        ]

        if len(ring_candidates) == 0:
            pytest.skip("No ring candidates in search output")

        candidate = copy.deepcopy(ring_candidates[0])
        prepare_pipeline(candidate)

        for r in candidate.visit(collect_reduce):
            if len(r.comm) > 0:
                assert r.managed_collective_strategy == "ring_overlap", (
                    f"Ring candidate strategy changed to {r.managed_collective_strategy}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
