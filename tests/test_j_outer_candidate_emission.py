# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Task 4.1 — Regression tests for J_outer async candidate emission.

Covers:
- J_outer async overlap candidates are emitted from GEMM search
- Tiled J_outer/J_inner axes appear in candidate ReduceOp.indices
- Example IR/code output makes async candidates visible
- Non-legalized candidates fall back to blocking_collective
"""

import ast
import copy
import textwrap

import pytest

from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.legalization import (
    fallback_failed_async_candidates,
    legalize_async_reductions,
    prepare_pipeline,
)
from mercury.ir.utils import collect_reduce
from mercury.search.search import search
from utils.gemm_dsl import format_gemm_template


def _build_gemm_program(m_len: int = 64, n_len: int = 128, k_len: int = 64):
    """Parse a GEMM template into an IR program."""
    source = format_gemm_template(m_len, n_len, k_len)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return builder.visit(node)
    raise ValueError("Could not find function definition")


# ---------------------------------------------------------------------------
# 4.1.1 — J_outer candidate emission from search
# ---------------------------------------------------------------------------


class TestJOuterCandidateEmission:
    """Verify GEMM search emits async overlap candidates containing J_outer."""

    def test_search_emits_async_collective_overlap_candidates(self):
        """At least one search candidate uses async_collective_overlap strategy."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        async_candidates = [
            c
            for c in candidates
            if any(
                r.managed_collective_strategy == "async_collective_overlap"
                for r in c.visit(collect_reduce)
            )
        ]
        assert len(async_candidates) > 0, (
            "Search should emit at least one async_collective_overlap candidate"
        )

    def test_async_candidate_has_j_outer_in_indices(self):
        """Async candidates should have J_outer (tiled J axis) in ReduceOp.indices."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        found_j_outer = False
        for candidate in candidates:
            for reduce_op in candidate.visit(collect_reduce):
                if reduce_op.managed_collective_strategy != "async_collective_overlap":
                    continue
                if reduce_op.indices is None:
                    continue
                from mercury.ir.elements import Axis

                axis_names = [
                    idx.name for idx in reduce_op.indices if isinstance(idx, Axis)
                ]
                if any("J" in name for name in axis_names):
                    found_j_outer = True
                    break
            if found_j_outer:
                break

        assert found_j_outer, (
            "At least one async candidate should have a J-related axis in ReduceOp.indices"
        )

    def test_async_candidate_indices_not_none(self):
        """Every async_collective_overlap candidate must have non-None ReduceOp.indices."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        for candidate in candidates:
            for reduce_op in candidate.visit(collect_reduce):
                if reduce_op.managed_collective_strategy != "async_collective_overlap":
                    continue
                assert reduce_op.indices is not None, (
                    f"Async overlap candidate has ReduceOp.indices=None "
                    f"(overlap axis={reduce_op.async_collective_overlap_axis})"
                )

    def test_async_candidate_has_stage_count_ge_2(self):
        """Async candidates must have stage_count >= 2."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        async_candidates = [
            c
            for c in candidates
            if any(
                r.managed_collective_strategy == "async_collective_overlap"
                for r in c.visit(collect_reduce)
            )
        ]
        assert len(async_candidates) > 0

        for candidate in async_candidates:
            for reduce_op in candidate.visit(collect_reduce):
                if reduce_op.managed_collective_strategy != "async_collective_overlap":
                    continue
                assert reduce_op.async_collective_stage_count >= 2, (
                    f"Async candidate stage_count={reduce_op.async_collective_stage_count}"
                )


# ---------------------------------------------------------------------------
# 4.1.2 — Example IR/code output visibility of async candidates
# ---------------------------------------------------------------------------


class TestAsyncCandidateCodeVisibility:
    """Verify that async candidates produce visible async constructs in generated code."""

    def test_async_candidate_codegen_emits_async_op(self):
        """Generated code for async candidate includes async_op=True."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        async_candidates = [
            c
            for c in candidates
            if any(
                r.managed_collective_strategy == "async_collective_overlap"
                for r in c.visit(collect_reduce)
            )
        ]
        assert len(async_candidates) > 0

        candidate = copy.deepcopy(async_candidates[0])
        eliminate_loops(candidate)
        code = generate_pytorch_code(candidate)

        assert "async_op=True" in code, (
            "Async candidate codegen must include async_op=True"
        )

    def test_async_candidate_codegen_emits_slot_arrays(self):
        """Generated code for async candidate includes slot and work arrays."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        async_candidates = [
            c
            for c in candidates
            if any(
                r.managed_collective_strategy == "async_collective_overlap"
                for r in c.visit(collect_reduce)
            )
        ]
        assert len(async_candidates) > 0

        candidate = copy.deepcopy(async_candidates[0])
        eliminate_loops(candidate)
        code = generate_pytorch_code(candidate)

        assert "_async_slots" in code, "Code must declare async slots"
        assert "_async_works" in code, "Code must declare async works"

    def test_legalized_candidate_codegen_emits_pending_array(self):
        """A legalized async candidate produces pending-tile tracking in codegen."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        async_candidates = [
            c
            for c in candidates
            if any(
                r.managed_collective_strategy == "async_collective_overlap"
                for r in c.visit(collect_reduce)
            )
        ]
        assert len(async_candidates) > 0

        candidate = copy.deepcopy(async_candidates[0])
        regions = prepare_pipeline(candidate)
        if len(regions) == 0:
            pytest.skip("No legalized regions produced for this candidate")

        eliminate_loops(candidate)
        code = generate_pytorch_code(candidate)

        assert "_pending" in code, (
            "Legalized codegen must emit pending-tile tracking array"
        )

    def test_prepared_pipeline_produces_pipeline_region_comment(self):
        """Prepared pipeline produces a pipeline region comment in codegen."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        async_candidates = [
            c
            for c in candidates
            if any(
                r.managed_collective_strategy == "async_collective_overlap"
                for r in c.visit(collect_reduce)
            )
        ]
        assert len(async_candidates) > 0

        candidate = copy.deepcopy(async_candidates[0])
        regions = prepare_pipeline(candidate)
        if len(regions) == 0:
            pytest.skip("No legalized regions produced for this candidate")

        eliminate_loops(candidate)
        code = generate_pytorch_code(candidate)

        assert "pipeline region" in code, (
            "Legalized codegen must emit pipeline region comment"
        )


# ---------------------------------------------------------------------------
# 4.1.3 — Blocking fallback for non-legalized candidates
# ---------------------------------------------------------------------------


class TestBlockingFallback:
    """Verify that non-legalizable async candidates fall back to blocking_collective."""

    def test_single_tile_candidate_falls_back_to_blocking(self):
        """GEMM with tile_count=1 on J axis cannot legalize → blocking fallback."""
        program = _build_gemm_program(m_len=64, n_len=32, k_len=64)
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        for candidate in candidates:
            regions = legalize_async_reductions(candidate)
            fallback_failed_async_candidates(candidate, regions)

            for reduce_op in candidate.visit(collect_reduce):
                if reduce_op.shard_dim and len(reduce_op.comm) == 0:
                    if reduce_op.async_collective_overlap_axis is not None:
                        overlap_axis = reduce_op.async_collective_overlap_axis
                        tile_count = int(overlap_axis.size) // int(
                            overlap_axis.min_block_size
                        )
                        if tile_count < 2:
                            assert (
                                reduce_op.managed_collective_strategy
                                == "blocking_collective"
                            ), (
                                f"Single-tile candidate should be blocking but is "
                                f"{reduce_op.managed_collective_strategy}"
                            )

    def test_fallback_does_not_touch_ring_candidates(self):
        """Ring-overlap candidates remain ring_overlap after fallback."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        ring_candidates = [
            c
            for c in candidates
            if any(len(r.comm) > 0 for r in c.visit(collect_reduce))
        ]

        for candidate in ring_candidates:
            original_strategies = {
                id(r): r.managed_collective_strategy
                for r in candidate.visit(collect_reduce)
                if len(r.comm) > 0
            }
            regions = legalize_async_reductions(candidate)
            fallback_failed_async_candidates(candidate, regions)

            for reduce_op in candidate.visit(collect_reduce):
                if len(reduce_op.comm) > 0:
                    orig = original_strategies.get(id(reduce_op))
                    if orig is not None:
                        assert reduce_op.managed_collective_strategy == orig, (
                            f"Ring candidate strategy changed from {orig} "
                            f"to {reduce_op.managed_collective_strategy}"
                        )

    def test_prepare_pipeline_is_idempotent(self):
        """Calling prepare_pipeline twice does not duplicate regions."""
        program = _build_gemm_program()
        mesh = DeviceMesh([0, 1], (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))

        async_candidates = [
            c
            for c in candidates
            if any(
                r.managed_collective_strategy == "async_collective_overlap"
                for r in c.visit(collect_reduce)
            )
        ]
        if len(async_candidates) == 0:
            pytest.skip("No async candidates to test idempotency")

        candidate = copy.deepcopy(async_candidates[0])
        regions1 = prepare_pipeline(candidate)
        regions2 = prepare_pipeline(candidate)

        assert len(regions1) == len(regions2), (
            f"prepare_pipeline not idempotent: first={len(regions1)}, second={len(regions2)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
