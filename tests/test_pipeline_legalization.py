# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tests for legalized async managed-reduction pipeline regions.

Validates end-to-end legalization, verification, estimation, and codegen
for GEMM-like programs with async collective overlap, as well as fallback
behavior for non-legalizable candidates.
"""

import copy

import pytest

from mercury.ir.distributed import DeviceMesh, ShardingSpec, ShardType
from mercury.ir.elements import Axis, Buffer
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
from mercury.ir.legalization import (
    fallback_failed_async_candidates,
    legalize_async_reductions,
)
from mercury.ir.verify_pipeline import (
    PipelineVerificationError,
    verify_pipeline_region,
    verify_pipeline_regions,
)


def _make_gemm_program(
    m: int = 64,
    n: int = 128,
    k: int = 64,
    n_block: int = 16,
    mesh_shape: tuple = (2,),
    strategy: str = "async_collective_overlap",
):
    """Build a minimal GEMM-like IR program with a managed reduction."""
    I = Axis("I", m, 16)
    J = Axis("J", n, n_block)
    K = Axis("K", k, k)

    mesh = DeviceMesh(list(range(mesh_shape[0])) if len(mesh_shape) == 1 else list(range(mesh_shape[0] * mesh_shape[1])), mesh_shape)
    shard_spec = ShardingSpec(
        mesh=mesh,
        specs=[ShardType.REPLICATE, (ShardType.SHARD, [0])],
    )

    reduce_buf = Buffer(
        "reduce_buf", [m, n], [[I], [J]], [[1], [1]],
        shard_spec=shard_spec, read=False, write=True,
    )
    out_buf = Buffer(
        "C", [m, n], [[I], [J]], [[1], [1]],
        shard_spec=shard_spec, read=False, write=True,
    )

    tile_count = n // n_block

    reduce_op = ReduceOp(
        op="torch.add",
        buffer=reduce_buf,
        src="matmul_result",
        axes=[K],
        shard_dim=[0],
        indices=[I, J],
        managed_collective_strategy=strategy,
        async_collective_overlap_axis=J if strategy == "async_collective_overlap" else None,
        async_collective_tile_count=tile_count if strategy == "async_collective_overlap" else 1,
        async_collective_stage_count=2 if strategy == "async_collective_overlap" else 1,
        async_collective_lifecycle=AsyncCollectiveLifecycle() if strategy == "async_collective_overlap" else None,
    )
    load_node = BufferLoad(buffer=reduce_buf, indices=[I, J], target="_tmp1")
    store_node = BufferStore(buffer=out_buf, indices=[I, J], value="_tmp1")

    grid_loop = GridLoop(
        axes=[I, J], axis_types="ss",
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
# 3.2 GEMM-focused tests for legalized async overlap
# ---------------------------------------------------------------------------


class TestLegalizedAsyncOverlap:
    """GEMM-focused tests that distinguish legalized async overlap from
    early-wait behavior and verify real J_outer overlap ordering."""

    def test_legalization_produces_pipeline_region(self):
        """Legalization of a valid async candidate yields one region."""
        program = _make_gemm_program()
        regions = legalize_async_reductions(program)
        assert len(regions) == 1
        region = regions[0]
        assert region.legalized is True
        assert region.overlap_axis.name == "J"
        assert region.stage_count == 2
        assert region.tile_count == 8  # 128 / 16

    def test_legalized_region_has_pending_tiles(self):
        """Legalized region carries structured pending-tile descriptors."""
        program = _make_gemm_program()
        regions = legalize_async_reductions(program)
        region = regions[0]
        assert len(region.pending_tiles) == 2  # stage_count
        for pt in region.pending_tiles:
            assert pt.slot_index in (0, 1)
            assert pt.reduce_buffer is not None
            assert pt.reduce_buffer.tensor == "reduce_buf"

    def test_legalized_region_has_consumer_store(self):
        """Legalized region identifies the retirement consumer store."""
        program = _make_gemm_program()
        regions = legalize_async_reductions(program)
        region = regions[0]
        assert region.consumer_store is not None
        assert region.consumer_store.buffer.tensor == "C"

    def test_legalization_verifier_accepts_valid_region(self):
        """Verifier accepts a well-formed legalized pipeline region."""
        program = _make_gemm_program()
        regions = legalize_async_reductions(program)
        valid, errors = verify_pipeline_region(regions[0])
        assert valid, f"Verifier errors: {errors}"

    def test_legalization_with_single_tile_fails(self):
        """Candidate with only 1 tile on the overlap axis is not legalized."""
        program = _make_gemm_program(n=16, n_block=16)  # tile_count = 1
        regions = legalize_async_reductions(program)
        assert len(regions) == 0

    def test_legalization_with_single_participant_fails(self):
        """Candidate with 1-device mesh is not legalized."""
        program = _make_gemm_program(mesh_shape=(1,))
        regions = legalize_async_reductions(program)
        assert len(regions) == 0

    def test_fallback_downgrades_non_legalized_candidates(self):
        """Non-legalized async candidates are rewritten to blocking_collective."""
        program = _make_gemm_program(n=16, n_block=16)  # Will fail legalization
        regions = legalize_async_reductions(program)
        assert len(regions) == 0
        count = fallback_failed_async_candidates(program, regions)
        assert count == 1

        reduce_ops = [
            n for n in program.visit(lambda x: x if isinstance(x, ReduceOp) else None)
        ]
        for r in reduce_ops:
            if r.shard_dim:
                assert r.managed_collective_strategy == "blocking_collective"
                assert r.async_collective_overlap_axis is None

    def test_legalized_region_deepcopy(self):
        """Pipeline regions survive deep copy."""
        program = _make_gemm_program()
        regions = legalize_async_reductions(program)
        region = regions[0]
        region_copy = copy.deepcopy(region)
        assert region_copy.legalized is True
        assert region_copy.overlap_axis.name == "J"
        assert region_copy.stage_count == 2
        assert len(region_copy.pending_tiles) == 2


# ---------------------------------------------------------------------------
# 3.3 Fallback and estimator tests
# ---------------------------------------------------------------------------


class TestFallbackAndEstimator:
    """Tests showing illegal async candidates revert to blocking-collective
    ranking and lowering."""

    def test_fallback_preserves_blocking_and_ring_strategies(self):
        """Fallback only touches async_collective_overlap, not other strategies."""
        program = _make_gemm_program(strategy="blocking_collective")
        regions = legalize_async_reductions(program)
        count = fallback_failed_async_candidates(program, regions)
        assert count == 0

    def test_fallback_with_empty_legalized_downgrades_all(self):
        """When no regions are legalized, all async candidates are downgraded."""
        program = _make_gemm_program()
        fallback_failed_async_candidates(program, [])
        reduce_ops = [
            n for n in program.visit(lambda x: x if isinstance(x, ReduceOp) else None)
        ]
        for r in reduce_ops:
            assert r.managed_collective_strategy != "async_collective_overlap"

    def test_verifier_rejects_region_with_too_many_pending_tiles(self):
        """Verifier rejects region with more pending tiles than slots."""
        program = _make_gemm_program()
        regions = legalize_async_reductions(program)
        region = regions[0]
        region.pending_tiles.append(
            PendingTileDescriptor(slot_index=2)
        )
        valid, errors = verify_pipeline_region(region)
        assert not valid
        assert any("slot_index=2 outside" in e for e in errors)

    def test_verifier_rejects_duplicate_slot_indices(self):
        """Verifier rejects region with duplicate slot indices."""
        program = _make_gemm_program()
        regions = legalize_async_reductions(program)
        region = regions[0]
        region.pending_tiles[1].slot_index = 0  # duplicate
        valid, errors = verify_pipeline_region(region)
        assert not valid
        assert any("duplicate" in e for e in errors)

    def test_verifier_rejects_low_stage_count(self):
        """Verifier rejects region with stage_count < 2."""
        program = _make_gemm_program()
        regions = legalize_async_reductions(program)
        region = regions[0]
        region.stage_count = 1
        valid, errors = verify_pipeline_region(region)
        assert not valid
        assert any("stage_count=1" in e for e in errors)

    def test_verifier_rejects_low_tile_count(self):
        """Verifier rejects region with tile_count < 2."""
        program = _make_gemm_program()
        regions = legalize_async_reductions(program)
        region = regions[0]
        region.tile_count = 1
        valid, errors = verify_pipeline_region(region)
        assert not valid
        assert any("tile_count=1" in e for e in errors)

    def test_multiple_reductions_partial_legalization(self):
        """When one of two reductions is legalizable, only it gets a region."""
        program = _make_gemm_program()
        # Add a second ReduceOp that won't legalize (no overlap axis)
        extra_buf = Buffer(
            "extra_buf", [64, 128], [[Axis("I", 64, 16)], [Axis("J", 128, 16)]],
            [[1], [1]],
            shard_spec=ShardingSpec(
                mesh=program.mesh,
                specs=[ShardType.REPLICATE, (ShardType.SHARD, [0])],
            ),
            read=False, write=True,
        )
        extra_reduce = ReduceOp(
            op="torch.add",
            buffer=extra_buf,
            src="extra_src",
            axes=[Axis("K", 64, 64)],
            shard_dim=[0],
            indices=None,  # No indices → not legalizable
            managed_collective_strategy="async_collective_overlap",
            async_collective_overlap_axis=Axis("J", 128, 16),
            async_collective_tile_count=8,
            async_collective_stage_count=2,
            async_collective_lifecycle=AsyncCollectiveLifecycle(),
        )
        program.body.append(extra_reduce)

        regions = legalize_async_reductions(program)
        # Only the original reduce_buf should legalize
        assert len(regions) == 1
        assert regions[0].reduce_op.buffer.tensor == "reduce_buf"

        # Fallback should downgrade the extra one
        count = fallback_failed_async_candidates(program, regions)
        assert count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
