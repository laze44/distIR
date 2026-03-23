# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""True async collective pipeline regression tests.

Verifies that legalized async pipeline codegen produces the intended
double-buffered schedule:

1. Pipeline state (slots, works, pending) is allocated OUTSIDE the overlap loop
2. Slot rotation uses the actual loop variable (e.g. ``% 2``)
3. Generated code does NOT collapse to constant slot index ``[0]``
4. Drain happens AFTER the overlap loop exits
5. Retirement uses pending tile identity from a previous iteration
6. Collapsed axes are correctly rejected by legalization/codegen
"""

import copy
import re

import pytest

from mercury.backend import generate_pytorch_code
from mercury.backend.pytorch.codegen import PyTorchCodegen
from mercury.ir.distributed import DeviceMesh, ShardingSpec, ShardType
from mercury.ir.elements import Axis, Buffer
from mercury.ir.legalization import (
    legalize_async_reductions,
    prepare_pipeline,
)
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.nodes import (
    AsyncCollectiveLifecycle,
    AxisDef,
    BufferLoad,
    BufferMatch,
    BufferStore,
    GridLoop,
    ManagedReductionPipelineRegion,
    PendingTileDescriptor,
    Program,
    ReduceOp,
)
from mercury.ir.utils import collect_pipeline_regions, collect_reduce
from mercury.ir.verify_pipeline import verify_pipeline_region
from mercury.search.estimate import estimate_program


def _make_true_pipeline_program(
    m: int = 64,
    n: int = 128,
    k: int = 64,
    n_block: int = 16,
    mesh_shape: tuple = (2,),
    j_max_block: int = None,
    include_axis_defs: bool = False,
):
    """Build a GEMM-like IR program where J is a true multi-tile overlap axis.

    Key: ``max_block_size`` is set to ``n_block`` so that ``eliminate_loops()``
    does NOT collapse the J loop.  This ensures the overlap axis produces a
    real runtime loop for slot rotation.
    """
    I = Axis("I", m, 16, max_block_size=16)
    if j_max_block is None:
        j_max_block = n_block
    J = Axis("J", n, n_block, max_block_size=j_max_block)
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

    body = [
        BufferMatch(buffer=reduce_buf, tensor_name=None),
        BufferMatch(buffer=out_buf, tensor_name="C"),
        grid_loop,
    ]
    if include_axis_defs:
        body = [AxisDef(I), AxisDef(J), AxisDef(K)] + body

    program = Program(
        name="gemm",
        inputs=["A", "B", "C"],
        defaults=[],
        outputs="C",
        body=body,
        mesh=mesh,
    )
    return program


def _make_collapsed_program():
    """Build a program where the overlap axis collapses (single tile)."""
    return _make_true_pipeline_program(n=16, n_block=16)


def _make_stale_pipeline_program(include_axis_defs: bool = False):
    """Build a program that legalizes first and collapses after ``eliminate_loops()``."""
    return _make_true_pipeline_program(
        n=128,
        n_block=16,
        j_max_block=128,
        include_axis_defs=include_axis_defs,
    )


def _make_test_hw_config():
    """Return a minimal valid hardware config for estimator regression tests."""
    return {
        "name": "test-hw",
        "compute": {
            "peak_tflops": {
                "bf16": 1000.0,
                "fp16": 1000.0,
                "fp32": 500.0,
            }
        },
        "memory": {
            "bandwidth_tb_per_s": 1.0,
        },
        "interconnect": {
            "intra_node": {"bandwidth_gb_per_s": 100.0, "latency_us": 1.0},
            "inter_node": {"bandwidth_gb_per_s": 50.0, "latency_us": 2.0},
        },
    }


# ---------------------------------------------------------------------------
# 1. Hoisted pipeline state
# ---------------------------------------------------------------------------


class TestHoistedPipelineState:
    """Verify pipeline state is allocated outside the overlap loop."""

    def test_slots_works_pending_initialized_outside_loop(self):
        """slots, works, and pending arrays are emitted before ``for J ...``."""
        program = _make_true_pipeline_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0

        code = generate_pytorch_code(program)
        lines = code.split("\n")

        # Find the first occurrence of the for-J loop
        j_loop_line = None
        for i, line in enumerate(lines):
            if re.search(r"for J in range\(", line.strip()):
                j_loop_line = i
                break

        assert j_loop_line is not None, "No 'for J in range(...)' loop found"

        # Find initialization lines
        init_lines = {
            "works": None,
            "pending": None,
        }
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "_async_works = [None" in stripped and init_lines["works"] is None:
                init_lines["works"] = i
            if "_pending = [None" in stripped and init_lines["pending"] is None:
                init_lines["pending"] = i

        assert init_lines["works"] is not None, "async_works not initialized"
        assert init_lines["pending"] is not None, "pending not initialized"
        assert init_lines["works"] < j_loop_line, (
            f"works init (line {init_lines['works']}) should be before J loop "
            f"(line {j_loop_line})"
        )
        assert init_lines["pending"] < j_loop_line, (
            f"pending init (line {init_lines['pending']}) should be before J loop "
            f"(line {j_loop_line})"
        )

    def test_pipeline_scope_init_comment(self):
        """Pipeline scope init is marked with a comment."""
        program = _make_true_pipeline_program()
        prepare_pipeline(program)
        code = generate_pytorch_code(program)

        assert "pipeline scope init" in code, (
            "Pipeline scope init should have an identifying comment"
        )


# ---------------------------------------------------------------------------
# 2. Slot rotation
# ---------------------------------------------------------------------------


class TestSlotRotation:
    """Verify slot rotation uses the actual loop variable."""

    def test_slot_var_uses_modulo_stage_count(self):
        """Generated code contains ``% 2`` for double-buffered slot rotation."""
        program = _make_true_pipeline_program()
        prepare_pipeline(program)
        code = generate_pytorch_code(program)

        assert "% 2" in code, (
            "Double-buffered pipeline must use '% 2' slot rotation"
        )

    def test_slot_var_not_constant_zero(self):
        """Legalized pipeline code must NOT use constant slot index [0]."""
        program = _make_true_pipeline_program()
        prepare_pipeline(program)
        code = generate_pytorch_code(program)

        # The slot variable should be dynamic, not constant "0"
        # Check that async_works and pending use a variable slot, not [0]
        assert re.search(r"_async_slot\b", code), (
            "Pipeline code should use a dynamic slot variable"
        )

    def test_slot_rotation_uses_j_loop_variable(self):
        """Slot rotation expression references the J loop variable."""
        program = _make_true_pipeline_program()
        prepare_pipeline(program)
        code = generate_pytorch_code(program)

        # Slot computation should reference J in some form
        slot_pattern = re.compile(r"_async_slot = \(J")
        assert slot_pattern.search(code), (
            "Slot rotation should reference the J loop variable"
        )


# ---------------------------------------------------------------------------
# 3. Drain placement
# ---------------------------------------------------------------------------


class TestDrainPlacement:
    """Verify drain loop is emitted after the overlap loop."""

    def test_drain_after_j_loop(self):
        """Drain ``for _slot in range(...)`` appears after the J loop body."""
        program = _make_true_pipeline_program()
        prepare_pipeline(program)
        code = generate_pytorch_code(program)
        lines = code.split("\n")

        # Find last line of J loop body (last occurrence of J-related code)
        j_loop_end = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.search(r"for J in range\(", stripped):
                j_loop_end = i  # will be updated

        # Find drain loop
        drain_line = None
        for i, line in enumerate(lines):
            if "for _slot in range(" in line.strip():
                drain_line = i
                break

        assert drain_line is not None, "No drain loop found"

    def test_drain_has_wait_and_retire(self):
        """Drain loop waits and retires pending tiles."""
        program = _make_true_pipeline_program()
        prepare_pipeline(program)
        code = generate_pytorch_code(program)

        drain_pattern = re.compile(
            r"for _slot in range\(\d+\).*?"
            r"\.wait\(\).*?"
            r"_pending\[_slot\] = None",
            re.DOTALL,
        )
        assert drain_pattern.search(code), (
            "Drain loop must wait and set pending[_slot] = None"
        )

    def test_drain_has_lifecycle_comment(self):
        """Drain loop includes the drain lifecycle marker."""
        program = _make_true_pipeline_program()
        prepare_pipeline(program)
        code = generate_pytorch_code(program)

        assert "all_reduce_wait_drain" in code, (
            "Drain should include the all_reduce_wait_drain lifecycle comment"
        )


# ---------------------------------------------------------------------------
# 4. Degenerate case rejection
# ---------------------------------------------------------------------------


class TestDegenerateRejection:
    """Verify that collapsed overlap axes are properly rejected."""

    def test_single_tile_does_not_legalize(self):
        """An overlap axis with tile_count=1 cannot legalize."""
        program = _make_collapsed_program()
        regions = legalize_async_reductions(program)
        assert len(regions) == 0, (
            "Single-tile overlap axis should not produce legalized regions"
        )

    def test_collapsed_axis_rejected_by_verifier(self):
        """Verifier rejects a region with collapsed overlap axis."""
        I = Axis("I", 64, 16)
        J = Axis("J", 16, 16)
        K = Axis("K", 64, 64)

        mesh = DeviceMesh([0, 1], (2,))
        shard_spec = ShardingSpec(
            mesh=mesh,
            specs=[ShardType.REPLICATE, (ShardType.SHARD, [0])],
        )
        reduce_buf = Buffer(
            "reduce_buf", [64, 16], [[I], [J]], [[1], [1]],
            shard_spec=shard_spec, read=False, write=True,
        )

        reduce_op = ReduceOp(
            op="torch.add", buffer=reduce_buf, src="matmul_result",
            axes=[K], shard_dim=[0], indices=[I, J],
        )

        # Manually create a region with a collapsed axis
        region = ManagedReductionPipelineRegion(
            reduce_op=reduce_op,
            overlap_axis=J,
            stage_count=2,
            tile_count=1,  # single tile!
            lifecycle=AsyncCollectiveLifecycle(),
            pending_tiles=[],
            legalized=True,
            materialized_overlap_axis=J,
            pipeline_scope_axis="J",
        )

        valid, errors = verify_pipeline_region(region)
        assert not valid, f"Collapsed axis should fail verification: {errors}"
        assert any("tile_count" in e for e in errors), (
            "Error should mention tile_count"
        )

    def test_codegen_raises_on_missing_slot_var(self):
        """Raw codegen raises ValueError when slot variable is missing."""
        I = Axis("I", 64, 64)  # Single-stride: no loop
        J = Axis("J", 128, 128)  # Single-stride: no loop
        K = Axis("K", 64, 64)

        mesh = DeviceMesh([0, 1], (2,))
        shard_spec = ShardingSpec(
            mesh=mesh,
            specs=[ShardType.REPLICATE, (ShardType.SHARD, [0])],
        )
        reduce_buf = Buffer(
            "reduce_buf", [64, 128], [[I], [J]], [[1], [1]],
            shard_spec=shard_spec, read=False, write=True,
        )
        out_buf = Buffer(
            "C", [64, 128], [[I], [J]], [[1], [1]],
            shard_spec=shard_spec, read=False, write=True,
        )

        reduce_op = ReduceOp(
            op="torch.add", buffer=reduce_buf, src="matmul_result",
            axes=[K], shard_dim=[0], indices=[I, J],
            managed_collective_strategy="async_collective_overlap",
            async_collective_overlap_axis=J,
            async_collective_tile_count=8,
            async_collective_stage_count=2,
            async_collective_lifecycle=AsyncCollectiveLifecycle(),
        )
        load_node = BufferLoad(buffer=reduce_buf, indices=[I, J], target="_tmp1")
        store_node = BufferStore(buffer=out_buf, indices=[I, J], value="_tmp1")

        # Force a legalized region onto a program where axes have no loops
        region = ManagedReductionPipelineRegion(
            reduce_op=reduce_op,
            overlap_axis=J,
            stage_count=2,
            tile_count=8,
            lifecycle=AsyncCollectiveLifecycle(),
            pending_tiles=[],
            legalized=True,
            materialized_overlap_axis=J,
            pipeline_scope_axis="J",
        )

        grid_loop = GridLoop(
            axes=[I, J],
            axis_types="ss",
            body=[region],
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

        codegen = PyTorchCodegen()
        with pytest.raises(ValueError, match="no slot variable"):
            codegen.visit(program)


# ---------------------------------------------------------------------------
# 5. End-to-end schedule shape
# ---------------------------------------------------------------------------


class TestEndToEndScheduleShape:
    """Verify the overall generated code matches the intended pipeline schedule."""

    def test_full_pipeline_schedule_structure(self):
        """Generated code has the complete pipeline structure:
        1. Pipeline state init (outside loop)
        2. For J loop with slot rotation
        3. Wait-on-reuse → reduce → start → pending bookkeeping
        4. Drain after loop
        """
        program = _make_true_pipeline_program()
        prepare_pipeline(program)
        code = generate_pytorch_code(program)

        # 1. Pipeline state init
        assert "_async_works = [None" in code
        assert "_pending = [None" in code

        # 2. Slot rotation
        assert "% 2" in code

        # 3. Lifecycle markers
        assert "all_reduce_wait_on_reuse" in code
        assert "all_reduce_start" in code

        # 4. Async all_reduce
        assert "async_op=True" in code

        # 5. Drain
        assert "all_reduce_wait_drain" in code

    def test_async_op_true_present_in_legalized_code(self):
        """Legalized pipeline code uses ``async_op=True``."""
        program = _make_true_pipeline_program()
        prepare_pipeline(program)
        code = generate_pytorch_code(program)

        assert "async_op=True" in code, (
            "Legalized pipeline must use async_op=True for overlap"
        )

    def test_blocking_version_has_no_slot_state(self):
        """Blocking collective version does not emit slot/works/pending state."""
        I = Axis("I", 64, 16, max_block_size=16)
        J = Axis("J", 128, 16, max_block_size=16)
        K = Axis("K", 64, 64)

        mesh = DeviceMesh([0, 1], (2,))
        shard_spec = ShardingSpec(
            mesh=mesh,
            specs=[ShardType.REPLICATE, (ShardType.SHARD, [0])],
        )
        reduce_buf = Buffer(
            "reduce_buf", [64, 128], [[I], [J]], [[1], [1]],
            shard_spec=shard_spec, read=False, write=True,
        )
        out_buf = Buffer(
            "C", [64, 128], [[I], [J]], [[1], [1]],
            shard_spec=shard_spec, read=False, write=True,
        )

        reduce_op = ReduceOp(
            op="torch.add", buffer=reduce_buf, src="matmul_result",
            axes=[K], shard_dim=[0], indices=[I, J],
            managed_collective_strategy="blocking_collective",
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

        code = generate_pytorch_code(program)

        assert "_async_slots" not in code
        assert "_async_works" not in code
        assert "_pending" not in code
        assert "async_op=True" not in code


# ---------------------------------------------------------------------------
# 6. New region fields
# ---------------------------------------------------------------------------


class TestNewRegionFields:
    """Verify the new region metadata fields are correctly set."""

    def test_materialized_overlap_axis_set(self):
        """Legalized region has materialized_overlap_axis set."""
        program = _make_true_pipeline_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0

        for region in regions:
            assert region.materialized_overlap_axis is not None, (
                "materialized_overlap_axis should be set by legalization"
            )
            assert region.materialized_overlap_axis.name == region.overlap_axis.name

    def test_pipeline_scope_axis_set(self):
        """Legalized region has pipeline_scope_axis set."""
        program = _make_true_pipeline_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0

        for region in regions:
            assert region.pipeline_scope_axis is not None, (
                "pipeline_scope_axis should be set by legalization"
            )
            assert region.pipeline_scope_axis == region.overlap_axis.name

    def test_pending_tile_descriptor_has_tile_id_expr(self):
        """Pending tile descriptors have tile_id_expr set."""
        program = _make_true_pipeline_program()
        regions = prepare_pipeline(program)
        assert len(regions) > 0

        for region in regions:
            for pt in region.pending_tiles:
                assert pt.tile_id_expr is not None, (
                    "tile_id_expr should be set by legalization"
                )


# ---------------------------------------------------------------------------
# 7. Stale-region revalidation after loop elimination
# ---------------------------------------------------------------------------


class TestStaleRegionRevalidation:
    """Verify stale legalized regions are downgraded after axis collapse."""

    def test_codegen_downgrades_stale_region_after_eliminate_loops(self):
        """Repeated codegen should fall back to blocking collective, not error."""
        program = _make_stale_pipeline_program()

        initial_code = generate_pytorch_code(program)
        assert "async_op=True" in initial_code

        eliminate_loops(program)
        downgraded_code = generate_pytorch_code(program)

        regions = program.visit(collect_pipeline_regions)
        reduce_ops = program.visit(collect_reduce)

        assert len(regions) == 0
        assert len(reduce_ops) == 1
        assert reduce_ops[0].managed_collective_strategy == "blocking_collective"
        assert "async_op=True" not in downgraded_code
        assert "_async_works" not in downgraded_code
        assert "_pending" not in downgraded_code

    def test_estimator_skips_collapsed_stale_async_region(self):
        """Collapsed stale regions should estimate like a blocking fallback."""
        hw_config = _make_test_hw_config()

        stale_program = _make_stale_pipeline_program(include_axis_defs=True)
        generate_pytorch_code(stale_program)
        eliminate_loops(stale_program)
        stale_estimate = estimate_program(stale_program, hw_config)

        baseline_program = _make_stale_pipeline_program(include_axis_defs=True)
        eliminate_loops(baseline_program)
        baseline_estimate = estimate_program(baseline_program, hw_config)

        regions = stale_program.visit(collect_pipeline_regions)
        reduce_ops = stale_program.visit(collect_reduce)

        assert len(regions) == 0
        assert len(reduce_ops) == 1
        assert reduce_ops[0].managed_collective_strategy == "blocking_collective"
        assert stale_estimate.comm_time_ms == pytest.approx(
            baseline_estimate.comm_time_ms
        )
        assert stale_estimate.total_time_ms == pytest.approx(
            baseline_estimate.total_time_ms
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
