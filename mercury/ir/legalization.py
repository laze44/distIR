# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Legalization pass for async managed-reduction pipeline regions.

Recognizes eligible ``async_collective_overlap`` candidates and materializes
them into ``ManagedReductionPipelineRegion`` nodes.  Candidates that fail
legalization are downgraded to ``blocking_collective``.
"""

from typing import Dict, List, Optional, Set, Tuple, Union

from mercury.ir.elements import Axis, Buffer
from mercury.ir.nodes import (
    AsyncCollectiveLifecycle,
    BufferLoad,
    BufferStore,
    GridLoop,
    IRNode,
    ManagedReductionPipelineRegion,
    PendingTileDescriptor,
    Program,
    ReduceOp,
)
from mercury.ir.utils import collect_reduce


def _find_consumer_store(
    program: Program,
    reduce_buffer: Buffer,
) -> Optional[BufferStore]:
    """Find a direct ``BufferStore`` that loads from *reduce_buffer* and writes to an output.

    Returns the store node if there is exactly one direct consumer path
    (load from reduce_buffer → store to an output-like buffer), or ``None``
    otherwise.
    """
    loads_from_reduce: List[BufferLoad] = []
    stores: List[BufferStore] = []

    def _collector(node: IRNode):
        if isinstance(node, BufferLoad) and node.buffer.tensor == reduce_buffer.tensor:
            loads_from_reduce.append(node)
        elif isinstance(node, BufferStore) and node.buffer.write:
            stores.append(node)
        return None

    program.visit(_collector)

    if len(loads_from_reduce) != 1:
        return None

    load_target = loads_from_reduce[0].target
    for store in stores:
        if isinstance(store.value, str) and store.value == load_target:
            return store
    return None


def _overlap_axis_has_enough_tiles(overlap_axis: Axis, min_tiles: int = 2) -> bool:
    tile_count = int(overlap_axis.size) // int(overlap_axis.min_block_size)
    return tile_count >= min_tiles


def _has_multiple_participants(
    program: Program,
    reduce_op: ReduceOp,
) -> bool:
    if program.mesh is None:
        return False
    ring_dims = set(int(comm.shard_dim) for comm in reduce_op.comm)
    shard_dims = [
        int(dim)
        for dim in reduce_op.shard_dim
        if int(dim) < len(program.mesh.shape) and int(dim) not in ring_dims
    ]
    if len(shard_dims) == 0:
        return False
    participants = 1
    for dim in shard_dims:
        participants *= int(program.mesh.shape[dim])
    return participants > 1


def _no_same_iteration_consumer(
    program: Program,
    reduce_buffer: Buffer,
    allowed_load_count: int = 1,
) -> bool:
    """Check that the reduce buffer has at most *allowed_load_count* load consumers."""
    count = 0

    def _counter(node: IRNode):
        nonlocal count
        if isinstance(node, BufferLoad) and node.buffer.tensor == reduce_buffer.tensor:
            count += 1
        return None

    program.visit(_counter)
    return count <= allowed_load_count


def _build_pending_tiles(
    overlap_axis: Axis,
    stage_count: int,
    reduce_buffer: Buffer,
    consumer_store: Optional[BufferStore] = None,
) -> List[PendingTileDescriptor]:
    tile_count = int(overlap_axis.size) // int(overlap_axis.min_block_size)
    output_buffer = consumer_store.buffer if consumer_store is not None else None
    retire_indices: Optional[List[Union[int, Axis]]] = None
    if consumer_store is not None:
        retire_indices = [
            idx for idx in consumer_store.indices if isinstance(idx, (int, Axis))
        ]
    descriptors = []
    for slot in range(min(stage_count, tile_count)):
        descriptors.append(
            PendingTileDescriptor(
                slot_index=slot,
                tile_coords=[overlap_axis],
                reduce_buffer=reduce_buffer,
                output_buffer=output_buffer,
                retire_indices=retire_indices,
            )
        )
    return descriptors


def legalize_async_reductions(program: Program) -> List[ManagedReductionPipelineRegion]:
    """Legalize eligible async managed reductions into pipeline regions.

    For each ``ReduceOp`` with ``managed_collective_strategy ==
    'async_collective_overlap'``, check realizability and produce a
    ``ManagedReductionPipelineRegion``.  Returns the list of successfully
    legalized regions.

    Candidates that fail legalization are **not** downgraded here; use
    :func:`fallback_failed_async_candidates` afterwards.

    Args:
        program: The IR program to legalize.

    Returns:
        List of legalized ``ManagedReductionPipelineRegion`` nodes.
    """
    regions: List[ManagedReductionPipelineRegion] = []
    reduce_ops = program.visit(collect_reduce)

    for reduce_op in reduce_ops:
        if reduce_op.managed_collective_strategy != "async_collective_overlap":
            continue

        overlap_axis = reduce_op.async_collective_overlap_axis
        if overlap_axis is None:
            continue

        # Check 1: overlap axis has at least two tiles
        if not _overlap_axis_has_enough_tiles(overlap_axis):
            continue

        # Check 2: collective has more than one participant
        if not _has_multiple_participants(program, reduce_op):
            continue

        # Check 3: the reduction-buffer consumer can be retimed
        consumer_store = _find_consumer_store(program, reduce_op.buffer)
        if consumer_store is None:
            continue

        # Check 4: no additional same-iteration consumer forces an early wait
        if not _no_same_iteration_consumer(program, reduce_op.buffer):
            continue

        tile_count = int(overlap_axis.size) // int(overlap_axis.min_block_size)
        stage_count = max(2, int(reduce_op.async_collective_stage_count))

        pending_tiles = _build_pending_tiles(
            overlap_axis, stage_count, reduce_op.buffer, consumer_store
        )

        lifecycle = reduce_op.async_collective_lifecycle
        if lifecycle is None:
            lifecycle = AsyncCollectiveLifecycle()

        region = ManagedReductionPipelineRegion(
            reduce_op=reduce_op,
            overlap_axis=overlap_axis,
            stage_count=stage_count,
            tile_count=tile_count,
            lifecycle=lifecycle,
            pending_tiles=pending_tiles,
            consumer_store=consumer_store,
            legalized=True,
        )
        regions.append(region)

    return regions


def prepare_pipeline(program: Program) -> List[ManagedReductionPipelineRegion]:
    """Shared pipeline-preparation step for estimation and code generation.

    Runs managed-reduction legalization, verifier checks, and blocking
    fallback in a single deterministic pass.  The resulting program has
    legalized ``ManagedReductionPipelineRegion`` nodes inserted into
    ``program.body`` so that downstream visitors (estimator, codegen) can
    discover them via ``collect_pipeline_regions``.  Candidates that fail
    legalization or verification are downgraded to ``blocking_collective``.

    This function is idempotent: calling it on a program that already
    contains pipeline regions is safe (existing regions are preserved
    and no duplicates are added).

    Args:
        program: The IR program to prepare.

    Returns:
        List of successfully legalized and verified pipeline regions.
    """
    from mercury.ir.utils import collect_pipeline_regions
    from mercury.ir.verify_pipeline import verify_pipeline_region

    existing = program.visit(collect_pipeline_regions)
    existing_legalized = [r for r in existing if r.legalized]
    if existing_legalized:
        return existing_legalized

    regions = legalize_async_reductions(program)

    for region in regions:
        program.body.append(region)

    verified_regions: List[ManagedReductionPipelineRegion] = []
    for region in regions:
        valid, errors = verify_pipeline_region(region)
        if valid:
            verified_regions.append(region)
        else:
            region.legalized = False
            if region in program.body:
                program.body.remove(region)

    fallback_failed_async_candidates(program, verified_regions)

    return verified_regions


def fallback_failed_async_candidates(
    program: Program,
    legalized_regions: List[ManagedReductionPipelineRegion],
) -> int:
    """Downgrade non-legalized async candidates to ``blocking_collective``.

    Any ``ReduceOp`` with ``managed_collective_strategy ==
    'async_collective_overlap'`` that does **not** appear in
    *legalized_regions* is rewritten to ``blocking_collective``.

    Args:
        program: The IR program.
        legalized_regions: Regions that passed legalization.

    Returns:
        The number of candidates that were downgraded.
    """
    legalized_reduce_ids = set(id(r.reduce_op) for r in legalized_regions)
    reduce_ops = program.visit(collect_reduce)
    downgraded = 0

    for reduce_op in reduce_ops:
        if reduce_op.managed_collective_strategy != "async_collective_overlap":
            continue
        if id(reduce_op) in legalized_reduce_ids:
            continue

        reduce_op.managed_collective_strategy = "blocking_collective"
        reduce_op.async_collective_overlap_axis = None
        reduce_op.async_collective_tile_count = 1
        reduce_op.async_collective_stage_count = 1
        reduce_op.async_collective_lifecycle = None
        downgraded += 1

    return downgraded
