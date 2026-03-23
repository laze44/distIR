# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Legalization pass for async managed-reduction pipeline regions.

Recognizes eligible ``async_collective_overlap`` candidates and materializes
them into ``ManagedReductionPipelineRegion`` nodes.  Candidates that fail
legalization are downgraded to ``blocking_collective``.
"""

import ast
from dataclasses import dataclass
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
    PyNode,
    ReduceOp,
)
from mercury.ir.utils import collect_reduce


@dataclass
class ConsumerMatch:
    """Result of consumer-store matching for a reduce buffer.

    Carries enough information for the pipeline lowering to retire completed
    tiles, including any epilogue expression (e.g. ``.to(c.dtype)``) that
    must be replayed at retire time.

    Args:
        consumer_load: The ``BufferLoad`` that reads from the reduce buffer.
        consumer_store: The ``BufferStore`` that writes to the output buffer.
        consumer_source_name: The base variable name that links load to store
            (e.g. ``"reduce_res"``).
        consumer_epilogue: If the store value wraps the load target in an
            expression (e.g. ``reduce_res.to(c.dtype)``), this holds the
            full ``PyNode`` so codegen can replay it at retire time.
            ``None`` when the store value is a bare variable reference.
    """

    consumer_load: BufferLoad
    consumer_store: BufferStore
    consumer_source_name: str
    consumer_epilogue: Optional[PyNode] = None


def _extract_consumer_source_name(value: Union[PyNode, str]) -> Optional[str]:
    """Extract the base variable name from a ``BufferStore.value``.

    Handles three forms:
    - Bare string: ``"reduce_res"`` → ``"reduce_res"``
    - ``PyNode`` wrapping a bare ``ast.Name``: ``PyNode(reduce_res)`` → ``"reduce_res"``
    - ``PyNode`` wrapping a method call on a name:
      ``PyNode(reduce_res.to(c.dtype))`` → ``"reduce_res"``

    Returns:
        The extracted base variable name, or ``None`` if the expression
        structure is not recognized.
    """
    if isinstance(value, str):
        return value

    if not isinstance(value, PyNode):
        return None

    node = value.node

    if isinstance(node, ast.Name):
        return node.id

    # Method call on a Name, e.g. reduce_res.to(c.dtype)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        receiver = node.func.value
        if isinstance(receiver, ast.Name):
            return receiver.id

    return None


def _extract_consumer_epilogue(
    value: Union[PyNode, str],
    load_target: str,
) -> Optional[PyNode]:
    """Extract the epilogue expression from a ``BufferStore.value``, if any.

    If *value* wraps *load_target* inside a method-call expression
    (e.g. ``reduce_res.to(c.dtype)``), returns the full ``PyNode`` as the
    epilogue that must be replayed at retire time.  Returns ``None`` when
    the value is a bare variable reference (no epilogue needed).

    Args:
        value: The ``BufferStore.value`` to inspect.
        load_target: The expected base variable name from the ``BufferLoad``.

    Returns:
        The epilogue ``PyNode``, or ``None`` if no epilogue is present.
    """
    if isinstance(value, str):
        return None

    if not isinstance(value, PyNode):
        return None

    node = value.node

    if isinstance(node, ast.Name) and node.id == load_target:
        return None

    # Method call wrapping load_target, e.g. reduce_res.to(c.dtype)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        receiver = node.func.value
        if isinstance(receiver, ast.Name) and receiver.id == load_target:
            return value

    return None


def _find_consumer_store(
    program: Program,
    reduce_buffer: Buffer,
) -> Optional[ConsumerMatch]:
    """Find the direct consumer path from *reduce_buffer* to an output store.

    Scans the program for exactly one ``BufferLoad`` from *reduce_buffer* and
    a matching ``BufferStore`` whose value references the load target — either
    directly (bare string) or through a simple wrapper expression such as
    ``reduce_res.to(c.dtype)``.

    Returns:
        A ``ConsumerMatch`` with consumer metadata, or ``None`` if no unique
        consumer path is found.
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

    consumer_load = loads_from_reduce[0]
    load_target = consumer_load.target
    # load_target may be a str ("reduce_res") or a PyNode; normalize to str
    if isinstance(load_target, str):
        load_target_name = load_target
    elif isinstance(load_target, PyNode) and isinstance(load_target.node, ast.Name):
        load_target_name = load_target.node.id
    else:
        return None

    for store in stores:
        source_name = _extract_consumer_source_name(store.value)
        if source_name is not None and source_name == load_target_name:
            epilogue = _extract_consumer_epilogue(store.value, load_target_name)
            return ConsumerMatch(
                consumer_load=consumer_load,
                consumer_store=store,
                consumer_source_name=source_name,
                consumer_epilogue=epilogue,
            )
    return None


def _overlap_axis_has_enough_tiles(overlap_axis: Axis, min_tiles: int = 2) -> bool:
    tile_count = int(overlap_axis.size) // int(overlap_axis.min_block_size)
    return tile_count >= min_tiles


def _overlap_axis_materializes_as_runtime_loop(overlap_axis: Axis) -> bool:
    """Check whether *overlap_axis* will produce a real ``for`` loop in codegen.

    Codegen emits a loop only when ``axis.size > axis.min_block_size``.
    An axis where ``size == min_block_size`` becomes a single static slice
    and cannot drive slot rotation.
    """
    return int(overlap_axis.size) > int(overlap_axis.min_block_size)


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
    # Build a stable tile-id expression for codegen retire-time reconstruction
    block_size = int(overlap_axis.min_block_size)
    if block_size > 1:
        tile_id_expr = f"{overlap_axis.name} // {block_size}"
    else:
        tile_id_expr = overlap_axis.name
    descriptors = []
    for slot in range(min(stage_count, tile_count)):
        descriptors.append(
            PendingTileDescriptor(
                slot_index=slot,
                tile_coords=[overlap_axis],
                reduce_buffer=reduce_buffer,
                output_buffer=output_buffer,
                retire_indices=retire_indices,
                tile_id_expr=tile_id_expr,
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

        # Check 1b: overlap axis materializes as a real runtime loop in codegen
        if not _overlap_axis_materializes_as_runtime_loop(overlap_axis):
            continue

        # Check 2: collective has more than one participant
        if not _has_multiple_participants(program, reduce_op):
            continue

        # Check 3: the reduction-buffer consumer can be retimed
        consumer_match = _find_consumer_store(program, reduce_op.buffer)
        if consumer_match is None:
            continue

        # Check 4: no additional same-iteration consumer forces an early wait
        if not _no_same_iteration_consumer(program, reduce_op.buffer):
            continue

        tile_count = int(overlap_axis.size) // int(overlap_axis.min_block_size)
        stage_count = max(2, int(reduce_op.async_collective_stage_count))

        pending_tiles = _build_pending_tiles(
            overlap_axis, stage_count, reduce_op.buffer, consumer_match.consumer_store
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
            consumer_store=consumer_match.consumer_store,
            consumer_load=consumer_match.consumer_load,
            consumer_epilogue=consumer_match.consumer_epilogue,
            retire_target_buffer=consumer_match.consumer_store.buffer,
            retire_target_indices=[
                idx
                for idx in consumer_match.consumer_store.indices
                if isinstance(idx, (int, Axis))
            ],
            legalized=True,
            materialized_overlap_axis=overlap_axis,
            pipeline_scope_axis=overlap_axis.name,
        )
        regions.append(region)

    return regions


def _replace_legalized_nodes_in_loop(
    program: Program,
    region: ManagedReductionPipelineRegion,
) -> bool:
    """Replace the original ReduceOp + BufferLoad + BufferStore in the GridLoop body with *region*.

    Finds the GridLoop containing *region.reduce_op* and replaces the
    contiguous subsequence [ReduceOp, BufferLoad, BufferStore] with the
    single ``ManagedReductionPipelineRegion`` node.

    Returns ``True`` if the replacement was successful.
    """
    target_reduce_id = id(region.reduce_op)
    target_load_id = (
        id(region.consumer_load) if region.consumer_load is not None else None
    )
    target_store_id = (
        id(region.consumer_store) if region.consumer_store is not None else None
    )

    grid_loops: List[GridLoop] = []

    def _find_grids(node: IRNode):
        if isinstance(node, GridLoop):
            grid_loops.append(node)
        return None

    program.visit(_find_grids)

    for grid_loop in grid_loops:
        ids_to_remove = set()
        reduce_found = False
        for i, child in enumerate(grid_loop.body):
            if id(child) == target_reduce_id:
                reduce_found = True
                ids_to_remove.add(id(child))
            elif target_load_id is not None and id(child) == target_load_id:
                ids_to_remove.add(id(child))
            elif target_store_id is not None and id(child) == target_store_id:
                ids_to_remove.add(id(child))

        if not reduce_found:
            continue

        new_body: List[IRNode] = []
        region_inserted = False
        for child in grid_loop.body:
            if id(child) in ids_to_remove:
                if not region_inserted:
                    new_body.append(region)
                    region_inserted = True
            else:
                new_body.append(child)

        grid_loop.body = new_body
        return True

    return False


def _restore_invalidated_region_in_loop(
    program: Program,
    region: ManagedReductionPipelineRegion,
) -> bool:
    """Restore a stale pipeline region back to its original loop-body nodes."""
    target_region_id = id(region)

    grid_loops: List[GridLoop] = []

    def _find_grids(node: IRNode):
        if isinstance(node, GridLoop):
            grid_loops.append(node)
        return None

    program.visit(_find_grids)

    restored_children: List[IRNode] = []
    if region.reduce_op is not None:
        restored_children.append(region.reduce_op)
    if region.consumer_load is not None:
        restored_children.append(region.consumer_load)
    if region.consumer_store is not None:
        restored_children.append(region.consumer_store)

    # Collect ids of consumer nodes that will be restored into the region's
    # GridLoop.  These same nodes may also exist as stale references in OTHER
    # GridLoop bodies (the outer loop kept them when _replace_legalized_nodes_
    # in_loop only operated on the inner loop that contained the ReduceOp).
    consumer_ids_to_dedupe: Set[int] = set()
    if region.consumer_load is not None:
        consumer_ids_to_dedupe.add(id(region.consumer_load))
    if region.consumer_store is not None:
        consumer_ids_to_dedupe.add(id(region.consumer_store))

    restored_in_loop = None
    for grid_loop in grid_loops:
        new_body: List[IRNode] = []
        restored = False
        for child in grid_loop.body:
            if id(child) == target_region_id:
                new_body.extend(restored_children)
                restored = True
            else:
                new_body.append(child)
        if restored:
            grid_loop.body = new_body
            restored_in_loop = grid_loop
            break

    if restored_in_loop is None:
        return False

    # Remove stale consumer references from OTHER GridLoop bodies to prevent
    # duplicate code emission when codegen visits both the inner and outer loop.
    if consumer_ids_to_dedupe:
        for grid_loop in grid_loops:
            if grid_loop is restored_in_loop:
                continue
            grid_loop.body = [
                child for child in grid_loop.body
                if id(child) not in consumer_ids_to_dedupe
            ]

    return True


def _downgrade_async_reduce_op(reduce_op: Optional[ReduceOp]) -> bool:
    """Rewrite one async-overlap reduce op to blocking mode."""
    if reduce_op is None:
        return False
    if reduce_op.managed_collective_strategy != "async_collective_overlap":
        return False

    reduce_op.managed_collective_strategy = "blocking_collective"
    reduce_op.async_collective_overlap_axis = None
    reduce_op.async_collective_tile_count = 1
    reduce_op.async_collective_stage_count = 1
    reduce_op.async_collective_lifecycle = None
    return True


def _revalidate_existing_regions(
    program: Program,
    existing_legalized: List[ManagedReductionPipelineRegion],
) -> List[ManagedReductionPipelineRegion]:
    """Re-check previously legalized regions after IR mutation.

    ``eliminate_loops()`` mutates shared ``Axis`` objects in place, so a region
    that was valid during an earlier legalization pass may later lose the
    runtime loop needed for slot rotation.  Revalidate those regions before
    reusing them via the idempotency guard in ``prepare_pipeline()``.
    """
    still_valid: List[ManagedReductionPipelineRegion] = []
    invalidated = False

    for region in existing_legalized:
        check_axis = (
            region.materialized_overlap_axis
            if region.materialized_overlap_axis is not None
            else region.overlap_axis
        )
        if (
            check_axis is not None
            and _overlap_axis_materializes_as_runtime_loop(check_axis)
        ):
            still_valid.append(region)
            continue

        region.legalized = False
        _downgrade_async_reduce_op(region.reduce_op)
        _restore_invalidated_region_in_loop(program, region)
        invalidated = True

    if invalidated:
        fallback_failed_async_candidates(program, still_valid)

    return still_valid


def prepare_pipeline(program: Program) -> List[ManagedReductionPipelineRegion]:
    """Shared pipeline-preparation step for estimation and code generation.

    Runs managed-reduction legalization, verifier checks, and blocking
    fallback in a single deterministic pass.  Legalized regions replace
    the original ``ReduceOp + BufferLoad + BufferStore`` subsequence inside
    ``GridLoop.body`` so that codegen visits the region instead of the
    original nodes.  Candidates that fail legalization or verification are
    downgraded to ``blocking_collective``.

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
        return _revalidate_existing_regions(program, existing_legalized)

    regions = legalize_async_reductions(program)

    verified_regions: List[ManagedReductionPipelineRegion] = []
    for region in regions:
        valid, errors = verify_pipeline_region(region)
        if valid:
            replaced = _replace_legalized_nodes_in_loop(program, region)
            if replaced:
                verified_regions.append(region)
            else:
                region.legalized = False
        else:
            region.legalized = False

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
        if id(reduce_op) in legalized_reduce_ids:
            continue
        if _downgrade_async_reduce_op(reduce_op):
            downgraded += 1

    return downgraded
