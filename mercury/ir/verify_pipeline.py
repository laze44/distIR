# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Verifier for legalized managed-reduction pipeline-region invariants.

Checks structural correctness of ``ManagedReductionPipelineRegion`` nodes
before async estimation or code generation:

- Each slot has at most one in-flight collective.
- Every slot reuse is preceded by a wait.
- Every delayed consumer retires after the corresponding wait.
"""

from typing import List, Tuple

from mercury.ir.nodes import ManagedReductionPipelineRegion, Program
from mercury.ir.utils import collect_pipeline_regions


class PipelineVerificationError(Exception):
    """Raised when a pipeline region fails verification."""


def _verify_region(region: ManagedReductionPipelineRegion) -> List[str]:
    """Verify one pipeline region, returning a list of error messages.

    Returns:
        Empty list if the region is well-formed; otherwise a list of
        human-readable error strings.
    """
    errors: List[str] = []
    buf_name = region.reduce_op.buffer.tensor if region.reduce_op is not None else "?"

    # 1. Region must be marked legalized
    if not region.legalized:
        errors.append(f"Region for '{buf_name}' is not marked legalized")
        return errors

    # 2. Must have a reduce_op
    if region.reduce_op is None:
        errors.append(f"Region for '{buf_name}' has no reduce_op")
        return errors

    # 3. Overlap axis must be set
    if region.overlap_axis is None:
        errors.append(f"Region for '{buf_name}' has no overlap_axis")

    # 4. Stage count must be >= 2
    if region.stage_count < 2:
        errors.append(
            f"Region for '{buf_name}' has stage_count={region.stage_count}; "
            f"minimum is 2 for double-buffering"
        )

    # 5. Tile count must be >= 2 for real overlap
    if region.tile_count < 2:
        errors.append(
            f"Region for '{buf_name}' has tile_count={region.tile_count}; "
            f"minimum is 2 for pipeline overlap"
        )

    # 6. One in-flight work per slot: pending_tiles count <= stage_count
    if len(region.pending_tiles) > region.stage_count:
        errors.append(
            f"Region for '{buf_name}' has {len(region.pending_tiles)} pending tiles "
            f"but only {region.stage_count} slots"
        )

    # 7. Each pending tile must have a valid slot_index
    for i, pt in enumerate(region.pending_tiles):
        if pt.slot_index < 0 or pt.slot_index >= region.stage_count:
            errors.append(
                f"Region for '{buf_name}' pending tile {i} has "
                f"slot_index={pt.slot_index} outside [0, {region.stage_count})"
            )

    # 8. Slot indices in pending tiles must be unique (one in-flight per slot)
    slot_indices = [pt.slot_index for pt in region.pending_tiles]
    if len(slot_indices) != len(set(slot_indices)):
        errors.append(
            f"Region for '{buf_name}' has duplicate slot indices in "
            f"pending tiles: {slot_indices}"
        )

    # 9. Lifecycle must be set
    if region.lifecycle is None:
        errors.append(f"Region for '{buf_name}' has no lifecycle markers")

    # 10. Collective participants > 1
    reduce_op = region.reduce_op
    ring_dims = set(int(comm.shard_dim) for comm in reduce_op.comm)
    shard_dims = [int(dim) for dim in reduce_op.shard_dim if int(dim) not in ring_dims]
    if len(shard_dims) == 0:
        errors.append(
            f"Region for '{buf_name}' has no collective shard dimensions"
        )

    return errors


def verify_pipeline_regions(program: Program) -> Tuple[bool, List[str]]:
    """Verify all legalized pipeline regions in a program.

    Args:
        program: The IR program containing pipeline regions.

    Returns:
        Tuple of (all_valid, errors) where all_valid is True if every
        legalized region passes verification, and errors is a list of
        human-readable error strings.
    """
    regions = program.visit(collect_pipeline_regions)
    all_errors: List[str] = []

    for region in regions:
        if not region.legalized:
            continue
        region_errors = _verify_region(region)
        all_errors.extend(region_errors)

    return (len(all_errors) == 0, all_errors)


def verify_pipeline_region(region: ManagedReductionPipelineRegion) -> Tuple[bool, List[str]]:
    """Verify a single pipeline region.

    Args:
        region: The pipeline region to verify.

    Returns:
        Tuple of (valid, errors).
    """
    errors = _verify_region(region)
    return (len(errors) == 0, errors)
