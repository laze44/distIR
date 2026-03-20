# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import copy
import itertools
from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Optional, Tuple
from tqdm import tqdm

from mercury.ir.distributed import DeviceMesh
from mercury.ir.elements import Axis
from mercury.ir.init_distributed import init_distributed
from mercury.ir.nodes import AsyncCollectiveLifecycle, Program
from mercury.ir.primitives import parallelize, shift
from mercury.ir.tile import tile_loop
from mercury.ir.utils import (
    collect_axis,
    collect_loops,
    collect_parallelizeable_axes,
    collect_reduce,
    get_buffers,
    get_element_size,
)
from mercury.search.mapping_constraints import (
    TensorMappingConstraints,
    program_satisfies_tensor_mapping_constraints,
)
from mercury.search.topology_policy import FlatMeshShapePolicy, MeshShapePolicy


MAX_AXIS_TILE_FACTOR = 16
DEFAULT_ASYNC_COLLECTIVE_STAGE_COUNT = 2
DEFAULT_ASYNC_COLLECTIVE_MEMORY_BUDGET_BYTES = 80 * (2**30)


def _linear_to_coords(rank: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    coords = [0] * len(shape)
    remainder = rank
    for dim_id in range(len(shape) - 1, -1, -1):
        dim = shape[dim_id]
        coords[dim_id] = remainder % dim
        remainder //= dim
    return tuple(coords)


def _infer_topology_metadata(
    origin_mesh: DeviceMesh, mesh: DeviceMesh
) -> Dict[str, List[int]]:
    """Infer inter/intra-node mesh dimensions after reshaping."""
    ndim = len(mesh.shape)
    if len(origin_mesh.shape) <= 1:
        return {
            "inter_node_dims": [],
            "intra_node_dims": list(range(ndim)),
            "mixed_dims": [],
        }

    inter_node_dims: List[int] = []
    intra_node_dims: List[int] = []
    mixed_dims: List[int] = []

    for dim in range(ndim):
        groups: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = defaultdict(list)
        for coords in itertools.product(*[range(v) for v in mesh.shape]):
            key = tuple(coords[idx] for idx in range(ndim) if idx != dim)
            groups[key].append(coords)

        inter_changes = False
        intra_changes = False
        for grouped_coords in groups.values():
            inter_vals = set()
            intra_vals = set()
            for coords in grouped_coords:
                rank = mesh.get_device(coords)
                origin_coords = _linear_to_coords(rank, origin_mesh.shape)
                inter_vals.add(origin_coords[0])
                intra_vals.add(origin_coords[1:])
            inter_changes = inter_changes or len(inter_vals) > 1
            intra_changes = intra_changes or len(intra_vals) > 1

        if inter_changes and not intra_changes:
            inter_node_dims.append(dim)
        elif intra_changes and not inter_changes:
            intra_node_dims.append(dim)
        elif inter_changes and intra_changes:
            mixed_dims.append(dim)
        else:
            intra_node_dims.append(dim)

    return {
        "inter_node_dims": inter_node_dims,
        "intra_node_dims": intra_node_dims,
        "mixed_dims": mixed_dims,
    }


def enumerate_mesh_shapes(
    mesh_size: int, max_dim: int, current_shape: List[int] = None, remaining: int = None
) -> Iterator[Tuple[int, ...]]:
    """
    enumerate all possible device mesh shapes.
    Parameters
    ----------
    mesh_size : int
        The size of the original 1D mesh
    max_dim : int
        The maximum number of dimensions allowed
    current_shape : List[int], optional
        The current shape being constructed, used for recursion
    remaining : int, optional
        The remaining size to be allocated, used for recursion
    Yields
    -------
    Tuple[int, ...]
        A possible mesh shape, e.g., (4,), (2,2), (2,2,2), etc.
    """
    if current_shape is None:
        current_shape = []
        remaining = mesh_size

    # base case: if the current shape has reached max_dim but there is still remaining size, return
    if len(current_shape) >= max_dim and remaining > 1:
        return

    # if no remaining size to allocate, yield the current shape
    if remaining == 1:
        yield tuple(current_shape)
        return
    elif remaining < 1:
        return

    # try all possible factors
    for factor in range(2, remaining + 1):
        if remaining % factor == 0:
            # for each factor, recursively enumerate the remaining size
            for shape in enumerate_mesh_shapes(
                mesh_size, max_dim, current_shape + [factor], remaining // factor
            ):
                yield shape


def enumerate_mesh_assignment(
    ndim: int, axes_num: int
) -> Iterator[List[Tuple[int, int]]]:
    """
    enumerate all possible mesh dimension assignment for a given number of axes and mesh dimensions.
    each axis must be assigned to a continuous range of dimensions, and all dimensions must be assigned.
    For example, for a 2D mesh and 2 axes, the possible assignments include:
    - axis1: [0,1], axis2: []
    - axis1: [], axis2: [0,1]
    - axis1: [0], axis2: [1]
    - axis1: [1], axis2: [0]
    Parameters
    ----------
    ndim : int
        The total number of dimensions in the mesh
    axes_num : int
        The number of axes to be assigned
    Yields
    -------
    List[Tuple[int, int]]
        A possible assignment of axes to dimensions, where each tuple represents (start_dim, num_dims)
    """

    def _recursive_assign(
        remaining_dims: List[int],
        axis_idx: int,
        current_assignment: List[Tuple[int, int]],
    ) -> Iterator[List[Tuple[int, int]]]:
        # base case: if all axes have been assigned
        if axis_idx == axes_num:
            # only if all dimensions are assigned, it is a valid assignment
            if not remaining_dims:
                yield copy.deepcopy(current_assignment)
            return

        # try to assign no dimensions to the current axis
        yield from _recursive_assign(
            remaining_dims, axis_idx + 1, current_assignment + [(0, 0)]
        )

        # try to assign a continuous range of dimensions to the current axis
        for start in range(len(remaining_dims)):
            for length in range(1, len(remaining_dims) - start + 1):
                # get the dimensions to be assigned
                assigned_dims = remaining_dims[start : start + length]
                # check if the assigned dimensions are continuous
                if assigned_dims != list(
                    range(assigned_dims[0], assigned_dims[-1] + 1)
                ):
                    continue

                remaining = remaining_dims[:start] + remaining_dims[start + length :]

                yield from _recursive_assign(
                    remaining,
                    axis_idx + 1,
                    current_assignment + [(assigned_dims[0], length)],
                )

    yield from _recursive_assign(list(range(ndim)), 0, [])


def enumerate_axis_split(
    axes: List[Axis], res_cards: int, cur_split: List[int]
) -> Iterator[List[int]]:
    """Enumerate all valid split factors for target axes.

    The split candidates are independent from ``res_cards`` and are constrained by:
    1) exact divisibility, 2) minimum block size, and 3) a maximum tile-factor cap.

    For GEMM small axes (size < 32 and min_block_size == size), only factor 1
    (unsplit) and factor 2 (one binary split, if even) are allowed.
    """
    if len(cur_split) == len(axes):
        yield copy.deepcopy(cur_split)
        return

    cur_axis = axes[len(cur_split)]

    # GEMM small-axis path: whole-axis tile with at most one binary split
    if cur_axis.size < 32 and cur_axis.min_block_size == cur_axis.size:
        # Always allow unsplit
        yield from enumerate_axis_split(axes, res_cards, cur_split + [1])
        # Allow one binary split only for even dimensions
        if cur_axis.size % 2 == 0 and cur_axis.size >= 2:
            yield from enumerate_axis_split(axes, res_cards, cur_split + [2])
        return

    max_split_factor = min(MAX_AXIS_TILE_FACTOR, cur_axis.size)
    for factor in range(1, max_split_factor + 1):
        if (
            cur_axis.size % factor == 0
            and cur_axis.size // factor >= cur_axis.min_block_size
        ):
            yield from enumerate_axis_split(axes, res_cards, cur_split + [factor])


def _collective_shard_dims(reduce_op, mesh_ndim: int) -> List[int]:
    ring_dims = set(int(comm.shard_dim) for comm in reduce_op.comm)
    shard_dims = sorted(
        set(
            int(dim)
            for dim in reduce_op.shard_dim
            if 0 <= int(dim) < mesh_ndim and int(dim) not in ring_dims
        )
    )
    return shard_dims


# ---------------------------------------------------------------------------
# Logical-factor enumeration helpers for flat mesh policies
# ---------------------------------------------------------------------------


def _valid_axis_factors(axis: Axis, domain_size: int) -> List[int]:
    """Return all valid shard factors for *axis* within a domain of *domain_size*.

    A factor ``f`` is valid when:
    - ``f`` divides ``axis.size``
    - ``axis.size / f >= axis.min_block_size``
    - ``f`` divides ``domain_size``
    - ``f <= MAX_AXIS_TILE_FACTOR`` (unless ``f == axis.size``)

    For small axes (size < 32 and min_block_size == size), only factor 1
    and factor 2 (if even) are allowed, matching ``enumerate_axis_split``.
    """
    factors: List[int] = [1]

    if axis.size < 32 and axis.min_block_size == axis.size:
        if axis.size % 2 == 0 and axis.size >= 2 and domain_size % 2 == 0:
            factors.append(2)
        return factors

    max_factor = min(MAX_AXIS_TILE_FACTOR, axis.size)
    for f in range(2, max_factor + 1):
        if (
            axis.size % f == 0
            and axis.size // f >= axis.min_block_size
            and domain_size % f == 0
        ):
            factors.append(f)
    return factors


def _enumerate_logical_factor_assignments(
    axes: List[Axis],
    domain_size: int,
) -> Iterator[List[int]]:
    """Enumerate per-axis logical shard factor combinations for a domain.

    Yields lists of length ``len(axes)`` where ``factors[i]`` is the shard
    factor assigned to ``axes[i]``.  The product of all factors must divide
    ``domain_size``.

    This replaces the role of ``enumerate_mesh_assignment`` in the flat
    search path: instead of assigning contiguous mesh dim ranges to axes,
    each axis independently chooses a shard factor.
    """
    per_axis_factors = [_valid_axis_factors(ax, domain_size) for ax in axes]

    for combo in itertools.product(*per_axis_factors):
        product = 1
        for f in combo:
            product *= f
        if domain_size % product == 0:
            yield list(combo)


def _build_virtual_mesh_shape(
    factor_list: List[int],
) -> Tuple[Tuple[int, ...], List[Tuple[int, int]]]:
    """Build a virtual mesh shape and axis assignment from factor list.

    Given a factor list (one per axis), constructs a virtual mesh shape
    containing only the non-1 factors, and returns the corresponding
    mesh assignment list for ``parallelize`` calls.

    Returns:
        (virtual_mesh_shape, assignment_list) where assignment_list has
        one ``(start_dim, num_dims)`` entry per axis.
    """
    virtual_dims: List[int] = []
    assign: List[Tuple[int, int]] = []
    dim_offset = 0

    for factor in factor_list:
        if factor > 1:
            virtual_dims.append(factor)
            assign.append((dim_offset, 1))
            dim_offset += 1
        else:
            assign.append((0, 0))

    return tuple(virtual_dims) if virtual_dims else (1,), assign


def _search_flat_path(
    splited_program: Program,
    origin_mesh: DeviceMesh,
    all_axes: List[Axis],
    flat_policy: "FlatMeshShapePolicy",
    _set_metadata_and_match: "Callable",
    _should_yield: "Callable",
) -> Iterator[Program]:
    """Flat search path: enumerate logical shard factors per domain.

    For each domain with size > 1, independently enumerate shard factor
    combinations for all axes.  For each combination, construct a virtual
    mesh and run the standard parallelize + shift flow.
    """
    topology = flat_policy.topology

    # Collect domain sizes and labels for domains with size > 1
    active_domains: List[Tuple[str, int]] = []
    for domain, label in zip(topology.domains, topology.domain_labels):
        if domain.size > 1:
            active_domains.append((label, domain.size))

    if len(active_domains) == 0:
        # Single-device case: yield just the base program
        program = copy.deepcopy(splited_program)
        mesh = origin_mesh.reshape((1,))
        init_distributed(program, mesh)
        if _set_metadata_and_match(program, mesh, (1,)):
            if _should_yield(program):
                yield program
        return

    # For each domain, enumerate factor assignments over all axes.
    # For multi-domain case, the virtual mesh is the concatenation of
    # per-domain virtual dims.
    per_domain_factor_lists: List[List[List[int]]] = []
    for _label, domain_size in active_domains:
        domain_factors = list(
            _enumerate_logical_factor_assignments(all_axes, domain_size)
        )
        per_domain_factor_lists.append(domain_factors)

    for domain_combo in itertools.product(*per_domain_factor_lists):
        # domain_combo is a tuple of factor lists, one per active domain
        # Check: no axis is sharded more than once across domains on the
        # *same tensor dimension* — but since each axis maps to exactly
        # one tensor dimension, having factor > 1 on the same axis in
        # multiple domains is fine (it compounds).  However, the total
        # factor per axis across all domains must still allow valid
        # parallelize calls.

        # Build virtual mesh shape: concatenate per-domain virtual dims
        virtual_dims: List[int] = []
        per_axis_assign: List[List[Tuple[int, int]]] = [
            [] for _ in range(len(all_axes))
        ]
        dim_offset = 0

        for domain_idx, factor_list in enumerate(domain_combo):
            for axis_idx, factor in enumerate(factor_list):
                if factor > 1:
                    virtual_dims.append(factor)
                    per_axis_assign[axis_idx].append((dim_offset, 1))
                    dim_offset += 1

        if len(virtual_dims) == 0:
            virtual_mesh_shape = (1,)
        else:
            virtual_mesh_shape = tuple(virtual_dims)

        # Check that virtual mesh product divides total devices
        vm_product = 1
        for v in virtual_mesh_shape:
            vm_product *= v
        if vm_product > len(origin_mesh.devices):
            continue

        # Reshape origin mesh to virtual shape
        # Need to pad with 1s if vm_product < total devices
        # Actually, the virtual mesh must have the same number of devices.
        # If some devices are not used, we need to handle that.
        # For now, only yield when vm_product == total devices.
        # (Partial use = devices would be replicated, but that's handled
        #  by the standard search path already.)
        if vm_product != len(origin_mesh.devices):
            continue

        mesh = origin_mesh.reshape(virtual_mesh_shape)

        program_shape = copy.deepcopy(splited_program)
        init_distributed(program_shape, mesh)

        # Build flattened assignment: for each axis, merge dims from all domains
        flat_assign: List[Tuple[int, int]] = []
        for axis_idx in range(len(all_axes)):
            domain_assigns = per_axis_assign[axis_idx]
            if len(domain_assigns) == 0:
                flat_assign.append((0, 0))
            elif len(domain_assigns) == 1:
                flat_assign.append(domain_assigns[0])
            else:
                # Multiple domains contribute to this axis.
                # They should form a contiguous range in the virtual mesh.
                all_start_dims = [a[0] for a in domain_assigns]
                min_dim = min(all_start_dims)
                max_dim_end = max(a[0] + a[1] for a in domain_assigns)
                flat_assign.append((min_dim, max_dim_end - min_dim))

        # Run parallelize + shift flow (same as standard path)
        program = copy.deepcopy(program_shape)
        seed_program = copy.deepcopy(program)

        loops = program.visit(collect_loops)
        axes_list_inner = program.visit(collect_parallelizeable_axes)

        all_axes_inner = program.visit(collect_axis)
        axes_names = set(axis.name for axis in all_axes_inner)
        ringable_axes = set()

        # collect the axes to be parallelized
        parallelize_axes = []
        cnt = 0
        for loop, axes in zip(loops, axes_list_inner):
            for axis in axes:
                if flat_assign[cnt][1] != 0:
                    parallelize_axes.append(axis.name)
                cnt += 1

        cnt = 0
        succ = True
        for loop, axes in zip(loops, axes_list_inner):
            for axis in axes:
                usd_axes = set(parallelize_axes)
                succ &= parallelize(
                    program,
                    loop,
                    axis,
                    mesh,
                    flat_assign[cnt][0],
                    flat_assign[cnt][0] + flat_assign[cnt][1],
                    usd_axes,
                )
                if not succ:
                    break
                if flat_assign[cnt][1] != 0:
                    ringable_axes |= axes_names - usd_axes
                shift(
                    program,
                    axis,
                    mesh,
                    flat_assign[cnt][0],
                    flat_assign[cnt][0] + flat_assign[cnt][1],
                    1,
                    usd_axes,
                )
                cnt += 1
            if not succ:
                break

        if succ:
            for variant in _enumerate_collective_strategy_variants(program):
                if _set_metadata_and_match(
                    variant, mesh, virtual_mesh_shape
                ) and _should_yield(variant):
                    yield variant

            # enumerate all subsets of ringable axes
            for i in range(1, len(ringable_axes) + 1):
                for ringable_axes_subset in itertools.combinations(
                    ringable_axes, i
                ):
                    program = copy.deepcopy(seed_program)
                    loops = program.visit(collect_loops)
                    axes_list_inner = program.visit(collect_parallelizeable_axes)
                    cnt = 0
                    for loop, axes in zip(loops, axes_list_inner):
                        for axis in axes:
                            usd_axes = set(ringable_axes_subset)
                            usd_axes.update(parallelize_axes)
                            succ = parallelize(
                                program,
                                loop,
                                axis,
                                mesh,
                                flat_assign[cnt][0],
                                flat_assign[cnt][0] + flat_assign[cnt][1],
                                usd_axes,
                            )
                            assert succ is True, (
                                "should be able to parallelize as we have "
                                "checked before"
                            )
                            shift(
                                program,
                                axis,
                                mesh,
                                flat_assign[cnt][0],
                                flat_assign[cnt][0] + flat_assign[cnt][1],
                                1,
                                usd_axes,
                            )
                            cnt += 1
                    for variant in _enumerate_collective_strategy_variants(
                        program
                    ):
                        if _set_metadata_and_match(
                            variant, mesh, virtual_mesh_shape
                        ) and _should_yield(variant):
                            yield variant


def _collective_participants(mesh_shape: Tuple[int, ...], shard_dims: List[int]) -> int:
    participants = 1
    for dim in shard_dims:
        participants *= int(mesh_shape[int(dim)])
    return participants


def _buffer_bytes(buffer) -> float:
    numel = 1
    for dim in buffer.get_shape():
        numel *= int(dim)
    return float(numel * get_element_size(buffer.dtype))


def _layout_allows_async_overlap(
    reduce_op, overlap_axis: Axis, collective_shard_dims: List[int]
) -> bool:
    if not reduce_op.buffer.has_axis(overlap_axis):
        return False
    if reduce_op.buffer.shard_spec is None:
        return False
    dim_id, _ = reduce_op.buffer.get_axis(overlap_axis)
    spec = reduce_op.buffer.shard_spec.specs[dim_id]
    if not isinstance(spec, tuple):
        return True
    overlap_mesh_dims = set(int(dim) for dim in spec[1])
    return (
        len(
            overlap_mesh_dims.intersection(
                set(int(dim) for dim in collective_shard_dims)
            )
        )
        == 0
    )


def _eligible_async_overlap_axes(
    program: Program, reduce_op, collective_shard_dims: List[int]
) -> List[Axis]:
    if reduce_op.indices is None:
        return []

    reduce_axis_names = set(axis.name for axis in reduce_op.axes)
    seen: set = set()
    eligible_axes: List[Axis] = []
    for index in reduce_op.indices:
        if not isinstance(index, Axis):
            continue
        if index.name in reduce_axis_names:
            continue
        if index.name in seen:
            continue
        seen.add(index.name)
        tile_count = int(index.size) // int(index.min_block_size)
        if tile_count < 2:
            continue
        if not _layout_allows_async_overlap(reduce_op, index, collective_shard_dims):
            continue
        eligible_axes.append(index)
    return eligible_axes


def _async_memory_budget_bytes(program: Program) -> float:
    budget = getattr(
        program,
        "async_collective_memory_budget_bytes",
        DEFAULT_ASYNC_COLLECTIVE_MEMORY_BUDGET_BYTES,
    )
    try:
        budget_value = float(budget)
    except (TypeError, ValueError):
        return float(DEFAULT_ASYNC_COLLECTIVE_MEMORY_BUDGET_BYTES)
    if budget_value <= 0:
        return 0.0
    return budget_value


def _annotate_default_collective_strategy(program: Program) -> None:
    if program.mesh is None:
        return
    mesh_ndim = len(program.mesh.shape)
    for reduce_op in program.visit(collect_reduce):
        shard_dims = _collective_shard_dims(reduce_op, mesh_ndim)
        if len(shard_dims) > 0 and len(reduce_op.comm) == 0:
            reduce_op.managed_collective_strategy = "blocking_collective"
        elif len(reduce_op.comm) > 0:
            reduce_op.managed_collective_strategy = "ring_overlap"
        else:
            reduce_op.managed_collective_strategy = "blocking_collective"
        reduce_op.async_collective_overlap_axis = None
        reduce_op.async_collective_tile_count = 1
        reduce_op.async_collective_stage_count = 1
        reduce_op.async_collective_lifecycle = None


def _enumerate_collective_strategy_variants(program: Program) -> List[Program]:
    if program.mesh is None:
        return [program]

    _annotate_default_collective_strategy(program)
    variants: List[Program] = [program]
    mesh_ndim = len(program.mesh.shape)
    memory_budget_bytes = _async_memory_budget_bytes(program)
    reduce_ops = program.visit(collect_reduce)

    for reduce_id, reduce_op in enumerate(reduce_ops):
        shard_dims = _collective_shard_dims(reduce_op, mesh_ndim)
        if len(shard_dims) == 0:
            continue
        participants = _collective_participants(program.mesh.shape, shard_dims)
        if participants <= 1:
            continue

        stage_count = DEFAULT_ASYNC_COLLECTIVE_STAGE_COUNT
        extra_bytes = (stage_count - 1) * _buffer_bytes(reduce_op.buffer)
        if extra_bytes > memory_budget_bytes:
            continue

        overlap_axes = _eligible_async_overlap_axes(program, reduce_op, shard_dims)
        if len(overlap_axes) == 0:
            continue

        for overlap_axis in overlap_axes:
            variant = copy.deepcopy(program)
            variant_axis_map = {axis.name: axis for axis in variant.visit(collect_axis)}
            variant_reduce = variant.visit(collect_reduce)[reduce_id]
            variant_overlap_axis = variant_axis_map.get(overlap_axis.name, overlap_axis)
            variant_reduce.managed_collective_strategy = "async_collective_overlap"
            variant_reduce.async_collective_overlap_axis = variant_overlap_axis
            variant_reduce.async_collective_tile_count = int(
                variant_overlap_axis.size
            ) // int(variant_overlap_axis.min_block_size)
            variant_reduce.async_collective_stage_count = stage_count
            variant_reduce.async_collective_lifecycle = copy.deepcopy(
                reduce_op.async_collective_lifecycle
            )
            if variant_reduce.async_collective_lifecycle is None:
                variant_reduce.async_collective_lifecycle = AsyncCollectiveLifecycle()
            variants.append(variant)

    return variants


# def detect_conflict_axis(assign, axes_list, mutex_pair) -> bool:
#     axes = [axis.name for axes in axes_list for axis in axes]
#     non_zero_axes = []
#     for axis, (start, length) in zip(axes, assign):
#         if length > 0:
#             non_zero_axes.append(axis)

#     for axis in non_zero_axes:
#         if mutex_pair.get(axis) is None:
#             continue
#         for banned in mutex_pair[axis]:
#             if banned in non_zero_axes:
#                 return True
#     return False


def search(
    input_program: Program,
    origin_mesh: DeviceMesh,
    split_axis_names: List[str] = list(),
    tensor_mapping_constraints: Optional[TensorMappingConstraints] = None,
    program_filter: Optional[Callable[[Program], bool]] = None,
    dedupe_key_fn: Optional[Callable[[Program], object]] = None,
    mesh_shape_policy: Optional[MeshShapePolicy] = None,
) -> Iterator[Program]:
    """
    Search for the best schedule for a given program.

    Parameters
    ----------
    program : Program
        The program to search for the best schedule.

    mesh : DeviceMesh
        The device mesh to search for the best schedule.
        we assume it is a 1D mesh for now
        TODO: support 2D mesh

    split_axis_names : List[str]
        The axes can be split to multiple dimensions.

    program_filter : Optional[Callable[[Program], bool]]
        Additional runtime predicate applied after tensor mapping constraints.

    dedupe_key_fn : Optional[Callable[[Program], object]]
        Optional callback that maps a candidate ``Program`` to a canonical
        pruning key.  When provided, the search keeps only the first candidate
        whose key has not been seen before.  A ``None`` return from the
        callback means the candidate is not eligible for deduplication and is
        always yielded.

    mesh_shape_policy : Optional[MeshShapePolicy]
        When provided, uses topology-aware mesh shape enumeration instead of
        blind world_size factorization.  Topology metadata is generated
        directly from the policy rather than inferred post-hoc.

    Returns
    -------
    Iterator[Program]
        An iterator of programs with different schedules.
    """

    axes_list = input_program.visit(
        collect_parallelizeable_axes
    )  # list of list of axes, each list coresponds to a GridLoop
    # print(f"axes_list: {axes_list}")
    axis_num = sum(len(axes) for axes in axes_list)
    # print(f"axis_num: {axis_num}")

    max_dim = axis_num + len(
        split_axis_names
    )  # allow one axis to be assigned to two dimensions, for double ring

    # the following code is found to be wrong, q and kv can be parallelized together
    # # find the axis pairs that can't be parallelized together
    # # theortically, we can analyze the data dependency to get the pairs
    # # here we will simplify the condition to be if match the
    # # pattern of q[axis0, axis1], k[axis0, axis2] then axis1 and axis2 can't be parallelized together
    # mutex_pair = {}

    # buffers = input_program.visit(get_buffers)
    # for buffer1 in buffers:
    #     for buffer2 in buffers:
    #         if buffer1 == buffer2:
    #             continue
    #         axes1 = []
    #         for axes in buffer1.bound_axes:
    #             for axis in axes:
    #                 axes1.append(axis.name)
    #         axes2 = []
    #         for axes in buffer2.bound_axes:
    #             for axis in axes:
    #                 axes2.append(axis.name)

    #         # compute common axes
    #         common_axes = set(axes1) & set(axes2)
    #         if len(common_axes) < len(axes1) and len(common_axes) < len(axes2):
    #             axes1 = list(set(axes1) - common_axes)
    #             axes2 = list(set(axes2) - common_axes)
    #             for axis in axes1:
    #                 if mutex_pair.get(axis) is None:
    #                     mutex_pair[axis] = set()
    #                 mutex_pair[axis].update(axes2)
    #             for axis in axes2:
    #                 if mutex_pair.get(axis) is None:
    #                     mutex_pair[axis] = set()
    #                 mutex_pair[axis].update(axes1)

    axes_split = []
    for axes in axes_list:
        for axis in axes:
            if axis.name in split_axis_names:
                axes_split.append(axis)

    assert len(axes_split) == len(split_axis_names), (
        "split axis not found in the program"
    )

    seen_keys: set = set()

    def _should_yield(variant: Program) -> bool:
        if dedupe_key_fn is None:
            return True
        key = dedupe_key_fn(variant)
        if key is None:
            return True
        if key in seen_keys:
            return False
        seen_keys.add(key)
        return True

    def _set_metadata_and_match(
        program: Program,
        mesh: DeviceMesh,
        mesh_shape: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        if mesh_shape_policy is not None and mesh_shape is not None:
            program.topology_metadata = mesh_shape_policy.topology_metadata_for_shape(
                mesh_shape
            )
        else:
            program.topology_metadata = _infer_topology_metadata(origin_mesh, mesh)

        # Attach logical shard factors when a topology-aware policy is active.
        if mesh_shape_policy is not None:
            from mercury.search.topology_policy import (
                compute_program_logical_shard_factors,
            )

            program._logical_shard_factors = compute_program_logical_shard_factors(
                program
            )

        matches_constraints = program_satisfies_tensor_mapping_constraints(
            program,
            tensor_mapping_constraints,
        )
        if not matches_constraints:
            return False
        if program_filter is None:
            return True
        return bool(program_filter(program))

    for split_num in enumerate_axis_split(axes_split, len(origin_mesh.devices), []):
        # try all possible axis split

        splited_program = copy.deepcopy(input_program)

        # do the split
        for axis_name, split in zip(split_axis_names, split_num):
            if split == 1:
                continue
            axes = splited_program.visit(collect_axis)
            target_axis = None
            for axis in axes:
                if axis.name == axis_name:
                    target_axis = axis
                    break
            assert target_axis is not None, "split axis not found in the program"
            tile_loop(splited_program, target_axis, target_axis.size // split)

        axes = splited_program.visit(collect_axis)
        axis_num = len(axes)

        # ---------------------------------------------------------------
        # Flat search path: enumerate logical factors per domain
        # ---------------------------------------------------------------
        if isinstance(mesh_shape_policy, FlatMeshShapePolicy):
            yield from _search_flat_path(
                splited_program,
                origin_mesh,
                axes,
                mesh_shape_policy,
                _set_metadata_and_match,
                _should_yield,
            )
            continue

        if mesh_shape_policy is not None:
            mesh_shapes_iter = iter(mesh_shape_policy.enumerate_shapes())
        else:
            mesh_shapes_iter = enumerate_mesh_shapes(len(origin_mesh.devices), max_dim)

        for mesh_shape in mesh_shapes_iter:
            # try all possible mesh shapes

            program_shape = copy.deepcopy(splited_program)
            mesh = origin_mesh.reshape(mesh_shape)

            init_distributed(program_shape, mesh)

            # we need to partition the mesh for each axis

            for assign in enumerate_mesh_assignment(len(mesh_shape), axis_num):
                # try all possible assignments
                program = copy.deepcopy(program_shape)

                seed_program = copy.deepcopy(program)

                loops = program.visit(collect_loops)
                axes_list = program.visit(collect_parallelizeable_axes)

                all_axes = program.visit(collect_axis)
                axes_names = set([axis.name for axis in all_axes])
                ringable_axes = set()

                # collect the axes to be parallelized
                parallelize_axes = []
                cnt = 0
                for loop, axes in zip(loops, axes_list):
                    for axis in axes:
                        if assign[cnt][1] != 0:
                            parallelize_axes.append(axis.name)
                        cnt += 1

                cnt = 0
                succ = True
                for loop, axes in zip(loops, axes_list):
                    for axis in axes:
                        usd_axes = set(parallelize_axes)
                        succ &= parallelize(
                            program,
                            loop,
                            axis,
                            mesh,
                            assign[cnt][0],
                            assign[cnt][0] + assign[cnt][1],
                            usd_axes,
                        )
                        if not succ:
                            break
                        if assign[cnt][1] != 0:
                            ringable_axes |= axes_names - usd_axes
                        shift(
                            program,
                            axis,
                            mesh,
                            assign[cnt][0],
                            assign[cnt][0] + assign[cnt][1],
                            1,
                            usd_axes,
                        )
                        cnt += 1
                    if not succ:
                        break

                if succ:
                    # print(f"mesh_shape: {mesh_shape}")
                    # print(f"assign: {assign}")
                    for variant in _enumerate_collective_strategy_variants(program):
                        if _set_metadata_and_match(
                            variant, mesh, mesh_shape
                        ) and _should_yield(variant):
                            yield variant

                    # enumerate all subset of ringable axes
                    for i in range(
                        1, len(ringable_axes) + 1
                    ):  # start from 1, as empty set is the same as above
                        for ringable_axes_subset in itertools.combinations(
                            ringable_axes, i
                        ):
                            program = copy.deepcopy(seed_program)
                            loops = program.visit(collect_loops)
                            axes_list = program.visit(collect_parallelizeable_axes)
                            cnt = 0
                            for loop, axes in zip(loops, axes_list):
                                for axis in axes:
                                    usd_axes = set(ringable_axes_subset)
                                    usd_axes.update(parallelize_axes)
                                    succ = parallelize(
                                        program,
                                        loop,
                                        axis,
                                        mesh,
                                        assign[cnt][0],
                                        assign[cnt][0] + assign[cnt][1],
                                        usd_axes,
                                    )
                                    assert succ == True, (
                                        "should be able to parallelize as we have checked before"
                                    )
                                    shift(
                                        program,
                                        axis,
                                        mesh,
                                        assign[cnt][0],
                                        assign[cnt][0] + assign[cnt][1],
                                        1,
                                        usd_axes,
                                    )
                                    cnt += 1
                            for variant in _enumerate_collective_strategy_variants(
                                program
                            ):
                                if _set_metadata_and_match(
                                    variant, mesh, mesh_shape
                                ) and _should_yield(variant):
                                    yield variant


def search_with_progress(
    input_program: Program,
    origin_mesh: DeviceMesh,
    split_axis_names: List[str] = list(),
    tensor_mapping_constraints: Optional[TensorMappingConstraints] = None,
    program_filter: Optional[Callable[[Program], bool]] = None,
    progress_desc: Optional[str] = None,
    show_progress: bool = True,
    miniters: int = 32,
    mininterval: float = 0.5,
    dedupe_key_fn: Optional[Callable[[Program], object]] = None,
    mesh_shape_policy: Optional[MeshShapePolicy] = None,
) -> Iterator[Program]:
    """Wrap ``search`` with a streaming progress bar for generated candidates.

    The search space size is not known ahead of time, so this progress bar reports
    the current number of generated candidates and throughput instead of a percentage.

    Args:
        dedupe_key_fn: Optional callback mapping a candidate ``Program`` to a
            canonical pruning key.  When provided, ``search`` keeps only the
            first candidate seen for each key and suppresses later duplicates.
            Return ``None`` from the callback to leave a candidate unpruned.
        mesh_shape_policy: Optional topology-aware mesh shape policy.
    """
    iterator = search(
        input_program,
        origin_mesh,
        split_axis_names,
        tensor_mapping_constraints,
        program_filter,
        dedupe_key_fn=dedupe_key_fn,
        mesh_shape_policy=mesh_shape_policy,
    )
    if not show_progress:
        yield from iterator
        return

    with tqdm(
        desc=progress_desc or "search",
        unit="cand",
        dynamic_ncols=True,
        miniters=max(1, miniters),
        mininterval=mininterval,
    ) as progress_bar:
        for program in iterator:
            progress_bar.update(1)
            yield program
