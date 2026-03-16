# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""GEMM-specific candidate deduplication for the search pipeline."""

from typing import Dict, List, Optional, Tuple

from mercury.ir.elements import Buffer
from mercury.ir.nodes import Program, ReduceOp
from mercury.ir.utils import collect_loops, collect_reduce, get_io_buffers
from mercury.search.mapping_constraints import (
    ExactLayoutSignature,
    exact_layout_signature_from_buffer,
)


_GEMM_BOUNDARY_NAMES = ("a", "b", "c")
_K_FAMILY_PREFIXES = ("K", "K_outer", "K_inner")


def _is_k_family(name: str) -> bool:
    return name in _K_FAMILY_PREFIXES


def _find_gemm_boundary_buffers(
    program: Program,
) -> Optional[Dict[str, Buffer]]:
    buffers: Dict[str, Buffer] = {}
    for buf in program.visit(get_io_buffers):
        if buf.tensor in _GEMM_BOUNDARY_NAMES:
            buffers[buf.tensor] = buf
    if set(buffers.keys()) != set(_GEMM_BOUNDARY_NAMES):
        return None
    for buf in buffers.values():
        if buf.ndim != 2:
            return None
    return buffers


def _collect_managed_reduce_ops(program: Program) -> List[ReduceOp]:
    ops: List[ReduceOp] = []
    for loop in program.visit(collect_loops):
        if "m" not in loop.axis_types:
            continue
        for node in program.visit(collect_reduce):
            has_k_axis = any(_is_k_family(ax.name) for ax in node.axes)
            if has_k_axis and node not in ops:
                ops.append(node)
    return ops


def _is_safe_gemm_blocking_collective(
    managed_reduce_ops: List[ReduceOp],
) -> bool:
    if len(managed_reduce_ops) != 1:
        return False
    reduce_op = managed_reduce_ops[0]
    if not all(_is_k_family(ax.name) for ax in reduce_op.axes):
        return False
    if len(reduce_op.comm) > 0:
        return False
    if reduce_op.managed_collective_strategy != "blocking_collective":
        return False
    return True


def _layout_signature(buf: Buffer) -> ExactLayoutSignature:
    return exact_layout_signature_from_buffer(buf)


def _visible_non_k_loop_structure(
    program: Program,
) -> Tuple[Tuple[Tuple[str, int, int], ...], ...]:
    result = []
    for loop in program.visit(collect_loops):
        axes_info = []
        for ax, ax_type in zip(loop.axes, loop.axis_types):
            if _is_k_family(ax.name):
                continue
            axes_info.append((ax.name, int(ax.size), int(ax.min_block_size)))
        if axes_info:
            result.append(tuple(axes_info))
    return tuple(result)


def _normalized_collective_shard_dims(
    reduce_op: ReduceOp, mesh_ndim: int
) -> Tuple[int, ...]:
    ring_dims = set(int(comm.shard_dim) for comm in reduce_op.comm)
    dims = sorted(
        set(
            int(dim)
            for dim in reduce_op.shard_dim
            if 0 <= int(dim) < mesh_ndim and int(dim) not in ring_dims
        )
    )
    return tuple(dims)


def _effective_local_k_extent(reduce_op: ReduceOp) -> int:
    extent = 1
    for ax in reduce_op.axes:
        if _is_k_family(ax.name):
            extent *= int(ax.size) // int(ax.min_block_size)
    return extent


def gemm_canonical_dedupe_key(program: Program) -> Optional[object]:
    """Return a canonical pruning key for a GEMM search candidate.

    Returns ``None`` when the candidate does not match the safe
    blocking-collective GEMM pattern, leaving it unpruned.
    """
    buffers = _find_gemm_boundary_buffers(program)
    if buffers is None:
        return None

    managed_reduce_ops = _collect_managed_reduce_ops(program)
    if not _is_safe_gemm_blocking_collective(managed_reduce_ops):
        return None

    reduce_op = managed_reduce_ops[0]

    if program.mesh is None:
        return None

    mesh_ndim = len(program.mesh.shape)

    layout_a = _layout_signature(buffers["a"])
    layout_b = _layout_signature(buffers["b"])
    layout_c = _layout_signature(buffers["c"])

    topology = tuple(
        sorted(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in program.topology_metadata.items()
        )
    )

    loop_structure = _visible_non_k_loop_structure(program)

    shard_dims = _normalized_collective_shard_dims(reduce_op, mesh_ndim)

    k_extent = _effective_local_k_extent(reduce_op)

    return (
        layout_a,
        layout_b,
        layout_c,
        topology,
        loop_structure,
        shard_dims,
        k_extent,
    )
