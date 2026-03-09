# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""CPU-only estimation for activation resharding edges."""

from typing import Any, Dict, List, Optional, Tuple

from mercury.ir.distributed import DeviceMesh, ShardType
from mercury.ir.elements import Buffer
from mercury.ir.utils import get_element_size


def _read_positive(config: Dict[str, Any], path: List[str]) -> float:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"Missing required config field: {'.'.join(path)}")
        node = node[key]
    if not isinstance(node, (int, float)):
        raise ValueError(f"Config field must be numeric: {'.'.join(path)}")
    if node <= 0:
        raise ValueError(f"Config field must be > 0: {'.'.join(path)}")
    return float(node)


def _read_non_negative(config: Dict[str, Any], path: List[str]) -> float:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"Missing required config field: {'.'.join(path)}")
        node = node[key]
    if not isinstance(node, (int, float)):
        raise ValueError(f"Config field must be numeric: {'.'.join(path)}")
    if node < 0:
        raise ValueError(f"Config field must be >= 0: {'.'.join(path)}")
    return float(node)


def _one_dim_to_n_dim(one_dim_index: int, dimensions: Tuple[int, ...]) -> Tuple[int, ...]:
    n_dim_index = []
    remainder = one_dim_index

    total_device = 1
    for dim_size in dimensions:
        total_device *= int(dim_size)

    for dim_size in reversed(dimensions):
        total_device //= int(dim_size)
        index = remainder // total_device
        remainder %= total_device
        n_dim_index.insert(0, index)

    return tuple(n_dim_index)


def _n_dim_to_one_dim(n_dim_index: Tuple[int, ...], dimensions: Tuple[int, ...]) -> int:
    one_dim_index = 0
    for index, dim_size in zip(reversed(n_dim_index), reversed(dimensions)):
        one_dim_index = one_dim_index * int(dim_size) + int(index)
    return one_dim_index


def _mesh_world_size(mesh: DeviceMesh) -> int:
    world_size = 1
    for dim_size in mesh.shape:
        world_size *= int(dim_size)
    return world_size


def _global_shape(buffer: Buffer) -> Tuple[int, ...]:
    if buffer.shard_spec is None:
        raise ValueError(f"Buffer {buffer.tensor} shard_spec must be initialized")

    local_shape = buffer.get_shape()
    global_shape: List[int] = []
    for dim, (local_size, spec) in enumerate(zip(local_shape, buffer.shard_spec.specs)):
        if isinstance(spec, tuple) and spec[0] == ShardType.SHARD:
            shard_cards = 1
            for mesh_dim in spec[1]:
                shard_cards *= int(buffer.shard_spec.mesh.shape[int(mesh_dim)])
            global_shape.append(int(local_size) * shard_cards)
        elif spec == ShardType.REPLICATE:
            global_shape.append(int(local_size))
        else:
            raise ValueError(f"Unsupported shard spec at dim {dim}: {spec}")
    return tuple(global_shape)


def _get_shard_coords(buffer: Buffer, rank: int, mesh_shape: Tuple[int, ...]) -> List[int]:
    if buffer.shard_spec is None:
        raise ValueError(f"Buffer {buffer.tensor} shard_spec must be initialized")

    coords = []
    rank_indices = _one_dim_to_n_dim(rank, mesh_shape)

    for spec in buffer.shard_spec.specs:
        if isinstance(spec, tuple) and spec[0] == ShardType.SHARD:
            shard_mesh_dims = tuple(int(v) for v in spec[1])
            shard_coord = tuple(rank_indices[mesh_dim] for mesh_dim in shard_mesh_dims)
            shard_mesh = tuple(int(mesh_shape[mesh_dim]) for mesh_dim in shard_mesh_dims)
            coords.append(_n_dim_to_one_dim(shard_coord, shard_mesh))
        else:
            coords.append(0)
    return coords


def _get_shard_ranges(buffer: Buffer, rank: int, mesh_shape: Tuple[int, ...]) -> List[Tuple[int, int]]:
    local_shape = buffer.get_shape()
    shard_coords = _get_shard_coords(buffer, rank, mesh_shape)

    ranges = []
    for dim, shard_coord in enumerate(shard_coords):
        local_size = int(local_shape[dim])
        start = int(shard_coord) * local_size
        end = start + local_size
        ranges.append((start, end))
    return ranges


def _ranges_overlap(
    ranges1: List[Tuple[int, int]],
    ranges2: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    if len(ranges1) != len(ranges2):
        return None

    overlap = []
    for (start1, end1), (start2, end2) in zip(ranges1, ranges2):
        if end1 <= start2 or end2 <= start1:
            return None
        overlap.append((max(start1, start2), min(end1, end2)))
    return overlap


def _calculate_volume(range_: List[Tuple[int, int]]) -> int:
    volume = 1
    for start, end in range_:
        volume *= int(end) - int(start)
    return volume


def _is_fully_covered(
    target_range: List[Tuple[int, int]],
    covering_ranges: List[List[Tuple[int, int]]],
) -> bool:
    for covering_range in covering_ranges:
        overlap = _ranges_overlap(target_range, covering_range)
        if overlap is None:
            return False
        if overlap != covering_range:
            return False

    for i in range(len(covering_ranges)):
        for j in range(i + 1, len(covering_ranges)):
            if _ranges_overlap(covering_ranges[i], covering_ranges[j]) is not None:
                return False

    target_volume = _calculate_volume(target_range)
    covering_volume = sum(_calculate_volume(covering_range) for covering_range in covering_ranges)
    return target_volume == covering_volume


def _mesh_distance(rank_a: int, rank_b: int, mesh_shape: Tuple[int, ...]) -> int:
    coords_a = _one_dim_to_n_dim(rank_a, mesh_shape)
    coords_b = _one_dim_to_n_dim(rank_b, mesh_shape)
    return sum(abs(int(a) - int(b)) for a, b in zip(coords_a, coords_b))


def _pick_nearest_source(
    current_src_rank: int,
    candidate_src_rank: int,
    target_rank: int,
    origin_mesh: DeviceMesh,
) -> int:
    cur_distance = _mesh_distance(current_src_rank, target_rank, origin_mesh.shape)
    candidate_distance = _mesh_distance(candidate_src_rank, target_rank, origin_mesh.shape)
    if candidate_distance < cur_distance:
        return candidate_src_rank
    if candidate_distance > cur_distance:
        return current_src_rank
    return min(current_src_rank, candidate_src_rank)


def _is_inter_node(src_rank: int, dst_rank: int, origin_mesh: DeviceMesh) -> bool:
    if src_rank == dst_rank:
        return False
    if len(origin_mesh.shape) <= 1:
        return False
    src_coords = _one_dim_to_n_dim(src_rank, origin_mesh.shape)
    dst_coords = _one_dim_to_n_dim(dst_rank, origin_mesh.shape)
    return src_coords[0] != dst_coords[0]


def estimate_reshard_time(
    src_buffer: Buffer,
    dst_buffer: Buffer,
    hw_config: Dict[str, Any],
    origin_mesh: DeviceMesh,
) -> float:
    """Estimate one reshard edge latency in milliseconds."""
    if src_buffer.shard_spec is None or dst_buffer.shard_spec is None:
        raise ValueError("src_buffer and dst_buffer shard_spec must be initialized")

    src_mesh = src_buffer.shard_spec.mesh
    dst_mesh = dst_buffer.shard_spec.mesh
    if _mesh_world_size(src_mesh) != _mesh_world_size(dst_mesh):
        raise ValueError("src_buffer and dst_buffer mesh world sizes must match")
    if _mesh_world_size(src_mesh) != _mesh_world_size(origin_mesh):
        raise ValueError("origin_mesh world size must match src/dst world size")

    src_global_shape = _global_shape(src_buffer)
    dst_global_shape = _global_shape(dst_buffer)
    if src_global_shape != dst_global_shape:
        raise ValueError(
            f"src/dst global shape mismatch: {src_global_shape} vs {dst_global_shape}"
        )

    if (
        tuple(int(v) for v in src_mesh.shape) == tuple(int(v) for v in dst_mesh.shape)
        and src_buffer.shard_spec.specs == dst_buffer.shard_spec.specs
    ):
        return 0.0

    world_size = _mesh_world_size(src_mesh)
    src_mesh_shape = tuple(int(v) for v in src_mesh.shape)
    dst_mesh_shape = tuple(int(v) for v in dst_mesh.shape)

    element_size = int(get_element_size(src_buffer.dtype))
    inter_bw = _read_positive(
        hw_config, ["interconnect", "inter_node", "bandwidth_gb_per_s"]
    ) * (10 ** 9)
    intra_bw = _read_positive(
        hw_config, ["interconnect", "intra_node", "bandwidth_gb_per_s"]
    ) * (10 ** 9)
    inter_latency = _read_non_negative(
        hw_config, ["interconnect", "inter_node", "latency_us"]
    ) / (10 ** 6)
    intra_latency = _read_non_negative(
        hw_config, ["interconnect", "intra_node", "latency_us"]
    ) / (10 ** 6)

    max_rank_time_s = 0.0
    for target_rank in range(world_size):
        target_range = _get_shard_ranges(dst_buffer, target_rank, dst_mesh_shape)
        source_for_overlap: Dict[Tuple[Tuple[int, int], ...], int] = {}

        for src_rank in range(world_size):
            src_range = _get_shard_ranges(src_buffer, src_rank, src_mesh_shape)
            overlap = _ranges_overlap(target_range, src_range)
            if overlap is None:
                continue

            overlap_key = tuple((int(start), int(end)) for start, end in overlap)
            if overlap_key in source_for_overlap:
                source_for_overlap[overlap_key] = _pick_nearest_source(
                    source_for_overlap[overlap_key],
                    src_rank,
                    target_rank,
                    origin_mesh,
                )
            else:
                source_for_overlap[overlap_key] = src_rank

        overlap_ranges = [list(range_key) for range_key in source_for_overlap.keys()]
        if len(overlap_ranges) == 0:
            raise ValueError("No source range overlaps with destination shard")
        if not _is_fully_covered(target_range, overlap_ranges):
            raise ValueError("Source ranges cannot fully cover destination shard")

        inter_bytes = 0
        intra_bytes = 0
        inter_msgs = 0
        intra_msgs = 0
        for overlap_key, src_rank in source_for_overlap.items():
            if src_rank == target_rank:
                continue

            transfer_bytes = _calculate_volume(list(overlap_key)) * element_size
            if _is_inter_node(src_rank, target_rank, origin_mesh):
                inter_bytes += transfer_bytes
                inter_msgs += 1
            else:
                intra_bytes += transfer_bytes
                intra_msgs += 1

        rank_time_s = (
            (inter_bytes / inter_bw)
            + (inter_msgs * inter_latency)
            + (intra_bytes / intra_bw)
            + (intra_msgs * intra_latency)
        )
        max_rank_time_s = max(max_rank_time_s, rank_time_s)

    return max_rank_time_s * 1000.0
