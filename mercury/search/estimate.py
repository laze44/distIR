# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Theoretical performance estimation based on roofline and communication models."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from mercury.ir.elements import Buffer
from mercury.ir.nodes import (
    BufferLoad,
    BufferStore,
    ManagedReductionPipelineRegion,
    Program,
    ReduceOp,
)
from mercury.ir.utils import (
    collect_axis,
    collect_pipeline_regions,
    collect_reduce,
    get_element_size,
)


_DTYPE_TO_CONFIG_KEY: Dict[torch.dtype, str] = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float8_e4m3fn: "fp8",
    torch.float8_e5m2: "fp8",
}


@dataclass
class EstimateResult:
    """Estimated execution time breakdown for one IR program."""

    compute_time_ms: float
    comm_time_ms: float
    total_time_ms: float

    def __post_init__(self) -> None:
        if self.compute_time_ms < 0 or self.comm_time_ms < 0 or self.total_time_ms < 0:
            raise ValueError("Estimated times must be non-negative")
        if self.total_time_ms - (self.compute_time_ms + self.comm_time_ms) > 1e-9:
            raise ValueError(
                "total_time_ms cannot exceed compute_time_ms + comm_time_ms"
            )


@dataclass
class _CommEvent:
    """One communication event used by overlap estimation."""

    time_ms: float
    buffer_name: str
    overlaps_compute: bool


def _get_required(config: Dict[str, Any], path: Sequence[str]) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"Missing required config field: {'.'.join(path)}")
        node = node[key]
    return node


def _read_positive(config: Dict[str, Any], path: Sequence[str]) -> float:
    value = _get_required(config, path)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Config field must be numeric: {'.'.join(path)}")
    if value <= 0:
        raise ValueError(f"Config field must be > 0: {'.'.join(path)}")
    return float(value)


def _read_non_negative(config: Dict[str, Any], path: Sequence[str]) -> float:
    value = _get_required(config, path)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Config field must be numeric: {'.'.join(path)}")
    if value < 0:
        raise ValueError(f"Config field must be >= 0: {'.'.join(path)}")
    return float(value)


_KNOWN_DTYPE_KEYS = {"bf16", "fp16", "fp32", "tf32", "fp8"}


def _validate_hardware_config(config: Dict[str, Any]) -> None:
    name = _get_required(config, ["name"])
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Config field must be a non-empty string: name")

    peak_tflops = _get_required(config, ["compute", "peak_tflops"])
    if not isinstance(peak_tflops, dict) or not peak_tflops:
        raise ValueError("compute.peak_tflops must be a non-empty mapping")

    for key in peak_tflops:
        if key not in _KNOWN_DTYPE_KEYS:
            raise ValueError(f"Unknown dtype key in peak_tflops: {key}")
        _read_positive(config, ["compute", "peak_tflops", key])

    _read_positive(config, ["memory", "bandwidth_tb_per_s"])
    if "capacity_gb" in config.get("memory", {}):
        _read_positive(config, ["memory", "capacity_gb"])

    _read_positive(config, ["interconnect", "intra_node", "bandwidth_gb_per_s"])
    _read_non_negative(config, ["interconnect", "intra_node", "latency_us"])
    _read_positive(config, ["interconnect", "inter_node", "bandwidth_gb_per_s"])
    _read_non_negative(config, ["interconnect", "inter_node", "latency_us"])


def load_hardware_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a hardware configuration JSON file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Hardware config root must be a JSON object")

    _validate_hardware_config(config)
    return config


def _buffer_numel(buffer: Buffer) -> int:
    numel = 1
    for dim in buffer.get_shape():
        numel *= int(dim)
    return numel


def _collect_unique_buffers(program: Program) -> List[Buffer]:
    seen: Dict[str, Buffer] = {}

    def _collector(node: Any) -> Optional[Buffer]:
        if hasattr(node, "buffer") and isinstance(node.buffer, Buffer):
            return node.buffer
        return None

    for buffer in program.visit(_collector):
        if buffer.tensor not in seen:
            seen[buffer.tensor] = buffer
    return list(seen.values())


def _first_output_buffer(program: Program) -> Optional[Buffer]:
    for buffer in _collect_unique_buffers(program):
        if buffer.write:
            return buffer
    return None


def _extract_local_mnk(program: Program) -> List[int]:
    output_buffer = _first_output_buffer(program)
    if output_buffer is None:
        raise ValueError("Program has no writable output buffer")

    out_shape = output_buffer.get_shape()
    if len(out_shape) < 2:
        raise ValueError("Output buffer must be at least 2D for GEMM estimation")
    m_local, n_local = int(out_shape[0]), int(out_shape[1])

    k_local = None
    reduce_ops = program.visit(collect_reduce)
    for reduce_op in reduce_ops:
        if len(reduce_op.axes) == 0:
            continue
        k_local = 1
        for axis in reduce_op.axes:
            k_local *= int(axis.size)
        break

    if k_local is None:
        all_axes = program.visit(collect_axis)
        for axis in all_axes:
            if axis.name.upper().startswith("K"):
                k_local = int(axis.size)
                break

    if k_local is None:
        raise ValueError("Unable to infer K dimension from program")

    return [m_local, n_local, k_local]


def _default_dtype_key(hw_config: Dict[str, Any]) -> str:
    """Return the first available dtype key from the hardware config."""
    peak_tflops = hw_config["compute"]["peak_tflops"]
    for key in ("bf16", "fp16", "fp8", "fp32", "tf32"):
        if key in peak_tflops:
            return key
    raise ValueError("No supported dtype found in compute.peak_tflops")


def _peak_flops_per_second(hw_config: Dict[str, Any], dtype: torch.dtype) -> float:
    dtype_key = _DTYPE_TO_CONFIG_KEY.get(dtype)
    if dtype_key is None or dtype_key not in hw_config["compute"]["peak_tflops"]:
        dtype_key = _default_dtype_key(hw_config)
    peak_tflops = _read_positive(hw_config, ["compute", "peak_tflops", dtype_key])
    return peak_tflops * (10**12)


def _memory_bandwidth_bytes_per_second(hw_config: Dict[str, Any]) -> float:
    bandwidth_tb_per_s = _read_positive(hw_config, ["memory", "bandwidth_tb_per_s"])
    return bandwidth_tb_per_s * (10**12)


def _estimate_gemm_data_bytes(program: Program) -> float:
    """Estimate single-execution memory traffic for local GEMM."""
    output_buffer = _first_output_buffer(program)
    if output_buffer is None:
        raise ValueError("Program has no writable output buffer")

    output_bytes = float(
        _buffer_numel(output_buffer) * get_element_size(output_buffer.dtype)
    )
    buffers = _collect_unique_buffers(program)

    read_buffers = [
        buffer
        for buffer in buffers
        if buffer.tensor != output_buffer.tensor and buffer.read
    ]
    if len(read_buffers) < 2:
        fallback_buffers = [
            buffer for buffer in buffers if buffer.tensor != output_buffer.tensor
        ]
        fallback_buffers.sort(key=_buffer_numel, reverse=True)
        read_buffers = fallback_buffers[:2]
    else:
        read_buffers.sort(key=_buffer_numel, reverse=True)
        read_buffers = read_buffers[:2]

    read_bytes = 0.0
    for buffer in read_buffers:
        read_bytes += float(_buffer_numel(buffer) * get_element_size(buffer.dtype))

    # C local is read and written at least once in GEMM accumulation.
    return read_bytes + 2.0 * output_bytes


def _estimate_compute_time_ms(program: Program, hw_config: Dict[str, Any]) -> float:
    m_local, n_local, k_local = _extract_local_mnk(program)
    flops = 2.0 * float(m_local) * float(n_local) * float(k_local)

    output_buffer = _first_output_buffer(program)
    if output_buffer is None:
        raise ValueError("Program has no writable output buffer")

    compute_bound_s = flops / _peak_flops_per_second(hw_config, output_buffer.dtype)
    bytes_moved = _estimate_gemm_data_bytes(program)
    memory_bound_s = bytes_moved / _memory_bandwidth_bytes_per_second(hw_config)
    return max(compute_bound_s, memory_bound_s) * 1000.0


def _collect_comm_nodes(program: Program) -> List[Any]:
    def _collector(node: Any) -> Optional[Any]:
        if isinstance(node, (BufferLoad, BufferStore, ReduceOp)):
            if isinstance(node, ReduceOp):
                if len(node.comm) > 0 or len(node.shard_dim) > 0:
                    return node
            elif len(node.comm) > 0:
                return node
        return None

    return program.visit(_collector)


def _normalize_mesh_dims(mesh_dims: List[int], ndim: int) -> List[int]:
    unique = sorted(set(int(dim) for dim in mesh_dims))
    return [dim for dim in unique if 0 <= dim < ndim]


def _normalize_topology_metadata(
    program: Program,
    num_inter_dims: Optional[int],
) -> Dict[str, List[int]]:
    ndim = len(program.mesh.shape)
    metadata = (
        dict(program.topology_metadata) if program.topology_metadata is not None else {}
    )

    if "inter_node_dims" not in metadata:
        if num_inter_dims is None:
            inter_node_dims = [0] if ndim > 1 else []
        else:
            inter_node_dims = list(range(max(0, min(ndim, num_inter_dims))))
        metadata["inter_node_dims"] = inter_node_dims

    inter_node_dims = _normalize_mesh_dims(metadata.get("inter_node_dims", []), ndim)
    intra_node_dims = _normalize_mesh_dims(metadata.get("intra_node_dims", []), ndim)
    mixed_dims = _normalize_mesh_dims(metadata.get("mixed_dims", []), ndim)

    covered = set(inter_node_dims) | set(intra_node_dims) | set(mixed_dims)
    for dim in range(ndim):
        if dim not in covered:
            intra_node_dims.append(dim)

    return {
        "inter_node_dims": sorted(set(inter_node_dims)),
        "intra_node_dims": sorted(set(intra_node_dims)),
        "mixed_dims": sorted(set(mixed_dims)),
    }


def _link_params(hw_config: Dict[str, Any], inter_node: bool) -> Tuple[float, float]:
    if inter_node:
        bandwidth = _read_positive(
            hw_config, ["interconnect", "inter_node", "bandwidth_gb_per_s"]
        ) * (10**9)
        latency_s = _read_non_negative(
            hw_config, ["interconnect", "inter_node", "latency_us"]
        ) / (10**6)
    else:
        bandwidth = _read_positive(
            hw_config, ["interconnect", "intra_node", "bandwidth_gb_per_s"]
        ) * (10**9)
        latency_s = _read_non_negative(
            hw_config, ["interconnect", "intra_node", "latency_us"]
        ) / (10**6)
    return bandwidth, latency_s


def _is_inter_mesh_dim(mesh_dim: int, topology_metadata: Dict[str, List[int]]) -> bool:
    if mesh_dim in topology_metadata["inter_node_dims"]:
        return True
    if mesh_dim in topology_metadata["intra_node_dims"]:
        return False
    if mesh_dim in topology_metadata["mixed_dims"]:
        return True
    return True


def _estimate_collective_reduce_events(
    program: Program,
    hw_config: Dict[str, Any],
    topology_metadata: Dict[str, List[int]],
) -> List[_CommEvent]:
    events: List[_CommEvent] = []
    reduce_ops = program.visit(collect_reduce)

    legalized_reduce_bufs: Set[str] = set()
    pipeline_regions = program.visit(collect_pipeline_regions)
    for region in pipeline_regions:
        if region.legalized and region.reduce_op is not None:
            legalized_reduce_bufs.add(region.reduce_op.buffer.tensor)
    for reduce_op in reduce_ops:
        if len(reduce_op.shard_dim) == 0:
            continue
        strategy = getattr(
            reduce_op, "managed_collective_strategy", "blocking_collective"
        )
        if strategy == "async_collective_overlap":
            # After prepare_pipeline(), legalized reductions are handled by
            # _estimate_async_collective_pipeline_overhead_ms; skip them here.
            if reduce_op.buffer.tensor in legalized_reduce_bufs:
                continue
            # Non-legalized async candidates fall through as blocking collectives.

        ring_dims = set(int(comm.shard_dim) for comm in reduce_op.comm)
        shard_dims = sorted(
            set(
                int(dim)
                for dim in reduce_op.shard_dim
                if int(dim) < len(program.mesh.shape)
            )
        )
        shard_dims = [dim for dim in shard_dims if dim not in ring_dims]
        if len(shard_dims) == 0:
            continue

        participants = 1
        for mesh_dim in shard_dims:
            participants *= int(program.mesh.shape[mesh_dim])
        if participants <= 1:
            continue

        data_bytes = float(
            _buffer_numel(reduce_op.buffer) * get_element_size(reduce_op.buffer.dtype)
        )
        inter_node = any(
            _is_inter_mesh_dim(mesh_dim, topology_metadata) for mesh_dim in shard_dims
        )
        bandwidth, latency_s = _link_params(hw_config, inter_node)

        rounds = 2.0 * (participants - 1.0)
        transfer_factor = 2.0 * (participants - 1.0) / participants
        comm_s = rounds * latency_s + transfer_factor * (data_bytes / bandwidth)
        events.append(
            _CommEvent(
                time_ms=comm_s * 1000.0,
                buffer_name=reduce_op.buffer.tensor,
                overlaps_compute=False,
            )
        )

    return events


def _estimate_async_collective_pipeline_overhead_ms(
    program: Program,
    hw_config: Dict[str, Any],
    topology_metadata: Dict[str, List[int]],
    compute_time_ms: float,
) -> float:
    overhead_ms = 0.0
    memory_bandwidth = _memory_bandwidth_bytes_per_second(hw_config)

    pipeline_regions = program.visit(collect_pipeline_regions)
    legalized_regions = [r for r in pipeline_regions if r.legalized]

    for region in legalized_regions:
        reduce_op = region.reduce_op
        if reduce_op is None:
            continue

        ring_dims = set(int(comm.shard_dim) for comm in reduce_op.comm)
        shard_dims = sorted(
            set(
                int(dim)
                for dim in reduce_op.shard_dim
                if int(dim) < len(program.mesh.shape)
            )
        )
        shard_dims = [dim for dim in shard_dims if dim not in ring_dims]
        if len(shard_dims) == 0:
            continue

        participants = 1
        for mesh_dim in shard_dims:
            participants *= int(program.mesh.shape[mesh_dim])
        if participants <= 1:
            continue

        # Use materialized_overlap_axis tile count when available to ensure
        # the estimator uses the same realizable loop as codegen.
        mat_axis = getattr(region, "materialized_overlap_axis", None)
        if mat_axis is not None and int(mat_axis.size) <= int(mat_axis.min_block_size):
            continue
        if mat_axis is not None and int(mat_axis.size) > int(mat_axis.min_block_size):
            tile_count = max(
                1, int(mat_axis.size) // int(mat_axis.min_block_size)
            )
        else:
            tile_count = max(1, region.tile_count)
        if tile_count < 2:
            continue
        stage_count = max(2, region.stage_count)

        data_bytes = float(
            _buffer_numel(reduce_op.buffer) * get_element_size(reduce_op.buffer.dtype)
        )
        inter_node = any(
            _is_inter_mesh_dim(mesh_dim, topology_metadata) for mesh_dim in shard_dims
        )
        bandwidth, latency_s = _link_params(hw_config, inter_node)

        rounds = 2.0 * (participants - 1.0)
        transfer_factor = 2.0 * (participants - 1.0) / participants
        tile_comm_ms = (
            rounds * latency_s + transfer_factor * (data_bytes / bandwidth)
        ) * 1000.0
        tile_compute_ms = compute_time_ms / float(tile_count)

        warmup_ms = tile_compute_ms
        steady_state_ms = float(tile_count - 1) * max(tile_compute_ms, tile_comm_ms)
        drain_ms = tile_comm_ms
        pipeline_total_ms = warmup_ms + steady_state_ms + drain_ms

        baseline_compute_ms = float(tile_count) * tile_compute_ms
        pipeline_overhead_ms = max(0.0, pipeline_total_ms - baseline_compute_ms)

        extra_stage_bytes = float(stage_count - 1) * data_bytes
        memory_overhead_ms = (extra_stage_bytes / memory_bandwidth) * 1000.0

        overhead_ms += pipeline_overhead_ms + memory_overhead_ms

    return overhead_ms


def _estimate_ring_events(
    program: Program,
    hw_config: Dict[str, Any],
    topology_metadata: Dict[str, List[int]],
) -> List[_CommEvent]:
    events: List[_CommEvent] = []
    comm_nodes = _collect_comm_nodes(program)

    for node in comm_nodes:
        if not hasattr(node, "buffer") or not hasattr(node, "comm"):
            continue
        data_bytes = float(
            _buffer_numel(node.buffer) * get_element_size(node.buffer.dtype)
        )

        for ring_comm in node.comm:
            shard_dim = int(getattr(ring_comm, "shard_dim", 0))
            participants = int(getattr(ring_comm, "num_cards", 1))
            if 0 <= shard_dim < len(program.mesh.shape):
                participants = max(participants, int(program.mesh.shape[shard_dim]))
            if participants <= 1:
                continue

            inter_node = _is_inter_mesh_dim(shard_dim, topology_metadata)
            bandwidth, latency_s = _link_params(hw_config, inter_node)

            rounds = float(participants - 1)
            if isinstance(node, ReduceOp) and bool(
                getattr(ring_comm, "write_back", False)
            ):
                rounds += 1.0

            comm_s = rounds * (latency_s + (data_bytes / bandwidth))

            overlaps_compute = (
                isinstance(node, (BufferLoad, BufferStore)) and not node.buffer.write
            )
            events.append(
                _CommEvent(
                    time_ms=comm_s * 1000.0,
                    buffer_name=node.buffer.tensor,
                    overlaps_compute=overlaps_compute,
                )
            )

    return events


def _estimate_total_with_overlap(
    compute_time_ms: float,
    comm_events: List[_CommEvent],
    output_buffers: Set[str],
) -> float:
    comm_time_ms = sum(event.time_ms for event in comm_events)
    overlappable_comm_ms = 0.0
    for event in comm_events:
        if event.overlaps_compute and event.buffer_name not in output_buffers:
            overlappable_comm_ms += event.time_ms
    blocking_comm_ms = comm_time_ms - overlappable_comm_ms
    return blocking_comm_ms + max(compute_time_ms, overlappable_comm_ms)


def estimate_program(
    program: Program,
    hw_config: Dict[str, Any],
    num_inter_dims: Optional[int] = None,
) -> EstimateResult:
    """Estimate compute and communication time for one IR program."""
    from mercury.ir.legalization import prepare_pipeline

    prepare_pipeline(program)

    if program.mesh is None:
        raise ValueError("Program mesh is not initialized")

    axes = program.visit(collect_axis)
    if len(axes) == 0:
        raise ValueError("Program has no axes to estimate")

    _validate_hardware_config(hw_config)
    topology_metadata = _normalize_topology_metadata(program, num_inter_dims)

    compute_time_ms = _estimate_compute_time_ms(program, hw_config)
    comm_events = _estimate_collective_reduce_events(
        program, hw_config, topology_metadata
    )
    comm_events.extend(_estimate_ring_events(program, hw_config, topology_metadata))
    async_pipeline_overhead_ms = _estimate_async_collective_pipeline_overhead_ms(
        program,
        hw_config,
        topology_metadata,
        compute_time_ms,
    )

    comm_time_ms = (
        sum(event.time_ms for event in comm_events) + async_pipeline_overhead_ms
    )
    output_buffers = {
        buffer.tensor for buffer in _collect_unique_buffers(program) if buffer.write
    }
    total_time_ms = _estimate_total_with_overlap(
        compute_time_ms, comm_events, output_buffers
    )
    total_time_ms += async_pipeline_overhead_ms

    return EstimateResult(
        compute_time_ms=compute_time_ms,
        comm_time_ms=comm_time_ms,
        total_time_ms=total_time_ms,
    )
