# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Theoretical performance estimation based on roofline and communication models."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from mercury.ir.calculate_memory import get_buffer_size
from mercury.ir.elements import Axis, Buffer
from mercury.ir.nodes import BufferLoad, BufferStore, Program, ReduceOp
from mercury.ir.utils import collect_axis, collect_reduce, get_element_size


_DTYPE_TO_CONFIG_KEY: Dict[torch.dtype, str] = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
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
        expected_total = self.compute_time_ms + self.comm_time_ms
        if abs(self.total_time_ms - expected_total) > 1e-9:
            raise ValueError("total_time_ms must equal compute_time_ms + comm_time_ms")


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


def _validate_hardware_config(config: Dict[str, Any]) -> None:
    name = _get_required(config, ["name"])
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Config field must be a non-empty string: name")

    _read_positive(config, ["compute", "peak_tflops", "bf16"])
    _read_positive(config, ["compute", "peak_tflops", "fp16"])
    _read_positive(config, ["compute", "peak_tflops", "fp32"])

    peak_tflops = config["compute"]["peak_tflops"]
    if "tf32" in peak_tflops:
        _read_positive(config, ["compute", "peak_tflops", "tf32"])

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


def _first_output_buffer(program: Program) -> Optional[Buffer]:
    buffers = program.visit(lambda node: node.buffer if hasattr(node, "buffer") and isinstance(node.buffer, Buffer) else None)
    for buffer in buffers:
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
            k_local *= int(axis.min_block_size)
        break

    if k_local is None:
        all_axes = program.visit(collect_axis)
        for axis in all_axes:
            if axis.name.upper().startswith("K"):
                k_local = int(axis.min_block_size)
                break

    if k_local is None:
        raise ValueError("Unable to infer K dimension from program")

    return [m_local, n_local, k_local]


def _peak_flops_per_second(hw_config: Dict[str, Any], dtype: torch.dtype) -> float:
    dtype_key = _DTYPE_TO_CONFIG_KEY.get(dtype, "bf16")
    peak_tflops = _read_positive(hw_config, ["compute", "peak_tflops", dtype_key])
    return peak_tflops * (10 ** 12)


def _memory_bandwidth_bytes_per_second(hw_config: Dict[str, Any]) -> float:
    bandwidth_tb_per_s = _read_positive(hw_config, ["memory", "bandwidth_tb_per_s"])
    return bandwidth_tb_per_s * (10 ** 12)


def _estimate_compute_time_ms(program: Program, hw_config: Dict[str, Any]) -> float:
    m_local, n_local, k_local = _extract_local_mnk(program)
    flops = 2.0 * float(m_local) * float(n_local) * float(k_local)

    output_buffer = _first_output_buffer(program)
    if output_buffer is None:
        raise ValueError("Program has no writable output buffer")

    compute_bound_s = flops / _peak_flops_per_second(hw_config, output_buffer.dtype)
    memory_bytes = float(get_buffer_size(program))
    memory_bound_s = memory_bytes / _memory_bandwidth_bytes_per_second(hw_config)
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


def _link_params(hw_config: Dict[str, Any], inter_node: bool) -> List[float]:
    if inter_node:
        bandwidth = _read_positive(hw_config, ["interconnect", "inter_node", "bandwidth_gb_per_s"]) * (10 ** 9)
        latency_s = _read_non_negative(hw_config, ["interconnect", "inter_node", "latency_us"]) / (10 ** 6)
    else:
        bandwidth = _read_positive(hw_config, ["interconnect", "intra_node", "bandwidth_gb_per_s"]) * (10 ** 9)
        latency_s = _read_non_negative(hw_config, ["interconnect", "intra_node", "latency_us"]) / (10 ** 6)
    return [bandwidth, latency_s]


def _estimate_reduce_comm_ms(program: Program, hw_config: Dict[str, Any], num_inter_dims: int) -> float:
    comm_s = 0.0
    reduce_ops = program.visit(collect_reduce)

    for reduce_op in reduce_ops:
        if len(reduce_op.shard_dim) == 0:
            continue

        data_bytes = float(_buffer_numel(reduce_op.buffer) * get_element_size(reduce_op.buffer.dtype))
        for mesh_dim in reduce_op.shard_dim:
            if mesh_dim >= len(program.mesh.shape):
                continue
            participants = int(program.mesh.shape[mesh_dim])
            if participants <= 1:
                continue
            inter_node = mesh_dim < num_inter_dims
            bandwidth, latency_s = _link_params(hw_config, inter_node)
            comm_s += latency_s + (2.0 * (participants - 1.0) / participants) * (data_bytes / bandwidth)

    return comm_s * 1000.0


def _estimate_ring_comm_ms(program: Program, hw_config: Dict[str, Any], num_inter_dims: int) -> float:
    comm_s = 0.0
    comm_nodes = _collect_comm_nodes(program)

    for node in comm_nodes:
        if not hasattr(node, "buffer") or not hasattr(node, "comm"):
            continue
        data_bytes = float(_buffer_numel(node.buffer) * get_element_size(node.buffer.dtype))
        for ring_comm in node.comm:
            participants = int(getattr(ring_comm, "num_cards", 1))
            if participants <= 1:
                continue

            shard_dim = int(getattr(ring_comm, "shard_dim", 0))
            inter_node = shard_dim < num_inter_dims
            bandwidth, latency_s = _link_params(hw_config, inter_node)

            comm_name = str(getattr(ring_comm, "name", "")).lower()
            if "all_reduce" in comm_name or "allreduce" in comm_name:
                factor = 2.0 * (participants - 1.0) / participants
            elif "all_gather" in comm_name or "allgather" in comm_name:
                factor = (participants - 1.0) / participants
            elif "reduce_scatter" in comm_name or "reducescatter" in comm_name:
                factor = (participants - 1.0) / participants
            else:
                factor = 1.0

            comm_s += latency_s + factor * (data_bytes / bandwidth)

    return comm_s * 1000.0


def estimate_program(program: Program, hw_config: Dict[str, Any], num_inter_dims: int = 1) -> EstimateResult:
    """Estimate compute and communication time for one IR program."""
    if program.mesh is None:
        raise ValueError("Program mesh is not initialized")

    axes = program.visit(collect_axis)
    if len(axes) == 0:
        raise ValueError("Program has no axes to estimate")

    _validate_hardware_config(hw_config)

    compute_time_ms = _estimate_compute_time_ms(program, hw_config)
    comm_time_ms = _estimate_reduce_comm_ms(program, hw_config, num_inter_dims)
    comm_time_ms += _estimate_ring_comm_ms(program, hw_config, num_inter_dims)
    total_time_ms = compute_time_ms + comm_time_ms

    return EstimateResult(
        compute_time_ms=compute_time_ms,
        comm_time_ms=comm_time_ms,
        total_time_ms=total_time_ms,
    )
