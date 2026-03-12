# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""FFN-specific exact graph-level joint search."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from mercury.ir.distributed import DeviceMesh, ShardType
from mercury.ir.elements import Buffer
from mercury.ir.nodes import Program
from mercury.ir.utils import get_io_buffers
from mercury.search.estimate import estimate_program
from mercury.search.reshard_estimate import estimate_reshard_time


_FFN_OPERATORS = ("gate", "up", "down")


@dataclass
class FFNOperatorCandidate:
    """One FFN operator candidate with its estimated execution time."""

    operator_name: str
    candidate_id: int
    program: Program
    exec_time_ms: float
    buffer_a: Buffer
    buffer_c: Buffer


@dataclass
class FFNJointSearchResult:
    """Best FFN joint optimization result."""

    total_time_ms: float
    selected_programs: Dict[str, Program]
    selected_indices: Dict[str, int]
    selected_layouts: Dict[str, str]
    edge_costs_ms: Dict[str, float]
    exec_costs_ms: Dict[str, float]
    candidate_counts: Dict[str, int]


def _normalize_spec(spec: Union[ShardType, Tuple[ShardType, List[int]]]) -> Tuple[str, Tuple[int, ...]]:
    if isinstance(spec, tuple):
        if spec[0] != ShardType.SHARD:
            raise ValueError(f"Unsupported shard type in spec tuple: {spec[0]}")
        return ("S", tuple(int(v) for v in spec[1]))
    if spec == ShardType.REPLICATE:
        return ("R", tuple())
    raise ValueError(f"Unsupported shard spec: {spec}")


def _layout_signature(buffer: Buffer) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[str, Tuple[int, ...]], ...]]:
    if buffer.shard_spec is None:
        raise ValueError(f"Buffer {buffer.tensor} has no shard spec")

    mesh_shape = tuple(int(v) for v in buffer.shard_spec.mesh.shape)
    local_shape = tuple(int(v) for v in buffer.get_shape())
    specs = tuple(_normalize_spec(spec) for spec in buffer.shard_spec.specs)
    return (mesh_shape, local_shape, specs)


def _layout_summary(buffer: Buffer) -> str:
    if buffer.shard_spec is None:
        raise ValueError(f"Buffer {buffer.tensor} has no shard spec")

    specs = []
    for spec in buffer.shard_spec.specs:
        normalized = _normalize_spec(spec)
        if normalized[0] == "R":
            specs.append("R")
        else:
            specs.append("S(" + ",".join(str(v) for v in normalized[1]) + ")")
    mesh = tuple(int(v) for v in buffer.shard_spec.mesh.shape)
    local_shape = tuple(int(v) for v in buffer.get_shape())
    return f"mesh={mesh}, local_shape={local_shape}, specs=[{', '.join(specs)}]"


def _extract_matrix_buffers(program: Program) -> Dict[str, Buffer]:
    io_buffers = program.visit(get_io_buffers)
    matrix_buffers: Dict[str, Buffer] = {}
    for buffer in io_buffers:
        matrix_name = buffer.tensor.upper()
        if matrix_name in ("A", "B", "C") and matrix_name not in matrix_buffers:
            matrix_buffers[matrix_name] = buffer

    for matrix_name in ("A", "B", "C"):
        if matrix_name not in matrix_buffers:
            raise ValueError(f"Program '{program.name}' missing matrix buffer '{matrix_name}'")
        if matrix_buffers[matrix_name].shard_spec is None:
            raise ValueError(f"Program '{program.name}' matrix '{matrix_name}' has no shard spec")
    return matrix_buffers


def _build_candidates(
    operator_name: str,
    programs: List[Program],
    hw_config: Dict[str, Any],
) -> List[FFNOperatorCandidate]:
    candidates: List[FFNOperatorCandidate] = []
    for candidate_id, program in enumerate(programs):
        estimate = estimate_program(program, hw_config)
        matrix_buffers = _extract_matrix_buffers(program)
        candidates.append(
            FFNOperatorCandidate(
                operator_name=operator_name,
                candidate_id=candidate_id,
                program=program,
                exec_time_ms=estimate.total_time_ms,
                buffer_a=matrix_buffers["A"],
                buffer_c=matrix_buffers["C"],
            )
        )
    if len(candidates) == 0:
        raise ValueError(f"Operator '{operator_name}' has no candidates")
    return candidates


def _best_candidate(
    candidates: List[FFNOperatorCandidate],
    score_fn,
) -> Tuple[int, float]:
    best_idx = 0
    best_score = float("inf")
    for idx, candidate in enumerate(candidates):
        score = float(score_fn(candidate))
        if score < best_score:
            best_idx = idx
            best_score = score
    return best_idx, best_score


def search_ffn(
    candidate_programs: Dict[str, List[Program]],
    hw_config: Dict[str, Any],
    origin_mesh: DeviceMesh,
) -> FFNJointSearchResult:
    """Find the exact FFN joint optimum over shared L_in/L_mid layout nodes."""
    missing = [operator for operator in _FFN_OPERATORS if operator not in candidate_programs]
    if len(missing) > 0:
        raise ValueError(f"Missing FFN operator candidates: {', '.join(missing)}")

    gate_candidates = _build_candidates("gate", candidate_programs["gate"], hw_config)
    up_candidates = _build_candidates("up", candidate_programs["up"], hw_config)
    down_candidates = _build_candidates("down", candidate_programs["down"], hw_config)

    l_in_layouts: Dict[
        Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[str, Tuple[int, ...]], ...]],
        Buffer,
    ] = {}
    for candidate in gate_candidates + up_candidates:
        l_in_layouts[_layout_signature(candidate.buffer_a)] = candidate.buffer_a

    l_mid_layouts: Dict[
        Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[str, Tuple[int, ...]], ...]],
        Buffer,
    ] = {}
    for candidate in gate_candidates + up_candidates:
        l_mid_layouts[_layout_signature(candidate.buffer_c)] = candidate.buffer_c
    for candidate in down_candidates:
        l_mid_layouts[_layout_signature(candidate.buffer_a)] = candidate.buffer_a

    edge_cache: Dict[
        Tuple[
            Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[str, Tuple[int, ...]], ...]],
            Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[str, Tuple[int, ...]], ...]],
        ],
        float,
    ] = {}

    def edge_cost(src_buffer: Buffer, dst_buffer: Buffer) -> float:
        src_sig = _layout_signature(src_buffer)
        dst_sig = _layout_signature(dst_buffer)
        cache_key = (src_sig, dst_sig)
        if cache_key not in edge_cache:
            edge_cache[cache_key] = estimate_reshard_time(
                src_buffer,
                dst_buffer,
                hw_config,
                origin_mesh,
            )
        return edge_cache[cache_key]

    best_total = float("inf")
    best_result: Optional[FFNJointSearchResult] = None

    for l_in_sig, l_in_buffer in l_in_layouts.items():
        for l_mid_sig, l_mid_buffer in l_mid_layouts.items():
            gate_idx, gate_score = _best_candidate(
                gate_candidates,
                lambda candidate: (
                    candidate.exec_time_ms
                    + edge_cost(l_in_buffer, candidate.buffer_a)
                    + edge_cost(candidate.buffer_c, l_mid_buffer)
                ),
            )
            up_idx, up_score = _best_candidate(
                up_candidates,
                lambda candidate: (
                    candidate.exec_time_ms
                    + edge_cost(l_in_buffer, candidate.buffer_a)
                    + edge_cost(candidate.buffer_c, l_mid_buffer)
                ),
            )
            down_idx, down_score = _best_candidate(
                down_candidates,
                lambda candidate: (
                    edge_cost(l_mid_buffer, candidate.buffer_a) + candidate.exec_time_ms
                ),
            )

            total_score = gate_score + up_score + down_score
            if total_score >= best_total:
                continue

            best_total = total_score
            selected_gate = gate_candidates[gate_idx]
            selected_up = up_candidates[up_idx]
            selected_down = down_candidates[down_idx]

            best_result = FFNJointSearchResult(
                total_time_ms=total_score,
                selected_programs={
                    "gate": selected_gate.program,
                    "up": selected_up.program,
                    "down": selected_down.program,
                },
                selected_indices={
                    "gate": selected_gate.candidate_id,
                    "up": selected_up.candidate_id,
                    "down": selected_down.candidate_id,
                },
                selected_layouts={
                    "L_in": _layout_summary(l_in_layouts[l_in_sig]),
                    "L_mid": _layout_summary(l_mid_layouts[l_mid_sig]),
                },
                edge_costs_ms={
                    "L_in->gate.A": edge_cost(l_in_buffer, selected_gate.buffer_a),
                    "gate.C->L_mid": edge_cost(selected_gate.buffer_c, l_mid_buffer),
                    "L_in->up.A": edge_cost(l_in_buffer, selected_up.buffer_a),
                    "up.C->L_mid": edge_cost(selected_up.buffer_c, l_mid_buffer),
                    "L_mid->down.A": edge_cost(l_mid_buffer, selected_down.buffer_a),
                },
                exec_costs_ms={
                    "gate": selected_gate.exec_time_ms,
                    "up": selected_up.exec_time_ms,
                    "down": selected_down.exec_time_ms,
                },
                candidate_counts={
                    "gate": len(gate_candidates),
                    "up": len(up_candidates),
                    "down": len(down_candidates),
                },
            )

    if best_result is None:
        raise ValueError("Failed to find a valid FFN joint solution")
    return best_result
