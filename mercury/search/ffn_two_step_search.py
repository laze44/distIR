# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Two-step FFN search with boundary-layout step-1 and constrained step-2."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mercury.backend import generate_pytorch_code
from mercury.ir.elements import Buffer
from mercury.ir.nodes import Program
from mercury.ir.utils import get_io_buffers
from mercury.search.estimate import estimate_program
from mercury.search.mapping_constraints import (
    LogicalBoundaryLayoutSignature,
    LogicalTensorLayoutConstraints,
    OperatorTensorMappingConstraints,
    logical_layout_signature_equal,
    logical_layout_signature_from_buffer,
    program_satisfies_logical_layout_constraints,
)
from mercury.search.reshard_estimate import estimate_reshard_time_from_logical_layout
from mercury.search.search import search_with_progress


_FFN_OPERATORS = ("gate", "up", "down")


@dataclass(frozen=True)
class EdgeReshardTransition:
    """Explicit graph edge transition between two logical boundary layouts."""

    edge_name: str
    src_layout: LogicalBoundaryLayoutSignature
    dst_layout: LogicalBoundaryLayoutSignature
    cost_ms: float


@dataclass(frozen=True)
class FFNLayoutPlan:
    """One step-1 FFN graph plan with operator-local and edge costs."""

    activation_layouts: Dict[str, LogicalBoundaryLayoutSignature]
    weight_layouts: Dict[str, LogicalBoundaryLayoutSignature]
    operator_layouts: Dict[str, Dict[str, LogicalBoundaryLayoutSignature]]
    edge_transitions: List[EdgeReshardTransition]
    step1_operator_costs_ms: Dict[str, float]
    step1_edge_costs_ms: Dict[str, float]
    step1_total_time_ms: float


@dataclass
class FFNTwoStepSearchResult:
    """Final FFN result after step-1 ranking and step-2 rerun."""

    selected_plan: FFNLayoutPlan
    ranked_plans: List[FFNLayoutPlan]
    selected_programs: Dict[str, Program]
    selected_indices: Dict[str, int]
    step2_operator_costs_ms: Dict[str, float]
    total_time_ms: float
    candidate_counts: Dict[str, int]


@dataclass(frozen=True)
class _FFNOperatorCandidate:
    """One operator candidate annotated with logical A/B/C layouts."""

    operator_name: str
    candidate_id: int
    program: Program
    total_time_ms: float
    layout_a: LogicalBoundaryLayoutSignature
    layout_b: LogicalBoundaryLayoutSignature
    layout_c: LogicalBoundaryLayoutSignature


@dataclass(frozen=True)
class _OperatorStep2Result:
    """Best step-2 lowering for one logical operator boundary."""

    candidate_id: int
    program: Program
    total_time_ms: float


def _extract_matrix_buffers(program: Program) -> Dict[str, Buffer]:
    matrix_buffers: Dict[str, Buffer] = {}
    for buffer in program.visit(get_io_buffers):
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
    programs: Iterable[Program],
    hw_config: Dict[str, Any],
) -> List[_FFNOperatorCandidate]:
    candidates: List[_FFNOperatorCandidate] = []
    for candidate_id, program in enumerate(programs):
        estimate = estimate_program(program, hw_config)
        matrix_buffers = _extract_matrix_buffers(program)
        candidates.append(
            _FFNOperatorCandidate(
                operator_name=operator_name,
                candidate_id=candidate_id,
                program=program,
                total_time_ms=estimate.total_time_ms,
                layout_a=logical_layout_signature_from_buffer(matrix_buffers["A"]),
                layout_b=logical_layout_signature_from_buffer(matrix_buffers["B"]),
                layout_c=logical_layout_signature_from_buffer(matrix_buffers["C"]),
            )
        )
    if len(candidates) == 0:
        raise ValueError(f"Operator '{operator_name}' has no candidates")
    return candidates


def _edge_cost(
    edge_name: str,
    src_layout: LogicalBoundaryLayoutSignature,
    dst_layout: LogicalBoundaryLayoutSignature,
    edge_cache: Dict[
        Tuple[LogicalBoundaryLayoutSignature, LogicalBoundaryLayoutSignature],
        float,
    ],
    hw_config: Dict[str, Any],
    origin_mesh,
) -> Tuple[float, Optional[EdgeReshardTransition]]:
    if logical_layout_signature_equal(src_layout, dst_layout):
        return 0.0, None

    cache_key = (src_layout, dst_layout)
    if cache_key not in edge_cache:
        edge_cache[cache_key] = estimate_reshard_time_from_logical_layout(
            src_layout,
            dst_layout,
            hw_config,
            origin_mesh,
        )
    cost_ms = edge_cache[cache_key]
    return cost_ms, EdgeReshardTransition(
        edge_name=edge_name,
        src_layout=src_layout,
        dst_layout=dst_layout,
        cost_ms=cost_ms,
    )


def _best_operator_candidate(
    operator_name: str,
    candidates: List[_FFNOperatorCandidate],
    l_in_layout: LogicalBoundaryLayoutSignature,
    l_mid_layout: LogicalBoundaryLayoutSignature,
    edge_cache: Dict[
        Tuple[LogicalBoundaryLayoutSignature, LogicalBoundaryLayoutSignature],
        float,
    ],
    hw_config: Dict[str, Any],
    origin_mesh,
) -> Tuple[_FFNOperatorCandidate, List[EdgeReshardTransition], Dict[str, float], float]:
    best_tuple = None

    for candidate in candidates:
        transitions: List[EdgeReshardTransition] = []
        edge_costs: Dict[str, float] = {}

        if operator_name in ("gate", "up"):
            lhs_name = f"L_in->{operator_name}.A"
            rhs_name = f"{operator_name}.C->L_mid"

            lhs_cost, lhs_transition = _edge_cost(
                lhs_name,
                l_in_layout,
                candidate.layout_a,
                edge_cache,
                hw_config,
                origin_mesh,
            )
            rhs_cost, rhs_transition = _edge_cost(
                rhs_name,
                candidate.layout_c,
                l_mid_layout,
                edge_cache,
                hw_config,
                origin_mesh,
            )
            edge_costs[lhs_name] = lhs_cost
            edge_costs[rhs_name] = rhs_cost
            if lhs_transition is not None:
                transitions.append(lhs_transition)
            if rhs_transition is not None:
                transitions.append(rhs_transition)
            total_score = candidate.total_time_ms + lhs_cost + rhs_cost
        elif operator_name == "down":
            edge_name = "L_mid->down.A"
            edge_ms, transition = _edge_cost(
                edge_name,
                l_mid_layout,
                candidate.layout_a,
                edge_cache,
                hw_config,
                origin_mesh,
            )
            edge_costs[edge_name] = edge_ms
            if transition is not None:
                transitions.append(transition)
            total_score = candidate.total_time_ms + edge_ms
        else:
            raise ValueError(f"Unsupported operator '{operator_name}'")

        ordering_key = (
            total_score,
            candidate.total_time_ms,
            candidate.layout_a.to_summary(),
            candidate.layout_b.to_summary(),
            candidate.layout_c.to_summary(),
            candidate.candidate_id,
        )
        if best_tuple is None or ordering_key < best_tuple[0]:
            best_tuple = (ordering_key, candidate, transitions, edge_costs, total_score)

    assert best_tuple is not None
    return best_tuple[1], best_tuple[2], best_tuple[3], float(best_tuple[4])


def _step1_plan_order_key(plan: FFNLayoutPlan) -> Tuple[str, ...]:
    return (
        plan.activation_layouts["L_in"].to_summary(),
        plan.activation_layouts["L_mid"].to_summary(),
        plan.activation_layouts["L_out"].to_summary(),
        plan.weight_layouts["W_gate"].to_summary(),
        plan.weight_layouts["W_up"].to_summary(),
        plan.weight_layouts["W_down"].to_summary(),
        plan.operator_layouts["gate"]["A"].to_summary(),
        plan.operator_layouts["gate"]["C"].to_summary(),
        plan.operator_layouts["up"]["A"].to_summary(),
        plan.operator_layouts["up"]["C"].to_summary(),
        plan.operator_layouts["down"]["A"].to_summary(),
    )


def _top_plans_from_candidates(
    candidate_programs: Dict[str, List[Program]],
    hw_config: Dict[str, Any],
    origin_mesh,
    layout_top_k: int,
) -> Tuple[List[FFNLayoutPlan], Dict[str, int]]:
    missing = [operator for operator in _FFN_OPERATORS if operator not in candidate_programs]
    if len(missing) > 0:
        raise ValueError(f"Missing FFN operator candidates: {', '.join(missing)}")
    if layout_top_k <= 0:
        raise ValueError("layout_top_k must be a positive integer")

    gate_candidates = _build_candidates("gate", candidate_programs["gate"], hw_config)
    up_candidates = _build_candidates("up", candidate_programs["up"], hw_config)
    down_candidates = _build_candidates("down", candidate_programs["down"], hw_config)

    l_in_layouts = {
        candidate.layout_a
        for candidate in gate_candidates + up_candidates
    }
    l_mid_layouts = {
        candidate.layout_c
        for candidate in gate_candidates + up_candidates
    }.union({candidate.layout_a for candidate in down_candidates})

    edge_cache: Dict[
        Tuple[LogicalBoundaryLayoutSignature, LogicalBoundaryLayoutSignature],
        float,
    ] = {}
    all_plans: List[FFNLayoutPlan] = []

    for l_in_layout in l_in_layouts:
        for l_mid_layout in l_mid_layouts:
            gate_best = _best_operator_candidate(
                "gate",
                gate_candidates,
                l_in_layout,
                l_mid_layout,
                edge_cache,
                hw_config,
                origin_mesh,
            )
            up_best = _best_operator_candidate(
                "up",
                up_candidates,
                l_in_layout,
                l_mid_layout,
                edge_cache,
                hw_config,
                origin_mesh,
            )
            down_best = _best_operator_candidate(
                "down",
                down_candidates,
                l_in_layout,
                l_mid_layout,
                edge_cache,
                hw_config,
                origin_mesh,
            )

            gate_candidate, gate_edges, gate_edge_costs, _ = gate_best
            up_candidate, up_edges, up_edge_costs, _ = up_best
            down_candidate, down_edges, down_edge_costs, _ = down_best

            step1_operator_costs_ms = {
                "gate": gate_candidate.total_time_ms,
                "up": up_candidate.total_time_ms,
                "down": down_candidate.total_time_ms,
            }
            step1_edge_costs_ms = {
                **gate_edge_costs,
                **up_edge_costs,
                **down_edge_costs,
            }
            step1_total = sum(step1_operator_costs_ms.values()) + sum(
                step1_edge_costs_ms.values()
            )

            all_plans.append(
                FFNLayoutPlan(
                    activation_layouts={
                        "L_in": l_in_layout,
                        "L_mid": l_mid_layout,
                        "L_out": down_candidate.layout_c,
                    },
                    weight_layouts={
                        "W_gate": gate_candidate.layout_b,
                        "W_up": up_candidate.layout_b,
                        "W_down": down_candidate.layout_b,
                    },
                    operator_layouts={
                        "gate": {
                            "A": gate_candidate.layout_a,
                            "B": gate_candidate.layout_b,
                            "C": gate_candidate.layout_c,
                        },
                        "up": {
                            "A": up_candidate.layout_a,
                            "B": up_candidate.layout_b,
                            "C": up_candidate.layout_c,
                        },
                        "down": {
                            "A": down_candidate.layout_a,
                            "B": down_candidate.layout_b,
                            "C": down_candidate.layout_c,
                        },
                    },
                    edge_transitions=gate_edges + up_edges + down_edges,
                    step1_operator_costs_ms=step1_operator_costs_ms,
                    step1_edge_costs_ms=step1_edge_costs_ms,
                    step1_total_time_ms=step1_total,
                )
            )

    if len(all_plans) == 0:
        raise ValueError("Failed to find a valid FFN layout plan")

    all_plans.sort(key=lambda plan: (plan.step1_total_time_ms, _step1_plan_order_key(plan)))
    return all_plans[:layout_top_k], {
        "gate": len(gate_candidates),
        "up": len(up_candidates),
        "down": len(down_candidates),
    }


def _operator_constraints_for_plan(
    operator_name: str,
    plan: FFNLayoutPlan,
) -> LogicalTensorLayoutConstraints:
    operator_layout = plan.operator_layouts.get(operator_name)
    if operator_layout is None:
        raise ValueError(f"Unsupported FFN operator '{operator_name}'")
    return LogicalTensorLayoutConstraints(matrices=operator_layout)


def _step2_memo_key(
    operator_name: str,
    constraints: LogicalTensorLayoutConstraints,
) -> Tuple[str, Tuple[Tuple[str, LogicalBoundaryLayoutSignature], ...]]:
    return (
        operator_name,
        tuple(sorted(constraints.matrices.items(), key=lambda item: item[0])),
    )


def search_ffn_two_step(
    operator_programs: Dict[str, Program],
    origin_mesh,
    split_axis_names: List[str],
    hw_config: Dict[str, Any],
    tensor_mapping_constraints: Optional[OperatorTensorMappingConstraints] = None,
    layout_top_k: int = 10,
    candidate_programs: Optional[Dict[str, List[Program]]] = None,
    show_progress: bool = False,
) -> FFNTwoStepSearchResult:
    """Run FFN two-step search with explicit edge-reshard-aware step-1 ranking."""
    missing = [operator for operator in _FFN_OPERATORS if operator not in operator_programs]
    if len(missing) > 0:
        raise ValueError(f"Missing FFN operator programs: {', '.join(missing)}")

    if candidate_programs is None:
        candidate_programs = {}
        for operator_name in _FFN_OPERATORS:
            operator_constraints = None
            if tensor_mapping_constraints is not None:
                operator_constraints = tensor_mapping_constraints.get(operator_name)
            candidate_programs[operator_name] = list(
                search_with_progress(
                    operator_programs[operator_name],
                    origin_mesh,
                    split_axis_names,
                    tensor_mapping_constraints=operator_constraints,
                    progress_desc=f"step1[{operator_name}]",
                    show_progress=show_progress,
                    miniters=32,
                    mininterval=0.5,
                )
            )

    ranked_plans, candidate_counts = _top_plans_from_candidates(
        candidate_programs,
        hw_config,
        origin_mesh,
        layout_top_k,
    )

    step2_cache: Dict[
        Tuple[str, Tuple[Tuple[str, LogicalBoundaryLayoutSignature], ...]],
        _OperatorStep2Result,
    ] = {}
    best_result: Optional[FFNTwoStepSearchResult] = None
    best_total_time_ms = float("inf")

    for plan in ranked_plans:
        selected_programs: Dict[str, Program] = {}
        selected_indices: Dict[str, int] = {}
        step2_operator_costs_ms: Dict[str, float] = {}
        supported = True

        for operator_name in _FFN_OPERATORS:
            operator_constraints = None
            if tensor_mapping_constraints is not None:
                operator_constraints = tensor_mapping_constraints.get(operator_name)
            boundary_constraints = _operator_constraints_for_plan(operator_name, plan)
            cache_key = _step2_memo_key(operator_name, boundary_constraints)
            cached_result = step2_cache.get(cache_key)
            if cached_result is None:
                filtered_candidates = list(
                    search_with_progress(
                        operator_programs[operator_name],
                        origin_mesh,
                        split_axis_names,
                        tensor_mapping_constraints=operator_constraints,
                        program_filter=lambda program, constraints=boundary_constraints: (
                            program_satisfies_logical_layout_constraints(program, constraints)
                        ),
                        progress_desc=f"step2[{operator_name}]",
                        show_progress=show_progress,
                        miniters=32,
                        mininterval=0.5,
                    )
                )
                if len(filtered_candidates) == 0:
                    supported = False
                    break

                ranked_candidates = []
                for candidate_id, candidate_program in enumerate(filtered_candidates):
                    estimate = estimate_program(candidate_program, hw_config)
                    ranked_candidates.append(
                        (
                            estimate.total_time_ms,
                            generate_pytorch_code(candidate_program),
                            candidate_id,
                            candidate_program,
                        )
                    )
                ranked_candidates.sort(key=lambda item: (item[0], item[1]))
                best_candidate = ranked_candidates[0]
                cached_result = _OperatorStep2Result(
                    candidate_id=best_candidate[2],
                    program=best_candidate[3],
                    total_time_ms=best_candidate[0],
                )
                step2_cache[cache_key] = cached_result

            selected_programs[operator_name] = cached_result.program
            selected_indices[operator_name] = cached_result.candidate_id
            step2_operator_costs_ms[operator_name] = cached_result.total_time_ms

        if not supported:
            continue

        total_time_ms = sum(step2_operator_costs_ms.values())
        if total_time_ms >= best_total_time_ms:
            continue

        best_total_time_ms = total_time_ms
        best_result = FFNTwoStepSearchResult(
            selected_plan=plan,
            ranked_plans=ranked_plans,
            selected_programs=selected_programs,
            selected_indices=selected_indices,
            step2_operator_costs_ms=step2_operator_costs_ms,
            total_time_ms=total_time_ms,
            candidate_counts=candidate_counts,
        )

    if best_result is None:
        raise ValueError("Failed to find a valid FFN two-step solution")
    return best_result
