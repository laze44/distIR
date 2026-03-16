# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Two-step FFN search with boundary-layout step-1 and segment-aware step-2."""

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
_DOWN_EDGE_NAME = "L_mid->down.A"
_OWNER_STANDALONE_RESHARD = "standalone_reshard_segment"
_OWNER_CONSUMER_SEGMENT = "consumer_segment"
_DOWN_ISOLATED_STRATEGY = "isolated_operator"
_DOWN_CONSUMER_FUSED_STRATEGY = "consumer_chain_edge_consumer"


@dataclass(frozen=True)
class EdgeReshardTransition:
    """Explicit graph edge transition between two logical boundary layouts."""

    edge_name: str
    src_layout: LogicalBoundaryLayoutSignature
    dst_layout: LogicalBoundaryLayoutSignature
    cost_ms: float


@dataclass(frozen=True)
class FFNEdgeLoweringOwnership:
    """Final step-2 ownership decision for one explicit edge obligation."""

    edge_name: str
    owner_kind: str
    owner_segment_id: str
    materialized_as_standalone: bool
    cost_ms: float


@dataclass(frozen=True)
class FFNSegmentSelection:
    """Selected executable segment with logical/materialized boundary metadata."""

    segment_id: str
    segment_kind: str
    logical_operators: Tuple[str, ...]
    logical_boundary_obligations: Tuple[str, ...]
    materialized_boundaries: Tuple[str, ...]
    selected_program: Optional[Program]
    selected_candidate_id: Optional[int]
    total_time_ms: float
    owned_edge_obligations: Tuple[str, ...]


@dataclass(frozen=True)
class FFNOperatorBoundaryClass:
    """One exact logical (A, B, C) boundary class for an FFN operator."""

    operator_name: str
    layout_a: LogicalBoundaryLayoutSignature
    layout_b: LogicalBoundaryLayoutSignature
    layout_c: LogicalBoundaryLayoutSignature
    candidate_count: int
    representative_candidate_ids: Tuple[int, ...]
    best_step1_exec_time_ms: float


@dataclass(frozen=True)
class FFNLayoutPlan:
    """One step-1 FFN graph plan with canonical boundary classes and projections."""

    activation_layouts: Dict[str, LogicalBoundaryLayoutSignature]
    weight_layouts: Dict[str, LogicalBoundaryLayoutSignature]
    boundary_classes: Dict[str, FFNOperatorBoundaryClass]
    operator_layouts: Dict[str, Dict[str, LogicalBoundaryLayoutSignature]]
    edge_transitions: List[EdgeReshardTransition]
    step1_operator_costs_ms: Dict[str, float]
    step1_edge_costs_ms: Dict[str, float]
    step1_total_time_ms: float


@dataclass(frozen=True)
class FFNStep1LayoutStats:
    """Summary counts for the current step-1 layout enumeration strategy."""

    unique_l_in_count: int
    unique_l_mid_count: int
    gate_boundary_class_count: int
    up_boundary_class_count: int
    down_boundary_class_count: int
    projected_l_out_count: int
    projected_w_gate_count: int
    projected_w_up_count: int
    projected_w_down_count: int
    total_plan_count: int


@dataclass
class FFNTwoStepSearchResult:
    """Final FFN result after step-1 ranking and step-2 rerun."""

    selected_plan: FFNLayoutPlan
    ranked_plans: List[FFNLayoutPlan]
    selected_segments: List[FFNSegmentSelection]
    explicit_edge_obligations: Dict[str, EdgeReshardTransition]
    edge_ownership: Dict[str, FFNEdgeLoweringOwnership]
    selected_programs: Dict[str, Program]
    selected_indices: Dict[str, int]
    step2_operator_costs_ms: Dict[str, float]
    step2_segment_costs_ms: Dict[str, float]
    total_time_ms: float
    candidate_counts: Dict[str, int]
    step1_layout_stats: FFNStep1LayoutStats


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


@dataclass
class _FFNStep2PlanCandidate:
    """One concrete step-2 realization for a fixed step-1 plan."""

    selected_segments: List[FFNSegmentSelection]
    explicit_edge_obligations: Dict[str, EdgeReshardTransition]
    edge_ownership: Dict[str, FFNEdgeLoweringOwnership]
    selected_programs: Dict[str, Program]
    selected_indices: Dict[str, int]
    step2_operator_costs_ms: Dict[str, float]
    step2_segment_costs_ms: Dict[str, float]
    total_time_ms: float
    strategy_rank: int


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


def _layout_order_key(layout: LogicalBoundaryLayoutSignature) -> str:
    return layout.to_summary()


def _boundary_class_order_key(
    boundary_class: FFNOperatorBoundaryClass,
) -> Tuple[str, str, str]:
    return (
        boundary_class.layout_a.to_summary(),
        boundary_class.layout_b.to_summary(),
        boundary_class.layout_c.to_summary(),
    )


def _group_operator_boundary_classes(
    operator_name: str,
    candidates: List[_FFNOperatorCandidate],
) -> List[FFNOperatorBoundaryClass]:
    grouped: Dict[
        Tuple[
            LogicalBoundaryLayoutSignature,
            LogicalBoundaryLayoutSignature,
            LogicalBoundaryLayoutSignature,
        ],
        List[_FFNOperatorCandidate],
    ] = {}
    for candidate in candidates:
        signature = (candidate.layout_a, candidate.layout_b, candidate.layout_c)
        grouped.setdefault(signature, []).append(candidate)

    boundary_classes: List[FFNOperatorBoundaryClass] = []
    for signature, grouped_candidates in grouped.items():
        layout_a, layout_b, layout_c = signature
        candidate_ids = tuple(
            sorted(candidate.candidate_id for candidate in grouped_candidates)
        )
        boundary_classes.append(
            FFNOperatorBoundaryClass(
                operator_name=operator_name,
                layout_a=layout_a,
                layout_b=layout_b,
                layout_c=layout_c,
                candidate_count=len(grouped_candidates),
                representative_candidate_ids=candidate_ids,
                best_step1_exec_time_ms=min(
                    candidate.total_time_ms for candidate in grouped_candidates
                ),
            )
        )

    boundary_classes.sort(key=_boundary_class_order_key)
    return boundary_classes


def _step1_plan_order_key(plan: FFNLayoutPlan) -> Tuple[str, ...]:
    gate_class = plan.boundary_classes["gate"]
    up_class = plan.boundary_classes["up"]
    down_class = plan.boundary_classes["down"]
    return (
        plan.activation_layouts["L_in"].to_summary(),
        plan.activation_layouts["L_mid"].to_summary(),
        gate_class.layout_a.to_summary(),
        gate_class.layout_b.to_summary(),
        gate_class.layout_c.to_summary(),
        up_class.layout_a.to_summary(),
        up_class.layout_b.to_summary(),
        up_class.layout_c.to_summary(),
        down_class.layout_a.to_summary(),
        down_class.layout_b.to_summary(),
        down_class.layout_c.to_summary(),
        plan.activation_layouts["L_out"].to_summary(),
        plan.weight_layouts["W_gate"].to_summary(),
        plan.weight_layouts["W_up"].to_summary(),
        plan.weight_layouts["W_down"].to_summary(),
    )


def _top_plans_from_candidates(
    candidate_programs: Dict[str, List[Program]],
    hw_config: Dict[str, Any],
    origin_mesh,
    layout_top_k: int,
) -> Tuple[List[FFNLayoutPlan], Dict[str, int], FFNStep1LayoutStats]:
    missing = [operator for operator in _FFN_OPERATORS if operator not in candidate_programs]
    if len(missing) > 0:
        raise ValueError(f"Missing FFN operator candidates: {', '.join(missing)}")
    if layout_top_k <= 0:
        raise ValueError("layout_top_k must be a positive integer")

    gate_candidates = _build_candidates("gate", candidate_programs["gate"], hw_config)
    up_candidates = _build_candidates("up", candidate_programs["up"], hw_config)
    down_candidates = _build_candidates("down", candidate_programs["down"], hw_config)

    gate_boundary_classes = _group_operator_boundary_classes("gate", gate_candidates)
    up_boundary_classes = _group_operator_boundary_classes("up", up_candidates)
    down_boundary_classes = _group_operator_boundary_classes("down", down_candidates)

    l_in_layouts = sorted(
        {boundary_class.layout_a for boundary_class in gate_boundary_classes}.union(
            {boundary_class.layout_a for boundary_class in up_boundary_classes}
        ),
        key=_layout_order_key,
    )
    l_mid_layouts = sorted(
        {boundary_class.layout_c for boundary_class in gate_boundary_classes}
        .union({boundary_class.layout_c for boundary_class in up_boundary_classes})
        .union({boundary_class.layout_a for boundary_class in down_boundary_classes}),
        key=_layout_order_key,
    )

    edge_cache: Dict[
        Tuple[LogicalBoundaryLayoutSignature, LogicalBoundaryLayoutSignature],
        float,
    ] = {}
    all_plans: List[FFNLayoutPlan] = []

    for l_in_layout in l_in_layouts:
        for l_mid_layout in l_mid_layouts:
            for gate_class in gate_boundary_classes:
                for up_class in up_boundary_classes:
                    for down_class in down_boundary_classes:
                        edge_transitions: List[EdgeReshardTransition] = []
                        step1_edge_costs_ms: Dict[str, float] = {}

                        l_in_gate_cost, l_in_gate_transition = _edge_cost(
                            "L_in->gate.A",
                            l_in_layout,
                            gate_class.layout_a,
                            edge_cache,
                            hw_config,
                            origin_mesh,
                        )
                        step1_edge_costs_ms["L_in->gate.A"] = l_in_gate_cost
                        if l_in_gate_transition is not None:
                            edge_transitions.append(l_in_gate_transition)

                        gate_mid_cost, gate_mid_transition = _edge_cost(
                            "gate.C->L_mid",
                            gate_class.layout_c,
                            l_mid_layout,
                            edge_cache,
                            hw_config,
                            origin_mesh,
                        )
                        step1_edge_costs_ms["gate.C->L_mid"] = gate_mid_cost
                        if gate_mid_transition is not None:
                            edge_transitions.append(gate_mid_transition)

                        l_in_up_cost, l_in_up_transition = _edge_cost(
                            "L_in->up.A",
                            l_in_layout,
                            up_class.layout_a,
                            edge_cache,
                            hw_config,
                            origin_mesh,
                        )
                        step1_edge_costs_ms["L_in->up.A"] = l_in_up_cost
                        if l_in_up_transition is not None:
                            edge_transitions.append(l_in_up_transition)

                        up_mid_cost, up_mid_transition = _edge_cost(
                            "up.C->L_mid",
                            up_class.layout_c,
                            l_mid_layout,
                            edge_cache,
                            hw_config,
                            origin_mesh,
                        )
                        step1_edge_costs_ms["up.C->L_mid"] = up_mid_cost
                        if up_mid_transition is not None:
                            edge_transitions.append(up_mid_transition)

                        mid_down_cost, mid_down_transition = _edge_cost(
                            "L_mid->down.A",
                            l_mid_layout,
                            down_class.layout_a,
                            edge_cache,
                            hw_config,
                            origin_mesh,
                        )
                        step1_edge_costs_ms["L_mid->down.A"] = mid_down_cost
                        if mid_down_transition is not None:
                            edge_transitions.append(mid_down_transition)

                        step1_operator_costs_ms = {
                            "gate": gate_class.best_step1_exec_time_ms,
                            "up": up_class.best_step1_exec_time_ms,
                            "down": down_class.best_step1_exec_time_ms,
                        }
                        step1_total = sum(step1_operator_costs_ms.values()) + sum(
                            step1_edge_costs_ms.values()
                        )

                        operator_layouts = {
                            "gate": {
                                "A": gate_class.layout_a,
                                "B": gate_class.layout_b,
                                "C": gate_class.layout_c,
                            },
                            "up": {
                                "A": up_class.layout_a,
                                "B": up_class.layout_b,
                                "C": up_class.layout_c,
                            },
                            "down": {
                                "A": down_class.layout_a,
                                "B": down_class.layout_b,
                                "C": down_class.layout_c,
                            },
                        }

                        all_plans.append(
                            FFNLayoutPlan(
                                activation_layouts={
                                    "L_in": l_in_layout,
                                    "L_mid": l_mid_layout,
                                    "L_out": down_class.layout_c,
                                },
                                weight_layouts={
                                    "W_gate": gate_class.layout_b,
                                    "W_up": up_class.layout_b,
                                    "W_down": down_class.layout_b,
                                },
                                boundary_classes={
                                    "gate": gate_class,
                                    "up": up_class,
                                    "down": down_class,
                                },
                                operator_layouts=operator_layouts,
                                edge_transitions=edge_transitions,
                                step1_operator_costs_ms=step1_operator_costs_ms,
                                step1_edge_costs_ms=step1_edge_costs_ms,
                                step1_total_time_ms=step1_total,
                            )
                        )

    if len(all_plans) == 0:
        raise ValueError("Failed to find a valid FFN layout plan")

    step1_layout_stats = FFNStep1LayoutStats(
        unique_l_in_count=len(l_in_layouts),
        unique_l_mid_count=len(l_mid_layouts),
        gate_boundary_class_count=len(gate_boundary_classes),
        up_boundary_class_count=len(up_boundary_classes),
        down_boundary_class_count=len(down_boundary_classes),
        projected_l_out_count=len({boundary.layout_c for boundary in down_boundary_classes}),
        projected_w_gate_count=len({boundary.layout_b for boundary in gate_boundary_classes}),
        projected_w_up_count=len({boundary.layout_b for boundary in up_boundary_classes}),
        projected_w_down_count=len({boundary.layout_b for boundary in down_boundary_classes}),
        total_plan_count=len(all_plans),
    )
    all_plans.sort(key=lambda plan: (plan.step1_total_time_ms, _step1_plan_order_key(plan)))
    return all_plans[:layout_top_k], {
        "gate": len(gate_candidates),
        "up": len(up_candidates),
        "down": len(down_candidates),
    }, step1_layout_stats


def _operator_constraints_for_plan(
    operator_name: str,
    plan: FFNLayoutPlan,
) -> LogicalTensorLayoutConstraints:
    boundary_class = plan.boundary_classes.get(operator_name)
    if boundary_class is None:
        raise ValueError(f"Unsupported FFN operator '{operator_name}'")
    return LogicalTensorLayoutConstraints(
        matrices={
            "A": boundary_class.layout_a,
            "B": boundary_class.layout_b,
            "C": boundary_class.layout_c,
        }
    )


def _step2_memo_key(
    operator_name: str,
    constraints: LogicalTensorLayoutConstraints,
    ownership_strategy: str,
    edge_obligation: Optional[EdgeReshardTransition] = None,
) -> Tuple[
    str,
    str,
    Tuple[Tuple[str, LogicalBoundaryLayoutSignature], ...],
    Optional[
        Tuple[
            str,
            LogicalBoundaryLayoutSignature,
            LogicalBoundaryLayoutSignature,
        ]
    ],
]:
    edge_signature = None
    if edge_obligation is not None:
        edge_signature = (
            edge_obligation.edge_name,
            edge_obligation.src_layout,
            edge_obligation.dst_layout,
        )
    return (
        operator_name,
        ownership_strategy,
        tuple(sorted(constraints.matrices.items(), key=lambda item: item[0])),
        edge_signature,
    )


def _search_best_operator_candidate(
    operator_name: str,
    operator_program: Program,
    operator_constraints,
    boundary_constraints: LogicalTensorLayoutConstraints,
    origin_mesh,
    split_axis_names: List[str],
    hw_config: Dict[str, Any],
    show_progress: bool,
    ownership_strategy: str,
    step2_cache: Dict[
        Tuple[
            str,
            str,
            Tuple[Tuple[str, LogicalBoundaryLayoutSignature], ...],
            Optional[
                Tuple[
                    str,
                    LogicalBoundaryLayoutSignature,
                    LogicalBoundaryLayoutSignature,
                ]
            ],
        ],
        Optional[_OperatorStep2Result],
    ],
    edge_obligation: Optional[EdgeReshardTransition] = None,
) -> Optional[_OperatorStep2Result]:
    cache_key = _step2_memo_key(
        operator_name,
        boundary_constraints,
        ownership_strategy,
        edge_obligation,
    )
    if cache_key in step2_cache:
        return step2_cache[cache_key]

    filtered_candidates = list(
        search_with_progress(
            operator_program,
            origin_mesh,
            split_axis_names,
            tensor_mapping_constraints=operator_constraints,
            program_filter=lambda program, constraints=boundary_constraints: (
                program_satisfies_logical_layout_constraints(program, constraints)
            ),
            progress_desc=f"step2[{operator_name}:{ownership_strategy}]",
            show_progress=show_progress,
            miniters=32,
            mininterval=0.5,
        )
    )
    if len(filtered_candidates) == 0:
        step2_cache[cache_key] = None
        return None

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
    result = _OperatorStep2Result(
        candidate_id=best_candidate[2],
        program=best_candidate[3],
        total_time_ms=best_candidate[0],
    )
    step2_cache[cache_key] = result
    return result


def _make_operator_segment(
    operator_name: str,
    result: _OperatorStep2Result,
) -> FFNSegmentSelection:
    return FFNSegmentSelection(
        segment_id=operator_name,
        segment_kind="isolated_operator",
        logical_operators=(operator_name,),
        logical_boundary_obligations=(
            f"{operator_name}.A",
            f"{operator_name}.B",
            f"{operator_name}.C",
        ),
        materialized_boundaries=(
            f"{operator_name}.A",
            f"{operator_name}.B",
            f"{operator_name}.C",
        ),
        selected_program=result.program,
        selected_candidate_id=result.candidate_id,
        total_time_ms=result.total_time_ms,
        owned_edge_obligations=tuple(),
    )


def _make_down_segment(
    result: _OperatorStep2Result,
    fused_edge: Optional[EdgeReshardTransition],
) -> FFNSegmentSelection:
    if fused_edge is None:
        return _make_operator_segment("down", result)
    return FFNSegmentSelection(
        segment_id="down_chain",
        segment_kind="layout_preserving_chain_edge_consumer",
        logical_operators=("silu_and_mul", "down"),
        logical_boundary_obligations=(fused_edge.edge_name, "down.B", "down.C"),
        materialized_boundaries=("down.B", "down.C"),
        selected_program=result.program,
        selected_candidate_id=result.candidate_id,
        total_time_ms=result.total_time_ms + fused_edge.cost_ms,
        owned_edge_obligations=(fused_edge.edge_name,),
    )


def _make_standalone_edge_segment(
    edge_obligation: EdgeReshardTransition,
) -> FFNSegmentSelection:
    return FFNSegmentSelection(
        segment_id=f"edge:{edge_obligation.edge_name}",
        segment_kind="standalone_edge_reshard",
        logical_operators=tuple(),
        logical_boundary_obligations=(edge_obligation.edge_name,),
        materialized_boundaries=(edge_obligation.edge_name,),
        selected_program=None,
        selected_candidate_id=None,
        total_time_ms=edge_obligation.cost_ms,
        owned_edge_obligations=(edge_obligation.edge_name,),
    )


def _build_step2_plan_candidate(
    plan: FFNLayoutPlan,
    gate_result: _OperatorStep2Result,
    up_result: _OperatorStep2Result,
    down_result: _OperatorStep2Result,
    fuse_down_edge: bool,
) -> _FFNStep2PlanCandidate:
    explicit_edge_obligations = {
        transition.edge_name: transition
        for transition in plan.edge_transitions
    }
    fused_edge = None
    if fuse_down_edge:
        fused_edge = explicit_edge_obligations.get(_DOWN_EDGE_NAME)
        if fused_edge is None:
            raise ValueError("Cannot fuse down edge when no explicit L_mid->down.A obligation exists")

    selected_segments: List[FFNSegmentSelection] = [
        _make_operator_segment("gate", gate_result),
        _make_operator_segment("up", up_result),
        _make_down_segment(down_result, fused_edge),
    ]
    selected_programs = {
        "gate": gate_result.program,
        "up": up_result.program,
        "down": down_result.program,
    }
    selected_indices = {
        "gate": gate_result.candidate_id,
        "up": up_result.candidate_id,
        "down": down_result.candidate_id,
    }
    step2_operator_costs_ms = {
        "gate": gate_result.total_time_ms,
        "up": up_result.total_time_ms,
        "down": down_result.total_time_ms,
    }
    step2_segment_costs_ms = {
        segment.segment_id: segment.total_time_ms for segment in selected_segments
    }
    edge_ownership: Dict[str, FFNEdgeLoweringOwnership] = {}

    for edge_name in sorted(explicit_edge_obligations.keys()):
        obligation = explicit_edge_obligations[edge_name]
        if fuse_down_edge and edge_name == _DOWN_EDGE_NAME:
            edge_ownership[edge_name] = FFNEdgeLoweringOwnership(
                edge_name=edge_name,
                owner_kind=_OWNER_CONSUMER_SEGMENT,
                owner_segment_id="down_chain",
                materialized_as_standalone=False,
                cost_ms=obligation.cost_ms,
            )
            continue

        edge_segment = _make_standalone_edge_segment(obligation)
        selected_segments.append(edge_segment)
        step2_segment_costs_ms[edge_segment.segment_id] = edge_segment.total_time_ms
        edge_ownership[edge_name] = FFNEdgeLoweringOwnership(
            edge_name=edge_name,
            owner_kind=_OWNER_STANDALONE_RESHARD,
            owner_segment_id=edge_segment.segment_id,
            materialized_as_standalone=True,
            cost_ms=obligation.cost_ms,
        )

    strategy_rank = 1
    if fuse_down_edge:
        strategy_rank = 0
    total_time_ms = sum(segment.total_time_ms for segment in selected_segments)
    return _FFNStep2PlanCandidate(
        selected_segments=selected_segments,
        explicit_edge_obligations=explicit_edge_obligations,
        edge_ownership=edge_ownership,
        selected_programs=selected_programs,
        selected_indices=selected_indices,
        step2_operator_costs_ms=step2_operator_costs_ms,
        step2_segment_costs_ms=step2_segment_costs_ms,
        total_time_ms=total_time_ms,
        strategy_rank=strategy_rank,
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
    """Run FFN two-step search with explicit edge obligations and segment ownership."""
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

    ranked_plans, candidate_counts, step1_layout_stats = _top_plans_from_candidates(
        candidate_programs,
        hw_config,
        origin_mesh,
        layout_top_k,
    )

    step2_cache: Dict[
        Tuple[
            str,
            str,
            Tuple[Tuple[str, LogicalBoundaryLayoutSignature], ...],
            Optional[
                Tuple[
                    str,
                    LogicalBoundaryLayoutSignature,
                    LogicalBoundaryLayoutSignature,
                ]
            ],
        ],
        Optional[_OperatorStep2Result],
    ] = {}
    best_result: Optional[FFNTwoStepSearchResult] = None
    best_score = (float("inf"), float("inf"), float("inf"))

    for plan in ranked_plans:
        gate_constraints = _operator_constraints_for_plan("gate", plan)
        up_constraints = _operator_constraints_for_plan("up", plan)
        down_constraints = _operator_constraints_for_plan("down", plan)

        gate_operator_constraints = None
        up_operator_constraints = None
        down_operator_constraints = None
        if tensor_mapping_constraints is not None:
            gate_operator_constraints = tensor_mapping_constraints.get("gate")
            up_operator_constraints = tensor_mapping_constraints.get("up")
            down_operator_constraints = tensor_mapping_constraints.get("down")

        gate_result = _search_best_operator_candidate(
            operator_name="gate",
            operator_program=operator_programs["gate"],
            operator_constraints=gate_operator_constraints,
            boundary_constraints=gate_constraints,
            origin_mesh=origin_mesh,
            split_axis_names=split_axis_names,
            hw_config=hw_config,
            show_progress=show_progress,
            ownership_strategy=_DOWN_ISOLATED_STRATEGY,
            step2_cache=step2_cache,
        )
        up_result = _search_best_operator_candidate(
            operator_name="up",
            operator_program=operator_programs["up"],
            operator_constraints=up_operator_constraints,
            boundary_constraints=up_constraints,
            origin_mesh=origin_mesh,
            split_axis_names=split_axis_names,
            hw_config=hw_config,
            show_progress=show_progress,
            ownership_strategy=_DOWN_ISOLATED_STRATEGY,
            step2_cache=step2_cache,
        )
        if gate_result is None or up_result is None:
            continue

        explicit_edge_obligations = {
            transition.edge_name: transition
            for transition in plan.edge_transitions
        }
        down_edge_obligation = explicit_edge_obligations.get(_DOWN_EDGE_NAME)

        candidate_builds: List[_FFNStep2PlanCandidate] = []
        down_isolated_result = _search_best_operator_candidate(
            operator_name="down",
            operator_program=operator_programs["down"],
            operator_constraints=down_operator_constraints,
            boundary_constraints=down_constraints,
            origin_mesh=origin_mesh,
            split_axis_names=split_axis_names,
            hw_config=hw_config,
            show_progress=show_progress,
            ownership_strategy=_DOWN_ISOLATED_STRATEGY,
            step2_cache=step2_cache,
            edge_obligation=down_edge_obligation,
        )
        if down_isolated_result is not None:
            candidate_builds.append(
                _build_step2_plan_candidate(
                    plan=plan,
                    gate_result=gate_result,
                    up_result=up_result,
                    down_result=down_isolated_result,
                    fuse_down_edge=False,
                )
            )

        if down_edge_obligation is not None:
            down_fused_result = _search_best_operator_candidate(
                operator_name="down",
                operator_program=operator_programs["down"],
                operator_constraints=down_operator_constraints,
                boundary_constraints=down_constraints,
                origin_mesh=origin_mesh,
                split_axis_names=split_axis_names,
                hw_config=hw_config,
                show_progress=show_progress,
                ownership_strategy=_DOWN_CONSUMER_FUSED_STRATEGY,
                step2_cache=step2_cache,
                edge_obligation=down_edge_obligation,
            )
            if down_fused_result is not None:
                candidate_builds.append(
                    _build_step2_plan_candidate(
                        plan=plan,
                        gate_result=gate_result,
                        up_result=up_result,
                        down_result=down_fused_result,
                        fuse_down_edge=True,
                    )
                )

        if len(candidate_builds) == 0:
            continue

        candidate_builds.sort(
            key=lambda candidate: (
                candidate.total_time_ms,
                candidate.strategy_rank,
            )
        )
        best_plan_candidate = candidate_builds[0]

        candidate_score = (
            best_plan_candidate.total_time_ms,
            float(best_plan_candidate.strategy_rank),
            float(plan.step1_total_time_ms),
        )
        if candidate_score >= best_score:
            continue

        best_score = candidate_score
        best_result = FFNTwoStepSearchResult(
            selected_plan=plan,
            ranked_plans=ranked_plans,
            selected_segments=best_plan_candidate.selected_segments,
            explicit_edge_obligations=best_plan_candidate.explicit_edge_obligations,
            edge_ownership=best_plan_candidate.edge_ownership,
            selected_programs=best_plan_candidate.selected_programs,
            selected_indices=best_plan_candidate.selected_indices,
            step2_operator_costs_ms=best_plan_candidate.step2_operator_costs_ms,
            step2_segment_costs_ms=best_plan_candidate.step2_segment_costs_ms,
            total_time_ms=best_plan_candidate.total_time_ms,
            candidate_counts=candidate_counts,
            step1_layout_stats=step1_layout_stats,
        )

    if best_result is None:
        raise ValueError("Failed to find a valid FFN two-step solution")
    return best_result
