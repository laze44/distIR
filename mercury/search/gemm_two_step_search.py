# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Two-step GEMM search over logical boundary layouts and constrained lowering."""

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mercury.backend import generate_pytorch_code
from mercury.ir.distributed import DeviceMesh
from mercury.ir.nodes import Program
from mercury.ir.utils import get_io_buffers
from mercury.search.estimate import _default_dtype_key, estimate_program
from mercury.search.gemm_dedupe import gemm_canonical_dedupe_key
from mercury.search.mapping_constraints import (
    LogicalBoundaryLayoutSignature,
    LogicalTensorLayoutConstraints,
    MatrixMappingConstraint,
    MatrixDimMapping,
    TensorMappingConstraints,
    derive_logical_local_shape,
    program_satisfies_logical_layout_constraints,
    resolve_topology_tokens_from_metadata,
)
from mercury.search.search import search_with_progress


_GEMM_MATRICES = ("A", "B", "C")


@dataclass(frozen=True)
class GEMMLayoutPlan:
    """One step-1 GEMM logical boundary layout plan."""

    problem_shape: Tuple[int, int, int]
    topology_shape: Tuple[int, ...]
    boundary_layouts: Dict[str, LogicalBoundaryLayoutSignature]
    step1_obligations_bytes: Dict[str, float]
    step1_cost_terms_ms: Dict[str, float]
    step1_total_time_ms: float


@dataclass
class GEMMTwoStepSearchResult:
    """Selected result after step-1 plan ranking and step-2 constrained lowering."""

    selected_plan: GEMMLayoutPlan
    ranked_plans: List[GEMMLayoutPlan]
    selected_program: Program
    selected_index: int
    selected_step2_total_time_ms: float
    plan_step2_costs_ms: Dict[str, float]
    candidate_count: int
    unsupported_plan_count: int


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


def _mesh_cards(mesh_shape: Tuple[int, ...], mesh_dims: Tuple[int, ...]) -> int:
    cards = 1
    for mesh_dim in mesh_dims:
        cards *= int(mesh_shape[int(mesh_dim)])
    return cards


def _matrix_buffers(program: Program) -> Dict[str, Any]:
    buffers: Dict[str, Any] = {}
    for buffer in program.visit(get_io_buffers):
        matrix_name = buffer.tensor.upper()
        if matrix_name in _GEMM_MATRICES and matrix_name not in buffers:
            buffers[matrix_name] = buffer
    for matrix_name in _GEMM_MATRICES:
        if matrix_name not in buffers:
            raise ValueError(
                f"Program '{program.name}' missing matrix buffer '{matrix_name}'"
            )
    return buffers


def _buffer_dim_size(dim: Any) -> int:
    if hasattr(dim, "size"):
        return int(dim.size)
    return int(dim)


def _infer_gemm_problem_shape(program: Program) -> Tuple[int, int, int]:
    buffers = _matrix_buffers(program)
    a_shape = tuple(_buffer_dim_size(dim) for dim in buffers["A"].shape)
    b_shape = tuple(_buffer_dim_size(dim) for dim in buffers["B"].shape)
    c_shape = tuple(_buffer_dim_size(dim) for dim in buffers["C"].shape)

    if len(a_shape) != 2 or len(b_shape) != 2 or len(c_shape) != 2:
        raise ValueError("GEMM two-step search expects rank-2 A/B/C matrices")
    m_len, k_len = a_shape
    k_len_b, n_len = b_shape
    m_len_c, n_len_c = c_shape
    if m_len != m_len_c or n_len != n_len_c or k_len != k_len_b:
        raise ValueError(
            f"Inconsistent GEMM shapes: A={a_shape}, B={b_shape}, C={c_shape}"
        )
    return int(m_len), int(n_len), int(k_len)


def _fixed_topology_metadata(mesh_shape: Tuple[int, ...]) -> Dict[str, List[int]]:
    # TODO: migrate to MeshShapePolicy (mercury/search/topology_policy.py)
    if len(mesh_shape) == 0:
        raise ValueError("mesh_shape must be non-empty")
    if len(mesh_shape) == 1:
        return {
            "inter_node_dims": [],
            "intra_node_dims": [0],
            "mixed_dims": [],
        }
    return {
        "inter_node_dims": [0] if int(mesh_shape[0]) > 1 else [],
        "intra_node_dims": [
            dim for dim in range(1, len(mesh_shape)) if int(mesh_shape[dim]) > 1
        ],
        "mixed_dims": [],
    }


def _flexible_dim_options(
    topology_metadata: Dict[str, List[int]],
) -> List[Optional[Tuple[int, ...]]]:
    options: List[Optional[Tuple[int, ...]]] = [None]
    seen = {None}

    inter_dims = tuple(int(dim) for dim in topology_metadata.get("inter_node_dims", []))
    intra_dims = tuple(int(dim) for dim in topology_metadata.get("intra_node_dims", []))
    mixed_dims = tuple(int(dim) for dim in topology_metadata.get("mixed_dims", []))
    combined_dims = tuple(sorted(set(inter_dims + intra_dims)))

    for dims in (inter_dims, intra_dims, combined_dims, mixed_dims):
        normalized = tuple(sorted(set(int(dim) for dim in dims)))
        if len(normalized) == 0 or normalized in seen:
            continue
        options.append(normalized)
        seen.add(normalized)
    return options


def _fixed_dim_options(
    dim_mapping: MatrixDimMapping,
    topology_metadata: Dict[str, List[int]],
) -> List[Optional[Tuple[int, ...]]]:
    if dim_mapping.is_replicate:
        return [None]

    assert dim_mapping.shard_topology is not None
    resolved = resolve_topology_tokens_from_metadata(
        topology_metadata,
        dim_mapping.shard_topology,
    )
    if len(resolved) == 0:
        return []
    return [resolved]


def _enumerate_matrix_layouts(
    matrix_name: str,
    global_shape: Tuple[int, int],
    constraint: MatrixMappingConstraint,
    mesh_shape: Tuple[int, ...],
    topology_metadata: Dict[str, List[int]],
) -> List[LogicalBoundaryLayoutSignature]:
    if len(global_shape) != 2:
        raise ValueError(f"{matrix_name} global_shape must be rank-2")

    if constraint.mode == "flexible":
        dim_options = [
            _flexible_dim_options(topology_metadata),
            _flexible_dim_options(topology_metadata),
        ]
    elif constraint.mode == "fixed":
        if constraint.mapping is None or len(constraint.mapping) != 2:
            raise ValueError(f"Fixed matrix {matrix_name} must define rank-2 mapping")
        dim_options = [
            _fixed_dim_options(constraint.mapping[0], topology_metadata),
            _fixed_dim_options(constraint.mapping[1], topology_metadata),
        ]
    else:
        raise ValueError(f"Unsupported matrix mode '{constraint.mode}'")

    layouts: List[LogicalBoundaryLayoutSignature] = []
    seen = set()
    for dim0, dim1 in itertools.product(dim_options[0], dim_options[1]):
        used_mesh_dims = set()
        shard_specs: List[Tuple[str, Tuple[int, ...]]] = []
        valid = True
        for shard_dims in (dim0, dim1):
            if shard_dims is None:
                shard_specs.append(("R", tuple()))
                continue
            normalized_dims = tuple(sorted(set(int(dim) for dim in shard_dims)))
            if len(normalized_dims) == 0:
                valid = False
                break
            if len(used_mesh_dims.intersection(normalized_dims)) > 0:
                valid = False
                break
            used_mesh_dims.update(normalized_dims)
            shard_specs.append(("S", normalized_dims))

        if not valid:
            continue

        signature = LogicalBoundaryLayoutSignature(
            mesh_shape=tuple(int(dim) for dim in mesh_shape),
            global_shape=tuple(int(dim) for dim in global_shape),
            shard_specs=tuple(shard_specs),
        )
        key = (signature.mesh_shape, signature.global_shape, signature.shard_specs)
        if key in seen:
            continue
        seen.add(key)
        layouts.append(signature)
    return layouts


def _dim_uses_topology(
    layout: LogicalBoundaryLayoutSignature,
    tensor_dim: int,
    topology_dims: List[int],
) -> bool:
    if layout.shard_specs[tensor_dim][0] != "S":
        return False
    shard_dims = set(int(dim) for dim in layout.shard_specs[tensor_dim][1])
    return len(shard_dims.intersection(int(dim) for dim in topology_dims)) > 0


def _bytes_to_comm_ms(
    inter_bytes: float,
    intra_bytes: float,
    hw_config: Dict[str, Any],
) -> float:
    inter_bw = _read_positive(
        hw_config, ["interconnect", "inter_node", "bandwidth_gb_per_s"]
    ) * (10**9)
    intra_bw = _read_positive(
        hw_config, ["interconnect", "intra_node", "bandwidth_gb_per_s"]
    ) * (10**9)
    inter_latency = _read_non_negative(
        hw_config, ["interconnect", "inter_node", "latency_us"]
    ) / (10**6)
    intra_latency = _read_non_negative(
        hw_config, ["interconnect", "intra_node", "latency_us"]
    ) / (10**6)

    return (
        (inter_bytes / inter_bw)
        + (intra_bytes / intra_bw)
        + (inter_latency if inter_bytes > 0 else 0.0)
        + (intra_latency if intra_bytes > 0 else 0.0)
    ) * 1000.0


def _classify_obligation_bytes(
    bytes_moved: float,
    layout_a: LogicalBoundaryLayoutSignature,
    dim_a: int,
    layout_b: LogicalBoundaryLayoutSignature,
    dim_b: int,
    topology_metadata: Dict[str, List[int]],
) -> Tuple[float, float]:
    inter_dims = topology_metadata.get("inter_node_dims", [])
    uses_inter = _dim_uses_topology(layout_a, dim_a, inter_dims) or _dim_uses_topology(
        layout_b, dim_b, inter_dims
    )
    if uses_inter:
        return bytes_moved, 0.0
    return 0.0, bytes_moved


def _estimate_step1_cost(
    plan: GEMMLayoutPlan,
    hw_config: Dict[str, Any],
    topology_metadata: Dict[str, List[int]],
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    layout_a = plan.boundary_layouts["A"]
    layout_b = plan.boundary_layouts["B"]
    layout_c = plan.boundary_layouts["C"]

    m_len, n_len, k_len = plan.problem_shape
    elem_bytes = 2.0
    a_total_bytes = float(m_len * k_len) * elem_bytes
    b_total_bytes = float(k_len * n_len) * elem_bytes
    c_total_bytes = float(m_len * n_len) * elem_bytes

    local_a = derive_logical_local_shape(
        layout_a.global_shape,
        layout_a.mesh_shape,
        layout_a.shard_specs,
    )
    local_b = derive_logical_local_shape(
        layout_b.global_shape,
        layout_b.mesh_shape,
        layout_b.shard_specs,
    )
    local_c = derive_logical_local_shape(
        layout_c.global_shape,
        layout_c.mesh_shape,
        layout_c.shard_specs,
    )

    local_m = int(local_c[0])
    local_n = int(local_c[1])
    local_k = int(min(local_a[1], local_b[0]))
    flops = float(2 * local_m * local_n * local_k)
    dtype_key = _default_dtype_key(hw_config)
    peak_tflops = _read_positive(hw_config, ["compute", "peak_tflops", dtype_key])
    compute_ms = (flops / (peak_tflops * (10**12))) * 1000.0

    input_materialization_bytes = 0.0
    input_inter_bytes = 0.0
    input_intra_bytes = 0.0
    if layout_a.shard_specs[0] != layout_c.shard_specs[0]:
        moved = a_total_bytes * 0.5
        input_materialization_bytes += moved
        inter_bytes, intra_bytes = _classify_obligation_bytes(
            moved,
            layout_a,
            0,
            layout_c,
            0,
            topology_metadata,
        )
        input_inter_bytes += inter_bytes
        input_intra_bytes += intra_bytes
    if layout_b.shard_specs[1] != layout_c.shard_specs[1]:
        moved = b_total_bytes * 0.5
        input_materialization_bytes += moved
        inter_bytes, intra_bytes = _classify_obligation_bytes(
            moved,
            layout_b,
            1,
            layout_c,
            1,
            topology_metadata,
        )
        input_inter_bytes += inter_bytes
        input_intra_bytes += intra_bytes

    reduction_shard_cards = max(
        _mesh_cards(layout_a.mesh_shape, layout_a.shard_specs[1][1])
        if layout_a.shard_specs[1][0] == "S"
        else 1,
        _mesh_cards(layout_b.mesh_shape, layout_b.shard_specs[0][1])
        if layout_b.shard_specs[0][0] == "S"
        else 1,
    )
    reduction_finalize_bytes = 0.0
    reduction_inter_bytes = 0.0
    reduction_intra_bytes = 0.0
    if reduction_shard_cards > 1:
        reduction_finalize_bytes = c_total_bytes * (
            float(reduction_shard_cards - 1) / float(reduction_shard_cards)
        )
        inter_bytes, intra_bytes = _classify_obligation_bytes(
            reduction_finalize_bytes,
            layout_a,
            1,
            layout_b,
            0,
            topology_metadata,
        )
        reduction_inter_bytes += inter_bytes
        reduction_intra_bytes += intra_bytes

    output_materialization_bytes = 0.0
    output_inter_bytes = 0.0
    output_intra_bytes = 0.0
    if layout_c.shard_specs[0][0] == "R" or layout_c.shard_specs[1][0] == "R":
        output_materialization_bytes = c_total_bytes * 0.25
        if _dim_uses_topology(
            layout_c, 0, topology_metadata.get("inter_node_dims", [])
        ) or _dim_uses_topology(
            layout_c, 1, topology_metadata.get("inter_node_dims", [])
        ):
            output_inter_bytes = output_materialization_bytes
        else:
            output_intra_bytes = output_materialization_bytes

    overlapable_comm_ms = _bytes_to_comm_ms(
        input_inter_bytes, input_intra_bytes, hw_config
    )
    blocking_comm_ms = _bytes_to_comm_ms(
        reduction_inter_bytes + output_inter_bytes,
        reduction_intra_bytes + output_intra_bytes,
        hw_config,
    )
    edge_ms = 0.0
    total_ms = edge_ms + blocking_comm_ms + max(compute_ms, overlapable_comm_ms)

    obligations = {
        "input_materialization_bytes": input_materialization_bytes,
        "reduction_finalize_bytes": reduction_finalize_bytes,
        "output_materialization_bytes": output_materialization_bytes,
    }
    cost_terms = {
        "compute_ms": compute_ms,
        "overlapable_comm_ms": overlapable_comm_ms,
        "blocking_comm_ms": blocking_comm_ms,
        "edge_ms": edge_ms,
    }
    return obligations, cost_terms, total_ms


def _plan_order_key(plan: GEMMLayoutPlan) -> Tuple[str, str, str]:
    return (
        plan.boundary_layouts["A"].to_summary(),
        plan.boundary_layouts["B"].to_summary(),
        plan.boundary_layouts["C"].to_summary(),
    )


def enumerate_gemm_step1_layout_plans(
    problem_shape: Tuple[int, int, int],
    origin_mesh: DeviceMesh,
    hw_config: Dict[str, Any],
    tensor_mapping_constraints: Optional[TensorMappingConstraints],
    layout_top_k: int,
) -> List[GEMMLayoutPlan]:
    """Enumerate and rank GEMM step-1 logical boundary plans under fixed topology."""
    if layout_top_k <= 0:
        raise ValueError("layout_top_k must be a positive integer")

    topology_metadata = _fixed_topology_metadata(
        tuple(int(dim) for dim in origin_mesh.shape)
    )
    m_len, n_len, k_len = problem_shape
    constraints = tensor_mapping_constraints or TensorMappingConstraints(matrices={})

    layouts_a = _enumerate_matrix_layouts(
        "A",
        (m_len, k_len),
        constraints.get("A"),
        tuple(int(dim) for dim in origin_mesh.shape),
        topology_metadata,
    )
    layouts_b = _enumerate_matrix_layouts(
        "B",
        (k_len, n_len),
        constraints.get("B"),
        tuple(int(dim) for dim in origin_mesh.shape),
        topology_metadata,
    )
    layouts_c = _enumerate_matrix_layouts(
        "C",
        (m_len, n_len),
        constraints.get("C"),
        tuple(int(dim) for dim in origin_mesh.shape),
        topology_metadata,
    )

    ranked_plans: List[GEMMLayoutPlan] = []
    for layout_a, layout_b, layout_c in itertools.product(
        layouts_a, layouts_b, layouts_c
    ):
        seed_plan = GEMMLayoutPlan(
            problem_shape=(m_len, n_len, k_len),
            topology_shape=tuple(int(dim) for dim in origin_mesh.shape),
            boundary_layouts={"A": layout_a, "B": layout_b, "C": layout_c},
            step1_obligations_bytes={},
            step1_cost_terms_ms={},
            step1_total_time_ms=0.0,
        )
        try:
            obligations, cost_terms, total_ms = _estimate_step1_cost(
                seed_plan,
                hw_config,
                topology_metadata,
            )
        except ValueError:
            continue
        ranked_plans.append(
            GEMMLayoutPlan(
                problem_shape=seed_plan.problem_shape,
                topology_shape=seed_plan.topology_shape,
                boundary_layouts=seed_plan.boundary_layouts,
                step1_obligations_bytes=obligations,
                step1_cost_terms_ms=cost_terms,
                step1_total_time_ms=total_ms,
            )
        )

    if len(ranked_plans) == 0:
        raise ValueError("Failed to enumerate any GEMM logical layout plan")

    ranked_plans.sort(
        key=lambda plan: (plan.step1_total_time_ms, _plan_order_key(plan))
    )
    return ranked_plans[:layout_top_k]


def search_gemm_two_step(
    input_program: Program,
    origin_mesh: DeviceMesh,
    split_axis_names: List[str],
    hw_config: Dict[str, Any],
    tensor_mapping_constraints: Optional[TensorMappingConstraints] = None,
    layout_top_k: int = 10,
    candidate_programs: Optional[List[Program]] = None,
    show_progress: bool = False,
) -> GEMMTwoStepSearchResult:
    """Run GEMM two-step search with boundary-plan step-1 and constrained step-2."""
    problem_shape = _infer_gemm_problem_shape(input_program)
    ranked_plans = enumerate_gemm_step1_layout_plans(
        problem_shape=problem_shape,
        origin_mesh=origin_mesh,
        hw_config=hw_config,
        tensor_mapping_constraints=tensor_mapping_constraints,
        layout_top_k=layout_top_k,
    )

    if candidate_programs is None:
        candidate_programs = list(
            search_with_progress(
                input_program,
                origin_mesh,
                split_axis_names,
                tensor_mapping_constraints=tensor_mapping_constraints,
                progress_desc="gemm.step2.seed",
                show_progress=show_progress,
                miniters=32,
                mininterval=0.5,
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )
    if len(candidate_programs) == 0:
        raise ValueError("GEMM step-2 search has no candidates")

    best_program: Optional[Program] = None
    best_index = -1
    best_plan: Optional[GEMMLayoutPlan] = None
    best_total = float("inf")
    plan_step2_costs_ms: Dict[str, float] = {}
    unsupported_plan_count = 0

    for plan in ranked_plans:
        logical_constraints = LogicalTensorLayoutConstraints(
            matrices=plan.boundary_layouts
        )
        matched_candidates = [
            (candidate_idx, program)
            for candidate_idx, program in enumerate(candidate_programs)
            if program_satisfies_logical_layout_constraints(
                program, logical_constraints
            )
        ]
        if len(matched_candidates) == 0:
            unsupported_plan_count += 1
            continue

        ranked_candidates = []
        for candidate_idx, program in matched_candidates:
            estimate = estimate_program(program, hw_config)
            ranked_candidates.append(
                (
                    estimate.total_time_ms,
                    generate_pytorch_code(program),
                    candidate_idx,
                    program,
                )
            )
        ranked_candidates.sort(key=lambda item: (item[0], item[1]))
        best_candidate = ranked_candidates[0]
        plan_step2_costs_ms[" | ".join(_plan_order_key(plan))] = float(
            best_candidate[0]
        )

        if float(best_candidate[0]) >= best_total:
            continue

        best_total = float(best_candidate[0])
        best_index = int(best_candidate[2])
        best_program = best_candidate[3]
        best_plan = plan

    if best_program is None or best_plan is None:
        raise ValueError(
            "No GEMM step-2 lowering satisfies any ranked logical boundary plan"
        )

    return GEMMTwoStepSearchResult(
        selected_plan=best_plan,
        ranked_plans=ranked_plans,
        selected_program=best_program,
        selected_index=best_index,
        selected_step2_total_time_ms=best_total,
        plan_step2_costs_ms=plan_step2_costs_ms,
        candidate_count=len(candidate_programs),
        unsupported_plan_count=unsupported_plan_count,
    )
