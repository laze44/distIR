# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Search FFN gate/up/down GEMM candidates and run two-step graph-level selection."""

import argparse
import ast
import contextlib
import copy
import io
import os
import textwrap
from typing import Dict, List, Tuple

from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.search.dump import dump
from mercury.search.estimate import estimate_program, load_hardware_config
from mercury.search.mapping_constraints import load_operator_tensor_mapping_constraints
from mercury.search.ffn_two_step_search import search_ffn_two_step
from mercury.search.search import search_with_progress


_OPERATORS = ("gate", "up", "down")


def _extract_template_from_file(file_name: str, template_name: str) -> str:
    file_path = os.path.join(os.path.dirname(__file__), "utils", file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == template_name:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    raise ValueError(f"Cannot find template '{template_name}' in {file_path}")


def _load_ffn_templates() -> Dict[str, str]:
    try:
        from utils.ffn_dsl import (
            ffn_down_gemm_manage_reduction,
            ffn_gate_gemm_manage_reduction,
            ffn_up_gemm_manage_reduction,
        )

        return {
            "gate": ffn_gate_gemm_manage_reduction,
            "up": ffn_up_gemm_manage_reduction,
            "down": ffn_down_gemm_manage_reduction,
        }
    except (ModuleNotFoundError, ImportError):
        return {
            "gate": _extract_template_from_file(
                "ffn_dsl.py",
                "ffn_gate_gemm_manage_reduction",
            ),
            "up": _extract_template_from_file(
                "ffn_dsl.py",
                "ffn_up_gemm_manage_reduction",
            ),
            "down": _extract_template_from_file(
                "ffn_dsl.py",
                "ffn_down_gemm_manage_reduction",
            ),
        }


def _build_program(source: str):
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return builder.visit(node)
    raise ValueError("Could not find function definition in FFN GEMM template")


def _capture_dump(program) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dump(program)
    return buf.getvalue()


def _rank_operator_candidates(
    operator_candidates: List,
    hw_config: Dict[str, object],
) -> List[Tuple[int, object, float, float, float]]:
    """Rank one operator's candidates by estimated total time."""
    ranked_candidates: List[Tuple[int, object, float, float, float]] = []
    for candidate_idx, candidate_program in enumerate(operator_candidates):
        estimate = estimate_program(candidate_program, hw_config)
        ranked_candidates.append(
            (
                candidate_idx,
                candidate_program,
                estimate.total_time_ms,
                estimate.compute_time_ms,
                estimate.comm_time_ms,
            )
        )
    ranked_candidates.sort(
        key=lambda item: (
            item[2],
            generate_pytorch_code(item[1]),
        )
    )
    return ranked_candidates


def _serialize_program_for_export(program) -> Tuple[str, str]:
    program_copy = copy.deepcopy(program)
    eliminate_loops(program_copy)
    code = generate_pytorch_code(program_copy)
    ir_text = _capture_dump(program_copy)
    return ir_text, code


def _write_program_artifacts(
    result_dir: str,
    file_prefix: str,
    program,
) -> Tuple[str, str]:
    ir_text, code = _serialize_program_for_export(program)
    ir_name = f"{file_prefix}_ir.txt"
    code_name = f"{file_prefix}_code.py"
    with open(
        os.path.join(result_dir, ir_name),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(ir_text)
    with open(
        os.path.join(result_dir, code_name),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(code)
    return ir_name, code_name


def search_ffn(
    batch: int,
    seq_len: int,
    d_model: int,
    d_ffn: int,
    num_devices: int,
    output_dir: str,
    top_k: int,
    layout_top_k: int,
    hw_config_path: str,
    mapping_config_path: str = "config/ffn_tensor_mapping.json",
    show_progress: bool = True,
) -> None:
    """Run FFN search and save the best two-step result plus top-k candidates."""
    if batch <= 0 or seq_len <= 0 or d_model <= 0 or d_ffn <= 0:
        raise ValueError("batch/seq_len/d_model/d_ffn must be positive integers")
    if num_devices <= 0:
        raise ValueError("num_devices must be a positive integer")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if layout_top_k <= 0:
        raise ValueError("layout_top_k must be a positive integer")

    m_len = batch * seq_len
    world_size = num_devices
    mesh = DeviceMesh(list(range(world_size)), (world_size,))

    templates = _load_ffn_templates()
    operator_sources = {
        "gate": templates["gate"].format(M_LEN=m_len, DM_LEN=d_model, DFFN_LEN=d_ffn),
        "up": templates["up"].format(M_LEN=m_len, DM_LEN=d_model, DFFN_LEN=d_ffn),
        "down": templates["down"].format(M_LEN=m_len, DM_LEN=d_model, DFFN_LEN=d_ffn),
    }
    operator_programs = {
        operator_name: _build_program(operator_sources[operator_name])
        for operator_name in _OPERATORS
    }

    constraints = load_operator_tensor_mapping_constraints(
        mapping_config_path,
        list(_OPERATORS),
    )
    hw_config = load_hardware_config(hw_config_path)

    candidate_programs: Dict[str, List] = {}
    ranked_candidates_by_operator: Dict[str, List[Tuple[int, object, float, float, float]]] = {}
    for operator_name in _OPERATORS:
        operator_candidates = list(
            search_with_progress(
                operator_programs[operator_name],
                mesh,
                ["I", "J", "K"],
                tensor_mapping_constraints=constraints.get(operator_name),
                progress_desc=f"search[{operator_name}]",
                show_progress=show_progress,
                miniters=32,
                mininterval=0.5,
            )
        )
        operator_candidates.sort(key=generate_pytorch_code)
        if len(operator_candidates) == 0:
            raise ValueError(f"Operator '{operator_name}' has no candidate after filtering")
        candidate_programs[operator_name] = operator_candidates
        ranked_candidates_by_operator[operator_name] = _rank_operator_candidates(
            operator_candidates,
            hw_config,
        )

    two_step_result = search_ffn_two_step(
        operator_programs=operator_programs,
        origin_mesh=mesh,
        split_axis_names=["I", "J", "K"],
        hw_config=hw_config,
        tensor_mapping_constraints=constraints,
        layout_top_k=layout_top_k,
        candidate_programs=candidate_programs,
        show_progress=show_progress,
    )
    selected_plan = two_step_result.selected_plan

    result_dir = os.path.join(
        output_dir,
        (
            f"ffn_b{batch}_l{seq_len}_dm{d_model}_df{d_ffn}_"
            f"devices{num_devices}"
        ),
    )
    os.makedirs(result_dir, exist_ok=True)

    selected_segment_files: Dict[str, Tuple[str, str]] = {}
    for segment in two_step_result.selected_segments:
        if segment.selected_program is None:
            continue
        selected_segment_files[segment.segment_id] = _write_program_artifacts(
            result_dir,
            f"{segment.segment_id}_selected",
            segment.selected_program,
        )

    canonical_selected_files: Dict[str, Tuple[str, str]] = {}
    for operator_name in _OPERATORS:
        canonical_selected_files[operator_name] = _write_program_artifacts(
            result_dir,
            operator_name,
            two_step_result.selected_programs[operator_name],
        )

    summary_lines = [
        "FFN Two-Step Search Results",
        f"Input: batch={batch}, seq_len={seq_len}, d_model={d_model}, d_ffn={d_ffn}",
        f"Mesh: num_devices={num_devices}, world_size={world_size}",
        f"Hardware config: {hw_config['name']}",
        f"Mapping config: {mapping_config_path}",
        f"Requested top_k per operator: {top_k}",
        f"Requested layout_top_k: {layout_top_k}",
        "",
        "Operator Candidate Counts:",
        f"  gate: {two_step_result.candidate_counts['gate']}",
        f"  up: {two_step_result.candidate_counts['up']}",
        f"  down: {two_step_result.candidate_counts['down']}",
        "",
        "Step-1 Layout Plan Counts:",
        f"  unique L_in: {two_step_result.step1_layout_stats.unique_l_in_count}",
        f"  unique L_mid: {two_step_result.step1_layout_stats.unique_l_mid_count}",
        f"  gate boundary classes: {two_step_result.step1_layout_stats.gate_boundary_class_count}",
        f"  up boundary classes: {two_step_result.step1_layout_stats.up_boundary_class_count}",
        f"  down boundary classes: {two_step_result.step1_layout_stats.down_boundary_class_count}",
        f"  projected unique L_out: {two_step_result.step1_layout_stats.projected_l_out_count}",
        f"  projected unique W_gate: {two_step_result.step1_layout_stats.projected_w_gate_count}",
        f"  projected unique W_up: {two_step_result.step1_layout_stats.projected_w_up_count}",
        f"  projected unique W_down: {two_step_result.step1_layout_stats.projected_w_down_count}",
        f"  total evaluated plans: {two_step_result.step1_layout_stats.total_plan_count}",
        f"  retained top-k plans: {len(two_step_result.ranked_plans)}",
        "",
        "Operator Tensor Mapping Constraints:",
    ]
    for operator_name, matrix_summary in constraints.summary_by_operator().items():
        summary_lines.append(f"  {operator_name}:")
        for matrix_name in ("A", "B", "C"):
            summary_lines.append(f"    {matrix_name}: {matrix_summary[matrix_name]}")

    summary_lines.extend(
        [
            "",
            "Two-Step Search Results:",
            f"  Step-1 retained top-k plans: {len(two_step_result.ranked_plans)}",
            "",
            "Step-1 Top Layout Plans:",
        ]
    )
    for plan_rank, layout_plan in enumerate(two_step_result.ranked_plans, start=1):
        summary_lines.append(
            f"  Plan {plan_rank}: total={layout_plan.step1_total_time_ms:.6f} ms"
        )
        for layout_name in ("L_in", "L_mid", "L_out"):
            summary_lines.append(
                f"    {layout_name}: {layout_plan.activation_layouts[layout_name].to_summary()}"
            )
        for weight_name in ("W_gate", "W_up", "W_down"):
            summary_lines.append(
                f"    {weight_name}: {layout_plan.weight_layouts[weight_name].to_summary()}"
            )
        for operator_name in _OPERATORS:
            boundary_class = layout_plan.boundary_classes[operator_name]
            summary_lines.append(
                (
                    f"    {operator_name} boundary class: "
                    f"A={boundary_class.layout_a.to_summary()}, "
                    f"B={boundary_class.layout_b.to_summary()}, "
                    f"C={boundary_class.layout_c.to_summary()}"
                )
            )
        summary_lines.append(
            "    step1_operator_costs="
            + ", ".join(
                f"{operator_name}:{layout_plan.step1_operator_costs_ms[operator_name]:.6f}"
                for operator_name in _OPERATORS
            )
        )
        summary_lines.append(
            "    step1_edge_costs="
            + ", ".join(
                f"{edge_name}:{edge_cost:.6f}"
                for edge_name, edge_cost in sorted(layout_plan.step1_edge_costs_ms.items())
            )
        )
        summary_lines.append(
            f"    explicit_edge_transitions={len(layout_plan.edge_transitions)}"
        )

    summary_lines.extend(
        [
            "",
            "Selected Layout Plan:",
            f"  L_in: {selected_plan.activation_layouts['L_in'].to_summary()}",
            f"  L_mid: {selected_plan.activation_layouts['L_mid'].to_summary()}",
            f"  L_out: {selected_plan.activation_layouts['L_out'].to_summary()}",
            f"  Selected L_out: {selected_plan.activation_layouts['L_out'].to_summary()}",
            f"  W_gate: {selected_plan.weight_layouts['W_gate'].to_summary()}",
            f"  W_up: {selected_plan.weight_layouts['W_up'].to_summary()}",
            f"  W_down: {selected_plan.weight_layouts['W_down'].to_summary()}",
            (
                "  gate boundary class: "
                f"A={selected_plan.boundary_classes['gate'].layout_a.to_summary()}, "
                f"B={selected_plan.boundary_classes['gate'].layout_b.to_summary()}, "
                f"C={selected_plan.boundary_classes['gate'].layout_c.to_summary()}"
            ),
            (
                "  up boundary class: "
                f"A={selected_plan.boundary_classes['up'].layout_a.to_summary()}, "
                f"B={selected_plan.boundary_classes['up'].layout_b.to_summary()}, "
                f"C={selected_plan.boundary_classes['up'].layout_c.to_summary()}"
            ),
            (
                "  down boundary class: "
                f"A={selected_plan.boundary_classes['down'].layout_a.to_summary()}, "
                f"B={selected_plan.boundary_classes['down'].layout_b.to_summary()}, "
                f"C={selected_plan.boundary_classes['down'].layout_c.to_summary()}"
            ),
            (
                "  down.C == L_out: "
                f"{selected_plan.boundary_classes['down'].layout_c == selected_plan.activation_layouts['L_out']}"
            ),
            "",
            "Step-1 Operator Costs (ms):",
            f"  gate: {selected_plan.step1_operator_costs_ms['gate']:.6f}",
            f"  up: {selected_plan.step1_operator_costs_ms['up']:.6f}",
            f"  down: {selected_plan.step1_operator_costs_ms['down']:.6f}",
            "Step-1 Edge Costs (ms):",
            f"  L_in->gate.A: {selected_plan.step1_edge_costs_ms.get('L_in->gate.A', 0.0):.6f}",
            f"  gate.C->L_mid: {selected_plan.step1_edge_costs_ms.get('gate.C->L_mid', 0.0):.6f}",
            f"  L_in->up.A: {selected_plan.step1_edge_costs_ms.get('L_in->up.A', 0.0):.6f}",
            f"  up.C->L_mid: {selected_plan.step1_edge_costs_ms.get('up.C->L_mid', 0.0):.6f}",
            f"  L_mid->down.A: {selected_plan.step1_edge_costs_ms.get('L_mid->down.A', 0.0):.6f}",
            "",
            "Step-2 Operator Costs (ms):",
            f"  gate: {two_step_result.step2_operator_costs_ms['gate']:.6f}",
            f"  up: {two_step_result.step2_operator_costs_ms['up']:.6f}",
            f"  down: {two_step_result.step2_operator_costs_ms['down']:.6f}",
            "Step-2 Segment Costs (ms):",
        ]
    )
    for segment in two_step_result.selected_segments:
        summary_lines.append(f"  {segment.segment_id}: {segment.total_time_ms:.6f}")

    summary_lines.extend(
        [
            "",
            "Selected Step-2 Segments:",
        ]
    )
    for segment in two_step_result.selected_segments:
        summary_lines.append(
            (
                f"  {segment.segment_id}: kind={segment.segment_kind}, "
                f"operators={list(segment.logical_operators)}, "
                f"logical_obligations={list(segment.logical_boundary_obligations)}, "
                f"materialized_boundaries={list(segment.materialized_boundaries)}, "
                f"cost={segment.total_time_ms:.6f} ms"
            )
        )
        if segment.selected_program is None:
            summary_lines.append("    selected_program: none (logical edge-only segment)")
        else:
            file_names = selected_segment_files[segment.segment_id]
            summary_lines.append(
                (
                    f"    selected_program: candidate_index={segment.selected_candidate_id}, "
                    f"IR={file_names[0]}, code={file_names[1]}"
                )
            )

    summary_lines.extend(
        [
            "",
            "Explicit Edge Obligations And Ownership:",
        ]
    )
    if len(two_step_result.explicit_edge_obligations) == 0:
        summary_lines.append("  (none)")
    else:
        for edge_name in sorted(two_step_result.explicit_edge_obligations.keys()):
            edge = two_step_result.explicit_edge_obligations[edge_name]
            ownership = two_step_result.edge_ownership.get(edge_name)
            if ownership is None:
                summary_lines.append(
                    (
                        f"  {edge_name}: src={edge.src_layout.to_summary()}, "
                        f"dst={edge.dst_layout.to_summary()}, "
                        f"obligation_cost={edge.cost_ms:.6f} ms, ownership=missing"
                    )
                )
                continue
            summary_lines.append(
                (
                    f"  {edge_name}: src={edge.src_layout.to_summary()}, "
                    f"dst={edge.dst_layout.to_summary()}, "
                    f"obligation_cost={edge.cost_ms:.6f} ms, "
                    f"owner={ownership.owner_kind}({ownership.owner_segment_id}), "
                    f"materialized_standalone={ownership.materialized_as_standalone}"
                )
            )

    summary_lines.extend(
        [
            "",
            f"Step-2 Total Cost (ms): {two_step_result.total_time_ms:.6f}",
            "",
            "Selected Candidate Indices:",
            f"  gate: {two_step_result.selected_indices['gate']}",
            f"  up: {two_step_result.selected_indices['up']}",
            f"  down: {two_step_result.selected_indices['down']}",
            "",
            "Canonical Selected Operator Files:",
            (
                f"  gate: IR={canonical_selected_files['gate'][0]}, "
                f"code={canonical_selected_files['gate'][1]}"
            ),
            (
                f"  up: IR={canonical_selected_files['up'][0]}, "
                f"code={canonical_selected_files['up'][1]}"
            ),
            (
                f"  down: IR={canonical_selected_files['down'][0]}, "
                f"code={canonical_selected_files['down'][1]}"
            ),
        ]
    )

    summary_lines.extend(
        [
            "",
            "Top-k Candidate Files (Non-Canonical Debug):",
            "  These per-operator rankings are auxiliary and may differ from selected segments.",
        ]
    )

    for operator_name in _OPERATORS:
        ranked_candidates = ranked_candidates_by_operator[operator_name]
        save_count = min(top_k, len(ranked_candidates))
        summary_lines.append(f"  {operator_name}: saved {save_count} candidate(s)")
        for rank, (
            candidate_idx,
            candidate_program,
            total_time_ms,
            compute_time_ms,
            comm_time_ms,
        ) in enumerate(ranked_candidates[:save_count], start=1):
            rank_ir_name = f"{operator_name}_candidate_{rank}_ir.txt"
            rank_code_name = f"{operator_name}_candidate_{rank}_code.py"
            ir_text, code = _serialize_program_for_export(candidate_program)
            with open(
                os.path.join(result_dir, rank_ir_name),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(ir_text)
            with open(
                os.path.join(result_dir, rank_code_name),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(code)

            summary_lines.append(
                (
                    f"    Rank {rank}: candidate_index={candidate_idx}, "
                    f"total={total_time_ms:.6f} ms, compute={compute_time_ms:.6f} ms, "
                    f"comm={comm_time_ms:.6f} ms, IR={rank_ir_name}, code={rank_code_name}"
                )
            )

    summary_path = os.path.join(result_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines))

    print(
        "FFN search completed: "
        f"batch={batch}, seq_len={seq_len}, d_model={d_model}, d_ffn={d_ffn}, "
        f"num_devices={num_devices}, "
        f"top_k={top_k}, layout_top_k={layout_top_k}"
    )
    print(f"Results saved to: {result_dir}/")
    print(f"Summary: {summary_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run FFN gate/up/down search and two-step layout optimization."
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 64)")
    parser.add_argument("--d-model", type=int, default=4096, help="Model hidden dim (default: 256)")
    parser.add_argument("--d-ffn", type=int, default=4096, help="FFN hidden dim (default: 1024)")
    parser.add_argument(
        "--num-devices",
        type=int,
        default=4,
        help="Number of devices (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory root (default: results)",
    )
    parser.add_argument(
        "--hw-config",
        type=str,
        default="config/h100.json",
        help="Hardware config JSON path (default: config/h100.json)",
    )
    parser.add_argument(
        "--mapping-config",
        type=str,
        default="config/ffn_tensor_mapping.json",
        help="Operator mapping config JSON path (default: config/ffn_tensor_mapping.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of per-operator IR/code candidates to save (default: 1)",
    )
    parser.add_argument(
        "--layout-top-k",
        type=int,
        default=10,
        help="Number of step-1 layout plans to keep for step-2 (default: 10)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable dynamic progress bars during candidate generation",
    )
    args = parser.parse_args()

    search_ffn(
        batch=args.batch,
        seq_len=args.seq_len,
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        num_devices=args.num_devices,
        output_dir=args.output_dir,
        top_k=args.top_k,
        layout_top_k=args.layout_top_k,
        hw_config_path=args.hw_config,
        mapping_config_path=args.mapping_config,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
