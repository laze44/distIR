# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Search all possible tiling and parallelism configurations for GEMM and save
the IR dumps and generated PyTorch code to the results directory."""

import argparse
import ast
import contextlib
import io
import json
import os
import textwrap

from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.search.dump import dump
from mercury.search.estimate import estimate_program, load_hardware_config
from mercury.search.search import search


def _extract_template_from_file(template_name: str) -> str:
    """Extract a template string from utils/gemm_dsl.py via AST without importing."""
    file_path = os.path.join(os.path.dirname(__file__), "utils", "gemm_dsl.py")
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


def _load_gemm_template() -> str:
    """Load the GEMM DSL template with a dependency-safe fallback."""
    template_name = "gemm_manage_reduction"
    try:
        from utils.gemm_dsl import gemm_manage_reduction
        return gemm_manage_reduction
    except (ModuleNotFoundError, ImportError):
        return _extract_template_from_file(template_name)


def _capture_dump(program) -> str:
    """Capture the output of dump() as a string."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dump(program)
    return buf.getvalue()


def search_gemm(
    m: int,
    n: int,
    k: int,
    inter_node: int,
    intra_node: int,
    output_dir: str,
    top_k: int,
    hw_config_path: str,
) -> None:
    """Search all tiling/parallelism configurations and save results.

    Args:
        m: M dimension of the matrix multiplication.
        n: N dimension of the matrix multiplication.
        k: K dimension of the matrix multiplication.
        inter_node: Number of inter-node partitions in the mesh.
        intra_node: Number of intra-node partitions in the mesh.
        output_dir: Directory to write result files into.
        top_k: Number of best estimated programs to persist.
        hw_config_path: Path to hardware configuration JSON file.
    """
    if inter_node <= 0 or intra_node <= 0:
        raise ValueError("inter_node and intra_node must be positive integers")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    world_size = inter_node * intra_node

    gemm_template = _load_gemm_template()
    source = gemm_template.format(M_LEN=m, N_LEN=n, K_LEN=k)

    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        raise ValueError("Could not find function definition in GEMM template")

    devices = list(range(world_size))
    mesh = DeviceMesh(devices, (inter_node, intra_node))

    searched_programs = list(search(program, mesh, ["I", "J", "K"]))
    # Sort for deterministic ordering (same approach as test_search_gemm.py)
    searched_programs.sort(key=lambda x: generate_pytorch_code(x))

    hw_config = load_hardware_config(hw_config_path)

    estimated_programs = []
    num_inter_dims = 1 if inter_node > 1 else 0
    for res_program in searched_programs:
        eliminate_loops(res_program)
        estimate = estimate_program(
            res_program,
            hw_config,
            num_inter_dims=num_inter_dims,
        )
        estimated_programs.append((res_program, estimate))

    estimated_programs.sort(key=lambda item: item[1].total_time_ms)

    save_count = min(top_k, len(estimated_programs))
    selected_programs = estimated_programs[:save_count]

    result_dir = os.path.join(
        output_dir,
        f"gemm_{m}x{n}x{k}_inter{inter_node}_intra{intra_node}",
    )
    os.makedirs(result_dir, exist_ok=True)

    summary_lines = [
        (
            f"GEMM Search Results: M={m}, N={n}, K={k}, "
            f"inter_node={inter_node}, intra_node={intra_node}, world_size={world_size}"
        ),
        f"Total programs searched: {len(searched_programs)}",
        f"Top-k saved: {save_count}",
        f"Hardware config: {hw_config['name']}",
        "",
    ]

    summary_json = {
        "config": {
            "m": m,
            "n": n,
            "k": k,
            "inter_node": inter_node,
            "intra_node": intra_node,
            "world_size": world_size,
            "hardware": hw_config["name"],
            "top_k": top_k,
        },
        "total_searched": len(searched_programs),
        "programs": [],
    }

    for idx, (res_program, estimate) in enumerate(selected_programs):
        code = generate_pytorch_code(res_program)
        ir_text = _capture_dump(res_program)

        code_filename = f"program_{idx + 1}_code.py"
        ir_filename = f"program_{idx + 1}_ir.txt"

        with open(os.path.join(result_dir, code_filename), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(result_dir, ir_filename), "w", encoding="utf-8") as f:
            f.write(ir_text)

        summary_lines.append(f"Program {idx + 1} (Rank {idx + 1}):")
        summary_lines.append(f"  Code: {code_filename}")
        summary_lines.append(f"  IR:   {ir_filename}")
        summary_lines.append(f"  Estimated compute time: {estimate.compute_time_ms:.6f} ms")
        summary_lines.append(f"  Estimated communication time: {estimate.comm_time_ms:.6f} ms")
        summary_lines.append(f"  Estimated total time: {estimate.total_time_ms:.6f} ms")
        summary_lines.append("")

        summary_json["programs"].append({
            "rank": idx + 1,
            "code_file": code_filename,
            "ir_file": ir_filename,
            "compute_time_ms": estimate.compute_time_ms,
            "comm_time_ms": estimate.comm_time_ms,
            "total_time_ms": estimate.total_time_ms,
        })

    summary_path = os.path.join(result_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    summary_json_path = os.path.join(result_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(
        f"Found {len(searched_programs)} program(s) for GEMM {m}x{n}x{k} "
        f"with inter_node={inter_node}, intra_node={intra_node} "
        f"(world_size={world_size})"
    )
    print(f"Saved top-{save_count} program(s) ranked by estimated total time")
    print(f"Results saved to: {result_dir}/")
    print(f"Summary: {summary_path}")
    print(f"Summary JSON: {summary_json_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Search all GEMM tiling/parallelism configurations and save "
            "IR + PyTorch code."
        )
    )
    parser.add_argument("--m", type=int, default=512, help="M dimension (default: 512)")
    parser.add_argument("--n", type=int, default=256, help="N dimension (default: 256)")
    parser.add_argument("--k", type=int, default=1024, help="K dimension (default: 1024)")
    parser.add_argument(
        "--inter-node",
        type=int,
        default=1,
        help="Inter-node mesh dimension (default: 1)",
    )
    parser.add_argument(
        "--intra-node",
        type=int,
        default=2,
        help="Intra-node mesh dimension (default: 2)",
    )
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results (default: results)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of best estimated programs to save (default: 10)",
    )
    parser.add_argument(
        "--hw-config",
        type=str,
        default="config/h100.json",
        help="Hardware configuration JSON path (default: config/h100.json)",
    )
    args = parser.parse_args()
    search_gemm(
        args.m,
        args.n,
        args.k,
        args.inter_node,
        args.intra_node,
        args.output_dir,
        args.top_k,
        args.hw_config,
    )


if __name__ == "__main__":
    main()
