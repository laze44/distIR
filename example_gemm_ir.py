# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Search all possible tiling and parallelism configurations for GEMM and save
the IR dumps and generated PyTorch code to the results directory."""

import argparse
import ast
import contextlib
import io
import os
import textwrap
from typing import Any, Dict, List, Tuple

from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.utils import get_io_buffers
from mercury.search.dump import dump
from mercury.search.estimate import estimate_program, load_hardware_config
from mercury.search.gemm_dedupe import gemm_canonical_dedupe_key
from mercury.search.mapping_constraints import load_tensor_mapping_constraints
from mercury.search.search import search_with_progress
from mercury.search.topology_policy import make_gemm_mesh_shape_policy


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
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str
                    ):
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


def _format_gemm_source(m: int, n: int, k: int) -> str:
    """Format the GEMM template with validated block sizes."""
    try:
        from utils.gemm_dsl import format_gemm_template

        return format_gemm_template(m, n, k)
    except (ModuleNotFoundError, ImportError):
        from utils.gemm_dsl import gemm_block_size

        template = _load_gemm_template()
        return template.format(
            M_LEN=m,
            N_LEN=n,
            K_LEN=k,
            M_BLOCK=gemm_block_size(m),
            N_BLOCK=gemm_block_size(n),
            K_BLOCK=gemm_block_size(k),
        )


def _capture_dump(program) -> str:
    """Capture the output of dump() as a string."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dump(program)
    return buf.getvalue()


def _product(values: Tuple[int, ...]) -> int:
    """Compute product of integer tuple values."""
    result = 1
    for value in values:
        result *= int(value)
    return result


def _n_dim_to_one_dim(n_dim_index: Tuple[int, ...], dimensions: Tuple[int, ...]) -> int:
    """Convert an N-D index into a flattened index using row-major order."""
    one_dim_index = 0
    for index, dim_size in zip(reversed(n_dim_index), reversed(dimensions)):
        one_dim_index = one_dim_index * int(dim_size) + int(index)
    return one_dim_index


def _extract_abc_device_mapping(program) -> Dict[str, Any]:
    """Extract A/B/C mapping info on each device for one IR program."""
    if program.mesh is None:
        return {}

    buffers = program.visit(get_io_buffers)
    abc_buffers: Dict[str, Any] = {}
    for buffer in buffers:
        if buffer.tensor in ("a", "b", "c") and buffer.tensor not in abc_buffers:
            abc_buffers[buffer.tensor] = buffer

    if len(abc_buffers) == 0:
        return {}

    mesh = program.mesh
    device_coords = {
        int(mesh.get_device(coords)): tuple(int(v) for v in coords)
        for coords in mesh.all_coords()
    }

    tensor_mapping: Dict[str, Any] = {}
    for tensor_name in ("a", "b", "c"):
        buffer = abc_buffers.get(tensor_name)
        if buffer is None or buffer.shard_spec is None:
            continue

        local_shape = [int(dim_size) for dim_size in buffer.get_shape()]
        per_device = {}
        for device_id in sorted(int(dev) for dev in mesh.devices):
            coords = device_coords[device_id]
            dim_mappings: List[Dict[str, Any]] = []
            for dim, spec in enumerate(buffer.shard_spec.specs):
                local_size = local_shape[dim]
                if isinstance(spec, tuple):
                    _, mesh_dims = spec
                    mesh_dims_tuple = tuple(int(v) for v in mesh_dims)
                    shard_coord = tuple(coords[i] for i in mesh_dims_tuple)
                    shard_mesh = tuple(int(mesh.shape[i]) for i in mesh_dims_tuple)
                    shard_index = _n_dim_to_one_dim(shard_coord, shard_mesh)
                    num_shards = _product(shard_mesh)
                    start = shard_index * local_size
                    end = start + local_size
                    dim_mappings.append(
                        {
                            "dim": dim,
                            "local_size": local_size,
                            "global_range": [start, end],
                            "sharding": "S",
                            "mesh_dims": list(mesh_dims_tuple),
                            "shard_index": shard_index,
                            "num_shards": num_shards,
                        }
                    )
                else:
                    dim_mappings.append(
                        {
                            "dim": dim,
                            "local_size": local_size,
                            "global_range": [0, local_size],
                            "sharding": "R",
                            "mesh_dims": [],
                            "shard_index": 0,
                            "num_shards": 1,
                        }
                    )

            per_device[str(device_id)] = {
                "mesh_coord": list(coords),
                "dim_mappings": dim_mappings,
            }

        serialized_shard_spec = []
        for spec in buffer.shard_spec.specs:
            if isinstance(spec, tuple):
                _, mesh_dims = spec
                serialized_shard_spec.append(["S", [int(v) for v in mesh_dims]])
            else:
                serialized_shard_spec.append("R")

        tensor_mapping[tensor_name.upper()] = {
            "shape": local_shape,
            "shard_spec": serialized_shard_spec,
            "per_device": per_device,
        }

    return tensor_mapping


def _validate_mapping_constraints_topology(
    constraints,
    inter_node: int,
    intra_node: int,
    mapping_config_path: str,
) -> None:
    """Fail fast when fixed shard topology tokens cannot exist on the topology."""
    topology_capacity = {
        "inter_node": int(inter_node) > 1,
        "intra_node": int(intra_node) > 1,
        "mixed": int(inter_node) > 1 and int(intra_node) > 1,
    }

    incompatibilities: List[str] = []
    for matrix_name in ("A", "B", "C"):
        matrix_constraint = constraints.get(matrix_name)
        if matrix_constraint.mode != "fixed" or matrix_constraint.mapping is None:
            continue

        for dim_id, dim_mapping in enumerate(matrix_constraint.mapping):
            if dim_mapping.is_replicate or dim_mapping.shard_topology is None:
                continue

            unavailable_tokens = [
                token
                for token in dim_mapping.shard_topology
                if not topology_capacity[token]
            ]
            if len(unavailable_tokens) == 0:
                continue

            missing = ", ".join(unavailable_tokens)
            incompatibilities.append(
                f"{matrix_name}[dim {dim_id}] requires {dim_mapping.to_summary()} "
                f"but topology inter_node={inter_node}, intra_node={intra_node} "
                f"has no shardable {missing} dimension"
            )

    if len(incompatibilities) == 0:
        return

    detail_text = "; ".join(incompatibilities)
    raise ValueError(
        f"Tensor mapping config '{mapping_config_path}' is incompatible with the "
        f"requested topology: {detail_text}. Use '--mapping-config "
        "config/gemm_tensor_mapping.json' for flexible layouts, or choose a "
        "topology with positive cardinality for the required shard dimensions."
    )


def search_gemm(
    m: int,
    n: int,
    k: int,
    inter_node: int,
    intra_node: int,
    output_dir: str,
    top_k: int,
    hw_config_path: str,
    mapping_config_path: str = "config/gemm_tensor_mapping.json",
    show_progress: bool = True,
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
        mapping_config_path: Path to tensor mapping constraint JSON file.
    """
    if inter_node <= 0 or intra_node <= 0:
        raise ValueError("inter_node and intra_node must be positive integers")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    world_size = inter_node * intra_node

    source = _format_gemm_source(m, n, k)

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
    tensor_mapping_constraints = load_tensor_mapping_constraints(mapping_config_path)
    _validate_mapping_constraints_topology(
        tensor_mapping_constraints,
        inter_node,
        intra_node,
        mapping_config_path,
    )

    mesh_shape_policy = make_gemm_mesh_shape_policy(inter_node, intra_node)

    searched_programs = list(
        search_with_progress(
            program,
            mesh,
            ["I", "J", "K"],
            tensor_mapping_constraints=tensor_mapping_constraints,
            progress_desc=f"search[gemm {m}x{n}x{k}]",
            show_progress=show_progress,
            miniters=32,
            mininterval=0.5,
            dedupe_key_fn=gemm_canonical_dedupe_key,
            mesh_shape_policy=mesh_shape_policy,
        )
    )
    # Sort for deterministic ordering (same approach as test_search_gemm.py)
    searched_programs.sort(key=lambda x: generate_pytorch_code(x))

    hw_config = load_hardware_config(hw_config_path)

    estimated_programs = []
    for res_program in searched_programs:
        eliminate_loops(res_program)
        estimate = estimate_program(res_program, hw_config)
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
        f"Requested top_k: {top_k}",
        f"Total programs searched: {len(searched_programs)}",
        f"Top-k saved: {save_count}",
        f"Hardware config: {hw_config['name']}",
        f"Tensor mapping config: {mapping_config_path}",
        "",
    ]

    summary_lines.extend(
        [
            "Summary Metadata (migrated from summary.json):",
            f"  config.m={m}",
            f"  config.n={n}",
            f"  config.k={k}",
            f"  config.inter_node={inter_node}",
            f"  config.intra_node={intra_node}",
            f"  config.world_size={world_size}",
            f"  config.hardware={hw_config['name']}",
            f"  config.top_k={top_k}",
            f"  config.mapping_config={mapping_config_path}",
            f"  total_searched={len(searched_programs)}",
            "",
            "Tensor Mapping Constraints:",
        ]
    )

    for tensor_name, summary in tensor_mapping_constraints.summary_by_matrix().items():
        summary_lines.append(f"  {tensor_name}: {summary}")

    for idx, (res_program, estimate) in enumerate(selected_programs):
        code = generate_pytorch_code(res_program)
        ir_text = _capture_dump(res_program)
        tensor_device_mapping = _extract_abc_device_mapping(res_program)

        code_filename = f"program_{idx + 1}_code.py"
        ir_filename = f"program_{idx + 1}_ir.txt"

        with open(os.path.join(result_dir, code_filename), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(result_dir, ir_filename), "w", encoding="utf-8") as f:
            f.write(ir_text)

        summary_lines.append(f"Program {idx + 1} (Rank {idx + 1}):")
        summary_lines.append(f"  Code: {code_filename}")
        summary_lines.append(f"  IR:   {ir_filename}")
        summary_lines.append(f"  JSON rank field: {idx + 1}")
        summary_lines.append(
            f"  Estimated compute time: {estimate.compute_time_ms:.6f} ms"
        )
        summary_lines.append(
            f"  Estimated communication time: {estimate.comm_time_ms:.6f} ms"
        )
        summary_lines.append(f"  Estimated total time: {estimate.total_time_ms:.6f} ms")
        if len(tensor_device_mapping) > 0:
            summary_lines.append("  Tensor Mapping (A/B/C on each device):")
            for tensor_name in ("A", "B", "C"):
                tensor_info = tensor_device_mapping.get(tensor_name)
                if tensor_info is None:
                    continue
                summary_lines.append(f"    {tensor_name}:")
                for device_id, device_info in tensor_info["per_device"].items():
                    dim_segments = []
                    for dim_info in device_info["dim_mappings"]:
                        seg = (
                            f"d{dim_info['dim']}=[{dim_info['global_range'][0]},{dim_info['global_range'][1]})"
                            f"/{dim_info['sharding']}"
                        )
                        if dim_info["sharding"] == "S":
                            seg += (
                                f"(mesh_dims={dim_info['mesh_dims']},"
                                f" shard={dim_info['shard_index']}/{dim_info['num_shards']})"
                            )
                        dim_segments.append(seg)
                    summary_lines.append(
                        f"      device {device_id}, mesh_coord={device_info['mesh_coord']}: "
                        + "; ".join(dim_segments)
                    )
        summary_lines.append("")

    summary_path = os.path.join(result_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(
        f"Found {len(searched_programs)} program(s) for GEMM {m}x{n}x{k} "
        f"with inter_node={inter_node}, intra_node={intra_node} "
        f"(world_size={world_size})"
    )
    print(f"Saved top-{save_count} program(s) ranked by estimated total time")
    print(f"Results saved to: {result_dir}/")
    print(f"Summary: {summary_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Search all GEMM tiling/parallelism configurations and save "
            "IR + PyTorch code."
        )
    )
    parser.add_argument("--m", type=int, default=16, help="M dimension (default: 512)")
    parser.add_argument(
        "--n", type=int, default=4096, help="N dimension (default: 256)"
    )
    parser.add_argument(
        "--k", type=int, default=4096, help="K dimension (default: 1024)"
    )
    parser.add_argument(
        "--inter-node",
        type=int,
        default=1,
        help="Inter-node mesh dimension (default: 1)",
    )
    parser.add_argument(
        "--intra-node",
        type=int,
        default=4,
        help="Intra-node mesh dimension (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
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
    parser.add_argument(
        "--mapping-config",
        type=str,
        default="config/gemm_tensor_mapping.json",
        help=(
            "Tensor mapping constraint JSON path "
            "(default: config/gemm_tensor_mapping.json)"
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable dynamic progress bars during candidate generation",
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
        args.mapping_config,
        not args.no_progress,
    )


if __name__ == "__main__":
    main()
