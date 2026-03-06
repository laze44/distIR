# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Generate and save Ring Attention PyTorch code without running execution."""

import argparse
import ast
import os
import textwrap

from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.init_distributed import init_distributed
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.primitives import parallelize, shift
from mercury.ir.utils import collect_axis, collect_loops


def _extract_template_from_file(template_name: str) -> str:
    """Extract template string from utils/flash_attn_dsl.py without importing it."""
    file_path = os.path.join(os.path.dirname(__file__), "utils", "flash_attn_dsl.py")
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


def _load_flash_attn_template() -> str:
    """Load flash attention template with a dependency-safe fallback."""
    template_name = "flash_attn_pack_kv_template"
    try:
        from utils.flash_attn_dsl import flash_attn_pack_kv_template

        return flash_attn_pack_kv_template
    except ModuleNotFoundError as error:
        if error.name != "flash_attn":
            raise
        return _extract_template_from_file(template_name)


def build_ring_attn_code(world_size: int) -> str:
    """Generate transformed Ring Attention PyTorch code.

    Args:
        world_size: Number of devices in the logical mesh.

    Returns:
        Generated PyTorch code as a string.
    """
    flash_attn_pack_kv_template = _load_flash_attn_template()
    source = flash_attn_pack_kv_template.format(BATCH=4, HEADS=5, SEQ_LEN=4096, HEAD_DIM=128)
    tree = ast.parse(textwrap.dedent(source))

    builder = IRBuilder()
    program = builder.visit(tree.body[0])

    axes = program.visit(collect_axis)
    loops = program.visit(collect_loops)

    axis = axes[2]
    outer_loop = loops[0]

    devices = list(range(world_size))
    mesh = DeviceMesh(devices, (world_size,))

    init_distributed(program, mesh)
    parallelize(program, outer_loop, axis, mesh, 0, len(mesh.shape))
    shift(program, axis, mesh, 0, len(mesh.shape), 1)
    eliminate_loops(program)

    return generate_pytorch_code(program)


def save_code(code_text: str, output_path: str) -> None:
    """Save generated code text to file.

    Args:
        code_text: Generated code content.
        output_path: Target file path.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(code_text)


def main() -> None:
    """CLI entry for generating and saving Ring Attention PyTorch code."""
    parser = argparse.ArgumentParser(description="Generate Ring Attention PyTorch code only.")
    parser.add_argument("--world-size", type=int, default=2, help="Logical device count in mesh.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/ring_attn_codegen.py",
        help="Output path for dumped generated PyTorch code.",
    )
    args = parser.parse_args()

    code_text = build_ring_attn_code(args.world_size)
    save_code(code_text, args.output)
    print(f"Generated code saved to: {args.output}")


if __name__ == "__main__":
    main()