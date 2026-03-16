# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""
Tests for IR generation and manipulation.
"""
import torch
import pytest
import ast
import inspect
import textwrap
from typing import Any

from mercury.ir.elements import Axis, Buffer, grid, match_buffer, store_buffer, load_buffer
from mercury.frontend.parser import IRBuilder, auto_schedule
from mercury.ir.nodes import (
    Program, AxisDef, BufferMatch, GridLoop,
)

def test_matmul_ir_gen():
    """Test IR generation for matrix multiplication."""

    def simple_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        # Define axes
        I = Axis("I", 128, min_block_size=32)  # M dimension
        J = Axis("J", 256, min_block_size=32)  # K dimension
        K = Axis("K", 128, min_block_size=32)  # N dimension

        # Match buffers
        A = match_buffer(a, [128, 256], [I, J])  # [M, K]
        B = match_buffer(b, [256, 128], [J, K])  # [K, N]
        C = match_buffer(c, [128, 128], [I, K])  # [M, N]

        # Grid pattern shows J is reduction axis
        for i, j, k in grid([I, J, K], "srs"):
            _c = load_buffer(C[i, k])
            _a = load_buffer(A[i, j])
            _b = load_buffer(B[j, k])
            _c += _a @ _b
            C[i, k] = store_buffer(_c)

    # Create test inputs
    a = torch.randn(128, 256)
    b = torch.randn(256, 128)
    c = torch.zeros(128, 128)

    # Get IR from function
    source = inspect.getsource(simple_matmul)
    source = textwrap.dedent(source)

    print(source)
    
    # Parse the function and build IR
    tree = ast.parse(source)
    builder = IRBuilder()

    # Find and parse the function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    # print IR structure
    print(program)

    # Verify IR structure
    assert isinstance(program, Program)
    
    # Count node types
    node_types = {}
    for node in program.body:
        node_type = type(node).__name__
        node_types[node_type] = node_types.get(node_type, 0) + 1

    # Verify axis definitions
    assert node_types.get('AxisDef', 0) == 3, "Should have 3 axis definitions"
    axis_defs = [n for n in program.body if isinstance(n, AxisDef)]
    assert {ax.axis.name for ax in axis_defs} == {"I", "J", "K"}

    # Verify buffer matching
    assert node_types.get('BufferMatch', 0) == 3, "Should have 3 buffer matches"
    buffer_matches = [n for n in program.body if isinstance(n, BufferMatch)]
    assert {m.tensor_name for m in buffer_matches} == {"A", "B", "C"}

    # Verify grid loop
    grid_loops = [n for n in program.body if isinstance(n, GridLoop)]
    assert len(grid_loops) == 1, "Should have 1 grid loop"
    grid_loop = grid_loops[0]
    assert grid_loop.axis_types == "srs", "Grid should have srs pattern"

    print("✓ Matmul IR generation test passed")

def test_ir_errors():
    """Test error handling in IR generation."""
    def invalid_grid_func(a: torch.Tensor, b: torch.Tensor):
        # Test invalid grid pattern
        I = Axis("I", 10)
        for i in grid([I], "x"):  # Invalid axis type
            pass

    def invalid_buffer_func(a: torch.Tensor):
        # Test mismatched buffer dimensions
        I = Axis("I", 10)
        A = match_buffer(a, [10,], [I, I])  # Too many axes

    # Test invalid grid pattern
    with pytest.raises(ValueError, match="Axis types must be 's' or 'r'"):
        source = inspect.getsource(invalid_grid_func)
        tree = ast.parse(textwrap.dedent(source))
        builder = IRBuilder()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                builder.visit(node)

    # Test mismatched buffer dimensions
    with pytest.raises(ValueError, match="dimensions"):
        source = inspect.getsource(invalid_buffer_func)
        tree = ast.parse(textwrap.dedent(source))
        builder = IRBuilder()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                builder.visit(node)

if __name__ == "__main__":
    test_matmul_ir_gen()
    test_ir_errors()
    print("\nAll tests passed! ✨")

def test_small_even_dimensions_build_ir():
    """Even dimensions below 32 should build GEMM IR successfully."""
    from utils.gemm_dsl import format_gemm_template

    for m, n, k in [(16, 8, 16), (8, 8, 8), (2, 4, 6)]:
        source = format_gemm_template(m, n, k)
        tree = ast.parse(textwrap.dedent(source))
        builder = IRBuilder()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                program = builder.visit(node)
                break
        else:
            pytest.fail(f"Could not find function definition for m={m}, n={n}, k={k}")

        assert isinstance(program, Program)
        axis_defs = [n for n in program.body if isinstance(n, AxisDef)]
        assert len(axis_defs) == 3


def test_small_axis_block_size_equals_dim():
    """For dimensions < 32, min_block_size should equal the dimension size."""
    from utils.gemm_dsl import format_gemm_template

    source = format_gemm_template(8, 16, 8)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break

    axis_defs = {ad.axis.name: ad.axis for ad in program.body if isinstance(ad, AxisDef)}
    assert axis_defs["I"].min_block_size == 8
    assert axis_defs["J"].min_block_size == 16
    assert axis_defs["K"].min_block_size == 8


def test_large_dimensions_keep_block_size_32():
    """Dimensions >= 32 should keep min_block_size=32."""
    from utils.gemm_dsl import format_gemm_template

    source = format_gemm_template(64, 128, 256)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break

    axis_defs = {ad.axis.name: ad.axis for ad in program.body if isinstance(ad, AxisDef)}
    assert axis_defs["I"].min_block_size == 32
    assert axis_defs["J"].min_block_size == 32
    assert axis_defs["K"].min_block_size == 32


def test_odd_small_dimensions_rejected():
    """Odd dimensions below 32 should be rejected."""
    from utils.gemm_dsl import format_gemm_template

    with pytest.raises(ValueError, match="even"):
        format_gemm_template(15, 64, 64)

    with pytest.raises(ValueError, match="even"):
        format_gemm_template(64, 7, 64)

    with pytest.raises(ValueError, match="even"):
        format_gemm_template(64, 64, 3)


def test_search_small_axis_split_candidates():
    """Search on small axes should only produce unsplit or one binary-split candidates."""
    from utils.gemm_dsl import format_gemm_template
    from mercury.ir.distributed import DeviceMesh
    from mercury.ir.loop_eliminating import eliminate_loops
    from mercury.ir.utils import collect_axis
    from mercury.search.search import search

    source = format_gemm_template(64, 64, 8)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break

    mesh = DeviceMesh(list(range(2)), (2,))
    candidates = list(search(program, mesh, ["I", "J", "K"]))
    assert len(candidates) > 0

    for candidate in candidates:
        axes = candidate.visit(collect_axis)
        k_axes = [a for a in axes if a.name.startswith("K")]
        # Either one K axis (unsplit) or K_outer + K_inner (binary split)
        assert len(k_axes) in (1, 2), f"Expected 1 or 2 K axes, got {len(k_axes)}: {[a.name for a in k_axes]}"
        if len(k_axes) == 2:
            names = {a.name for a in k_axes}
            assert names == {"K_outer", "K_inner"}, f"Unexpected K axis names: {names}"


def test_search_small_axis_no_repeated_splits():
    """Search should never produce repeated subdivision of a small axis."""
    from utils.gemm_dsl import format_gemm_template
    from mercury.ir.distributed import DeviceMesh
    from mercury.ir.utils import collect_axis
    from mercury.search.search import search

    source = format_gemm_template(8, 8, 8)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break

    mesh = DeviceMesh(list(range(2)), (2,))
    candidates = list(search(program, mesh, ["I", "J", "K"]))

    for candidate in candidates:
        axes = candidate.visit(collect_axis)
        # No axis should have _inner_inner or _outer_outer (no repeated splits)
        for axis in axes:
            assert "_inner_inner" not in axis.name, f"Repeated inner split detected: {axis.name}"
            assert "_outer_outer" not in axis.name, f"Repeated outer split detected: {axis.name}"


def test_search_small_axis_no_non_binary_splits():
    """Small axes should never be split by non-binary factors (4-way, 8-way, etc)."""
    from utils.gemm_dsl import format_gemm_template
    from mercury.ir.distributed import DeviceMesh
    from mercury.ir.utils import collect_axis
    from mercury.search.search import enumerate_axis_split
    from mercury.ir.elements import Axis

    # Simulate a small axis (size=8, min_block_size=8)
    small_axis = Axis("K", 8, min_block_size=8)
    splits = list(enumerate_axis_split([small_axis], 4, []))
    # Should only produce [1] and [2]
    assert splits == [[1], [2]], f"Expected [[1], [2]], got {splits}"

    # For odd small axis (size=6), should only produce [1] and [2]
    small_axis_6 = Axis("K", 6, min_block_size=6)
    splits_6 = list(enumerate_axis_split([small_axis_6], 4, []))
    assert splits_6 == [[1], [2]], f"Expected [[1], [2]], got {splits_6}"


def test_small_dim_through_search_lowering_codegen():
    """Exercise small-dimension GEMM through search, lowering, and code generation."""
    from utils.gemm_dsl import format_gemm_template
    from mercury.ir.distributed import DeviceMesh
    from mercury.ir.loop_eliminating import eliminate_loops
    from mercury.backend import generate_pytorch_code
    from mercury.search.search import search
    from mercury.search.estimate import estimate_program, load_hardware_config

    for m, n, k in [(16, 8, 16), (8, 8, 8)]:
        source = format_gemm_template(m, n, k)
        tree = ast.parse(textwrap.dedent(source))
        builder = IRBuilder()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                program = builder.visit(node)
                break

        mesh = DeviceMesh(list(range(2)), (2,))
        candidates = list(search(program, mesh, ["I", "J", "K"]))
        assert len(candidates) > 0, f"No candidates for {m}x{n}x{k}"

        hw_config = load_hardware_config("config/h100.json")
        for idx, candidate in enumerate(candidates):
            eliminate_loops(candidate)
            code = generate_pytorch_code(candidate)
            assert len(code) > 0, f"Empty code for candidate {idx} of {m}x{n}x{k}"
            assert "def matmul" in code, f"Missing function def in candidate {idx}"

            estimate = estimate_program(candidate, hw_config)
            assert estimate.compute_time_ms > 0
            assert estimate.total_time_ms >= 0


def test_small_dim_single_device_codegen():
    """Single-device small GEMM generates valid code without communication."""
    from utils.gemm_dsl import format_gemm_template
    from mercury.ir.distributed import DeviceMesh
    from mercury.ir.loop_eliminating import eliminate_loops
    from mercury.backend import generate_pytorch_code
    from mercury.search.search import search

    source = format_gemm_template(8, 8, 8)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break

    mesh = DeviceMesh([0], (1,))
    candidates = list(search(program, mesh, ["I", "J", "K"]))
    assert len(candidates) > 0

    for candidate in candidates:
        eliminate_loops(candidate)
        code = generate_pytorch_code(candidate)
        assert "def matmul" in code
        # Single device shouldn't have all_reduce
        assert "all_reduce" not in code
