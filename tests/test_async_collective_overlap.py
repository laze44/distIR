# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import ast
import textwrap

from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.utils import collect_reduce
from mercury.search.search import search
from utils.gemm_dsl import format_gemm_template


def _build_gemm_program(m_len: int = 64, n_len: int = 128, k_len: int = 64):
    source = format_gemm_template(m_len, n_len, k_len)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return builder.visit(node)
    raise ValueError("Could not find function definition")


def test_gemm_search_emits_async_collective_overlap_candidate_and_codegen():
    program = _build_gemm_program()
    mesh = DeviceMesh([0, 1], (2,))
    candidates = list(search(program, mesh, ["I", "J", "K"]))

    async_candidates = [
        candidate
        for candidate in candidates
        if any(
            reduce_op.managed_collective_strategy == "async_collective_overlap"
            for reduce_op in candidate.visit(collect_reduce)
        )
    ]
    assert len(async_candidates) > 0

    candidate = async_candidates[0]
    reduce_ops = candidate.visit(collect_reduce)
    assert any(reduce_op.async_collective_stage_count == 2 for reduce_op in reduce_ops)

    eliminate_loops(candidate)
    code = generate_pytorch_code(candidate)
    assert "all_reduce" in code


def test_search_keeps_ring_overlap_candidates_for_managed_reduction():
    program = _build_gemm_program()
    mesh = DeviceMesh([0, 1], (2,))
    candidates = list(search(program, mesh, ["I", "J", "K"]))

    ring_candidates = [
        candidate
        for candidate in candidates
        if any(len(reduce_op.comm) > 0 for reduce_op in candidate.visit(collect_reduce))
    ]
    assert len(ring_candidates) > 0
    assert any(
        reduce_op.managed_collective_strategy == "ring_overlap"
        for candidate in ring_candidates
        for reduce_op in candidate.visit(collect_reduce)
    )
