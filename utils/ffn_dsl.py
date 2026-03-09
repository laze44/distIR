# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""FFN GEMM DSL templates."""

import torch
import torch.distributed as dist


def add_collective(tensor, op, group, dst=None):
    tensor_red = tensor.contiguous()
    if op == dist.all_reduce:
        dist.all_reduce(tensor_red, dist.ReduceOp.SUM, group=group)
    else:
        raise ValueError("not support other op than all reduce")
    tensor = tensor_red


ffn_gate_gemm_manage_reduction = """
def gate_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    I = Axis("I", {M_LEN}, min_block_size=32)
    K = Axis("K", {DM_LEN}, min_block_size=32)
    J = Axis("J", {DFFN_LEN}, min_block_size=32)

    A = match_buffer(a, [{M_LEN}, {DM_LEN}], [I, K])
    B = match_buffer(b, [{DM_LEN}, {DFFN_LEN}], [K, J])
    C = match_buffer(c, [{M_LEN}, {DFFN_LEN}], [I, J])

    for i, j in grid([I, J], "ss"):
        reduce_buf = temp_buffer([i, j], [I, J], dtype=torch.float32)
        for k in grid([K], "m"):
            _a = load_buffer(A[i, k])
            _b = load_buffer(B[k, j])
            _c = torch.matmul(_a, _b)
            block_res = _c.to(torch.float32)
            reduce(
                op=torch.add,
                buffer=reduce_buf,
                collective_op=add_collective,
                src=block_res,
                axis=k,
            )
        reduce_res = load_buffer(reduce_buf[:, :])
        C[i, j] = store_buffer(reduce_res.to(c.dtype))
"""


ffn_up_gemm_manage_reduction = """
def up_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    I = Axis("I", {M_LEN}, min_block_size=32)
    K = Axis("K", {DM_LEN}, min_block_size=32)
    J = Axis("J", {DFFN_LEN}, min_block_size=32)

    A = match_buffer(a, [{M_LEN}, {DM_LEN}], [I, K])
    B = match_buffer(b, [{DM_LEN}, {DFFN_LEN}], [K, J])
    C = match_buffer(c, [{M_LEN}, {DFFN_LEN}], [I, J])

    for i, j in grid([I, J], "ss"):
        reduce_buf = temp_buffer([i, j], [I, J], dtype=torch.float32)
        for k in grid([K], "m"):
            _a = load_buffer(A[i, k])
            _b = load_buffer(B[k, j])
            _c = torch.matmul(_a, _b)
            block_res = _c.to(torch.float32)
            reduce(
                op=torch.add,
                buffer=reduce_buf,
                collective_op=add_collective,
                src=block_res,
                axis=k,
            )
        reduce_res = load_buffer(reduce_buf[:, :])
        C[i, j] = store_buffer(reduce_res.to(c.dtype))
"""


ffn_down_gemm_manage_reduction = """
def down_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    I = Axis("I", {M_LEN}, min_block_size=32)
    K = Axis("K", {DFFN_LEN}, min_block_size=32)
    J = Axis("J", {DM_LEN}, min_block_size=32)

    A = match_buffer(a, [{M_LEN}, {DFFN_LEN}], [I, K])
    B = match_buffer(b, [{DFFN_LEN}, {DM_LEN}], [K, J])
    C = match_buffer(c, [{M_LEN}, {DM_LEN}], [I, J])

    for i, j in grid([I, J], "ss"):
        reduce_buf = temp_buffer([i, j], [I, J], dtype=torch.float32)
        for k in grid([K], "m"):
            _a = load_buffer(A[i, k])
            _b = load_buffer(B[k, j])
            _c = torch.matmul(_a, _b)
            block_res = _c.to(torch.float32)
            reduce(
                op=torch.add,
                buffer=reduce_buf,
                collective_op=add_collective,
                src=block_res,
                axis=k,
            )
        reduce_res = load_buffer(reduce_buf[:, :])
        C[i, j] = store_buffer(reduce_res.to(c.dtype))
"""
