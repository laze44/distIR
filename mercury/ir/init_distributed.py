# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import copy

from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.nodes import Program
from mercury.ir.utils import get_buffers


def init_distributed(program: Program, mesh: DeviceMesh):
    """Initialize distributed IR with device mesh.

    Args:
        program: IR program
        mesh: device mesh
    """
    # Add device mesh to program
    program.mesh = mesh

    # Initialize distributed buffers
    buffers = program.visit(get_buffers)
    for buffer in buffers:
        base_spec = ShardingSpec(mesh, [ShardType.REPLICATE] * len(buffer.shape))
        buffer.shard_spec = base_spec
        buffer.logical_shard_spec = copy.deepcopy(base_spec)
        buffer.global_shape = [
            int(dim.size) if hasattr(dim, "size") else int(dim) for dim in buffer.shape
        ]
