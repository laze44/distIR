# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tests for CPU-only reshard cost estimation."""

import math

import torch

from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.elements import Axis, Buffer
from mercury.search.estimate import load_hardware_config
from mercury.search.reshard_estimate import estimate_reshard_time


def _build_buffer(
    name: str,
    mesh_shape,
    local_shape,
    specs,
) -> Buffer:
    mesh = DeviceMesh(list(range(int(math.prod(mesh_shape)))), tuple(mesh_shape))
    axis_i = Axis("I", int(local_shape[0]), int(local_shape[0]))
    axis_j = Axis("J", int(local_shape[1]), int(local_shape[1]))
    return Buffer(
        tensor=name,
        shape=[int(local_shape[0]), int(local_shape[1])],
        bound_axes=[[axis_i], [axis_j]],
        axes_factor=[[1], [1]],
        shard_spec=ShardingSpec(mesh=mesh, specs=specs),
        read=True,
        write=False,
        dtype=torch.bfloat16,
    )


def test_estimate_reshard_same_layout_zero():
    hw = load_hardware_config("config/h100.json")
    origin_mesh = DeviceMesh(list(range(4)), (2, 2))

    src = _build_buffer(
        "src",
        mesh_shape=(2, 2),
        local_shape=(32, 64),
        specs=[(ShardType.SHARD, [0]), ShardType.REPLICATE],
    )
    dst = _build_buffer(
        "dst",
        mesh_shape=(2, 2),
        local_shape=(32, 64),
        specs=[(ShardType.SHARD, [0]), ShardType.REPLICATE],
    )

    assert estimate_reshard_time(src, dst, hw, origin_mesh) == 0.0


def test_estimate_reshard_inter_node_larger_than_intra_node():
    hw = load_hardware_config("config/h100.json")
    src = _build_buffer(
        "src",
        mesh_shape=(2, 2),
        local_shape=(32, 64),
        specs=[(ShardType.SHARD, [0]), ShardType.REPLICATE],
    )
    dst = _build_buffer(
        "dst",
        mesh_shape=(2, 2),
        local_shape=(32, 64),
        specs=[(ShardType.SHARD, [1]), ShardType.REPLICATE],
    )

    inter_origin = DeviceMesh(list(range(4)), (2, 2))
    intra_origin = DeviceMesh(list(range(4)), (1, 4))

    inter_cost = estimate_reshard_time(src, dst, hw, inter_origin)
    intra_cost = estimate_reshard_time(src, dst, hw, intra_origin)

    assert inter_cost > intra_cost


def test_estimate_reshard_partial_replicated_to_sharded_is_finite_non_zero():
    hw = load_hardware_config("config/h100.json")
    origin_mesh = DeviceMesh(list(range(4)), (2, 2))

    src = _build_buffer(
        "src",
        mesh_shape=(2, 2),
        local_shape=(32, 64),
        specs=[(ShardType.SHARD, [0]), ShardType.REPLICATE],
    )
    dst = _build_buffer(
        "dst",
        mesh_shape=(2, 2),
        local_shape=(64, 32),
        specs=[ShardType.REPLICATE, (ShardType.SHARD, [1])],
    )

    cost = estimate_reshard_time(src, dst, hw, origin_mesh)
    assert math.isfinite(cost)
    assert cost > 0.0


def test_estimate_reshard_different_mesh_shapes_same_world_size():
    hw = load_hardware_config("config/h100.json")
    origin_mesh = DeviceMesh(list(range(4)), (2, 2))

    src = _build_buffer(
        "src",
        mesh_shape=(4,),
        local_shape=(16, 64),
        specs=[(ShardType.SHARD, [0]), ShardType.REPLICATE],
    )
    dst = _build_buffer(
        "dst",
        mesh_shape=(2, 2),
        local_shape=(32, 64),
        specs=[(ShardType.SHARD, [1]), ShardType.REPLICATE],
    )

    cost = estimate_reshard_time(src, dst, hw, origin_mesh)
    assert math.isfinite(cost)
    assert cost >= 0.0
