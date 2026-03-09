# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import ast
import copy
import json
import subprocess
import sys
import textwrap

import pytest
import torch

from example_gemm_ir import search_gemm
from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.elements import Axis, Buffer
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.nodes import AxisDef, BufferLoad, BufferStore, Program, ReduceOp, RingComm
from mercury.ir.utils import collect_reduce
from mercury.search.estimate import estimate_program, load_hardware_config
from mercury.search.search import search
from utils.gemm_dsl import gemm_manage_reduction


def _build_programs(m: int, n: int, k: int, mesh_shape):
    source = gemm_manage_reduction.format(M_LEN=m, N_LEN=n, K_LEN=k)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        raise ValueError("Could not find function definition")

    world_size = 1
    for dim in mesh_shape:
        world_size *= dim
    devices = list(range(world_size))
    mesh = DeviceMesh(devices, mesh_shape)

    programs = list(search(program, mesh, ["I", "J", "K"]))
    programs.sort(key=lambda prog: generate_pytorch_code(prog))

    for prog in programs:
        eliminate_loops(prog)

    return programs


def _has_all_reduce(program) -> bool:
    reduce_ops = program.visit(collect_reduce)
    for reduce_op in reduce_ops:
        if len(reduce_op.shard_dim) > 0:
            return True
    return False


def _build_mock_gemm_program(
    name: str,
    mesh_shape,
    ring_dim=None,
    topology_metadata=None,
):
    axis_i = Axis("I", 64, 64)
    axis_j = Axis("J", 64, 64)
    axis_k = Axis("K", 64, 64)

    world_size = 1
    for dim in mesh_shape:
        world_size *= dim
    mesh = DeviceMesh(list(range(world_size)), tuple(mesh_shape))

    replicate_specs = [ShardType.REPLICATE, ShardType.REPLICATE]
    buffer_a = Buffer(
        tensor=f"{name}_a",
        shape=[64, 64],
        bound_axes=[[axis_i], [axis_k]],
        axes_factor=[[1], [1]],
        shard_spec=ShardingSpec(mesh, copy.deepcopy(replicate_specs)),
        read=True,
        write=False,
        dtype=torch.float16,
    )
    buffer_b = Buffer(
        tensor=f"{name}_b",
        shape=[64, 64],
        bound_axes=[[axis_k], [axis_j]],
        axes_factor=[[1], [1]],
        shard_spec=ShardingSpec(mesh, copy.deepcopy(replicate_specs)),
        read=True,
        write=False,
        dtype=torch.float16,
    )
    buffer_c = Buffer(
        tensor=f"{name}_c",
        shape=[64, 64],
        bound_axes=[[axis_i], [axis_j]],
        axes_factor=[[1], [1]],
        shard_spec=ShardingSpec(mesh, copy.deepcopy(replicate_specs)),
        read=True,
        write=True,
        dtype=torch.float16,
    )

    ring_comms = []
    if ring_dim is not None:
        ring_comms.append(
            RingComm(
                axis=axis_k,
                num_cards=int(mesh.shape[ring_dim]),
                name=f"{name}_ring",
                shard_dim=int(ring_dim),
            )
        )

    body = [
        AxisDef(axis_i),
        AxisDef(axis_j),
        AxisDef(axis_k),
        BufferLoad(buffer=buffer_a, indices=[axis_i, axis_k], target=f"{name}_va", comm=ring_comms),
        BufferLoad(buffer=buffer_b, indices=[axis_k, axis_j], target=f"{name}_vb"),
        ReduceOp(
            op="torch.add",
            buffer=buffer_c,
            src=f"{name}_va * {name}_vb",
            axes=[axis_k],
            collective_op="all_reduce",
            shard_dim=[],
            indices=[axis_i, axis_j],
        ),
        BufferStore(buffer=buffer_c, indices=[axis_i, axis_j], value=f"{name}_vc"),
    ]

    program = Program(
        name=name,
        inputs=[],
        defaults=[],
        outputs=[],
        body=body,
        mesh=mesh,
    )
    if topology_metadata is not None:
        program.topology_metadata = topology_metadata
    return program


def test_load_hardware_config_success():
    config = load_hardware_config("config/h100.json")
    assert config["name"] == "H100-SXM"
    assert config["compute"]["peak_tflops"]["bf16"] > 0


def test_h100_required_fields_are_positive():
    config = load_hardware_config("config/h100.json")

    assert config["compute"]["peak_tflops"]["bf16"] > 0
    assert config["compute"]["peak_tflops"]["fp16"] > 0
    assert config["compute"]["peak_tflops"]["fp32"] > 0
    assert config["memory"]["bandwidth_tb_per_s"] > 0
    assert config["interconnect"]["intra_node"]["bandwidth_gb_per_s"] > 0
    assert config["interconnect"]["intra_node"]["latency_us"] >= 0
    assert config["interconnect"]["inter_node"]["bandwidth_gb_per_s"] > 0
    assert config["interconnect"]["inter_node"]["latency_us"] >= 0


def test_load_hardware_config_invalid_field(tmp_path):
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps({"name": "broken", "compute": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_hardware_config(str(invalid_path))


def test_load_hardware_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_hardware_config("config/not_exist.json")


def test_load_hardware_config_alternate_schema_compatible(tmp_path):
    alt_config = {
        "name": "A100-SXM",
        "compute": {
            "peak_tflops": {
                "bf16": 312.0,
                "fp16": 312.0,
                "fp32": 156.0,
            }
        },
        "memory": {
            "bandwidth_tb_per_s": 2.0,
            "capacity_gb": 80.0,
        },
        "interconnect": {
            "intra_node": {
                "bandwidth_gb_per_s": 600.0,
                "latency_us": 3.0,
            },
            "inter_node": {
                "bandwidth_gb_per_s": 200.0,
                "latency_us": 5.0,
            },
        },
    }
    alt_path = tmp_path / "a100.json"
    alt_path.write_text(json.dumps(alt_config), encoding="utf-8")

    loaded = load_hardware_config(str(alt_path))
    assert loaded["name"] == "A100-SXM"


def test_estimate_invariant_and_non_negative():
    programs = _build_programs(64, 64, 64, (2,))
    assert len(programs) > 0

    config = load_hardware_config("config/h100.json")
    estimate = estimate_program(programs[0], config)

    assert estimate.compute_time_ms >= 0
    assert estimate.comm_time_ms >= 0
    assert estimate.total_time_ms >= 0
    assert estimate.total_time_ms <= estimate.compute_time_ms + estimate.comm_time_ms + 1e-9


def test_zero_communication_single_device():
    programs = _build_programs(64, 64, 64, (1,))
    assert len(programs) > 0

    config = load_hardware_config("config/h100.json")
    estimate = estimate_program(programs[0], config)
    assert estimate.comm_time_ms == pytest.approx(0.0)


def test_communication_heavier_than_local_for_comm_cost():
    local_programs = _build_programs(64, 64, 64, (1,))
    dist_programs = _build_programs(64, 64, 64, (2,))

    comm_program = None
    for program in dist_programs:
        if _has_all_reduce(program):
            comm_program = program
            break

    if comm_program is None:
        pytest.skip("No communication-heavy program found in distributed search")

    config = load_hardware_config("config/h100.json")
    local_est = estimate_program(local_programs[0], config)
    comm_est = estimate_program(comm_program, config)

    assert comm_est.comm_time_ms > local_est.comm_time_ms


def test_search_candidates_record_topology_metadata():
    programs = _build_programs(64, 64, 64, (2, 2))
    assert len(programs) > 0

    metadata = programs[0].topology_metadata
    assert "inter_node_dims" in metadata
    assert "intra_node_dims" in metadata
    for dim in metadata["inter_node_dims"] + metadata["intra_node_dims"]:
        assert 0 <= dim < len(programs[0].mesh.shape)


def test_inter_intra_cost_differs_across_mesh_shape():
    config = load_hardware_config("config/h100.json")

    inter_program = _build_mock_gemm_program(
        name="mesh_inter",
        mesh_shape=(4, 2),
        ring_dim=0,
        topology_metadata={"inter_node_dims": [0], "intra_node_dims": [1]},
    )
    intra_program = _build_mock_gemm_program(
        name="mesh_intra",
        mesh_shape=(2, 4),
        ring_dim=1,
        topology_metadata={"inter_node_dims": [0], "intra_node_dims": [1]},
    )

    inter_est = estimate_program(inter_program, config)
    intra_est = estimate_program(intra_program, config)

    assert inter_est.comm_time_ms > intra_est.comm_time_ms


def test_ring_candidate_has_higher_comm_time_than_non_ring():
    config = load_hardware_config("config/h100.json")
    topology = {"inter_node_dims": [0], "intra_node_dims": [1]}

    non_ring_program = _build_mock_gemm_program(
        name="no_ring",
        mesh_shape=(2, 2),
        ring_dim=None,
        topology_metadata=topology,
    )
    ring_program = _build_mock_gemm_program(
        name="ring",
        mesh_shape=(2, 2),
        ring_dim=1,
        topology_metadata=topology,
    )

    non_ring_est = estimate_program(non_ring_program, config)
    ring_est = estimate_program(ring_program, config)
    assert ring_est.comm_time_ms > non_ring_est.comm_time_ms


def test_topk_ranking_changes_stably_with_topology_and_comm_mode():
    config = load_hardware_config("config/h100.json")

    no_comm = _build_mock_gemm_program("rank_no_comm", (2, 2), ring_dim=None)
    ring_dim0 = _build_mock_gemm_program("rank_ring_dim0", (2, 2), ring_dim=0)
    ring_dim1 = _build_mock_gemm_program("rank_ring_dim1", (2, 2), ring_dim=1)
    programs = [no_comm, ring_dim0, ring_dim1]

    topo_a = {"inter_node_dims": [0], "intra_node_dims": [1]}
    for program in programs:
        program.topology_metadata = topo_a
    ranked_a = sorted(programs, key=lambda p: estimate_program(p, config).total_time_ms)
    ranked_a_names = [program.name for program in ranked_a]
    assert ranked_a_names[0] == "rank_no_comm"
    assert ranked_a_names[1:] == ["rank_ring_dim1", "rank_ring_dim0"]

    topo_b = {"inter_node_dims": [1], "intra_node_dims": [0]}
    for program in programs:
        program.topology_metadata = topo_b
    ranked_b = sorted(programs, key=lambda p: estimate_program(p, config).total_time_ms)
    ranked_b_names = [program.name for program in ranked_b]
    assert ranked_b_names[0] == "rank_no_comm"
    assert ranked_b_names[1:] == ["rank_ring_dim0", "rank_ring_dim1"]


def test_top_k_output_and_summary_schema(tmp_path):
    out_dir = tmp_path / "results"
    search_gemm(
        m=64,
        n=64,
        k=64,
        inter_node=1,
        intra_node=2,
        output_dir=str(out_dir),
        top_k=3,
        hw_config_path="config/h100.json",
    )

    result_dir = out_dir / "gemm_64x64x64_inter1_intra2"
    summary_json_path = result_dir / "summary.json"
    summary_txt_path = result_dir / "summary.txt"

    assert summary_json_path.exists()
    assert summary_txt_path.exists()

    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    saved = len(payload["programs"])
    assert saved == min(3, payload["total_searched"])

    for entry in payload["programs"]:
        assert (result_dir / entry["code_file"]).exists()
        assert (result_dir / entry["ir_file"]).exists()
        assert "compute_time_ms" in entry
        assert "comm_time_ms" in entry
        assert "total_time_ms" in entry


def test_cli_default_top_k(tmp_path):
    out_dir = tmp_path / "cli_results"
    cmd = [
        sys.executable,
        "example_gemm_ir.py",
        "--m",
        "64",
        "--n",
        "64",
        "--k",
        "64",
        "--inter-node",
        "1",
        "--intra-node",
        "2",
        "--output-dir",
        str(out_dir),
        "--hw-config",
        "config/h100.json",
    ]
    subprocess.run(cmd, check=True)

    result_dir = out_dir / "gemm_64x64x64_inter1_intra2"
    payload = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["top_k"] == 10
    assert len(payload["programs"]) == min(10, payload["total_searched"])


def test_top_k_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        search_gemm(
            m=64,
            n=64,
            k=64,
            inter_node=1,
            intra_node=1,
            output_dir=str(tmp_path / "results"),
            top_k=0,
            hw_config_path="config/h100.json",
        )
