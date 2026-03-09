# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import ast
import json
import subprocess
import sys
import textwrap

import pytest

from example_gemm_ir import search_gemm
from mercury.backend import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.loop_eliminating import eliminate_loops
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
    estimate = estimate_program(programs[0], config, num_inter_dims=0)

    assert estimate.compute_time_ms >= 0
    assert estimate.comm_time_ms >= 0
    assert estimate.total_time_ms >= 0
    assert estimate.total_time_ms == pytest.approx(
        estimate.compute_time_ms + estimate.comm_time_ms
    )


def test_zero_communication_single_device():
    programs = _build_programs(64, 64, 64, (1,))
    assert len(programs) > 0

    config = load_hardware_config("config/h100.json")
    estimate = estimate_program(programs[0], config, num_inter_dims=0)
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
    local_est = estimate_program(local_programs[0], config, num_inter_dims=0)
    comm_est = estimate_program(comm_program, config, num_inter_dims=0)

    assert comm_est.comm_time_ms > local_est.comm_time_ms


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
