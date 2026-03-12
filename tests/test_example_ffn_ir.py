# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Smoke test for the FFN example CLI."""

import subprocess
import sys


def test_example_ffn_ir_cli_smoke(tmp_path):
    output_root = tmp_path / "results"
    cmd = [
        sys.executable,
        "example_ffn_ir.py",
        "--batch",
        "1",
        "--seq-len",
        "32",
        "--d-model",
        "32",
        "--d-ffn",
        "64",
        "--inter-node",
        "1",
        "--intra-node",
        "2",
        "--output-dir",
        str(output_root),
        "--hw-config",
        "config/h100.json",
        "--mapping-config",
        "config/ffn_tensor_mapping.json",
        "--top-k",
        "2",
        "--layout-top-k",
        "2",
    ]
    subprocess.run(cmd, check=True)

    result_dir = output_root / "ffn_b1_l32_dm32_df64_inter1_intra2"
    assert result_dir.exists()

    expected_files = [
        "summary.txt",
        "gate_ir.txt",
        "gate_code.py",
        "gate_candidate_1_ir.txt",
        "gate_candidate_1_code.py",
        "gate_candidate_2_ir.txt",
        "gate_candidate_2_code.py",
        "up_ir.txt",
        "up_code.py",
        "up_candidate_1_ir.txt",
        "up_candidate_1_code.py",
        "up_candidate_2_ir.txt",
        "up_candidate_2_code.py",
        "down_ir.txt",
        "down_code.py",
        "down_candidate_1_ir.txt",
        "down_candidate_1_code.py",
        "down_candidate_2_ir.txt",
        "down_candidate_2_code.py",
    ]
    for file_name in expected_files:
        assert (result_dir / file_name).exists()

    summary_text = (result_dir / "summary.txt").read_text(encoding="utf-8")
    assert "Two-Step Search Results:" in summary_text
    assert "Step-1 Top Layout Plans:" in summary_text
    assert "Selected Layout Plan:" in summary_text
    assert "Selected L_out:" in summary_text
    assert "Step-2 Operator Costs (ms):" in summary_text
    assert "Requested top_k per operator: 2" in summary_text
    assert "Requested layout_top_k: 2" in summary_text
    assert "Top-k Candidate Files:" in summary_text
    assert "Rank 2: candidate_index=" in summary_text
    assert "gate:" in summary_text
    assert "up:" in summary_text
    assert "down:" in summary_text
