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
        "gate_selected_ir.txt",
        "gate_selected_code.py",
        "gate_candidate_1_ir.txt",
        "gate_candidate_1_code.py",
        "gate_candidate_2_ir.txt",
        "gate_candidate_2_code.py",
        "up_ir.txt",
        "up_code.py",
        "up_selected_ir.txt",
        "up_selected_code.py",
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
    assert (
        (result_dir / "down_selected_ir.txt").exists()
        or (result_dir / "down_chain_selected_ir.txt").exists()
    )
    assert (
        (result_dir / "down_selected_code.py").exists()
        or (result_dir / "down_chain_selected_code.py").exists()
    )

    summary_text = (result_dir / "summary.txt").read_text(encoding="utf-8")
    assert "Two-Step Search Results:" in summary_text
    assert "Step-1 Top Layout Plans:" in summary_text
    assert "Selected Layout Plan:" in summary_text
    assert "Selected L_out:" in summary_text
    assert "Step-2 Operator Costs (ms):" in summary_text
    assert "Requested top_k per operator: 2" in summary_text
    assert "Requested layout_top_k: 2" in summary_text
    assert "Operator Candidate Counts:" in summary_text
    assert "Step-1 Layout Plan Counts:" in summary_text
    assert "unique L_in:" in summary_text
    assert "unique L_mid:" in summary_text
    assert "gate boundary classes:" in summary_text
    assert "up boundary classes:" in summary_text
    assert "down boundary classes:" in summary_text
    assert "projected unique L_out:" in summary_text
    assert "projected unique W_gate:" in summary_text
    assert "projected unique W_up:" in summary_text
    assert "projected unique W_down:" in summary_text
    assert "total evaluated plans:" in summary_text
    assert "retained top-k plans:" in summary_text
    assert "gate boundary class:" in summary_text
    assert "up boundary class:" in summary_text
    assert "down boundary class:" in summary_text
    assert "down.C == L_out: True" in summary_text
    assert "Selected Step-2 Segments:" in summary_text
    assert "Explicit Edge Obligations And Ownership:" in summary_text
    assert "Canonical Selected Operator Files:" in summary_text
    assert "Top-k Candidate Files (Non-Canonical Debug):" in summary_text
    assert "auxiliary and may differ from selected segments" in summary_text
    assert "Rank 2: candidate_index=" in summary_text
    assert "gate:" in summary_text
    assert "up:" in summary_text
    assert "down:" in summary_text
