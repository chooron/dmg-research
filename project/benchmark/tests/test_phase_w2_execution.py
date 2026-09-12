"""Unit tests for Phase W2 (730d Warmup + 365d Scored) execution, 2x2 decomposition, and report generation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from project.benchmark.scripts.ablation.ablation_runner import execute_ablation_job
from project.benchmark.scripts.ablation.evaluate_2x2_warmup import evaluate_inner_val_kge, run_2x2_evaluation
from project.benchmark.scripts.ablation.run_ablation_queue import load_manifest


def test_phase_w2_manifest_structure():
    """Verify that Phase W2 has exactly 8 jobs covering the 8 target models with 730d warmup and 365d scored."""
    ablation_root = Path(__file__).resolve().parents[1] / "results/dpl_protocol_ablation_v2_20260831"
    manifest_p = ablation_root / "experiment_manifest.yaml"
    assert manifest_p.exists()

    manifest = load_manifest(manifest_p)
    w2_jobs = [j for j in manifest["jobs"] if j.get("phase") == "W2"]
    assert len(w2_jobs) == 8

    expected_models = {"flexb", "topmodel", "collie1", "gr4j", "hbv96", "mopex4", "xinanjiang", "penman"}
    actual_models = {j["model"] for j in w2_jobs}
    assert actual_models == expected_models

    for j in w2_jobs:
        assert j["horizon_days"] == 1095
        assert j["warmup_days"] == 730
        assert j["scored_days"] == 365
        assert j["max_optimizer_updates"] > 0
        assert j["parent_h1_job"].startswith("H")


def test_phase_w2_runner_cpu_smoke():
    """Run a 3-step CPU micro-smoke on W2 configuration (1095d horizon, 730d warmup)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)

        job_config = {
            "job_id": "TEST_W201_gr4j_smoke",
            "phase": "W2",
            "model": "gr4j",
            "horizon_days": 1095,
            "warmup_days": 730,
            "scored_days": 365,
            "mapping": "auto",
            "lr": 1e-3,
            "max_optimizer_updates": 3,
            "selection_metric": "train_loss",
            "inner_val_freq": 1,
        }

        res = execute_ablation_job(job_config, tmp_root, torch.device("cpu"))
        assert res["status"] == "DONE"
        assert res["actual_optimizer_updates"] == 3

        run_dir = tmp_root / "runs" / "TEST_W201_gr4j_smoke"
        assert (run_dir / "best.pt").exists()
        assert (run_dir / "DONE").exists()


def test_2x2_warmup_decomposition_logic():
    """Verify that 2x2 warmup decomposition correctly computes Train Warmup and Eval Warmup effects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)

        # Mock audit CSV
        df_audit = pd.DataFrame([
            {
                "model": "gr4j",
                "category": "control_structure",
                "H1_job_id": "H01_gr4j_730",
                "W2_job_id": "W204_gr4j_w730_s365",
                "W2_target_updates": 300,
            }
        ])
        df_audit.to_csv(tmp_root / "PHASE_W2_UPDATE_BUDGET_AUDIT.csv", index=False)

        mock_manifest = {
            "experiment_id": "test_w2",
            "jobs": [
                {
                    "job_id": "W204_gr4j_w730_s365",
                    "phase": "W2",
                    "model": "gr4j",
                }
            ],
        }
        with open(tmp_root / "experiment_manifest.yaml", "w") as f:
            yaml.dump(mock_manifest, f)

        # Create dummy runs directories and dummy best.pt checkpoints
        runs_dir = tmp_root / "runs"
        (runs_dir / "H01_gr4j_730").mkdir(parents=True, exist_ok=True)
        (runs_dir / "W204_gr4j_w730_s365").mkdir(parents=True, exist_ok=True)

        from dpl.nn_parameterizer import CatchmentParameterizer
        net = CatchmentParameterizer(35, 4, hidden_dims=[256, 256], dropout=0.05)
        torch.save({"network": net.state_dict()}, runs_dir / "H01_gr4j_730" / "best.pt")
        torch.save({"network": net.state_dict()}, runs_dir / "W204_gr4j_w730_s365" / "best.pt")

        run_2x2_evaluation(tmp_root, device_str="cpu")

        assert (tmp_root / "PHASE_W2_2X2_WARMUP_DECOMPOSITION.csv").exists()
        assert (tmp_root / "PHASE_W2_SUMMARY.csv").exists()
        assert (tmp_root / "PHASE_W2_REPORT.md").exists()

        df_res = pd.read_csv(tmp_root / "PHASE_W2_2X2_WARMUP_DECOMPOSITION.csv")
        assert len(df_res) == 1
        assert "delta_train_warmup_med (T730-T365 @ E730)" in df_res.columns
        assert "verdict" in df_res.columns
