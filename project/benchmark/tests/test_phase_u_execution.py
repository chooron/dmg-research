"""Unit tests for Phase U Update-Budget Matched execution, stopping precision, and report generation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from project.benchmark.scripts.ablation.ablation_runner import execute_ablation_job
from project.benchmark.scripts.ablation.run_ablation_queue import generate_phase_u_report, load_manifest


def test_phase_u_exact_update_budget_stopping():
    """Verify that setting max_optimizer_updates stops at the exact update number."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)

        job_config = {
            "job_id": "TEST_U01_exact_stop",
            "phase": "U",
            "model": "gr4j",
            "horizon_days": 1825,
            "warmup_days": 365,
            "scored_days": 1460,
            "mapping": "auto",
            "lr": 1e-3,
            "max_optimizer_updates": 5,  # Exactly 5 updates
            "selection_metric": "train_loss",
            "inner_val_freq": 1,
        }

        res = execute_ablation_job(job_config, tmp_root, torch.device("cpu"))
        assert res["status"] == "DONE"
        assert res["actual_optimizer_updates"] == 5

        run_dir = tmp_root / "runs" / "TEST_U01_exact_stop"
        assert (run_dir / "DONE").exists()
        assert (run_dir / "inner_validation_by_update.csv").exists()

        df_up = pd.read_csv(run_dir / "inner_validation_by_update.csv")
        assert len(df_up) >= 1
        assert "inner_val_median_kge" in df_up.columns
        assert df_up["update"].iloc[-1] == 5


def test_phase_u_report_generation():
    """Verify that generate_phase_u_report correctly computes 3-point contrast delta_H, delta_U, delta_MATCHED."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)

        # Mock manifest with Phase H and Phase U jobs
        mock_manifest = {
            "experiment_id": "test_phase_u",
            "jobs": [
                {
                    "job_id": "U01_hbv96_1825_update_matched",
                    "phase": "U",
                    "model": "hbv96",
                    "max_optimizer_updates": 295,
                    "status": "ELIGIBLE",
                }
            ],
        }
        import yaml
        with open(tmp_root / "experiment_manifest.yaml", "w") as f:
            yaml.dump(mock_manifest, f)

        # Mock Phase H summary
        df_h = pd.DataFrame([
            {"model": "hbv96", "horizon_days": 730, "inner_val_median_kge": 0.7000, "inner_val_q25_kge": 0.5800, "best_train_loss": 0.3942},
            {"model": "hbv96", "horizon_days": 1825, "inner_val_median_kge": 0.6760, "inner_val_q25_kge": 0.5416, "best_train_loss": 0.3522},
        ])
        df_h.to_csv(tmp_root / "PHASE_H_SUMMARY.csv", index=False)

        # Mock Phase U run output
        u_run_dir = tmp_root / "runs" / "U01_hbv96_1825_update_matched"
        u_run_dir.mkdir(parents=True, exist_ok=True)
        (u_run_dir / "DONE").write_text("COMPLETED\n")
        (u_run_dir / "inner_validation_metrics.json").write_text(json.dumps({"median": 0.7100, "q25": 0.5900, "mean": 0.6800}))
        (u_run_dir / "runtime.json").write_text(json.dumps({"total_runtime_s": 100.0, "actual_optimizer_updates": 295}))
        (u_run_dir / "best_metadata.json").write_text(json.dumps({"best_selection_value": 0.3300, "actual_optimizer_updates": 295}))

        generate_phase_u_report(tmp_root)

        assert (tmp_root / "PHASE_U_SUMMARY.csv").exists()
        assert (tmp_root / "PHASE_U_MATCHED_COMPARISON.csv").exists()
        assert (tmp_root / "PHASE_U_REPORT.md").exists()

        df_matched = pd.read_csv(tmp_root / "PHASE_U_MATCHED_COMPARISON.csv")
        assert len(df_matched) == 1
        row = df_matched.iloc[0]
        assert row["model"] == "hbv96"
        assert row["h1_730d_med_kge"] == 0.7000
        assert row["h2_1825d_med_kge"] == 0.6760
        assert row["u_1825d_matched_med_kge"] == 0.7100
        # Delta U = 0.7100 - 0.6760 = +0.0340
        assert pytest.approx(row["delta_U_med (U - H2)"], 1e-4) == 0.0340
        # Delta MATCHED = 0.7100 - 0.7000 = +0.0100 -> LONG_HORIZON_WINS_MATCHED
        assert pytest.approx(row["delta_MATCHED_med (U - H1)"], 1e-4) == 0.0100
        assert row["verdict"] == "LONG_HORIZON_WINS_MATCHED"
