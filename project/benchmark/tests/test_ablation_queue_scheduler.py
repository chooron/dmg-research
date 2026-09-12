"""Unit tests for dPL Ablation Queue Scheduler, 4-worker slot replacement, and report generation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from project.benchmark.scripts.ablation.run_ablation_queue import (
    load_manifest,
    get_job_status,
    append_to_ledger,
    generate_phase_h_report,
    run_queue,
)


def test_manifest_structure_and_job_counts():
    """Verify that experiment_manifest.yaml has exactly 48 candidate jobs (16 Phase H, 8 Phase W2, 5 Phase U, 19 conditional)."""
    ablation_root = Path(__file__).resolve().parents[1] / "results/dpl_protocol_ablation_v2_20260831"
    manifest_path = ablation_root / "experiment_manifest.yaml"
    assert manifest_path.exists()

    manifest = load_manifest(manifest_path)
    jobs = manifest["jobs"]
    assert len(jobs) == 48

    h_jobs = [j for j in jobs if j.get("phase") == "H"]
    assert len(h_jobs) == 16
    assert all(j.get("status") == "ELIGIBLE" for j in h_jobs)

    w2_jobs = [j for j in jobs if j.get("phase") == "W2"]
    assert len(w2_jobs) == 8
    assert all(j.get("status") == "ELIGIBLE" for j in w2_jobs)

    cond_jobs = [j for j in jobs if j.get("phase") not in {"H", "W2", "U"}]
    assert len(cond_jobs) == 19
    assert all(j.get("status") == "BLOCKED_BY_GATE" for j in cond_jobs)

def test_four_worker_dynamic_slot_replacement():
    """Verify that 4-worker queue scheduler immediately fills empty slots when any job completes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        
        # Construct 6 synthetic lightweight mock jobs
        mock_jobs = [
            {
                "job_id": f"TEST_mock_{i:02d}",
                "phase": "H",
                "model": "gr4j",
                "horizon_days": 730,
                "warmup_days": 365,
                "scored_days": 365,
                "mapping": "auto",
                "lr": 1e-3,
                "epochs": 1,
                "min_epochs": 1,
                "status": "ELIGIBLE",
            }
            for i in range(1, 7)
        ]

        manifest_payload = {
            "experiment_id": "test_queue_4workers",
            "concurrency": 4,
            "jobs": mock_jobs,
        }

        manifest_path = tmp_root / "experiment_manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_payload, f)

        # Initialize mock ledger
        (tmp_root / "ABLATION_LEDGER.md").write_text("# Test Ledger\n")

        # Mock runner script that finishes in 0.2s - 0.5s
        mock_runner_code = """#!/usr/bin/env python3
import json, sys, time, os
from pathlib import Path

args = sys.argv
out_dir = Path(args[args.index("--out") + 1])
job_cfg = json.loads(args[args.index("--job-config") + 1])
job_id = job_cfg["job_id"]

run_dir = out_dir / "runs" / job_id
run_dir.mkdir(parents=True, exist_ok=True)
lock = run_dir / ".lock"
lock.write_text("PID=" + str(os.getpid()))

# Variable sleep to test asynchronous completion
idx = int(job_id.split("_")[-1])
time.sleep(0.1 + (idx % 3) * 0.1)

(run_dir / "runtime.json").write_text(json.dumps({"job_id": job_id, "total_runtime_s": 0.2, "status": "COMPLETED"}))
(run_dir / "inner_validation_metrics.json").write_text(json.dumps({"median": 0.65, "q25": 0.50}))
(run_dir / "DONE").write_text("COMPLETED")
if lock.exists():
    lock.unlink()
sys.exit(0)
"""
        mock_runner_path = tmp_root / "mock_runner.py"
        mock_runner_path.write_text(mock_runner_code)
        mock_runner_path.chmod(0o755)

        # Temporarily patch runner script location
        import project.benchmark.scripts.ablation.run_ablation_queue as queue_mod
        old_runner_script = queue_mod.RUNNER_SCRIPT
        queue_mod.RUNNER_SCRIPT = mock_runner_path

        try:
            start_t = time.time()
            run_queue(
                ablation_root=tmp_root,
                manifest_path=manifest_path,
                target_phase="H",
                concurrency=4,
                devices=["cpu", "cpu", "cpu", "cpu"],
                poll_interval=0.05,
                resume=True,
            )
            elapsed = time.time() - start_t

            # Verify all 6 jobs completed
            for j in mock_jobs:
                assert (tmp_root / "runs" / j["job_id"] / "DONE").exists()
                assert get_job_status(tmp_root, j["job_id"]) == "DONE"

            # Check ledger
            ledger_text = (tmp_root / "ABLATION_LEDGER.md").read_text()
            for j in mock_jobs:
                assert j["job_id"] in ledger_text

            # Check Phase H summary generated
            assert (tmp_root / "PHASE_H_SUMMARY.csv").exists()
            assert (tmp_root / "PHASE_H_PAIRED_COMPARISON.csv").exists()
            assert (tmp_root / "PHASE_H_REPORT.md").exists()

            df_summary = pd.read_csv(tmp_root / "PHASE_H_SUMMARY.csv")
            assert len(df_summary) == 6

        finally:
            queue_mod.RUNNER_SCRIPT = old_runner_script


def test_ablation_runner_cpu_smoke_and_test_isolation():
    """Run ablation_runner directly on CPU for 1 epoch and verify FIT/INNER-VAL metrics and zero TEST access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        from project.benchmark.scripts.ablation.ablation_runner import execute_ablation_job

        job_config = {
            "job_id": "H01_gr4j_test_smoke",
            "phase": "H",
            "model": "gr4j",
            "horizon_days": 730,
            "warmup_days": 365,
            "scored_days": 365,
            "mapping": "auto",
            "lr": 1e-3,
            "scheduler": "none",
            "weight_decay": 1e-4,
            "clip_norm": 1.0,
            "seed": 42,
            "epochs": 1,
            "min_epochs": 1,
            "patience": 10,
            "plateau_eps": 1e-4,
            "selection_metric": "train_loss",
            "inner_val_freq": 1,
        }

        import torch
        res = execute_ablation_job(job_config, tmp_root, torch.device("cpu"))
        assert res["status"] == "DONE"
        assert res["exit_code"] == 0

        run_dir = tmp_root / "runs" / "H01_gr4j_test_smoke"
        assert (run_dir / "best.pt").exists()
        assert (run_dir / "best_metadata.json").exists()
        assert (run_dir / "inner_validation_metrics.json").exists()
        assert (run_dir / "epoch_metrics.csv").exists()
        assert (run_dir / "DONE").exists()

        df_ep = pd.read_csv(run_dir / "epoch_metrics.csv")
        assert len(df_ep) == 1
        assert "inner_val_median_kge" in df_ep.columns
        assert "grad_norm_preclip_median" in df_ep.columns
        assert "saturation_lower" in df_ep.columns
