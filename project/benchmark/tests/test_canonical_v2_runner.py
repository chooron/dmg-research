"""Unit tests for Canonical dPL v2 Runner, 36-Model Manifest, and Post-Hoc Test Evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from project.benchmark.scripts.canonical_v2.run_canonical_v2_model import (
    Phase,
    CURRENT_PHASE,
    execute_canonical_v2_model,
    evaluate_test_partition,
)
from project.benchmark.scripts.canonical_v2.run_canonical_v2_queue import load_manifest


def test_canonical_v2_manifest_structure():
    """Verify that canonical_manifest.yaml contains all 36 models, seed 42, with Penman warmup-length exception.

    NOTE: canonical_manifest.yaml is the FROZEN launch record of the 2026-08-31 run.
    It historically declared `warmup_grad_mode: truncate:90` for penman; that label
    was never implemented (no-op; see PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md)
    and is preserved here only as historical provenance. Active code rejects it.
    """
    canonical_root = Path(__file__).resolve().parents[1] / "results/dpl_canonical_v2_20260831"
    manifest_p = canonical_root / "canonical_manifest.yaml"
    assert manifest_p.exists()

    manifest = load_manifest(manifest_p)
    jobs = manifest["jobs"]
    assert len(jobs) == 36
    assert manifest["seed"] == 42
    assert manifest["kge_eps"] == 0.1

    # Penman warmup-length exception vs 35 standard models (frozen launch record)
    penman_job = [j for j in jobs if j["model"] == "penman"][0]
    assert penman_job["horizon_days"] == 730
    assert penman_job["warmup_days"] == 365
    # Historical launch record preservation: the frozen manifest declared the dead
    # "truncate:90" label. It is kept unchanged as provenance of what the run was
    # launched with; it had zero effect on training (see audit report).
    assert penman_job["warmup_grad_mode"] == "truncate:90"

    for j in jobs:
        if j["model"] != "penman":
            assert j["horizon_days"] == 1095
            assert j["warmup_days"] == 730
            assert j["warmup_grad_mode"] == "detach"
        assert j["optimizer"] == "AdamW"
        assert j["lr"] == 1.0e-3
        assert j["scheduler"] == "none"
        assert j["clip_norm"] == 1.0
        assert j["selection_metric"] == "train_loss"


def test_canonical_v2_test_isolation_gate():
    """Verify that calling evaluate_test_partition during Phase.TRAIN strictly raises RuntimeError."""
    import project.benchmark.scripts.canonical_v2.run_canonical_v2_model as runner_mod
    runner_mod.CURRENT_PHASE = Phase.TRAIN

    dummy_hydro = torch.nn.Identity()
    dummy_net = torch.nn.Linear(10, 4)
    dummy_attrs = torch.zeros((2, 10))
    dummy_x = torch.zeros((10, 2, 3))
    dummy_y = torch.zeros((10, 2))

    with pytest.raises(RuntimeError, match="Test period evaluation attempted during training phase"):
        evaluate_test_partition("gr4j", dummy_hydro, dummy_net, dummy_attrs, dummy_x, dummy_y)

    runner_mod.CURRENT_PHASE = Phase.EVAL


def test_canonical_v2_runner_rejects_dead_warmup_mode():
    """Active runner must fail fast if a job config declares a warmup_grad_mode other than 'detach'."""
    cfg = {
        "model": "penman",
        "warmup_grad_mode": "truncate:90",
    }
    with pytest.raises(RuntimeError, match="Unsupported warmup_grad_mode"):
        execute_canonical_v2_model(cfg, Path(tempfile.mkdtemp()), torch.device("cpu"))


def test_canonical_v2_runner_cpu_smoke():
    """Run a 1-epoch CPU micro-smoke on gr4j and penman under canonical v2 runner."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)

        # 1. Standard model: GR4J (1095d horizon, 730d warmup, full backprop)
        cfg_gr4j = {
            "model": "gr4j",
            "horizon_days": 1095,
            "warmup_days": 730,
            "scored_days": 365,
            "mapping": "auto",
            "lr": 1e-3,
            "epochs": 1,
            "min_epochs": 1,
            "selection_metric": "train_loss",
        }

        res_gr4j = execute_canonical_v2_model(cfg_gr4j, tmp_root, torch.device("cpu"))
        assert res_gr4j["status"] == "DONE"
        assert res_gr4j["exit_code"] == 0

        run_dir_gr4j = tmp_root / "runs" / "gr4j"
        assert (run_dir_gr4j / "best.pt").exists()
        assert (run_dir_gr4j / "best_metadata.json").exists()
        assert (run_dir_gr4j / "test_evaluation.json").exists()
        assert (run_dir_gr4j / "basin_test_kge.csv").exists()
        assert (run_dir_gr4j / "DONE").exists()

        # 2. Penman exception model: 730d horizon, 365d warmup, full backprop
        #    (warmup-LENGTH exception; gradient semantics identical to all models)
        cfg_penman = {
            "model": "penman",
            "horizon_days": 730,
            "warmup_days": 365,
            "scored_days": 365,
            "mapping": "auto",
            "lr": 1e-3,
            "epochs": 1,
            "min_epochs": 1,
            "selection_metric": "train_loss",
        }

        res_penman = execute_canonical_v2_model(cfg_penman, tmp_root, torch.device("cpu"))
        assert res_penman["status"] == "DONE"
        assert res_penman["exit_code"] == 0

        run_dir_penman = tmp_root / "runs" / "penman"
        assert (run_dir_penman / "best.pt").exists()
        assert (run_dir_penman / "test_evaluation.json").exists()
        assert (run_dir_penman / "DONE").exists()