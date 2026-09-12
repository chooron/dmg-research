"""Unit tests for remediated dPL runner protocol, test isolation, exact best checkpointing, and clipping telemetry."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCHMARK_DIR), str(BENCHMARK_DIR / "src"), str(Path(__file__).resolve().parents[2] / "dmotpy")]

def _load_runner():
    runner_path = BENCHMARK_DIR / "scripts/diagnostics/k_full_retrain.py"
    spec = importlib.util.spec_from_file_location("k_full_retrain_mod", runner_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

runner = _load_runner()
Phase = runner.Phase
CURRENT_PHASE = runner.CURRENT_PHASE
evaluate_test_period = runner.evaluate_test_period
save_best_checkpoint = runner.save_best_checkpoint
best_checkpoint = runner.best_checkpoint
run_model = runner.run_model
from dpl.nn_parameterizer import CatchmentParameterizer
from src.model_registry import build_model

def test_test_isolation_gate():
    """Verify that calling evaluate_test_period during TRAIN phase strictly raises RuntimeError."""
    import project.benchmark.scripts.diagnostics.k_full_retrain as runner
    runner.CURRENT_PHASE = Phase.TRAIN
    
    dummy_hydro = nn.Identity()
    dummy_network = nn.Linear(10, 4)
    dummy_attrs = torch.zeros((2, 10))
    dummy_x = torch.zeros((10, 2, 3))
    dummy_y = torch.zeros((10, 2))

    with pytest.raises(RuntimeError, match="Test period evaluation attempted during training phase"):
        evaluate_test_period("gr4j", dummy_hydro, dummy_network, dummy_attrs, dummy_x, dummy_y)

    # In EVAL phase, it does not raise phase error
    runner.CURRENT_PHASE = Phase.EVAL
    # (function may proceed normally or fail on mock inputs, but won't raise the phase guard RuntimeError)


def test_exact_best_checkpoint_provenance():
    """Verify that best.pt is saved with exact metadata and can be restored identically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import project.benchmark.scripts.diagnostics.k_full_retrain as runner
        old_out = runner.OUT
        runner.OUT = Path(tmpdir)
        try:
            arm = "auto100"
            model = "gr4j"
            net = nn.Linear(10, 4)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            
            # Save at a non-multiple of 10, e.g. epoch 7
            saved_path = save_best_checkpoint(
                arm=arm,
                model=model,
                epoch=7,
                network=net,
                optimizer=opt,
                metric_name="train_loss",
                metric_value=0.3456,
                extra_metadata={"custom_flag": True},
            )
            assert saved_path.exists()
            assert saved_path.name == "best.pt"

            payload = torch.load(saved_path, map_location="cpu", weights_only=False)
            assert payload["epoch"] == 7
            assert payload["metadata"]["best_epoch"] == 7
            assert payload["metadata"]["selection_metric_name"] == "train_loss"
            assert payload["metadata"]["selection_metric_value"] == 0.3456
            assert payload["metadata"]["kge_eps"] == 0.1
            assert payload["metadata"]["custom_flag"] is True

            # Verify parameter equality
            restored_net = nn.Linear(10, 4)
            restored_net.load_state_dict(payload["network"])
            for p1, p2 in zip(net.parameters(), restored_net.parameters()):
                torch.testing.assert_close(p1, p2)
        finally:
            runner.OUT = old_out


def test_cpu_micro_smoke_and_clipping_telemetry():
    """Run 1-step CPU micro-smoke on gr4j and flexb and verify clipping telemetry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_out = runner.OUT
        old_device = runner.DEVICE
        old_steps = runner.STEPS
        old_min_ep = runner.MIN_EPOCHS
        runner.OUT = Path(tmpdir)
        runner.DEVICE = torch.device("cpu")
        runner.MIN_EPOCHS = 1

        try:
            # 1 epoch, 2 steps, batch 2 on CPU
            res_gr4j = runner.run_model("auto100", "gr4j", epochs=1, lr=1e-3, steps_per_epoch=2, batch_size=2)
            assert res_gr4j["model"] == "gr4j"
            assert res_gr4j["best_epoch"] == 1
            assert "train_loss_improvement" in res_gr4j
            
            epoch_csv = runner.OUT / "auto100" / "epochs.csv"
            assert epoch_csv.exists()
            
            import csv
            with open(epoch_csv) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) >= 1
            row = rows[0]
            assert "grad_norm_preclip_median" in row
            assert "grad_clip_fraction" in row
            assert float(row["grad_norm_preclip_median"]) >= 0.0

            # Verify best.pt exists
            assert runner.best_checkpoint("auto100", "gr4j").exists()

            # Test flexb micro-smoke on CPU as well
            res_flexb = runner.run_model("auto100", "flexb", epochs=1, lr=1e-3, steps_per_epoch=2, batch_size=2)
            assert res_flexb["model"] == "flexb"
            assert runner.best_checkpoint("auto100", "flexb").exists()
        finally:
            runner.OUT = old_out
            runner.DEVICE = old_device
            runner.STEPS = old_steps
            runner.MIN_EPOCHS = old_min_ep
