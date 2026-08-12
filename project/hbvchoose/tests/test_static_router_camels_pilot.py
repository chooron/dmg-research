"""Tests for CAMELS StaticRouter pilot script."""

import sys
from pathlib import Path

import pytest
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))


class TestCamelsPilotScript:

    def test_synthetic_fallback_runs(self):
        from scripts.train_static_router_camels_pilot import run_camels_pilot
        import argparse
        out = _PROJECT / "validation_results" / "static_router_camels_pilot" / "test_synth"
        args = argparse.Namespace(
            num_basins=2, steps=2, warmup=20, eval_len=40,
            seed=42, lr=1e-3, synthetic_fallback=True,
            output_dir=str(out),
        )
        success, _ = run_camels_pilot(args)
        assert success, "synthetic pilot should succeed"
        assert (out / "camels_pilot_training_steps.csv").exists()
        assert (out / "camels_pilot_basin_metrics.csv").exists()
        assert (out / "camels_pilot_report.md").exists()

    def test_loss_finite_in_synthetic(self):
        from scripts.train_static_router_camels_pilot import run_camels_pilot
        import argparse, csv, math
        out = _PROJECT / "validation_results" / "static_router_camels_pilot" / "test_finite"
        args = argparse.Namespace(
            num_basins=2, steps=3, warmup=20, eval_len=40,
            seed=42, lr=1e-3, synthetic_fallback=True,
            output_dir=str(out),
        )
        run_camels_pilot(args)
        with open(out / "camels_pilot_training_steps.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1
        for r in rows:
            assert math.isfinite(float(r["loss_total"]))
            assert float(r["nan_in_loss"]) == 0
            assert float(r["nan_in_grad"]) == 0

    def test_basin_metrics_has_nse(self):
        from scripts.train_static_router_camels_pilot import run_camels_pilot
        import argparse, csv
        out = _PROJECT / "validation_results" / "static_router_camels_pilot" / "test_nse"
        args = argparse.Namespace(
            num_basins=2, steps=2, warmup=20, eval_len=40,
            seed=42, lr=1e-3, synthetic_fallback=True,
            output_dir=str(out),
        )
        run_camels_pilot(args)
        with open(out / "camels_pilot_basin_metrics.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1
        for r in rows:
            assert "default_NSE" in r
            assert "router_NSE" in r

    def test_does_not_change_hbv_static(self):
        from model.hbv_static import HbvStatic
        import inspect
        hbv = HbvStatic()
        sig = inspect.signature(HbvStatic.forward)
        params = list(sig.parameters)
        assert "x_dict" in params
        assert "parameters" in params

    def test_q5_not_in_main_router(self):
        from model.static_formula_router import StaticFormulaRouter
        router = StaticFormulaRouter(attr_dim=8)
        fids = router.formula_ids
        for node in fids:
            assert "Q5" not in fids[node], f"Q5 found in {node} main formulas"

    def test_router_weights_finite_after_training(self):
        from scripts.train_static_router_camels_pilot import run_camels_pilot
        import argparse, csv, math
        out = _PROJECT / "validation_results" / "static_router_camels_pilot" / "test_weights"
        args = argparse.Namespace(
            num_basins=2, steps=3, warmup=20, eval_len=40,
            seed=42, lr=1e-3, synthetic_fallback=True,
            output_dir=str(out),
        )
        run_camels_pilot(args)
        with open(out / "camels_pilot_training_steps.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1
        for r in rows:
            for key in ["entropy_snow", "entropy_recharge", "entropy_aet", "entropy_response"]:
                assert math.isfinite(float(r[key])), f"{key} not finite"
