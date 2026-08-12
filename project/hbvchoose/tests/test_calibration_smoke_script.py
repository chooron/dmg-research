"""Quick smoke test for calibration script."""

import sys, os, csv
from pathlib import Path
import pytest

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from scripts.calibrate_formula_combinations_smoke import run_calibration


def test_calibration_smoke_runs():
    out = _PROJECT / "validation_results" / "formula_calibration_smoke" / "test"
    out.mkdir(parents=True, exist_ok=True)
    run_calibration(max_combos=2, steps=2, cases=["case_01_dry"], output_dir=str(out))
    assert (out / "calibration_smoke_summary.csv").exists()
    assert (out / "calibration_smoke_raw_steps.csv").exists()


def test_calibration_has_finite_loss():
    out = _PROJECT / "validation_results" / "formula_calibration_smoke" / "test"
    with open(out / "calibration_smoke_summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    for r in rows:
        import math
        assert float(r["initial_loss"]) >= 0
        assert math.isfinite(float(r["final_loss"]))


def test_no_nan_grad_in_smoke():
    out = _PROJECT / "validation_results" / "formula_calibration_smoke" / "test"
    with open(out / "calibration_smoke_summary.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert r["nan_in_grad"] == "False" or r["nan_in_grad"] == "0"
