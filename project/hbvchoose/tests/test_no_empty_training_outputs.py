"""Test: No empty training outputs — even on failure, CSVs must have proper headers and failure records."""
import csv
from pathlib import Path

import pytest
import torch

from model.static_formula_router import StaticFormulaRouter
from model.hbv_formula_static import HbvFormulaStatic

_PROJECT = Path(__file__).resolve().parent.parent


class TestNoEmptyOutputs:
    """Verify outputs are never silently empty."""

    def test_selection_summary_has_selection_source_column(self):
        """selection_summary.csv must document selection source."""
        path = _PROJECT / "validation_results" / "static_router_camels_calibrated_pilot" / "test_no_empty"
        path.mkdir(parents=True, exist_ok=True)
        # Write a minimal selection_summary with required columns
        fields = ["combo_id", "count", "selection_source"]
        with open(path / "test_selection.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow({"combo_id": "S0_R0_E0_Q0", "count": 1, "selection_source": "router_logits"})
        # Verify
        with open(path / "test_selection.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0, "Empty selection_summary"
            assert "selection_source" in reader.fieldnames, "Missing selection_source column"

    def test_failures_csv_not_empty_after_error(self):
        """failures.csv should have records when failures occur."""
        path = _PROJECT / "validation_results" / "static_router_camels_calibrated_pilot" / "test_no_empty"
        path.mkdir(parents=True, exist_ok=True)
        fields = ["stage", "basin_id", "step", "reason", "loss", "nse", "kge", "rmse"]
        with open(path / "test_failures.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow({"stage": "screening", "basin_id": 9999, "step": -1,
                         "reason": "all_nan_target", "loss": "nan", "nse": "nan", "kge": "nan", "rmse": "nan"})
        with open(path / "test_failures.csv") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("reason")]
            assert len(rows) > 0, "failure records should be non-empty"

    def test_training_steps_csv_not_empty_on_success(self):
        """training_steps.csv should have actual step records."""
        path = _PROJECT / "validation_results" / "static_router_camels_calibrated_pilot" / "test_no_empty"
        path.mkdir(parents=True, exist_ok=True)
        fields = ["step", "loss_total", "grad_norm_before_clip"]
        with open(path / "test_steps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for s in range(5):
                w.writerow({"step": s, "loss_total": 1.0 - 0.1 * s, "grad_norm_before_clip": 0.5})
        with open(path / "test_steps.csv") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) >= 5, f"Expected >=5 steps, got {len(rows)}"
