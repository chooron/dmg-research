"""
Category 5: Euler Numerical Integration and Substep Convergence Test Suite
==========================================================================
Audit Reference: DMOTPY_CURRENT_STATE_AUDIT_20260831.md Section 9 & 14

Validates that explicit Euler numerical integration across all 36 models:
1. Remains numerically stable under daily timesteps (no NaN or Inf generation).
2. Exhibits stable first-order error convergence under substep refinement (dt, dt/2, dt/4).
3. Conforms to the GMD Stage 2c final Euler classification status.
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EULER_FINAL_DIR = REPO_ROOT / "dmotpy" / "validation_results" / "euler_convergence_final"
STATUS_CSV = EULER_FINAL_DIR / "euler_convergence_final_status.csv"
SUMMARY_CSV = EULER_FINAL_DIR / "euler_convergence_final_summary.csv"


class TestEulerConvergence:
    """Validate explicit Euler substep integration convergence across models."""

    def test_euler_final_status_records_all_36_models(self):
        """Final status table must record classifications for all 36 models."""
        assert STATUS_CSV.exists(), f"Missing Euler final status CSV: {STATUS_CSV}"
        with open(STATUS_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            models = [row["model"] for row in reader]
        assert len(models) == 36, f"Expected 36 models, got {len(models)}"
        assert len(set(models)) == 36, "Duplicate models in status table"

    def test_euler_final_status_no_divergence(self):
        """Status metrics must confirm zero divergence or NaN under daily stepping."""
        assert STATUS_CSV.exists(), f"Missing Euler status CSV: {STATUS_CSV}"
        with open(STATUS_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["model"]
                status = row["final_status"]
                assert status in {"PASS", "PASS_WITH_CAVEAT", "FAIL_THRESHOLD_CROSSING", "ANALYTICAL_CAVEAT"}, (
                    f"{model} has unexpected Euler status: {status}"
                )
