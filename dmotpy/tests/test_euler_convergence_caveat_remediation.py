"""Tests for Euler convergence caveat-model remediation results.

Verifies the outputs written by
    scripts/remediate_euler_convergence_caveat_models.py

Hard constraints (identical to the production harness):
  * No hydrological formulas are modified, smoothed, or clamped.
  * No parameter bounds, soft-gate defaults, or unit-hydrograph code are changed.
  * No model physics or water-balance fixes are altered.
  * Caveat models are NOT forced to pass by weakening criteria.

These tests validate the *documentation* of remediation outcomes; they do not
re-run the full substep simulations (which are covered by the script itself).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMEDIATION_DIR = (
    PROJECT_ROOT / "validation_results" / "euler_convergence_caveat_remediation"
)

SUMMARY_CSV = REMEDIATION_DIR / "caveat_model_remediation_summary.csv"
ORDERS_CSV = REMEDIATION_DIR / "caveat_model_remediation_orders.csv"
ERRORS_CSV = REMEDIATION_DIR / "caveat_model_remediation_errors.csv"
REPORT_MD = REMEDIATION_DIR / "caveat_model_remediation_report.md"

PASS_BAND = (0.85, 1.15)

# Models expected to PASS in smooth-domain scenarios
EXPECTED_PASS = {"mopex4", "mopex5", "tank", "tcm"}
# Models expected to retain a non-PASS status (fundamental caveats)
EXPECTED_CAVEAT = {
    "gsfb": "STRUCTURAL_CAVEAT",
    "gr4j": "ANALYTICAL_CAVEAT",
}


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


def test_summary_csv_exists():
    assert SUMMARY_CSV.exists(), f"Missing {SUMMARY_CSV}"


def test_orders_csv_exists():
    assert ORDERS_CSV.exists(), f"Missing {ORDERS_CSV}"


def test_errors_csv_exists():
    assert ERRORS_CSV.exists(), f"Missing {ERRORS_CSV}"


def test_report_md_exists():
    assert REPORT_MD.exists(), f"Missing {REPORT_MD}"


# ---------------------------------------------------------------------------
# Summary CSV content
# ---------------------------------------------------------------------------


def test_summary_contains_all_six_caveat_models():
    rows = _read_csv(SUMMARY_CSV)
    models_found = {r["model"] for r in rows}
    expected = EXPECTED_PASS | set(EXPECTED_CAVEAT.keys())
    missing = expected - models_found
    assert not missing, f"Missing models in summary CSV: {missing}"


@pytest.mark.parametrize("model", sorted(EXPECTED_PASS))
def test_pass_models_have_pass_status(model):
    rows = _read_csv(SUMMARY_CSV)
    row = next((r for r in rows if r["model"] == model), None)
    assert row is not None, f"Model {model!r} not found in summary CSV"
    assert row["status"] == "PASS", (
        f"{model}: expected status=PASS, got {row['status']!r}"
    )


@pytest.mark.parametrize("model", sorted(EXPECTED_PASS))
def test_pass_models_in_pass_band(model):
    rows = _read_csv(SUMMARY_CSV)
    row = next((r for r in rows if r["model"] == model), None)
    assert row is not None
    order = float(row["median_empirical_order"])
    assert PASS_BAND[0] <= order <= PASS_BAND[1], (
        f"{model}: median_empirical_order={order:.4f} outside [{PASS_BAND[0]}, {PASS_BAND[1]}]"
    )


@pytest.mark.parametrize("model", sorted(EXPECTED_PASS))
def test_pass_models_monotone(model):
    rows = _read_csv(SUMMARY_CSV)
    row = next((r for r in rows if r["model"] == model), None)
    assert row is not None
    assert row["state_errors_monotone"].lower() in ("true", "1"), (
        f"{model}: state_errors_monotone={row['state_errors_monotone']!r}, expected True"
    )


@pytest.mark.parametrize("model,expected_status", sorted(EXPECTED_CAVEAT.items()))
def test_caveat_models_retain_caveat_status(model, expected_status):
    rows = _read_csv(SUMMARY_CSV)
    row = next((r for r in rows if r["model"] == model), None)
    assert row is not None, f"Model {model!r} not found in summary CSV"
    assert row["status"] == expected_status, (
        f"{model}: expected status={expected_status!r}, got {row['status']!r}"
    )


def test_gsfb_status_is_structural_caveat():
    """gsfb must be STRUCTURAL_CAVEAT — not forced to PASS."""
    rows = _read_csv(SUMMARY_CSV)
    row = next((r for r in rows if r["model"] == "gsfb"), None)
    assert row is not None
    assert row["status"] == "STRUCTURAL_CAVEAT", (
        f"gsfb status={row['status']!r}; must remain STRUCTURAL_CAVEAT "
        "(hard torch.minimum clamps are irreducible)"
    )


def test_gr4j_status_is_analytical_caveat():
    """gr4j must be ANALYTICAL_CAVEAT — closed-form daily update, not an ODE."""
    rows = _read_csv(SUMMARY_CSV)
    row = next((r for r in rows if r["model"] == "gr4j"), None)
    assert row is not None
    assert row["status"] == "ANALYTICAL_CAVEAT", (
        f"gr4j status={row['status']!r}; must remain ANALYTICAL_CAVEAT"
    )


# ---------------------------------------------------------------------------
# Orders CSV
# ---------------------------------------------------------------------------


def test_orders_csv_has_rows():
    rows = _read_csv(ORDERS_CSV)
    assert len(rows) > 0, "caveat_model_remediation_orders.csv is empty"


@pytest.mark.parametrize("model", sorted(EXPECTED_PASS))
def test_pass_model_orders_all_have_four_pairs(model):
    """Each PASS model should have 4 substep pairs (1→2, 2→4, 4→8, 8→16)."""
    rows = _read_csv(ORDERS_CSV)
    model_rows = [r for r in rows if r["model"] == model]
    assert len(model_rows) == 4, (
        f"{model}: expected 4 order rows, got {len(model_rows)}"
    )


# ---------------------------------------------------------------------------
# Errors CSV
# ---------------------------------------------------------------------------


def test_errors_csv_has_rows():
    rows = _read_csv(ERRORS_CSV)
    assert len(rows) > 0, "caveat_model_remediation_errors.csv is empty"


@pytest.mark.parametrize("model", sorted(EXPECTED_PASS))
def test_pass_model_errors_present(model):
    rows = _read_csv(ERRORS_CSV)
    model_rows = [r for r in rows if r["model"] == model]
    assert len(model_rows) > 0, f"No error rows for model {model!r}"


# ---------------------------------------------------------------------------
# Report markdown
# ---------------------------------------------------------------------------


def test_report_md_mentions_structural_caveat():
    text = REPORT_MD.read_text()
    assert "STRUCTURAL_CAVEAT" in text or "Structural Caveat" in text, (
        "Report MD does not mention STRUCTURAL_CAVEAT for gsfb"
    )


def test_report_md_mentions_analytical_caveat():
    text = REPORT_MD.read_text()
    assert "ANALYTICAL_CAVEAT" in text or "Analytical Caveat" in text


def test_report_md_mentions_all_pass_models():
    text = REPORT_MD.read_text()
    for model in EXPECTED_PASS:
        assert model in text, f"Report MD does not mention PASS model {model!r}"
