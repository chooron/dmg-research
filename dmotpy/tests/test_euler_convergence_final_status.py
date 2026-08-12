"""Tests for the final Euler convergence status synthesis.

Verifies that finalize_euler_convergence_validation.py produces consistent,
complete, and correctly classified final-status files.

Post-diagnostics reclassification (2026-06-26):
  - collie3: FAIL_THRESHOLD_CROSSING → PASS_WITH_CAVEAT (b exponent rate-param fix)
  - susannah1: FAIL_THRESHOLD_CROSSING → PASS_WITH_CAVEAT (precision floor)
  - us1: FAIL_THRESHOLD_CROSSING → PASS_WITH_CAVEAT (precision floor)
  - australia, hbv96, mopex2, mopex3, vic: retain FAIL_THRESHOLD_CROSSING
"""
import csv
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "validation_results" / "euler_convergence_final"

FINAL_STATUS_CSV = FINAL_DIR / "euler_convergence_final_status.csv"
FINAL_SUMMARY_CSV = FINAL_DIR / "euler_convergence_final_summary.csv"
FINAL_REPORT_MD = FINAL_DIR / "euler_convergence_final_report.md"
UNRESOLVED_CSV = FINAL_DIR / "euler_convergence_unresolved_models.csv"

VALID_STATUSES = {
    "PASS",
    "PASS_WITH_CAVEAT",
    "ANALYTICAL_CAVEAT",
    "FAIL_THRESHOLD_CROSSING",
    "FAIL_PRECISION_FLOOR",
    "NOT_RUN",
}

EXPECTED_CAVEAT_CLASSIFICATIONS = {
    "gr4j": "ANALYTICAL_CAVEAT",
    "gsfb": "PASS",
    "mopex4": "PASS",
    "mopex5": "PASS",
    "tank": "PASS",
    "tcm": "PASS",
}

# Models reclassified from FAIL_THRESHOLD_CROSSING to PASS_WITH_CAVEAT
RECLASSIFIED_MODELS = {
    "collie3": ("PASS_WITH_CAVEAT", "rate-parameter misclassification fixed"),
    "susannah1": ("PASS_WITH_CAVEAT", "precision floor dominated"),
    "us1": ("PASS_WITH_CAVEAT", "precision floor dominated"),
}

# Models that should remain FAIL_THRESHOLD_CROSSING
RETAINED_FAIL_MODELS = {"australia", "hbv96", "mopex2", "mopex3", "vic"}


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------

def test_final_status_csv_exists():
    assert FINAL_STATUS_CSV.exists(), f"Missing: {FINAL_STATUS_CSV}"


def test_final_summary_csv_exists():
    assert FINAL_SUMMARY_CSV.exists(), f"Missing: {FINAL_SUMMARY_CSV}"


def test_final_report_md_exists():
    assert FINAL_REPORT_MD.exists(), f"Missing: {FINAL_REPORT_MD}"


def test_unresolved_csv_exists():
    assert UNRESOLVED_CSV.exists(), f"Missing: {UNRESOLVED_CSV}"


# ---------------------------------------------------------------------------
# Final status CSV content
# ---------------------------------------------------------------------------

def test_final_status_csv_has_models():
    rows = _read_csv(FINAL_STATUS_CSV)
    assert len(rows) >= 10, f"Expected ≥10 models in final status, got {len(rows)}"


def test_final_status_csv_valid_statuses():
    rows = _read_csv(FINAL_STATUS_CSV)
    for row in rows:
        status = row["final_status"]
        assert status in VALID_STATUSES, (
            f"Model {row['model']}: unexpected status '{status}'"
        )


def test_final_status_csv_no_missing_model_field():
    rows = _read_csv(FINAL_STATUS_CSV)
    for row in rows:
        assert row.get("model", "").strip(), f"Empty model field in row: {row}"


def test_final_status_caveat_models_correct():
    rows = _read_csv(FINAL_STATUS_CSV)
    status_map = {row["model"]: row["final_status"] for row in rows}
    for model, expected in EXPECTED_CAVEAT_CLASSIFICATIONS.items():
        assert model in status_map, f"Model '{model}' not found in final status CSV"
        assert status_map[model] == expected, (
            f"Model '{model}': expected '{expected}', got '{status_map[model]}'"
        )


def test_final_status_no_duplicate_models():
    rows = _read_csv(FINAL_STATUS_CSV)
    models = [row["model"] for row in rows]
    assert len(models) == len(set(models)), (
        f"Duplicate models in final status: {[m for m in models if models.count(m) > 1]}"
    )


def test_final_status_pass_models_have_valid_orders():
    rows = _read_csv(FINAL_STATUS_CSV)
    for row in rows:
        if row["final_status"] != "PASS":
            continue  # only strict PASS models are expected in [0.85, 1.15]
        order_str = row.get("median_order", "")
        if order_str and order_str != "N/A":
            try:
                order = float(order_str)
                assert 0.0 < order < 10.0, (
                    f"Model {row['model']}: implausible order {order}"
                )
            except ValueError:
                pass  # N/A or empty is fine


# ---------------------------------------------------------------------------
# Summary CSV content
# ---------------------------------------------------------------------------

def test_final_summary_csv_has_count_pass():
    rows = _read_csv(FINAL_SUMMARY_CSV)
    groups = {row["final_status"]: int(row["model_count"]) for row in rows}
    total = sum(groups.values())
    assert total >= 10, f"Expected ≥10 total models in summary, got {total}"
    # At least some models should pass
    pass_count = groups.get("PASS", 0) + groups.get("PASS_WITH_CAVEAT", 0)
    assert pass_count >= 1, "Expected at least 1 PASS or PASS_WITH_CAVEAT model"


def test_final_summary_csv_has_caveat_entries():
    rows = _read_csv(FINAL_SUMMARY_CSV)
    groups = {row["final_status"]: int(row["model_count"]) for row in rows}
    analytical = groups.get("ANALYTICAL_CAVEAT", 0)
    assert analytical >= 1, "Expected at least 1 ANALYTICAL_CAVEAT (gr4j)"
    # Note: gsfb was renamed from gsfb_smooth (smooth variant) on 2026-06-25;
    # the original gsfb_original.py is archived. gsfb now has PASS status.


# ---------------------------------------------------------------------------
# Report markdown content
# ---------------------------------------------------------------------------

def test_final_report_md_has_sections():
    text = FINAL_REPORT_MD.read_text()
    assert "PASS" in text
    assert "ANALYTICAL_CAVEAT" in text or "analytical" in text.lower()
    # After gsfb_smooth renamed to gsfb, STRUCTURAL_CAVEAT is no longer in status report
    assert "gr4j" in text
    assert "gsfb" in text


def test_final_report_md_not_empty():
    text = FINAL_REPORT_MD.read_text()
    assert len(text) > 200, "Final report MD seems too short"


# ---------------------------------------------------------------------------
# Unresolved CSV
# ---------------------------------------------------------------------------

def test_unresolved_csv_valid_statuses():
    rows = _read_csv(UNRESOLVED_CSV)
    for row in rows:
        status = row["final_status"]
        assert status not in {"PASS", "PASS_WITH_CAVEAT"}, (
            f"Model {row['model']} with status '{status}' should not appear in unresolved CSV"
        )


def test_unresolved_csv_only_6_models():
    """Should be 6: australia, gr4j, hbv96, mopex2, mopex3, vic."""
    rows = _read_csv(UNRESOLVED_CSV)
    assert len(rows) == 6, (
        f"Expected 6 unresolved models (5 FAIL + 1 ANALYTICAL), got {len(rows)}"
    )


def test_unresolved_csv_has_no_reclassified_models():
    """collie3, susannah1, us1 should NOT be in unresolved CSV."""
    rows = _read_csv(UNRESOLVED_CSV)
    models = {row["model"] for row in rows}
    for m in ("collie3", "susannah1", "us1"):
        assert m not in models, f"{m} should not be in unresolved CSV"


# ---------------------------------------------------------------------------
# Reclassification tests
# ---------------------------------------------------------------------------

def test_reclassified_models_are_pass_with_caveat():
    rows = _read_csv(FINAL_STATUS_CSV)
    status_map = {row["model"]: row["final_status"] for row in rows}
    for model, (expected, _reason) in RECLASSIFIED_MODELS.items():
        assert status_map.get(model) == expected, (
            f"Model '{model}': expected {expected}, got {status_map.get(model)}"
        )


def test_reclassified_models_have_notes():
    rows = _read_csv(FINAL_STATUS_CSV)
    for row in rows:
        if row["model"] in RECLASSIFIED_MODELS:
            assert row.get("notes", ""), (
                f"Model '{row['model']}': should have reclassification notes"
            )


def test_retained_fail_models():
    rows = _read_csv(FINAL_STATUS_CSV)
    status_map = {row["model"]: row["final_status"] for row in rows}
    for model in RETAINED_FAIL_MODELS:
        assert status_map.get(model) == "FAIL_THRESHOLD_CROSSING", (
            f"Model '{model}': expected FAIL_THRESHOLD_CROSSING, "
            f"got {status_map.get(model)}"
        )


def test_final_summary_has_12_pass_with_caveat():
    rows = _read_csv(FINAL_SUMMARY_CSV)
    for row in rows:
        if row["final_status"] == "PASS_WITH_CAVEAT":
            count = int(row["model_count"])
            assert count == 12, (
                f"Expected 12 PASS_WITH_CAVEAT models, got {count}"
            )


def test_final_summary_has_5_fail_threshold_crossing():
    rows = _read_csv(FINAL_SUMMARY_CSV)
    for row in rows:
        if row["final_status"] == "FAIL_THRESHOLD_CROSSING":
            count = int(row["model_count"])
            assert count == 5, (
                f"Expected 5 FAIL_THRESHOLD_CROSSING models, got {count}"
            )
