"""Tests for threshold-isolation diagnostics artifacts and consistency.

Verifies:
  - Diagnostic CSV and MD exist
  - All 40 scenarios (8 models x 5 scenarios) are present
  - All 8 target models are correctly diagnosed
  - No NaN in any scenario
  - ASYMPTOTIC_FIRST_ORDER_CONFIRMED implies fine-level order in [0.85, 1.15]
  - Final status CSV has correct classifications:
    * 5 models retained FAIL_THRESHOLD_CROSSING (australia, hbv96, mopex2, mopex3, vic)
    * 3 models reclassified as PASS_WITH_CAVEAT (collie3, susannah1, us1)
  - Final status CSV still has 36 models
"""

import csv
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DGN_DIR = PROJECT_ROOT / "validation_results" / "euler_threshold_isolation"
FINAL_DIR = PROJECT_ROOT / "validation_results" / "euler_convergence_final"
PF_DIR = PROJECT_ROOT / "validation_results" / "euler_precision_floor"

DGN_CSV = DGN_DIR / "threshold_isolation_diagnostics.csv"
DGN_MD = DGN_DIR / "threshold_isolation_diagnostics.md"
PF_CSV = PF_DIR / "precision_floor_diagnostics.csv"
PF_MD = PF_DIR / "precision_floor_diagnostics.md"

FINAL_STATUS_CSV = FINAL_DIR / "euler_convergence_final_status.csv"

TARGET_MODELS = {
    "mopex2", "mopex3", "hbv96",
    "australia", "collie3", "susannah1", "us1", "vic",
}
EXPECTED_SCENARIOS = {
    "A_original", "B_threshold_separated", "C_high_storage",
    "D_stress", "E_fine_asymptotic",
}

# Models that should REMAIN FAIL_THRESHOLD_CROSSING
FAIL_MODELS_IN_FINAL = {"australia", "hbv96", "mopex2", "mopex3", "vic"}

# Models reclassified as PASS_WITH_CAVEAT
RECLASSIFIED_MODELS = {"collie3", "susannah1", "us1"}


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Diagnostic artifact existence
# ---------------------------------------------------------------------------

def test_diagnostic_csv_exists():
    assert DGN_CSV.exists(), f"Missing: {DGN_CSV}"


def test_diagnostic_md_exists():
    assert DGN_MD.exists(), f"Missing: {DGN_MD}"


def test_precision_floor_csv_exists():
    assert PF_CSV.exists(), f"Missing: {PF_CSV}"


def test_precision_floor_md_exists():
    assert PF_MD.exists(), f"Missing: {PF_MD}"


# ---------------------------------------------------------------------------
# Diagnostic CSV content
# ---------------------------------------------------------------------------

def test_diagnostic_csv_has_all_scenarios():
    rows = _read_csv(DGN_CSV)
    assert len(rows) == 40, f"Expected 40 rows (8 models x 5 scenarios), got {len(rows)}"

    models_seen = {row["model"] for row in rows}
    assert models_seen == TARGET_MODELS, (
        f"Expected models {TARGET_MODELS}, got {models_seen}"
    )

    for model in TARGET_MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        scenarios = {r["scenario"].split("_", 1)[1] for r in model_rows
                     if "_" in r["scenario"]}
        expected = {f"{model}_{s}" for s in EXPECTED_SCENARIOS}
        model_scenarios = {r["scenario"] for r in model_rows}
        assert model_scenarios == expected, (
            f"Model {model}: expected scenarios {expected}, got {model_scenarios}"
        )


def test_diagnostic_csv_recommended_statuses():
    rows = _read_csv(DGN_CSV)
    valid_recs = {"FAIL_THRESHOLD_CROSSING", "PASS", "PASS_WITH_CAVEAT",
                  "FAIL_UNEXPECTED"}
    for row in rows:
        rec = row.get("recommended_status", "")
        assert rec in valid_recs, (
            f"Model {row['model']} scenario {row['scenario']}: "
            f"unexpected recommended_status '{rec}'"
        )


def test_diagnostic_csv_no_nan_inf():
    rows = _read_csv(DGN_CSV)
    for row in rows:
        bad_str = row.get("any_nan_inf", "False")
        assert bad_str.strip().lower() in ("false", "0", ""), (
            f"Model {row['model']} scenario {row['scenario']}: unexpected NaN/Inf"
        )


def test_diagnostic_csv_diagnostic_subtype_valid():
    valid_subtypes = {
        "ASYMPTOTIC_FIRST_ORDER_CONFIRMED",
        "PERSISTENT_THRESHOLD_OR_STRUCTURAL_FAILURE",
        "UNEXPECTED_NAN_INF",
        "NON_MONOTONE_THRESHOLD",
    }
    rows = _read_csv(DGN_CSV)
    for row in rows:
        subtype = row.get("diagnostic_subtype", "")
        assert subtype in valid_subtypes, (
            f"Model {row['model']} scenario {row['scenario']}: "
            f"unexpected diagnostic_subtype '{subtype}'"
        )


def test_asymptotic_first_order_has_valid_final_order():
    """If diagnostic_subtype=ASYMPTOTIC_FIRST_ORDER_CONFIRMED,
    final_local_order must be in [0.85, 1.15]."""
    rows = _read_csv(DGN_CSV)
    for row in rows:
        subtype = row.get("diagnostic_subtype", "")
        if subtype != "ASYMPTOTIC_FIRST_ORDER_CONFIRMED":
            continue
        flo_str = row.get("final_local_order", "")
        if not flo_str or flo_str.strip().upper() == "N/A":
            continue
        try:
            flo = float(flo_str)
            assert 0.85 <= flo <= 1.15, (
                f"Model {row['model']} scenario {row['scenario']}: "
                f"ASYMPTOTIC_FIRST_ORDER_CONFIRMED but "
                f"final_local_order={flo:.4f} not in [0.85,1.15]"
            )
        except ValueError:
            pass


def test_australia_confirmed():
    """australia: E_fine_asymptotic should be ASYMPTOTIC_FIRST_ORDER_CONFIRMED."""
    rows = _read_csv(DGN_CSV)
    for row in rows:
        if row["model"] == "australia" and "E_fine_asymptotic" in row["scenario"]:
            assert row["diagnostic_subtype"] == "ASYMPTOTIC_FIRST_ORDER_CONFIRMED", (
                "australia E_fine_asymptotic: expected ASYMPTOTIC_FIRST_ORDER_CONFIRMED"
            )
            assert row["in_pass_band_by_final_local_order"] == "True", (
                "australia E_fine_asymptotic: final_local_order should be in band"
            )


def test_vic_fine_confirmed():
    """vic: E_fine_asymptotic should have final_local_order in band."""
    rows = _read_csv(DGN_CSV)
    for row in rows:
        if row["model"] == "vic" and "E_fine_asymptotic" in row["scenario"]:
            flo_str = row.get("final_local_order", "")
            try:
                flo = float(flo_str)
                assert 0.85 <= flo <= 1.15, (
                    f"vic E_fine_asymptotic: final_local_order={flo:.4f}"
                )
            except ValueError:
                pytest.fail(f"vic E_fine_asymptotic: bad final_local_order={flo_str}")


def test_all_models_have_correct_number_of_scenarios():
    rows = _read_csv(DGN_CSV)
    for model in TARGET_MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        assert len(model_rows) == 5, (
            f"Model {model}: expected 5 scenarios, got {len(model_rows)}"
        )


def test_substep_levels_field_present():
    rows = _read_csv(DGN_CSV)
    for row in rows:
        levels_str = row.get("substep_levels", "")
        assert levels_str, (
            f"Model {row['model']} scenario {row['scenario']}: missing substep_levels"
        )


def test_detected_threshold_crossing_flags_field_present():
    rows = _read_csv(DGN_CSV)
    for row in rows:
        flags = row.get("detected_threshold_crossing_flags", "")
        if row["model"] in {"australia", "collie3", "susannah1", "us1", "vic"}:
            assert flags, (
                f"Model {row['model']}: missing threshold crossing flags"
            )


def test_final_disposition_field_present():
    rows = _read_csv(DGN_CSV)
    for row in rows:
        disp = row.get("final_disposition", "")
        assert disp, f"Model {row['model']}: missing final_disposition"


# ---------------------------------------------------------------------------
# Consistency with final status CSV
# ---------------------------------------------------------------------------

def test_final_status_remains_fail_threshold_crossing():
    """Verify that the 5 retained FAIL models are correctly classified."""
    rows = _read_csv(FINAL_STATUS_CSV)
    status_map = {row["model"]: row["final_status"] for row in rows}

    for model in FAIL_MODELS_IN_FINAL:
        assert model in status_map, f"Model '{model}' missing from final status CSV"
        assert status_map[model] == "FAIL_THRESHOLD_CROSSING", (
            f"Model '{model}': expected FAIL_THRESHOLD_CROSSING, "
            f"got {status_map[model]}"
        )


def test_reclassified_models_pass_with_caveat():
    """Verify collie3, susannah1, us1 are now PASS_WITH_CAVEAT."""
    rows = _read_csv(FINAL_STATUS_CSV)
    status_map = {row["model"]: row["final_status"] for row in rows}

    for model in RECLASSIFIED_MODELS:
        assert model in status_map, f"Model '{model}' missing from final status CSV"
        assert status_map[model] == "PASS_WITH_CAVEAT", (
            f"Model '{model}': expected PASS_WITH_CAVEAT, "
            f"got {status_map[model]}"
        )


def test_collie3_has_rate_fix_note():
    """collie3 should have the rate-parameter misclassification note."""
    rows = _read_csv(FINAL_STATUS_CSV)
    for row in rows:
        if row["model"] == "collie3":
            assert "dimensionless exponent" in row.get("notes", ""), (
                "collie3 notes should mention rate-parameter fix"
            )


def test_susannah1_us1_precision_floor_note():
    """susannah1 and us1 should have precision floor notes."""
    rows = _read_csv(FINAL_STATUS_CSV)
    for row in rows:
        if row["model"] in ("susannah1", "us1"):
            assert "precision floor" in row.get("notes", ""), (
                f"{row['model']}: notes should mention precision floor"
            )


def test_final_status_count_consistent():
    """Total count should still be 36."""
    rows = _read_csv(FINAL_STATUS_CSV)
    assert len(rows) == 36, f"Expected 36 models, got {len(rows)}"


def test_no_unexpected_reclassifications():
    """Verify that PASS models remain PASS, gr4j remains ANALYTICAL_CAVEAT."""
    rows = _read_csv(FINAL_STATUS_CSV)
    status_map = {row["model"]: row["final_status"] for row in rows}

    # gr4j should remain ANALYTICAL_CAVEAT
    assert status_map.get("gr4j") == "ANALYTICAL_CAVEAT"

    # PASS models should remain PASS
    pass_models = {
        "flexb", "flexi", "gsfb", "hillslope", "hymod", "mopex1",
        "mopex4", "mopex5", "newzealand1", "newzealand2", "penman",
        "plateau", "simhyd", "susannah2", "tank", "tcm", "wetland", "xinanjiang",
    }
    for model in pass_models:
        assert status_map.get(model) == "PASS", (
            f"Model '{model}': expected PASS, got {status_map.get(model)}"
        )


# ---------------------------------------------------------------------------
# Diagnostic MD content
# ---------------------------------------------------------------------------

def test_diagnostic_md_has_new_models():
    text = DGN_MD.read_text()
    for model in ("australia", "collie3", "susannah1", "us1", "vic"):
        assert model in text, f"Model '{model}' not mentioned in diagnostic MD"


def test_diagnostic_md_has_reclassification():
    text = DGN_MD.read_text()
    assert "reclassification" in text.lower() or "Reclassification" in text, (
        "Diagnostic MD should mention reclassification"
    )


def test_diagnostic_md_has_summary_table():
    text = DGN_MD.read_text()
    assert "Summary Table" in text


def test_diagnostic_md_has_rate_parameter_notes():
    text = DGN_MD.read_text()
    assert "rate" in text.lower() or "Rate" in text


def test_diagnostic_md_has_australia_thresholds():
    text = DGN_MD.read_text()
    assert "saturation_1" in text or "excess_1" in text or "interflow_3" in text


# ---------------------------------------------------------------------------
# Precision floor diagnostics
# ---------------------------------------------------------------------------

def test_precision_floor_csv_has_susannah1_us1():
    rows = _read_csv(PF_CSV)
    models_in_pf = {row["model"] for row in rows}
    assert "susannah1" in models_in_pf, "susannah1 missing from precision floor CSV"
    assert "us1" in models_in_pf, "us1 missing from precision floor CSV"


def test_precision_floor_csv_no_nan():
    rows = _read_csv(PF_CSV)
    for row in rows:
        for key in ("state_error", "flux_error"):
            val = row.get(key, "")
            if val and val != "":
                assert "nan" not in val.lower(), (
                    f"Model {row['model']} n={row['n_substeps']}: NaN in {key}"
                )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_errors_finite():
    rows = _read_csv(DGN_CSV)
    for row in rows:
        assert row.get("any_nan_inf", "True").strip().lower() != "true", (
            f"Model {row['model']} {row['scenario']}: unexpected NaN/Inf"
        )


def test_collie3_original_shows_pass():
    """collie3 passes in diagnostics with correct rate params (levels 2..16)."""
    rows = _read_csv(DGN_CSV)
    for row in rows:
        if row["model"] == "collie3" and "A_original" in row["scenario"]:
            mo_str = row.get("median_order", "")
            try:
                mo = float(mo_str)
                assert 0.85 <= mo <= 1.15, (
                    f"collie3 A_original median_order={mo} not in band"
                )
            except ValueError:
                pytest.fail(f"collie3 A_original: bad median_order={mo_str}")
