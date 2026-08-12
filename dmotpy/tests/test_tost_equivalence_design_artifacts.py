"""
Test: TOST equivalence design artifacts
=======================================
Verifies that all required design-lock files exist and contain
the pre-specified equivalence margins, fallback rules, and
result categories BEFORE any TOST results are computed.
"""
import pytest
import csv
from pathlib import Path

import yaml
import pandas as pd

TOST_DIR = Path(__file__).parent.parent / "validation_results" / "tost_equivalence"


def _read_design_yaml():
    path = TOST_DIR / "tost_design.yaml"
    assert path.exists(), f"Missing {path}"
    with open(path) as f:
        return yaml.safe_load(f.read())


def _read_design_md():
    path = TOST_DIR / "tost_design_lock.md"
    assert path.exists(), f"Missing {path}"
    return path.read_text()


def _assert_csv_nonempty(path: Path):
    assert path.exists(), f"Missing {path}"
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) > 1, f"{path} is empty or has only header"


# ---------------------------------------------------------------------------
# Design lock file existence
# ---------------------------------------------------------------------------

def test_design_lock_md_exists():
    """tost_design_lock.md must exist."""
    assert (TOST_DIR / "tost_design_lock.md").exists()


def test_design_yaml_exists():
    """tost_design.yaml must exist."""
    assert (TOST_DIR / "tost_design.yaml").exists()


# ---------------------------------------------------------------------------
# Design YAML content checks
# ---------------------------------------------------------------------------

def test_design_yaml_contains_fixed_margins():
    design = _read_design_yaml()
    eq = design.get("equivalence", {})
    primary = eq.get("primary_metrics", {})
    assert primary["kge_difference"]["margin"] == 0.02
    assert primary["nse_difference"]["margin"] == 0.02
    assert primary["volume_bias"]["margin"] == 0.01
    assert primary["mean_bias"]["margin"] == 0.01


def test_design_yaml_contains_alpha():
    design = _read_design_yaml()
    st = design.get("statistical_test", {})
    assert st["alpha"] == 0.05


def test_design_yaml_contains_eval_period():
    design = _read_design_yaml()
    d = design.get("data", {})
    tr = d.get("time_range", {})
    eval_period = tr.get("eval_period", {})
    assert eval_period["start"] == "1999-01-01"
    assert eval_period["end"] == "2009-12-31"


def test_design_yaml_contains_train_period():
    design = _read_design_yaml()
    d = design.get("data", {})
    tr = d.get("time_range", {})
    train_period = tr.get("train_period", {})
    assert train_period["start"] == "1989-01-01"
    assert train_period["end"] == "1998-12-31"


def test_design_yaml_contains_uh_disabled():
    design = _read_design_yaml()
    analysis = design.get("analysis", {})
    assert analysis.get("uh_disabled") is True


def test_design_yaml_contains_parameter_source():
    design = _read_design_yaml()
    analysis = design.get("analysis", {})
    assert "marrmot_obj1" in analysis.get("parameter_source", "")


def test_design_yaml_contains_near_zero_rules():
    design = _read_design_yaml()
    nz = design.get("near_zero_rules", {})
    assert nz.get("near_zero_mean_threshold") == 1.0e-6
    assert nz.get("near_zero_std_threshold") == 1.0e-8
    assert nz.get("min_nonzero_fraction") == 0.05
    assert nz.get("min_nonzero_days") == 30
    assert nz.get("eps") == 1.0e-8


def test_design_yaml_contains_result_categories():
    design = _read_design_yaml()
    rc = design.get("result_categories", {})
    required = [
        "EQUIVALENT",
        "NOT_EQUIVALENT",
        "INCONCLUSIVE_LOW_POWER",
        "INCONCLUSIVE_METRIC_INVALID",
        "RUN_FAILED",
    ]
    for cat in required:
        assert cat in rc, f"Missing result category: {cat}"


# ---------------------------------------------------------------------------
# Planning matrix existence and content
# ---------------------------------------------------------------------------

def test_planning_matrix_csv_exists_and_nonempty():
    _assert_csv_nonempty(TOST_DIR / "tost_planning_matrix.csv")


def test_planning_matrix_has_required_columns():
    path = TOST_DIR / "tost_planning_matrix.csv"
    df = pd.read_csv(path)
    required_cols = [
        "model", "reliability_class", "model_group",
        "pymarrmot_model", "param_file_obj1",
        "equiv_margin_kge", "equiv_margin_nse",
        "equiv_margin_vol_bias", "equiv_margin_mean_bias",
        "n_testable_basins", "status", "reporting_level",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_planning_matrix_contains_testable_models():
    path = TOST_DIR / "tost_planning_matrix.csv"
    df = pd.read_csv(path)
    testable = df[df["status"] == "planned"]
    assert len(testable) > 0, "No testable models in planning matrix"
    # All testable models should have >0 testable basins
    assert (testable["n_testable_basins"] > 0).all()


def test_planning_matrix_margins_are_locked():
    path = TOST_DIR / "tost_planning_matrix.csv"
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        assert row["equiv_margin_kge"] == 0.02
        assert row["equiv_margin_nse"] == 0.02
        assert row["equiv_margin_vol_bias"] == 0.01
        assert row["equiv_margin_mean_bias"] == 0.01


# ---------------------------------------------------------------------------
# Basin coverage CSV
# ---------------------------------------------------------------------------

def test_basin_coverage_csv_exists_and_nonempty():
    _assert_csv_nonempty(TOST_DIR / "basin_model_param_coverage.csv")


def test_basin_coverage_has_overlap():
    path = TOST_DIR / "basin_model_param_coverage.csv"
    df = pd.read_csv(path)
    planned = df[df["status"] == "planned"]
    # All planned models should have basin overlap with CAMELS
    assert (planned["n_overlap_basins"] > 0).all()
    # Overlap count should be 559 for all (CAMELS has 671, params have 559)
    assert (planned["n_overlap_basins"] == 559).all()


# ---------------------------------------------------------------------------
# Sanity check artifacts (if sanity was run)
# ---------------------------------------------------------------------------

def test_sanity_check_dir_exists():
    sanity_dir = TOST_DIR / "sanity_check"
    assert sanity_dir.is_dir(), "Missing sanity_check directory"


def test_sanity_check_report_exists():
    report = TOST_DIR / "sanity_check" / "sanity_check_report.md"
    assert report.exists(), "Missing sanity check report"


def test_sanity_check_has_required_files():
    sanity_dir = TOST_DIR / "sanity_check"
    required = [
        "sanity_model_mapping.csv",
        "sanity_parameter_transfer.csv",
        "sanity_timeseries_head_tail.csv",
        "sanity_metric_summary.csv",
        "sanity_check_report.md",
    ]
    for fname in required:
        assert (sanity_dir / fname).exists(), f"Missing: {fname}"


# ---------------------------------------------------------------------------
# Negative check: full TOST result file must NOT exist in this planning stage
# ---------------------------------------------------------------------------

def test_no_full_tost_results_yet():
    """Planning stage: no full TOST result files should exist."""
    results_dir = TOST_DIR / "results"
    global_summary = TOST_DIR / "tost_global_summary.csv"
    assert not global_summary.exists(), \
        "tost_global_summary.csv should not exist in planning stage"
    # results/ dir may exist but should be empty or contain only placeholder
    if results_dir.exists():
        csv_files = list(results_dir.glob("**/*.csv"))
        assert len(csv_files) == 0, \
            f"Found unexpected result CSV files in {results_dir}: {csv_files}"


# ---------------------------------------------------------------------------
# Design lock content cross-verification
# ---------------------------------------------------------------------------

def test_design_lock_md_contains_equivalence_margins():
    text = _read_design_md()
    assert "±0.02" in text, "Missing ΔKGE/ΔNSE margin in design lock"
    assert "±1%" in text, "Missing volume/mean bias margin in design lock"


def test_design_lock_md_contains_alpha():
    text = _read_design_md()
    assert "alpha = 0.05" in text or "α = 0.05" in text, "Missing alpha in design lock"


def test_design_lock_md_contains_result_categories():
    text = _read_design_md()
    for cat in ["EQUIVALENT", "NOT_EQUIVALENT", "INCONCLUSIVE_LOW_POWER"]:
        assert cat in text, f"Missing result category {cat} in design lock"
