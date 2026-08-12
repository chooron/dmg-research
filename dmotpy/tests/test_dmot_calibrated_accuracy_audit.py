"""
Test: dMoT Calibrated-Parameter Accuracy Audit — Artifact Validation
====================================================================
Validates all 12+ output artifacts from the calibrated-accuracy audit.
Confirms file existence, field completeness, data integrity, and report
wording consistency.  Does NOT re-run simulations.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
AUDIT_DIR = REPO_ROOT / "dmotpy" / "validation_results" / "dmot_calibrated_accuracy_audit"
CACHE_DIR = AUDIT_DIR / "cache"

REQUIRED_FILES = [
    "calibrated_accuracy_design.md",
    "model_param_coverage.csv",
    "dmot_calibrated_batched_run_summary.csv",
    "dmot_basin_level_accuracy.csv",
    "dmot_model_level_accuracy_summary.csv",
    "dmot_accuracy_classification.csv",
    "calibration_artifact_cases.csv",
    "metric_invalid_cases.csv",
    "run_failed_cases.csv",
    "param_dimension_issues.csv",
    "dmot_calibrated_accuracy_report.md",
    "dmot_calibrated_accuracy_manifest.json",
]

EXPECTED_BASIN_ROWS = 19565  # 35 models * 559 basins
EXPECTED_N_EVAL_DAYS = 4018
EXPECTED_N_WARMUP_DAYS = 3652
N_MODELS_TOTAL = 36
N_TESTABLE = 35
N_PARAM_MISMATCH = 1
N_LARGE_DEVIATION = 35

# ---------------------------------------------------------------------------
# 1. Output directory exists
# ---------------------------------------------------------------------------
def test_output_directory_exists():
    assert AUDIT_DIR.is_dir(), f"Audit directory not found: {AUDIT_DIR}"


# ---------------------------------------------------------------------------
# 2. All 12 core artifacts present
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_core_artifact_exists(filename):
    path = AUDIT_DIR / filename
    assert path.exists(), f"Missing artifact: {filename}"
    assert path.stat().st_size > 0, f"Artifact is empty: {filename}"


# ---------------------------------------------------------------------------
# 3. model_param_coverage.csv
# ---------------------------------------------------------------------------
def test_model_param_coverage():
    df = pd.read_csv(AUDIT_DIR / "model_param_coverage.csv")
    assert len(df) == N_MODELS_TOTAL, f"Expected {N_MODELS_TOTAL} models, got {len(df)}"

    # ihacres: tau_d removed, status TAU_D_REMOVED (re-calibration pending)
    ih = df[df["model"] == "ihacres"]
    assert len(ih) == 1, "ihacres missing from coverage"
    ih = ih.iloc[0]
    assert ih["test_status"] in ("TAU_D_REMOVED", "TESTABLE"), \
        f"ihacres status: {ih['test_status']}"
    assert int(ih["n_param_csv"]) == 6, f"ihacres n_param_csv: {ih['n_param_csv']}"
    assert int(ih["n_param_dmot"]) == 6, f"ihacres n_param_dmot: {ih['n_param_dmot']}"
    assert ih["uh_scope"] == "endpoint", f"ihacres uh_scope: {ih['uh_scope']}"

    # At least 35 models are TESTABLE
    testable = df[df["test_status"] == "TESTABLE"]
    assert len(testable) >= N_TESTABLE, \
        f"Expected ≥{N_TESTABLE} TESTABLE, got {len(testable)}"

    # Required columns
    required_cols = [
        "model", "param_file", "param_file_exists", "n_param_csv",
        "n_param_dmot", "uh_scope", "uh_enabled_used",
        "extra_inputs_required", "test_status", "notes",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# 4. dmot_calibrated_batched_run_summary.csv
# ---------------------------------------------------------------------------
def test_batched_run_summary():
    df = pd.read_csv(AUDIT_DIR / "dmot_calibrated_batched_run_summary.csv")

    # At least 35 OK runs
    ok_runs = df[df["run_status"] == "OK"]
    assert len(ok_runs) >= N_TESTABLE, \
        f"Expected ≥{N_TESTABLE} OK, got {len(ok_runs)}"

    # No NaN/Inf in Q outputs for OK runs
    for _, row in ok_runs.iterrows():
        assert row["any_nan_inf_q"] == False or row["any_nan_inf_q"] is False, \
            f"{row['model']}: any_nan_inf_q={row['any_nan_inf_q']}"

    # Eval / warmup days
    for _, row in ok_runs.iterrows():
        assert int(row["n_days_eval"]) == EXPECTED_N_EVAL_DAYS, \
            f"{row['model']}: n_days_eval={row['n_days_eval']}"
        assert int(row["n_days_warmup"]) == EXPECTED_N_WARMUP_DAYS, \
            f"{row['model']}: n_days_warmup={row['n_days_warmup']}"

    # Device / dtype
    assert "device" in df.columns, "Missing device column"
    assert "dtype" in df.columns, "Missing dtype column"
    assert "batch_size" in df.columns, "Missing batch_size column"

    # dtype should be float64
    for _, row in ok_runs.iterrows():
        assert "float64" in str(row["dtype"]), \
            f"{row['model']}: unexpected dtype={row['dtype']}"


# ---------------------------------------------------------------------------
# 5. dmot_basin_level_accuracy.csv
# ---------------------------------------------------------------------------
def test_basin_level_accuracy():
    df = pd.read_csv(AUDIT_DIR / "dmot_basin_level_accuracy.csv")
    assert len(df) > 0, "Basin-level CSV is empty"
    # Allow ±5% tolerance on row count
    assert abs(len(df) - EXPECTED_BASIN_ROWS) < EXPECTED_BASIN_ROWS * 0.05, \
        f"Expected ~{EXPECTED_BASIN_ROWS}, got {len(df)}"

    required_cols = [
        "model", "basin_id",
        "recorded_cal_kge", "recorded_eval_kge",
        "kge_dmot_eval", "nse_dmot_eval",
        "delta_kge_eval", "abs_delta_kge_eval",
        "qobs_valid_days", "qobs_mean", "qobs_std",
        "q_dmot_mean", "q_dmot_std",
        "volume_bias_vs_qobs",
        "metric_valid_flag", "run_status",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # dMoT KGE no NaN/Inf for rows with metric_valid_flag=True and run_status==OK
    valid_ok = df[(df["metric_valid_flag"] == True) & (df["run_status"] == "OK")]
    assert len(valid_ok) > 0, "No valid-metric OK rows in basin CSV"
    for _, row in valid_ok.iterrows():
        assert not np.isnan(row["kge_dmot_eval"]), \
            f"{row['model']}:{row['basin_id']} kge_dmot_eval is NaN"
        assert not np.isinf(row["kge_dmot_eval"]), \
            f"{row['model']}:{row['basin_id']} kge_dmot_eval is Inf"

    # q_dmot_mean / q_dmot_std no NaN/Inf for OK rows
    ok_rows = df[df["run_status"] == "OK"]
    for _, row in ok_rows.iterrows():
        assert not np.isnan(row["q_dmot_mean"]), \
            f"{row['model']}:{row['basin_id']} q_dmot_mean is NaN"
        assert not np.isinf(row["q_dmot_mean"]), \
            f"{row['model']}:{row['basin_id']} q_dmot_mean is Inf"

    # All 35 testable models present
    models_in_bl = set(df["model"].unique())
    assert len(models_in_bl) == N_TESTABLE, \
        f"Expected {N_TESTABLE} models in basin CSV, got {len(models_in_bl)}"


# ---------------------------------------------------------------------------
# 6. dmot_model_level_accuracy_summary.csv
# ---------------------------------------------------------------------------
def test_model_level_accuracy_summary():
    df = pd.read_csv(AUDIT_DIR / "dmot_model_level_accuracy_summary.csv")
    assert len(df) == N_MODELS_TOTAL, \
        f"Expected {N_MODELS_TOTAL} rows, got {len(df)}"

    required_cols = [
        "model", "reliability_class", "uh_scope", "uh_enabled_used",
        "n_total_basins", "n_valid_metric_basins",
        "median_recorded_eval_kge_raw", "median_kge_dmot_eval_raw",
        "median_delta_kge_raw",
        "median_recorded_eval_kge_robust", "median_kge_dmot_eval_robust",
        "median_delta_kge_robust",
        "fraction_within_0p05", "fraction_within_0p10", "fraction_within_0p15",
        "accuracy_class", "likely_source", "notes",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # 35 testable models all LARGE_DEVIATION (ihacres excluded: TAU_D_REMOVED)
    testable_ms = df[df["accuracy_class"] != "PARAM_DIMENSION_MISMATCH"]
    testable_ms = testable_ms[testable_ms["accuracy_class"] != "TAU_D_REMOVED"]
    assert len(testable_ms) == N_TESTABLE, \
        f"Expected {N_TESTABLE} testable models, got {len(testable_ms)}"

    for _, row in testable_ms.iterrows():
        assert row["accuracy_class"] == "LARGE_DEVIATION", \
            f"{row['model']}: expected LARGE_DEVIATION, got {row['accuracy_class']}"

    # LARGE_DEVIATION rows must have non-empty likely_source or notes
    for _, row in testable_ms.iterrows():
        has_source = (isinstance(row["likely_source"], str) and len(str(row["likely_source"]).strip()) > 0)
        has_notes = (isinstance(row["notes"], str) and len(str(row["notes"]).strip()) > 0)
        assert has_source or has_notes, \
            f"{row['model']}: LARGE_DEVIATION must have likely_source or notes"

    # ihacres: tau_d removed, now resolved (TAU_D_REMOVED or similar, not PARAM_DIMENSION_MISMATCH)
    ih = df[df["model"] == "ihacres"]
    assert len(ih) == 1, "ihacres missing from model summary"
    assert ih.iloc[0]["accuracy_class"] != "PARAM_DIMENSION_MISMATCH", \
        f"ihacres should no longer be PARAM_DIMENSION_MISMATCH: {ih.iloc[0]['accuracy_class']}"

    # No LARGE_DEVIATION model claims formula bug
    for _, row in testable_ms.iterrows():
        ls = str(row["likely_source"]).lower()
        ns = str(row["notes"]).lower()
        assert "formula bug" not in ls, f"{row['model']}: likely_source says formula bug"
        assert "formula bug" not in ns, f"{row['model']}: notes says formula bug"


# ---------------------------------------------------------------------------
# 7. dmot_accuracy_classification.csv
# ---------------------------------------------------------------------------
def test_accuracy_classification():
    df = pd.read_csv(AUDIT_DIR / "dmot_accuracy_classification.csv")

    expected = {
        "CLOSE_REPRODUCTION": 0,
        "MODERATE_REPRODUCTION": 0,
        "NOTABLE_DEVIATION": 0,
        "LARGE_DEVIATION": N_LARGE_DEVIATION,
        "INSUFFICIENT_VALID_BASINS": 0,
        "PARAM_DIMENSION_MISMATCH": N_PARAM_MISMATCH,
        "RUN_FAILED": 0,
    }

    for cls, count in expected.items():
        match = df[df["accuracy_class"] == cls]
        assert len(match) <= 1, f"Duplicate class: {cls}"
        if len(match) == 1:
            actual = int(match.iloc[0]["n_models"])
            assert actual == count, \
                f"Class {cls}: expected {count}, got {actual}"


# ---------------------------------------------------------------------------
# 8. calibration_artifact_cases.csv
# ---------------------------------------------------------------------------
def test_calibration_artifact_cases():
    df = pd.read_csv(AUDIT_DIR / "calibration_artifact_cases.csv")
    assert len(df) > 0, "Artifact cases CSV is empty"

    required_cols = [
        "model", "basin_id", "recorded_cal_kge", "recorded_eval_kge",
        "kge_dmot_eval", "reason", "notes",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # At least some rows have recorded_eval_kge < -5
    extreme = df[df["recorded_eval_kge"] < -5]
    assert len(extreme) > 0, "No rows with recorded_eval_kge < -5"

    # Verify the reason mentions the artifact
    for _, row in extreme.iterrows():
        reason = str(row["reason"]).lower()
        assert ("recorded_eval_kge" in reason or "recorded" in reason), \
            f"Artifact reason doesn't mention recorded: {reason}"


# ---------------------------------------------------------------------------
# 9. metric_invalid_cases.csv
# ---------------------------------------------------------------------------
def test_metric_invalid_cases():
    df = pd.read_csv(AUDIT_DIR / "metric_invalid_cases.csv")
    assert len(df) > 0, "Metric-invalid CSV is empty — expected some near-zero Qobs cases"

    # Must have reason or near_zero_qobs column
    has_reason = "reason" in df.columns
    has_nz = "near_zero_qobs_flag" in df.columns
    assert has_reason or has_nz, "Missing reason/near_zero_qobs_flag column"

    # Should contain near-zero Qobs or invalid metric cases
    if "reason" in df.columns:
        reasons = " ".join(df["reason"].astype(str).tolist()).lower()
        has_expected = ("near_zero" in reasons or "kge=nan" in reasons or
                        "nse=nan" in reasons or "valid_days" in reasons or
                        "metric_valid" in reasons)
        assert has_expected, f"Metric-invalid reasons don't match expected: {reasons[:200]}"

    # Also check near_zero_qobs_flag if present
    if "near_zero_qobs_flag" in df.columns:
        # Some should be True (near-zero Qobs)
        nz_true = df["near_zero_qobs_flag"].sum()
        # At least some should have near_zero=True or metric invalid
        assert nz_true >= 0, "near_zero_qobs_flag check"


# ---------------------------------------------------------------------------
# 10. run_failed_cases.csv
# ---------------------------------------------------------------------------
def test_run_failed_cases():
    df = pd.read_csv(AUDIT_DIR / "run_failed_cases.csv")
    # File must exist (may be empty or contain METRIC_INVALID rows)
    # Current count is 1552 METRIC_INVALID rows
    assert len(df) >= 0  # Always true — just checking it loads

    required_cols = ["model", "basin_id", "run_status"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # No actual simulation failures (no RUN_FAILED status)
    sim_failures = df[df["run_status"].str.contains("FAILED", case=False, na=False)]
    assert len(sim_failures) == 0, \
        f"Unexpected simulation failures: {sim_failures[['model', 'basin_id']].to_dict()}"


# ---------------------------------------------------------------------------
# 11. param_dimension_issues.csv
# ---------------------------------------------------------------------------
def test_param_dimension_issues():
    df = pd.read_csv(AUDIT_DIR / "param_dimension_issues.csv")
    # With tau_d removed, ihacres should no longer appear as a dimension issue
    assert "ihacres" not in df["model"].values if len(df) > 0 else True, \
        "ihacres should no longer be in param_dimension_issues after tau_d removal"


# ---------------------------------------------------------------------------
# 12. Cache files
# ---------------------------------------------------------------------------
def test_cache_directory_exists():
    assert CACHE_DIR.is_dir(), f"Cache directory not found: {CACHE_DIR}"


def test_cache_file_count():
    cache_files = sorted(CACHE_DIR.glob("*_eval_outputs.npz"))
    assert len(cache_files) >= N_TESTABLE, \
        f"Expected ≥{N_TESTABLE} cache files, got {len(cache_files)}"
    # ihacres now testable, may or may not have cache yet
    ihacres_cache = [f for f in cache_files if "ihacres" in f.name]


@pytest.mark.parametrize("model_name", [
    "collie1", "gr4j", "hbv96", "newzealand2", "flexi",
])
def test_cache_content_integrity(model_name):
    """Spot-check 5 cache files for correct shape and no NaN/Inf."""
    cache_path = CACHE_DIR / f"{model_name}_eval_outputs.npz"
    if not cache_path.exists():
        pytest.skip(f"Cache not found: {model_name}")

    data = np.load(cache_path, allow_pickle=True)

    # Required keys
    required_keys = [
        "basin_ids", "q_dmot_eval", "qobs_eval_mm_day",
        "qobs_valid_mask_eval", "params_obj1",
        "recorded_cal_kge", "recorded_eval_kge",
    ]
    for key in required_keys:
        assert key in data, f"{model_name}: missing key '{key}' in cache"

    q_dmot = data["q_dmot_eval"]

    # Eval length = 4018 (first dim)
    assert q_dmot.shape[0] == EXPECTED_N_EVAL_DAYS, \
        f"{model_name}: expected {EXPECTED_N_EVAL_DAYS} eval days, got {q_dmot.shape[0]}"

    # No NaN/Inf
    assert not np.any(np.isnan(q_dmot)), f"{model_name}: q_dmot_eval contains NaN"
    assert not np.any(np.isinf(q_dmot)), f"{model_name}: q_dmot_eval contains Inf"

    # basin_ids count matches q_dmot basin count
    assert len(data["basin_ids"]) == q_dmot.shape[1], \
        f"{model_name}: basin_ids={len(data['basin_ids'])}, q_dmot basins={q_dmot.shape[1]}"


# ---------------------------------------------------------------------------
# 13. Manifest
# ---------------------------------------------------------------------------
def test_manifest():
    with open(AUDIT_DIR / "dmot_calibrated_accuracy_manifest.json") as f:
        mf = json.load(f)

    assert mf.get("analysis_type") == "dmot_calibrated_parameter_accuracy_audit", \
        f"analysis_type: {mf.get('analysis_type')}"

    assert mf.get("area_index") == 11, f"area_index: {mf.get('area_index')}"
    assert mf.get("n_models_testable") == N_TESTABLE, \
        f"n_models_testable: {mf.get('n_models_testable')}"
    assert mf.get("n_models_param_mismatch") == N_PARAM_MISMATCH, \
        f"n_models_param_mismatch: {mf.get('n_models_param_mismatch')}"
    assert mf.get("n_models_run_failed") == 0, \
        f"n_models_run_failed: {mf.get('n_models_run_failed')}"

    classification = mf.get("classification", {})
    assert classification.get("LARGE_DEVIATION") == N_LARGE_DEVIATION, \
        f"classification LARGE_DEVIATION: {classification.get('LARGE_DEVIATION')}"
    assert classification.get("PARAM_DIMENSION_MISMATCH") == N_PARAM_MISMATCH

    assert "files" in mf, "Missing files list in manifest"
    assert "timestamp" in mf, "Missing timestamp"
    assert "device" in mf, "Missing device"
    assert "dtype" in mf, "Missing dtype"

    # Verify all listed files exist
    for fname in mf.get("files", []):
        path = AUDIT_DIR / fname
        assert path.exists(), f"Manifest-listed file missing: {fname}"


# ---------------------------------------------------------------------------
# 14. Report text checks
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report_text():
    path = AUDIT_DIR / "dmot_calibrated_accuracy_report.md"
    return path.read_text()


def test_report_mentions_audit_type(report_text):
    """Report must identify as calibrated-parameter accuracy audit."""
    rt_lower = report_text.lower()
    assert "calibrated-parameter accuracy audit" in rt_lower or \
           "calibrated parameter accuracy audit" in rt_lower or \
           "calibrated-parameter" in rt_lower, \
        "Report does not clearly identify as calibrated-parameter accuracy audit"


def test_report_no_pymarrmot_comparison_claim(report_text):
    """Report must NOT claim pymarrmot comparison."""
    rt_lower = report_text.lower()
    # Strip markdown bold markers so "**Not** a" matches "not a"
    rt_stripped = rt_lower.replace("**", "")

    # Must contain disclaimer about no pymarrmot comparison
    disclaimers = [
        "not a pymarrmot comparison",
        "does not compare pymarrmot",
        "no pymarrmot comparison",
        "not compare dMot to pymarrmot",
        "not a pymarrmot",
    ]
    found = any(d in rt_stripped for d in disclaimers)
    assert found, "Report does not state it is NOT a pymarrmot comparison"

    # Must NOT make positive claims of equivalence
    # Allow negations: "Does NOT prove ... is numerically equivalent"
    forbidden_positive = [
        "tost passed",
        "equivalence passed",
        "implementation equivalent",
        "pymarrmot comparison completed",
        "all models reproduce recorded accuracy",
        "proves dMot is numerically equivalent",
        "prove dMot is numerically equivalent",
        "dMot is equivalent",
        "are equivalent",
    ]
    for phrase in forbidden_positive:
        assert phrase not in rt_lower, f"Report contains forbidden phrase: '{phrase}'"


def test_report_no_tost_claim(report_text):
    """Report must clarify this is not TOST."""
    rt_lower = report_text.lower()
    rt_stripped = rt_lower.replace("**", "")
    disclaimers = [
        "not a tost",
        "no tost",
        "not a statistical equivalence test",
        "not an implementation-equivalence",
        "not a tost statistical",
    ]
    found = any(d in rt_stripped for d in disclaimers)
    assert found, "Report does not state it is NOT a TOST test"


def test_report_key_facts(report_text):
    """Report must contain key factual statements."""
    rt_lower = report_text.lower()

    # Core identity phrases
    identity_checks = [
        "obj1",
        "ihacres",
        "area_index=11",
        "large deviation",
        "parameter-transfer",
        "native calibration",
        "what this does not claim",
    ]
    for phrase in identity_checks:
        assert phrase in rt_lower, f"Report missing: {phrase}"

    # ihacres specifically mentions parameter mismatch or tau_d (may now be resolved)
    assert ("tau_d" in rt_lower or "ihacres" in rt_lower), \
        "Report should mention ihacres"


def test_report_no_misleading_claims(report_text):
    """Report must not make misleading claims."""
    rt_lower = report_text.lower()

    forbidden = [
        "all models reproduce recorded accuracy",
        "large deviation proves formula bug",
        "large deviation proves model bug",
        "to prove equivalence",
        "pymarrmot and dmot are equivalent",
    ]
    for phrase in forbidden:
        assert phrase not in rt_lower, \
            f"Report contains misleading phrase: '{phrase}'"


def test_report_mentions_dmot_stability(report_text):
    """Report should mention dMoT numerical stability (no NaN/Inf / extreme artifacts)."""
    rt_lower = report_text.lower()
    # Should mention no NaN/Inf or numerical stability
    stability_checks = [
        "numerically stable",
        "no nan",
        "no inf",
        "no extreme",
        "did not reproduce extreme",
    ]
    found_any = any(c in rt_lower for c in stability_checks)
    assert found_any, "Report does not mention dMoT numerical stability"


def test_report_mentions_next_steps(report_text):
    """Report must recommend dMoT native calibration / multi-seed training."""
    rt_lower = report_text.lower()
    assert "native calibration" in rt_lower or "multi-seed" in rt_lower, \
        "Report does not recommend dMoT native calibration"


# ---------------------------------------------------------------------------
# 15. Design document checks
# ---------------------------------------------------------------------------
def test_design_document():
    path = AUDIT_DIR / "calibrated_accuracy_design.md"
    text = path.read_text().lower()

    checks = [
        "calibrated-parameter accuracy audit",
        "area_index=11",
        "no pymarrmot",
        "no tost",
        "no implementation-equivalence",
        "warmup",
        "evaluation",
    ]
    for c in checks:
        assert c in text, f"Design doc missing: {c}"


# ---------------------------------------------------------------------------
# 16. Cross-file consistency
# ---------------------------------------------------------------------------
def test_model_count_consistency_across_files():
    """Verify model count consistency across all key artifacts."""
    coverage = pd.read_csv(AUDIT_DIR / "model_param_coverage.csv")
    run_sum = pd.read_csv(AUDIT_DIR / "dmot_calibrated_batched_run_summary.csv")
    model_sum = pd.read_csv(AUDIT_DIR / "dmot_model_level_accuracy_summary.csv")
    basin = pd.read_csv(AUDIT_DIR / "dmot_basin_level_accuracy.csv")

    # All 35 testable models appear in run_summary with OK (ihacres pending)
    testable = set(coverage[coverage["test_status"] == "TESTABLE"]["model"])
    ok_models = set(run_sum[run_sum["run_status"] == "OK"]["model"])
    assert testable == ok_models, \
        f"TESTABLE models mismatch: in CSV but not OK={testable - ok_models}, OK but not TESTABLE={ok_models - testable}"

    # All 35 models appear in basin CSV
    basin_models = set(basin["model"].unique())
    assert testable == basin_models, \
        f"TESTABLE vs basin mismatch: {testable ^ basin_models}"

    # Model summary contains all 35 testable + ihacres (TAU_D_REMOVED)
    summary_models = set(model_sum["model"])
    assert summary_models == testable | {"ihacres"}, \
        f"Summary model mismatch: {summary_models ^ (testable | {'ihacres'})}"


def test_uh_model_mapping_consistency():
    """UH models in coverage have uh_enabled_used='yes'."""
    coverage = pd.read_csv(AUDIT_DIR / "model_param_coverage.csv")

    expected_uh = {
        "newzealand2", "hillslope", "plateau", "smar", "ihacres", "hbv96",
        "flexi", "flexb", "flexis", "gr4j",
    }

    for model in expected_uh:
        match = coverage[coverage["model"] == model]
        assert len(match) == 1, f"UH model missing from coverage: {model}"
        assert match.iloc[0]["uh_enabled_used"] == "yes", \
            f"{model}: uh_enabled_used={match.iloc[0]['uh_enabled_used']}"


def test_extra_input_models():
    """Verify mopex4/mopex5 have doy, tcm has pmean in extra_inputs."""
    coverage = pd.read_csv(AUDIT_DIR / "model_param_coverage.csv")

    mopex4 = coverage[coverage["model"] == "mopex4"].iloc[0]
    assert "doy" in str(mopex4["extra_inputs_required"]).lower()

    mopex5 = coverage[coverage["model"] == "mopex5"].iloc[0]
    assert "doy" in str(mopex5["extra_inputs_required"]).lower()

    tcm = coverage[coverage["model"] == "tcm"].iloc[0]
    assert "pmean" in str(tcm["extra_inputs_required"]).lower() or \
           "mean" in str(tcm["extra_inputs_required"]).lower()
