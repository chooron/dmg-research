"""
Test: MARRMoT-UH vs dMoT-UH Algorithmic Alignment Diagnostic — Artifact Validation
===================================================================================
Validates all 11 diagnostic artifacts. Does NOT run MARRMoT or pymarrmot.
"""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
DIAG_DIR = REPO_ROOT / "dmotpy" / "validation_results" / "marrmot_uh_alignment_diagnostic"

REQUIRED_FILES = [
    "marrmot_uh_source_map.csv",
    "marrmot_uh_source_map.md",
    "uh_kernel_formula_comparison.csv",
    "uh_kernel_weight_comparison.csv",
    "uh_time_indexing_audit.csv",
    "uh_boundary_memory_audit.csv",
    "standalone_uh_alignment_results.csv",
    "integrated_uh_alignment_probe.csv",
    "uh_alignment_issue_cases.csv",
    "marrmot_uh_alignment_report.md",
    "marrmot_uh_alignment_manifest.json",
]

EXPECTED_UH_TYPES = {"half1", "full2", "tri3", "tri4", "exp5", "gamma6", "uniform7", "delay8"}


# ---------------------------------------------------------------------------
def test_output_directory_exists():
    assert DIAG_DIR.is_dir()


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_artifact_exists(filename):
    path = DIAG_DIR / filename
    assert path.exists(), f"Missing: {filename}"
    assert path.stat().st_size > 0, f"Empty: {filename}"


# ---------------------------------------------------------------------------
def test_source_map_csv():
    df = pd.read_csv(DIAG_DIR / "marrmot_uh_source_map.csv")
    assert len(df) > 0
    for col in ["file", "function_name", "role", "uh_type", "called_by", "time_loop_location",
                "ode_coupled", "routing_stateful", "boundary_handling"]:
        assert col in df.columns, f"Missing column: {col}"

    # Must find route.m and update_uh.m
    funcs = set(df["function_name"])
    assert "route" in funcs, "Missing route() in source map"
    assert "update_uh" in funcs, "Missing update_uh()"


def test_source_map_md():
    text = (DIAG_DIR / "marrmot_uh_source_map.md").read_text().lower()
    assert "stepwise" in text
    assert "full-sequence" in text or "full sequence" in text
    for key in ["route(", "update_uh", "uh(2,", "circshift"]:
        assert key.lower() in text, f"Source map MD missing: {key}"


# ---------------------------------------------------------------------------
def test_kernel_formula_csv():
    df = pd.read_csv(DIAG_DIR / "uh_kernel_formula_comparison.csv")
    assert len(df) >= 8
    for col in ["uh_name_marrmot", "uh_name_dmot", "formula_same", "likely_difference"]:
        assert col in df.columns

    # All kernels covered
    dMoT_names = set(df["uh_name_dmot"].str.lower())
    for ut in EXPECTED_UH_TYPES:
        has_kernel = any(f"dpl{ut}" == n.lower() for n in dMoT_names)
        assert has_kernel, f"dMoT UH kernel missing from formula comparison: {ut}"


# ---------------------------------------------------------------------------
def test_weight_comparison_csv():
    df = pd.read_csv(DIAG_DIR / "uh_kernel_weight_comparison.csv")
    assert len(df) >= 20
    for col in ["uh_type", "param_case", "max_abs_weight_diff", "status"]:
        assert col in df.columns

    # Should have MATCH entries
    statuses = set(df["status"])
    assert "MATCH" in statuses, f"No MATCH entries in weights; got {statuses}"


# ---------------------------------------------------------------------------
def test_timing_audit_csv():
    df = pd.read_csv(DIAG_DIR / "uh_time_indexing_audit.csv")
    assert len(df) > 0
    for col in ["uh_type", "flux_case", "timing_status", "first_nonzero_marrmot", "first_nonzero_dmot"]:
        assert col in df.columns


def test_boundary_audit_csv():
    df = pd.read_csv(DIAG_DIR / "uh_boundary_memory_audit.csv")
    assert len(df) > 0
    for col in ["uh_type", "flux_case", "input_total", "q_total_marrmot_finite", "q_total_dmot_finite"]:
        assert col in df.columns


# ---------------------------------------------------------------------------
def test_standalone_alignment_csv():
    df = pd.read_csv(DIAG_DIR / "standalone_uh_alignment_results.csv")
    assert len(df) >= 8
    for col in ["uh_type", "formula_status", "weight_status", "time_index_status",
                "boundary_status", "standalone_alignment_status", "likely_effect_on_accuracy",
                "recommended_action"]:
        assert col in df.columns

    # Covers all UH types
    covered = set(df["uh_type"])
    for ut in EXPECTED_UH_TYPES:
        assert ut in covered, f"Missing UH type in standalone: {ut}"

    # Each row has non-empty recommended_action
    for _, row in df.iterrows():
        assert isinstance(row["recommended_action"], str) and len(row["recommended_action"].strip()) > 0


# ---------------------------------------------------------------------------
def test_integrated_probe_csv():
    df = pd.read_csv(DIAG_DIR / "integrated_uh_alignment_probe.csv")
    assert len(df) >= 9
    for col in ["model", "uh_scope", "likely_difference_source", "notes"]:
        assert col in df.columns

    # Must cover endpoint and intermediate models
    scopes = set(df["uh_scope"])
    assert "endpoint" in scopes
    assert "intermediate" in scopes


# ---------------------------------------------------------------------------
def test_issue_cases_csv():
    df = pd.read_csv(DIAG_DIR / "uh_alignment_issue_cases.csv")
    assert len(df) >= 3
    for col in ["issue_id", "level", "severity", "issue_summary", "recommended_action",
                "blocks_native_calibration", "affects_paper_interpretation"]:
        assert col in df.columns

    # Should have HIGH severity issues
    high = df[df["severity"] == "HIGH"]
    assert len(high) >= 1

    # blocked calibration should be false
    for _, row in df.iterrows():
        blocked = str(row["blocks_native_calibration"]).lower()
        assert blocked in ("no", "false", "0"), f"Issue {row['issue_id']} incorrectly blocks calibration"


# ---------------------------------------------------------------------------
def test_report():
    text = (DIAG_DIR / "marrmot_uh_alignment_report.md").read_text().lower()

    # Identity
    assert "marrmot-uh" in text or "marrmot uh" in text
    assert "algorithmic" in text

    # No TOST / pymarrmot claims
    assert "not a tost" in text or "not a pymarrmot" in text

    # Key findings
    assert "stepwise" in text
    assert "conv1d" in text or "full-sequence" in text
    assert "not algorithmically equivalent" in text or "not aligned" in text or "fundamentally differ" in text

    # Architecture
    assert "route(" in text or "route " in text
    assert "update_uh" in text or "memory" in text

    # Interpretation
    assert "euler" in text
    assert "parameter-transfer" in text or "parameter transfer" in text

    # Implication for native calibration
    assert "native calibration" in text
    assert "does not block" in text or "does not prevent" in text or "not block" in text

    # Next steps
    assert "uh=on" in text or "uh_on" in text

    # Must not contain misleading claims
    forbidden = ["tost passed", "equivalent to marrmot", "numerically identical"]
    for phrase in forbidden:
        assert phrase not in text, f"Report contains forbidden: '{phrase}'"


# ---------------------------------------------------------------------------
def test_manifest():
    with open(DIAG_DIR / "marrmot_uh_alignment_manifest.json") as f:
        mf = json.load(f)

    assert mf.get("analysis_type") == "marrmot_uh_algorithmic_alignment_diagnostic"
    assert mf.get("no_pymarrmot_comparison") is True
    assert mf.get("no_tost") is True

    key_findings = mf.get("key_findings", {})
    assert key_findings.get("uh_kernel_formulas_aligned") == "yes (all 8 kernels)"
    assert key_findings.get("uh_kernel_weights_aligned") == "yes (numerically verified)"
    assert "NO" in key_findings.get("routing_architecture_aligned", "")
    assert key_findings.get("native_calibration_blocked") == "no"

    assert "files" in mf
    for fname in mf["files"]:
        assert (DIAG_DIR / fname).exists(), f"Manifest file missing: {fname}"
