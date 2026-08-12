"""
Test: Conv1d vs Stepwise UH Routing Diagnostic — Artifact Validation
=======================================================================
"""
import json, os
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
DIAG_DIR = REPO_ROOT / "dmotpy" / "validation_results" / "uh_conv1d_vs_stepwise_diagnostic"

REQUIRED_FILES = [
    "diagnostic_design.md", "diagnostic_design.yaml",
    "stepwise_uh_reference.py",
    "boundary_tail_effect_results.csv",
    "max_lag_truncation_audit.csv",
    "routing_linearity_check.csv",
    "endpoint_vs_intermediate_group_contrast.csv",
    "endpoint_vs_intermediate_group_summary.csv",
    "gr4j_exchange_coupling_diagnostic.csv",
    "mechanism_attribution_by_model.csv",
    "conv1d_vs_stepwise_diagnostic_report.md",
    "conv1d_vs_stepwise_diagnostic_manifest.json",
]


def test_output_directory_exists():
    assert DIAG_DIR.is_dir()


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_artifact_exists(filename):
    path = DIAG_DIR / filename
    assert path.exists(), f"Missing: {filename}"
    assert path.stat().st_size > 0, f"Empty: {filename}"


def test_stepwise_backend():
    path = DIAG_DIR / "stepwise_uh_reference.py"
    text = path.read_text()
    assert "route_step" in text
    assert "update_step" in text
    assert "route_sequence_stepwise" in text
    assert "conv1d_route_sequence" in text


def test_boundary_tail_csv():
    df = pd.read_csv(DIAG_DIR / "boundary_tail_effect_results.csv")
    assert len(df) > 0
    for col in ["model", "uh_scope", "branch", "boundary_contribution_fraction", "status"]:
        assert col in df.columns
    assert "NEGLIGIBLE" in set(df["status"])


def test_max_lag_truncation_csv():
    df = pd.read_csv(DIAG_DIR / "max_lag_truncation_audit.csv")
    assert len(df) > 100  # Many basins tested
    for col in ["model", "uh_scope", "uh_type", "dynamic_support_length", "dmot_max_lag", "status"]:
        assert col in df.columns
    statuses = set(df["status"])
    assert "NO_TRUNCATION" in statuses


def test_routing_linearity_csv():
    df = pd.read_csv(DIAG_DIR / "routing_linearity_check.csv")
    assert len(df) > 0
    assert "LINEAR" in set(df["status"])


def test_endpoint_vs_intermediate_contrast():
    df = pd.read_csv(DIAG_DIR / "endpoint_vs_intermediate_group_contrast.csv")
    assert len(df) > 0
    for col in ["model", "uh_scope", "status"]:
        assert col in df.columns
    scopes = set(df["uh_scope"])
    assert "endpoint" in scopes
    assert "intermediate" in scopes


def test_endpoint_vs_intermediate_summary():
    df = pd.read_csv(DIAG_DIR / "endpoint_vs_intermediate_group_summary.csv")
    assert len(df) == 2
    groups = set(df["group"])
    assert "endpoint" in groups
    assert "intermediate" in groups


def test_gr4j_exchange_csv():
    df = pd.read_csv(DIAG_DIR / "gr4j_exchange_coupling_diagnostic.csv")
    assert len(df) >= 0
    for col in ["basin_id", "difference_amplification_stage"]:
        assert col in df.columns


def test_mechanism_attribution():
    df = pd.read_csv(DIAG_DIR / "mechanism_attribution_by_model.csv")
    assert len(df) >= 8
    for col in ["model", "uh_scope", "dominant_mechanism", "blocks_native_calibration", "recommended_action"]:
        assert col in df.columns

    # All models have non-empty recommended_action
    for _, row in df.iterrows():
        assert isinstance(row["recommended_action"], str) and len(row["recommended_action"].strip()) > 0

    # No model blocks native calibration except ihacres (if present)
    blocked = df[df["blocks_native_calibration"] == "yes"]
    if len(blocked) > 0:
        assert set(blocked["model"]) <= {"ihacres"}


def test_report():
    text = (DIAG_DIR / "conv1d_vs_stepwise_diagnostic_report.md").read_text().lower()

    assert "not pymarrmot" in text or "not a pymarrmot" in text
    assert "not tost" in text or "not a tost" in text
    assert "not native calibration" in text

    assert "stepwise" in text
    assert "conv1d" in text
    assert "boundary" in text
    assert "linear" in text

    # Key questions answered
    assert "(a)" in text or "faithful" in text or "endpoint" in text
    assert "(b)" in text or "feedback" in text or "decoupling" in text

    assert "not block" in text or "does not block" in text

    # No misleading claims
    forbidden = ["tost passed", "equivalent to marrmot"]
    for phrase in forbidden:
        assert phrase not in text


def test_manifest():
    with open(DIAG_DIR / "conv1d_vs_stepwise_diagnostic_manifest.json") as f:
        mf = json.load(f)

    assert "analysis_type" in mf
    assert mf.get("no_pymarrmot") is True
    assert mf.get("no_tost") is True
    assert mf.get("no_native_calibration") is True
    assert "main_hypothesis_decision" in mf
    assert "output_files" in mf
    assert "key_findings" in mf

    for fname in mf.get("output_files", []):
        assert (DIAG_DIR / fname).exists(), f"Manifest file missing: {fname}"


def test_design_yaml():
    import yaml
    with open(DIAG_DIR / "diagnostic_design.yaml") as f:
        ds = yaml.safe_load(f)
    assert ds["main_hypothesis"] == "intermediate_uh_differences_dominated_by_routing_storage_feedback_decoupling"
    assert ds["no_pymarrmot"] is True
    assert ds["no_tost"] is True
