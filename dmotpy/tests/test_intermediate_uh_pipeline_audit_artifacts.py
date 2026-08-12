"""
Test: IntermediateUHModel Pipeline Wiring Audit — Artifact Validation
=======================================================================
"""
import json, os
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
AUDIT_DIR = REPO_ROOT / "dmotpy" / "validation_results" / "intermediate_uh_pipeline_audit"

REQUIRED_FILES = [
    "intermediate_pre_post_signature_audit.csv",
    "intermediate_passthru_alignment_audit.csv",
    "intermediate_param_broadcast_audit.csv",
    "intermediate_identity_uh_limit_results.csv",
    "intermediate_ea_water_balance_audit.csv",
    "intermediate_uh_pipeline_audit_report.md",
    "intermediate_uh_pipeline_audit_manifest.json",
]


def test_output_directory():
    assert AUDIT_DIR.is_dir()


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_artifact_exists(filename):
    path = AUDIT_DIR / filename
    assert path.exists(), f"Missing: {filename}"
    assert path.stat().st_size > 0, f"Empty: {filename}"


def test_signature_audit():
    df = pd.read_csv(AUDIT_DIR / "intermediate_pre_post_signature_audit.csv")
    assert len(df) == 4
    assert set(df["model"]) == {"flexi", "flexb", "flexis", "gr4j"}
    for _, row in df.iterrows():
        assert row["pre_reads_post_states"] == "no"
        assert "ALIGNED" in row["signature_status"]


def test_passthru_audit():
    df = pd.read_csv(AUDIT_DIR / "intermediate_passthru_alignment_audit.csv")
    assert len(df) == 6
    for _, row in df.iterrows():
        assert "ALIGNED" in row["order_status"]


def test_broadcast_audit():
    df = pd.read_csv(AUDIT_DIR / "intermediate_param_broadcast_audit.csv")
    assert len(df) >= 6
    for _, row in df.iterrows():
        assert row["broadcast_status"] == "ALIGNED"


def test_identity_limit():
    df = pd.read_csv(AUDIT_DIR / "intermediate_identity_uh_limit_results.csv")
    assert len(df) >= 9
    flex_rows = df[df["model"].isin(["flexi", "flexb", "flexis"])]
    for _, row in flex_rows.iterrows():
        assert "PASS" in row["status"], f"{row['model']}/{row['test_case']}: {row['status']}"


def test_ea_audit():
    df = pd.read_csv(AUDIT_DIR / "intermediate_ea_water_balance_audit.csv")
    assert len(df) == 4
    for _, row in df.iterrows():
        assert row["ea_returned_by_step_post"] == "yes"


def test_report():
    text = (AUDIT_DIR / "intermediate_uh_pipeline_audit_report.md").read_text().lower()
    assert "wiring" in text
    assert "no" in text  # "no code changes"
    assert "verified" in text or "aligned" in text
    assert "identity" in text
    assert "ea" in text
    assert "native calibration" in text
    # Must not claim pymarrmot or TOST
    forbidden = ["pymarrmot comparison", "tost passed"]
    for fb in forbidden:
        assert fb not in text


def test_manifest():
    with open(AUDIT_DIR / "intermediate_uh_pipeline_audit_manifest.json") as f:
        mf = json.load(f)
    assert mf["analysis_type"] == "intermediate_uh_pipeline_wiring_audit"
    assert mf["no_pymarrmot"] is True
    assert mf["no_tost"] is True
    assert mf["no_production_formula_change"] is True
    assert mf["key_findings"]["wiring_bugs_found"] == "NONE"
    assert mf["key_findings"]["blocks_native_calibration"] is False
