"""
Test: dMoT vs pymarrmot UH Value Comparison — Artifact Validation
===================================================================
"""
import json, os
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
AUDIT_DIR = REPO_ROOT / "dmotpy" / "validation_results" / "uh_value_comparison_dmot_vs_pymarrmot"

REQUIRED_FILES = [
    "selected_basins.csv",
    "pymarrmot_uh_behavior_check.csv",
    "dmot_uh_before_after_values.csv",
    "pymarrmot_uh_before_after_values.csv",
    "daily_before_after_comparison.csv",
    "model_basin_difference_summary.csv",
    "group_difference_summary.csv",
    "uh_before_after_value_comparison_report.md",
    "uh_value_comparison_manifest.json",
]


def test_output_directory():
    assert AUDIT_DIR.is_dir()


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_artifact_exists(filename):
    path = AUDIT_DIR / filename
    assert path.exists(), f"Missing: {filename}"
    assert path.stat().st_size > 0, f"Empty: {filename}"


def test_selected_basins():
    df = pd.read_csv(AUDIT_DIR / "selected_basins.csv")
    assert len(df) >= 3
    assert "basin_id" in df.columns
    assert "selection_reason" in df.columns


def test_pymarrmot_behavior():
    df = pd.read_csv(AUDIT_DIR / "pymarrmot_uh_behavior_check.csv")
    assert len(df) > 0
    assert "detected_behavior" in df.columns
    # All should be IDENTITY_ROUTING (pymarrmot UH entirely stubbed)
    for _, row in df.iterrows():
        assert "IDENTITY" in row["detected_behavior"]


def test_dmot_values():
    df = pd.read_csv(AUDIT_DIR / "dmot_uh_before_after_values.csv")
    assert len(df) > 0
    for col in ["model", "basin_id", "pre_routing_flux_dmot", "q_final_dmot"]:
        assert col in df.columns


def test_daily_comparison():
    df = pd.read_csv(AUDIT_DIR / "daily_before_after_comparison.csv")
    assert len(df) > 0
    for col in ["model", "basin_id", "q_final_dmot", "q_final_pymarrmot", "q_final_diff"]:
        assert col in df.columns


def test_model_basin_summary():
    df = pd.read_csv(AUDIT_DIR / "model_basin_difference_summary.csv")
    assert len(df) > 0
    for col in ["model", "basin_id", "dominant_difference_stage"]:
        assert col in df.columns
    # Should have endpoint and intermediate models
    models = set(df["model"])
    assert "hbv96" in models or "newzealand2" in models or "smar" in models  # endpoint
    assert "flexi" in models or "flexb" in models or "gr4j" in models  # intermediate


def test_group_summary():
    df = pd.read_csv(AUDIT_DIR / "group_difference_summary.csv")
    groups = set(df["group"])
    assert "endpoint" in groups
    assert "intermediate" in groups


def test_report():
    text = (AUDIT_DIR / "uh_before_after_value_comparison_report.md").read_text().lower()
    assert "not tost" in text or "not a tost" in text
    assert "not native calibration" in text
    assert "identity" in text
    assert "stubbed" in text or "stub" in text
    assert "pymarrmot" in text
    assert "cannot serve" in text or "cannot be used" in text
    assert "native calibration" in text


def test_manifest():
    with open(AUDIT_DIR / "uh_value_comparison_manifest.json") as f:
        mf = json.load(f)
    assert mf["no_tost"] is True
    assert mf["no_native_calibration"] is True
    assert mf["key_finding"] == "pymarrmot UH entirely stubbed to identity routing"
    assert len(mf["selected_basins"]) >= 3
    assert len(mf["selected_models"]) >= 5
