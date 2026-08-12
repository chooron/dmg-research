"""
Test: Native Calibration Readiness Audit — Artifact Validation
===============================================================
"""
import json, os
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
AUDIT_DIR = REPO_ROOT / "dmotpy" / "validation_results" / "native_calibration_readiness"

REQUIRED = [
    "native_calibration_readiness_design.md",
    "native_calibration_readiness_design.yaml",
    "parameter_transform_audit.csv",
    "quick_gradient.csv",
    "quick_loss.csv",
    "quick_optimizer.csv",
    "quick_uh_grad.csv",
    "quick_multi.csv",
    "native_calibration_readiness_report.md",
    "native_calibration_readiness_manifest.json",
]


def test_dir(): assert AUDIT_DIR.is_dir()

@pytest.mark.parametrize("f", REQUIRED)
def test_exists(f):
    p = AUDIT_DIR / f
    assert p.exists(), f"Missing: {f}"
    assert p.stat().st_size > 0, f"Empty: {f}"


def test_param_transform():
    df = pd.read_csv(AUDIT_DIR / "parameter_transform_audit.csv")
    assert len(df) > 0
    assert "INVALID_BOUND" not in set(df["status"])


def test_optimizer_smoke():
    df = pd.read_csv(AUDIT_DIR / "quick_optimizer.csv")
    assert len(df) >= 5
    passes = df[df["status"].str.startswith("PASS")]
    assert len(passes) >= 4, f"Only {len(passes)} optimizer smoke pass"


def test_loss_gradient():
    df = pd.read_csv(AUDIT_DIR / "quick_loss.csv")
    passes = df[df["status"] == "PASS"]
    assert len(passes) >= 5, f"Only {len(passes)} loss gradients pass"


def test_uh_gradient():
    df = pd.read_csv(AUDIT_DIR / "quick_uh_grad.csv")
    assert len(df) >= 4


def test_multi_seed():
    df = pd.read_csv(AUDIT_DIR / "quick_multi.csv")
    models = set(df["model"])
    assert "collie1" in models
    assert "hbv96" in models or "flexi" in models


def test_report():
    text = (AUDIT_DIR / "native_calibration_readiness_report.md").read_text().lower()
    assert "not tost" in text or "not a tost" in text
    assert "not pymarrmot" in text
    assert "native calibration" in text
    assert "decision" in text


def test_manifest():
    with open(AUDIT_DIR / "native_calibration_readiness_manifest.json") as f:
        mf = json.load(f)
    assert mf["no_pymarrmot"] is True
    assert mf["no_tost"] is True
    assert "final_decision" in mf
