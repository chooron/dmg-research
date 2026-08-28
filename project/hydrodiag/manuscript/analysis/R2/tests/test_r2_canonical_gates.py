"""Tests for all 12 R2 canonical validation gates."""
import sys
from pathlib import Path

import pytest

R2_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R2_DIR))

from r2_canonical_gates import verify_r2_canonical_gates
from run_r2 import run_r2_pipeline


def test_all_12_r2_canonical_gates_pass():
    summary = run_r2_pipeline(draws=200)
    assert summary["canonical_gates"] == "PASS"

    gate_report = verify_r2_canonical_gates()
    assert gate_report["overall_status"] == "PASS"
    for g_name, g_info in gate_report["gates"].items():
        assert g_info["status"] == "PASS", f"Gate {g_name} failed: {g_info.get('failures')}"
