"""Tests for 5 canonical gates verification."""
import sys
from pathlib import Path

import pytest

R1_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R1_DIR))

from canonical_gates import verify_canonical_gates
from run_all import run_pipeline


def test_all_canonical_gates_pass():
    # Run full pipeline to ensure all output files are populated
    summary = run_pipeline(draws=200)
    assert summary["canonical_gates"] == "PASS"

    gate_report = verify_canonical_gates()
    assert gate_report["overall_status"] == "PASS"
    for g_name, g_info in gate_report["gates"].items():
        assert g_info["status"] == "PASS", f"Gate {g_name} failed: {g_info.get('failures')}"
