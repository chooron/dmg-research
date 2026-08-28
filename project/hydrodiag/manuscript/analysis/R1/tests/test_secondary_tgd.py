"""Tests for secondary TGD structural control and boundaries."""
import sys
from pathlib import Path

import pytest
import torch

R1_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R1_DIR))

from secondary_tgd_control import analyze_secondary_tgd_control


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_secondary_tgd_control():
    rows, meta = analyze_secondary_tgd_control(draws=500)

    assert meta["status"] == "PASS"
    assert "No F_TGD calculation" in meta["exclusions"]
    assert "No irreducible snow contribution claim" in meta["exclusions"]

    for r in rows:
        assert r["role"] == "secondary_output_structural_control"
