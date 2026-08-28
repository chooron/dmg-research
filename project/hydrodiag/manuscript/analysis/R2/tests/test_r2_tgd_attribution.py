"""Tests for TGD attribution control and paired Delta_beta bootstrap."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R2_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R2_DIR))

from tgd_attribution_control import analyze_tgd_attribution_control
from r2_config import TOTAL_BASINS


def test_tgd_attribution_control_and_paired_delta_beta():
    b_rows, s_rows, r_rows, d_rows, meta = analyze_tgd_attribution_control(draws=500)

    assert meta["status"] == "PASS"
    assert len(b_rows) == TOTAL_BASINS * 2 * 3
    assert len(d_rows) == 4  # 2 paradigms x 2 subsets (Full, Excl. S5)

    # Check paired bootstrap Delta_beta results
    for r in d_rows:
        assert r["paired_bootstrap"] is True
        assert np.isfinite(r["delta_beta"])
        assert np.isfinite(r["delta_beta_ci_lower"])
        assert np.isfinite(r["delta_beta_ci_upper"])
        assert r["delta_beta_ci_lower"] <= r["delta_beta"] <= r["delta_beta_ci_upper"]
