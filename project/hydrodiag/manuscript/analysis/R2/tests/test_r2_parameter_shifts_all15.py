"""Tests for all 15 parameter signed shifts and directional consistency."""
import sys
from pathlib import Path

import pandas as pd
import pytest

R2_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R2_DIR))

from parameter_shifts_all15 import analyze_parameter_shifts_all15
from r2_config import TOTAL_BASINS


def test_parameter_shifts_all15_dimensions_and_signatures():
    b_shifts, full_sum, strata_sum, rob_sum, meta = analyze_parameter_shifts_all15(draws=500)

    assert meta["status"] == "PASS"
    assert len(b_shifts) == TOTAL_BASINS * 2 * 15
    assert len(full_sum) == 30
    assert len(strata_sum) == 30 * 5

    # Check key signatures: um positive slope, ki negative slope, ci negative slope
    full_df = pd.DataFrame(full_sum)
    for p in ["IC", "dPL"]:
        um_row = full_df[(full_df["paradigm"] == p) & (full_df["parameter"] == "xaj_um")].iloc[0]
        ki_row = full_df[(full_df["paradigm"] == p) & (full_df["parameter"] == "xaj_ki")].iloc[0]
        ci_row = full_df[(full_df["paradigm"] == p) & (full_df["parameter"] == "xaj_ci")].iloc[0]

        assert um_row["slope_beta"] > 0.30  # positive compensation
        assert ki_row["slope_beta"] < -0.20  # negative compensation
        assert ci_row["slope_beta"] < -0.20  # negative compensation
