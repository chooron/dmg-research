"""Tests for macro whole-space response, ensemble excess, and prevalence."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R2_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R2_DIR))

from macro_whole_space import analyze_macro_whole_space
from r2_config import TOTAL_BASINS


def test_macro_whole_space_prevalences_and_slopes():
    cd_b, cd_sum, ens_b, ens_sum, meta = analyze_macro_whole_space(draws=500)

    assert meta["status"] == "PASS"
    assert meta["total_basins"] == TOTAL_BASINS

    # 1. Prevalence of between_all > within_pooled
    ic_prev = meta["prevalence_between_gt_within"]["IC_Full531"]
    dpl_prev = meta["prevalence_between_gt_within"]["dPL_Full531"]

    # Exact expected: IC = 335/531 = 63.09%, dPL = 445/531 = 83.80%
    assert np.isclose(ic_prev, 335 / 531, atol=1e-3)
    assert np.isclose(dpl_prev, 445 / 531, atol=1e-3)

    # 2. Excess OLS slopes on frac_snow
    ic_slope = meta["excess_slope"]["IC_Full531"]
    dpl_slope = meta["excess_slope"]["dPL_Full531"]

    # Both slopes must be clearly positive
    assert ic_slope > 0.10
    assert dpl_slope > 0.10
    assert np.isclose(ic_slope, 0.1542, atol=1e-3)
