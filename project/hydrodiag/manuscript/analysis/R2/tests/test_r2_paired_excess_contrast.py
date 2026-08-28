"""Regression test for Stage 7: Direct Basin-Paired Base-CN vs Base-TGD Macro Excess Contrast."""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import pytest

R2_DIR = Path(__file__).parents[1]
if str(R2_DIR) not in sys.path:
    sys.path.insert(0, str(R2_DIR))

from paired_excess_contrast import compute_paired_excess_contrast
from r2_config import RESULTS_DIR, TOTAL_BASINS
def test_paired_excess_contrast_computation(tmp_path: Path):
    b_df, s_df, p_df, meta = compute_paired_excess_contrast(output_dir=tmp_path, draws=500)

    assert meta["status"] == "PASS"
    assert meta["verdict"] == "VERDICT_A_INTERMEDIATE_EMERGENCE"

    # Basin level checks
    assert len(b_df) == TOTAL_BASINS * 2  # 531 for IC + 531 for dPL = 1062
    for paradigm in ["IC", "dPL"]:
        sub = b_df[b_df["paradigm"] == paradigm]
        assert len(sub) == TOTAL_BASINS
        assert len(sub["basin_id"].unique()) == TOTAL_BASINS

    # Summary level checks
    assert len(s_df) == 14  # 7 strata splits * 2 paradigms
    assert len(p_df) == 14

    # dPL S2/S3 positive delta_excess check
    dpl_s2 = s_df[(s_df["paradigm"] == "dPL") & (s_df["stratum"] == "S2")].iloc[0]
    assert dpl_s2["median_delta_excess"] > 0.02
    assert dpl_s2["prop_positive"] > 0.60

    dpl_s3 = s_df[(s_df["paradigm"] == "dPL") & (s_df["stratum"] == "S3")].iloc[0]
    assert dpl_s3["median_delta_excess"] > 0.015
    assert dpl_s3["prop_positive"] > 0.60

    # IC S1 negative / near-zero check
    ic_s1 = s_df[(s_df["paradigm"] == "IC") & (s_df["stratum"] == "S1")].iloc[0]
    assert abs(ic_s1["median_delta_excess"]) < 0.01

    # Prevalence checks
    ic_full_prev = p_df[(p_df["paradigm"] == "IC") & (p_df["stratum"] == "Full531")].iloc[0]
    assert 0.62 < ic_full_prev["base_cn_prevalence"] < 0.64

    dpl_full_prev = p_df[(p_df["paradigm"] == "dPL") & (p_df["stratum"] == "Full531")].iloc[0]
    assert 0.82 < dpl_full_prev["base_cn_prevalence"] < 0.85
