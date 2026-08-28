"""Tests for R2 final completeness & statistical validity closure audit."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R2_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R2_DIR))

from r2_final_audit import run_r2_final_closure_audit


def test_r2_final_closure_audit_verdict_and_trace():
    closure = run_r2_final_closure_audit(draws=500)

    assert closure["status"] == "COMPLETED"
    assert closure["R2_FINAL_STATUS"] == "READY"
    assert closure["lopo_domination_audit"]["status"] == "PASS"

    # Check 4-basin trace file
    out_dir = R2_DIR / "results"
    trace_df = pd.read_csv(out_dir / "r2_four_basin_calculation_trace.csv")
    assert len(trace_df) == 4
    for _, tr in trace_df.iterrows():
        assert np.isclose(tr["within_pooled"], (tr["within_base_median"] + tr["within_cn_median"]) / 2.0, atol=1e-6)
        assert np.isclose(tr["excess"], tr["between_all_median"] - tr["within_pooled"], atol=1e-6)
        assert tr["between_gt_within"] == (tr["between_all_median"] > tr["within_pooled"])


def test_basin_paired_cn_tgd_delta_excess():
    out_dir = R2_DIR / "results"
    de_df = pd.read_csv(out_dir / "r2_paired_cn_tgd_delta_excess_summary.csv")

    assert len(de_df) == 2 * 7  # 2 paradigms x 7 strata/subsets

    # IC Full531 delta_excess is near zero
    ic_full = de_df[(de_df["paradigm"] == "IC") & (de_df["stratum"] == "Full531")].iloc[0]
    assert np.abs(ic_full["median_delta_excess"]) < 0.01

    # dPL Full531 delta_excess is positive with CI > 0
    dpl_full = de_df[(de_df["paradigm"] == "dPL") & (de_df["stratum"] == "Full531")].iloc[0]
    assert dpl_full["median_delta_excess"] > 0.01
    assert dpl_full["ci_lower"] > 0.005


def test_leave_one_parameter_out_domination_robustness():
    out_dir = R2_DIR / "results"
    lopo_df = pd.read_csv(out_dir / "r2_leave_one_parameter_out_sensitivity.csv")

    assert len(lopo_df) == 2 * 16  # 2 paradigms x (1 baseline + 15 parameter omissions)

    # In IC, all 14-D slopes should be within [0.13, 0.18]
    ic_14d = lopo_df[(lopo_df["paradigm"] == "IC") & (lopo_df["dimension"] == 14)]
    assert (ic_14d["excess_slope_beta"] >= 0.13).all()
    assert (ic_14d["excess_slope_beta"] <= 0.18).all()

    # In dPL, all 14-D slopes should be within [0.15, 0.23]
    dpl_14d = lopo_df[(lopo_df["paradigm"] == "dPL") & (lopo_df["dimension"] == 14)]
    assert (dpl_14d["excess_slope_beta"] >= 0.15).all()
    assert (dpl_14d["excess_slope_beta"] <= 0.23).all()
