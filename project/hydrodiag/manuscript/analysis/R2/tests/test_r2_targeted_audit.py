"""Tests for R2 targeted audit: S1-S5 macro trajectory and TGD leverage diagnostics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R2_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R2_DIR))

from r2_targeted_audit import run_r2_targeted_audit


def test_r2_targeted_audit_execution_and_verdicts():
    audit_summary = run_r2_targeted_audit(draws=500)

    assert audit_summary["status"] == "PASS"
    assert audit_summary["sample_and_pairing_verification"].startswith("PASS")
    assert audit_summary["paired_delta_beta_verification"].startswith("PASS")

    # Check IC monotonicity
    ic_v = audit_summary["wording_verdicts"]["IC"]
    assert ic_v["verdict"] == "MONOTONIC / NEAR-MONOTONIC ORGANIZATION"

    # Check dPL ordered but nonlinear
    dpl_v = audit_summary["wording_verdicts"]["dPL"]
    assert dpl_v["verdict"] == "ORDERED BUT NONLINEAR"


def test_s1_s5_trajectory_invariants():
    out_dir = R2_DIR / "results"
    traj_df = pd.read_csv(out_dir / "r2_s1_s5_macro_trajectory.csv")

    assert len(traj_df) == 2 * 2 * 5  # 2 paradigms x 2 contrasts x 5 strata

    # Check IC Base-CN monotonic increase in excess
    ic_cn = traj_df[(traj_df["paradigm"] == "IC") & (traj_df["contrast"] == "Base-CN")].set_index("snow_stratum")
    e_vals = [ic_cn.loc[s, "excess_median"] for s in ["S1", "S2", "S3", "S4", "S5"]]
    assert all(x < y for x, y in zip(e_vals, e_vals[1:]))

    # Check dPL Base-CN S1-S4 steep rise and S5 plateau
    dpl_cn = traj_df[(traj_df["paradigm"] == "dPL") & (traj_df["contrast"] == "Base-CN")].set_index("snow_stratum")
    assert dpl_cn.loc["S1", "excess_median"] < dpl_cn.loc["S2", "excess_median"] < dpl_cn.loc["S3", "excess_median"]
    assert dpl_cn.loc["S4", "excess_median"] > dpl_cn.loc["S5", "excess_median"] - 0.02


def test_leverage_diagnostics_s5_high_leverage():
    out_dir = R2_DIR / "results"
    diag_df = pd.read_csv(out_dir / "r2_leverage_influence_diagnostics.csv")

    for p in ["IC", "dPL"]:
        for c in ["Base-CN", "Base-TGD"]:
            sub = diag_df[(diag_df["paradigm"] == p) & (diag_df["contrast"] == c)].set_index("snow_stratum")
            # S5 leverage must be markedly higher than S1, S2, S3 leverage (at least 3x higher)
            assert sub.loc["S5", "mean_leverage"] > 3 * sub.loc["S2", "mean_leverage"]
            assert sub.loc["S5", "mean_leverage"] > 3 * sub.loc["S3", "mean_leverage"]
