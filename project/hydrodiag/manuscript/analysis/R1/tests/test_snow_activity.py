"""Tests for snow activity primary summaries, Spearman associations, and endpoint contrasts."""
import sys
from pathlib import Path

import pytest
import torch

R1_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R1_DIR))

from snow_activity_analysis import analyze_snow_activity
from config import PARADIGMS, STRATA, STRATA_COUNTS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_snow_activity_strata_and_endpoints():
    strat_rows, sp_rows, ep_rows, meta = analyze_snow_activity(draws=1000)

    assert meta["status"] == "PASS"
    assert meta["strata_counts"] == STRATA_COUNTS

    # Endpoint D_activity (S5 - S1) must be positive for both IC and dPL
    for ep in ep_rows:
        if ep["metric"] == "delta_absCT_Base_CN":
            assert ep["D_activity"] > 40.0  # Around ~46-47 days
            assert ep["ci_low"] > 35.0

    # Spearman rank correlation must be positive for both IC and dPL
    for sp in sp_rows:
        if sp["metric"] == "delta_absCT_Base_CN":
            assert sp["spearman_rho"] > 0.40
            assert sp["ci_low"] > 0.30

    # Stratified counts check
    for p in PARADIGMS:
        for s in STRATA:
            match = [r for r in strat_rows if r["paradigm"] == p and r["stratum"] == s and r["metric"] == "delta_absCT_Base_CN" and r["table"] == "snow_stratified_contrast"]
            assert len(match) == 1
            assert match[0]["n_basins"] == STRATA_COUNTS[s]
