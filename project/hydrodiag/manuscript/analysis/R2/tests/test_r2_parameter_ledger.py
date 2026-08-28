"""Tests for raw long-form parameter ledger construction and completeness."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R2_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R2_DIR))

from parameter_ledger import build_raw_parameter_ledger
from r2_config import TOTAL_BASINS


def test_parameter_ledger_completeness_and_bounds():
    rows, audit = build_raw_parameter_ledger()

    assert audit["status"] == "PASS"
    assert len(rows) == 310635
    assert audit["total_rows"] == 310635

    df = pd.DataFrame(rows)
    assert len(df["basin_id"].unique()) == TOTAL_BASINS
    assert set(df["paradigm"].unique()) == {"IC", "dPL"}
    assert set(df["structure"].unique()) == {"Base", "CN", "TGD"}
    assert len(df["parameter"].unique()) == 15

    # Check normalized coordinate precision
    norm_calc = (df["physical_value"] - df["lower_bound"]) / (df["upper_bound"] - df["lower_bound"])
    assert np.allclose(df["normalized_value"], norm_calc, atol=1e-12)
    assert (df["normalized_value"] >= 0.0).all()
    assert (df["normalized_value"] <= 1.0).all()
