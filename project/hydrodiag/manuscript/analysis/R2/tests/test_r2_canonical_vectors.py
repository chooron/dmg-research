"""Tests for canonical basin-level parameter vector reduction rules."""
import sys
from pathlib import Path

import pandas as pd
import pytest

R2_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R2_DIR))

from canonical_vectors import build_canonical_parameter_vectors
from r2_config import STRATA_COUNTS, TOTAL_BASINS


def test_canonical_vectors_rules_and_dimensions():
    c_rows, audit = build_canonical_parameter_vectors()

    assert audit["status"] == "PASS"
    assert len(c_rows) == 3186
    assert audit["total_basins"] == TOTAL_BASINS

    df = pd.DataFrame(c_rows)
    assert len(df["basin_id"].unique()) == TOTAL_BASINS
    assert set(df["paradigm"].unique()) == {"IC", "dPL"}
    assert set(df["structure"].unique()) == {"Base", "CN", "TGD"}

    # Verify IC rule
    ic_df = df[df["paradigm"] == "IC"]
    assert (ic_df["reduction_rule"] == "best_train_kge_restart").all()

    # Verify dPL rule
    dpl_df = df[df["paradigm"] == "dPL"]
    assert (dpl_df["reduction_rule"] == "median_across_seeds").all()

    # Verify snow strata counts
    snow_counts = df.drop_duplicates("basin_id")["snow_stratum"].value_counts().to_dict()
    assert snow_counts == STRATA_COUNTS
