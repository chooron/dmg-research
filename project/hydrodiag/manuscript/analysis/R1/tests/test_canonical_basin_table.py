"""Tests for canonical basin-level table construction and schema audit."""
import sys
from pathlib import Path

import pytest

R1_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R1_DIR))

from canonical_basin_table import build_canonical_basin_table
from config import STRATA, STRATA_COUNTS, TOTAL_BASINS


def test_canonical_basin_table_dimensions_and_strata():
    test_rows, all_rows, audit = build_canonical_basin_table()

    # 531 basins x 3 structures x 2 regimes = 3186 test rows
    assert len(test_rows) == 3186
    # 531 basins x 3 structures x 2 regimes x 2 periods = 6372 all-period rows
    assert len(all_rows) == 6372

    assert audit["total_unique_basins"] == TOTAL_BASINS
    assert audit["strata_counts"] == STRATA_COUNTS

    # Unique basins in test rows
    unique_test_basins = {r["basin_id"] for r in test_rows}
    assert len(unique_test_basins) == 531

    # Check dPL seed aggregation provenance
    dpl_test_rows = [r for r in test_rows if r["paradigm"] == "dPL-MLP"]
    assert len(dpl_test_rows) == 531 * 3
    for r in dpl_test_rows:
        assert r["seed_or_restart"] == "median_across_seeds"

    # Check IC restart selection provenance
    ic_test_rows = [r for r in test_rows if r["paradigm"] == "IC-CMA-ES"]
    assert len(ic_test_rows) == 531 * 3
    for r in ic_test_rows:
        assert r["seed_or_restart"] == "selected_restart"


def test_required_columns_present():
    test_rows, _, _ = build_canonical_basin_table()
    required = [
        "basin_id", "regime", "paradigm", "structure", "frac_snow", "snow_stratum",
        "KGE", "signed_CT_error", "absolute_CT_error", "valid_year_count"
    ]
    for r in test_rows[:10]:
        for col in required:
            assert col in r
            assert r[col] is not None
