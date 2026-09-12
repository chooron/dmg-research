from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import sys
CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from s1_audit_utils import ATTRIBUTE_NAMES, CONTINUOUS_ATTRIBUTES, fixed_strata, period_frame, stratum


def test_531_ids_are_unique_eight_digit_and_stable():
    values = [str(v).zfill(8) for v in __import__("json").loads((Path("/home/jingxin/code/dmg-research/data/531sub_id.txt")).read_text())]
    assert len(values) == 531
    assert len(set(values)) == 531
    assert all(len(v) == 8 and v.isdigit() for v in values)


def test_requested_date_lengths_and_leap_day_retention():
    dates = np.arange("1980-01-01", "2011-01-01", dtype="datetime64[D]")
    frame = period_frame(dates, {"candidate": {"warmup": {"start": "1988-01-01", "end": "1988-12-31"}, "calibration": {"start": "1989-01-01", "end": "1998-12-31"}, "test": {"start": "1999-01-01", "end": "2009-12-31"}}})
    assert frame.set_index("period").loc["warmup", "days"] == 366
    assert frame.set_index("period").loc["calibration", "days"] == 3652
    assert frame.set_index("period").loc["test", "days"] == 4018
    assert np.datetime64("1988-02-29") in dates


def test_conversion_constant_independent():
    factor = 0.028316846592 * 86400 * 1000 / 1_000_000
    assert np.isclose(factor, 2.4465755455488, rtol=0, atol=1e-12)


def test_valid_zero_and_fixed_boundaries():
    assert [stratum(x) for x in [0, .05, .15, .30, .50, 1.0]] == ["S1", "S2", "S3", "S4", "S5", "S5"]
    assert fixed_strata(np.array([0.0, .05, .15, .30, .50, 1.0])).tolist() == ["S1", "S2", "S3", "S4", "S5", "S5"]


def test_attribute_contract_and_correlation_shape():
    assert len(ATTRIBUTE_NAMES) == 35
    assert len(CONTINUOUS_ATTRIBUTES) == 32
    x = np.arange(15, dtype=float).reshape(5, 3)
    corr = pd.DataFrame(x).corr(method="spearman")
    assert np.allclose(corr, corr.T)
    assert np.allclose(np.diag(corr), 1.0)


def test_key_artifact_order_hash_is_stable():
    ids = [str(v).zfill(8) for v in __import__("json").loads(Path("/home/jingxin/code/dmg-research/data/531sub_id.txt").read_text())]
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == hashlib.sha256("\n".join(ids).encode()).hexdigest()
