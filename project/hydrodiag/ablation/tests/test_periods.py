import numpy as np
from ablation.ic_core.periods import resolve_periods


def _config():
    return {
        "warmup": {"start": "1980-10-01", "end": "1981-09-30"},
        "train": {"start": "1981-10-01", "end": "1995-09-30"},
        "test": {"start": "1995-10-01", "end": "2010-09-30"},
    }


def test_period_lengths_and_adjacency() -> None:
    dates = np.arange(
        np.datetime64("1980-10-01"), np.datetime64("2014-10-01"), np.timedelta64(1, "D")
    )
    result = resolve_periods(dates, _config(), warmup_days=365)
    assert result.warmup.days == 365
    assert result.train.days == 5113
    assert result.test.days == 5479
    assert result.train.start_index == result.warmup.end_index + 1
    assert result.test.start_index == result.train.end_index + 1


def test_periods_are_disjoint_and_test_has_preceding_warmup() -> None:
    dates = np.arange(
        np.datetime64("1980-10-01"), np.datetime64("2014-10-01"), np.timedelta64(1, "D")
    )
    result = resolve_periods(dates, _config(), warmup_days=365)
    assert result.test_forcing_start_index == result.test.start_index - 365
    assert result.test_forcing_end_index == result.test.end_index + 1


def test_noncontiguous_dates_fail() -> None:
    dates = np.arange(
        np.datetime64("1980-10-01"), np.datetime64("2014-10-01"), np.timedelta64(1, "D")
    )
    dates = np.delete(dates, 100)
    try:
        resolve_periods(dates, _config(), warmup_days=365)
    except ValueError:
        return
    raise AssertionError("non-contiguous dates must fail")
