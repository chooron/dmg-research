from __future__ import annotations

from typing import Any

import numpy as np

from .schemas import PeriodResolution, PeriodSlice


def _as_day(value: Any) -> np.datetime64:
    return np.datetime64(str(value), "D")


def _resolve_one(name: str, dates: np.ndarray, spec: dict[str, Any]) -> PeriodSlice:
    start = _as_day(spec["start"])
    end = _as_day(spec["end"])
    if end < start:
        raise ValueError(f"{name} period ends before it starts")
    days = (end - start).astype("timedelta64[D]").astype(int) + 1
    matches_start = np.flatnonzero(dates.astype("datetime64[D]") == start)
    matches_end = np.flatnonzero(dates.astype("datetime64[D]") == end)
    if len(matches_start) != 1 or len(matches_end) != 1:
        raise ValueError(f"{name} dates are not both present in the dataset: {start}..{end}")
    start_index = int(matches_start[0])
    end_index = int(matches_end[0])
    if end_index - start_index + 1 != days:
        raise ValueError(f"{name} date index is not contiguous")
    return PeriodSlice(
        name=name,
        start=str(start),
        end=str(end),
        start_index=start_index,
        end_index=end_index,
        days=int(days),
    )


def resolve_periods(
    dates: np.ndarray,
    periods: dict[str, dict[str, Any]],
    *,
    warmup_days: int,
) -> PeriodResolution:
    dates_day = np.asarray(dates).astype("datetime64[D]")
    if dates_day.ndim != 1 or len(dates_day) < 2:
        raise ValueError("dates must be a one-dimensional sequence")
    diffs = np.diff(dates_day.astype("int64"))
    if not np.all(diffs == 1):
        raise ValueError("dataset dates must be strictly daily and contiguous")
    if len(np.unique(dates_day)) != len(dates_day):
        raise ValueError("dataset dates contain duplicates")
    warmup = _resolve_one("warmup", dates_day, periods["warmup"])
    train = _resolve_one("train", dates_day, periods["train"])
    if periods.get("test") is not None:
        test = _resolve_one("test", dates_day, periods["test"])
    else:
        test = None
    if train.start_index != warmup.end_index + 1:
        raise ValueError("warmup and train must be adjacent")
        
    if test is not None:
        if test.start_index != train.end_index + 1:
            raise ValueError("train and test must be adjacent")
    if abs(warmup.days - warmup_days) > 1:
        raise ValueError(f"warmup must have ~{warmup_days} days, got {warmup.days}")
    if train.start_index - warmup.start_index != warmup.days:
        raise ValueError("train start must follow the configured warmup length")
        
    test_forcing_start_index = -1
    test_forcing_end_index = -1
    test_warmup_days = 0
    if test is not None:
        test_forcing_start_index = test.start_index - warmup.days
        test_forcing_end_index = test.end_index + 1
        test_warmup_days = warmup_days
        if test_forcing_start_index < 0:
            raise ValueError("not enough preceding dates for test warmup")
            
    return PeriodResolution(
        warmup=warmup,
        train=train,
        test=test,
        train_forcing_start_index=warmup.start_index,
        train_forcing_end_index=train.end_index + 1,
        test_forcing_start_index=test_forcing_start_index,
        test_forcing_end_index=test_forcing_end_index,
        test_warmup_days=test_warmup_days,
    )
