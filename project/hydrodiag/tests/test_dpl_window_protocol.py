"""Regression test for the released Lite-v2 dPL 730-day sampling protocol."""

import numpy as np


def test_random_window_has_365_warmup_and_365_scored_days():
    warmup_days, prediction_days = 365, 365
    # A target starting at forcing index 365 has a complete preceding year.
    target_start = np.asarray([365, 912], dtype=np.int64)
    forcing_offsets = np.arange(-warmup_days, prediction_days, dtype=np.int64)
    target_offsets = np.arange(prediction_days, dtype=np.int64)
    forcing_index = target_start[:, None] + forcing_offsets[None, :]
    target_index = (target_start - warmup_days)[:, None] + target_offsets[None, :]

    assert forcing_index.shape == (2, 730)
    assert target_index.shape == (2, 365)
    assert np.array_equal(
        forcing_index[:, :365], target_start[:, None] - 365 + np.arange(365)
    )
    assert np.array_equal(
        forcing_index[:, 365:], target_start[:, None] + np.arange(365)
    )
    assert np.array_equal(target_index, target_start[:, None] - 365 + np.arange(365))
