"""Regression tests for bettermodel-compatible dPL time sampling."""

from __future__ import annotations

import math

import numpy as np
from training.dpl.run_dpl_model import (
    bettermodel_training_iterations,
    build_valid_window_catalog,
    sample_bettermodel_window,
)


def test_iteration_count_matches_bettermodel_grid() -> None:
    n_basins, n_time, batch, warmup, prediction = 531, 5478, 128, 365, 365
    probability = batch * prediction / (n_basins * (n_time - warmup))
    expected = math.ceil(math.log(0.01) / math.log(1.0 - probability))
    assert (
        bettermodel_training_iterations(n_basins, n_time, batch, warmup, prediction)
        == expected
        == 266
    )


def test_random_window_is_730_days_and_targets_prediction_suffix() -> None:
    np.random.seed(42)
    basin_index, target_start = sample_bettermodel_window(
        n_basins=531,
        n_time=5478,
        batch_size=128,
        warmup_days=365,
        prediction_days=365,
    )
    assert basin_index.shape == (128,)
    assert target_start.shape == (128,)
    assert basin_index.min() >= 0 and basin_index.max() < 531
    # Matches dmg.HydroSampler.random_index: [warmup, n_time-rho).
    assert target_start.min() >= 365
    assert target_start.max() < 5478 - 365

    forcing_offsets = np.arange(-365, 365)
    target_offsets = np.arange(365)
    forcing_index = target_start[:, None] + forcing_offsets[None, :]
    target_index = (target_start - 365)[:, None] + target_offsets[None, :]
    assert forcing_index.shape == (128, 730)
    assert target_index.shape == (128, 365)
    # The forcing array is absolute-from-1980, while calibration targets are
    # indexed from 1981; subtract the one-year prefix before comparing.
    assert np.all(forcing_index[:, 365:] - 365 == target_index)


def test_valid_window_catalog_preserves_uniform_basin_sampling() -> None:
    observations = np.array(
        [
            [0.0] * 8 + [1.0] * 8,
            [np.nan, 1.0, 0.0, 2.0] * 4,
        ],
        dtype=np.float32,
    )
    catalog, summary = build_valid_window_catalog(
        observations,
        warmup_days=2,
        prediction_days=4,
        min_valid_points=3,
        min_observation_std=0.1,
    )
    assert len(catalog) == 2
    assert summary["fallback_basins"] == 0
    np.random.seed(7)
    basin_index, target_start = sample_bettermodel_window(
        n_basins=2,
        n_time=18,
        batch_size=200,
        warmup_days=2,
        prediction_days=4,
        window_catalog=catalog,
    )
    assert set(basin_index.tolist()) == {0, 1}
    for basin, start in zip(basin_index, target_start):
        assert start in set(catalog[int(basin)].tolist())
