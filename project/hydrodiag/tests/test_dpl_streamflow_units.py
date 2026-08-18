"""Tests for the CAMELS observed-flow unit conversion used by dPL."""

from __future__ import annotations

import numpy as np
import pytest
from training.dpl.run_dpl_model import convert_streamflow_ft3s_to_mm_day


def test_ft3s_target_is_converted_to_basin_average_mm_day() -> None:
    # 1 ft^3/s over 1 km^2 equals 2.44657152 mm/day with the CAMELS
    # conversion constant used throughout this project.
    streamflow = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32)
    area_km2 = np.array([1.0, 10.0], dtype=np.float32)

    converted = convert_streamflow_ft3s_to_mm_day(streamflow, area_km2)

    expected = np.array([[2.446572, 4.893143], [2.446572, 4.893143]], dtype=np.float32)
    np.testing.assert_allclose(converted, expected, rtol=1e-6, atol=1e-6)


def test_ft3s_target_conversion_rejects_invalid_area() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        convert_streamflow_ft3s_to_mm_day(
            np.ones((2, 3), dtype=np.float32),
            np.array([100.0, 0.0], dtype=np.float32),
        )
