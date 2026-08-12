import numpy as np
import pytest

from ablation.ic_core.units import FT3S_TO_MMDAY_FACTOR, convert_ft3s_to_mm_day


def test_flow_conversion_scalar() -> None:
    result = convert_ft3s_to_mm_day(np.array([[1.0]]), np.array([1.0]))
    assert np.isclose(result[0, 0], FT3S_TO_MMDAY_FACTOR)


def test_flow_conversion_second_scalar() -> None:
    result = convert_ft3s_to_mm_day(np.array([[100.0]]), np.array([1000.0]))
    assert np.isclose(result[0, 0], 0.24465755455488, rtol=1e-12)


def test_flow_conversion_vectorized_and_zero() -> None:
    result = convert_ft3s_to_mm_day(np.array([[1.0, 0.0], [2.0, 4.0]]), np.array([1.0, 2.0]))
    assert result.shape == (2, 2)
    assert result[0, 1] == 0.0
    assert np.isclose(result[1, 0], result[0, 0])


def test_nan_preserved_and_negative_not_clipped() -> None:
    result = convert_ft3s_to_mm_day(np.array([[np.nan, -1.0, 0.0]]), np.array([10.0]))
    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])
    assert result[0, 2] == 0.0


def test_nonpositive_area_fails() -> None:
    with pytest.raises(ValueError):
        convert_ft3s_to_mm_day(np.array([[1.0]]), np.array([0.0]))


def test_float32_float64_consistency() -> None:
    a = convert_ft3s_to_mm_day(np.ones((2, 3), dtype=np.float32), np.array([10, 20], dtype=np.float32))
    b = convert_ft3s_to_mm_day(np.ones((2, 3), dtype=np.float64), np.array([10, 20], dtype=np.float64))
    assert np.allclose(a, b, rtol=1e-6)
