"""Test: Static attribute normalization produces no NaN/Inf."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))


def normalize_attrs(attr_raw):
    """NaN-safe: impute NaN with column median, handle constant columns."""
    attr = attr_raw.astype(np.float32).copy()
    n_cols = attr.shape[1]
    for j in range(n_cols):
        col = attr[:, j]
        col[np.isinf(col)] = np.nan
        nan_mask = np.isnan(col)
        n_imp = int(nan_mask.sum())
        if n_imp > 0 and n_imp < len(col):
            col[nan_mask] = np.nanmedian(col)
        elif n_imp == len(col):
            col[:] = 0.0
        cmin, cmax = float(col.min()), float(col.max())
        constant = abs(cmax - cmin) < 1e-10
        attr[:, j] = col
    a_min = attr.min(axis=0, keepdims=True)
    a_rng = np.maximum(attr.max(axis=0, keepdims=True) - a_min, 1e-8)
    result = (attr - a_min) / a_rng
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return result.astype(np.float32)


class TestAttributeNormalizationNoNaN:

    def test_normal_attributes_pass(self):
        attrs = np.random.randn(10, 35).astype(np.float32)
        normed = normalize_attrs(attrs)
        assert not np.isnan(normed).any(), "No NaN expected for normal data"
        assert not np.isinf(normed).any(), "No Inf expected for normal data"

    def test_nan_values_imputed(self):
        attrs = np.random.randn(10, 5).astype(np.float32)
        attrs[0, 0] = np.nan
        attrs[3, 2] = np.nan
        normed = normalize_attrs(attrs)
        assert not np.isnan(normed).any(), f"NaN should be imputed, got {normed}"
        assert not np.isinf(normed).any()

    def test_inf_values_converted(self):
        attrs = np.random.randn(10, 5).astype(np.float32)
        attrs[0, 0] = np.inf
        attrs[3, 2] = -np.inf
        normed = normalize_attrs(attrs)
        assert not np.isinf(normed).any(), f"Inf should be converted, got inf at {(np.isinf(normed)).sum()}"
        assert not np.isnan(normed).any()

    def test_all_nan_column_filled_zero(self):
        attrs = np.full((10, 5), np.nan, dtype=np.float32)
        normed = normalize_attrs(attrs)
        assert not np.isnan(normed).any()
        assert not np.isinf(normed).any()
        for j in range(5):
            assert abs(normed[:, j].max() - normed[:, j].min()) < 1e-8, "All-NaN column should be constant after fill"

    def test_constant_column_becomes_zero(self):
        attrs = np.random.randn(10, 5).astype(np.float32)
        attrs[:, 2] = 3.14
        normed = normalize_attrs(attrs)
        assert abs(normed[:, 2].max()) < 1e-8, f"Constant column should be 0, got max={normed[:, 2].max()}"
        assert abs(normed[:, 2].min()) < 1e-8

    def test_output_in_01_range(self):
        attrs = np.random.randn(100, 20).astype(np.float32)
        normed = normalize_attrs(attrs)
        assert normed.min() >= -1e-4, f"Normalized min {normed.min()} < 0"
        assert normed.max() <= 1.0 + 1e-4, f"Normalized max {normed.max()} > 1"

    def test_mixed_nan_inf_zero_columns(self):
        attrs = np.random.randn(50, 8).astype(np.float32)
        attrs[0:10, 0] = np.nan
        attrs[20:25, 1] = np.inf
        attrs[30:35, 2] = -np.inf
        attrs[:, 3] = 5.0
        attrs[:, 4] = np.nan
        normed = normalize_attrs(attrs)
        assert not np.isnan(normed).any(), "Mixed NaN/Inf should be handled"
        assert not np.isinf(normed).any()

    def test_preserves_batch_dimension(self):
        attrs = np.random.randn(7, 35).astype(np.float32)
        normed = normalize_attrs(attrs)
        assert normed.shape == (7, 35), f"Shape changed: {normed.shape}"

    def test_real_camels_attributes(self):
        try:
            import pickle
            _PROJECT = Path(__file__).resolve().parent.parent
            data_path = _PROJECT.parent.parent / "data" / "camels_dataset"
            with open(data_path, "rb") as f:
                _, _, attributes = pickle.load(f)
            attrs = attributes.astype(np.float32)
            normed = normalize_attrs(attrs)
            assert not np.isnan(normed).any(), f"Real CAMELS attrs have NaN after normalization: {np.isnan(normed).sum()}"
            assert not np.isinf(normed).any(), f"Real CAMELS attrs have Inf after normalization: {np.isinf(normed).sum()}"
        except FileNotFoundError:
            pytest.skip("CAMELS data not available")
