"""Test: StaticFormulaRouter logits remain NaN/Inf free with missing attributes."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.static_formula_router import StaticFormulaRouter


def normalize_safe(attrs):
    attr = attrs.astype(np.float32).copy()
    n_cols = attr.shape[1]
    for j in range(n_cols):
        col = attr[:, j]
        col[np.isinf(col)] = np.nan
        mask = np.isnan(col)
        n_imp = int(mask.sum())
        if n_imp > 0 and n_imp < len(col):
            col[mask] = np.nanmedian(col)
        elif n_imp == len(col):
            col[:] = 0.0
        attr[:, j] = col
    a_min = attr.min(axis=0, keepdims=True)
    a_rng = np.maximum(attr.max(axis=0, keepdims=True) - a_min, 1e-8)
    result = (attr - a_min) / a_rng
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return result.astype(np.float32)


class TestRouterLogitsNoNaN:

    def test_logits_finite_with_normalized_attrs(self):
        router = StaticFormulaRouter(attr_dim=10)
        attrs = normalize_safe(np.random.randn(8, 10))
        attrs_t = torch.from_numpy(attrs)
        out = router(attrs_t)
        for node in ["recharge", "snow", "aet", "response"]:
            logits = out["logits"][node]
            assert not torch.isnan(logits).any(), f"{node} logits have NaN"
            assert not torch.isinf(logits).any(), f"{node} logits have Inf"

    def test_logits_finite_with_nan_before_normalization(self):
        router = StaticFormulaRouter(attr_dim=8)
        raw = np.random.randn(16, 8).astype(np.float32)
        raw[0:3, 1] = np.nan
        raw[5:7, 3] = np.inf
        raw[10:12, 5] = -np.inf
        attrs = normalize_safe(raw)
        attrs_t = torch.from_numpy(attrs)
        out = router(attrs_t)
        for node in ["recharge", "snow", "aet", "response"]:
            logits = out["logits"][node]
            assert not torch.isnan(logits).any(), f"{node} logits have NaN after NaN->median imputation"
            assert not torch.isinf(logits).any(), f"{node} logits have Inf"

    def test_selection_indices_valid(self):
        router = StaticFormulaRouter(attr_dim=10)
        attrs = normalize_safe(np.random.randn(8, 10))
        out = router(torch.from_numpy(attrs))
        for node in ["recharge", "snow", "aet", "response"]:
            sel = out["selected"][node]
            n_f = router.num_formulas[node]
            assert (sel >= 0).all(), f"{node} selected has negative indices"
            assert (sel < n_f).all(), f"{node} selected has index >= {n_f} (max {sel.max()})"

    def test_entropy_finite(self):
        router = StaticFormulaRouter(attr_dim=10)
        attrs = normalize_safe(np.random.randn(8, 10))
        out = router(torch.from_numpy(attrs))
        for node in ["recharge", "snow", "aet", "response"]:
            ent = out[f"entropy_{node}"]
            assert not torch.isnan(ent).any(), f"{node} entropy has NaN"
            assert not torch.isinf(ent).any(), f"{node} entropy has Inf"

    def test_logits_with_all_nan_column(self):
        router = StaticFormulaRouter(attr_dim=5)
        raw = np.random.randn(12, 5).astype(np.float32)
        raw[:, 2] = np.nan
        attrs = normalize_safe(raw)
        out = router(torch.from_numpy(attrs))
        for node in ["recharge", "snow", "aet", "response"]:
            logits = out["logits"][node]
            assert not torch.isnan(logits).any(), "All-NaN column should not cause NaN logits"

    def test_logits_with_constant_column(self):
        router = StaticFormulaRouter(attr_dim=5)
        raw = np.random.randn(8, 5).astype(np.float32)
        raw[:, 3] = 42.0
        attrs = normalize_safe(raw)
        out = router(torch.from_numpy(attrs))
        for node in ["recharge", "snow", "aet", "response"]:
            logits = out["logits"][node]
            assert not torch.isnan(logits).any(), "Constant column should not cause NaN logits"

    def test_weights_sum_to_one(self):
        router = StaticFormulaRouter(attr_dim=10)
        attrs = normalize_safe(np.random.randn(8, 10))
        out = router(torch.from_numpy(attrs))
        for node in ["recharge", "snow", "aet", "response"]:
            w = out["weights"][node]
            row_sums = w.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"{node} weights do not sum to 1"

    def test_real_camels_attrs_router_logits_finite(self):
        try:
            import pickle
            _PROJECT = Path(__file__).resolve().parent.parent
            data_path = _PROJECT.parent.parent / "data" / "camels_dataset"
            with open(data_path, "rb") as f:
                _, _, attributes = pickle.load(f)
            attrs = attributes[:20].astype(np.float32)
            attrs_norm = normalize_safe(attrs)
            router = StaticFormulaRouter(attr_dim=attrs_norm.shape[1])
            out = router(torch.from_numpy(attrs_norm))
            for node in ["recharge", "snow", "aet", "response"]:
                logits = out["logits"][node]
                assert not torch.isnan(logits).any(), f"{node} logits have NaN on real CAMELS data"
                assert not torch.isinf(logits).any(), f"{node} logits have Inf on real CAMELS data"
        except FileNotFoundError:
            pytest.skip("CAMELS data not available")
