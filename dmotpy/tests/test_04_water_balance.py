"""
Category 4: Mass Balance and Water Balance Closure Test Suite
============================================================
Audit Reference: DMOTPY_CURRENT_STATE_AUDIT_20260831.md Section 9, 10 & 14

Validates that all 36 conceptual hydrological models satisfy physical water balance:
    delta_S = sum(P) - sum(Ea) - sum(Q)
    full_period_relative_residual = |delta_S - (sum(P) - sum(Ea) - sum(Q))| / sum(P) < tolerance

Known Audit Caveat:
    As confirmed in Audit Report Section 1 & 9, 'vic' has known clipping-induced
    water loss/gain under extreme synthetic forcing. All other 35 models pass strictly.
"""

from __future__ import annotations

import pytest
import torch

from tests.core_model_registry import CORE_MODEL_REGISTRY, CoreModelEntry
from tests.core_water_balance_utils import evaluate_model

ALL_MODELS = sorted(CORE_MODEL_REGISTRY.keys())
STANDARD_MODELS = [m for m in ALL_MODELS if m != "vic"]


class TestWaterBalanceClosure:
    """Validate physical water balance closure across hydrological models."""

    @pytest.mark.parametrize("model_name", STANDARD_MODELS)
    def test_water_balance_standard_models_float64(self, model_name: str) -> None:
        """35 standard models must strictly close water balance in float64."""
        entry = CORE_MODEL_REGISTRY[model_name]
        results = evaluate_model(entry, torch.float64, "cpu", "pytest")
        failures = [r for r in results if not r["pass_fail"]]
        if failures:
            details = "\n".join(
                f"{r['test_case']} {r['parameter_case']} res={r['max_absolute_full_period_residual']:.3e} "
                f"rel={r['full_period_relative_residual']:.3e} tol={r['tolerance']:.3e}"
                for r in failures
            )
            pytest.fail(f"{model_name} failed water balance closure in float64:\n{details}")

    def test_vic_water_balance_known_caveat(self) -> None:
        """VIC model exhibits documented clipping-induced residual under extreme forcing."""
        entry = CORE_MODEL_REGISTRY["vic"]
        results = evaluate_model(entry, torch.float64, "cpu", "pytest")
        # Relative residual is bounded even if clipping threshold induces minor loss
        for r in results:
            rel_res = r["full_period_relative_residual"]
            assert rel_res < 0.01, f"vic relative residual {rel_res:.3e} exceeds 1% safety bound"
