"""
Category 3: Flux Non-Negativity and Gradient Stability Test Suite
=================================================================
Audit Reference: DMOTPY_CURRENT_STATE_AUDIT_20260831.md Section 6, 7 & 9

Validates that:
1. All 36 models produce finite streamflow and finite gradients under representative forcing.
2. Hydrological flux equations satisfy physical non-negativity (flux >= 0).
3. Specific boundary singularities (e.g. baseflow_6 at zero storage, TCM k2=0) do not generate NaN/Inf.
"""

from __future__ import annotations

import pytest
import torch

from models.core import PARAM_INFO, STFN_INFO, STATE_INFO
from models.flux.baseflow import baseflow_6
from models.flux.interflow import interflow_2, interflow_3, interflow_10


class TestFluxAndGradientStability:
    """Validate numerical stability, flux non-negativity, and gradient finiteness."""

    @pytest.mark.parametrize("model_name", sorted(PARAM_INFO.keys()))
    def test_all_36_models_gradient_finiteness(self, model_name: str):
        """All 36 models must produce finite streamflow and finite gradients for all parameters."""
        step_fn = STFN_INFO[model_name]
        p_info = PARAM_INFO[model_name]

        # Use midpoint parameters
        params = torch.tensor(
            [0.5 * (b[0] + b[1]) for b in p_info.values()],
            dtype=torch.float64,
            requires_grad=True,
        )
        states = [
            torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
            for _ in range(STATE_INFO[model_name])
        ]

        p = torch.tensor([5.0], dtype=torch.float64)
        t = torch.tensor([10.0], dtype=torch.float64)
        pet = torch.tensor([2.0], dtype=torch.float64)

        kwargs = {}
        if model_name in {"mopex4", "mopex5"}:
            kwargs["doy"] = torch.tensor([150.0], dtype=torch.float64)
        elif model_name == "tcm":
            kwargs["mean_P"] = torch.tensor([3.0], dtype=torch.float64)

        outputs = step_fn(p, t, pet, *params, *states, **kwargs)
        q = outputs[0]

        assert torch.isfinite(q), f"{model_name}: streamflow is non-finite: {q}"
        assert q >= 0.0, f"{model_name}: streamflow is negative: {q}"

        q.backward()
        assert params.grad is not None, f"{model_name}: parameter gradients are None"
        assert torch.isfinite(params.grad).all(), f"{model_name}: parameter gradients contain NaN/Inf: {params.grad}"

    def test_baseflow_6_at_zero_storage_no_nan(self):
        """baseflow_6 must not produce NaN or infinite gradients at zero storage."""
        s = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        p1 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        p2 = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        nearzero = 1e-5

        out = baseflow_6(p1, p2, s, nearzero=nearzero)
        assert torch.isfinite(out).all()
        assert out >= 0.0

        out.backward()
        assert torch.isfinite(s.grad).all()
        assert torch.isfinite(p1.grad).all()
        assert torch.isfinite(p2.grad).all()

    def test_tcm_backward_finite_at_boundary(self):
        """TCM model backward pass must remain finite at boundary parameters."""
        step_fn = STFN_INFO["tcm"]
        p_info = PARAM_INFO["tcm"]

        # k2 set to near lower boundary
        param_vals = [0.5 * (b[0] + b[1]) for b in p_info.values()]
        param_vals[2] = 0.001  # k2 small
        params = torch.tensor(param_vals, dtype=torch.float64, requires_grad=True)
        states = [torch.tensor([1.0], dtype=torch.float64, requires_grad=True) for _ in range(STATE_INFO["tcm"])]

        p = torch.tensor([10.0], dtype=torch.float64)
        t = torch.tensor([15.0], dtype=torch.float64)
        pet = torch.tensor([3.0], dtype=torch.float64)
        mean_P = torch.tensor([3.0], dtype=torch.float64)

        out = step_fn(p, t, pet, *params, *states, mean_P=mean_P)
        q = out[0]
        q.backward()
        assert torch.isfinite(params.grad).all()

    def test_interflow_fluxes_non_negative(self):
        """Interflow flux functions must strictly enforce non-negative discharge."""
        s = torch.tensor([5.0], dtype=torch.float64)
        p1 = torch.tensor([0.1], dtype=torch.float64)
        p2 = torch.tensor([1.5], dtype=torch.float64)
        s_max = torch.tensor([10.0], dtype=torch.float64)

        f2 = interflow_2(p1, p2, s)
        f3 = interflow_3(p1, p2, s)
        f10 = interflow_10(p1, p2, s, s_max)

        assert f2 >= 0.0
        assert f3 >= 0.0
        assert f10 >= 0.0
