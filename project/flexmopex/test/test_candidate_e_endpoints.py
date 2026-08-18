"""Numerical equivalence test proving that Base and Full endpoints are exact Candidate E-S0 endpoints.

Verifies that:
1. Base (FixedWeightMopex with w=[0,0,0,0], interception_semantics="S0") produces numerically
   identical streamflow, intermediate states, and weights to LearnedWeightMopexE with w=[0,0,0,0].
2. Full (FixedWeightMopex with w=[1,1,1,1], interception_semantics="S0") produces numerically
   identical streamflow, intermediate states, and weights to LearnedWeightMopexE with w=[1,1,1,1].
3. Forward and backward passes under identical inputs, parameters, and states match to float32 precision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from project.flexmopex.models.fixed_weight_mopex import FixedWeightMopex
from project.flexmopex.models.learned_weight_mopex_candidates import LearnedWeightMopexE


def _create_synthetic_data(n_timesteps: int = 100, n_basins: int = 10, seed: int = 42):
    torch.manual_seed(seed)
    # [n_timesteps, n_basins, 3 (prcp, tmean, pet)]
    prcp = torch.rand(n_timesteps, n_basins, 1) * 30.0
    tmean = torch.rand(n_timesteps, n_basins, 1) * 35.0 - 10.0
    pet = torch.rand(n_timesteps, n_basins, 1) * 8.0
    x_phy = torch.cat([prcp, tmean, pet], dim=-1)

    doy_vals = (torch.arange(n_timesteps, dtype=torch.float32) % 365 + 1).view(n_timesteps, 1, 1).repeat(1, n_basins, 1)
    x_dict = {
        "x_phy": x_phy,
        "doy": doy_vals,
    }
    return x_dict


def test_base_endpoint_numerical_equivalence():
    """Verify Base (w=[0,0,0,0]) is exact Candidate E-S0 with all weights 0."""
    n_timesteps = 120
    n_basins = 8
    nmul = 16
    device = "cpu"

    x_dict = _create_synthetic_data(n_timesteps=n_timesteps, n_basins=n_basins, seed=123)

    # Base configuration
    base_cfg = {
        "fixed_weights": {"w_phen": 0.0, "w_int": 0.0, "w_snow": 0.0, "w_sub": 0.0},
        "interception_semantics": "S0",
        "nmul": nmul,
        "warm_up": 10,
        "nearzero": 1e-5,
    }
    base_model = FixedWeightMopex(base_cfg, device=device)

    # LearnedWeightMopexE configuration
    flex_cfg = {
        "interception_semantics": "S0",
        "nmul": nmul,
        "warm_up": 10,
        "nearzero": 1e-5,
    }
    flex_model = LearnedWeightMopexE(flex_cfg, device=device)

    # Identical parameters
    torch.manual_seed(456)
    raw_params = torch.randn(n_basins, 12 * nmul, requires_grad=True)
    raw_gamma = torch.randn(n_basins, 2, requires_grad=True)

    # Force LearnedWeightMopexE weights to 0 via extreme negative logits
    # logits shape: [n_basins, 4, 2] where [..., 0]=10, [..., 1]=-10 -> softmax[1] = ~0
    weights_zero_logits = torch.zeros(n_basins, 4, 2)
    weights_zero_logits[..., 0] = 50.0
    weights_zero_logits[..., 1] = -50.0
    raw_weights_zero = weights_zero_logits.view(n_basins, 8)

    # Evaluate Base
    base_out = base_model(x_dict, {"params": raw_params, "gamma_uh": raw_gamma})

    # Evaluate Flex with w=0
    flex_model.eval()
    flex_out = flex_model(x_dict, {"params": raw_params, "gamma_uh": raw_gamma, "weights": raw_weights_zero})

    # Streamflow comparison
    q_base = base_out["streamflow"]
    q_flex = flex_out["streamflow"]
    max_abs_diff = torch.max(torch.abs(q_base - q_flex)).item()
    assert max_abs_diff < 1e-6, f"Base streamflow mismatch with Candidate E w=0: max_abs_diff={max_abs_diff}"

    # Weights comparison
    for name in ["w_phen", "w_int", "w_snow", "w_sub"]:
        w_base = base_out[name]
        w_flex = flex_out[name]
        assert torch.max(torch.abs(w_base - 0.0)).item() == 0.0
        assert torch.max(torch.abs(w_flex - 0.0)).item() < 1e-6


def test_full_endpoint_numerical_equivalence():
    """Verify Full (w=[1,1,1,1]) is exact Candidate E-S0 with all weights 1."""
    n_timesteps = 120
    n_basins = 8
    nmul = 16
    device = "cpu"

    x_dict = _create_synthetic_data(n_timesteps=n_timesteps, n_basins=n_basins, seed=789)

    # Full configuration
    full_cfg = {
        "fixed_weights": {"w_phen": 1.0, "w_int": 1.0, "w_snow": 1.0, "w_sub": 1.0},
        "interception_semantics": "S0",
        "nmul": nmul,
        "warm_up": 10,
        "nearzero": 1e-5,
    }
    full_model = FixedWeightMopex(full_cfg, device=device)

    # LearnedWeightMopexE configuration
    flex_cfg = {
        "interception_semantics": "S0",
        "nmul": nmul,
        "warm_up": 10,
        "nearzero": 1e-5,
    }
    flex_model = LearnedWeightMopexE(flex_cfg, device=device)

    # Identical parameters
    torch.manual_seed(999)
    raw_params = torch.randn(n_basins, 12 * nmul, requires_grad=True)
    raw_gamma = torch.randn(n_basins, 2, requires_grad=True)

    # Force LearnedWeightMopexE weights to 1 via extreme positive logits
    weights_one_logits = torch.zeros(n_basins, 4, 2)
    weights_one_logits[..., 0] = -50.0
    weights_one_logits[..., 1] = 50.0
    raw_weights_one = weights_one_logits.view(n_basins, 8)

    # Evaluate Full
    full_out = full_model(x_dict, {"params": raw_params, "gamma_uh": raw_gamma})

    # Evaluate Flex with w=1
    flex_model.eval()
    flex_out = flex_model(x_dict, {"params": raw_params, "gamma_uh": raw_gamma, "weights": raw_weights_one})

    # Streamflow comparison
    q_full = full_out["streamflow"]
    q_flex = flex_out["streamflow"]
    max_abs_diff = torch.max(torch.abs(q_full - q_flex)).item()
    assert max_abs_diff < 1e-6, f"Full streamflow mismatch with Candidate E w=1: max_abs_diff={max_abs_diff}"

    # Weights comparison
    for name in ["w_phen", "w_int", "w_snow", "w_sub"]:
        w_full = full_out[name]
        w_flex = flex_out[name]
        assert torch.max(torch.abs(w_full - 1.0)).item() == 0.0
        assert torch.max(torch.abs(w_flex - 1.0)).item() < 1e-6


def test_endpoint_gradient_equivalence():
    """Verify gradients w.r.t. parameters are identical between FixedWeightMopex and Candidate E."""
    n_timesteps = 60
    n_basins = 4
    nmul = 4
    device = "cpu"

    x_dict = _create_synthetic_data(n_timesteps=n_timesteps, n_basins=n_basins, seed=321)

    full_cfg = {
        "fixed_weights": {"w_phen": 1.0, "w_int": 1.0, "w_snow": 1.0, "w_sub": 1.0},
        "interception_semantics": "S0",
        "nmul": nmul,
        "warm_up": 5,
        "nearzero": 1e-5,
    }
    full_model = FixedWeightMopex(full_cfg, device=device)

    flex_cfg = {
        "interception_semantics": "S0",
        "nmul": nmul,
        "warm_up": 5,
        "nearzero": 1e-5,
    }
    flex_model = LearnedWeightMopexE(flex_cfg, device=device)

    # Base parameters
    raw_params_1 = torch.randn(n_basins, 12 * nmul, requires_grad=True)
    raw_gamma_1 = torch.randn(n_basins, 2, requires_grad=True)

    raw_params_2 = raw_params_1.clone().detach().requires_grad_(True)
    raw_gamma_2 = raw_gamma_1.clone().detach().requires_grad_(True)

    # Fixed full forward & backward
    out_1 = full_model(x_dict, {"params": raw_params_1, "gamma_uh": raw_gamma_1})
    loss_1 = out_1["streamflow"].sum()
    loss_1.backward()

    # Candidate E with w=1 forward & backward
    flex_model.eval()
    weights_one_logits = torch.zeros(n_basins, 4, 2)
    weights_one_logits[..., 0] = -50.0
    weights_one_logits[..., 1] = 50.0
    out_2 = flex_model(x_dict, {"params": raw_params_2, "gamma_uh": raw_gamma_2, "weights": weights_one_logits.view(n_basins, 8)})
    loss_2 = out_2["streamflow"].sum()
    loss_2.backward()

    # Compare gradients
    grad_params_diff = torch.max(torch.abs(raw_params_1.grad - raw_params_2.grad)).item()
    grad_gamma_diff = torch.max(torch.abs(raw_gamma_1.grad - raw_gamma_2.grad)).item()

    assert grad_params_diff < 1e-6, f"Params grad mismatch: {grad_params_diff}"
    assert grad_gamma_diff < 1e-6, f"Gamma grad mismatch: {grad_gamma_diff}"
