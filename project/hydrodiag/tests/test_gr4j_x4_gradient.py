"""Test that GR4J x4 parameter is differentiable and has nonzero gradient.

Verifies:
- x4.grad is not None after loss.backward()
- x4.grad is finite
- x4.grad is nonzero (x4 actually affects the output)
- unit_hydro ordinates are differentiable in x4
- batch independence of UH routing
"""

import pytest
import torch
from models.gr4j import GR4J
from models.unit_hydro import apply_unit_hydrograph_routing, compute_gr4j_uh_ordinates

BATCH = 3
TIME = 20


def make_synthetic_forcings(batch, time, device, dtype):
    torch.manual_seed(42)
    precip = torch.rand(batch, time, device=device, dtype=dtype) * 10.0
    pet = torch.rand(batch, time, device=device, dtype=dtype) * 5.0
    temp = torch.randn(batch, time, device=device, dtype=dtype) * 10.0
    return {"precip": precip, "pet": pet, "temp": temp}


def make_params(batch, device, dtype):
    """Create params with x4 requiring gradient."""
    x1 = torch.rand(batch, device=device, dtype=dtype) * 500.0 + 100.0
    x2 = torch.randn(batch, device=device, dtype=dtype) * 2.0
    x3 = torch.rand(batch, device=device, dtype=dtype) * 1000.0 + 100.0
    x4 = torch.rand(batch, device=device, dtype=dtype) * 5.0 + 1.5
    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
    }


def test_gr4j_x4_has_nonzero_gradient():
    """Verify x4 has non-None, finite, nonzero gradient through full forward."""
    device = torch.device("cpu")
    dtype = torch.float32

    model = GR4J()
    forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
    params = make_params(BATCH, device, dtype)

    params["x4"].requires_grad_(True)

    qsim, aux = model(forcings=forcings, params=params)
    loss = qsim.mean()
    loss.backward()

    assert params["x4"].grad is not None, "x4.grad is None"
    assert torch.isfinite(params["x4"].grad).all(), "x4.grad contains NaN/Inf"
    assert params["x4"].grad.abs().sum() > 0, (
        "x4.grad is zero (x4 doesn't affect output)"
    )

    print(f"x4 grad abs mean: {params['x4'].grad.abs().mean().item():.6f}")


def test_gr4j_unit_hydro_ordinates_are_differentiable():
    """Verify UH ordinate function is differentiable in x4.

    Since normalized UH ordinates always sum to 1, we use a position-weighted
    loss so that the shape (not just the sum) of the distribution matters.
    """
    device = torch.device("cpu")
    dtype = torch.float32

    max_len = 15
    x4 = torch.rand(BATCH, device=device, dtype=dtype) * 5.0 + 1.5
    x4.requires_grad_(True)

    uh1, uh2 = compute_gr4j_uh_ordinates(x4, max_len=max_len)

    # Position-weighted loss: earlier positions weight differently than later
    # This makes the "shape" of the UH matter, not just the sum
    pos_weights = torch.arange(max_len, device=device, dtype=dtype).view(1, -1)
    loss = (uh1 * pos_weights).sum() + (uh2 * pos_weights).sum()
    loss.backward()

    assert x4.grad is not None, "x4.grad is None after UH ordinate computation"
    assert torch.isfinite(x4.grad).all(), "x4.grad contains NaN/Inf"
    assert x4.grad.abs().sum() > 0, "x4.grad is zero"


def test_apply_unit_hydrograph_routing_differentiable():
    """Verify conv1d routing is differentiable."""
    device = torch.device("cpu")
    dtype = torch.float32

    x4 = torch.rand(BATCH, device=device, dtype=dtype) * 3.0 + 1.5
    x4.requires_grad_(True)

    flux = torch.rand(BATCH, TIME, device=device, dtype=dtype) * 5.0
    uh, _ = compute_gr4j_uh_ordinates(x4, max_len=15)
    routed = apply_unit_hydrograph_routing(flux, uh)
    loss = routed.mean()
    loss.backward()

    assert x4.grad is not None, "x4.grad is None after conv1d routing"
    assert torch.isfinite(x4.grad).all(), "x4.grad contains NaN/Inf"
    assert x4.grad.abs().sum() > 0, "x4.grad is zero"


def test_gr4j_unit_hydro_batch_independence():
    """Verify UH routing doesn't mix basins."""
    device = torch.device("cpu")
    dtype = torch.float32

    x4 = torch.tensor([2.0, 8.0, 4.0], device=device, dtype=dtype)
    flux = torch.rand(BATCH, TIME, device=device, dtype=dtype) * 5.0

    uh, _ = compute_gr4j_uh_ordinates(x4, max_len=15)
    routed_batch = apply_unit_hydrograph_routing(flux, uh)

    for b in range(BATCH):
        uh_single = uh[b : b + 1]
        flux_single = flux[b : b + 1]
        routed_single = apply_unit_hydrograph_routing(flux_single, uh_single)
        assert torch.allclose(
            routed_batch[b], routed_single[0], atol=1e-5, rtol=1e-4
        ), f"Basin {b} output differs between batch and single-basin runs"


def test_gr4j_x4_affects_timing():
    """Verify that changing x4 changes the shape/timing of the hydrograph."""
    device = torch.device("cpu")
    dtype = torch.float32

    model = GR4J()
    forcings = make_synthetic_forcings(1, TIME, device, dtype)

    params_small = {
        "x1": torch.tensor([300.0], device=device, dtype=dtype),
        "x2": torch.tensor([0.0], device=device, dtype=dtype),
        "x3": torch.tensor([500.0], device=device, dtype=dtype),
        "x4": torch.tensor([1.5], device=device, dtype=dtype),
    }
    params_large = {
        "x1": torch.tensor([300.0], device=device, dtype=dtype),
        "x2": torch.tensor([0.0], device=device, dtype=dtype),
        "x3": torch.tensor([500.0], device=device, dtype=dtype),
        "x4": torch.tensor([8.0], device=device, dtype=dtype),
    }

    qsim_small, _ = model(forcings=forcings, params=params_small)
    qsim_large, _ = model(forcings=forcings, params=params_large)

    # Different x4 should produce different outflow (unless no runoff generated)
    diff = torch.abs(qsim_small - qsim_large).sum()
    assert diff > 1e-6, f"x4 has no effect on output (diff={diff.item():.6e})"
