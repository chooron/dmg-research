"""Focused invariants for the frozen XAJ structure-diagnosis kernels."""

from __future__ import annotations

import pytest
import torch
from models import (
    DE,
    DR,
    GE,
    GR,
    DELite,
    DRLite,
    GELite,
    GRLite,
    normalized_to_beta,
    normalized_to_gamma,
    normalized_to_tau0,
)
from models.structure_evaporation import _parallel_evaporation_step
from models.structure_response import _subsurface_response_step

DTYPES = (torch.float32, torch.float64)


def _evap_inputs(dtype):
    return (
        torch.tensor([0.0, 4.0, 12.0, 4.0], dtype=dtype),
        torch.tensor([0.0, 0.0, 20.0, 80.0], dtype=dtype),
        torch.tensor([0.0, 10.0, 40.0, 40.0], dtype=dtype),
        torch.full((4,), 80.0, dtype=dtype),
        torch.full((4,), 40.0, dtype=dtype),
    )


def _response_inputs(dtype):
    return (
        torch.tensor([0.0, 2.0, 8.0, 20.0], dtype=dtype),
        torch.tensor([0.0, 0.0, 10.0, 50.0], dtype=dtype),
        torch.full((4,), 10.0, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_evaporation_ladder_formula_boundaries_and_full_lite(dtype):
    inputs = _evap_inputs(dtype)
    gamma = normalized_to_gamma(torch.tensor([0.0, 0.5, 1.0, 0.5], dtype=dtype))
    de = DE(compile_step=False)(*inputs)
    ge = GE(compile_step=False)(*inputs, gamma)
    de_lite = DELite(compile_step=False)(*inputs)
    ge_lite = GELite(compile_step=False)(*inputs, gamma)

    assert torch.equal(de[0], de_lite[0]) and torch.equal(de[1], de_lite[1])
    assert torch.equal(ge[0], ge_lite[0]) and torch.equal(ge[1], ge_lite[1])
    assert torch.all(de[0] >= 0.0) and torch.all(de[1] >= 0.0)
    assert torch.all(de[0] <= inputs[1]) and torch.all(de[1] <= inputs[2])
    assert de[0][0] == 0.0 and de[1][0] == 0.0

    # The normalized midpoint is the exact D_E/G_E reduction point.
    gamma_one = normalized_to_gamma(torch.full((4,), 0.5, dtype=dtype))
    ge_one = GE(compile_step=False)(*inputs, gamma_one)
    assert torch.equal(ge_one[0], de[0])
    assert torch.equal(ge_one[1], de[1])

    # D_E's uncapped linear allocation is pooled linear moisture stress.
    wl, wd, lm, dm = inputs[1:]
    expected = (wl + wd) / (lm + dm)
    uncapped = (de[0] + de[1]) / inputs[0].clamp_min(1.0)
    assert torch.allclose(uncapped[1:], expected[1:], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_response_ladder_conservation_reduction_and_full_lite(dtype):
    r_ss, z, tau = _response_inputs(dtype)
    beta = normalized_to_beta(torch.tensor([0.5, 0.0, 0.5, 1.0], dtype=dtype))
    dr = DR(compile_step=False)(r_ss, z, tau)
    gr = GR(compile_step=False)(r_ss, z, tau, beta)
    assert torch.equal(dr[0], DRLite(compile_step=False)(r_ss, z, tau)[0])
    assert torch.equal(gr[0], GRLite(compile_step=False)(r_ss, z, tau, beta)[0])
    assert torch.all(dr[0] <= z + r_ss)
    assert torch.all(gr[0] <= z + r_ss)
    assert torch.all(dr[1] >= 0.0) and torch.all(gr[1] >= 0.0)
    assert torch.allclose(dr[1], z + r_ss - dr[0], atol=0.0, rtol=0.0)
    assert torch.allclose(gr[1], z + r_ss - gr[0], atol=2e-6, rtol=2e-6)

    beta_one = normalized_to_beta(torch.full((4,), 0.5, dtype=dtype))
    gr_one = GR(compile_step=False)(r_ss, z, tau, beta_one)
    assert torch.equal(gr_one[0], dr[0])
    assert torch.equal(gr_one[1], dr[1])


@pytest.mark.parametrize("dtype", DTYPES)
def test_compiled_kernels_match_eager_and_backward_is_finite(dtype):
    evap = _evap_inputs(dtype)
    gamma = normalized_to_gamma(torch.full((4,), 0.35, dtype=dtype)).requires_grad_()
    eager = GE(compile_step=False)(*evap, gamma)
    compiled = GE()(*evap, gamma)
    assert all(
        torch.allclose(a, b, atol=2e-6, rtol=2e-6) for a, b in zip(eager, compiled)
    )

    response = _response_inputs(dtype)
    beta = normalized_to_beta(torch.full((4,), 0.65, dtype=dtype)).requires_grad_()
    tau_grad = response[2].clone().requires_grad_()
    eager_r = GR(compile_step=False)(response[0], response[1], tau_grad, beta)
    compiled_r = GR()(response[0], response[1], tau_grad, beta)
    assert all(
        torch.allclose(a, b, atol=2e-6, rtol=2e-6) for a, b in zip(eager_r, compiled_r)
    )

    (eager[0].sum() + eager_r[0].sum()).backward()
    assert torch.isfinite(gamma.grad).all()
    assert torch.isfinite(beta.grad).all()
    assert torch.isfinite(tau_grad.grad).all()


def test_power_kernel_has_zero_output_without_nan_gradient():
    dtype = torch.float64
    wl = torch.tensor([0.0, 1e-12], dtype=dtype, requires_grad=True)
    wd = torch.tensor([0.0, 1e-12], dtype=dtype)
    gamma = torch.tensor([0.2, 0.2], dtype=dtype, requires_grad=True)
    out = _parallel_evaporation_step(
        torch.ones(2, dtype=dtype),
        wl,
        wd,
        torch.full((2,), 80.0, dtype=dtype),
        torch.full((2,), 40.0, dtype=dtype),
        gamma,
        1e-8,
    )
    out[0].sum().backward()
    assert out[0][0] == 0.0
    assert torch.isfinite(wl.grad).all() and torch.isfinite(gamma.grad).all()

    z = torch.tensor([0.0, 1e-12], dtype=dtype, requires_grad=True)
    beta = torch.tensor([0.5, 0.5], dtype=dtype, requires_grad=True)
    response = _subsurface_response_step(
        torch.zeros(2, dtype=dtype),
        z,
        torch.full((2,), 10.0, dtype=dtype),
        beta,
        torch.ones(2, dtype=dtype),
        1e-8,
    )
    response[0].sum().backward()
    assert response[0][0] == 0.0
    assert torch.isfinite(z.grad).all() and torch.isfinite(beta.grad).all()


def test_normalized_midpoints_and_tau_mapping():
    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    assert torch.allclose(
        normalized_to_gamma(x), torch.tensor([0.2, 1.0, 5.0], dtype=x.dtype)
    )
    assert torch.allclose(
        normalized_to_beta(x), torch.tensor([0.5, 1.0, 2.0], dtype=x.dtype)
    )
    assert torch.allclose(
        normalized_to_tau0(x),
        torch.tensor(
            [0.43429448190325187, 14.72854443778428, 499.49983316645478], dtype=x.dtype
        ),
        rtol=1e-9,
        atol=1e-9,
    )
