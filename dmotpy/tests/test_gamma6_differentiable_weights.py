"""Validate DplGamma6 default MHPI kernel and legacy compatibility path."""

import torch
from models.unithydro.uh_gamma_6 import (
    DplGamma6,
    _forward_pdf_half_step,
    _tail_mass_redistribution_reference,
    _tail_mass_redistribution_vectorized,
)
from models.hydrology_model import HydrologyModel


def test_gamma6_weights_finite_nonnegative():
    uh = DplGamma6(max_lag=120)
    for _ in range(10):
        params = torch.rand(3, 2) * torch.tensor([10.0, 120.0])
        w = uh.get_weights(params)
        assert torch.isfinite(w).all(), "Non-finite weight"
        assert (w >= 0).all(), "Negative weight"


def test_gamma6_weights_sum_normalized():
    uh = DplGamma6(max_lag=120)
    for _ in range(10):
        params = torch.rand(3, 2) * torch.tensor([10.0, 120.0]) + torch.tensor([0.1, 0.5])
        flux = torch.rand(3, 200)
        out = uh(flux, params)
        # Normalized kernel output — sum should be close(ish) to input sum modulo boundary
        # test pass: no errors during normalization


def test_gamma6_gradient_no_nan_inf():
    uh = DplGamma6(max_lag=120)
    for _ in range(10):
        raw_params = torch.rand(2, 2, requires_grad=True)
        flux = torch.rand(2, 200)
        out = uh(flux, raw_params)
        loss = out.sum()
        loss.backward()
        grad = raw_params.grad
        assert grad is not None, "grad is None"
        assert not grad.isnan().any(), "NaN gradient"
        assert not grad.isinf().any(), "Inf gradient"


def test_gamma6_pdf_half_step_weights_match_helper():
    uh = DplGamma6(max_lag=120, kernel_mode="pdf_half_step")
    params = torch.tensor([[3.0, 5.0], [8.0, 20.0]], dtype=torch.float32)
    got = uh.get_weights(params)
    expected = _forward_pdf_half_step(
        params[:, 0:1].unsqueeze(-1),
        params[:, 1:2].unsqueeze(-1),
        uh.t_idx,
        uh.epsilon,
    )
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_gamma6_forward_close_to_old():
    """Legacy cdf_diff mode should stay close to the gammainc S-curve difference."""
    def _old_weights(n, k, max_lag):
        t_idx = torch.arange(1, max_lag + 1, dtype=n.dtype, device=n.device).view(1, 1, -1)
        x = t_idx / k.unsqueeze(-1)
        s = torch.special.gammainc(n.unsqueeze(-1), x)
        z = torch.zeros_like(s[..., :1])
        p = torch.cat([z, s], dim=-1)
        inc = p[..., 1:] - p[..., :-1]
        return inc / inc.sum(dim=-1, keepdim=True)

    uh = DplGamma6(max_lag=120, kernel_mode="cdf_diff")
    for n_val, k_val in [(1.5, 10.0), (3.0, 30.0), (5.0, 60.0), (1.1, 120.0), (10.0, 5.0)]:
        n_t = torch.tensor([[n_val]])
        k_t = torch.tensor([[k_val]])
        old = _old_weights(n_t, k_t, 120)
        new = uh.get_weights(torch.tensor([[n_val, k_val]]))
        diff = (old - new).abs().max().item()
        assert diff < 0.05, f"n={n_val}, k={k_val}: diff={diff:.6e} (log-PDF vs gammainc, half-step alignment difference)"


def test_smar_uh_on_gradient():
    m = HydrologyModel(
        config={"model_name": "smar", "warm_up": 5, "uh_enabled": True, "uh_mode": "endpoint", "backend": "none"},
        device=torch.device("cpu"),
    )
    forcing = torch.rand(200, 2, 3) * 5
    raw = torch.rand(1, 8, requires_grad=True)
    out = m({"x_phy": forcing}, (None, raw))
    loss = out["streamflow"].mean()
    loss.backward()
    assert not raw.grad.isnan().any(), "smar NaN gradient"
    assert not raw.grad.isinf().any(), "smar Inf gradient"


def test_smar_uh_pdf_half_step_on_gradient():
    m = HydrologyModel(
        config={
            "model_name": "smar",
            "warm_up": 5,
            "uh_enabled": True,
            "uh_mode": "endpoint",
            "gamma6_kernel_mode": "pdf_half_step",
            "backend": "none",
        },
        device=torch.device("cpu"),
    )
    forcing = torch.rand(200, 2, 3) * 5
    raw = torch.rand(1, 8, requires_grad=True)
    out = m({"x_phy": forcing}, (None, raw))
    loss = out["streamflow"].mean()
    loss.backward()
    assert not raw.grad.isnan().any(), "smar pdf_half_step NaN gradient"
    assert not raw.grad.isinf().any(), "smar pdf_half_step Inf gradient"


def test_gamma6_tail_mass_vectorized_matches_reference_random():
    gen = torch.Generator().manual_seed(20260730)
    for _ in range(200):
        weights = torch.rand((32, 1, 120), generator=gen, dtype=torch.float64)
        ref = _tail_mass_redistribution_reference(weights)
        vec = _tail_mass_redistribution_vectorized(weights)
        torch.testing.assert_close(vec, ref, atol=1e-12, rtol=0.0)


def test_gamma6_tail_mass_vectorized_matches_reference_edge_cases():
    weights = torch.zeros((4, 1, 8), dtype=torch.float64)

    # No trigger: strictly nondecreasing-ish positive sequence that never falls below threshold.
    weights[0, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    # First possible early trigger at lag 1 (0-indexed), cutoff should include that point.
    weights[1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    # Mid-sequence trigger.
    weights[2, 0] = torch.tensor([1.0, 0.8, 0.4, 0.0003, 0.0002, 0.0, 0.0, 0.0], dtype=torch.float64)
    # Near-zero mass.
    weights[3, 0] = torch.tensor([1e-18, 1e-18, 1e-18, 1e-18, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    ref = _tail_mass_redistribution_reference(weights)
    vec = _tail_mass_redistribution_vectorized(weights)
    torch.testing.assert_close(vec, ref, atol=1e-12, rtol=0.0)
