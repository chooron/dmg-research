"""Regression tests for the differentiable dPL KGE loss."""

from __future__ import annotations

import torch

from training.dpl.run_dpl_model import kge_per_basin


EPS = 1e-6
TIME = 96


def _observations(time: int = TIME) -> torch.Tensor:
    return torch.linspace(0.1, 4.0, time, dtype=torch.float32)


def _assert_finite_backward(qsim: torch.Tensor, qobs: torch.Tensor) -> None:
    qsim.retain_grad()
    kge = kge_per_basin(qsim.unsqueeze(0), qobs.unsqueeze(0), eps=EPS)
    loss = (1.0 - kge).mean()
    assert torch.isfinite(loss).all()
    loss.backward()
    assert qsim.grad is not None
    assert torch.isfinite(qsim.grad).all()
    assert not torch.isnan(qsim.grad).any()
    assert not torch.isinf(qsim.grad).any()


def test_all_zero_prediction_kge_backward_is_finite() -> None:
    torch.manual_seed(1001)
    qsim = torch.zeros(TIME, dtype=torch.float32, requires_grad=True)
    _assert_finite_backward(qsim, _observations())


def test_constant_nonzero_prediction_kge_backward_is_finite() -> None:
    torch.manual_seed(1002)
    qsim = torch.full((TIME,), 5.0, dtype=torch.float32, requires_grad=True)
    _assert_finite_backward(qsim, _observations())


def test_near_constant_prediction_kge_backward_is_finite() -> None:
    torch.manual_seed(1003)
    # Around zero, float32 retains the 1e-9 perturbation instead of rounding
    # it away as it would for a value near 5.0.
    qsim = (torch.linspace(0.0, 1.0, TIME, dtype=torch.float32) * 1e-9).requires_grad_()
    _assert_finite_backward(qsim, _observations())


def test_perfect_prediction_outer_sqrt_is_finite() -> None:
    torch.manual_seed(1004)
    qobs = _observations()
    qsim = qobs.clone().requires_grad_()
    _assert_finite_backward(qsim, qobs)


def _plain_kge_reference(qsim: torch.Tensor, qobs: torch.Tensor) -> torch.Tensor:
    mean_sim = qsim.mean(dim=1)
    mean_obs = qobs.mean(dim=1)
    dp = qsim - mean_sim[:, None]
    do = qobs - mean_obs[:, None]
    sim_ss = dp.square().sum(dim=1)
    obs_ss = do.square().sum(dim=1)
    r = (dp * do).sum(dim=1) / (torch.sqrt(sim_ss) * torch.sqrt(obs_ss))
    alpha = torch.sqrt(sim_ss / qsim.shape[1]) / torch.sqrt(obs_ss / qobs.shape[1])
    beta = mean_sim / mean_obs
    return 1.0 - torch.sqrt(
        (r - 1.0).square() + (alpha - 1.0).square() + (beta - 1.0).square()
    )


def test_normal_basin_matches_plain_kge_within_tolerance() -> None:
    torch.manual_seed(1005)
    qobs = torch.rand(3, TIME, dtype=torch.float32) * 5.0 + 0.1
    qsim = torch.rand(3, TIME, dtype=torch.float32) * 5.0 + 0.1
    qsim.requires_grad_()
    actual = kge_per_basin(qsim, qobs, eps=EPS)
    with torch.no_grad():
        reference = _plain_kge_reference(qsim.detach(), qobs)
    assert torch.allclose(actual.detach(), reference, atol=1e-5, rtol=0.0)
    actual.mean().backward()
    assert qsim.grad is not None
    assert torch.isfinite(qsim.grad).all()


def test_mixed_batch_kge_reduction_has_finite_upstream_gradients() -> None:
    torch.manual_seed(1006)
    qobs = _observations().expand(2, -1).clone()
    normal = torch.rand(TIME, dtype=torch.float32) * 5.0 + 0.1
    scales = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    base = torch.stack((torch.zeros(TIME, dtype=torch.float32), normal))
    qsim = scales[:, None] * base
    kge = kge_per_basin(qsim, qobs, eps=EPS)
    loss = (1.0 - kge).mean()
    assert torch.isfinite(loss).all()
    loss.backward()
    assert scales.grad is not None
    assert torch.isfinite(scales.grad).all()
    assert not torch.isnan(scales.grad).any()
    assert not torch.isinf(scales.grad).any()
