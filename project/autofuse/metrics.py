"""Metrics matching the paper's R implementation."""

from __future__ import annotations

import torch
from torch import Tensor


def _tensor(value: object) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.get_default_dtype())


def kge(sim: object, obs: object) -> Tensor:
    """Kling-Gupta efficiency using sample standard deviation (R ``sd``)."""
    simulated, observed = _tensor(sim), _tensor(obs)
    if simulated.shape != observed.shape:
        raise ValueError("sim and obs must have equal shapes")
    valid = torch.isfinite(simulated) & torch.isfinite(observed)
    n = int(valid.sum().item())
    if n < 2:
        return simulated.sum() * torch.nan
    sim_v, obs_v = simulated[valid], observed[valid]
    sim_mean, obs_mean = sim_v.mean(), obs_v.mean()
    sim_std, obs_std = sim_v.std(unbiased=True), obs_v.std(unbiased=True)
    covariance = ((sim_v - sim_mean) * (obs_v - obs_mean)).sum() / (n - 1)
    denominator = (sim_std * obs_std).clamp_min(torch.finfo(simulated.dtype).eps)
    correlation = covariance / denominator
    alpha = sim_std / obs_std.clamp_min(torch.finfo(simulated.dtype).eps)
    beta = sim_v.sum() / obs_v.sum().clamp_min(torch.finfo(simulated.dtype).eps)
    return 1.0 - torch.sqrt((correlation - 1.0) ** 2 + (beta - 1.0) ** 2 + (alpha - 1.0) ** 2)



def kgecomp_batched(sim: object, obs: object, *, epsilon: float | Tensor | None = None) -> Tensor:
    """Compute paper KGECOMP independently for each leading basin row."""
    simulated, observed = _tensor(sim), _tensor(obs)
    if simulated.ndim != 2 or observed.shape != simulated.shape:
        raise ValueError("batched sim and obs must both have shape [basin, time]")
    def _kge_rows(left: Tensor, right: Tensor) -> Tensor:
        valid = torch.isfinite(left) & torch.isfinite(right)
        count_raw = valid.sum(dim=-1)
        count = count_raw.clamp_min(2).to(dtype=left.dtype)
        left_safe = torch.where(valid, left, torch.zeros_like(left))
        right_safe = torch.where(valid, right, torch.zeros_like(right))
        left_mean = left_safe.sum(dim=-1) / count
        right_mean = right_safe.sum(dim=-1) / count
        left_centered = torch.where(valid, left - left_mean.unsqueeze(-1), torch.zeros_like(left))
        right_centered = torch.where(valid, right - right_mean.unsqueeze(-1), torch.zeros_like(right))
        eps_dtype = torch.finfo(left.dtype).eps
        left_std = torch.sqrt(((left_centered * left_centered).sum(dim=-1) / (count - 1.0)).clamp_min(eps_dtype))
        right_std = torch.sqrt(((right_centered * right_centered).sum(dim=-1) / (count - 1.0)).clamp_min(eps_dtype))
        covariance = (left_centered * right_centered).sum(dim=-1) / (count - 1.0)
        denominator = (left_std * right_std).clamp_min(eps_dtype)
        correlation = covariance / denominator
        alpha = left_std / right_std.clamp_min(eps_dtype)
        beta = left_safe.sum(dim=-1) / right_safe.sum(dim=-1).clamp_min(eps_dtype)
        kge_dist_sq = ((correlation - 1.0) ** 2 + (beta - 1.0) ** 2 + (alpha - 1.0) ** 2).clamp_min(eps_dtype)
        value = 1.0 - torch.sqrt(kge_dist_sq)
        nan_val = torch.full_like(value, float("nan"))
        return torch.where(count_raw >= 2, value, nan_val)
    if epsilon is None:
        valid_obs = torch.isfinite(observed)
        count = valid_obs.sum(dim=-1).clamp_min(1).to(dtype=observed.dtype)
        epsilon_value = torch.where(valid_obs, observed, torch.zeros_like(observed)).sum(dim=-1) / count / 100.0
    else:
        epsilon_value = _tensor(epsilon).to(device=observed.device, dtype=observed.dtype)
        if epsilon_value.ndim == 0:
            epsilon_value = epsilon_value.expand(simulated.shape[0])
    direct = _kge_rows(simulated, observed)
    inverse = _kge_rows(1.0 / (epsilon_value.unsqueeze(-1) + simulated), 1.0 / (epsilon_value.unsqueeze(-1) + observed))
    return (direct + inverse) / 2.0
def kgecomp(sim: object, obs: object, *, epsilon: float | Tensor | None = None) -> Tensor:
    """Paper metric ``[KGE(Q) + KGE(1/Q)] / 2``.

    For the inverse-flow term the paper uses ``epsilon = mean(Qobs_cal)/100``
    to avoid division by zero.  Callers evaluating a held-out period should
    pass the calibration-period epsilon explicitly.
    """
    simulated, observed = _tensor(sim), _tensor(obs)
    if epsilon is None:
        valid = torch.isfinite(observed)
        epsilon_value = observed[valid].mean() / 100.0 if bool(valid.any()) else observed.sum() * torch.nan
    else:
        epsilon_value = _tensor(epsilon).to(device=observed.device, dtype=observed.dtype)
    direct = kge(simulated, observed)
    inverse = kge(1.0 / (epsilon_value + simulated), 1.0 / (epsilon_value + observed))
    return (direct + inverse) / 2.0
