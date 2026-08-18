"""GPU-batched FP64 KGE(Q), kept separate from the CPU reference function."""

from __future__ import annotations

import torch


def compute_kge_fp64_batch_gpu(
    sim_fp64: torch.Tensor,
    obs_fp64: torch.Tensor,
    min_samples: int = 30,
) -> torch.Tensor:
    """Compute one KGE(Q) per row of ``sim_fp64`` on the same device.

    This follows ``optimization.pycma_calibrator_v3.compute_kge_fp64``:
    finite/nonnegative mask, population mean and std with population
    normalization, Pearson correlation, alpha, beta, and the standard KGE
    combination.  The explicit reductions are intentionally visible so this
    function can be compared term-by-term with the CPU reference.
    """
    if sim_fp64.ndim != 2 or obs_fp64.ndim != 1:
        raise ValueError("Expected sim [population, time] and obs [time]")
    if sim_fp64.shape[1] != obs_fp64.shape[0]:
        raise ValueError("Simulation and observation time dimensions differ")
    if sim_fp64.dtype != torch.float64 or obs_fp64.dtype != torch.float64:
        raise TypeError("GPU KGE requires torch.float64 inputs")

    mask = (
        torch.isfinite(sim_fp64)
        & torch.isfinite(obs_fp64).unsqueeze(0)
        & (sim_fp64 >= 0.0)
        & (obs_fp64 >= 0.0).unsqueeze(0)
    )
    count = mask.sum(dim=1).to(torch.float64)
    safe_count = count.clamp_min(1.0)
    sim_valid = torch.where(mask, sim_fp64, torch.zeros_like(sim_fp64))
    obs_valid = torch.where(mask, obs_fp64.unsqueeze(0), torch.zeros_like(sim_fp64))

    sim_mean = sim_valid.sum(dim=1) / safe_count
    obs_mean = obs_valid.sum(dim=1) / safe_count
    d_sim = torch.where(
        mask, sim_fp64 - sim_mean.unsqueeze(1), torch.zeros_like(sim_fp64)
    )
    d_obs = torch.where(
        mask, obs_fp64.unsqueeze(0) - obs_mean.unsqueeze(1), torch.zeros_like(sim_fp64)
    )

    sim_var = d_sim.square().sum(dim=1) / safe_count
    obs_var = d_obs.square().sum(dim=1) / safe_count
    sim_std = torch.sqrt(sim_var)
    obs_std = torch.sqrt(obs_var)
    denom = torch.sqrt(d_sim.square().sum(dim=1)) * torch.sqrt(
        d_obs.square().sum(dim=1)
    )
    r = (d_sim * d_obs).sum(dim=1) / denom
    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean
    kge = 1.0 - torch.sqrt(
        (r - 1.0).square() + (alpha - 1.0).square() + (beta - 1.0).square()
    )

    invalid = (count < float(min_samples)) | (obs_std < 1e-10)
    return torch.where(invalid, torch.full_like(kge, -999.0), kge)


def compute_kge_fp64_matrix_gpu(
    sim_fp64: torch.Tensor,
    obs_fp64: torch.Tensor,
    min_samples: int = 30,
) -> torch.Tensor:
    """Compute one KGE per row when both simulation and observations are batched.

    ``sim_fp64`` and ``obs_fp64`` are ``[N, T]``.  This is the same reduction
    as :func:`compute_kge_fp64_batch_gpu`, but avoids a Python loop over
    basins when each basin has its own observation series.
    """
    if sim_fp64.ndim != 2 or obs_fp64.ndim != 2:
        raise ValueError("Expected sim and obs [population, time]")
    if sim_fp64.shape != obs_fp64.shape:
        raise ValueError("Simulation and observation shapes differ")
    if sim_fp64.dtype != torch.float64 or obs_fp64.dtype != torch.float64:
        raise TypeError("GPU KGE requires torch.float64 inputs")

    mask = (
        torch.isfinite(sim_fp64)
        & torch.isfinite(obs_fp64)
        & (sim_fp64 >= 0.0)
        & (obs_fp64 >= 0.0)
    )
    count = mask.sum(dim=1).to(torch.float64)
    safe_count = count.clamp_min(1.0)
    sim_valid = torch.where(mask, sim_fp64, torch.zeros_like(sim_fp64))
    obs_valid = torch.where(mask, obs_fp64, torch.zeros_like(obs_fp64))

    sim_mean = sim_valid.sum(dim=1) / safe_count
    obs_mean = obs_valid.sum(dim=1) / safe_count
    d_sim = torch.where(
        mask, sim_fp64 - sim_mean.unsqueeze(1), torch.zeros_like(sim_fp64)
    )
    d_obs = torch.where(
        mask, obs_fp64 - obs_mean.unsqueeze(1), torch.zeros_like(obs_fp64)
    )

    sim_var = d_sim.square().sum(dim=1) / safe_count
    obs_var = d_obs.square().sum(dim=1) / safe_count
    sim_std = torch.sqrt(sim_var)
    obs_std = torch.sqrt(obs_var)
    denom = torch.sqrt(d_sim.square().sum(dim=1)) * torch.sqrt(
        d_obs.square().sum(dim=1)
    )
    r = (d_sim * d_obs).sum(dim=1) / denom
    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean
    kge = 1.0 - torch.sqrt(
        (r - 1.0).square() + (alpha - 1.0).square() + (beta - 1.0).square()
    )

    invalid = (count < float(min_samples)) | (obs_std < 1e-10)
    return torch.where(invalid, torch.full_like(kge, -999.0), kge)
