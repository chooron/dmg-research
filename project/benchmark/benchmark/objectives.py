"""Differentiable calibration objectives."""

from __future__ import annotations

import torch


def align_target(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    observed = target[..., 0] if target.dim() == 3 else target
    return observed[-prediction.shape[0] :, :]


def nse_per_start(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    observed = align_target(prediction, target)
    if observed.shape[1] == 1 and prediction.shape[1] > 1:
        observed = observed.expand(-1, prediction.shape[1])

    mask = torch.isfinite(prediction) & torch.isfinite(observed)
    pred = torch.where(mask, prediction, torch.nan)
    obs = torch.where(mask, observed, torch.nan)

    obs_mean = torch.nanmean(obs, dim=0, keepdim=True)
    numerator = torch.nansum((pred - obs) ** 2, dim=0)
    denominator = torch.nansum((obs - obs_mean) ** 2, dim=0).clamp_min(eps)
    return 1.0 - numerator / denominator


def kge_per_start(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 0.1,
) -> torch.Tensor:
    """KGE computed independently for each multi-start member.

    Parameters
    ----------
    prediction : Tensor [T, num_starts]
    target     : Tensor [T, 1, 1] or [T, num_starts] (or broadcastable)
    eps        : stability constant

    Returns
    -------
    Tensor [num_starts]  — KGE per start (NaN for degenerate starts)
    """
    observed = align_target(prediction, target)  # [T, num_starts or 1]
    if observed.shape[1] == 1 and prediction.shape[1] > 1:
        observed = observed.expand(-1, prediction.shape[1])

    T, S = prediction.shape
    kge_scores = torch.full((S,), float("nan"), dtype=prediction.dtype, device=prediction.device)

    for s in range(S):
        p = prediction[:, s]
        o = observed[:, s]
        mask = torch.isfinite(p) & torch.isfinite(o)
        if mask.sum() < 2:
            continue
        p_m = p[mask]
        o_m = o[mask]
        mean_p = p_m.mean()
        mean_o = o_m.mean()
        std_p = p_m.std()
        std_o = o_m.std()
        num = ((p_m - mean_p) * (o_m - mean_o)).sum()
        den = (
            torch.sqrt(((p_m - mean_p) ** 2).sum())
            * torch.sqrt(((o_m - mean_o) ** 2).sum())
        )
        r = num / (den + eps)
        beta = mean_p / (mean_o + eps)
        gamma = std_p / (std_o + eps)
        kge_scores[s] = 1.0 - torch.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)

    return kge_scores


def objective_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    log_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute scalar loss for independent calibration (multi-start).

    Parameters
    ----------
    prediction  : Tensor [T, num_starts]
    target      : Tensor [T, 1, 1] or [T, num_starts]
    objective   : 'NSE', 'LOG_NSE', 'KGE', or 'KGE_LOG'
    log_epsilon : epsilon for log transform (for LOG_NSE / KGE_LOG)

    Returns
    -------
    Tensor (scalar) — negative mean score across starts (to minimize)
    """
    objective_key = objective.lower().replace("-", "_")

    if objective_key in {"nse", "high_nse"}:
        score = nse_per_start(prediction, target)
    elif objective_key in {"lognse", "log_nse", "log_transformed_nse"}:
        pred = torch.log(torch.clamp(prediction, min=0.0) + log_epsilon)
        observed = align_target(prediction, target)
        observed = torch.log(torch.clamp(observed, min=0.0) + log_epsilon)
        score = nse_per_start(pred, observed)
    elif objective_key in {"kge"}:
        score = kge_per_start(prediction, target)
    elif objective_key in {"kge_log", "kge_log_transform", "kgelog"}:
        pred_log = torch.log(torch.clamp(prediction, min=0.0) + log_epsilon)
        observed = align_target(prediction, target)
        if observed.shape[1] == 1 and prediction.shape[1] > 1:
            observed = observed.expand(-1, prediction.shape[1])
        obs_log = torch.log(torch.clamp(observed, min=0.0) + log_epsilon)
        # build a fake "prediction/target" pair reusing kge_per_start
        score = kge_per_start(pred_log, obs_log.unsqueeze(-1).transpose(0, 1).transpose(0, 1))
    else:
        raise ValueError(
            f"Unsupported objective: '{objective}'. "
            "Supported: NSE, LOG_NSE, KGE, KGE_LOG"
        )

    return -torch.nanmean(score)
