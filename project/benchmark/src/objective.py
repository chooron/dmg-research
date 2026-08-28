from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class KGEStats:
    count: torch.Tensor
    sum_pred: torch.Tensor
    sum_obs: torch.Tensor
    sum_pred2: torch.Tensor
    sum_obs2: torch.Tensor
    sum_cross: torch.Tensor


@dataclass(frozen=True)
class StreamingKGEState:
    """FP64 sufficient statistics matching :func:`streaming_kge`."""
    count: torch.Tensor
    sum_pred: torch.Tensor
    sum_obs: torch.Tensor
    sum_pred2: torch.Tensor
    sum_obs2: torch.Tensor
    sum_cross: torch.Tensor
    invalid_prediction: torch.Tensor


def initialize_streaming_kge(shape: tuple[int, ...], device: torch.device) -> StreamingKGEState:
    zeros = torch.zeros(shape, dtype=torch.float64, device=device)
    return StreamingKGEState(
        zeros, zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(),
        torch.zeros(shape, dtype=torch.bool, device=device),
    )


def update_streaming_kge_tensors(
    count: torch.Tensor,
    sum_pred: torch.Tensor,
    sum_obs: torch.Tensor,
    sum_pred2: torch.Tensor,
    sum_obs2: torch.Tensor,
    sum_cross: torch.Tensor,
    invalid_prediction: torch.Tensor,
    prediction: torch.Tensor,
    observation: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Update KGE statistics for one timestep without retaining predictions."""
    pred = prediction if prediction.dtype == torch.float64 else prediction.to(torch.float64)
    obs = observation if observation.dtype == torch.float64 else observation.to(torch.float64)
    obs = obs.unsqueeze(-1)
    finite_pred = torch.isfinite(pred)
    finite = finite_pred & torch.isfinite(obs)
    p = torch.where(finite, pred, torch.zeros_like(pred))
    o = torch.where(finite, obs, torch.zeros_like(pred))
    return (
        count + finite.to(torch.float64),
        sum_pred + p,
        sum_obs + o,
        sum_pred2 + p.square(),
        sum_obs2 + o.square(),
        sum_cross + p * o,
        invalid_prediction | ~finite_pred,
    )


def finalize_streaming_kge_tensors(
    count: torch.Tensor,
    sum_pred: torch.Tensor,
    sum_obs: torch.Tensor,
    sum_pred2: torch.Tensor,
    sum_obs2: torch.Tensor,
    sum_cross: torch.Tensor,
    invalid_prediction: torch.Tensor,
    *,
    eps: float = 0.1,
    invalid_penalty: float = -1_000_000.0,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Finalize the exact project KGE convention from sufficient statistics."""
    variance_floor = torch.as_tensor(1.0e-24, dtype=torch.float64, device=count.device)
    safe_n = count.clamp_min(1.0)
    centered_pred = (sum_pred2 - sum_pred.square() / safe_n).clamp_min(0.0)
    centered_obs = (sum_obs2 - sum_obs.square() / safe_n).clamp_min(0.0)
    denom_n = (count - 1.0).clamp_min(1.0)
    std_pred = torch.sqrt((centered_pred / denom_n).clamp_min(variance_floor))
    std_obs = torch.sqrt((centered_obs / denom_n).clamp_min(variance_floor))
    covariance_scale = torch.sqrt((centered_pred * centered_obs).clamp_min(variance_floor))
    r = (sum_cross - sum_pred * sum_obs / safe_n) / (covariance_scale + eps)
    beta = (sum_pred / safe_n) / (sum_obs / safe_n + eps)
    gamma = std_pred / (std_obs + eps)
    distance_sq = (r - 1.0).square() + (beta - 1.0).square() + (gamma - 1.0).square()
    score = 1.0 - torch.sqrt(distance_sq.clamp_min(variance_floor))
    invalid = (count < 2) | ~torch.isfinite(score) | invalid_prediction
    return torch.where(invalid, torch.full_like(score, invalid_penalty), score), invalid


def streaming_kge(prediction: torch.Tensor, observation: torch.Tensor, *, eps: float = 0.1,
                  invalid_penalty: float = -1_000_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Project-compatible KGE over time without retaining candidate time series.

    prediction is [T,B,S,P], observation is [T,B]. Returns score and invalid masks [B,S,P].
    """
    pred = prediction if prediction.dtype == torch.float64 else prediction.to(torch.float64)
    obs = observation if observation.dtype == torch.float64 else observation.to(torch.float64)
    obs = obs.unsqueeze(-1).unsqueeze(-1)
    finite = torch.isfinite(pred) & torch.isfinite(obs)
    p = torch.where(finite, pred, torch.zeros_like(pred))
    o = torch.where(finite, obs, torch.zeros_like(obs))
    n = finite.sum(dim=0).to(torch.float64)
    sp, so = p.sum(0), o.sum(0)
    sp2, so2, sc = (p * p).sum(0), (o * o).sum(0), (p * o).sum(0)
    variance_floor = torch.as_tensor(1.0e-24, dtype=torch.float64, device=pred.device)
    safe_n = n.clamp_min(1.0)
    centered_p = (sp2 - sp.square() / safe_n).clamp_min(0.0)
    centered_o = (so2 - so.square() / safe_n).clamp_min(0.0)
    denom_n = (n - 1.0).clamp_min(1.0)
    # Keep degenerate/near-degenerate basins differentiable.  The floor is
    # far below the IC metric's eps=0.1 and does not change finite scores at
    # ordinary precision, but prevents sqrt(0) from producing infinite
    # backward derivatives during long Flex recurrent graphs.
    std_p = torch.sqrt((centered_p / denom_n).clamp_min(variance_floor))
    std_o = torch.sqrt((centered_o / denom_n).clamp_min(variance_floor))
    covariance_scale = torch.sqrt((centered_p * centered_o).clamp_min(variance_floor))
    r = (sc - sp * so / safe_n) / (covariance_scale + eps)
    beta = (sp / safe_n) / (so / safe_n + eps)
    gamma = std_p / (std_o + eps)
    distance_sq = (r - 1.0).square() + (beta - 1.0).square() + (gamma - 1.0).square()
    score = 1.0 - torch.sqrt(distance_sq.clamp_min(variance_floor))
    invalid = (n < 2) | ~torch.isfinite(score) | ~torch.isfinite(pred).all(dim=0)
    return torch.where(invalid, torch.full_like(score, invalid_penalty), score), invalid


def full_kge_reference(prediction: torch.Tensor, observation: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """Reference implementation matching the existing `kge_per_start` convention."""
    return streaming_kge(prediction, observation, eps=eps)[0]
