"""Self-contained loss functions for the dual-evidence benchmark.

Includes:
- KgeBatchLoss   : Global KGE across all basins (masked)
- KgeLogBatchLoss: Same but on log-transformed streamflow (replaces KgeInverseLoss)
- NseBatchLoss   : Std-scaled NSE across basins
- LogNseBatchLoss: Std-scaled log-NSE across basins
- build_loss()   : Registry factory
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# KGE-based losses
# ---------------------------------------------------------------------------

class KgeBatchLoss(nn.Module):
    """Batch KGE loss computed globally across all basins.

    Flattens all valid (pred, obs) pairs from the entire batch, then
    computes a single KGE value.  Gradient flows through all basins.

    Parameters
    ----------
    eps : float
        Small stability constant to avoid division by zero.  Default 0.1.
    """

    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 1 - KGE.

        Parameters
        ----------
        y_pred : Tensor, shape [T, B] or [T, B, 1]
        y_obs  : Tensor, shape [T, B] or [T, B, 1]

        Returns
        -------
        Tensor (scalar)
        """
        pred, obs = _prepare(y_pred, y_obs)
        return _kge_loss(pred, obs, self.eps)


class KgeLogBatchLoss(nn.Module):
    """Batch KGE loss on log-transformed streamflow.

    Applies ``log(Q + eps_basin)`` to both pred and obs before computing
    KGE, where ``eps_basin`` is per-basin (defaults to 1 % of mean train
    obs if not provided).

    Parameters
    ----------
    basin_eps : Tensor, shape [B] or None
        Per-basin epsilon values.  If None, ``global_eps`` is used.
    global_eps : float
        Fallback epsilon when ``basin_eps`` is None.  Default 1e-3.
    kge_eps : float
        KGE numerical stability constant.  Default 0.1.
    """

    def __init__(
        self,
        basin_eps: torch.Tensor | None = None,
        global_eps: float = 1e-3,
        kge_eps: float = 0.1,
    ) -> None:
        super().__init__()
        self.global_eps = global_eps
        self.kge_eps = kge_eps
        if basin_eps is not None:
            self.register_buffer("basin_eps", basin_eps.float())
        else:
            self.basin_eps = None

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 1 - KGE on log-transformed streamflow.

        Parameters
        ----------
        y_pred : Tensor, shape [T, B] or [T, B, 1]
        y_obs  : Tensor, shape [T, B] or [T, B, 1]

        Returns
        -------
        Tensor (scalar)
        """
        pred, obs = _prepare(y_pred, y_obs)  # [T, B]
        T, B = pred.shape

        if self.basin_eps is not None:
            eps = self.basin_eps.to(pred.device)  # [B]
            if eps.shape[0] != B:
                raise ValueError(
                    f"basin_eps has {eps.shape[0]} elements but batch has {B} basins."
                )
        else:
            eps = torch.full((B,), self.global_eps, dtype=pred.dtype, device=pred.device)

        # log-transform: clamp negatives to 0, then add per-basin eps
        pred_log = torch.log(torch.clamp(pred, min=0.0) + eps.unsqueeze(0))
        obs_log = torch.log(torch.clamp(obs, min=0.0) + eps.unsqueeze(0))

        return _kge_loss(pred_log, obs_log, self.kge_eps)

    @classmethod
    def from_obs(
        cls,
        y_obs: torch.Tensor,
        frac: float = 0.01,
        global_eps: float = 1e-3,
        kge_eps: float = 0.1,
    ) -> "KgeLogBatchLoss":
        """Build from training observations: eps = frac * mean(obs) per basin.

        Parameters
        ----------
        y_obs : Tensor, shape [T, B] or [T, B, 1]
        frac  : fraction of mean obs used as eps (default 0.01 = 1 %)
        """
        if y_obs.dim() == 3:
            y_obs = y_obs[..., 0]
        mean_obs = torch.nanmean(torch.clamp(y_obs, min=0.0), dim=0)  # [B]
        basin_eps = torch.clamp(mean_obs * frac, min=global_eps)
        return cls(basin_eps=basin_eps, global_eps=global_eps, kge_eps=kge_eps)


# ---------------------------------------------------------------------------
# NSE-based losses
# ---------------------------------------------------------------------------

class _BatchStdScaledLoss(nn.Module):
    """Base class for std-scaled NSE-style batch losses."""

    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def _transform(
        self,
        pred: torch.Tensor,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional transform before computing residuals (identity by default)."""
        return pred, obs

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> torch.Tensor:
        pred, obs = _prepare(y_pred, y_obs)  # [T, B]
        pred, obs = self._transform(pred, obs)

        mask = torch.isfinite(pred) & torch.isfinite(obs)
        T, B = pred.shape

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        n_valid = 0

        for b in range(B):
            m = mask[:, b]
            if m.sum() < 2:
                continue
            p_b = pred[m, b]
            o_b = obs[m, b]
            std_b = o_b.std().clamp_min(self.eps)
            res = (p_b - o_b) / std_b
            total_loss = total_loss + (res ** 2).mean()
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
        return total_loss / n_valid


class NseBatchLoss(_BatchStdScaledLoss):
    """Std-scaled NSE loss across basins."""

    pass  # identity transform


class LogNseBatchLoss(_BatchStdScaledLoss):
    """Std-scaled NSE loss on log-transformed streamflow."""

    def __init__(self, eps: float = 0.1, log_eps: float = 1e-6) -> None:
        super().__init__(eps)
        self.log_eps = log_eps

    def _transform(
        self,
        pred: torch.Tensor,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_log = torch.log(torch.clamp(pred, min=0.0) + self.log_eps)
        obs_log = torch.log(torch.clamp(obs, min=0.0) + self.log_eps)
        return pred_log, obs_log


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "KgeBatchLoss": KgeBatchLoss,
    "KgeLogBatchLoss": KgeLogBatchLoss,
    "NseBatchLoss": NseBatchLoss,
    "LogNseBatchLoss": LogNseBatchLoss,
}


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a loss by name.

    Parameters
    ----------
    name : str
        Key in the loss registry.
    **kwargs
        Forwarded to the loss constructor.

    Returns
    -------
    nn.Module
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {sorted(_LOSS_REGISTRY)}"
        )
    return _LOSS_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare(
    y_pred: torch.Tensor,
    y_obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squeeze trailing singleton dims and align lengths."""
    if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]
    if y_obs.dim() == 3 and y_obs.shape[-1] == 1:
        y_obs = y_obs[..., 0]
    # Align T dimension (prediction may be shorter due to warm-up)
    T = min(y_pred.shape[0], y_obs.shape[0])
    return y_pred[-T:], y_obs[-T:]


def _kge_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    eps: float = 0.1,
) -> torch.Tensor:
    """Compute 1 - KGE on already-prepared [T, B] tensors."""
    mask = torch.isfinite(pred) & torch.isfinite(obs)
    p_sub = pred[mask]
    o_sub = obs[mask]

    mean_p = p_sub.mean()
    mean_o = o_sub.mean()
    std_p = p_sub.std()
    std_o = o_sub.std()

    num = ((p_sub - mean_p) * (o_sub - mean_o)).sum()
    den = (
        torch.sqrt(((p_sub - mean_p) ** 2).sum())
        * torch.sqrt(((o_sub - mean_o) ** 2).sum())
    )
    r = num / (den + eps)
    beta = mean_p / (mean_o + eps)
    gamma = std_p / (std_o + eps)
    kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)
    return 1.0 - kge
