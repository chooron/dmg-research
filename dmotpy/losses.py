"""Training losses with one explicit, auditable tensor contract.

The training contract deliberately separates the data-validity mask from
numeric finiteness.  Observation NaNs may be excluded by the data mask, but a
non-finite prediction is always a failed forward pass and is never filtered.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _prepare(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prediction and target must be torch.Tensor instances")

    if not torch.isfinite(prediction).all():
        raise FloatingPointError("prediction contains NaN or Inf; refusing to mask it")

    if prediction.ndim > 2 and prediction.shape[-1] == 1:
        prediction = prediction.squeeze(-1)
    if target.ndim > 2 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "Loss inputs must have shape [time, basin] (a trailing singleton "
            f"dimension is allowed), got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}."
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"Loss inputs must have identical shapes, got {tuple(prediction.shape)} "
            f"vs {tuple(target.shape)}."
        )

    if mask is None:
        # This is an observation/data-validity mask, not a prediction mask.
        mask = torch.ones_like(target, dtype=torch.bool)
    else:
        if mask.ndim > 2 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.shape != target.shape:
            raise ValueError(
                f"mask must match target shape {tuple(target.shape)}, got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
        mask = mask.to(device=target.device)

    # Missing observations are invalid data even if the caller supplied a
    # broad padding mask.  Prediction finiteness was checked above.
    effective_mask = mask & torch.isfinite(target)
    return prediction, target, effective_mask


def _resolve_reduction(reduction: str | None) -> str:
    value = "mean" if reduction is None else str(reduction).lower()
    if value not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")
    return value


def _columnwise_kge_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float,
    reduction: str = "mean",
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for idx in range(pred.shape[1]):
        valid = mask[:, idx]
        if int(valid.sum().item()) < 2:
            continue
        p = pred[:, idx][valid]
        o = obs[:, idx][valid]
        mean_p = p.mean()
        mean_o = o.mean()
        std_p = p.std()
        std_o = o.std()
        num = ((p - mean_p) * (o - mean_o)).sum()
        den = torch.sqrt(((p - mean_p) ** 2).sum()) * torch.sqrt(
            ((o - mean_o) ** 2).sum()
        )
        r = num / (den + eps)
        beta = mean_p / (mean_o + eps)
        gamma = std_p / (std_o + eps)
        kge = 1.0 - torch.sqrt(
            (r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2
        )
        values.append(1.0 - kge)

    if not values:
        raise ValueError("fewer than two valid observations for every basin")
    result = torch.stack(values)
    return result if reduction == "none" else result.mean()


class LossContract(nn.Module):
    """Base class documenting the common loss call protocol."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
        basin_ids: torch.Tensor | None = None,
        time_index: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class KgeLoss(LossContract):
    """KGE metric loss for validation or explicitly complete sequences."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
        basin_ids: torch.Tensor | None = None,
        time_index: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        del sample_ids, basin_ids, time_index
        prediction, target, mask = _prepare(prediction, target, mask)
        return _columnwise_kge_loss(
            prediction,
            target,
            mask,
            eps=self.eps,
            reduction=_resolve_reduction(reduction),
        )


class KgeBatchLoss(KgeLoss):
    """Compatibility name; semantics are now explicit-mask KGE."""


class KgeInverseLoss(LossContract):
    def __init__(self, stability_eps: float = 1e-5, floor_eps: float = 1e-3) -> None:
        super().__init__()
        self.stability_eps = float(stability_eps)
        self.floor_eps = float(floor_eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
        basin_ids: torch.Tensor | None = None,
        time_index: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        del sample_ids, basin_ids, time_index
        prediction, target, mask = _prepare(prediction, target, mask)
        valid_target = torch.where(mask, target, torch.zeros_like(target))
        count = mask.sum(dim=0).clamp_min(1)
        eps = (valid_target.sum(dim=0) / count) * 0.01
        eps = torch.clamp(eps, min=self.floor_eps)
        pred_inv = 1.0 / (torch.clamp(prediction, min=0.0) + eps.unsqueeze(0))
        target_inv = 1.0 / (torch.clamp(valid_target, min=0.0) + eps.unsqueeze(0))
        return _columnwise_kge_loss(
            pred_inv,
            target_inv,
            mask,
            eps=self.stability_eps,
            reduction=_resolve_reduction(reduction),
        )


class KgeLogLoss(LossContract):
    def __init__(self, stability_eps: float = 1e-5, floor_eps: float = 1e-3) -> None:
        super().__init__()
        self.stability_eps = float(stability_eps)
        self.floor_eps = float(floor_eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
        basin_ids: torch.Tensor | None = None,
        time_index: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        del sample_ids, basin_ids, time_index
        prediction, target, mask = _prepare(prediction, target, mask)
        eps = torch.as_tensor(self.floor_eps, dtype=prediction.dtype, device=prediction.device)
        pred_log = torch.log(torch.clamp(prediction, min=0.0) + eps)
        target_log = torch.log(torch.clamp(torch.where(mask, target, torch.zeros_like(target)), min=0.0) + eps)
        return _columnwise_kge_loss(
            pred_log,
            target_log,
            mask,
            eps=self.stability_eps,
            reduction=_resolve_reduction(reduction),
        )


class KgeInvQPerBasinLoss(KgeInverseLoss):
    pass


class KgeLogPerBasinLoss(KgeLogLoss):
    pass


class NseBatchLoss(LossContract):
    """Decomposable normalized squared-error training objective."""

    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
        basin_ids: torch.Tensor | None = None,
        time_index: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        del sample_ids, basin_ids, time_index
        prediction, target, mask = _prepare(prediction, target, mask)
        values: list[torch.Tensor] = []
        for idx in range(prediction.shape[1]):
            valid = mask[:, idx]
            if int(valid.sum().item()) < 2:
                continue
            p = prediction[:, idx][valid]
            o = target[:, idx][valid]
            std = o.std().clamp_min(self.eps)
            values.append((((p - o) / std) ** 2).mean())
        if not values:
            raise ValueError("no valid observations available for NSE loss")
        result = torch.stack(values)
        return result if _resolve_reduction(reduction) == "none" else result.mean()


class LogNseBatchLoss(NseBatchLoss):
    def __init__(self, eps: float = 0.1, log_eps: float = 1e-6) -> None:
        super().__init__(eps=eps)
        self.log_eps = float(log_eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
        basin_ids: torch.Tensor | None = None,
        time_index: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        del sample_ids, basin_ids, time_index
        prediction, target, mask = _prepare(prediction, target, mask)
        prediction = torch.log(torch.clamp(prediction, min=0.0) + self.log_eps)
        target = torch.log(torch.clamp(target, min=0.0) + self.log_eps)
        return super().forward(prediction, target, mask=mask, reduction=reduction)


class HybridNseBatchLoss(LossContract):
    def __init__(self, eps: float = 0.1, log_eps: float = 1e-6, weight: float = 0.5) -> None:
        super().__init__()
        self.nse = NseBatchLoss(eps=eps)
        self.log_nse = LogNseBatchLoss(eps=eps, log_eps=log_eps)
        self.weight = float(weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
        basin_ids: torch.Tensor | None = None,
        time_index: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        del sample_ids, basin_ids, time_index
        prediction, target, mask = _prepare(prediction, target, mask)
        nse = self.nse(prediction, target, mask=mask, reduction=reduction)
        log_nse = self.log_nse(prediction, target, mask=mask, reduction=reduction)
        return self.weight * nse + (1.0 - self.weight) * log_nse


_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "KgeLoss": KgeLoss,
    "KgeBatchLoss": KgeBatchLoss,
    "KgeInverseLoss": KgeInverseLoss,
    "KgeLogLoss": KgeLogLoss,
    "KgeInvQPerBasinLoss": KgeInvQPerBasinLoss,
    "KgeLogPerBasinLoss": KgeLogPerBasinLoss,
    "NseBatchLoss": NseBatchLoss,
    "LogNseBatchLoss": LogNseBatchLoss,
    "HybridNseBatchLoss": HybridNseBatchLoss,
}


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(_LOSS_REGISTRY)}")
    return _LOSS_REGISTRY[name](**kwargs)


def build_loss_from_config(config: dict[str, Any]) -> nn.Module:
    name = str(config.get("model") or config.get("name") or "NseBatchLoss")
    kwargs = {key: value for key, value in config.items() if key not in {"model", "name"}}
    return build_loss(name, **kwargs)


def full_sequence_kge(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Named validation metric; never used as a window-averaged train proxy."""
    return KgeLoss()(prediction, target, mask=mask, **kwargs)
