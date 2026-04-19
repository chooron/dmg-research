"""Parameterize-local custom losses."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dmg.models.criterion.base import BaseCriterion


class LogNseBatchLoss(BaseCriterion):
    """Batch loss based on NSE after log-transforming the runoff series."""

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, device)
        self.name = "Log NSE Batch Loss"
        self.eps = float(kwargs.get("eps", config.get("eps", 1e-6)))
        self.log_eps = float(kwargs.get("log_eps", config.get("log_nse_eps", 1e-6)))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        prediction, target = self._format(y_pred, y_obs)
        mask = ~torch.isnan(target)
        prediction = prediction[mask]
        target = target[mask]

        pred_log = torch.log(torch.clamp(prediction, min=0.0) + self.log_eps)
        target_log = torch.log(torch.clamp(target, min=0.0) + self.log_eps)

        target_mean = torch.mean(target_log)
        numerator = torch.sum((pred_log - target_log) ** 2)
        denominator = torch.sum((target_log - target_mean) ** 2)
        nse = 1.0 - numerator / (denominator + self.eps)
        return 1.0 - nse


class HybridNseBatchLoss(BaseCriterion):
    """Hybrid NSE loss with equal weight on NSE and log-NSE."""

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, device)
        self.name = "Hybrid NSE Batch Loss"
        self.eps = float(kwargs.get("eps", config.get("eps", 1e-6)))
        self.nse_weight = float(kwargs.get("nse_weight", 0.5))
        self.log_nse_weight = float(kwargs.get("log_nse_weight", 0.5))
        self.log_nse = LogNseBatchLoss(config, device=device, **kwargs)

    def _nse_loss(self, y_pred: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        prediction, target = self._format(y_pred, y_obs)
        mask = ~torch.isnan(target)
        prediction = prediction[mask]
        target = target[mask]

        target_mean = torch.mean(target)
        numerator = torch.sum((prediction - target) ** 2)
        denominator = torch.sum((target - target_mean) ** 2)
        nse = 1.0 - numerator / (denominator + self.eps)
        return 1.0 - nse

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        nse_loss = self._nse_loss(y_pred, y_obs)
        log_nse_loss = self.log_nse(y_pred=y_pred, y_obs=y_obs, **kwargs)
        return self.nse_weight * nse_loss + self.log_nse_weight * log_nse_loss
