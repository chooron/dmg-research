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
