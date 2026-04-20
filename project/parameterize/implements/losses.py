"""Parameterize-local custom losses aligned to dMG batch-style criteria."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

from dmg.models.criterion.base import BaseCriterion


class _BatchStdScaledLoss(BaseCriterion):
    """Official batch-NSE style loss based on per-basin residual scaling."""

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: Union[torch.Tensor, float],
    ) -> None:
        super().__init__(config, device)
        self.config = config
        self.device = device
        self.eps = float(kwargs.get("eps", config.get("eps", 0.1)))

        try:
            y_obs = kwargs["y_obs"]
        except KeyError as e:
            raise KeyError("'y_obs' is not provided in kwargs") from e

        self.std = np.nanstd(self._reference_series(y_obs), axis=0)

    @staticmethod
    def _extract_target_array(y_obs: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(y_obs, torch.Tensor):
            target = y_obs.detach().cpu().numpy()
        else:
            target = np.asarray(y_obs)

        if target.ndim == 3:
            return target[:, :, 0]
        if target.ndim == 2:
            return target
        raise ValueError(f"Expected target with 2 or 3 dimensions, got {target.ndim}.")

    def _reference_series(self, y_obs: torch.Tensor | np.ndarray) -> np.ndarray:
        return self._extract_target_array(y_obs)

    def _transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values

    @staticmethod
    def _resolve_sample_ids(sample_ids: Any) -> np.ndarray:
        if isinstance(sample_ids, torch.Tensor):
            return sample_ids.detach().cpu().numpy().astype(int)
        return np.asarray(sample_ids).astype(int)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        prediction, target = self._format(y_pred, y_obs)

        try:
            sample_ids = self._resolve_sample_ids(kwargs["sample_ids"])
        except KeyError as e:
            raise KeyError("'sample_ids' is not provided in kwargs") from e

        if len(target) > 0:
            prediction = self._transform_tensor(prediction)
            target = self._transform_tensor(target)
            n_timesteps = target.shape[0]
            std_batch = torch.tensor(
                np.tile(self.std[sample_ids].T, (n_timesteps, 1)),
                dtype=prediction.dtype,
                requires_grad=False,
                device=self.device,
            )

            mask = ~torch.isnan(target)
            p_sub = prediction[mask]
            t_sub = target[mask]
            std_sub = std_batch[mask]

            sq_res = (p_sub - t_sub) ** 2
            norm_res = sq_res / (std_sub + self.eps) ** 2
            loss = torch.mean(norm_res)
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss


class NseBatchLoss(_BatchStdScaledLoss):
    """Normalized squared error (NSE) loss function."""

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: Union[torch.Tensor, float],
    ) -> None:
        super().__init__(config, device, **kwargs)
        self.name = "Batch NSE Loss"


class LogNseBatchLoss(_BatchStdScaledLoss):
    """Official batch-NSE style loss after log-transforming runoff."""

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: Any,
    ) -> None:
        self.log_eps = float(kwargs.get("log_eps", config.get("log_nse_eps", 1e-6)))
        super().__init__(config, device, **kwargs)
        self.name = "Log NSE Batch Loss"

    def _reference_series(self, y_obs: torch.Tensor | np.ndarray) -> np.ndarray:
        target = self._extract_target_array(y_obs)
        return np.log(np.clip(target, a_min=0.0, a_max=None) + self.log_eps)

    def _transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(values, min=0.0) + self.log_eps)


class HybridNseBatchLoss(BaseCriterion):
    """Weighted sum of official NSE-batch and log-NSE-batch losses."""

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, device)
        self.name = "Hybrid NSE Batch Loss"
        self.nse_weight = float(kwargs.get("nse_weight", 0.5))
        self.log_nse_weight = float(kwargs.get("log_nse_weight", 0.5))
        self.nse = NseBatchLoss(config, device=device, **kwargs)
        self.log_nse = LogNseBatchLoss(config, device=device, **kwargs)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        nse_loss = self.nse(y_pred=y_pred, y_obs=y_obs, **kwargs)
        log_nse_loss = self.log_nse(y_pred=y_pred, y_obs=y_obs, **kwargs)
        return self.nse_weight * nse_loss + self.log_nse_weight * log_nse_loss
