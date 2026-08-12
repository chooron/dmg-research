from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from dmg.models.criterion.base import BaseCriterion


class NseL0BatchLoss(BaseCriterion):
    """NSE Loss with basin-mean expected-L0 regularization for BinaryWeightMopex.

    Minimizes:
        L = NSE_Loss + alpha * complexity_loss

    where:
        NSE_Loss = mean((pred - obs)^2 / (std_obs + eps)^2)
        p_nonzero_k = sigmoid(log_alpha_k - temperature * log(-gamma / zeta))
        complexity_per_basin = sum_k p_nonzero_k          (expected active count)
        complexity_loss = mean_over_basins(complexity_per_basin)

    Alpha direction is consistent with NseAicBatchLoss: larger alpha → sparser structure.
    Unlike AIC loss, process costs are uniform (no parameter-count weighting).

    Parameters
    ----------
    config
        Configuration dictionary. Reads: aic_alpha, eps, concrete_temperature,
        concrete_gamma, concrete_zeta.
    device
        Device string.
    **kwargs
        y_obs: Tensor [n_time, n_grid, 1] — full observations for std computation.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: Union[torch.Tensor, float],
    ) -> None:
        super().__init__(config, device)
        self.name = "NSE + L0 Batch Loss"
        self.config = config
        self.device = device

        y_obs = kwargs["y_obs"]
        self.std = np.nanstd(y_obs[:, :, 0].cpu().detach().numpy(), axis=0)
        self.eps = float(kwargs.get("eps", config.get("eps", 0.1)))
        self.aic_alpha = float(kwargs.get("aic_alpha", config.get("aic_alpha", 0.0)))

        self.temperature = float(config.get("concrete_temperature", 0.5))
        self.gamma = float(config.get("concrete_gamma", -0.1))
        self.zeta = float(config.get("concrete_zeta", 1.1))
        self._pnz_offset = self.temperature * math.log(-self.gamma / self.zeta)

        # Diagnostics accumulated during training (reset each epoch by trainer)
        self._diag: dict[str, list[float]] = {
            "mean_p_nonzero": [],
            "mean_active_count": [],
            "fraction_all_zero": [],
            "fraction_all_one": [],
            "loss_fit": [],
            "loss_complexity": [],
        }

    def reset_diagnostics(self) -> None:
        for v in self._diag.values():
            v.clear()

    def get_diagnostics(self) -> dict[str, float]:
        out = {}
        for k, v in self._diag.items():
            out[k] = float(np.mean(v)) if v else float("nan")
        return out

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        prediction, target = self._format(y_pred, y_obs)

        sample_ids = kwargs["sample_ids"]
        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.cpu().numpy().astype(int)
        else:
            sample_ids = np.asarray(sample_ids).astype(int)

        # ── 1. NSE fit loss ──────────────────────────────────────────────────
        if len(target) > 0:
            n_timesteps = target.shape[0]
            std_batch = torch.tensor(
                np.tile(self.std[sample_ids].T, (n_timesteps, 1)),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            mask = ~torch.isnan(target)
            p_sub = prediction[mask]
            t_sub = target[mask]
            std_sub = std_batch[mask]
            sq_res = (p_sub - t_sub) ** 2
            norm_res = sq_res / (std_sub + self.eps) ** 2
            loss_fit = torch.mean(norm_res)
        else:
            loss_fit = torch.tensor(0.0, device=self.device)

        # ── 2. Expected-L0 complexity loss ───────────────────────────────────
        # log_alpha is passed via the full model output dict under key "log_alpha"
        # shape: (n_grid, 4)
        log_alpha = kwargs.get("log_alpha", None)
        if log_alpha is None:
            # Fallback: try extracting from weights dict (not expected in normal use)
            raise ValueError(
                "NseL0BatchLoss requires 'log_alpha' in kwargs. "
                "Ensure BinaryWeightMopex returns 'log_alpha' and the handler passes it."
            )

        p_nz = torch.sigmoid(log_alpha - self._pnz_offset)  # (n_grid, 4)
        complexity_per_basin = p_nz.sum(dim=-1)              # (n_grid,)
        loss_complexity = complexity_per_basin.mean()         # scalar

        # ── 3. Combined loss ─────────────────────────────────────────────────
        final_loss = loss_fit + self.aic_alpha * loss_complexity

        # ── 4. Accumulate diagnostics (detached) ─────────────────────────────
        with torch.no_grad():
            p_nz_d = p_nz.detach()
            active = (p_nz_d >= 0.5).float()
            active_count = active.sum(dim=-1)  # (n_grid,)
            self._diag["mean_p_nonzero"].append(float(p_nz_d.mean()))
            self._diag["mean_active_count"].append(float(active_count.mean()))
            self._diag["fraction_all_zero"].append(
                float((active_count == 0).float().mean())
            )
            self._diag["fraction_all_one"].append(
                float((active_count == 4).float().mean())
            )
            self._diag["loss_fit"].append(float(loss_fit.detach()))
            self._diag["loss_complexity"].append(float(loss_complexity.detach()))

        return final_loss
