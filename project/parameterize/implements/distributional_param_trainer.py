"""Distributional paper trainer with KL regularization."""

from __future__ import annotations

from typing import Any

import torch

from .my_trainer import MyTrainer
from .param_models import DistributionalParamModel


class DistributionalParamTrainer(MyTrainer):
    """Paper trainer that adds KL regularization to the hydro loss."""

    def _distribution_config(self) -> dict[str, Any]:
        cfg = self.config.setdefault("distribution", {})
        cfg.setdefault("beta_kl", 1e-3)
        cfg.setdefault("kl_warmup_epochs", 10)
        return cfg

    def _beta_effective(self, epoch: int) -> float:
        distribution_cfg = self._distribution_config()
        beta_kl = float(distribution_cfg["beta_kl"])
        warmup_epochs = int(distribution_cfg["kl_warmup_epochs"])
        if warmup_epochs <= 0:
            return beta_kl
        return beta_kl * min(1.0, float(epoch) / float(warmup_epochs))

    @staticmethod
    def _latent_logstd_stats(latent_logstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return latent_logstd.mean(), latent_logstd.std(unbiased=False)

    def _compute_step_metrics(
        self,
        dataset_sample: dict[str, torch.Tensor],
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        _, y_pred, y_obs = self._forward_train_batch(dataset_sample)
        loss_hydro = self.loss_func(y_pred=y_pred, y_obs=y_obs)
        param_model = self.model.nn_model
        if not isinstance(param_model, DistributionalParamModel):
            raise TypeError("DistributionalParamTrainer requires DistributionalParamModel.")

        beta_effective = self._beta_effective(epoch)
        loss_kl = param_model.kl_divergence(dataset_sample["xc_nn_norm"])
        _, latent_logstd, _ = param_model._distribution_stats(dataset_sample["xc_nn_norm"])
        latent_logstd_mean, latent_logstd_std = self._latent_logstd_stats(latent_logstd)
        loss_total = loss_hydro + beta_effective * loss_kl
        return {
            "loss_total": loss_total,
            "loss_hydro": loss_hydro.detach(),
            "loss_kl": loss_kl.detach(),
            "beta_effective": torch.as_tensor(beta_effective, device=loss_hydro.device),
            "latent_logstd_mean": latent_logstd_mean.detach(),
            "latent_logstd_std": latent_logstd_std.detach(),
        }
