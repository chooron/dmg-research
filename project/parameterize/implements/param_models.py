"""Parameter models for the parameterize paper stack."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _StaticParamBase(nn.Module):
    def __init__(self, config: dict[str, Any], nx: int, ny: int) -> None:
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.hidden_size = int(config.get("hidden_size", 128))
        self.output_activation = str(config.get("output_activation", "sigmoid")).lower()
        self.static_pool = str(config.get("static_pool", "last")).lower()

    def _pool_static_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int | None]:
        if x.ndim == 2:
            if x.shape[-1] != self.nx:
                raise ValueError(
                    f"{type(self).__name__} expected {self.nx} input features, got {x.shape[-1]}."
                )
            return x, None
        if x.ndim != 3:
            raise ValueError(
                f"{type(self).__name__} expects a 2D or 3D tensor, got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.nx:
            raise ValueError(
                f"{type(self).__name__} expected {self.nx} input features, got {x.shape[-1]}."
            )
        nt = x.shape[0]
        pooled = x.mean(dim=0) if self.static_pool == "mean" else x[-1]
        return pooled, nt

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "softplus":
            return F.softplus(x)
        if self.output_activation in {"identity", "none"}:
            return x
        raise ValueError(f"Unsupported output activation '{self.output_activation}'.")

    def _repeat_over_time(self, output: torch.Tensor, nt: int | None) -> torch.Tensor:
        if nt is None:
            return output
        return output.unsqueeze(0).repeat(nt, 1, 1).contiguous()


class DeterministicParamModel(_StaticParamBase):
    def __init__(self, config: dict[str, Any], nx: int, ny: int) -> None:
        super().__init__(config, nx, ny)
        self.name = "DeterministicParamModel"
        self.layers = nn.Sequential(
            nn.Linear(self.nx, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.ny),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, nt = self._pool_static_input(x)
        output = self._apply_output_activation(self.layers(pooled))
        if output.shape[-1] != self.ny:
            raise RuntimeError(
                f"DeterministicParamModel produced {output.shape[-1]} outputs, expected {self.ny}."
            )
        return self._repeat_over_time(output, nt)


class DistributionalParamModel(_StaticParamBase):
    def __init__(self, config: dict[str, Any], nx: int, ny: int) -> None:
        super().__init__(config, nx, ny)
        self.name = "DistributionalParamModel"
        distribution_cfg = config.get("distribution", {})
        self.logstd_min = float(distribution_cfg.get("logstd_min", -5.0))
        self.logstd_max = float(distribution_cfg.get("logstd_max", 2.0))
        self.backbone = nn.Sequential(
            nn.Linear(self.nx, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.latent_mu_head = nn.Linear(self.hidden_size, self.ny)
        self.latent_logstd_head = nn.Linear(self.hidden_size, self.ny)

    def _distribution_stats(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        pooled, nt = self._pool_static_input(x)
        hidden = self.backbone(pooled)
        latent_mu = self.latent_mu_head(hidden)
        latent_logstd = torch.clamp(
            self.latent_logstd_head(hidden),
            self.logstd_min,
            self.logstd_max,
        )
        return latent_mu, latent_logstd, nt

    def kl_divergence(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar mean KL( N(mu, sigma^2) || N(0, I) ) over the batch."""
        latent_mu, latent_logstd, _ = self._distribution_stats(x)
        kl = -0.5 * (
            1 + 2 * latent_logstd - latent_mu.pow(2) - (2 * latent_logstd).exp()
        )
        return kl.mean()

    def sample_parameters(self, x: torch.Tensor) -> torch.Tensor:
        parameters, _, _ = self.sample_parameters_with_stats(x)
        return parameters

    def sample_parameters_with_stats(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_mu, latent_logstd, nt = self._distribution_stats(x)
        eps = torch.randn_like(latent_mu)
        latent_sample = latent_mu + torch.exp(latent_logstd) * eps
        parameters = self._apply_output_activation(latent_sample)
        parameters = self._repeat_over_time(parameters, nt)

        latent_mu = self._repeat_over_time(latent_mu, nt)
        latent_logstd = self._repeat_over_time(latent_logstd, nt)
        return parameters, latent_mu, latent_logstd

    def estimate_parameter_moments(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
        samples = [self.sample_parameters(x) for _ in range(int(n_samples))]
        parameter_samples = torch.stack(samples, dim=0)
        return (
            parameter_samples.mean(dim=0),
            parameter_samples.std(dim=0, unbiased=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample_parameters(x)
