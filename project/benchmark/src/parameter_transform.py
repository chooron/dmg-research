from __future__ import annotations

import torch


class LatentBoundTransform:
    """Smooth unbounded latent <-> existing normalized physical-bound mapping."""

    def __init__(self, bounds: torch.Tensor, eps: float = 1e-7) -> None:
        self.bounds = bounds
        self.eps = float(eps)

    def latent_to_normalized(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(latent)

    def normalized_to_latent(self, normalized: torch.Tensor) -> torch.Tensor:
        p = normalized.clamp(self.eps, 1.0 - self.eps)
        return torch.logit(p)

    def latent_to_physical(self, latent: torch.Tensor) -> torch.Tensor:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        while lo.ndim < latent.ndim:
            lo, hi = lo.unsqueeze(0), hi.unsqueeze(0)
        return lo + self.latent_to_normalized(latent) * (hi - lo)

    def physical_to_latent(self, physical: torch.Tensor) -> torch.Tensor:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        while lo.ndim < physical.ndim:
            lo, hi = lo.unsqueeze(0), hi.unsqueeze(0)
        return self.normalized_to_latent((physical - lo) / (hi - lo))
