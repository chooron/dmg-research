from __future__ import annotations

from typing import Any

import numpy as np
import torch

from training.ic.gpu_kge import compute_kge_fp64_matrix_gpu


class KGEObjective:
    """Optimizer-neutral maximize KGE(Q) objective using target mm/day."""

    def __init__(self, *, min_samples: int = 30, invalid_fitness: float = -999.0):
        self.min_samples = int(min_samples)
        self.invalid_fitness = float(invalid_fitness)

    def evaluate(self, simulation: torch.Tensor, target_mm_day: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if simulation.ndim == 2:
            simulation = simulation.unsqueeze(0)
        if target_mm_day.ndim == 1:
            target_mm_day = target_mm_day.unsqueeze(0)
        if simulation.ndim != 3 or target_mm_day.ndim not in {2, 3}:
            raise ValueError("simulation must be [B,P,T] and target [B,T] or [B,P,T]")
        if target_mm_day.ndim == 2:
            target_mm_day = target_mm_day.unsqueeze(1).expand(-1, simulation.shape[1], -1)
        if simulation.shape != target_mm_day.shape:
            raise ValueError(f"simulation/target shape mismatch: {simulation.shape} vs {target_mm_day.shape}")
        sim64 = simulation.reshape(-1, simulation.shape[-1]).to(torch.float64)
        target64 = target_mm_day.reshape(-1, target_mm_day.shape[-1]).to(torch.float64)
        fitness = compute_kge_fp64_matrix_gpu(sim64, target64, min_samples=self.min_samples)
        mask = torch.isfinite(sim64) & torch.isfinite(target64) & (sim64 >= 0.0) & (target64 >= 0.0)
        valid_count = mask.sum(dim=1).reshape(simulation.shape[:2])
        valid = torch.isfinite(fitness).reshape(simulation.shape[:2]) & (fitness.reshape(simulation.shape[:2]) > self.invalid_fitness)
        return fitness.reshape(simulation.shape[:2]), {
            "valid": valid,
            "valid_count": valid_count,
            "maximize": torch.tensor(True, device=fitness.device),
            "metric_dtype": torch.tensor(64, device=fitness.device),
        }

    def evaluate_numpy(self, simulation: np.ndarray, target_mm_day: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        sim = torch.as_tensor(simulation)
        target = torch.as_tensor(target_mm_day)
        fitness, diagnostics = self.evaluate(sim, target)
        return fitness.cpu().numpy(), {key: value.detach().cpu().numpy() for key, value in diagnostics.items() if value.ndim > 0}
