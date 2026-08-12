from __future__ import annotations

import torch


def plateau(history: torch.Tensor, *, improvement_window: int = 30, improvement_threshold: float = 1e-4,
            range_window: int = 10, range_threshold: float = 1e-3) -> torch.Tensor:
    """history [generation, unit] best-of-pop objective; returns per-unit certification."""
    if history.shape[0] < improvement_window:
        return torch.zeros(history.shape[1], dtype=torch.bool, device=history.device)
    improve = history[-1] - history[-improvement_window]
    recent_range = history[-range_window:].amax(0) - history[-range_window:].amin(0)
    return (improve < improvement_threshold) & (recent_range < range_threshold)


def cross_start_certified(best_by_start: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    values = torch.topk(best_by_start, min(2, best_by_start.shape[1]), dim=1).values
    return torch.ones(values.shape[0], dtype=torch.bool, device=values.device) if values.shape[1] == 1 else (values[:, 0] - values[:, 1] <= threshold)
