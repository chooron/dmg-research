"""Sacramento-specific reusable process formulas."""

from __future__ import annotations

import torch


def deficit_based_distribution(
    S1: torch.Tensor,
    S1max: torch.Tensor,
    S2: torch.Tensor,
    S2max: torch.Tensor,
    nearzero: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split inflow according to the two stores' relative deficits."""
    S1_safe = torch.minimum(S1, S1max)
    S2_safe = torch.minimum(S2, S2max)
    rd1 = (S1max - S1_safe) / (S1max + nearzero)
    rd2 = (S2max - S2_safe) / (S2max + nearzero)
    total_deficit = rd1 + rd2
    deficit_fraction = rd1 / (total_deficit + nearzero)
    capacity_fraction = S1max / (S1max + S2max + nearzero)
    f1 = torch.where(total_deficit > nearzero, deficit_fraction, capacity_fraction)
    f1 = torch.clamp(f1, 0.0, 1.0)
    return f1, 1.0 - f1
