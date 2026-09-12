"""Stochastic time-window loader and batch constructor for shared-dPL."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class TimeWindowConfig:
    total_days: int = 730
    warmup_days: int = 365
    scored_days: int = 365
    forcing_start: date = date(1987, 1, 1)
    calibration_start: date = date(1989, 1, 1)
    calibration_end: date = date(1998, 12, 31)

    def __post_init__(self) -> None:
        if self.total_days != self.warmup_days + self.scored_days:
            raise ValueError(f"total_days ({self.total_days}) must equal warmup_days ({self.warmup_days}) + scored_days ({self.scored_days})")


@dataclass
class TimeWindowBatch:
    """Container holding one mini-batch of stochastic basin-time windows."""
    forcing: Tensor
    target_full: Tensor
    target_scored: Tensor
    attributes: Tensor
    basin_ids: tuple[str, ...]
    start_indices: tuple[int, ...]
    epsilon: Tensor


class StochasticTimeWindowLoader:
    """Samples stochastic 730-day (365 warmup + 365 scored) windows for a batch of basins."""

    def __init__(
        self,
        basin_data: Mapping[str, Mapping[str, Any]],
        dates: Sequence[date],
        config: TimeWindowConfig | None = None,
        *,
        seed: int = 20260903,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.basin_data = basin_data
        self.dates = tuple(dates)
        self.config = config or TimeWindowConfig()
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        # Precompute valid random sampling range within calibration period:
        # Start index t must allow at least warmup days prior or be within calibration range
        # Window of length 730 days fits within the 1987-01-01 .. 1998-12-31 forcing window (4383 days)
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.calib_start_idx = self.date_to_idx.get(self.config.calibration_start, 731)
        self.calib_end_idx = self.date_to_idx.get(self.config.calibration_end, 4382)
        
        # Minimum start index is 0 (1987-01-01), maximum start index is calib_end_idx - total_days + 1
        self.min_start_idx = 0
        self.max_start_idx = max(0, self.calib_end_idx - self.config.total_days + 1)

    def sample_batch(self, basin_ids: Sequence[str]) -> TimeWindowBatch:
        """Sample a stochastic 730-day time window for each basin in the batch."""
        B = len(basin_ids)
        if B < 1:
            raise ValueError("basin_ids must not be empty")

        # Sample random start indices
        # Independent integer random sampling per batch item
        start_indices = [
            int(torch.randint(self.min_start_idx, self.max_start_idx + 1, (1,), generator=self.generator).item())
            for _ in range(B)
        ]

        forcing_slices = []
        target_full_slices = []
        target_scored_slices = []
        attr_slices = []
        epsilons = []

        for b_idx, b_id in enumerate(basin_ids):
            b_info = self.basin_data[b_id]
            s_idx = start_indices[b_idx]
            e_idx = s_idx + self.config.total_days
            scored_s_idx = s_idx + self.config.warmup_days

            # Forcing [730, 3] (ppt, pet, temp)
            ppt = b_info["ppt"][s_idx:e_idx]
            pet = b_info["pet"][s_idx:e_idx]
            temp = b_info["temp"][s_idx:e_idx]
            f_slice = np.stack([ppt, pet, temp], axis=-1)
            forcing_slices.append(f_slice)

            # Target [730] and [365]
            q_obs_full = b_info["q_obs"][s_idx:e_idx]
            q_obs_scored = b_info["q_obs"][scored_s_idx:e_idx]
            target_full_slices.append(q_obs_full)
            target_scored_slices.append(q_obs_scored)

            # Attributes [35]
            if "attributes" in b_info:
                attr_slices.append(np.asarray(b_info["attributes"], dtype=np.float64))
            else:
                # Mock / default 35-dim attributes
                attr_slices.append(np.zeros(35, dtype=np.float64))

            # Epsilon for KGECOMP: mean(Qobs_cal) / 100.0
            if "epsilon" in b_info:
                epsilons.append(float(b_info["epsilon"]))
            else:
                valid_q = q_obs_full[np.isfinite(q_obs_full)]
                eps = float(np.mean(valid_q) / 100.0) if len(valid_q) > 0 else 0.05
                epsilons.append(eps)

        forcing_tensor = torch.as_tensor(np.stack(forcing_slices, axis=0), device=self.device, dtype=self.dtype)
        target_full_tensor = torch.as_tensor(np.stack(target_full_slices, axis=0), device=self.device, dtype=self.dtype)
        target_scored_tensor = torch.as_tensor(np.stack(target_scored_slices, axis=0), device=self.device, dtype=self.dtype)
        attributes_tensor = torch.as_tensor(np.stack(attr_slices, axis=0), device=self.device, dtype=self.dtype)
        epsilon_tensor = torch.as_tensor(epsilons, device=self.device, dtype=self.dtype)

        return TimeWindowBatch(
            forcing=forcing_tensor,
            target_full=target_full_tensor,
            target_scored=target_scored_tensor,
            attributes=attributes_tensor,
            basin_ids=tuple(basin_ids),
            start_indices=tuple(start_indices),
            epsilon=epsilon_tensor,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "generator_state": self.generator.get_state(),
            "config": {
                "total_days": self.config.total_days,
                "warmup_days": self.config.warmup_days,
                "scored_days": self.config.scored_days,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.seed = int(state["seed"])
        gen_state = state["generator_state"]
        if isinstance(gen_state, torch.Tensor):
            gen_state = gen_state.cpu().to(torch.uint8)
        self.generator.set_state(gen_state)
