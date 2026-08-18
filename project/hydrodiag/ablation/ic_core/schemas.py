from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PeriodSlice:
    name: str
    start: str
    end: str
    start_index: int
    end_index: int
    days: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "days": self.days,
        }


@dataclass(frozen=True)
class PeriodResolution:
    warmup: PeriodSlice
    train: PeriodSlice
    test: PeriodSlice
    train_forcing_start_index: int
    train_forcing_end_index: int
    test_forcing_start_index: int
    test_forcing_end_index: int
    test_warmup_days: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "warmup": self.warmup.as_dict(),
            "train": self.train.as_dict(),
            "test": self.test.as_dict(),
            "train_forcing": {
                "start_index": self.train_forcing_start_index,
                "end_index": self.train_forcing_end_index,
                "days": self.train_forcing_end_index - self.train_forcing_start_index,
            },
            "test_forcing": {
                "start_index": self.test_forcing_start_index,
                "end_index": self.test_forcing_end_index,
                "days": self.test_forcing_end_index - self.test_forcing_start_index,
                "preceding_warmup_days": self.test_warmup_days,
            },
            "train_target_days": self.train.days,
            "test_target_days": self.test.days,
        }


@dataclass(frozen=True)
class ICDataBundle:
    basin_ids: tuple[str, ...]
    source_indices: np.ndarray
    metadata_indices: np.ndarray
    dates: np.ndarray
    forcing: np.ndarray
    target_cfs: np.ndarray
    target_mm_day: np.ndarray
    valid_target_mask: np.ndarray
    raw_attributes: np.ndarray
    raw_area_km2: np.ndarray
    forcing_names: tuple[str, ...]
    target_unit_raw: str
    target_unit_ic: str
    area_field: str
    area_unit: str
    periods: PeriodResolution
    temp_mean_train: np.ndarray
    temp_std_train: np.ndarray
    source_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        n_basins = len(self.basin_ids)
        if n_basins != 531:
            raise ValueError(
                f"IC foundation requires exactly 531 basins, got {n_basins}"
            )
        if len(set(self.basin_ids)) != n_basins:
            raise ValueError("basin_ids must be unique")
        if self.source_indices.shape != (n_basins,) or self.metadata_indices.shape != (
            n_basins,
        ):
            raise ValueError(
                "source_indices and metadata_indices must have shape [531]"
            )
        if self.forcing.ndim != 3 or self.forcing.shape[0] != n_basins:
            raise ValueError(
                f"forcing must have shape [531,time,features], got {self.forcing.shape}"
            )
        if self.target_cfs.shape[:1] != (n_basins,) or self.target_cfs.ndim != 2:
            raise ValueError(
                f"target_cfs must have shape [531,time], got {self.target_cfs.shape}"
            )
        if self.target_mm_day.shape != self.target_cfs.shape:
            raise ValueError("target_mm_day must align with target_cfs")
        if self.valid_target_mask.shape != self.target_cfs.shape:
            raise ValueError("valid_target_mask must align with target_cfs")
        if self.raw_attributes.shape[0] != n_basins:
            raise ValueError("raw_attributes must align with basin_ids")
        if self.raw_area_km2.shape != (n_basins,):
            raise ValueError("raw_area_km2 must have shape [531]")
        if self.temp_mean_train.shape != (n_basins,) or self.temp_std_train.shape != (
            n_basins,
        ):
            raise ValueError("temperature statistics must have shape [531]")


@dataclass(frozen=True)
class CandidateEvaluation:
    fitness: np.ndarray
    valid: np.ndarray
    valid_count: np.ndarray
    candidate_evaluations: int
    split: str
    q_shape: tuple[int, ...]
    forcing_shape: tuple[int, ...]
    metric_dtype: str
    runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitness": self.fitness.tolist(),
            "valid": self.valid.tolist(),
            "valid_count": self.valid_count.tolist(),
            "candidate_evaluations": self.candidate_evaluations,
            "split": self.split,
            "q_shape": list(self.q_shape),
            "forcing_shape": list(self.forcing_shape),
            "metric_dtype": self.metric_dtype,
            "runtime_seconds": self.runtime_seconds,
        }


RESULT_FIELDS = (
    "run_id",
    "experiment_name",
    "basin_id",
    "model_key",
    "optimizer",
    "optimizer_seed",
    "start_id",
    "generation",
    "population",
    "candidate_evaluations_generation",
    "candidate_evaluations_cumulative",
    "best_fitness_generation",
    "best_fitness_so_far",
    "best_theta_normalized",
    "best_theta_physical",
    "status",
    "failure_reason",
    "runtime_seconds",
)
