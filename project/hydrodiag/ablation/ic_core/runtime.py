from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch

from .model_adapter import ModelAdapter
from .objective_adapter import KGEObjective
from .parameter_adapter import normalized_to_physical, validate_parameter_shape
from .schemas import CandidateEvaluation, ICDataBundle


class ICObjectiveRuntime:
    """Evaluate independent basin/candidate batches without optimizer assumptions."""

    def __init__(
        self,
        bundle: ICDataBundle,
        config: dict[str, Any],
        model_key: str,
        *,
        model_variant: str | None = None,
    ):
        self.bundle = bundle
        self.config = config
        self.model_key = model_key
        if model_variant is None:
            model_variant = config.get("model_variant", "lite")
        self.model_variant = model_variant
        self.device = torch.device(config["device"])
        self.model_adapter = ModelAdapter(
            model_key,
            device=self.device,
            dtype=torch.float32,
            variant=model_variant,
        )
        self.objective = KGEObjective(min_samples=int(config.get("objective", {}).get("min_samples", 30)))
        self.basin_batch_size = int(config["batching"].get("basin_batch_size", 4))
        self.cache_device_data = bool(config.get("batching", {}).get("cache_device_data", False))
        self._device_split_cache: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        # Canonical CN forcing-derived quantity (R3 synthetic protocol):
        # when requested, the CN snow-cover threshold uses the mean annual
        # solid precipitation computed once from the FULL basin record, so
        # split/window evaluations reproduce the generating truth exactly.
        # Default (flag absent) keeps the historical per-sequence estimate.
        self._cn_psol_annual: np.ndarray | None = None
        if bool(config.get("canonical_cn_psol_annual", False)) and model_key.endswith("_CN"):
            from models.cemaneige import _estimate_psol_annual

            forcing_t = torch.from_numpy(self.bundle.forcing)
            precip = forcing_t[:, :, 0]
            temp = forcing_t[:, :, 1]
            with torch.no_grad():
                self._cn_psol_annual = _estimate_psol_annual(precip, temp).numpy().astype(
                    np.float32, copy=False
                )

    def _split_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, int]:
        if split == "train":
            p = self.bundle.periods
            return (
                self.bundle.forcing[:, p.train_forcing_start_index:p.train_forcing_end_index, :],
                self.bundle.target_mm_day[:, p.train.start_index:p.train.end_index + 1],
                p.warmup.days,
            )
        if split == "test":
            p = self.bundle.periods
            return (
                self.bundle.forcing[:, p.test_forcing_start_index:p.test_forcing_end_index, :],
                self.bundle.target_mm_day[:, p.test.start_index:p.test.end_index + 1],
                p.test_warmup_days,
            )
        raise ValueError(f"unknown split: {split}")

    def _normalize_candidates(self, theta_01: Any, basin_indices: Sequence[int] | None) -> tuple[torch.Tensor, np.ndarray]:
        theta = torch.as_tensor(theta_01, dtype=torch.float64)
        validate_parameter_shape(self.model_key, theta)
        if theta.ndim == 1:
            theta = theta.reshape(1, 1, -1)
            default_basins = np.asarray([0], dtype=np.int64)
        elif theta.ndim == 2:
            theta = theta.unsqueeze(0)
            default_basins = np.asarray([0], dtype=np.int64)
        elif theta.ndim == 3:
            default_basins = np.arange(theta.shape[0], dtype=np.int64)
        else:
            raise ValueError("theta_01 must have shape [D], [P,D], or [B,P,D]")
        basin_array = default_basins if basin_indices is None else np.asarray(list(basin_indices), dtype=np.int64)
        if len(basin_array) != theta.shape[0]:
            raise ValueError("basin_indices length must match candidate basin dimension")
        if (basin_array < 0).any() or (basin_array >= len(self.bundle.basin_ids)).any():
            raise IndexError("basin index out of range")
        return theta, basin_array

    def _split_tensors(self, split: str, basin_array: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Select forcings/targets, optionally from a device-resident 531-basin cache."""
        if self.cache_device_data:
            cached = self._device_split_cache.get(split)
            if cached is None:
                forcing, target, warmup = self._split_arrays(split)
                cached = (
                    torch.from_numpy(forcing).to(device=self.device, dtype=torch.float32),
                    torch.from_numpy(target).to(device=self.device, dtype=torch.float64),
                    warmup,
                )
                self._device_split_cache[split] = cached
            forcing, target, warmup = cached
            index = torch.as_tensor(basin_array, device=self.device, dtype=torch.long)
            return forcing.index_select(0, index), target.index_select(0, index), warmup
        forcing, target, warmup = self._split_arrays(split)
        return (
            torch.from_numpy(forcing[basin_array]).to(device=self.device, dtype=torch.float32),
            torch.from_numpy(target[basin_array]).to(device=self.device, dtype=torch.float64),
            warmup,
        )

    def evaluate_candidates_tensor(
        self,
        theta_01: Any,
        *,
        basin_indices: Sequence[int] | None = None,
        split: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """GPU-resident KGE used by batched optimizers; no host round-trip."""
        theta, basin_array = self._normalize_candidates(theta_01, basin_indices)
        physical = normalized_to_physical(self.model_key, theta, clip=True)
        n_basins, population, d = theta.shape
        forcing_tensor, target_tensor, warmup_days = self._split_tensors(split, basin_array)
        physical_flat = physical.reshape(n_basins * population, d).to(device=self.device, dtype=torch.float32)
        temp_mean_train = torch.from_numpy(self.bundle.temp_mean_train[basin_array]).to(device=self.device, dtype=torch.float32)
        temp_std_train = torch.from_numpy(self.bundle.temp_std_train[basin_array]).to(device=self.device, dtype=torch.float32)
        forcing_expanded = forcing_tensor.repeat_interleave(population, dim=0)
        q_full, _ = self.model_adapter.run_model(
            forcing_expanded, physical_flat, forcing_names=self.bundle.forcing_names,
            temp_mean_train=temp_mean_train, temp_std_train=temp_std_train,
            cn_psol_annual=(torch.from_numpy(self._cn_psol_annual[basin_array])
                            .to(device=self.device, dtype=torch.float32)
                            if self._cn_psol_annual is not None else None),
        )
        q_eval = q_full[:, warmup_days:].reshape(n_basins, population, -1)
        return self.objective.evaluate(q_eval, target_tensor)

    def evaluate_candidates(
        self,
        theta_01: Any,
        *,
        basin_indices: Sequence[int] | None = None,
        split: str = "train",
    ) -> CandidateEvaluation:
        started = time.perf_counter()
        theta, basin_array = self._normalize_candidates(theta_01, basin_indices)
        n_basins, population, d = theta.shape
        fitness_tensor, diagnostics = self.evaluate_candidates_tensor(theta, basin_indices=basin_array, split=split)
        split_forcing, _split_target, warmup_days = self._split_arrays(split)
        evaluation_steps = split_forcing.shape[1] - warmup_days

        fitness_np = fitness_tensor.cpu().numpy()
        valid_np = diagnostics["valid"].cpu().numpy()
        valid_count_np = diagnostics["valid_count"].cpu().numpy()

        return CandidateEvaluation(
            fitness=fitness_np,
            valid=valid_np,
            valid_count=valid_count_np,
            candidate_evaluations=n_basins * population,
            split=split,
            q_shape=(n_basins, population, evaluation_steps),
            forcing_shape=(n_basins * population, split_forcing.shape[1], len(self.bundle.forcing_names)),
            metric_dtype="float64",
            runtime_seconds=time.perf_counter() - started,
        )
