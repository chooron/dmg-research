"""
Production Differentiable Parameter Learning (dPL) Runner for 36 Hydrological Models using Native DMG Framework.

Strictly inherits and uses DMG's native architecture:
- dmg.core.utils.set_randomseed
- dmg.core.data.create_training_grid (calculates ~170-200 minibatches per epoch for batch_size=100 & rho=730)
- dmg.core.data.samplers.hydro_sampler.HydroSampler (or random_index for 100 basins & 730d sequence sampling)
- dmg.core.calc.metrics.Metrics
- dmg.trainers.base.BaseTrainer / Trainer structure
"""
import argparse
from contextlib import nullcontext
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Setup repo paths
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

# DMG Framework Imports
from dmg.core.calc.metrics import Metrics
from dmg.core.data.data import create_training_grid, random_index
from dmg.core.data.samplers.hydro_sampler import HydroSampler
from dmg.core.utils import set_randomseed
from dmg.trainers.base import BaseTrainer

from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing, calendar_features
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dpl.optimizer_transaction import FiniteOptimizerTransaction, validate_finite_training_state
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model, get_spec
from src.checkpointing import atomic_torch_save
from src.objective import streaming_kge

DATA_DIR = BENCHMARK_ROOT.parents[1] / "data"
RESULTS_DIR = BENCHMARK_ROOT / "results" / "dpl_results"
CHECKPOINTS_DIR = BENCHMARK_ROOT / "checkpoints" / "dpl_production_20260730"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def load_camels_time_series(
    ids: list[int],
    *,
    train_start: str = "1980-10-01",
    train_end: str = "1995-09-30",
    validation_start: str = "1995-10-01",
    validation_end: str = "2010-09-30",
    warmup_days: int = 365,
):
    """Load date-aligned CAMELS forcing and mm/day streamflow targets.

    The source pickle starts at 1980-10-01.  Training forcing includes the
    warm-up prefix and the training target keeps that prefix so the existing
    ``compute_differentiable_kge`` call can discard it.  Validation forcing
    likewise includes the preceding warm-up period, while validation targets
    start exactly at ``validation_start``.
    """
    dataset_pkl = DATA_DIR / "camels_dataset"
    if not dataset_pkl.exists():
        dataset_pkl = DATA_DIR / "camels_dataset.pkl"

    import pickle
    with open(dataset_pkl, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        forcings_raw = data["forcings"]
        streamflow_raw = data["streamflow"]
    elif isinstance(data, (tuple, list)):
        forcings_raw = data[0]
        streamflow_raw = data[1]

    gage_ids_all = np.asarray(np.load(DATA_DIR / "gage_id.npy"), dtype=np.int64)
    id_map = {int(gid): idx for idx, gid in enumerate(gage_ids_all)}
    missing = [int(basin_id) for basin_id in ids if int(basin_id) not in id_map]
    if missing:
        raise ValueError(f"Basin IDs missing from gage_id.npy: {missing[:5]}")
    sub_indices = [id_map[int(basin_id)] for basin_id in ids]

    forcings = forcings_raw[sub_indices]  # (531, 12418, 3)
    streamflow = streamflow_raw[sub_indices]  # (531, 12418, 1)

    forcings = np.transpose(forcings, (1, 0, 2))  # (12418, 531, 3)
    streamflow = np.transpose(streamflow[:, :, 0], (1, 0))  # (12418, 531)

    # Convert observed streamflow from ft3/s to mm/day using area_gages2 (km2)
    attr_builder = CatchmentAttributeBuilder()
    raw_attr = attr_builder.load_raw_attributes(ids)
    area_km2 = raw_attr[:, 11]  # area_gages2 at col 11
    FT3S_TO_MMD_NUMERATOR = 0.0283168 * 86400.0 * 1000.0
    conversion_factor = FT3S_TO_MMD_NUMERATOR / (area_km2 * 1.0e6)  # shape (531,)
    streamflow_mmd = streamflow * conversion_factor[None, :]  # shape (12418, 531)

    dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")

    def bounds(start: str, end: str) -> tuple[int, int]:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts < dates[0] or end_ts > dates[-1] or start_ts > end_ts:
            raise ValueError(f"Requested period is outside CAMELS dates: {start}..{end}")
        left = int(dates.get_loc(start_ts))
        right = int(dates.get_loc(end_ts)) + 1
        return left, right

    train_left, train_right = bounds(train_start, train_end)
    validation_left, validation_right = bounds(validation_start, validation_end)
    validation_forcing_left = validation_left - int(warmup_days)
    if validation_forcing_left < 0:
        raise ValueError("Validation period does not have enough preceding warm-up days")

    train_x = forcings[train_left:train_right].copy()
    train_y = streamflow_mmd[train_left:train_right].copy()
    val_x = forcings[validation_forcing_left:validation_right].copy()
    val_y = streamflow_mmd[validation_left:validation_right].copy()

    expected_train_days = train_right - train_left
    expected_validation_days = validation_right - validation_left
    if train_x.shape[0] != expected_train_days or train_y.shape[0] != expected_train_days:
        raise RuntimeError("Training forcing and target lengths are not date-aligned")
    if val_x.shape[0] != expected_validation_days + warmup_days or val_y.shape[0] != expected_validation_days:
        raise RuntimeError("Validation warm-up forcing and target lengths are not aligned")

    return train_x, train_y, val_x, val_y


def build_informative_kge_catalog(
    observations: np.ndarray,
    prediction_days: int = 365,
    min_valid_points: int = 30,
    min_observation_std: float = 0.01,
) -> list[np.ndarray]:
    """Build calibration-window starts using vectorized cumulative statistics.
    
    Filters out flat/zero-variance observation windows where std(Q_obs) < min_observation_std (0.01 mm/day).
    For basins where no window meets the threshold, falls back to the highest-variance window.
    """
    values = np.asarray(observations, dtype=np.float64)  # (531, 5478)
    n_basins, calibration_days = values.shape

    valid = np.isfinite(values) & (values >= 0.0)
    clean = np.where(valid, values, 0.0)

    count_cs = np.concatenate((np.zeros((n_basins, 1)), np.cumsum(valid, axis=1)), axis=1)
    sum_cs = np.concatenate((np.zeros((n_basins, 1)), np.cumsum(clean, axis=1)), axis=1)
    square_cs = np.concatenate((np.zeros((n_basins, 1)), np.cumsum(clean * clean, axis=1)), axis=1)

    count = count_cs[:, prediction_days:] - count_cs[:, :-prediction_days]
    total = sum_cs[:, prediction_days:] - sum_cs[:, :-prediction_days]
    square_total = square_cs[:, prediction_days:] - square_cs[:, :-prediction_days]

    safe_count = np.maximum(count, 1.0)
    variance = np.maximum(square_total / safe_count - (total / safe_count) ** 2, 0.0)
    eligible = (count >= min_valid_points) & (variance >= float(min_observation_std) ** 2)

    catalog = []
    for b in range(n_basins):
        starts = np.flatnonzero(eligible[b])
        if starts.size == 0:
            score = np.where(count[b] >= min_valid_points, variance[b], -1.0)
            starts = np.asarray([int(np.argmax(score))], dtype=np.int64)
        
        # ``observations`` is calibration-only.  These starts index the
        # corresponding post-warm-up position in the full forcing sequence.
        catalog.append(starts.astype(np.int64))

    return catalog


def compute_differentiable_kge(
    q_sim: torch.Tensor,
    q_obs: torch.Tensor,
    warmup_days: int = 365,
    eps: float = 0.1,
):
    """Compute the IC-compatible differentiable 1 - KGE loss.

    The canonical IC evaluator uses ``streaming_kge``: float64 moment
    accumulation, sample standard deviation, and ``eps=0.1`` in the
    correlation and bias denominators.  Keeping the dPL trainer on that same
    convention makes reported validation KGE directly comparable to IC while
    retaining autograd through the simulated discharge.
    """
    if q_obs.shape[0] == q_sim.shape[0] + warmup_days:
        q_obs = q_obs[warmup_days:]
    elif q_sim.shape[0] == q_obs.shape[0] + warmup_days:
        q_sim = q_sim[warmup_days:]

    if q_obs.ndim == 3 and q_obs.shape[-1] == 1:
        q_obs = q_obs.squeeze(-1)
    if q_obs.ndim != 2:
        raise ValueError(f"q_obs must have shape [time, basin], got {tuple(q_obs.shape)}")

    if q_sim.ndim == 2:
        prediction = q_sim.unsqueeze(-1).unsqueeze(-1)
    elif q_sim.ndim == 3 and q_sim.shape[-1] == 1:
        prediction = q_sim.unsqueeze(-1)
    elif q_sim.ndim == 4:
        prediction = q_sim
    else:
        raise ValueError(f"q_sim must have shape [time, basin] or singleton-group variants, got {tuple(q_sim.shape)}")

    kge, invalid = streaming_kge(prediction, q_obs, eps=eps)
    kge = kge.squeeze(-1).squeeze(-1)
    invalid = invalid.squeeze(-1).squeeze(-1)
    valid_basin_mask = ~invalid & torch.isfinite(kge)
    kge_valid = torch.where(valid_basin_mask, kge, torch.zeros_like(kge))
    loss = 1.0 - kge_valid.sum() / valid_basin_mask.sum().clamp_min(1.0)
    return loss, kge


class DmgNativeBenchmarkTrainer(BaseTrainer):
    """Native DMG Trainer implementation inheriting from BaseTrainer."""

    def __init__(self, model_name: str, config: dict, device: str = "cuda"):
        super().__init__(config=config, model=None)
        self.model_name = model_name
        self.device = device
        train_config = config["train"]
        self.epochs = train_config["epochs"]
        self.min_epochs = train_config.get("min_epochs", 50)
        self.early_stopping_patience = train_config.get("early_stopping_patience", 10)
        self.early_stopping_min_delta = train_config.get("early_stopping_min_delta", 1.0e-4)
        self.detect_anomaly = train_config.get("detect_anomaly", False)
        self.resume_checkpoint = train_config.get("resume_checkpoint")
        self.batch_size = train_config["batch_size"]
        self.lr = train_config["lr"]
        # Preserve the historical production clip threshold unless explicitly
        # configured; unlike the old inline code this is transaction-gated.
        self.grad_clip_norm = train_config.get("grad_clip_norm", 1.0)
        self.failure_policy = train_config.get("finite_failure_policy", "raise")
        self.parameterizer_architecture = train_config.get("parameterizer_architecture", "legacy")
        self.parameterizer_output_transform = train_config.get("parameterizer_output_transform", "sigmoid")
        self.parameterizer_head_hidden_dims = train_config.get("parameterizer_head_hidden_dims")
        self.saturation_floor = float(train_config.get("saturation_floor", 0.01))
        self.saturation_regularizer_weight = float(train_config.get("saturation_regularizer_weight", 0.0))
        self.saturation_diagnostics = bool(train_config.get("saturation_diagnostics", False))
        self.mapping_telemetry: list[dict] = []
        self.checkpoint_root = Path(train_config.get("checkpoint_root", CHECKPOINTS_DIR))
        self.results_root = Path(train_config.get("results_root", RESULTS_DIR))
        if not self.checkpoint_root.is_absolute():
            self.checkpoint_root = BENCHMARK_ROOT / self.checkpoint_root
        if not self.results_root.is_absolute():
            self.results_root = BENCHMARK_ROOT / self.results_root
        self.results_root.mkdir(parents=True, exist_ok=True)
        # Set DMG random seed for reproducibility
        set_randomseed(config.get("random_seed", 42))

        # 1. Load Basin IDs & Caravan 35 Attributes Matrix
        sub531_path = DATA_DIR / "531sub_id.txt"
        self.ids = load_ids(sub531_path)
        self.n_basins = len(self.ids)

        attr_builder = CatchmentAttributeBuilder()
        self.norm_attr = attr_builder.build_normalized_attributes(self.ids, device=device, method="zscore")
        self.n_attr = self.norm_attr.shape[1]  # 35

        # 2. Load Train & Validation Time Series and Preload onto GPU VRAM for 100% GPU speed
        self.train_x, self.train_y, self.val_x, self.val_y = load_camels_time_series(self.ids)
        self.n_train_days = self.train_x.shape[0]

        self.train_x_t = torch.as_tensor(self.train_x, dtype=torch.float64, device=device)
        self.train_y_t = torch.as_tensor(self.train_y, dtype=torch.float64, device=device)
        self.train_dates = pd.date_range("1980-10-01", "1995-09-30", freq="D")
        self.validation_dates = pd.date_range("1994-10-01", "2010-09-30", freq="D")
        self.is_calendar_model = self.model_name in CALENDAR_MODELS
        if self.is_calendar_model:
            self.train_doy_t = calendar_features(
                self.train_dates,
                dtype=torch.float64,
                device=torch.device(device),
            ).reshape(-1)
        else:
            self.train_doy_t = None

        # 3. Build Informative KGE Window Catalog (Filters out std(Q_obs) < 0.01 mm/day windows)
        min_std = config["model"].get("min_observation_std", 0.01)
        warmup_days = config["model"].get("warmup", 365)
        self.catalog = build_informative_kge_catalog(
            self.train_y[warmup_days:].T,
            prediction_days=365,
            min_observation_std=min_std,
        )

        # 4. Hydrological Model & Parameterizer Setup
        self.spec = get_spec(model_name, device=device)
        n_params = self.spec.dimension

        # Use DMG's compiled step path.  The eager path executes every daily
        # step through Python and leaves the GPU underutilized.
        backend = config["model"].get("backend", "compile")
        self.hydro_model = build_model(
            model_name, device, warm_up=365, backend=backend, dtype=torch.float64
        )
        self.parameterizer = CatchmentParameterizer(
            in_features=self.n_attr,
            out_features=n_params,
            hidden_dims=[256, 256, 256],
            # DMG HydrologyModel owns normalized-to-physical parameter
            # mapping in _descale_params.  Passing bounds here would map the
            # parameters twice and collapse runoff near zero.
            param_bounds=None,
            dropout=0.05,
            architecture=self.parameterizer_architecture,
            output_transform=self.parameterizer_output_transform,
            parameter_names=list(self.spec.parameter_names),
            parameter_groups=self.spec.parameter_groups,
            head_hidden_dims=self.parameterizer_head_hidden_dims,
            saturation_floor=self.saturation_floor,
            saturation_regularizer_weight=self.saturation_regularizer_weight,
        ).to(device, dtype=torch.float64)
        self.optimizer = self.init_optimizer()
        self.transaction = FiniteOptimizerTransaction(
            self.optimizer,
            self.parameterizer.parameters(),
            clip_norm=self.grad_clip_norm,
            failure_policy=self.failure_policy,
            named_parameters=self.parameterizer.named_parameters(),
        )
        # Build dummy tensor for DMG create_training_grid calculation
        n_forcing_features = 4 if self.is_calendar_model else 3
        dummy_xc = np.zeros(
            (self.n_train_days, self.n_basins, n_forcing_features + self.n_attr),
            dtype=np.float32,
        )
        self.n_samples, self.n_minibatch, self.n_timesteps = create_training_grid(dummy_xc, config)

    def init_optimizer(self) -> torch.optim.Optimizer:
        return optim.AdamW(self.parameterizer.parameters(), lr=self.lr, weight_decay=1e-4)

    def train(self) -> None:
        self.train_benchmark()

    def evaluate(self) -> None:
        pass

    def inference(self) -> torch.Tensor:
        with torch.no_grad():
            params = self.parameterizer(self.norm_attr).unsqueeze(-1)
            val_x_t = torch.as_tensor(self.val_x, dtype=torch.float32, device=self.device)
            val_x_t, _ = add_calendar_forcing(
                val_x_t,
                self.validation_dates,
                model_name=self.model_name,
            )
            return self.hydro_model({"x_phy": val_x_t}, (None, params))["streamflow"]

    def calc_metrics(self, obs: np.ndarray, sim: np.ndarray) -> Metrics:
        return Metrics(obs, sim)

    def train_benchmark(self) -> dict:
        print(f"\n========================================================")
        print(f"   DMG Native dPL Training for [{self.model_name.upper()}]")
        print(
            f"   Max Epochs: {self.epochs} | Min Epochs: {self.min_epochs} | "
            f"Patience: {self.early_stopping_patience} | Batch Size: {self.batch_size} | "
            f"Steps/Epoch: {self.n_minibatch}"
        )
        print(
            f"   Backend: {self.config['model'].get('backend', 'compile')} | "
            f"Anomaly Detection: {self.detect_anomaly}"
        )
        print(f"========================================================", flush=True)

        if self.min_epochs > self.epochs:
            raise ValueError("min_epochs cannot exceed epochs")

        model_ckpt_dir = self.checkpoint_root / self.model_name
        model_ckpt_dir.mkdir(parents=True, exist_ok=True)
        resume_epoch = 0
        resume_loss = float("inf")
        if self.resume_checkpoint:
            resume_path = Path(self.resume_checkpoint)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            checkpoint_model = checkpoint.get("model_name")
            if checkpoint_model is not None and str(checkpoint_model).lower() != self.model_name:
                raise ValueError(f"checkpoint model mismatch: checkpoint={checkpoint_model!r}, requested={self.model_name!r}")
            checkpoint_names = checkpoint.get("parameter_names")
            if checkpoint_names is not None and tuple(checkpoint_names) != tuple(self.spec.parameter_names):
                raise ValueError("checkpoint canonical parameter names/order mismatch")
            checkpoint_groups = checkpoint.get("parameter_groups")
            current_groups = self.spec.parameter_groups
            if checkpoint_groups is not None:
                normalized_checkpoint_groups = {
                    str(group): tuple(names) for group, names in checkpoint_groups.items()
                }
                if normalized_checkpoint_groups != current_groups:
                    raise ValueError("checkpoint process-group metadata mismatch")
            checkpoint_mapping = checkpoint.get("parameter_mapping")
            if checkpoint_mapping is not None and str(checkpoint_mapping).lower() != self.hydro_model.parameter_mapping:
                raise ValueError("checkpoint parameter mapping mismatch")
            checkpoint_span = checkpoint.get("log_mapping_span_threshold")
            if checkpoint_span is not None and not np.isclose(float(checkpoint_span), self.hydro_model.log_mapping_span_threshold):
                raise ValueError("checkpoint log-mapping threshold mismatch")
            checkpoint_architecture = checkpoint.get("parameterizer_architecture")
            if checkpoint_architecture is not None and checkpoint_architecture != self.parameterizer_architecture:
                raise ValueError(
                    f"checkpoint architecture mismatch: checkpoint={checkpoint_architecture!r}, "
                    f"requested={self.parameterizer_architecture!r}"
                )
            checkpoint_transform = checkpoint.get("parameterizer_output_transform")
            if checkpoint_transform is not None and checkpoint_transform != self.parameterizer_output_transform:
                raise ValueError(
                    f"checkpoint output transform mismatch: checkpoint={checkpoint_transform!r}, "
                    f"requested={self.parameterizer_output_transform!r}"
                )
            # Strict loading preserves legacy key semantics and refuses silent
            # partial loads for the process-head architecture.
            self.parameterizer.load_state_dict(checkpoint["parameterizer_state"], strict=True)
            if "optimizer_state" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(self.device)
            validate_finite_training_state(self.parameterizer.parameters(), self.optimizer)
            resume_epoch = int(checkpoint.get("epoch", 0))
            resume_loss = float(checkpoint.get("loss", float("inf")))
            if not np.isfinite(resume_loss):
                raise FloatingPointError(f"refusing non-finite resume loss from {resume_path}")
            print(
                f"Resuming from {resume_path} at epoch {resume_epoch} "
                f"with train loss {resume_loss:.6f}",
                flush=True,
            )
        def assert_finite(label: str, tensor: torch.Tensor) -> None:
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{self.model_name}: non-finite {label}")

        def assert_parameter_bounds(raw_params: torch.Tensor) -> None:
            assert_finite("normalized parameters", raw_params)
            if bool((raw_params < 0.0).any() or (raw_params > 1.0).any()):
                raise FloatingPointError(f"{self.model_name}: normalized parameter outside [0, 1]")
            physical = self.hydro_model._descale_params(raw_params)
            for name, values in physical.items():
                assert_finite(f"physical parameter {name}", values)
                lower, upper = self.hydro_model.parameter_bounds[name]
                tolerance = 1.0e-5 * max(abs(float(upper) - float(lower)), 1.0)
                if bool((values < lower - tolerance).any() or (values > upper + tolerance).any()):
                    raise FloatingPointError(f"{self.model_name}: physical parameter {name} outside bounds")

        def save_checkpoint(path: Path, epoch: int, loss_value: float, stop_reason: str = "") -> None:
            # Only successful finite transactions reach this function.  Keep
            # the check here as the serialization boundary as well.
            validate_finite_training_state(self.parameterizer.parameters(), self.optimizer, loss=loss_value)
            payload = {
                "epoch": epoch,
                "model_name": self.model_name,
                "parameterizer_architecture": self.parameterizer_architecture,
                "parameterizer_output_transform": self.parameterizer_output_transform,
                "parameter_names": list(self.spec.parameter_names),
                "parameter_groups": self.spec.parameter_groups,
                "parameter_mapping": self.hydro_model.parameter_mapping,
                "log_mapping_span_threshold": self.hydro_model.log_mapping_span_threshold,
                "parameterizer_state": self.parameterizer.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss": loss_value,
                "stop_reason": stop_reason,
                "finite_optimizer_steps": self.transaction.successful_steps,
            }
            atomic_torch_save(payload, path)
            print(f"  --> Checkpoint Saved -> {path.name}", flush=True)

        history = []
        window_length = self.config["model"]["rho"]
        warmup_days = self.config["model"].get("warmup", 365)
        best_loss = resume_loss
        best_epoch = resume_epoch
        stale_epochs = 0
        best_parameterizer_state = {
            name: value.detach().cpu().clone()
            for name, value in self.parameterizer.state_dict().items()
        } if self.resume_checkpoint else None
        stop_reason = "max_epochs"
        t0 = time.time()

        for epoch in range(resume_epoch + 1, self.epochs + 1):

            self.parameterizer.train()
            epoch_loss_sum = 0.0
            epoch_successes = 0

            for mb in range(self.n_minibatch):
                self.optimizer.zero_grad()
                b_indices = np.random.choice(self.n_basins, size=self.batch_size, replace=False)
                norm_attr_batch = self.norm_attr[b_indices]
                window_starts = []
                sub_x_list = []
                sub_y_list = []
                for b_idx in b_indices:
                    t_start = int(np.random.choice(self.catalog[b_idx]))
                    window_starts.append(t_start)
                    sub_x_list.append(self.train_x_t[t_start : t_start + window_length, b_idx, :])
                    sub_y_list.append(self.train_y_t[t_start : t_start + window_length, b_idx])

                sub_x = torch.stack(sub_x_list, dim=1)
                sub_y = torch.stack(sub_y_list, dim=1)
                if self.is_calendar_model:
                    sub_doy = torch.stack(
                        [self.train_doy_t[start : start + window_length] for start in window_starts],
                        dim=1,
                    ).unsqueeze(-1)
                    sub_x = torch.cat((sub_x, sub_doy), dim=-1)

                anomaly_context = torch.autograd.detect_anomaly() if self.detect_anomaly else nullcontext()
                with anomaly_context:
                    pred_params, mapping_diagnostics = self.parameterizer(
                        norm_attr_batch, return_diagnostics=True
                    )
                    total_mapping_jacobian = mapping_diagnostics["transform_jacobian"] * self.hydro_model.normalized_parameter_mapping_jacobian(pred_params)
                    mapping_diagnostics["total_mapping_jacobian"] = total_mapping_jacobian
                    mapping_diagnostics["normalized_jacobian"] = total_mapping_jacobian
                    raw_params = pred_params.unsqueeze(-1)
                    assert_parameter_bounds(raw_params)
                    q_sim = self.hydro_model(
                        {"x_phy": sub_x}, (None, raw_params)
                    )["streamflow"].squeeze(-1).squeeze(-1)
                    assert_finite("simulated streamflow", q_sim)
                    loss, _ = compute_differentiable_kge(
                        q_sim,
                        sub_y[warmup_days:],
                        warmup_days=0,
                    )
                    if self.saturation_regularizer_weight:
                        loss = loss + self.saturation_regularizer_weight * self.parameterizer.saturation_regularizer_from_diagnostics(
                            mapping_diagnostics
                        )
                    assert_finite("loss", loss)
                    result = self.transaction.step(
                        loss,
                        epoch=epoch,
                        batch_index=mb,
                        basin_ids=b_indices,
                    )
                if not result.success:
                    continue
                if self.saturation_diagnostics:
                    self.mapping_telemetry.append(
                        self.parameterizer.summarize_mapping_diagnostics(mapping_diagnostics)
                    )
                epoch_successes += 1
                epoch_loss_sum += float(loss.detach().item())

            if epoch_successes == 0:
                raise RuntimeError(f"{self.model_name}: no finite optimizer transactions completed in epoch {epoch}")
            avg_epoch_loss = epoch_loss_sum / epoch_successes
            history.append({"epoch": epoch, "loss_1_minus_kge": avg_epoch_loss})
            improved = avg_epoch_loss < best_loss - self.early_stopping_min_delta
            if improved:
                best_loss = avg_epoch_loss
                best_epoch = epoch
                stale_epochs = 0
                best_parameterizer_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.parameterizer.state_dict().items()
                }
                save_checkpoint(model_ckpt_dir / "best.pt", epoch, avg_epoch_loss)
            else:
                stale_epochs += 1

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] Train Loss (1-KGE): {avg_epoch_loss:.4f} "
                f"(best={best_loss:.4f}, stale={stale_epochs})",
                flush=True,
            )
            if epoch % 5 == 0 or epoch == self.epochs:
                save_checkpoint(model_ckpt_dir / f"epoch_{epoch:02d}.pt", epoch, avg_epoch_loss)
            if epoch >= self.min_epochs and stale_epochs >= self.early_stopping_patience:
                stop_reason = "early_stopping"
                save_checkpoint(model_ckpt_dir / "early_stop.pt", epoch, avg_epoch_loss, stop_reason)
                break

        if best_parameterizer_state is None:
            raise RuntimeError(f"{self.model_name}: no finite training checkpoint was produced")
        self.parameterizer.load_state_dict(best_parameterizer_state)
        actual_epochs = history[-1]["epoch"] if history else resume_epoch
        elapsed_time = time.time() - t0

        print(f"\nEvaluating Model [{self.model_name}] on 1995-2010 Validation Set...", flush=True)
        self.parameterizer.eval()
        with torch.no_grad():
            val_pred_params = self.parameterizer(self.norm_attr)
            val_raw_params = val_pred_params.unsqueeze(-1)
            assert_parameter_bounds(val_raw_params)
            val_x_t = torch.as_tensor(self.val_x, dtype=torch.float32, device=self.device)
            val_x_t, _ = add_calendar_forcing(
                val_x_t,
                self.validation_dates,
                model_name=self.model_name,
            )
            val_y_t = torch.as_tensor(self.val_y, dtype=torch.float32, device=self.device)
            val_q_sim = self.hydro_model(
                {"x_phy": val_x_t}, (None, val_raw_params)
            )["streamflow"].squeeze(-1).squeeze(-1)
            assert_finite("validation streamflow", val_q_sim)
            val_loss, per_basin_kge = compute_differentiable_kge(
                val_q_sim, val_y_t, warmup_days=warmup_days
            )
            assert_finite("validation loss", val_loss)

        val_kge_np = per_basin_kge.cpu().numpy()
        val_kge_median = float(np.nanmedian(val_kge_np))
        val_kge_mean = float(np.nanmean(val_kge_np))
        print(f"=== DMG Validation Complete for [{self.model_name}] in {elapsed_time:.1f}s ===", flush=True)
        print(f"Validation KGE Median: {val_kge_median:.4f} | Validation KGE Mean: {val_kge_mean:.4f}", flush=True)

        summary_data = {
            "model_name": self.model_name,
            "configured_epochs": self.epochs,
            "actual_epochs": actual_epochs,
            "best_epoch": best_epoch,
            "stop_reason": stop_reason,
            "train_loss_final": history[-1]["loss_1_minus_kge"],
            "train_loss_best": best_loss,
            "val_kge_median": val_kge_median,
            "val_kge_mean": val_kge_mean,
            "elapsed_seconds": elapsed_time,
            "saturation_telemetry": self.mapping_telemetry[-1] if self.mapping_telemetry else None,
        }

        by_basin_df = pd.DataFrame({"basin_id": [f"{b:08d}" for b in self.ids], "val_kge": val_kge_np})
        by_basin_csv = self.results_root / f"dpl_{actual_epochs}ep_{self.model_name}_by_basin.csv"
        by_basin_df.to_csv(by_basin_csv, index=False, float_format="%.4f")
        return summary_data


def main():
    parser = argparse.ArgumentParser(description="Run DMG Native Production dPL Benchmark for 36 Models")
    parser.add_argument("--model", default="simhyd", help="Target model name or 'all'")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--min_epochs", type=int, default=50, help="Minimum epochs before early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs")
    parser.add_argument("--min_delta", type=float, default=1.0e-4, help="Minimum train-loss improvement")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rho", type=int, default=730, help="Training window length")
    parser.add_argument("--warmup", type=int, default=365)
    parser.add_argument("--backend", choices=("compile", "eager"), default="compile")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--parameterizer_architecture",
        choices=("legacy", "process_heads", "residual_process", "residual_selective"),
        default="legacy",
    )
    parser.add_argument(
        "--parameterizer_output_transform",
        choices=("sigmoid", "softsign", "arctan", "identity", "linear"),
        default="sigmoid",
    )
    parser.add_argument("--saturation_floor", type=float, default=0.01)
    parser.add_argument("--saturation_regularizer_weight", type=float, default=0.0)
    parser.add_argument("--saturation_diagnostics", action="store_true")
    parser.add_argument("--resume_checkpoint", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--checkpoint_root", default=None, help="Optional isolated checkpoint root")
    parser.add_argument("--results_root", default=None, help="Optional isolated results root")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = {
        "random_seed": 42,
        "device": args.device,
        "train_time": ["1980/10/01", "1995/09/30"],
        "train": {
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "early_stopping_patience": args.patience,
            "early_stopping_min_delta": args.min_delta,
            "detect_anomaly": args.detect_anomaly,
            "resume_checkpoint": args.resume_checkpoint,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "grad_clip_norm": args.grad_clip_norm,
            "parameterizer_architecture": args.parameterizer_architecture,
            "parameterizer_output_transform": args.parameterizer_output_transform,
            "saturation_floor": args.saturation_floor,
            "saturation_regularizer_weight": args.saturation_regularizer_weight,
            "saturation_diagnostics": args.saturation_diagnostics,
            "checkpoint_root": args.checkpoint_root,
            "results_root": args.results_root,
        },
        "model": {
            "rho": args.rho,
            "warmup": args.warmup,
            "backend": args.backend,
        },
    }

    models_to_run = [args.model] if args.model != "all" else list(NPARAM_INFO_36.keys())
    all_summaries = []

    for m in models_to_run:
        try:
            trainer = DmgNativeBenchmarkTrainer(m, config=config, device=args.device)
            res = trainer.train_benchmark()
            all_summaries.append(res)
        except Exception as e:
            print(f"ERROR running DMG Native dPL for model [{m}]: {e}")

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_root = Path(config["train"].get("results_root") or RESULTS_DIR)
        if not summary_root.is_absolute():
            summary_root = BENCHMARK_ROOT / summary_root
        summary_root.mkdir(parents=True, exist_ok=True)
        summary_csv = summary_root / f"dpl_{args.epochs}ep_model_summary.csv"
        summary_df.to_csv(summary_csv, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
