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
import os
import random
import sys
import time
from pathlib import Path

import json
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
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model, get_spec

DATA_DIR = BENCHMARK_ROOT.parents[1] / "data"
# Results follow the canonical layout: results/{method}/{model}/{loss}/{seed}/.
RESULTS_DIR = BENCHMARK_ROOT / "results" / "dpl"
CHECKPOINTS_DIR = BENCHMARK_ROOT / "checkpoints" / "dpl"

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


def compute_differentiable_kge(q_sim: torch.Tensor, q_obs: torch.Tensor, warmup_days: int = 365, eps: float = 1e-6):
    """Compute differentiable 1 - KGE loss with eps_sq inside sqrt (matching hydrodiag)."""
    if q_obs.shape[0] == q_sim.shape[0] + warmup_days:
        q_obs = q_obs[warmup_days:]
    elif q_sim.shape[0] == q_obs.shape[0] + warmup_days:
        q_sim = q_sim[warmup_days:]

    eps_sq = eps * eps
    mask = torch.isfinite(q_obs) & torch.isfinite(q_sim) & (q_obs >= 0.0) & (q_sim >= 0.0)
    mask_f = mask.to(dtype=q_sim.dtype)
    n_valid = mask_f.sum(dim=0).clamp_min(1.0)

    obs_safe = torch.where(mask, q_obs, torch.zeros_like(q_obs))
    sim_safe = torch.where(mask, q_sim, torch.zeros_like(q_sim))

    mean_obs = obs_safe.sum(dim=0) / n_valid
    mean_sim = sim_safe.sum(dim=0) / n_valid

    obs_diff = (obs_safe - mean_obs[None, :]) * mask_f
    sim_diff = (sim_safe - mean_sim[None, :]) * mask_f

    var_obs = (obs_diff ** 2).sum(dim=0) / n_valid
    var_sim = (sim_diff ** 2).sum(dim=0) / n_valid

    std_obs = torch.sqrt(var_obs + eps_sq)
    std_sim = torch.sqrt(var_sim + eps_sq)

    cov = (obs_diff * sim_diff).sum(dim=0) / n_valid
    r = cov / (std_obs * std_sim)

    alpha = std_sim / std_obs
    beta = mean_sim / (mean_obs + eps)

    distance_sq = (r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2 + eps_sq
    kge = 1.0 - torch.sqrt(distance_sq)

    valid_basin_mask = (n_valid > 30) & torch.isfinite(kge)
    kge_valid = torch.where(valid_basin_mask, kge, torch.zeros_like(kge))

    mean_kge = kge_valid.sum() / valid_basin_mask.sum().clamp_min(1.0)
    loss = 1.0 - mean_kge
    return loss, kge


class DmgNativeBenchmarkTrainer(BaseTrainer):
    """Native DMG Trainer implementation inheriting from BaseTrainer."""

    def __init__(self, model_name: str, config: dict, device: str = "cuda"):
        super().__init__(config=config, model=None)
        self.model_name = model_name
        self.device = device
        self.epochs = config["train"]["epochs"]
        self.batch_size = config["train"]["batch_size"]
        self.lr = config["train"]["lr"]

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

        self.train_x_t = torch.as_tensor(self.train_x, dtype=torch.float32, device=device)
        self.train_y_t = torch.as_tensor(self.train_y, dtype=torch.float32, device=device)
        self.train_dates = pd.date_range("1980-10-01", "1995-09-30", freq="D")
        self.validation_dates = pd.date_range("1994-10-01", "2010-09-30", freq="D")
        self.is_calendar_model = self.model_name in CALENDAR_MODELS
        if self.is_calendar_model:
            self.train_doy_t = calendar_features(
                self.train_dates,
                dtype=torch.float32,
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
        self.hydro_model = build_model(model_name, device, warm_up=365, backend=backend)
        self.parameterizer = CatchmentParameterizer(
            in_features=self.n_attr,
            out_features=n_params,
            hidden_dims=[256, 256, 256],
            # DMG HydrologyModel owns normalized-to-physical parameter
            # mapping in _descale_params.  Passing bounds here would map the
            # parameters twice and collapse HBV runoff near zero.
            param_bounds=None,
            dropout=0.05,
        ).to(device)

        self.optimizer = self.init_optimizer()

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

    def _evaluate_validation(self) -> tuple[float, float, np.ndarray]:
        """Evaluate frozen parameterizer on the full 1995-2010 validation period.

        Returns (val_kge_median, val_kge_mean, per_basin_kge_np).
        """
        warmup_days = int(self.config["model"].get("warmup", 365))
        self.parameterizer.eval()
        with torch.no_grad():
            val_pred_params = self.parameterizer(self.norm_attr)
            val_raw_params = val_pred_params.unsqueeze(-1)
            val_x_t = torch.as_tensor(self.val_x, dtype=torch.float32, device=self.device)
            val_x_t, _ = add_calendar_forcing(
                val_x_t,
                self.validation_dates,
                model_name=self.model_name,
            )
            val_y_t = torch.as_tensor(self.val_y, dtype=torch.float32, device=self.device)
            val_q_sim = self.hydro_model({"x_phy": val_x_t}, (None, val_raw_params))["streamflow"].squeeze(-1).squeeze(-1)
            _, per_basin_kge = compute_differentiable_kge(val_q_sim, val_y_t, warmup_days=warmup_days)
            val_kge_np = per_basin_kge.cpu().numpy()
            val_kge_median = float(np.nanmedian(val_kge_np))
            val_kge_mean = float(np.nanmean(val_kge_np))
        return val_kge_median, val_kge_mean, val_kge_np

    def train_benchmark(self) -> dict:
        print(f"\n========================================================")
        print(f"   DMG Native dPL Training for [{self.model_name.upper()}]")
        print(f"   Epochs: {self.epochs} | Batch Size: {self.batch_size} Basins | Steps/Epoch: {self.n_minibatch}")
        print(f"   Variance Filter: min_observation_std = {self.config['model'].get('min_observation_std', 0.01)} mm/day")
        print(f"========================================================", flush=True)

        # Canonical layout: results|checkpoints/{method}/{model}/{loss}/{seed}/
        model_ckpt_dir = CHECKPOINTS_DIR / self.model_name / "1-kge" / f"seed{self.config['random_seed']}"
        model_ckpt_dir.mkdir(parents=True, exist_ok=True)
        result_dir = RESULTS_DIR / self.model_name / "1-kge" / f"seed{self.config['random_seed']}"
        (result_dir / "final").mkdir(parents=True, exist_ok=True)

        history = []
        window_length = self.config["model"]["rho"]  # 730
        warmup_days = self.config["model"].get("warmup", 365)

        t0 = time.time()

        # Early-stopping policy: never stop before min_epochs; afterwards stop
        # when validation median KGE has not improved for `patience` epochs.
        min_epochs = int(self.config["train"].get("min_epochs", 50))
        patience = int(self.config["train"].get("patience", 10))
        best_val_kge = -float("inf")
        best_epoch = 0
        stop_reason = ""
        for epoch in range(1, self.epochs + 1):
            self.parameterizer.train()
            epoch_loss_sum = 0.0

            for mb in range(self.n_minibatch):
                self.optimizer.zero_grad()

                # Mini-batch sampling: Randomly select batch_size (100) basins
                b_indices = np.random.choice(self.n_basins, size=self.batch_size, replace=False)
                norm_attr_batch = self.norm_attr[b_indices]  # (100, 35)

                pred_params = self.parameterizer(norm_attr_batch)
                raw_params = pred_params.unsqueeze(-1)  # (100, N_params, 1)

                # Direct GPU Tensor Slicing & Stacking (100% In-VRAM, 0 CPU Bottleneck)
                sub_x_list = []
                sub_y_list = []
                window_starts = []
                for b_idx in b_indices:
                    t_start = int(np.random.choice(self.catalog[b_idx]))
                    window_starts.append(t_start)
                    sub_x_list.append(self.train_x_t[t_start : t_start + window_length, b_idx, :])
                    sub_y_list.append(self.train_y_t[t_start : t_start + window_length, b_idx])

                sub_x = torch.stack(sub_x_list, dim=1)  # (730, 100, 3) on GPU
                sub_y = torch.stack(sub_y_list, dim=1)  # (730, 100) on GPU
                if self.is_calendar_model:
                    sub_doy = torch.stack(
                        [self.train_doy_t[start : start + window_length] for start in window_starts],
                        dim=1,
                    ).unsqueeze(-1)
                    sub_x = torch.cat((sub_x, sub_doy), dim=-1)

                q_sim = self.hydro_model({"x_phy": sub_x}, (None, raw_params))["streamflow"].squeeze(-1).squeeze(-1)

                # The first 365 days initialize the physical states only;
                # compare simulated and observed runoff on the prediction
                # interval that follows the warm-up.
                loss, _ = compute_differentiable_kge(
                    q_sim,
                    sub_y[warmup_days:],
                    warmup_days=0,
                )
                loss.backward()

                nn.utils.clip_grad_norm_(self.parameterizer.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss_sum += loss.item()

            avg_epoch_loss = epoch_loss_sum / self.n_minibatch
            history.append({"epoch": epoch, "loss_1_minus_kge": avg_epoch_loss})

            # Full-period validation every epoch: drives best-checkpoint selection and early stopping
            val_kge_median, _, _ = self._evaluate_validation()
            improved = val_kge_median > best_val_kge
            if improved:
                best_val_kge = val_kge_median
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_name": self.model_name,
                    "parameterizer_state": self.parameterizer.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "val_kge_median": val_kge_median,
                }, model_ckpt_dir / "best.pt")
            history[-1]["val_median_kge"] = val_kge_median

            print(f"Epoch [{epoch:02d}/{self.epochs:02d}] Train Loss (1-KGE): {avg_epoch_loss:.4f} | Val KGE: {val_kge_median:.4f}{' *' if improved else ''}", flush=True)
            # Save Checkpoints Every 5 Epochs
            if epoch % 5 == 0 or epoch == self.epochs:
                ckpt_path = model_ckpt_dir / f"epoch_{epoch:02d}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_name": self.model_name,
                    "parameterizer_state": self.parameterizer.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "val_median_kge": val_kge_median,
                }, ckpt_path)
                print(f"  --> Checkpoint Saved -> {ckpt_path.name}", flush=True)

            # Early stopping: only after min_epochs, stop after `patience` epochs without improvement
            if epoch >= min_epochs and (epoch - best_epoch) >= patience:
                stop_reason = f"early_stop_patience_{patience}_no_improve_since_epoch_{best_epoch}"
                print(f"  --> Early stopping at epoch {epoch} (best epoch {best_epoch}, val KGE {best_val_kge:.4f})", flush=True)
                break

        elapsed_time = time.time() - t0
        actual_epochs = history[-1]["epoch"] if history else 0

        # Reload the best checkpoint and re-evaluate for canonical per-basin metrics
        best_ckpt = model_ckpt_dir / "best.pt"
        if best_ckpt.exists():
            payload = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.parameterizer.load_state_dict(payload["parameterizer_state"])
            best_epoch = int(payload["epoch"])
            best_val_kge = float(payload.get("val_kge_median", best_val_kge))
            print(f"Loaded best checkpoint epoch {best_epoch} (val KGE {best_val_kge:.4f})", flush=True)
        if not stop_reason:
            stop_reason = "completed_all_epochs"

        # Full 15-year Validation Evaluation using DMG Metrics
        print(f"\nEvaluating Model [{self.model_name}] on 1995-2010 Validation Set...", flush=True)
        val_kge_median, val_kge_mean, val_kge_np = self._evaluate_validation()

        print(f"=== DMG Validation Complete for [{self.model_name}] in {elapsed_time:.1f}s ===", flush=True)
        print(f"Validation KGE Median: {val_kge_median:.4f} | Validation KGE Mean: {val_kge_mean:.4f}", flush=True)

        summary_data = {
            "model_name": self.model_name,
            "epochs": self.epochs,
            "actual_epochs": actual_epochs,
            "min_epochs": min_epochs,
            "patience": patience,
            "best_epoch": best_epoch,
            "stop_reason": stop_reason,
            "train_loss_final": history[-1]["loss_1_minus_kge"] if history else None,
            "val_kge_median": val_kge_median,
            "val_kge_mean": val_kge_mean,
            "val_kge_median_best": best_val_kge,
            "elapsed_seconds": elapsed_time,
        }

        # Save per-basin validation KGE CSV + run summary (canonical layout)
        by_basin_df = pd.DataFrame({"basin_id": [f"{b:08d}" for b in self.ids], "val_kge": val_kge_np})
        by_basin_csv = result_dir / "final" / "basin_metrics.csv"
        by_basin_df.to_csv(by_basin_csv, index=False, float_format="%.4f")
        pd.DataFrame(history).to_csv(result_dir / "epochs.csv", index=False, float_format="%.4f")
        (result_dir / "summary.json").write_text(json.dumps(summary_data, indent=2) + "\n")
        # DONE marker: worker-pool completion signal (replaces epoch-file check)
        (result_dir / "DONE").write_text(json.dumps(summary_data, indent=2) + "\n")

        return summary_data


def main():
    parser = argparse.ArgumentParser(description="Run DMG Native Production dPL Benchmark for 36 Models")
    parser.add_argument("--model", default="simhyd", help="Target model name or 'all'")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs (upper budget)")
    parser.add_argument("--min-epochs", type=int, default=50, help="Early stopping never triggers before this epoch")
    parser.add_argument("--patience", type=int, default=10, help="Stop after N epochs without validation-KGE improvement (after min-epochs)")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    config = {
        "random_seed": 42,
        "device": args.device,
        "train_time": ["1980/10/01", "1995/09/30"],
        "train": {
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "model": {
            "rho": 730,
            "warmup": 365,
            "backend": "compile",
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
        (RESULTS_DIR / "_summary").mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(all_summaries)
        summary_csv = RESULTS_DIR / "_summary" / "dpl_model_summary.csv"
        summary_df.to_csv(summary_csv, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
