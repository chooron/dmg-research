"""
Production Differentiable Parameter Learning (dPL) Runner for 36 Hydrological Models.

Features:
- Time periods: Train (1980-1995, 15yr) | Validation (1995-2010, 15yr)
- Subsequence Random Sampling: 730 days (365 warmup + 365 prediction) per training step
- Input attributes: 35-dimensional Caravan attributes matrix (aligned to 671/531 basins)
- Training schedule: 20 Epochs, saving PyTorch checkpoints every 5 epochs (epochs 5, 10, 15, 20)
- Validation: Evaluates frozen MLP parameters on 1995-2010 validation set across 531 basins
- Results: Saved to project/benchmark/results/dpl/{model}/1-kge/seed42/ (canonical layout) alongside IC (CMA-ES) benchmark
"""
import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Setup paths
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from dmg.core.calc.metrics import Metrics
from dmg.core.utils import set_randomseed

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


def convert_streamflow_ft3s_to_mm_day(streamflow: np.ndarray, area_km2: np.ndarray) -> np.ndarray:
    """Convert CAMELS streamflow from ft³/s to mm/day using drainage area in km²."""
    # streamflow: (Time, Basins), area_km2: (Basins,)
    factor = (0.0283168 * 86400 * 1000.0) / (area_km2 * 1.0e6)
    return streamflow * factor[None, :]


def load_camels_time_series(ids_531: np.ndarray):
    """Load forcing and streamflow targets for 531 basins from 1980 to 2010."""
    dataset_path = DATA_DIR / "camels_dataset"
    with open(dataset_path, "rb") as f:
        forcings, streamflow, attributes = pickle.load(f)

    reference_ids = np.load(DATA_DIR / "gage_id.npy")
    reference_ids_str = [str(g).zfill(8) for g in reference_ids]
    target_ids_str = [str(b).zfill(8) for b in ids_531]

    indices = [reference_ids_str.index(b) for b in target_ids_str]

    # Sub-select 531 basins: forcings in pkl is (Basins=671, Time=12418, Features=3)
    sub_forcings_b_t_f = forcings[indices, :, :]  # Shape: (531, 12418, 3)
    sub_streamflow_ft3s_b_t = streamflow[indices, :, 0]  # Shape: (531, 12418)

    # Transpose to (Time, Basins, Features) and (Time, Basins)
    sub_forcings = sub_forcings_b_t_f.transpose(1, 0, 2)  # Shape: (12418, 531, 3)
    sub_streamflow_ft3s = sub_streamflow_ft3s_b_t.transpose(1, 0)  # Shape: (12418, 531)
    
    # Area column is index 11 (area_gages2)
    areas_km2 = attributes[indices, 11]
    sub_streamflow_mmd = convert_streamflow_ft3s_to_mm_day(sub_streamflow_ft3s, areas_km2)

    # Dataset date range starting at 1980-10-01
    # Train: 1980-10-01 to 1995-09-30 (15 years = 5478 days, index 0:5478)
    # Val:   1995-10-01 to 2010-09-30 (15 years = 5479 days, index 5478:10957)
    train_slice = slice(0, 5478)
    val_slice = slice(5478, 10957)

    train_x = sub_forcings[train_slice].copy()
    train_y = sub_streamflow_mmd[train_slice].copy()

    val_x = sub_forcings[val_slice].copy()
    val_y = sub_streamflow_mmd[val_slice].copy()

    return train_x, train_y, val_x, val_y


def compute_differentiable_kge(
    q_sim: torch.Tensor,
    q_obs: torch.Tensor,
    warmup_days: int = 365,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute differentiable 1 - KGE loss and per-basin KGE array.
    q_sim, q_obs: (Time, Basins)
    """
    # Align time dimensions if model already stripped warmup
    if q_obs.shape[0] == q_sim.shape[0] + warmup_days:
        q_obs = q_obs[warmup_days:]
    elif q_sim.shape[0] == q_obs.shape[0] + warmup_days:
        q_sim = q_sim[warmup_days:]

    mask = torch.isfinite(q_obs) & torch.isfinite(q_sim)
    n_valid = mask.sum(dim=0, keepdim=True).clamp_min(1.0)

    q_sim_clean = torch.where(mask, q_sim, torch.zeros_like(q_sim))
    q_obs_clean = torch.where(mask, q_obs, torch.zeros_like(q_obs))

    mean_obs = q_obs_clean.sum(dim=0, keepdim=True) / n_valid
    mean_sim = q_sim_clean.sum(dim=0, keepdim=True) / n_valid

    var_obs = (torch.where(mask, (q_obs - mean_obs) ** 2, torch.zeros_like(q_obs)).sum(dim=0, keepdim=True) / n_valid).clamp_min(eps)
    var_sim = (torch.where(mask, (q_sim - mean_sim) ** 2, torch.zeros_like(q_sim)).sum(dim=0, keepdim=True) / n_valid).clamp_min(eps)

    std_obs = torch.sqrt(var_obs)
    std_sim = torch.sqrt(var_sim)

    cov = torch.where(mask, (q_obs - mean_obs) * (q_sim - mean_sim), torch.zeros_like(q_obs)).sum(dim=0, keepdim=True) / n_valid
    r = cov / (std_obs * std_sim + eps)

    alpha = std_sim / (std_obs + eps)
    beta = mean_sim / (mean_obs + eps)

    kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    kge = kge.squeeze(0)  # Shape: (Basins,)

    valid_basins = torch.isfinite(kge)
    if valid_basins.any():
        loss = torch.mean(1.0 - kge[valid_basins])
    else:
        loss = torch.tensor(0.0, device=q_sim.device, requires_grad=True)

    return loss, kge


def train_dpl_for_model(
    model_name: str,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 100,
    batch_steps_per_epoch: int = 40,
) -> dict:
    """Train dPL for a specific hydrological model over 20 epochs with batch_size=100 basin sampling and 730d subsequence sampling."""
    print(f"\n========================================================")
    print(f"   Starting dPL Training for Model [{model_name.upper()}] ({epochs} Epochs, Batch Size = {batch_size} Basins)")
    print(f"========================================================")
    set_randomseed(42)

    # 1. Load Basin IDs & Caravan 35 Attributes Matrix
    sub531_path = DATA_DIR / "531sub_id.txt"
    ids = load_ids(sub531_path)
    n_basins = len(ids)

    attr_builder = CatchmentAttributeBuilder()
    norm_attr = attr_builder.build_normalized_attributes(ids, device=device, method="zscore")
    n_attr = norm_attr.shape[1]  # 35

    # 2. Load Train (1980-1995) & Val (1995-2010) Time Series
    train_x, train_y, val_x, val_y = load_camels_time_series(ids)
    n_train_days = train_x.shape[0]  # 5478 days

    # 3. Model Spec & Parameterizer Setup
    spec = get_spec(model_name, device=device)
    min_b = spec.bounds[:, 0].to(dtype=torch.float32)
    max_b = spec.bounds[:, 1].to(dtype=torch.float32)
    n_params = spec.dimension

    hydro_model = build_model(model_name, device, warm_up=365, backend="eager")
    parameterizer = CatchmentParameterizer(
        in_features=n_attr,
        out_features=n_params,
        hidden_dims=[256, 256, 256],
        param_bounds=(min_b, max_b),
        dropout=0.05,
    ).to(device)

    optimizer = optim.AdamW(parameterizer.parameters(), lr=lr, weight_decay=1e-4)

    # Output folders follow the canonical layout: results|checkpoints/{method}/{model}/{loss}/{seed}/
    seed_dir = RESULTS_DIR / model_name / "1-kge" / "seed42"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "final").mkdir(parents=True, exist_ok=True)
    model_ckpt_dir = CHECKPOINTS_DIR / model_name / "1-kge" / "seed42"
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []

    # 4. Training Loop (20 Epochs with Random 730-day Subsequence & 100-Basin Mini-batch Sampling)
    window_length = 730  # 365 warmup + 365 prediction
    max_start_idx = n_train_days - window_length  # e.g., 5478 - 730 = 4748

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        parameterizer.train()
        epoch_loss_sum = 0.0

        for step in range(batch_steps_per_epoch):
            optimizer.zero_grad()

            # Mini-batch sampling: Randomly select batch_size (100) basins
            if batch_size < n_basins:
                b_indices = np.random.choice(n_basins, size=batch_size, replace=False)
            else:
                b_indices = np.arange(n_basins)

            norm_attr_batch = norm_attr[b_indices]  # (batch_size, 35)

            # Predict model parameters for the 100 sampled basins
            pred_params = parameterizer(norm_attr_batch)
            raw_params = pred_params.unsqueeze(-1)  # Shape: (batch_size, Params, 1)

            # Random 730-day subsequence start index
            t_start = random.randint(0, max_start_idx)
            sub_x_np = train_x[t_start : t_start + window_length, b_indices, :]  # (730, batch_size, 3)
            sub_y_np = train_y[t_start : t_start + window_length, b_indices]  # (730, batch_size)

            sub_x = torch.as_tensor(sub_x_np, dtype=torch.float32, device=device)
            sub_y = torch.as_tensor(sub_y_np, dtype=torch.float32, device=device)

            # Hydrological Model Forward Pass
            q_sim = hydro_model({"x_phy": sub_x}, (None, raw_params))["streamflow"].squeeze(-1).squeeze(-1)

            # Compute Loss (365 warmup excluded)
            loss, _ = compute_differentiable_kge(q_sim, sub_y, warmup_days=365)
            loss.backward()

            nn.utils.clip_grad_norm_(parameterizer.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_sum += loss.item()

        avg_epoch_loss = epoch_loss_sum / batch_steps_per_epoch
        history.append({"epoch": epoch, "loss_1_minus_kge": avg_epoch_loss})

        # Save Checkpoint Every 5 Epochs
        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = model_ckpt_dir / f"epoch_{epoch:02d}.pt"
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "parameterizer_state": parameterizer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_epoch_loss,
            }, ckpt_path)
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss (1-KGE): {avg_epoch_loss:.4f} | Checkpoint Saved -> {ckpt_path.name}", flush=True)

    elapsed_time = time.time() - t0

    # 5. Validation Evaluation on Full 1995-2010 Validation Period
    print(f"\nEvaluating Model [{model_name}] on Validation Set (1995-2010)...", flush=True)
    parameterizer.eval()
    with torch.no_grad():
        val_pred_params = parameterizer(norm_attr)
        val_raw_params = val_pred_params.unsqueeze(-1)

        val_x_t = torch.as_tensor(val_x, dtype=torch.float32, device=device)
        val_y_t = torch.as_tensor(val_y, dtype=torch.float32, device=device)

        # Full 15-year validation forward pass
        val_q_sim = hydro_model({"x_phy": val_x_t}, (None, val_raw_params))["streamflow"].squeeze(-1).squeeze(-1)
        val_loss, per_basin_kge = compute_differentiable_kge(val_q_sim, val_y_t, warmup_days=365)

        val_kge_np = per_basin_kge.cpu().numpy()
        val_kge_median = float(np.nanmedian(val_kge_np))
        val_kge_mean = float(np.nanmean(val_kge_np))

    print(f"=== dPL Validation Complete for [{model_name}] in {elapsed_time:.1f}s ===", flush=True)
    print(f"Validation KGE Median: {val_kge_median:.4f} | Validation KGE Mean: {val_kge_mean:.4f}", flush=True)

    # 6. Save Summary & Per-basin KGE Results
    summary_data = {
        "model_name": model_name,
        "epochs": epochs,
        "train_loss_final": history[-1]["loss_1_minus_kge"],
        "val_kge_median": val_kge_median,
        "val_kge_mean": val_kge_mean,
        "elapsed_seconds": elapsed_time,
    }

    # Save per-basin validation KGE CSV + run summary (canonical layout)
    by_basin_df = pd.DataFrame({"basin_id": [f"{b:08d}" for b in ids], "val_kge": val_kge_np})
    by_basin_csv = seed_dir / "final" / "basin_metrics.csv"
    by_basin_df.to_csv(by_basin_csv, index=False, float_format="%.4f")
    with (seed_dir / "summary.json").open("w") as fh:
        json.dump(summary_data, fh, indent=2)
        fh.write("\n")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Run Production dPL Benchmark for 36 Hydrological Models")
    parser.add_argument("--model", default="simhyd", help="Target model name or 'all'")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=100, help="Number of basins per training step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model != "all" else list(NPARAM_INFO_36.keys())

    all_summaries = []
    for m in models_to_run:
        try:
            res = train_dpl_for_model(
                m, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device
            )
            all_summaries.append(res)
        except Exception as e:
            print(f"ERROR running dPL for model [{m}]: {e}")

    # Save Overall Summary CSV
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_csv = RESULTS_DIR / "_summary" / "dpl_model_summary.csv"
        summary_df.to_csv(summary_csv, index=False, float_format="%.4f")
        print(f"\n========================================================")
        print(f"Overall dPL Benchmark Summary Saved -> {summary_csv}")
        print(f"========================================================")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
