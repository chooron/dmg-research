"""Differentiable parameter learning for the dual-evidence benchmark.

Trains an MLP from CAMELS basin attributes to normalized dmotpy model parameters,
then backpropagates KGE / KGE_LOG / NSE losses through HydrologyModel.

Usage
-----
from benchmark.parameter_learning import run_parameter_learning, ParameterLearningConfig
cfg = ParameterLearningConfig(model_id="hbv96", objective="KGE", ...)
run_parameter_learning(cfg)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local imports (benchmark is self-contained)
from .data import CamelsStore
from .losses import KgeBatchLoss, KgeLogBatchLoss, NseBatchLoss, LogNseBatchLoss
from .models import build_hydrology_model, available_model_ids
from .param_models import DeterministicParamModel

try:
    import dmotpy
    from dmotpy.models import HydrologyModel
    DMOTPY_AVAILABLE = True
except ImportError:
    DMOTPY_AVAILABLE = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParameterLearningConfig:
    """Full configuration for a parameter learning run."""

    # Required
    model_id: str
    objective: str                          # "KGE" or "KGE_LOG"

    # Data
    basin_ids_path: str = "data/559sub_id.txt"
    data_root: str = ""                     # passed to load_basin_data
    attributes_path: str = ""              # path to basin attributes CSV/parquet

    # Time splits
    train_start: str = "1989-01-01"
    train_end: str   = "1998-12-31"
    test_start: str  = "1999-01-01"
    test_end: str    = "2009-12-31"
    warmup_days: int = 365

    # Training
    epochs: int = 100
    lr: float = 1e-3
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1234])

    # MLP architecture
    hidden_size: int = 128

    # Loss-specific
    kge_log_eps_frac: float = 0.01          # per-basin eps = frac * mean_train_obs
    kge_log_global_eps: float = 1e-3

    # Output
    output_dir: str = "outputs/parameter_learning"

    # Device
    device: str = "cpu"

    # Misc
    num_workers: int = 0
    log_interval: int = 10

    # Full benchmark config dict (used by CamelsStore / build_hydrology_model)
    # If provided, data loading uses CamelsStore; otherwise falls back to legacy paths.
    benchmark_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Attribute loading helpers
# ---------------------------------------------------------------------------

def _load_attributes_from_store(store: "CamelsStore", basin_ids: list[str]) -> torch.Tensor:
    """Load and normalize CAMELS basin attributes from CamelsStore.

    Returns
    -------
    Tensor [n_basins, n_attrs]  float32, NaN-filled gaps replaced by column mean.
    """
    from .basins import basin_index
    idx_list = [basin_index(store.reference_ids, bid) for bid in basin_ids]
    arr = store.attributes[idx_list, :].astype(np.float32)  # [B, n_attrs]
    # Fill NaN with column mean
    for j in range(arr.shape[1]):
        col = arr[:, j]
        nan_mask = ~np.isfinite(col)
        if nan_mask.any() and (~nan_mask).any():
            arr[nan_mask, j] = np.nanmean(col)
    # Normalize per column
    mu = arr.mean(axis=0, keepdims=True)
    sigma = arr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    arr = (arr - mu) / sigma
    return torch.tensor(arr, dtype=torch.float32)


def _load_attributes(attributes_path: str, basin_ids: list[str]) -> torch.Tensor:
    """Load and normalize CAMELS basin attributes from a file (CSV or parquet).

    Returns
    -------
    Tensor [n_basins, n_attrs]  float32, NaN-filled gaps replaced by column mean.
    """
    p = Path(attributes_path)
    if p.suffix == ".parquet":
        df = pd.read_csv(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p, index_col=0)
    else:
        raise ValueError(f"Unsupported attributes file: {p}")

    # Align to basin_ids order
    df = df.loc[basin_ids]
    df = df.apply(lambda col: col.fillna(col.mean()))

    arr = df.values.astype(np.float32)
    mu = arr.mean(axis=0, keepdims=True)
    sigma = arr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    arr = (arr - mu) / sigma
    return torch.tensor(arr, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Batch hydrology helpers
# ---------------------------------------------------------------------------

def _build_batch_inputs_from_store(
    store: "CamelsStore",
    basin_ids: list[str],
    split_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load forcing + observed streamflow for all basins using CamelsStore.

    Returns
    -------
    forcing  : [T, n_basins, n_forcing]
    q_obs    : [T, n_basins, 1]
    """
    forcing_list, qobs_list = [], []
    for bid in basin_ids:
        period = store.period(bid, split_name, device="cpu")
        # period.x_phy: [T, 1, nf], period.target: [T, 1, 1]
        forcing_list.append(period.x_phy[:, 0, :])   # [T, nf]
        qobs_list.append(period.target[:, 0, :])      # [T, 1]

    forcing = torch.stack(forcing_list, dim=1).to(device)  # [T, B, nf]
    qobs    = torch.stack(qobs_list,   dim=1).to(device)   # [T, B, 1]
    return forcing, qobs


def _run_hydro_model(
    hydro_model: nn.Module,
    params_normalized: torch.Tensor,   # [n_basins, ny]
    forcing: torch.Tensor,             # [T, n_basins, nf]
    warmup_days: int,
) -> torch.Tensor:
    """Run hydrology model for all basins and return streamflow.

    Returns
    -------
    q_sim : [T - warmup_days, n_basins, 1] or [T - warmup_days, n_basins]
    """
    # Pass params as [B, ny] — dmotpy's 2D unpack_parameters branch handles this:
    #   if raw.shape[1] == static_count: return raw[:, :static_count]
    # This correctly gives per-basin static parameters.
    out = hydro_model(
        {"x_phy": forcing},
        (None, params_normalized),  # [B, ny] — 2D, no transpose needed
    )
    q_sim = out["streamflow"]                               # [T, B, ...] or [T, B]
    # Drop warmup
    q_sim = q_sim[warmup_days:]
    return q_sim


# ---------------------------------------------------------------------------
# Per-basin metric computation
# ---------------------------------------------------------------------------

def _per_basin_metrics(
    q_pred: torch.Tensor,    # [T, B] or [T, B, 1]
    q_obs: torch.Tensor,     # [T, B] or [T, B, 1]
    basin_ids: list[str],
    eps: float = 1e-6,
    kge_log_eps: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Compute per-basin KGE, KGE_LOG, NSE metrics.

    Returns DataFrame indexed by basin_id.
    """
    if q_pred.ndim == 3:
        q_pred = q_pred[..., 0]
    if q_obs.ndim == 3:
        q_obs = q_obs[..., 0]

    q_pred = q_pred.detach().cpu()
    q_obs  = q_obs.detach().cpu()

    T, B = q_pred.shape
    rows = []
    for b, bid in enumerate(basin_ids):
        p = q_pred[:, b]
        o = q_obs[:, b]
        mask = torch.isfinite(p) & torch.isfinite(o)
        if mask.sum() < 2:
            rows.append({"basin_id": bid, "KGE": float("nan"), "KGE_LOG": float("nan"), "NSE": float("nan")})
            continue

        pm = p[mask]; om = o[mask]

        # KGE
        kge_val = _kge_numpy(pm.numpy(), om.numpy())

        # KGE_LOG
        b_eps = float(kge_log_eps[b]) if kge_log_eps is not None else eps
        pm_log = np.log(np.clip(pm.numpy(), 0, None) + b_eps)
        om_log = np.log(np.clip(om.numpy(), 0, None) + b_eps)
        kge_log_val = _kge_numpy(pm_log, om_log)

        # NSE
        om_mean = om.mean().item()
        nse_num = ((pm - om) ** 2).sum().item()
        nse_den = ((om - om_mean) ** 2).sum().item()
        nse_val = 1.0 - nse_num / (nse_den + 1e-10)

        rows.append({"basin_id": bid, "KGE": kge_val, "KGE_LOG": kge_log_val, "NSE": nse_val})

    return pd.DataFrame(rows).set_index("basin_id")


def _kge_numpy(pred: np.ndarray, obs: np.ndarray, eps: float = 1e-10) -> float:
    """KGE on numpy arrays."""
    mean_p = pred.mean(); mean_o = obs.mean()
    std_p  = pred.std();  std_o  = obs.std()
    num = ((pred - mean_p) * (obs - mean_o)).sum()
    den = (np.sqrt(((pred - mean_p) ** 2).sum()) * np.sqrt(((obs - mean_o) ** 2).sum()))
    r = num / (den + eps)
    beta  = mean_p / (mean_o + eps)
    gamma = std_p  / (std_o  + eps)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _train_one_seed(
    seed: int,
    cfg: ParameterLearningConfig,
    basin_ids: list[str],
    attrs: torch.Tensor,
    hydro_model: nn.Module,
    train_forcing: torch.Tensor,
    train_qobs: torch.Tensor,
    test_forcing: torch.Tensor,
    test_qobs: torch.Tensor,
    kge_log_eps: torch.Tensor | None,
    device: torch.device,
    out_seed_dir: Path,
) -> dict[str, Any]:
    """Train MLP for one seed, return summary dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_basins = len(basin_ids)
    nx = attrs.shape[1]
    ny = hydro_model.num_parameters if hasattr(hydro_model, "num_parameters") else _infer_ny(hydro_model, cfg.model_id)

    mlp_cfg = {"hidden_size": cfg.hidden_size, "output_activation": "sigmoid"}
    mlp = DeterministicParamModel(config=mlp_cfg, nx=nx, ny=ny).to(device)
    optimizer = Adam(mlp.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Build loss
    obj_key = cfg.objective.upper()
    if obj_key == "KGE":
        loss_fn = KgeBatchLoss(eps=0.1).to(device)
    elif obj_key in {"KGE_LOG", "KGE_LOG_TRANSFORM"}:
        if kge_log_eps is not None:
            loss_fn = KgeLogBatchLoss(basin_eps=kge_log_eps, kge_eps=0.1).to(device)
        else:
            loss_fn = KgeLogBatchLoss(global_eps=cfg.kge_log_global_eps, kge_eps=0.1).to(device)
    elif obj_key == "NSE":
        loss_fn = NseBatchLoss(eps=0.1).to(device)
    elif obj_key in {"LOG_NSE", "LOGNSE"}:
        loss_fn = LogNseBatchLoss(eps=0.1, log_eps=1e-6).to(device)
    else:
        raise ValueError(f"Unsupported objective: {cfg.objective}")

    attrs_dev = attrs.to(device)
    epoch_log = []

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        mlp.train()
        optimizer.zero_grad()

        # Forward: attrs -> params [B, ny]
        params = mlp(attrs_dev)                                          # [B, ny]

        # Run hydro model on train period
        q_sim_train = _run_hydro_model(
            hydro_model, params, train_forcing, cfg.warmup_days
        )                                                                # [T_train, B, ...]

        # Align qobs
        T_sim = q_sim_train.shape[0]
        q_obs_aligned = train_qobs[-T_sim:].to(device)

        loss = loss_fn(q_sim_train, q_obs_aligned)
        if not torch.isfinite(loss):
            log.warning(f"  seed={seed} epoch={epoch}: non-finite loss, skipping backward")
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

        if epoch % cfg.log_interval == 0 or epoch == 1:
            log.info(f"  seed={seed} epoch={epoch}/{cfg.epochs} loss={loss.item():.4f}")
            epoch_log.append({"epoch": epoch, "train_loss": loss.item()})

    runtime = time.time() - t0

    # ---------- Evaluate ----------
    mlp.eval()
    with torch.no_grad():
        params_final = mlp(attrs_dev)                                    # [B, ny]

        # Train metrics
        q_sim_train_eval = _run_hydro_model(
            hydro_model, params_final, train_forcing, cfg.warmup_days
        )
        T_tr = q_sim_train_eval.shape[0]
        train_metrics_df = _per_basin_metrics(
            q_sim_train_eval, train_qobs[-T_tr:],
            basin_ids, kge_log_eps=kge_log_eps
        )

        # Test metrics
        q_sim_test_eval = _run_hydro_model(
            hydro_model, params_final, test_forcing, cfg.warmup_days
        )
        T_te = q_sim_test_eval.shape[0]
        test_metrics_df = _per_basin_metrics(
            q_sim_test_eval, test_qobs[-T_te:],
            basin_ids, kge_log_eps=kge_log_eps
        )

    # ---------- Save outputs ----------
    out_seed_dir.mkdir(parents=True, exist_ok=True)

    # Predicted parameters [B, ny]
    params_np = params_final.cpu().numpy()
    np.save(out_seed_dir / "predicted_params.npy", params_np)

    # Per-basin results table
    results = pd.DataFrame(index=basin_ids)
    results.index.name = "basin_id"
    results["model_id"]   = cfg.model_id
    results["objective"]  = cfg.objective
    results["seed"]       = seed
    results["train_start"] = cfg.train_start
    results["train_end"]   = cfg.train_end
    results["test_start"]  = cfg.test_start
    results["test_end"]    = cfg.test_end
    results["train_KGE"]     = train_metrics_df["KGE"]
    results["train_KGE_LOG"] = train_metrics_df["KGE_LOG"]
    results["train_NSE"]     = train_metrics_df["NSE"]
    results["test_KGE"]      = test_metrics_df["KGE"]
    results["test_KGE_LOG"]  = test_metrics_df["KGE_LOG"]
    results["test_NSE"]      = test_metrics_df["NSE"]

    # Record per-basin KGE_LOG epsilon
    if kge_log_eps is not None:
        eps_dict = {bid: float(kge_log_eps[i]) for i, bid in enumerate(basin_ids)}
        results["kge_log_epsilon"] = results.index.map(eps_dict)
    else:
        results["kge_log_epsilon"] = cfg.kge_log_global_eps

    # Predicted param columns
    for j in range(params_np.shape[1]):
        results[f"param_{j:02d}"] = params_np[:, j]

    # Success flags
    results["success"] = results["train_KGE"].apply(lambda x: np.isfinite(x))
    results["failure_reason"] = results["success"].apply(
        lambda s: "" if s else "non_finite_metrics"
    )

    results.to_csv(out_seed_dir / "per_basin_results.csv", index=False)

    # Epoch log
    pd.DataFrame(epoch_log).to_csv(out_seed_dir / "epoch_log.csv", index=False)

    # Summary
    summary = {
        "model_id": cfg.model_id,
        "objective": cfg.objective,
        "seed": seed,
        "mean_train_KGE": float(np.nanmean(results["train_KGE"])),
        "median_train_KGE": float(np.nanmedian(results["train_KGE"])),
        "mean_test_KGE": float(np.nanmean(results["test_KGE"])),
        "median_test_KGE": float(np.nanmedian(results["test_KGE"])),
        "mean_train_KGE_LOG": float(np.nanmean(results["train_KGE_LOG"])),
        "mean_test_KGE_LOG": float(np.nanmean(results["test_KGE_LOG"])),
        "mean_train_NSE": float(np.nanmean(results["train_NSE"])),
        "mean_test_NSE": float(np.nanmean(results["test_NSE"])),
        "basin_success_rate": float(results["success"].mean()),
        "basin_failure_rate": float(1.0 - results["success"].mean()),
        "runtime_s": runtime,
    }
    with open(out_seed_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        f"  seed={seed} done | train_KGE={summary['mean_train_KGE']:.3f} "
        f"test_KGE={summary['mean_test_KGE']:.3f} | runtime={runtime:.1f}s"
    )
    return summary


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_parameter_learning(cfg: ParameterLearningConfig) -> None:
    """Train MLP parameter learning for one model × objective across all seeds.

    Saves per-seed outputs under:
        {cfg.output_dir}/{cfg.model_id}/{cfg.objective}/seed{seed}/
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info(f"Parameter learning: model={cfg.model_id} objective={cfg.objective}")

    if not DMOTPY_AVAILABLE:
        raise ImportError("dmotpy is required for parameter learning.")

    device = torch.device(cfg.device)

    # Build CamelsStore from the benchmark config
    store = CamelsStore(cfg.benchmark_config)
    basin_ids = store.selected_ids
    log.info(f"  Loaded {len(basin_ids)} basins from CamelsStore")

    # Load basin attributes from CamelsStore
    attrs = _load_attributes_from_store(store, basin_ids)
    log.info(f"  Attributes shape: {tuple(attrs.shape)}")

    # Build HydrologyModel using benchmark's build_hydrology_model
    hydro_model = build_hydrology_model(cfg.benchmark_config, cfg.model_id, cfg.device)
    hydro_model.eval()
    # Freeze hydro model weights (only MLP params are trained)
    for p in hydro_model.parameters():
        p.requires_grad_(False)

    # Load forcing and qobs for train and test periods using CamelsStore
    log.info("  Loading train data...")
    train_forcing, train_qobs = _build_batch_inputs_from_store(
        store, basin_ids, "train", device
    )
    log.info("  Loading test data...")
    test_forcing, test_qobs = _build_batch_inputs_from_store(
        store, basin_ids, "test", device
    )

    # Compute per-basin KGE_LOG epsilon from training observations
    if cfg.objective.upper() in {"KGE_LOG", "KGE_LOG_TRANSFORM"}:
        kge_log_eps = _compute_basin_eps(train_qobs, cfg.kge_log_eps_frac, cfg.kge_log_global_eps)
        kge_log_eps = kge_log_eps.to(device)
        log.info(f"  KGE_LOG eps: min={kge_log_eps.min().item():.4f} max={kge_log_eps.max().item():.4f}")
    else:
        kge_log_eps = None

    # Output root
    out_root = Path(cfg.output_dir) / cfg.model_id / cfg.objective
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for seed in cfg.seeds:
        out_seed_dir = out_root / f"seed{seed}"
        log.info(f"  Starting seed {seed} → {out_seed_dir}")
        summary = _train_one_seed(
            seed=seed,
            cfg=cfg,
            basin_ids=basin_ids,
            attrs=attrs,
            hydro_model=hydro_model,
            train_forcing=train_forcing,
            train_qobs=train_qobs,
            test_forcing=test_forcing,
            test_qobs=test_qobs,
            kge_log_eps=kge_log_eps,
            device=device,
            out_seed_dir=out_seed_dir,
        )
        all_summaries.append(summary)

    # Save all-seed summary
    pd.DataFrame(all_summaries).to_csv(out_root / "all_seeds_summary.csv", index=False)
    log.info(f"  Saved all-seeds summary → {out_root / 'all_seeds_summary.csv'}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_basin_eps(
    train_qobs: torch.Tensor,
    frac: float = 0.01,
    global_eps: float = 1e-3,
) -> torch.Tensor:
    """Compute per-basin epsilon = frac * mean_train_obs.

    Parameters
    ----------
    train_qobs : [T, B] or [T, B, 1]
    """
    if train_qobs.ndim == 3:
        qobs = train_qobs[..., 0]
    else:
        qobs = train_qobs
    q_clamped = qobs.clamp(min=0.0).cpu()
    mean_obs = q_clamped.nanmean(dim=0)              # [B]
    eps = (mean_obs * frac).clamp(min=global_eps)
    return eps


def _infer_ny(hydro_model: nn.Module, model_id: str) -> int:
    """Fallback: try to infer number of parameters from model."""
    if hasattr(hydro_model, "n_params"):
        return int(hydro_model.n_params)
    if hasattr(hydro_model, "num_params"):
        return int(hydro_model.num_params)
    # Common MARRMoT model param counts (fallback table)
    _FALLBACK = {
        "hbv96": 14, "m01": 2, "m02": 3, "m03": 5, "m04": 4, "m05": 5,
        "m06": 6, "m07": 4, "m08": 4, "m09": 6, "m10": 7, "m11": 8,
        "m12": 6, "m13": 6, "m14": 6, "m15": 6, "m16": 5, "m17": 5,
        "m18": 5, "m19": 4, "m20": 6, "m21": 7, "m22": 7, "m23": 8,
        "m24": 9, "m25": 8, "m26": 7, "m27": 7, "m28": 8, "m29": 8,
        "m30": 7, "m31": 14, "m32": 8, "m33": 9, "m34": 9, "m35": 10,
        "m36": 10,
    }
    return _FALLBACK.get(model_id.lower(), 10)
