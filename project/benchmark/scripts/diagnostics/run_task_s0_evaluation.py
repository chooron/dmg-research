#!/usr/bin/env python3
"""Task S0: Re-evaluate 36 Models IC-dPL Gap on Validation Period (1995-10..2010-09).

- Protocol: Continuous long sequence with 365d warmup (1994-10-01..2010-09-30)
- Basins: 531 Caravan/CAMELS basins
- Objective: FP64 streaming KGE, sample variance, eps=0.1
- Checkpoint rule: Best validation median KGE available checkpoint per model
  (SIMHYD <= 63, VIC <= 56, flexb CROSS_RUN).
"""
from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.data_selection import frozen_parameters, load_ids
from src.model_registry import NPARAM_INFO_36, build_model, get_spec
from src.objective import streaming_kge

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = BENCHMARK_ROOT / "results/ic_dpl_validation_recompute_20260831"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
DATA_DIR = REPO_ROOT / "data"
basin_ids = load_ids(DATA_DIR / "531sub_id.txt")
n_basins = len(basin_ids)
print(f"Loaded {n_basins} basin IDs")

dataset_pkl = DATA_DIR / "camels_dataset.pkl" if (DATA_DIR / "camels_dataset.pkl").exists() else DATA_DIR / "camels_dataset"
with open(dataset_pkl, "rb") as f:
    bundle = pickle.load(f)

if isinstance(bundle, dict):
    forcings, target, attrs = bundle["forcings"], bundle["streamflow"], bundle["attributes"]
else:
    forcings, target, attrs = bundle

reference = np.load(DATA_DIR / "gage_id.npy")
idx = np.array([np.where(reference == int(b))[0][0] for b in basin_ids])
dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
wl = dates.get_loc(pd.Timestamp("1994-10-01"))
ys = dates.get_loc(pd.Timestamp("1995-10-01"))
wr = dates.get_loc(pd.Timestamp("2010-09-30")) + 1

fx = forcings[idx, wl:wr, :3]
fy = target[idx, ys:wr, 0].copy()
area = attrs[idx, 11].astype(float)
fy *= (0.0283168 * 86400.0 * 1e3 / (area * 1e6))[:, None]

x_val_base = torch.as_tensor(np.transpose(fx, (1, 0, 2)), dtype=torch.float64, device=DEVICE)
y_val = torch.as_tensor(fy.T, dtype=torch.float64, device=DEVICE)

attr_builder = CatchmentAttributeBuilder()
attrs_cuda = attr_builder.build_normalized_attributes(basin_ids, device=DEVICE, method="zscore")

# 2. Checkpoint and Archive Inventory
ic_root = BENCHMARK_ROOT / "results/ic_dpl_aligned_full300_20260819_final/checkpoints/ic_dpl_aligned_full300_20260819"
best_root = BENCHMARK_ROOT / "results/ic_dpl_aligned_full300_20260819_final/best_training"
ckpt_main_dir = BENCHMARK_ROOT / "results/dpl_full_retrain_20260813/auto100/checkpoints"
ckpt_flexb_dir = BENCHMARK_ROOT / "results/dpl_flexb_retrain_20260830/auto100/checkpoints"

models = sorted(list(NPARAM_INFO_36.keys()))

def get_ic_frozen_data(model_name: str):
    m_dir = ic_root / model_name
    pieces = sorted(m_dir.glob("chunk_*_gen_*.pt"), key=lambda p: int(p.name.split("_")[1]))
    if not pieces:
        m_best_dir = best_root / model_name
        pieces = sorted(m_best_dir.glob("chunk_*_best.pt"), key=lambda p: int(p.name.split("_")[1]))
        
    basin_parts, latent_parts, fitness_parts = [], [], []
    dim = NPARAM_INFO_36[model_name]
    used_paths = [str(p) for p in pieces]
    
    for p in pieces:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        b_ids = np.asarray(payload["basin_ids"], dtype=np.int64)
        if "solver" in payload:
            state = payload["solver"]["state"]
            latent = state["best_latent"]
            fitness = state["best_fitness"]
        else:
            latent = payload["best_latent"]
            fitness = payload["best_fitness"]
            
        b_count = len(b_ids)
        if latent.ndim == 2 and latent.shape[0] == b_count * 10:
            latent = latent.reshape(b_count, 10, dim)
            fitness = fitness.reshape(b_count, 10)
        elif latent.ndim == 2 and latent.shape[0] == b_count:
            basin_parts.append(b_ids)
            latent_parts.append(latent)
            fitness_parts.append(fitness)
            continue
            
        best_start = fitness.argmax(axis=1)
        sel_latent = latent[torch.arange(b_count), torch.as_tensor(best_start)]
        sel_fitness = fitness[np.arange(b_count), best_start]
        basin_parts.append(b_ids)
        latent_parts.append(sel_latent)
        fitness_parts.append(sel_fitness)
        
    all_bids = np.concatenate(basin_parts)
    all_latent = torch.cat(latent_parts, dim=0)
    all_fitness = np.concatenate(fitness_parts) if isinstance(fitness_parts[0], np.ndarray) else torch.cat(fitness_parts, dim=0).numpy()
    return all_bids, all_latent, all_fitness, used_paths

def compute_kge_components(pred: torch.Tensor, obs: torch.Tensor, eps: float = 0.1):
    p_orig = pred if pred.dtype == torch.float64 else pred.to(torch.float64)
    o_orig = obs if obs.dtype == torch.float64 else obs.to(torch.float64)
    if p_orig.dim() == 4:
        p_orig = p_orig.squeeze(-1).squeeze(-1)
        
    finite = torch.isfinite(p_orig) & torch.isfinite(o_orig)
    p = torch.where(finite, p_orig, torch.zeros_like(p_orig))
    o = torch.where(finite, o_orig, torch.zeros_like(o_orig))
    n = finite.sum(dim=0).to(torch.float64)
    
    sp, so = p.sum(0), o.sum(0)
    sp2, so2, sc = (p * p).sum(0), (o * o).sum(0), (p * o).sum(0)
    variance_floor = torch.as_tensor(1.0e-24, dtype=torch.float64, device=p_orig.device)
    safe_n = n.clamp_min(1.0)
    centered_p = (sp2 - sp.square() / safe_n).clamp_min(0.0)
    centered_o = (so2 - so.square() / safe_n).clamp_min(0.0)
    denom_n = (n - 1.0).clamp_min(1.0)
    
    std_p = torch.sqrt((centered_p / denom_n).clamp_min(variance_floor))
    std_o = torch.sqrt((centered_o / denom_n).clamp_min(variance_floor))
    covariance_scale = torch.sqrt((centered_p * centered_o).clamp_min(variance_floor))
    
    r = (sc - sp * so / safe_n) / (covariance_scale + eps)
    beta = (sp / safe_n) / (so / safe_n + eps)
    alpha = std_p / (std_o + eps)
    distance_sq = (r - 1.0).square() + (beta - 1.0).square() + (alpha - 1.0).square()
    score = 1.0 - torch.sqrt(distance_sq.clamp_min(variance_floor))
    invalid = (n < 2) | ~torch.isfinite(score) | ~torch.isfinite(p_orig).all(dim=0)
    return score, r, alpha, beta, invalid

# Helper for bootstrap CI
def bootstrap_mean_ci(arr, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    low = np.percentile(boot_means, (100 - ci) / 2)
    high = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return low, high

def bootstrap_median_ci(arr, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    boot_meds = [np.median(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    low = np.percentile(boot_meds, (100 - ci) / 2)
    high = np.percentile(boot_meds, 100 - (100 - ci) / 2)
    return low, high

def sha256_of_file(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return "FILE_NOT_FOUND"
    return hashlib.sha256(p.read_bytes()).hexdigest()

per_basin_rows = []
model_summary_rows = []
provenance_rows = []

print("\n=== Running 36-Model Continuous Validation Rescore ===")

for model_name in models:
    t0 = time.time()
    dim = NPARAM_INFO_36[model_name]
    
    # 1. Setup Forcing (add DOY for calendar models)
    if model_name in CALENDAR_MODELS:
        x_val, _ = add_calendar_forcing(
            x_val_base, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name=model_name
        )
    else:
        x_val = x_val_base
        
    # 2. Evaluate IC
    ic_bids, ic_latent, ic_train_fitness, ic_paths = get_ic_frozen_data(model_name)
    ic_norm = torch.sigmoid(ic_latent.to(device=DEVICE, dtype=torch.float64)).unsqueeze(-1)
    
    m_ic = build_model(model_name, DEVICE, warm_up=365, backend="eager", parameter_mapping="linear", dtype=torch.float64)
    with torch.inference_mode():
        q_ic = m_ic({"x_phy": x_val}, (None, ic_norm))["streamflow"].squeeze(-1).squeeze(-1)
        kge_ic, r_ic, a_ic, b_ic, _ = compute_kge_components(q_ic, y_val)
        
    # 3. Select and Evaluate dPL Checkpoint
    ckpt_dir = ckpt_flexb_dir / model_name if model_name == "flexb" else ckpt_main_dir / model_name
    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem and p.stem.split("_")[1].isdigit() else 0)
    
    # Apply constraints
    if model_name == "simhyd":
        pts = [p for p in pts if int(p.stem.split("_")[1]) <= 63]
    elif model_name == "vic":
        pts = [p for p in pts if int(p.stem.split("_")[1]) <= 56]
        
    m_dpl = build_model(model_name, DEVICE, warm_up=365, backend="eager", parameter_mapping="auto", dtype=torch.float64)
    
    ckpt_scores = []
    for pt in pts:
        ep_num = int(pt.stem.split("_")[1])
        ckpt_payload = torch.load(pt, map_location=DEVICE, weights_only=False)
        net = CatchmentParameterizer(attrs_cuda.shape[1], dim, hidden_dims=[256, 256], dropout=0.05).to(DEVICE, dtype=torch.float64)
        net.load_state_dict(ckpt_payload["network"])
        net.eval()
        with torch.no_grad():
            dpl_norm = net(attrs_cuda.to(dtype=torch.float64)).unsqueeze(-1)
        with torch.inference_mode():
            q_d = m_dpl({"x_phy": x_val}, (None, dpl_norm))["streamflow"].squeeze(-1).squeeze(-1)
            score_d, r_d, a_d, b_d, _ = compute_kge_components(q_d, y_val)
            med = float(score_d.median().item())
            mean = float(score_d.mean().item())
            ckpt_scores.append((ep_num, med, mean, pt, score_d, r_d, a_d, b_d))
            
    # Sort to find best validation median KGE, tie-breaker earliest epoch
    ckpt_scores.sort(key=lambda x: (-x[1], x[0]))
    best_ep, best_med, best_mean, best_pt, kge_dpl, r_dpl, a_dpl, b_dpl = ckpt_scores[0]
    
    # Sort by epoch for adjacent range
    by_ep = sorted(ckpt_scores, key=lambda x: x[0])
    avail_eps = [x[0] for x in by_ep]
    idx_in_avail = avail_eps.index(best_ep)
    adj_eps = []
    if idx_in_avail > 0:
        adj_eps.append(by_ep[idx_in_avail - 1][0])
    adj_eps.append(best_ep)
    if idx_in_avail < len(by_ep) - 1:
        adj_eps.append(by_ep[idx_in_avail + 1][0])
    adj_meds = [x[1] for x in by_ep if x[0] in adj_eps]
    adj_range_str = f"[{min(adj_meds):.4f}, {max(adj_meds):.4f}] (epochs {adj_eps})"
    
    tag = "CHECKPOINT_GRID_LIMITED"
    if model_name == "flexb":
        tag = "CROSS_RUN; CHECKPOINT_GRID_LIMITED"
        
    # 4. Compute Per-Basin Rows and Gap
    kge_ic_np = kge_ic.detach().cpu().numpy()
    r_ic_np = r_ic.detach().cpu().numpy()
    a_ic_np = a_ic.detach().cpu().numpy()
    b_ic_np = b_ic.detach().cpu().numpy()
    
    kge_dpl_np = kge_dpl.detach().cpu().numpy()
    r_dpl_np = r_dpl.detach().cpu().numpy()
    a_dpl_np = a_dpl.detach().cpu().numpy()
    b_dpl_np = b_dpl.detach().cpu().numpy()
    
    gap_np = kge_ic_np - kge_dpl_np
    
    for b_idx in range(n_basins):
        bid = int(basin_ids[b_idx])
        per_basin_rows.append({
            "model": model_name,
            "basin_id": bid,
            "kge_ic": float(kge_ic_np[b_idx]),
            "kge_dpl": float(kge_dpl_np[b_idx]),
            "gap": float(gap_np[b_idx]),
            "r_ic": float(r_ic_np[b_idx]),
            "alpha_ic": float(a_ic_np[b_idx]),
            "beta_ic": float(b_ic_np[b_idx]),
            "r_dpl": float(r_dpl_np[b_idx]),
            "alpha_dpl": float(a_dpl_np[b_idx]),
            "beta_dpl": float(b_dpl_np[b_idx]),
        })
        
    # 5. Bootstrap CIs on Gap
    mean_ci_low, mean_ci_high = bootstrap_mean_ci(gap_np)
    med_ci_low, med_ci_high = bootstrap_median_ci(gap_np)
    
    model_summary_rows.append({
        "model": model_name,
        "gap_median": float(np.median(gap_np)),
        "gap_median_ci95_low": float(med_ci_low),
        "gap_median_ci95_high": float(med_ci_high),
        "gap_mean": float(np.mean(gap_np)),
        "gap_mean_ci95_low": float(mean_ci_low),
        "gap_mean_ci95_high": float(mean_ci_high),
        "pos_fraction": float((gap_np > 0).mean()),
        "N": n_basins,
        "ic_median": float(np.median(kge_ic_np)),
        "ic_mean": float(np.mean(kge_ic_np)),
        "dpl_median": float(np.median(kge_dpl_np)),
        "dpl_mean": float(np.mean(kge_dpl_np)),
        "dpl_chosen_epoch": best_ep,
        "dpl_tag": tag,
        "dpl_adjacent_range": adj_range_str,
    })
    
    # 6. Record Provenance (paths + SHA256)
    for ic_p in ic_paths:
        provenance_rows.append({
            "model": model_name,
            "side": "IC",
            "epoch_or_gen": "gen300",
            "path": ic_p,
            "sha256": sha256_of_file(ic_p),
        })
    provenance_rows.append({
        "model": model_name,
        "side": "dPL",
        "epoch_or_gen": f"epoch_{best_ep:03d}",
        "path": str(best_pt),
        "sha256": sha256_of_file(str(best_pt)),
    })
    
    print(f"[{model_name:12s}] IC={np.median(kge_ic_np):.4f}, dPL(ep{best_ep:02d})={np.median(kge_dpl_np):.4f} -> Gap Median={np.median(gap_np):+.4f} (Mean={np.mean(gap_np):+.4f}) in {time.time()-t0:.2f}s")

# Save 1. per_basin.csv
df_per_basin = pd.DataFrame(per_basin_rows)
csv_per_basin = OUT_DIR / "per_basin.csv"
df_per_basin.to_csv(csv_per_basin, index=False)
print(f"\nSaved {len(df_per_basin)} rows to {csv_per_basin}")

# Save 2. model_summary.csv
df_summary = pd.DataFrame(model_summary_rows)
df_summary = df_summary.sort_values(by="gap_median", ascending=False).reset_index(drop=True)
csv_summary = OUT_DIR / "model_summary.csv"
df_summary.to_csv(csv_summary, index=False)
print(f"Saved model summary ({len(df_summary)} models) to {csv_summary}")

# Save Provenance CSV
df_prov = pd.DataFrame(provenance_rows)
csv_prov = OUT_DIR / "provenance_sha256.csv"
df_prov.to_csv(csv_prov, index=False)
print(f"Saved provenance ({len(df_prov)} items) to {csv_prov}")

print("\n=== Top 10 Models by Gap Median ===")
print(df_summary[["model", "gap_median", "gap_median_ci95_low", "gap_median_ci95_high", "gap_mean", "ic_median", "dpl_median", "dpl_chosen_epoch", "dpl_tag"]].head(10).to_string())

print("\n=== Full 36-Model Ranking Table ===")
for idx, r in df_summary.iterrows():
    print(f"{idx+1:2d}. {r['model']:12s} | Gap Median: {r['gap_median']:+.4f} [{r['gap_median_ci95_low']:+.4f}, {r['gap_median_ci95_high']:+.4f}] | IC: {r['ic_median']:.4f} | dPL(ep{r['dpl_chosen_epoch']:02d}): {r['dpl_median']:.4f}")
