#!/usr/bin/env python3
"""Agent-3 Parameter-Level Paired Audit Runner.

1. Parameter boundary fraction pairing (IC vs dPL) on normalized theta <= 0.01 or >= 0.99.
2. IC cross-start parameter divergence (10 starts distance and train KGE spread).
3. Slow flow time constant (1/ks) distribution and threshold exceedances (1/3, 1/2, 1 year).
Across 5 models: flexb, flexi, flexis, hbv96, xinanjiang.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.model_registry import get_spec, NPARAM_INFO_36
from src.data_selection import load_ids

OUT_DIR = BENCHMARK_ROOT / "results/flex_protocol_diagnosis_20260831"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = REPO_ROOT / "data"
basin_ids = load_ids(DATA_DIR / "531sub_id.txt")
n_basins = len(basin_ids)

attr_builder = CatchmentAttributeBuilder()
attrs_cuda = attr_builder.build_normalized_attributes(basin_ids, device="cpu", method="zscore")

models = ["flexb", "flexi", "flexis", "hbv96", "xinanjiang"]
ic_base = BENCHMARK_ROOT / "results/ic_dpl_aligned_full300_20260819_final/checkpoints/ic_dpl_aligned_full300_20260819"
dpl_ckpts = {
    "flexb": BENCHMARK_ROOT / "results/dpl_flexb_retrain_20260830/auto100/checkpoints/flexb/epoch_060.pt",
    "flexi": BENCHMARK_ROOT / "results/dpl_full_retrain_20260813/auto100/checkpoints/flexi/epoch_060.pt",
    "flexis": BENCHMARK_ROOT / "results/dpl_full_retrain_20260813/auto100/checkpoints/flexis/epoch_070.pt",
    "hbv96": BENCHMARK_ROOT / "results/dpl_full_retrain_20260813/auto100/checkpoints/hbv96/epoch_050.pt",
    "xinanjiang": BENCHMARK_ROOT / "results/dpl_full_retrain_20260813/auto100/checkpoints/xinanjiang/epoch_050.pt",
}

# Helper to compute physical parameters
def to_physical(norm_val: float, bounds: list[float], mapping: str = "linear") -> float:
    lower, upper = bounds[0], bounds[1]
    if mapping == "auto" and lower > 0 and upper / lower >= 100.0:
        return float(np.exp(np.log(lower) + norm_val * (np.log(upper) - np.log(lower))))
    return float(lower + norm_val * (upper - lower))

# ----------------------------------------------------
# 1. Boundary Paired Audit & 2. IC Restart Divergence
# ----------------------------------------------------
boundary_rows = []
divergence_rows = []
slowflow_rows = []

for model_name in models:
    spec = get_spec(model_name)
    param_names = spec.parameter_names
    dim = len(param_names)
    bounds_dict = {name: [float(spec.bounds[i, 0]), float(spec.bounds[i, 1])] for i, name in enumerate(param_names)}
    
    # --- Load IC 10 starts ---
    m_ic_dir = ic_base / model_name
    pieces = sorted(m_ic_dir.glob("chunk_*_gen_300.pt"), key=lambda p: int(p.name.split("_")[1]))
    
    basin_parts = []
    latent_parts = []
    fitness_parts = []
    for p in pieces:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        b_ids = np.asarray(payload["basin_ids"], dtype=np.int64)
        state = payload["solver"]["state"]
        latent = state["best_latent"]  # (b_count * 10, dim)
        fitness = state["best_fitness"] # (b_count * 10)
        
        b_count = len(b_ids)
        latent = latent.reshape(b_count, 10, dim)
        fitness = fitness.reshape(b_count, 10)
        
        basin_parts.append(b_ids)
        latent_parts.append(latent)
        fitness_parts.append(fitness)
        
    all_bids = np.concatenate(basin_parts)
    all_latent = torch.cat(latent_parts, dim=0) # (531, 10, dim)
    all_fitness = torch.cat(fitness_parts, dim=0).numpy() # (531, 10)
    
    # Pick best start for each basin
    best_start_idx = all_fitness.argmax(axis=1) # (531,)
    ic_best_latent = all_latent[torch.arange(n_basins), torch.as_tensor(best_start_idx)] # (531, dim)
    ic_best_norm = torch.sigmoid(ic_best_latent).numpy() # (531, dim)
    
    # --- Load dPL ---
    ckpt = torch.load(dpl_ckpts[model_name], map_location="cpu", weights_only=False)
    net = CatchmentParameterizer(attrs_cuda.shape[1], NPARAM_INFO_36[model_name], hidden_dims=[256, 256], dropout=0.05).to(dtype=torch.float64)
    net.load_state_dict(ckpt["network"])
    net.eval()
    with torch.no_grad():
        dpl_norm = net(attrs_cuda.to(dtype=torch.float64)).numpy() # (531, dim)
        
    # 1. Boundary audit per parameter (normalized <= 0.01 or >= 0.99)
    for p_idx, p_name in enumerate(param_names):
        ic_vals = ic_best_norm[:, p_idx]
        dpl_vals = dpl_norm[:, p_idx]
        
        ic_low = (ic_vals <= 0.01).mean()
        ic_high = (ic_vals >= 0.99).mean()
        ic_tot = ic_low + ic_high
        
        dpl_low = (dpl_vals <= 0.01).mean()
        dpl_high = (dpl_vals >= 0.99).mean()
        dpl_tot = dpl_low + dpl_high
        
        boundary_rows.append({
            "model": model_name,
            "parameter": p_name,
            "bounds": f"[{bounds_dict[p_name][0]}, {bounds_dict[p_name][1]}]",
            "ic_lower_frac": ic_low,
            "ic_upper_frac": ic_high,
            "ic_total_frac": ic_tot,
            "dpl_lower_frac": dpl_low,
            "dpl_upper_frac": dpl_high,
            "dpl_total_frac": dpl_tot,
            "ic_gt_dpl": ic_tot >= dpl_tot,
        })
        
    # 2. IC restart divergence (10 starts distance and train KGE spread)
    all_norm_starts = torch.sigmoid(all_latent).numpy() # (531, 10, dim)
    pairwise_dists = []
    for b in range(n_basins):
        starts_b = all_norm_starts[b] # (10, dim)
        # Pairwise euclidean distances
        d_mat = []
        for i in range(10):
            for j in range(i+1, 10):
                d_mat.append(np.linalg.norm(starts_b[i] - starts_b[j]))
        pairwise_dists.append(np.mean(d_mat))
    pairwise_dists = np.array(pairwise_dists)
    
    # Train KGE spread across 10 starts
    kge_spreads = all_fitness.max(axis=1) - all_fitness.min(axis=1) # (531,)
    kge_stds = all_fitness.std(axis=1)
    
    divergence_rows.append({
        "model": model_name,
        "param_dist_mean": pairwise_dists.mean(),
        "param_dist_median": np.median(pairwise_dists),
        "param_dist_iqr": stats.iqr(pairwise_dists),
        "kge_spread_median": np.median(kge_spreads),
        "kge_spread_mean": kge_spreads.mean(),
        "kge_spread_iqr": stats.iqr(kge_spreads),
        "kge_std_median": np.median(kge_stds),
    })
    
    # 3. Slow flow time constant (1/ks)
    # Identify slow flow recession parameter
    slow_param_name = "ks" if "ks" in param_names else "k1" if "k1" in param_names else "kg" if "kg" in param_names else None
    if slow_param_name is not None:
        p_idx = param_names.index(slow_param_name)
        # IC physical ks
        ic_norm_ks = ic_best_norm[:, p_idx]
        ic_phys_ks = np.array([to_physical(v, bounds_dict[slow_param_name], "linear") for v in ic_norm_ks])
        
        # Avoid division by zero
        safe_ks = np.clip(ic_phys_ks, 1e-6, None)
        tau_slow = 1.0 / safe_ks # days
        
        # Exceedance fractions
        gt_122 = (tau_slow > (365.0 / 3.0)).mean()
        gt_183 = (tau_slow > (365.0 / 2.0)).mean()
        gt_365 = (tau_slow > 365.0).mean()
        
        slowflow_rows.append({
            "model": model_name,
            "slow_parameter": slow_param_name,
            "bounds": f"[{bounds_dict[slow_param_name][0]}, {bounds_dict[slow_param_name][1]}]",
            "ks_phys_median": np.median(ic_phys_ks),
            "ks_phys_mean": ic_phys_ks.mean(),
            "tau_median_days": np.median(tau_slow),
            "tau_iqr_days": stats.iqr(tau_slow),
            "tau_min_days": tau_slow.min(),
            "tau_max_days": tau_slow.max(),
            "frac_gt_122d_1_3yr": gt_122,
            "frac_gt_183d_1_2yr": gt_183,
            "frac_gt_365d_1yr": gt_365,
        })

# Save boundary paired CSV
df_boundary = pd.DataFrame(boundary_rows)
csv_boundary = OUT_DIR / "agent3_boundary_paired.csv"
df_boundary.to_csv(csv_boundary, index=False)
print(f"Saved {len(df_boundary)} parameter boundary rows to {csv_boundary}")

df_div = pd.DataFrame(divergence_rows)
df_slow = pd.DataFrame(slowflow_rows)

print("\n=== IC Restart Divergence Summary ===")
print(df_div.to_string())

print("\n=== Slow Flow Time Constant (1/ks) Summary ===")
print(df_slow.to_string())

# Write agent3_parameter_audit.md
audit_md = OUT_DIR / "agent3_parameter_audit.md"
with open(audit_md, "w") as f:
    f.write("# Agent-3: 参数层配对对照审计报告 (agent3_parameter_audit.md)\n\n")
    f.write("- **评估日期**：2026-08-31\n")
    f.write("- **样本量**：N = 531 个基准流域\n")
    f.write("- **贴边定义**：Normalized parameter $\\theta_{\\text{norm}} \\le 0.01$（下界）或 $\\ge 0.99$（上界）\n\n")
    
    f.write("## 1. 逐参数边界比例配对 (Boundary Fraction Paired)\n\n")
    f.write("| 模型 | 参数 | 参数物理范围 | IC 下界% | IC 上界% | IC 总贴边% | dPL 下界% | dPL 上界% | dPL 总贴边% | IC ≥ dPL (H4判定) |\n")
    f.write("|---|---|---|---:|---:|---:|---:|---:|---:|:---:|\n")
    for r in boundary_rows:
        h4_tag = "★ IC贴边更高" if r["ic_gt_dpl"] else "dPL贴边更高"
        f.write(f"| `{r['model']}` | `{r['parameter']}` | {r['bounds']} | {r['ic_lower_frac']*100:.1f}% | {r['ic_upper_frac']*100:.1f}% | **{r['ic_total_frac']*100:.1f}%** | {r['dpl_lower_frac']*100:.1f}% | {r['dpl_upper_frac']*100:.1f}% | **{r['dpl_total_frac']*100:.1f}%** | {h4_tag} |\n")
        
    f.write("\n## 2. IC 重启间多解性散度 (Cross-Start Parameter Dispersion)\n\n")
    f.write("统计 10 个独立 CMA-ES restart 之间的欧式参数距离与训练 KGE 差异：\n\n")
    f.write("| 模型 | 参数维度 | 10-start 参数距离中位数 (IQR) | 10-start 参数距离均值 | 10-start 训练KGE Spread 中位数 (IQR) | 10-start 训练KGE Std 中位数 |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for r in divergence_rows:
        f.write(f"| `{r['model']}` | {NPARAM_INFO_36[r['model']]} | {r['param_dist_median']:.4f} ({r['param_dist_iqr']:.4f}) | {r['param_dist_mean']:.4f} | {r['kge_spread_median']:.4f} ({r['kge_spread_iqr']:.4f}) | {r['kge_std_median']:.4f} |\n")
        
    f.write("\n## 3. 慢流消退时间常数分布 ($\\tau_{\\text{slow}} = 1/k_s$)\n\n")
    f.write("在 IC 最优 $\\theta$ 下，慢流蓄水库出流时间常数 $\\tau = 1/k_s$（天）的分位数与跨尺度超标统计：\n\n")
    f.write("| 模型 | 慢流参数 | 物理参数范围 | $k_s$ 物理中位数 | $\\tau$ 中位数 (天) | $\\tau$ IQR (天) | $\\tau > 122$天 (>1/3年) | $\\tau > 183$天 (>1/2年) | $\\tau > 365$天 (>1年) |\n")
    f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in slowflow_rows:
        f.write(f"| `{r['model']}` | `{r['slow_parameter']}` | {r['bounds']} | {r['ks_phys_median']:.4f} | {r['tau_median_days']:.1f} | {r['tau_iqr_days']:.1f} | **{r['frac_gt_122d_1_3yr']*100:.1f}%** | **{r['frac_gt_183d_1_2yr']*100:.1f}%** | **{r['frac_gt_365d_1yr']*100:.1f}%** |\n")

print(f"Wrote {audit_md}")
