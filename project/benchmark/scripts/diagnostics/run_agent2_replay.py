#!/usr/bin/env python3
"""Agent-2 2x2 Protocol Replay Runner.

Runs 5 models (flexb, flexi, flexis, hbv96, xinanjiang) on:
1. Continuous Protocol (1981-10..1995-09, single 365d warmup from 1980-10)
2. Windowed Protocol (14 annual 730d windows: 365d warmup + 365d scoring, averaged)
using both IC frozen best-of-10 theta and dPL best-checkpoint theta.
"""
from __future__ import annotations

import os
import sys
import time
import pickle
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from dmotpy.models.hydrology_model import HydrologyModel
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.model_registry import build_model, get_spec, NPARAM_INFO_36
from src.data_selection import load_ids, frozen_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = BENCHMARK_ROOT / "results/flex_protocol_diagnosis_20260831"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
DATA_DIR = Path("/home/jingxin/code/dmg-research/data")
basin_ids = load_ids(DATA_DIR / "531sub_id.txt")
n_basins = len(basin_ids)
print(f"Loaded {n_basins} basin IDs")

dataset_pkl = DATA_DIR / "camels_dataset.pkl" if (DATA_DIR / "camels_dataset.pkl").exists() else DATA_DIR / "camels_dataset"
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
sub_indices = [id_map[int(b)] for b in basin_ids]

forcings = forcings_raw[sub_indices]  # (531, 12418, 3)
streamflow = streamflow_raw[sub_indices]  # (531, 12418, 1)

forcings = np.transpose(forcings, (1, 0, 2))  # (12418, 531, 3)
streamflow = np.transpose(streamflow[:, :, 0], (1, 0))  # (12418, 531)

# Area conversion
attr_builder = CatchmentAttributeBuilder()
raw_attr = attr_builder.load_raw_attributes(basin_ids)
area_km2 = raw_attr[:, 11]  # area_gages2 at col 11
FT3S_TO_MMD = 0.0283168 * 86400.0 * 1000.0
conv_factor = FT3S_TO_MMD / (area_km2 * 1.0e6)
streamflow_mmd = streamflow * conv_factor[None, :]

dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
def get_slice(s_str, e_str):
    left = int(dates.get_loc(pd.Timestamp(s_str)))
    right = int(dates.get_loc(pd.Timestamp(e_str))) + 1
    return left, right

# Define 14 hydrological years
hydrological_years = []
curr_start = pd.Timestamp("1980-10-01")
for yr in range(14):
    ws = curr_start
    we = ws + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    ss = we + pd.Timedelta(days=1)
    se = ss + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    hydrological_years.append((
        ws.strftime("%Y-%m-%d"), we.strftime("%Y-%m-%d"),
        ss.strftime("%Y-%m-%d"), se.strftime("%Y-%m-%d")
    ))
    curr_start = curr_start + pd.DateOffset(years=1)

# Continuous full slice
c_warm_start, c_warm_end = "1980-10-01", "1981-09-30"
c_score_start, c_score_end = "1981-10-01", "1995-09-30"
c_full_left, c_full_right = get_slice(c_warm_start, c_score_end)
c_score_left, c_score_right = get_slice(c_score_start, c_score_end)
c_warmup_days = get_slice(c_warm_start, c_warm_end)[1] - get_slice(c_warm_start, c_warm_end)[0]

cont_x = torch.as_tensor(forcings[c_full_left:c_full_right], dtype=torch.float64, device=DEVICE)
cont_y = torch.as_tensor(streamflow_mmd[c_score_left:c_score_right], dtype=torch.float64, device=DEVICE)

# Prepare 14 window slices
window_slices = []
for ws, we, ss, se in hydrological_years:
    w_left, w_right = get_slice(ws, se)
    s_left, s_right = get_slice(ss, se)
    w_days = get_slice(ws, we)[1] - get_slice(ws, we)[0]
    wx = torch.as_tensor(forcings[w_left:w_right], dtype=torch.float64, device=DEVICE)
    wy = torch.as_tensor(streamflow_mmd[s_left:s_right], dtype=torch.float64, device=DEVICE)
    window_slices.append((wx, wy, w_days))

# KGE component evaluation
def compute_kge_metrics(pred: torch.Tensor, obs: torch.Tensor, eps: float = 0.1):
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

# Model setup
models = ["flexb", "flexi", "flexis", "hbv96", "xinanjiang"]
ic_base = Path("/home/jingxin/code/dmg-research/project/benchmark/results/ic_dpl_aligned_full300_20260819_final/checkpoints/ic_dpl_aligned_full300_20260819")
dpl_ckpts = {
    "flexb": Path("/home/jingxin/code/dmg-research/project/benchmark/results/dpl_flexb_retrain_20260830/auto100/checkpoints/flexb/epoch_060.pt"),
    "flexi": Path("/home/jingxin/code/dmg-research/project/benchmark/results/dpl_full_retrain_20260813/auto100/checkpoints/flexi/epoch_060.pt"),
    "flexis": Path("/home/jingxin/code/dmg-research/project/benchmark/results/dpl_full_retrain_20260813/auto100/checkpoints/flexis/epoch_070.pt"),
    "hbv96": Path("/home/jingxin/code/dmg-research/project/benchmark/results/dpl_full_retrain_20260813/auto100/checkpoints/hbv96/epoch_050.pt"),
    "xinanjiang": Path("/home/jingxin/code/dmg-research/project/benchmark/results/dpl_full_retrain_20260813/auto100/checkpoints/xinanjiang/epoch_050.pt"),
}

attrs_cuda = attr_builder.build_normalized_attributes(basin_ids, device=DEVICE, method="zscore")

# Helper to compute physical parameters
def to_physical(norm_val: float, bounds: list[float], mapping: str = "linear") -> float:
    lower, upper = bounds[0], bounds[1]
    if mapping == "auto" and lower > 0 and upper / lower >= 100.0:
        return float(np.exp(np.log(lower) + norm_val * (np.log(upper) - np.log(lower))))
    return float(lower + norm_val * (upper - lower))

all_rows = []

print("\n=== Executing 2x2 Protocol Replay ===")

for model_name in models:
    t0 = time.time()
    spec = get_spec(model_name)
    param_names = spec.parameter_names
    bounds_dict = {name: [float(spec.bounds[i, 0]), float(spec.bounds[i, 1])] for i, name in enumerate(param_names)}
    
    # 1. Load IC parameters
    m_ic_dir = ic_base / model_name
    _, ic_latent, _ = frozen_parameters(m_ic_dir, 300, 10)
    ic_norm = torch.sigmoid(ic_latent.to(device=DEVICE, dtype=torch.float64))  # (531, D)
    
    # 2. Load dPL parameters
    ckpt_path = dpl_ckpts[model_name]
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    net = CatchmentParameterizer(attrs_cuda.shape[1], NPARAM_INFO_36[model_name], hidden_dims=[256, 256], dropout=0.05).to(DEVICE, dtype=torch.float64)
    net.load_state_dict(ckpt["network"])
    net.eval()
    with torch.no_grad():
        dpl_norm = net(attrs_cuda.to(dtype=torch.float64))  # (531, D)
        
    # Extract physical parameters for logging
    def extract_phys_params(norm_tensor, mapping_mode):
        norm_np = norm_tensor.detach().cpu().numpy()
        phys_records = []
        for b_idx in range(n_basins):
            rec = {"ks": np.nan, "nlags": np.nan, "nlagf": np.nan}
            for p_idx, p_name in enumerate(param_names):
                val = norm_np[b_idx, p_idx]
                phys = to_physical(val, bounds_dict[p_name], mapping_mode)
                if p_name in ("ks", "k1", "kg"):
                    rec["ks"] = phys
                elif p_name in ("nlags", "maxbas"):
                    rec["nlags"] = phys
                elif p_name == "nlagf":
                    rec["nlagf"] = phys
            phys_records.append(rec)
        return phys_records

    ic_phys = extract_phys_params(ic_norm, "linear")
    dpl_phys = extract_phys_params(dpl_norm, "auto")
    
    # Run Quadrants
    # ----------------------------------------------------
    # Quadrant (a): IC theta + Continuous protocol (linear mapping)
    # ----------------------------------------------------
    m_cont_ic = build_model(model_name, DEVICE, warm_up=c_warmup_days, backend="eager", parameter_mapping="linear", dtype=torch.float64)
    with torch.inference_mode():
        raw_ic = ic_norm.unsqueeze(-1)
        q_cont_ic = m_cont_ic({"x_phy": cont_x}, (None, raw_ic))["streamflow"].squeeze(-1).squeeze(-1)
        kge_a, r_a, alpha_a, beta_a, _ = compute_kge_metrics(q_cont_ic, cont_y)
        
    # ----------------------------------------------------
    # Quadrant (b): IC theta + Windowed protocol (linear mapping)
    # ----------------------------------------------------
    kge_b_list, r_b_list, alpha_b_list, beta_b_list = [], [], [], []
    for wx, wy, w_days in window_slices:
        m_win_ic = build_model(model_name, DEVICE, warm_up=w_days, backend="eager", parameter_mapping="linear", dtype=torch.float64)
        with torch.inference_mode():
            q_win_ic = m_win_ic({"x_phy": wx}, (None, raw_ic))["streamflow"].squeeze(-1).squeeze(-1)
            k_val, r_val, a_val, b_val, _ = compute_kge_metrics(q_win_ic, wy)
            kge_b_list.append(k_val)
            r_b_list.append(r_val)
            alpha_b_list.append(a_val)
            beta_b_list.append(b_val)
    kge_b = torch.stack(kge_b_list, dim=0).mean(dim=0)
    r_b = torch.stack(r_b_list, dim=0).mean(dim=0)
    alpha_b = torch.stack(alpha_b_list, dim=0).mean(dim=0)
    beta_b = torch.stack(beta_b_list, dim=0).mean(dim=0)
    
    # ----------------------------------------------------
    # Quadrant (c): dPL theta + Continuous protocol (auto mapping)
    # ----------------------------------------------------
    m_cont_dpl = build_model(model_name, DEVICE, warm_up=c_warmup_days, backend="eager", parameter_mapping="auto", dtype=torch.float64)
    with torch.inference_mode():
        raw_dpl = dpl_norm.unsqueeze(-1)
        q_cont_dpl = m_cont_dpl({"x_phy": cont_x}, (None, raw_dpl))["streamflow"].squeeze(-1).squeeze(-1)
        kge_c, r_c, alpha_c, beta_c, _ = compute_kge_metrics(q_cont_dpl, cont_y)
        
    # ----------------------------------------------------
    # Quadrant (d): dPL theta + Windowed protocol (auto mapping)
    # ----------------------------------------------------
    kge_d_list, r_d_list, alpha_d_list, beta_d_list = [], [], [], []
    for wx, wy, w_days in window_slices:
        m_win_dpl = build_model(model_name, DEVICE, warm_up=w_days, backend="eager", parameter_mapping="auto", dtype=torch.float64)
        with torch.inference_mode():
            q_win_dpl = m_win_dpl({"x_phy": wx}, (None, raw_dpl))["streamflow"].squeeze(-1).squeeze(-1)
            k_val, r_val, a_val, b_val, _ = compute_kge_metrics(q_win_dpl, wy)
            kge_d_list.append(k_val)
            r_d_list.append(r_val)
            alpha_d_list.append(a_val)
            beta_d_list.append(b_val)
    kge_d = torch.stack(kge_d_list, dim=0).mean(dim=0)
    r_d = torch.stack(r_d_list, dim=0).mean(dim=0)
    alpha_d = torch.stack(alpha_d_list, dim=0).mean(dim=0)
    beta_d = torch.stack(beta_d_list, dim=0).mean(dim=0)
    
    # Store per-basin records
    for b_idx in range(n_basins):
        bid = int(basin_ids[b_idx])
        # (a) IC continuous
        all_rows.append({
            "model": model_name, "param_source": "IC", "protocol": "continuous",
            "basin_id": bid, "kge": float(kge_a[b_idx].item()),
            "r": float(r_a[b_idx].item()), "alpha": float(alpha_a[b_idx].item()), "beta": float(beta_a[b_idx].item()),
            "ks": ic_phys[b_idx]["ks"], "nlags": ic_phys[b_idx]["nlags"], "nlagf": ic_phys[b_idx]["nlagf"],
        })
        # (b) IC windowed
        all_rows.append({
            "model": model_name, "param_source": "IC", "protocol": "windowed",
            "basin_id": bid, "kge": float(kge_b[b_idx].item()),
            "r": float(r_b[b_idx].item()), "alpha": float(alpha_b[b_idx].item()), "beta": float(beta_b[b_idx].item()),
            "ks": ic_phys[b_idx]["ks"], "nlags": ic_phys[b_idx]["nlags"], "nlagf": ic_phys[b_idx]["nlagf"],
        })
        # (c) dPL continuous
        all_rows.append({
            "model": model_name, "param_source": "dPL", "protocol": "continuous",
            "basin_id": bid, "kge": float(kge_c[b_idx].item()),
            "r": float(r_c[b_idx].item()), "alpha": float(alpha_c[b_idx].item()), "beta": float(beta_c[b_idx].item()),
            "ks": dpl_phys[b_idx]["ks"], "nlags": dpl_phys[b_idx]["nlags"], "nlagf": dpl_phys[b_idx]["nlagf"],
        })
        # (d) dPL windowed
        all_rows.append({
            "model": model_name, "param_source": "dPL", "protocol": "windowed",
            "basin_id": bid, "kge": float(kge_d[b_idx].item()),
            "r": float(r_d[b_idx].item()), "alpha": float(alpha_d[b_idx].item()), "beta": float(beta_d[b_idx].item()),
            "ks": dpl_phys[b_idx]["ks"], "nlags": dpl_phys[b_idx]["nlags"], "nlagf": dpl_phys[b_idx]["nlagf"],
        })
        
    print(f"[{model_name}] completed in {time.time() - t0:.2f}s | IC cont/win: {kge_a.median():.4f}/{kge_b.median():.4f} | dPL cont/win: {kge_c.median():.4f}/{kge_d.median():.4f}")

# 3. Save CSV
df_replay = pd.DataFrame(all_rows)
csv_out = OUT_DIR / "agent2_protocol_replay.csv"
df_replay.to_csv(csv_out, index=False)
print(f"\nSaved per-basin replay results ({len(df_replay)} rows) to {csv_out}")

# 4. Statistical Summary
# Bootstrap helper
def bootstrap_ci(arr, n_boot=10000, ci=95):
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    low = np.percentile(boot_means, (100 - ci) / 2)
    high = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return low, high

summary_rows = []
for model_name in models:
    sub = df_replay[df_replay["model"] == model_name]
    
    # Extract (a), (b), (c), (d) per basin
    a_df = sub[(sub["param_source"] == "IC") & (sub["protocol"] == "continuous")].set_index("basin_id")
    b_df = sub[(sub["param_source"] == "IC") & (sub["protocol"] == "windowed")].set_index("basin_id")
    c_df = sub[(sub["param_source"] == "dPL") & (sub["protocol"] == "continuous")].set_index("basin_id")
    d_df = sub[(sub["param_source"] == "dPL") & (sub["protocol"] == "windowed")].set_index("basin_id")
    
    delta_ic = a_df["kge"] - b_df["kge"]  # (a) - (b)
    delta_dpl = c_df["kge"] - d_df["kge"] # (c) - (d)
    
    # Bootstrap CIs
    np.random.seed(42)
    ic_ci_low, ic_ci_high = bootstrap_ci(delta_ic.values)
    dpl_ci_low, dpl_ci_high = bootstrap_ci(delta_dpl.values)
    
    # Asymmetry test: Compare delta_ic vs delta_dpl
    # If dPL is relatively better in windowed protocol while IC is worse in windowed protocol:
    # delta_ic > 0 (IC continuous > IC windowed) whereas delta_dpl < delta_ic
    ttest_res = stats.ttest_rel(delta_ic.values, delta_dpl.values)
    wilcoxon_res = stats.wilcoxon(delta_ic.values, delta_dpl.values)
    
    summary_rows.append({
        "model": model_name,
        "IC_cont_median": a_df["kge"].median(), "IC_cont_mean": a_df["kge"].mean(),
        "IC_win_median": b_df["kge"].median(), "IC_win_mean": b_df["kge"].mean(),
        "dPL_cont_median": c_df["kge"].median(), "dPL_cont_mean": c_df["kge"].mean(),
        "dPL_win_median": d_df["kge"].median(), "dPL_win_mean": d_df["kge"].mean(),
        "Delta_IC_median": delta_ic.median(), "Delta_IC_mean": delta_ic.mean(),
        "Delta_IC_CI95": f"[{ic_ci_low:.4f}, {ic_ci_high:.4f}]",
        "Delta_IC_pos_frac": (delta_ic > 0).mean(),
        "Delta_dPL_median": delta_dpl.median(), "Delta_dPL_mean": delta_dpl.mean(),
        "Delta_dPL_CI95": f"[{dpl_ci_low:.4f}, {dpl_ci_high:.4f}]",
        "Delta_dPL_pos_frac": (delta_dpl > 0).mean(),
        "Asym_paired_t_stat": ttest_res.statistic, "Asym_paired_t_p": ttest_res.pvalue,
        "Asym_wilcoxon_stat": wilcoxon_res.statistic, "Asym_wilcoxon_p": wilcoxon_res.pvalue,
    })

df_summary = pd.DataFrame(summary_rows)
print("\n=== Protocol Effect Summary ===")
print(df_summary[["model", "Delta_IC_median", "Delta_IC_mean", "Delta_IC_CI95", "Delta_dPL_median", "Delta_dPL_mean", "Delta_dPL_CI95", "Asym_paired_t_p"]].to_string())

# Write summary markdown
summary_md = OUT_DIR / "agent2_summary.md"
with open(summary_md, "w") as f:
    f.write("# Agent-2: 2×2 协议回放实验报告 (agent2_summary.md)\n\n")
    f.write("- **评估日期**：2026-08-31\n")
    f.write("- **样本量**：N = 531 个基准流域\n")
    f.write("- **时间区间**：1980-10-01 至 1995-09-30（365天预热 + 14年率定期评分）\n")
    f.write("- **回放四格定义**：\n")
    f.write("  - (a) IC 最优 θ + 连续协议（1981-10 起 14 年连续模拟，单次预热）\n")
    f.write("  - (b) IC 最优 θ + 窗口协议（14 个 730 天年度窗口：365天预热 + 365天记分，逐年平均）\n")
    f.write("  - (c) dPL θ + 连续协议（1981-10 起 14 年连续模拟，单次预热）\n")
    f.write("  - (d) dPL θ + 窗口协议（14 个 730 天年度窗口：365天预热 + 365天记分，逐年平均）\n\n")
    f.write("## 1. 2×2 协议矩阵详细得分\n\n")
    f.write("| 模型 | (a) IC 连续 Median (Mean) | (b) IC 窗口 Median (Mean) | (c) dPL 连续 Median (Mean) | (d) dPL 窗口 Median (Mean) |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for r in summary_rows:
        f.write(f"| `{r['model']}` | {r['IC_cont_median']:.4f} ({r['IC_cont_mean']:.4f}) | {r['IC_win_median']:.4f} ({r['IC_win_mean']:.4f}) | {r['dPL_cont_median']:.4f} ({r['dPL_cont_mean']:.4f}) | {r['dPL_win_median']:.4f} ({r['dPL_win_mean']:.4f}) |\n")
        
    f.write("\n## 2. 协议效应统计量 Δ_proto\n\n")
    f.write("协议效应定义：$\\Delta_{\\text{proto}} = \\text{Continuous} - \\text{Windowed}$。\n\n")
    f.write("| 模型 | IC $\\Delta_{\\text{proto}}$ Median | IC $\\Delta_{\\text{proto}}$ Mean [95% CI] | IC 正号比例 (>0) | dPL $\\Delta_{\\text{proto}}$ Median | dPL $\\Delta_{\\text{proto}}$ Mean [95% CI] | dPL 正号比例 (>0) |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in summary_rows:
        f.write(f"| `{r['model']}` | {r['Delta_IC_median']:+.4f} | {r['Delta_IC_mean']:+.4f} {r['Delta_IC_CI95']} | {r['Delta_IC_pos_frac']*100:.1f}% | {r['Delta_dPL_median']:+.4f} | {r['Delta_dPL_mean']:+.4f} {r['Delta_dPL_CI95']} | {r['Delta_dPL_pos_frac']*100:.1f}% |\n")
        
    f.write("\n## 3. 目标不对称性检验 (Asymmetry Test)\n\n")
    f.write("检验假设：dPL 在窗口协议（其训练目标）下的相对表现是否系统性优于连续协议，而 IC 参数在窗口协议下是否表现退化。\n\n")
    f.write("| 模型 | 配对 t 统计量 (t-stat) | 配对 t 检验 p 值 | Wilcoxon W 统计量 | Wilcoxon p 值 | 不对称性结论 |\n")
    f.write("|---|---:|---:|---:|---:|:---:|\n")
    for r in summary_rows:
        sig = "极显著 (p<0.001)" if r["Asym_paired_t_p"] < 0.001 else "显著 (p<0.05)" if r["Asym_paired_t_p"] < 0.05 else "不显著"
        f.write(f"| `{r['model']}` | {r['Asym_paired_t_stat']:.3f} | {r['Asym_paired_t_p']:.3e} | {r['Asym_wilcoxon_stat']:.1f} | {r['Asym_wilcoxon_p']:.3e} | {sig} |\n")

print(f"Wrote summary to {summary_md}")
