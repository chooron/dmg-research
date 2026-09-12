#!/usr/bin/env python3
"""Agent-4 Gap and Protocol Effect Decomposition Runner.

1. Decompose IC-dPL KGE gap on Continuous Protocol into r, alpha, beta components.
2. Decompose Protocol Effect Delta_proto (Continuous - Windowed) into r, alpha, beta components.
3. Perform Spearman correlation tests (rho + p, N=531):
   - Delta_proto(IC theta) vs 1/ks
   - Delta_proto(IC theta) vs nlags
   - Delta_proto(IC theta) vs CV_annual (annual runoff variation coefficient)
Across 5 models: flexb, flexi, flexis, hbv96, xinanjiang.
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from src.data_selection import load_ids

OUT_DIR = BENCHMARK_ROOT / "results/flex_protocol_diagnosis_20260831"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Replay CSV from Agent-2
csv_replay = OUT_DIR / "agent2_protocol_replay.csv"
df_replay = pd.read_csv(csv_replay)

# 2. Compute Annual Runoff CV per Basin
DATA_DIR = REPO_ROOT / "data"
basin_ids = load_ids(DATA_DIR / "531sub_id.txt")
n_basins = len(basin_ids)

dataset_pkl = DATA_DIR / "camels_dataset.pkl" if (DATA_DIR / "camels_dataset.pkl").exists() else DATA_DIR / "camels_dataset"
with open(dataset_pkl, "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict):
    streamflow_raw = data["streamflow"]
elif isinstance(data, (tuple, list)):
    streamflow_raw = data[1]

gage_ids_all = np.asarray(np.load(DATA_DIR / "gage_id.npy"), dtype=np.int64)
id_map = {int(gid): idx for idx, gid in enumerate(gage_ids_all)}
sub_indices = [id_map[int(b)] for b in basin_ids]
streamflow = streamflow_raw[sub_indices, :, 0] # (531, 12418)

dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
curr_start = pd.Timestamp("1981-10-01")
annual_q = []
for yr in range(14):
    ss = curr_start
    se = ss + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    s_left = int(dates.get_loc(ss))
    s_right = int(dates.get_loc(se)) + 1
    y_slice = streamflow[:, s_left:s_right]
    y_slice_masked = np.where(np.isfinite(y_slice), y_slice, np.nan)
    y_mean = np.nanmean(y_slice_masked, axis=1) * 365.25
    annual_q.append(y_mean)
    curr_start = curr_start + pd.DateOffset(years=1)

annual_q = np.stack(annual_q, axis=1) # (531, 14)
mean_q = np.nanmean(annual_q, axis=1)
std_q = np.nanstd(annual_q, axis=1, ddof=1)
cv_annual_arr = std_q / np.clip(mean_q, 1e-6, None)
basin_cv_map = {int(basin_ids[i]): float(cv_annual_arr[i]) for i in range(n_basins)}

models = ["flexb", "flexi", "flexis", "hbv96", "xinanjiang"]

# ----------------------------------------------------
# 1. IC - dPL Gap Decomposition on Continuous Protocol
# ----------------------------------------------------
gap_decomp_rows = []
for m in models:
    sub = df_replay[df_replay["model"] == m]
    a_df = sub[(sub["param_source"] == "IC") & (sub["protocol"] == "continuous")].set_index("basin_id")
    c_df = sub[(sub["param_source"] == "dPL") & (sub["protocol"] == "continuous")].set_index("basin_id")
    
    kge_ic, r_ic, a_ic, b_ic = a_df["kge"].values, a_df["r"].values, a_df["alpha"].values, a_df["beta"].values
    kge_dpl, r_dpl, a_dpl, b_dpl = c_df["kge"].values, c_df["r"].values, c_df["alpha"].values, c_df["beta"].values
    
    # Distance components squared: D^2 = (r-1)^2 + (alpha-1)^2 + (beta-1)^2
    d2_ic_r = (r_ic - 1.0)**2
    d2_ic_a = (a_ic - 1.0)**2
    d2_ic_b = (b_ic - 1.0)**2
    d2_ic_tot = d2_ic_r + d2_ic_a + d2_ic_b
    
    d2_dpl_r = (r_dpl - 1.0)**2
    d2_dpl_a = (a_dpl - 1.0)**2
    d2_dpl_b = (b_dpl - 1.0)**2
    d2_dpl_tot = d2_dpl_r + d2_dpl_a + d2_dpl_b
    
    delta_d2_r = d2_dpl_r - d2_ic_r
    delta_d2_a = d2_dpl_a - d2_ic_a
    delta_d2_b = d2_dpl_b - d2_ic_b
    delta_d2_tot = d2_dpl_tot - d2_ic_tot
    
    # Mean contributions
    mean_delta_r = delta_d2_r.mean()
    mean_delta_a = delta_d2_a.mean()
    mean_delta_b = delta_d2_b.mean()
    mean_delta_tot = delta_d2_tot.mean()
    
    frac_r = mean_delta_r / mean_delta_tot if mean_delta_tot != 0 else np.nan
    frac_a = mean_delta_a / mean_delta_tot if mean_delta_tot != 0 else np.nan
    frac_b = mean_delta_b / mean_delta_tot if mean_delta_tot != 0 else np.nan
    
    # Linear differences in components
    diff_kge = kge_ic - kge_dpl
    diff_r = r_ic - r_dpl
    diff_abs_a = np.abs(a_dpl - 1.0) - np.abs(a_ic - 1.0)
    diff_abs_b = np.abs(b_dpl - 1.0) - np.abs(b_ic - 1.0)
    
    gap_decomp_rows.append({
        "model": m,
        "delta_kge_median": np.median(diff_kge),
        "delta_kge_mean": diff_kge.mean(),
        "delta_r_mean": diff_r.mean(),
        "delta_r_median": np.median(diff_r),
        "delta_alpha_err_mean": diff_abs_a.mean(),
        "delta_beta_err_mean": diff_abs_b.mean(),
        "frac_r_contribution": frac_r,
        "frac_alpha_contribution": frac_a,
        "frac_beta_contribution": frac_b,
    })

df_gap_decomp = pd.DataFrame(gap_decomp_rows)
print("=== IC-dPL Gap Decomposition (Continuous Protocol) ===")
print(df_gap_decomp.to_string())

# ----------------------------------------------------
# 2. Protocol Effect Delta_proto Decomposition
# ----------------------------------------------------
proto_decomp_rows = []
for m in models:
    sub = df_replay[df_replay["model"] == m]
    a_df = sub[(sub["param_source"] == "IC") & (sub["protocol"] == "continuous")].set_index("basin_id")
    b_df = sub[(sub["param_source"] == "IC") & (sub["protocol"] == "windowed")].set_index("basin_id")
    
    kge_cont, r_cont, a_cont, b_cont = a_df["kge"].values, a_df["r"].values, a_df["alpha"].values, a_df["beta"].values
    kge_win, r_win, a_win, b_win = b_df["kge"].values, b_df["r"].values, b_df["alpha"].values, b_df["beta"].values
    
    # Squared distance in windowed vs continuous: Delta_proto = Cont - Win -> Win error - Cont error
    d2_cont_r = (r_cont - 1.0)**2
    d2_cont_a = (a_cont - 1.0)**2
    d2_cont_b = (b_cont - 1.0)**2
    d2_cont_tot = d2_cont_r + d2_cont_a + d2_cont_b
    
    d2_win_r = (r_win - 1.0)**2
    d2_win_a = (a_win - 1.0)**2
    d2_win_b = (b_win - 1.0)**2
    d2_win_tot = d2_win_r + d2_win_a + d2_win_b
    
    delta_d2_r = d2_win_r - d2_cont_r
    delta_d2_a = d2_win_a - d2_cont_a
    delta_d2_b = d2_win_b - d2_cont_b
    delta_d2_tot = d2_win_tot - d2_cont_tot
    
    mean_delta_r = delta_d2_r.mean()
    mean_delta_a = delta_d2_a.mean()
    mean_delta_b = delta_d2_b.mean()
    mean_delta_tot = delta_d2_tot.mean()
    
    frac_r = mean_delta_r / mean_delta_tot if mean_delta_tot != 0 else np.nan
    frac_a = mean_delta_a / mean_delta_tot if mean_delta_tot != 0 else np.nan
    frac_b = mean_delta_b / mean_delta_tot if mean_delta_tot != 0 else np.nan
    
    diff_kge = kge_cont - kge_win
    diff_r = r_cont - r_win
    diff_a = a_win - a_cont
    diff_b = b_win - b_cont
    
    proto_decomp_rows.append({
        "model": m,
        "delta_proto_kge_median": np.median(diff_kge),
        "delta_proto_kge_mean": diff_kge.mean(),
        "delta_r_mean": diff_r.mean(),
        "delta_r_median": np.median(diff_r),
        "delta_alpha_mean": diff_a.mean(),
        "delta_beta_mean": diff_b.mean(),
        "frac_r_contribution": frac_r,
        "frac_alpha_contribution": frac_a,
        "frac_beta_contribution": frac_b,
    })

df_proto_decomp = pd.DataFrame(proto_decomp_rows)
print("\n=== Protocol Effect Delta_proto Decomposition (IC theta: Cont vs Win) ===")
print(df_proto_decomp.to_string())

# ----------------------------------------------------
# 3. Correlation Tests (Spearman rho + p, N=531)
# ----------------------------------------------------
corr_rows = []
for m in models:
    sub = df_replay[df_replay["model"] == m]
    a_df = sub[(sub["param_source"] == "IC") & (sub["protocol"] == "continuous")].set_index("basin_id")
    b_df = sub[(sub["param_source"] == "IC") & (sub["protocol"] == "windowed")].set_index("basin_id")
    
    delta_proto = (a_df["kge"] - b_df["kge"]).values # (531,)
    ks_vals = a_df["ks"].values # (531,)
    nlags_vals = a_df["nlags"].values # (531,)
    cv_annual_vals = np.array([basin_cv_map[bid] for bid in a_df.index])
    
    # 1. Delta_proto vs 1/ks
    valid_ks = np.isfinite(ks_vals) & (ks_vals > 0)
    tau_slow = np.where(valid_ks, 1.0 / np.clip(ks_vals, 1e-6, None), np.nan)
    mask_tau = np.isfinite(delta_proto) & np.isfinite(tau_slow)
    n_tau = int(mask_tau.sum())
    if n_tau > 10:
        rho_tau, p_tau = stats.spearmanr(delta_proto[mask_tau], tau_slow[mask_tau])
    else:
        rho_tau, p_tau = np.nan, np.nan
        
    # 2. Delta_proto vs nlags
    mask_nlags = np.isfinite(delta_proto) & np.isfinite(nlags_vals)
    n_nlags = int(mask_nlags.sum())
    if n_nlags > 10:
        rho_nlags, p_nlags = stats.spearmanr(delta_proto[mask_nlags], nlags_vals[mask_nlags])
    else:
        rho_nlags, p_nlags = np.nan, np.nan
        
    # 3. Delta_proto vs CV_annual
    mask_cv = np.isfinite(delta_proto) & np.isfinite(cv_annual_vals)
    n_cv = int(mask_cv.sum())
    rho_cv, p_cv = stats.spearmanr(delta_proto[mask_cv], cv_annual_vals[mask_cv])
    
    corr_rows.append({
        "model": m,
        "N_tau": n_tau, "rho_tau": rho_tau, "p_tau": p_tau,
        "N_nlags": n_nlags, "rho_nlags": rho_nlags, "p_nlags": p_nlags,
        "N_cv": n_cv, "rho_cv": rho_cv, "p_cv": p_cv,
    })

df_corr = pd.DataFrame(corr_rows)
print("\n=== Correlation Tests (Spearman rho + p, N=531) ===")
print(df_corr.to_string())

# Write agent4_decomposition.md
decomp_md = OUT_DIR / "agent4_decomposition.md"
with open(decomp_md, "w") as f:
    f.write("# Agent-4: 精度落差与协议效应分量分解报告 (agent4_decomposition.md)\n\n")
    f.write("- **评估日期**：2026-08-31\n")
    f.write("- **样本量**：N = 531 个基准流域\n")
    f.write("- **分解对象**：\n")
    f.write("  1. 连续协议下 IC − dPL 的 KGE 落差分解到相关系数 $r$、变异比 $\\alpha$ (或 $\\gamma$)、均值比 $\\beta$ 三分量；\n")
    f.write("  2. IC 参数在连续协议 vs 窗口协议下的协议效应 $\\Delta_{\\text{proto}}$ 分解；\n")
    f.write("  3. 协议效应与慢流时间尺度 $1/k_s$、汇流滞时 `nlags` 及年际径流变异系数 $\\text{CV}_{\\text{annual}}$ 的 Spearman 相关性检验。\n\n")
    
    f.write("## 1. 连续协议下 IC − dPL 落差三分量贡献分解\n\n")
    f.write("| 模型 | IC − dPL $\\Delta\\text{KGE}$ 中位数 (均值) | $r$ 贡献占比 | $\\alpha$ (变异) 贡献占比 | $\\beta$ (水量) 贡献占比 | $\\Delta r$ 均值 (IC−dPL) | $\\Delta |1-\\alpha|$ 均值 | $\\Delta |1-\\beta|$ 均值 |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in gap_decomp_rows:
        f.write(f"| `{r['model']}` | {r['delta_kge_median']:+.4f} ({r['delta_kge_mean']:+.4f}) | **{r['frac_r_contribution']*100:.1f}%** | **{r['frac_alpha_contribution']*100:.1f}%** | **{r['frac_beta_contribution']*100:.1f}%** | {r['delta_r_mean']:+.4f} | {r['delta_alpha_err_mean']:+.4f} | {r['delta_beta_err_mean']:+.4f} |\n")
        
    f.write("\n## 2. IC 协议效应 $\\Delta_{\\text{proto}}$ (连续 − 窗口) 三分量贡献分解\n\n")
    f.write("| 模型 | $\\Delta_{\\text{proto}}$ 中位数 (均值) | $r$ 贡献占比 | $\\alpha$ (变异) 贡献占比 | $\\beta$ (水量) 贡献占比 | $\\Delta r$ 均值 (Cont−Win) | $\\Delta \\alpha$ 均值 (Win−Cont) | $\\Delta \\beta$ 均值 (Win−Cont) |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in proto_decomp_rows:
        f.write(f"| `{r['model']}` | {r['delta_proto_kge_median']:+.4f} ({r['delta_proto_kge_mean']:+.4f}) | **{r['frac_r_contribution']*100:.1f}%** | **{r['frac_alpha_contribution']*100:.1f}%** | **{r['frac_beta_contribution']*100:.1f}%** | {r['delta_r_mean']:+.4f} | {r['delta_alpha_mean']:+.4f} | {r['delta_beta_mean']:+.4f} |\n")
        
    f.write("\n## 3. 协议效应相关性检验 (Spearman Rank Correlation)\n\n")
    f.write("| 模型 | 对比物理量 | 样本量 N | Spearman $\\rho$ | p-value | 预注册判据判定 |\n")
    f.write("|---|---|---:|---:|---:|:---:|\n")
    for r in corr_rows:
        # Check H2 criteria: |rho| >= 0.3 and p < 0.01 for tau
        h2_pass = (abs(r["rho_tau"]) >= 0.3 and r["p_tau"] < 0.01) if not np.isnan(r["rho_tau"]) else False
        h2_note = "满足 H2 阈值" if h2_pass else "未达 H2 阈值 (|ρ|<0.3)"
        f.write(f"| `{r['model']}` | $\\Delta_{{\\text{{proto}}}}$ vs $1/k_s$ (慢流尺度) | {r['N_tau']} | {r['rho_tau']:+.4f} | {r['p_tau']:.3e} | {h2_note} |\n")
        if not np.isnan(r["rho_nlags"]):
            f.write(f"| `{r['model']}` | $\\Delta_{{\\text{{proto}}}}$ vs `nlags` (汇流滞时) | {r['N_nlags']} | {r['rho_nlags']:+.4f} | {r['p_nlags']:.3e} | — |\n")
        f.write(f"| `{r['model']}` | $\\Delta_{{\\text{{proto}}}}$ vs $\\text{{CV}}_{{\\text{{annual}}}}$ (年际变异) | {r['N_cv']} | {r['rho_cv']:+.4f} | {r['p_cv']:.3e} | — |\n")

print(f"Wrote {decomp_md}")
