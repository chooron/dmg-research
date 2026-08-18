#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jingxin/code/dmg-research/project/hydrodiag")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

df_master = pd.read_csv(OUTPUTS_DIR / "aligned_median_of_3_master_531.csv", index_col=0)
df_master.index = df_master.index.astype(str).str.zfill(8)

df_audit_base = pd.read_csv(OUTPUTS_DIR / "XAJ_convergence_audit.csv", index_col=0)
df_audit_base.index = df_audit_base.index.astype(str).str.zfill(8)

df_audit_cn = pd.read_csv(OUTPUTS_DIR / "XAJ_CN_convergence_audit.csv", index_col=0)
df_audit_cn.index = df_audit_cn.index.astype(str).str.zfill(8)

df_audit_tgd = pd.read_csv(OUTPUTS_DIR / "XAJ_TGD_convergence_audit.csv", index_col=0)
df_audit_tgd.index = df_audit_tgd.index.astype(str).str.zfill(8)

fixed_labels = [
    "[0.00, 0.05)",
    "[0.05, 0.15)",
    "[0.15, 0.30)",
    "[0.30, 0.50)",
    "[0.50, 1.00]",
]


def iqr_str(series):
    v = series.dropna()
    med = v.median()
    q75, q25 = np.percentile(v, [75, 25])
    return f"{med:.4f} [{q75 - q25:.4f}]"


print(
    "================================================================================"
)
print("1. MULTI-START SPREAD BY SNOW STRATA (CRITERION 1)")
print(
    "================================================================================"
)

for bin_label in fixed_labels:
    b_ids = df_master[df_master["snow_bin"] == bin_label].index
    n = len(b_ids)

    b_sp = df_audit_base.loc[b_ids, "kge_spread"]
    cn_sp = df_audit_cn.loc[b_ids, "kge_spread"]
    tgd_sp = df_audit_tgd.loc[b_ids, "kge_spread"]

    b_sub01 = (b_sp < 0.01).mean() * 100
    cn_sub01 = (cn_sp < 0.01).mean() * 100
    tgd_sub01 = (tgd_sp < 0.01).mean() * 100

    print(f"Stratum {bin_label:15s} (N={n:3d}):")
    print(f"  XAJ Base: Spread Med = {iqr_str(b_sp)} | Spread < 0.01: {b_sub01:.1f}%")
    print(f"  XAJ_CN:   Spread Med = {iqr_str(cn_sp)} | Spread < 0.01: {cn_sub01:.1f}%")
    print(
        f"  XAJ_TGD:  Spread Med = {iqr_str(tgd_sp)} | Spread < 0.01: {tgd_sub01:.1f}%\n"
    )

print(
    "================================================================================"
)
print("2. HIGH SNOW (f_snow >= 0.30, N=89) DETAILED AUDIT")
print(
    "================================================================================"
)
high_ids = df_master[df_master["frac_snow"] >= 0.30].index

print("XAJ Base High Snow:")
print(
    f"  KGE Spread Med: {df_audit_base.loc[high_ids, 'kge_spread'].median():.6f} | Spread < 0.01: {(df_audit_base.loc[high_ids, 'kge_spread'] < 0.01).mean() * 100:.1f}%"
)
print(
    f"  Plateau Gen Med: {df_audit_base.loc[high_ids, 'avg_plateau_gen'].median():.1f} / 300"
)
print(
    f"  Param Dist Med: {df_audit_base.loc[high_ids, 'param_dist_mean'].median():.4f}"
)

print("\nXAJ_CN High Snow:")
print(
    f"  KGE Spread Med: {df_audit_cn.loc[high_ids, 'kge_spread'].median():.6f} | Spread < 0.01: {(df_audit_cn.loc[high_ids, 'kge_spread'] < 0.01).mean() * 100:.1f}%"
)
print(
    f"  Plateau Gen Med: {df_audit_cn.loc[high_ids, 'avg_plateau_gen'].median():.1f} / 300"
)
print(f"  Param Dist Med: {df_audit_cn.loc[high_ids, 'param_dist_mean'].median():.4f}")

print("\nXAJ_TGD High Snow:")
print(
    f"  KGE Spread Med: {df_audit_tgd.loc[high_ids, 'kge_spread'].median():.6f} | Spread < 0.01: {(df_audit_tgd.loc[high_ids, 'kge_spread'] < 0.01).mean() * 100:.1f}%"
)
print(
    f"  Plateau Gen Med: {df_audit_tgd.loc[high_ids, 'avg_plateau_gen'].median():.1f} / 300"
)
print(f"  Param Dist Med: {df_audit_tgd.loc[high_ids, 'param_dist_mean'].median():.4f}")
