#!/usr/bin/env python3
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/home/jingxin/code/dmg-research/project/hydrodiag")
DATA_DIR = Path("/home/jingxin/code/dmg-research/data")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results" / "dpl_camels_531_lite_v2"

# 1. Load 531 basin IDs
with open(DATA_DIR / "531sub_id.txt", "r") as f:
    text = f.read().strip()
    try:
        sub531_ids = [str(x).zfill(8) for x in json.loads(text)]
    except Exception:
        sub531_ids = [line.strip().zfill(8) for line in text.splitlines() if line.strip()]

print(f"1. 531 Basin IDs loaded: {len(sub531_ids)}")

# 2. Load frac_snow from R1's per_basin_snow_stratified_gain.csv
snow_gain_csv = RESULTS_DIR / "per_basin_snow_stratified_gain.csv"
if snow_gain_csv.exists():
    df_snow_gain = pd.read_csv(snow_gain_csv)
    df_snow_gain["basin_id"] = df_snow_gain["basin_id"].astype(str).str.zfill(8)
    snow_map = df_snow_gain.set_index("basin_id")["frac_snow"]
else:
    from scripts.analyze_snow_stratified_gain import load_dataset_attributes
    sub_ids, snow_vals, _ = load_dataset_attributes()
    snow_map = pd.Series(snow_vals, index=sub_ids)

frac_snow = snow_map.reindex(sub531_ids)
print(f"2. frac_snow mapped for {len(frac_snow.dropna())} / 531 basins.")

# 3. Load IC Median-of-3 Starts files
ic_xaj = pd.read_csv(OUTPUTS_DIR / "XAJ_ic_median_and_best_of_3.csv")
ic_xaj["basin_id"] = ic_xaj["basin_id"].astype(str).str.zfill(8)
ic_xaj = ic_xaj.set_index("basin_id")

ic_xaj_cn = pd.read_csv(OUTPUTS_DIR / "XAJ_CN_ic_median_and_best_of_3.csv")
ic_xaj_cn["basin_id"] = ic_xaj_cn["basin_id"].astype(str).str.zfill(8)
ic_xaj_cn = ic_xaj_cn.set_index("basin_id")

ic_xaj_tgd = pd.read_csv(OUTPUTS_DIR / "XAJ_TGD_ic_median_and_best_of_3.csv")
ic_xaj_tgd["basin_id"] = ic_xaj_tgd["basin_id"].astype(str).str.zfill(8)
ic_xaj_tgd = ic_xaj_tgd.set_index("basin_id")

# 4. Load dPL 3-Seed Median files & calculate basin-means for sanity check
def get_dpl_basin_data(model_name):
    dfs = []
    for s in [42, 123, 2026]:
        p = RESULTS_DIR / model_name / f"seed_{s}" / "train_test_kge_by_basin.csv"
        df = pd.read_csv(p)
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
        dfs.append(df.set_index("basin_id"))
    
    # Per-basin 3-seed median
    tr_med = pd.concat([d["train_kge"] for d in dfs], axis=1).median(axis=1)
    te_med = pd.concat([d["test_kge"] for d in dfs], axis=1).median(axis=1)
    
    # Per-basin 3-seed mean (for sanity check verification against three_seed_train_test_kge_summary.csv)
    tr_mean = pd.concat([d["train_kge"] for d in dfs], axis=1).mean(axis=1)
    te_mean = pd.concat([d["test_kge"] for d in dfs], axis=1).mean(axis=1)
    
    return pd.DataFrame({"tr_med": tr_med, "te_med": te_med, "tr_mean": tr_mean, "te_mean": te_mean})

dpl_xaj = get_dpl_basin_data("XAJ")
dpl_xaj_cn = get_dpl_basin_data("XAJ_CN")
dpl_xaj_tgd = get_dpl_basin_data("XAJ_TGD")

print("\n=== SANITY CHECK: dPL Basin-Mean Verification ===")
print("Expected Official values:")
print("  XAJ:     Train Mean = 0.6429, Test Mean = 0.5960")
print("  XAJ_CN:  Train Mean = 0.7710, Test Mean = 0.6981")
print("  XAJ_TGD: Train Mean = 0.7462, Test Mean = 0.6821")
print("Calculated dPL values:")
print(f"  XAJ:     Train Mean = {dpl_xaj['tr_mean'].mean():.4f}, Test Mean = {dpl_xaj['te_mean'].mean():.4f}")
print(f"  XAJ_CN:  Train Mean = {dpl_xaj_cn['tr_mean'].mean():.4f}, Test Mean = {dpl_xaj_cn['te_mean'].mean():.4f}")
print(f"  XAJ_TGD: Train Mean = {dpl_xaj_tgd['tr_mean'].mean():.4f}, Test Mean = {dpl_xaj_tgd['te_mean'].mean():.4f}")

# 5. Build Aligned Master Dataframe (Median-of-3 for BOTH IC and dPL)
df_master = pd.DataFrame({"frac_snow": frac_snow}, index=sub531_ids)

# Fixed snow strata bins
fixed_edges = [0.0, 0.05, 0.15, 0.30, 0.50, 1.0001]
fixed_labels = ["[0.00, 0.05)", "[0.05, 0.15)", "[0.15, 0.30)", "[0.30, 0.50)", "[0.50, 1.00]"]
df_master["snow_bin"] = pd.cut(df_master["frac_snow"], bins=fixed_edges, right=False, labels=fixed_labels)

# IC Median-of-3 columns
df_master["ic_xaj_tr"] = ic_xaj["median_train_kge"]
df_master["ic_xaj_te"] = ic_xaj["median_test_kge"]
df_master["ic_xaj_cn_tr"] = ic_xaj_cn["median_train_kge"]
df_master["ic_xaj_cn_te"] = ic_xaj_cn["median_test_kge"]
df_master["ic_xaj_tgd_tr"] = ic_xaj_tgd["median_train_kge"]
df_master["ic_xaj_tgd_te"] = ic_xaj_tgd["median_test_kge"]

# Also record Best-of-3 for comparison delta
df_master["ic_xaj_tr_best"] = ic_xaj["best_train_kge"]
df_master["ic_xaj_te_best"] = ic_xaj["best_test_kge"]
df_master["ic_xaj_cn_tr_best"] = ic_xaj_cn["best_train_kge"]
df_master["ic_xaj_cn_te_best"] = ic_xaj_cn["best_test_kge"]
df_master["ic_xaj_tgd_tr_best"] = ic_xaj_tgd["best_train_kge"]
df_master["ic_xaj_tgd_te_best"] = ic_xaj_tgd["best_test_kge"]

# dPL Median-of-3 columns
df_master["dpl_xaj_tr"] = dpl_xaj["tr_med"]
df_master["dpl_xaj_te"] = dpl_xaj["te_med"]
df_master["dpl_xaj_cn_tr"] = dpl_xaj_cn["tr_med"]
df_master["dpl_xaj_cn_te"] = dpl_xaj_cn["te_med"]
df_master["dpl_xaj_tgd_tr"] = dpl_xaj_tgd["tr_med"]
df_master["dpl_xaj_tgd_te"] = dpl_xaj_tgd["te_med"]

# Generalization drops (tr - te)
df_master["ic_xaj_drop"] = df_master["ic_xaj_tr"] - df_master["ic_xaj_te"]
df_master["ic_xaj_cn_drop"] = df_master["ic_xaj_cn_tr"] - df_master["ic_xaj_cn_te"]
df_master["ic_xaj_tgd_drop"] = df_master["ic_xaj_tgd_tr"] - df_master["ic_xaj_tgd_te"]

df_master["dpl_xaj_drop"] = df_master["dpl_xaj_tr"] - df_master["dpl_xaj_te"]
df_master["dpl_xaj_cn_drop"] = df_master["dpl_xaj_cn_tr"] - df_master["dpl_xaj_cn_te"]
df_master["dpl_xaj_tgd_drop"] = df_master["dpl_xaj_tgd_tr"] - df_master["dpl_xaj_tgd_te"]

# Save aligned master dataset
master_csv_path = OUTPUTS_DIR / "aligned_median_of_3_master_531.csv"
df_master.to_csv(master_csv_path)
print(f"\n3. Aligned Master Dataset saved to {master_csv_path} (Merge count: {len(df_master)} / 531 basins)")

def format_med_iqr(series):
    v = series.dropna()
    med = v.median()
    q75, q25 = np.percentile(v, [75, 25])
    return f"{med:.4f} [{q75-q25:.4f}]"

print("\n=== ALL 531 BASINS ALIGNED MEDIAN-OF-3 MATRIX ===")
print(f"{'Paradigm / Model':20s} | {'Train Med [IQR]':20s} | {'Test Med [IQR]':20s} | {'Drop Med [IQR]':20s}")
print("-" * 88)

models_cfg = [
    ("IC XAJ Base", "ic_xaj_tr", "ic_xaj_te", "ic_xaj_drop"),
    ("IC XAJ_CN", "ic_xaj_cn_tr", "ic_xaj_cn_te", "ic_xaj_cn_drop"),
    ("IC XAJ_TGD", "ic_xaj_tgd_tr", "ic_xaj_tgd_te", "ic_xaj_tgd_drop"),
    ("dPL XAJ Base", "dpl_xaj_tr", "dpl_xaj_te", "dpl_xaj_drop"),
    ("dPL XAJ_CN", "dpl_xaj_cn_tr", "dpl_xaj_cn_te", "dpl_xaj_cn_drop"),
    ("dPL XAJ_TGD", "dpl_xaj_tgd_tr", "dpl_xaj_tgd_te", "dpl_xaj_tgd_drop"),
]

for label, tr_c, te_c, dr_c in models_cfg:
    print(f"{label:20s} | {format_med_iqr(df_master[tr_c]):20s} | {format_med_iqr(df_master[te_c]):20s} | {format_med_iqr(df_master[dr_c]):20s}")

print("\n=== MATERIAL CHANGES: BEST-OF-3 vs MEDIAN-OF-3 IC ===")
for m_name, tr_med_col, tr_best_col, te_med_col, te_best_col in [
    ("XAJ Base", "ic_xaj_tr", "ic_xaj_tr_best", "ic_xaj_te", "ic_xaj_te_best"),
    ("XAJ_CN", "ic_xaj_cn_tr", "ic_xaj_cn_tr_best", "ic_xaj_cn_te", "ic_xaj_cn_te_best"),
    ("XAJ_TGD", "ic_xaj_tgd_tr", "ic_xaj_tgd_tr_best", "ic_xaj_tgd_te", "ic_xaj_tgd_te_best"),
]:
    tr_diff = df_master[tr_med_col].median() - df_master[tr_best_col].median()
    te_diff = df_master[te_med_col].median() - df_master[te_best_col].median()
    print(f"  {m_name:10s} | Train KGE: Best-of-3 = {df_master[tr_best_col].median():.4f} -> Median-of-3 = {df_master[tr_med_col].median():.4f} (Shift: {tr_diff:+.4f})")
    print(f"             | Test  KGE: Best-of-3 = {df_master[te_best_col].median():.4f} -> Median-of-3 = {df_master[te_med_col].median():.4f} (Shift: {te_diff:+.4f})")
