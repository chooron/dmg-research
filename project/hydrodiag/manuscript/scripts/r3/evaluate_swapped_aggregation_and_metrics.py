#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
        sub531_ids = [
            line.strip().zfill(8) for line in text.splitlines() if line.strip()
        ]

# 2. Load IC Median-of-3 and Best-of-3 data
ic_xaj = pd.read_csv(OUTPUTS_DIR / "XAJ_ic_median_and_best_of_3.csv").set_index(
    "basin_id"
)
ic_xaj_cn = pd.read_csv(OUTPUTS_DIR / "XAJ_CN_ic_median_and_best_of_3.csv").set_index(
    "basin_id"
)
ic_xaj_tgd = pd.read_csv(OUTPUTS_DIR / "XAJ_TGD_ic_median_and_best_of_3.csv").set_index(
    "basin_id"
)

ic_xaj.index = ic_xaj.index.astype(str).str.zfill(8)
ic_xaj_cn.index = ic_xaj_cn.index.astype(str).str.zfill(8)
ic_xaj_tgd.index = ic_xaj_tgd.index.astype(str).str.zfill(8)


# 3. Load dPL 3 Seeds data to compute BOTH Median-of-3 seeds AND Best-of-3 seeds
def get_dpl_seeds_data(model_name):
    dfs = []
    for s in [42, 123, 2026]:
        p = RESULTS_DIR / model_name / f"seed_{s}" / "train_test_kge_by_basin.csv"
        df = pd.read_csv(p)
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
        dfs.append(df.set_index("basin_id"))

    tr_med = pd.concat([d["train_kge"] for d in dfs], axis=1).median(axis=1)
    te_med = pd.concat([d["test_kge"] for d in dfs], axis=1).median(axis=1)

    tr_best = pd.concat([d["train_kge"] for d in dfs], axis=1).max(axis=1)
    te_best = pd.concat([d["test_kge"] for d in dfs], axis=1).max(axis=1)

    return pd.DataFrame(
        {"tr_med": tr_med, "te_med": te_med, "tr_best": tr_best, "te_best": te_best}
    )


dpl_xaj = get_dpl_seeds_data("XAJ")
dpl_xaj_cn = get_dpl_seeds_data("XAJ_CN")
dpl_xaj_tgd = get_dpl_seeds_data("XAJ_TGD")

# 4. Master Dataframe with all 4 aggregation combinations
df = pd.DataFrame(index=sub531_ids)

# IC columns
df["ic_xaj_tr_med"] = ic_xaj["median_train_kge"]
df["ic_xaj_te_med"] = ic_xaj["median_test_kge"]
df["ic_xaj_tr_best"] = ic_xaj["best_train_kge"]
df["ic_xaj_te_best"] = ic_xaj["best_test_kge"]

df["ic_cn_tr_med"] = ic_xaj_cn["median_train_kge"]
df["ic_cn_te_med"] = ic_xaj_cn["median_test_kge"]
df["ic_cn_tr_best"] = ic_xaj_cn["best_train_kge"]
df["ic_cn_te_best"] = ic_xaj_cn["best_test_kge"]

df["ic_tgd_tr_med"] = ic_xaj_tgd["median_train_kge"]
df["ic_tgd_te_med"] = ic_xaj_tgd["median_test_kge"]
df["ic_tgd_tr_best"] = ic_xaj_tgd["best_train_kge"]
df["ic_tgd_te_best"] = ic_xaj_tgd["best_test_kge"]

# dPL columns
df["dpl_xaj_tr_med"] = dpl_xaj["tr_med"]
df["dpl_xaj_te_med"] = dpl_xaj["te_med"]
df["dpl_xaj_tr_best"] = dpl_xaj["tr_best"]
df["dpl_xaj_te_best"] = dpl_xaj["te_best"]

df["dpl_cn_tr_med"] = dpl_xaj_cn["tr_med"]
df["dpl_cn_te_med"] = dpl_xaj_cn["te_med"]
df["dpl_cn_tr_best"] = dpl_xaj_cn["tr_best"]
df["dpl_cn_te_best"] = dpl_xaj_cn["te_best"]

df["dpl_tgd_tr_med"] = dpl_xaj_tgd["tr_med"]
df["dpl_tgd_te_med"] = dpl_xaj_tgd["te_med"]
df["dpl_tgd_tr_best"] = dpl_xaj_tgd["tr_best"]
df["dpl_tgd_te_best"] = dpl_xaj_tgd["te_best"]

print(
    "================================================================================"
)
print("COMPREHENSIVE 4-COMBINATION AGGREGATION SWAP MATRIX (531 BASINS)")
print(
    "================================================================================"
)

combinations = [
    (
        "Combination 1: Original Asymmetric (IC Best-of-3 vs dPL Median-of-3)",
        "tr_best",
        "tr_med",
        "te_best",
        "te_med",
    ),
    (
        "Combination 2: Aligned Median-of-3 (IC Median-of-3 vs dPL Median-of-3)",
        "tr_med",
        "tr_med",
        "te_med",
        "te_med",
    ),
    (
        "Combination 3: Swapped Asymmetric (IC Median-of-3 vs dPL Best-of-3)",
        "tr_med",
        "tr_best",
        "te_med",
        "te_best",
    ),
    (
        "Combination 4: Aligned Best-of-3 (IC Best-of-3 vs dPL Best-of-3)",
        "tr_best",
        "tr_best",
        "te_best",
        "te_best",
    ),
]

for title, ic_tr_suf, dpl_tr_suf, ic_te_suf, dpl_te_suf in combinations:
    print(f"\n--- {title} ---")
    print(
        f"{'Model':10s} | {'IC Train Med':12s} | {'dPL Train Med':13s} | {'IC Test Med':12s} | {'dPL Test Med':13s} | {'Test Winner':12s}"
    )
    print("-" * 80)
    for m_short, m_key in [("XAJ Base", "xaj"), ("XAJ_CN", "cn"), ("XAJ_TGD", "tgd")]:
        ic_tr = df[f"ic_{m_key}_{ic_tr_suf}"].median()
        dpl_tr = df[f"dpl_{m_key}_{dpl_tr_suf}"].median()
        ic_te = df[f"ic_{m_key}_{ic_te_suf}"].median()
        dpl_te = df[f"dpl_{m_key}_{dpl_te_suf}"].median()
        winner = (
            "dPL Wins" if dpl_te > ic_te else ("IC Wins" if ic_te > dpl_te else "Tie")
        )
        diff_str = f"({dpl_te - ic_te:+.4f})"
        print(
            f"{m_short:10s} | {ic_tr:12.4f} | {dpl_tr:13.4f} | {ic_te:12.4f} | {dpl_te:13.4f} | {winner:8s} {diff_str}"
        )

# Save master swapped dataset
df.to_csv(OUTPUTS_DIR / "swapped_aggregation_master_531.csv")
print(
    f"\nSwapped Master Data saved to {OUTPUTS_DIR / 'swapped_aggregation_master_531.csv'}"
)
