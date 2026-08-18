#!/usr/bin/env python3
"""Snow-stratified, basin-paired TGD2 comparison: CMA-ES vs paused dPL."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
CMA_RAW = ROOT / "results/xaj_tgd2_cmaes_531_batched_v1/raw/xaj_tgd2"
DPL = (
    ROOT
    / "results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/interim_epoch040_evaluation/median_of_3_per_basin.csv"
)
SNOW = ROOT / "results/dpl_camels_531_lite_v2/per_basin_snow_stratified_gain.csv"
OUT = (
    ROOT
    / "results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/interim_epoch040_evaluation/cmaes_vs_dpl_snow_stratified"
)
BINS = (0.0, 0.05, 0.15, 0.30, 0.50, 1.0000001)
LABELS = ("[0, 0.05)", "[0.05, 0.15)", "[0.15, 0.30)", "[0.30, 0.50)", "[0.50, 1.0]")


def q(x: pd.Series) -> str:
    return f"{x.median():.4f} [{x.quantile(0.25):.4f}, {x.quantile(0.75):.4f}]"


def paired(d: pd.DataFrame, left: str, right: str) -> tuple[float, float, float]:
    values = (d[left] - d[right]).dropna()
    if not len(values):
        return np.nan, np.nan, np.nan
    nonzero = values[values != 0]
    p = (
        1.0
        if len(nonzero) == 0
        else float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    )
    return float(values.median()), float((values > 0).mean()), p


def main() -> None:
    rows = []
    for path in CMA_RAW.glob("*_start*.json"):
        x = json.loads(path.read_text())
        rows.append(
            {
                "basin_id": str(x["basin_id"]).zfill(8),
                "start": int(x["start"]),
                "cma_train_kge": float(x["train_metrics"]["kge"]),
                "cma_validation_kge": float(x["test_metrics"]["kge"]),
            }
        )
    cma_all = pd.DataFrame(rows)
    if len(cma_all) != 5310:
        raise RuntimeError(f"Expected 5310 CMA records, found {len(cma_all)}")
    chosen = cma_all.loc[cma_all.groupby("basin_id")["cma_train_kge"].idxmax()].copy()
    dpl = pd.read_csv(DPL, dtype={"basin_id": str}).rename(
        columns={
            "train_kge": "dpl_train_kge",
            "validation_kge": "dpl_validation_kge",
        }
    )
    dpl["basin_id"] = dpl["basin_id"].str.zfill(8)
    snow = pd.read_csv(SNOW, dtype={"basin_id": str})[["basin_id", "frac_snow"]]
    snow["basin_id"] = snow["basin_id"].str.zfill(8)
    paired_data = chosen.merge(dpl, on="basin_id", validate="one_to_one").merge(
        snow, on="basin_id", validate="one_to_one"
    )
    paired_data["snow_stratum"] = pd.cut(
        paired_data["frac_snow"],
        bins=BINS,
        labels=LABELS,
        right=False,
        include_lowest=True,
    )
    paired_data["dpl_minus_cma_train"] = (
        paired_data["dpl_train_kge"] - paired_data["cma_train_kge"]
    )
    paired_data["dpl_minus_cma_validation"] = (
        paired_data["dpl_validation_kge"] - paired_data["cma_validation_kge"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    paired_data.sort_values("basin_id").to_csv(
        OUT / "paired_per_basin.csv", index=False
    )

    groups = [("All", paired_data)] + [
        (label, paired_data[paired_data.snow_stratum == label]) for label in LABELS
    ]
    summary = []
    for label, data in groups:
        tr_delta, tr_sign, tr_p = paired(data, "dpl_train_kge", "cma_train_kge")
        va_delta, va_sign, va_p = paired(
            data, "dpl_validation_kge", "cma_validation_kge"
        )
        summary.append(
            {
                "snow_stratum": label,
                "n": len(data),
                "cma_train_kge_median_iqr": q(data.cma_train_kge),
                "dpl_train_kge_median_iqr": q(data.dpl_train_kge),
                "dpl_minus_cma_train_median": tr_delta,
                "dpl_train_win_fraction": tr_sign,
                "dpl_minus_cma_train_wilcoxon_p": tr_p,
                "cma_validation_kge_median_iqr": q(data.cma_validation_kge),
                "dpl_validation_kge_median_iqr": q(data.dpl_validation_kge),
                "dpl_minus_cma_validation_median": va_delta,
                "dpl_validation_win_fraction": va_sign,
                "dpl_minus_cma_validation_wilcoxon_p": va_p,
            }
        )
    table = pd.DataFrame(summary)
    table.to_csv(OUT / "snow_stratified_summary.csv", index=False)
    with (OUT / "README.md").open("w") as f:
        f.write("# Interim TGD2: CMA-ES versus dPL snow-stratified comparison\n\n")
        f.write(
            "CMA-ES: best of 10 starts selected solely by train KGE. dPL: paused epoch-40 "
            + "median of three seeds, each using its currently best validation checkpoint. "
            + "Thus the comparison is diagnostic, not a final unbiased optimizer contest.\n\n"
        )
        f.write(
            "`dPL - CMA` is positive when dPL has higher KGE. `win_fraction` is the fraction of paired basins with positive difference.\n\n"
        )
        # Avoid an optional tabulate dependency in the training environment.
        f.write("```text\n")
        f.write(table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        f.write("\n```\n")


if __name__ == "__main__":
    main()
