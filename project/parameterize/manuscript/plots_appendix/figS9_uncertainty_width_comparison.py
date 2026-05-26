"""Fig. S9 — Direct uncertainty width comparison: delta_mcd vs delta_dist.
Style: horizontal point-range plot identical to Fig02 panel style
(grouped by parameter, two models side by side).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (
    setup_style, clean_axes, MM,
    APP_FIG_DIR, save_fig, MODEL_COLORS,
    PARAM_ORDER, PARAMETER_TABLE, BASIN_LIST, REFERENCE_LOSS,
)

OUT_STEM = "figS9_uncertainty_width_comparison"

SPATIAL_STD_FILE = (Path(__file__).resolve().parents[2]
                    / "manuscript" / "analysis" / "06_uncertainty_spatial_data"
                    / "data" / "distributional_parameter_uncertainty_maps_long.csv")

PARAM_LABELS = {
    "parBETA": "BETA", "parFC": "FC", "parLP": "LP",
    "parPERC": "PERC", "parUZL": "UZL", "parK0": "K0",
    "parK1": "K1", "parK2": "K2", "route_a": "route_a",
    "route_b": "route_b", "parTT": "TT", "parCFMAX": "CFMAX",
    "parCFR": "CFR", "parCWH": "CWH",
}


def load_basin_ids() -> list[int]:
    text = BASIN_LIST.read_text(encoding="utf-8").strip()
    return [int(x) for x in ast.literal_eval(text)]


def load_mcd_spread(basin_ids: list[int]) -> pd.DataFrame:
    df = pd.read_csv(PARAMETER_TABLE)
    mc = "model_raw" if "model_raw" in df.columns else "model"
    vc = "estimate_norm" if "estimate_norm" in df.columns else "mean"
    sub = df[(df[mc] == "mc_dropout") & (df["loss"] == REFERENCE_LOSS)]
    agg = (sub.groupby(["basin_id", "parameter"])[vc]
           .std().reset_index().rename(columns={vc: "spread"}))
    agg["model"] = "mc_dropout"
    return agg[agg["basin_id"].isin(basin_ids)]


def load_dist_spread(basin_ids: list[int]) -> pd.DataFrame:
    std_df = pd.read_csv(SPATIAL_STD_FILE)
    sub = std_df[["basin_id", "parameter", "parameter_std_unit"]].copy()
    sub = sub.rename(columns={"parameter_std_unit": "spread"})
    sub["model"] = "distributional"
    return sub[sub["basin_id"].isin(basin_ids)]


def main() -> None:
    setup_style()
    basin_ids = load_basin_ids()
    mcd  = load_mcd_spread(basin_ids)
    dist = load_dist_spread(basin_ids)
    combined = pd.concat([mcd, dist], ignore_index=True)

    params = [p for p in PARAM_ORDER if p in combined["parameter"].values]
    models_shown = ["mc_dropout", "distributional"]
    offsets = {"mc_dropout": -0.18, "distributional": 0.18}
    model_labels_plot = {
        "mc_dropout":     r"$\delta_{mcd}$ spread (seed SD)",
        "distributional": r"$\delta_{dist}$ parameter-scale std",
    }

    fig_w = 120 * MM
    fig_h = 130 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.22, 0.12, 0.72, 0.76])

    y_pos = np.arange(len(params))

    for model in models_shown:
        offset = offsets[model]
        color  = MODEL_COLORS[model]
        sub    = combined[combined["model"] == model]

        for i, param in enumerate(params):
            vals = sub[sub["parameter"] == param]["spread"].dropna().values
            if len(vals) == 0:
                continue
            med = np.nanmedian(vals)
            q25 = np.nanpercentile(vals, 25)
            q75 = np.nanpercentile(vals, 75)
            yi  = y_pos[i] + offset

            ax.plot([q25, q75], [yi, yi], color=color, linewidth=1.8,
                    solid_capstyle="round", zorder=3)
            ax.scatter([med], [yi], s=22, facecolor="white",
                       edgecolor=color, linewidth=1.0, zorder=4)

    # Significance markers (Mann–Whitney)
    for i, param in enumerate(params):
        mcd_v  = combined[(combined["model"] == "mc_dropout") &
                          (combined["parameter"] == param)]["spread"].dropna().values
        dist_v = combined[(combined["model"] == "distributional") &
                          (combined["parameter"] == param)]["spread"].dropna().values
        if len(mcd_v) > 5 and len(dist_v) > 5:
            _, pval = mannwhitneyu(mcd_v, dist_v, alternative="two-sided")
            if pval < 0.001:
                marker = "***"
            elif pval < 0.01:
                marker = "**"
            elif pval < 0.05:
                marker = "*"
            else:
                marker = ""
            if marker:
                ax.text(1.01, y_pos[i], marker, transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=7.5, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in params], fontsize=8.0)
    ax.set_xlabel("Parameter spread (normalised range)", fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color="#D8D8D8", lw=0.75, ls=(0, (3, 3)), zorder=0)
    clean_axes(ax, grid_axis="x")

    handles = [
        mlines.Line2D([], [], color=MODEL_COLORS[m], linewidth=1.8,
                      marker="o", markersize=5, markerfacecolor="white",
                      markeredgecolor=MODEL_COLORS[m], markeredgewidth=1.0,
                      label=model_labels_plot[m])
        for m in models_shown
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.985),
               handlelength=2.0, handletextpad=0.5, columnspacing=1.4)

    fig.text(0.5, -0.04,
             "MCD spread = seed SD of sampled means; Dist. std = parameter-scale std.\n"
             "Scales differ — comparison is qualitative. * p<0.05, ** p<0.01, *** p<0.001 (Mann–Whitney).",
             ha="center", fontsize=6.5, color="#666666", style="italic")

    save_fig(fig, OUT_STEM)
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
