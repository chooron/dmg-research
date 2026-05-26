"""Fig. S2 — Boundary-threshold sensitivity.
Style: identical to Fig02 grouped boxplot (fig02_cross_seed_parameter_stability.py).
Shows near_boundary_rate distributions per parameter under three thresholds,
recomputed from raw normalized parameter values for each threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (
    setup_style, clean_axes, add_panel_label,
    MM, DPI, MODEL_ORDER, MODEL_COLORS, ANALYSIS_ROOT,
    math_model_labels, APP_FIG_DIR, save_fig, PARAM_ORDER, p_label,
)

OUT_STEM = "figS2_boundary_threshold_sensitivity"

THRESHOLDS    = [0.02, 0.05, 0.10]
THRESH_COLORS = {
    0.02: MODEL_COLORS["deterministic"],
    0.05: MODEL_COLORS["mc_dropout"],
    0.10: MODEL_COLORS["distributional"],
}
THRESH_LABELS = {0.02: "≤ 2 %",   0.05: "≤ 5 %",   0.10: "≤ 10 %"}

LONG_TABLE = (ANALYSIS_ROOT.parent.parent / "results2_parameter_reliability"
              / "tables" / "parameter_long_table.csv")


def _compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, loss, seed, parameter) group compute near_boundary_rate
    at each threshold from raw normalized parameter values."""
    records = []
    grp_cols = ["model_name", "loss_function", "seed", "parameter_name"]
    for keys, grp in df.groupby(grp_cols):
        theta = grp["normalized_parameter_value"].dropna().values
        dist  = np.minimum(theta, 1.0 - theta)
        model, loss, seed, param = keys
        for thresh in THRESHOLDS:
            rate = (dist <= thresh).mean()
            records.append({"model": model, "loss": loss, "seed": seed,
                            "parameter": param, "threshold": thresh,
                            "near_boundary_rate": rate})
    return pd.DataFrame(records)


def main() -> None:
    setup_style()
    raw = pd.read_csv(LONG_TABLE)
    raw = raw[raw["model_name"].isin(MODEL_ORDER)]
    rates = _compute_rates(raw)

    fig_w = 205 * MM
    fig_h = 118 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, len(MODEL_ORDER),
                          left=0.08, right=0.99, top=0.93, bottom=0.15,
                          hspace=0.0, wspace=0.16)

    labels = math_model_labels()
    shared_y_ax = None

    for col, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[0, col], sharey=shared_y_ax)
        if shared_y_ax is None:
            shared_y_ax = ax
        sub = rates[rates["model"] == model]

        params = [p for p in PARAM_ORDER if p in sub["parameter"].values]
        group_gap = 0.85
        box_gap   = 0.22
        width     = 0.18
        centers   = []
        positions, plot_data, colors = [], [], []

        for idx, param in enumerate(params):
            center = idx * group_gap
            centers.append(center)
            psub = sub[sub["parameter"] == param]
            for ti, thresh in enumerate(THRESHOLDS):
                pos = center + (ti - 1) * box_gap
                vals = psub[psub["threshold"] == thresh]["near_boundary_rate"].dropna().values
                positions.append(pos)
                plot_data.append(vals)
                colors.append(THRESH_COLORS[thresh])

        bp = ax.boxplot(
            plot_data, positions=positions, widths=width,
            vert=False, patch_artist=True, showfliers=False,
            medianprops={"color": "#2A2A2A", "lw": 1.0},
            whiskerprops={"color": "#666666", "lw": 0.7},
            capprops={"color": "#666666", "lw": 0.7},
            boxprops={"edgecolor": "#666666", "lw": 0.7},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)

        ax.set_xlabel("Near-boundary basin fraction")
        ax.set_xlim(-0.05, 1.05)
        ax.axvline(0.5, color="#D8D8D8", lw=0.75, ls=(0, (3, 3)), zorder=0)
        ax.set_title(labels[model], fontsize=11.0, pad=5)
        clean_axes(ax, grid_axis="x")
        ax.yaxis.set_major_locator(mticker.FixedLocator(centers))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter([p_label(p) for p in params]))
        ax.tick_params(axis="y", which="major", left=True, labelleft=(col == 0), labelsize=8.2)
        add_panel_label(ax, f"({'abc'[col]})", x=0.98, y=0.98,
                        ha="right", va="top", fontweight="normal", fontsize=10.5)

    title_handle = mpatches.Patch(visible=False, label="Boundary threshold:")
    handles = [title_handle] + [
        mpatches.Patch(facecolor=THRESH_COLORS[t], alpha=0.80,
                       edgecolor="none", label=THRESH_LABELS[t])
        for t in THRESHOLDS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, 0.01),
               handlelength=1.2, handletextpad=0.4, columnspacing=1.2)

    save_fig(fig, OUT_STEM)
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
