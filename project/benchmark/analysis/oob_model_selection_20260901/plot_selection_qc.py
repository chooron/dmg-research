#!/usr/bin/env python3
"""Create two lightweight QC figures for the frozen representative selection."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from selection_common import OUT, write_csv  # noqa: E402


def main() -> None:
    features = pd.read_csv(OUT / "OOB_MODEL_SELECTION_FEATURES.csv")
    selected = features[features.selected_primary_8].copy().sort_values("primary_role")
    scatter_data = features[["model", "G", "R", "quadrant", "primary_role", "selected_primary_8", "selected_fallback_6"]]
    write_csv(OUT / "qc_G_vs_reproducibility_selected_data.csv", scatter_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(features.G, features.R, s=22, color="lightgray", label="candidate pool")
    ax.scatter(selected.G, selected.R, s=48, color="tab:red", edgecolor="black", label="primary 8")
    for row in selected.itertuples():
        ax.annotate(row.model, (row.G, row.R), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.axvline(features.G.median(), color="black", lw=.8, ls="--")
    ax.axhline(features.R.median(), color="black", lw=.8, ls="--")
    ax.set_xlabel("G_seen = median basin-level (IC KGE - dPL KGE)")
    ax.set_ylabel("Parameter–attribute reproducibility R")
    ax.set_title("Rule-based OOB candidate selection in G×R space")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "qc_G_vs_reproducibility_selected.png", dpi=170); plt.close(fig)

    rank_cols = ["G_pct", "R_pct", "D_pct", "U_pct", "K_joint_pct", "P_pct", "A_pct"]
    heat = selected.set_index("primary_role")[rank_cols]
    write_csv(OUT / "qc_selected_feature_heatmap_data.csv", heat.reset_index())
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(heat))); ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_xticks(range(len(rank_cols))); ax.set_xticklabels([x.removesuffix("_pct") for x in rank_cols], rotation=35, ha="right")
    ax.set_title("Selected-model percentile-rank feature coverage")
    fig.colorbar(im, ax=ax, label="0–1 percentile rank")
    fig.tight_layout(); fig.savefig(OUT / "qc_selected_feature_heatmap.png", dpi=170); plt.close(fig)
    print("Selection QC figures complete", len(selected))


if __name__ == "__main__":
    main()
