#!/usr/bin/env python3
"""Render Supplementary Figure S1: the complete unsorted 36×531 ΔKGE matrix."""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from R1X_fig_utils import save_figure, setup
from r1_config import FIGURES_DIR, MODEL_REGISTRY, TABLES_DIR


def main() -> None:
    setup()
    full = pd.read_csv(TABLES_DIR / "R1_model_basin_delta_kge.csv", dtype={"model": str, "basin_id": str})
    basin_order = pd.read_csv(TABLES_DIR / "Fig1c_basin_crossmodel_tendency.csv", dtype={"basin_id": str}).sort_values("plot_order")["basin_id"].tolist()
    matrix = full.pivot(index="model", columns="basin_id", values="delta_KGE").loc[list(MODEL_REGISTRY), basin_order].to_numpy(float)
    lim = max(float(np.quantile(np.abs(matrix), 0.98)), 0.01)
    cmap = mcolors.LinearSegmentedColormap.from_list("delta_kge", ["#2166AC", "#F7F7F7", "#B2182B"], N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    fig, ax = plt.subplots(figsize=(10.0, 5.3))
    mesh = ax.pcolormesh(np.arange(matrix.shape[1] + 1), np.arange(matrix.shape[0] + 1), matrix, cmap=cmap, norm=norm, shading="flat", rasterized=False)
    ax.set_yticks(np.arange(len(MODEL_REGISTRY)) + 0.5); ax.set_yticklabels(MODEL_REGISTRY)
    ax.invert_yaxis()
    ax.set_xticks([0.5, matrix.shape[1] / 2, matrix.shape[1] - 0.5]); ax.set_xticklabels(["west", "basin order", "east"])
    ax.set_xlabel("Basin order by longitude (outcome-independent)")
    ax.set_ylabel("Canonical model registry order")
    ax.set_title(r"Supplementary Figure S1: raw $\Delta KGE$ model–basin landscape", pad=8)
    ax.text(0.01, -0.16, rf"$\Delta KGE=KGE_{{dPL}}-KGE_{{IC}}$; color clipping at symmetric 98th absolute percentile (±{lim:.3f}) for display only", transform=ax.transAxes, fontsize=7, va="top")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.012, fraction=0.025)
    cbar.set_label(r"$\Delta KGE$", fontsize=8); cbar.ax.tick_params(labelsize=7, length=2)
    fig.tight_layout()
    save_figure(fig, "FigS1_R1_raw_model_basin_deltaKGE")
    print(f"PASS: wrote Supplementary Figure S1 under {FIGURES_DIR}; cells={matrix.size}; symmetric_clip={lim:.6f}")


if __name__ == "__main__":
    main()
