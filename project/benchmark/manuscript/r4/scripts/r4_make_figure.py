#!/usr/bin/env python3
"""Render the single R4 figure from frozen, already-generated figure data."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SCRIPT = Path(__file__).resolve()
R4 = SCRIPT.parents[1]
FIGURES = R4 / "figures"
DATA = R4 / "figure_data"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    viability = pd.read_csv(DATA / "R4_PANEL_A_KGE_VIABILITY.csv")
    population = pd.read_csv(DATA / "R4_PANEL_B_SEEN_VS_OOB.csv")
    cases = pd.read_csv(DATA / "R4_PANEL_C_FROZEN_CASES.csv")
    clusters = pd.read_csv(DATA / "R4_PANEL_D_CLUSTER_RETENTION.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    for row in viability.itertuples(index=False):
        ax.scatter(row.seen_dpl_median_kge, row.oob_pooled_median_kge, s=42, label=row.model)
        ax.annotate(row.model, (row.seen_dpl_median_kge, row.oob_pooled_median_kge), xytext=(3, 3), textcoords="offset points", fontsize=7)
    limits = [float(np.nanmin([viability.seen_dpl_median_kge.min(), viability.oob_pooled_median_kge.min()])) - 0.03, float(np.nanmax([viability.seen_dpl_median_kge.max(), viability.oob_pooled_median_kge.max()])) + 0.03]
    ax.plot(limits, limits, "k--", linewidth=0.8)
    ax.set(xlim=limits, ylim=limits, xlabel="seen dPL median KGE", ylabel="OOB pooled median KGE", title="A  Predictive viability control")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    colors = {"persistent/reproduced": "#2166ac", "attenuated": "#f4a582", "dPL-emergent": "#1b7837", "sign-changing": "#d6604d", "weak/unresolved": "#bdbdbd"}
    for label, frame in population.groupby("behavior_class", sort=False):
        ax.scatter(frame.rho_dpl_seen, frame.rho_dpl_oob, s=8, alpha=0.34, label=label, color=colors.get(label, "#999999"), linewidths=0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.8)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel=r"$\rho_{dPL,seen}$", ylabel=r"$\rho_{dPL,OOB}$", title="B  Population relationship persistence")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(alpha=0.15)

    ax = axes[1, 0]
    y_positions = np.arange(len(cases))[::-1]
    for y, row in zip(y_positions, cases.itertuples(index=False)):
        ax.errorbar([0, 1, 2], [row.rho_IC, row.rho_dPL_seen, row.rho_dPL_OOB], yerr=[[0, 0, row.rho_dPL_OOB - row.OOB_CI_low], [0, 0, row.OOB_CI_high - row.rho_dPL_OOB]], fmt="o", color="#2166ac", capsize=3, linewidth=1)
        folds = [row.rho_fold1, row.rho_fold2, row.rho_fold3, row.rho_fold4, row.rho_fold5]
        ax.scatter(np.full(5, 2.10), folds, color="#555555", s=14, alpha=0.8)
        ax.text(2.28, row.rho_dPL_OOB, f"{row.model}/{row.parameter}/{row.attribute}", va="center", fontsize=7)
    ax.axvline(2, color="#cccccc", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(xticks=[0, 1, 2], xticklabels=["IC", "dPL seen", "dPL OOB"], xlabel="parameterization", ylabel=r"$\rho$", title="C  Frozen representative relationships", ylim=(-1.05, 1.05))
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    if len(clusters):
        clusters = clusters.sort_values("cluster_retention_rate", ascending=True)
        y = np.arange(len(clusters))
        ax.barh(y - 0.18, clusters.raw_retention_rate, height=0.34, label="raw proxy", color="#92c5de")
        ax.barh(y + 0.18, clusters.cluster_retention_rate, height=0.34, label="information cluster", color="#0571b0")
        labels = [f"{row.information_cluster}\n({row.cluster_representative})" for row in clusters.itertuples(index=False)]
        ax.set(yticks=y, yticklabels=labels, xlim=(0, 1.05), xlabel="retention rate", title="D  Raw proxy vs information-cluster retention")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No non-weak cluster rows", ha="center", va="center")
        ax.set_title("D  Information-cluster retention")
    ax.grid(axis="x", alpha=0.2)

    fig.suptitle("R4 held-out-basin challenge: IC/seen dPL relationships versus OOB dPL", fontsize=14)
    png = FIGURES / "R4_MAIN_FIGURE.png"
    pdf = FIGURES / "R4_MAIN_FIGURE.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
