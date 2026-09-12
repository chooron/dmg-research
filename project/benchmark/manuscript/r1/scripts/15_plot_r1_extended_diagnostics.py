#!/usr/bin/env python3
"""R1X exploratory figures; formal Fig1 files are not touched."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from r1_config import FIGURES_DIR, MODEL_REGISTRY, TABLES_DIR
from r1_plot_utils import BLUE, DARK, LIGHT, MID, ORANGE, configure_matplotlib



def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_model_tendency() -> None:
    data = pd.read_csv(TABLES_DIR / "R1X_model_estimator_tendency.csv").set_index("model").loc[list(MODEL_REGISTRY)].reset_index()
    y = np.arange(len(data))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.2, 7.0), sharey=True, gridspec_kw={"width_ratios": [2.4, 1.0]})
    ax0.hlines(y, data.delta_Q10, data.delta_Q90, color=MID, lw=1.0)
    ax0.hlines(y, data.delta_Q25, data.delta_Q75, color=BLUE, lw=4.0)
    ax0.scatter(data.delta_median, y, color=DARK, s=16, zorder=3)
    ax0.axvline(0.0, color="black", lw=0.8)
    ax0.set_yticks(y); ax0.set_yticklabels(data.model); ax0.invert_yaxis()
    ax0.set_xlabel(r"Model median $\Delta KGE$")
    ax0.set_title("Model tendency", pad=8)
    ax0.text(0.02, 1.02, "Q10–Q90 / Q25–Q75 / median", transform=ax0.transAxes, fontsize=7, va="bottom")
    ax1.barh(y, data.frac_dPL_gt_IC, color=ORANGE, alpha=0.85)
    ax1.axvline(0.5, color=MID, lw=0.8, ls="--")
    ax1.set_xlim(0.0, 1.0); ax1.set_xlabel(r"$P_m^+$")
    ax1.set_title("dPL-positive basin fraction", pad=8)
    ax1.grid(False)
    fig.suptitle("R1X: model-level estimator-relative tendency", y=0.995)
    fig.tight_layout()
    save_figure(fig, "R1X_model_estimator_tendency")


def plot_basin_tendency() -> None:
    data = pd.read_csv(TABLES_DIR / "R1X_basin_estimator_tendency.csv")
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.scatter(x, data.frac_models_dPL_gt_IC, s=8, color=ORANGE, alpha=0.8, linewidths=0)
    ax.axhline(0.5, color=MID, lw=0.8, ls="--")
    ax.set_xlim(-1, len(data)); ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Canonical basin registry order (531 basins)")
    ax.set_ylabel(r"$P_b^+$ across 36 models")
    ax.set_title("R1X: basin-level cross-model estimator tendency", pad=8)
    ax.text(0.01, 0.04, "Continuous descriptive distribution; no basin ranking", transform=ax.transAxes, fontsize=7, va="bottom")
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "R1X_basin_tendency_strip")


def plot_two_way() -> None:
    data = pd.read_csv(TABLES_DIR / "R1X_two_way_decomposition.csv")
    data = data[data.component != "total"].copy()
    labels = ["Model main effect", "Basin main effect", "Residual model–basin specificity"]
    values = [float(data.loc[data.component == key, "fraction_of_total"].iloc[0]) for key in ("model_main_effect", "basin_main_effect", "residual_model_basin_specificity")]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    bars = ax.bar(labels, values, color=[BLUE, ORANGE, DARK], width=0.62)
    ax.set_ylim(0.0, max(1.0, max(values) * 1.18)); ax.set_ylabel("Descriptive SS / total SS")
    ax.set_title("R1X: descriptive two-way decomposition", pad=8)
    ax.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(0.02, 0.02, "DESCRIPTIVE ONLY; no p-values; residual is not inferential interaction variance", transform=ax.transAxes, fontsize=6.8, va="bottom")
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "R1X_two_way_decomposition")


def plot_temporal_persistence() -> None:
    data = pd.read_csv(TABLES_DIR / "R1X_temporal_basin_tendency.csv")
    summary = pd.read_csv(TABLES_DIR / "R1X_temporal_basin_persistence_summary.csv").set_index("metric")
    x = data.delta_crossmodel_median_A.to_numpy(float); y = data.delta_crossmodel_median_B.to_numpy(float)
    bound = max(float(np.max(np.abs(np.r_[x, y]))), 0.01) + 0.01
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(x, y, s=12, color=ORANGE, alpha=0.75, edgecolor="white", linewidth=0.2)
    ax.plot([-bound, bound], [-bound, bound], color=DARK, lw=0.9, ls="--")
    ax.axhline(0.0, color=MID, lw=0.65); ax.axvline(0.0, color=MID, lw=0.65)
    ax.set_xlim(-bound, bound); ax.set_ylim(-bound, bound); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$M_b^A$: median $\Delta KGE$, partition A")
    ax.set_ylabel(r"$M_b^B$: median $\Delta KGE$, partition B")
    ax.set_title("R1X: basin-level temporal persistence", pad=8)
    rho = float(summary.loc["rho_M_spearman", "value"])
    same = int(summary.loc["same_sign_numerator", "value"]); denom = int(summary.loc["same_sign_denominator", "value"])
    ax.text(0.04, 0.96, f"Spearman ρ={rho:.3f}\nsame sign={same}/{denom}\nN=531 basins", transform=ax.transAxes, va="top", fontsize=7, bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3, "alpha": 0.9})
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "R1X_temporal_basin_persistence")


def main() -> None:
    configure_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_tendency()
    plot_basin_tendency()
    plot_two_way()
    plot_temporal_persistence()
    print(f"PASS: wrote R1X exploratory figures under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
