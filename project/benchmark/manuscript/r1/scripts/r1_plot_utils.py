"""Matplotlib-only figure builders for R1 Fig. 1."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r1_config import MODEL_REGISTRY


BLUE = "#1f5a85"
DARK = "#243447"
MID = "#6b7280"
LIGHT = "#d8dee6"
ORANGE = "#b35c2e"


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
        "ytick.labelsize": 6.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def draw_fig1a(ax: plt.Axes, pair: pd.DataFrame, model_summary: pd.DataFrame, *, label: str = "a") -> None:
    x = model_summary.IC_median.to_numpy(float)
    y = model_summary.dPL_median.to_numpy(float)
    ax.scatter(x, y, s=25, color=BLUE, edgecolor="white", linewidth=0.35, alpha=0.95, zorder=3)
    low = float(min(x.min(), y.min()) - 0.03)
    high = float(max(x.max(), y.max()) + 0.03)
    ax.plot([low, high], [low, high], color=DARK, lw=0.9, ls="--", zorder=1)
    ax.set_xlim(low, high); ax.set_ylim(low, high); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("IC median KGE"); ax.set_ylabel("dPL median KGE")
    ax.set_title("Aggregate outlet performance", pad=8)
    ic_ens = float(model_summary.IC_median.median())
    dpl_ens = float(model_summary.dPL_median.median())
    ax.text(0.04, 0.04, f"36 models\n531 basins/model\nmedian of model medians: IC={ic_ens:.3f}, dPL={dpl_ens:.3f}",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3, "alpha": 0.9})
    ax.grid(False)
    panel_label(ax, label)


def draw_fig1b(ax: plt.Axes, model_summary: pd.DataFrame, *, label: str = "b") -> None:
    by_model = model_summary.set_index("model").loc[list(MODEL_REGISTRY)]
    y = np.arange(len(MODEL_REGISTRY))
    ax.hlines(y, by_model.delta_Q10, by_model.delta_Q90, color=MID, lw=1.0, zorder=1, label="Q10–Q90")
    ax.hlines(y, by_model.delta_Q25, by_model.delta_Q75, color=BLUE, lw=4.0, zorder=2, label="Q25–Q75")
    ax.scatter(by_model.delta_median, y, s=18, color=DARK, zorder=3, label="median")
    ax.axvline(0, color="black", lw=0.8, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(MODEL_REGISTRY)
    ax.invert_yaxis()
    lo = float(by_model.delta_Q10.min() - 0.015); hi = float(by_model.delta_Q90.max() + 0.015)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$\Delta KGE = KGE_{dPL} - KGE_{IC}$")
    ax.set_title("Basin-level ΔKGE heterogeneity", pad=8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor=LIGHT, handlelength=2.2)
    ax.grid(False)
    panel_label(ax, label)


def draw_fig1c(ax: plt.Axes, temporal_summary: pd.DataFrame, *, label: str = "c") -> None:
    x = temporal_summary.delta_median_A.to_numpy(float)
    y = temporal_summary.delta_median_B.to_numpy(float)
    ax.scatter(x, y, s=25, color=ORANGE, edgecolor="white", linewidth=0.35, alpha=0.95, zorder=3)
    max_abs = float(max(np.max(np.abs(x)), np.max(np.abs(y))) + 0.012)
    ax.plot([-max_abs, max_abs], [-max_abs, max_abs], color=DARK, lw=0.9, ls="--", zorder=1)
    ax.axhline(0, color=MID, lw=0.65, zorder=0); ax.axvline(0, color=MID, lw=0.65, zorder=0)
    ax.set_xlim(-max_abs, max_abs); ax.set_ylim(-max_abs, max_abs); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Median ΔKGE, partition A"); ax.set_ylabel("Median ΔKGE, partition B")
    ax.set_title("Temporal A/B reproducibility", pad=8)
    a = temporal_summary.delta_median_A.to_numpy(float); b = temporal_summary.delta_median_B.to_numpy(float)
    neutral = (a == 0) | (b == 0); denominator = int((~neutral).sum())
    same = int(((np.sign(a[~neutral]) == np.sign(b[~neutral]))).sum())
    rank_a = pd.Series(a).rank(method="average").to_numpy(); rank_b = pd.Series(b).rank(method="average").to_numpy()
    rho = float(np.corrcoef(rank_a, rank_b)[0, 1])
    ax.text(0.04, 0.96, f"Spearman ρ={rho:.3f}\nsame sign={same}/{denominator}\nN=36 models",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3, "alpha": 0.9})
    ax.grid(False)
    panel_label(ax, label)


def save_figure(fig: plt.Figure, stem) -> None:
    stem = str(stem)
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem + ".pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
