"""Shared drawing primitives for the formal R1 Figure 1 panels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from r1_config import FIGURES_DIR
from r1_plot_utils import BLUE, DARK, LIGHT, MID, ORANGE, configure_matplotlib


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_panel_a(ax: plt.Axes, data: pd.DataFrame, *, label: str = "(a)", show_labels: bool = True) -> None:
    y = np.arange(len(data))
    x_ic = data["IC_median_KGE"].to_numpy(float)
    x_dpl = data["dPL_median_KGE"].to_numpy(float)
    for yi, ic, dpl in zip(y, x_ic, x_dpl):
        ax.plot([ic, dpl], [yi, yi], color="#C9CED4", lw=0.75, zorder=1)
    ax.scatter(x_ic, y, s=18, color=BLUE, edgecolor="white", linewidth=0.3, label="IC", zorder=3)
    ax.scatter(x_dpl, y, s=18, color=ORANGE, edgecolor="white", linewidth=0.3, marker="o", label="dPL", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(data["model"] if show_labels else [])
    ax.invert_yaxis()
    lo = float(min(x_ic.min(), x_dpl.min()) - 0.025); hi = float(max(x_ic.max(), x_dpl.max()) + 0.025)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Median KGE across basins")
    ax.set_title("Aggregate outlet performance", pad=7)
    ax.text(-0.18, 1.02, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    ax.grid(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor=LIGHT, fontsize=7, handletextpad=0.3)


def draw_panel_b(ax: plt.Axes, data: pd.DataFrame, *, label: str = "(b)", show_labels: bool = False) -> None:
    y = np.arange(len(data))
    q10 = data["delta_Q10"].to_numpy(float); q25 = data["delta_Q25"].to_numpy(float)
    med = data["delta_median"].to_numpy(float); q75 = data["delta_Q75"].to_numpy(float); q90 = data["delta_Q90"].to_numpy(float)
    ax.hlines(y, q10, q90, color=MID, lw=1.0, zorder=1)
    ax.hlines(y, q25, q75, color=BLUE, lw=4.0, zorder=2)
    ax.scatter(med, y, s=16, color=DARK, zorder=3)
    ax.axvline(0.0, color="black", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(data["model"] if show_labels else [])
    ax.invert_yaxis()
    lo = float(q10.min() - 0.015); hi = float(q90.max() + 0.015)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$\Delta KGE = KGE_{dPL} - KGE_{IC}$")
    ax.set_title("Basin-level ΔKGE distribution", pad=7)
    ax.text(-0.18, 1.02, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    ax.grid(False)


def draw_panel_d(ax: plt.Axes, data: pd.DataFrame, *, label: str = "(d)") -> None:
    order = ["model_main_effect", "basin_main_effect", "residual_model_basin_specificity"]
    colors = [BLUE, ORANGE, DARK]
    names = ["model main effect", "basin main effect", "model–basin specificity"]
    left = 0.0
    legend_parts = []
    for key, color, name in zip(order, colors, names):
        value = float(data.loc[data.component == key, "fraction_of_total"].iloc[0])
        ax.barh([0], [value], left=left, color=color, edgecolor="white", linewidth=0.5)
        if value > 0.06:
            ax.text(left + value / 2, 0, f"{100 * value:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        else:
            ax.text(left + value + 0.006, 0.17, f"{100 * value:.1f}%", ha="left", va="bottom", color=DARK, fontsize=7.0, fontweight="bold", clip_on=False)
        legend_parts.append(f"{100 * value:.1f}% {name}")
        left += value
    ax.set_xlim(0, 1); ax.set_ylim(-0.31, 0.30); ax.set_yticks([]); ax.set_xticks([])
    ax.set_title("Descriptive decomposition", pad=7, fontsize=9)
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    ax.text(0.5, -0.25, "  |  ".join(legend_parts), transform=ax.transAxes, ha="center", va="top", fontsize=6.0)
    ax.axis("off")


def draw_panel_e(ax: plt.Axes, data: pd.DataFrame, *, label: str = "(e)") -> None:
    x = data["delta_median_A"].to_numpy(float); y = data["delta_median_B"].to_numpy(float)
    bound = max(float(np.max(np.abs(np.r_[x, y]))), 0.01) + 0.012
    ax.scatter(x, y, s=19, color=ORANGE, edgecolor="white", linewidth=0.3, alpha=0.92)
    ax.plot([-bound, bound], [-bound, bound], color=DARK, lw=0.85, ls="--")
    ax.axhline(0, color=MID, lw=0.6); ax.axvline(0, color=MID, lw=0.6)
    ax.set_xlim(-bound, bound); ax.set_ylim(-bound, bound); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Median ΔKGE, A", fontsize=8); ax.set_ylabel("Median ΔKGE, B", fontsize=8)
    ax.set_title("Temporal A/B persistence", pad=7, fontsize=9)
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    rank_a = pd.Series(x).rank(method="average").to_numpy(); rank_b = pd.Series(y).rank(method="average").to_numpy()
    rho = float(np.corrcoef(rank_a, rank_b)[0, 1])
    same = int(np.sum(np.sign(x) == np.sign(y)))
    ax.text(0.04, 0.96, f"Spearman ρ={rho:.3f}\nsame sign={same}/36\nN=36 models", transform=ax.transAxes, va="top", fontsize=6.5, bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 2.5, "alpha": 0.9})
    ax.grid(False)


def setup() -> None:
    configure_matplotlib()
    plt.rcParams.update({"axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 6.5})
