"""Reproducible R2 Figure 2 plots and Figure 3 prototypes.

This script consumes only the frozen R2 plotting caches.  It deliberately does
not draw a final Figure 3: 3a remains an estimand placeholder and 3c remains a
matched-set audit placeholder.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

BLUE = "#1f5a85"
DARK = "#243447"
MID = "#6b7280"
LIGHT = "#d8dee6"
PALE_BLUE = "#c7dbea"
ORANGE = "#b35c2e"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def load_order() -> list[str]:
    order = pd.read_csv(TABLES / "R2_MODEL_ORDER_CANDIDATES.csv")
    return order.sort_values("r1_order").model_id.tolist()


def ordered(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    out = df.copy()
    rank = {model: i for i, model in enumerate(model_order)}
    out["_order"] = out["model_id"].map(rank)
    return out.sort_values("_order").drop(columns="_order")


def finish_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def draw_2a(ax: plt.Axes, data: pd.DataFrame, model_order: list[str], label: str = "a") -> None:
    summaries = (
        data.groupby("model_id", as_index=False)
        .agg(
            q10=("D_theta_cross", lambda x: x.quantile(0.10)),
            q25=("D_theta_cross", lambda x: x.quantile(0.25)),
            median=("D_theta_cross", "median"),
            q75=("D_theta_cross", lambda x: x.quantile(0.75)),
            q90=("D_theta_cross", lambda x: x.quantile(0.90)),
            ic_median=("D_theta_IC_self", "median"),
        )
    )
    summaries = ordered(summaries, model_order)
    y = np.arange(len(summaries))
    ax.hlines(y, summaries.q10, summaries.q90, color=MID, lw=1.0, zorder=1)
    ax.hlines(y, summaries.q25, summaries.q75, color=BLUE, lw=4.0, zorder=2)
    ax.scatter(summaries["median"], y, color=DARK, s=16, zorder=3)
    ax.scatter(
        summaries.ic_median,
        y,
        color="white",
        edgecolor=MID,
        marker="D",
        s=14,
        linewidth=0.75,
        zorder=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(summaries.model_id)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel(r"$D_\theta$ (normalized RMS displacement)")
    ax.set_title("Cross-paradigm parameter displacement", pad=8)
    headline = float(summaries["median"].median())
    ax.text(
        0.98,
        0.03,
        f"36 models\nmedian = {headline:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3, "alpha": 0.92},
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color=MID, lw=1, label="Q10–Q90"),
            Line2D([0], [0], color=BLUE, lw=4, label="Q25–Q75"),
            Line2D([0], [0], marker="o", color=DARK, lw=0, markersize=4, label="median"),
            Line2D([0], [0], marker="D", markerfacecolor="white", markeredgecolor=MID, color="white", lw=0, markersize=4, label="IC-self median"),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        edgecolor=LIGHT,
    )
    finish_axis(ax)
    panel_label(ax, label)


def draw_2b(ax: plt.Axes, data: pd.DataFrame, model_order: list[str], label: str = "b") -> None:
    models = ordered(data.loc[data.scope == "model"], model_order)
    summary = data.loc[data.scope == "all36_summary"].iloc[0]
    y = np.arange(len(models))
    ax.scatter(models.cross_minus_self_median, y, color=BLUE, edgecolor="white", linewidth=0.35, s=18, zorder=3)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.axvspan(float(summary.ci_low), float(summary.ci_high), color=PALE_BLUE, alpha=0.55, zorder=0)
    ax.axvline(float(summary.cross_minus_self_median), color=BLUE, lw=1.2, ls="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(models.model_id)
    ax.invert_yaxis()
    ax.set_xlabel(r"$D_{cross} - D_{IC-self}$")
    ax.set_title("Excess over IC-self", pad=8)
    ax.text(
        0.98,
        0.03,
        f"median = {float(summary.cross_minus_self_median):+.4f}\n95% CI [{float(summary.ci_low):.4f}, {float(summary.ci_high):.4f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3, "alpha": 0.92},
    )
    finish_axis(ax)
    panel_label(ax, label)


def draw_2c(ax: plt.Axes, data: pd.DataFrame, model_order: list[str], label: str = "c") -> None:
    data = ordered(data, model_order)
    groups = [group.R_rank.dropna().to_numpy(float) for _, group in data.groupby("model_id", sort=False)]
    labels = data.model_id.drop_duplicates().tolist()
    y = np.arange(len(groups))
    ax.boxplot(
        groups,
        vert=False,
        positions=y,
        widths=0.62,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": PALE_BLUE, "edgecolor": BLUE, "linewidth": 0.65},
        medianprops={"color": DARK, "linewidth": 1.0},
        whiskerprops={"color": BLUE, "linewidth": 0.65},
        capprops={"color": BLUE, "linewidth": 0.65},
    )
    model_medians = data.groupby("model_id", sort=False).R_rank.median()
    ax.scatter(model_medians.to_numpy(), y, color=DARK, s=10, zorder=3)
    ax.axvline(0, color=MID, lw=0.65, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-1.02, 1.02)
    ax.set_xlabel(r"Coordinate-level Spearman $R_{rank}$")
    ax.set_title("Cross-catchment rank correspondence", pad=8)
    headline = float(model_medians.median())
    iqr = float(model_medians.quantile(0.75) - model_medians.quantile(0.25))
    ax.text(
        0.99,
        0.03,
        f"model-equal median = {headline:.4f}\nIQR = {iqr:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3, "alpha": 0.92},
    )
    finish_axis(ax)
    panel_label(ax, label)


def draw_2d(ax: plt.Axes, data: pd.DataFrame, model_order: list[str], label: str = "d") -> None:
    data = ordered(data, model_order)
    y = np.arange(len(data))
    ax.hlines(y, data.R_cross, data.R_self, color=LIGHT, lw=1.0, zorder=1)
    ax.scatter(data.R_cross, y, color=BLUE, s=16, zorder=3, label=r"$R_{cross}$")
    ax.scatter(data.R_self, y, color="white", edgecolor=MID, s=16, zorder=3, label=r"$R_{self}$")
    ax.set_yticks(y)
    ax.set_yticklabels(data.model_id)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rank correspondence")
    ax.set_title("Strict IC-self benchmark", pad=8)
    ax.text(
        0.98,
        0.03,
        f"strict subset: {len(data)}/36 models\nall {int((data.DeltaR_self_minus_cross > 0).sum())} have $R_{{self}}>R_{{cross}}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3, "alpha": 0.92},
    )
    finish_axis(ax)
    panel_label(ax, label)


def draw_3b_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    model_order: list[str],
    metric: str,
    label: str,
    title: str,
    xlabel: str,
) -> None:
    models = ordered(data.loc[data.scope == "model"], model_order)
    value_col = {"C_eff": "C_eff_median", "top1": "top1_share_median", "top2": "top2_share_median"}[metric]
    low_col = {"C_eff": "C_eff_bootstrap_ci_low", "top1": "top1_share_bootstrap_ci_low", "top2": "top2_share_bootstrap_ci_low"}[metric]
    high_col = {"C_eff": "C_eff_bootstrap_ci_high", "top1": "top1_share_bootstrap_ci_high", "top2": "top2_share_bootstrap_ci_high"}[metric]
    y = np.arange(len(models))
    ax.hlines(y, models[low_col], models[high_col], color=MID, lw=0.8, zorder=1)
    ax.scatter(models[value_col], y, color=BLUE, s=14, zorder=3)
    summary = data.loc[(data.scope == "model_equal") & (data.summary_metric == ("C_eff" if metric == "C_eff" else f"{metric}_share"))].iloc[0]
    ax.axvline(float(summary.summary_value), color=BLUE, lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(models.model_id)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=7)
    ax.text(
        0.98,
        0.03,
        f"median = {float(summary.summary_value):.4f}\n95% CI [{float(summary.summary_ci_low):.4f}, {float(summary.summary_ci_high):.4f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3, "alpha": 0.92},
    )
    finish_axis(ax)
    panel_label(ax, label)


def draw_3d(ax: plt.Axes, data: pd.DataFrame, model_order: list[str], label: str = "d") -> None:
    data = ordered(data, model_order)
    cols = ["CR_canonical", "CR_consensus", "CR_ICself"]
    labels = ["canonical", "consensus", "IC-self"]
    x = np.arange(3)
    for _, row in data.iterrows():
        ax.plot(x, row[cols].to_numpy(float), color=LIGHT, lw=0.8, alpha=0.85, zorder=1)
    medians = data[cols].median()
    ax.plot(x, medians.to_numpy(float), color=BLUE, marker="o", markersize=5, lw=2.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Contraction ratio")
    ax.set_title("Reference-dependent contraction", pad=8)
    ax.set_ylim(bottom=0)
    ax.text(
        0.03,
        0.97,
        "23 strict models\n" + "\n".join(f"{name}: {value:.4f}" for name, value in zip(labels, medians)),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3, "alpha": 0.92},
    )
    finish_axis(ax)
    panel_label(ax, label)


def make_figure2(data: dict[str, pd.DataFrame], order: list[str]) -> None:
    panels = [
        ("F2a_parameter_displacement.png", draw_2a, data["2a"]),
        ("F2b_cross_minus_self.png", draw_2b, data["2b"]),
        ("F2c_rank_correspondence.png", draw_2c, data["2c"]),
        ("F2d_strict_rank_benchmark.png", draw_2d, data["2d"]),
    ]
    for filename, drawer, frame in panels:
        fig, ax = plt.subplots(figsize=(8.8, 8.2))
        drawer(ax, frame, order)
        save_png(fig, FIGURES / filename)

    fig = plt.figure(figsize=(13.5, 11.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.78], height_ratios=[1.12, 1])
    axes = [
        fig.add_subplot(grid[0, :2]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, :2]),
        fig.add_subplot(grid[1, 2]),
    ]
    draw_2a(axes[0], data["2a"], order)
    draw_2b(axes[1], data["2b"], order)
    draw_2c(axes[2], data["2c"], order)
    draw_2d(axes[3], data["2d"], order)
    fig.suptitle("R2 Figure 2 — parameter-space separation and rank reorganization", fontsize=13, y=1.01)
    save_png(fig, FIGURES / "Figure2_R2_parameter_reorganization.png")


def make_figure3_prototypes(data: dict[str, pd.DataFrame], order: list[str]) -> None:
    metrics = [("C_eff", "F3b1_ceff_summary.png", "a", "Effective coordinate number", r"$C_{eff}$") ,
               ("top1", "F3b2_top1_summary.png", "b", "Top-1 displacement share", "Top-1 share"),
               ("top2", "F3b3_top2_summary.png", "c", "Top-2 cumulative displacement share", "Top-2 share")]
    for metric, filename, label, title, xlabel in metrics:
        fig, ax = plt.subplots(figsize=(8.8, 8.2))
        draw_3b_metric(ax, data["3b"], order, metric, label, title, xlabel)
        save_png(fig, FIGURES / filename)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    draw_3d(ax, data["3d"], order, label="d")
    save_png(fig, FIGURES / "F3d_contraction_reference_sensitivity.png")

    fig = plt.figure(figsize=(13.5, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, width_ratios=[1.35, 1.0, 0.95], height_ratios=[1.2, 0.9, 0.95])
    ax_a = fig.add_subplot(grid[0, :2])
    ax_a.axis("off")
    ax_a.add_patch(Rectangle((0, 0), 1, 1, transform=ax_a.transAxes, facecolor="#f7f8fa", edgecolor=MID, hatch="///", linewidth=0.9))
    ax_a.text(0.5, 0.55, "3a  HERO PLACEHOLDER", ha="center", va="center", fontsize=15, fontweight="bold", color=DARK)
    ax_a.text(0.5, 0.38, "Model-level composition estimand is not frozen", ha="center", va="center", fontsize=9, color=MID)
    panel_label(ax_a, "a")

    draw_3b_metric(fig.add_subplot(grid[0, 2]), data["3b"], order, "C_eff", "b1", "", r"$C_{eff}$")
    draw_3b_metric(fig.add_subplot(grid[1, 2]), data["3b"], order, "top1", "b2", "", "Top-1")
    draw_3b_metric(fig.add_subplot(grid[2, 2]), data["3b"], order, "top2", "b3", "", "Top-2")

    ax_c = fig.add_subplot(grid[1, :2])
    ax_c.axis("off")
    ax_c.add_patch(Rectangle((0, 0), 1, 1, transform=ax_c.transAxes, facecolor="#f7f8fa", edgecolor=MID, hatch="\\\\", linewidth=0.9))
    ax_c.text(0.5, 0.55, "3c  ROBUSTNESS PLACEHOLDER", ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)
    ax_c.text(0.5, 0.38, "Raw matched basin definition must be repaired before headline values are locked", ha="center", va="center", fontsize=8.5, color=MID)
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(grid[2, :2])
    draw_3d(ax_d, data["3d"], order, label="d")
    fig.suptitle("R2 Figure 3 — prototype layout only; 3a/3c not final", fontsize=13, y=1.01)
    save_png(fig, FIGURES / "Figure3_layout_prototype.png")


def main() -> None:
    configure_matplotlib()
    FIGURES.mkdir(parents=True, exist_ok=True)
    order = load_order()
    data = {
        "2a": pd.read_parquet(CACHE / "fig2a_parameter_displacement.parquet"),
        "2b": pd.read_csv(CACHE / "fig2b_cross_minus_self.csv"),
        "2c": pd.read_parquet(CACHE / "fig2c_rank_correspondence.parquet"),
        "2d": pd.read_csv(CACHE / "fig2d_strict_rank_benchmark.csv"),
        "3b": pd.read_csv(CACHE / "fig3b_localization_metrics.csv"),
        "3d": pd.read_csv(CACHE / "fig3d_contraction_reference.csv"),
    }
    make_figure2(data, order)
    make_figure3_prototypes(data, order)
    print("wrote Figure 2 formal panels/composite and Figure 3 prototype panels/layout")
    print("outputs: 10 PNG files, each saved at 600 dpi")


if __name__ == "__main__":
    main()
