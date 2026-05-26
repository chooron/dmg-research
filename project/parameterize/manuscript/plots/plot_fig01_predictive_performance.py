from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    FigureSpec,
    MAIN_FIG_DIR,
    MODEL_COLORS,
    MODEL_ORDER,
    TABLE_ROOT,
    add_panel_label,
    clean_axes,
    math_model_labels,
    save,
)


def _tick_label(key: str) -> str:
    return math_model_labels()[key]


def _boxplot(ax: plt.Axes, frame: pd.DataFrame, metric: str, panel: str) -> None:
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    data = [frame.loc[frame["model"].eq(model), metric].dropna().clip(-1.0, 1.0).to_numpy() for model in MODEL_ORDER]
    medians = [float(np.median(values)) for values in data]
    colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#2A2A2A", "lw": 1.0},
        whiskerprops={"color": "#666666", "lw": 0.8},
        capprops={"color": "#666666", "lw": 0.8},
        boxprops={"edgecolor": "#666666", "lw": 0.8},
        flierprops={
            "marker": "o",
            "markersize": 2.2,
            "markerfacecolor": "#6F6F6F",
            "markeredgecolor": "#6F6F6F",
            "markeredgewidth": 0.0,
            "alpha": 0.5,
        },
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for x, median, color in zip(positions, medians, colors):
        ax.scatter([x], [median], s=18, facecolor="white", edgecolor=color, linewidth=0.9, zorder=4)
        ax.text(x, -0.92, f"Median\n{median:.2f}", ha="center", va="bottom", fontsize=10.2, color=color)
    for y in (0.0, 0.5):
        ax.axhline(y, color="#D8D8D8", lw=0.75, ls=(0, (3, 3)), zorder=0)
    ax.set_xlim(0.45, len(MODEL_ORDER) + 0.55)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks(np.linspace(-1.0, 1.0, 5))
    ax.set_xticks(positions)
    ax.set_xticklabels([_tick_label(model) for model in MODEL_ORDER])
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", length=0)
    add_panel_label(ax, f"({panel.lower()})", x=0.98, y=0.98, ha="right", va="top", fontweight="normal", fontsize=12.0)
    clean_axes(ax)


def _plot_cdf(ax: plt.Axes, frame: pd.DataFrame, metric: str, panel: str) -> None:
    linestyles = {
        "deterministic": "-",
        "mc_dropout": (0, (4, 2.2)),
        "distributional": (0, (5, 2, 1.2, 2)),
    }
    fractions = {}
    labels = math_model_labels()
    for model in MODEL_ORDER:
        values = np.sort(frame.loc[frame["model"].eq(model), metric].dropna().clip(-1.0, 1.0).to_numpy())
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1, dtype=float) / len(values)
        ax.plot(values, y, color=MODEL_COLORS[model], lw=1.25, ls=linestyles[model], label=labels[model])
        fractions[model] = float(np.mean(values > 0.5))
    ax.axvline(0.5, color="#D8D8D8", lw=0.75, ls=(0, (3, 3)))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    ax.grid(True, color="#E6E6E6", lw=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, handlelength=1.8, borderpad=0.15, labelspacing=0.22, fontsize=10.0)
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("CDF")
    ax.text(-0.75, 0.50, f"{metric.upper()} > 0.5", ha="left", va="top", fontsize=10.2, color="#2A2A2A")
    y0 = 0.38
    step = 0.10
    for idx, model in enumerate(MODEL_ORDER):
        ax.text(-0.75, y0 - idx * step, f"{labels[model]}: {fractions.get(model, np.nan):.2f}", ha="left", va="top", fontsize=10.2, color=MODEL_COLORS[model])
    add_panel_label(ax, f"({panel.lower()})", x=0.98, y=0.06, ha="right", va="bottom", fontweight="normal", fontsize=12.0)
    clean_axes(ax)


def _plot_qq(ax: plt.Axes, frame: pd.DataFrame, metric: str, panel: str, *, show_xlabel: bool) -> None:
    labels = math_model_labels()
    base = frame.loc[frame["model"].eq("deterministic"), metric].dropna().clip(-1.0, 1.0).to_numpy()
    if len(base) == 0:
        return

    for model in ("mc_dropout", "distributional"):
        values = frame.loc[frame["model"].eq(model), metric].dropna().clip(-1.0, 1.0).to_numpy()
        n = min(len(base), len(values))
        if n == 0:
            continue
        probabilities = np.linspace(0.0, 1.0, n)
        ax.scatter(
            np.quantile(base, probabilities),
            np.quantile(values, probabilities),
            s=8,
            color=MODEL_COLORS[model],
            alpha=0.44,
            edgecolors="none",
            label=labels[model],
        )

    ax.plot([-1.0, 1.0], [-1.0, 1.0], color="#B8B8B8", lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.grid(True, color="#E6E6E6", lw=0.55)
    ax.set_axisbelow(True)
    ax.set_title(metric.upper(), pad=2, fontsize=11.2)
    ax.set_ylabel(f"{labels['mc_dropout']}, {labels['distributional']}", labelpad=1.5)
    if show_xlabel:
        ax.set_xlabel(labels["deterministic"])
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.legend(
        loc="lower right",
        frameon=False,
        markerscale=1.2,
        handletextpad=0.25,
        borderpad=0.05,
        labelspacing=0.15,
        fontsize=9.8,
    )
    add_panel_label(ax, f"({panel.lower()})", x=0.04, y=0.96, ha="left", va="top", fontweight="normal", fontsize=12.0)
    clean_axes(ax)


def draw() -> None:
    metrics = pd.read_csv(TABLE_ROOT / "metrics_long.csv")
    reference_loss = "HybridNseBatchLoss"
    basin_metrics = (
        metrics.loc[metrics["loss"].eq(reference_loss), ["basin_id", "model", "nse", "kge"]]
        .groupby(["basin_id", "model"], as_index=False)[["nse", "kge"]]
        .median()
    )
    spec = FigureSpec("Fig01_predictive_performance", MAIN_FIG_DIR, 224, 116)
    fig = plt.figure(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), constrained_layout=True)
    outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 2.55])
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    right = outer[0, 2].subgridspec(2, 2, width_ratios=[1.18, 1.02], hspace=0.16, wspace=0.02)
    ax_c = fig.add_subplot(right[0, 0])
    ax_d = fig.add_subplot(right[1, 0])
    ax_e = fig.add_subplot(right[0, 1])
    ax_f = fig.add_subplot(right[1, 1])

    _boxplot(ax_a, basin_metrics, "nse", "A")
    _boxplot(ax_b, basin_metrics, "kge", "B")
    _plot_cdf(ax_c, basin_metrics, "nse", "C")
    _plot_cdf(ax_d, basin_metrics, "kge", "D")
    _plot_qq(ax_e, basin_metrics, "nse", "E", show_xlabel=False)
    _plot_qq(ax_f, basin_metrics, "kge", "F", show_xlabel=True)
    save(fig, spec)
