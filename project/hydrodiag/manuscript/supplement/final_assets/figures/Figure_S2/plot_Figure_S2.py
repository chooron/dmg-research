#!/usr/bin/env python3
"""Plot frozen TGD response curves and response-shape sensitivity metrics.

Only CSV outputs from the completed reviewer-2 robustness work are read.  No
model objects, checkpoints, simulations, or training/evaluation pipelines are
loaded or called.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
MANUSCRIPT = PROJECT / "manuscript"
RESPONSE_DATA = PROJECT / "results" / "reviewer2_robustness" / "tgd_response" / "tgd_response_data.csv"
SHAPE_DATA = PROJECT / "results" / "reviewer2_robustness" / "tgd_shape_sensitivity" / "tgd_shape_sensitivity_basin_metrics.csv"
DEFAULT_OUT = HERE / "Figure_S2.png"

sys.path.insert(0, str(MANUSCRIPT / "scripts" / "shared"))
from r1_plot_style import (  # noqa: E402
    COLOR_DARK_NEUTRAL,
    COLOR_LIGHT_REF,
    COLOR_TGD,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)

RESPONSE_SPECS = (
    ("Default", "tau_Default_tau_w_0.25_Deltatau_c_10", "retention_Default_tau_w_0.25_Deltatau_c_10", "(0.25, 10)"),
    ("P10", "tau_P10_Fitted_tau_w_0.05_Deltatau_c_2.0", "retention_P10_Fitted_tau_w_0.05_Deltatau_c_2.0", "(0.05, 2)"),
    ("Median", "tau_Median_Fitted_tau_w_0.30_Deltatau_c_15.0", "retention_Median_Fitted_tau_w_0.30_Deltatau_c_15.0", "(0.30, 15)"),
    ("P90", "tau_P90_Fitted_tau_w_1.20_Deltatau_c_60.0", "retention_P90_Fitted_tau_w_1.20_Deltatau_c_60.0", "(1.20, 60)"),
    ("Upper bound", "tau_Upper_Bound_tau_w_3.00_Deltatau_c_180.0", "retention_Upper_Bound_tau_w_3.00_Deltatau_c_180.0", "(3, 180)"),
    ("Lower bound", "tau_Lower_Bound_tau_w_0.001_Deltatau_c_0.1", "retention_Lower_Bound_tau_w_0.001_Deltatau_c_0.1", "(0.001, 0.1)"),
)
VARIANT_ORDER = (
    "Sharp (T_ref=0, s_T=1)",
    "Canonical (T_ref=0, s_T=2)",
    "Warm-shifted (T_ref=+2, s_T=2)",
    "Broad (T_ref=0, s_T=4)",
)
VARIANT_LABELS = ("Sharp\n$s_T=1$", "Canonical\n$s_T=2$", "Warm-shifted\n$T_{ref}=+2$", "Broad\n$s_T=4$")


def _iqr(frame: pd.DataFrame, column: str) -> tuple[float, float, float, int]:
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError(f"No finite values for {column}")
    q25, median, q75 = values.quantile([0.25, 0.50, 0.75])
    return float(median), float(q25), float(q75), int(values.size)


def _point_interval(ax, x: float, frame: pd.DataFrame, column: str, marker: str, *, open_marker: bool) -> tuple[float, float, float, int]:
    median, q25, q75, n = _iqr(frame, column)
    ax.errorbar(
        x,
        median,
        yerr=[[median - q25], [q75 - median]],
        fmt=marker,
        color=COLOR_TGD,
        markerfacecolor="white" if open_marker else COLOR_TGD,
        markeredgecolor=COLOR_TGD,
        markeredgewidth=1.0,
        markersize=5.6,
        capsize=2.8,
        elinewidth=1.1,
        zorder=4,
    )
    return median, q25, q75, n


def _plot_response(ax_tau, ax_ret, response: pd.DataFrame) -> None:
    temperature = pd.to_numeric(response["temperature_c"], errors="coerce")
    line_styles = ("-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (2, 2)))
    line_colors = (COLOR_TGD, "#2f6f63", "#4d8d7d", "#6fa394", COLOR_DARK_NEUTRAL, "#9aa4a4")
    for (label, tau_col, retention_col, parameter_label), linestyle, color in zip(RESPONSE_SPECS, line_styles, line_colors):
        if tau_col not in response or retention_col not in response:
            raise KeyError(f"Missing response columns: {tau_col}, {retention_col}")
        tau = pd.to_numeric(response[tau_col], errors="coerce")
        retention = pd.to_numeric(response[retention_col], errors="coerce")
        ax_tau.plot(temperature, tau, linestyle=linestyle, color=color, linewidth=1.35, label=f"{label} {parameter_label}")
        ax_ret.plot(temperature, retention, linestyle=linestyle, color=color, linewidth=1.35, label=f"{label} {parameter_label}")


def build_figure(out_path: Path) -> Path:
    setup_publication_style()
    response = pd.read_csv(RESPONSE_DATA)
    shape = pd.read_csv(SHAPE_DATA)
    fig, (ax_tau, ax_ret, ax_shape) = plt.subplots(1, 3, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.0, 1.0, 1.18]})

    for ax in (ax_tau, ax_ret, ax_shape):
        apply_clean_spines(ax)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.42, color=COLOR_LIGHT_REF)
        ax.axhline(0.0, color=COLOR_ZERO_LINE, linewidth=0.75, zorder=1)

    _plot_response(ax_tau, ax_ret, response)
    ax_tau.set_title(r"(a) Residence time $\tau_t$", loc="left", weight="bold", pad=6, fontsize=9.0)
    ax_tau.set_xlabel("Temperature (°C)")
    ax_tau.set_ylabel("Residence time (d)")
    ax_tau.set_xlim(-15, 20)
    ax_tau.set_ylim(0.05, 200)
    ax_tau.set_yscale("log")
    ax_tau.text(0.03, 0.04, "Continuous thermal gate", transform=ax_tau.transAxes, fontsize=6.6, color="#555555")

    ax_ret.set_title(r"(b) Daily retention $r_t$", loc="left", weight="bold", pad=6, fontsize=9.0)
    ax_ret.set_xlabel("Temperature (°C)")
    ax_ret.set_ylabel("Retention fraction")
    ax_ret.set_xlim(-15, 20)
    ax_ret.set_ylim(-0.03, 1.03)
    ax_ret.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_ret.text(0.03, 0.04, r"$r_t=\exp[-1/\tau_t]$", transform=ax_ret.transAxes, fontsize=6.6, color="#555555")

    ax_shape.set_title("(c) Response-shape sensitivity", loc="left", weight="bold", pad=6, fontsize=9.0)
    ax_shape.set_ylabel(r"Median $\Delta F = F_{\mathrm{TGD}}^* - F_{\mathrm{close}}$")
    ax_shape.set_xticks(np.arange(len(VARIANT_ORDER)))
    ax_shape.set_xticklabels(VARIANT_LABELS, fontsize=6.9)
    ax_shape.set_xlim(-0.55, len(VARIANT_ORDER) - 0.45)
    ax_shape.set_ylim(-2.55, 0.65)
    ax_shape.text(0.03, 0.04, "Intervals = Q25–Q75; IC open, dPL filled\nN = 427 (IC), 460 (dPL)", transform=ax_shape.transAxes, fontsize=6.3, color="#555555")
    for i, variant in enumerate(VARIANT_ORDER):
        frame = shape.loc[shape["variant"].eq(variant)]
        if frame.empty:
            raise ValueError(f"Missing shape variant {variant}")
        ic = _point_interval(ax_shape, i - 0.10, frame, "delta_F_ic", "o", open_marker=True)
        dpl = _point_interval(ax_shape, i + 0.10, frame, "delta_F_dpl", "^", open_marker=False)
        ax_shape.text(i + 0.02, 0.98, f"IC {ic[0]:+.3f}\ndPL {dpl[0]:+.3f}", transform=ax_shape.get_xaxis_transform(), ha="center", va="top", fontsize=5.9, color=COLOR_DARK_NEUTRAL)

    response_line_styles = ("-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (2, 2)))
    response_handles = [
        Line2D([0], [0], color=COLOR_TGD, lw=1.5, ls=linestyle, label=f"{label} {parameter_label}")
        for (label, _tau_col, _retention_col, parameter_label), linestyle in zip(RESPONSE_SPECS, response_line_styles)
    ]
    response_handles.extend([
        Line2D([0], [0], marker="o", color=COLOR_TGD, markerfacecolor="white", lw=0, markersize=5.2, label="IC (open circle)"),
        Line2D([0], [0], marker="^", color=COLOR_TGD, markerfacecolor=COLOR_TGD, lw=0, markersize=5.2, label="dPL (filled triangle)"),
    ])
    fig.legend(response_handles, [h.get_label() for h in response_handles], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=6.5)
    fig.text(
        0.5, -0.01,
        "Panels (a–b) use the frozen mathematical response table (351 temperatures). "
        "Panel (c) uses frozen basin-level test metrics; fractions are unclipped and denominator-valid.",
        ha="center", va="top", fontsize=6.6, color="#555555",
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.21, top=0.78, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_figure(args.out)


if __name__ == "__main__":
    main()
