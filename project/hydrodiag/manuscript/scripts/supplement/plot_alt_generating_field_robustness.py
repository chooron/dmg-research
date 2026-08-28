#!/usr/bin/env python3
"""Plot alternative-generating-field sensitivity from frozen R3 basin outputs.

This script only reads the canonical and alternative basin-level CSV files.  It
never loads a model or invokes a simulation/training/evaluation pipeline.
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
PROJECT = HERE.parents[2]
MANUSCRIPT = PROJECT / "manuscript"
CANONICAL = MANUSCRIPT / "results" / "R3" / "figure5_basin_seedmedian.csv"
ALTERNATIVE = PROJECT / "results" / "reviewer2_robustness" / "alt_generating_field" / "alt_generating_field_basin_seedmedian.csv"
DEFAULT_OUT = MANUSCRIPT / "supplement" / "figures" / "FigureS3_alt_generating_field_robustness.png"

sys.path.insert(0, str(MANUSCRIPT / "scripts" / "shared"))
from r1_plot_style import (  # noqa: E402
    COLOR_BASE,
    COLOR_DARK_NEUTRAL,
    COLOR_TGD,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)


FIELDS = ("Canonical PCA/Ridge", "Direct CN–IC")
PARADIGMS = ("IC", "dPL")


def _read_test(path: Path, field: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame = frame.loc[frame["period"].eq("test")].copy()
    frame["field"] = field
    frame["tgd_column"] = "G_TGD" if "G_TGD" in frame.columns else "G_tgd"
    return frame


def _summary(frame: pd.DataFrame, column: str, valid_only: bool = False) -> tuple[float, float, float, int]:
    values = pd.to_numeric(frame[column], errors="coerce")
    if valid_only:
        values = values.loc[pd.to_numeric(frame["D"], errors="coerce").gt(1e-6)]
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError(f"No finite values for {column}")
    q25, median, q75 = values.quantile([0.25, 0.50, 0.75])
    return float(median), float(q25), float(q75), int(values.size)


def _plot_point_interval(ax, x: float, frame: pd.DataFrame, column: str, color: str, marker: str, *, valid_only: bool = False, open_marker: bool = False) -> tuple[float, float, float, int]:
    median, q25, q75, n = _summary(frame, column, valid_only=valid_only)
    ax.errorbar(
        x,
        median,
        yerr=[[median - q25], [q75 - median]],
        fmt=marker,
        color=color,
        markerfacecolor="white" if open_marker else color,
        markeredgecolor=color,
        markeredgewidth=1.0,
        markersize=5.5,
        capsize=2.8,
        elinewidth=1.1,
        zorder=4,
    )
    return median, q25, q75, n


def build_figure(out_path: Path) -> Path:
    setup_publication_style()
    canonical = _read_test(CANONICAL, FIELDS[0])
    alternative = _read_test(ALTERNATIVE, FIELDS[1])
    frames = {(field, paradigm): frame.loc[frame["paradigm"].eq(paradigm)].copy()
              for field, frame in ((FIELDS[0], canonical), (FIELDS[1], alternative))
              for paradigm in PARADIGMS}

    fig, (ax_raw, ax_fraction) = plt.subplots(
        1, 2, figsize=(8.9, 4.2), gridspec_kw={"width_ratios": [1.0, 1.12]}
    )
    x = np.arange(4, dtype=float)
    labels = ["Canonical\nIC", "Canonical\ndPL", "Direct CN–IC\nIC", "Direct CN–IC\ndPL"]
    entries = [(field, paradigm) for field in FIELDS for paradigm in PARADIGMS]

    for ax in (ax_raw, ax_fraction):
        apply_clean_spines(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.3)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.45, color="#C8CDD1")
        ax.axhline(0.0, color=COLOR_ZERO_LINE, linewidth=0.8, zorder=1)

    ax_raw.set_title("(a) Raw recovery gains", loc="left", weight="bold", pad=6, fontsize=9.0)
    ax_raw.set_ylabel("Test-period gain in KGE", labelpad=3)
    ax_raw.set_ylim(-0.10, 0.29)
    ax_raw.set_xlim(-0.55, 3.55)
    raw_results = []
    for i, key in enumerate(entries):
        frame = frames[key]
        base = _plot_point_interval(ax_raw, i - 0.09, frame, "G_base", COLOR_BASE, "o")
        tgd = _plot_point_interval(ax_raw, i + 0.09, frame, frame["tgd_column"].iloc[0], COLOR_TGD, "^")
        raw_results.extend([base, tgd])
    ax_raw.text(
        0.03, 0.97, "Points = median; intervals = Q25–Q75\nall 531 catchments",
        transform=ax_raw.transAxes, ha="left", va="top", fontsize=6.6, color="#555555",
    )

    ax_fraction.set_title("(b) Normalized recovery contrast", loc="left", weight="bold", pad=6, fontsize=9.0)
    ax_fraction.set_ylabel(r"Catchment-wise recovery fraction ($F$)", labelpad=3)
    ax_fraction.set_xlim(-0.55, 3.55)
    ax_fraction.set_ylim(-2.48, 1.10)
    for i, key in enumerate(entries):
        frame = frames[key]
        close = _plot_point_interval(
            ax_fraction, i - 0.09, frame, "F_close", COLOR_BASE, "o", valid_only=True, open_marker=True
        )
        tgd = _plot_point_interval(
            ax_fraction, i + 0.09, frame, "F_tgd_star" if "F_tgd_star" in frame.columns else "F_TGD_star",
            COLOR_TGD, "^", valid_only=True,
        )
        valid = frame.loc[pd.to_numeric(frame["D"], errors="coerce").gt(1e-6)]
        if "delta_F" in valid.columns:
            delta = pd.to_numeric(valid["delta_F"], errors="coerce")
        else:
            f_tgd_col = "F_tgd_star" if "F_tgd_star" in valid.columns else "F_TGD_star"
            delta = pd.to_numeric(valid[f_tgd_col], errors="coerce") - pd.to_numeric(valid["F_close"], errors="coerce")
        delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
        if delta.empty:
            raise ValueError(f"No finite delta_F values for {key}")
        positive = float((delta > 0).mean())
        ax_fraction.text(
            i, 0.98, f"ΔF={delta.median():+.3f}\nP+={positive:.1%}\nN={len(delta)}",
            transform=ax_fraction.get_xaxis_transform(), ha="center", va="top", fontsize=6.2,
            color=COLOR_DARK_NEUTRAL,
        )
    ax_fraction.text(
        0.03, 0.03, "Unclipped fractions; denominator-valid sample $D_b>10^{-6}$",
        transform=ax_fraction.transAxes, ha="left", va="bottom", fontsize=6.4, color="#555555",
    )

    handles = [
        Line2D([0], [0], marker="o", color=COLOR_BASE, markerfacecolor=COLOR_BASE, lw=0, markersize=5.5, label=r"$G_{\mathrm{Base}}$ / $F_{\mathrm{close}}$"),
        Line2D([0], [0], marker="^", color=COLOR_TGD, markerfacecolor=COLOR_TGD, lw=0, markersize=5.5, label=r"$G_{\mathrm{TGD}}$ / $F_{\mathrm{TGD}}^*$"),
        Line2D([0], [0], marker="o", color=COLOR_BASE, markerfacecolor="white", lw=0, markersize=5.5, label="open marker = fraction panel"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 1.015), ncol=3, frameon=False, fontsize=7.2)
    fig.text(
        0.5, -0.005,
        "Canonical = PCA/Ridge-smoothed field; Direct CN–IC = un-smoothed basin-wise CN–IC field. "
        "IC and dPL are separate panels of the comparison, not a ranking.",
        ha="center", va="top", fontsize=6.8, color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.19, top=0.82, wspace=0.28)
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
