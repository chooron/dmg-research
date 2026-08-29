#!/usr/bin/env python3
"""Reproduce the frozen HUC-2 leave-one-region-out forest plot from CSVs.

The submission asset is rendered from the CSVs below, retaining only source
HUC_11–HUC_18 and displaying them as HUC_01–HUC_08. This renderer does
not recompute any LORO result.
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
ROBUSTNESS = PROJECT / "results" / "reviewer2_robustness" / "regional_loro"
DEFAULT_OUT = MANUSCRIPT / "supplement" / "figures" / "FigureS5_huc2_loro_robustness.png"

sys.path.insert(0, str(MANUSCRIPT / "scripts" / "shared"))
from r1_plot_style import (  # noqa: E402
    COLOR_BASE,
    COLOR_CN,
    COLOR_LIGHT_REF,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)


REGIONS = [f"HUC_{i:02d}" for i in range(11, 19)]
DISPLAY_REGION_LABELS = [f"HUC_{i:02d}" for i in range(1, 9)]


def _load(name: str) -> pd.DataFrame:
    path = ROBUSTNESS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _region_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["region_removed"].astype(str).isin(REGIONS)].copy()



def _plot_panel(ax: plt.Axes, frame: pd.DataFrame, value_column: str, x_label: str, title: str, unit_scale: float = 1.0) -> None:
    apply_clean_spines(ax)
    rows = _region_rows(frame)
    y = np.arange(len(REGIONS))
    all_plotted_values = []
    for paradigm, color, marker in (("IC", COLOR_CN, "o"), ("dPL", COLOR_BASE, "^")):
        sub = rows.loc[rows["paradigm"].eq(paradigm)].set_index("region_removed").reindex(REGIONS)
        if sub[value_column].isna().any():
            raise ValueError(f"Missing regions for {paradigm} in {value_column}")
        vals = sub[value_column].to_numpy(dtype=float) * unit_scale
        all_plotted_values.extend(vals.tolist())
        ax.scatter(vals, y, color=color, marker=marker, s=29, zorder=3, label=f"{paradigm} (omit HUC)")
        full_paradigm = frame.loc[(~frame["region_removed"].astype(str).str.startswith("HUC_")) & frame["paradigm"].eq(paradigm)]
        if len(full_paradigm) != 1:
            raise ValueError(f"Expected one full row for {paradigm}")
        ref = float(full_paradigm.iloc[0][value_column]) * unit_scale
        all_plotted_values.append(ref)
        linestyle = "--" if paradigm == "IC" else ":"
        ax.axvline(ref, color=color, linestyle=linestyle, linewidth=1.25, alpha=0.85, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(DISPLAY_REGION_LABELS, fontsize=7.0)
    ax.set_xlabel(x_label, labelpad=3)
    ax.set_title(title, loc="left", weight="bold", pad=6, fontsize=8.8)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.45, color=COLOR_LIGHT_REF)
    ax.set_ylim(-0.6, len(REGIONS) - 0.4)
    # Adaptive X-axis calculation with padding
    vmin, vmax = min(all_plotted_values), max(all_plotted_values)
    span = vmax - vmin if vmax > vmin else 1.0
    pad = span * 0.15
    ax.set_xlim(vmin - pad, vmax + pad)

def build_figure(out_path: Path) -> Path:
    setup_publication_style()
    r1 = _load("r1_huc2_loro.csv")
    r3 = _load("r3_huc2_loro.csv")
    r5 = _load("r5_huc2_loro.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.5), sharey=False)
    _plot_panel(axes[0], r1, "S5_minus_S1_contrast", r"S5–S1 Timing Contrast $\Delta CT$ (days)", "(a) R1: High- vs. Low-Snow Timing Effect", 1.0)
    _plot_panel(axes[1], r3, "Delta_F_median", r"Paired Recovery Contrast $\Delta F = F_{\mathrm{TGD}}^* - F_{\mathrm{close}}$", "(b) R3: TGD vs. Base Gap Recovery", 1.0)
    _plot_panel(axes[2], r5, "P_majority_positive", r"Majority Host Agreement $P(A \geq 2)$ in S5 (%)", "(c) R5: High-Snow Cross-Host Coherence", 100.0)
    handles = [
        Line2D([0], [0], marker="o", color=COLOR_CN, markerfacecolor=COLOR_CN, lw=0, markersize=5.4, label="IC (omit HUC)"),
        Line2D([0], [0], marker="^", color=COLOR_BASE, markerfacecolor=COLOR_BASE, lw=0, markersize=5.4, label="dPL (omit HUC)"),
        Line2D([0], [0], color=COLOR_CN, lw=1.3, ls="--", label="Full IC"),
        Line2D([0], [0], color=COLOR_BASE, lw=1.3, ls=":", label="Full dPL"),
    ]
    axes[0].legend(handles=[handles[2], handles[3], handles[0], handles[1]], loc="lower left", frameon=False, fontsize=7.0)
    axes[1].legend(handles=[handles[2], handles[3], handles[0], handles[1]], loc="lower left", frameon=False, fontsize=7.0)
    axes[2].legend(handles=[handles[2], handles[3], handles[0], handles[1]], loc="lower left", frameon=False, fontsize=7.0)
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.10, top=0.94, wspace=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, facecolor="white", edgecolor="none")
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
