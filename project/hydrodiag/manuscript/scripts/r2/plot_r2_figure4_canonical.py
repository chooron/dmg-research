#!/usr/bin/env python3
"""Canonical two-panel R2 Figure 4 renderer.

The legacy R2 renderer remains available for provenance, while this renderer
is the only one allowed to write the canonical Figure 4 filename.  It uses
frozen R2 CSV summaries and performs no upstream statistical recomputation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.scripts.r2.plot_r2_figure4 import (  # noqa: E402
    DISPLAY,
    KEY_PARAMS,
    REGIMES,
    load_data,
    ridge_density,
)
from manuscript.scripts.shared.r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    apply_clean_spines,
    setup_publication_style,
)

PARAM_ORDER = list(DISPLAY)


def render(out_dir: Path | None = None) -> Path:
    setup_publication_style()
    df_g, df_c, df_p = load_data()
    out_dir = out_dir or (PROJECT / "manuscript" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "Figure4_R2_final.png"

    fig = plt.figure(figsize=(8.2, 5.1))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.35),
        wspace=0.36,
        left=0.09,
        right=0.98,
        bottom=0.14,
        top=0.91,
    )

    # (a) All-parameter snow-gradient summary, aligned IC/dPL facets.
    ax = fig.add_subplot(gs[0, 0])
    apply_clean_spines(ax)
    y = np.arange(len(PARAM_ORDER))
    for offset, paradigm in ((0.16, "IC"), (-0.16, "dPL")):
        sub = df_g[df_g["paradigm"] == paradigm].copy()
        if "parameter" not in sub or "beta" not in sub:
            raise ValueError("R2 snow-gradient source lacks parameter/beta columns")
        sub = sub.set_index("parameter").reindex(PARAM_ORDER)
        x = sub["beta"].to_numpy(float)
        lo = sub["ci95_low"].to_numpy(float)
        hi = sub["ci95_high"].to_numpy(float)
        valid = np.isfinite(x)
        ax.errorbar(
            x[valid],
            y[valid] + offset,
            xerr=[x[valid] - lo[valid], hi[valid] - x[valid]],
            fmt="o" if paradigm == "IC" else "^",
            color=MODEL_COLORS["CN" if paradigm == "IC" else "Base"],
            ms=3.2,
            lw=0.8,
            capsize=1.8,
            label=paradigm,
        )
    for idx, param in enumerate(PARAM_ORDER):
        if param in KEY_PARAMS:
            ax.axhspan(idx - 0.42, idx + 0.42, color="#F5F5F5", zorder=0)
    ax.axvline(0, color="#555555", lw=0.7, ls=":")
    ax.set_yticks(y, [DISPLAY[p] for p in PARAM_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Base − CN snow-gradient slope")
    ax.set_title(
        "(a) All shared parameters",
        loc="left",
        weight="bold",
    )
    ax.legend(frameon=False, fontsize=7, title="constraint regime")

    # (b) Key directional signatures: three aligned rows, S1–S5 distributions.
    ax = fig.add_subplot(gs[0, 1])
    apply_clean_spines(ax)
    grid = np.linspace(-1.0, 1.0, 240)
    for row, param in enumerate(KEY_PARAMS):
        for paradigm, color, ls in (("IC", MODEL_COLORS["CN"], "-"), ("dPL", MODEL_COLORS["Base"], "--")):
            for regime_idx, regime in enumerate(REGIMES):
                sub = df_p[
                    (df_p["paradigm"] == paradigm)
                    & (df_p["parameter"] == param)
                    & (df_p["snow_regime"] == regime)
                ]
                if sub.empty:
                    continue
                values = sub["delta_base_minus_cn"].to_numpy(float)
                density = ridge_density(values, grid, 0.075)
                baseline = row * 2.4 + (0.34 if paradigm == "IC" else -0.34)
                scale = 0.75 * density
                ax.fill_between(
                    grid,
                    baseline,
                    baseline + (scale if paradigm == "IC" else -scale),
                    color=color,
                    alpha=0.12 if paradigm == "dPL" else 0.25,
                )
                ax.plot(grid, baseline + (scale if paradigm == "IC" else -scale), color=color, ls=ls, lw=0.65)
        ax.axhline(row * 2.4, color="#DDDDDD", lw=0.6)
        ax.text(-1.03, row * 2.4, DISPLAY[param], ha="right", va="center", fontsize=8, weight="bold")
    ax.axvline(0, color="#555555", lw=0.7, ls=":")
    ax.set_xlim(-1, 1); ax.set_ylim(-1.5, 6.5); ax.set_yticks([])
    ax.set_xlabel("Δz = z$_{Base}$ − z$_{CN}$")
    ax.set_title(
        "(b) Key paired shifts across S1–S5",
        loc="left",
        weight="bold",
    )
    ax.plot([], [], color=MODEL_COLORS["CN"], label="IC")
    ax.plot([], [], color=MODEL_COLORS["Base"], ls="--", label="dPL")
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    fig.suptitle(
        "R2 parameter-space reorganization under snow-process omission",
        fontsize=10,
        weight="bold",
    )
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    print(render(args.out_dir))
