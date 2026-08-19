#!/usr/bin/env python3
"""Canonical seven-panel R1 Figure 2 renderer.

This is a plot-only renderer.  It consumes the existing basin-level KGE and
water-year CT summaries; it never recomputes model outputs.  A legacy
``XAJ-TGD`` source is explicitly labelled as legacy and writes an interim
asset rather than silently calling it TGD2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.scripts.r1.plot_r1_figure2 import (  # noqa: E402
    SCREEN_THRESHOLD,
    SNOW_BINS,
    SNOW_STRATA,
    load_ct_error,
    load_performance,
    load_snow_attributes,
)
from manuscript.scripts.shared.r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    MODEL_LABELS,
    apply_clean_spines,
    setup_publication_style,
)

STRUCTURES = ("Base", "TGD", "CN")


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[np.isfinite(values)])
    return x, np.arange(1, len(x) + 1) / len(x)


def _load_frames() -> tuple[dict, dict, pd.DataFrame, str]:
    perf_path = (
        PROJECT
        / "manuscript"
        / "results"
        / "R1"
        / "r1_basin_level_performance.csv"
    )
    if not perf_path.exists():
        raise FileNotFoundError(f"missing canonical R1 source: {perf_path}")
    raw = pd.read_csv(perf_path)
    source_models = set(raw["model"].astype(str))
    has_tgd2 = any("TGD2" in value for value in source_models)
    control_label = "TGD2" if has_tgd2 else MODEL_LABELS["TGD"]
    return load_performance(), load_ct_error(), load_snow_attributes(), control_label


def render(out_dir: Path | None = None) -> Path:
    setup_publication_style()
    kge, ct, snow, control_label = _load_frames()
    out_dir = out_dir or (PROJECT / "manuscript" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = control_label == "TGD2"
    out = out_dir / ("Figure2_R1_final.png" if canonical else "Figure2_R1_interim.png")

    fig = plt.figure(figsize=(8.2, 8.0))
    gs = fig.add_gridspec(3, 3, width_ratios=(1.0, 1.35, 1.0), hspace=0.48, wspace=0.28)
    # Left: screened ECDFs, one row per structure.
    for row, structure in enumerate(STRUCTURES):
        ax = fig.add_subplot(gs[row, 0])
        apply_clean_spines(ax)
        for paradigm, ls, marker in (("IC-CMA-ES", "-", "o"), ("dPL-MLP", "--", "^") ):
            frame = kge[(paradigm, structure)].to_frame("kge").join(ct[(paradigm, structure)].rename("ct"), how="inner")
            vals = frame.loc[frame["kge"] >= SCREEN_THRESHOLD, "ct"].to_numpy(float)
            x, y = _ecdf(vals)
            paradigm_label = paradigm.replace("-CMA-ES", "").replace("-MLP", "")
            ax.step(
                x,
                y,
                where="post",
                color=MODEL_COLORS[structure],
                ls=ls,
                lw=1.35,
                label=paradigm_label,
            )
        ax.axvline(0, color="#555555", lw=0.75, ls=":")
        ax.axvline(-15, color="#AAAAAA", lw=0.7, ls="--")
        ax.axvline(15, color="#AAAAAA", lw=0.7, ls="--")
        title = f"({chr(97 + row)}) {MODEL_LABELS.get(structure, structure)} screened CT error"
        ax.set_title(
            title,
            loc="left",
            fontsize=8.5,
            weight="bold",
        )
        ax.set_xlim(-120, 120)
        ax.set_ylim(0, 1)
        ax.set_ylabel("ECDF" if row == 1 else "")
        ax.set_xlabel("ΔCT = CT$_{sim}$ − CT$_{obs}$ (days)")
        if row == 0:
            ax.legend(fontsize=6.4, frameon=False, loc="lower right")

    # Center: direct population-level threshold response.
    ax = fig.add_subplot(gs[:, 1])
    apply_clean_spines(ax)
    for structure in STRUCTURES:
        for paradigm, marker, ls in (("IC-CMA-ES", "o", "-"), ("dPL-MLP", "^", "--")):
            frame = kge[(paradigm, structure)].to_frame("kge").join(
                ct[(paradigm, structure)].rename("ct"),
                how="inner",
            )
            rows = []
            for threshold in np.arange(0.40, 0.8001, 0.01):
                screened = frame.loc[frame["kge"] >= threshold, "ct"]
                rows.append(float((screened.abs() >= 15).mean()) if len(screened) else np.nan)
            ax.plot(np.arange(0.40, 0.8001, 0.01), rows, color=MODEL_COLORS[structure], marker=marker, markevery=10, ms=3.2, lw=1.15, ls=ls, label=f"{MODEL_LABELS.get(structure, structure)} / {paradigm.replace('-CMA-ES','').replace('-MLP','')}" )
    ax.set_title("(d) Large timing-error prevalence after KGE screening", loc="left", fontsize=8.5, weight="bold")
    ax.set_xlabel("KGE screening threshold")
    ax.set_ylabel("P(|ΔCT| ≥ 15 d | KGE ≥ threshold)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.0, frameon=False, ncol=2)

    # Right: full-population S1–S5 distributions, one row per structure.
    snow = snow.copy()
    snow["snow_bin"] = pd.cut(
        snow["frac_snow"],
        bins=SNOW_BINS,
        labels=SNOW_STRATA,
        right=False,
    )
    for row, structure in enumerate(STRUCTURES):
        ax = fig.add_subplot(gs[row, 2])
        apply_clean_spines(ax)
        for paradigm, marker, ls in (("IC-CMA-ES", "o", "-"), ("dPL-MLP", "^", "--")):
            vals_by_stratum = []
            medians = []
            for stratum in SNOW_STRATA:
                frame = kge[(paradigm, structure)].to_frame("kge").join(
                    ct[(paradigm, structure)].rename("ct"),
                    how="inner",
                ).join(
                    snow[["snow_bin"]],
                    how="inner",
                )
                vals = frame.loc[frame["snow_bin"].astype(str).eq(stratum), "ct"].to_numpy(float)
                vals_by_stratum.append(vals)
                medians.append(np.nanmedian(vals) if len(vals) else np.nan)
            positions = np.arange(1, 6) + (-0.12 if paradigm == "IC-CMA-ES" else 0.12)
            ax.boxplot(vals_by_stratum, positions=positions, widths=0.18, showfliers=False, patch_artist=False, boxprops={"color": MODEL_COLORS[structure]}, whiskerprops={"color": MODEL_COLORS[structure]}, capprops={"color": MODEL_COLORS[structure]}, medianprops={"color": MODEL_COLORS[structure], "lw": 1.2}, manage_ticks=False)
            ax.plot(positions, medians, color=MODEL_COLORS[structure], marker=marker, ls=ls, lw=0.9, ms=3.0)
        ax.axhline(0, color="#555555", lw=0.75, ls=":")
        ax.axhline(-15, color="#AAAAAA", lw=0.7, ls="--"); ax.axhline(15, color="#AAAAAA", lw=0.7, ls="--")
        ax.set_title(f"({chr(101+row)}) {MODEL_LABELS.get(structure, structure)} by snow stratum", loc="left", fontsize=8.5, weight="bold")
        ax.set_xticks(range(1, 6), SNOW_STRATA); ax.set_xlabel("frac_snow stratum")
        ax.set_ylabel("ΔCT (days)" if row == 1 else "")
        ax.set_ylim(-120, 120)

    fig.suptitle(
        "R1 timing error: screened distribution, threshold response, "
        "and snow stratification",
        y=0.995,
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
