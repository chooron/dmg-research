#!/usr/bin/env python3
"""Canonical four-panel R4 Figure 8 renderer.

This renderer uses the existing three-structure R4 state/timing exports.  It
is intentionally plot-only and writes an explicit ``_interim`` asset while
TGD2 provenance is pending.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.scripts.r4.common import default_results_root  # noqa: E402
from manuscript.scripts.r4.soil_analysis import calendar_month_anomaly  # noqa: E402
from manuscript.scripts.r4.plot_r4_figure8 import (  # noqa: E402
    _select_illustrative_basin_year,
 )
from manuscript.scripts.shared.r1_plot_style import (  # noqa: E402
    apply_clean_spines,
    setup_publication_style,
 )


def _z(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    month = dates.month.to_numpy()
    anomaly = calendar_month_anomaly(values.astype(float), month)
    return (anomaly - np.nanmean(anomaly)) / (np.nanstd(anomaly) + 1e-12)


def generate_figure8_canonical(
    results_root: Path,
    out_dir: Path,
    suffix: str = "_interim",
) -> Path:
    setup_publication_style()
    official = results_root / "r4_phase1_soil_official"
    basin_id, water_year, audit = _select_illustrative_basin_year(results_root)
    audit["renderer"] = "plot_r4_figure8_canonical.py"
    audit["state_chain"] = "W_total = wu + wl + wd"
    (HERE / "figure8_r4_selection_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    caravan = np.load(
        results_root
        / "r4_caravan_soil_reference_v1"
        / "caravan_soil_ensemble.npz"
    )
    basin_ids = [str(x).zfill(8) for x in caravan["basin_ids"]]
    idx = basin_ids.index(basin_id)
    test = slice(int(caravan["test_slice_start"]), int(caravan["test_slice_stop"]))
    dates = pd.to_datetime(caravan["dates"][test])
    ref = caravan["SM100"][idx, test].astype(float)
    swe = np.load(results_root / "r4_swe_reference_v1" / "swe_ensemble.npz")["swe_median"][idx, test].astype(float)

    arrays = {}
    for key, directory, filename in (("Base", "r4_official_dpl_XAJ_seed42", "official_dpl_XAJ_seed42_full_arrays.npz"), ("TGD2", "r4_official_dpl_XAJ_TGD2_seed42", "official_dpl_XAJ_TGD2_seed42_full_arrays.npz"), ("CN", "r4_official_dpl_XAJ_CN_seed42", "official_dpl_XAJ_CN_seed42_full_arrays.npz")):
        z = np.load(results_root / directory / filename)
        arrays[key] = (z["wu"][idx, test] + z["wl"][idx, test] + z["wd"][idx, test]).astype(float)

    df = pd.DataFrame(
        {
            "date": dates,
            "ref": ref,
            "swe": swe,
            **{f"{k}_z": _z(v, dates) for k, v in arrays.items()},
            "ref_z": _z(ref, dates),
        }
    )
    df["wy"] = df["date"].map(lambda d: d.year if d.month < 10 else d.year + 1)
    wy = df[df["wy"] == water_year].copy()
    timing = pd.read_csv(official / "three_structure_timing_metrics_basin_year.csv", dtype={"basin_id": str})
    timing = timing[(timing["regime"] == "dPL_seed42") & (timing["basin_id"] == basin_id) & (timing["water_year"] == water_year)]

    summary = pd.read_csv(official / "three_structure_timing_metrics_basin_summary.csv", dtype={"basin_id": str})
    paired = pd.read_csv(official / "three_structure_paired_structural_effects.csv", dtype={"basin_id": str})
    paired = paired[paired["regime"] == "dPL_seed42"].drop_duplicates("basin_id")
    q3 = float(paired["snow_burden_swe_mm"].quantile(0.75))
    q3_ids = set(paired.loc[paired["snow_burden_swe_mm"] >= q3, "basin_id"])
    pop = summary[(summary["regime"] == "dPL_seed42") & (summary["basin_id"].isin(q3_ids))]

    fig = plt.figure(figsize=(8.2, 6.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.12, 1.0),
        height_ratios=(1.15, 1.0),
        wspace=0.30,
        hspace=0.38,
        left=0.08,
        right=0.98,
        top=0.93,
        bottom=0.10,
    )

    # (a) One composite representative basin-year panel.
    ax = fig.add_subplot(gs[0, 0]); apply_clean_spines(ax)
    ax2 = ax.twinx()
    ax2.plot(wy["date"], wy["swe"], color="#9ECAE1", lw=1.0, alpha=0.9, label="Snow-17 SWE")
    ax2.fill_between(wy["date"], 0, wy["swe"], color="#9ECAE1", alpha=0.20)
    for key, color, ls in (("ref_z", "#555555", "-"), ("Base_z", "#EE7733", "--"), ("TGD2_z", "#009988", "-.") , ("CN_z", "#0077BB", "-")):
        label = "ERA5-Land SM100" if key == "ref_z" else key.replace("_z", "") + " Wtotal"
        ax.plot(wy["date"], wy[key], color=color, ls=ls, lw=1.15, label=label)
    ax.set_title("(a) Representative Q3 water year: snow → state response", loc="left", weight="bold")
    ax.set_ylabel("Standardized state anomaly"); ax2.set_ylabel("SWE [mm]")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(fontsize=5.8, frameon=False, ncol=2, loc="upper left")
    ax.text(0.99, 0.03, f"basin {basin_id}, WY {water_year}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)

    # (b)/(c) Population timing distributions.
    for col, metric, title, xlabel in (
        (0, "median_wetup_error_days", "(b) Spring wet-up timing", "Signed wet-up error (days)"),
        (1, "median_peak_error_days", "(c) Soil-water peak timing", "Signed peak error (days)"),
    ):
        ax = fig.add_subplot(gs[0, 1] if col == 0 else gs[1, 0])
        apply_clean_spines(ax)
        for key, color, ls in (
            ("Base", "#EE7733", "--"),
            ("TGD2", "#009988", "-."),
            ("CN", "#0077BB", "-"),
        ):
            vals = pop.loc[pop["structure"] == key, metric].dropna().to_numpy(float)
            if len(vals) == 0: continue
            x = np.sort(vals); y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=color, ls=ls, lw=1.25, label=key)
        ax.axvline(0, color="#555555", lw=0.7, ls=":")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative fraction (Q3)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6.0, frameon=False)

    # (d) Compact sensitivity rail. Prefer the three-structure table emitted
    # by the current run; fall back to the formal Base/CN raw sensitivity file.
    ax = fig.add_subplot(gs[1, 1])
    apply_clean_spines(ax)
    table_s7 = out_dir.parent / "tables" / f"TableS7_R4{suffix}.csv"
    sens = pd.read_csv(table_s7 if table_s7.exists() else official / "robustness_timing_sensitivity.csv")
    if "Definition" in sens:
        labels = sens["Definition"].astype(str).to_list()
        series = [("Base−TGD2 gain", "#009988", "^"), ("Base−CN gain", "#0077BB", "D")]
    else:
        labels = (sens["peak_definition"].astype(str) + " / " + sens["wetup_definition"].astype(str)).to_list()
        sens = sens.assign(**{"Base−CN gain": sens["peak_abs_error_improvement_days"].astype(float)})
        series = [("Base−CN gain", "#0077BB", "D")]
    labels = labels[:10]; y = np.arange(len(labels))
    for key, color, marker in series:
        if key not in sens: continue
        vals = pd.to_numeric(sens[key], errors="coerce").to_numpy()[:len(labels)]
        ax.plot(vals, y, color=color, marker=marker, lw=1.0, ms=3.5, label=key)
    ax.axvline(0, color="#555555", lw=0.7, ls=":"); ax.set_yticks(y, labels, fontsize=6.0); ax.invert_yaxis()
    ax.set_xlabel("Base MAE − model MAE (days)")
    ax.set_title(
        "(d) Timing-definition sensitivity",
        loc="left",
        weight="bold",
    )
    ax.legend(fontsize=6.0, frameon=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"Figure8_R4{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "manuscript" / "figures",
    )
    parser.add_argument("--suffix", default="_interim")
    args = parser.parse_args()
    print(generate_figure8_canonical(args.results_root, args.out_dir, args.suffix))
