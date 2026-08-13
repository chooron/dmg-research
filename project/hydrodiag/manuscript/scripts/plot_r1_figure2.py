"""
Plot R1 Figure 2 (final): seasonal center-of-timing errors among basins passing
the aggregate outlet-performance screen and environmental organization along snow gradient.

Three-column population-level evidence chain (3 x 3, column-major panel labels),
rows strictly Base / TGD / CN:

    Column 1 (a-c)  Screened timing error
                    ECDF of signed center-of-timing error, DeltaCT = CT_sim - CT_obs
                    among basins with standard KGE >= 0.60
    Column 2 (d-f)  Threshold robustness
                    fraction with |DeltaCT| >= 15 d vs. the KGE screening
                    threshold (y = KGE screening threshold, x = fraction;
                    grid t = 0.40..0.80 step 0.01)
    Column 3 (g-i)  Timing across snow regimes
                    population-level horizontal paired boxplots + connected medians
                    of DeltaCT across the five fixed snow regimes S1-S5
                    (all 531 basins per combination)

Terminology: Center of timing (CT) is the water-year day at which cumulative
runoff reaches 50% of the annual total; DeltaCT = CT_sim - CT_obs
(negative = simulated runoff timing earlier than observed). The canonical
data column `ct_error_signed` in r1_snow_signatures_basin_level.csv is
this signed error (days); code-internal names keep the canonical schema,
manuscript-facing text strictly uses CT / DeltaCT.

Canonical data (strict, unchanged):
  - Performance screen: standard KGE column `kge` from
    manuscript/results/R1/r1_basin_level_performance.csv (test rows)
  - Timing data: canonical `ct_error_signed` (DeltaCT), water year Oct-Sep,
    test period 1995-10-01 .. 2010-09-30. IC uses canonical selected_restart;
    dPL uses canonical median across seeds.
  - Snow fraction: `frac_snow` from manuscript/results/R1/r1_snow_attributes.csv.
  - Manuscript main screen threshold: KGE >= 0.60.
  - Large center-timing error reference: |DeltaCT| >= 15 d.

Output (PNG only): manuscript/plots/figures/Figure2_R1_ct_error_snow_regimes.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ── path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
SCRIPTS = HERE.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from r1_plot_style import (
    MODEL_COLORS,
    RESOLVED_FONT,
    setup_publication_style,
    apply_clean_spines,
)

# ── constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
R1_DIR = PROJECT_ROOT / "manuscript" / "results" / "R1"
PLOTS_FIG_DIR = PROJECT_ROOT / "manuscript" / "plots" / "figures"

SCREEN_THRESHOLD = 0.60   # predefined operational aggregate-performance screen
TIMING_THRESHOLD = 15.0   # |DeltaCT| large center-timing-error reference (days)
THRESHOLD_GRID = np.arange(0.40, 0.8001, 0.01)  # fixed sensitivity grid

PARADIGM_ORDER = ["IC-CMA-ES", "dPL-MLP"]
STRUCTURE_ORDER = ["Base", "TGD", "CN"]
MODEL_CODE = {"Base": "XAJ-Base", "TGD": "XAJ-TGD", "CN": "XAJ-CN"}
IC_DPL_LABEL = {"IC-CMA-ES": "IC", "dPL-MLP": "dPL"}

# Line-style encoding
IC_LINESTYLE = "-"
DPL_LINESTYLE = (0, (6.0, 3.0))   # clearly visible publication dash pattern

# Snow-fraction regime definitions (identical to Figure 1): S1-S5 by frac_snow
SNOW_BINS = [0.0, 0.05, 0.15, 0.30, 0.50, 1.0001]
SNOW_STRATA = ["S1", "S2", "S3", "S4", "S5"]
SNOW_REGIME_LABELS = ["S1", "S2", "S3", "S4", "S5"]


# ── display helpers (CT / DeltaCT terminology) ────────────────────────────────
def fmt_days(value: float) -> str:
    """Signed day value with a proper minus sign, e.g. -5 -> "-5 d"."""
    v = int(round(value))
    if v < 0:
        return f"\u2212{abs(v)} d"
    if v > 0:
        return f"+{v} d"
    return "0 d"


def fmt_pct(fraction: float) -> str:
    """One-decimal percentage, e.g. 0.1692 -> '16.9%'."""
    return f"{100.0 * fraction:.1f}%"


# ── canonical data loading ────────────────────────────────────────────────────
def load_performance() -> dict[tuple[str, str], pd.Series]:
    """Canonical test-period standard KGE per (paradigm, structure)."""
    perf = pd.read_csv(R1_DIR / "r1_basin_level_performance.csv")
    perf["basin_id"] = perf["basin_id"].astype(str).str.zfill(8)
    if not bool((perf["selected_run"] == True).all()):
        raise ValueError("performance table must contain only canonical selected-run rows")
    out: dict[tuple[str, str], pd.Series] = {}
    for paradigm in PARADIGM_ORDER:
        for structure in STRUCTURE_ORDER:
            sub = perf[(perf["paradigm"] == paradigm)
                       & (perf["model"] == MODEL_CODE[structure])
                       & (perf["period"] == "test")]
            out[(paradigm, structure)] = sub.set_index("basin_id")["kge"]
    return out


def load_ct_error() -> dict[tuple[str, str], pd.Series]:
    """Canonical basin-level signed center-timing error (DeltaCT, days)."""
    sig = pd.read_csv(R1_DIR / "r1_snow_signatures_basin_level.csv")
    sig["basin_id"] = sig["basin_id"].astype(str).str.zfill(8)
    out: dict[tuple[str, str], pd.Series] = {}
    for paradigm in PARADIGM_ORDER:
        for structure in STRUCTURE_ORDER:
            sub = sig[(sig["paradigm"] == paradigm)
                      & (sig["model"] == MODEL_CODE[structure])
                      & (sig["period"] == "test")]
            if paradigm == "IC-CMA-ES":
                series = sub[sub["seed_or_restart"] == "selected_restart"] \
                    .set_index("basin_id")["ct_error_signed"]
            else:
                series = sub.groupby("basin_id")["ct_error_signed"].median()
            out[(paradigm, structure)] = series
    return out


def load_snow_attributes() -> pd.DataFrame:
    """Canonical basin-level snow attributes."""
    attr = pd.read_csv(R1_DIR / "r1_snow_attributes.csv")
    attr["basin_id"] = attr["basin_id"].astype(str).str.zfill(8)
    attr["snow_bin"] = pd.cut(attr["frac_snow"], bins=SNOW_BINS, labels=SNOW_STRATA, right=False)
    return attr.set_index("basin_id")[["frac_snow", "snow_bin"]]


def combine(kge: dict, ct_err: dict, snow_attr: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Joined (basin_id, standard KGE, signed DeltaCT, frac_snow, snow_bin) frame per combination."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for paradigm in PARADIGM_ORDER:
        for structure in STRUCTURE_ORDER:
            k, d = kge[(paradigm, structure)], ct_err[(paradigm, structure)]
            common = k.index.intersection(d.index).intersection(snow_attr.index)
            df = pd.DataFrame({
                "kge": k[common],
                "ct_error_signed": d[common],
                "frac_snow": snow_attr.loc[common, "frac_snow"],
                "snow_bin": snow_attr.loc[common, "snow_bin"],
            }).dropna()
            out[(paradigm, structure)] = df
    return out


# ── pre-implementation check (canonical validation anchors) ──────────────────
def run_precheck(frames: dict[tuple[str, str], pd.DataFrame]) -> dict:
    """Print/verify the canonical data audit table required before plotting."""
    print("=" * 100)
    print("PRE-IMPLEMENTATION CHECK (canonical standard KGE + canonical ct_error_signed / DeltaCT)")
    print("=" * 100)
    header = (f"{'combination':18s} {'N_valid':>7s} {'N_screen':>8s} "
              f"{'N_large':>7s} {'total%':>7s} {'med_dCT':>7s} {'IQR':>5s}")
    print(header)
    stats: dict[tuple[str, str], dict] = {}
    for paradigm in PARADIGM_ORDER:
        for structure in STRUCTURE_ORDER:
            df = frames[(paradigm, structure)]
            n_valid = int(len(df))
            n_screen = int((df["kge"] >= SCREEN_THRESHOLD).sum())
            scr = df[df["kge"] >= SCREEN_THRESHOLD]
            n_large = int((scr["ct_error_signed"].abs() >= TIMING_THRESHOLD).sum())
            frac = n_large / n_screen if n_screen else float("nan")
            med = float(scr["ct_error_signed"].median())
            iqr = float(scr["ct_error_signed"].quantile(0.75) - scr["ct_error_signed"].quantile(0.25))
            combo = f"{IC_DPL_LABEL[paradigm]}-{structure}"
            print(f"{combo:18s} {n_valid:7d} {n_screen:8d} {n_large:7d} "
                  f"{100*frac:6.1f} {med:+7.0f} {iqr:5.1f}")
            stats[(paradigm, structure)] = {
                "n_valid": n_valid, "n_screen": n_screen, "n_screen_large": n_large,
                "fraction": frac, "median": med, "iqr": iqr,
            }
    # hard checks
    expected_counts = {
        ("IC-CMA-ES", "Base"): (331, 56, -5.0),
        ("IC-CMA-ES", "TGD"):  (394, 49, -3.0),
        ("IC-CMA-ES", "CN"):   (427, 25, 0.0),
        ("dPL-MLP", "Base"):   (344, 46, -2.0),
        ("dPL-MLP", "TGD"):    (404, 37, -1.0),
        ("dPL-MLP", "CN"):     (426, 20, 1.0),
    }
    for key, (exp_ns, exp_nl, exp_med) in expected_counts.items():
        st = stats[key]
        if st["n_valid"] != 531:
            raise SystemExit(f"BLOCKED: {key} N_valid={st['n_valid']} != 531")
        if st["n_screen"] != exp_ns or st["n_screen_large"] != exp_nl:
            raise SystemExit(f"BLOCKED: {key} screened ({st['n_screen']}/{st['n_screen_large']}) != expected ({exp_ns}/{exp_nl})")
        if abs(st["median"] - exp_med) > 1e-6:
            raise SystemExit(f"BLOCKED: {key} median={st['median']} != expected {exp_med}")
    print("  basin universe: 531 per combination for all six combinations .... OK")
    print("  test period:    1995-10-01 .. 2010-09-30 (canonical CSV) ........ OK")
    print("  timing data:    canonical ct_error_signed (DeltaCT) ............. OK")
    print("  screen metric:  canonical standard KGE (Figure 1 `kge` column) .. OK")
    print("=" * 100)
    return stats


# ── figure ────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_publication_style()
    os.makedirs(PLOTS_FIG_DIR, exist_ok=True)

    kge = load_performance()
    ct_err = load_ct_error()
    snow_attr = load_snow_attributes()
    frames = combine(kge, ct_err, snow_attr)
    stats = run_precheck(frames)

    # ── shared panel (a-c) x limits from all six screened subsets ────────────
    screened_vals = [frames[(p, s)].loc[frames[(p, s)]["kge"] >= SCREEN_THRESHOLD, "ct_error_signed"]
                     for p in PARADIGM_ORDER for s in STRUCTURE_ORDER]
    all_scr = np.concatenate([v.to_numpy(float) for v in screened_vals])
    x_pad = max(2.0, 0.03 * (all_scr.max() - all_scr.min()))
    ecdf_xlim = (float(all_scr.min()) - x_pad, float(all_scr.max()) + x_pad)

    # ── sensitivity curves (d-f), statistics unchanged ───────────────────────
    sens_curves: dict[tuple[str, str], np.ndarray] = {}
    denominators: dict[tuple[str, str], np.ndarray] = {}
    for p in PARADIGM_ORDER:
        for s in STRUCTURE_ORDER:
            df = frames[(p, s)]
            vals, dens = [], []
            for t in THRESHOLD_GRID:
                den = int((df["kge"] >= t).sum())
                num = int(((df["kge"] >= t) & (df["ct_error_signed"].abs() >= TIMING_THRESHOLD)).sum())
                dens.append(den)
                vals.append(num / den if den > 0 else np.nan)
            sens_curves[(p, s)] = np.array(vals)
            denominators[(p, s)] = np.array(dens)
    sens_max = np.nanmax(np.concatenate(list(sens_curves.values())))
    x_sens_max = max(0.10, float(np.ceil(sens_max * 20.0) / 20.0))  # common upper bound

    # ── figure layout: GridSpec(3, 3), equal columns ─────────────────────────
    fig_w = 18.2 / 2.54
    fig_h = 16.6 / 2.54
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        3, 3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.28, hspace=0.30,
        top=0.860, bottom=0.090, left=0.085, right=0.985,
    )
    ax_a = fig.add_subplot(gs[0, 0])   # column 1 (ECDF) Base
    ax_b = fig.add_subplot(gs[1, 0])   #                         TGD
    ax_c = fig.add_subplot(gs[2, 0])   #                         CN
    ax_d = fig.add_subplot(gs[0, 1])   # column 2 (threshold) Base
    ax_e = fig.add_subplot(gs[1, 1])   #                         TGD
    ax_f = fig.add_subplot(gs[2, 1])   #                         CN
    ax_g = fig.add_subplot(gs[0, 2])   # column 3 (snow boxplots) Base
    ax_h = fig.add_subplot(gs[1, 2])   #                          TGD
    ax_i = fig.add_subplot(gs[2, 2])   #                          CN

    # ── column headers (grey subtitles removed; details live in manuscript captions) ─
    pos_y1 = ax_a.get_position().y1
    header_y = pos_y1 + 0.024
    for ax, head in (
        (ax_a, "Screened timing error"),
        (ax_d, "Threshold robustness"),
        (ax_g, "Timing across snow regimes"),
    ):
        pos = ax.get_position()
        cx = 0.5 * (pos.x0 + pos.x1)
        fig.text(cx, header_y, head, ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold")

    # ── overall legend (IC / dPL only) ───────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color="#333333", linestyle=IC_LINESTYLE, lw=1.8,
               marker="o", ms=5.5, markerfacecolor="#333333",
               markeredgecolor="#333333", label="IC"),
        Line2D([0], [0], color="#333333", linestyle=DPL_LINESTYLE, lw=1.8,
               marker="o", ms=5.5, markerfacecolor="white",
               markeredgecolor="#333333", markeredgewidth=1.1, label="dPL"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.968), ncol=2, frameon=False,
               fontsize=8.5, handlelength=2.6, columnspacing=1.6)

    # ── column 1 (a-c): screened center-timing-error ECDF ────────────────────
    for r, (ax, structure, lab) in enumerate(zip(
            [ax_a, ax_b, ax_c], STRUCTURE_ORDER, ["(a)", "(b)", "(c)"])):
        apply_clean_spines(ax)
        color = MODEL_COLORS[structure]
        ax.axvspan(ecdf_xlim[0], -TIMING_THRESHOLD, facecolor="#B3B3B3",
                   alpha=0.09, edgecolor="none", zorder=0)
        ax.axvspan(TIMING_THRESHOLD, ecdf_xlim[1], facecolor="#B3B3B3",
                   alpha=0.09, edgecolor="none", zorder=0)
        ax.axvline(-TIMING_THRESHOLD, color="#999999", linewidth=0.7,
                   linestyle=(0, (3.0, 2.0)), zorder=1)
        ax.axvline(TIMING_THRESHOLD, color="#999999", linewidth=0.7,
                   linestyle=(0, (3.0, 2.0)), zorder=1)
        # F = 0.50 median guide (light dotted)
        ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":",
                   alpha=0.55, zorder=1)
        medians: dict[str, float] = {}
        for p in PARADIGM_ORDER:
            df = frames[(p, structure)]
            scr = np.sort(df.loc[df["kge"] >= SCREEN_THRESHOLD, "ct_error_signed"].to_numpy(float))
            y = np.arange(1, len(scr) + 1) / len(scr)
            ax.step(scr, y, where="post", color=color,
                    linestyle=IC_LINESTYLE if p == "IC-CMA-ES" else DPL_LINESTYLE,
                    linewidth=1.8, zorder=2)
            med = float(np.median(scr))
            medians[p] = med
            if p == "IC-CMA-ES":
                ax.plot([med], [0.5], marker="o", ms=4.5, color=color,
                        markerfacecolor=color, zorder=4)
            else:
                ax.plot([med], [0.5], marker="o", ms=4.5, color=color,
                        markerfacecolor="white", markeredgewidth=1.0, zorder=4)
        # median value labels
        ax.annotate(fmt_days(medians["IC-CMA-ES"]),
                    xy=(medians["IC-CMA-ES"], 0.5), xytext=(-5, 8),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=7.0, color="#333333", zorder=5)
        ax.annotate(fmt_days(medians["dPL-MLP"]),
                    xy=(medians["dPL-MLP"], 0.5), xytext=(5, -9),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=7.0, color="#333333", zorder=5)
        # +-15 d threshold labels
        ax.text(-TIMING_THRESHOLD, 0.935, "\u221215 d",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.5, color="#777777", zorder=5)
        ax.text(TIMING_THRESHOLD, 0.935, "+15 d",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.5, color="#777777", zorder=5)
        # structure identity (top-right)
        ax.text(0.965, 0.82, structure, transform=ax.transAxes, ha="right",
                va="center", fontsize=9.0, fontweight="bold", color=color, zorder=4)
        # compact two-line statistics box
        st_ic = stats[("IC-CMA-ES", structure)]
        st_dp = stats[("dPL-MLP", structure)]
        box = (f"IC\nn={st_ic['n_screen']}\nlarge={fmt_pct(st_ic['fraction'])}\n\n"
               f"dPL\nn={st_dp['n_screen']}\nlarge={fmt_pct(st_dp['fraction'])}")
        ax.text(0.035, 0.965, box, transform=ax.transAxes, va="top", ha="left",
                fontsize=6.8, linespacing=1.4, zorder=4,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF",
                          edgecolor="#E0E0E0", linewidth=0.5, alpha=0.85))
        ax.text(0.04, 0.07, lab, transform=ax.transAxes,
                fontsize=10.0, fontweight="bold", va="bottom", ha="left", zorder=4)
        ax.set_xlim(*ecdf_xlim)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        if r != 2:
            ax.set_xticklabels([])
    ax_b.set_ylabel("Empirical cumulative probability", fontsize=9.0)
    ax_c.set_xlabel("Center-of-timing error, \u0394CT (days)", fontsize=9.0)

    # ── column 2 (d-f): threshold robustness, rotated (y = threshold) ────────
    for r, (ax, structure, lab) in enumerate(zip(
            [ax_d, ax_e, ax_f], STRUCTURE_ORDER, ["(d)", "(e)", "(f)"])):
        apply_clean_spines(ax)
        color = MODEL_COLORS[structure]
        ax.axhline(SCREEN_THRESHOLD, color="#555555", linewidth=0.8, zorder=1)
        for p in PARADIGM_ORDER:
            vals = sens_curves[(p, structure)]
            ls = IC_LINESTYLE if p == "IC-CMA-ES" else DPL_LINESTYLE
            ax.plot(vals, THRESHOLD_GRID, color=color, linestyle=ls,
                    linewidth=1.8, zorder=2)
            t60 = int(np.argmin(np.abs(THRESHOLD_GRID - SCREEN_THRESHOLD)))
            v60 = vals[t60]
            if np.isfinite(v60):
                if p == "IC-CMA-ES":
                    ax.plot([v60], [SCREEN_THRESHOLD], marker="o", ms=4.0,
                            color=color, markerfacecolor=color, zorder=3)
                    ax.annotate(fmt_pct(v60), xy=(v60, SCREEN_THRESHOLD),
                                xytext=(5, 8), textcoords="offset points",
                                ha="left", va="bottom", fontsize=7.0,
                                color=color, zorder=4)
                else:
                    ax.plot([v60], [SCREEN_THRESHOLD], marker="o", ms=4.0,
                            color=color, markerfacecolor="white",
                            markeredgewidth=1.0, zorder=3)
                    ax.annotate(fmt_pct(v60), xy=(v60, SCREEN_THRESHOLD),
                                xytext=(-5, -9), textcoords="offset points",
                                ha="right", va="top", fontsize=7.0,
                                color=color, zorder=4)
            # small endpoint markers at KGE = 0.40 and 0.80
            for t_idx in (0, len(THRESHOLD_GRID) - 1):
                v = vals[t_idx]
                if np.isfinite(v):
                    if p == "IC-CMA-ES":
                        ax.plot([v], [THRESHOLD_GRID[t_idx]], marker="o", ms=2.5,
                                color=color, markerfacecolor=color, zorder=3)
                    else:
                        ax.plot([v], [THRESHOLD_GRID[t_idx]], marker="o", ms=2.5,
                                color=color, markerfacecolor="white",
                                markeredgewidth=0.8, zorder=3)
        ax.text(0.96, 0.90, structure, transform=ax.transAxes, ha="right",
                va="top", fontsize=9.0, fontweight="bold", color=color, zorder=4)
        ax.text(0.04, 0.07, lab, transform=ax.transAxes,
                fontsize=10.0, fontweight="bold", va="bottom", ha="left", zorder=4)
        ax.set_xlim(0.0, x_sens_max)
        ax.set_ylim(0.40, 0.80)
        ax.set_yticks([0.40, 0.50, 0.60, 0.70, 0.80])
        yticks = np.arange(0.0, x_sens_max + 1e-9, 0.05)
        ax.set_xticks(yticks)
        ax.set_xticklabels([f"{int(100 * v)}%" for v in yticks])
        if r != 2:
            ax.set_xticklabels([])
    ax_e.set_ylabel("KGE screening threshold", fontsize=9.0)
    ax_f.set_xlabel("Fraction with |\u0394CT| \u2265 15 d", fontsize=9.0)

    # ── column 3 (g-i): horizontal paired boxplots + connected medians ──────
    y_strata = np.arange(len(SNOW_STRATA))
    pos_ic = y_strata - 0.13
    pos_dp = y_strata + 0.13
    col3_xlim = (-102.0, 38.0)

    for r, (ax, structure, lab) in enumerate(zip(
            [ax_g, ax_h, ax_i], STRUCTURE_ORDER, ["(g)", "(h)", "(i)"])):
        apply_clean_spines(ax)
        color = MODEL_COLORS[structure]

        # Main reference line at DeltaCT = 0 d
        ax.axvline(0.0, color="#555555", linewidth=0.9, zorder=1)
        # Operational threshold lines at +-15 d
        ax.axvline(-TIMING_THRESHOLD, color="#BBBBBB", linewidth=0.6,
                   linestyle=(0, (3.0, 2.0)), zorder=1)
        ax.axvline(TIMING_THRESHOLD, color="#BBBBBB", linewidth=0.6,
                   linestyle=(0, (3.0, 2.0)), zorder=1)

        df_ic = frames[("IC-CMA-ES", structure)]
        df_dp = frames[("dPL-MLP", structure)]

        data_ic = [df_ic.loc[df_ic["snow_bin"] == s_bin, "ct_error_signed"].to_numpy(float)
                   for s_bin in SNOW_STRATA]
        data_dp = [df_dp.loc[df_dp["snow_bin"] == s_bin, "ct_error_signed"].to_numpy(float)
                   for s_bin in SNOW_STRATA]

        # Draw IC boxplots (structure-color light fill, alpha=0.22)
        bp_ic = ax.boxplot(data_ic, vert=False, positions=pos_ic, widths=0.20,
                           showfliers=False, patch_artist=True, manage_ticks=False,
                           zorder=2)
        for box in bp_ic["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.22)
            box.set_edgecolor(color)
            box.set_linewidth(1.0)
        for element in ("whiskers", "caps"):
            for line in bp_ic[element]:
                line.set_color(color)
                line.set_linewidth(1.0)
        for median in bp_ic["medians"]:
            median.set_color(color)
            median.set_linewidth(1.2)

        # Draw dPL boxplots (white fill, structure-color edge)
        bp_dp = ax.boxplot(data_dp, vert=False, positions=pos_dp, widths=0.20,
                           showfliers=False, patch_artist=True, manage_ticks=False,
                           zorder=2)
        for box in bp_dp["boxes"]:
            box.set_facecolor("white")
            box.set_edgecolor(color)
            box.set_linewidth(1.0)
        for element in ("whiskers", "caps"):
            for line in bp_dp[element]:
                line.set_color(color)
                line.set_linewidth(1.0)
        for median in bp_dp["medians"]:
            median.set_color(color)
            median.set_linewidth(1.2)

        # Connected median trajectories
        meds_ic = [float(np.median(d)) for d in data_ic]
        meds_dp = [float(np.median(d)) for d in data_dp]

        ax.plot(meds_ic, pos_ic, color=color, linestyle=IC_LINESTYLE, linewidth=1.1,
                marker="o", ms=4.5, markerfacecolor=color, markeredgecolor=color, zorder=4)
        ax.plot(meds_dp, pos_dp, color=color, linestyle=DPL_LINESTYLE, linewidth=1.1,
                marker="o", ms=4.5, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.0, zorder=4)

        # Structure identity (top-right) and panel label
        ax.text(0.96, 0.92, structure, transform=ax.transAxes, ha="right",
                va="top", fontsize=9.0, fontweight="bold", color=color, zorder=4)
        ax.text(0.04, 0.07, lab, transform=ax.transAxes,
                fontsize=10.0, fontweight="bold", va="bottom", ha="left", zorder=4)

        ax.set_xlim(*col3_xlim)
        ax.set_yticks(y_strata)
        ax.set_yticklabels(SNOW_REGIME_LABELS, fontsize=7.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.invert_yaxis()  # low-snow (0-0.05) at top, high-snow (0.50-1.00) at bottom

        if r != 2:
            ax.set_xticklabels([])

    ax_h.set_ylabel("Snow-fraction interval", fontsize=9.0)
    ax_i.set_xlabel("Center-of-timing error, \u0394CT (days)", fontsize=9.0)

    # ── save PNG only ────────────────────────────────────────────────────────
    png_path = PLOTS_FIG_DIR / "Figure2_R1_ct_error_snow_regimes.png"
    fig.savefig(png_path, dpi=600, format="png", bbox_inches="tight",
                facecolor="#FFFFFF")
    plt.close(fig)

    size_mb = os.path.getsize(png_path) / 1024 / 1024
    print(f"\nFigure 2 generated successfully.")
    print(f"  PNG   : {png_path}  ({size_mb:.2f} MB)")
    print(f"  Font  : {RESOLVED_FONT}")

    # ── post-hoc consistency validation ──────────────────────────────────────
    print("\nPOST-HOC VALIDATION")
    for p in PARADIGM_ORDER:
        for s in STRUCTURE_ORDER:
            t60 = int(np.argmin(np.abs(THRESHOLD_GRID - SCREEN_THRESHOLD)))
            v60 = sens_curves[(p, s)][t60]
            ann = stats[(p, s)]["fraction"]
            ok = abs(v60 - ann) < 1e-12
            print(f"  {IC_DPL_LABEL[p]:3s}-{s:5s} KGE=0.60 marker {fmt_pct(v60)} "
                  f"vs stats {fmt_pct(ann)} -> {'OK' if ok else 'MISMATCH'}")
            med = float(frames[(p, s)].loc[frames[(p, s)]["kge"] >= SCREEN_THRESHOLD,
                                            "ct_error_signed"].median())
            ok_m = abs(med - stats[(p, s)]["median"]) < 1e-9
            print(f"      screened median {med:+.0f} d (marker) vs stats "
                  f"{stats[(p, s)]['median']:+.0f} d -> {'OK' if ok_m else 'MISMATCH'}")
    print(f"  threshold grid: {THRESHOLD_GRID[0]:.2f}..{THRESHOLD_GRID[-1]:.2f} "
          f"step={THRESHOLD_GRID[1]-THRESHOLD_GRID[0]:.2f}, "
          f"n={len(THRESHOLD_GRID)}  (fixed, unchanged)")
    print(f"  min denominator at t=0.80: "
          f"{min(denominators[(p, s)][40] for p in PARADIGM_ORDER for s in STRUCTURE_ORDER)}")
    print(f"  column-2 common x fraction upper bound: {x_sens_max:.2f}")

    print("\nCOLUMN 3 PAIRED BOXPLOT & MEDIAN TRAJECTORY VALIDATION")
    for s in STRUCTURE_ORDER:
        print(f"--- Structure: {s} ---")
        for p in PARADIGM_ORDER:
            df = frames[(p, s)]
            meds, q1s, q3s, iqrs = [], [], [], []
            for s_bin in SNOW_STRATA:
                vals = df.loc[df["snow_bin"] == s_bin, "ct_error_signed"].to_numpy(float)
                q25, q50, q75 = np.percentile(vals, [25, 50, 75])
                meds.append(q50)
                q1s.append(q25)
                q3s.append(q75)
                iqrs.append(q75 - q25)
            med_str = ", ".join([f"{m:+.1f}" for m in meds])
            print(f"  {IC_DPL_LABEL[p]:3s} box medians = connected line points: [{med_str}]")


if __name__ == "__main__":
    main()
