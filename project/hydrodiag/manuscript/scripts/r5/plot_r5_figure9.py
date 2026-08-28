#!/usr/bin/env python3
"""Render the HESS/Copernicus-compliant Figure 9 (R5 cross-model structural replication).

Layout (6 panels across 3 rows, 19 cm width, 14.8 cm height):
    Row 1: (a) IC primary timing effect along snow gradient (Orange/Green/Blue host colors, shared y-axis)
           (b) dPL primary timing effect along snow gradient (Orange/Green/Blue host colors, hollow)
           (c) Spearman rho strip (grouped by host, matching host colors, starts at 0.0)
    Row 2: (d) Distribution across S1, S3, S5 (Orange/Green/Blue palette, thicker lines, all strata presented)
           (e) High snow signed CT (89 individual basin lines + Base->TGD->CN slopes)
    Row 3: (f) Compressed cross-host coherence composition bars (shifted up, single-row legend outside top-right)

Outputs PNG only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Path setup
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.scripts.shared.r1_plot_style import (  # noqa: E402
    COLOR_DARK_NEUTRAL,
    COLOR_SECONDARY_NEUTRAL,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)

RESULTS_DIR = HERE.parents[1] / "results" / "R5"
FIGURES_DIR = HERE.parents[1] / "figures"
OUTPUT_NAME = "Figure9_R5_cross_model_replication.png"

HOSTS = ("XAJ", "GR4J", "SIMHYD")
REGIMES = ("IC", "dPL")
STRUCTURES = ("Base", "TGD", "CN")
STRATA = ("S1", "S2", "S3", "S4", "S5")
STRATUM_COUNTS = {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}
ROW_ORDER = [(host, regime) for host in HOSTS for regime in REGIMES]

# Colors aligned with Figures 1-4
COLOR_BASE = "#D55E00"  # Orange (Base / XAJ / S1)
COLOR_TGD = "#009E73"   # Green (TGD / GR4J / S3)
COLOR_CN = "#0072B2"    # Blue (CN / SIMHYD / S5)
COLOR_TEXT = "#303438"  # Dark neutral
COLOR_REF = "#70767B"   # Zero line grey
COLOR_BAND = "#D0D5DA"  # Equivalence band

HOST_COLORS = {"XAJ": COLOR_BASE, "GR4J": COLOR_TGD, "SIMHYD": COLOR_CN}
HOST_MARKERS = {"XAJ": "o", "GR4J": "s", "SIMHYD": "D"}
MODEL_COLORS = {"Base": COLOR_BASE, "TGD": COLOR_TGD, "CN": COLOR_CN}
STRATUM_PALETTE = {"S1": COLOR_BASE, "S3": COLOR_TGD, "S5": COLOR_CN}

COMPOSITION_COLORS = {
    "all three positive": "#1F4E79",
    "exactly 2/3 positive": "#6BAED6",
    "≤1 host positive": "#D0D5DA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Build the R1-aligned statistical assets before rendering.",
    )
    parser.add_argument("--device", default=None, help="Device passed to the preparation script.")
    return parser.parse_args()


def load_assets(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = results_dir / "r5_figure9_canonical_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run prepare_r5_figure9_canonical.py first or use --prepare."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("estimand") != "abs(signed_CT_Base) - abs(signed_CT_CN)":
        raise ValueError("Figure 9 assets do not declare the R1 primary Base-CN CT estimand")
    primary = pd.read_csv(results_dir / "r5_figure9_primary_effects.csv")
    matrix = pd.read_csv(results_dir / "r5_figure9_primary_effect_matrix.csv")
    continuous = pd.read_csv(results_dir / "r5_figure9_continuous_summary.csv")
    timing = pd.read_csv(results_dir / "r5_figure9_timing_distributions.csv")
    agreement = pd.read_csv(results_dir / "r5_figure9_primary_agreement.csv")
    validate_assets(primary, matrix, continuous, timing, agreement)
    return primary, matrix, continuous, timing, agreement, manifest


def validate_assets(
    primary: pd.DataFrame,
    matrix: pd.DataFrame,
    continuous: pd.DataFrame,
    timing: pd.DataFrame,
    agreement: pd.DataFrame,
) -> None:
    expected_effects = [f"delta_abs_CT_Base_CN_{host}_{regime}" for host, regime in ROW_ORDER]
    missing = [column for column in expected_effects if column not in primary]
    if missing:
        raise ValueError(f"Missing primary effect columns: {missing}")
    if len(primary) != 531 or primary["basin_id"].nunique() != 531:
        raise ValueError("Primary asset must contain exactly 531 unique basins")
    if set(matrix["stratum"]) != set(STRATA) or len(matrix) != 30:
        raise ValueError("Primary matrix must contain 6 rows x 5 strata")
    if len(continuous) != 6 or set(continuous["host"]) != set(HOSTS):
        raise ValueError("Continuous summary must contain six host-regime rows")
    if set(timing["structure"]) != set(STRUCTURES):
        raise ValueError("Timing distribution asset is missing Base, TGD, or CN")
    if len(agreement) != 10 or set(agreement["stratum"]) != set(STRATA):
        raise ValueError("Agreement summary must contain five strata for each regime")


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 1200) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap CI for a stratum median."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(draws, values.size))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def primary_effect_summary(primary: pd.DataFrame) -> pd.DataFrame:
    """Summarize basin-level primary effects with bootstrap median intervals."""
    rng = np.random.default_rng(20260904)
    records: list[dict[str, object]] = []
    for host, regime in ROW_ORDER:
        column = f"delta_abs_CT_Base_CN_{host}_{regime}"
        for stratum in STRATA:
            values = primary.loc[primary["snow_stratum"] == stratum, column].dropna().to_numpy(float)
            ci_low, ci_high = bootstrap_median_ci(values, rng)
            records.append(
                {
                    "host": host,
                    "regime": regime,
                    "stratum": stratum,
                    "N": int(values.size),
                    "median": float(np.median(values)) if values.size else float("nan"),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame.from_records(records)


def draw_half_violin_bounded(
    ax: plt.Axes,
    center: float,
    values: np.ndarray,
    side: int,
    color: str,
    y_grid: np.ndarray,
) -> None:
    """Draw one half of a compact vertical violin truncated to data min/max."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    vmin, vmax = float(np.min(values)), float(np.max(values))
    sub_grid = y_grid[(y_grid >= vmin) & (y_grid <= vmax)]
    if len(sub_grid) < 2:
        return
    spread = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    bandwidth = max(2.0, 1.06 * spread * values.size ** (-0.2))
    kernel = np.exp(-0.5 * ((sub_grid[:, None] - values[None, :]) / bandwidth) ** 2).mean(axis=1)
    kmax = float(kernel.max())
    if kmax <= 0:
        return
    width = 0.35 * kernel / kmax
    # 1. Very light transparent fill inside
    ax.fill_betweenx(
        sub_grid,
        center,
        center + side * width,
        facecolor=color,
        edgecolor="none",
        alpha=0.16,
        zorder=1,
    )
    # 2. Fully closed boundary line in the corresponding stratum color
    x_boundary = np.concatenate([[center], center + side * width, [center], [center]])
    y_boundary = np.concatenate([[sub_grid[0]], sub_grid, [sub_grid[-1]], [sub_grid[0]]])
    ax.plot(
        x_boundary,
        y_boundary,
        color=color,
        linewidth=1.0,
        alpha=0.95,
        zorder=2,
    )


def composition_table(primary: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Classify each valid basin into one exhaustive 3-category host-state."""
    columns = [f"delta_abs_CT_Base_CN_{host}_{regime}" for host in HOSTS]
    values = primary[columns].to_numpy(float)
    valid = np.isfinite(values).all(axis=1)
    states = values[valid] > 0
    strata = primary.loc[valid, "snow_stratum"].to_numpy()
    rows: list[dict[str, object]] = []
    for stratum in STRATA:
        subset = states[strata == stratum]
        denominator = int(subset.shape[0])
        if denominator:
            count_positive = subset.sum(axis=1)
            masks = {
                "all three positive": count_positive == 3,
                "exactly 2/3 positive": count_positive == 2,
                "≤1 host positive": count_positive <= 1,
            }
        else:
            masks = {category: np.array([], dtype=bool) for category in ("all three positive", "exactly 2/3 positive", "≤1 host positive")}
        for category in ("all three positive", "exactly 2/3 positive", "≤1 host positive"):
            rows.append(
                {
                    "stratum": stratum,
                    "category": category,
                    "fraction": float(masks[category].mean()) if denominator else float("nan"),
                    "N": denominator,
                }
            )
    return pd.DataFrame(rows)


def export_si_host_pair_coherence(primary: pd.DataFrame, out_path: Path) -> None:
    """Export detailed host-pair decomposition table for Supplementary Information."""
    rows: list[dict[str, object]] = []
    for regime in REGIMES:
        columns = [f"delta_abs_CT_Base_CN_{host}_{regime}" for host in HOSTS]
        values = primary[columns].to_numpy(float)
        valid = np.isfinite(values).all(axis=1)
        states = values[valid] > 0
        strata = primary.loc[valid, "snow_stratum"].to_numpy()
        for stratum in STRATA:
            subset = states[strata == stratum]
            denominator = int(subset.shape[0])
            if denominator:
                count_positive = subset.sum(axis=1)
                le1 = float(np.mean(count_positive <= 1))
                eq2 = float(np.mean(count_positive == 2))
                eq3 = float(np.mean(count_positive == 3))
                ge2 = float(np.mean(count_positive >= 2))
                xaj_gr4j = float(np.mean(subset[:, 0] & subset[:, 1] & ~subset[:, 2]))
                xaj_simhyd = float(np.mean(subset[:, 0] & ~subset[:, 1] & subset[:, 2]))
                gr4j_simhyd = float(np.mean(~subset[:, 0] & subset[:, 1] & subset[:, 2]))
            else:
                le1 = eq2 = eq3 = ge2 = xaj_gr4j = xaj_simhyd = gr4j_simhyd = float("nan")
            rows.append(
                {
                    "regime": regime,
                    "stratum": stratum,
                    "N": denominator,
                    "P_le1_positive": le1,
                    "P_exactly_2_of_3": eq2,
                    "P_3_of_3": eq3,
                    "P_at_least_2": ge2,
                    "P_XAJ_GR4J_only": xaj_gr4j,
                    "P_XAJ_SIMHYD_only": xaj_simhyd,
                    "P_GR4J_SIMHYD_only": gr4j_simhyd,
                }
            )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Exported SI host-pair coherence table: {out_path}")


def render_figure9(out_dir: Path | None = None, prepare: bool = False, device: str | None = None) -> Path:
    setup_publication_style()
    results_dir = RESULTS_DIR
    if prepare:
        from manuscript.scripts.r5.prepare_r5_figure9_canonical import build_assets
        build_assets(results_dir, device)
    primary, _matrix, continuous, timing, agreement, _manifest = load_assets(results_dir)

    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_NAME

    export_si_host_pair_coherence(primary, out_dir / "SI_host_pair_coherence.csv")

    summary = primary_effect_summary(primary)

    # 19 cm width, 14.8 cm height
    fig_w_in = 19.0 / 2.54
    fig_h_in = 14.8 / 2.54
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # 3-row layout with comfortable vertical spacing (hspace=0.52)
    gs_main = fig.add_gridspec(
        3, 1,
        height_ratios=[1.0, 1.05, 0.72],
        hspace=0.52,
        left=0.070,
        right=0.985,
        top=0.950,
        bottom=0.065,
    )

    # Row 1: [ (a) IC + (b) dPL (compact gap) ] + (c) Spearman rho
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], width_ratios=[3.3, 1.0], wspace=0.22)
    gs_row1_ab = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_row1[0], width_ratios=[1.0, 1.0], wspace=0.08)
    ax_a = fig.add_subplot(gs_row1_ab[0])
    ax_b = fig.add_subplot(gs_row1_ab[1])
    ax_c = fig.add_subplot(gs_row1[1])

    # Row 2: (d) 60% : (e) 40% (width_ratios = [1.5, 1.0])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], width_ratios=[1.5, 1.0], wspace=0.20)
    ax_d = fig.add_subplot(gs_row2[0])
    ax_e = fig.add_subplot(gs_row2[1])

    # Row 3: (f) full width
    ax_f = fig.add_subplot(gs_main[2])

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL (a): IC Primary Timing Effect across S1-S5 (3 Hosts)
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_a)
    ax_a.set_title("(a) IC timing effect", loc="left", fontsize=9.2, fontweight="bold", pad=6)
    ax_a.axvspan(-5, 5, color=COLOR_BAND, alpha=0.25, zorder=0)
    ax_a.axvline(0, color=COLOR_REF, lw=0.8, ls="--", zorder=1)

    # Horizontal dashed line for each snow stratum tick
    for i in range(len(STRATA)):
        ax_a.axhline(i, color="#CBD5E1", ls="--", lw=0.75, alpha=0.8, zorder=0)

    y_offsets_hosts = {"XAJ": +0.18, "GR4J": 0.00, "SIMHYD": -0.18}
    y_positions = {s: i for i, s in enumerate(STRATA)}

    for stratum in STRATA:
        y_base = y_positions[stratum]
        for host in HOSTS:
            m = HOST_MARKERS[host]
            col = HOST_COLORS[host]
            row = summary[(summary["host"] == host) & (summary["regime"] == "IC") & (summary["stratum"] == stratum)].iloc[0]
            med = float(row["median"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            y_pos = y_base + y_offsets_hosts[host]

            ax_a.errorbar(
                med, y_pos, xerr=[[med - lo], [hi - med]],
                fmt=m, color=col, ecolor=col,
                markerfacecolor=col,
                markeredgecolor=col, markeredgewidth=1.0,
                markersize=4.6 if host != "SIMHYD" else 4.2,
                elinewidth=1.0, capsize=2.0, capthick=0.8,
                zorder=3,
            )

    ax_a.set_yticks(range(len(STRATA)))
    ax_a.set_yticklabels(list(STRATA), fontsize=8.0)
    ax_a.set_ylabel("Snow stratum", fontsize=8.5)
    ax_a.set_xlim(-10, 65)
    ax_a.set_xticks([0, 20, 40, 60])
    ax_a.set_xticklabels(["0", "20", "40", "60"], fontsize=8.0)
    ax_a.set_xlabel(r"$\Delta |CT|^{\mathrm{Base-CN}}$ (days)", fontsize=8.5)
    ax_a.grid(True, axis="x", linestyle=":", alpha=0.35, color=COLOR_BAND)

    # Shared Host Legend in (a) at lower right
    leg_a = [
        Line2D([0], [0], marker="o", color=COLOR_BASE, markerfacecolor=COLOR_BASE, markeredgecolor=COLOR_BASE, lw=1.2, markersize=4.4, label="XAJ"),
        Line2D([0], [0], marker="s", color=COLOR_TGD, markerfacecolor=COLOR_TGD, markeredgecolor=COLOR_TGD, lw=1.2, markersize=4.4, label="GR4J"),
        Line2D([0], [0], marker="D", color=COLOR_CN, markerfacecolor=COLOR_CN, markeredgecolor=COLOR_CN, lw=1.2, markersize=4.0, label="SIMHYD"),
    ]
    ax_a.legend(
        handles=leg_a,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=6.8,
        ncol=3,
        columnspacing=0.5,
        handletextpad=0.2,
        framealpha=0.92,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL (b): dPL Primary Timing Effect across S1-S5 (3 Hosts)
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_b)
    ax_b.set_title("(b) dPL timing effect", loc="left", fontsize=9.2, fontweight="bold", pad=6)
    ax_b.axvspan(-5, 5, color=COLOR_BAND, alpha=0.25, zorder=0)
    ax_b.axvline(0, color=COLOR_REF, lw=0.8, ls="--", zorder=1)

    # Horizontal dashed line for each snow stratum tick
    for i in range(len(STRATA)):
        ax_b.axhline(i, color="#CBD5E1", ls="--", lw=0.75, alpha=0.8, zorder=0)

    for stratum in STRATA:
        y_base = y_positions[stratum]
        for host in HOSTS:
            m = HOST_MARKERS[host]
            col = HOST_COLORS[host]
            row = summary[(summary["host"] == host) & (summary["regime"] == "dPL") & (summary["stratum"] == stratum)].iloc[0]
            med = float(row["median"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            y_pos = y_base + y_offsets_hosts[host]

            ax_b.errorbar(
                med, y_pos, xerr=[[med - lo], [hi - med]],
                fmt=m, color=col, ecolor=col,
                markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=1.1,
                markersize=4.6 if host != "SIMHYD" else 4.2,
                elinewidth=1.0, capsize=2.0, capthick=0.8,
                zorder=3,
            )

    ax_b.set_yticks(range(len(STRATA)))
    ax_b.tick_params(labelleft=False)
    ax_b.set_xlim(-10, 65)
    ax_b.set_xticks([0, 20, 40, 60])
    ax_b.set_xticklabels(["0", "20", "40", "60"], fontsize=8.0)
    ax_b.set_xlabel(r"$\Delta |CT|^{\mathrm{Base-CN}}$ (days)", fontsize=8.5)
    ax_b.grid(True, axis="x", linestyle=":", alpha=0.35, color=COLOR_BAND)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL (c): Grouped Model Spearman rho strip (Matching Host Colors)
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_c)
    ax_c.set_title(r"(c) Spearman $\rho$", loc="left", fontsize=9.2, fontweight="bold", pad=6)

    # Horizontal dashed line for each host tick
    for i in range(3):
        ax_c.axhline(i, color="#CBD5E1", ls="--", lw=0.75, alpha=0.8, zorder=0)

    y_positions_c = {"XAJ": 2, "GR4J": 1, "SIMHYD": 0}
    for host in HOSTS:
        y_base = y_positions_c[host]
        m = HOST_MARKERS[host]
        col = HOST_COLORS[host]
        for regime in REGIMES:
            row = continuous[(continuous["host"] == host) & (continuous["regime"] == regime)].iloc[0]
            rho = float(row["spearman_rho"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            is_ic = (regime == "IC")
            y_pos = y_base + (+0.14 if is_ic else -0.14)

            ax_c.errorbar(
                rho, y_pos, xerr=[[rho - lo], [hi - rho]],
                fmt=m, color=col, ecolor=col,
                markerfacecolor=col if is_ic else "white",
                markeredgecolor=col, markeredgewidth=1.1,
                markersize=4.6 if host != "SIMHYD" else 4.0,
                elinewidth=1.0, capsize=2.0, capthick=0.8,
                zorder=3,
            )
            ax_c.text(max(hi, rho) + 0.04, y_pos, f"{rho:+.2f}", va="center", ha="left", fontsize=6.8, color=col, weight="bold", zorder=4)

    ax_c.set_yticks([0, 1, 2])
    ax_c.set_yticklabels(["SIMHYD", "GR4J", "XAJ"], fontsize=8.0)
    ax_c.set_xlim(0.0, 1.05)
    ax_c.set_ylim(-0.5, 2.5)
    ax_c.set_xticks([0.0, 0.5, 1.0])
    ax_c.set_xticklabels(["0", "0.5", "1"], fontsize=8.0)
    ax_c.set_xlabel(r"Spearman $\rho$", fontsize=8.5)
    ax_c.grid(True, axis="x", linestyle=":", alpha=0.35, color=COLOR_BAND)

    leg_c = [
        Line2D([0], [0], marker="o", color=COLOR_TEXT, linestyle="none", markerfacecolor=COLOR_TEXT, markeredgecolor=COLOR_TEXT, markersize=4.2, label="IC"),
        Line2D([0], [0], marker="o", color=COLOR_TEXT, linestyle="none", markerfacecolor="white", markeredgecolor=COLOR_TEXT, markeredgewidth=1.0, markersize=4.2, label="dPL"),
    ]
    ax_c.legend(
        handles=leg_c,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        frameon=False,
        facecolor="none",
        edgecolor="none",
        fontsize=6.5,
        ncol=1,
        labelspacing=0.25,
        borderpad=0.3,
        handletextpad=0.2,
        framealpha=0.0,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL (d): Distribution across S1, S3, S5 (Orange/Green/Blue, 60% width)
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_d)
    ax_d.set_title("(d) Distribution across S1, S3, S5", loc="left", fontsize=9.2, fontweight="bold", pad=6)
    ax_d.axhspan(-5, 5, color=COLOR_BAND, alpha=0.25, zorder=0)
    ax_d.axhline(0, color=COLOR_REF, lw=0.8, ls="--", zorder=1)

    y_min_d, y_max_d = -55.0, 125.0
    y_grid_d = np.linspace(-60.0, 150.0, 400)
    host_centers_d = np.arange(len(HOSTS), dtype=float) * 3.0
    stratum_offsets = {"S1": -0.75, "S3": 0.0, "S5": 0.75}
    regime_sides = {"IC": -1, "dPL": 1}

    for host_idx, host in enumerate(HOSTS):
        host_center = host_centers_d[host_idx]
        for stratum in ("S1", "S3", "S5"):
            center = host_center + stratum_offsets[stratum]
            c_col = STRATUM_PALETTE[stratum]
            for regime in REGIMES:
                values = primary.loc[primary["snow_stratum"] == stratum, f"delta_abs_CT_Base_CN_{host}_{regime}"].dropna().to_numpy(float)
                if values.size == 0:
                    continue
                side = regime_sides[regime]
                draw_half_violin_bounded(ax_d, center, values, side, c_col, y_grid_d)
                med = float(np.median(values))

                is_ic = (regime == "IC")
                x_dot = center + side * 0.22
                ax_d.scatter(
                    [x_dot], [med],
                    marker="o" if is_ic else "^", s=22,
                    facecolor=c_col if is_ic else "white",
                    edgecolor=c_col, linewidth=1.0,
                    zorder=3,
                )
                pos_pct = float(np.mean(values > 0))
                x_annot = center + side * 0.35
                if stratum == "S1":
                    y_annot = -46.0 if is_ic else -37.0
                    ax_d.text(x_annot, y_annot, f"{pos_pct:.0%}", ha="center", va="top", fontsize=6.5, color=c_col, weight="normal", zorder=4)
                elif stratum == "S3":
                    if host == "XAJ":
                        y_annot = 56.0 if is_ic else 64.0
                    elif host == "SIMHYD":
                        y_annot = 64.0 if is_ic else 72.0
                    else:  # GR4J
                        y_annot = 80.0 if is_ic else 89.0
                    ax_d.text(x_annot, y_annot, f"{pos_pct:.0%}", ha="center", va="bottom", fontsize=6.5, color=c_col, weight="normal", zorder=4)
                elif stratum == "S5":
                    y_annot = 104.0 if is_ic else 113.0
                    ax_d.text(x_annot, y_annot, f"{pos_pct:.0%}", ha="center", va="bottom", fontsize=6.5, color=c_col, weight="normal", zorder=4)

    ax_d.set_xlim(-1.45, host_centers_d[-1] + 1.45)
    ax_d.set_ylim(y_min_d, y_max_d)
    ax_d.set_xticks(host_centers_d)
    ax_d.set_xticklabels(HOSTS, fontsize=8.0)
    ax_d.set_ylabel(r"$\Delta |CT|^{\mathrm{Base-CN}}$ (days)", fontsize=8.5)
    ax_d.grid(True, axis="y", linestyle=":", alpha=0.35, color=COLOR_BAND)

    leg_d = [
        Patch(facecolor=STRATUM_PALETTE[s], edgecolor=STRATUM_PALETTE[s], alpha=0.35, linewidth=1.0, label=s)
        for s in ("S1", "S3", "S5")
    ] + [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_TEXT, markeredgecolor=COLOR_TEXT, label="IC", markersize=4.2),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor=COLOR_TEXT, markeredgewidth=1.0, label="dPL", markersize=4.2),
    ]
    ax_d.legend(
        handles=leg_d,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.98),
        frameon=False,
        facecolor="none",
        edgecolor="none",
        fontsize=6.5,
        ncol=1,
        labelspacing=0.18,
        borderpad=0.15,
        handletextpad=0.20,
        framealpha=0.0,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL (e): High snow signed CT (40% width, 89 Individual Basin Lines)
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_e)
    ax_e.set_title("(e) High snow signed CT", loc="left", fontsize=9.2, fontweight="bold", pad=6)
    ax_e.axhline(0, color=COLOR_REF, lw=0.8, ls="--", zorder=1)

    high = timing[timing["frac_snow"] >= 0.30]
    host_centers_e = np.arange(len(HOSTS), dtype=float) * 3.6 + 1.2
    regime_offsets_e = {"IC": -0.65, "dPL": 0.65}
    structure_offsets_e = {"Base": -0.25, "TGD": 0.0, "CN": 0.25}

    for i, center in enumerate(host_centers_e):
        if i % 2 == 1:
            ax_e.axvspan(center - 1.2, center + 1.2, color="#F1F5F9", alpha=0.5, zorder=0)

    for host_idx, host in enumerate(HOSTS):
        center = host_centers_e[host_idx]
        for regime in REGIMES:
            is_ic = (regime == "IC")
            sub_hr = high[(high["host"] == host) & (high["regime"] == regime)]
            piv_hr = sub_hr.pivot(index="basin_id", columns="structure", values="signed_CT_error")

            pos_b = center + regime_offsets_e[regime] + structure_offsets_e["Base"]
            pos_t = center + regime_offsets_e[regime] + structure_offsets_e["TGD"]
            pos_c = center + regime_offsets_e[regime] + structure_offsets_e["CN"]

            # 1. 89 Individual Basin Connecting Lines
            for _, r_b in piv_hr.iterrows():
                ax_e.plot([pos_b, pos_t, pos_c], [r_b["Base"], r_b["TGD"], r_b["CN"]], color="#CBD5E1", lw=0.40, alpha=0.15, zorder=1)

            # 2. Individual basin scatter points
            ax_e.scatter(np.full(len(piv_hr), pos_b), piv_hr["Base"], s=5.0, color=COLOR_BASE, alpha=0.15, edgecolors="none", zorder=2)
            ax_e.scatter(np.full(len(piv_hr), pos_t), piv_hr["TGD"], s=5.0, color=COLOR_TGD, alpha=0.15, edgecolors="none", zorder=2)
            ax_e.scatter(np.full(len(piv_hr), pos_c), piv_hr["CN"], s=5.0, color=COLOR_CN, alpha=0.15, edgecolors="none", zorder=2)

            # 3. Medians + IQR whiskers
            med_list = []
            pos_list = [pos_b, pos_t, pos_c]
            for struct, pos in zip(STRUCTURES, pos_list):
                vals = piv_hr[struct].dropna().to_numpy(float)
                q25, med, q75 = np.quantile(vals, [0.25, 0.50, 0.75])
                med_list.append(med)

                c_s = MODEL_COLORS[struct]
                ax_e.errorbar(
                    pos, med, yerr=[[med - q25], [q75 - med]],
                    fmt="o" if is_ic else "^",
                    color=c_s, ecolor=c_s,
                    markerfacecolor=c_s if is_ic else "white",
                    markeredgecolor=c_s, markeredgewidth=1.1,
                    markersize=4.6, elinewidth=1.2, capsize=2.2, capthick=0.9,
                    zorder=4,
                )

            # 4. Bold Median Connecting Line
            ax_e.plot(pos_list, med_list, color=COLOR_TEXT, linestyle="-" if is_ic else "--", linewidth=1.5, alpha=0.90, zorder=3)

    ax_e.set_xlim(0.0, host_centers_e[-1] + 1.8)
    ax_e.set_ylim(-80, 15)
    ax_e.set_yticks([-80, -60, -40, -20, 0])
    ax_e.set_xticks(host_centers_e)
    ax_e.set_xticklabels(HOSTS, fontsize=8.0)
    ax_e.tick_params(axis="x", pad=2)
    ax_e.set_ylabel(r"Signed CT error (days)", fontsize=8.5)
    ax_e.grid(True, axis="y", linestyle=":", alpha=0.35, color=COLOR_BAND)

    leg_e = [
        Patch(facecolor=MODEL_COLORS[s], edgecolor=MODEL_COLORS[s], label=s)
        for s in STRUCTURES
    ] + [
        Line2D([0], [0], marker="o", color=COLOR_TEXT, linestyle="-", markerfacecolor=COLOR_TEXT, markeredgecolor=COLOR_TEXT, lw=1.0, label="IC", markersize=4.2),
        Line2D([0], [0], marker="^", color=COLOR_TEXT, linestyle="--", markerfacecolor="white", markeredgecolor=COLOR_TEXT, markeredgewidth=1.0, lw=1.0, label="dPL", markersize=4.2),
    ]
    ax_e.legend(
        handles=leg_e,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=6.5,
        ncol=5,
        columnspacing=0.5,
        handletextpad=0.20,
        framealpha=0.92,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL (f): Cross-host coherence composition (Single-row Legend at Upper Right Outside)
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_f)
    ax_f.set_title("(f) Cross-host coherence", loc="left", fontsize=9.2, fontweight="bold", pad=12)

    tables = {regime: composition_table(primary, regime) for regime in REGIMES}
    x_f = np.arange(len(STRATA), dtype=float) * 1.5
    offsets_f = {"IC": -0.22, "dPL": 0.22}
    width_f = 0.38

    for regime in REGIMES:
        is_ic = (regime == "IC")
        table = tables[regime]
        agr_sub = agreement[agreement["regime"] == regime].set_index("stratum")
        xs_f = x_f + offsets_f[regime]

        f_3 = np.array([float(table[(table["stratum"] == s) & (table["category"] == "all three positive")]["fraction"].iloc[0]) for s in STRATA])
        f_2 = np.array([float(table[(table["stratum"] == s) & (table["category"] == "exactly 2/3 positive")]["fraction"].iloc[0]) for s in STRATA])
        f_le1 = np.array([float(table[(table["stratum"] == s) & (table["category"] == "≤1 host positive")]["fraction"].iloc[0]) for s in STRATA])

        bottom_2 = f_3
        bottom_le1 = f_3 + f_2

        if is_ic:
            ax_f.bar(xs_f, f_3, width=width_f, bottom=0, color=COMPOSITION_COLORS["all three positive"], edgecolor="white", linewidth=0.45, zorder=2)
            ax_f.bar(xs_f, f_2, width=width_f, bottom=bottom_2, color=COMPOSITION_COLORS["exactly 2/3 positive"], edgecolor="white", linewidth=0.45, zorder=2)
            ax_f.bar(xs_f, f_le1, width=width_f, bottom=bottom_le1, color=COMPOSITION_COLORS["≤1 host positive"], edgecolor="white", linewidth=0.45, zorder=2)
        else:
            ax_f.bar(xs_f, f_3, width=width_f, bottom=0, facecolor="white", edgecolor=COMPOSITION_COLORS["all three positive"], linewidth=1.3, zorder=2)
            ax_f.bar(xs_f, f_2, width=width_f, bottom=bottom_2, facecolor="white", edgecolor=COMPOSITION_COLORS["exactly 2/3 positive"], linewidth=1.3, zorder=2)
            ax_f.bar(xs_f, f_le1, width=width_f, bottom=bottom_le1, facecolor="white", edgecolor=COMPOSITION_COLORS["≤1 host positive"], linewidth=1.3, zorder=2)

        for i, s in enumerate(STRATA):
            xi = xs_f[i]
            text_color = "white" if is_ic else COLOR_TEXT
            if f_3[i] >= 0.16:
                ax_f.text(xi, f_3[i] / 2, f"{f_3[i]:.0%}", ha="center", va="center", fontsize=6.2, color=text_color, weight="bold", zorder=4)
            if f_2[i] >= 0.16:
                ax_f.text(xi, bottom_2[i] + f_2[i] / 2, f"{f_2[i]:.0%}", ha="center", va="center", fontsize=6.2, color=text_color, weight="bold", zorder=4)

            pge2 = f_3[i] + f_2[i]
            ci_low = float(agr_sub.loc[s, "P_at_least_2_ci_low"])
            ci_high = float(agr_sub.loc[s, "P_at_least_2_ci_high"])
            ax_f.errorbar(xi, pge2, yerr=[[max(0.0, pge2 - ci_low)], [max(0.0, ci_high - pge2)]], fmt="none", ecolor=COLOR_TEXT, elinewidth=0.85, capsize=1.8, capthick=0.7, zorder=5)
            # Annotate percentage above bar without "IC" or "dPL" prefix
            ax_f.text(xi, 1.06, f"{pge2:.0%}", ha="center", va="bottom", fontsize=6.5, color=COLOR_TEXT, weight="bold", zorder=4)

    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(list(STRATA), fontsize=8.0)
    ax_f.set_xlim(-0.7, 7.2)
    ax_f.set_ylim(0, 1.05)
    ax_f.set_yticks([0, 0.5, 1.0])
    ax_f.set_yticklabels(["0", "0.5", "1.0"], fontsize=8.0)
    ax_f.set_ylabel("Coherence", fontsize=8.5)
    ax_f.set_xlabel("Snow-activity stratum", fontsize=8.5)

    # Single row legend in upper right of panel (f) outside
    leg_f = [
        Patch(facecolor=COMPOSITION_COLORS["all three positive"], edgecolor="none", label="all 3 positive"),
        Patch(facecolor=COMPOSITION_COLORS["exactly 2/3 positive"], edgecolor="none", label="exactly 2/3 positive"),
        Patch(facecolor=COMPOSITION_COLORS["≤1 host positive"], edgecolor="none", label="≤1 host positive"),
        Patch(facecolor="#64748B", edgecolor="none", label="IC (solid)"),
        Patch(facecolor="white", edgecolor=COLOR_TEXT, linewidth=1.3, label="dPL (open)"),
    ]
    ax_f.legend(
        handles=leg_f,
        loc="lower right",
        bbox_to_anchor=(0.99, 1.08),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=6.5,
        ncol=5,
        columnspacing=0.6,
        handletextpad=0.25,
        framealpha=0.92,
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Rendered Figure 9 (PNG only): {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    render_figure9(args.out_dir, args.prepare, args.device)


if __name__ == "__main__":
    main()
