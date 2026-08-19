#!/usr/bin/env python3
"""R3 SI component-level truth diagnostics (the only new SI figure for R3).

One compact TWO-PANEL figure answering:

> Are the aggregate parameter/state conclusions (C_theta, C_state) distributed
> across multiple components rather than driven by a single isolated parameter
> or state?

Panel (a): parameter-level CN-adjusted excess errors
  * all 15 shared XAJ parameters (COMMON_XAJ order, top = xaj_k)
  * Base vs TGD2, IC/dPL facets sharing one x-axis (median |e_M - e_CN|)
  * paired-basin bootstrap 95 % CI (2000 reps, seed 20260730)
  * frozen tier encoding from paired_parameters.csv 'tier' column
    (primary solid+bold, secondary hollow, exploratory grey) -- per facet
Panel (b): state-level CN-adjusted excess errors (delta_E, NRMSE, test)
  * primary states wu, wl, s, qi, qg (solid) + secondary wd (hollow, distinct)
  * same Base-vs-TGD2 / IC-dPL grammar as (a)

Data (canonical, read-only):
  results/r3_misspec_analysis_v1/paired_parameters.csv   |delta_e| per basin/param
  results/r3_misspec_analysis_v1/state_excess.csv        delta_E (nrmse, test)
  r3/protocol_misspec_v1.json                            frozen tiers + states

dPL handling follows the established R3 convention: per-basin median over the
three seeds (42/123/2026) BEFORE the basin-level median/bootstrap.

Output: manuscript/supplement/figures/Fig_S5_R3_components.png (PNG only, 600 dpi).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

sys.path.insert(0, str(HERE.parents[1] / "shared"))
from r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    apply_clean_spines,
    setup_publication_style,
)

DEFAULT_RESULTS_ROOT = PROJECT / "results"
RUN_ID = "r3_misspec_analysis_v1"
SUPP_FIG_DIR = PROJECT / "manuscript" / "figures"
OUT_NAME = "Fig_S5_R3_components.png"

SEEDS = (42, 123, 2026)
BOOT_N = 2000
BOOT_SEED = 20260730

C_BASE = MODEL_COLORS["Base"]  # #EE7733  omitted-process baseline (fitted)
C_TGD = MODEL_COLORS["TGD"]  # #009988  generic temperature-memory control
C_TEXT = "#333333"
C_GREY = "#999999"

# Frozen shared-XAJ parameter order (r3/common.py COMMON_XAJ) - top to bottom
COMMON_XAJ = [
    "xaj_k",
    "xaj_b",
    "xaj_im",
    "xaj_um",
    "xaj_lm",
    "xaj_dm",
    "xaj_c",
    "xaj_sm",
    "xaj_ex",
    "xaj_ki",
    "xaj_kg",
    "xaj_ci",
    "xaj_cg",
    "xaj_a",
    "xaj_theta",
]
# Primary states of the C_state aggregate (protocol state_estimands).
PRIMARY_STATES = ["wu", "wl", "s", "qi", "qg"]
SECONDARY_STATES = ["wd"]


def boot_ci_median(values: np.ndarray, n_boot: int = BOOT_N, seed: int = BOOT_SEED):
    """Paired basin-level bootstrap 95% CI of the median (R3 convention)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = np.median(values[idx])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def per_basin_values_long(df, value_col, group_cols, reg):
    """Per-basin values with dPL seed-median aggregation (IC passthrough)."""
    if reg == "dPL":
        sub = df[df["paradigm"] == "dPL"]
        return sub.groupby(["basin_id"] + group_cols, as_index=False)[
            value_col
        ].median()
    sub = df[df["paradigm"] == "IC"]
    return sub[["basin_id"] + group_cols + [value_col]].copy()


def load_tiers(protocol_path: Path) -> dict:
    proto = json.loads(protocol_path.read_text())
    return {
        "ic_primary": set(proto["predeclared_parameter_tiers"]["ic_primary"]),
        "ic_secondary": set(
            proto["predeclared_parameter_tiers"]["ic_secondary_supporting"]
        ),
        "dpl_primary": set(proto["predeclared_parameter_tiers"]["dpl_primary"]),
        "dpl_secondary": set(
            proto["predeclared_parameter_tiers"]["dpl_secondary_supporting"]
        ),
    }


def tier_style(reg: str, param: str, tiers: dict) -> dict:
    """Per-facet tier style: primary bold/solid, secondary hollow, exploratory grey."""
    pri = tiers["ic_primary" if reg == "IC" else "dpl_primary"]
    sec = tiers["ic_secondary" if reg == "IC" else "dpl_secondary"]
    if param in pri:
        return {"bold": True, "size": 7.5, "fill": True, "grey": False}
    if param in sec:
        return {"bold": False, "size": 6.5, "fill": False, "grey": False}
    return {"bold": False, "size": 6.0, "fill": False, "grey": True}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_parameter_data(results_root: Path, tiers: dict) -> dict:
    pp = pd.read_csv(results_root / "paired_parameters.csv")
    pp["abs_d"] = pp["delta_e"].abs()
    out = {}
    for reg in ("IC", "dPL"):
        rows = []
        g = per_basin_values_long(pp, "abs_d", ["structure", "parameter"], reg)
        for struct in ("Base", "TGD2"):
            sub = g[g["structure"] == struct]
            for p in COMMON_XAJ:
                vals = sub.loc[sub["parameter"] == p, "abs_d"].to_numpy()
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                lo, hi = boot_ci_median(vals)
                st = tier_style(reg, p, tiers)
                rows.append(
                    {
                        "reg": reg,
                        "structure": struct,
                        "parameter": p,
                        "median": float(np.median(vals)),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "bold": st["bold"],
                        "size": st["size"],
                        "fill": st["fill"],
                        "grey": st["grey"],
                    }
                )
        out[reg] = pd.DataFrame(rows)
    return out


def prepare_state_data(results_root: Path) -> dict:
    se = pd.read_csv(results_root / "state_excess.csv")
    se = se[(se["period"] == "test") & (se["metric"] == "nrmse")]
    states = PRIMARY_STATES + SECONDARY_STATES
    out = {}
    for reg in ("IC", "dPL"):
        rows = []
        g = per_basin_values_long(se, "delta_E", ["structure", "variable"], reg)
        for struct in ("Base", "TGD2"):
            sub = g[g["structure"] == struct]
            for v in states:
                vals = sub.loc[sub["variable"] == v, "delta_E"].to_numpy()
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                lo, hi = boot_ci_median(vals)
                is_sec = v in SECONDARY_STATES
                rows.append(
                    {
                        "reg": reg,
                        "structure": struct,
                        "variable": v,
                        "median": float(np.median(vals)),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "bold": not is_sec,
                        "size": 7.5 if not is_sec else 6.5,
                        "fill": not is_sec,
                        "grey": False,
                    }
                )
        out[reg] = pd.DataFrame(rows)
    return out


def shared_xlim(df_ic: pd.DataFrame, df_dpl: pd.DataFrame) -> tuple:
    """Common x-limits across the IC/dPL facets of one panel."""
    all_x = np.concatenate(
        [
            np.concatenate([df_ic["ci_hi"].to_numpy(), df_ic["ci_lo"].to_numpy()]),
            np.concatenate([df_dpl["ci_hi"].to_numpy(), df_dpl["ci_lo"].to_numpy()]),
        ]
    )
    all_x = all_x[np.isfinite(all_x)]
    xmax = float(np.max(np.abs(all_x)))
    xlo = min(0.0, float(np.min(all_x)) - 0.06 * xmax)
    return xlo, xmax * 1.12


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _draw_forest(ax, df, y_order, y_key, reg, show_labels, xlim, xlabel, show_legend):
    """Forest plot: y = component (top-to-bottom), x = median effect + CI.

    Base = orange circles, TGD2 = teal triangles; rows offset slightly so both
    markers are visible.  Tier styling (bold/solid vs hollow/grey) is applied
    per facet from the frozen 'tier' column.
    """
    ypos = {name: len(y_order) - 1 - i for i, name in enumerate(y_order)}

    for struct, marker, color, off in (
        ("Base", "o", C_BASE, -0.18),
        ("TGD2", "^", C_TGD, +0.18),
    ):
        sub = df[(df["structure"] == struct) & (df["reg"] == reg)]
        xs, ys, xerr_lo, xerr_hi, fills, sizes = [], [], [], [], [], []
        for _, row in sub.iterrows():
            y = ypos[row[y_key]] + off
            xs.append(row["median"])
            ys.append(y)
            xerr_lo.append(row["median"] - row["ci_lo"])
            xerr_hi.append(row["ci_hi"] - row["median"])
            fills.append(row["fill"])
            sizes.append(row["size"])
        ax.errorbar(
            xs,
            ys,
            xerr=[xerr_lo, xerr_hi],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            capthick=1.0,
            alpha=0.9,
            zorder=3,
        )
        ax.scatter(
            xs,
            ys,
            marker=marker,
            s=np.asarray(sizes) ** 2,
            color=color,
            facecolors=[color if f else "white" for f in fills],
            edgecolors=color,
            linewidths=0.9,
            zorder=4,
        )

    # y tick labels (tier styling per facet)
    if show_labels:
        labels, colors_lab, weights = [], [], []
        for name in y_order:
            tier_row = df[(df["reg"] == reg) & (df[y_key] == name)]
            if len(tier_row):
                colors_lab.append(C_GREY if tier_row.iloc[0]["grey"] else C_TEXT)
                weights.append("bold" if tier_row.iloc[0]["bold"] else "normal")
            else:
                colors_lab.append(C_TEXT)
                weights.append("normal")
            labels.append(name)
        ax.set_yticks([ypos[n] for n in y_order])
        ax.set_yticklabels(labels, fontsize=8.5)
        for t, c, w in zip(ax.get_yticklabels(), colors_lab, weights):
            t.set_color(c)
            t.set_fontweight(w)
    else:
        ax.set_yticks([ypos[n] for n in y_order])
        ax.set_yticklabels(["" for _ in y_order])

    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.axvline(0.0, color=C_GREY, linestyle="--", linewidth=0.9, zorder=1)
    ax.grid(True, axis="x", linestyle=":", alpha=0.25)
    ax.text(
        0.02,
        0.97,
        f"{reg} regime",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=C_TEXT,
    )

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=C_BASE,
                linestyle="none",
                markerfacecolor=C_BASE,
                markersize=6.5,
                label="Base",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color=C_TGD,
                linestyle="none",
                markerfacecolor=C_TGD,
                markersize=7.0,
                label="TGD2",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.97),
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            fontsize=8.0,
        )


def build_figure(results_root: Path, protocol_path: Path, out_path: Path) -> None:
    tiers = load_tiers(protocol_path)
    param_data = prepare_parameter_data(results_root, tiers)
    state_data = prepare_state_data(results_root)

    fig = plt.figure(figsize=(8.8, 9.6))
    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[1.0, 0.92],
        hspace=0.34,
        left=0.11,
        right=0.985,
        top=0.95,
        bottom=0.06,
    )

    # --- Panel (a): parameters (IC | dPL facets sharing one x-axis) ---
    gsa = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0], width_ratios=[1.0, 1.0], wspace=0.10
    )
    ax_p_ic = fig.add_subplot(gsa[0, 0])
    apply_clean_spines(ax_p_ic)
    ax_p_dp = fig.add_subplot(gsa[0, 1])
    apply_clean_spines(ax_p_dp)
    ax_p_ic.set_title(
        "(a) Parameter-level excess errors", weight="bold", loc="left", pad=6
    )
    xlim_p = shared_xlim(param_data["IC"], param_data["dPL"])
    xlab_p = "Median $|e_M - e_{CN}|$"
    _draw_forest(
        ax_p_ic,
        param_data["IC"],
        COMMON_XAJ,
        "parameter",
        "IC",
        show_labels=True,
        xlim=xlim_p,
        xlabel=xlab_p,
        show_legend=True,
    )
    _draw_forest(
        ax_p_dp,
        param_data["dPL"],
        COMMON_XAJ,
        "parameter",
        "dPL",
        show_labels=True,
        xlim=xlim_p,
        xlabel=xlab_p,
        show_legend=False,
    )
    ax_p_dp.text(
        0.98,
        0.03,
        "filled = primary (aggregate members)\nopen = secondary / exploratory",
        transform=ax_p_dp.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=C_GREY,
        linespacing=1.4,
    )

    # --- Panel (b): states (same grammar) ---
    gsb = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1], width_ratios=[1.0, 1.0], wspace=0.10
    )
    ax_s_ic = fig.add_subplot(gsb[0, 0])
    apply_clean_spines(ax_s_ic)
    ax_s_dp = fig.add_subplot(gsb[0, 1])
    apply_clean_spines(ax_s_dp)
    ax_s_ic.set_title("(b) State-level excess errors", weight="bold", loc="left", pad=6)
    state_order = PRIMARY_STATES + SECONDARY_STATES
    xlim_s = shared_xlim(state_data["IC"], state_data["dPL"])
    xlab_s = "Median $\\Delta$NRMSE (test)"
    _draw_forest(
        ax_s_ic,
        state_data["IC"],
        state_order,
        "variable",
        "IC",
        show_labels=True,
        xlim=xlim_s,
        xlabel=xlab_s,
        show_legend=True,
    )
    _draw_forest(
        ax_s_dp,
        state_data["dPL"],
        state_order,
        "variable",
        "dPL",
        show_labels=True,
        xlim=xlim_s,
        xlabel=xlab_s,
        show_legend=False,
    )
    ax_s_dp.text(
        0.98,
        0.03,
        "open = secondary state (wd)",
        transform=ax_s_dp.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=C_GREY,
    )

    SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600)
    print(f"saved: {out_path}")
    plt.close()


def main() -> None:
    setup_publication_style()
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
        }
    )
    results_root = DEFAULT_RESULTS_ROOT / RUN_ID
    protocol_path = PROJECT / "r3" / "protocol_misspec_v1.json"
    out_path = SUPP_FIG_DIR / OUT_NAME
    build_figure(results_root, protocol_path, out_path)
    print("Figure S5 (R3 component diagnostics) generated successfully.")


if __name__ == "__main__":
    main()
