#!/usr/bin/env python3
"""Render component-wise R3 truth-relative excess errors from frozen CSVs only."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent / "Figure_S3.png"
RESULTS = PROJECT / "results" / "r3_misspec_analysis_v1"
PROTOCOL = PROJECT / "manuscript" / "scripts" / "r3" / "protocol_misspec_v1.json"
SHARED = PROJECT / "manuscript" / "scripts" / "shared"
sys.path.insert(0, str(SHARED))
from r1_plot_style import (  # noqa: E402
    COLOR_BASE,
    COLOR_CN,
    COLOR_LIGHT_REF,
    COLOR_TGD,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)

PARAMETERS = [
    "xaj_k", "xaj_b", "xaj_im", "xaj_um", "xaj_lm", "xaj_dm", "xaj_c",
    "xaj_sm", "xaj_ex", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg", "xaj_a", "xaj_theta",
]
STATES = ["wu", "wl", "wd", "wt", "qi", "qg"]
STATE_LABELS = {
    "wu": r"$W_{U,t}$",
    "wl": r"$W_{L,t}$",
    "wd": r"$W_{D,t}$",
    "wt": r"$W_t$",
    "qi": r"$Q_{i,t}$",
    "qg": r"$Q_{g,t}$",
}
PARAMETER_LABELS = {
    "xaj_k": r"$k$", "xaj_b": r"$b$", "xaj_im": r"$i_m$", "xaj_um": r"$u_m$",
    "xaj_lm": r"$l_m$", "xaj_dm": r"$d_m$", "xaj_c": r"$c$", "xaj_sm": r"$s_m$",
    "xaj_ex": r"$ex$", "xaj_ki": r"$k_i$", "xaj_kg": r"$k_g$", "xaj_ci": r"$c_i$",
    "xaj_cg": r"$c_g$", "xaj_a": r"$a$", "xaj_theta": r"$\theta$",
}
BOOT_SEED = 20260730
BOOT_N = 2000


def basin_level(frame: pd.DataFrame, value: str, group: str, paradigm: str) -> pd.DataFrame:
    sub = frame.loc[frame["paradigm"].eq(paradigm)].copy()
    if sub.empty:
        raise ValueError(f"No {paradigm} rows for {group}")
    # IC has one selected restart; dPL is median across seeds before basin-level summaries.
    return sub.groupby(["basin_id", "structure", group], as_index=False)[value].median()


def median_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite values available for component")
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.median(values[rng.integers(0, len(values), size=(BOOT_N, len(values)))], axis=1)
    return float(np.median(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summaries(frame: pd.DataFrame, value: str, group: str, ordered: list[str], paradigm: str) -> pd.DataFrame:
    grouped = basin_level(frame, value, group, paradigm)
    rows = []
    for structure in ("Base", "TGD2"):
        for component in ordered:
            values = grouped.loc[(grouped["structure"].eq(structure)) & grouped[group].eq(component), value].to_numpy(float)
            median, lo, hi = median_ci(values)
            rows.append({"structure": structure, "component": component, "median": median, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def plot_facet(ax, data: pd.DataFrame, ordered: list[str], labels: dict[str, str], paradigm: str, xlabel: str, show_ylabel: bool) -> None:
    apply_clean_spines(ax)
    y = np.arange(len(ordered), dtype=float)
    for structure, color, offset in (("Base", COLOR_BASE, -0.16), ("TGD2", COLOR_TGD, 0.16)):
        sub = data.loc[data["structure"].eq(structure)].set_index("component").reindex(ordered)
        if sub[["median", "lo", "hi"]].isna().any().any():
            raise ValueError(f"Missing component summary for {paradigm}/{structure}")
        ax.errorbar(
            sub["median"].to_numpy(), y + offset,
            xerr=[sub["median"].to_numpy() - sub["lo"].to_numpy(), sub["hi"].to_numpy() - sub["median"].to_numpy()],
            fmt="o" if paradigm == "IC" else "^",
            color=color,
            markerfacecolor=color if paradigm == "IC" else "white",
            markeredgecolor=color,
            markersize=4.5,
            markeredgewidth=0.9,
            capsize=2.0,
            elinewidth=1.0,
            linestyle="none",
            zorder=3,
        )
    ax.axvline(0.0, color=COLOR_ZERO_LINE, linewidth=0.8, zorder=1)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.45, color=COLOR_LIGHT_REF)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in ordered], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, labelpad=3)
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)


def main() -> None:
    setup_publication_style()
    parameters = pd.read_csv(RESULTS / "paired_parameters.csv")
    states = pd.read_csv(RESULTS / "state_excess.csv")
    if not set(PARAMETERS).issubset(set(parameters["parameter"].unique())):
        raise ValueError("The canonical R3 parameter source does not contain all 15 shared XAJ parameters")
    available_states = set(states["variable"].unique())
    missing_states = set(STATES) - available_states
    if missing_states:
        raise ValueError(f"Missing canonical R3 state/flux variables: {sorted(missing_states)}")
    states = states.loc[(states["period"].eq("test")) & (states["metric"].eq("nrmse")) & states["variable"].isin(STATES)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), sharex="row", gridspec_kw={"wspace": 0.08, "hspace": 0.28})
    param_data = {reg: summaries(parameters.loc[parameters["parameter"].isin(PARAMETERS)], "delta_abs_e", "parameter", PARAMETERS, reg) for reg in ("IC", "dPL")}
    state_data = {reg: summaries(states, "delta_E", "variable", STATES, reg) for reg in ("IC", "dPL")}
    xparam = np.concatenate([d[["lo", "hi"]].to_numpy().ravel() for d in param_data.values()])
    xstate = np.concatenate([d[["lo", "hi"]].to_numpy().ravel() for d in state_data.values()])
    param_lim = max(abs(np.nanmin(xparam)), abs(np.nanmax(xparam))) * 1.15
    state_lim = max(abs(np.nanmin(xstate)), abs(np.nanmax(xstate))) * 1.15
    for ax, reg in zip(axes[0], ("IC", "dPL")):
        plot_facet(ax, param_data[reg], PARAMETERS, PARAMETER_LABELS, reg, r"Excess parameter error $\Delta|e| = |e_M|-|e_{CN}|$", ax is axes[0, 0])
        ax.set_xlim(-param_lim, param_lim)
    for ax, reg in zip(axes[1], ("IC", "dPL")):
        plot_facet(ax, state_data[reg], STATES, STATE_LABELS, reg, r"Excess state/flux error $\Delta E = \mathrm{NRMSE}_M-\mathrm{NRMSE}_{CN}$", ax is axes[1, 0])
        ax.set_xlim(-state_lim, state_lim)
    axes[0, 0].set_title("(a) Shared XAJ parameter errors", loc="left", weight="bold", pad=6, fontsize=9.0)
    axes[0, 1].set_title("dPL facet", loc="left", weight="bold", pad=6, fontsize=8.5)
    axes[1, 0].set_title("(b) Common-state and flux errors", loc="left", weight="bold", pad=6, fontsize=9.0)
    axes[1, 1].set_title("dPL facet", loc="left", weight="bold", pad=6, fontsize=8.5)
    handles = [
        Line2D([0], [0], marker="o", color=COLOR_BASE, markerfacecolor=COLOR_BASE, lw=0, markersize=5, label="Base; IC filled"),
        Line2D([0], [0], marker="^", color=COLOR_TGD, markerfacecolor=COLOR_TGD, lw=0, markersize=5, label="TGD; IC filled"),
        Line2D([0], [0], marker="o", color=COLOR_BASE, markerfacecolor="white", lw=0, markersize=5, label="Base; dPL open"),
        Line2D([0], [0], marker="^", color=COLOR_TGD, markerfacecolor="white", lw=0, markersize=5, label="TGD; dPL open"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.88)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
