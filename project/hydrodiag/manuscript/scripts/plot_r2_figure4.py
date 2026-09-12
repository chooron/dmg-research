#!/usr/bin/env python3
"""Final R2 Figure 4 (F4): parameter-specific compensation signatures + spatial organization.

F4 answers:
  1. Which shared parameters concentrate the Base–CN (compensatory) reorganization;
  2. How the key parameters' Base–CN differences evolve with snow regime;
  3. Whether the key parameter differences show geographic organization.

F4 is NOT an IC-vs-dPL ranking figure, NOT a parameter-truth figure, NOT a raw scatter
regression collage, and NOT a Base/CN/TGD2 comparison grid. IC and dPL are contrasting
parameter-constraint regimes (IC = main independent environmental anchor; dPL = shared
cross-basin mapping expression). TGD2 stays in the Supplement.

Final layout (compact, not wide; full-width overview + GIS column):
  Row 0 (full width): (a) Shared-parameter paired shifts (IC | dPL facets, all 15
                      shared parameters) - the overview panel.
  Row 1: left column  (b) um / (c) ki / (d) ci  - snow-conditioned parameter
                      OCCUPATION DISTRIBUTIONS (ridgeline-style: per S1-S5 regime the
                      Base and CN normalized-value densities are drawn as overlaid
                      filled ridges with median dots; IC and dPL as two strips).
         right column (e) Geographic organization of paired shifts (header + three
                      vertical mini-maps um/ki/ci, dPL regime, shared blue-orange
                      diverging scale centred at 0).

Colour semantics:
  * Blue  = CN structure (F1/F2); in (a), the ordinary shared parameters.
  * Orange= Base structure (F1/F2); in (a), the KEY compensation parameters (um, ki, ci).
    Colour encodes parameter class in (a), not IC vs dPL.
  * IC/dPL are separated by facet/strip position.
  * Maps use a blue (negative) - neutral (0) - orange (positive) diverging scale.

Statistics (all frozen-style, read-only) and GIS (read-only shapefiles, EPSG:5070):
  r2_snow_gradients_summary.csv, r2_parameter_values_canonical.csv,
  r2_paired_shifts_basin_level.csv, r2_snow_gradient_robustness.csv,
  r2_gd_diagnostic_summary.csv,
  data/camels_loc/conus_clipped/{camels_671_loc_conus_clipped,s_18mr25_conus}.shp.
Only deterministic descriptive aggregations are computed locally (regime medians/IQR,
bounded KDE of stored canonical z, map colour scales); no upstream analysis is
recomputed or modified.

Output: high-resolution PNG only (600 DPI), saved to manuscript/plots/figures/;
IC versions of the maps are also written to the Supplement (Fig_S4_IC_maps.png).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    setup_publication_style, apply_clean_spines,
)

PROJECT = Path(__file__).resolve().parents[2]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R2 = MANUSCRIPT / "results" / "R2"
PLOTS_FIG_DIR = MANUSCRIPT / "plots" / "figures"
SUPP_FIG_DIR = MANUSCRIPT / "supplement" / "figures"
PLOTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_NAME = "Figure4_R2"

# ---------------------------------------------------------------------------
# Canonical parameter definitions
# ---------------------------------------------------------------------------
PARAM_ORDER = [
    "xaj_k", "xaj_b", "xaj_im", "xaj_um", "xaj_lm", "xaj_dm", "xaj_c",
    "xaj_sm", "xaj_ex", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg",
    "xaj_a", "xaj_theta",
]
DISPLAY = {
    "xaj_k": "k", "xaj_b": "b", "xaj_im": "im", "xaj_um": "um",
    "xaj_lm": "lm", "xaj_dm": "dm", "xaj_c": "c", "xaj_sm": "sm",
    "xaj_ex": "ex", "xaj_ki": "ki", "xaj_kg": "kg", "xaj_ci": "ci",
    "xaj_cg": "cg", "xaj_a": "a", "xaj_theta": "θ",
}
KEY_PARAMS = ["xaj_um", "xaj_ki", "xaj_ci"]   # key compensation parameters
REGIMES = ["S1", "S2", "S3", "S4", "S5"]

# ---------------------------------------------------------------------------
# Visual grammar (F1/F2/F3 system)
# ---------------------------------------------------------------------------
COLOR_BASE = MODEL_COLORS["Base"]   # #EE7733  Base structure / key-compensation emphasis
COLOR_CN = MODEL_COLORS["CN"]       # #0077BB  CN structure / ordinary shared parameters
COLOR_REF = "#999999"               # reference lines
GEO_BASIN_SHP = PROJECT.parents[1] / "data" / "camels_loc" / "conus_clipped" / "camels_671_loc_conus_clipped.shp"
GEO_STATES_SHP = PROJECT.parents[1] / "data" / "camels_loc" / "conus_clipped" / "s_18mr25_conus.shp"
GEO_CRS = "EPSG:5070"

# Blue (negative) -> neutral (0) -> orange (positive) diverging map scale.
MAP_CMAP = LinearSegmentedColormap.from_list(
    "cn_base_div", ["#0077BB", "#F2F2F2", "#EE7733"], N=256)
MAP_VLIM = 1.0  # delta z bounded by [-1, +1]; shared across the three maps

RIDGE_PEAK = 0.42  # ridgeline height in regime-row units (Base up, CN down split)


# ---------------------------------------------------------------------------
# Data loading (frozen-style, read-only)
# ---------------------------------------------------------------------------
def load_data():
    df_g = pd.read_csv(RESULTS_R2 / "r2_snow_gradients_summary.csv")        # slopes + CI
    df_c = pd.read_csv(RESULTS_R2 / "r2_parameter_values_canonical.csv")    # canonical z
    df_c["basin_id"] = df_c["basin_id"].astype(str).str.zfill(8)
    df_p = pd.read_csv(RESULTS_R2 / "r2_paired_shifts_basin_level.csv")     # basin delta
    df_p["basin_id"] = df_p["basin_id"].astype(str).str.zfill(8)
    return df_g, df_c, df_p


def slope_ci_param(df_g, paradigm, parameter):
    row = df_g[(df_g["paradigm"] == paradigm) & (df_g["parameter"] == parameter)]
    assert len(row) == 1, f"missing slope {paradigm} {parameter}"
    r = row.iloc[0]
    return float(r["beta"]), float(r["ci95_low"]), float(r["ci95_high"])


def bounded_kde(vals, grid):
    """KDE on [0,1] with boundary reflection (robust to boundary concentration)."""
    v = np.asarray(vals, dtype=float)
    v = v[(v >= 0.0) & (v <= 1.0)]
    if len(v) < 2 or np.ptp(v) < 1e-12:
        d = np.zeros_like(grid)
        d[np.abs(grid - float(np.median(v))) < 1e-6] = 1.0
        return d
    refl = np.concatenate([-v, v, 2.0 - v])          # reflect at 0 and 1
    try:
        kde = gaussian_kde(refl)
    except Exception:
        d = np.zeros_like(grid)
        d[np.abs(grid - float(np.median(v))) < 1e-6] = 1.0
        return d
    d = kde(grid)
    d[grid < 0.0] = 0.0
    d[grid > 1.0] = 0.0
    if d.max() > 0:
        d = d / d.max()
    return d


# ---------------------------------------------------------------------------
# Panel (a): shared-parameter paired shifts (15 params, key ones in orange)
# ---------------------------------------------------------------------------
def panel_a_forest(ax, df_g, paradigm, show_legend=False):
    y_pos = np.arange(len(PARAM_ORDER))
    for i, p in enumerate(PARAM_ORDER):
        slope, lo, hi = slope_ci_param(df_g, paradigm, p)
        y = y_pos[i]
        key = p in KEY_PARAMS
        color = COLOR_BASE if key else COLOR_CN
        ax.errorbar(slope, y, xerr=[[slope - lo], [hi - slope]], fmt="s",
                    color=color, ecolor=color, elinewidth=1.4,
                    capsize=3.0, capthick=1.1, markersize=5.5,
                    markerfacecolor=color, zorder=3)
    ax.axvline(0, color=COLOR_REF, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([DISPLAY[p] for p in PARAM_ORDER], fontsize=7.5)
    for tick, p in zip(ax.get_yticklabels(), PARAM_ORDER):
        if p in KEY_PARAMS:
            tick.set_fontweight("bold")
            tick.set_color(COLOR_BASE)
    ax.set_xlabel("Snow gradient of paired shift, β", labelpad=3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.25)
    cilo = min(slope_ci_param(df_g, paradigm, p)[1] for p in PARAM_ORDER)
    cihi = max(slope_ci_param(df_g, paradigm, p)[2] for p in PARAM_ORDER)
    pad = max(0.05, (cihi - cilo) * 0.08)
    ax.set_xlim(cilo - pad, cihi + pad)
    if show_legend:
        handles = [
            Line2D([0], [0], marker="s", color=COLOR_BASE, markerfacecolor=COLOR_BASE,
                   markersize=5.5, linestyle="none",
                   label="Key compensation parameters"),
            Line2D([0], [0], marker="s", color=COLOR_CN, markerfacecolor=COLOR_CN,
                   markersize=5.5, linestyle="none", label="Other shared parameters"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.90,
                  edgecolor="none", fontsize=6.2)


# ---------------------------------------------------------------------------
# Panels (b)-(d): snow-conditioned occupation ridgelines
# ---------------------------------------------------------------------------
def _strip_ridgeline(ax, df_c, paradigm, parameter, xlabel=False):
    """One regime strip: per S1-S5 row, Base (up) and CN (down) occupation densities."""
    grid = np.linspace(0.0, 1.0, 120)
    y_pos = np.arange(len(REGIMES))  # S1 at bottom, S5 at top
    for i, reg in enumerate(REGIMES):
        y = y_pos[i]
        for structure, color, mk, sign in [("Base", COLOR_BASE, "o", +1.0),
                                           ("CN", COLOR_CN, "s", -1.0)]:
            vals = df_c[(df_c["paradigm"] == paradigm)
                        & (df_c["structure"] == structure)
                        & (df_c["parameter"] == parameter)
                        & (df_c["snow_regime"] == reg)]["z"].to_numpy()
            d = bounded_kde(vals, grid)
            curve = y + sign * RIDGE_PEAK * d
            ax.fill_between(grid, y, curve, color=color, alpha=0.42,
                            linewidth=0, zorder=2)
            ax.plot(grid, curve, color=color, linewidth=0.7, zorder=3)
            med = float(np.median(vals))
            ax.plot([med], [y], marker=mk, color=color, markersize=3.6,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(REGIMES, fontsize=7.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.62, 4.62)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="x", linestyle=":", alpha=0.25)
    if xlabel:
        ax.set_xlabel("Normalized parameter value, z", labelpad=3)


def panel_bcd_param(ax_ic, ax_dpl, df_c, parameter, title, show_legend=False):
    """One parameter panel: two strips (IC top, dPL bottom) of occupation ridgelines."""
    ax_ic.set_title(title, weight="bold", loc="left", pad=6)
    _strip_ridgeline(ax_ic, df_c, "IC", parameter)
    _strip_ridgeline(ax_dpl, df_c, "dPL", parameter, xlabel=True)
    ax_ic.set_xticklabels([])
    ax_ic.text(0.02, 0.97, "IC regime", transform=ax_ic.transAxes, ha="left",
               va="top", fontsize=7.5, fontweight="bold", color="#333333")
    ax_dpl.text(0.02, 0.97, "dPL regime", transform=ax_dpl.transAxes, ha="left",
                va="top", fontsize=7.5, fontweight="bold", color="#333333")
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color=COLOR_BASE, markerfacecolor=COLOR_BASE,
                   markersize=4.0, linestyle="none", label="Base"),
            Line2D([0], [0], marker="s", color=COLOR_CN, markerfacecolor=COLOR_CN,
                   markersize=4.0, linestyle="none", label="CN"),
        ]
        ax_ic.legend(handles=handles, loc="upper right", frameon=True,
                     framealpha=0.90, edgecolor="none", fontsize=6.2)


# ---------------------------------------------------------------------------
# Panel (e): GIS column - three vertical mini-maps (blue-orange diverging)
# ---------------------------------------------------------------------------
def _load_geo():
    import geopandas as gpd  # noqa: WPS433
    basins = gpd.read_file(GEO_BASIN_SHP).to_crs(GEO_CRS)
    states = gpd.read_file(GEO_STATES_SHP).to_crs(GEO_CRS)
    basins = basins.copy()
    basins["basin_id"] = basins["gage_id"].astype(str).str.zfill(8)
    states["geometry"] = states.geometry.simplify(3000, preserve_topology=True)
    rp = basins.geometry.representative_point()
    basins["x"] = rp.x
    basins["y"] = rp.y
    return basins, states


def _draw_map(ax, basins, states, delta, title):
    states.plot(ax=ax, facecolor="#F6F6F3", edgecolor="#D5D5D0", linewidth=0.25,
                zorder=1)
    ax.scatter(basins["x"], basins["y"], c=delta, cmap=MAP_CMAP, vmin=-MAP_VLIM,
               vmax=MAP_VLIM, s=11, alpha=0.85, edgecolors="none", zorder=3)
    minx, miny, maxx, maxy = states.total_bounds
    px, py = (maxx - minx) * 0.015, (maxy - miny) * 0.02
    ax.set_xlim(minx - px, maxx + px)
    ax.set_ylim(miny - py, maxy + py)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontweight="bold", pad=2, fontsize=8.5)


def _maps_data(df_p, paradigm):
    sub = df_p[df_p["paradigm"] == paradigm]
    out = {}
    for p in KEY_PARAMS:
        s = sub[sub["parameter"] == p][["basin_id", "delta_base_minus_cn"]]
        out[p] = s.set_index("basin_id")["delta_base_minus_cn"]
    return out


def panel_e_maps(axs, df_p, paradigm):
    basins, states = _load_geo()
    delta_map = _maps_data(df_p, paradigm)
    for ax, p in zip(axs, KEY_PARAMS):
        delta = basins["basin_id"].map(delta_map[p])
        _draw_map(ax, basins, states, delta, DISPLAY[p])
    return basins, states


def _maps_data(df_p, paradigm):
    sub = df_p[df_p["paradigm"] == paradigm]
    out = {}
    for p in KEY_PARAMS:
        s = sub[sub["parameter"] == p][["basin_id", "delta_base_minus_cn"]]
        out[p] = s.set_index("basin_id")["delta_base_minus_cn"]
    return out


def panel_e_maps(axs, df_p, paradigm):
    """Three vertical mini-maps (um, ki, ci) with the shared blue-orange scale."""
    basins, states = _load_geo()
    delta_map = _maps_data(df_p, paradigm)
    for ax, p in zip(axs, KEY_PARAMS):
        delta = basins["basin_id"].map(delta_map[p])
        _draw_map(ax, basins, states, delta, DISPLAY[p])
    return basins, states

# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_figure(df_g, df_c, df_p) -> None:
    fig = plt.figure(figsize=(10.2, 12.2))
    # Row 0: (a) full width; Row 1: left (b)(c)(d) + right GIS column
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.75], hspace=0.50,
                           left=0.07, right=0.98, top=0.96, bottom=0.05)

    # ---- (a) full-width overview ----
    gsa = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.30)
    ax_a1 = fig.add_subplot(gsa[0, 0]); apply_clean_spines(ax_a1)
    ax_a2 = fig.add_subplot(gsa[0, 1]); apply_clean_spines(ax_a2)
    ax_a1.set_title("(a) Shared-parameter paired shifts", weight="bold",
                    loc="left", pad=6)
    panel_a_forest(ax_a1, df_g, "IC", show_legend=True)
    panel_a_forest(ax_a2, df_g, "dPL")
    ax_a1.set_ylabel("Parameter", labelpad=3)
    ax_a2.set_yticklabels([])
    ax_a1.text(0.02, 0.97, "IC regime", transform=ax_a1.transAxes, ha="left",
               va="top", fontsize=8.0, fontweight="bold", color="#333333")
    ax_a2.text(0.02, 0.97, "dPL regime", transform=ax_a2.transAxes, ha="left",
               va="top", fontsize=8.0, fontweight="bold", color="#333333")

    # ---- Row 1: left (b)(c)(d) | right GIS ----
    gs_row = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],
                                              width_ratios=[1.45, 1.0], wspace=0.16)
    # left: three parameter panels
    gsL = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_row[0, 0],
                                           hspace=0.55)
    for row, (param, letter) in enumerate(zip(KEY_PARAMS, ["b", "c", "d"])):
        gsp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsL[row, 0],
                                               hspace=0.20)
        ax_t = fig.add_subplot(gsp[0, 0]); apply_clean_spines(ax_t)
        ax_b = fig.add_subplot(gsp[1, 0]); apply_clean_spines(ax_b)
        panel_bcd_param(ax_t, ax_b, df_c, param, f"({letter}) {DISPLAY[param]}",
                        show_legend=(row == 0))

    # right: GIS column (header + three maps) with the (e) title at the top
    gsRcol = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_row[0, 1],
                                              width_ratios=[0.93, 0.07], wspace=0.03)
    gsR = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gsRcol[0, 0],
                                           height_ratios=[0.4, 1.0, 1.0, 1.0],
                                           hspace=0.42)
    header_ax = fig.add_subplot(gsR[0, 0])
    header_ax.axis("off")
    header_ax.text(0.5, 0.95, "(e) Geographic organization\nof paired shifts",
                   transform=header_ax.transAxes, ha="center", va="top",
                   fontsize=8.5, fontweight="bold", linespacing=1.3)
    map_axes = [fig.add_subplot(gsR[i, 0]) for i in [1, 2, 3]]
    panel_e_maps(map_axes, df_p, "dPL")
    # shared vertical colour bar in its own sub-column of the GIS column
    mappable = ScalarMappable(norm=Normalize(-MAP_VLIM, MAP_VLIM), cmap=MAP_CMAP)
    cax = fig.add_subplot(gsRcol[0, 1])
    cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cbar.set_label("Δz (Base − CN)", fontsize=7.0, labelpad=2)
    cbar.ax.tick_params(labelsize=6.0, pad=1, length=2)

    # Output: PNG only, saved to manuscript/plots/figures/ (no PDF, no figures/ copy).
    out = PLOTS_FIG_DIR / f"{OUT_NAME}.png"
    plt.savefig(out, dpi=600)
    print("saved:", out)
    plt.close()


def build_ic_maps_supplement(df_p) -> None:
    """IC versions of the (e) maps -> Supplement (Fig_S4_IC_maps.png)."""
    fig = plt.figure(figsize=(8.8, 3.4))
    gs = gridspec.GridSpec(1, 3, wspace=0.10, left=0.02, right=0.92,
                           top=0.92, bottom=0.08)
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_e_maps(map_axes, df_p, "IC")
    mappable = ScalarMappable(norm=Normalize(-MAP_VLIM, MAP_VLIM), cmap=MAP_CMAP)
    cbar = fig.colorbar(mappable, ax=map_axes, orientation="vertical",
                        fraction=0.04, pad=0.01)
    cbar.set_label("Δz (Base − CN)", fontsize=7.5, labelpad=2)
    out = SUPP_FIG_DIR / "Fig_S4_IC_maps.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    print("saved:", out)
    plt.close(fig)


def main() -> None:
    setup_publication_style()
    df_g, df_c, df_p = load_data()
    # sanity: all 15 params present with finite slopes; canonical z in [0,1]; 531 basins
    for paradigm in ["IC", "dPL"]:
        for p in PARAM_ORDER:
            row = df_g[(df_g["paradigm"] == paradigm) & (df_g["parameter"] == p)]
            assert len(row) == 1 and np.isfinite(row["beta"].iloc[0])
        assert len(df_p[df_p["paradigm"] == paradigm]) == 531 * 15
    zc = df_c[df_c["parameter"].isin(PARAM_ORDER)]["z"]
    assert float(zc.min()) >= -1e-9 and float(zc.max()) <= 1.0 + 1e-9
    assert (df_p[(df_p["paradigm"] == "dPL") & (df_p["parameter"].isin(KEY_PARAMS))]
               .groupby("basin_id").ngroups) == 531
    build_figure(df_g, df_c, df_p)
    build_ic_maps_supplement(df_p)
    print("Figure 4 (full-width overview + occupation ridgelines + GIS column) generated.")


if __name__ == "__main__":
    main()
