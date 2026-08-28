#!/usr/bin/env python3
"""Final R2 Figure 4 (F4) v5: parameter-specific compensation signatures + spatial organization.

F4 answers:
  1. Which shared parameters concentrate the Base-CN (compensatory) reorganization;
  2. How the key parameters' paired shifts dZ = z_Base - z_CN evolve with snow influence;
  3. Whether these paired shifts show geographic organization;
  4. How the same structural absence is expressed differently under the IC and dPL
     parameter-constraint regimes.

F4 is NOT an IC-vs-dPL ranking figure, NOT a parameter-truth figure, NOT a raw scatter
regression collage, and NOT a Base/CN occupation figure. IC and dPL are contrasting
parameter-constraint regimes (IC = main independent environmental anchor; dPL = shared
cross-basin mapping expression). TGD2 stays in the Supplement.

Unified core quantity across the WHOLE figure (a, b-d, GIS):
    dZ_{p,i} = z_{Base,p,i} - z_{CN,p,i}    (normalized paired shift; z in [0,1])

Final layout (three balanced columns + bottom shared legend):
  Col 1 (panel a only):  (a) Snow gradients of paired shifts - split vertically
                         into (a1) IC regime (top) and (a2) dPL regime (bottom);
                         all 15 shared parameters, single structural contrast
                         Base-CN (deep-blue squares, the primary estimand).
                         um / ki / ci rows carry a very light band + bold labels
                         (highlight = expanded in panels b-d).
  Col 2 (ridges):        (b) um / (c) ki / (d) ci - ridgeline distributions of
                         basin-level dZ = z_Base - z_CN across snow regimes S1-S5,
                         stacked (Base-CN only).
  Col 3 (GIS):           (e) Geographic patterns of paired shifts - three enlarged
                         Base-CN maps (um | ki | ci, dPL regime), full column width.
  Bottom:                the (e) horizontal colour bar now hugs the GIS maps; the
                         IC/dPL ridge legend sits inside the upper-right of panel (b).

Ridgeline grammar (b-d), restrained HESS style:
  * One shared baseline per regime row; IC density mirrored above it (deep blue fill,
    solid outline), dPL density mirrored below (orange fill, thin dashed outline) -
    direct within-regime comparison of the two parameter-constraint regimes.
  * Fixed-bandwidth boundary-reflected KDE, peak-normalized; very light grey
    per-regime baselines; faint vertical grid; thin x = 0 reference; small median
    dot + thin short IQR line as auxiliary overlays; regime n in the y-axis labels;
    snow-influence cue on panel (b) only.

Colour semantics:
  * In (a): uniform deep blue = the single Base-CN contrast; um / ki / ci are
    highlighted only by a very light row band + bold labels (a manuscript-focus
    highlight, not a significance/threshold encoding).
  * In (b-d): IC = deep blue (above baseline), dPL = orange (below baseline).
  * Maps: blue (negative) - neutral (0) - orange (positive) diverging scale.

Statistics (all frozen-style, read-only) and GIS (read-only shapefiles, EPSG:5070):
  r2_snow_gradients_summary.csv (slopes + CI), r2_paired_shifts_basin_level.csv
  (canonical per-basin dZ = delta_base_minus_cn, with snow_regime),
  r2_parameter_values_canonical.csv (canonical z, used for the [0,1] sanity check),
  data/camels_loc/conus_clipped/{camels_671_loc_conus_clipped,s_18mr25_conus}.shp.
Only deterministic descriptive aggregations are computed locally (fixed-bandwidth
boundary-reflected KDE of stored canonical dZ, regime medians/IQR, map colour scales);
no upstream analysis is recomputed or modified.

Output: high-resolution PNG (600 DPI) only, saved to manuscript/plots/figures/;
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
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    setup_publication_style, apply_clean_spines,
)

PROJECT = Path(__file__).resolve().parents[3]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R2 = MANUSCRIPT / "results" / "R2"
PLOTS_FIG_DIR = MANUSCRIPT / "figures"
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
    "xaj_cg": "cg", "xaj_a": "a", "xaj_theta": "\u03b8",
}
KEY_PARAMS = ["xaj_um", "xaj_ki", "xaj_ci"]   # key compensation parameters
REGIMES = ["S1", "S2", "S3", "S4", "S5"]
REGIME_N = [165, 156, 121, 34, 55]

# ---------------------------------------------------------------------------
# Visual grammar (F1/F2/F3 system)
# ---------------------------------------------------------------------------
COLOR_BASE = MODEL_COLORS["Base"]   # #EE7733  warm orange (dPL ridges in (b-d))
COLOR_CN = MODEL_COLORS["CN"]       # #0077BB  deep blue (Base-CN contrast in (a); IC ridges)
COLOR_REF = "#999999"               # reference lines
RIDGE_EDGE = "#0B4C79"              # darker blue ridge outline (IC)
KEY_ROW_SHADE = "#FEF8F1"           # very light warm shading for highlighted-parameter rows in (a)

GEO_BASIN_SHP = PROJECT.parents[1] / "data" / "camels_loc" / "conus_clipped" / "camels_671_loc_conus_clipped.shp"
GEO_STATES_SHP = PROJECT.parents[1] / "data" / "camels_loc" / "conus_clipped" / "s_18mr25_conus.shp"
GEO_CRS = "EPSG:5070"

# Blue (negative) -> neutral (0) -> orange (positive) diverging map scale.
MAP_CMAP = LinearSegmentedColormap.from_list(
    "cn_base_div", ["#0077BB", "#F2F2F2", "#EE7733"], N=256)
MAP_VLIM = 1.0  # dZ bounded by [-1, +1]; shared across the three maps

# Ridgeline geometry for (b)-(d): IC and dPL share one baseline per regime row,
# mirrored as IC density above / dPL density below (restrained HESS style)
DZ_GRID = np.linspace(-1.0, 1.0, 240)   # fixed symmetric axis [-1, 1]
DZ_BW = 0.075                           # fixed absolute KDE bandwidth on the dZ scale (mild smoothing to avoid small-n fragmentation)
RIDGE_HEIGHT = 0.36                     # peak ridge height from the shared baseline
IC_FILL_ALPHA = 0.42                    # deeper blue fill for IC
DPL_FILL_ALPHA = 0.20                   # lighter orange fill for dPL
DPL_LINESTYLE = (0, (4.0, 2.0))         # thin dashed outline for dPL
BASELINE_COLOR = "#DDDDDD"              # very light grey per-regime baseline
ZERO_LINE_COLOR = "#888888"             # x = 0 reference (slightly darker, thin)
GRID_ALPHA = 0.12                       # very faint vertical reference grid
DZ_XTICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]

# ---------------------------------------------------------------------------
# Data loading (frozen-style, read-only)
# ---------------------------------------------------------------------------
def load_data():
    df_g = pd.read_csv(RESULTS_R2 / "r2_snow_gradients_summary.csv")        # slopes + CI
    df_c = pd.read_csv(RESULTS_R2 / "r2_parameter_values_canonical.csv")    # canonical z
    df_c["basin_id"] = df_c["basin_id"].astype(str).str.zfill(8)
    df_p = pd.read_csv(RESULTS_R2 / "r2_paired_shifts_basin_level.csv")     # basin dZ
    df_p["basin_id"] = df_p["basin_id"].astype(str).str.zfill(8)
    return df_g, df_c, df_p


def slope_ci_param(df_g, paradigm, parameter):
    row = df_g[(df_g["paradigm"] == paradigm) & (df_g["parameter"] == parameter)]
    assert len(row) == 1, f"missing slope {paradigm} {parameter}"
    r = row.iloc[0]
    return float(r["beta"]), float(r["ci95_low"]), float(r["ci95_high"])


def ridge_density(vals, grid, h):
    """Fixed-bandwidth Gaussian KDE on [-1, 1] with boundary reflection.

    All regimes share the same absolute bandwidth ``h`` (DZ_BW) so ridge shapes are
    directly comparable; each ridge is peak-normalized to 1 so sample size does not
    control peak height.
    """
    v = np.asarray(vals, dtype=float)
    v = v[(v >= -1.0) & (v <= 1.0)]
    if len(v) < 2:
        d = np.zeros_like(grid)
        d[np.argmin(np.abs(grid - float(np.median(v))))] = 1.0
        return d
    refl = np.concatenate([-2.0 - v, v, 2.0 - v])        # reflect at -1 and +1
    d = np.exp(-0.5 * ((grid[:, None] - refl[None, :]) / h) ** 2).sum(axis=1)
    d[grid < -1.0] = 0.0
    d[grid > 1.0] = 0.0
    if d.max() > 0:
        d = d / d.max()
    return d


# ---------------------------------------------------------------------------
# Panel (a): snow gradients of paired parameter shifts (Base-CN only)
#   Single structural contrast per parameter row (deep-blue squares). The
#   um / ki / ci rows carry a very light band + bold labels (highlight =
#   expanded in panels b-d); colours are uniform because there is one contrast.
# ---------------------------------------------------------------------------
def panel_a_forest(ax, df_g, paradigm, shared_lim=None):
    """One (a) subplot: snow gradients of paired shifts (Base-CN) for 15 parameters.

    The primary estimand Base-CN is drawn as a deep-blue square per parameter with
    the established 95 % bootstrap CI; um/ki/ci rows carry a very light band + bold
    labels as a focus highlight (expanded in panels b-d) - no significance encoding.
    ``shared_lim`` (lo, hi, ticks) unifies the x-axis across the IC and dPL subplots.
    """
    y_pos = np.arange(len(PARAM_ORDER))
    # very light background band behind the highlighted rows
    for i, p in enumerate(PARAM_ORDER):
        if p in KEY_PARAMS:
            ax.axhspan(i - 0.5, i + 0.5, color=KEY_ROW_SHADE, linewidth=0, zorder=0)
    for i, p in enumerate(PARAM_ORDER):
        slope, lo, hi = slope_ci_param(df_g, paradigm, p)
        y = y_pos[i]
        # um / ki / ci rows carry the orange highlight (marker + CI) matching
        # their bold orange y-tick labels; all other rows stay deep blue.
        col = COLOR_BASE if p in KEY_PARAMS else COLOR_CN
        ax.errorbar(slope, y, xerr=[[slope - lo], [hi - slope]], fmt="s",
                    color=col, ecolor=col, elinewidth=1.1,
                    capsize=2.4, capthick=0.9, markersize=4.8,
                    markerfacecolor=col, markeredgecolor="none",
                    linestyle="none", zorder=3)
    ax.axvline(0, color=COLOR_REF, linestyle="--", linewidth=0.85, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([DISPLAY[p] for p in PARAM_ORDER], fontsize=11.2)
    for tick, p in zip(ax.get_yticklabels(), PARAM_ORDER):
        if p in KEY_PARAMS:
            tick.set_fontweight("bold")
            tick.set_color(COLOR_BASE)   # orange highlight for um/ki/ci
    ax.set_ylim(-0.75, len(PARAM_ORDER) - 0.25)
    ax.set_xlabel("Snow gradient of paired shift, \u03b2", labelpad=3,
                  fontsize=11.5)
    ax.tick_params(axis="x", labelsize=11.2)
    ax.grid(True, axis="x", linestyle=":", alpha=0.12)
    if shared_lim is not None:
        lo_r, hi_r, ticks = shared_lim
        ax.set_xlim(lo_r, hi_r)
        ax.set_xticks(ticks)
    else:
        cilo = min(slope_ci_param(df_g, paradigm, p)[1] for p in PARAM_ORDER)
        cihi = max(slope_ci_param(df_g, paradigm, p)[2] for p in PARAM_ORDER)
        pad = max(0.05, (cihi - cilo) * 0.08)
        # round outward to a 0.5 grid and add a label buffer so the outermost tick
        # labels are never clipped at the axes / figure edge
        lo_r = np.floor((cilo - pad) / 0.5) * 0.5
        hi_r = np.ceil((cihi + pad) / 0.5) * 0.5
        ax.set_xlim(lo_r, hi_r + 0.10)
        ax.set_xticks(np.arange(lo_r, hi_r + 0.5 + 1e-9, 0.5))

# ---------------------------------------------------------------------------
# Panels (b)-(d): ridgeline distributions of paired shifts dZ = z_Base - z_CN
#   One regime row holds the IC ridge (up, solid, deeper) and the dPL ridge
#   (down, dashed, lighter) so both regimes are compared within each snow regime.
# ---------------------------------------------------------------------------
def _dz_ridge_pair(ax, df_p, parameter, xlabel=False):
    """Per S1-S5 row: IC density mirrored above the shared baseline, dPL below.

    Restrained HESS-style grammar: very light grey per-regime baseline, faint
    vertical grid, thin x=0 reference; the density is the first visual layer and
    the summary overlays (small median dot, thin short IQR line) stay auxiliary.
    """
    y_pos = np.arange(len(REGIMES))  # S1 at bottom, S5 at top
    for i, reg in enumerate(REGIMES):
        y = y_pos[i]
        ax.axhline(y, color=BASELINE_COLOR, linewidth=0.6, zorder=1)
        for paradigm, sign, alpha, fill, edge, ls in [
            ("IC", +1.0, IC_FILL_ALPHA, COLOR_CN, RIDGE_EDGE, "-"),
            ("dPL", -1.0, DPL_FILL_ALPHA, COLOR_BASE, COLOR_BASE, DPL_LINESTYLE),
        ]:
            vals = df_p[(df_p["paradigm"] == paradigm)
                        & (df_p["parameter"] == parameter)
                        & (df_p["snow_regime"] == reg)]["delta_base_minus_cn"].to_numpy()
            d = ridge_density(vals, DZ_GRID, DZ_BW)
            curve = y + sign * RIDGE_HEIGHT * d
            ax.fill_between(DZ_GRID, y, curve, color=fill, alpha=alpha,
                            linewidth=0, zorder=2)
            ax.plot(DZ_GRID, curve, color=edge, linestyle=ls, linewidth=0.7, zorder=3)
            med = float(np.median(vals))
            # small median dot on the ridge, in the ridge's own colour (auxiliary)
            ax.plot([med], [y + sign * RIDGE_HEIGHT * d[np.argmin(np.abs(DZ_GRID - med))]],
                    marker="o", color=edge, markersize=2.4,
                    markerfacecolor=edge, markeredgecolor="none", zorder=5)
            # thin short IQR line at the baseline, in the ridge's own colour
            q1, q3 = np.percentile(vals, [25, 75])
            ax.plot([q1, q3], [y + sign * 0.015, y + sign * 0.015],
                    color=edge, linewidth=0.9, solid_capstyle="butt", zorder=5)
    # x = 0 reference: slightly darker than the grid, still thin and restrained
    ax.axvline(0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(REGIMES, fontsize=11.0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.5, float(len(REGIMES)) + 0.5)
    ax.set_xticks(DZ_XTICKS)
    ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=11.0)
    ax.grid(True, axis="x", linestyle=":", alpha=GRID_ALPHA)
    if xlabel:
        ax.set_xlabel(r"$\Delta z = z_{\mathrm{Base}} - z_{\mathrm{CN}}$", labelpad=3,
                      fontsize=11.5)

def panel_ridge(ax, df_p, parameter, letter, xlabel=False):
    """One middle-column parameter panel: (letter) name + per-regime IC/dPL ridges.

    Panel label is drawn in the axes' upper-left corner (no set_title).
    """
    _dz_ridge_pair(ax, df_p, parameter, xlabel=xlabel)
    ax.text(0.02, 0.975, f"({letter}) {DISPLAY[parameter]}",
            transform=ax.transAxes, ha="left", va="top", fontsize=11.8,
            fontweight="bold", color="#333333")


# ---------------------------------------------------------------------------
# GIS row: horizontal row of three maps of the same paired shift dZ
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
               vmax=MAP_VLIM, s=15, alpha=0.90, edgecolors="none", zorder=3)
    minx, miny, maxx, maxy = states.total_bounds
    px, py = (maxx - minx) * 0.015, (maxy - miny) * 0.02
    ax.set_xlim(minx - px, maxx + px)
    ax.set_ylim(miny - py, maxy + py)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontweight="bold", pad=2, fontsize=12.3)


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


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_figure(df_g, df_c, df_p) -> None:
    fig = plt.figure(figsize=(13.0, 10.6))
    TOP, BOT = 0.955, 0.096
    # legend strip sits ~1 character below the panels (BOT small).
    c1 = (0.050, 0.314)   # panel (a)
    c2 = (0.372, 0.647)   # ridgelines (b)(c)(d)
    c3 = (0.676, 0.982)   # GIS (e)
    gsL = gridspec.GridSpec(2, 1, left=c1[0], right=c1[1], top=TOP, bottom=BOT, hspace=0.12)
    gsM = gridspec.GridSpec(3, 1, left=c2[0], right=c2[1], top=TOP, bottom=BOT, hspace=0.22)
    # GIS block sits slightly higher than the (a) and (b-d) columns (top += 0.012)
    gsR = gridspec.GridSpec(3, 1, left=c3[0], right=c3[1], top=TOP + 0.012, bottom=BOT,
                            height_ratios=[1.0, 1.0, 1.0], hspace=0.015)

    # ---- Column 1: (a) split into (a1) IC regime / (a2) dPL regime (vertical) ----
    ax_a1 = fig.add_subplot(gsL[0, 0]); apply_clean_spines(ax_a1)
    ax_a2 = fig.add_subplot(gsL[1, 0]); apply_clean_spines(ax_a2)
    ax_a1.text(0.985, 0.975, "(a)", transform=ax_a1.transAxes, ha="right",
               va="top", fontsize=12.3, fontweight="bold")
    # shared x-axis across the two regimes (Base-CN only)
    cilo = min(slope_ci_param(df_g, p_, par)[1] for p_ in ["IC", "dPL"] for par in PARAM_ORDER)
    cihi = max(slope_ci_param(df_g, p_, par)[2] for p_ in ["IC", "dPL"] for par in PARAM_ORDER)
    pad = max(0.05, (cihi - cilo) * 0.08)
    lo_r = np.floor((cilo - pad) / 0.5) * 0.5
    hi_r = np.ceil((cihi + pad) / 0.5) * 0.5
    shared_lim = (lo_r, hi_r + 0.10, np.arange(lo_r, hi_r + 0.5 + 1e-9, 0.5))
    panel_a_forest(ax_a1, df_g, "IC", shared_lim=shared_lim)
    panel_a_forest(ax_a2, df_g, "dPL", shared_lim=shared_lim)
    # a1/a2 share one x-axis: ticks+label only on the bottom panel (a2)
    ax_a1.set_xlabel(None)
    ax_a1.tick_params(labelbottom=False)
    ax_a1.set_ylabel("Parameter", labelpad=3)
    ax_a2.set_ylabel("Parameter", labelpad=3)
    ax_a1.text(0.02, 0.97, "IC regime", transform=ax_a1.transAxes, ha="left",
               va="top", fontsize=11.2, fontweight="bold", color="#333333")
    ax_a2.text(0.02, 0.97, "dPL regime", transform=ax_a2.transAxes, ha="left",
               va="top", fontsize=11.2, fontweight="bold", color="#333333")

    # ---- Column 2: (b) um / (c) ki / (d) ci ridgelines (vertical) ----
    ridge_axes = []
    for row, (param, letter) in enumerate(zip(KEY_PARAMS, ["b", "c", "d"])):
        ax = fig.add_subplot(gsM[row, 0]); apply_clean_spines(ax)
        ridge_axes.append(ax)
        panel_ridge(ax, df_p, param, letter, xlabel=(row == 2))

    # ---- Column 3: (e) GIS column - three enlarged maps (um | ki | ci), full width
    map_axes = [fig.add_subplot(gsR[i, 0]) for i in [0, 1, 2]]
    panel_e_maps(map_axes, df_p, "dPL")
    map_axes[0].text(0.005, 0.985, "(e)", transform=map_axes[0].transAxes,
                     ha="left", va="top", fontsize=11.8, fontweight="bold")

    # ---- IC/dPL ridge legend: in-figure, inside the upper-right of panel (b) ----
    regime_handles = [
        Patch(facecolor=COLOR_CN, alpha=IC_FILL_ALPHA, edgecolor=RIDGE_EDGE,
              linewidth=0.8, label="IC"),
        Patch(facecolor=COLOR_BASE, alpha=DPL_FILL_ALPHA, edgecolor=COLOR_BASE,
              linewidth=0.7, linestyle=DPL_LINESTYLE, label="dPL"),
    ]
    leg1 = ridge_axes[0].legend(handles=regime_handles, loc="upper right",
                                ncol=2, frameon=True,
                                facecolor="white", framealpha=0.85, edgecolor="none",
                                fontsize=10.4, columnspacing=1.2, handlelength=1.6)
    # horizontal colour bar for (e): hug the bottom edge of the GIS maps
    fig.canvas.draw()
    gis_bottom = map_axes[2].get_position()
    mappable = ScalarMappable(norm=Normalize(-MAP_VLIM, MAP_VLIM), cmap=MAP_CMAP)
    cax = fig.add_axes([gis_bottom.x0, gis_bottom.y0 - 0.014, gis_bottom.width, 0.014])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label("\u0394z (Base \u2212 CN)", fontsize=10.2, labelpad=1)
    # Output: high-resolution PNG only, saved to manuscript/plots/figures/
    out_png = PLOTS_FIG_DIR / f"{OUT_NAME}.png"
    fig.savefig(out_png, dpi=600)
    print("saved:", out_png)
    plt.close(fig)


def build_ic_maps_supplement(df_p) -> None:
    """IC versions of the GIS maps -> Supplement (Fig_S4_IC_maps.png)."""
    fig = plt.figure(figsize=(8.8, 3.4))
    gs = gridspec.GridSpec(1, 3, wspace=0.10, left=0.02, right=0.92,
                           top=0.92, bottom=0.08)
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_e_maps(map_axes, df_p, "IC")
    mappable = ScalarMappable(norm=Normalize(-MAP_VLIM, MAP_VLIM), cmap=MAP_CMAP)
    cbar = fig.colorbar(mappable, ax=map_axes, orientation="vertical",
                        fraction=0.04, pad=0.01)
    cbar.set_label("\u0394z (Base \u2212 CN)", fontsize=11.0, labelpad=2)
    out = SUPP_FIG_DIR / "Fig_S4_IC_maps.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    print("saved:", out)
    plt.close(fig)


def main() -> None:
    raise SystemExit(
        "Legacy Figure 4 renderer is disabled; use "
        "plot_r2_figure4_canonical.py for the manuscript-facing output."
    )
if __name__ == "__main__":
    main()
