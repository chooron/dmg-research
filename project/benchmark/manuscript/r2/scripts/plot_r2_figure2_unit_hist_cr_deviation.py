"""Render the finished R2 Figure 2 composite as one publication PNG.

Panel (a) is the displacement hero, panel (b) is the model-level paired
excess histogram, and panel (c) is the reference-sensitivity CR distribution.
No panel-wise image, PDF, or Figure 3 output is produced.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

PAIRED_CACHE = CACHE / "fig2a_paired_displacement_plane.parquet"
EXCESS_CACHE = CACHE / "fig2b_cross_minus_self.csv"
CR_CACHE = CACHE / "fig3d_contraction_reference.csv"
RECHECK = TABLES / "F2_MODEL_EXCESS_RECHECK.csv"
ZERO_AUDIT = TABLES / "F2_ICSELF_ZERO_AUDIT.csv"
COLLIE_SENSITIVITY = TABLES / "F2_COLLIE1_SENSITIVITY.csv"
AUDIT_REPORT = TABLES / "F2_ICSELF_AUDIT_REPORT.md"
CR_PROVENANCE_REPORT = TABLES / "F2_CR_PROVENANCE_AUDIT_REPORT.md"
CR_MODEL_LIST = TABLES / "F2_CR_CURRENT23_MODEL_LIST.csv"

PURPLE = "#6a3d9a"
PALE_PURPLE = "#d9c6e8"
GREEN = "#2f8f5b"
PALE_GREEN = "#c4e3d0"
GREY = "#7c8793"
PALE_GREY = "#d9dee3"
DARK = "#2d2933"
NEUTRAL_LIGHT = "#c8cdd2"
NEUTRAL_MID = "#9aa3ad"
NEUTRAL_DARK = "#596673"
MEDIAN_PURPLE = "#5a2d83"
MEDIAN_GREEN = "#267349"
RIDGE_FILLS = ["#d8c2e8", "#b8d9c3", "#94c9a5"]
RIDGE_TOP_COLORS = ["#6a3d9a", "#2e8050", "#23643d"]
OUTPUT = FIGURES / "Figure2_R2_parameter_space_response_final.png"
BIN_WIDTH = 0.05


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.2,
            "axes.linewidth": 0.75,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_png(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def label_panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.18, 1.02, f"({label}) {title}", transform=ax.transAxes, fontsize=10.3, fontweight="bold", va="bottom", ha="left", color="black")


def finish(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    paired = pd.read_parquet(PAIRED_CACHE)
    excess = pd.read_csv(EXCESS_CACHE)
    recheck = pd.read_csv(RECHECK)
    zero = pd.read_csv(ZERO_AUDIT)
    cr = pd.read_csv(CR_CACHE)
    if "IC-SELF REFERENCE = VALID" not in AUDIT_REPORT.read_text():
        raise RuntimeError("F2 REDRAW = BLOCKED: Phase A IC-self audit is not valid")
    if "CR DATA VERDICT = STRICT23 ONLY" not in CR_PROVENANCE_REPORT.read_text():
        raise RuntimeError("F2 REDRAW = BLOCKED: CR provenance audit is not resolved")
    sensitivity = pd.read_csv(COLLIE_SENSITIVITY)
    sensitivity_row = sensitivity.loc[sensitivity.analysis == "exclude_collie1"]
    if len(sensitivity_row) != 1 or int(sensitivity_row.positive_models.iloc[0]) != 35:
        raise RuntimeError("F2 REDRAW = BLOCKED: collie1 sensitivity did not reproduce 35/35")

    if paired[["model_id", "basin_id"]].duplicated().any() or len(paired) != 36 * 531:
        raise RuntimeError("F2 REDRAW = BLOCKED: paired cache is not the audited 36 x 531 object")
    groups = paired.groupby("model_id", sort=False)
    if groups.basin_id.nunique().ne(531).any() or len(groups) != 36:
        raise RuntimeError("F2 REDRAW = BLOCKED: paired cache has nonuniform model coverage")
    if len(recheck) != 36 or recheck.model_id.duplicated().any():
        raise RuntimeError("F2 REDRAW = BLOCKED: excess recheck does not contain 36 unique models")
    if set(recheck.model_id) != set(paired.model_id.unique()):
        raise RuntimeError("F2 REDRAW = BLOCKED: model IDs differ between paired cache and excess recheck")
    if len(excess.loc[excess.scope == "model"]) != 36:
        raise RuntimeError("F2 REDRAW = BLOCKED: frozen excess cache does not contain 36 model rows")

    # Hard QC: every plotted model/quantity uses one matched subset for all five quantiles.
    for model, group in groups:
        for column, quantity in [("D_theta_cross", "D_cross"), ("D_theta_IC_self", "D_self")]:
            values = group[column].to_numpy(float)
            if len(values) != 531 or not np.isfinite(values).all():
                raise RuntimeError(f"F2 REDRAW = BLOCKED: nonfinite or incomplete {quantity} values for {model}")
            q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
            if not (q05 <= q25 and q25 <= median and median <= q75 and q75 <= q95):
                raise RuntimeError(
                    f"F2 REDRAW = BLOCKED: quantile order failed for {model} {quantity}: "
                    f"Q05={q05}, Q25={q25}, median={median}, Q75={q75}, Q95={q95}"
                )

    summary = excess.loc[excess.scope == "all36_summary"].iloc[0]
    frozen_point = recheck.set_index("model_id").frozen_median_of_basin_differences
    cache_point = excess.loc[excess.scope == "model"].set_index("model_id").cross_minus_self_median
    if not np.allclose(frozen_point.sort_index(), cache_point.sort_index(), rtol=0, atol=1e-9):
        raise RuntimeError("F2 REDRAW = BLOCKED: paired excess points do not match the frozen cache")
    if not np.isclose(float(frozen_point.median()), float(summary.cross_minus_self_median), rtol=0, atol=1e-9):
        raise RuntimeError("F2 REDRAW = BLOCKED: frozen paired median excess is not reproduced")
    if int((recheck.frozen_median_of_basin_differences > 0).sum()) != 36:
        raise RuntimeError("F2 REDRAW = BLOCKED: frozen 36/36 positive statement is not reproduced")

    # One frozen order is used for the hero rows; panel (b) is an order-free strip.
    cross_medians = groups.D_theta_cross.median().rename("median_D_cross")
    order = cross_medians.sort_values().index.tolist()
    delta_by_model = recheck.set_index("model_id").frozen_median_of_basin_differences
    order_table = pd.DataFrame({
        "model_id": order,
        "plot_order": np.arange(1, len(order) + 1),
        "median_D_cross": cross_medians.loc[order].to_numpy(float),
        "Delta_m": delta_by_model.loc[order].to_numpy(float),
    }).merge(zero[["model_id", "reference_status"]], on="model_id", validate="one_to_one")
    order_table.to_csv(TABLES / "F2_MODEL_ORDER.csv", index=False, float_format="%.10f")

    cr_columns = ["CR_canonical", "CR_consensus", "CR_ICself"]
    cr_model_list = pd.read_csv(CR_MODEL_LIST)
    if len(cr) != 23 or len(cr_model_list) != 23 or cr["model_id"].duplicated().any() or cr_model_list["model_id"].duplicated().any():
        raise RuntimeError("F2 REDRAW = BLOCKED: strict23 panel (c) requires 23 unique model-level rows")
    if set(cr.model_id.astype(str)) != set(cr_model_list.model_id.astype(str)) or not cr_model_list[["present_canonical", "present_consensus", "present_icself"]].all().all():
        raise RuntimeError("F2 REDRAW = BLOCKED: panel (c) strict23 model set is not identical across all references")
    if cr[cr_columns].isna().any().any():
        raise RuntimeError("F2 REDRAW = BLOCKED: panel (c) CR values contain missing entries")
    cr_values = cr[cr_columns].to_numpy(float)
    if not np.isfinite(cr_values).all() or (cr_values <= 0).any():
        raise RuntimeError("F2 REDRAW = BLOCKED: CR contains nonpositive or nonfinite values")
    full_range = {"min": float(cr_values.min()), "max": float(cr_values.max())}
    final_values = pd.DataFrame(
        [
            {"quantity": "across_model_median_D_cross", "value": float(cross_medians.median()), "role": "panel_a", "source": str(PAIRED_CACHE.relative_to(ROOT))},
            {"quantity": "paired_model_equal_median_Delta", "value": float(summary.cross_minus_self_median), "role": "panel_b_primary", "source": str(EXCESS_CACHE.relative_to(ROOT))},
            {"quantity": "paired_model_equal_CI_low", "value": float(summary.ci_low), "role": "panel_b_primary", "source": str(EXCESS_CACHE.relative_to(ROOT))},
            {"quantity": "paired_model_equal_CI_high", "value": float(summary.ci_high), "role": "panel_b_primary", "source": str(EXCESS_CACHE.relative_to(ROOT))},
            {"quantity": "positive_models", "value": int((recheck.frozen_median_of_basin_differences > 0).sum()), "role": "panel_b_primary", "source": str(RECHECK.relative_to(ROOT))},
            {"quantity": "difference_of_marginal_medians", "value": float(recheck.Delta_m.median()), "role": "descriptive_not_primary", "source": str(RECHECK.relative_to(ROOT))},
            {"quantity": "matched_cells_above_1to1_fraction", "value": float((paired.D_theta_cross > paired.D_theta_IC_self).mean()), "role": "caption_only", "source": str(PAIRED_CACHE.relative_to(ROOT))},
            {"quantity": "CR_canonical", "value": float(cr.CR_canonical.median()), "role": "panel_c", "source": str(CR_CACHE.relative_to(ROOT))},
            {"quantity": "CR_consensus", "value": float(cr.CR_consensus.median()), "role": "panel_c", "source": str(CR_CACHE.relative_to(ROOT))},
            {"quantity": "CR_ICself", "value": float(cr.CR_ICself.median()), "role": "panel_c", "source": str(CR_CACHE.relative_to(ROOT))},
            {"quantity": "CR_data_min", "value": full_range["min"], "role": "range_check", "source": str(CR_CACHE.relative_to(ROOT))},
            {"quantity": "CR_data_max", "value": full_range["max"], "role": "range_check", "source": str(CR_CACHE.relative_to(ROOT))},
        ]
    )
    final_values.to_csv(TABLES / "F2_V4_FINAL_VALUES.csv", index=False, float_format="%.10f")
    return paired, excess, recheck, cr, {"cr_min": full_range["min"], "cr_max": full_range["max"], "above_fraction": float((paired.D_theta_cross > paired.D_theta_IC_self).mean())}


def write_build_note(paired: pd.DataFrame, excess: pd.DataFrame, recheck: pd.DataFrame, cr: pd.DataFrame, order: list[str], checks: dict[str, float]) -> None:
    summary = excess.loc[excess.scope == "all36_summary"].iloc[0]
    marginal = float(recheck.Delta_m.median())
    sensitivity = pd.read_csv(COLLIE_SENSITIVITY).set_index("analysis")
    status_counts = pd.read_csv(ZERO_AUDIT).reference_status.value_counts().to_dict()
    order_text = " → ".join(order)
    n_above = int((paired.D_theta_cross > paired.D_theta_IC_self).sum())
    note = f"""# R2 Figure 2 final build note

## Provenance and frozen estimands

Phase A is resolved as **IC-SELF REFERENCE = VALID**. The canonical IC-self reference is compared with eligible archived IC multi-start vectors under the predeclared `within_0.01` training-fitness rule; the raw archive contains ten IC restart slots per basin for 36 models and 531 basins/model. The CR provenance audit verdict is **CR DATA VERDICT = STRICT23 ONLY**: panel (c) uses the identical 23-model primary set across canonical, consensus, and IC-self. `collie1` is a valid one-parameter model (`P=1`); its invariant localization behavior is structural, not a corruption or restart/cache failure.

The primary paired quantity is:

```text
delta_m,b = D_cross(m,b) - D_self(m,b)
Delta_m    = median_b(delta_m,b)
```

The model-equal primary median excess is `{float(summary.cross_minus_self_median):.10f}` with 95% bootstrap CI `[{float(summary.ci_low):.10f}, {float(summary.ci_high):.10f}]`; 36/36 model-level estimates are positive. The distinct marginal contrast is `{marginal:.10f}` and is not interchangeable with the paired estimand. Excluding `collie1` leaves `{int(sensitivity.loc['exclude_collie1', 'positive_models'])}/35` positive models with median paired excess `{float(sensitivity.loc['exclude_collie1', 'median_paired_excess']):.10f}`.

The hard quantile gate `Q05 <= Q25 <= median <= Q75 <= Q95` passed for both plotted displacement quantities in all 36 models, using the same 531-basin matched subset per model. The unit-histogram conclusion was checked privately at bin widths 0.04, 0.05, and 0.06; width 0.05 is used in the figure.

## Final visual grammar

- **(a) HERO:** the existing interval landscape, with Q05–Q95 whiskers, Q25–Q75 bands, medians, slight vertical dodge, the fixed ascending-`median D_cross` order, and a `-0.05–0.8` x-axis. Purple and green encode the contrasting `D_self` and `D_cross` quantities; the in-panel legend uses only the `D_self` and `D_cross` symbols.
- **(b) UNIT HISTOGRAM:** fixed-width bins of `0.05` in `Delta_m`; each model contributes exactly one stacked square, with a visible negative `Delta_m < 0` region that is empty. The main histogram retains only the black dashed `Delta_m = 0` reference. Green squares encode the positive paired excess; collie1 is plotted with the same symbol as the other valid model estimates. Its one-parameter structural status is documented separately.
- **(c) RIDGELINE / BOUNDARY:** the identical 23-model primary set contributes three distributions of model-level CR values, one for each alternative IC reference. KDEs are estimated in log-CR space with the same bandwidth rule and common ridge-height normalization; short baseline rug ticks show the individual model-level values. A very pale neutral background marks only `CR < 1` and is labeled `apparent contraction`; a fine bracket groups consensus and IC-self as `alternative references`. Purple marks the canonical reference, while green shades mark consensus and IC-self; the common black dashed line marks `CR = 1`, and right-side labels report `0.614`, `0.979`, and `1.004`.

The exact panel (a) model order is:

```text
{order_text}
```

Required caption wording: panel (a) displays the marginal basin-level distributions of `D_cross` and `D_self`, whereas panel (b) displays `Delta_m = median_b(D_cross - D_self)`. Therefore the paired median excess `0.2188` and marginal-median difference `0.2531` are not expected to be identical. The scope is 36 models, 531 basins/model, and 19,116 exact matched model–basin cells. The pooled cell-level check is `{100 * checks['above_fraction']:.1f}%` ({n_above}/{len(paired)}) with `D_cross > D_self`; this is not the model-equal primary inference. The references in panel (c) are alternatives, not a temporal sequence.
Panel (c) caption wording: CR is shown relative to the common `CR = 1` boundary using the identical strict primary subset of `23/36` models for all three reference constructions; the common subset requires primary IC restart coverage `>= 0.90`. The canonical median (`0.614`) lies clearly below the boundary, whereas consensus (`0.979`) and IC-self (`1.004`) are near unity; `CR ≈ 1` means relative parameter spread comparable to the reference, not expansion. KDEs were estimated in log-CR space using the same bandwidth rule, and short rug ticks show individual model-level CR values.
The panel (c) ridgelines are across the same 23 models for all references: `canonical` is the naive reference, while `consensus` and `IC-self` are the alternative primary constructions. The panel must be read as strict-subset reference sensitivity, not a full-36 model-equal claim.
The panel (a) model order is the frozen ascending `median D_cross` order in `F2_MODEL_ORDER.csv`; any later Figure 3 or R4 reuse must preserve that order and distinguish the strict23 CR subset from the 36-model displacement/excess scope.

## Output scope

The formal output is one complete composite PNG only. No panel-wise image, PDF, fourth panel, 1:1 plane, near-zero strip, restart audit, rank/localization metric, map, or performance bridge is included. The output is 600 dpi at approximately 18.2 × 17.0 cm.

- Script: `scripts/plot_r2_figure2_unit_hist_cr_deviation.py`
- Figure: `figures/Figure2_R2_parameter_space_response_final.png`
- Model order: `tables/F2_MODEL_ORDER.csv`
- Values: `tables/F2_V4_FINAL_VALUES.csv`
"""
    (TABLES / "F2_V4_BUILD_NOTE.md").write_text(note)


def draw_a(ax: plt.Axes, paired: pd.DataFrame, order: list[str]) -> None:
    grouped = paired.groupby("model_id")
    y = np.arange(len(order), dtype=float)
    max_displayed = 0.0
    for i, model in enumerate(order):
        if i % 2 == 0:
            ax.axhspan(i - 0.48, i + 0.48, color="#f7f9fb", zorder=0)
        group = grouped.get_group(model)
        for column, color, edge, marker, offset in [
            ("D_theta_IC_self", PALE_PURPLE, PURPLE, "o", 0.14),
            ("D_theta_cross", PALE_GREEN, GREEN, "s", -0.14),
        ]:
            values = group[column].to_numpy(float)
            q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
            max_displayed = max(max_displayed, float(q95))
            yy = i + offset
            ax.hlines(yy, q05, q95, color=edge, lw=0.75, alpha=0.82, zorder=1)
            ax.hlines(yy, q25, q75, color=color, lw=2.6, alpha=0.88, zorder=2)
            if marker == "o":
                ax.scatter(median, yy, s=20, facecolor="white", edgecolor=edge, marker=marker, linewidth=0.85, zorder=4)
            else:
                ax.scatter(median, yy, s=20, facecolor=edge, edgecolor="white", marker=marker, linewidth=0.4, zorder=4)
    self_medians = grouped.D_theta_IC_self.median()
    cross_medians = grouped.D_theta_cross.median()
    self_reference = float(self_medians.median())
    cross_reference = float(cross_medians.median())
    ax.axvline(self_reference, color=MEDIAN_PURPLE, lw=0.95, ls="--", alpha=0.95, zorder=1)
    ax.axvline(cross_reference, color=MEDIAN_GREEN, lw=0.95, ls="--", alpha=0.95, zorder=1)
    ax.text(self_reference + 0.008, -1.00, rf"$D_{{self}}$ median = {self_reference:.3f}", ha="left", va="top", fontsize=9.0, color="black", zorder=4)
    ax.text(cross_reference + 0.008, -1.00, rf"$D_{{cross}}$ median = {cross_reference:.3f}", ha="left", va="top", fontsize=9.0, color="black", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_ylim(len(order) - 0.35, -1.34)
    if max_displayed > 0.8:
        raise RuntimeError("F2 REDRAW = BLOCKED: plotted panel (a) quantiles exceed fixed 0.0–0.8 x-axis")
    ax.set_xlim(-0.05, 0.8)
    ax.set_xlabel("Normalized RMS parameter displacement")
    finish(ax)
    label_panel(ax, "a", "Parameter-space displacement by model")


def draw_b(ax: plt.Axes, recheck: pd.DataFrame, excess: pd.DataFrame) -> None:
    summary = excess.loc[excess.scope == "all36_summary"].iloc[0]
    d = recheck.sort_values("frozen_median_of_basin_differences").reset_index(drop=True)
    values = d.frozen_median_of_basin_differences.to_numpy(float)
    if len(values) != 36:
        raise RuntimeError("F2 REDRAW = BLOCKED: unit histogram requires exactly 36 model estimates")
    x_min, x_max = -0.02, 0.60
    bin_min = -0.05
    edges = np.arange(bin_min, x_max + BIN_WIDTH * 0.5, BIN_WIDTH)
    if values.min() < x_min or values.max() > x_max:
        raise RuntimeError("F2 REDRAW = BLOCKED: Delta_m values fall outside the fixed histogram range")
    bin_indices = np.digitize(values, edges, right=False) - 1
    bin_indices[values == x_max] = len(edges) - 2
    counts = np.bincount(bin_indices, minlength=len(edges) - 1)
    if int(counts.sum()) != 36:
        raise RuntimeError("F2 REDRAW = BLOCKED: panel (b) bin count sum is not 36")
    negative_count = int((values < 0).sum())
    if negative_count != 0 or int(counts[edges[:-1] < 0].sum()) != 0:
        raise RuntimeError("F2 REDRAW = BLOCKED: panel (b) negative-half count is not zero")

    max_count = int(counts.max())
    for bin_index, center in enumerate(edges[:-1] + BIN_WIDTH / 2):
        members = d.iloc[np.flatnonzero(bin_indices == bin_index)]
        for stack_y, (_, row) in enumerate(members.iterrows(), start=1):
            ax.scatter(center, stack_y, marker="s", s=27, facecolor=GREEN, edgecolor=GREEN, linewidth=0.35, zorder=4)
    median = float(summary.cross_minus_self_median)
    ax.axvline(median, color="black", lw=1.0, ls="--", zorder=3)
    positive_count = int((values > 0).sum())
    ax.text(
        0.98,
        0.98,
        f"median = {median:+.3f}\n{positive_count}/{len(values)} $\\Delta_m > 0$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        color="black",
        zorder=5,
    )
    ax.set_ylim(0.0, 11.5)
    ax.set_yticks(np.arange(0, 12, 1))
    ax.set_ylabel("Model count")
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(0.0, x_max + 0.001, 0.10))
    ax.set_xlabel("Paired median excess, $\\Delta_m$")
    finish(ax)
    label_panel(ax, "b", "Paired excess")
def draw_c(ax: plt.Axes, cr: pd.DataFrame) -> None:
    columns = ["CR_canonical", "CR_consensus", "CR_ICself"]
    labels = ["canonical", "consensus", "IC-self"]
    y = np.array([0.0, 0.90, 1.65])
    samples = [np.log2(cr[column].to_numpy(float)) for column in columns]
    raw_z_values = np.concatenate(samples)
    pooled_bandwidth = max(1.06 * float(np.std(raw_z_values, ddof=1)) * len(raw_z_values) ** (-1.0 / 5.0), 0.035)
    grid = np.linspace(float(raw_z_values.min()) - 0.04, float(raw_z_values.max()) + 0.04, 600)
    ridge_height = 0.24

    # Estimate all three densities in the common log-CR coordinate system.
    plot_left = min(float(np.log2(0.40)), float(raw_z_values.min()) - 0.04)
    label_x = float(np.log2(2.0) + 0.28)
    plot_right = max(label_x + 0.08, float(raw_z_values.max()) + 0.04)
    ax.axvspan(plot_left, 0.0, color=NEUTRAL_LIGHT, alpha=0.10, zorder=0)
    ax.axvline(0.0, color="black", lw=1.0, ls="--", zorder=2)

    for i, z_values, fill, top_color in zip(y, samples, RIDGE_FILLS, RIDGE_TOP_COLORS):
        distances = (grid[:, None] - z_values[None, :]) / pooled_bandwidth
        density = np.exp(-0.5 * distances**2).sum(axis=1)
        density /= density.max()
        ridge_top = i - ridge_height * density
        ridge = ax.fill_between(grid, i, ridge_top, color=fill, alpha=0.90, zorder=3, linewidth=0)
        ridge.set_edgecolor("none")
        # The dark series-colored upper boundary and fine black baseline define each ridge.
        ax.plot(grid, ridge_top, color=top_color, lw=0.85, zorder=5)
        ax.hlines(i, grid[0], grid[-1], color="black", lw=0.55, zorder=5)
        # The individual model-level values are shown as short baseline rugs.
        ax.vlines(z_values, i + 0.015, i + 0.12, color="black", lw=0.45, alpha=0.95, zorder=6)
        median = float(np.median(z_values))
        ax.vlines(median, i - ridge_height * 0.82, i + 0.02, color="black", lw=0.85, zorder=7)
        ax.text(label_x, i - 0.22, f"{2.0**median:.3f}", ha="right", va="center", fontsize=10.0, color="black", zorder=8)

    # The compact bracket makes the 1+2 reference structure explicit without
    # introducing a second background encoding.
    bracket_x = float(np.log2(0.45))
    bracket_top = float(y[1] - 0.22)
    bracket_bottom = float(y[2] + 0.22)
    ax.plot([bracket_x, bracket_x + 0.035, bracket_x + 0.035, bracket_x], [bracket_top, bracket_top, bracket_bottom, bracket_bottom], color="black", lw=0.50, zorder=4)
    ax.text(bracket_x + 0.055, (bracket_top + bracket_bottom) / 2, "alternative\nreferences", ha="left", va="center", fontsize=10.5, color="black", zorder=4)
    ax.text(float(np.log2(0.43)), 0.22, "apparent contraction", ha="left", va="center", fontsize=10.5, color="black")

    ax.set_xlim(plot_left, plot_right)
    tick_cr = [0.4, 0.6, 1.0, 1.4, 2.0]
    tick_labels = ["0.4", "0.6", "1.0", "1.4", "2.0"]
    tick_z = [float(np.log2(tick)) for tick in tick_cr if plot_left <= np.log2(tick) <= plot_right]
    visible_labels = [label for tick, label in zip(tick_cr, tick_labels) if plot_left <= np.log2(tick) <= plot_right]
    ax.set_xticks(tick_z)
    ax.set_xticklabels(visible_labels)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, rotation=0, ha="right", fontsize=9.8)
    ax.tick_params(axis="y", pad=4, colors="black")
    ax.set_ylim(2.0, -0.25)
    ax.set_xlabel("Relative parameter spread (CR)")
    finish(ax)
    label_panel(ax, "c", "Relative spread by IC reference")


def main() -> None:
    configure()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    paired, excess, recheck, cr, checks = prepare()
    order = pd.read_csv(TABLES / "F2_MODEL_ORDER.csv").sort_values("plot_order").model_id.tolist()
    if len(order) != 36 or len(set(order)) != 36:
        raise RuntimeError("F2 REDRAW = BLOCKED: model order is not 36 unique models")
    if set(order) != set(paired.model_id.unique()) or set(order) != set(recheck.model_id):
        raise RuntimeError("F2 REDRAW = BLOCKED: model order does not cover the audited data")
    if "collie1" not in order:
        raise RuntimeError("F2 REDRAW = BLOCKED: collie1 is missing from the frozen model order")
    write_build_note(paired, excess, recheck, cr, order, checks)

    # Widened canvas allocates the added width to the right B/C column.
    fig = plt.figure(figsize=(20.0 / 2.54, 17.0 / 2.54), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.08, 1.30], left=0.12, right=0.99, bottom=0.07, top=0.95, wspace=0.16)
    ax_a = fig.add_subplot(grid[0, 0])
    right = grid[0, 1].subgridspec(2, 1, height_ratios=[1.02, 1.0], hspace=0.28)
    ax_b = fig.add_subplot(right[0, 0])
    ax_c = fig.add_subplot(right[1, 0])
    # Nudge the left panel down slightly to align its plotting band with panel (b).
    a_position = ax_a.get_position()
    ax_a.set_position([a_position.x0, a_position.y0 - 0.012, a_position.width, a_position.height])
    draw_a(ax_a, paired, order)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", color=PURPLE, markerfacecolor="white", markeredgecolor=PURPLE, lw=0, markersize=5.0, label="$D_{self}$"),
            Line2D([0], [0], marker="s", color=GREEN, markerfacecolor=GREEN, markeredgecolor="white", lw=0, markersize=5.0, label="$D_{cross}$"),
        ],
        loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=1, fontsize=9.2, frameon=False, handletextpad=0.35, labelspacing=0.25, borderaxespad=0.0, labelcolor="black",
    )
    draw_b(ax_b, recheck, excess)
    draw_c(ax_c, cr)
    if len(fig.axes) != 3:
        raise RuntimeError("F2 REDRAW = BLOCKED: expected exactly three formal panels")
    cr_values = cr[["CR_canonical", "CR_consensus", "CR_ICself"]].to_numpy(float).ravel()
    z_values = np.log2(cr_values)
    if float(z_values.min()) <= ax_c.get_xlim()[0] or float(z_values.max()) >= ax_c.get_xlim()[1]:
        raise RuntimeError("F2 REDRAW = BLOCKED: CR values are clipped")
    fig.savefig(OUTPUT, dpi=600, facecolor="white")
    plt.close(fig)
    print({
        "status": "COMPLETE",
        "figure": str(OUTPUT),
        "layout": "full-height left hero + compact B/C right stack",
        "models": 36,
        "cr_models_per_reference": len(cr),
        "paired_cells": len(paired),
        "quantile_qc": "PASS for D_cross and D_self in all 36 models",
        "above_line_fraction": round(checks["above_fraction"], 6),
        "cr_range": [checks["cr_min"], checks["cr_max"]],
        "cr_axis": "log2",
        "dpi": 600,
    })


if __name__ == "__main__":
    main()
