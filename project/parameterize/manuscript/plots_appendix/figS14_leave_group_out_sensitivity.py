"""Fig. S14 - Leave-one-group-out and within-group sensitivity diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (  # noqa: E402
    APP_FIG_DIR,
    MM,
    add_panel_label,
    save_fig,
    setup_style,
)

OUT_STEM = "figS14_leave_group_out_and_within_group_sensitivity"
DATA_STEM = "figS14_data_used"
CAPTION_NOTE = "figS14_caption_note.md"

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
EXTENDS_DIR = MANUSCRIPT_ROOT / "extends"
LEAVE_GROUP_OUT_FILE = EXTENDS_DIR / "leave_group_out_relationships.csv"
WITHIN_GROUP_FILE = EXTENDS_DIR / "groupwise_relationships.csv"

RELATIONSHIP_ORDER = [
    ("parBETA", "slope_mean"),
    ("parFC", "pet_mean"),
    ("parPERC", "aridity"),
    ("parUZL", "soil_conductivity"),
    ("parCWH", "frac_snow"),
    ("parCFR", "frac_snow"),
]

RELATIONSHIP_LABELS = {
    ("parBETA", "slope_mean"): "BETA-slope",
    ("parFC", "pet_mean"): "FC-PET",
    ("parPERC", "aridity"): "PERC-aridity",
    ("parUZL", "soil_conductivity"): "UZL-soil cond.",
    ("parCWH", "frac_snow"): "CWH-snow frac.",
    ("parCFR", "frac_snow"): "CFR-snow frac.",
}

GROUP_ORDER = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
GROUP_LABELS = {
    "G1": "G1 Humid steep",
    "G2": "G2 Low-snow humid lowland",
    "G3": "G3 Arid lowland",
    "G4": "G4 Arid seasonal",
    "G5": "G5 Low-snow arid steep",
    "G6": "G6 Snow arid steep",
    "G7": "G7 Snow humid steep",
}

INSUFFICIENT_N = 3


def relationship_label(parameter: str, attribute: str) -> str:
    """Return the compact relationship label used on the figure y-axis."""
    return RELATIONSHIP_LABELS.get((parameter, attribute), f"{parameter}-{attribute}")


def relationship_order(parameter: str, attribute: str) -> int:
    """Return the fixed row order for the selected relationships."""
    pair = (parameter, attribute)
    return RELATIONSHIP_ORDER.index(pair) if pair in RELATIONSHIP_ORDER else len(RELATIONSHIP_ORDER)


def group_order(group_id: str) -> int:
    """Return the fixed hydroclimatic group order."""
    return GROUP_ORDER.index(str(group_id)) if str(group_id) in GROUP_ORDER else len(GROUP_ORDER)


def load_leave_group_out_data() -> pd.DataFrame:
    """Load selected leave-one-group-out sensitivities."""
    data = pd.read_csv(LEAVE_GROUP_OUT_FILE)
    data = data.loc[
        data.apply(lambda row: (row["parameter"], row["attribute"]) in RELATIONSHIP_ORDER, axis=1)
    ].copy()
    data["relationship_label"] = data.apply(
        lambda row: relationship_label(row["parameter"], row["attribute"]), axis=1
    )
    data["relationship_order"] = data.apply(
        lambda row: relationship_order(row["parameter"], row["attribute"]), axis=1
    )
    data["group_order"] = data["excluded_group_id"].map(group_order)
    data["panel"] = "leave_one_group_out"
    data["group_id"] = data["excluded_group_id"]
    data["group_name_figure4"] = data["group_id"].map(
        {
            "G1": "G1 Humid steep",
            "G2": "G2 Low-snow humid lowland",
            "G3": "G3 Arid lowland",
            "G4": "G4 Arid seasonal",
            "G5": "G5 Low-snow arid steep",
            "G6": "G6 Snow arid steep",
            "G7": "G7 Snow humid steep",
        }
    )
    return data.sort_values(["relationship_order", "group_order"]).reset_index(drop=True)


def load_within_group_data() -> pd.DataFrame:
    """Load selected within-group relationships."""
    data = pd.read_csv(WITHIN_GROUP_FILE)
    data = data.loc[
        data.apply(lambda row: (row["parameter"], row["attribute"]) in RELATIONSHIP_ORDER, axis=1)
    ].copy()
    data["relationship_label"] = data.apply(
        lambda row: relationship_label(row["parameter"], row["attribute"]), axis=1
    )
    data["relationship_order"] = data.apply(
        lambda row: relationship_order(row["parameter"], row["attribute"]), axis=1
    )
    data["group_order"] = data["group_id"].map(group_order)
    data["group_name_figure4"] = data["group_id"].map(
        {
            "G1": "G1 Humid steep",
            "G2": "G2 Low-snow humid lowland",
            "G3": "G3 Arid lowland",
            "G4": "G4 Arid seasonal",
            "G5": "G5 Low-snow arid steep",
            "G6": "G6 Snow arid steep",
            "G7": "G7 Snow humid steep",
        }
    )
    data["panel"] = "within_group"
    data.loc[data["within_group_n"] < INSUFFICIENT_N, "within_group_rho"] = np.nan
    return data.sort_values(["relationship_order", "group_order"]).reset_index(drop=True)


def save_data_used(leave_out: pd.DataFrame, within_group: pd.DataFrame) -> pd.DataFrame:
    """Save the long-form values used in both panels."""
    leave_used = leave_out[
        [
            "panel",
            "parameter",
            "parameter_label",
            "attribute",
            "relationship_label",
            "all_basins_rho",
            "all_basins_n",
            "group_id",
            "group_name_figure4",
            "leave_group_out_rho",
            "leave_group_out_n",
            "delta_vs_all",
            "sign_matches_all",
        ]
    ].copy()
    leave_used["within_group_rho"] = np.nan
    leave_used["within_group_n"] = np.nan
    leave_used["within_group_p_value"] = np.nan

    within_used = within_group[
        [
            "panel",
            "parameter",
            "parameter_label",
            "attribute",
            "relationship_label",
            "all_basins_rho",
            "group_id",
            "group_name_figure4",
            "within_group_rho",
            "within_group_n",
            "within_group_p_value",
            "delta_vs_all",
            "sign_matches_all",
        ]
    ].copy()
    within_used["all_basins_n"] = np.nan
    within_used["leave_group_out_rho"] = np.nan
    within_used["leave_group_out_n"] = np.nan

    data_used = pd.concat([leave_used, within_used], ignore_index=True, sort=False)
    ordered_columns = [
        "panel",
        "parameter",
        "parameter_label",
        "attribute",
        "relationship_label",
        "all_basins_rho",
        "all_basins_n",
        "group_id",
        "group_name_figure4",
        "leave_group_out_rho",
        "leave_group_out_n",
        "within_group_rho",
        "within_group_n",
        "within_group_p_value",
        "delta_vs_all",
        "sign_matches_all",
    ]
    data_used = data_used[ordered_columns]
    data_used.to_csv(APP_FIG_DIR / f"{DATA_STEM}.csv", index=False)
    return data_used


def matrix_from_leave_out(data: pd.DataFrame) -> pd.DataFrame:
    """Build panel-a matrix with an all-basin reference column."""
    rows = []
    for parameter, attribute in RELATIONSHIP_ORDER:
        panel = data.loc[(data["parameter"] == parameter) & (data["attribute"] == attribute)]
        row = {"All": float(panel["all_basins_rho"].iloc[0])}
        for group_id in GROUP_ORDER:
            value = panel.loc[panel["group_id"] == group_id, "leave_group_out_rho"]
            row[group_id] = float(value.iloc[0]) if not value.empty else np.nan
        rows.append(row)
    labels = [relationship_label(parameter, attribute) for parameter, attribute in RELATIONSHIP_ORDER]
    return pd.DataFrame(rows, index=labels)


def matrix_from_within_group(data: pd.DataFrame) -> pd.DataFrame:
    """Build panel-b matrix."""
    rows = []
    for parameter, attribute in RELATIONSHIP_ORDER:
        panel = data.loc[(data["parameter"] == parameter) & (data["attribute"] == attribute)]
        row = {}
        for group_id in GROUP_ORDER:
            value = panel.loc[panel["group_id"] == group_id, "within_group_rho"]
            row[group_id] = float(value.iloc[0]) if not value.empty else np.nan
        rows.append(row)
    labels = [relationship_label(parameter, attribute) for parameter, attribute in RELATIONSHIP_ORDER]
    return pd.DataFrame(rows, index=labels)


def draw_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    panel_label: str,
    show_ylabels: bool,
) -> None:
    """Draw a compact annotated heatmap with fixed Spearman rho scale."""
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = LinearSegmentedColormap.from_list(
        "s14_purple_white_green",
        ["#6A3D9A", "#F7F7F7", "#1B9E77"],
        N=256,
    )
    cmap.set_bad(color="#F2F2F2")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_title(title, fontsize=8.2, pad=6)
    ax.set_xticks(np.arange(matrix.shape[1]))
    if "All" in matrix.columns:
        xticklabels = ["All"] + GROUP_ORDER
    else:
        xticklabels = GROUP_ORDER
    ax.set_xticklabels(xticklabels, fontsize=7.2)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)

    ax.set_yticks(np.arange(matrix.shape[0]))
    if show_ylabels:
        ax.set_yticklabels(matrix.index.tolist(), fontsize=7.4)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=3)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                ax.text(col_idx, row_idx, "NA", ha="center", va="center", fontsize=6.2, color="#777777")
                continue
            text_color = "white" if abs(value) >= 0.68 else "#222222"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=6.0, color=text_color)

    if "All" in matrix.columns:
        ax.axvline(0.5, color="#333333", linewidth=0.95)

    add_panel_label(ax, panel_label, x=-0.08 if show_ylabels else -0.03, y=1.12, fontsize=11.0)
    return im


def write_caption_note() -> None:
    """Write the requested caption-ready note."""
    note = (
        "Panel (a) evaluates whether the full-sample relationship is driven by one "
        "hydroclimatic group. Panel (b) shows how the same relationships vary within "
        "individual groups. This is a hydroclimatic-group sensitivity diagnostic, not "
        "ungauged validation.\n"
    )
    (APP_FIG_DIR / CAPTION_NOTE).write_text(note, encoding="utf-8")


def main() -> None:
    setup_style()
    leave_out = load_leave_group_out_data()
    within_group = load_within_group_data()
    save_data_used(leave_out, within_group)
    write_caption_note()

    leave_matrix = matrix_from_leave_out(leave_out)
    within_matrix = matrix_from_within_group(within_group)

    fig = plt.figure(figsize=(184 * MM, 112 * MM))
    ax_a = fig.add_axes([0.15, 0.27, 0.40, 0.56])
    ax_b = fig.add_axes([0.60, 0.27, 0.35, 0.56])
    cax = fig.add_axes([0.34, 0.080, 0.42, 0.026])

    im = draw_heatmap(
        ax_a,
        leave_matrix,
        title=r"Leave-one-group-out Spearman $\rho$",
        panel_label="(a)",
        show_ylabels=True,
    )
    draw_heatmap(
        ax_b,
        within_matrix,
        title=r"Within-group Spearman $\rho$",
        panel_label="(b)",
        show_ylabels=False,
    )

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Spearman rho", fontsize=7.5, labelpad=2)
    cbar.ax.tick_params(labelsize=6.8, length=2)

    group_note = (
        "Groups follow Fig. 4 order (G1-G7); All = full-sample rho."
    )
    fig.text(0.50, 0.145, group_note, ha="center", va="center", fontsize=6.2, color="#555555")

    save_fig(fig, OUT_STEM)
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")
    print(f"Saved {APP_FIG_DIR / DATA_STEM}.csv")
    print(f"Saved {APP_FIG_DIR / CAPTION_NOTE}")


if __name__ == "__main__":
    main()
