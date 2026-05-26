"""Fig. S4 — Regional stability comparison across hydroclimatic strata.
Style: boxplot panels identical to Fig02 (grouped by relationship class,
stratified by aridity / snow fraction / precip. seasonality terciles).

Computes Spearman rho between learned parameter values and basin attributes
within each tercile group, then reports seed SD of rho across seeds.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (
    setup_style, clean_axes, add_panel_label,
    MM, MODEL_ORDER, TABLE_ROOT, CORR_ROOT,
    math_model_labels, APP_FIG_DIR, save_fig,
)

OUT_STEM      = "figS4_regional_stability_comparison"
MIN_BASINS    = 20   # skip tercile groups smaller than this

STRATA_ATTRS  = ["aridity", "frac_snow", "p_seasonality"]
STRATA_LABELS = {"aridity": "Aridity", "frac_snow": "Snow fraction",
                 "p_seasonality": "Precip. seasonality"}
CLASS_ORDER   = ["robust", "loss-sensitive", "model-sensitive"]
CLASS_LABELS  = {"robust": "Robust", "loss-sensitive": "Loss-sens.",
                 "model-sensitive": "Model-sens."}
GROUP_LABELS  = {"low": "Low", "middle": "Mid", "high": "High"}
GROUP_COLORS  = {"low": "#4C78A8", "middle": "#AAB7C4", "high": "#F58518"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_params(table_root: Path) -> pd.DataFrame:
    """Basin-level parameter values: one row per (basin, model, loss, seed, param)."""
    df = pd.read_csv(table_root / "params_long.csv")
    # distributional model has duplicate rows per basin — average them
    df = (df.groupby(["basin_id", "model", "loss", "seed", "parameter"],
                     as_index=False)["mean"].mean())
    df = df.rename(columns={"mean": "param_value"})
    return df


def _assign_terciles(basin_df: pd.DataFrame,
                     strata_attrs: list[str]) -> pd.DataFrame:
    """Add tercile group columns for each strata attribute."""
    out = basin_df[["basin_id"] + strata_attrs].copy()
    for attr in strata_attrs:
        if attr not in out.columns:
            continue
        out[f"{attr}_group"] = pd.qcut(
            out[attr], q=3, labels=["low", "middle", "high"],
            duplicates="drop"
        ).astype(str)
    return out


def _compute_strata_seed_std(params: pd.DataFrame,
                              basin_terciles: pd.DataFrame,
                              rel: pd.DataFrame,
                              strata_attrs: list[str]) -> pd.DataFrame:
    """
    For each (model, loss, seed, strata_attr, tercile_group, parameter, attribute)
    compute Spearman rho within the tercile group, then aggregate across seeds
    to get seed_std_rho.
    """
    # attributes to correlate = those appearing in relationship_classes
    corr_attrs = rel["attribute"].unique().tolist()
    basin_attr_df = pd.read_csv(TABLE_ROOT / "basin_attributes.csv")

    records = []
    params_merged = params.merge(basin_terciles, on="basin_id", how="inner")

    for strata_attr in strata_attrs:
        group_col = f"{strata_attr}_group"
        if group_col not in params_merged.columns:
            continue

        for model in params_merged["model"].unique():
            m_df = params_merged[params_merged["model"] == model]

            for loss in m_df["loss"].unique():
                ml_df = m_df[m_df["loss"] == loss]

                for seed in ml_df["seed"].unique():
                    s_df = ml_df[ml_df["seed"] == seed]

                    for grp in ["low", "middle", "high"]:
                        g_df = s_df[s_df[group_col] == grp]
                        if len(g_df["basin_id"].unique()) < MIN_BASINS:
                            warnings.warn(
                                f"Skipping {model}/{loss}/seed={seed}/"
                                f"{strata_attr}={grp}: "
                                f"only {len(g_df['basin_id'].unique())} basins"
                            )
                            continue

                        # pivot to basin × parameter
                        piv = g_df.pivot_table(
                            index="basin_id", columns="parameter",
                            values="param_value", aggfunc="mean"
                        )
                        # basin attributes for this group
                        ba = basin_attr_df[
                            basin_attr_df["basin_id"].isin(piv.index)
                        ].set_index("basin_id")

                        for param in piv.columns:
                            if param not in rel["parameter"].values:
                                continue
                            p_vals = piv[param].dropna()
                            for attr in corr_attrs:
                                if attr not in ba.columns:
                                    continue
                                a_vals = ba[attr].reindex(p_vals.index).dropna()
                                common = p_vals.index.intersection(a_vals.index)
                                if len(common) < MIN_BASINS:
                                    continue
                                rho, _ = stats.spearmanr(
                                    p_vals.loc[common], a_vals.loc[common]
                                )
                                records.append({
                                    "model": model, "loss": loss, "seed": seed,
                                    "strata_attribute": strata_attr,
                                    "tercile_group": grp,
                                    "parameter": param, "attribute": attr,
                                    "rho": rho,
                                    "n_basins": len(common),
                                })

    rho_df = pd.DataFrame(records)
    if rho_df.empty:
        return rho_df

    # seed SD of rho
    seed_std = (
        rho_df.groupby(
            ["model", "loss", "strata_attribute", "tercile_group",
             "parameter", "attribute"]
        ).agg(
            seed_std_rho=("rho", "std"),
            n_basins=("n_basins", "mean"),
        ).reset_index()
    )

    # merge relationship class (model-level)
    seed_std = seed_std.merge(
        rel[["model", "parameter", "attribute", "relationship_class"]],
        on=["model", "parameter", "attribute"], how="left"
    )
    return seed_std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(seed_std: pd.DataFrame) -> None:
    setup_style()
    labels  = math_model_labels()
    n_rows  = len(STRATA_ATTRS)
    n_cols  = len(MODEL_ORDER)

    fig_w = 200 * MM
    fig_h = n_rows * 52 * MM + 20 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(n_rows, n_cols,
                           left=0.10, right=0.99, top=0.96, bottom=0.12,
                           hspace=0.38, wspace=0.30)

    group_gap = 1.0
    box_gap   = 0.26
    width     = 0.22
    panel_idx = 0

    for row, strata_attr in enumerate(STRATA_ATTRS):
        sub_strata = seed_std[seed_std["strata_attribute"] == strata_attr]

        for col, model in enumerate(MODEL_ORDER):
            ax  = fig.add_subplot(gs[row, col])
            sub = sub_strata[sub_strata["model"] == model]

            centers = []
            positions, plot_data, colors = [], [], []

            for ci, cls in enumerate(CLASS_ORDER):
                center = ci * group_gap
                centers.append(center)
                csub = sub[sub["relationship_class"] == cls]

                for gi, grp in enumerate(["low", "middle", "high"]):
                    pos  = center + (gi - 1) * box_gap
                    vals = (csub[csub["tercile_group"] == grp]
                            ["seed_std_rho"].dropna().values)
                    positions.append(pos)
                    plot_data.append(vals)
                    colors.append(GROUP_COLORS[grp])

            bp = ax.boxplot(
                plot_data, positions=positions, widths=width,
                patch_artist=True, showfliers=False,
                medianprops={"color": "#2A2A2A", "lw": 1.0},
                whiskerprops={"color": "#666666", "lw": 0.7},
                capprops={"color": "#666666", "lw": 0.7},
                boxprops={"edgecolor": "#666666", "lw": 0.7},
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.72)

            ax.set_xticks(centers)
            ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER],
                               fontsize=7.0)
            ax.set_ylabel("Seed SD of ρ" if col == 0 else "", fontsize=7.5)
            if row == 0:
                ax.set_title(labels[model], fontsize=9, pad=4)
            if col == 0:
                ax.text(-0.30, 0.5, STRATA_LABELS[strata_attr],
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=7.5, color="#333333")
            clean_axes(ax, grid_axis="y")
            add_panel_label(ax, f"({chr(ord('a') + panel_idx)})",
                            x=0.98, y=0.98, ha="right", va="top",
                            fontweight="normal", fontsize=10.5)
            panel_idx += 1

    title_handle = mpatches.Patch(visible=False, label="Tercile group:")
    handles = [title_handle] + [
        mpatches.Patch(facecolor=GROUP_COLORS[g], alpha=0.80,
                       edgecolor="none", label=GROUP_LABELS[g])
        for g in ["low", "middle", "high"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, 0.01),
               handlelength=1.2, handletextpad=0.4, columnspacing=1.2)

    save_fig(fig, OUT_STEM)
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    params        = _load_params(TABLE_ROOT)
    basin_df      = pd.read_csv(TABLE_ROOT / "basin_attributes.csv")
    basin_terciles = _assign_terciles(basin_df, STRATA_ATTRS)
    rel           = pd.read_csv(CORR_ROOT / "relationship_classes.csv")

    print("Computing strata-stratified Spearman rho (this may take a minute)…")
    seed_std = _compute_strata_seed_std(params, basin_terciles, rel, STRATA_ATTRS)

    if seed_std.empty:
        print("ERROR: no data computed — check MIN_BASINS threshold or data paths.")
        return

    # save stats table
    out_table = APP_FIG_DIR.parent / "tables" / f"{OUT_STEM}_stats.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    seed_std.to_csv(out_table, index=False)
    print(f"Stats saved to {out_table}")

    # sanity check: low/middle/high should differ
    for strata_attr in STRATA_ATTRS:
        for model in MODEL_ORDER:
            sub = seed_std[
                (seed_std["strata_attribute"] == strata_attr) &
                (seed_std["model"] == model)
            ]
            medians = sub.groupby("tercile_group")["seed_std_rho"].median()
            if medians.nunique() == 1:
                print(f"WARNING: {model}/{strata_attr} — all tercile medians identical!")

    _plot(seed_std)


if __name__ == "__main__":
    main()
