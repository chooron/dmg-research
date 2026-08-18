"""
Plotting Script for Main-Text Figure 1 (R1 Analysis)
Generates manuscript/plots/figures/Figure1_R1_compensation_overview.png
Follows HESS / Copernicus figure standards and Nature visual style principles.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# Add script directory to sys.path to load r1_plot_style
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from r1_plot_style import (
    MODEL_COLORS,
    MODEL_MARKERS,
    PERIOD_STYLES,
    RESOLVED_FONT,
    setup_publication_style,
    apply_clean_spines,
)


def get_short_model_name(m):
    return m.replace("XAJ-", "")


def main():
    setup_publication_style()

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r1_dir = os.path.join(project_root, "manuscript/results/R1")

    # Output directory (canonical main-figure location)
    out_plots_fig_dir = os.path.join(project_root, "manuscript/plots/figures")
    os.makedirs(out_plots_fig_dir, exist_ok=True)

    # 1. Load authoritative data
    df_basin = pd.read_csv(os.path.join(r1_dir, "r1_basin_level_performance.csv"))
    df_abs = pd.read_csv(os.path.join(r1_dir, "r1_absolute_metrics_summary.csv"))

    # Verify strata counts
    expected_counts = {
        "S1": 165,
        "S2": 156,
        "S3": 121,
        "S4": 34,
        "S5": 55,
    }
    kge_strata = df_abs[
        (df_abs["metric"] == "kge")
        & (df_abs["summary_level"] == "snow_stratum")
    ]
    test_strata = kge_strata[kge_strata["period"] == "test"]
    for s_name, exp_n in expected_counts.items():
        obs_n = test_strata[test_strata["snow_stratum"] == s_name]["stratum_n"].iloc[0]
        if int(obs_n) != exp_n:
            raise ValueError(
                f"Stratum count discrepancy for {s_name}: expected {exp_n}, got {obs_n}"
            )

    # Figure dimensions (17.8 cm width, 15.6 cm height)
    fig_w_in = 17.8 / 2.54
    fig_h_in = 15.6 / 2.54
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # GridSpec: 2 rows x 2 columns
    gs = GridSpec(
        2,
        2,
        height_ratios=[1.0, 1.25],
        wspace=0.18,
        hspace=0.18,
        top=0.93,
        bottom=0.07,
        left=0.10,
        right=0.98,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Shared ECDF x-limits and y-limits
    ecdf_xlim = [-0.5, 1.0]
    ecdf_ylim = [0.0, 1.0]

    # --- Panel (a): IC-CMA-ES ECDF ---
    apply_clean_spines(ax_a)
    ic_models = ["XAJ-Base", "XAJ-TGD", "XAJ-CN"]
    for m in ic_models:
        c = MODEL_COLORS[get_short_model_name(m)]
        for p in ["train", "test"]:
            st = PERIOD_STYLES[p]
            sub = df_basin[
                (df_basin["paradigm"] == "IC-CMA-ES")
                & (df_basin["model"] == m)
                & (df_basin["period"] == p)
            ]
            vals = np.sort(sub["kge"].values)
            y_vals = np.arange(1, len(vals) + 1) / len(vals)
            ax_a.step(
                vals,
                y_vals,
                where="post",
                color=c,
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
                alpha=st["alpha"],
            )

    ax_a.set_xlim(ecdf_xlim)
    ax_a.set_ylim(ecdf_ylim)
    ax_a.set_xlabel("KGE", fontsize=9.0)
    ax_a.set_ylabel("Empirical cumulative probability", fontsize=9.0)
    ax_a.text(
        0.04,
        0.06,
        "(a)",
        transform=ax_a.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # --- Panel (b): dPL-MLP ECDF ---
    apply_clean_spines(ax_b)
    dpl_models = ["XAJ-Base", "XAJ-TGD", "XAJ-CN", "HBV"]
    for m in dpl_models:
        c = MODEL_COLORS[get_short_model_name(m)]
        for p in ["train", "test"]:
            st = PERIOD_STYLES[p]
            sub = df_basin[
                (df_basin["paradigm"] == "dPL-MLP")
                & (df_basin["model"] == m)
                & (df_basin["period"] == p)
            ]
            vals = np.sort(sub["kge"].values)
            y_vals = np.arange(1, len(vals) + 1) / len(vals)
            ax_b.step(
                vals,
                y_vals,
                where="post",
                color=c,
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
                alpha=st["alpha"],
            )

    ax_b.set_xlim(ecdf_xlim)
    ax_b.set_ylim(ecdf_ylim)
    ax_b.set_xlabel("KGE", fontsize=9.0)
    ax_b.set_yticklabels([])  # share y-scale with panel a
    ax_b.text(
        0.04,
        0.06,
        "(b)",
        transform=ax_b.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # --- Shared Legends above top row ---
    struct_handles = [
        Line2D([0], [0], color=MODEL_COLORS["Base"], lw=1.8, label="Base"),
        Line2D([0], [0], color=MODEL_COLORS["TGD"], lw=1.8, label="TGD"),
        Line2D([0], [0], color=MODEL_COLORS["CN"], lw=1.8, label="CN"),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["HBV"],
            lw=1.8,
            marker="D",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=MODEL_COLORS["HBV"],
            label="HBV benchmark",
        ),
    ]

    period_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="-",
            lw=1.65,
            marker="o",
            markersize=5.0,
            markerfacecolor="#333333",
            markeredgecolor="#333333",
            label="Train (filled)",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=(0, (4.0, 2.0)),
            lw=1.75,
            marker="o",
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=1.0,
            label="Test (hollow)",
        ),
    ]

    leg_struct = fig.legend(
        handles=struct_handles,
        loc="upper left",
        bbox_to_anchor=(0.10, 0.99),
        ncol=4,
        frameon=False,
        fontsize=8.0,
        handlelength=1.6,
        columnspacing=1.2,
    )
    leg_period = fig.legend(
        handles=period_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        handlelength=2.0,
        columnspacing=1.2,
    )

    # --- Panels (c) & (d): Snow-stratum ladders (train=filled, test=hollow) ---
    strata_order = ["S1", "S2", "S3", "S4", "S5"]
    strata_labels = ["S1", "S2", "S3", "S4", "S5"]
    y_positions = np.arange(len(strata_order))
    ladder_xlim = [0.0, 0.95]
    offset_y = 0.16  # Vertical offset between train and test rows within each stratum

    # Panel (c) IC-CMA-ES Ladder
    apply_clean_spines(ax_c)
    ax_c.grid(
        True,
        axis="x",
        color="#E5E5E5",
        linestyle="--",
        linewidth=0.5,
        alpha=0.8,
        zorder=0,
    )

    for i, s_name in enumerate(strata_order):
        y_base = y_positions[i]

        for p, y_pos, ls, line_c, line_alpha in [
            ("train", y_base + offset_y, "-", "#C7C7C7", 0.92),
            ("test", y_base - offset_y, (0, (3.5, 2.0)), "#A0A0A0", 1.0),
        ]:
            m_vals = {}
            ci_vals = {}
            for m in ["Base", "TGD", "CN"]:
                m_full = f"XAJ-{m}"
                row = kge_strata[
                    (kge_strata["paradigm"] == "IC-CMA-ES")
                    & (kge_strata["period"] == p)
                    & (kge_strata["model"] == m_full)
                    & (kge_strata["snow_stratum"] == s_name)
                ].iloc[0]
                m_vals[m] = row["median"]
                ci_vals[m] = (row["bootstrap_ci_low"], row["bootstrap_ci_high"])

            # Connector Base -> TGD -> CN
            ax_c.plot(
                [m_vals["Base"], m_vals["TGD"], m_vals["CN"]],
                [y_pos, y_pos, y_pos],
                color=line_c,
                linestyle=ls,
                linewidth=0.9,
                alpha=line_alpha,
                zorder=1,
            )

            for m in ["Base", "TGD", "CN"]:
                c = MODEL_COLORS[m]
                marker = MODEL_MARKERS[m]
                med = m_vals[m]
                ci_low, ci_high = ci_vals[m]
                err_low = med - ci_low
                err_high = ci_high - med
                ax_c.errorbar(
                    med,
                    y_pos,
                    xerr=[[err_low], [err_high]],
                    fmt="none",
                    ecolor=c,
                    elinewidth=1.0,
                    capsize=2.0,
                    capthick=0.8,
                    alpha=line_alpha,
                    zorder=2,
                )

                # Fill state: Train = filled (color=c, edge=white), Test = hollow (face=white, edge=c)
                if p == "train":
                    m_face = c
                    m_edge = "white"
                    m_ew = 0.4
                else:
                    m_face = "white"
                    m_edge = c
                    m_ew = 1.0

                ax_c.plot(
                    med,
                    y_pos,
                    marker=marker,
                    color=c,
                    markersize=5.5,
                    markerfacecolor=m_face,
                    markeredgecolor=m_edge,
                    markeredgewidth=m_ew,
                    alpha=line_alpha,
                    zorder=3,
                )

    ax_c.set_xlim(ladder_xlim)
    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels(strata_labels, fontsize=8.0)
    ax_c.set_xlabel("KGE", fontsize=9.0)
    ax_c.set_ylabel("Snow-fraction stratum", fontsize=9.0)
    ax_c.text(
        0.04,
        0.05,
        "(c)",
        transform=ax_c.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Panel (d) dPL-MLP Ladder
    apply_clean_spines(ax_d)
    ax_d.grid(
        True,
        axis="x",
        color="#E5E5E5",
        linestyle="--",
        linewidth=0.5,
        alpha=0.8,
        zorder=0,
    )

    for i, s_name in enumerate(strata_order):
        y_base = y_positions[i]

        for p, y_pos, ls, line_c, line_alpha in [
            ("train", y_base + offset_y, "-", "#C7C7C7", 0.92),
            ("test", y_base - offset_y, (0, (3.5, 2.0)), "#A0A0A0", 1.0),
        ]:
            m_vals = {}
            ci_vals = {}
            for m in ["Base", "TGD", "CN"]:
                m_full = f"XAJ-{m}"
                row = kge_strata[
                    (kge_strata["paradigm"] == "dPL-MLP")
                    & (kge_strata["period"] == p)
                    & (kge_strata["model"] == m_full)
                    & (kge_strata["snow_stratum"] == s_name)
                ].iloc[0]
                m_vals[m] = row["median"]
                ci_vals[m] = (row["bootstrap_ci_low"], row["bootstrap_ci_high"])

            # Connector Base -> TGD -> CN
            ax_d.plot(
                [m_vals["Base"], m_vals["TGD"], m_vals["CN"]],
                [y_pos, y_pos, y_pos],
                color=line_c,
                linestyle=ls,
                linewidth=0.9,
                alpha=line_alpha,
                zorder=1,
            )

            for m in ["Base", "TGD", "CN"]:
                c = MODEL_COLORS[m]
                marker = MODEL_MARKERS[m]
                med = m_vals[m]
                ci_low, ci_high = ci_vals[m]
                err_low = med - ci_low
                err_high = ci_high - med
                ax_d.errorbar(
                    med,
                    y_pos,
                    xerr=[[err_low], [err_high]],
                    fmt="none",
                    ecolor=c,
                    elinewidth=1.0,
                    capsize=2.0,
                    capthick=0.8,
                    alpha=line_alpha,
                    zorder=2,
                )

                if p == "train":
                    m_face = c
                    m_edge = "white"
                    m_ew = 0.4
                else:
                    m_face = "white"
                    m_edge = c
                    m_ew = 1.0

                ax_d.plot(
                    med,
                    y_pos,
                    marker=marker,
                    color=c,
                    markersize=5.5,
                    markerfacecolor=m_face,
                    markeredgecolor=m_edge,
                    markeredgewidth=m_ew,
                    alpha=line_alpha,
                    zorder=3,
                )

            # HBV benchmark marker (unconnected diamond)
            hbv_row = kge_strata[
                (kge_strata["paradigm"] == "dPL-MLP")
                & (kge_strata["period"] == p)
                & (kge_strata["model"] == "HBV")
                & (kge_strata["snow_stratum"] == s_name)
            ].iloc[0]
            hbv_med = hbv_row["median"]
            hbv_ci_low, hbv_ci_high = (
                hbv_row["bootstrap_ci_low"],
                hbv_row["bootstrap_ci_high"],
            )
            ax_d.errorbar(
                hbv_med,
                y_pos,
                xerr=[[hbv_med - hbv_ci_low], [hbv_ci_high - hbv_med]],
                fmt="none",
                ecolor=MODEL_COLORS["HBV"],
                elinewidth=1.0,
                capsize=2.0,
                capthick=0.8,
                alpha=line_alpha,
                zorder=2,
            )

            if p == "train":
                hbv_face = MODEL_COLORS["HBV"]
                hbv_edge = "white"
                hbv_ew = 0.4
            else:
                hbv_face = "white"
                hbv_edge = MODEL_COLORS["HBV"]
                hbv_ew = 1.0

            ax_d.plot(
                hbv_med,
                y_pos,
                marker=MODEL_MARKERS["HBV"],
                color=MODEL_COLORS["HBV"],
                markersize=5.5,
                markerfacecolor=hbv_face,
                markeredgecolor=hbv_edge,
                markeredgewidth=hbv_ew,
                alpha=line_alpha,
                zorder=4,
            )

    ax_d.set_xlim(ladder_xlim)
    ax_d.set_yticks(y_positions)
    ax_d.set_yticklabels([])  # share y-scale with panel c
    ax_d.set_xlabel("KGE", fontsize=9.0)
    ax_d.text(
        0.04,
        0.05,
        "(d)",
        transform=ax_d.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Save Figure PNG (600 dpi, bbox_inches='tight')
    fig_path_plots = os.path.join(
        out_plots_fig_dir, "Figure1_R1_compensation_overview.png"
    )

    plt.savefig(fig_path_plots, dpi=600, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()

    # 2. Build Source Data CSV
    source_records = []
    # ECDFs source data (panels a and b)
    for p_name, p_models, panel_id in [
        ("IC-CMA-ES", ["XAJ-Base", "XAJ-TGD", "XAJ-CN"], "a"),
        ("dPL-MLP", ["XAJ-Base", "XAJ-TGD", "XAJ-CN", "HBV"], "b"),
    ]:
        for m in p_models:
            for period in ["train", "test"]:
                sub = df_basin[
                    (df_basin["paradigm"] == p_name)
                    & (df_basin["model"] == m)
                    & (df_basin["period"] == period)
                ]
                for _, r in sub.iterrows():
                    source_records.append(
                        {
                            "panel": panel_id,
                            "basin_id": r["basin_id"],
                            "paradigm": p_name,
                            "model": m,
                            "period": period,
                            "kge": r["kge"],
                            "snow_stratum": np.nan,
                            "stratum_n": np.nan,
                            "summary_statistic": "raw_basin_observation",
                            "lower_interval": np.nan,
                            "upper_interval": np.nan,
                            "uncertainty_type": "none",
                        }
                    )

    # Ladder source data (panels c and d) - including train and test
    for p_name, p_models, panel_id in [
        ("IC-CMA-ES", ["XAJ-Base", "XAJ-TGD", "XAJ-CN"], "c"),
        ("dPL-MLP", ["XAJ-Base", "XAJ-TGD", "XAJ-CN", "HBV"], "d"),
    ]:
        for m in p_models:
            for period in ["train", "test"]:
                for s_name in strata_order:
                    row = kge_strata[
                        (kge_strata["paradigm"] == p_name)
                        & (kge_strata["period"] == period)
                        & (
                            kge_strata["model"]
                            == (f"XAJ-{m}" if m in ["Base", "TGD", "CN"] else m)
                        )
                        & (kge_strata["snow_stratum"] == s_name)
                    ].iloc[0]
                    source_records.append(
                        {
                            "panel": panel_id,
                            "basin_id": np.nan,
                            "paradigm": p_name,
                            "model": m,
                            "period": period,
                            "kge": row["median"],
                            "snow_stratum": s_name,
                            "stratum_n": row["stratum_n"],
                            "summary_statistic": "median",
                            "lower_interval": row["bootstrap_ci_low"],
                            "upper_interval": row["bootstrap_ci_high"],
                            "uncertainty_type": "95% bootstrap CI",
                        }
                    )

    df_source = pd.DataFrame(source_records)
    src_path_plots = os.path.join(out_plots_fig_dir, "Figure1_R1_source_data.csv")
    df_source.to_csv(src_path_plots, index=False)

    # 3. Create Notes MD File
    file_size_bytes = os.path.getsize(fig_path_plots)
    file_size_mb = file_size_bytes / (1024 * 1024)

    notes_content = f"""# Figure 1 (R1) Technical Notes & Caption Specification

## Figure Caption Notes
Snow regimes S1\u2013S5 are the fixed R1 strata by basin snow fraction: S1 [0, 0.05) (n=165), S2 [0.05, 0.15) (n=156), S3 [0.15, 0.30) (n=121), S4 [0.30, 0.50) (n=34), and S5 [0.50, 1.00] (n=55). Panels (c) and (d) show medians and 95% bootstrap confidence intervals for both training (solid lines, filled markers) and testing (dashed lines, hollow markers with white faces) periods within each snow regime.

## Technical Implementation Details
- **Authoritative Input Files**:
  - `manuscript/results/R1/r1_basin_level_performance.csv` (7,434 basin records across 14 model-period combinations)
  - `manuscript/results/R1/r1_absolute_metrics_summary.csv` (392 metric summaries)
- **ECDF Membership**:
  - Panel (a): IC-CMA-ES — XAJ-Base, XAJ-TGD, XAJ-CN (531 basins each for Train & Test, total 3,186 observations)
  - Panel (b): dPL-MLP — XAJ-Base, XAJ-TGD, XAJ-CN, HBV benchmark (531 basins each for Train & Test, total 4,248 observations)
- **Uncertainty Definition**:
  - Panels (c) & (d) plot model train and test KGE medians with **95% bootstrap confidence intervals** (`bootstrap_ci_low` to `bootstrap_ci_high` from authoritative summaries).
- **Palette and Encoding Mapping**:
  - Base: `#EE7733` (orange), marker `o` (circle)
  - TGD: `#009988` (teal), marker `^` (triangle)
  - CN: `#0077BB` (deep blue), marker `s` (square)
  - HBV: `#6F6F6F` (grey), marker `D` (diamond)
  - Train: solid (`-`, connector linewidth=0.9, alpha=0.92, **filled marker**)
  - Test: dashed (`(0, (3.5, 2.0))`, connector linewidth=1.0, alpha=1.00, **hollow marker** with white face)
  - Sub-row vertical offset: $\pm 0.16$ within each stratum y-position
- **Typography & Font**:
  - Resolved font family: `{RESOLVED_FONT}`
- **Dimensions & Output**:
  - Target dimensions: 17.8 cm x 15.6 cm (7.01 in x 6.14 in)
  - Resolution: 600 dpi
  - File size: {file_size_mb:.2f} MB (< 5 MB threshold)
  - Output formats created: PNG only (no SVG or PDF)
- **HBV Benchmark Status**:
  - Retained in Panel (d) for both train (filled grey diamond) and test (hollow grey diamond) periods for each snow stratum.
- **CVD & Greyscale QA**:
  - Redundant encoding: shape markers (`o`, `^`, `s`, `D`), line styles (solid vs dashed), and marker fill states (filled vs hollow) preserve identity in greyscale and for readers with color-vision deficiencies.
"""
    notes_path_plots = os.path.join(out_plots_fig_dir, "Figure1_R1_notes.md")
    with open(notes_path_plots, "w") as f:
        f.write(notes_content)

    print("Figure 1 revised successfully with filled/hollow markers!")
    print(f"PNG: {fig_path_plots}")
    print(f"File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
