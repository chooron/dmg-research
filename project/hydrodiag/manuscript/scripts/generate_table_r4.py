"""Generate Main and Supplementary Tables for Manuscript R4.

Outputs:
  - Table 4 (Main Manuscript): Quantitative summary of soil-water state consistency & timing diagnostics
  - Table S6 (Supplementary): Robustness checks summary (performance control, LORO, extreme trimming)
  - Table S7 (Supplementary): Timing-definition sensitivity across wet-up thresholds and peak windows

Formats:
  LaTeX (.tex), Markdown (.md), and machine-readable CSV (.csv)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r4.common import default_results_root  # noqa: E402
from r4.robustness_analysis import bootstrap_median_ci  # noqa: E402

TABLES_DIR = HERE.parents[0] / "tables"
Q3_QUANTILE = 0.75


def _df_to_markdown_clean(df: pd.DataFrame, title: str, note: str = "") -> str:
    headers = list(df.columns)
    lines = [
        f"# {title}\n",
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join([":---" if i == 0 else ":---:" for i in range(len(headers))])
        + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
    if note:
        lines.append(f"\n*{note}*")
    return "\n".join(lines) + "\n"


def generate_table4_main(r4_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Generate Table 4: R4 Main-text Quantitative Summary."""
    df_phase = pd.read_csv(r4_dir / "robustness_process_phase_consistency.csv")
    df_paired = pd.read_csv(
        r4_dir / "paired_structural_effects.csv", dtype={"basin_id": str}
    )
    df_summary = pd.read_csv(
        r4_dir / "timing_metrics_basin_summary.csv", dtype={"basin_id": str}
    )
    df_sens = pd.read_csv(r4_dir / "robustness_timing_sensitivity.csv")

    swe_q3 = df_paired[df_paired["regime"] == "dPL_seed42"].drop_duplicates("basin_id")
    q3_val = float(swe_q3["snow_burden_swe_mm"].quantile(Q3_QUANTILE))
    q3_basins = set(swe_q3[swe_q3["snow_burden_swe_mm"] >= q3_val]["basin_id"])
    q3_n = len(q3_basins)

    rows = []

    # 1. Active-melt state consistency difference
    row_1 = {
        "Quantity": "Active-melt state-consistency difference ($\Delta\\rho_{\\text{anom}}$, CN − Base)"
    }
    # 2. Summer dry-down state consistency difference
    row_2 = {
        "Quantity": "Summer dry-down state-consistency difference ($\Delta\\rho_{\\text{anom}}$, CN − Base)"
    }
    # 3. Active-melt consistency superiority in Q3
    row_3 = {
        "Quantity": f"Active-melt consistency superiority in Q3 (% catchments with CN > Base, n = {q3_n})"
    }
    # 4. Spring wet-up median signed timing error
    row_4 = {
        "Quantity": "Spring wet-up median signed error (Base $\\rightarrow$ CN, days)"
    }
    # 5. Soil-water peak median signed timing error
    row_5 = {
        "Quantity": "Soil-water peak median signed error (Base $\\rightarrow$ CN, days)"
    }
    # 6. Spring wet-up baseline MAE reduction
    row_6 = {
        "Quantity": "Spring wet-up baseline MAE reduction (Base MAE − CN MAE, days)"
    }
    # 7. Soil-water peak baseline MAE reduction
    row_7 = {
        "Quantity": "Soil-water peak baseline MAE reduction (Base MAE − CN MAE, days)"
    }

    for reg, col_name in [("dPL_seed42", "dPL-42"), ("IC_fused", "IC fused")]:
        # Row 1
        sub_am = df_phase[
            (df_phase["regime"] == reg)
            & (df_phase["phase_name"] == "Phase_2_Active_Melt_Recharge")
        ]
        vals_am = sub_am["delta_anomaly_corr"].to_numpy(float)
        m_am, l_am, h_am = bootstrap_median_ci(vals_am)
        row_1[col_name] = f"{m_am:+.3f} [{l_am:+.3f}, {h_am:+.3f}]"

        # Row 2
        sub_dd = df_phase[
            (df_phase["regime"] == reg)
            & (df_phase["phase_name"] == "Phase_4_Summer_Dry_Down")
        ]
        vals_dd = sub_dd["delta_anomaly_corr"].to_numpy(float)
        m_dd, l_dd, h_dd = bootstrap_median_ci(vals_dd)
        row_2[col_name] = f"{m_dd:+.3f} [{l_dd:+.3f}, {h_dd:+.3f}]"

        # Row 3
        sub_am_q3 = sub_am[sub_am["snow_burden_swe_mm"] >= q3_val]
        pct_am_q3 = (
            100.0
            * (sub_am_q3["cn_anomaly_corr"] > sub_am_q3["base_anomaly_corr"]).mean()
        )
        row_3[col_name] = f"{pct_am_q3:.1f}%"

        # Row 4 & 5
        sub_sum = df_summary[
            (df_summary["regime"] == reg) & (df_summary["basin_id"].isin(q3_basins))
        ]
        piv_w = sub_sum.pivot(
            index="basin_id", columns="structure", values="median_wetup_error_days"
        )
        row_4[col_name] = (
            f"{piv_w['Base'].median():+.1f} $\\rightarrow$ {piv_w['CN'].median():+.1f}"
        )

        piv_p = sub_sum.pivot(
            index="basin_id", columns="structure", values="median_peak_error_days"
        )
        row_5[col_name] = (
            f"{piv_p['Base'].median():+.1f} $\\rightarrow$ {piv_p['CN'].median():+.1f}"
        )

        # Row 6 & 7 (canonical: Wetup 14d, Peak FullWY)
        canon_row = df_sens[
            (df_sens["regime"] == reg)
            & (df_sens["wetup_definition"] == "Wetup_14d_Spring")
            & (df_sens["peak_definition"] == "Peak_Annual_FullWY")
        ].iloc[0]
        row_6[col_name] = f"{canon_row['wetup_abs_error_improvement_days']:.1f}"
        row_7[col_name] = f"{canon_row['peak_abs_error_improvement_days']:.1f}"

    df_t4 = pd.DataFrame([row_1, row_2, row_3, row_4, row_5, row_6, row_7])

    # Save CSV, Markdown, LaTeX
    csv_path = out_dir / "Table4_soil_state_consistency.csv"
    md_path = out_dir / "Table4_soil_state_consistency.md"
    tex_path = out_dir / "Table4_soil_state_consistency.tex"

    df_t4.to_csv(csv_path, index=False)
    md_content = _df_to_markdown_clean(
        df_t4,
        "Table 4: Quantitative Summary of Soil-Water State Consistency and Timing Diagnostics (R4)",
        "Note: Reference is ERA5-Land SM100 composite (0–100 cm). Model state is total soil water storage W_total = wu + wl + wd. Q3 denotes the upper snow-burden quartile (SWE ≥ 133.4 mm, n = 133 catchments). Bracketed values report 95% bootstrap confidence intervals of the median across catchments. Timing metrics are evaluated across valid snow years in the 1995–2010 test period using canonical 14-day wet-up and annual peak definitions.",
    )
    md_path.write_text(md_content, encoding="utf-8")
    df_t4.to_latex(tex_path, index=False, escape=False)
    return df_t4


def generate_tables6_robustness(r4_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Generate Table S6: Supplementary Robustness Checks."""
    df_reg = pd.read_csv(r4_dir / "robustness_controlled_regressions.csv")
    df_loro = pd.read_csv(r4_dir / "robustness_leave_one_region_out.csv")
    df_trim = pd.read_csv(r4_dir / "robustness_extreme_swe_trimming.csv")

    reg_anom = df_reg[df_reg["target_metric"] == "delta_anomaly_corr"]

    rows = []

    # Block A: Performance Control
    rows.append(
        {
            "Robustness Check / Specification": "A. Performance control ($\Delta$KGE-controlled OLS regression)",
            "dPL-42": "",
            "IC fused": "",
        }
    )
    r_dpl_a = reg_anom[reg_anom["regime"] == "dPL_seed42"].iloc[0]
    r_ic_a = reg_anom[reg_anom["regime"] == "IC_fused"].iloc[0]
    rows.append(
        {
            "Robustness Check / Specification": "  Controlled SWE $\\beta_1$ [std.] [95% CI]",
            "dPL-42": f"{r_dpl_a['beta1_swe_burden_std']:.3f} [{r_dpl_a['beta1_ci_lower']:.3f}, {r_dpl_a['beta1_ci_upper']:.3f}]",
            "IC fused": f"{r_ic_a['beta1_swe_burden_std']:.3f} [{r_ic_a['beta1_ci_lower']:.3f}, {r_ic_a['beta1_ci_upper']:.3f}]",
        }
    )

    # Block B: Leave-one-HUC02-out
    rows.append(
        {
            "Robustness Check / Specification": "B. Leave-one-HUC02-out cross-region stability (18 regions)",
            "dPL-42": "",
            "IC fused": "",
        }
    )
    sub_dpl_loro = df_loro[df_loro["regime"] == "dPL_seed42"]
    sub_ic_loro = df_loro[df_loro["regime"] == "IC_fused"]

    full_dpl = sub_dpl_loro[sub_dpl_loro["dropped_region"] == "NONE (Full Sample)"][
        "rho_delta_anomaly_swe"
    ].iloc[0]
    full_ic = sub_ic_loro[sub_ic_loro["dropped_region"] == "NONE (Full Sample)"][
        "rho_delta_anomaly_swe"
    ].iloc[0]

    loro_dpl = sub_dpl_loro[sub_dpl_loro["dropped_region"] != "NONE (Full Sample)"][
        "rho_delta_anomaly_swe"
    ]
    loro_ic = sub_ic_loro[sub_ic_loro["dropped_region"] != "NONE (Full Sample)"][
        "rho_delta_anomaly_swe"
    ]

    rows.append(
        {
            "Robustness Check / Specification": "  Full-sample Spearman $\\rho$(SWE, $\\Delta\\text{Anom.}$)",
            "dPL-42": f"{full_dpl:.3f}",
            "IC fused": f"{full_ic:.3f}",
        }
    )
    rows.append(
        {
            "Robustness Check / Specification": "  Leave-one-region-out range [min, max]",
            "dPL-42": f"[{loro_dpl.min():.3f}, {loro_dpl.max():.3f}]",
            "IC fused": f"[{loro_ic.min():.3f}, {loro_ic.max():.3f}]",
        }
    )
    rows.append(
        {
            "Robustness Check / Specification": "  Evaluated HUC02 regions / Sign flips",
            "dPL-42": f"{len(loro_dpl)} / {(loro_dpl < 0).sum()}",
            "IC fused": f"{len(loro_ic)} / {(loro_ic < 0).sum()}",
        }
    )

    # Block C: Extreme-SWE Trimming
    rows.append(
        {
            "Robustness Check / Specification": "C. Extreme-SWE trimming (Spearman $\\rho$(SWE, $\\Delta\\text{Anom.}$))",
            "dPL-42": "",
            "IC fused": "",
        }
    )
    for scheme, label in [
        ("full_sample", "Full sample (n = 531)"),
        ("trim_top_1pct", "Trim top 1% SWE (n = 525)"),
        ("trim_top_5pct", "Trim top 5% SWE (n = 504)"),
    ]:
        val_dpl = df_trim[
            (df_trim["regime"] == "dPL_seed42") & (df_trim["trimming_scheme"] == scheme)
        ]["rho_delta_anomaly_swe"].iloc[0]
        val_ic = df_trim[
            (df_trim["regime"] == "IC_fused") & (df_trim["trimming_scheme"] == scheme)
        ]["rho_delta_anomaly_swe"].iloc[0]
        rows.append(
            {
                "Robustness Check / Specification": f"  {label}",
                "dPL-42": f"{val_dpl:.3f}",
                "IC fused": f"{val_ic:.3f}",
            }
        )

    df_ts6 = pd.DataFrame(rows)

    csv_path = out_dir / "TableS6_robustness_checks.csv"
    md_path = out_dir / "TableS6_robustness_checks.md"
    tex_path = out_dir / "TableS6_robustness_checks.tex"

    df_ts6.to_csv(csv_path, index=False)
    md_content = _df_to_markdown_clean(
        df_ts6,
        "Table S6: Robustness Checks for Soil-Water State-Consistency Separation (Figure 7f numerical summary)",
        "Note: Performance control regresses basin-level delta anomaly correlation against standardized SWE burden while controlling for delta KGE. Leave-one-region-out omits each of the 18 CAMELS-US HUC02 regions in turn. Extreme-SWE trimming removes catchments in the top 1% (SWE > 838 mm) and top 5% (SWE > 465 mm).",
    )
    md_path.write_text(md_content, encoding="utf-8")
    df_ts6.to_latex(tex_path, index=False, escape=False)
    return df_ts6


def generate_tables7_timing_sensitivity(r4_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Generate Table S7: Supplementary Timing-Definition Sensitivity."""
    df_sens = pd.read_csv(r4_dir / "robustness_timing_sensitivity.csv")

    rows_def = [
        ("Wet-up 7 d", "Peak_Annual_FullWY", "Wetup_07d_Spring", "wetup"),
        ("Wet-up 14 d (canonical)", "Peak_Annual_FullWY", "Wetup_14d_Spring", "wetup"),
        ("Wet-up 21 d", "Peak_Annual_FullWY", "Wetup_21d_Spring", "wetup"),
        ("Peak full WY (canonical)", "Peak_Annual_FullWY", "Wetup_14d_Spring", "peak"),
        ("Peak Mar–Aug", "Peak_SpringSummer_MarAug", "Wetup_14d_Spring", "peak"),
    ]

    rows = []
    for label, p_def, w_def, m_type in rows_def:
        row_dict = {"Timing Definition": label}
        for reg, prefix in [("dPL_seed42", "dPL-42"), ("IC_fused", "IC")]:
            sub = df_sens[
                (df_sens["regime"] == reg)
                & (df_sens["peak_definition"] == p_def)
                & (df_sens["wetup_definition"] == w_def)
            ].iloc[0]
            if m_type == "wetup":
                b_mae = sub["base_abs_wetup_error_median"]
                c_mae = sub["cn_abs_wetup_error_median"]
                d_mae = sub["wetup_abs_error_improvement_days"]
            else:
                b_mae = sub["base_abs_peak_error_median"]
                c_mae = sub["cn_abs_peak_error_median"]
                d_mae = sub["peak_abs_error_improvement_days"]
            row_dict[f"{prefix} Base MAE [d]"] = f"{b_mae:.1f}"
            row_dict[f"{prefix} CN MAE [d]"] = f"{c_mae:.1f}"
            row_dict[f"{prefix} $\Delta$MAE [d]"] = f"{d_mae:+.1f}"
        rows.append(row_dict)

    df_ts7 = pd.DataFrame(rows)

    csv_path = out_dir / "TableS7_timing_sensitivity.csv"
    md_path = out_dir / "TableS7_timing_sensitivity.md"
    tex_path = out_dir / "TableS7_timing_sensitivity.tex"

    df_ts7.to_csv(csv_path, index=False)
    md_content = _df_to_markdown_clean(
        df_ts7,
        "Table S7: Sensitivity of Timing Metrics Across Alternative Event Thresholds and Windows (Figure 8f numerical summary)",
        "Note: MAE denotes catchment median absolute timing error relative to the ERA5-Land SM100 reference evaluated across valid snow years in the 1995–2010 test period. Delta MAE = Base MAE − CN MAE (positive values indicate CN timing is closer to reference). All values are in days.",
    )
    md_path.write_text(md_content, encoding="utf-8")
    df_ts7.to_latex(tex_path, index=False, escape=False)
    return df_ts7


def generate_all_tables(results_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    r4_dir = results_root / "r4_phase1_soil_official"

    t4 = generate_table4_main(r4_dir, out_dir)
    ts6 = generate_tables6_robustness(r4_dir, out_dir)
    ts7 = generate_tables7_timing_sensitivity(r4_dir, out_dir)

    print("Generated Tables successfully:")
    print(f"  Table 4:  {out_dir / 'Table4_soil_state_consistency.md'}")
    print(f"  Table S6: {out_dir / 'TableS6_robustness_checks.md'}")
    print(f"  Table S7: {out_dir / 'TableS7_timing_sensitivity.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()
    generate_all_tables(args.results_root or default_results_root(), args.out_dir)
