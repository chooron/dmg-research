#!/usr/bin/env python3
"""Build the minimal R1 Supplement tables and caption drafts."""
from __future__ import annotations

import pandas as pd

from r1_config import TABLES_DIR


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    rows = [frame[columns].copy()]
    table = rows[0]
    def value(x: object) -> str:
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            return f"{x:.{digits}f}"
        return str(x)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(value(x) for x in row) + " |" for row in table[columns].itertuples(index=False, name=None)]
    return "\n".join([header, divider] + body) + "\n"


def main() -> None:
    import numpy as np
    full = pd.read_csv(TABLES_DIR / "R1_model_basin_delta_kge.csv")
    abs_delta = np.abs(full["delta_KGE"].to_numpy(float))
    dpl_pos = full.loc[full["delta_KGE"] > 0, "delta_KGE"].to_numpy(float)
    ic_pos = full.loc[full["delta_KGE"] < 0, "delta_KGE"].to_numpy(float)

    s1a_rows = [
        {
            "subset": "All model–basin pairs",
            "N": len(full),
            "fraction": 1.0,
            "median_abs_delta_KGE": float(np.median(abs_delta)),
            "Q25_abs_delta_KGE": float(np.quantile(abs_delta, 0.25)),
            "Q75_abs_delta_KGE": float(np.quantile(abs_delta, 0.75)),
            "median_signed_delta_KGE": float(np.median(full["delta_KGE"])),
            "description": "Overall typical magnitude of absolute IC–dPL difference across all 19,116 pairs.",
        },
        {
            "subset": "dPL higher (ΔKGE > 0)",
            "N": len(dpl_pos),
            "fraction": len(dpl_pos) / len(full),
            "median_abs_delta_KGE": float(np.median(dpl_pos)),
            "Q25_abs_delta_KGE": float(np.quantile(dpl_pos, 0.25)),
            "Q75_abs_delta_KGE": float(np.quantile(dpl_pos, 0.75)),
            "median_signed_delta_KGE": float(np.median(dpl_pos)),
            "description": "Conditional effect magnitude when dPL achieves higher evaluation KGE.",
        },
        {
            "subset": "IC higher (ΔKGE < 0)",
            "N": len(ic_pos),
            "fraction": len(ic_pos) / len(full),
            "median_abs_delta_KGE": float(np.median(np.abs(ic_pos))),
            "Q25_abs_delta_KGE": float(np.quantile(np.abs(ic_pos), 0.25)),
            "Q75_abs_delta_KGE": float(np.quantile(np.abs(ic_pos), 0.75)),
            "median_signed_delta_KGE": float(np.median(ic_pos)),
            "description": "Conditional effect magnitude when IC achieves higher evaluation KGE.",
        },
    ]
    df_s1a = pd.DataFrame(s1a_rows)
    s1a_cols = ["subset", "N", "fraction", "median_abs_delta_KGE", "Q25_abs_delta_KGE", "Q75_abs_delta_KGE", "median_signed_delta_KGE", "description"]
    df_s1a[s1a_cols].to_csv(TABLES_DIR / "TableS1A_R1_effect_magnitude_summary.csv", index=False, float_format="%.6f")

    fig_a = pd.read_csv(TABLES_DIR / "Fig1a_model_aggregate_performance.csv")
    fig_b = pd.read_csv(TABLES_DIR / "Fig1b_model_delta_distribution.csv")
    fig_e = pd.read_csv(TABLES_DIR / "Fig1e_temporal_model_persistence.csv")
    s1b = fig_a.merge(fig_b, on=["model", "plot_order"], how="inner", validate="one_to_one").merge(fig_e, on=["model", "plot_order"], how="inner", validate="one_to_one").sort_values("plot_order")
    s1b = s1b.rename(columns={"P_m_positive": "P_m+", "same_sign_non_neutral": "same_direction_AB"})
    s1b_cols = ["model", "plot_order", "n_params", "IC_median_KGE", "dPL_median_KGE", "delta_model_median", "delta_Q10", "delta_Q25", "delta_Q75", "delta_Q90", "P_m+", "delta_median_A", "delta_median_B", "same_direction_AB"]
    df_s1b = s1b[s1b_cols].copy()
    df_s1b.to_csv(TABLES_DIR / "TableS1B_R1_model_summary.csv", index=False, float_format="%.6f")
    df_s1b.to_csv(TABLES_DIR / "TableS1_R1_model_summary.csv", index=False, float_format="%.6f")

    s1_doc = f"""# Table S1. Numerical performance context for R1 (IC vs. dPL)

## Table S1A. Overall ΔKGE effect magnitude across 19,116 model–basin cases

{markdown_table(df_s1a, ["subset", "N", "fraction", "median_abs_delta_KGE", "Q25_abs_delta_KGE", "Q75_abs_delta_KGE", "median_signed_delta_KGE"], digits=4)}
*Note: Throughout, `ΔKGE = KGE_dPL − KGE_IC`. Across all 19,116 cases, the overall median absolute difference is 0.0456 KGE units. Conditional on dPL being higher (40.07% of cases), the median advantage is +0.0425; conditional on IC being higher (59.93% of cases), the median advantage is +0.0477 (i.e. median signed difference −0.0477). Ties (`ΔKGE = 0`) occurred in 0 cases.*

## Table S1B. Canonical 36-model outlet performance and ΔKGE distribution summary

{markdown_table(df_s1b, ["model", "n_params", "IC_median_KGE", "dPL_median_KGE", "delta_model_median", "delta_Q10", "delta_Q25", "delta_Q75", "delta_Q90", "P_m+", "delta_median_A", "delta_median_B", "same_direction_AB"], digits=4)}
*Note: Model rows are shown in canonical registry order. `IC_median_KGE` and `dPL_median_KGE` are median test KGE values across 531 basins; `delta_model_median` is the median of within-model ΔKGE across 531 basins. `P_m+` is the fraction of the 531 basins where `ΔKGE > 0`. `delta_median_A` and `delta_median_B` are the median ΔKGE in temporal splits A (1995–2003) and B (2003–2010), evaluated forward-only on frozen results with no retraining.*
"""
    TABLES_DIR.joinpath("TableS1_R1_performance_context.md").write_text(s1_doc)
    TABLES_DIR.joinpath("TableS1_R1_model_summary.md").write_text(s1_doc)

    simhyd = pd.read_csv(TABLES_DIR / "R1_sensitivity_exclude_simhyd.csv")
    mean_median = pd.read_csv(TABLES_DIR / "R1_sensitivity_mean_vs_median.csv")
    ensemble = mean_median.loc[mean_median.model == "__ENSEMBLE__"].iloc[0]
    def sim(metric: str) -> pd.Series:
        return simhyd.loc[simhyd.metric == metric].iloc[0]
    temporal = pd.read_csv(TABLES_DIR / "R1X_temporal_basin_persistence_summary.csv").set_index("metric")
    rows = [
        {"check": "primary_full_36", "metric": "IC ensemble median KGE", "primary_value": sim("Fig1a_IC_median_of_model_medians").full_36, "alternative_value": "", "contrast": "", "notes": "Median aggregation across 36 model-level basin medians."},
        {"check": "primary_full_36", "metric": "dPL ensemble median KGE", "primary_value": sim("Fig1a_dPL_median_of_model_medians").full_36, "alternative_value": "", "contrast": "", "notes": "Median aggregation across 36 model-level basin medians."},
        {"check": "primary_full_36", "metric": "median model-level ΔKGE", "primary_value": sim("Fig1a_delta_median_of_model_medians").full_36, "alternative_value": "", "contrast": "", "notes": "ΔKGE = dPL − IC."},
        {"check": "exclude_simhyd", "metric": "median model-level ΔKGE", "primary_value": sim("Fig1a_delta_median_of_model_medians").full_36, "alternative_value": sim("Fig1a_delta_median_of_model_medians").exclude_simhyd_35, "contrast": sim("Fig1a_delta_median_of_model_medians").exclude_minus_full, "notes": "Full 36 versus exclude-simhyd 35; sensitivity only."},
        {"check": "mean_vs_median", "metric": "IC ensemble aggregation", "primary_value": ensemble.IC_basin_median, "alternative_value": ensemble.IC_basin_mean, "contrast": ensemble.IC_basin_mean - ensemble.IC_basin_median, "notes": "Basin median versus basin mean; model remains the ensemble unit."},
        {"check": "mean_vs_median", "metric": "dPL ensemble aggregation", "primary_value": ensemble.dPL_basin_median, "alternative_value": ensemble.dPL_basin_mean, "contrast": ensemble.dPL_basin_mean - ensemble.dPL_basin_median, "notes": "Basin median versus basin mean; model remains the ensemble unit."},
        {"check": "mean_vs_median", "metric": "median model-level ΔKGE", "primary_value": ensemble.delta_basin_median, "alternative_value": ensemble.delta_basin_mean, "contrast": ensemble.delta_mean_minus_median, "notes": "Basin median versus basin mean sensitivity."},
        {"check": "temporal_basin", "metric": "Spearman rho of basin M_A/M_B", "primary_value": temporal.loc["rho_M_spearman", "value"], "alternative_value": "", "contrast": "", "notes": "N=531 basins; descriptive temporal persistence."},
        {"check": "temporal_basin", "metric": "same-sign basin fraction", "primary_value": temporal.loc["same_sign_fraction", "value"], "alternative_value": "", "contrast": "", "notes": "Neutral count and denominator are in the R1X temporal summary."},
        {"check": "tolerance", "metric": "comparable-or-better C_b(τ)", "primary_value": "τ=0, 0.01, 0.02, 0.05", "alternative_value": "", "contrast": "tolerance-dependent", "notes": "Full sensitivity table remains R1X internal support; do not treat τ>0 as a primary threshold."},
    ]
    s2 = pd.DataFrame(rows)
    s2.to_csv(TABLES_DIR / "TableS2_R1_compact_robustness_summary.csv", index=False, float_format="%.10f")
    s2_columns = ["check", "metric", "primary_value", "alternative_value", "contrast", "notes"]
    TABLES_DIR.joinpath("TableS2_R1_compact_robustness_summary.md").write_text("# Table S2. Compact R1 robustness summary\n\n" + markdown_table(s2, s2_columns, digits=6) + "\nRecommendation: retain as internal support unless the manuscript explicitly cites these sensitivity checks.\n")

    print(f"PASS: wrote Table S1 (S1A, S1B, and Markdown), Table S2 under {TABLES_DIR}")


if __name__ == "__main__":
    main()
