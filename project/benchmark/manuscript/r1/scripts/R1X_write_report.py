#!/usr/bin/env python3
"""Write the data-backed R1X supplemental analysis report."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from R1X_common import FULL_TABLE, TEMPORAL_TABLE
from r1_config import MODEL_REGISTRY, R1_ROOT, TABLES_DIR, TEMPORAL_AB

REPORT_PATH = R1_ROOT / "R1_EXTENDED_ANALYSIS_REPORT.md"


def f(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def main() -> None:
    model = pd.read_csv(TABLES_DIR / "R1X_model_estimator_tendency.csv")
    basin = pd.read_csv(TABLES_DIR / "R1X_basin_estimator_tendency.csv")
    comparable = pd.read_csv(TABLES_DIR / "R1X_comparable_or_better_population_summary.csv")
    decomposition = pd.read_csv(TABLES_DIR / "R1X_two_way_decomposition.csv")
    temporal = pd.read_csv(TABLES_DIR / "R1X_temporal_basin_persistence_summary.csv")
    difficulty = pd.read_csv(TABLES_DIR / "R1X_basin_difficulty_checks.csv")

    model_p = model["frac_dPL_gt_IC"]
    basin_p = basin["frac_models_dPL_gt_IC"]
    basin_m = basin["delta_crossmodel_median"]
    model_gt_half = int((model_p > 0.5).sum())
    p_ge_75 = int((basin_p >= 0.75).sum())
    p_ge_90 = int((basin_p >= 0.90).sum())
    p_le_25 = int((basin_p <= 0.25).sum())
    p_le_10 = int((basin_p <= 0.10).sum())
    p_eq_zero = int((basin_p == 0.0).sum())
    p_eq_one = int((basin_p == 1.0).sum())
    m_positive = int((basin_m > 0.0).sum())
    m_negative = int((basin_m < 0.0).sum())

    def metric(name: str) -> float:
        return float(temporal.loc[temporal.metric == name, "value"].iloc[0])

    def component(name: str) -> float:
        return float(decomposition.loc[decomposition.component == name, "fraction_of_total"].iloc[0])

    difficulty_lines = "\n".join(
        f"- `{row.covariate} ↔ {row.outcome}`: Spearman={f(row.spearman)}, Pearson={f(row.pearson)}, N={int(row.N)}."
        for row in difficulty.itertuples()
    )
    sensitivity_lines = "\n".join(
        f"- τ={row['tau']:.2f}: median C_b={f(row['median_C_b'])}; Q25/Q75={f(row['C_b_Q25'])}/{f(row['C_b_Q75'])}; "
        f"C_b>0.5={pct(row['frac_basins_C_gt_0.5'])}; C_b≥0.75={pct(row['frac_basins_C_ge_0.75'])}; C_b≥0.90={pct(row['frac_basins_C_ge_0.90'])}."
        for row in comparable.to_dict("records")
    )

    report = f"""# R1 extended supplemental analysis report

## Scope and input checks

This report uses existing R1 tables only; no training, recalibration, CMA-ES, or temporal forward evaluation was run for R1X. The estimand remains:

`delta_KGE = KGE_dPL - KGE_IC`

Inputs: `{FULL_TABLE}` and `{TEMPORAL_TABLE}`. The full table was checked as a balanced {len(MODEL_REGISTRY)} × 531 matrix with 19,116 unique `(model, basin_id)` cells, no duplicates or missing cells, finite KGE values, and the stated ΔKGE identity. The temporal table contains 38,232 unique `(model, basin_id, partition)` cells with A=`{TEMPORAL_AB[0]['start_date']}..{TEMPORAL_AB[0]['end_date']}` and B=`{TEMPORAL_AB[1]['start_date']}..{TEMPORAL_AB[1]['end_date']}`.

All results below are descriptive. No cell-level, model-level, or basin-level significance tests, p-values, multiple-testing corrections, or causal interpretations are used.

## Q1. Model-level estimator tendency

Across the 36 model structures, the continuous distribution of `P_m+` (fraction of basins with dPL > IC) has Q10/Q25/median/Q75/Q90 = **{f(model_p.quantile(.10))} / {f(model_p.quantile(.25))} / {f(model_p.median())} / {f(model_p.quantile(.75))} / {f(model_p.quantile(.90))}**, with range **{f(model_p.min())}..{f(model_p.max())}**. `{model_gt_half}/36` models have `P_m+ > 0.5`.

The model-level pattern therefore varies across model structures, but the dPL-positive fraction is not uniformly high. This is estimator-relative behavior, not model suitability or a recommendation.

## Q2. Basin-level cross-model estimator tendency

The continuous `P_b+` distribution across 531 basins has Q10/Q25/median/Q75/Q90 = **{f(basin_p.quantile(.10))} / {f(basin_p.quantile(.25))} / {f(basin_p.median())} / {f(basin_p.quantile(.75))} / {f(basin_p.quantile(.90))}**, with range **{f(basin_p.min())}..{f(basin_p.max())}**. The corresponding `M_b` distribution has Q10/Q25/median/Q75/Q90 = **{f(basin_m.quantile(.10))} / {f(basin_m.quantile(.25))} / {f(basin_m.median())} / {f(basin_m.quantile(.75))} / {f(basin_m.quantile(.90))}**; `M_b` is positive for {m_positive} basins and negative for {m_negative} basins.

Descriptively, {p_ge_75}/531 basins have `P_b+ ≥ 0.75`, {p_ge_90}/531 have `P_b+ ≥ 0.90`, {p_le_25}/531 have `P_b+ ≤ 0.25`, and {p_le_10}/531 have `P_b+ ≤ 0.10`. Exact extremes are `P_b+=0` for {p_eq_zero} basins and `P_b+=1` for {p_eq_one} basins.

**Answer:** YES, the full-period matrix contains a descriptive basin-level tendency: some basins have dPL > IC in most models, while others have IC > dPL in most models. However, this is not a basin-wide suitability claim. The basin-level tendency is heterogeneous and must be read together with the model–basin specificity and temporal checks below.

## Q3. Comparable-or-better sensitivity

`C_b(τ)` is the fraction of models with `ΔKGE ≥ -τ`. τ=0 is the primary no-tolerance description; positive τ values are sensitivity only.

{sensitivity_lines}

The prevalence increases as tolerance widens, so the apparent extent of comparable-or-better basins is tolerance-dependent. No single positive τ is selected as a preferred threshold.

## Q4. Descriptive two-way decomposition

The descriptive sums-of-squares shares are:

- Model main effect: **{f(component('model_main_effect'))}**.
- Basin main effect: **{f(component('basin_main_effect'))}**.
- Residual model–basin specificity: **{f(component('residual_model_basin_specificity'))}**.

The additive identity error is below numerical precision. The largest share is residual model–basin specificity, not a statistically identified interaction variance. Thus, estimator differences are predominantly organized by specific model–basin combinations, with a nonzero but smaller basin main-effect contribution and a small model main effect. This decomposition is descriptive only: no p-values and no inferential interaction claim.

## Q5. Basin-level temporal persistence

- Spearman correlation of basin median tendencies, `rho_M`: **{f(metric('rho_M_spearman'))}**, N=531.
- Same-sign fraction for `M_b^A` and `M_b^B`: **{int(metric('same_sign_numerator'))}/{int(metric('same_sign_denominator'))} = {pct(metric('same_sign_fraction'))}**; neutral count={int(metric('neutral_count'))}.
- Spearman correlation of basin dPL-positive fractions, `rho_P`: **{f(metric('rho_P_spearman'))}**, N=531.

**Answer:** Basin-level tendency is only moderately persistent across the two temporal partitions compared with the stronger model-level persistence in the primary R1 analysis. It supports a descriptive full-period pattern, but does not establish a strong temporally stable basin preference.

## Q6. Basin difficulty / ceiling-effect checks

{difficulty_lines}

The largest absolute Spearman correlation among these four checks is **{f(difficulty.spearman.abs().max())}**. These descriptive correlations do not show that basin tendency is simply determined by baseline IC skill or IC cross-model spread; they also do not establish independence from those factors. No catchment attributes or causal interpretation were introduced.

## Outcome decision

**Outcome B with a qualified full-period basin pattern:** basin-level `P_b+` is clearly heterogeneous and includes basins that are dPL-positive or IC-positive across many model structures. Nevertheless, residual model–basin specificity accounts for the largest descriptive share ({pct(component('residual_model_basin_specificity'))}), and temporal persistence is moderate rather than strong. The defensible R1 conclusion is therefore that estimator differences are predominantly model–basin specific, with a descriptive basin-level tendency that is not promoted to a universal or suitability interpretation.

## Figure/content gate

- **Model `P_m+`: SI ONLY.** It adds a compact estimator-relative descriptor but substantially overlaps the existing model-level ΔKGE heterogeneity panel.
- **Basin `P_b+`: SI ONLY.** It answers a new question and shows tails, but the continuous pattern is descriptive and not sufficient for a main-text basin preference claim.
- **Two-way decomposition: MAIN-TEXT CANDIDATE.** It directly addresses whether variation is model-, basin-, or combination-organized and can be shown with one simple descriptive component plot.
- **Basin A/B persistence: SI ONLY.** It adds useful basin-level persistence information, but the moderate correlations do not change the R1 temporal headline.
- **Difficulty checks: SI ONLY.** They are robustness context, not a new primary result.
- **Basin GIS visualization: MAIN FIGURE 1 PANEL (c).** The later formal figure build uses the read-only CAMELS location DBF and a cached hydrodiag boundary file; the map remains a descriptive point display with no spatial inference.

## Reproducibility

Scripts `10_model_estimator_tendency.py` through `14_basin_difficulty_checks.py` read only existing R1 tables and write only `R1X_` outputs. `15_plot_r1_extended_diagnostics.py` writes exploratory `R1X_` figures only; formal `Fig1*` files are not modified. The generated tables and figures are under `{R1_ROOT}`.
"""
    REPORT_PATH.write_text(report)
    print(f"PASS: wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
