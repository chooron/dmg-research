# R1 extended supplemental analysis report

## Scope and input checks

This report uses existing R1 tables only; no training, recalibration, CMA-ES, or temporal forward evaluation was run for R1X. The estimand remains:

`delta_KGE = KGE_dPL - KGE_IC`

Inputs: `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r1/tables/R1_model_basin_delta_kge.csv` and `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r1/tables/R1_temporal_AB_model_basin.csv`. The full table was checked as a balanced 36 × 531 matrix with 19,116 unique `(model, basin_id)` cells, no duplicates or missing cells, finite KGE values, and the stated ΔKGE identity. The temporal table contains 38,232 unique `(model, basin_id, partition)` cells with A=`1995-10-01..2003-03-31` and B=`2003-04-01..2010-09-30`.

All results below are descriptive. No cell-level, model-level, or basin-level significance tests, p-values, multiple-testing corrections, or causal interpretations are used.

## Q1. Model-level estimator tendency

Across the 36 model structures, the continuous distribution of `P_m+` (fraction of basins with dPL > IC) has Q10/Q25/median/Q75/Q90 = **0.276836 / 0.367232 / 0.408663 / 0.438795 / 0.493409**, with range **0.214689..0.587571**. `4/36` models have `P_m+ > 0.5`.

The model-level pattern therefore varies across model structures, but the dPL-positive fraction is not uniformly high. This is estimator-relative behavior, not model suitability or a recommendation.

## Q2. Basin-level cross-model estimator tendency

The continuous `P_b+` distribution across 531 basins has Q10/Q25/median/Q75/Q90 = **0.083333 / 0.166667 / 0.361111 / 0.611111 / 0.805556**, with range **0.000000..1.000000**. The corresponding `M_b` distribution has Q10/Q25/median/Q75/Q90 = **-0.092454 / -0.041822 / -0.010959 / 0.009879 / 0.058894**; `M_b` is positive for 175 basins and negative for 356 basins.

Descriptively, 75/531 basins have `P_b+ ≥ 0.75`, 25/531 have `P_b+ ≥ 0.90`, 199/531 have `P_b+ ≤ 0.25`, and 71/531 have `P_b+ ≤ 0.10`. Exact extremes are `P_b+=0` for 14 basins and `P_b+=1` for 3 basins.

**Answer:** YES, the full-period matrix contains a descriptive basin-level tendency: some basins have dPL > IC in most models, while others have IC > dPL in most models. However, this is not a basin-wide suitability claim. The basin-level tendency is heterogeneous and must be read together with the model–basin specificity and temporal checks below.

## Q3. Comparable-or-better sensitivity

`C_b(τ)` is the fraction of models with `ΔKGE ≥ -τ`. τ=0 is the primary no-tolerance description; positive τ values are sensitivity only.

- τ=0.00: median C_b=0.361111; Q25/Q75=0.166667/0.611111; C_b>0.5=32.8%; C_b≥0.75=14.1%; C_b≥0.90=4.7%.
- τ=0.01: median C_b=0.500000; Q25/Q75=0.250000/0.694444; C_b>0.5=46.0%; C_b≥0.75=20.2%; C_b≥0.90=7.2%.
- τ=0.02: median C_b=0.583333; Q25/Q75=0.319444/0.777778; C_b>0.5=58.8%; C_b≥0.75=30.5%; C_b≥0.90=10.0%.
- τ=0.05: median C_b=0.777778; Q25/Q75=0.555556/0.916667; C_b>0.5=77.0%; C_b≥0.75=57.3%; C_b≥0.90=29.9%.

The prevalence increases as tolerance widens, so the apparent extent of comparable-or-better basins is tolerance-dependent. No single positive τ is selected as a preferred threshold.

## Q4. Descriptive two-way decomposition

The descriptive sums-of-squares shares are:

- Model main effect: **0.040412**.
- Basin main effect: **0.297738**.
- Residual model–basin specificity: **0.661850**.

The additive identity error is below numerical precision. The largest share is residual model–basin specificity, not a statistically identified interaction variance. Thus, estimator differences are predominantly organized by specific model–basin combinations, with a nonzero but smaller basin main-effect contribution and a small model main effect. This decomposition is descriptive only: no p-values and no inferential interaction claim.

## Q5. Basin-level temporal persistence

- Spearman correlation of basin median tendencies, `rho_M`: **0.345336**, N=531.
- Same-sign fraction for `M_b^A` and `M_b^B`: **348/531 = 65.5%**; neutral count=0.
- Spearman correlation of basin dPL-positive fractions, `rho_P`: **0.371143**, N=531.

**Answer:** Basin-level tendency is only moderately persistent across the two temporal partitions compared with the stronger model-level persistence in the primary R1 analysis. It supports a descriptive full-period pattern, but does not establish a strong temporally stable basin preference.

## Q6. Basin difficulty / ceiling-effect checks

- `baseline_IC_KGE_median ↔ delta_crossmodel_median`: Spearman=-0.138110, Pearson=-0.178743, N=531.
- `baseline_IC_KGE_median ↔ frac_models_dPL_gt_IC`: Spearman=-0.206428, Pearson=-0.138612, N=531.
- `baseline_IC_KGE_IQR ↔ delta_crossmodel_median`: Spearman=0.011300, Pearson=0.196444, N=531.
- `baseline_IC_KGE_IQR ↔ frac_models_dPL_gt_IC`: Spearman=0.094168, Pearson=0.116656, N=531.

The largest absolute Spearman correlation among these four checks is **0.206428**. These descriptive correlations do not show that basin tendency is simply determined by baseline IC skill or IC cross-model spread; they also do not establish independence from those factors. No catchment attributes or causal interpretation were introduced.

## Outcome decision

**Outcome B with a qualified full-period basin pattern:** basin-level `P_b+` is clearly heterogeneous and includes basins that are dPL-positive or IC-positive across many model structures. Nevertheless, residual model–basin specificity accounts for the largest descriptive share (66.2%), and temporal persistence is moderate rather than strong. The defensible R1 conclusion is therefore that estimator differences are predominantly model–basin specific, with a descriptive basin-level tendency that is not promoted to a universal or suitability interpretation.

## Figure/content gate

- **Model `P_m+`: SI ONLY.** It adds a compact estimator-relative descriptor but substantially overlaps the existing model-level ΔKGE heterogeneity panel.
- **Basin `P_b+`: SI ONLY.** It answers a new question and shows tails, but the continuous pattern is descriptive and not sufficient for a main-text basin preference claim.
- **Two-way decomposition: MAIN-TEXT CANDIDATE.** It directly addresses whether variation is model-, basin-, or combination-organized and can be shown with one simple descriptive component plot.
- **Basin A/B persistence: SI ONLY.** It adds useful basin-level persistence information, but the moderate correlations do not change the R1 temporal headline.
- **Difficulty checks: SI ONLY.** They are robustness context, not a new primary result.
- **Basin GIS visualization: MAIN FIGURE 1 PANEL (c).** The later formal figure build uses the read-only CAMELS location DBF and a cached hydrodiag boundary file; the map remains a descriptive point display with no spatial inference.

## Reproducibility

Scripts `10_model_estimator_tendency.py` through `14_basin_difficulty_checks.py` read only existing R1 tables and write only `R1X_` outputs. `15_plot_r1_extended_diagnostics.py` writes exploratory `R1X_` figures only; formal `Fig1*` files are not modified. The generated tables and figures are under `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r1`.
