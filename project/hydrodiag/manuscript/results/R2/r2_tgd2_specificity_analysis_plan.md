# R2 TGD2 Structure-Specificity Analysis Plan (Pre-Registered)

Status: pre-registered before any TGD2-specific computation (Step 1 of the TGD2
structure-specificity work). Interpretation rules fixed here are applied to the
results afterwards without post-hoc revision.

## 0. Verified repository facts (Step 0 audit)

| Fact | Verified value | Source |
|---|---|---|
| Base parameter count | 15 (all shared XAJ) | `results/xaj_base_cmaes_531_batched_paired_v2/raw/xaj/*.json` |
| CN parameter count | 17 (`cn_ctg`, `cn_kf` + 15 shared) | `results/xaj_cn_cmaes_531_batched_paired_v2/raw/xaj_cn/*.json` |
| TGD2 parameter count | 17 (`tgd_tau_warm`, `tgd_delta_tau_cold` + 15 shared) | `results/xaj_tgd2_cmaes_531_batched_v1/raw/xaj_tgd2/*.json` |
| TGD2 vs CN dimension | **matched: both D = 17** | model `parameter_specs.py` (`XAJ_CN_PARAM_SPECS`, `XAJ_TGD2_PARAM_SPECS = {**TGD2_PARAM_SPECS, **XAJ_PARAM_SPECS}`) |
| TGD2 vs Base dimension | TGD2 has exactly 2 additional parameters | same |
| 15 shared parameters | `k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, a, theta` | frozen `COMMON_XAJ` in `run_r2_within_structure_baseline.py` |
| IC TGD2 coverage | 531 basins × 10 restarts (5310 raw JSON) | `results/xaj_tgd2_cmaes_531_batched_v1/raw/xaj_tgd2/` |
| dPL TGD2 coverage | 531 basins × 3 seeds (42/123/2026), z in [0,1] | `r2_parameter_values_seed_level.csv` (structure `GD`) |
| dPL canonical rule (GD) | common maximum periodic checkpoint epoch 100 (fixed by R1) | `run_r2_parameter_statistics.py`; canonical median NOT used for within baseline here |
| Bounds | all 15 shared params present, identical across structures | `s2_parameter_bounds_from_code.csv` |

Consequence: because TGD2 and CN share the same total parameter dimension (D=17),
TGD2 provides a **parameter-count-matched generic-flexibility control** relative to
CN (only used in writing once the computation confirms the numbers above).

## 1. Frozen definitions reused without change

- Basin set = 531 (statistical unit = basin).
- Snow regimes: S1=165, S2=156, S3=121, S4=34, S5=55; Exclude-S5 = 476.
- 15 shared parameters normalized to [0,1] physical space, `z=(theta-lower)/(upper-lower)`.
- `d_rms(x,y) = sqrt(mean_p (x_p - y_p)^2)` over the 15 shared parameters.
- IC within-structure distance: median over all C(10,2)=45 restart pairs.
- dPL within-structure distance: median over all C(3,2)=3 seed pairs (seed-level values,
  NOT the canonical per-parameter median — canonical medians are reserved for Figure 4).
- `within_pooled(A,B) = (within_A + within_B) / 2`.
- `between_all(A,B)` = median over all cross-restart (10×10) or cross-seed (3×3) RMS distances.
- `excess(A,B) = between_all(A,B) - within_pooled(A,B)`.
- Bootstrap: 10,000 resamples, seed `20260730`, basin as resampling unit.
- Regression: OLS `excess ~ frac_snow` (also `within_pooled`, `between_all`); report slope,
  95% bootstrap CI, Spearman rho.
- Frozen Base–CN numbers are NOT recomputed with new definitions; my implementation must
  reproduce the existing `r2_within_structure_*.csv` values exactly (validation gate).

## 2. New contrasts (this work)

- Base–CN (frozen reference; recomputed only to validate the pipeline reproduces frozen numbers)
- Base–TGD2 (new)
- TGD2–CN (new, secondary diagnostic)

All three use the identical RMS / within / between / excess definitions above.

## 3. Primary specificity question

Compare, within each parameter-estimation regime (IC, dPL):

- `excess(Base–CN)` vs `excess(Base–TGD2)` across S1–S5;
- `beta(excess ~ frac_snow)` for Base–CN vs Base–TGD2;
- `delta_beta = beta(Base–CN) - beta(Base–TGD2)` with a **paired basin bootstrap** CI
  (same resample fits both slopes, difference taken within each resample), for:
  - IC Full (n=531)
  - IC Exclude-S5 (n=476)
  - dPL Full (n=531)
  - dPL Exclude-S5 (n=476)

`delta_beta` is the quantitative structural-specificity estimand.

## 4. Pre-registered interpretation rules (applied verbatim)

1. If Base–TGD2 snow-dependent excess slope is clearly weaker than Base–CN AND the
   paired slope-difference CI supports a positive difference:
   R2 may be written as: explicit snow-process representation produces a more distinct,
   snow-organized parameter-space contrast than a generic temperature-dependent memory.
   This is a statement about the *structure-conditioned organization of parameter space*,
   NOT about parameter truth or physical correctness.
2. If Base–TGD2 and Base–CN slopes are close, or the slope-difference CI crosses zero:
   do NOT claim snow-process specificity for the R2 parameter-space evidence;
   R2 main claim reverts to `structure-conditioned, snow-organized parameter
   reorganization`; process specificity is carried by R1 and future R3;
   the result is not hidden or softened.
3. If results fall in between: classify as `PARTIAL / QUALIFIED`; no binary packaging.

Classification output: `SUPPORTED` | `PARTIAL / QUALIFIED` | `NOT SUPPORTED`.

## 5. Regime weighting

- IC specificity = primary evidence (IC is the main independent anchor for snow-gradient
  evidence; its frac_snow is a static basin attribute, not a model input).
- dPL specificity = secondary replication / expression under a shared parameter-mapping
  (frac_snow is an attribute input to the dPL mapping, so dPL snow dependence is NOT an
  independent environmental discovery).
- No IC-vs-dPL slope-magnitude ranking anywhere.

## 6. TGD2–CN secondary diagnostic

- Computed fully; enters Figure 3 only if it provides independent structural information
  and does not crowd the panel; otherwise Table / Supplement.
- If highly redundant with Base–CN / Base–TGD2, it goes to Table / Supplement.

## 7. Interpretation constraints (writing rules)

- `compensatory` allowed only as interpretation combined with R1 outlet-level masking
  evidence, e.g. "consistent with compensatory parameter adjustment".
- No parameter transplant, no post-selected low-dimensional distance, no
  remove-frac_snow retraining, no new experiments.
- No internal-state / flux / parameter-truth mechanism discussion (R3).
- `delta_beta` for dPL is never interpreted as an attribute-independent discovery.
- IC's weak 15D global excess vs strong single-parameter snow gradients is a measurement /
  aggregation property (15D RMS is a conservative all-parameter summary; dimensions without
  systematic directional signal dilute a few stable signatures); parameter-specific evidence
  is presented separately in Figure 4. No claim that the remaining parameters are "pure
  noise" unless the data explicitly supports it.

## 8. Outputs (all new files; frozen Base–CN files untouched)

- `r2_tgd2_specificity_basin_level.csv` — basin-level three-contrast within/between/excess
- `r2_tgd2_specificity_summary.csv` — regime summaries (Full531, S1–S5, ExcludeS5)
- `r2_tgd2_specificity_regressions.csv` — full / exclude-S5 regressions per contrast
- `r2_tgd2_slope_difference_summary.csv` — paired-bootstrap delta_beta (IC/dPL × Full/Excl)
- `r2_tgd2_specificity_interpretation_report.md` — interpretation applied from §4
- `r2_tgd2_specificity_analysis_plan.md` — this file
