# Agent D — R2→R3 linkage hostile check and conditioned support

## Disposition

**PASS WITH LIMITATION.** This is a read-only audit of frozen R2/R3 products. It does not change the R3 primary estimator, rerun training/calibration/simulation, add a dPL seed, or inspect held-out targets.

## Question and scope

R2 measures raw parameter-value/rank reorganization. R3 measures the correspondence of each parameter's 20-dimensional catchment-information association profile. Agent D tests whether the R3 same-coordinate result can be reduced to raw parameter-rank continuity, and separately audits the pre-specified outlet-performance-conditioned supporting result.

The frozen universe is 36 models, 531 common basins/model, 271 model–parameter coordinates, and 20 threshold-0.70 information dimensions. Model-level summaries are the inferential units; coordinate and pair rows are descriptive. Canonical dPL is seed 42 only.

## Frozen R2↔R3 rank linkage

The first two rows below are coordinate-level comparisons across the 271 model–parameter coordinates; the next two are full parameter-pair comparisons with model labels retained. Model-centered values are descriptive centering within model, and the within-model values are medians of model-level correlations (the frozen tables have 35 finite model summaries for this linkage statistic).

| comparison | pooled Spearman | model-centered | median within-model | source |
|---|---:|---:|---:|---|
| `R_rank` ↔ `R_info` | 0.743830 | 0.683142 | 0.678571 | `r2_parameter_axis_audit_20260906/agent_D_r2_r3_rank_linkage/tables/correlation_summaries.csv` |
| `Q_rank` ↔ `Q_info` | 0.863927 | 0.860931 | 0.859901 | same |

Thus raw rank continuity is genuinely associated with profile correspondence. It must not be denied or silently treated as noise removal.

## Rank-matched residual same-coordinate test

The frozen primary comparator is within model: each diagonal `(p,p)` is compared with the three nearest eligible off-diagonal alternatives `(p,q)/(q,p)` by absolute `Q_rank` distance, with `k=3` and **no caliper**. The one-parameter model is structurally unmatched; no row-level independence claim is made.

- Matched coverage: `270/271 = 0.996310`.
- Residual estimand: `A_info|rank = Q_info(diagonal) - mean(Q_info(matched off-diagonals))`.
- Model-equal observed median: `0.295238`.
- Model-clustered 95% CI: `[0.254386, 0.429073]`.
- Positive fraction of matched model summaries: `1.000000`.
- Sign-flip p-value: `0.000400`.
- The frozen R3 bridge values remain distinct: `R_paired≈0.715789` is profile retention, whereas `A_diag≈0.615038` is the unconditioned diagonal-vs-off-diagonal specificity contrast.

The residual advantage remains after matching on raw rank similarity. The frozen hostile verdict is:

> **R3 INFORMATION SPECIFICITY REMAINS BEYOND RANK CONTINUITY**

This does **not** establish that dPL removes noisy ranking, preserves informative ranking, or causes information retention. It says only that same-coordinate profile similarity exceeds typical raw-rank-matched alternatives under the declared descriptive comparison.

### Matching sensitivities

The source table retains the frozen `k=1`, `k=3`, and `k=5` matching sensitivities and pre-specified rank-distance calipers (`0.025`, `0.05`, `0.10`). The primary remains `k=3`, no caliper; these sensitivities are not substituted for it.

## Performance-conditioned supporting result

The primary conditioning variable is `abs(DeltaKGE)`, ordered **within each model** into equal-size low/middle/high strata (`177` basins per stratum). The model-equal high-minus-low contrasts are:

| metric | observed | bootstrap 95% CI | bootstrap draws | source |
|---|---:|---:|---:|---|
| profile divergence (`median abs rho`) | `+0.028033` | `[0.017163, 0.043610]` | 1,000 model resamples | `r3/tables/r3_bootstrap_summary.csv` |
| paired profile correspondence | `-0.141353` | `[-0.215038, -0.078947]` | 1,000 model resamples | same |

The balanced within-model equal-size random-subset null uses `36,000` model draws (`1,000` replicates × 36 models), seed `20261005`, and gives one-sided exceedance `p=0.000999` for the high-minus-low divergence contrast. The null is a sampling/composition control, not a causal test.

The direction is consistent with larger outlet-performance differences being associated with greater profile divergence and lower paired profile correspondence. Basin composition can contribute to this association; the result does not show that performance differences cause information-profile divergence.

`D_theta` conditioning is secondary only. It uses parameter displacement in the same parameter realizations that define profile changes, so its direction is subject to mathematical coupling. It is a bridge/sensitivity, not an independent causal parameter-to-information test. The source reports the secondary direction but it is not the R3 primary conditioning estimand.

## Primary, supporting, and rejected boundaries

### Primary evidence

1. The frozen R3 profile correspondence result (`R_paired≈0.715789`) at model-equal profile level.
2. The rank-matched residual same-coordinate result (`A_info|rank≈0.295238`, CI `[0.254386,0.429073]`, sign-flip `p≈0.000400`) as the hostile linkage protection.

### Supporting evidence

- Rank-linkage correlations (`R_rank↔R_info` and `Q_rank↔Q_info`) show that raw rank continuity accompanies part of profile correspondence.
- `abs(DeltaKGE)` equal-stratum contrasts show a performance-conditioned association with profile divergence.
- `D_theta` conditioning is secondary and mathematically coupled.

### Rejected or unavailable interpretations

- No claim that dPL removes noise or preserves informative ranking.
- No claim that `R_paired` proves parameter identity, physical meaning, or functional-role preservation.
- No causal interpretation of the `abs(DeltaKGE)` conditioning result.
- No causal parameter-to-information pathway from `D_theta` conditioning.
- No dPL multi-seed ensemble; symmetric paradigm variability is unavailable.
- No claim that IC is truth, dPL is physically superior, or parameter changes represent compensation.
- No naïve significance test treating 271 coordinates, parameter pairs, or basin rows as independent replicates.

## Provenance and hashes

The authoritative linkage table and uncertainty table are:

- `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_parameter_axis_audit_20260906/agent_D_r2_r3_rank_linkage/tables/correlation_summaries.csv` — SHA256 `0ef3b1560899e7326dc2da2e734f50364bfb411e0d055f65328f17351cc148a1`
- `.../tables/uncertainty_summary.csv` — SHA256 `fd31c1e86044186daf13d77deea30439bb0a949a6b5733d653354509f76e3c50`
- `.../tables/model_matching_summaries.csv` — SHA256 `b54a8c778ddc1c1437f6bbd8001bd5e3f61e131ea0e76799a570fb70405c97cd`
- Frozen R3 metric contract: `.../agent_D_r2_r3_rank_linkage/r3_metric_contract.md` — source hashes include the unchanged R3 implementation `ba52604b0d7d938896b8d333151e26d5786a99f664b8839ab547874672d3ce69` and bridge `ab591130b4e5177f3a800115b9326e74cb883e1824535bc9e203ae34b2c0b4ed`.

The conditioned supporting inputs are:

- `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r3/tables/conditioned_association_summary.csv` — SHA256 `715de19791fba4f1baa9233dc6d43f210b2c1ac2884c95b70bb8ad517fe4c190`
- `.../r3/tables/r3_bootstrap_summary.csv` — SHA256 `a1a2be7f4795e53d7f576237c4face5c227f8f0e2514aabfd434462df926d9a4`
- `.../r3/tables/random_subset_null.csv` — SHA256 `860177ff86c0895b142022765928f3390ba3cd701542b8522ae7f59ff0ba546c`
- Functional-role boundary (not a positive D finding): `project/benchmark/results/joh_functional_role_diagnostic_20260905/tables/03_INFORMATION_ROLE_AUDIT.csv` — SHA256 `b8d4bf2d17b200702dc45c9145cd6f964604fcd8cb777928f6ccd864554c3645`; HESS-prior negative test `.../tables/22_HESS_PRIOR_ROLE_TEST.csv` — SHA256 `9c68316fb9761d860641952f32321ba9ce356585d69971e38903b53e8e77f4d4`.

## Final Agent D statement

Raw rank continuity statistically accompanies part of the R3 correspondence, but the rank-matched same-coordinate information advantage remains positive and model-consistent. The performance-conditioned result is a supporting association with composition and mathematical-coupling limitations. The evidence supports a narrow **coordinate-specific information organization** claim, not rank denoising, physical information retention, functional-role preservation, or causality.
