# R3 Within-IC Information-Profile Reference

## Verdict

A within-IC calibration reference was constructed from archived ten-start IC parameter vectors without training or recalibration. It is a **within-IC calibration reference**, not an upper bound, ceiling, or theoretical maximum.

The frozen performance-comparable rule is:

```text
KGE_best - KGE_restart <= 0.01
```

Fitness direction was verified as maximize: the archived benchmark uses `argmax` and KGE. For each basin, the canonical IC vector is the archived best restart; the alternative is the highest-fitness eligible restart after excluding that canonical start. No selection uses parameter distance, profile similarity, or any R3 outcome.

## Restart support

`tables/R3_IC_SELF_RESTART_SUPPORT.csv` contains all 36 × 531 rows. Each self profile uses only basins with at least two eligible starts, so the canonical start and an independent alternative both exist.

| Quantity | Value |
|---|---:|
| Models | 36 |
| Basins/model | 531 |
| Starts/basin | 10 |
| Total model–basin rows | 19116 |
| Rows with ≥2 eligible starts | 17420 |
| Rows with only one eligible start | 1696 |
| Canonical-to-best max absolute normalized-u error | 1.110e-16 |
| Models with ≥400 self-eligible basins | 35/36 |
| R2 strict model list | 23 |

The common basin support is model-specific `B_m^self`; the canonical-versus-dPL matched comparison uses the same `B_m^self` for that model.

## Representation and estimands

Both IC profiles use the frozen R3 20-D information-cluster scores at threshold 0.70. No raw 35-attribute or R4 13-cluster representation is used. For each model:

```text
rho_IC_A[p,k] = Spearman_b(IC canonical u[b,p], information[b,k])
rho_IC_B[p,k] = Spearman_b(IC alternative u[b,p], information[b,k])
rho_dPL[p,k]  = Spearman_b(dPL canonical u[b,p], information[b,k])
R_self[p]      = Spearman_k(rho_IC_A[p,k], rho_IC_B[p,k])
R_cross[p]     = Spearman_k(rho_IC_A[p,k], rho_dPL[p,k])
```

`R_model` is the median across parameters and the ensemble value is the median across equally weighted eligible models. `C` and `A_diag` use the exact existing R3 row/off-diagonal definitions.

## Model sets

- **All-model available set:** models with at least 400 common-support basins and estimable profile values; R and A denominators are reported separately because `collie1` has no off-diagonal A contrast.
- **R2-matched strict set:** the frozen 23-model R2 primary list, intersected with the same profile/support rules; no new model threshold was invented.
- Model-by-model coverage and estimability flags are in `tables/R3_F6_IC_SELF_REFERENCE_MODEL.csv`.

## Results

| Population | Estimand | N models | Median self | Median cross matched | Self − cross | 95% model-bootstrap CI | Positive model differences | Median basin support |
|---|---|---:|---:|---:|---:|---|---:|---:|
| all_model_available | R_paired_IC_self | 35 | 0.967669 | 0.731579 | 0.217293 | [0.156391, 0.284211] | 35/35 | 496.0 |
| all_model_available | A_diag_IC_self | 34 | 0.981203 | 0.603759 | 0.339474 | [0.202256, 0.428571] | 31/34 | 490.5 |
| r2_strict_23 | R_paired_IC_self | 23 | 0.983459 | 0.742105 | 0.216541 | [0.127820, 0.253383] | 23/23 | 511.0 |
| r2_strict_23 | A_diag_IC_self | 22 | 1.026316 | 0.701504 | 0.285338 | [0.121053, 0.405263] | 20/22 | 510.5 |

Bootstrap uncertainty resamples eligible models with replacement and preserves the paired self/cross difference within each model (`5000` draws, seed `20260910`). It is a model-level calibration-scale uncertainty, not a cell-independence significance test.

## Interpretation boundary

The result is reported as observed, without presupposing self > cross. The within-IC reference quantifies repeatability under the archived performance-comparable restart rule. It does not prove parameter identity, a physical upper bound, causality, or that any cross/self difference is exclusively caused by the calibration paradigm.

## Provenance

- Restart source: `results/ic_dpl_aligned_full300_20260819_final/best_training/*/chunk_*_best.pt`.
- Loader: `analysis/seenbasin_remaining_20260901/common.py::load_ic_restart`.
- Restart gate: `results/seenbasin_remaining_analysis_20260901/agent_C/C05_RESTART_DATA_AVAILABILITY_GATE.csv`.
- Frozen aligned R3 basin arrays: `results/joh_direct_parameter_change_diagnostic_20260905/r3/cache/model_arrays/*.npz`.
- Frozen source audit: `results/joh_direct_parameter_change_diagnostic_20260905/r3/R3_PROVENANCE_AUDIT.md` and `RUN_MANIFEST.json`.
- Information representation: archived `info_scores` arrays, threshold-0.70 20-D PC1 scores.
- Reproduction check: 10840 frozen information-space rho entries, maximum absolute error `5.000e-11`.
- Figure-ready profiles: `tables/R3_IC_SELF_PROFILE_VALUES_LONG.csv` and per-model `cache/R3_IC_SELF_PROFILE_REFERENCE/*.npz`.
