# R2 Figure Data Audit

## SUPERSEDED / HISTORICAL NOTICE

> This audit predates the exact-common-support F3 branch. It is retained for provenance only. Its Figure 3a BLOCKED / Figure 3c PARTIAL statuses and the raw 0.346893 / 0.573874 comparison are superseded by the exact-common-support artifacts listed in `tables/R2_F3_ARTIFACT_SYNCHRONIZATION.md`. Do not use this file as the active manuscript-readiness gate.

## 1. Executive verdict

- **Figure 2 2a — READY**
- **Figure 2 2b — READY**
- **Figure 2 2c — READY**
- **Figure 2 2d — READY**
- **Figure 3 3a — BLOCKED**
- **Figure 3 3b — READY**
- **Figure 3 3c — PARTIAL**
- **Figure 3 3d — READY**

Maximum blockers: Figure 3a has exact basin-wise squared contribution data, but the requested model-level top-1/second/remaining composition is not a frozen estimand; Figure 3c has a raw-matched aggregation conflict (.346893/.573874 versus exact-overlap .345217/.571177).

Overall plotting gate: **NOT READY FOR PLOTTING** until the Figure 3a aggregation and Figure 3c raw-matched definition are resolved. No new experiment is required for these decisions.

## 2. Canonical source map

The canonical source map is the unique source assignment in `tables/R2_FIGURE_DATA_MANIFEST.csv`; source hashes are also retained in `tables/R2_SOURCE_PROVENANCE.json`.

- Fig 2a D_theta: direct frozen normalized basin table `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2/tables/10_BASIN_PARAMETER_VECTOR_DISPLACEMENT.csv`, cross-checked against the pinned IC-self/model summary and `11_MODEL_LEVEL_PARAMETER_DISPLACEMENT.csv`.
- Fig 2b: pinned `MODEL_IC_SELF_SUMMARY.csv` and `BOOTSTRAP_SUMMARY.csv`; generator `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/claim_audit_multiaudit_20260905/agent_B/ic_self_benchmark.py`.
- Fig 2c: `R_rank_all_271.csv` plus occupancy QC; model-equal headline from `R_rank_model_equal_summaries.csv`.
- Fig 2d: `model_rank_self_primary_5000.csv`; corrected strict primary rank benchmark.
- Fig 3a/b: `basin_coordinate_weights.csv` and the exact frozen coordinate-concentration summaries.
- Fig 3c: coordinate-specific adjusted output plus matched raw comparison from the final coordinate IC-self audit.
- Fig 3d: corrected 23-model contraction branch only.

UNRESOLVED PROVENANCE: none for the existing frozen quantities. The only unresolved item is the proposed Fig 3a model-level composition definition, not a file identity.

## 3. Figure 2 data readiness

- **2a:** 19,116 rows = 36 models x 531 basins; D_RMS is computed as RMS over same-model normalized coordinates. The source script uses the frozen [0,1] normalized-u arrays, canonical IC best archived restart, and dPL seed 42. Model medians reproduce the expected range `0.150576–0.592668` and headline median `0.384375`.
- **2b:** 36 model rows plus a bootstrap summary. IC-self is the median distance to eligible non-canonical archived IC restarts within training-KGE tolerance 0.01, with canonical restart excluded. The source reports 1,000 paired basin resamples, seed 20261005, and model-equal aggregation.
- **2c:** 271 coordinate rows across 36 models. R_rank is tie-corrected Spearman over the same 531 basins, coordinate by coordinate. Constant/tie/boundary diagnostics exist; no coordinate is silently relabeled. `reach 0.5/0.8/0.9` means the model-level median R_rank summary is at least that threshold, not a pooled coordinate percentage.
- **2d:** strict rank benchmark is exactly 23 models at the 0.90 primary coverage gate. The model-level paired R_cross/R_self table exists. Formal full-ensemble verdict remains `INCONCLUSIVE`, not a positive universal rank-preservation claim.

## 4. Figure 3 data readiness

- **3a:** `basin_coordinate_weights.csv` has the exact source object: `weight = DeltaTheta^2 / sum_p(DeltaTheta^2)`, with squared displacement and basin IDs. Existing code ranks coordinates within each basin and reports cumulative basin-wise top-k shares. It does not define one model-level total vector composition with a unique top coordinate across basins. Therefore the requested hero composition is BLOCKED pending an explicit aggregation choice; the raw data are not missing.
- **3b:** C_eff uses `N_eff = 1/sum(w^2)` and `C_eff=N_eff/P`; top1/top2 are cumulative shares from basin-wise sorted weights, followed by within-model medians and equal model weighting. 36-model and paired-basin bootstrap objects exist.
- **3c:** exact adjustment is coordinate-wise `X=abs(dPL-IC)`, `S_med=median(abs(IC_restart-IC))` over eligible non-canonical restarts, `E=X-S_med`, and `E_plus=max(E,0)`. The adjusted primary is 23 models. The frozen raw .346893/.573874 values reproduce from all-basins raw summaries restricted to those 23 models, whereas the exact adjusted-valid basin-overlap table gives .345217/.571177; this is a real aggregation/provenance conflict and must be resolved before plotting the raw-to-adjusted comparison.
- **3d:** CR is `IQR(reference field)/IQR(IC)` coordinate-wise. Canonical uses dPL, consensus uses basin-wise median eligible non-canonical restart fields, and IC-self uses the frozen synthetic restart fields. Paired 23-model rows are present for all three references.

## 5. Strict subset audit

- rank strict N = 23
- localization strict N = 23
- intersection N = 23
- symmetric difference = []
- rank strict IDs = `alpine1, alpine2, collie1, collie2, collie3, flexi, flexis, gr4j, hillslope, hymod, ihacres, modhydrolog, mopex1, mopex2, mopex3, newzealand1, simhyd, susannah1, tank, us1, vic, wetland, xinanjiang`
- localization strict IDs = `alpine1, alpine2, collie1, collie2, collie3, flexi, flexis, gr4j, hillslope, hymod, ihacres, modhydrolog, mopex1, mopex2, mopex3, newzealand1, simhyd, susannah1, tank, us1, vic, wetland, xinanjiang`

The two strict subsets are identical in the current frozen outputs; they must nevertheless remain separately named in figure code (`strict_rank_subset` and `strict_localization_subset`) rather than being aliased as an unexplained `strict23`.

Excluded 13 models and reasons:

- `australia` — primary coverage 0.888889 < 0.90; insufficient archived non-canonical IC restart reference
- `flexb` — primary coverage 0.860640 < 0.90; insufficient archived non-canonical IC restart reference
- `gsfb` — primary coverage 0.839925 < 0.90; insufficient archived non-canonical IC restart reference
- `hbv96` — primary coverage 0.822976 < 0.90; insufficient archived non-canonical IC restart reference
- `mopex4` — primary coverage 0.890772 < 0.90; insufficient archived non-canonical IC restart reference
- `mopex5` — primary coverage 0.871940 < 0.90; insufficient archived non-canonical IC restart reference
- `newzealand2` — primary coverage 0.672316 < 0.90; insufficient archived non-canonical IC restart reference
- `penman` — primary coverage 0.779661 < 0.90; insufficient archived non-canonical IC restart reference
- `plateau` — primary coverage 0.836158 < 0.90; insufficient archived non-canonical IC restart reference
- `smar` — primary coverage 0.807910 < 0.90; insufficient archived non-canonical IC restart reference
- `susannah2` — primary coverage 0.813559 < 0.90; insufficient archived non-canonical IC restart reference
- `tcm` — primary coverage 0.877589 < 0.90; insufficient archived non-canonical IC restart reference
- `topmodel` — primary coverage 0.868173 < 0.90; insufficient archived non-canonical IC restart reference

## 6. Headline reproduction

`tables/R2_HEADLINE_REPRO_CHECK.csv` contains 52 checks. PASS rows use absolute tolerance 1e-6; bootstrap rows retain the recovered source seed and resampling description. Failing rows: 2.

                                                     metric  expected_value  recomputed_value  absolute_difference  relative_difference                                                                                                                                                                           source                                                                   aggregation status                                                                                                                                           notes
fig3c_raw_matched_exact_basin_C_eff_against_frozen_headline        0.346893          0.345217             0.001676             0.004832 project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/raw_vs_adjusted_comparison.csv median of 23 model summaries recomputed on exact adjusted-valid basin overlap   FAIL FAIL is substantive: the frozen .346893 headline is not reproduced by the exact basin-matched table.; check_type=deterministic; tolerance=1e-06
 fig3c_raw_matched_exact_basin_top1_against_frozen_headline        0.573874          0.571177             0.002697             0.004700 project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/raw_vs_adjusted_comparison.csv median of 23 model summaries recomputed on exact adjusted-valid basin overlap   FAIL FAIL is substantive: the frozen .573874 headline is not reproduced by the exact basin-matched table.; check_type=deterministic; tolerance=1e-06

The two failing headline checks are intentional audit findings, not rounding noise: the frozen raw .346893/.573874 reproduces from the 23-model subset of the all-basins raw table, but the exact adjusted-valid basin overlap table is .345217/.571177. The source code and cache therefore do not currently support calling those values both 'raw matched'.

Important definitions recovered from source code:

- Parameter normalization is physical-bound normalized `[0,1]`; IC values are verified against linear `(physical-lower)/(upper-lower)`, while dPL auto-log mappings remain canonical network-u coordinates. All 271 registry coordinates are included; no equal-bound fixed coordinates were found.
- IC is the best archived restart per model-basin, canonical restart excluded from self-reference, no fallback, with one shared restart index across parameters. dPL has seed 42 only; no dPL multi-seed result exists.
- IC-self separation bootstrap: 1,000 paired basin resamples, seed 20261005, basin resampling within model followed by model-equal medians.
- Rank bootstrap: 1,500 basin resamples, seed 20260906, coordinate rank recomputed within each sampled basin set, then model-equal median.
- Localization raw bootstrap: 1,000 paired basin resamples, seed 20260906, then model-equal median. Adjusted summaries use the final coordinate audit's fixed 1,000-draw model-summary bootstrap.
- Contraction robustness: fixed 5,000 synthetic restart fields with draw seed 20260913; corrected primary summaries use 23 models and bootstrap seed 42 for model summaries.

## 7. Stale / conflicting result files

- `results/.../r2_final_robustness_20260906/agent_B_contraction_robustness/SUPERSEDED_DO_NOT_USE.md`: parent Agent B outputs are superseded; only `corrected_primary_23models/` is canonical for Fig 3d.
- Older exploratory manuscript/r2 branches are recorded in `manuscript/r2/CLEANUP_REPORT.md` and must not be called by figure scripts.
- Direct diagnostic figures under `results/.../r2/figures/` are exploratory 180-dpi PNGs, not formal Figure 2/3 sources; this audit generated no figures.
- The dated direct R2 tables and manuscript/r2 frozen cache are related but not interchangeable; the cache input manifest/hash contract controls frozen summary claims.

The 36-model direct diagnostic is useful for basin-level D_theta and raw coordinate contributions, but it is not a license to replace the manuscript frozen 23-model strict summaries or corrected contraction branch. All figure caches written by the companion builder include source hashes to prevent silent mixing.

## 8. Minimum additional work

1. Freeze the Figure 3a aggregation: either use the existing basin-wise composition distribution (no new analysis) or explicitly document a model-level coordinate aggregation from the already available weights.
2. Resolve Figure 3c raw matched semantics: choose exact adjusted-valid basin overlap or 23-model all-basins raw values, document it, and do not update the frozen headline silently.
3. Use `build_r2_figure_cache.py` to regenerate plotting caches after any source-hash change; no training, CMA-ES, OOB/PUB/PUR, or hydrological simulation is needed.
4. Do not promote the optional bridge or the 23-model strict result to a full-ensemble causal or superiority claim.

## 9. Recommendation for plotting stage

- READY hero/support: Figure 2a, 2c, Figure 3b, 3d.
- READY but strict/inconclusive boundary: Figure 2b and 2d.
- PARTIAL pending aggregation/provenance resolution: Figure 3c.
- BLOCKED pending estimand wording: Figure 3a.
- Formal plotting condition: **not met** until Figure 3a and Figure 3c are resolved.

## Appendix: model-order candidates

`tables/R2_MODEL_ORDER_CANDIDATES.csv` preserves R1/registry order and displacement-sorted order without making a final layout decision. R1's source explicitly identifies its order as the canonical registry order.

## Appendix: scan and provenance

Candidate scan rows: 1557. Required source files: 24. Repository pre-existing changes were not modified.
No PDF/SVG/EPS or temporary figure was generated.
