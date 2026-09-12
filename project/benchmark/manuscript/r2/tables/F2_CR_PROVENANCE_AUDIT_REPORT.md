# F2 CR provenance audit report

## CR DATA VERDICT = STRICT23 ONLY

The provenance-valid primary CR dataset contains **23 models × 3 references = 69 values**. A full-36 primary ridgeline is not valid because 13 models fail the frozen primary IC restart-reference coverage gate. Their upstream diagnostics retain numeric coverage-incomplete rows, but those rows are explicitly marked `INSUFFICIENT_REFERENCE`; treating them as primary-valid would change the estimand. Therefore Phase B, if performed, must use the same strict 23-model set for all three ridgelines and must disclose `23/36`.

## Current 23-model table

- Cache: `manuscript/r2/cache/fig3d_contraction_reference.csv`
- Rows: `23`
- Current IDs match the primary eligible set: `True`
- Current IDs match strict rank set: `True`
- Current IDs match strict localization set: `True`
- Source script: `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/corrected_primary_23models.py`
- Source files: `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/matched_model_comparison_23.csv;project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/corrected_model_equal_summary.csv`
- Current cache vs upstream `model_summary.csv` maximum absolute difference across the 23 models and 3 references: `0.000e+00`.

## Full-36 availability interpretation

- Canonical inputs and canonical CR calculations are available for all 36 models.
- Consensus and IC-self primary fields are formally available for the 23 models passing the >=0.90 primary coverage gate.
- For the 13 excluded models, upstream `model_summary.csv` contains coverage-incomplete diagnostic values, but the formal status is `INSUFFICIENT_REFERENCE`; these cells are classified `STRICT_FILTER_ONLY`, not promoted to primary-valid values.
- The shared contract explicitly retains unavailable primary basins and prohibits canonical fallback.

Excluded 13 models: `australia, flexb, gsfb, hbv96, mopex4, mopex5, newzealand2, penman, plateau, smar, susannah2, tcm, topmodel`.

## Headline reproduction

The frozen headline values are strict23 quantities:

- canonical: `0.6140173622`
- consensus: `0.9794559561`
- IC-self: `1.0043244360`

The headline table records both these accepted strict23 values and the non-primary full36 diagnostic medians. The latter are not used for the final panel.

- **canonical:** strict23 median `0.6140173622`; coverage-incomplete full36 diagnostic median `0.5407368501`; difference `-0.0732805121`.
- **consensus:** strict23 median `0.9794559561`; coverage-incomplete full36 diagnostic median `0.9770873953`; difference `-0.0023685608`.
- **IC-self:** strict23 median `1.0043244360`; coverage-incomplete full36 diagnostic median `1.0040112735`; difference `-0.0003131625`.

## Reconstruction decision

No new full36 CR file was created. Although raw 36-model artifacts exist (`raw_replicate_arrays.npz` has 36 model axes and primary/all-valid fields), the 13 insufficient-reference models do not satisfy the frozen primary eligibility condition. The all-valid sensitivity pool is a different reference construction and cannot be silently substituted. The provenance-valid action is therefore a strict23 ridgeline, not a fabricated full36 ridgeline.

## Source artifacts and hashes

- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/qc.csv` — SHA256 `36481f3139e473c690fe98a26d32db5dfe3759cec6f24c0ebe8feee5d26d2b79`
- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/model_summary.csv` — SHA256 `2542e0eefb18e99f5cd39180e28e8fada36be83de43e5e35cb40baea6b675c97`
- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/raw_replicate_arrays.npz` — SHA256 `8d17569157304d7342ea4af8d3f193adbed7c2481cfbce033f9821db737df690`
- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/canonical_coordinate_summary.csv` — SHA256 `44f61e03c3f527607b312a8813630ca1f4f19b3957bd316c9b129078f85a4ff0`
- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/consensus_coordinate_summary.csv` — SHA256 `5547be0c31a886fc22eec8cb3266d406e27422bcec096b8a31ffb830064399fa`
- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/shared/IC_RESTART_REFERENCE_CONTRACT.md` — SHA256 `09a733f579411ced6142ef7c50cdd2be7bef35c89efd4aebd6c8b148e8ce62ce`
- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/shared/ic_restart_eligibility.csv` — SHA256 `60bd5bd2e80b0b982d94f34a5ad9c22c436668b199cf46ff7f3db8294fd31e2e`
- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/shared/ic_self_draw_plan.npz` — SHA256 `9281802d8022960f719a8f2c45b0a123d6b43db935120b0f97d85ea7508fc52a`
- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/corrected_primary_23models.py` — SHA256 `71a70acab6738aa2047c8fa1baae2321f215b05e80fd2a1321b60ce3826e931a`
- `results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/matched_model_comparison_23.csv` — SHA256 `e4dfd4bfb42bf7d459ac1cfedc394d73184befb98a15cba15502c0b9827e166a`

## Required Phase B disclosure

If redrawn, panel (c) must use the identical 23-model set in all three rows and include either the title suffix `(strict subset)` or an explicit `strict subset: 23/36 models` annotation/caption. It must use the three 23-value distributions, not the coverage-incomplete full36 diagnostic rows.
