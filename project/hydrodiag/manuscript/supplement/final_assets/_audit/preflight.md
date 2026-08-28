# Supplement asset preflight

- Generated: 2026-08-27
- Project root: /home/jingxin/code/dmg-research/project/hydrodiag
- Git HEAD: 322fb932c922a131a67800bcbc6aa7eb704c7605
- Python executable: /home/jingxin/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10
- Python version: Python 3.10.20
- CPU logical count: 4
- RAM: 8133228 kB
- Thread limits: OMP_NUM_THREADS=2, MKL_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2, NUMEXPR_NUM_THREADS=2

## Git status summary

- Modified/staged paths: 42
- Untracked paths (all): 6728

The complete status was collected with git status --short --untracked-files=all; generated cache paths are summarized below to keep this audit readable.

### Modified or staged paths

```text
 M ../../.gitignore
 M ../../dmotpy/models/core/penman.py
 M ../../dmotpy/models/core/sacramento.py
 M ../../dmotpy/models/core/smar.py
 M ../../dmotpy/models/core/tcm.py
 M ../../dmotpy/models/core/topmodel.py
 M ../../dmotpy/models/core/vic.py
 M ../../dmotpy/models/hydrology_model.py
 M ../benchmark/dpl/attributes.py
 M ../benchmark/dpl/train_dpl.py
 M ../benchmark/scripts/evaluate_benchmark_metrics.py
 M ../benchmark/scripts/evaluate_ic_aligned_gen300.py
 M ../benchmark/scripts/freeze_model_version.py
 M ../benchmark/scripts/run_36model_benchmark.py
 M ../benchmark/scripts/run_dpl_benchmark_dmg_native.py
 M ../benchmark/src/batched_cmaes.py
 M ../benchmark/src/checkpoint_guard.py
 M ../benchmark/src/data_selection.py
 M ../benchmark/src/model_adapter.py
 M ../benchmark/src/model_registry.py
 M ../benchmark/src/objective.py
 M ../benchmark/src/production_config.py
 M manuscript/figure_manifests/canonical_assets.json
 M manuscript/scripts/r1/generate_table1.py
 M manuscript/scripts/r1/generate_table_s1.py
 M manuscript/scripts/r1/plot_r1_figure1.py
 M manuscript/scripts/r1/plot_r1_figure2.py
 M manuscript/scripts/r1/plot_r1_figure2_canonical.py
 M manuscript/scripts/r1/r1_daily_inference.py
 M manuscript/scripts/r2/generate_table_s4.py
 M manuscript/scripts/r2/plot_r2_figure3_final.py
 M manuscript/scripts/r2/plot_r2_figure4.py
 M manuscript/scripts/r2/plot_r2_figure4_canonical.py
 M manuscript/scripts/r3/generate_table_r3_main.py
 M manuscript/scripts/r3/plot_figure5.py
 M manuscript/scripts/r3/plot_figure6.py
 M manuscript/scripts/r3/prepare_figure5_data.py
 M manuscript/scripts/r3/prepare_figure6_data.py
 M manuscript/scripts/r4/export_all_tgd2_states.py
 M manuscript/scripts/r4/plot_r4_figure7.py
 M manuscript/scripts/r4/plot_r4_figure8.py
 M manuscript/scripts/shared/r1_plot_style.py
```

### Untracked paths by top-level project area

```text
      1 ../audit_scout.md
      1 benchmark/configs
      3 benchmark/dpl
      1 benchmark/frozen_versions
      2 benchmark/scripts
      1 benchmark/src
      1 ../context.md
    135 manuscript/cache
      2 manuscript/figures
      1 "manuscript/scripts
     31 manuscript/scripts
      4 manuscript/tables
   6544 ../release
      1 ../reviewer2_current_evidence_audit.md
```

### Relevant untracked Supplement / robustness paths

```text
?? manuscript/scripts/supplement/plot_alt_generating_field_robustness.py
?? manuscript/scripts/supplement/plot_huc2_loro_robustness.py
?? manuscript/scripts/supplement/plot_tgd_response_sensitivity.py
```

## Current Supplement assets

```text
FigureS1_caption_notes.md
FigureS1_provenance.md
FigureS2_caption_notes.md
FigureS2_provenance.md
FigureS3_alt_generating_field_provenance.md
FigureS3_caption_notes.md
FigureS4_caption_notes.md
FigureS4_tgd_response_shape_sensitivity_provenance.md
FigureS5_caption_notes.md
FigureS5_huc2_loro_provenance.md
figures/FigureS1_R4_multibasin_validation.png
figures/FigureS1_R4_population_audit.csv
figures/FigureS1_R4_selection_audit.json
figures/FigureS2_R3_seasonal_trajectories.png
figures/FigureS3_alt_generating_field_robustness.png
figures/FigureS4_tgd_response_shape_sensitivity.png
figures/FigureS5_huc2_loro_robustness.png
final_assets/_audit/preflight.md
results/s2_parameter_bounds_from_code.csv
supplement_asset_inventory.md
supplement_asset_plan.md
supplement_execution_report.md
supplement_final_numbering_map.md
tables/TableS1_parameter_bounds.csv
tables/TableS1_parameter_bounds.md
tables/TableS1_parameter_bounds.tex
tables/TableS2_sensitivity_audits.md
tables/TableS2_sensitivity_audits.tex
```

## Current plotting scripts

```text
r0/plot_r0_figure10.py
r1/build_r1_statistics.py
r1/generate_table1.py
r1/generate_table_s1.py
r1/generate_table_s2.py
r1/generate_table_s3.py
r1/plot_r13_root_cause_diagnostics.py
r1/plot_r14_feasibility_diagnostics.py
r1/plot_r1_figure1.py
r1/plot_r1_figure2_canonical.py
r1/plot_r1_figure2 copy.py
r1/plot_r1_figure2.py
r1/r1_daily_inference.py
r1/r1_metrics.py
r1/r1_statistics.py
r1/rebuild_r1_statistics_staged.py
r1/rebuild_r1_statistics_streaming.py
r1/run_r1_missing_dpl_gpu.py
r2/compute_r2_tgd2_parameter_gradients.py
r2/generate_table_s4.py
r2/generate_table_s5.py
r2/plot_r2_fig_s5_tgd2_matched_control.py
r2/plot_r2_fig_s6_tgd2_parallel.py
r2/plot_r2_figure3_final.py
r2/plot_r2_figure4_canonical.py
r2/plot_r2_figure4.py
r2/run_r2_parameter_statistics.py
r2/run_r2_robustness_checks.py
r2/run_r2_tgd2_specificity.py
r2/run_r2_within_structure_baseline.py
r3/analyze_pilot.py
r3/common.py
r3/docs/estimand_audit/derive_estimand_audit.py
r3/docs/estimand_audit/derive_param_truth_error.py
r3/export_figure6_process_data.py
r3/gate_analysis.py
r3/gate_report_md.py
r3/generate_table_r3_main.py
r3/generate_table_r3_si.py
r3/generate_truth.py
r3/__init__.py
r3/launch_d2_parallel.py
r3/misspec_analysis.py
r3/misspec_states.py
r3/oracle_dpl_audit.py
r3/oracle_identity.py
r3/pilot.py
r3/plot_figure5.py
r3/plot_figure6.py
r3/plot_r3_si_components.py
r3/plot_r3_si_seasonal_trajectories.py
r3/posthoc_stats.py
r3/posthoc_validation.py
r3/prepare_figure5_data.py
r3/prepare_figure6_data.py
r3/recorded_forward.py
r3/run_base_no_refit.py
r3/run_gate_531.py
r3/truth_generator.py
r4/audit_huc02_from_daymet.py
r4/audit_tgd_dpl_seed_failures.py
r4/build_complete_three_structure_r4.py
r4/build_figure_s1_population_audit.py
r4/build_r4_soil_statistics.py
r4/common.py
r4/export_all_tgd2_states.py
r4/extract_caravan_soil.py
r4/forward_export.py
r4/generate_table_r4.py
r4/generate_three_structure_r4_all.py
r4/__init__.py
r4/input_adapters.py
r4/phase1_dpl_analysis.py
r4/phase1_ic_fused_analysis.py
r4/plot_r4_figure4.py
r4/plot_r4_figure7.py
r4/plot_r4_figure8_canonical.py
r4/plot_r4_figure8.py
r4/plot_r4_figure_s1_multibasin.py
r4/plot_r4_figure_s6.py
r4/rebuild_figure7_canonical.py
r4/robustness_analysis.py
r4/smoke_test.py
r4/snow_reference.py
r4/soil_analysis.py
r4/state_export.py
r5/build_r5_formal_analysis.py
r5/plot_r5_figure9.py
r5/prepare_r5_figure9_canonical.py
shared/build_discussion_readiness_audit.py
shared/build_local_draft_results_audit.py
shared/build_results_freeze_audit.py
shared/canonical_assets.py
shared/generate_table1_structural_configurations.py
shared/generate_table2_controlled_recovery.py
shared/generate_table_s1_parameter_bounds.py
shared/generate_table_s2_sensitivity.py
shared/r1_plot_style.py
supplement/plot_alt_generating_field_robustness.py
supplement/plot_huc2_loro_robustness.py
supplement/plot_tgd_response_sensitivity.py
```

## Reviewer-2 robustness inputs

```text
alt_generating_field/alt_generating_field_basin_seedmedian.csv
alt_generating_field/alt_generating_field_basin_table.csv
alt_generating_field/alt_generating_field_report.md
alt_generating_field/alt_generating_field_summary.json
alt_generating_field/q_star_alt.npz
alt_generating_field/theta_star_alt.npz
logs/preflight.md
p0_reporting/dpl_valid_n_resolution.md
p0_reporting/invalid_denominator_strata_breakdown.csv
p0_reporting/recovery_denominator_tail_audit.csv
p0_reporting/recovery_denominator_tail_audit.md
regional_loro/r1_huc2_loro.csv
regional_loro/r3_huc2_loro.csv
regional_loro/r5_huc2_loro.csv
regional_loro/regional_loro_forest.png
regional_loro/regional_loro_summary.md
REVIEWER2_ROBUSTNESS_FINAL_REPORT.md
summaries/canonical_registry.csv
tgd_response/tgd_response_curves.png
tgd_response/tgd_response_data.csv
tgd_response/tgd_response_summary.md
tgd_shape_sensitivity/tgd_shape_sensitivity_basin_metrics.csv
tgd_shape_sensitivity/tgd_shape_sensitivity_report.md
tgd_shape_sensitivity/tgd_shape_sensitivity_summary.json
```

## Candidate canonical R1-R5 result directories

```text
manuscript/results/R1
manuscript/results/R2
manuscript/results/R3
manuscript/results/R3/fig6_seasonal
manuscript/results/R5
results/r3_base_no_refit_v1
results/r3_gate_dpl_xaj_cn_localcheck_seed_42
results/r3_gate_dpl_xaj_cn_seed_123
results/r3_gate_dpl_xaj_cn_seed_2026
results/r3_gate_dpl_xaj_cn_seed_42
results/r3_gate_ic_xaj_cn_531_v1
results/r3_gate_ic_xaj_cn_531_v1/checkpoints
results/r3_gate_ic_xaj_cn_531_v1/raw
results/r3_gate_ic_xaj_cn_531_v1/raw/xaj_cn
results/r3_gate_ic_xaj_cn_531_v1/summaries
results/r3_gate_v1
results/r3_misspec_analysis_v1
results/r3_misspec_dpl_xaj_seed_123
results/r3_misspec_dpl_xaj_seed_2026
results/r3_misspec_dpl_xaj_seed_42
results/r3_misspec_dpl_xaj_tgd2_seed_123
results/r3_misspec_dpl_xaj_tgd2_seed_2026
results/r3_misspec_dpl_xaj_tgd2_seed_42
results/r3_misspec_ic_xaj_531_v1
results/r3_misspec_ic_xaj_531_v1/checkpoints
results/r3_misspec_ic_xaj_531_v1/raw
results/r3_misspec_ic_xaj_531_v1/raw/xaj
results/r3_misspec_ic_xaj_531_v1/summaries
results/r3_misspec_ic_xaj_tgd2_531_v1
results/r3_misspec_ic_xaj_tgd2_531_v1/checkpoints
results/r3_misspec_ic_xaj_tgd2_531_v1/raw
results/r3_misspec_ic_xaj_tgd2_531_v1/raw/xaj_tgd2
results/r3_misspec_ic_xaj_tgd2_531_v1/summaries
results/r3_runlogs
results/r3_synthetic_truth_v1
results/r4_caravan_soil_reference_v1
results/r4_ic_fused_XAJ
results/r4_ic_fused_XAJ_CN
results/r4_ic_fused_XAJ_TGD2
results/r4_official_dpl_XAJ_CN_seed123
results/r4_official_dpl_XAJ_CN_seed42
results/r4_official_dpl_XAJ_seed123
results/r4_official_dpl_XAJ_seed42
results/r4_official_dpl_XAJ_TGD2_seed123
results/r4_official_dpl_XAJ_TGD2_seed2026
results/r4_official_dpl_XAJ_TGD2_seed42
results/r4_phase1_dpl_official
results/r4_phase1_ic_fused_sensitivity
results/r4_phase1_soil_official
results/r4_phase1_soil_official/figure7_rebuilt_canonical
results/r4_phase1_soil_official/figure7_rebuilt_canonical/seed123_sensitivity
results/r4_phase1_soil_official/tgd_dpl_seed_failure_audit
results/r4_r4_smoke
results/r4_replay_dpl_XAJ_TGD2_seed42
results/r4_replay_dpl_XAJ_TGD2_seed42/f7_replay_stage__ki7jjqu
results/r4_replay_dpl_XAJ_TGD2_seed42/f7_replay_stage__ki7jjqu/arrays
results/r4_swe_reference_v1
results/reviewer2_robustness
results/reviewer2_robustness/alt_generating_field
results/reviewer2_robustness/dpl_no_fsnow
results/reviewer2_robustness/logs
results/reviewer2_robustness/p0_reporting
results/reviewer2_robustness/regional_loro
results/reviewer2_robustness/summaries
results/reviewer2_robustness/tgd_response
results/reviewer2_robustness/tgd_shape_sensitivity
```

## Preservation notes

- No reset, checkout, deletion, result overwrite, training, recalibration, state export, full evaluation, pytest, or unittest was run during preflight.
- The repository has substantial pre-existing modifications and generated caches; they remain untouched.
- manuscript/supplement/final_assets/ is the new submission-asset layer and will not replace original results.
