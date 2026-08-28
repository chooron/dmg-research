# HESS Supplement asset inventory

**Audit scope.** Read-only inventory of the current `hydrodiag` manuscript and the
frozen Reviewer-2 robustness outputs. Existing user work under `manuscript/` was
preserved. The fixed project interpreter was
`/home/jingxin/code/dmg-research/.venv/bin/python`.

## Important discovery

The specifically named current Supplement planning manuscript containing
`S1 Data, model implementation, and parameter estimation`, `S2 Diagnostic
definitions and controlled experiments`, and `S3 Additional diagnostic evidence`
was not present in the current tree. The production audit references the absent
placeholders file `HESS_Supplementary_Methods_with_placeholders.md`; the closest
present document is `manuscript/methods_supplement_production_audit.md`. The
inventory below therefore treats the existing tables, figures, reports, scripts,
and the manuscript Results draft as the current evidence record rather than
assuming that an absent Supplement body exists.

## Asset map

| Asset | Scientific role | Existing? | Input data | Plot script | Current manuscript reference | Action |
|---|---|---:|---|---|---|---|
| Supplement planning body | S1–S3 methods/evidence organization | **No** | — | — | Referenced indirectly by `methods_supplement_production_audit.md` | Keep this audit/plan as an independent handoff; do not invent or merge a new Supplement body in this pass. |
| Table S1 | Parameter bounds, units, and structural applicability | **Yes** | `models/parameter_specs.py` and the generated table assets | `manuscript/scripts/shared/generate_table_s1_parameter_bounds.py` | `hess_results_R1_R5_reframed_v2.md` cites Table S1 for parameter definitions | Keep unchanged. Current canonical output is in `manuscript/supplement/tables/`; `canonical_assets.json` also declares a stale `manuscript/stats/tables/` location. |
| Table S2 Panel A | R1 KGE-screen and timing-threshold sensitivity | **Yes** | `manuscript/cache/r1_rebuild_audit_staged/r1_basin_level_ct.csv` | `manuscript/scripts/shared/generate_table_s2_sensitivity.py` | Results 3.1 / Figure 2 caption cite Table S2 Panel A | Keep the existing threshold grid; do not change the definition. |
| Table S2 Panel B | R3 denominator-cutoff sensitivity | **Yes** | `manuscript/results/discussion_audit/r3_denominator_sensitivity_audit.csv` and current R3 basin table | `manuscript/scripts/shared/generate_table_s2_sensitivity.py` | Results 3.3 and Table 2 refer to denominator sensitivity in Table S2 Panel B | Retain the cutoff grid. Add the frozen tail/invalid-stratum values as a compact note or small extension when the formal Supplement body is merged; no tail figure. |
| Figure S1 | R4 multi-catchment external-state corroboration across snow-burden terciles | **Yes; one numeric-label correction** | `results/r4_caravan_soil_reference_v1/caravan_soil_ensemble.npz`; `results/r4_swe_reference_v1/swe_ensemble.npz`; selection/audit CSV/JSON | `manuscript/scripts/r4/plot_r4_figure_s1_multibasin.py` | Results 3.4 refers to Fig. S1 | Re-rendered with the same layout and data after correcting the stale panel-g title from `N=443` to the audit-CSV count `N=442`; no new experiment. |
| R3 seasonal trajectory figure | Separates seasonal outlet/input and internal-storage evidence from main Figure 6 | **Yes** | `manuscript/results/R3/fig6_seasonal/fig6_seasonal_input.npz`, `fig6_seasonal_state.npz`, `fig6_seasonal_meta.json`, and `manuscript/results/R3/figure6_summary.json` | `manuscript/scripts/r3/plot_r3_si_seasonal_trajectories.py` | Results 3.3 cites Fig. S6; current source filename is `manuscript/figures/Fig_S6_R3_seasonal_trajectories.png` | Reuse unchanged and stage as final Figure S2 copy. The source file is not deleted. |
| Figure S3: alternative generating field | Tests whether the Base-refit/TGD recovery ordering depends on PCA/Ridge smoothing | **Newly rendered** | Canonical `manuscript/results/R3/figure5_basin_seedmedian.csv`; alternative `results/reviewer2_robustness/alt_generating_field/alt_generating_field_basin_seedmedian.csv` and summary JSON | `manuscript/scripts/supplement/plot_alt_generating_field_robustness.py` | Reviewer-2 report calls this alternative-field evidence; no finalized figure number | Include as a compact two-panel figure. Raw gains use all 531 basins; normalized fractions use `D_b > 10^-6` and are not clipped. |
| Figure S4: TGD response and shape sensitivity | Shows the continuous TGD response and the empirical effect of changing `T_ref`/`s_T` | **Newly rendered** | `results/reviewer2_robustness/tgd_response/tgd_response_data.csv`; `results/reviewer2_robustness/tgd_shape_sensitivity/tgd_shape_sensitivity_basin_metrics.csv` | `manuscript/scripts/supplement/plot_tgd_response_sensitivity.py` | Reviewer-2 report calls the response figure `Fig. S-TGD`; no finalized figure number | Include as one three-panel figure. Panel c reports the actual broad-transition degradation rather than hiding it. |
| Figure S5: HUC-2 LORO | Regional omission sensitivity for R1, R3, and R5 | **Re-rendered** | `results/reviewer2_robustness/regional_loro/r1_huc2_loro.csv`, `r3_huc2_loro.csv`, `r5_huc2_loro.csv` | `manuscript/scripts/supplement/plot_huc2_loro_robustness.py` | Reviewer-2 report calls this `Fig. S-LORO`; no finalized figure number | Retain only source HUC_11–HUC_18 and display them as HUC_01–HUC_08. The frozen `regional_loro_forest.png` remains preserved as a source asset; no map, heatmap, or HUC-2 connecting lines. |
| Main Figures 5–6 | Canonical R3 outlet/internal evidence that the Supplement must not duplicate or modify | **Yes** | `manuscript/results/R3/figure5_*`, `figure6_*`, and formal scripts | `manuscript/scripts/r3/plot_figure5.py`, `plot_figure6.py` | Results 3.3 / Figures 5–6 | Leave unchanged. Supplement figures provide only additional evidence and seasonal separation. |
| Reviewer-2 evidence record | Traceability for robustness decisions and residual limitations | **Yes** | `manuscript/reviewer2_response_evidence_matrix.md`; `results/reviewer2_robustness/REVIEWER2_ROBUSTNESS_FINAL_REPORT.md` | — | Used as audit sources | Use as evidence, but replace provisional labels (`Fig. S-TGD`, `Fig. S-LORO`) with the frozen map below. |

## Current conflicts requiring editorial attention

1. `manuscript/figure_manifests/canonical_assets.json` still lists only Figure S1
   in `submission_structure.supplement_figures`, although the working assets now
   support five final Supplement figures. It also declares main-table outputs in
   `manuscript/stats/tables/`, while the current generated Supplement tables are
   in `manuscript/supplement/tables/`. This file already has unrelated working-tree
   modifications and was not changed here.
2. The seasonal asset is explicitly called Figure S6 by its script and current
   Results draft, but the final numbering map below assigns it Figure S2.
3. The seasonal metadata string says “plot median and IQR,” while the plotting
   script uses the recorded `ci_lo`/`ci_hi` fields. The source generator documents
   these as 2,000-draw bootstrap 95% CIs of the monthly median. The asset was not
   redrawn; this metadata wording should be reconciled before submission.
4. The alternative-field dPL test has only **123 denominator-valid basins** under
   `D_b > 10^-6`, despite the canonical dPL seed-median sample being 460. This is
   a real consequence of the alternative truth field and is displayed explicitly
   in Figure S3, not harmonized away.
5. Figure S1 had a stale panel-g eligible-count label (`N=443`); the current
   audit CSV contains 442 eligible catchments. The same Figure S1 layout and
   data were re-rendered with the corrected count.
