# Supplement execution report

## Status

Completed as a read-only evidence audit plus minimal plotting/staging. Existing
user modifications and existing canonical/source assets were preserved.

## Files and evidence read

- `manuscript/figure_manifests/canonical_assets.json`
- `manuscript/hess_results_R1_R5_reframed_v2.md`
- `manuscript/methods_supplement_production_audit.md`
- `manuscript/reviewer2_response_evidence_matrix.md`
- `manuscript/reviewer2_current_evidence_audit.md`
- current `manuscript/supplement/tables/TableS1_parameter_bounds.{md,tex,csv}`
  and `TableS2_sensitivity_audits.{md,tex}`
- current R3 Figure 5/6 summaries and basin-level tables, including
  `manuscript/results/R3/fig6_seasonal/fig6_seasonal_meta.json`
- frozen Reviewer-2 outputs under
  `results/reviewer2_robustness/{p0_reporting,regional_loro,tgd_response,alt_generating_field,tgd_shape_sensitivity,summaries}`
- formal Figure 5/6/7/8/9 plotting scripts and
  `manuscript/scripts/shared/r1_plot_style.py`

## Reused assets

1. Figure S1 retained its existing layout and data; the stale panel-g eligible count
   was corrected from `N=443` to the audit-CSV count `N=442` by running the
   existing no-model plotting helper after a one-line dynamic-label fix.
2. The existing R3 seasonal PNG was copied byte-for-byte from
   `manuscript/figures/Fig_S6_R3_seasonal_trajectories.png` to
   `manuscript/supplement/figures/FigureS2_R3_seasonal_trajectories.png`.
3. The existing HUC-2 forest PNG was initially copied byte-for-byte from
   `results/reviewer2_robustness/regional_loro/regional_loro_forest.png` to
   `manuscript/supplement/figures/FigureS5_huc2_loro_robustness.png`; it was later
   re-rendered from the three regional LORO CSVs to retain source HUC_11–HUC_18
   and display them as HUC_01–HUC_08.
4. The existing `tgd_response_curves.png` was treated as a frozen source/audit
   asset, not duplicated as a standalone submission figure.
5. The two original copy operations were checked with SHA-256 equality against their sources.

## Newly rendered figures

- **Figure S3:**
  `manuscript/supplement/figures/FigureS3_alt_generating_field_robustness.png`
  from canonical and direct-field basin CSVs. Panel a uses raw KGE gains over
  all 531 basins; panel b uses unclipped denominator-valid fractions and
  annotates the field-specific valid N and positive paired fraction.
- **Figure S4:**
  `manuscript/supplement/figures/FigureS4_tgd_response_shape_sensitivity.png`
  from the frozen 351-row TGD response CSV and the completed shape-sensitivity
  basin CSV. It combines `tau(T)`, `r(T)`, and `ΔF` shape sensitivity.
- **Figure S5:** `manuscript/supplement/figures/FigureS5_huc2_loro_robustness.png` was re-rendered from the three regional LORO CSVs, retaining source HUC_11–HUC_18 and displaying them as HUC_01–HUC_08.
Both scripts are in `manuscript/scripts/supplement/` and have fixed input paths,
no model imports, no random resampling, and no input mutations.

## Evidence intentionally not added as figures

- Denominator tails, invalid-stratum concentration, and cutoff sensitivity stay
  in Table S2 Panel B / text. The frozen facts are `N_valid=427` (IC) and 460
  (dPL), invalid-basin S1 shares 91.4% and 91.5%, and unclipped negative-tail
  rates for `F_close` of 31.6% and 25.0%.
- No Table S3 was added: Figures S3–S5 plus Table S2 and provenance notes cover
  the robustness evidence without fragmenting the Supplement.
- No `dpl_no_fsnow` result or 34-attribute ablation was invented; that directory
  is empty and retraining was prohibited.
- No alternative-field NPZ, complete daily state series, map, heatmap, or
  duplicate standalone response plot was added to the final figure set.

## Conflicts and limitations found

1. The specified S1–S3 Supplement planning body is absent. The production audit
   references `HESS_Supplementary_Methods_with_placeholders.md`, which is not in
   the current tree.
2. `canonical_assets.json` still declares only Figure S1 and points some table
   outputs to `manuscript/stats/tables/`; it has pre-existing working-tree
   changes and was not edited.
3. The old seasonal filename/label is S6, while this handoff freezes it as S2.
   The seasonal data chain is complete (`f_snow >= Q75`, threshold
   `0.21769666937653748`, `N=133`), but the metadata says “median and IQR” while
   the source generator and plot use `ci_lo`/`ci_hi` bootstrap 95% intervals.
4. The alternative direct-field dPL test has only 123 denominator-valid basins,
   versus 460 for canonical dPL. This is displayed in Figure S3 and must remain
   explicit in the final caption.
5. Reviewer documents use provisional labels `Fig. S-TGD` and `Fig. S-LORO`;
   the final map resolves these to S4 and S5.
6. Figure S1's prior panel-g `N=443` label disagreed with the audit CSV's 442 eligible
   rows; the existing helper was corrected to compute and display `N=442`.

## Resource and execution record

- Interpreter: `/home/jingxin/code/dmg-research/.venv/bin/python` (Python 3.10.20).
- Environment: `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
  `OPENBLAS_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2`.
- Subagent concurrency: at most two read-only scouts, then serial plotting.
- RAM: the S3/S4 plots loaded only small basin-level CSVs and the 351-row response table.
  The one authorized S1 label correction invoked the existing plotting helper and loaded its
  frozen R4 reference/model-array inputs in one process; `q_star_alt.npz` and model checkpoints
  were not loaded. No memory pressure was observed.
- GPU: not used by this pass.
- No training, CMA-ES, parameter estimation, forward simulation, full test or
  evaluation pipeline, alternative-field rerun, shape-sensitivity rerun,
  HUC-2 recomputation, `pytest`, or `unittest` was run.

## Validation performed

- Both newly added Supplement plotting scripts completed successfully with the fixed project
  interpreter and generated one PNG each.
- The existing Figure S1 helper was run once only to correct the stale eligible-count label;
  its output was visually inspected. The new S3/S4 PNGs were visually inspected, and the
  S2 copy was visually inspected; the original S5 copy was visually inspected and copied unchanged.
- Source/copy SHA-256 equality was confirmed for the seasonal copy and the original LORO copy.
- Follow-up correction: the HUC-2 renderer was rerun with source HUC_11–HUC_18 only; the resulting Figure S5 was visually inspected and confirmed to contain eight displayed HUC_01–HUC_08 rows per panel.
- No project regression tests were run, by explicit task constraint.

## Independent review

- Two fresh-context read-only reviewer passes were performed.
- The first pass identified clipped lower Q25 caps in Figure S4; the shape-panel y-range was expanded and the PNG was regenerated.
- The follow-up pass confirmed that all eight Q25–Q75 intervals are visible and found no remaining blocker.
- Assembly follow-ups remain: update the pre-existing manifest, reconcile seasonal interval wording, standardize `F_TGD`/`F_TGD*`, and apply the final S2/S4/S5 labels.


## Final recommendation

Submit **five Supplement figures (S1–S5)**, retain **Table S1**, expand **Table
S2 Panel B** with a compact denominator/tail note, and do **not** add Table S3.
Before manuscript assembly, update only the necessary provisional figure/table
references and reconcile the seasonal interval wording and manifest paths.
