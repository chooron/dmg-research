# HESS Supplement asset plan

## Decision rule

Keep the Supplement as supporting evidence, not a second Results section. The
frozen plan contains five figures: one existing external-state figure, one
reused seasonal figure, two newly rendered robustness figures, and one reused
regional-omission forest plot. Denominator tails remain tabular/textual.
Figures 5 and 6 are not modified.

## A. Keep unchanged

- **Figure S1 — external-state corroboration.** Keep the existing selection, layout, and
  data. A stale panel-g label (`N=443`) was corrected to the audit-CSV count
  (`N=442`) using the existing `manuscript/scripts/r4/plot_r4_figure_s1_multibasin.py`;
  no new experiment or layout change was made.
- **Table S1 — parameter bounds.** Keep the current parameter definitions,
  bounds, units, and Base/TGD/CN applicability.
- **Table S2 Panel A — timing-screen sensitivity.** Keep the existing KGE and
  `|ΔCT|` threshold grid exactly as defined.
- **Main Figures 5 and 6.** Keep the canonical controlled-recovery and
  truth-relative internal-state figures unchanged. The seasonal evidence is
  separate and must not be folded back into Figure 6.
- **Existing source assets.** Do not delete the old seasonal source file, the frozen
  TGD response source image, or the frozen regional forest source image:
  `manuscript/figures/Fig_S6_R3_seasonal_trajectories.png`,
  `results/reviewer2_robustness/tgd_response/tgd_response_curves.png`, and
  `results/reviewer2_robustness/regional_loro/regional_loro_forest.png`. The Figure S1
  output was the one authorized numeric-label correction described above.

## B. Reuse / renumber

| Existing asset | Final asset | Decision |
|---|---|---|
| `manuscript/figures/Fig_S6_R3_seasonal_trajectories.png` | `manuscript/supplement/figures/FigureS2_R3_seasonal_trajectories.png` | Exact byte-for-byte copy; final number is S2. The old S6 file remains. |
| `results/reviewer2_robustness/regional_loro/regional_loro_forest.png` | `manuscript/supplement/figures/FigureS5_huc2_loro_robustness.png` | Re-rendered from the regional LORO CSVs, retaining source HUC_11–HUC_18 and displaying them as HUC_01–HUC_08. The frozen source image remains preserved. |
| `results/reviewer2_robustness/tgd_response/tgd_response_curves.png` | Included conceptually in final Figure S4 | Retain as a frozen audit/source image, but do not submit it as a duplicate standalone figure. |

The seasonal chain is complete: `f_snow >= Q75`, threshold
`0.21769666937653748`, `N=133`, water-year October–September axis, effective
liquid-water input, and common XAJ tension-water storage are recorded in
`fig6_seasonal_meta.json`. Its generator documents median plus 2,000-draw
catchment-resampling 95% intervals for the plotted `ci_lo`/`ci_hi` bands. The
metadata's conflicting “median and IQR” wording should be corrected in the
formal Supplement body, not by changing this copied PNG.

## C. New figures

### Figure S3 — alternative generating-field robustness

A compact two-panel figure compares the canonical PCA/Ridge field against the
un-smoothed basin-wise CN–IC field.

- **Panel a:** test-period raw `G_Base` and `G_TGD`, median with empirical
  Q25–Q75 interval, using all 531 basins.
- **Panel b:** unclipped `F_close`, `F_TGD*`, and the paired `ΔF` contrast;
  fractions use `D_b > 10^-6` and median/Q25–Q75. `P(ΔF > 0)` and valid N are
  annotated.
- **Important values:** canonical `ΔF` = +0.460 (IC; 91.6% positive,
  `N_valid=427`) and +0.443 (dPL; 92.8% positive, `N_valid=460`); direct-field
  `ΔF` = +0.195 (IC; 72.4% positive, `N_valid=522`) and +0.701 (dPL; 91.1%
  positive, `N_valid=123`).
- **Claim boundary:** recovery ordering is retained, but absolute fractions
  are generating-field sensitive. The direct-field dPL valid sample is not
  silently treated as the canonical 460.

Output: `FigureS3_alt_generating_field_robustness.png`.

### Figure S4 — TGD response and response-shape sensitivity

A single three-panel figure combines the existing mathematical response data
with the completed basin-level shape sensitivity.

- **Panel a:** `tau(T)` from the six recorded parameter settings.
- **Panel b:** `r(T)` from the same 351-temperature table; this shows
  continuous leakage rather than a discrete snow/rain partition.
- **Panel c:** median `ΔF` with Q25–Q75 interval for Sharp (`s_T=1`), Canonical
  (`s_T=2`), Warm-shifted (`T_ref=+2, s_T=2`), and Broad (`s_T=4`), with IC
  open markers and dPL filled markers.
- **Values shown:** dPL `ΔF` = +0.414, +0.441, +0.255, and −0.091 for those
  four variants, respectively; the IC values are −0.159, −0.134, −0.219,
  and −0.366. The broad transition is shown as an actual degradation.
- **Claim boundary:** stability is limited to the tested response family and
  reasonable transition scales; TGD is not described as a snow module or as a
  universal bound.

Output: `FigureS4_tgd_response_shape_sensitivity.png`.

### Figure S5 — HUC-2 regional omission robustness

Reuse the existing three-panel forest plot layout and regenerate it from the regional LORO CSVs. It displays R1 S5–S1 timing,
R3 `ΔF`, and R5 S5 majority host coherence for the 8 retained HUC-2 omissions (source HUC_11–HUC_18,
displayed as HUC_01–HUC_08) plus the full-sample reference. HUC-2 categories are not connected, and the result is
called **regional omission sensitivity**, not spatial correction.

## D. Tables

- **Table S1:** no expansion required.
- **Table S2:** retain Panels A and B. Add a compact Panel B note or adjacent
  note at formal assembly containing the already frozen denominator/tail facts:
  `D_b` valid N = 427 (IC) and 460 (dPL); invalid denominators are concentrated
  in S1 (91.4% and 91.5% of invalid basins); `P(F_close<0)` = 31.6% and
  25.0%; `P(F_TGD<0)` = 8.4% and 5.9%; `P(ΔF>0)` = 91.6% and 92.8%; and the
  `ΔF` contrast remains approximately +0.42–+0.46 across cutoffs from
  `10^-6` to `0.10`. These are unclipped fractions.
- **Table S3:** do **not** add. Alternative-field, shape, and LORO values are
  adequately represented by Figures S3–S5, provenance, and caption notes.

## E. Do not include

- **Standalone denominator-tail figure:** the frozen CSV and Table S2B grid
  fully support the disclosure; a separate tail plot would add little and
  would overemphasize a secondary ratio diagnostic.
- **Standalone TGD response PNG:** it would duplicate Panels a–b of Figure S4.
- **Alternative-field NPZ files or full basin distributions:** retain as
  provenance/source data; do not add bulky `q_star_alt.npz` or
  `theta_star_alt.npz` to the figure set.
- **A second LORO visualization, map, or heatmap:** the existing forest plot is
  sufficient and avoids implying a spatial correction or continuous ordering of
  HUC-2 categories.
- **Alternative generating-field, shape-sensitivity, or HUC-2 results in the
  main Results figures:** these remain Supplement robustness evidence.
- **The empty `dpl_no_fsnow/` lane or a 34-attribute dPL ablation:** no result
  exists, and the user explicitly prohibited retraining.
- **Any new alternative-generating-field, TGD-shape, HUC2-LORO, CMA-ES,
  training, full evaluation, or checkpoint-generation run.**
