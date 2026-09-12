# Figure 4 (F4) — Execution Report (final: Base–CN-only main figure + SI Base–TGD2 parallel)

Generated: 2026-08-13 by `manuscript/scripts/plot_r2_figure4.py` (final state) and
`manuscript/scripts/plot_r2_fig_s6_tgd2_parallel.py` (new SI figure).

## Final design decision implemented

- **Main-text F4 is Base–CN only** (the Base–TGD2 overlay that had been added to
  panel (a) was removed again): panel (a) shows the 15 shared parameters as a single
  deep-blue-square contrast per row (um/ki/ci highlighted only by the light row band
  + bold labels), and the bottom legend is reduced to the IC/dPL ridge group plus the
  (e) colour bar.
- **Base–TGD2 matched-control evidence moved to a new SI parallel figure (Fig. S6)**.

## Files

- Main F4 script: `manuscript/scripts/plot_r2_figure4.py` (reverted to Base–CN only;
  layout/spacing unchanged).
- New SI script: `manuscript/scripts/plot_r2_fig_s6_tgd2_parallel.py`.
- Main F4 figure: `manuscript/plots/figures/Figure4_R2.png` (PNG only, per project
  convention).
- SI figure: `manuscript/supplement/figures/Fig_S6_R2_tgd2_parallel_figure4.png`
  (+ vector PDF, matching the supplement figure convention).
- Captions: `manuscript/captions/Figure4_R2_caption.md` (updated),
  `manuscript/captions/FigureS6_R2_tgd2_parallel_caption.md` (new).

## SI figure content (Fig. S6)

- (a) Base–TGD2 snow gradients (teal triangles), IC / dPL regimes, 15 params,
  um/ki/ci rows shaded + bold.
- (b–d) ridgeline distributions of dZ = z_Base − z_TGD2 for um / ki / ci (identical
  mirrored IC/dPL grammar, S1–S5, x ∈ [−1, 1]).
- **GIS omitted deliberately**: Base–TGD2 paired shifts are small and diffuse with
  only weak snow-gradient spatial organization (|corr(dZ, frac_snow)| ≤ ~0.3 for the
  key parameters vs strong Base–CN signals), so maps would be noisy/redundant.

## Data reuse

- Base–CN: frozen `r2_snow_gradients_summary.csv` (a) and
  `r2_paired_shifts_basin_level.csv` (b–e), unchanged.
- Base–TGD2: verified `r2_snow_gradients_base_tgd2_summary.csv` (a;
  `validation_base_cn_max_abs_diff == 0` asserted); (b–d) derived in-script from the
  canonical `r2_parameter_values_canonical.csv` as z_Base − z_GD (GD == XAJ_TGD2) —
  no upstream file written or modified.
## Layout retained (unchanged from the previous round)

- Three balanced columns ((a) | (b–d) ridges | (e) GIS), asymmetric gaps
  ((a)→(bcd) ≈0.75 in, (bcd)→(e) ≈0.38 in); shared bottom legend row ≈1 character
  below the panels; GIS maps ≈3.98 × 2.50 in; main F4 exported as PNG only (no PDF).

## Validation

- Main F4: panel (a) contains only the 15 Base–CN deep-blue squares in both IC and
  dPL subplots (zero teal / zero orange point pixels; warm highlight band present
  behind `um / ki / ci`); identical shared x-limits (≈[−1.5, 2.0], Base–CN only);
  bottom legend reduced to the IC/dPL ridge group + (e) colour bar; 0 clipped text.
- Panels (b–e): unchanged, Base–CN only (blue IC / orange dPL ridges; GIS um
  orange-dominant, ki/ci blue-dominant).
- Fig. S6: (a) teal triangles only (zero pure-blue / zero pure-orange); (b–d) blue
  IC / orange dPL ridges of z_Base − z_TGD2 (TGD2 medians small/diffuse, e.g. dPL
  um S5 +0.10, ki S3 −0.16, ci ≈ 0); shared (a) x-limits; 0 clipped text; legend
  IC/dPL.
- No upstream analysis, result files or public interfaces modified; Supplement
  Fig. S5 and its data were left untouched; the new SI figure and captions are the
  only additions.

## Outputs

- `manuscript/plots/figures/Figure4_R2.png` — 600 DPI (PNG only)
- `manuscript/supplement/figures/Fig_S6_R2_tgd2_parallel_figure4.png` (+ `.pdf`)
- `manuscript/supplement/figures/Fig_S4_IC_maps.png` — IC maps (Supplement Fig. S4)
- `manuscript/captions/Figure4_R2_caption.md` — updated (Base–CN only)
- `manuscript/captions/FigureS6_R2_tgd2_parallel_caption.md` — new

## Non-blocking notes

- The SI figure omits GIS panels (small/diffuse Base–TGD2 shifts with weak snow
  gradient) and is explicitly framed in its caption as descriptive matched-control
  context, not a significance test against Base–CN.
- The KDE bandwidth (0.075) and all other plotting parameters are shared with F4
  via module reuse (`plot_r2_figure4` imported read-only by the S6 script).
