# Final Figure 3 (F3) — Execution Report

## What changed (final 5-panel version; new script `plot_r2_figure3_final.py`)

- **Layout**: compact, slightly vertical 5-panel composite (8.0 × 10.2 in figure,
  7.84 × 9.91 in content, aspect 0.79) — much narrower than the previous 11.5-in
  versions. Rows: (a)(b) top (equal width), (c) full-width HERO with two compact
  facets (IC left, dPL right), bottom (d):(e) at width ratio 1.3 : 0.7 (measured
  1.86 ≈ 1.3/0.7).
- **(a)/(b)** — unchanged in substance: both Base–CN (blue squares) and Base–TGD2
  (teal triangles) basin scatter vs within-structure variability, 1:1 line, spaced
  percentage prevalence annotations (IC 63.1 %/60.6 %; dPL 83.8 %/82.3 %),
  regime-appropriate axis ranges.
- **(c) HERO — upgraded to the continuous relationship**: each of the two facets
  (IC, dPL) now shows, for both contrasts on the same x = frac_snow axis:
  (i) basin-level low-alpha scatter (all 531 basins); (ii) OLS regression line with a
  95 % slope-CI wedge anchored at the sample centroid; (iii) S1–S5 binned median
  excess + 95 % bootstrap CI as larger markers at each bin's median frac_snow, with
  light dotted bin-boundary guides and S1–S5 labels at the top. Zero reference line.
  The regression slopes reproduce the frozen values exactly (asserted); the only
  locally computed quantities are the OLS intercepts (deterministic) and bin-median
  frac_snow positions. No Δβ annotations.
- **(d) Sources of excess separation — restored to line decomposition** (not forest):
  one panel with two internal facets (IC left, dPL right), each showing Base–CN
  `between_all` (blue) vs `within_pooled` (grey) across S1–S5 with 95 % CIs. Facets use
  separate ordinates (IC 0.42–0.62; dPL 0.05–0.52) so raw levels are not compared
  across regimes. (Independent review found and the script was fixed: the dPL facet
  must be drawn on an internal sub-axes of the (d) cell, not on the (e) axes.)
- **(e) Gradient summary** — the 8 slopes (Base–CN / Base–TGD2 × IC/dPL ×
  Full/Excl-S5) with CIs, x-axis `Snow-gradient slope, β`, numeric labels, zero line.
- **Old panel (f) removed** (paired Δβ no longer a main panel); the Δβ values are
  reported only in the caption and this report.

## Statistics reused / computed

Fully reused the frozen-style TGD2 specificity outputs:
`r2_tgd2_specificity_basin_level.csv`, `r2_tgd2_specificity_summary.csv`,
`r2_tgd2_specificity_regressions.csv`. No upstream analysis modified. Locally computed
(lightweight, deterministic):
- OLS intercepts for the (c) regression lines — slopes asserted to equal the frozen
  regression slopes (e.g. IC Base–CN +0.1542, dPL Base–CN +0.4721);
- descriptive above-1:1 prevalences;
- bin-median frac_snow positions for the binned markers (S1 0.0125 … S5 0.6823);
- the 95 % slope-CI wedge = lines through the sample centroid with the frozen slope CI
  bounds (a standard visualization of slope uncertainty when only slope intervals are
  stored; stated in the caption).

## Outputs

- `manuscript/figures/Figure3_R2_final.png` (+ `plots/figures/` mirror), 4705×5949 px,
  600 DPI, PNG-only convention plus vector PDF
- `manuscript/figures/Figure3_R2_final.pdf` (vector, fonts embedded FontFile2/Type42)
- Script: `manuscript/scripts/plot_r2_figure3_final.py`
- Caption: `manuscript/captions/Figure3_R2_final_caption.md`

## Validation performed

- 5-panel layout verified geometrically: (a)(b) equal width; (c) full-width hero with
  two facets in the same y-band; (d):(e) ratio 1.86; hero row tallest.
- Pixel audit: (c) both facets contain blue + teal (continuous relationship + lines +
  binned markers); (d) contains blue + grey with zero teal (Base–CN decomposition);
  (e) blue + teal; zero orange pixels anywhere (no IC-vs-dPL colour system); CVD-safe
  via marker shape + line style redundancy.
- Deterministic: identical MD5 across two runs (no RNG in the script).
- Frozen-value sanity checks pass inline (prevalences; local slope == frozen slope).
- PNG and PDF both generated; PDF fonts embedded.

## Remaining non-blocking issues

- The (c) facets share a continuous frac_snow axis but have separate y-ordinates
  (IC excess range smaller than dPL's); the caption states magnitudes are not ranked
  across regimes.
- The 95 % CI band in (c) is a slope-uncertainty wedge (only slope CIs are stored in
  the frozen regression outputs), not a full prediction/confidence band; stated in the
  caption.
- A human visual pass at print size is recommended to confirm the scatter/regression/
  binned-marker layering in (c) and the label density in the narrow (e) panel
  (geometry checks found no clipping).

## Final micro-adjustment pass (PNG-only export)

- (c)/(d) facet headers `IC constraint`/`dPL constraint` → `IC regime`/`dPL regime`.
- (c) basin scatter further lightened (s 6→4, alpha 0.16→0.11) so the regression line,
  CI wedge and binned markers are the visual main layer.
- (d) S1–S5 tick labels simplified to `S1`…`S5` (sample sizes defined once in the
  caption, no longer repeated on both facets).
- (e) title → `Snow-gradient summary`.
- Subplot vertical spacing (hspace 0.55→0.38) and the internal (d) facet gap
  (wspace 0.30→0.18) reduced for a tighter single-panel look.
- Output now PNG-only (600 DPI) per the final instruction; the earlier PDF is removed.
- Re-verified: 5 panels render, (c) main layers dominate, (d) two facets with zero
  teal, (e) clean slopes, zero orange, deterministic MD5.
