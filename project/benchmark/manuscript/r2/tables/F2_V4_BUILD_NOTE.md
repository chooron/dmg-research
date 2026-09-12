# R2 Figure 2 final build note

## Provenance and frozen estimands

Phase A is resolved as **IC-SELF REFERENCE = VALID**. The canonical IC-self reference is compared with eligible archived IC multi-start vectors under the predeclared `within_0.01` training-fitness rule; the raw archive contains ten IC restart slots per basin for 36 models and 531 basins/model. The CR provenance audit verdict is **CR DATA VERDICT = STRICT23 ONLY**: panel (c) uses the identical 23-model primary set across canonical, consensus, and IC-self. `collie1` is a valid one-parameter model (`P=1`); its invariant localization behavior is structural, not a corruption or restart/cache failure.

The primary paired quantity is:

```text
delta_m,b = D_cross(m,b) - D_self(m,b)
Delta_m    = median_b(delta_m,b)
```

The model-equal primary median excess is `0.2187745308` with 95% bootstrap CI `[0.2091360196, 0.2281453316]`; 36/36 model-level estimates are positive. The distinct marginal contrast is `0.2530788322` and is not interchangeable with the paired estimand. Excluding `collie1` leaves `35/35` positive models with median paired excess `0.2052773724`.

The hard quantile gate `Q05 <= Q25 <= median <= Q75 <= Q95` passed for both plotted displacement quantities in all 36 models, using the same 531-basin matched subset per model. The unit-histogram conclusion was checked privately at bin widths 0.04, 0.05, and 0.06; width 0.05 is used in the figure.

## Final visual grammar

- **(a) HERO:** the existing interval landscape, with Q05–Q95 whiskers, Q25–Q75 bands, medians, slight vertical dodge, the fixed ascending-`median D_cross` order, and a `-0.05–0.8` x-axis. Purple and green encode the contrasting `D_self` and `D_cross` quantities; the in-panel legend uses only the `D_self` and `D_cross` symbols.
- **(b) UNIT HISTOGRAM:** fixed-width bins of `0.05` in `Delta_m`; each model contributes exactly one stacked square, with a visible negative `Delta_m < 0` region that is empty. The main histogram retains only the black dashed `Delta_m = 0` reference. Green squares encode the positive paired excess; collie1 is plotted with the same symbol as the other valid model estimates. Its one-parameter structural status is documented separately.
- **(c) RIDGELINE / BOUNDARY:** the identical 23-model primary set contributes three distributions of model-level CR values, one for each alternative IC reference. KDEs are estimated in log-CR space with the same bandwidth rule and common ridge-height normalization; short baseline rug ticks show the individual model-level values. A very pale neutral background marks only `CR < 1` and is labeled `apparent contraction`; a fine bracket groups consensus and IC-self as `alternative references`. Purple marks the canonical reference, while green shades mark consensus and IC-self; the common black dashed line marks `CR = 1`, and right-side labels report `0.614`, `0.979`, and `1.004`.

The exact panel (a) model order is:

```text
penman → us1 → alpine2 → susannah1 → collie2 → newzealand1 → simhyd → alpine1 → collie3 → gsfb → australia → smar → xinanjiang → mopex2 → modhydrolog → newzealand2 → gr4j → flexis → hbv96 → ihacres → hillslope → tcm → hymod → susannah2 → vic → tank → mopex1 → flexi → plateau → mopex5 → flexb → mopex4 → wetland → mopex3 → collie1 → topmodel
```

Required caption wording: panel (a) displays the marginal basin-level distributions of `D_cross` and `D_self`, whereas panel (b) displays `Delta_m = median_b(D_cross - D_self)`. Therefore the paired median excess `0.2188` and marginal-median difference `0.2531` are not expected to be identical. The scope is 36 models, 531 basins/model, and 19,116 exact matched model–basin cells. The pooled cell-level check is `87.6%` (16741/19116) with `D_cross > D_self`; this is not the model-equal primary inference. The references in panel (c) are alternatives, not a temporal sequence.
Panel (c) caption wording: CR is shown relative to the common `CR = 1` boundary using the identical strict primary subset of `23/36` models for all three reference constructions; the common subset requires primary IC restart coverage `>= 0.90`. The canonical median (`0.614`) lies clearly below the boundary, whereas consensus (`0.979`) and IC-self (`1.004`) are near unity; `CR ≈ 1` means relative parameter spread comparable to the reference, not expansion. KDEs were estimated in log-CR space using the same bandwidth rule, and short rug ticks show individual model-level CR values.
The panel (c) ridgelines are across the same 23 models for all references: `canonical` is the naive reference, while `consensus` and `IC-self` are the alternative primary constructions. The panel must be read as strict-subset reference sensitivity, not a full-36 model-equal claim.
The panel (a) model order is the frozen ascending `median D_cross` order in `F2_MODEL_ORDER.csv`; any later Figure 3 or R4 reuse must preserve that order and distinguish the strict23 CR subset from the 36-model displacement/excess scope.

## Output scope

The formal output is one complete composite PNG only. No panel-wise image, PDF, fourth panel, 1:1 plane, near-zero strip, restart audit, rank/localization metric, map, or performance bridge is included. The output is 600 dpi at approximately 18.2 × 17.0 cm.

- Script: `scripts/plot_r2_figure2_unit_hist_cr_deviation.py`
- Figure: `figures/Figure2_R2_parameter_space_response_final.png`
- Model order: `tables/F2_MODEL_ORDER.csv`
- Values: `tables/F2_V4_FINAL_VALUES.csv`
