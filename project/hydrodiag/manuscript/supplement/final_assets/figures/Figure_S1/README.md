# Figure S1 — Leave-one-HUC2-out regional omission robustness

## 1. Scientific role

Figure S1 audits regional omission robustness for the R1 timing contrast, R3 recovery contrast, and R5 cross-host coherence. It belongs in the SI because it is a sensitivity analysis rather than a spatial process model or a main-text estimand.

## 2. What is shown

Panel (a) shows the R1 S5–S1 timing contrast for each retained HUC-2 omission. Panel (b) shows the R3 `Delta F = F_TGD* - F_close` contrast. Panel (c) shows the R5 S5 majority-positive cross-host agreement. IC and dPL points are separate, and full-sample values are vertical reference lines.

The source HUC labels HUC_11–HUC_18 are intentionally displayed as HUC_01–HUC_08. Source HUC_01–HUC_10 are the random ten-fold partitions and are excluded from the figure.

## 3. Source data

- `results/reviewer2_robustness/regional_loro/r1_huc2_loro.csv`
- `results/reviewer2_robustness/regional_loro/r3_huc2_loro.csv`
- `results/reviewer2_robustness/regional_loro/r5_huc2_loro.csv`
- Renderer base: `manuscript/scripts/supplement/plot_huc2_loro_robustness.py`

## 4. Sample definition

There are 8 retained source HUC-2 omissions per paradigm: HUC_11–HUC_18, displayed as HUC_01–HUC_08. Each panel has IC and dPL omission points plus one full-sample reference per paradigm. No random seeds or basin resampling are introduced by the renderer.

## 5. Metric definitions

Panel (a) uses `Delta CT` (the signed S5–S1 timing contrast) in days. Panel (b) uses paired recovery contrast `Delta F = F_TGD* - F_close`. Panel (c) uses `P(A >= 2)` in S5 converted to percent. The figure reports regional omission sensitivity, not spatial correction.

## 6. Aggregation and uncertainty

Each plotted point is the already summarized HUC-2 omission value from the source CSV. No interval or bootstrap band is added. The full-sample row is shown as a vertical reference line.

## 7. Generation method

- Script: `manuscript/supplement/final_assets/figures/Figure_S1/plot_Figure_S1.py`
- Command: `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S1/plot_Figure_S1.py`
- Input: three frozen regional LORO CSVs.
- Output: `Figure_S1.png`.
- **NO MODEL TRAINING. NO RECALIBRATION. NO FULL TEST PIPELINE.**

## 8. Visual encoding

IC uses blue circles and dPL uses orange triangles. Full IC and full dPL are dashed/dotted vertical lines. Categories are independent and are not connected by lines. The layout follows the existing Figure S5 HUC-2 asset.

## 9. Caption-ready factual statements

- Three panels show R1, R3, and R5 regional omission sensitivity.
- Points represent source HUC_11–HUC_18 omissions, displayed as HUC_01–HUC_08.
- Full-sample IC and dPL values are shown as reference lines.
- The figure contains no map and no connecting lines.

## 10. Interpretation boundary

The figure tests sensitivity to deleting retained regional categories. It does not estimate spatial dependence, spatial correction, or a formal regional uncertainty distribution.

## 11. Validation

The renderer was executed successfully. All three CSVs contain exactly 16 retained omission rows (8 regions × 2 paradigms), and a mutation-invariance check confirmed that changing excluded HUC_01–HUC_10 values leaves the output unchanged. The output was visually inspected for eight aligned rows per panel, legible labels, and non-clipped legends.
