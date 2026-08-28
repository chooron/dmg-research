# Figure S5 provenance

- **Scientific question:** Do the R1 timing contrast, R3 recovery contrast, and R5 high-snow cross-host coherence depend on omitting one HUC-2 region?
- **Input files:** `results/reviewer2_robustness/regional_loro/r1_huc2_loro.csv`, `r3_huc2_loro.csv`, and `r5_huc2_loro.csv`. The renderer retains source regions HUC_11–HUC_18; the frozen source image is preserved separately for provenance.
- **Input columns:** R1 `paradigm`, `region_removed`, `S5_minus_S1_contrast`; R3 `paradigm`, `period`, `region_removed`, `Delta_F_median`, `P_Delta_F_gt_0`; R5 `paradigm`, `region_removed`, `P_majority_positive`.
- **Aggregation:** One point per retained omitted HUC-2 region (source HUC_11–HUC_18, displayed as HUC_01–HUC_08), with the full-sample row shown as a vertical reference. Panel a uses R1 S5–S1 signed timing contrast; panel b uses R3 median `ΔF`; panel c uses R5 S5 `P(A>=2)` converted to percent.
- **N:** 8 retained HUC-2 omissions per paradigm. Full-sample references are R1 `N=531`, R3 valid `N=427` IC / `N=460` dPL, and R5 S5 `N=55` before omission.
- **IC/dPL handling:** IC and dPL use distinct markers and reference lines; HUC-2 categories are not connected.
- **Interval definition:** No interval is added; the figure is an omission-sensitivity point cloud with full-sample reference lines.
- **Plot script:** `manuscript/scripts/supplement/plot_huc2_loro_robustness.py` is the transparent CSV-based renderer. It filters out source regions HUC_01–HUC_10 and renumbers HUC_11–HUC_18 as HUC_01–HUC_08.
- **Output:** `manuscript/supplement/figures/FigureS5_huc2_loro_robustness.png`, rendered from the three regional LORO CSVs.
- **Canonical values checked:** Full-sample R1 contrasts are +47.4 d (IC) and +46.3 d (dPL); R3 `ΔF` is +0.460 and +0.443; R5 majority agreement is 90.9% for both regimes. Omission ranges remain directionally positive.
- **Claim boundary:** This is regional omission sensitivity, not formal spatial correction or spatial independence inference.
