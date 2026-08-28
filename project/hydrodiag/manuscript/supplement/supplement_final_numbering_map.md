# Recommended final Supplement numbering map

This map is frozen for the current asset handoff. It does not delete or rename
legacy files; it only defines the submission-facing copies.

| Final number | Scientific role | Existing/source filename | Final output filename | Status |
|---|---|---|---|---|
| Figure S1 | R4 multi-catchment external-state corroboration | `manuscript/supplement/figures/FigureS1_R4_multibasin_validation.png` | `manuscript/supplement/figures/FigureS1_R4_multibasin_validation.png` | Retain layout/data; corrected panel-g `N=442` label |
| Figure S2 | R3 high-snow seasonal effective input and truth-relative storage trajectory | `manuscript/figures/Fig_S6_R3_seasonal_trajectories.png` | `manuscript/supplement/figures/FigureS2_R3_seasonal_trajectories.png` | Exact copy; old S6 retained |
| Figure S3 | Alternative generating-field robustness | `results/reviewer2_robustness/alt_generating_field/*.csv` plus canonical R3 basin CSV | `manuscript/supplement/figures/FigureS3_alt_generating_field_robustness.png` | Newly rendered |
| Figure S4 | TGD response curves and response-shape sensitivity | `results/reviewer2_robustness/tgd_response/tgd_response_data.csv` and `tgd_shape_sensitivity/tgd_shape_sensitivity_basin_metrics.csv` | `manuscript/supplement/figures/FigureS4_tgd_response_shape_sensitivity.png` | Newly rendered combined figure |
| Figure S5 | R1/R3/R5 HUC-2 regional omission sensitivity | `results/reviewer2_robustness/regional_loro/r1_huc2_loro.csv`, `r3_huc2_loro.csv`, `r5_huc2_loro.csv` | `manuscript/supplement/figures/FigureS5_huc2_loro_robustness.png` | Re-rendered using source HUC_11–HUC_18, displayed as HUC_01–HUC_08 |

## Legacy/provisional label mapping

| Old/provisional label | Final number | Action |
|---|---|---|
| `Fig_S6_R3_seasonal_trajectories.png` / “Figure S6” | Figure S2 | Use the final S2 copy in submission references; retain old source file. |
| `Fig. S-TGD` in `reviewer2_response_evidence_matrix.md` and the final report | Figure S4 | Replace only at formal manuscript assembly. |
| `Fig. S-LORO` in `reviewer2_response_evidence_matrix.md` and the final report | Figure S5 | Replace only at formal manuscript assembly. |
| `FigureS1_R4_multibasin_validation.png` | Figure S1 | No change. |
| `tgd_response_curves.png` | — | Keep as frozen audit/source image; it is represented within Figure S4 rather than submitted separately. |
| `regional_loro_forest.png` | Figure S5 | Copy only; preserve the source asset. |

The numbering is intentionally sequential and keeps the seasonal evidence
separate from Reviewer-2 robustness figures. Main Figures 1–9 remain unchanged.
