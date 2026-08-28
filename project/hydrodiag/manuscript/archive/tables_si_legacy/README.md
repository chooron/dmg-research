# Legacy Tables and Supplementary Figures Archive

This directory stores archived, legacy, duplicated, or superseded tables and supplementary figures from previous iterations of the `hydrodiag` manuscript. None of these files are part of the final submission package, but they are preserved here for full provenance, auditability, and historical traceability.

---

## 1. Archived Tables (`tables/`)

| Original File Name | Original Location | Description / Original Purpose | Reason for Archiving / Replacement | Associated Generation Script |
| :--- | :--- | :--- | :--- | :--- |
| `Table1_absolute_performance.md` / `.tex` | `manuscript/stats/tables/` | Full-sample streamflow simulation performance (KGE, NSE, PBIAS, RMSE) for Base, TGD, CN under IC and dPL | Replaced by new **Table 1: Structural configurations and diagnostic roles** (Methods descriptive table; performance figures are in Fig. 1). | `manuscript/scripts/r1/generate_table1.py` |
| `Table5_R3_final.md` / `.tex` / `Table5_R3_summary.*` | `manuscript/stats/tables/` | R3 synthetic known-truth experiment multi-metric summary | Streamlined and replaced by new **Table 2: Controlled recovery of the imposed outlet gap** (focused on core gap recovery estimands). | `manuscript/scripts/r3/generate_table_r3_main.py` |
| `TableS1_paired_effects_and_sensitivity.md` / `.tex` | `manuscript/stats/tables/` | R1 paired structural $\Delta\text{KGE}$ effects across snow regimes and timing error sensitivity | Retired to minimize SI; core timing and KGE distributions are already in Figure 2. | `manuscript/scripts/r1/generate_table_s1.py` |
| `TableS2_paired_structural_kge_differences.md` / `.tex` | `manuscript/stats/tables/` | Exhaustive stratum-by-stratum paired KGE differences across train/test and IC/dPL | Retired to minimize SI; exhaustive splits are redundant with Figure 1 & 2. | `manuscript/scripts/r1/generate_table_s2.py` |
| `TableS3_ic_dpl_temporal_transfer.md` / `.tex` | `manuscript/stats/tables/` | IC vs dPL temporal transfer decay table | Retired; temporal persistence is already visualized in Figure 1c,d. | `manuscript/scripts/r1/generate_table_s3.py` |
| `TableS4_exact_estimates_f3_f4.md` / `.tex` | `manuscript/stats/tables/` | R2 exact slope contrasts and 15 parameter gradients underlying Figures 3 and 4 | Retired; Figure 3 and 4 visually convey these gradients; parameter bounds are in Table S1. | `manuscript/scripts/r2/generate_table_s4.py` |
| `TableS5_boundary_point_mass.md` / `.tex` | `manuscript/stats/tables/` | Boundary and point-mass statistics of $u_m, k_i, c_i$ | Retired; detailed ridgeline distributions are already shown in Figure 4e–h. | `manuscript/scripts/r2/generate_table_s5.py` |
| `TableS5_R3_statistics.md` / `.tex` / `TableS5_R3_final.*` | `manuscript/stats/tables/` | R3 5-block detailed SI statistics table | Streamlined into new Table 2 and Discussion text; exhaustive blocks retired from submission. | `manuscript/scripts/r3/generate_table_r3_si.py` |

---

## 2. Archived Supplementary Figures (`figures/`)

| Original File Name | Original Location | Description / Original Purpose | Reason for Archiving / Replacement | Associated Generation Script |
| :--- | :--- | :--- | :--- | :--- |
| `FigureS5_R3_final.png` | `manuscript/supplement/figures/` | R3 component-level parameter and state excess errors | Retired; Figure 6 already covers parameter distance, excess, and state/flux dynamics. | `manuscript/scripts/r3/plot_r3_si_components.py` |
| `FigureS6_R4_interim.png` / `legacy.png` | `manuscript/supplement/figures/` | R4 multi-seed (seed 42, seed 123) and IC fused replication curves | Replaced by new **Figure S1: R4 multi-catchment external-state consistency validation** (6-basin small multiples). | `manuscript/scripts/r4/plot_r4_figure_s6.py` |
| `figureS7_active_melt_ecdf.png` / `.csv` | `manuscript/figures/SI/` | R4 active-melt anomaly correlation difference ECDF | Retired; Figure 7c already includes active-melt ECDF heterogeneity. | `manuscript/scripts/r4/plot_r4_figure7_rebuilt.py` |

---

## 3. Policy Note

- All underlying machine-readable CSV files in `manuscript/results/` and `manuscript/analysis/` remain frozen and intact.
- The generation scripts remain in `manuscript/scripts/` for reproducibility, but their outputs are no longer part of the submission package.
