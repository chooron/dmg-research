# Results 3.2 (R2) Canonical Results Package

This directory contains the frozen, verified canonical datasets and statistics for **Section 3.2 (R2: Real-Catchment Parameter Response Layer)** of the manuscript.

All data are derived strictly from lowest-level raw parameter outputs (10 independent CMA-ES restarts per basin for IC; 3 independent seeds 42, 123, 2026 for dPL) across the 531 CAMELS basins, without launching any model training, calibration, inference, or forward simulations.

---

## 1. Directory & File Catalog

### A. Core Primary Datasets
| File | Rows | Description |
| :--- | :---: | :--- |
| `authoritative_15_parameter_specs.csv` | 15 | Definition, bounds, physical units, processes, and layout indices for all 15 shared XAJ parameters. |
| `raw_parameter_ledger.csv` | 310,635 | Complete long-form ledger ($531 \times 3 \times (10+3) \times 15$) with physical and normalized values. |
| `r2_parameter_values_canonical.csv` | 3,186 | Canonical basin-level parameter vectors ($531 \text{ basins} \times 3 \text{ structures} \times 2 \text{ paradigms}$). |
| `r2_within_structure_basin_level.csv` | 1,062 | Basin-level ensemble within/between/excess metrics for Base–CN ($531 \times 2$). |
| `r2_paired_shifts_basin_level.csv` | 15,930 | Basin-level signed parameter shifts $\Delta z = z_{\text{Base}} - z_{\text{CN}}$ ($531 \times 2 \times 15$). |
| `r2_tgd2_specificity_basin_level.csv` | 3,186 | Basin-level ensemble metrics for Base–CN, Base–TGD, and TGD–CN ($531 \times 2 \times 3$). |

### B. Statistical Summaries (Figure 3 & Figure 4 Direct Data Sources)
| File | Target in Manuscript | Description |
| :--- | :--- | :--- |
| `r2_within_structure_summary.csv` | **Figure 3** | Full531, ExcludeS5, and S1–S5 strata medians, IQRs, 95% CIs for within, between, and excess. |
| `r2_macro_regressions.csv` | **Figure 3 / Table S4** | OLS regression slopes and Spearman $\rho$ for within_pooled, between_all, and excess on $f_{\text{snow}}$. |
| `r2_s1_s5_macro_trajectory.csv` | **Figure 3 & Discussion** | Complete S1–S5 macro excess and prevalence trajectory with 95% bootstrap CIs. |
| `r2_parameter_shifts_full_summary.csv` | **Figure 4** | Full-sample (531) medians, IQRs, 95% CIs, OLS slopes, and Spearman $\rho$ for all 15 parameters. |
| `r2_parameter_shifts_strata_summary.csv` | **Figure 4** | S1–S5 stratified distributions and $S5-S1$ activity endpoint contrasts for all 15 parameters. |
| `r2_snow_gradient_robustness.csv` | **Figure 4 / Table S4** | ExcludeS5 and Leave-One-Stratum-Out (LOSO) sensitivity for all 15 parameter slopes. |
| `r2_tgd2_slope_difference_summary.csv` | **Figure 3 / Table S4** | Paired bootstrap $\Delta\beta = \beta(\text{Base-CN}) - \beta(\text{Base-TGD})$ across Full531 and ExcludeS5. |
| `r2_paired_cn_tgd_delta_excess_summary.csv` | **Figure 3 / Discussion** | Basin-paired $\Delta_{\text{excess}} = \text{excess}(\text{Base-CN}) - \text{excess}(\text{Base-TGD})$ across strata. |

### C. Quality Audits & Robustness Diagnostics
| File | Purpose |
| :--- | :--- |
| `r2_leave_one_parameter_out_sensitivity.csv` | 14-D LOPO whole-space sensitivity across all 15 parameter exclusions (`ROBUST_MULTIVARIATE`). |
| `r2_parameter_distance_contribution_shares.csv` | Individual parameter percentage contribution shares to total squared distance (Option B). |
| `r2_leverage_influence_diagnostics.csv` | Stratum-level Hat matrix leverage ($h_{ii}$), Cook's distance, and residuals explaining S5 plateau. |
| `r2_four_basin_calculation_trace.csv` | Step-by-step calculation trace for 4 representative basins (IC-S1, IC-S5, dPL-S1, dPL-S5). |
| `r2_historical_reconciliation.csv` | Explicit reconciliation of historical numbers vs rebuilt canonical table. |
| `r2_ic_restart_quality_audit.csv` | IC restart quality metrics (KGE IQR, best-minus-median, Top-3 vs Top-5 sensitivity). |
| `r2_dpl_seed_stability_audit.csv` | dPL across-seed standard deviation and stability per parameter. |
| `r2_boundary_mass_safeguards.csv` | Exact boundary hits and near-boundary point mass shares (1%, 2%, 5% tolerances). |
| `canonical_gates_summary.json` | Formal automated verification of all 12 R2 validation gates (**12/12 PASS**). |
| `machine_readable_summary.json` | Complete machine-readable summary. |
| `r2_final_closure_report.md` | Formal final statistical closure report (`R2_FINAL_STATUS = READY`). |

---

## 2. Key Canonical Statistics & Scientific Evidence

### A. Macro Whole-Space Response (Figure 3)
- **Prevalence ($\text{fraction}(\text{between\_all} > \text{within\_pooled})$):**
  - **IC-CMA-ES (10 restarts):** **63.09%** (335/531 basins) [59.13%, 67.04%] (95% CI)
  - **dPL-MLP (3 seeds):** **83.80%** (445/531 basins) [80.60%, 86.82%] (95% CI)
- **Macro Excess OLS Slope on Snow Fraction ($\beta(\text{excess} \sim f_{\text{snow}})$):**
  - **IC-CMA-ES:** Full531 $\beta = \mathbf{+0.1542}$ [+0.0898, +0.2185]; ExcludeS5 $\beta = \mathbf{+0.4042}$ [+0.2783, +0.5285]
  - **dPL-MLP:** Full531 $\beta = \mathbf{+0.1974}$ [+0.1578, +0.2372]; ExcludeS5 $\beta = \mathbf{+0.4267}$ [+0.3541, +0.4996]

### B. S1–S5 Stratified Trajectory
- **IC Base–CN excess:** S1 ($-0.0018$) $\to$ S2 ($+0.0022$) $\to$ S3 ($+0.0123$) $\to$ S4 ($+0.0430$) $\to$ S5 ($+0.0829$) (Strictly monotonic increase; prevalence $46.7\% \to 98.2\%$).
- **dPL Base–CN excess:** S1 ($+0.0186$) $\to$ S2 ($+0.0557$) $\to$ S3 ($+0.1267$) $\to$ S4 ($+0.1322$) $\to$ S5 ($+0.1252$) (Steep rise across moderate snow regimes S1–S4, plateauing in S5).

### C. TGD Attribution Control & Paired $\Delta\beta$
- **IC-CMA-ES:** Full531 $\Delta\beta = \mathbf{+0.000}$ [-0.032, +0.031]; ExcludeS5 $\Delta\beta = \mathbf{+0.023}$ [-0.013, +0.058]
- **dPL-MLP:** Full531 $\Delta\beta = \mathbf{+0.0411}$ [+0.008, +0.077]; ExcludeS5 $\Delta\beta = \mathbf{+0.0861}$ [+0.017, +0.157]
- **Basin-paired $\Delta_{\text{excess}} = \text{excess}(\text{Base-CN}) - \text{excess}(\text{Base-TGD})$ in dPL:**
  - S1: $+0.0047$ (near zero)
  - S2: $\mathbf{+0.0287}$ [+0.0153, +0.0369] (strictly positive)
  - S3: $\mathbf{+0.0239}$ [+0.0142, +0.0363]
  - S4: $\mathbf{+0.0363}$ [+0.0072, +0.1025]
  - S5: $\mathbf{+0.0317}$ [+0.0065, +0.0501]
  - *Conclusion*: CN establishes additional parameter reorganization over TGD starting in intermediate snow strata S2/S3.

### D. Key Parameter Signatures (Figure 4)
- $u_m$ (Tension water storage capacity): IC slope $\beta = \mathbf{+0.521}$, dPL slope $\beta = \mathbf{+0.566}$ (Consistent positive shift).
- $k_i$ (Interflow outflow coefficient): IC slope $\beta = \mathbf{-0.475}$, dPL slope $\beta = \mathbf{-0.315}$ (Consistent negative shift).
- $c_i$ (Interflow recession constant): IC slope $\beta = \mathbf{-0.414}$, dPL slope $\beta = \mathbf{-0.531}$ (Consistent negative shift).
- $i_m$ (Impervious area fraction): IC slope $\beta = \mathbf{-0.363}$, dPL slope $\beta = \mathbf{-0.142}$ (Consistent negative shift).

---

## 3. Rebuilding & Execution

To reproduce all artifacts from source:

```bash
.venv/bin/python project/hydrodiag/manuscript/analysis/R2/run_all.py --draws 10000
```

To run the complete automated test suite:

```bash
.venv/bin/python -m pytest project/hydrodiag/manuscript/analysis/R2/tests/ -v
```
