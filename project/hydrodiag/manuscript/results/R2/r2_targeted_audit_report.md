# Results 3.2 (R2) Targeted Statistical Audit Report

- **Status:** PASS / VERIFIED
- **Scope:** Targeted verification of (1) TGD attribution control Full531 vs ExcludeS5 $\Delta\beta$ enhancement and (2) Complete S1–S5 macro trajectory across snow-activity gradient.
- **Audit Protocol:** Read-mostly audit operating strictly on lowest-level raw parameters (IC: 10 restarts, dPL: 3 seeds) with verified paired bootstrap.

## 1. Sample, Pairing, and Bootstrap Implementation Verification

- **Full531 vs ExcludeS5 Subsets:** Full531 contains exactly 531 unique basins; ExcludeS5 contains exactly 476 basins (S1=165, S2=156, S3=121, S4=34; 55 S5 basins omitted).
- **Structural Pairing:** Base–CN and Base–TGD use identical 531 basins in Full531 and identical 476 basins in ExcludeS5.
- **`frac_snow` Alignment:** 100% matched with canonical R1 manifest.
- **Paired Bootstrap Implementation:** Verified at code level in `tgd_attribution_control.py`: each bootstrap draw resamples basin IDs with replacement, then *simultaneously* refits $\beta(\text{Base-CN})$ and $\beta(\text{Base-TGD})$ on the exact same resampled basins and computes $\Delta\beta = \beta_{\text{CN}} - \beta_{\text{TGD}}$ within the draw. No post-hoc independent CI subtraction is used.

## 2. S1–S5 Complete Macro Trajectory

### A. IC-CMA-ES Macro Trajectory

| Stratum | n | $f_{\text{snow}}$ Median | Base-CN within (median) | Base-CN between (median) | Base-CN excess [95% CI] | Base-CN Prevalence [95% CI] | Base-TGD excess [95% CI] | Base-TGD Prevalence [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 165 | 0.0125 | 0.473 | 0.470 | **-0.0018** [-0.0044, +0.0015] | **46.7%** [40.0%, 53.9%] | **+0.0006** [-0.0021, +0.0040] | **51.5%** [43.9%, 58.8%] |
| **S2** | 156 | 0.0845 | 0.485 | 0.488 | **+0.0022** [-0.0034, +0.0041] | **52.6%** [44.5%, 60.6%] | **+0.0022** [-0.0001, +0.0055] | **57.1%** [49.4%, 65.1%] |
| **S3** | 121 | 0.2021 | 0.475 | 0.494 | **+0.0123** [+0.0090, +0.0185] | **74.4%** [66.1%, 81.8%] | **+0.0133** [+0.0080, +0.0163] | **71.1%** [62.8%, 78.1%] |
| **S4** | 34 | 0.3652 | 0.462 | 0.509 | **+0.0430** [+0.0324, +0.0649] | **94.1%** [85.3%, 100.0%] | **+0.0308** [+0.0106, +0.0460] | **76.5%** [61.8%, 91.2%] |
| **S5** | 55 | 0.6823 | 0.452 | 0.560 | **+0.0829** [+0.0702, +0.1104] | **98.2%** [94.5%, 100.0%] | **+0.0937** [+0.0743, +0.1153] | **94.5%** [87.3%, 100.0%] |

### B. dPL-MLP Macro Trajectory

| Stratum | n | $f_{\text{snow}}$ Median | Base-CN within (median) | Base-CN between (median) | Base-CN excess [95% CI] | Base-CN Prevalence [95% CI] | Base-TGD excess [95% CI] | Base-TGD Prevalence [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 165 | 0.0125 | 0.105 | 0.132 | **+0.0186** [+0.0074, +0.0258] | **70.3%** [63.0%, 77.0%] | **+0.0195** [+0.0104, +0.0270] | **70.9%** [63.9%, 77.6%] |
| **S2** | 156 | 0.0845 | 0.098 | 0.165 | **+0.0557** [+0.0433, +0.0793] | **87.2%** [82.1%, 91.7%] | **+0.0310** [+0.0234, +0.0406] | **75.6%** [69.9%, 82.1%] |
| **S3** | 121 | 0.2021 | 0.106 | 0.233 | **+0.1267** [+0.1006, +0.1702] | **92.6%** [87.2%, 96.7%] | **+0.0791** [+0.0547, +0.1378] | **88.4%** [82.6%, 94.2%] |
| **S4** | 34 | 0.3652 | 0.139 | 0.346 | **+0.1322** [+0.0926, +0.2667] | **97.1%** [91.2%, 100.0%] | **+0.1070** [+0.0681, +0.1537] | **91.2%** [79.4%, 100.0%] |
| **S5** | 55 | 0.6823 | 0.119 | 0.307 | **+0.1252** [+0.0826, +0.2189] | **87.3%** [77.2%, 94.5%] | **+0.0851** [+0.0546, +0.1173] | **87.3%** [78.2%, 94.5%] |

## 3. Explanation of Full vs ExcludeS5 Slopes and dPL $\Delta\beta$ Enhancement

### A. Regression Comparisons

| Paradigm | Contrast | Full531 OLS Slope | ExcludeS5 OLS Slope | Slope Shift (Excl - Full) | Full531 Spearman $\rho$ | ExcludeS5 Spearman $\rho$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| IC | Base-CN | +0.1542 | +0.1387 | **-0.0155** | +0.549 | +0.406 |
| IC | Base-TGD | +0.1538 | +0.1155 | **-0.0383** | +0.438 | +0.282 |
| dPL | Base-CN | +0.1974 | +0.4271 | **+0.2297** | +0.441 | +0.465 |
| dPL | Base-TGD | +0.1563 | +0.3410 | **+0.1847** | +0.380 | +0.387 |

### B. Leverage & Cook's Distance Diagnostics

| Paradigm | Contrast | Stratum | n | Mean Leverage ($h_{ii}$) | Max Leverage | Mean Cook's D | Mean Residual (y - yhat) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| IC | Base-CN | S1 | 165 | 0.0030 | 0.0033 | 0.0006 | +0.0047 |
| IC | Base-CN | S2 | 156 | 0.0022 | 0.0026 | 0.0005 | -0.0034 |
| IC | Base-CN | S3 | 121 | 0.0020 | 0.0026 | 0.0011 | -0.0030 |
| IC | Base-CN | S4 | 34 | 0.0040 | 0.0068 | 0.0044 | +0.0008 |
| IC | Base-CN | S5 | 55 | 0.0140 | 0.0268 | 0.0275 | +0.0015 |
| IC | Base-TGD | S1 | 165 | 0.0030 | 0.0033 | 0.0005 | +0.0079 |
| IC | Base-TGD | S2 | 156 | 0.0022 | 0.0026 | 0.0003 | -0.0043 |
| IC | Base-TGD | S3 | 121 | 0.0020 | 0.0026 | 0.0007 | -0.0059 |
| IC | Base-TGD | S4 | 34 | 0.0040 | 0.0068 | 0.0066 | -0.0075 |
| IC | Base-TGD | S5 | 55 | 0.0140 | 0.0268 | 0.0352 | +0.0063 |
| dPL | Base-CN | S1 | 165 | 0.0030 | 0.0033 | 0.0012 | -0.0196 |
| dPL | Base-CN | S2 | 156 | 0.0022 | 0.0026 | 0.0007 | -0.0039 |
| dPL | Base-CN | S3 | 121 | 0.0020 | 0.0026 | 0.0013 | +0.0374 |
| dPL | Base-CN | S4 | 34 | 0.0040 | 0.0068 | 0.0029 | +0.0404 |
| dPL | Base-CN | S5 | 55 | 0.0140 | 0.0268 | 0.0140 | -0.0373 |
| dPL | Base-TGD | S1 | 165 | 0.0030 | 0.0033 | 0.0008 | -0.0160 |
| dPL | Base-TGD | S2 | 156 | 0.0022 | 0.0026 | 0.0008 | -0.0060 |
| dPL | Base-TGD | S3 | 121 | 0.0020 | 0.0026 | 0.0015 | +0.0377 |
| dPL | Base-TGD | S4 | 34 | 0.0040 | 0.0068 | 0.0027 | +0.0216 |
| dPL | Base-TGD | S5 | 55 | 0.0140 | 0.0268 | 0.0141 | -0.0316 |

### C. Mathematical and Physical Explanation of dPL $\Delta\beta$ Enhancement

1. **Nonlinear Plateauing in S5**: In dPL, excess structural separation rises rapidly across moderate snow regimes S1 $\to$ S2 $\to$ S3 $\to$ S4 ($f_{\text{snow}} \in [0, 0.50]$): Base-CN excess increases from $+0.0186$ to $+0.1322$. In S5 ($f_{\text{snow}} \in [0.50, 0.91]$), excess plateaus at $+0.1252$.
2. **High $x$-Leverage of S5**: S5 basins have high $x$-coordinates (mean $f_{\text{snow}} = 0.68$, mean leverage $h_{ii} = 0.0140$, 7x higher than S2/S3). Because excess levels off in S5 rather than rising linearly to $>0.30$, these high-leverage points exert negative torque on the global OLS line, flattening $\beta(\text{Base-CN})$ from $+0.4271$ in ExcludeS5 down to $+0.1974$ in Full531.
3. **TGD Behavior**: Base-TGD follows a similar plateauing profile, flattening from $+0.3410$ (ExcludeS5) to $+0.1563$ (Full531).
4. **Why $\Delta\beta$ increases in ExcludeS5**: Across the active steep transition in S1–S4, Base-CN separates at rate $+0.4271$, while Base-TGD separates at rate $+0.3410$. The rate difference in the active snow zone is $\Delta\beta = +0.4271 - 0.3410 = \mathbf{+0.0861}$ [+0.017, +0.157]. In Full531, because S5 flattens both slopes towards the plateau, the global linear fit compresses the difference to $\Delta\beta = \mathbf{+0.0411}$ [+0.008, +0.077].
5. **Scientific Implication**: Structural differentiation between CN and TGD is **not an S5 artifact**; CN separates from TGD throughout the moderate snow regimes (S2, S3, S4). The historical hypothesis that differentiation was driven by S5 is disproven by the stratified data.

## 4. Main Conclusion Wording Verdicts

- **IC-CMA-ES:** **`MONOTONIC / NEAR-MONOTONIC ORGANIZATION`** — IC Base-CN excess increases strictly monotonically across all strata: S1 (-0.002) -> S2 (+0.002) -> S3 (+0.012) -> S4 (+0.043) -> S5 (+0.083), and prevalence increases from 46.7% to 98.2%.
  - *Recommended wording:* "parameter-space reorganization became progressively stronger with snow activity"
- **dPL-MLP:** **`ORDERED BUT NONLINEAR`** — dPL Base-CN excess rises steeply from S1 (+0.019) -> S2 (+0.056) -> S3 (+0.127) -> S4 (+0.132), then plateaus in S5 (+0.125). S5 exerts high leverage that pulls down the linear slope across [0, 0.91].
  - *Recommended wording:* "parameter-space reorganization was increasingly organized across the snow-activity gradient, steep across moderate snow regimes (S2-S4) and plateauing in high-snow basins (S5)"

## 5. Artifact Manifest

- `r2_s1_s5_macro_trajectory.csv`: Full S1-S5 trajectory for Base-CN and Base-TGD across IC and dPL.
- `r2_leverage_influence_diagnostics.csv`: Stratum-level leverage, Cook's distance, and residuals.
- `r2_regression_comparison_full_vs_excl_s5.csv`: Full531 vs ExcludeS5 OLS slopes and Spearman correlations.
- `r2_targeted_audit_summary.json`: Complete machine-readable summary.
- `r2_targeted_audit_report.md`: Targeted audit report.