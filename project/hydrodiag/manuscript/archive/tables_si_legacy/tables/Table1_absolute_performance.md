# Table 1: Streamflow Simulation Performance Across Structural Configurations and Parameter-Estimation Regimes

| Configuration | Regime | Period | KGE | NSE | PBIAS (%) | RMSE (mm d⁻¹) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Base | IC | Train | 0.810 [0.788, 0.825] | 0.719 [0.699, 0.734] | 4.05 [3.94, 4.13] | 0.352 [0.340, 0.369] |
| Base | IC | Test | 0.672 [0.650, 0.692] | 0.589 [0.568, 0.607] | 3.36 [3.25, 3.46] | 0.462 [0.446, 0.480] |
| Base | dPL | Train | 0.734 [0.706, 0.755] | 0.647 [0.621, 0.667] | 3.67 [3.53, 3.77] | 0.413 [0.396, 0.435] |
| Base | dPL | Test | 0.678 [0.656, 0.702] | 0.594 [0.574, 0.617] | 3.39 [3.28, 3.51] | 0.457 [0.438, 0.475] |
| TGD | IC | Train | 0.833 [0.825, 0.844] | 0.741 [0.734, 0.752] | 4.16 [4.13, 4.22] | 0.334 [0.325, 0.340] |
| TGD | IC | Test | 0.709 [0.698, 0.727] | 0.624 [0.613, 0.641] | 3.55 [3.49, 3.64] | 0.432 [0.418, 0.442] |
| TGD | dPL | Train | 0.764 [0.753, 0.781] | 0.676 [0.665, 0.692] | 3.82 [3.76, 3.91] | 0.389 [0.375, 0.398] |
| TGD | dPL | Test | 0.718 [0.702, 0.733] | 0.632 [0.617, 0.646] | 3.59 [3.51, 3.67] | 0.426 [0.414, 0.439] |
| CN | IC | Train | 0.871 [0.866, 0.875] | 0.777 [0.773, 0.781] | 4.35 [4.33, 4.37] | 0.303 [0.300, 0.307] |
| CN | IC | Test | 0.760 [0.746, 0.772] | 0.672 [0.659, 0.683] | 3.80 [3.73, 3.86] | 0.392 [0.382, 0.403] |
| CN | dPL | Train | 0.824 [0.815, 0.833] | 0.733 [0.724, 0.741] | 4.12 [4.08, 4.16] | 0.341 [0.334, 0.348] |
| CN | dPL | Test | 0.760 [0.742, 0.769] | 0.672 [0.654, 0.681] | 3.80 [3.71, 3.85] | 0.392 [0.384, 0.407] |

*Note*: Values report basin-wise medians with 95% bootstrap confidence intervals [2.5th, 97.5th percentiles] across all n = 531 matched basins for calibration (1981–1995) and evaluation (1995–2010) periods. Units: PBIAS (%), RMSE (mm d⁻¹). KGE and NSE are dimensionless. HBV is reported as an external dPL reference benchmark and is not part of the controlled XAJ structural progression.