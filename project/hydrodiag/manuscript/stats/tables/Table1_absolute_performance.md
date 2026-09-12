# Table 1: Streamflow Simulation Performance Across Structural Configurations and Parameter-Estimation Regimes

| Configuration | Regime | Period | KGE | NSE | PBIAS (%) | RMSE (mm d⁻¹) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Base | IC | Train | 0.813 [0.792, 0.828] | 0.633 [0.602, 0.663] | 0.25 [0.17, 0.34] | 1.476 [1.352, 1.573] |
| Base | IC | Test | 0.670 [0.648, 0.693] | 0.482 [0.442, 0.522] | 7.78 [6.38, 8.96] | 1.649 [1.531, 1.780] |
| Base | dPL | Train | 0.733 [0.704, 0.752] | 0.613 [0.590, 0.636] | -1.70 [-2.27, -1.17] | 1.484 [1.351, 1.570] |
| Base | dPL | Test | 0.686 [0.667, 0.705] | 0.540 [0.503, 0.570] | 4.66 [3.18, 6.21] | 1.540 [1.453, 1.650] |
| TGD | IC | Train | 0.831 [0.824, 0.841] | 0.674 [0.661, 0.694] | 0.04 [-0.01, 0.08] | 1.317 [1.222, 1.414] |
| TGD | IC | Test | 0.712 [0.693, 0.726] | 0.556 [0.524, 0.583] | 7.33 [5.52, 8.27] | 1.534 [1.438, 1.636] |
| TGD | dPL | Train | 0.773 [0.761, 0.790] | 0.662 [0.642, 0.681] | -1.32 [-1.74, -0.80] | 1.320 [1.257, 1.413] |
| TGD | dPL | Test | 0.721 [0.703, 0.728] | 0.597 [0.575, 0.623] | 5.01 [3.18, 7.27] | 1.456 [1.383, 1.542] |
| CN | IC | Train | 0.873 [0.868, 0.878] | 0.750 [0.741, 0.762] | -0.05 [-0.09, -0.02] | 1.142 [1.062, 1.219] |
| CN | IC | Test | 0.754 [0.744, 0.772] | 0.647 [0.624, 0.665] | 7.74 [6.02, 8.61] | 1.370 [1.294, 1.459] |
| CN | dPL | Train | 0.825 [0.815, 0.836] | 0.720 [0.709, 0.736] | -1.45 [-1.81, -1.00] | 1.200 [1.113, 1.272] |
| CN | dPL | Test | 0.763 [0.746, 0.772] | 0.669 [0.653, 0.687] | 6.60 [4.69, 7.84] | 1.353 [1.286, 1.426] |
| HBV (reference) | dPL | Train | 0.784 [0.771, 0.792] | 0.668 [0.656, 0.678] | -3.09 [-3.65, -2.66] | 1.326 [1.245, 1.401] |
| HBV (reference) | dPL | Test | 0.731 [0.718, 0.743] | 0.616 [0.601, 0.630] | 2.13 [0.80, 3.01] | 1.481 [1.395, 1.546] |

*Note*: Values report basin-wise medians with 95% bootstrap confidence intervals [2.5th, 97.5th percentiles] across all n = 531 matched basins for calibration (1981–1995) and evaluation (1995–2010) periods. Units: PBIAS (%), RMSE (mm d⁻¹). KGE and NSE are dimensionless. HBV is reported as an external dPL reference benchmark and is not part of the controlled XAJ structural progression.