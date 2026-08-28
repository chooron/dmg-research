# Table 5: R3 Synthetic Known-Truth Experiment — Summary of Compensation and Structural-Surrogate Evidence (Figures 5–6)

| Estimand | Evidence role | IC median [95% CI] | dPL median [95% CI] | N (IC/dPL) |
| :--- | :--- | :--- | :--- | :---: |
| F_close,test | Limited compensation | 0.101 [0.080, 0.113] | 0.101 [0.089, 0.109] | 427/468 |
| decay_G_base | Generalization decay | 0.0137 [0.0119, 0.0151] | 0.0021 [0.0014, 0.0036] | 531/531 |
| F_tgd2 (test) | Generic mitigation | 0.510 [0.483, 0.541] | 0.504 [0.474, 0.529] | 466/528 |
| Delta C_theta (R_theta_tgd2) | Parameter relief | 0.0047 [0.0027, 0.0071] | 0.0067 [0.0042, 0.0096] | 531/531 |
| Delta C_state (R_state_tgd2) | State relief | 0.0749 [0.0459, 0.1037] | 0.1101 [0.0889, 0.1285] | 531/531 |
| Delta KGE_CN-TGD2 (test) | Residual explicit advantage | 0.0487 [0.0385, 0.0608] | 0.0466 [0.0380, 0.0573] | 531/531 |
| Delta RMSE_snow (mm d-1) | Residual on snow-active days | 0.479 [0.436, 0.523] | 0.465 [0.420, 0.503] | 521/521 |
| Delta RMSE_non-snow (mm d-1) | Residual on non-snow days | 0.077 [0.062, 0.092] | 0.063 [0.046, 0.074] | 476/476 |

*Note*: Values report basin-level medians with 95% bootstrap confidence intervals [2.5th, 97.5th percentiles] from paired basin resampling (2000 replicates, seed 20260730; repository R3 convention). dPL values are per-basin medians over seeds 42/123/2026 (seed-aggregated), reported as the median across basins. Sign conventions: decay_G_base = G_base(train) - G_base(test), positive means compensation is stronger in train; Delta C_theta and Delta C_state = Base - TGD2, positive means TGD2 reduces the CN-adjusted excess error; Delta KGE_CN-TGD2 = KGE_CN - KGE_TGD2, positive means CN retains an advantage; process residuals = RMSE_TGD2 - RMSE_CN (mm d-1) on truth snow-active / non-snow days, positive means CN has lower error. Valid basin counts (IC/dPL) by row: Limited compensation 427/468; Generalization decay 531/531; Generic mitigation 466/528; Parameter relief 531/531; State relief 531/531; Residual explicit advantage 531/531; Residual on snow-active days 521/521; Residual on non-snow days 476/476.
