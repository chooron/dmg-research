# Table S5: R3 Synthetic-Truth SI Statistics

### Block 1 — Correct-CN reference

| Regime | Estimand | Median | Note |
| :---: | :--- | :---: | :--- |
| IC | CN test KGE (vs Q*) | 0.9926 | deficit 1-KGE = 0.0074 |
| IC | CN parameter recovery D_theta | 0.0699 | gate equifinality median |
| IC | CN state NRMSE [wu] | 0.4486 | gate state summary, primary states |
| IC | CN state NRMSE [wl] | 0.4973 | gate state summary, primary states |
| IC | CN state NRMSE [s] | 0.3688 | gate state summary, primary states |
| IC | CN state NRMSE [qi] | 0.2369 | gate state summary, primary states |
| IC | CN state NRMSE [qg] | 0.1476 | gate state summary, primary states |
| dPL | CN test KGE (vs Q*) | 0.9955 | deficit 1-KGE = 0.0045 |
| dPL | CN parameter recovery D_theta | 0.0243 | gate equifinality median |
| dPL | CN state NRMSE [wu] | 0.0633 | gate state summary, primary states |
| dPL | CN state NRMSE [wl] | 0.0953 | gate state summary, primary states |
| dPL | CN state NRMSE [s] | 0.1604 | gate state summary, primary states |
| dPL | CN state NRMSE [qi] | 0.1334 | gate state summary, primary states |
| dPL | CN state NRMSE [qg] | 0.0778 | gate state summary, primary states |

### Block 2 — Main R3 estimands

| Estimand | Regime | N | Median [95% CI] |
| :--- | :---: | :---: | :--- |
| Limited compensation: F_close (test) | IC | 427 | 0.1015 [0.0773, 0.1126] |
| Limited compensation: F_close (test) | dPL | 468 | 0.1009 [0.0889, 0.1092] |
| Generic mitigation: F_tgd2 (test) | IC | 466 | 0.5105 [0.4867, 0.5412] |
| Generic mitigation: F_tgd2 (test) | dPL | 528 | 0.5036 [0.4732, 0.5295] |
| Parameter relief: R_theta_tgd2 | IC | 531 | 0.0047 [0.0027, 0.0071] |
| Parameter relief: R_theta_tgd2 | dPL | 531 | 0.0067 [0.0042, 0.0096] |
| State relief: R_state_tgd2 | IC | 531 | 0.0749 [0.0457, 0.1037] |
| State relief: R_state_tgd2 | dPL | 531 | 0.1101 [0.0887, 0.1284] |
| Residual CN advantage: G_CN_over_TGD2 (test) | IC | 531 | 0.0487 [0.0382, 0.0608] |
| Residual CN advantage: G_CN_over_TGD2 (test) | dPL | 531 | 0.0466 [0.0383, 0.0573] |
| Generalization decay: decay_G_base | IC | 531 | 0.0137 [0.0118, 0.0151] |
| Generalization decay: decay_G_base | dPL | 531 | 0.0021 [0.0014, 0.0035] |
| Process residual RMSE gap [snow_active] (TGD2-CN) | IC | 521 | 0.4785 [0.4371, 0.5228] |
| Process residual RMSE gap [snow_active] (TGD2-CN) | dPL | 521 | 0.4646 [0.4201, 0.5029] |
| Process residual RMSE gap [no_snow_active] (TGD2-CN) | IC | 476 | 0.0768 [0.0618, 0.0945] |
| Process residual RMSE gap [no_snow_active] (TGD2-CN) | dPL | 476 | 0.0632 [0.0461, 0.0745] |

### Block 3 — dPL seed stability

| Estimand | Seed | Median |
| :--- | :---: | :---: |
| F_close (test) | 42 | 0.095144 |
| F_close (test) | 123 | 0.103667 |
| F_close (test) | 2026 | 0.105491 |
| F_tgd2 (test) | 42 | 0.474660 |
| F_tgd2 (test) | 123 | 0.490981 |
| F_tgd2 (test) | 2026 | 0.520732 |
| R_theta_tgd2 | 42 | 0.011360 |
| R_theta_tgd2 | 123 | 0.006741 |
| R_theta_tgd2 | 2026 | 0.011483 |
| R_state_tgd2 | 42 | 0.107486 |
| R_state_tgd2 | 123 | 0.111559 |
| R_state_tgd2 | 2026 | 0.103669 |
| G_CN_over_TGD2 (test) | 42 | 0.046286 |
| G_CN_over_TGD2 (test) | 123 | 0.047381 |
| G_CN_over_TGD2 (test) | 2026 | 0.046928 |
| decay_G_base | 42 | 0.002180 |
| decay_G_base | 123 | 0.002320 |
| decay_G_base | 2026 | 0.002981 |

### Block 4 — Output–internal association boundary

| Pair | Regime | Seed | Raw ρ | Partial ρ (controlling frac_snow) |
| :--- | :---: | :---: | :---: | :---: |
| G_base|C_theta_primary | IC |  | 0.356 | -0.078 |
| G_base|C_theta_primary | dPL | 42 | 0.782 | 0.112 |
| G_base|C_theta_primary | dPL | 123 | 0.771 | 0.097 |
| G_base|C_theta_primary | dPL | 2026 | 0.770 | 0.081 |
| G_base|C_state_primary | IC |  | 0.661 | 0.005 |
| G_base|C_state_primary | dPL | 42 | 0.822 | -0.021 |
| G_base|C_state_primary | dPL | 123 | 0.817 | -0.056 |
| G_base|C_state_primary | dPL | 2026 | 0.823 | -0.007 |
| C_theta vs frac_snow | IC |  | 0.496 | - |
| C_state vs frac_snow | IC |  | 0.823 | - |
| C_theta vs frac_snow | dPL | 42 | 0.881 | - |
| C_state vs frac_snow | dPL | 42 | 0.963 | - |
| C_theta vs frac_snow | dPL | 123 | 0.873 | - |
| C_state vs frac_snow | dPL | 123 | 0.965 | - |
| C_theta vs frac_snow | dPL | 2026 | 0.880 | - |
| C_state vs frac_snow | dPL | 2026 | 0.967 | - |

### Block 5 — Aggregate definitions

| Estimand | Regime | Definition |
| :--- | :---: | :--- |
| C_theta definition | IC | median over frozen primary params of |e_M - e_CN|; IC primary = ['xaj_k']; secondary = ['xaj_b'] |
| C_theta definition | dPL | median over frozen primary params of |e_M - e_CN|; dPL primary = ['xaj_k', 'xaj_theta', 'xaj_um', 'xaj_cg', 'xaj_ci']; secondary = ['xaj_b', 'xaj_lm', 'xaj_c', 'xaj_sm', 'xaj_ki'] |
| C_state definition | IC/dPL | median over primary states of delta_NRMSE (test); primary = ['wu', 'wl', 's', 'qi', 'qg']; secondary = ['wd'] (wd excluded: correct-CN recovery of wd is already poor (gate NRMSE ~1.5-2.5), so it is secondary); derived total tension storage wt = wu+wl+wd (wt = wu + wl + wd (total tension-water storage)) |

*Note*: Values report basin-level medians; dPL rows aggregate the three seeds (42/123/2026) to per-basin seed medians before summarising, exactly as in Figures 5–6. 95% CIs are paired-basin bootstrap (2000 replicates, seed 20260730). Block 2 medians and Block 3 seed medians are asserted equal to the frozen post-hoc summaries (1e-6 / 1e-9). In Block 4, the raw output-recovery/internal-cost association largely disappears after controlling for frac_snow (partial ρ ≈ 0), i.e. the relationship is jointly organized by snow-process activity rather than an independent trade-off. D_theta is the correct-CN parameter-recovery dispersion (gate equifinality, KGE ≥ 0.99 basins).