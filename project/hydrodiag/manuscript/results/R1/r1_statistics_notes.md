# R1 Statistics Notes

- This extension reuses existing daily exports and existing partition summaries; it launches no inference, training, calibration, preprocessing, or plotting.
- The main metric is the repository's standard KGE(Q), using the standard-deviation ratio `alpha=std_sim/std_obs`; no KGE-prime relabeling is applied.
- Full mode computes metrics from aligned daily observed/simulated arrays during inference; statistics-only mode recomputes them from daily exports with matched valid-day masks.
- dPL seed-by-basin records remain separate; primary dPL effects summarize seed-specific basin effects by median.
- Generalization exposure is `(enhanced_test - base_test) - (enhanced_train - base_train)` for CN-Base, TGD-Base, and CN-TGD. IC-dPL transfer is `IC - median_seed(dPL)` for Base, TGD, and CN.
- Snow interaction uses `effect ~ frac_snow + paradigm_dPL + frac_snow:paradigm_dPL`, with IC-CMA-ES as reference and basin-clustered standard errors.
- CT and AMJJ effects use `|E_first| - |E_second|`, with primary five-year complete-water-year matching and three-year sensitivity.
- Bootstrap uses 10,000 basin resamples and fixed seed `20260730`; region block bootstrap uses seven authoritative LORO regions from `data/basin_groups/group_11.npy` through `group_17.npy`.
- dPL robustness is evaluated independently within seeds `42`, `123`, and `2026`; IC restart robustness uses stored restart KGE and does not select by test performance. IC signature restart sensitivity is unavailable without non-selected-restart daily series.
- SPO is excluded from active R1 under `excluded_from_R1_incomplete_prespecified_definition`; daily simulation data exist, but the prespecified definition is incomplete. CT is the primary timing signature and AMJJ is the secondary seasonal-volume signature.

## R1 full-basin statistical summary

All rows below are full-sample aggregates; the inferential unit is the basin and `n` is the valid matched basin count.

### Absolute performance

| paradigm | model | train KGE median [IQR] | test KGE median [IQR] | train-test gap median [95% CI] | n |
|---|---|---:|---:|---:|---:|
| IC-CMA-ES | XAJ-Base | 0.813 [0.652, 0.874] | 0.670 [0.489, 0.775] | 0.086 [0.071, 0.099] | 531 |
| IC-CMA-ES | XAJ-TGD | 0.831 [0.751, 0.889] | 0.712 [0.595, 0.792] | 0.096 [0.084, 0.110] | 531 |
| IC-CMA-ES | XAJ-CN | 0.873 [0.831, 0.907] | 0.754 [0.644, 0.829] | 0.108 [0.096, 0.120] | 531 |
| dPL-MLP | XAJ-Base | 0.733 [0.566, 0.834] | 0.686 [0.513, 0.786] | 0.033 [0.020, 0.040] | 531 |
| dPL-MLP | XAJ-TGD | 0.773 [0.675, 0.848] | 0.721 [0.608, 0.793] | 0.045 [0.035, 0.053] | 531 |
| dPL-MLP | XAJ-CN | 0.825 [0.739, 0.870] | 0.763 [0.647, 0.822] | 0.055 [0.045, 0.061] | 531 |
| dPL-MLP | HBV | 0.784 [0.688, 0.838] | 0.731 [0.629, 0.796] | NA | 531 |

### Structural effects

| paradigm | estimand | period | median effect [95% CI] | positive fraction | n | support status |
|---|---|---|---:|---:|---:|---|
| IC-CMA-ES | KGE_CN - KGE_Base | train | 0.022 [0.017, 0.032] | 0.829 | 531 | supported_positive |
| IC-CMA-ES | KGE_CN - KGE_Base | test | 0.024 [0.016, 0.037] | 0.731 | 531 | supported_positive |
| IC-CMA-ES | KGE_TGD - KGE_Base | train | 0.014 [0.012, 0.019] | 0.825 | 531 | supported_positive |
| IC-CMA-ES | KGE_TGD - KGE_Base | test | 0.015 [0.011, 0.019] | 0.723 | 531 | supported_positive |
| IC-CMA-ES | KGE_CN - KGE_TGD | train | 0.010 [0.007, 0.015] | 0.750 | 531 | supported_positive |
| IC-CMA-ES | KGE_CN - KGE_TGD | test | 0.010 [0.006, 0.013] | 0.637 | 531 | supported_positive |
| dPL-MLP | KGE_CN - KGE_Base | train | 0.029 [0.023, 0.036] | 0.821 | 531 | supported_positive |
| dPL-MLP | KGE_CN - KGE_Base | test | 0.016 [0.011, 0.023] | 0.657 | 531 | supported_positive |
| dPL-MLP | KGE_TGD - KGE_Base | train | 0.019 [0.016, 0.021] | 0.834 | 531 | supported_positive |
| dPL-MLP | KGE_TGD - KGE_Base | test | 0.011 [0.007, 0.015] | 0.687 | 531 | supported_positive |
| dPL-MLP | KGE_CN - KGE_TGD | train | 0.010 [0.007, 0.013] | 0.689 | 531 | supported_positive |
| dPL-MLP | KGE_CN - KGE_TGD | test | 0.008 [0.005, 0.012] | 0.606 | 531 | supported_positive |

### Generalization and paradigm transfer

| estimand | paradigm or model | median effect [95% CI] | positive fraction | n | support status |
|---|---|---:|---:|---:|---|
| E_CN-Base | IC-CMA-ES | -0.008 [-0.011, -0.005] | 0.373 | 531 | supported_negative |
| E_CN-Base | dPL-MLP | -0.014 [-0.018, -0.010] | 0.320 | 531 | supported_negative |
| E_TGD-Base | IC-CMA-ES | -0.004 [-0.006, -0.001] | 0.414 | 531 | supported_negative |
| E_TGD-Base | dPL-MLP | -0.008 [-0.010, -0.006] | 0.326 | 531 | supported_negative |
| E_CN-TGD | IC-CMA-ES | -0.004 [-0.006, -0.002] | 0.418 | 531 | supported_negative |
| E_CN-TGD | dPL-MLP | -0.006 [-0.009, -0.003] | 0.411 | 531 | supported_negative |
| (KGE_IC,test-KGE_dPL,test)-(KGE_IC,train-KGE_dPL,train) | IC-dPL / XAJ-Base | -0.030 [-0.035, -0.025] | 0.207 | 531 | supported_negative |
| (KGE_IC,test-KGE_dPL,test)-(KGE_IC,train-KGE_dPL,train) | IC-dPL / XAJ-TGD | -0.027 [-0.035, -0.022] | 0.228 | 531 | supported_negative |
| (KGE_IC,test-KGE_dPL,test)-(KGE_IC,train-KGE_dPL,train) | IC-dPL / XAJ-CN | -0.033 [-0.038, -0.028] | 0.226 | 531 | supported_negative |

### Snow specificity

| paradigm | CN-TGD test median [95% CI] | positive fraction | Spearman rho | robust slope [95% CI] | n |
|---|---:|---:|---:|---:|---:|
| IC-CMA-ES | 0.010 [0.006, 0.013] | 0.637 | 0.535 | 0.268 [0.231, 0.307] | 531 |
| dPL-MLP | 0.008 [0.005, 0.012] | 0.606 | 0.462 | 0.218 [0.184, 0.251] | 531 |

Interaction model coefficients:

| coefficient | estimate | standard error | 95% CI | p-value | reference | n |
|---|---:|---:|---:|---:|---|---:|
| eta_0 | -0.009 | 0.007 | [-0.024, 0.005] | 0.199 | IC-CMA-ES | 531 |
| eta_1_frac_snow | 0.268 | 0.030 | [0.209, 0.326] | 0.000 | IC-CMA-ES | 531 |
| eta_2_paradigm_dPL | 0.001 | 0.009 | [-0.017, 0.019] | 0.895 | IC-CMA-ES | 531 |
| eta_3_frac_snow_x_paradigm_dPL | -0.008 | 0.020 | [-0.047, 0.031] | 0.680 | IC-CMA-ES | 531 |

### Snowmelt signatures

| paradigm | signature | comparison | median error reduction [95% CI] | positive fraction | n | support status |
|---|---|---|---:|---:|---:|---|
| IC-CMA-ES | AMJJ | R_CN-Base | 0.011 [0.008, 0.014] | 0.729 | 531 | supported_positive |
| IC-CMA-ES | AMJJ | R_CN-TGD | 0.004 [0.003, 0.005] | 0.637 | 531 | supported_positive |
| IC-CMA-ES | AMJJ | R_TGD-Base | 0.005 [0.003, 0.006] | 0.674 | 531 | supported_positive |
| IC-CMA-ES | CT | R_CN-Base | 2.000 [1.000, 3.000] | 0.621 | 531 | supported_positive |
| IC-CMA-ES | CT | R_CN-TGD | 1.000 [0.000, 1.000] | 0.522 | 531 | inconclusive |
| IC-CMA-ES | CT | R_TGD-Base | 1.000 [1.000, 2.000] | 0.573 | 531 | supported_positive |
| dPL-MLP | AMJJ | R_CN-Base | 0.007 [0.005, 0.009] | 0.716 | 531 | supported_positive |
| dPL-MLP | AMJJ | R_CN-TGD | 0.002 [0.001, 0.003] | 0.576 | 531 | supported_positive |
| dPL-MLP | AMJJ | R_TGD-Base | 0.003 [0.002, 0.004] | 0.676 | 531 | supported_positive |
| dPL-MLP | CT | R_CN-Base | 1.000 [1.000, 1.000] | 0.554 | 531 | supported_positive |
| dPL-MLP | CT | R_CN-TGD | 0.000 [0.000, 1.000] | 0.469 | 531 | inconclusive |
| dPL-MLP | CT | R_TGD-Base | 1.000 [0.000, 1.000] | 0.507 | 531 | inconclusive |

### frac_snow fixed strata

Project-fixed strata: S1 `[0, 0.05)`, S2 `[0.05, 0.15)`, S3 `[0.15, 0.30)`, S4 `[0.30, 0.50)`, S5 `[0.50, 1.00]`.

| stratum | n | IC Base test KGE | IC TGD test KGE | IC CN test KGE | dPL Base test KGE | dPL TGD test KGE | dPL CN test KGE |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 [0, 0.05) | 165 | 0.746 [0.729, 0.772] | 0.750 [0.725, 0.777] | 0.748 [0.716, 0.775] | 0.769 [0.743, 0.790] | 0.770 [0.734, 0.790] | 0.760 [0.731, 0.784] |
| S2 [0.05, 0.15) | 156 | 0.722 [0.700, 0.748] | 0.743 [0.725, 0.759] | 0.746 [0.731, 0.765] | 0.738 [0.720, 0.757] | 0.751 [0.727, 0.763] | 0.754 [0.721, 0.776] |
| S3 [0.15, 0.30) | 121 | 0.621 [0.583, 0.648] | 0.687 [0.667, 0.706] | 0.753 [0.722, 0.781] | 0.629 [0.606, 0.664] | 0.686 [0.664, 0.714] | 0.743 [0.716, 0.768] |
| S4 [0.30, 0.50) | 34 | 0.502 [0.298, 0.607] | 0.647 [0.589, 0.688] | 0.748 [0.676, 0.807] | 0.514 [0.360, 0.604] | 0.627 [0.557, 0.673] | 0.693 [0.648, 0.787] |
| S5 [0.50, 1.00] | 55 | 0.099 [0.066, 0.223] | 0.634 [0.590, 0.680] | 0.814 [0.791, 0.844] | 0.094 [0.073, 0.214] | 0.653 [0.579, 0.680] | 0.804 [0.780, 0.825] |

| stratum | paradigm | CN-Base test effect | TGD-Base test effect | CN-TGD test effect |
|---|---|---:|---:|---:|
| S1 | IC-CMA-ES | 0.000 [-0.003, 0.002] | 0.000 [-0.000, 0.003] | -0.002 [-0.004, -0.000] |
| S1 | dPL-MLP | -0.003 [-0.006, -0.000] | 0.000 [-0.002, 0.002] | -0.001 [-0.005, 0.001] |
| S2 | IC-CMA-ES | 0.013 [0.008, 0.019] | 0.010 [0.006, 0.014] | 0.005 [0.002, 0.010] |
| S2 | dPL-MLP | 0.006 [0.001, 0.013] | 0.005 [0.003, 0.010] | 0.003 [-0.002, 0.007] |
| S3 | IC-CMA-ES | 0.112 [0.094, 0.129] | 0.036 [0.026, 0.046] | 0.067 [0.055, 0.080] |
| S3 | dPL-MLP | 0.080 [0.065, 0.097] | 0.031 [0.024, 0.037] | 0.035 [0.022, 0.046] |
| S4 | IC-CMA-ES | 0.188 [0.128, 0.327] | 0.046 [0.026, 0.266] | 0.091 [0.012, 0.132] |
| S4 | dPL-MLP | 0.195 [0.158, 0.227] | 0.038 [0.031, 0.165] | 0.103 [0.043, 0.129] |
| S5 | IC-CMA-ES | 0.670 [0.566, 0.755] | 0.491 [0.414, 0.600] | 0.163 [0.106, 0.212] |
| S5 | dPL-MLP | 0.642 [0.586, 0.734] | 0.531 [0.425, 0.587] | 0.145 [0.113, 0.211] |

### Robustness

| effect | primary direction | IC restart agreement | dPL seed agreement | basin-bootstrap conclusion | region-block-bootstrap conclusion |
|---|---|---|---|---|---|
| test CN-Base | supported_positive | yes | yes | supported_positive | supported_positive |
| test CN-TGD | supported_positive | yes | yes | supported_positive | supported_positive |
| E_CN-Base | supported_negative | yes | yes | supported_negative | supported_negative |
| E_CN-TGD | supported_negative | yes | yes | supported_negative | supported_negative |
| CT CN-TGD | inconclusive | no | yes | inconclusive | inconclusive |
| AMJJ CN-TGD | supported_positive | no | yes | supported_positive | supported_positive |

### Machine-generated claim status

| claim ID | statistical status | primary estimate | 95% CI | valid basin count |
|---|---|---:|---:|---:|
| amjj_repair_cn_base | supported_positive | 0.011 | [0.008, 0.014] | 531 |
| amjj_repair_cn_base | supported_positive | 0.007 | [0.005, 0.009] | 531 |
| amjj_repair_cn_tgd | supported_positive | 0.004 | [0.003, 0.005] | 531 |
| amjj_repair_cn_tgd | supported_positive | 0.002 | [0.001, 0.003] | 531 |
| amjj_repair_tgd_base | supported_positive | 0.005 | [0.003, 0.006] | 531 |
| amjj_repair_tgd_base | supported_positive | 0.003 | [0.002, 0.004] | 531 |
| calibration_masking_cn_base | supported_positive | 0.022 | [0.017, 0.032] | 531 |
| calibration_masking_cn_base | supported_positive | 0.029 | [0.023, 0.036] | 531 |
| calibration_masking_cn_tgd | supported_positive | 0.010 | [0.007, 0.015] | 531 |
| calibration_masking_cn_tgd | supported_positive | 0.010 | [0.007, 0.013] | 531 |
| ct_repair_cn_base | supported_positive | 2.000 | [1.000, 3.000] | 531 |
| ct_repair_cn_base | supported_positive | 1.000 | [1.000, 1.000] | 531 |
| ct_repair_cn_tgd | inconclusive | 1.000 | [0.000, 1.000] | 531 |
| ct_repair_cn_tgd | inconclusive | 0.000 | [0.000, 1.000] | 531 |
| ct_repair_tgd_base | supported_positive | 1.000 | [1.000, 2.000] | 531 |
| ct_repair_tgd_base | inconclusive | 1.000 | [0.000, 1.000] | 531 |
| generalization_exposure_cn_base | supported_negative | -0.008 | [-0.011, -0.005] | 531 |
| generalization_exposure_cn_base | supported_negative | -0.014 | [-0.018, -0.010] | 531 |
| generalization_exposure_cn_tgd | supported_negative | -0.004 | [-0.006, -0.002] | 531 |
| generalization_exposure_cn_tgd | supported_negative | -0.006 | [-0.009, -0.003] | 531 |
| generalization_exposure_tgd_base | supported_negative | -0.004 | [-0.006, -0.001] | 531 |
| generalization_exposure_tgd_base | supported_negative | -0.008 | [-0.010, -0.006] | 531 |
| ic_dpl_transfer_base | supported_negative | -0.030 | [-0.035, -0.025] | 531 |
| ic_dpl_transfer_cn | supported_negative | -0.033 | [-0.038, -0.028] | 531 |
| ic_dpl_transfer_tgd | supported_negative | -0.027 | [-0.035, -0.022] | 531 |
| snow_specificity_cn_tgd | supported_positive | 0.010 | [0.006, 0.013] | 531 |
| snow_specificity_cn_tgd | supported_positive | 0.008 | [0.005, 0.012] | 531 |
| structural_test_cn_base | supported_positive | 0.024 | [0.016, 0.037] | 531 |
| structural_test_cn_base | supported_positive | 0.016 | [0.011, 0.023] | 531 |
| structural_test_tgd_base | supported_positive | 0.015 | [0.011, 0.019] | 531 |
| structural_test_tgd_base | supported_positive | 0.014 | [0.012, 0.019] | 531 |
| structural_test_tgd_base | supported_positive | 0.011 | [0.007, 0.015] | 531 |
| structural_test_tgd_base | supported_positive | 0.019 | [0.016, 0.021] | 531 |
| train_test_gap | supported_positive | 0.086 | [0.071, 0.099] | 531 |
| train_test_gap | supported_positive | 0.096 | [0.084, 0.110] | 531 |
| train_test_gap | supported_positive | 0.108 | [0.096, 0.120] | 531 |
| train_test_gap | supported_positive | 0.033 | [0.020, 0.040] | 531 |
| train_test_gap | supported_positive | 0.045 | [0.035, 0.053] | 531 |
| train_test_gap | supported_positive | 0.055 | [0.045, 0.061] | 531 |

## TGD transfer and snow-stratified exposure checks

All quantities are basin-wise; no individual basins are listed.

### TGD IC versus dPL by snow stratum

| snow stratum | n | IC train KGE | dPL train KGE | train advantage A [95% CI] | IC test KGE | dPL test KGE | test advantage B [95% CI] | transfer D [95% CI] | support status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S1 | 165 | 0.883 | 0.817 | 0.052 [0.043, 0.067] | 0.750 | 0.770 | -0.002 [-0.016, 0.008] | -0.048 [-0.071, -0.030] | supported_negative |
| S2 | 156 | 0.856 | 0.813 | 0.032 [0.028, 0.040] | 0.743 | 0.751 | -0.000 [-0.005, 0.006] | -0.025 [-0.037, -0.018] | supported_negative |
| S3 | 121 | 0.755 | 0.724 | 0.027 [0.023, 0.032] | 0.687 | 0.686 | -0.003 [-0.009, 0.005] | -0.027 [-0.044, -0.019] | supported_negative |
| S4 | 34 | 0.767 | 0.701 | 0.036 [0.017, 0.072] | 0.647 | 0.627 | 0.013 [0.006, 0.028] | -0.012 [-0.044, -0.001] | supported_negative |
| S5 | 55 | 0.738 | 0.698 | 0.023 [0.017, 0.035] | 0.634 | 0.653 | 0.009 [0.000, 0.020] | -0.017 [-0.021, -0.011] | supported_negative |

### Transfer-loss snow gradient

| structure | Spearman rho | robust slope [95% CI] | OLS slope [95% CI] | n |
|---|---:|---:|---:|---:|
| XAJ-Base | 0.228 | 0.073 [0.028, 0.246] | 0.103 [0.023, 0.183] | 531 |
| XAJ-CN | 0.107 | 0.039 [-0.058, 0.170] | 0.017 [-0.102, 0.135] | 531 |
| XAJ-TGD | 0.152 | 0.046 [0.024, 0.200] | 0.089 [0.025, 0.154] | 531 |

Transfer-gradient interaction coefficients (TGD reference; basin-clustered standard errors):

| coefficient | estimate | SE | 95% CI | p-value | n |
|---|---:|---:|---:|---:|---:|
| beta0_intercept | -0.093 | 0.009 | [-0.110, -0.076] | 0.000 | 531 |
| beta1_frac_snow_TGD | 0.089 | 0.021 | [0.047, 0.131] | 0.000 | 531 |
| beta2_structure_Base | -0.012 | 0.007 | [-0.025, 0.001] | 0.076 | 531 |
| beta2_structure_CN | 0.005 | 0.009 | [-0.013, 0.023] | 0.585 | 531 |
| beta3_frac_snow_x_Base | 0.014 | 0.027 | [-0.038, 0.066] | 0.603 | 531 |
| beta3_frac_snow_x_CN | -0.073 | 0.020 | [-0.113, -0.033] | 0.000 | 531 |

### Snow-stratified structural effects

| paradigm | stratum | period | Base / TGD / CN KGE | CN-Base / TGD-Base / CN-TGD effects |
|---|---|---|---:|---:|
| IC-CMA-ES | S1 | train | 0.879 / 0.883 / 0.883 | 0.001 [0.000, 0.002] / 0.002 [0.001, 0.003] / 0.000 [-0.001, 0.001] |
| IC-CMA-ES | S1 | test | 0.746 / 0.750 / 0.748 | 0.000 [-0.003, 0.002] / 0.000 [-0.000, 0.003] / -0.002 [-0.004, -0.000] |
| dPL-MLP | S1 | train | 0.812 / 0.817 / 0.816 | 0.006 [0.005, 0.008] / 0.006 [0.004, 0.008] / 0.000 [-0.001, 0.002] |
| dPL-MLP | S1 | test | 0.769 / 0.770 / 0.760 | -0.003 [-0.006, -0.000] / 0.000 [-0.002, 0.002] / -0.001 [-0.005, 0.001] |
| IC-CMA-ES | S2 | train | 0.843 / 0.856 / 0.866 | 0.019 [0.014, 0.022] / 0.011 [0.009, 0.013] / 0.007 [0.006, 0.010] |
| IC-CMA-ES | S2 | test | 0.722 / 0.743 / 0.746 | 0.013 [0.008, 0.019] / 0.010 [0.006, 0.014] / 0.005 [0.002, 0.010] |
| dPL-MLP | S2 | train | 0.798 / 0.813 / 0.828 | 0.018 [0.012, 0.023] / 0.011 [0.009, 0.014] / 0.005 [0.003, 0.009] |
| dPL-MLP | S2 | test | 0.738 / 0.751 / 0.754 | 0.006 [0.001, 0.013] / 0.005 [0.003, 0.010] / 0.003 [-0.002, 0.007] |
| IC-CMA-ES | S3 | train | 0.689 / 0.755 / 0.869 | 0.157 [0.141, 0.173] / 0.047 [0.038, 0.061] / 0.100 [0.087, 0.111] |
| IC-CMA-ES | S3 | test | 0.621 / 0.687 / 0.753 | 0.112 [0.094, 0.129] / 0.036 [0.026, 0.046] / 0.067 [0.055, 0.080] |
| dPL-MLP | S3 | train | 0.664 / 0.724 / 0.835 | 0.148 [0.124, 0.176] / 0.037 [0.030, 0.048] / 0.100 [0.075, 0.114] |
| dPL-MLP | S3 | test | 0.629 / 0.686 / 0.743 | 0.080 [0.065, 0.097] / 0.031 [0.024, 0.037] / 0.035 [0.022, 0.046] |
| IC-CMA-ES | S4 | train | 0.564 / 0.767 / 0.855 | 0.241 [0.127, 0.374] / 0.103 [0.033, 0.318] / 0.093 [0.053, 0.108] |
| IC-CMA-ES | S4 | test | 0.502 / 0.647 / 0.748 | 0.188 [0.128, 0.327] / 0.046 [0.026, 0.266] / 0.091 [0.012, 0.132] |
| dPL-MLP | S4 | train | 0.512 / 0.701 / 0.767 | 0.267 [0.141, 0.402] / 0.149 [0.037, 0.375] / 0.075 [0.035, 0.094] |
| dPL-MLP | S4 | test | 0.514 / 0.627 / 0.693 | 0.195 [0.158, 0.227] / 0.038 [0.031, 0.165] / 0.103 [0.043, 0.129] |
| IC-CMA-ES | S5 | train | 0.124 / 0.738 / 0.913 | 0.765 [0.651, 0.807] / 0.559 [0.458, 0.637] / 0.183 [0.128, 0.241] |
| IC-CMA-ES | S5 | test | 0.099 / 0.634 / 0.814 | 0.670 [0.566, 0.755] / 0.491 [0.414, 0.600] / 0.163 [0.106, 0.212] |
| dPL-MLP | S5 | train | 0.105 / 0.698 / 0.821 | 0.655 [0.603, 0.720] / 0.571 [0.466, 0.617] / 0.135 [0.046, 0.189] |
| dPL-MLP | S5 | test | 0.094 / 0.653 / 0.804 | 0.642 [0.586, 0.734] / 0.531 [0.425, 0.587] / 0.145 [0.113, 0.211] |

### Snow-stratified exposure

| paradigm | stratum | estimand | median [95% CI] | positive fraction | n | support status |
|---|---|---|---:|---:|---:|---|
| IC-CMA-ES | S1 | E_CN-Base | -0.002 [-0.004, 0.000] | 0.442 | 165 | inconclusive |
| IC-CMA-ES | S1 | E_CN-TGD | -0.002 [-0.004, -0.000] | 0.424 | 165 | supported_negative |
| IC-CMA-ES | S1 | E_TGD-Base | -0.000 [-0.002, 0.001] | 0.473 | 165 | inconclusive |
| IC-CMA-ES | S2 | E_CN-Base | -0.005 [-0.008, 0.000] | 0.423 | 156 | inconclusive |
| IC-CMA-ES | S2 | E_CN-TGD | -0.003 [-0.007, 0.000] | 0.423 | 156 | inconclusive |
| IC-CMA-ES | S2 | E_TGD-Base | -0.000 [-0.004, 0.002] | 0.494 | 156 | inconclusive |
| IC-CMA-ES | S3 | E_CN-Base | -0.047 [-0.061, -0.031] | 0.248 | 121 | supported_negative |
| IC-CMA-ES | S3 | E_CN-TGD | -0.031 [-0.042, -0.014] | 0.306 | 121 | supported_negative |
| IC-CMA-ES | S3 | E_TGD-Base | -0.009 [-0.019, -0.005] | 0.339 | 121 | supported_negative |
| IC-CMA-ES | S4 | E_CN-Base | -0.024 [-0.071, 0.026] | 0.441 | 34 | inconclusive |
| IC-CMA-ES | S4 | E_CN-TGD | 0.016 [-0.011, 0.031] | 0.618 | 34 | inconclusive |
| IC-CMA-ES | S4 | E_TGD-Base | -0.008 [-0.041, -0.000] | 0.324 | 34 | supported_negative |
| IC-CMA-ES | S5 | E_CN-Base | -0.064 [-0.094, -0.036] | 0.255 | 55 | supported_negative |
| IC-CMA-ES | S5 | E_CN-TGD | 0.001 [-0.017, 0.020] | 0.509 | 55 | inconclusive |
| IC-CMA-ES | S5 | E_TGD-Base | -0.026 [-0.058, -0.019] | 0.236 | 55 | supported_negative |
| dPL-MLP | S1 | E_CN-Base | -0.008 [-0.014, -0.005] | 0.279 | 165 | supported_negative |
| dPL-MLP | S1 | E_CN-TGD | -0.003 [-0.006, 0.001] | 0.455 | 165 | inconclusive |
| dPL-MLP | S1 | E_TGD-Base | -0.008 [-0.011, -0.005] | 0.297 | 165 | supported_negative |
| dPL-MLP | S2 | E_CN-Base | -0.010 [-0.015, -0.005] | 0.372 | 156 | supported_negative |
| dPL-MLP | S2 | E_CN-TGD | -0.006 [-0.010, -0.002] | 0.391 | 156 | supported_negative |
| dPL-MLP | S2 | E_TGD-Base | -0.006 [-0.008, -0.002] | 0.353 | 156 | supported_negative |
| dPL-MLP | S3 | E_CN-Base | -0.060 [-0.082, -0.036] | 0.223 | 121 | supported_negative |
| dPL-MLP | S3 | E_CN-TGD | -0.046 [-0.054, -0.028] | 0.256 | 121 | supported_negative |
| dPL-MLP | S3 | E_TGD-Base | -0.011 [-0.018, -0.003] | 0.347 | 121 | supported_negative |
| dPL-MLP | S4 | E_CN-Base | -0.018 [-0.077, 0.029] | 0.471 | 34 | inconclusive |
| dPL-MLP | S4 | E_CN-TGD | 0.019 [-0.022, 0.037] | 0.588 | 34 | inconclusive |
| dPL-MLP | S4 | E_TGD-Base | -0.008 [-0.040, 0.001] | 0.353 | 34 | inconclusive |
| dPL-MLP | S5 | E_CN-Base | -0.024 [-0.039, 0.003] | 0.418 | 55 | inconclusive |
| dPL-MLP | S5 | E_CN-TGD | 0.031 [-0.009, 0.058] | 0.564 | 55 | inconclusive |
| dPL-MLP | S5 | E_TGD-Base | -0.036 [-0.050, -0.010] | 0.273 | 55 | supported_negative |

### CT and AMJJ

| paradigm | signature | analysis set | comparison | median [95% CI] | region-block interval | positive fraction | n | support status |
|---|---|---|---|---:|---:|---:|---:|---|
| IC-CMA-ES | AMJJ | r1_remaining_signature_effect | R_CN-Base | 0.011 [0.009, 0.013] | [0.004, 0.022] | 0.748 | 531 | supported_positive |
| IC-CMA-ES | AMJJ | r1_remaining_signature_effect | R_CN-Base | 0.011 [0.008, 0.014] | [0.004, 0.023] | 0.729 | 531 | supported_positive |
| IC-CMA-ES | AMJJ | r1_remaining_signature_effect | R_CN-TGD | 0.005 [0.003, 0.006] | [0.002, 0.009] | 0.685 | 531 | supported_positive |
| IC-CMA-ES | AMJJ | r1_remaining_signature_effect | R_CN-TGD | 0.004 [0.003, 0.005] | [0.001, 0.007] | 0.637 | 531 | supported_positive |
| IC-CMA-ES | AMJJ | r1_remaining_signature_effect | R_TGD-Base | 0.005 [0.003, 0.006] | [0.002, 0.009] | 0.697 | 531 | supported_positive |
| IC-CMA-ES | AMJJ | r1_remaining_signature_effect | R_TGD-Base | 0.005 [0.003, 0.006] | [0.002, 0.008] | 0.674 | 531 | supported_positive |
| IC-CMA-ES | CT | r1_remaining_signature_effect | R_CN-Base | 2.500 [2.000, 3.500] | [1.000, 6.500] | 0.701 | 531 | supported_positive |
| IC-CMA-ES | CT | r1_remaining_signature_effect | R_CN-Base | 2.000 [1.000, 3.000] | [1.000, 5.000] | 0.621 | 531 | supported_positive |
| IC-CMA-ES | CT | r1_remaining_signature_effect | R_CN-TGD | 1.000 [1.000, 1.500] | [0.500, 2.000] | 0.625 | 531 | supported_positive |
| IC-CMA-ES | CT | r1_remaining_signature_effect | R_CN-TGD | 1.000 [0.000, 1.000] | [0.000, 2.000] | 0.522 | 531 | inconclusive |
| IC-CMA-ES | CT | r1_remaining_signature_effect | R_TGD-Base | 1.500 [1.000, 2.000] | [0.000, 2.500] | 0.640 | 531 | supported_positive |
| IC-CMA-ES | CT | r1_remaining_signature_effect | R_TGD-Base | 1.000 [1.000, 2.000] | [0.000, 3.000] | 0.573 | 531 | supported_positive |
| dPL-MLP | AMJJ | r1_remaining_signature_effect | R_CN-Base | 0.007 [0.004, 0.009] | [0.002, 0.017] | 0.712 | 531 | supported_positive |
| dPL-MLP | AMJJ | r1_remaining_signature_effect | R_CN-Base | 0.007 [0.005, 0.009] | [0.002, 0.014] | 0.718 | 531 | supported_positive |
| dPL-MLP | AMJJ | r1_remaining_signature_effect | R_CN-TGD | 0.003 [0.002, 0.004] | [0.001, 0.007] | 0.614 | 531 | supported_positive |
| dPL-MLP | AMJJ | r1_remaining_signature_effect | R_CN-TGD | 0.002 [0.001, 0.003] | [0.001, 0.004] | 0.578 | 531 | supported_positive |
| dPL-MLP | AMJJ | r1_remaining_signature_effect | R_TGD-Base | 0.003 [0.002, 0.004] | [0.001, 0.005] | 0.670 | 531 | supported_positive |
| dPL-MLP | AMJJ | r1_remaining_signature_effect | R_TGD-Base | 0.003 [0.002, 0.004] | [0.001, 0.006] | 0.672 | 531 | supported_positive |
| dPL-MLP | CT | r1_remaining_signature_effect | R_CN-Base | 2.000 [1.500, 3.000] | [0.500, 5.500] | 0.676 | 531 | supported_positive |
| dPL-MLP | CT | r1_remaining_signature_effect | R_CN-Base | 1.000 [1.000, 1.000] | [0.000, 3.000] | 0.546 | 531 | supported_positive |
| dPL-MLP | CT | r1_remaining_signature_effect | R_CN-TGD | 0.500 [0.500, 1.000] | [0.000, 1.250] | 0.559 | 531 | supported_positive |
| dPL-MLP | CT | r1_remaining_signature_effect | R_CN-TGD | 0.000 [0.000, 1.000] | [0.000, 1.000] | 0.476 | 531 | inconclusive |
| dPL-MLP | CT | r1_remaining_signature_effect | R_TGD-Base | 1.000 [0.500, 1.000] | [0.000, 2.000] | 0.620 | 531 | supported_positive |
| dPL-MLP | CT | r1_remaining_signature_effect | R_TGD-Base | 1.000 [0.000, 1.000] | [0.000, 1.000] | 0.512 | 531 | inconclusive |

### Process-gradient evidence

| paradigm | signature | comparison | Spearman rho | robust slope [95% CI] | n |
|---|---|---|---:|---:|---:|
| IC-CMA-ES | AMJJ | R_CN-Base | 0.659 | 0.243 [0.383, 0.452] | 531 |
| IC-CMA-ES | AMJJ | R_CN-Base | 0.670 | 0.250 [0.383, 0.455] | 531 |
| IC-CMA-ES | AMJJ | R_CN-TGD | 0.387 | 0.056 [0.043, 0.090] | 531 |
| IC-CMA-ES | AMJJ | R_CN-TGD | 0.361 | 0.055 [0.039, 0.085] | 531 |
| IC-CMA-ES | AMJJ | R_TGD-Base | 0.518 | 0.122 [0.311, 0.394] | 531 |
| IC-CMA-ES | AMJJ | R_TGD-Base | 0.516 | 0.126 [0.305, 0.410] | 531 |
| IC-CMA-ES | CT | R_CN-Base | 0.665 | 55.534 [58.860, 85.396] | 531 |
| IC-CMA-ES | CT | R_CN-Base | 0.612 | 50.210 [58.653, 88.837] | 531 |
| IC-CMA-ES | CT | R_CN-TGD | 0.322 | 10.481 [8.425, 18.154] | 531 |
| IC-CMA-ES | CT | R_CN-TGD | 0.307 | 10.765 [10.461, 23.638] | 531 |
| IC-CMA-ES | CT | R_TGD-Base | 0.589 | 33.702 [48.454, 69.380] | 531 |
| IC-CMA-ES | CT | R_TGD-Base | 0.555 | 32.837 [44.815, 67.104] | 531 |
| dPL-MLP | AMJJ | R_CN-Base | 0.679 | 0.199 [0.319, 0.392] | 531 |
| dPL-MLP | AMJJ | R_CN-Base | 0.628 | 0.194 [0.312, 0.410] | 531 |
| dPL-MLP | AMJJ | R_CN-TGD | 0.297 | 0.037 [0.001, 0.043] | 531 |
| dPL-MLP | AMJJ | R_CN-TGD | 0.182 | 0.027 [-0.018, 0.056] | 531 |
| dPL-MLP | AMJJ | R_TGD-Base | 0.530 | 0.080 [0.300, 0.362] | 531 |
| dPL-MLP | AMJJ | R_TGD-Base | 0.535 | 0.091 [0.296, 0.384] | 531 |
| dPL-MLP | CT | R_CN-Base | 0.620 | 43.202 [53.231, 76.248] | 531 |
| dPL-MLP | CT | R_CN-Base | 0.507 | 35.895 [49.363, 82.926] | 531 |
| dPL-MLP | CT | R_CN-TGD | 0.232 | 6.484 [4.595, 15.867] | 531 |
| dPL-MLP | CT | R_CN-TGD | 0.185 | 3.918 [1.633, 21.217] | 531 |
| dPL-MLP | CT | R_TGD-Base | 0.570 | 27.925 [44.337, 62.326] | 531 |
| dPL-MLP | CT | R_TGD-Base | 0.557 | 24.867 [40.417, 62.142] | 531 |

## Reproducibility

Summary mode is deterministic across independent Python processes and `PYTHONHASHSEED` values. Deterministic CSV artifacts use stable row ordering, explicit UTF-8/LF serialization, and round-trip-safe `%.17g` floats. The manifest stores the canonical reproduction command and excludes itself from artifact hashes; `r1_execution.log` remains volatile because it contains timestamps, runtime paths, and git-status snapshots.
