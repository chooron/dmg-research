# R2 最后一轮 robustness checks（未生成正式 Figure 3）

## Scope and estimand
本报告只使用已验证的 `r2_paired_shifts_basin_level.csv`，不重选 IC/dPL 参数、不修改 R2 primary CSV、不读取或修改 production model。所有 slope 使用 basin-level statistical unit、10,000 次 percentile bootstrap、seed `20260730`。15D 距离使用当前 15 个 normalized Base−CN shifts。

## Validation
- full canonical gradient reproduction: `True`; maximum absolute difference to current R2 slope/CI/rho table: `4.441e-16`.
- paired rows: `15930`; full basins per paradigm: `{'IC': 531, 'dPL': 531}`; parameters per basin/paradigm: `15`.
- subset sizes: `{'full_531': 531, 'exclude_S5': 476, 'loo_S1': 366, 'loo_S2': 375, 'loo_S3': 410, 'loo_S4': 497, 'loo_S5': 476}`; all required fields finite: `True`.
- Existing `r2_primary_shift_summary.csv` and other primary R2 files were read only; no Figure 3 was generated.

## 1. Snow-gradient robustness

Full results are in `r2_snow_gradient_robustness.csv`; direction counts are in `r2_snow_gradient_direction_summary.csv`.

| paradigm | parameter | subset | n | slope | 95% CI | rho | direction | change vs full |
|---|---|---|---:|---:|---|---:|---|---:|
| IC | ci | exclude_S5 | 476 | -0.5483 | [-0.9083, -0.2018] | -0.1396 | negative | -0.1344 |
| IC | ci | full_531 | 531 | -0.4138 | [-0.6366, -0.1943] | -0.1832 | negative | 0.0000 |
| IC | ci | loo_S1 | 366 | -0.4080 | [-0.6657, -0.1433] | -0.1977 | negative | 0.0058 |
| IC | ci | loo_S2 | 375 | -0.3962 | [-0.6171, -0.1730] | -0.2028 | negative | 0.0176 |
| IC | ci | loo_S3 | 410 | -0.4136 | [-0.6355, -0.1890] | -0.1747 | negative | 0.0003 |
| IC | ci | loo_S4 | 497 | -0.3922 | [-0.6246, -0.1618] | -0.1646 | negative | 0.0216 |
| IC | ci | loo_S5 | 476 | -0.5483 | [-0.9034, -0.2030] | -0.1396 | negative | -0.1344 |
| IC | im | exclude_S5 | 476 | -0.1366 | [-0.3540, 0.0699] | -0.1367 | negative | 0.2269 |
| IC | im | full_531 | 531 | -0.3635 | [-0.5169, -0.2067] | -0.2477 | negative | 0.0000 |
| IC | im | loo_S1 | 366 | -0.4124 | [-0.5850, -0.2237] | -0.2443 | negative | -0.0490 |
| IC | im | loo_S2 | 375 | -0.3711 | [-0.5207, -0.2128] | -0.2901 | negative | -0.0076 |
| IC | im | loo_S3 | 410 | -0.3694 | [-0.5241, -0.2076] | -0.2800 | negative | -0.0060 |
| IC | im | loo_S4 | 497 | -0.3879 | [-0.5463, -0.2240] | -0.2482 | negative | -0.0244 |
| IC | im | loo_S5 | 476 | -0.1366 | [-0.3454, 0.0698] | -0.1367 | negative | 0.2269 |
| IC | ki | exclude_S5 | 476 | -0.8647 | [-1.2060, -0.5416] | -0.1942 | negative | -0.3901 |
| IC | ki | full_531 | 531 | -0.4746 | [-0.6562, -0.3131] | -0.2367 | negative | 0.0000 |
| IC | ki | loo_S1 | 366 | -0.4291 | [-0.6387, -0.2372] | -0.2706 | negative | 0.0454 |
| IC | ki | loo_S2 | 375 | -0.4377 | [-0.6156, -0.2731] | -0.2609 | negative | 0.0368 |
| IC | ki | loo_S3 | 410 | -0.4686 | [-0.6450, -0.3062] | -0.2260 | negative | 0.0060 |
| IC | ki | loo_S4 | 497 | -0.4298 | [-0.6083, -0.2644] | -0.1965 | negative | 0.0447 |
| IC | ki | loo_S5 | 476 | -0.8647 | [-1.2003, -0.5333] | -0.1942 | negative | -0.3901 |
| IC | um | exclude_S5 | 476 | 0.5632 | [0.1771, 0.9412] | 0.1494 | positive | 0.0421 |
| IC | um | full_531 | 531 | 0.5211 | [0.3229, 0.7198] | 0.2164 | positive | 0.0000 |
| IC | um | loo_S1 | 366 | 0.4701 | [0.2326, 0.7061] | 0.2001 | positive | -0.0510 |
| IC | um | loo_S2 | 375 | 0.5520 | [0.3512, 0.7583] | 0.2680 | positive | 0.0309 |
| IC | um | loo_S3 | 410 | 0.5253 | [0.3245, 0.7339] | 0.2391 | positive | 0.0042 |
| IC | um | loo_S4 | 497 | 0.5127 | [0.3032, 0.7250] | 0.1929 | positive | -0.0084 |
| IC | um | loo_S5 | 476 | 0.5632 | [0.1926, 0.9368] | 0.1494 | positive | 0.0421 |
| dPL | ci | exclude_S5 | 476 | -0.9815 | [-1.1804, -0.7751] | -0.3392 | negative | 0.2449 |
| dPL | ci | full_531 | 531 | -1.2264 | [-1.2951, -1.1538] | -0.5192 | negative | 0.0000 |
| dPL | ci | loo_S1 | 366 | -1.3619 | [-1.4513, -1.2763] | -0.6385 | negative | -0.1355 |
| dPL | ci | loo_S2 | 375 | -1.2034 | [-1.2741, -1.1306] | -0.6713 | negative | 0.0230 |
| dPL | ci | loo_S3 | 410 | -1.2216 | [-1.2943, -1.1480] | -0.4719 | negative | 0.0048 |
| dPL | ci | loo_S4 | 497 | -1.2375 | [-1.3047, -1.1668] | -0.4759 | negative | -0.0111 |
| dPL | ci | loo_S5 | 476 | -0.9815 | [-1.1788, -0.7630] | -0.3392 | negative | 0.2449 |
| dPL | im | exclude_S5 | 476 | -0.2156 | [-0.3862, -0.0657] | -0.4334 | negative | 0.2367 |
| dPL | im | full_531 | 531 | -0.4523 | [-0.5782, -0.3280] | -0.5685 | negative | 0.0000 |
| dPL | im | loo_S1 | 366 | -0.5260 | [-0.6710, -0.3783] | -0.6291 | negative | -0.0738 |
| dPL | im | loo_S2 | 375 | -0.4546 | [-0.5813, -0.3258] | -0.6532 | negative | -0.0023 |
| dPL | im | loo_S3 | 410 | -0.4706 | [-0.6002, -0.3416] | -0.5493 | negative | -0.0183 |
| dPL | im | loo_S4 | 497 | -0.4580 | [-0.5917, -0.3213] | -0.5321 | negative | -0.0057 |
| dPL | im | loo_S5 | 476 | -0.2156 | [-0.3889, -0.0636] | -0.4334 | negative | 0.2367 |
| dPL | ki | exclude_S5 | 476 | -1.3650 | [-1.6007, -1.1482] | -0.5513 | negative | -0.6247 |
| dPL | ki | full_531 | 531 | -0.7403 | [-0.8644, -0.6240] | -0.6130 | negative | 0.0000 |
| dPL | ki | loo_S1 | 366 | -0.6004 | [-0.7461, -0.4683] | -0.5454 | negative | 0.1399 |
| dPL | ki | loo_S2 | 375 | -0.7141 | [-0.8384, -0.6015] | -0.6762 | negative | 0.0261 |
| dPL | ki | loo_S3 | 410 | -0.6911 | [-0.8067, -0.5835] | -0.5547 | negative | 0.0492 |
| dPL | ki | loo_S4 | 497 | -0.7171 | [-0.8495, -0.5974] | -0.5884 | negative | 0.0232 |
| dPL | ki | loo_S5 | 476 | -1.3650 | [-1.6039, -1.1507] | -0.5513 | negative | -0.6247 |
| dPL | um | exclude_S5 | 476 | 0.7444 | [0.4530, 1.0077] | 0.2526 | positive | -0.2709 |
| dPL | um | full_531 | 531 | 1.0153 | [0.8676, 1.1524] | 0.4182 | positive | 0.0000 |
| dPL | um | loo_S1 | 366 | 1.1790 | [1.0109, 1.3452] | 0.5276 | positive | 0.1637 |
| dPL | um | loo_S2 | 375 | 0.9916 | [0.8467, 1.1316] | 0.4888 | positive | -0.0237 |
| dPL | um | loo_S3 | 410 | 1.0492 | [0.9047, 1.1916] | 0.4486 | positive | 0.0339 |
| dPL | um | loo_S4 | 497 | 0.9825 | [0.8324, 1.1253] | 0.3609 | positive | -0.0328 |
| dPL | um | loo_S5 | 476 | 0.7444 | [0.4637, 1.0054] | 0.2526 | positive | -0.2709 |

### Direction stability

| paradigm | parameter | full direction | matching LOO regimes | fraction |
|---|---|---|---:|---:|
| IC | um | positive | 5 / 5 | 1.00 |
| IC | ki | negative | 5 / 5 | 1.00 |
| IC | ci | negative | 5 / 5 | 1.00 |
| IC | im | negative | 5 / 5 | 1.00 |
| dPL | um | positive | 5 / 5 | 1.00 |
| dPL | ki | negative | 5 / 5 | 1.00 |
| dPL | ci | negative | 5 / 5 | 1.00 |
| dPL | im | negative | 5 / 5 | 1.00 |

Interpretation: all four parameters retain their full-data direction in every leave-one-regime-out result, but im is S5-sensitive: after excluding S5 its IC CI crosses zero, whereas dPL remains negative but attenuates by more than one-half. The report does not use a CI crossing zero as a deletion rule; it records this as partial robustness.

## 2. 15-dimensional parameter-reorganization distance

`D = sqrt(sum_p(delta_p^2))`; `D_rms` is also retained at basin level but does not replace D. Basin-level values are in `r2_15d_distance_basin_level.csv`.

| paradigm | subset | n | median D | IQR D | median 95% CI | slope | slope 95% CI | rho |
|---|---|---:|---:|---|---|---:|---|---:|
| IC | full_531 | 531 | 1.4389 | [1.0491, 1.7924] | [1.3817, 1.4959] | 1.2905 | [1.1216, 1.4643] | 0.4323 |
| dPL | full_531 | 531 | 0.6417 | [0.3902, 1.1363] | [0.5747, 0.6972] | 2.0848 | [1.9672, 2.2124] | 0.7500 |
| IC | exclude_S5 | 476 | 1.3656 | [1.0095, 1.7071] | [1.3194, 1.4295] | 1.5644 | [1.1553, 1.9707] | 0.3038 |
| dPL | exclude_S5 | 476 | 0.5640 | [0.3674, 0.9050] | [0.5192, 0.6264] | 2.6742 | [2.3737, 2.9782] | 0.6671 |
| IC | S1 | 165 | 1.2046 | [0.7461, 1.5155] | [1.0533, 1.2757] | NA | [NA, NA] | NA |
| dPL | S1 | 165 | 0.3902 | [0.2711, 0.5396] | [0.3419, 0.4232] | NA | [NA, NA] | NA |
| IC | S2 | 156 | 1.3277 | [1.0054, 1.7288] | [1.2300, 1.4347] | NA | [NA, NA] | NA |
| dPL | S2 | 156 | 0.5028 | [0.3603, 0.7095] | [0.4524, 0.5320] | NA | [NA, NA] | NA |
| IC | S3 | 121 | 1.5411 | [1.2665, 1.8347] | [1.4272, 1.6022] | NA | [NA, NA] | NA |
| dPL | S3 | 121 | 0.9171 | [0.7251, 1.2272] | [0.8733, 1.0379] | NA | [NA, NA] | NA |
| IC | S4 | 34 | 1.7838 | [1.3714, 2.0399] | [1.5227, 1.9540] | NA | [NA, NA] | NA |
| dPL | S4 | 34 | 1.2416 | [0.9415, 1.6832] | [0.9977, 1.5418] | NA | [NA, NA] | NA |
| IC | S5 | 55 | 2.0018 | [1.7603, 2.2538] | [1.8311, 2.1102] | NA | [NA, NA] | NA |
| dPL | S5 | 55 | 1.7090 | [1.5856, 1.9457] | [1.6343, 1.7996] | NA | [NA, NA] | NA |

## 3. IC signed-median≈0 diagnostic

`r2_ic_signed_median_diagnostic.csv` reports signed median, median absolute movement, exact-zero share, threshold exceedance and sign proportions. `zero_both_boundary_share` is the share of all IC basin rows that are exact zero and have both Base and CN at a numerical parameter boundary.

| parameter | signed median | median |delta| | exact zero | zero both boundary | zero nonboundary | P(|delta|>.01) | P(|delta|>.05) | positive | negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a (UH shape) | 0.0000 | 0.0293 | 0.403 | 0.403 | 0.000 | 0.557 | 0.458 | 0.307 | 0.290 |
| b | 0.0000 | 0.0599 | 0.241 | 0.241 | 0.000 | 0.712 | 0.522 | 0.337 | 0.422 |
| c | 0.0000 | 0.2185 | 0.380 | 0.380 | 0.000 | 0.620 | 0.591 | 0.330 | 0.290 |
| cg | 0.0000 | 0.1836 | 0.154 | 0.154 | 0.000 | 0.785 | 0.676 | 0.439 | 0.407 |
| ci | 0.0000 | 0.1878 | 0.117 | 0.117 | 0.000 | 0.846 | 0.714 | 0.394 | 0.490 |
| dm | 0.0000 | 0.0881 | 0.394 | 0.394 | 0.000 | 0.591 | 0.550 | 0.267 | 0.339 |
| ex | 0.0000 | 0.0952 | 0.315 | 0.315 | 0.000 | 0.659 | 0.574 | 0.311 | 0.375 |
| im | 0.0000 | 0.0000 | 0.627 | 0.627 | 0.000 | 0.354 | 0.292 | 0.139 | 0.234 |
| k | -0.0031 | 0.0201 | 0.141 | 0.141 | 0.000 | 0.620 | 0.331 | 0.311 | 0.548 |
| kg | 0.0000 | 0.1020 | 0.307 | 0.307 | 0.000 | 0.659 | 0.576 | 0.267 | 0.426 |
| ki | 0.0000 | 0.1221 | 0.186 | 0.186 | 0.000 | 0.763 | 0.640 | 0.350 | 0.463 |
| lm | 0.0000 | 0.1470 | 0.315 | 0.315 | 0.000 | 0.663 | 0.595 | 0.460 | 0.226 |
| sm | 0.0000 | 0.0658 | 0.139 | 0.139 | 0.000 | 0.768 | 0.567 | 0.403 | 0.458 |
| theta (UH scale) | 0.0000 | 0.0015 | 0.473 | 0.473 | 0.000 | 0.452 | 0.331 | 0.211 | 0.316 |
| um | 0.0000 | 0.0000 | 0.522 | 0.522 | 0.000 | 0.465 | 0.433 | 0.315 | 0.164 |

IC regime-level absolute-shift diagnostics are in `r2_ic_abs_shift_by_regime.csv`; they are descriptive only.

## Final classifications

- `SNOW_GRADIENT_ROBUSTNESS = PARTIAL`
- `GLOBAL_PARAMETER_REORGANIZATION = STRONG`
- `IC_ZERO_MEDIAN_INTERPRETATION = MIXED`

## R2 conclusion boundary
这些 robustness checks 支持将 R2 表述为参数层的 Base−CN reorganization 与 compensation signatures：候选参数的 snow-gradient 方向不是由 S5 单独决定，且完整 15 维参数距离提供了不依赖单个参数的整体诊断。IC 与 dPL 的整体距离和单参数组织若出现不同强度，应保持范式特异性表述。IC signed median 接近零不能单独当作低 movement 证据，应结合 absolute movement、exact-zero 和 boundary concentration 共同解读。任何参数如何影响内部状态、通量或流量机制，仍属于 R3。
