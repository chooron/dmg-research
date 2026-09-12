# R2 参数层统计分析报告（结果驱动，未生成正式 Figure 3）

## Status
本报告由 `scripts/run_r2_parameter_statistics.py` 实际运行生成。所有新增产物均位于 `manuscript/`；R1 active 图、表和统计产物未修改。R2 主比较锁定为同一流域的 Base − CN，方向不可反转。

## 1. 数据审计

- canonical basin set: `531` 个，来自 `/home/jingxin/code/dmg-research/data/531sub_id.txt`；重复 ID: `0`。
- canonical `frac_snow`: R1 `/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/manuscript/results/R1/r1_snow_attributes.csv`，来源字段为 CAMELS `attributes[:,3]`，未重新计算。固定分层：S1 [0,0.05): 165；S2 [0.05,0.15): 156；S3 [0.15,0.30): 121；S4 [0.30,0.50): 34；S5 [0.50,1.00]: 55。
- active IC: CMA-ES raw JSON（不是旧大纲中的 IC-XNES）；每个结构按已完成 restart 的 stored train-period KGE 最大值选择，start 最小值作为平局规则。
- active dPL: Base/CN 使用每 seed 的 `best_checkpoint.pt`；GD（代码模型名 XAJ_TGD2）使用 R1 固定的三 seed 共同最大 periodic checkpoint，即 epoch 100。dPL canonical parameter vector 是 checkpoint 经 `robust_normalize`、sigmoid-to-physical mapping 重建的物理参数。
- active GD is a generic temperature-dependent precipitation-memory structure (`XAJ_TGD2`), not an explicit snow accumulation/melt model; it is auxiliary only.

### Source coverage

| structure | IC valid raw records | IC selected basins | dPL seeds | IC–dPL paired basins | common public parameters |
|---|---:|---:|---:|---:|---:|
| Base | 5310 | 531 | 3 | 531 | 15 |
| CN | 5310 | 531 | 3 | 531 | 15 |
| GD | 5310 | 531 | 3 | 531 | 15 |

### Parameter comparability

当前 active XAJ public parameter 集合为 **15 个**：`xaj_k, xaj_b, xaj_im, xaj_um, xaj_lm, xaj_dm, xaj_c, xaj_sm, xaj_ex, xaj_ki, xaj_kg, xaj_ci, xaj_cg, xaj_a, xaj_theta`。Base/CN/GD 的这些名称、物理定义和 project-specific bounds 一致；因此没有使用旧计划中的 `lag`，也没有把 CN/TGD structure-specific parameters 放入 normalized shift 主分析。`xaj_a` 与 `xaj_theta` 是当前 Gamma-UH shape/scale routing parameters。统计统一使用物理空间 `z=(theta-lower)/(upper-lower)`。KI/KG 的主结果使用 checkpoint/raw JSON 中保存的 **pre-runtime mapped values**；模型运行时对 `KI+KG>=1` 的 joint rescaling 被单独审计，并在 seed/canonical value tables 中同时保留 effective values。

## 2. Primary paired Base − CN shifts

效应量为每个 basin 的 `(theta_Base - theta_CN)/(upper-lower)`，随后按 paradigm 汇总。CI 是固定 seed `20260730`、10,000 次 basin bootstrap 的 median CI。

| paradigm | parameter | n | median shift | IQR | 95% CI |
|---|---|---:|---:|---:|---|
| IC | k | 531 | -0.003 | [-0.042, 0.003] | [-0.006, -0.000] |
| IC | a (UH shape) | 531 | 0.000 | [-0.023, 0.030] | [0.000, 0.000] |
| IC | b | 531 | 0.000 | [-0.068, 0.038] | [0.000, 0.000] |
| IC | c | 531 | 0.000 | [-0.152, 0.260] | [0.000, 0.000] |
| IC | cg | 531 | 0.000 | [-0.148, 0.213] | [0.000, 0.000] |
| IC | ci | 531 | 0.000 | [-0.295, 0.100] | [-0.015, 0.000] |
| IC | dm | 531 | 0.000 | [-0.183, 0.031] | [0.000, 0.000] |
| IC | ex | 531 | 0.000 | [-0.142, 0.061] | [0.000, 0.000] |
| IC | im | 531 | 0.000 | [0.000, 0.000] | [0.000, 0.000] |
| IC | kg | 531 | 0.000 | [-0.170, 0.011] | [0.000, 0.000] |
| IC | ki | 531 | 0.000 | [-0.232, 0.049] | [-0.002, 0.000] |
| IC | lm | 531 | 0.000 | [0.000, 0.338] | [0.000, 0.000] |
| IC | sm | 531 | 0.000 | [-0.069, 0.057] | [-0.000, 0.000] |
| IC | theta (UH scale) | 531 | 0.000 | [-0.018, 0.000] | [0.000, 0.000] |
| IC | um | 531 | 0.000 | [0.000, 0.113] | [0.000, 0.000] |
| dPL | ki | 531 | -0.095 | [-0.360, 0.025] | [-0.125, -0.064] |
| dPL | cg | 531 | -0.084 | [-0.257, 0.000] | [-0.113, -0.066] |
| dPL | ci | 531 | -0.048 | [-0.229, 0.031] | [-0.066, -0.035] |
| dPL | kg | 531 | -0.031 | [-0.190, 0.012] | [-0.045, -0.020] |
| dPL | lm | 531 | 0.029 | [-0.010, 0.116] | [0.022, 0.038] |
| dPL | ex | 531 | -0.015 | [-0.111, 0.077] | [-0.024, -0.007] |
| dPL | c | 531 | 0.012 | [-0.000, 0.100] | [0.008, 0.018] |
| dPL | um | 531 | 0.011 | [-0.005, 0.164] | [0.008, 0.016] |
| dPL | b | 531 | 0.004 | [-0.023, 0.095] | [-0.000, 0.010] |
| dPL | dm | 531 | -0.003 | [-0.019, 0.001] | [-0.005, -0.002] |
| dPL | k | 531 | 0.002 | [-0.010, 0.017] | [0.001, 0.004] |
| dPL | im | 531 | -0.002 | [-0.018, 0.000] | [-0.003, -0.001] |
| dPL | a (UH shape) | 531 | -0.001 | [-0.054, 0.047] | [-0.007, 0.003] |
| dPL | theta (UH scale) | 531 | 0.001 | [-0.002, 0.007] | [0.000, 0.001] |
| dPL | sm | 531 | -0.000 | [-0.045, 0.079] | [-0.003, 0.004] |

## 3. Snow dependence

`delta_p ~ frac_snow` 用普通 OLS slope（每 basin 一条 paired shift）；CI 为同一固定 bootstrap 设置下的 slope percentile interval；Spearman rho 是单调性诊断。BH q-value 仅用于多重比较标记，不作为 Figure 3 选择的唯一规则。

| paradigm | parameter | beta | 95% CI | Spearman rho | Spearman p | BH q | n |
|---|---|---:|---|---:|---:|---:|---:
| IC | dm | -0.537 | [-0.756, -0.308] | -0.116 | 0.00762 | 0.0143 | 531 |
| IC | um | 0.521 | [0.323, 0.720] | 0.216 | 4.8e-07 | 2.4e-06 | 531 |
| IC | ki | -0.475 | [-0.656, -0.313] | -0.237 | 3.39e-08 | 2.54e-07 | 531 |
| IC | ci | -0.414 | [-0.637, -0.194] | -0.183 | 2.16e-05 | 5.39e-05 | 531 |
| IC | lm | 0.400 | [0.225, 0.576] | 0.189 | 1.15e-05 | 3.44e-05 | 531 |
| IC | ex | -0.370 | [-0.556, -0.179] | -0.077 | 0.0748 | 0.0935 | 531 |
| IC | im | -0.363 | [-0.517, -0.207] | -0.248 | 7.28e-09 | 1.09e-07 | 531 |
| IC | cg | 0.292 | [0.138, 0.447] | 0.007 | 0.869 | 0.869 | 531 |
| IC | c | 0.278 | [0.053, 0.510] | 0.085 | 0.0516 | 0.0774 | 531 |
| IC | kg | -0.203 | [-0.403, -0.008] | -0.205 | 1.91e-06 | 7.18e-06 | 531 |
| IC | sm | 0.161 | [-0.025, 0.350] | 0.106 | 0.0148 | 0.0246 | 531 |
| IC | a (UH shape) | 0.036 | [-0.162, 0.240] | 0.038 | 0.386 | 0.445 | 531 |
| IC | k | -0.028 | [-0.088, 0.030] | -0.126 | 0.0036 | 0.00771 | 531 |
| IC | theta (UH scale) | -0.017 | [-0.249, 0.220] | -0.081 | 0.0619 | 0.0845 | 531 |
| IC | b | -0.006 | [-0.196, 0.201] | 0.023 | 0.596 | 0.638 | 531 |
| dPL | ci | -1.226 | [-1.295, -1.154] | -0.519 | 5.57e-38 | 2.09e-37 | 531 |
| dPL | um | 1.015 | [0.868, 1.152] | 0.418 | 6.77e-24 | 2.03e-23 | 531 |
| dPL | ki | -0.740 | [-0.864, -0.624] | -0.613 | 4.14e-56 | 6.21e-55 | 531 |
| dPL | c | 0.625 | [0.479, 0.773] | 0.554 | 5.23e-44 | 2.61e-43 | 531 |
| dPL | sm | 0.595 | [0.510, 0.682] | 0.315 | 1.12e-13 | 2.8e-13 | 531 |
| dPL | im | -0.452 | [-0.578, -0.328] | -0.568 | 8.94e-47 | 6.7e-46 | 531 |
| dPL | lm | 0.324 | [0.216, 0.439] | 0.266 | 4.39e-10 | 9.41e-10 | 531 |
| dPL | theta (UH scale) | 0.265 | [0.150, 0.385] | 0.126 | 0.00364 | 0.00426 | 531 |
| dPL | cg | 0.238 | [0.099, 0.367] | -0.154 | 0.000366 | 0.000549 | 531 |
| dPL | dm | -0.130 | [-0.215, -0.049] | -0.187 | 1.39e-05 | 2.32e-05 | 531 |
| dPL | ex | -0.098 | [-0.161, -0.037] | -0.126 | 0.00369 | 0.00426 | 531 |
| dPL | k | 0.062 | [0.012, 0.120] | -0.043 | 0.325 | 0.348 | 531 |
| dPL | a (UH shape) | -0.036 | [-0.191, 0.124] | 0.001 | 0.978 | 0.978 | 531 |
| dPL | b | 0.015 | [-0.107, 0.156] | 0.129 | 0.00287 | 0.00392 | 531 |
| dPL | kg | 0.005 | [-0.141, 0.146] | -0.230 | 8.6e-08 | 1.61e-07 | 531 |

### Fixed R1 snow regimes

完整 regime-level 结果保存在 `r2_snow_regime_summary.csv`；S4/S5 样本较少，均作为不确定性较大的描述性证据。

本次实际结果中，按可复现选择规则（IC/dPL slope CI 均不跨零、方向一致，再按两范式较小绝对 slope 排名）选出的 4 个候选参数为 **um、ki、ci、im**：um 的 shift 随雪影响增强而上升，ki、ci 与 im 下降；它们覆盖 ET/tension-water storage、routing/recession 与 production 功能组。lm、dm、c、ex 作为次级候选保留到 Supplement。
## 4. Boundary use and cross-basin organization

连续 boundary distance 为 `min(z, 1-z)`；boundary concentration 同时报告 0.01/0.02/0.05 三个阈值，而不是选一个阈值偷换结论。

### Boundary concentration (Base/CN comparison)

| paradigm | parameter | threshold | Base rate | CN rate | CN−Base | CI |
|---|---|---:|---:|---:|---:|---|
| IC | a (UH shape) | 0.01 | 0.606 | 0.559 | -0.047 | [-0.094, 0.000] |
| IC | b | 0.01 | 0.492 | 0.420 | -0.072 | [-0.121, -0.023] |
| IC | c | 0.01 | 0.766 | 0.702 | -0.064 | [-0.113, -0.013] |
| IC | cg | 0.01 | 0.384 | 0.388 | 0.004 | [-0.047, 0.055] |
| IC | ci | 0.01 | 0.367 | 0.320 | -0.047 | [-0.100, 0.004] |
| IC | dm | 0.01 | 0.719 | 0.674 | -0.045 | [-0.092, 0.002] |
| IC | ex | 0.01 | 0.653 | 0.614 | -0.040 | [-0.090, 0.011] |
| IC | im | 0.01 | 0.825 | 0.746 | -0.079 | [-0.122, -0.036] |
| IC | k | 0.01 | 0.200 | 0.183 | -0.017 | [-0.041, 0.009] |
| IC | kg | 0.01 | 0.529 | 0.525 | -0.004 | [-0.053, 0.045] |
| IC | ki | 0.01 | 0.407 | 0.384 | -0.023 | [-0.068, 0.023] |
| IC | lm | 0.01 | 0.588 | 0.569 | -0.019 | [-0.068, 0.032] |
| IC | sm | 0.01 | 0.284 | 0.252 | -0.032 | [-0.072, 0.006] |
| IC | theta (UH scale) | 0.01 | 0.721 | 0.627 | -0.094 | [-0.137, -0.053] |
| IC | um | 0.01 | 0.836 | 0.746 | -0.090 | [-0.134, -0.049] |
| IC | a (UH shape) | 0.02 | 0.614 | 0.563 | -0.051 | [-0.098, -0.004] |
| IC | b | 0.02 | 0.508 | 0.429 | -0.079 | [-0.128, -0.032] |
| IC | c | 0.02 | 0.768 | 0.702 | -0.066 | [-0.115, -0.017] |
| IC | cg | 0.02 | 0.397 | 0.411 | 0.013 | [-0.038, 0.066] |
| IC | ci | 0.02 | 0.373 | 0.320 | -0.053 | [-0.105, 0.000] |
| IC | dm | 0.02 | 0.725 | 0.680 | -0.045 | [-0.092, 0.002] |
| IC | ex | 0.02 | 0.669 | 0.631 | -0.038 | [-0.089, 0.013] |
| IC | im | 0.02 | 0.834 | 0.763 | -0.072 | [-0.113, -0.028] |
| IC | k | 0.02 | 0.203 | 0.183 | -0.021 | [-0.047, 0.006] |
| IC | kg | 0.02 | 0.531 | 0.527 | -0.004 | [-0.053, 0.045] |
| IC | ki | 0.02 | 0.409 | 0.386 | -0.023 | [-0.070, 0.023] |
| IC | lm | 0.02 | 0.593 | 0.573 | -0.021 | [-0.073, 0.028] |
| IC | sm | 0.02 | 0.288 | 0.256 | -0.032 | [-0.070, 0.006] |
| IC | theta (UH scale) | 0.02 | 0.746 | 0.663 | -0.083 | [-0.124, -0.041] |
| IC | um | 0.02 | 0.836 | 0.751 | -0.085 | [-0.126, -0.041] |
| IC | a (UH shape) | 0.05 | 0.627 | 0.573 | -0.055 | [-0.102, -0.008] |
| IC | b | 0.05 | 0.559 | 0.461 | -0.098 | [-0.145, -0.051] |
| IC | c | 0.05 | 0.785 | 0.727 | -0.058 | [-0.107, -0.009] |
| IC | cg | 0.05 | 0.411 | 0.450 | 0.040 | [-0.015, 0.092] |
| IC | ci | 0.05 | 0.395 | 0.343 | -0.053 | [-0.102, -0.002] |
| IC | dm | 0.05 | 0.731 | 0.697 | -0.034 | [-0.079, 0.011] |
| IC | ex | 0.05 | 0.691 | 0.669 | -0.023 | [-0.072, 0.026] |
| IC | im | 0.05 | 0.851 | 0.787 | -0.064 | [-0.105, -0.023] |
| IC | k | 0.05 | 0.222 | 0.196 | -0.026 | [-0.053, 0.000] |
| IC | kg | 0.05 | 0.550 | 0.548 | -0.002 | [-0.049, 0.047] |
| IC | ki | 0.05 | 0.429 | 0.403 | -0.026 | [-0.073, 0.021] |
| IC | lm | 0.05 | 0.612 | 0.588 | -0.024 | [-0.073, 0.024] |
| IC | sm | 0.05 | 0.305 | 0.275 | -0.030 | [-0.066, 0.006] |
| IC | theta (UH scale) | 0.05 | 0.808 | 0.714 | -0.094 | [-0.134, -0.055] |
| IC | um | 0.05 | 0.846 | 0.766 | -0.079 | [-0.121, -0.038] |
| dPL | a (UH shape) | 0.01 | 0.021 | 0.013 | -0.008 | [-0.023, 0.006] |
| dPL | b | 0.01 | 0.087 | 0.056 | -0.030 | [-0.049, -0.011] |
| dPL | c | 0.01 | 0.650 | 0.360 | -0.290 | [-0.345, -0.234] |
| dPL | cg | 0.01 | 0.023 | 0.073 | 0.051 | [0.034, 0.070] |
| dPL | ci | 0.01 | 0.006 | 0.000 | -0.006 | [-0.013, 0.000] |
| dPL | dm | 0.01 | 0.620 | 0.420 | -0.200 | [-0.241, -0.160] |
| dPL | ex | 0.01 | 0.153 | 0.051 | -0.102 | [-0.134, -0.072] |
| dPL | im | 0.01 | 0.806 | 0.548 | -0.258 | [-0.298, -0.218] |
| dPL | k | 0.01 | 0.045 | 0.062 | 0.017 | [0.000, 0.034] |
| dPL | kg | 0.01 | 0.109 | 0.066 | -0.043 | [-0.073, -0.011] |
| dPL | ki | 0.01 | 0.166 | 0.000 | -0.166 | [-0.198, -0.136] |
| dPL | lm | 0.01 | 0.045 | 0.045 | 0.000 | [-0.019, 0.019] |
| dPL | sm | 0.01 | 0.122 | 0.026 | -0.096 | [-0.124, -0.070] |
| dPL | theta (UH scale) | 0.01 | 0.518 | 0.586 | 0.068 | [0.034, 0.104] |
| dPL | um | 0.01 | 0.373 | 0.235 | -0.137 | [-0.181, -0.094] |
| dPL | a (UH shape) | 0.02 | 0.036 | 0.038 | 0.002 | [-0.015, 0.019] |
| dPL | b | 0.02 | 0.128 | 0.117 | -0.011 | [-0.032, 0.009] |
| dPL | c | 0.02 | 0.770 | 0.495 | -0.275 | [-0.326, -0.224] |
| dPL | cg | 0.02 | 0.045 | 0.107 | 0.062 | [0.040, 0.085] |
| dPL | ci | 0.02 | 0.028 | 0.008 | -0.021 | [-0.038, -0.006] |
| dPL | dm | 0.02 | 0.719 | 0.554 | -0.166 | [-0.205, -0.128] |
| dPL | ex | 0.02 | 0.194 | 0.098 | -0.096 | [-0.128, -0.064] |
| dPL | im | 0.02 | 0.872 | 0.652 | -0.220 | [-0.258, -0.183] |
| dPL | k | 0.02 | 0.079 | 0.087 | 0.008 | [-0.008, 0.024] |
| dPL | kg | 0.02 | 0.277 | 0.124 | -0.153 | [-0.192, -0.113] |
| dPL | ki | 0.02 | 0.188 | 0.002 | -0.186 | [-0.220, -0.153] |
| dPL | lm | 0.02 | 0.107 | 0.100 | -0.008 | [-0.036, 0.019] |
| dPL | sm | 0.02 | 0.175 | 0.070 | -0.105 | [-0.136, -0.075] |
| dPL | theta (UH scale) | 0.02 | 0.672 | 0.687 | 0.015 | [-0.013, 0.043] |
| dPL | um | 0.02 | 0.480 | 0.356 | -0.124 | [-0.168, -0.083] |
| dPL | a (UH shape) | 0.05 | 0.104 | 0.100 | -0.004 | [-0.026, 0.019] |
| dPL | b | 0.05 | 0.237 | 0.213 | -0.024 | [-0.053, 0.004] |
| dPL | c | 0.05 | 0.876 | 0.648 | -0.228 | [-0.273, -0.183] |
| dPL | cg | 0.05 | 0.109 | 0.202 | 0.092 | [0.058, 0.126] |
| dPL | ci | 0.05 | 0.090 | 0.041 | -0.049 | [-0.075, -0.024] |
| dPL | dm | 0.05 | 0.815 | 0.721 | -0.094 | [-0.132, -0.056] |
| dPL | ex | 0.05 | 0.299 | 0.217 | -0.083 | [-0.122, -0.043] |
| dPL | im | 0.05 | 0.947 | 0.800 | -0.147 | [-0.181, -0.113] |
| dPL | k | 0.05 | 0.122 | 0.117 | -0.006 | [-0.024, 0.013] |
| dPL | kg | 0.05 | 0.469 | 0.239 | -0.230 | [-0.273, -0.186] |
| dPL | ki | 0.05 | 0.234 | 0.028 | -0.205 | [-0.241, -0.169] |
| dPL | lm | 0.05 | 0.211 | 0.215 | 0.004 | [-0.028, 0.034] |
| dPL | sm | 0.05 | 0.256 | 0.141 | -0.115 | [-0.147, -0.083] |
| dPL | theta (UH scale) | 0.05 | 0.812 | 0.789 | -0.023 | [-0.051, 0.006] |
| dPL | um | 0.05 | 0.595 | 0.520 | -0.075 | [-0.117, -0.034] |

boundary signal should be read as a diagnostic, not a claim of invalidity. Full GD and all threshold rows are retained in `r2_boundary_summary.csv`.

### Dispersion

| paradigm | structure | parameter | n | median z | IQR z |
|---|---|---|---:|---:|---:
| IC | Base | a (UH shape) | 531 | 0.363 | 0.863 |
| IC | Base | b | 531 | 0.101 | 0.413 |
| IC | Base | c | 531 | 1.000 | 0.831 |
| IC | Base | cg | 531 | 0.645 | 0.861 |
| IC | Base | ci | 531 | 0.411 | 0.730 |
| IC | Base | dm | 531 | 0.000 | 0.678 |
| IC | Base | ex | 531 | 0.083 | 0.585 |
| IC | Base | im | 531 | 0.000 | 0.000 |
| IC | Base | k | 531 | 0.167 | 0.225 |
| IC | Base | kg | 531 | 0.146 | 0.502 |
| IC | Base | ki | 531 | 0.299 | 0.546 |
| IC | Base | lm | 531 | 0.675 | 0.901 |
| IC | Base | sm | 531 | 0.265 | 0.394 |
| IC | Base | theta (UH scale) | 531 | 0.000 | 0.087 |
| IC | Base | um | 531 | 1.000 | 0.898 |
| IC | CN | a (UH shape) | 531 | 0.315 | 0.794 |
| IC | CN | b | 531 | 0.153 | 0.384 |
| IC | CN | c | 531 | 0.950 | 0.849 |
| IC | CN | cg | 531 | 0.582 | 0.843 |
| IC | CN | ci | 531 | 0.564 | 0.686 |
| IC | CN | dm | 531 | 0.151 | 0.874 |
| IC | CN | ex | 531 | 0.134 | 1.000 |
| IC | CN | im | 531 | 0.000 | 0.047 |
| IC | CN | k | 531 | 0.187 | 0.239 |
| IC | CN | kg | 531 | 0.211 | 0.735 |
| IC | CN | ki | 531 | 0.388 | 0.635 |
| IC | CN | lm | 531 | 0.414 | 0.991 |
| IC | CN | sm | 531 | 0.279 | 0.404 |
| IC | CN | theta (UH scale) | 531 | 0.000 | 0.156 |
| IC | CN | um | 531 | 0.864 | 1.000 |
| IC | GD | a (UH shape) | 531 | 0.448 | 0.960 |
| IC | GD | b | 531 | 0.123 | 0.377 |
| IC | GD | c | 531 | 0.923 | 0.969 |
| IC | GD | cg | 531 | 0.554 | 0.807 |
| IC | GD | ci | 531 | 0.474 | 0.800 |
| IC | GD | dm | 531 | 0.016 | 0.792 |
| IC | GD | ex | 531 | 0.040 | 0.563 |
| IC | GD | im | 531 | 0.000 | 0.000 |
| IC | GD | k | 531 | 0.170 | 0.213 |
| IC | GD | kg | 531 | 0.200 | 0.886 |
| IC | GD | ki | 531 | 0.384 | 0.763 |
| IC | GD | lm | 531 | 0.576 | 1.000 |
| IC | GD | sm | 531 | 0.234 | 0.464 |
| IC | GD | theta (UH scale) | 531 | 0.000 | 0.168 |
| IC | GD | um | 531 | 1.000 | 0.851 |
| dPL | Base | a (UH shape) | 531 | 0.519 | 0.540 |
| dPL | Base | b | 531 | 0.239 | 0.537 |
| dPL | Base | c | 531 | 0.995 | 0.014 |
| dPL | Base | cg | 531 | 0.617 | 0.590 |
| dPL | Base | ci | 531 | 0.600 | 0.503 |
| dPL | Base | dm | 531 | 0.005 | 0.031 |
| dPL | Base | ex | 531 | 0.205 | 0.587 |
| dPL | Base | im | 531 | 0.001 | 0.006 |
| dPL | Base | k | 531 | 0.178 | 0.156 |
| dPL | Base | kg | 531 | 0.058 | 0.175 |
| dPL | Base | ki | 531 | 0.281 | 0.483 |
| dPL | Base | lm | 531 | 0.722 | 0.461 |
| dPL | Base | sm | 531 | 0.249 | 0.448 |
| dPL | Base | theta (UH scale) | 531 | 0.009 | 0.032 |
| dPL | Base | um | 531 | 0.971 | 0.326 |
| dPL | CN | a (UH shape) | 531 | 0.539 | 0.490 |
| dPL | CN | b | 531 | 0.182 | 0.351 |
| dPL | CN | c | 531 | 0.976 | 0.137 |
| dPL | CN | cg | 531 | 0.684 | 0.451 |
| dPL | CN | ci | 531 | 0.717 | 0.271 |
| dPL | CN | dm | 531 | 0.016 | 0.075 |
| dPL | CN | ex | 531 | 0.268 | 0.516 |
| dPL | CN | im | 531 | 0.007 | 0.034 |
| dPL | CN | k | 531 | 0.166 | 0.143 |
| dPL | CN | kg | 531 | 0.177 | 0.378 |
| dPL | CN | ki | 531 | 0.513 | 0.417 |
| dPL | CN | lm | 531 | 0.659 | 0.577 |
| dPL | CN | sm | 531 | 0.262 | 0.371 |
| dPL | CN | theta (UH scale) | 531 | 0.007 | 0.034 |
| dPL | CN | um | 531 | 0.826 | 0.783 |
| dPL | GD | a (UH shape) | 531 | 0.567 | 0.521 |
| dPL | GD | b | 531 | 0.206 | 0.447 |
| dPL | GD | c | 531 | 0.958 | 0.562 |
| dPL | GD | cg | 531 | 0.602 | 0.597 |
| dPL | GD | ci | 531 | 0.593 | 0.446 |
| dPL | GD | dm | 531 | 0.011 | 0.082 |
| dPL | GD | ex | 531 | 0.186 | 0.631 |
| dPL | GD | im | 531 | 0.002 | 0.006 |
| dPL | GD | k | 531 | 0.163 | 0.148 |
| dPL | GD | kg | 531 | 0.140 | 0.513 |
| dPL | GD | ki | 531 | 0.429 | 0.428 |
| dPL | GD | lm | 531 | 0.669 | 0.537 |
| dPL | GD | sm | 531 | 0.213 | 0.423 |
| dPL | GD | theta (UH scale) | 531 | 0.007 | 0.046 |
| dPL | GD | um | 531 | 0.926 | 0.542 |

Boundary concentration provides independent evidence mainly for dPL: across threshold 0.01–0.05, CN lowers near-boundary rates for c, im, ki, dm, um and related routing/storage parameters, while the direction is not uniform for every parameter. IC also shows frequent boundary use, but Base−CN median shifts are mostly zero because both selected CMA-ES solutions are often at common bounds. Therefore boundary concentration can be Panel d only as a focused dPL diagnostic; normalized-IQR dispersion should accompany it, not be replaced by it.

## 5. IC versus dPL and GD

IC and dPL are compared using the same basin-level normalized physical values. dPL seed-level values are retained; canonical dPL values are the within-basin median across seeds, following R1. Full seed diagnostics and GD rows are in machine-readable outputs; dPL seed-specific Base-CN shifts and direction agreement are in `r2_dpl_seed_robustness_summary.csv`.

The compensation signatures are **clearly different rather than simply consistent**: IC has little global median Base−CN displacement for most public parameters but shows snow-dependent, boundary-heavy reorganization; dPL shows large global shifts in ki, cg and ci and much stronger monotonic snow gradients in ci, um, ki, c and sm. GD does not add an independent primary line: its dPL public-parameter shifts relative to Base/CN are generally smaller and diffuse (largest median GD−CN values are cg ≈ −0.061, ci ≈ −0.041 and ki ≈ −0.034), so GD remains Supplement-only.

## 6. Evidence-driven Figure 3 plan

### Recommended main panels

- **Panel a — global paired reorganization:** all 15 common XAJ parameters, Base−CN median and 95% CI, IC and dPL side-by-side. This is supported for the full parameter set and avoids post-hoc p-value filtering.
- **Panel b — snow gradients:** plot beta and 95% CI for the 15 parameters, IC/dPL side-by-side. Use the full set; visually emphasize only parameters whose effect size, CI, gradient, cross-paradigm evidence, and data quality jointly support a writing claim. Do not imply causality.
- **Panel c — five snow regimes:** use **um, ki, ci and im**, selected from the actual results because their IC/dPL snow-gradient directions agree and their slopes/intervals are among the clearest. Show all five R1 regimes and their actual n; interpret as descriptive parameter shifts only.
- **Panel d — boundary concentration with dispersion context:** use threshold-sensitive boundary rates for the strongest dPL boundary signals (especially c, im, ki, dm, um) and a compact normalized-IQR organization comparison. Do not claim universal boundary relief for IC or all parameters.

### Key-parameter rule

Use the intersection of: large absolute global shift with CI away from zero; clear beta or monotonicity; consistent IC/dPL direction or an explicitly meaningful divergence; low missing/invalid/boundary risk; and an unambiguous model-function definition. The final ranked candidates are in `r2_figure3_candidate_parameters.csv`. No formal Figure 3 layout or image was generated.

### GD and Supplement

GD remains Supplement-only because it does not show a distinct, stable public-parameter path beyond Base/CN. Include full basin-level shifts, seed-level dPL rows, regime summaries, threshold sensitivity, all 15-parameter tables, invalid/boundary audit, dispersion, and GD in Supplement. Do not add state/flux/causal interpretation to R2.

## 7. Unresolved blockers and limitations

- No blocker remained for the Base/CN paired shift calculation after the active source audit; all canonical source joins and public bounds checks are explicit in `r2_data_quality_checks.csv`.
- IC restart selection is based on stored train KGE, while R1 final performance is recomputed from exports; this provenance distinction is preserved.
- The project has no basin-level parameter truth. Results are reported only as parameter shifts, compensation signatures, boundary concentration, and cross-basin parameter organization.
- GD is XAJ-TGD2 generic temperature-dependent delay, not a physical snow model; it is not evidence for R3 mechanisms.
