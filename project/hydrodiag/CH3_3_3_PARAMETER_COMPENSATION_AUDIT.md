# 3.3 参数补偿数据核准报告

> **用途**：博士论文第三章 3.3“融雪过程结构缺失的参数补偿特征”的数据核准与写作准备。
>
> **范围**：531 个主体流域、Base/CN 两种融雪过程结构与 TGD 通用温度条件控制、XAJ 共同模型骨架、15 个共享 XAJ host parameters、IC 与 dPL 两种参数估计约束。本文档只使用已冻结的 R2 参数输出，不重新训练、不重新率定、不修改模型。

## 1. Canonical 数据来源与 provenance

### 1.1 权威性判定

本报告采用 `project/hydrodiag/manuscript/analysis/R2/results/` 作为 Chapter 3 R2 的 canonical machine-readable 结果目录。理由是：

1. `project/hydrodiag/manuscript/analysis/R2/README.md` 将该目录定义为运行时分析包，并明确 15 参数、规范化坐标、canonical vector rule、basin-level bootstrap 和范围边界。
2. `project/hydrodiag/manuscript/README.md` 区分了 `analysis/` 运行时分析包与 `results/` 结果索引；已完成结果不应被覆盖。
3. `r2_statistical_audit_report.md` 和 `r2_final_closure_report.md` 均给出 `R2_FINAL_STATUS = READY`、12/12 gates PASS，但其中部分 CI 是较早或不同 bootstrap 汇总的渲染文本；最终数值以直接 machine-readable CSV 为准。
4. `manuscript/results/R2/` 是高度重叠的镜像目录。关键结果的点估计相同，少数浮点序列化值存在约 `10^-8`–`10^-7` 量级差异；本报告不混用两棵树。

### 1.2 直接使用的 frozen 输出

| 用途 | canonical 路径（相对于 `project/hydrodiag/`） | 行数/说明 |
|---|---|---:|
| 15 参数身份、边界、单位、结构索引 | `manuscript/analysis/R2/results/authoritative_15_parameter_specs.csv` | 15 |
| canonical basin-level 参数向量 | `manuscript/analysis/R2/results/r2_parameter_values_canonical.csv` | 3,186 = 531×3×2 |
| Base–CN basin-level `D_within/D_between/E` | `manuscript/analysis/R2/results/r2_within_structure_basin_level.csv` | 1,062 |
| Base–CN 总体/分层摘要 | `manuscript/analysis/R2/results/r2_within_structure_summary.csv` | 98 |
| Base–TGD basin-level `D_within/D_between/E` | `manuscript/analysis/R2/results/r2_tgd2_specificity_basin_level.csv` | 3,186，含 Base–CN、Base–TGD、TGD–CN |
| Base–TGD 总体/分层摘要 | `manuscript/analysis/R2/results/r2_tgd2_specificity_summary.csv` | 294 |
| Base–CN 全 15 参数斜率 | `manuscript/analysis/R2/results/r2_snow_gradients_summary.csv` | 30 = 2×15 |
| Base–CN 全 15 参数 S1–S5 摘要 | `manuscript/analysis/R2/results/r2_parameter_shifts_strata_summary.csv` | 150 = 2×15×5 |
| Base–CN 全 15 参数 basin-level shift | `manuscript/analysis/R2/results/r2_paired_shifts_basin_level.csv` | 15,930 |
| Base–CN/Base–TGD 配对差异 | `manuscript/analysis/R2/results/r2_paired_cn_tgd_delta_excess_summary.csv` | 14 = 2×(Full、ExcludeS5、S1–S5) |
| Base–CN/Base–TGD 配对差异 basin-level | `manuscript/analysis/R2/results/r2_paired_cn_tgd_delta_excess_basin_level.csv` | 1,062 |
| 结构内边界质量控制 | `manuscript/analysis/R2/results/r2_boundary_mass_safeguards.csv` | 270 = 2×3×15×3 个 tolerance |
| R2 bootstrap/门禁汇总 | `manuscript/analysis/R2/results/results_summary.md`、`r2_statistical_audit_report.md`、`r2_final_closure_report.md` | 交叉检查 |

关键文件 SHA-256（用于本次核准的版本识别）：

- `r2_within_structure_summary.csv`: `b6918cc27b5b767ea7486a681177abbab5f6e9553fceb82939492b6413509b7a`
- `r2_tgd2_specificity_summary.csv`: `ada037f0a66b98f52564cc9bb32b657a7da673f57e741b890753527c70530a53`
- `r2_snow_gradients_summary.csv`: `c22a37713b0b08e8314cf161f32636cfb6f69dea3e7a30b21cfda4ac931cb149`
- `r2_parameter_shifts_strata_summary.csv`: `0b6fc9b416f1cff38909a8111d34bfb561ec2b5a7b5e178b1b22b7041bd13699`
- `r2_parameter_values_canonical.csv`: `bc4acf1bc5b274732a961770de9e1999790b48c77f8f5e9d9b333ac39ae2f14d`
- `authoritative_15_parameter_specs.csv`: `e6a01b2d2eb67018c8e8549454fa14f9eb343775dcce2c71801758216b0e4d83`

### 1.3 原始参数与 snow metadata provenance

由 `manuscript/analysis/R2/r2_config.py` 和 `parameter_ledger.py` 明确记录：

- IC 原始 10-restart 目录：
  - `results/xaj_base_cmaes_531_batched_paired_v2/raw/xaj`
  - `results/xaj_cn_cmaes_531_batched_paired_v2/raw/xaj_cn`
  - `results/xaj_tgd2_cmaes_531_batched_v1/raw/xaj_tgd2`
- dPL 原始 3-seed 目录：
  - `results/dpl_camels_531_lite_v2/XAJ`
  - `results/dpl_camels_531_lite_v2/XAJ_CN`
  - `results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2`
  - seeds 为 `42, 123, 2026`。
- 531 流域与 snow 分层：`data/531sub_id.txt`、`manuscript/analysis/R1/results/canonical_basin_level.csv`。
- 边界来源：`manuscript/supplement/results/s2_parameter_bounds_from_code.csv`，并由 `authoritative_15_parameter_specs.csv` 再次冻结。
- IC canonical vector：每流域选择 best train-KGE restart；dPL canonical vector：每流域对 3 个 seed 取 median。
- 15 个参数均使用 `z=(physical-lower)/(upper-lower)` 归一化到 `[0,1]`。
- frozen R2 宏观结果记录的 bootstrap 为 200 次、seed `20260730`、basin 为 bootstrap 单元。下文标有“审计派生”的 Base–TGD 15 参数 CI，是从同一 frozen canonical vector 轻量重汇总，使用 paired-basin bootstrap 10,000 次；没有任何模型运行。

### 1.4 manuscript 与图件核查

- manuscript-facing Figure 4 的脚本是 `manuscript/scripts/r2/plot_r2_figure4_canonical.py`，输出 `manuscript/figures/Figure4_R2_final.png`。脚本直接读取 `authoritative_15_parameter_specs.csv`、`r2_snow_gradients_summary.csv`、`r2_parameter_shifts_strata_summary.csv` 和 `r2_parameter_values_canonical.csv`，并突出 `xaj_um/xaj_ki/xaj_ci/xaj_im`。
- Figure 5 (`manuscript/figures/Figure5_R3_final.png`) 属于 R3 controlled recovery，不属于本节，未作为 3.3 证据。
- `manuscript/supplement/FigureS3_alt_generating_field_provenance.md` 与 `FigureS4_tgd_response_shape_sensitivity_provenance.md` 记录的是 R3 alternative generating field / TGD response-shape sensitivity；`manuscript/supplement/final_assets/figures/Figure_S3/provenance.md` 和 `Figure_S4/provenance.md` 又将 S3/S4 改编号为 R3 参数—状态误差与高雪季节状态图。它们都不是 R2 15 参数空间核准来源，本报告不使用。

> **指标边界**：`r2_canonical_15D_displacement_summary.csv` 的 `D_rms` 是 canonical selected-vector displacement，不能替代本报告冻结定义下的重复估计 `D_between`。因此表 A/B 使用 `within_structure_*` 与 `tgd2_specificity_*` 的 ensemble distance，而不是把 `D_rms` 混入。

## 2. 15维参数空间总体分离

表中 `D_between Q25–Q75` 与 `D_within Q25–Q75` 由对应 basin-level frozen CSV 的 531 个 basin 值计算；`E` 的 CI 与 `E>0` 使用对应 canonical summary。`D_within` 统一采用 `within_pooled`。

### 表 A

| Estimation | Contrast | N | Median D_between | D_between Q25–Q75 | Median D_within | D_within Q25–Q75 | Median E | 95% CI E | E>0 n | E>0 % | Source |
|---|---|---:|---:|---|---:|---|---:|---|---:|---:|---|
| IC | Base–CN | 531 | 0.491328 | [0.457852, 0.524025] | 0.475352 | [0.439974, 0.506916] | +0.007463 | [+0.004953, +0.010655] | 335 | 63.1% | `r2_within_structure_summary.csv` + basin-level |
| IC | Base–TGD | 531 | 0.491233 | [0.460755, 0.519483] | 0.474208 | [0.442490, 0.505259] | +0.007398 | [+0.005112, +0.009795] | 338 | 63.7% | `r2_tgd2_specificity_summary.csv` + basin-level |
| dPL | Base–CN | 531 | 0.181716 | [0.123074, 0.307550] | 0.105607 | [0.082820, 0.144061] | +0.061057 | [+0.050360, +0.079551] | 445 | 83.8% | `r2_within_structure_summary.csv` + basin-level |
| dPL | Base–TGD | 531 | 0.153922 | [0.107476, 0.288620] | 0.113681 | [0.087608, 0.153472] | +0.037647 | [+0.032404, +0.043981] | 421 | 79.3% | `r2_tgd2_specificity_summary.csv` + basin-level |

### 表 A 结果判断

1. **Base–CN**：IC 与 dPL 的 median `E` 均为正，且 `E>0` 分别为 63.1% 与 83.8%；因此全 531 流域上，结构间距离总体超过相应结构内重复估计离散。manuscript 的 63.1%（335/531）和 83.8%（445/531）核准通过。
2. **Base–TGD**：IC 与 dPL 也均为正，`E>0` 分别为 63.7% 与 79.3%；因此通用温度条件控制下同样存在总体结构间分离，但幅度和 basin-level 分布不同。
3. **来源分解**：全样本的正 `E` 不是由某一个统一的 `D_within` 异常减小造成的。Base–CN 的主要共同形态是 `D_between` 的结构间差异存在；而 IC Base–TGD 的高雪层级还伴随 `D_within` 下降。不得将 `D_within` 下降直接写为“结构间参数变化增强”。分层分解见表 B。

## 3. 参数空间分离的 S1–S5 梯度

本表中的 `E` Q25–Q75 和 `E>0 %` 直接由 basin-level `E` 计算；95% CI 使用对应 Base–CN / Base–TGD frozen summary。百分比的分子为该层 `E>0` 的 basin 数。

### 表 B

| Estimation | Contrast | Snow class | N | D_between | D_within | Median E | E Q25–Q75 | 95% CI E | E>0 % |
|---|---|---|---:|---:|---:|---:|---|---|---:|
| IC | Base–CN | S1 | 165 | 0.470462 | 0.472882 | −0.001811 | [−0.010101, +0.012524] | [−0.004415, +0.000908] | 46.7% |
| IC | Base–CN | S2 | 156 | 0.487502 | 0.484631 | +0.002161 | [−0.011643, +0.016398] | [−0.004192, +0.004567] | 52.6% |
| IC | Base–CN | S3 | 121 | 0.494330 | 0.475480 | +0.012314 | [−0.000041, +0.038611] | [+0.008598, +0.018483] | 74.4% |
| IC | Base–CN | S4 | 34 | 0.508664 | 0.461686 | +0.042987 | [+0.021924, +0.076607] | [+0.027108, +0.064304] | 94.1% |
| IC | Base–CN | S5 | 55 | 0.560495 | 0.452018 | +0.082934 | [+0.056622, +0.128836] | [+0.069946, +0.110391] | 98.2% |
| dPL | Base–CN | S1 | 165 | 0.132233 | 0.105193 | +0.018614 | [−0.003241, +0.075932] | [+0.007092, +0.025961] | 70.3% |
| dPL | Base–CN | S2 | 156 | 0.165261 | 0.097711 | +0.055702 | [+0.015289, +0.123920] | [+0.043571, +0.079986] | 87.2% |
| dPL | Base–CN | S3 | 121 | 0.233339 | 0.105556 | +0.126672 | [+0.049235, +0.232247] | [+0.100780, +0.171508] | 92.6% |
| dPL | Base–CN | S4 | 34 | 0.345845 | 0.138529 | +0.132172 | [+0.082491, +0.312886] | [+0.092606, +0.274811] | 97.1% |
| dPL | Base–CN | S5 | 55 | 0.306614 | 0.119413 | +0.125198 | [+0.036968, +0.297106] | [+0.062766, +0.218888] | 87.3% |
| IC | Base–TGD | S1 | 165 | 0.481043 | 0.477424 | +0.000594 | [−0.009012, +0.013041] | [−0.002618, +0.003526] | 51.5% |
| IC | Base–TGD | S2 | 156 | 0.487730 | 0.487045 | +0.002167 | [−0.009250, +0.012792] | [−0.000068, +0.005773] | 57.1% |
| IC | Base–TGD | S3 | 121 | 0.493701 | 0.472435 | +0.013347 | [−0.000725, +0.033858] | [+0.007980, +0.016111] | 71.1% |
| IC | Base–TGD | S4 | 34 | 0.507257 | 0.459551 | +0.030809 | [+0.003346, +0.065262] | [+0.010641, +0.046977] | 76.5% |
| IC | Base–TGD | S5 | 55 | 0.522130 | 0.428830 | +0.093662 | [+0.038903, +0.135280] | [+0.074280, +0.115324] | 94.5% |
| dPL | Base–TGD | S1 | 165 | 0.132853 | 0.119879 | +0.019465 | [−0.003675, +0.050268] | [+0.010917, +0.026509] | 70.9% |
| dPL | Base–TGD | S2 | 156 | 0.130354 | 0.101994 | +0.031016 | [+0.000760, +0.080166] | [+0.023313, +0.039026] | 75.6% |
| dPL | Base–TGD | S3 | 121 | 0.241799 | 0.121617 | +0.079076 | [+0.025433, +0.203489] | [+0.054661, +0.137168] | 88.4% |
| dPL | Base–TGD | S4 | 34 | 0.339597 | 0.180136 | +0.106986 | [+0.034387, +0.184242] | [+0.066668, +0.159941] | 91.2% |
| dPL | Base–TGD | S5 | 55 | 0.276747 | 0.120528 | +0.085137 | [+0.022059, +0.157972] | [+0.054567, +0.117311] | 87.3% |

### 表 B 结果形态

- **IC Base–CN**：`−0.0018 → +0.0022 → +0.0123 → +0.0430 → +0.0829`，为清楚的 S1→S5 逐步增加。
- **dPL Base–CN**：`+0.0186 → +0.0557 → +0.1267 → +0.1322 → +0.1252`，主要在 S1→S3 快速增加，S3–S5 处于平台/轻微回落。
- **IC Base–TGD**：`+0.0006 → +0.0022 → +0.0133 → +0.0308 → +0.0937`，与 IC Base–CN 相似，S4→S5 增长更明显。
- **dPL Base–TGD**：`+0.0195 → +0.0310 → +0.0791 → +0.1070 → +0.0851`，同样是低—中等 snow activity 增加，高雪层级平台/回落。

仅从分解看，Base–CN 的 IC 梯度由 `D_between` 增大（S1→S5 约 `+0.0900`）和 `D_within` 下降（约 `−0.0209`）共同构成；Base–CN 的 dPL 梯度主要由 `D_between` 增大（约 `+0.1744`），而 `D_within` 略增。Base–TGD 的 IC 梯度同时包含 `D_between` 增大约 `+0.0411` 与 `D_within` 下降约 `−0.0486`；dPL Base–TGD 主要是 `D_between` 增大约 `+0.1439`，`D_within` 近似不变。这里是距离分解的描述，不是机制解释。

## 4. Base–CN 与 Base–TGD 的配对差异

定义严格为：

\[
\delta E_b=E_{\mathrm{Base-CN},b}-E_{\mathrm{Base-TGD},b}.
\]

`deltaE` 只表示同一 basin、同一 estimation constraint 下两个结构对照的参数分离程度差异；不表示 CN 的“物理性”、CN 独有融雪贡献、过程贡献量或补偿因果量。

### 表 C

| Estimation | Snow class | N | Median deltaE | 95% CI | deltaE>0 n | deltaE>0 % |
|---|---|---:|---:|---|---:|---:|
| IC | Full | 531 | −0.001446 | [−0.003381, +0.001068] | 256 | 48.2% |
| IC | S1 | 165 | −0.004570 | [−0.009124, +0.000516] | 73 | 44.2% |
| IC | S2 | 156 | −0.000873 | [−0.003073, +0.002955] | 76 | 48.7% |
| IC | S3 | 121 | −0.001291 | [−0.004646, +0.001992] | 56 | 46.3% |
| IC | S4 | 34 | +0.015452 | [+0.001178, +0.020502] | 23 | 67.6% |
| IC | S5 | 55 | +0.001786 | [−0.024308, +0.021905] | 28 | 50.9% |
| dPL | Full | 531 | +0.017713 | [+0.012278, +0.023620] | 343 | 64.6% |
| dPL | S1 | 165 | +0.004676 | [−0.000395, +0.009104] | 93 | 56.4% |
| dPL | S2 | 156 | +0.028708 | [+0.016873, +0.036764] | 108 | 69.2% |
| dPL | S3 | 121 | +0.023893 | [+0.015058, +0.033391] | 79 | 65.3% |
| dPL | S4 | 34 | +0.036251 | [+0.007204, +0.102898] | 24 | 70.6% |
| dPL | S5 | 55 | +0.031728 | [+0.006488, +0.048997] | 39 | 70.9% |

### 表 C 结果判断

- IC 全样本 `median deltaE = −0.001446`，CI 跨 0，`deltaE>0 = 48.2%`，不支持 Base–CN 相对 Base–TGD 存在稳定的全样本额外分离。
- dPL 全样本 `median deltaE = +0.017713`，CI 为正，`deltaE>0 = 64.6%`；在当前 shared cross-basin parameter learning 约束下，Base–CN 的分离量高于 Base–TGD。该判断不应升级为物理性或过程贡献判断。
- IC 的 S4 为正但样本量仅 34，S5 CI 跨 0；不应据此写出稳定的高雪结构差异。
- dPL 的 S2–S5 点估计均为正，S2–S5 的 CI 亦为正，但这只是当前参数估计约束下的结构对照差异。

## 5. 15个共享参数的具体变化

### 5.1 参数名称、中文含义与方向

以下中文含义是依据 canonical Table S1 / `authoritative_15_parameter_specs.csv` 的 XAJ 参数定义作的忠实中文对应；代码字段保留为 `xaj_*`，不引入新的物理概念。

| Code field | Symbol | 中文含义 | Canonical English definition |
|---|---|---|---|
| `xaj_k` | k | 潜在蒸散发与参考作物蒸发比 | Ratio of potential ET to reference crop evaporation |
| `xaj_b` | b | 张力水容量曲线指数 | Exponent of tension water capacity curve |
| `xaj_im` | im | 不透水面积比例 | Impervious area fraction |
| `xaj_um` | um | 上层张力水容量 | Upper layer tension water capacity |
| `xaj_lm` | lm | 下层张力水容量 | Lower layer tension water capacity |
| `xaj_dm` | dm | 深层张力水容量 | Deep layer tension water capacity |
| `xaj_c` | c | 深层蒸发系数 | Deep layer evaporation coefficient |
| `xaj_sm` | sm | 表层自由水容量的面平均值 | Areal mean free water capacity of surface layer |
| `xaj_ex` | ex | 自由水容量曲线指数 | Exponent of free water capacity curve |
| `xaj_ki` | ki | 壤中流出流系数 | Outflow coefficient for interflow |
| `xaj_kg` | kg | 地下水出流系数 | Outflow coefficient for groundwater |
| `xaj_ci` | ci | 壤中流水库退水常数 | Recession constant for interflow reservoir |
| `xaj_cg` | cg | 地下水水库退水常数 | Recession constant for groundwater reservoir |
| `xaj_a` | a | Gamma-UH 形状参数 | Gamma-UH shape parameter |
| `xaj_theta` | theta | Gamma-UH 尺度参数 | Gamma-UH scale parameter |

方向定义固定为：

\[
\Delta\theta=\theta_{\mathrm{Base}}-\theta_{\mathrm{compared}},
\]

其中 `theta` 指归一化参数坐标 `z`，`compared` 为 CN 或 TGD。因而表 D 中的 slope 是 **归一化参数位移对连续 `f_snow` 的 OLS slope**，单位为“归一化参数单位 / `f_snow` 单位”；不是物理量斜率，也不是相对于参数真值的误差。

### 5.2 表 D：全 15 参数

Base–CN 的数值直接读取 `r2_snow_gradients_summary.csv` 与 `r2_parameter_shifts_strata_summary.csv`。Base–TGD 没有单独的 canonical all-15 summary 文件；以下 Base–TGD 行是从 `r2_parameter_values_canonical.csv` 逐 basin 配对计算 `z_Base-z_TGD` 的审计派生结果。其 slope CI 与 endpoint CI 为 paired-basin percentile bootstrap（10,000 次；seed schedule：slope=`20260730+16000+j`，endpoint=`20260730+17000+5j`，`j` 为 canonical 15 参数顺序索引）。

| Parameter | 中文含义 | Contrast | Estimation | OLS slope | slope 95% CI | Spearman rho | S1 median | S5 median | S5–S1 | endpoint 95% CI |
|---|---|---|---|---:|---|---:|---:|---:|---:|---|
| `xaj_k` (k) | 潜在蒸散发与参考作物蒸发比 | Base–CN | IC | −0.028460 | [−0.092388, +0.031860] | −0.126143 | 0.000000 | 0.000000 | 0.000000 | [0.000000, +0.002010] |
| `xaj_b` (b) | 张力水容量曲线指数 | Base–CN | IC | −0.005931 | [−0.184492, +0.193991] | +0.023066 | 0.000000 | −0.022771 | −0.022771 | [−0.242977, 0.000000] |
| `xaj_im` (im) | 不透水面积比例 | Base–CN | IC | −0.363479 | [−0.501575, −0.212196] | −0.247684 | 0.000000 | −0.081506 | −0.081506 | [−0.227982, 0.000000] |
| `xaj_um` (um) | 上层张力水容量 | Base–CN | IC | +0.521123 | [+0.334504, +0.705945] | +0.216372 | 0.000000 | +0.038135 | +0.038135 | [+0.000000, +0.524190] |
| `xaj_lm` (lm) | 下层张力水容量 | Base–CN | IC | +0.400108 | [+0.231230, +0.572064] | +0.189116 | 0.000000 | +0.006873 | +0.006873 | [+0.000000, +0.386704] |
| `xaj_dm` (dm) | 深层张力水容量 | Base–CN | IC | −0.537260 | [−0.748618, −0.272617] | −0.115677 | 0.000000 | −0.540617 | −0.540617 | [−0.931039, −0.101899] |
| `xaj_c` (c) | 深层蒸发系数 | Base–CN | IC | +0.278299 | [+0.048393, +0.518423] | +0.084518 | 0.000000 | 0.000000 | 0.000000 | [+0.000000, +0.144620] |
| `xaj_sm` (sm) | 表层自由水容量的面平均值 | Base–CN | IC | +0.161483 | [−0.007756, +0.347938] | +0.105768 | −0.000024 | +0.008740 | +0.008763 | [+0.000000, +0.152672] |
| `xaj_ex` (ex) | 自由水容量曲线指数 | Base–CN | IC | −0.370161 | [−0.565363, −0.190384] | −0.077379 | 0.000000 | −0.054870 | −0.054870 | [−0.220144, 0.000000] |
| `xaj_ki` (ki) | 壤中流出流系数 | Base–CN | IC | −0.474569 | [−0.638043, −0.312112] | −0.236691 | 0.000000 | 0.000000 | 0.000000 | [−0.144250, 0.000000] |
| `xaj_kg` (kg) | 地下水出流系数 | Base–CN | IC | −0.202880 | [−0.410933, +0.004566] | −0.204953 | 0.000000 | −0.068674 | −0.068674 | [−0.179649, −0.011529] |
| `xaj_ci` (ci) | 壤中流水库退水常数 | Base–CN | IC | −0.413816 | [−0.625345, −0.202897] | −0.183221 | 0.000000 | −0.240228 | −0.240228 | [−0.468681, 0.000000] |
| `xaj_cg` (cg) | 地下水水库退水常数 | Base–CN | IC | +0.292470 | [+0.147972, +0.449560] | +0.007194 | +0.004482 | +0.208669 | +0.204188 | [+0.081758, +0.332284] |
| `xaj_a` (a) | Gamma-UH 形状参数 | Base–CN | IC | +0.036294 | [−0.157337, +0.258977] | +0.037734 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| `xaj_theta` (theta) | Gamma-UH 尺度参数 | Base–CN | IC | −0.016638 | [−0.255455, +0.234684] | −0.081070 | 0.000000 | 0.000000 | 0.000000 | [−0.021169, 0.000000] |
| `xaj_k` (k) | 潜在蒸散发与参考作物蒸发比 | Base–CN | dPL | −0.003561 | [−0.046869, +0.037174] | −0.071871 | +0.005076 | −0.002377 | −0.007454 | [−0.020475, +0.004982] |
| `xaj_b` (b) | 张力水容量曲线指数 | Base–CN | dPL | +0.023791 | [−0.051389, +0.116792] | +0.087595 | +0.001702 | +0.009466 | +0.007764 | [−0.006639, +0.038730] |
| `xaj_im` (im) | 不透水面积比例 | Base–CN | dPL | −0.141777 | [−0.220506, −0.055934] | −0.345184 | −0.000128 | −0.020224 | −0.020096 | [−0.041637, −0.006125] |
| `xaj_um` (um) | 上层张力水容量 | Base–CN | dPL | +0.565542 | [+0.377397, +0.754283] | +0.252634 | +0.005412 | +0.332704 | +0.327292 | [+0.089233, +0.768792] |
| `xaj_lm` (lm) | 下层张力水容量 | Base–CN | dPL | +0.092166 | [+0.015728, +0.195223] | +0.094089 | +0.015632 | +0.026470 | +0.010838 | [−0.016063, +0.031402] |
| `xaj_dm` (dm) | 深层张力水容量 | Base–CN | dPL | −0.106122 | [−0.208600, +0.002147] | −0.070553 | −0.005055 | −0.013054 | −0.007999 | [−0.021578, +0.001730] |
| `xaj_c` (c) | 深层蒸发系数 | Base–CN | dPL | +0.253485 | [+0.115231, +0.380596] | +0.244465 | +0.001164 | +0.014498 | +0.013334 | [+0.001224, +0.058759] |
| `xaj_sm` (sm) | 表层自由水容量的面平均值 | Base–CN | dPL | +0.268218 | [+0.182194, +0.366514] | +0.147812 | +0.000306 | +0.117803 | +0.117498 | [+0.038607, +0.180332] |
| `xaj_ex` (ex) | 自由水容量曲线指数 | Base–CN | dPL | −0.095379 | [−0.156395, −0.034684] | −0.095551 | −0.002789 | −0.022671 | −0.019882 | [−0.037748, +0.000785] |
| `xaj_ki` (ki) | 壤中流出流系数 | Base–CN | dPL | −0.314584 | [−0.440408, −0.193804] | −0.327635 | −0.012114 | −0.265222 | −0.253108 | [−0.332464, −0.124131] |
| `xaj_kg` (kg) | 地下水出流系数 | Base–CN | dPL | +0.011042 | [−0.080698, +0.104603] | −0.088103 | −0.007575 | +0.001819 | +0.009394 | [−0.010666, +0.024651] |
| `xaj_ci` (ci) | 壤中流水库退水常数 | Base–CN | dPL | −0.531236 | [−0.704039, −0.365714] | −0.290449 | −0.023406 | −0.198383 | −0.174977 | [−0.636208, −0.062174] |
| `xaj_cg` (cg) | 地下水水库退水常数 | Base–CN | dPL | +0.059040 | [−0.037957, +0.158272] | −0.118731 | −0.016669 | −0.095962 | −0.079293 | [−0.108215, −0.010398] |
| `xaj_a` (a) | Gamma-UH 形状参数 | Base–CN | dPL | −0.012214 | [−0.121203, +0.103530] | −0.011317 | −0.001442 | −0.013417 | −0.011975 | [−0.075177, +0.006802] |
| `xaj_theta` (theta) | Gamma-UH 尺度参数 | Base–CN | dPL | +0.130386 | [+0.049679, +0.224312] | +0.109449 | −0.000099 | +0.000316 | +0.000415 | [−0.001112, +0.002176] |
| `xaj_k` (k) | 潜在蒸散发与参考作物蒸发比 | Base–TGD | IC | +0.000118 | [−0.041354, +0.040570] | −0.011687 | 0.000000 | 0.000000 | 0.000000 | [−0.000034, 0.000000] |
| `xaj_b` (b) | 张力水容量曲线指数 | Base–TGD | IC | +0.173210 | [+0.004929, +0.342462] | +0.107598 | 0.000000 | 0.000000 | 0.000000 | [0.000000, +0.088769] |
| `xaj_im` (im) | 不透水面积比例 | Base–TGD | IC | +0.096378 | [+0.002096, +0.202933] | +0.038758 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| `xaj_um` (um) | 上层张力水容量 | Base–TGD | IC | +0.256363 | [+0.061897, +0.454877] | +0.116658 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| `xaj_lm` (lm) | 下层张力水容量 | Base–TGD | IC | +0.185835 | [+0.011747, +0.361796] | +0.073280 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| `xaj_dm` (dm) | 深层张力水容量 | Base–TGD | IC | −0.098037 | [−0.326423, +0.130190] | −0.070295 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| `xaj_c` (c) | 深层蒸发系数 | Base–TGD | IC | +0.152715 | [−0.069431, +0.373669] | +0.103390 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| `xaj_sm` (sm) | 表层自由水容量的面平均值 | Base–TGD | IC | −0.238914 | [−0.391756, −0.080028] | +0.038552 | 0.000000 | 0.000000 | 0.000000 | [−0.003739, +0.000983] |
| `xaj_ex` (ex) | 自由水容量曲线指数 | Base–TGD | IC | +0.034857 | [−0.091307, +0.165146] | +0.069176 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| `xaj_ki` (ki) | 壤中流出流系数 | Base–TGD | IC | −0.097668 | [−0.217753, +0.010896] | −0.143155 | 0.000000 | 0.000000 | 0.000000 | [−0.000611, +0.004293] |
| `xaj_kg` (kg) | 地下水出流系数 | Base–TGD | IC | +0.030497 | [−0.171203, +0.211869] | −0.093609 | 0.000000 | −0.034134 | −0.034134 | [−0.067661, +0.002066] |
| `xaj_ci` (ci) | 壤中流水库退水常数 | Base–TGD | IC | −0.172807 | [−0.379695, +0.037994] | −0.111233 | +0.000897 | 0.000000 | −0.000897 | [−0.033118, 0.000000] |
| `xaj_cg` (cg) | 地下水水库退水常数 | Base–TGD | IC | +0.439135 | [+0.315526, +0.562425] | +0.148554 | 0.000000 | +0.193355 | +0.193355 | [+0.161661, +0.286455] |
| `xaj_a` (a) | Gamma-UH 形状参数 | Base–TGD | IC | −0.321102 | [−0.524558, −0.097638] | −0.070173 | 0.000000 | 0.000000 | 0.000000 | [−0.449830, 0.000000] |
| `xaj_theta` (theta) | Gamma-UH 尺度参数 | Base–TGD | IC | −0.280452 | [−0.485504, −0.070583] | −0.182090 | 0.000000 | 0.000000 | 0.000000 | [−0.153370, 0.000000] |
| `xaj_k` (k) | 潜在蒸散发与参考作物蒸发比 | Base–TGD | dPL | +0.032316 | [+0.010588, +0.055195] | +0.060046 | +0.006852 | +0.016237 | +0.009385 | [+0.001128, +0.024673] |
| `xaj_b` (b) | 张力水容量曲线指数 | Base–TGD | dPL | −0.104443 | [−0.192764, −0.016947] | −0.007532 | +0.001964 | −0.002669 | −0.004633 | [−0.043556, +0.027724] |
| `xaj_im` (im) | 不透水面积比例 | Base–TGD | dPL | −0.005326 | [−0.012473, +0.001961] | −0.096442 | −0.000000 | −0.000309 | −0.000309 | [−0.001638, +0.000120] |
| `xaj_um` (um) | 上层张力水容量 | Base–TGD | dPL | +0.309115 | [+0.131908, +0.485101] | +0.158352 | +0.002246 | +0.048659 | +0.046414 | [+0.011060, +0.200886] |
| `xaj_lm` (lm) | 下层张力水容量 | Base–TGD | dPL | +0.035437 | [−0.047979, +0.124371] | +0.000639 | +0.014237 | +0.002714 | −0.011523 | [−0.028046, +0.012244] |
| `xaj_dm` (dm) | 深层张力水容量 | Base–TGD | dPL | −0.087289 | [−0.148433, −0.038400] | −0.143369 | −0.000111 | −0.003719 | −0.003609 | [−0.009032, −0.000066] |
| `xaj_c` (c) | 深层蒸发系数 | Base–TGD | dPL | +0.478776 | [+0.308923, +0.671439] | +0.312353 | +0.003712 | +0.020027 | +0.016315 | [+0.004446, +0.606434] |
| `xaj_sm` (sm) | 表层自由水容量的面平均值 | Base–TGD | dPL | +0.076833 | [+0.027318, +0.127543] | +0.131975 | +0.009111 | +0.033169 | +0.024058 | [−0.004836, +0.047813] |
| `xaj_ex` (ex) | 自由水容量曲线指数 | Base–TGD | dPL | −0.143779 | [−0.222578, −0.073571] | −0.143384 | 0.000000 | −0.012858 | −0.012858 | [−0.040032, +0.002950] |
| `xaj_ki` (ki) | 壤中流出流系数 | Base–TGD | dPL | −0.006009 | [−0.076511, +0.058642] | −0.085252 | −0.028802 | −0.012595 | +0.016207 | [−0.023492, +0.048232] |
| `xaj_kg` (kg) | 地下水出流系数 | Base–TGD | dPL | −0.100679 | [−0.191901, −0.021792] | −0.195719 | −0.016345 | −0.058917 | −0.042573 | [−0.078547, −0.003616] |
| `xaj_ci` (ci) | 壤中流水库退水常数 | Base–TGD | dPL | −0.130984 | [−0.216470, −0.045163] | −0.069397 | −0.035180 | −0.122006 | −0.086826 | [−0.159931, −0.038140] |
| `xaj_cg` (cg) | 地下水水库退水常数 | Base–TGD | dPL | +0.161656 | [+0.077565, +0.255712] | +0.237366 | −0.013761 | −0.003309 | +0.010453 | [−0.017028, +0.044721] |
| `xaj_a` (a) | Gamma-UH 形状参数 | Base–TGD | dPL | −0.096197 | [−0.241975, +0.045686] | +0.010759 | −0.014294 | −0.032059 | −0.017765 | [−0.050087, +0.018839] |
| `xaj_theta` (theta) | Gamma-UH 尺度参数 | Base–TGD | dPL | −0.142202 | [−0.289892, −0.006537] | −0.123570 | +0.000592 | +0.000269 | −0.000324 | [−0.001089, +0.001240] |

> **读表注意**：许多 IC 的 S1/S5 median 恰为 0，是位移分布的中位数，不代表该参数在所有 basin 没有变化；它也不能被写成“参数完全不变”。同样，单个参数的方向不能被写成该参数单独替代了融雪过程。

## 6. 重点参数筛选与边界检查

### 6.1 表 E：Base–CN 主体下的 15 参数筛选

“Boundary mass”定义为 canonical vector 的 `z≤0.01` 或 `z≥0.99` 比例；表中数字为 Base、CN 两种融雪结构与 TGD 通用温度条件控制的三类 canonical vector 合并后、S1→S5 的百分比，分别列 IC 与 dPL。筛选按方向重复性、S1→S5 组织、物理含义和边界风险综合判断，不按 p 值排序。

| Parameter | IC direction | dPL direction | S1–S5 organization（Base–CN；IC / dPL） | Boundary concern（near-boundary mass S1→S5） | Suggested placement |
|---|---|---|---|---|---|
| `xaj_k` (k) | − | − | 0→0；+0.005→−0.002，弱/混合 | IC 20→57%；dPL 5→12%，中等 | Appendix |
| `xaj_b` (b) | − | + | 0→−0.023；+0.002→+0.009，方向不一致 | IC 41→48%；dPL 8→7%，中等/低 | Appendix |
| `xaj_im` (im) | − | − | 0→−0.082；0→−0.020，负向但边界主导 | IC 82→81%；dPL 75→59%，高 | Supporting / Appendix |
| `xaj_um` (um) | + | + | 0→+0.038；+0.005→+0.333，正向且 dPL 更清楚 | IC 75→88%；dPL 16→24%，IC 高 | Main（边界脚注） |
| `xaj_lm` (lm) | + | + | 0→+0.007；+0.016→+0.026，正向但 dPL 温和 | IC 61→73%；dPL 3→7%，IC 中高 | Supporting |
| `xaj_dm` (dm) | − | − | 0→−0.541；−0.005→−0.013，方向重复、IC 更强 | IC 57→79%；dPL 31→42%，中高/中等 | Supporting |
| `xaj_c` (c) | + | + | 0→0；+0.001→+0.014，正向但 IC 中位数钉扎 | IC 72→82%；dPL 47→45%，高/中等 | Supporting |
| `xaj_sm` (sm) | + | + | −0.000→+0.009；0→+0.118，dPL 端点移动清楚 | IC 21→72%；dPL 9→21%，IC S5 高 | Supporting |
| `xaj_ex` (ex) | − | − | 0→−0.055；−0.003→−0.023，负向重复 | IC 65→84%；dPL 7→29%，高/低中 | Supporting |
| `xaj_ki` (ki) | − | − | 0→0；−0.012→−0.265，方向重复但 IC endpoint 中位数为 0 | IC 32→85%；dPL 5→20%，IC S5 高 | Main（IC 边界脚注） |
| `xaj_kg` (kg) | − | + | 0→−0.069；−0.008→+0.002，方向不一致 | IC 55→27%；dPL 10→7%，中等/低 | Appendix |
| `xaj_ci` (ci) | − | − | 0→−0.240；−0.023→−0.198，方向和端点均重复 | IC 32→61%；dPL 1→0%，中等/低 | Main |
| `xaj_cg` (cg) | + | + | +0.004→+0.209；−0.017→−0.096，端点方向相反 | IC 40→12%；dPL 5→7%，中等/低 | Appendix |
| `xaj_a` (a) | + | − | 0→0；−0.001→−0.013，弱/方向混合 | IC 63→67%；dPL 2→5%，中等/低 | Appendix |
| `xaj_theta` (theta) | − | + | 0→0；−0.000→0，近零/混合 | IC 72→68%；dPL 62→44%，高/中等 | Appendix |

### 6.2 重点参数的 exact bound-hit audit

下列数字为各结构在 S1/S5 的 `z` 分布：`0:x` 表示 exact zero share，`1:x` 表示 exact one share，`near:x` 表示 `z≤0.01` 或 `z≥0.99` share。顺序为 S1 / S5。

| Parameter | Estimation | Base | CN | TGD |
|---|---|---|---|---|
| `xaj_um` | IC | 0:35.2% / 16.4%; 1:37.0% / 81.8%; near:73.3% / 98.2% | 0:33.9% / 34.5%; 1:38.2% / 36.4%; near:72.7% / 70.9% | 0:37.0% / 27.3%; 1:41.2% / 67.3%; near:79.4% / 94.5% |
| `xaj_um` | dPL | 0:0% / 0%; 1:0% / 0%; near:17.0% / 38.2% | 0:0% / 0%; 1:0% / 0%; near:14.5% / 10.9% | 0:0% / 0%; 1:0% / 0%; near:16.4% / 21.8% |
| `xaj_ki` | IC | 0:12.1% / 96.4%; 1:20.0% / 0%; near:32.7% / 96.4% | 0:12.7% / 49.1%; 1:18.8% / 12.7%; near:31.5% / 63.6% | 0:12.1% / 92.7%; 1:18.2% / 1.8%; near:30.3% / 96.4% |
| `xaj_ki` | dPL | 0:0% / 0%; 1:0% / 0%; near:9.1% / 36.4% | 0:0% / 0%; 1:0% / 0%; near:0% / 0% | 0:0% / 0%; 1:0% / 0%; near:5.5% / 23.6% |
| `xaj_ci` | IC | 0:25.5% / 30.9%; 1:3.0% / 30.9%; near:30.3% / 61.8% | 0:24.2% / 7.3%; 1:6.1% / 36.4%; near:31.5% / 47.3% | 0:32.1% / 27.3%; 1:1.8% / 43.6%; near:35.2% / 74.5% |
| `xaj_ci` | dPL | 0:0% / 0%; 1:0% / 0%; near:1.2% / 0% | 0:0% / 0%; 1:0% / 0%; near:0% / 0% | 0:0% / 0%; 1:0% / 0%; near:0.6% / 0% |
| `xaj_im` | IC | 0:78.2% / 87.3%; 1:1.8% / 5.5%; near:80.6% / 92.7% | 0:83.6% / 40.0%; 1:1.2% / 18.2%; near:85.5% / 58.2% | 0:77.0% / 90.9%; 1:2.4% / 1.8%; near:80.0% / 92.7% |
| `xaj_im` | dPL | 0:0% / 0%; 1:0% / 0%; near:77.6% / 78.2% | 0:0% / 0%; 1:0% / 0%; near:66.1% / 25.5% | 0:0% / 0%; 1:0% / 0%; near:81.8% / 74.5% |

**边界检查结论**：

- `u_m` 的 IC canonical vectors 在 S5 大量接近上界，因而正文可以保留其稳定的方向和雪活跃度组织，但不能把 slope 单独写成无约束的物理变化；应配合 boundary 脚注和 dPL 结果。
- `k_i` 的 IC Base/TGD 在 S5 大量接近下界，IC 的 S1/S5 median shift 为 0；正文如保留，须写成 slope/分布组织信号，不能写成 IC 中位数端点移动。
- `c_i` 在 IC 具有中等边界质量，但 dPL 几乎没有 bound-hit，且两个 estimation constraint 的 slope/endpoint 方向一致，适合作为主体方向之一。
- `i_m` 在 IC 与 dPL 的 near-boundary mass 都很高；其负向 slope 虽在 IC/dPL 重复，但更可能受参数边界与不透水比例的集中质量影响，降为 supporting/appendix，不作为最强的单参数示范。

## 7. IC 与 dPL 的可比与不可比内容

### 可以支持的比较

1. **方向重复性**：Base–CN 下，15 个参数 OLS slope 的 IC/dPL sign agreement 为 `11/15 = 73.3%`；四个候选重点参数 `u_m/k_i/c_i/i_m` 为 `4/4`。
2. **辅助排序统计**：15 个 Base–CN slope 的 IC 与 dPL Spearman correlation 为 `rho=0.854`。这是辅助一致性描述，不是新的主指标或假设检验。
3. **梯度形态**：IC 在 Base–CN 下为 S1→S5 逐步增加；dPL 在 S1→S3 快速增加并在 S3–S5 平台。Base–TGD 分别重复了“IC 增加”和“dPL 先升后平台/回落”的总体形态。
4. **重点参数**：`u_m` 正向、`k_i` 负向、`c_i` 负向在 IC/dPL 下重复；但 `i_m` 必须伴随边界风险限定。
5. **Base–CN 相对 Base–TGD**：两种 estimation constraint 下都可报告 `deltaE`，但只能说当前参数估计约束下两种结构对照的分离程度不同。IC Full 的 deltaE 接近 0，dPL Full 为正。

### 不可以支持的比较

- 不可因为 dPL 的 `E` 绝对值较大就说“dPL 参数补偿更强”。IC 的 `D_within` 来自 10 次 CMA-ES restart 优化离散，dPL 的 `D_within` 来自 3 个随机 seed 训练离散；两者的绝对离散尺度和重复机制不同。
- 不把 IC/dPL 的差异写成参数估计方式造成的因果效应；只能写为两种参数约束条件下的不同表达形式。
- 不把 `u_m/k_i/c_i/i_m` 的单参数 slope 写成该参数单独“替代融雪过程”。推荐用“共享参数的补偿性参数变化”“重复出现的方向性参数特征”或“参数空间中的 recurring directional signatures”。

### 3.3 推荐总表述

数据支持以下强度的表述：

> **IC 与 dPL 下均出现随积雪活动程度组织的共享参数变化，但具体参数变化的端点、幅度和边界受约束表达存在差异。**

可以进一步说明：Base–CN 是主体结构对照；TGD 是通用温度条件控制。不要写成 IC/dPL 优劣比较，也不要把 TGD 差异解释成物理性排序。

## 8. Manuscript / Supplement 已有证据 vs 博士论文仍缺内容

| 项目 | 状态 | 判断 |
|---|---|---|
| manuscript 已有 Base–CN IC `E>0=63.1%` | 已有且已核准 | `335/531`，canonical 直接支持 |
| manuscript 已有 Base–CN dPL `E>0=83.8%` | 已有且已核准 | `445/531`，canonical 直接支持 |
| manuscript 已有 Base–CN IC S1–S5 median E | 已有且已核准 | 五层点估计与梯度形态完全支持 |
| manuscript 已有 Base–CN dPL S1–S5 median E | 已有且已核准 | 五层点估计与平台形态完全支持 |
| Base–TGD 完整 E 梯度 | 需要从 canonical 补齐 | `r2_tgd2_specificity_*` 已有，本报告已整理表 B；不需新率定 |
| 表 A 的 `D_between/D_within` Q25–Q75 | 需要轻量汇总 | basin-level 文件已有，直接 quantile；不是新科学分析 |
| 表 C 全样本与 S1–S5 deltaE | 已有且已核准 | paired summary 已给出，本报告补上 n |
| 全 15 参数的 Base–CN 表 D | 已有且已核准 | full + strata summary 已冻结 |
| 全 15 参数的 Base–TGD 表 D | 需要轻量汇总 | canonical vector 已有；本报告以 audit-derived 行补齐，并明确 CI provenance |
| S1/S5 boundary-hit audit | 推荐补 | 不构成 blocker；本报告已完成重点参数和全 15 参数 near-boundary 摘要 |
| IC–dPL slope sign agreement | 推荐补 | 轻量辅助统计 `11/15`、`rho=0.854`；不建立新的主假设 |
| `D_between/D_within` 分解 | 必须保留 | 这是回答“结构间变化还是结构内离散变化”的关键，表 A/B 和正文都应保留 |
| 15 参数全部 S1–S5 数值 | 附录足够 | 不宜放正文；正文展示重点参数端点和 slope |
| region-omission robustness for R2 | 不需要作为 blocker | 现有 `r2_snow_gradient_robustness.csv` 是 ExcludeS5/leave-one-stratum-out，不是地理 region omission；当前没有证据要求为 3.3 新增地理分区扫描，可放附录或不补 |
| 其他环境属性与参数变化相关性 | 不建议新增 | 属于后续属性关系问题，不是 3.3 必要证据 |
| 新损失函数实验 | 不建议新增 | 超出本节范围 |
| 新参数率定 | 不建议新增 | frozen outputs 已足够 |
| 新 dPL 训练 | 不建议新增 | frozen 3-seed outputs 已足够 |

## 9. 3.3 正文图表规划

### 图 3-X：15维参数空间结构分离与积雪活动梯度

建议保留四个 panel，不再增加更多指标：

- **(a) `D_between` vs `D_within`**：按 IC/dPL 与 Base–CN/Base–TGD 展示 median 和 Q25–Q75；用零差异参考线或直接标注 median `E`。不要把 selected-vector `D_rms` 混入。
- **(b) IC 的 `E` S1–S5**：Base–CN 为主体线，Base–TGD 用低饱和度辅助线。
- **(c) dPL 的 `E` S1–S5**：同样 Base–CN 为主体，TGD 为 matched control；标出 dPL 的 S1→S3 上升和 S3–S5 平台。
- **(d) paired `deltaE`**：IC/dPL 两条分层线，带 0 参考线；避免把它画成 CN 的“过程贡献量”。

四 panel 是合适的最小配置：panel (a) 回答结构间与结构内距离的分解，(b)/(c) 回答 snow gradient，(d) 回答 TGD control。压缩为两个 panel 会丢失 D 分解或 paired control 的信息。

### 图 3-Y：重点共享参数的补偿性变化

建议 2×2 四个 panel，保留现有 Figure 4 已选的四个参数：`u_m, k_i, c_i, i_m`。

- 横轴为 S1–S5；纵轴为 `Delta theta=z_Base-z_compared`。
- Base–CN 为主体实线，IC/dPL 用不同线型；Base–TGD 用较细、低饱和度虚线作 matched-control 辅助。
- `i_m` 面板明确标为 secondary / boundary-sensitive；不要给四个参数完全相同的主叙事权重。
- 如果版面需要最简化，可把 TGD 放为同图辅助线，不单独占用与 Base–CN 相同的主体篇幅。

### 正文表（最多 1 张）

建议表题为“融雪过程结构变化与通用温度条件控制下主要共享参数的积雪活动响应”，只列 `u_m/k_i/c_i/i_m`：

- 中文含义；
- Base–CN slope（IC、dPL）；
- Base–CN S5–S1（IC、dPL）；
- 以一列简短文字注明 Base–TGD 方向；
- `i_m` 的边界敏感性放脚注。

完整 15 参数数值、所有 CI 和 Base–TGD audit-derived 结果放附录。

## 10. 建议进入正文的关键数字

控制在以下 **15 组**，其余数字放表格或附录：

1. **样本与分层**：`N=531`；S1–S5 分别为 `165/156/121/34/55`。
2. **IC Base–CN 总体分离**：median `E=+0.0075`，95% CI `[+0.0050,+0.0107]`，`E>0=63.1% (335/531)`。
3. **dPL Base–CN 总体分离**：median `E=+0.0611`，95% CI `[+0.0504,+0.0796]`，`E>0=83.8% (445/531)`。
4. **IC Base–TGD 总体控制**：median `E=+0.0074`，95% CI `[+0.0051,+0.0098]`，`E>0=63.7%`。
5. **dPL Base–TGD 总体控制**：median `E=+0.0376`，95% CI `[+0.0324,+0.0440]`，`E>0=79.3%`。
6. **IC Base–CN 梯度**：S1→S5 为 `−0.0018→+0.0829`，逐步增加。
7. **dPL Base–CN 梯度**：S1→S5 为 `+0.0186→+0.1252`，S1→S3 快速增加后平台。
8. **IC Base–TGD 梯度**：S1→S5 为 `+0.0006→+0.0937`，与 IC Base–CN 形态相似。
9. **dPL Base–TGD 梯度**：S1→S5 为 `+0.0195→+0.0851`，中等 snow activity 后平台/回落。
10. **IC paired deltaE**：Full median `−0.0014`，95% CI `[−0.0034,+0.0011]`，positive `48.2%`。
11. **dPL paired deltaE**：Full median `+0.0177`，95% CI `[+0.0123,+0.0236]`，positive `64.6%`。
12. **u_m**：Base–CN slope IC/dPL `+0.521/+0.566`；S5–S1 `+0.038/+0.327`。
13. **k_i**：Base–CN slope IC/dPL `−0.475/−0.315`；S5–S1 `0/−0.253`，IC 端点受下界集中影响。
14. **c_i**：Base–CN slope IC/dPL `−0.414/−0.531`；S5–S1 `−0.240/−0.175`。
15. **i_m**：Base–CN slope IC/dPL `−0.363/−0.142`；S5–S1 `−0.082/−0.020`，应同时注明高 boundary mass、降为 secondary。

## 11. 建议进入附录的数据

1. 表 A 的 `D_between/D_within` Q25–Q75、所有 basin-level quantile 与 `E` 分布。
2. 表 B 的 4 个 estimation×contrast 组合完整 S1–S5 数值、CI 和 `E>0` 分子。
3. 表 C 的全部 Full、ExcludeS5、S1–S5 paired deltaE 数值和 CI。
4. 表 D 的全 15 参数、两种 contrast、两种 estimation 的 slope、rho、S1、S5、endpoint 和 CI。
5. 表 E 全 15 参数方向筛选表，以及完整 boundary proximity / exact bound-hit audit。
6. IC 10-restart 与 dPL 3-seed 的重复估计分布、`within_base/within_cn/within_pooled` 的定义和原始计算追踪。
7. `r2_snow_gradient_robustness.csv` 的 ExcludeS5 与 leave-one-stratum-out 结果；若审稿人要求再放 geographic region-omission，则另列为方法稳健性，不混入 3.3 主证据。
8. canonical file path、hash、bootstrap seed/draw 说明和 manuscript-vs-canonical reconciliation。

## 12. 数据冲突、旧版本与待确认问题

### 12.1 已解决的版本冲突

- `analysis/R2/results/` 与 `manuscript/results/R2/` 的关键点估计一致；少数 CSV 浮点序列化差异最大约 `10^-7`，不影响判断。`analysis/R2/results/` 由 README 的运行时权威规则确定为本报告唯一主来源。
- `r2_within_structure_summary.csv` 与 `r2_tgd2_specificity_summary.csv` 都包含 Base–CN 行，但 CI 不完全相同。例如 IC Full `E` CI 分别为 `[0.004953,0.010655]` 与 `[0.005162,0.010334]`；dPL S5 分别为 `[0.062766,0.218888]` 与 `[0.078706,0.218888]`。两者 median、方向和 prevalence 相同，说明主要是不同汇总/bootstrap 版本，不是点估计冲突。处理规则为：Base–CN 使用其专用 primary `r2_within_structure_summary.csv`；Base–TGD 使用其专用 `r2_tgd2_specificity_summary.csv`；不跨文件替换 CI。
- narrative `r2_final_closure_report.md` / `results_summary.md` 与直接 CSV 存在 CI 末位或小样本差异，例如 IC Base–CN S1 报告约 `[−0.0044,+0.0021]`，而专用 CSV 为 `[−0.004415,+0.000908]`；dPL Base–CN S5 narrative 下界约 `+0.0826`，专用 CSV 为 `+0.062766`。这些 narrative 用于交叉检查，最终表采用 direct machine-readable source，并在正文避免过度强调小样本 CI。
- `r2_statistical_audit_report.md` / `r2_historical_reconciliation.csv` 中的部分 15 参数 slope CI 也与 `r2_snow_gradients_summary.csv` 不同，例如 `xaj_um` IC 约 `[0.3529,0.7064]` 对比 direct CSV `[0.334504,0.705945]`；slope point estimate 与方向相同。表 D 采用 Figure 4 直接读取的 `r2_snow_gradients_summary.csv`。

### 12.2 不能混入 3.3 的旧版或其他章节文件

- `r2_canonical_15D_displacement_summary.csv` 的 `D_rms` 不等于 ensemble `D_between`，不是数值冲突，而是 estimand 不同。
- Figure 5 及 final-assets S3/S4 的 R3 文件包括 F_close、状态、truth-relative error、seasonal arrays 等；这些均按用户边界排除，不可拿来填充 3.3。
- Supplement S3/S4 存在两个 numbering/content trees。该 assembly 冲突需要 manuscript 版本管理者另行确认，但不阻塞本节参数补偿数据。

### 12.3 本轮仍需在写作时确认的事项

1. 将表 D 中 Base–TGD 的 15 参数 audit-derived CI 若作为正式附录表，建议把本报告的 seed schedule 写入附录或另存一个不覆盖 canonical 的 audit CSV；不建议悄悄替换既有 R2 文件。
2. Figure 4 现有脚本的四个重点参数选择可以作为 manuscript 证据，但博士正文应明确 `i_m` 为 boundary-sensitive secondary。
3. 正文若引用 bootstrap CI，应统一采用本报告表 A–C 的 direct primary 口径，不同时引用 closure narrative 的另一套小数。
4. 本报告不对 `p` 值、parameter×attribute 相关性或新模型实验作扩展判断。

## 13. 最终判断

1. **现有数据是否足够撰写 3.3？** 足够。531 流域、Base/CN 两种融雪过程结构与 TGD 通用温度条件控制、两种 estimation constraint 的 frozen canonical outputs 已覆盖结构间分离、结构内离散、S1–S5 梯度、paired deltaE、全 15 参数和 boundary QC。缺口只是对已有 basin-level/vector 输出做轻量 reaggregation，已在本报告补齐。
2. **是否存在 blocker？** 没有 3.3 科学结论 blocker。唯一需要单独管理的是 Supplement S3/S4 numbering/content conflict，它不属于本节数据。
3. **是否需要重新训练/率定？** 不需要，且本轮没有训练或率定。
4. **是否只需要轻量统计核准？** 是。需要保留 direct canonical source、明确 CI 版本、附录放全 15 参数与 boundary audit；不需要改变模型、参数范围或主指标。
5. **推荐的 3.3 最终三级标题**：
   - **3.3.1 不同融雪过程结构间的参数变化**
   - **3.3.2 参数补偿特征随积雪活动程度的变化**
   - **3.3.3 共享参数的主要变化方向**
   - **3.3.4 参数估计约束与通用温度条件控制下的参数变化**

**最终写作主线**：以 Base–CN 为主体，先说明 15 维参数空间的结构间分离超过结构内重复估计离散，再说明该分离随 `f_snow` 组织；随后用 `u_m/k_i/c_i` 的重复方向作参数层面的例证，将 `i_m` 作为边界敏感的 supporting signal；最后用 TGD 说明通用温度条件下的相对表达差异。全程使用“参数补偿/补偿性参数变化/共享参数/参数空间”，不使用参数真值误差、单参数替代融雪过程或 IC/dPL 优劣表述。
