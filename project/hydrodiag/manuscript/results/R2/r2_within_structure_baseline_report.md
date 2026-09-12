# R2 Within-Structure Baseline & Directional Re-analysis Report

## Executive Summary

This report establishes the within-structure parameter baseline (restart/seed variability) for Base and CN across 531 basins under both IC (CMA-ES 10 restarts) and dPL (3 seeds). It evaluates whether the 15D Base–CN parameter distance exceeds同一结构内部由 restart/seed 随机性产生的散布, and assesses snow-gradient robustness.

### Key Findings:
1. **IC Parameter Dispersion**: IC restart variability within structure is large (median `within_pooled` = 0.4754 [0.4695, 0.4790]), making raw between-structure distance (`between_all` = 0.4913 [0.4864, 0.4958]) barely larger than within baseline (median `excess` = 0.0075 [0.0052, 0.0107]).
2. **dPL Parameter Separation**: dPL seed variability within structure is tight (median `within_pooled` = 0.1056 [0.1026, 0.1118]), and between-structure separation (`between_all` = 0.1817 [0.1680, 0.1984]) substantially exceeds within baseline (median `excess` = 0.0611 [0.0504, 0.0804], ratio = 1.5326 [1.4630, 1.6517]).
3. **Snow-Gradient Excess**: In both IC and dPL, parameter excess distance (`between_all - within_pooled`) increases strongly and monotonically with basin snow fraction (`frac_snow`), maintaining positive slopes whose 95% CIs strictly avoid 0 in both full 531 and exclude-S5 sensitivity.

---

## 1. 口径核对与参数聚合规则

### 1.1 旧 IC cross-start distance 1.75 / 1.85 核对结果
- **状态**: **`NOT_VERIFIED`**
- **Base/CN 统计项**: `NOT_VERIFIED`
- **参数数量**: `NOT_VERIFIED`
- **是否 0–1 归一化**: `NOT_VERIFIED`
- **距离度量类型**: `NOT_VERIFIED`
- **使用 Restarts**: `NOT_VERIFIED`
- **汇总规则**: `NOT_VERIFIED`
- **与当前 R2 D 同口径**: `NOT_VERIFIED`
- **审计记录**: 在全仓库代码、提交历史、文档及生成产物中进行了系统搜索，未发现定义或生成 `1.75 / 1.85` 参数距离的代码或结果文件。该数值标记为 `NOT_VERIFIED`。

### 1.2 dPL 3-seed 形成 Canonical Basin-level 参数规则
- **规则**: `r2_parameter_values_canonical.csv` 中 dPL 的 basin-level 参数 $z$ 是取 3 个 seed (42, 123, 2026) 在 active checkpoint 处的 **逐参数中位数 (elementwise median)**。
- **IC 对比**: IC canonical 参数取 10 次 completed restarts 中 **train-period KGE 最高** 的单次 restart (`selected_restart`)。
- **Seed Averaging 效应**: dPL 在计算 Base–CN 差异前对 3 个 seed 取中位数，有效地平滑了同结构内部的随机 seed 噪声（canonical $D_{RMS} = 0.1657$）。而 IC 的 canonical 参数是单次 best restart，保留了原始优化散布（canonical $D_{RMS} = 0.3715$）。若不建立 within-structure baseline 直接比较 raw canonical $D$, 会将 IC CMA-ES 较大的同结构优化散布误读为更大的结构间分离距离。

---

## 2. Ensemble-Level Within & Between Distance 结果

所有距离统一采用 15 个参数在 $[0,1]$ 物理边界归一化空间下的 RMS distance:
$$d_{rms}(x, y) = \sqrt{\frac{1}{15} \sum_{p=1}^{15} (x_p - y_p)^2}$$
独立统计单位严格为 531 basins。IC 为 $C(10,2)=45$ 个 within-pair 及 $10 \times 10 = 100$ 个 between-pair 的 median；dPL 为 $C(3,2)=3$ 个 within-pair 及 $3 \times 3 = 9$ 个 between-pair 的 median。

### 2.1 Full 531 及 S1–S5 Snow Regimes 汇总表 (Median [95% Bootstrap CI])

#### IC Paradigm (CMA-ES 10 Restarts)
| Stratum | N | `within_base` | `within_cn` | `within_pooled` | `between_all` | `excess` | `ratio` | `excess > 0` Prop | `between > within` Prop |
|---|---|---|---|---|---|---|---|---|---|
| Full531 | 531 | 0.4676 [0.4618, 0.4732] | 0.4821 [0.4779, 0.4872] | 0.4754 [0.4695, 0.4790] | 0.4913 [0.4864, 0.4958] | 0.0075 [0.0052, 0.0107] | 1.0161 [1.0109, 1.0223] | 63.1% [58.9%, 67.2%] | 63.1% [58.9%, 67.2%] |
| S1 | 8 | 0.4759 [0.4677, 0.4860] | 0.4697 [0.4629, 0.4777] | 0.4729 [0.4658, 0.4825] | 0.4705 [0.4622, 0.4818] | -0.0018 [-0.0041, 0.0021] | 0.9960 [0.9910, 1.0043] | 46.7% [38.8%, 53.9%] | 46.7% [38.8%, 53.9%] |
| S2 | 8 | 0.4783 [0.4669, 0.4951] | 0.4887 [0.4793, 0.4978] | 0.4846 [0.4772, 0.4950] | 0.4875 [0.4770, 0.4945] | 0.0022 [-0.0037, 0.0042] | 1.0043 [0.9922, 1.0096] | 52.6% [44.9%, 60.3%] | 52.6% [44.9%, 60.3%] |
| S3 | 8 | 0.4551 [0.4511, 0.4680] | 0.4946 [0.4811, 0.5050] | 0.4755 [0.4646, 0.4830] | 0.4943 [0.4853, 0.5028] | 0.0123 [0.0090, 0.0187] | 1.0243 [1.0188, 1.0377] | 74.4% [66.1%, 81.8%] | 74.4% [66.1%, 81.8%] |
| S4 | 8 | 0.4450 [0.4016, 0.4623] | 0.4754 [0.4653, 0.4863] | 0.4617 [0.4264, 0.4712] | 0.5087 [0.4899, 0.5332] | 0.0430 [0.0271, 0.0643] | 1.1001 [1.0616, 1.1521] | 94.1% [85.3%, 100.0%] | 94.1% [85.3%, 100.0%] |
| S5 | 8 | 0.4547 [0.4292, 0.4797] | 0.4885 [0.4743, 0.5086] | 0.4520 [0.4369, 0.4783] | 0.5605 [0.5438, 0.5755] | 0.0829 [0.0702, 0.1104] | 1.1856 [1.1477, 1.2341] | 98.2% [94.5%, 100.0%] | 98.2% [94.5%, 100.0%] |
| ExcludeS5 | 478 | 0.4679 [0.4623, 0.4758] | 0.4812 [0.4758, 0.4868] | 0.4760 [0.4703, 0.4814] | 0.4850 [0.4806, 0.4910] | 0.0050 [0.0028, 0.0071] | 1.0105 [1.0058, 1.0149] | 59.0% [54.6%, 63.4%] | 59.0% [54.6%, 63.4%] |

#### dPL Paradigm (Neural Net 3 Seeds)
| Stratum | N | `within_base` | `within_cn` | `within_pooled` | `between_all` | `excess` | `ratio` | `excess > 0` Prop | `between > within` Prop |
|---|---|---|---|---|---|---|---|---|---|
| Full531 | 531 | 0.0999 [0.0933, 0.1041] | 0.1007 [0.0969, 0.1070] | 0.1056 [0.1026, 0.1118] | 0.1817 [0.1680, 0.1984] | 0.0611 [0.0504, 0.0804] | 1.5326 [1.4630, 1.6517] | 83.8% [80.6%, 86.8%] | 83.8% [80.6%, 86.8%] |
| S1 | 8 | 0.1070 [0.1022, 0.1185] | 0.0991 [0.0919, 0.1087] | 0.1088 [0.0987, 0.1185] | 0.1232 [0.1157, 0.1321] | 0.0122 [0.0051, 0.0176] | 1.1252 [1.0500, 1.1861] | 67.9% [60.6%, 75.2%] | 67.9% [60.6%, 75.2%] |
| S2 | 8 | 0.0954 [0.0871, 0.1095] | 0.0899 [0.0820, 0.1006] | 0.1025 [0.0940, 0.1065] | 0.1376 [0.1325, 0.1541] | 0.0404 [0.0273, 0.0500] | 1.3843 [1.2959, 1.5130] | 82.1% [75.6%, 87.8%] | 82.1% [75.6%, 87.8%] |
| S3 | 8 | 0.0842 [0.0797, 0.0911] | 0.0969 [0.0879, 0.1097] | 0.0996 [0.0882, 0.1061] | 0.2520 [0.2287, 0.2707] | 0.1444 [0.1264, 0.1615] | 2.5268 [2.1978, 2.8880] | 96.7% [93.4%, 99.2%] | 96.7% [93.4%, 99.2%] |
| S4 | 8 | 0.1239 [0.0875, 0.1843] | 0.1258 [0.1003, 0.1590] | 0.1275 [0.1121, 0.1971] | 0.3335 [0.3075, 0.4072] | 0.1767 [0.1230, 0.2467] | 2.2569 [1.9578, 2.7891] | 97.1% [91.2%, 100.0%] | 97.1% [91.2%, 100.0%] |
| S5 | 8 | 0.0961 [0.0741, 0.1139] | 0.1395 [0.1125, 0.1582] | 0.1197 [0.1056, 0.1370] | 0.4411 [0.4174, 0.4771] | 0.3130 [0.3013, 0.3445] | 3.7611 [3.3370, 4.0230] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |
| ExcludeS5 | 478 | 0.0999 [0.0936, 0.1041] | 0.0980 [0.0937, 0.1022] | 0.1046 [0.1003, 0.1099] | 0.1667 [0.1558, 0.1774] | 0.0491 [0.0410, 0.0592] | 1.4412 [1.3766, 1.5180] | 81.9% [78.4%, 85.3%] | 81.9% [78.4%, 85.3%] |

### 2.2 Sensitivity: Matched Index / Seed Distance
- **IC Matched-Restart Distance (Base_r vs CN_r, r=0..9)**: Full 531 median = `0.4754` (matched pair distance median = `0.4875`), 与 `between_all` (0.4913) 基本一致。
- **dPL Matched-Seed Distance (Base_s vs CN_s, s in 42,123,2026)**: Full 531 median = `0.1822`, 与 `between_all` (0.1817) 基本一致。

---

## 3. Restart Quality Audit (IC CMA-ES)

评估 IC 10 个 restart 的训练期 KGE 质量是否系统性影响 `within_pooled` 宽度：
- **Base Train KGE Spread**: Median Best-minus-Median KGE Gap = `0.0136`, KGE IQR = `0.0186`。
- **CN Train KGE Spread**: Median Best-minus-Median KGE Gap = `0.0127`, KGE IQR = `0.0177`。
- **Snow Association**: KGE Gap 与 `frac_snow` 的 Spearman rho 为 `-0.0696` (Base) 和 `-0.2135` (CN)；KGE IQR 与 `frac_snow` 为 `-0.1144` (Base) 和 `-0.2403` (CN)。训练 KGE 散布未随积雪比例系统性扩大。
- **Parameter Dispersion vs KGE Spread**: `within_base` RMS 距离与 Base KGE IQR 的 Spearman rho 为 `0.3520`；`within_cn` 与 CN KGE IQR 为 `0.2821`。
- **Top 3 / Top 5 Restart Sensitivity**: 取 Train KGE 前 5 名 restart 时，IC `within_pooled` 降至 `0.4058` (`between_all` 降至 `0.4366`，`excess` = `0.0207`)；取前 3 名 restart 时，`within_pooled` 降至 `0.3748` (`between_all` 降至 `0.4136`，`excess` = `0.0203`)。
- **结论**: IC 较大的同结构参数散布源自 15 维自由参数空间的内在多解性（equifinality），而非少数极劣 restart 的异常拉宽。

---

## 4. Snow Gradient & Exclude-S5 Robustness

### 4.1 Regression Models (`y ~ frac_snow`)

| Paradigm | Stratum | Dependent Variable | OLS Slope [95% CI] | Spearman Rho [95% CI] |
|---|---|---|---|---|
| IC | Full531 | `within_pooled` | -0.0233 [-0.0461, -0.0013] | -0.0935 [-0.1795, -0.0079] |
| IC | Full531 | `between_all` | +0.1309 [+0.1101, +0.1515] | +0.3739 [+0.2932, +0.4511] |
| IC | Full531 | `excess` | +0.1542 [+0.1337, +0.1754] | +0.5485 [+0.4788, +0.6124] |
| IC | ExcludeS5 | `within_pooled` | -0.0340 [-0.0718, +0.0047] | -0.0689 [-0.1590, +0.0204] |
| IC | ExcludeS5 | `between_all` | +0.1047 [+0.0631, +0.1438] | +0.2150 [+0.1243, +0.3017] |
| IC | ExcludeS5 | `excess` | +0.1387 [+0.1101, +0.1686] | +0.4057 [+0.3252, +0.4818] |
| dPL | Full531 | `within_pooled` | +0.0330 [+0.0136, +0.0530] | +0.0634 [-0.0211, +0.1474] |
| dPL | Full531 | `between_all` | +0.5051 [+0.4787, +0.5342] | +0.7426 [+0.6933, +0.7844] |
| dPL | Full531 | `excess` | +0.4721 [+0.4405, +0.5055] | +0.7570 [+0.7087, +0.7988] |
| dPL | ExcludeS5 | `within_pooled` | +0.0592 [+0.0102, +0.1079] | +0.0003 [-0.0922, +0.0908] |
| dPL | ExcludeS5 | `between_all` | +0.6513 [+0.5839, +0.7184] | +0.6538 [+0.5929, +0.7082] |
| dPL | ExcludeS5 | `excess` | +0.5922 [+0.5129, +0.6750] | +0.6760 [+0.6144, +0.7301] |

---

## 5. Low-Cost Boundary Component Audit

评估每 basin $D_{RMS}^2$ 中由“至少一侧参数位于物理边界阈值 $\epsilon$ 内”的参数项贡献占比：

| Paradigm | Threshold $\epsilon$ | Full531 | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|---|
| IC | 0.01 | 93.1% | 95.0% | 93.1% | 90.2% | 89.8% | 94.7% |
| IC | 0.02 | 93.2% | 95.1% | 93.2% | 90.3% | 90.2% | 94.8% |
| IC | 0.05 | 94.2% | 95.2% | 93.6% | 92.0% | 91.1% | 95.8% |
| dPL | 0.01 | 0.4% | 0.1% | 0.1% | 1.1% | 26.1% | 46.4% |
| dPL | 0.02 | 2.1% | 0.4% | 0.5% | 15.3% | 43.8% | 58.2% |
| dPL | 0.05 | 18.8% | 4.8% | 7.5% | 34.4% | 63.8% | 78.8% |

**审计发现**: IC 中 >90% 的 $D_{RMS}^2$ 由贴边参数（如 `im`, `theta`, `a`, `ex`）贡献；而 dPL 在无雪/低雪流域贴边贡献极低（S1 仅 0.1%），随积雪增加平滑上升，体现了神经网络参数化良好的物理连续性。

---

## 6. Core Statistical Classifications

### 6.1 `STRUCTURE_SEPARATION_ABOVE_WITHIN_BASELINE`
- **判定分类**: **`SUPPORTED_BUT_QUALIFIED`**
- **依据**: dPL 的结构间分离显著高于同结构基线（Full 531 median excess = `+0.0611` [0.0504, 0.0804], 83.8% basins between > within）；而 IC 范式下由于 CMA-ES 优化多解性导致同结构基线极大 (`within_pooled` = 0.4754)，结构间超额分离较弱 (Full 531 median excess = `+0.0075` [0.0052, 0.0107], 63.1% basins between > within)。

### 6.2 `SNOW_DEPENDENT_EXCESS_SEPARATION`
- **判定分类**: **`STRONG`**
- **依据**: IC 和 dPL 的 `excess ~ frac_snow` 线性回归斜率均为正值（IC slope = `+0.1542` [0.1337, 0.1754], Spearman rho = `0.5485`；dPL slope = `+0.4721` [0.4405, 0.5055], Spearman rho = `0.7570`），且 95% Bootstrap CI 均严格不跨 0。在排除 S5 (N=478) 敏感性测试中，正向梯度与统计显著性完全保持（IC slope = `+0.1387`, dPL slope = `+0.5922`）。

---

## 7. Recommendations for Figure 3 Panel d

1. **禁止直接使用 Raw Unsigned 15D Distance (`canonical_best_d_rms`) 作为 Main Text 主结论承重**：IC 的原始 15D 距离 (0.3715) 主要受 CMA-ES 的高同结构散布 (0.4754) 支配，若不对比 within baseline，会造成结构间分离强度的误导。
2. **建议方案**: 优先采用 **`USE_BETWEEN_VS_WITHIN`**（在 Panel d 中并行展示 Between Distance 与 Within Baseline）或 **`USE_EXCESS_RMS`**（使用 $between\_all - within\_pooled$ 超额距离）；若版面受限，采用 **`DROP_UNSIGNED_DISTANCE_FROM_MAIN`** 将 Raw 15D Unsigned 距离移至 Supplementary，正文由 Signed Global Shifts + Signed Snow Gradients + `um/ki/ci` 跨范式方向一致性承重。
3. **严格止损点**: 不再启动任何新训练或重做实验，R2 结论已完整确立。

---

## 8. Final Numerical Conclusion (Strict Single Sentence)

> While dPL displays parameter separation substantially above its within-structure seed baseline (median excess = 0.0611, 83.8% basins positive), IC parameter separation is qualified by large within-structure optimization dispersion (median within_pooled = 0.4754, median excess = 0.0075), yet both paradigms demonstrate robust, snow-dependent excess parameter separation that increases significantly with basin snow fraction across the 531 basins.