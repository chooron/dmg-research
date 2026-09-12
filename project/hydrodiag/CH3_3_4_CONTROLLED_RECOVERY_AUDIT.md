# 3.4 合成流域结构缺口恢复数据核准报告

> **用途**：博士论文第三章 3.4“合成流域中的结构缺口恢复与定量归因”的数据核准与写作准备。
>
> **范围**：CN-generated synthetic system，531 个 CAMELS-US 流域，Base-no-refit、Base-refit、TGD-refit、CN-refit，evaluation-period outlet KGE、reference outlet gap、raw recovery、normalized recovery、denominator sensitivity、generating-field / TGD-shape / regional robustness，以及已知生成参数条件下的 15 个 shared XAJ parameter error。
>
> **严格边界**：本报告不重新生成 truth、不重新训练、不重新率定、不增加 TGD variant，不分析内部状态/通量、季节轨迹或 recovery–internal association。TGD 在本报告中是 **通用温度条件控制**，不是显式积融雪过程结构。

## 1. Canonical provenance

### 1.1 权威性规则

本报告采用以下优先级：

1. `manuscript/results/R3/` 的 canonical seed-median / summary 输出；
2. `manuscript/results/discussion_audit/` 的 Table 2 直接来源；
3. `results/r3_misspec_analysis_v1/` 的 basin-level posthoc 输出；
4. `results/reviewer2_robustness/` 中已冻结的 sensitivity 输出；
5. manuscript / Supplement narrative 仅作交叉核验，不覆盖 machine-readable 数值。

R3 代码、协议和结果说明表明：R3 的 canonical controlled experiment 已冻结；`posthoc_basin_table.csv`、`figure5_basin_seedmedian.csv`、`figure6_basin_seedmedian.csv` 和 discussion audit CSV 是本节的直接数据基础。旧版 `R3_RECOVERY_AUDIT.md` 记录过部分原始 gate artifact 的丢失风险，但不影响当前已保存、可重现核对的 manuscript-level 输出；该风险在第 14 节保留。

### 表 A0：Canonical provenance

| Data role | File path | Rows / objects | Main variables | Bootstrap | Canonical status | Notes |
|---|---|---:|---|---|---|---|
| Synthetic generating parameters | `results/r3_synthetic_truth_v1/theta_star.npz` | 531×17 | 15 shared XAJ + 2 CN-specific truth parameters | N/A | FROZEN_CANONICAL | 3.4 只使用其中 15 个 shared XAJ；不把 CN-specific parameters 当作 shared truth error |
| Synthetic outlet target | `results/r3_synthetic_truth_v1/q_star.npz` | 531×12,418 | `Q*` | N/A | FROZEN_CANONICAL | CN-generated evaluation target |
| Generating-field manifest | `results/r3_synthetic_truth_v1/gstar_manifest.json` | 1 | PCA/SVD–Ridge definition、bounds、seed | N/A | FROZEN_CANONICAL | `k=15`、ridge alpha=`316.227766`、CV split seed=`20260730` |
| Generating-field diagnostics | `results/r3_synthetic_truth_v1/gstar_diagnostics.json` | 1 | explained variance、5-fold CV、boundary audit | N/A | FROZEN_CANONICAL | 15 components explained variance=`0.9509457473` |
| Basin-level KGE source | `results/r3_misspec_analysis_v1/posthoc_basin_table.csv` | 4,248 | per basin×paradigm×seed×period KGE / gains | N/A | FROZEN_CANONICAL | IC best restart；dPL 3 seeds |
| Manuscript seed-median KGE table | `manuscript/results/R3/figure5_basin_seedmedian.csv` | 2,124 | KGE、`D`、`G_base`、`G_TGD`、`F_close`、`F_TGD` | 2,000 in summary protocol | FROZEN_CANONICAL | IC passthrough；dPL per-basin median across seeds |
| Recovery summary | `manuscript/results/discussion_audit/r3_gap_recovery_ratio_audit.csv` | 28 | Full / S1–S5 / S4+S5 recovery statistics | 2,000，seed `20260730` | FROZEN_CANONICAL | Table 2 的直接 machine-readable source |
| Denominator sensitivity | `manuscript/results/discussion_audit/r3_denominator_sensitivity_audit.csv`；`manuscript/stats/tables/TableS2_PanelB_denominator_sensitivity.csv` | 14 rows | threshold、Nvalid、fractions、`DeltaF`、positive fraction | frozen summary | FROZEN_SENSITIVITY | `1e-6`–`0.10` |
| Parameter truth error | `results/r3_misspec_analysis_v1/paired_parameters.csv`；`manuscript/results/R3/figure6_basin_seedmedian.csv` | 63,720；1,062 | `e`、`e_cn`、`delta_e`、`delta_abs_e`、`E_param` | 2,000，seed `20260730` | FROZEN_CANONICAL | 15 shared XAJ only；dPL seed-median before basin summary |
| Alternative field | `results/reviewer2_robustness/alt_generating_field/alt_generating_field_basin_seedmedian.csv`；`alt_generating_field_summary.json` | 2,124；4 regimes/periods | alternative field KGE / gains / fractions | frozen summary | FROZEN_SENSITIVITY | unsmoothed direct basin-wise calibrated CN–IC field |
| TGD shape sensitivity | `results/reviewer2_robustness/tgd_shape_sensitivity/tgd_shape_sensitivity_basin_metrics.csv`；`tgd_shape_sensitivity_summary.json` | 2,124；4 shapes | `T_ref`、`s_T`、KGE、gain、fraction、`DeltaF` | dPL can be reaggregated；IC raw artifact invalid | FROZEN_SENSITIVITY with IC caveat | IC positional-indexing mismatch documented in shape audit |
| Regional omission | `results/reviewer2_robustness/regional_loro/r3_huc2_loro.csv` | 16 rows | 7 omissions×2 paradigms + full reference | frozen summary | FROZEN_SENSITIVITY | direct CSV has 7 retained broad-region omissions |

Key SHA-256 identifiers for this audit:

- `figure5_basin_seedmedian.csv`: `87a9c4ee52749fff408d912ebda27fa21317028855d6de233dd365349d550960`
- `figure5_summary.json`: `21d50a4bfed4e35b84823cb55270dc62dd204144b9233814fae9477a9a7580de`
- `figure6_basin_seedmedian.csv`: `610fe88f10e67fc267aeaa95855ca721d561d325817f52efae5377a024aed2d6`
- `figure6_summary.json`: `941077c901af3af6a1377f6a7fa3b24c0042815a11e887f412789a0dace61739`
- `r3_gap_recovery_ratio_audit.csv`: `77dae8cb67f4cb774fd9071793e29e7462dbc7e7275902aaf2f92ff7eed76970`
- `r3_denominator_sensitivity_audit.csv`: `33abda1734340b98468bbac0fefaca573859029f7c5f0a61ac2885b330394375`
- `paired_parameters.csv`: `4a8c6f21774149ede5d0520c3bb3686a7629f95b49b17e8a92bbe1ec59af4f99`

### 1.2 固定实验设计核准

| 项目 | Frozen definition |
|---|---|
| Generating system | XAJ + CN，`generating_structure = XAJ_CN` |
| Generating field | 531 流域 CN–IC reference field，经 15 维 SVD/PCA 与 Ridge 属性映射 |
| Low-rank rule | `k=15`，target explained variance=`0.95`，实际=`0.9509457473` |
| Ridge | alpha=`316.22776601683796` |
| Cross-validation | 5 folds，CV seed=`20260730`；`cv_r2_total=0.9391314435` |
| Attribute input | CAMELS attributes `[:, :35]`，`frac_snow` 为固定诊断轴，不作因果变量 |
| Warm-up / train / evaluation | warm-up 1980-10-01–1981-09-30；train 1981-10-01–1995-09-30；evaluation/test 1995-10-01–2010-09-30 |
| IC | 逐流域 CMA-ES，10 random starts，取 best train-KGE；原始 gate 默认 300 generations |
| dPL | 3 个 canonical seeds `42, 123, 2026`；结果先按 basin 取 seed median |
| KGE | repository standard KGE `(r, alpha, beta)`，不是另行引入的 modified KGE′ |
| Bootstrap | basin 为单元，主要 summary 为 2,000 draws，seed=`20260730` |

## 2. 合成流域结构缺失基准

`Base-no-refit` 使用 generating 15-dimensional XAJ host parameters，删除 CN 后直接 forward；`CN-refit` 使用正确的 XAJ+CN 结构并重新估计，是已知结构条件下的 refit reference；`Base-refit` 和 `TGD-refit` 均为 evaluation-period outlet 表现。

表 A1 的 median、Q25、Q75 来自 `figure5_summary.json:panel_b_ladder`；该 frozen summary 未保存 KGE median CI，因此 CI 是基于相同的 531-basin seed-median vectors、同一 basin bootstrap 算法、2,000 draws、seed=`20260730` 的轻量 audit reaggregation，不是新模型运行。

### 表 A1：Evaluation-period outlet KGE

| Estimation | Configuration | N | Median KGE | Q25 | Q75 | 95% CI of median | Source |
|---|---|---:|---:|---:|---:|---|---|
| IC | Base-no-refit | 531 | 0.898015 | 0.639213 | 0.976972 | [0.877196, 0.922679] | `figure5_summary.json` + frozen basin table audit bootstrap |
| IC | Base-refit | 531 | 0.898845 | 0.669948 | 0.965912 | [0.883149, 0.916257] | same |
| IC | TGD-refit | 531 | 0.933705 | 0.851541 | 0.974557 | [0.916609, 0.944392] | same |
| IC | CN-refit | 531 | 0.992570 | 0.981149 | 0.996355 | [0.991005, 0.993531] | same |
| dPL | Base-no-refit | 531 | 0.898015 | 0.639213 | 0.976972 | [0.877196, 0.922679] | `figure5_summary.json` + frozen basin table audit bootstrap |
| dPL | Base-refit | 531 | 0.908099 | 0.684664 | 0.974549 | [0.888219, 0.930410] | same |
| dPL | TGD-refit | 531 | 0.944020 | 0.842205 | 0.985367 | [0.928680, 0.954389] | same |
| dPL | CN-refit | 531 | 0.995495 | 0.990614 | 0.997759 | [0.994826, 0.995859] | same |

### 2.1 结果判断

- `CN-refit` 的 evaluation median KGE 为 IC `0.9926`、dPL `0.9955`，接近 synthetic generating hydrograph 的正确结构 benchmark，但不应写成逐流域严格等于 truth 或 residual 为 0。
- `Base-no-refit` 的 evaluation median KGE 为 `0.8980`，相对于 CN-refit 形成清楚的结构缺失基准。
- Base-refit 的 KGE 仅由 IC `0.8980` 提升至 `0.8988`、dPL 提升至 `0.9081`；重新估计 shared host parameters 不能恢复大部分出口缺口。
- TGD-refit 达到 IC `0.9337`、dPL `0.9440`，位于 Base-refit 与 CN-refit 之间。
- train-period 的 KGE ladder 整体高于 evaluation-period。已有 per-basin train-to-test decay summary：IC `G_Base` median decay=`0.013688`、`G_TGD`=`0.002650`；dPL 分别为 `0.002143`、`0.001058`。该结果适合放附录，正文不必展开。

## 3. Reference outlet gap 与 raw recovery gains

### 3.1 定义核准

当前 manuscript/Table 2 使用的 primary common-reference estimands 为：

\[
D_b = KGE_{CN-refit,b}-KGE_{Base-no-refit,b},
\]

\[
G_{Base,b}=KGE_{Base-refit,b}-KGE_{Base-no-refit,b},
\]

\[
G_{TGD,b}=KGE_{TGD-refit,b}-KGE_{Base-no-refit,b}.
\]

`D_b`、`G_Base`、`G_TGD` 的 population median 和 Q25/Q75 使用全部 531 个流域；只有 ratio 及 `DeltaF` 使用 denominator-valid subset。

### 3.2 Table 2 / canonical 核准

### 表 B：Reference gap 与 raw recovery

| Estimation | Quantity | N | Median | 95% CI | Q25–Q75 | Positive fraction（若适用） | Source |
|---|---|---:|---:|---|---|---:|---|
| IC | `D_b` | 531 | +0.086733 | [+0.060769, +0.105721] | [0.009006, 0.356202] | 80.4% `D_b>1e-6` | `r3_gap_recovery_ratio_audit.csv` + `figure5_summary.json` |
| IC | `G_Base` | 531 | +0.002609 | [+0.000833, +0.007729] | [−0.006325, +0.052362] | 56.3% | same |
| IC | `G_TGD` | 531 | +0.038562 | [+0.025385, +0.049824] | [+0.001275, +0.195182] | 75.9% | same |
| dPL | `D_b` | 531 | +0.091071 | [+0.069773, +0.113706] | [0.018093, 0.355436] | 86.6% `D_b>1e-6` | same；dPL seed median |
| dPL | `G_Base` | 531 | +0.007325 | [+0.003854, +0.009876] | [−0.002489, +0.050300] | 65.5% | same |
| dPL | `G_TGD` | 531 | +0.035972 | [+0.026788, +0.044640] | [+0.004490, +0.179991] | 81.4% | same |

### 3.3 结果判断

- manuscript 旧值已核准：IC `D_b≈+0.087`、`G_Base≈+0.0026`、`G_TGD≈+0.0386`；dPL 分别为 `+0.091`、`+0.0073`、`+0.0360`。
- `G_Base` 相对于 `D_b` 较小，说明仅重新估计 Base 的 15 个 host parameters 只能恢复较小的 outlet KGE gain。
- `G_TGD` 的 raw gain 在 IC 为 `+0.0386`、dPL 为 `+0.0360`。raw gain 是本节主结果，不能只报告 normalized fraction。
- 不将 `G_TGD` 写成 TGD 恢复了 CN 的物理过程；它是相对于 Base-no-refit knockout 的 outlet KGE gain。

### 3.4 F_TGD estimand 的命名冲突

R3 工作区同时保留两个 TGD fraction estimand：

1. **本节 / Table 2 使用的 common-reference `F_TGD`（亦称 `F_TGD_star`）**：
   \[
   F_{TGD,b}=G_{TGD,b}/D_b,
   \]
   其中 `G_TGD = KGE(TGD-refit)-KGE(Base-no-refit)`。
2. `derive_estimand_audit.py` 中另有 **incremental `F_tgd2`**：`KGE(TGD)-KGE(Base-refit)` 除以 `KGE(CN)-KGE(Base-refit)`。它回答的是 TGD 在 Base-refit 之后的增量，不是本节 common-reference recovery fraction。

本报告严格使用第 1 种，避免把两个 denominator 或两个 recovery question 混为一谈。

## 4. Normalized recovery fractions

### 4.1 canonical denominator 与汇总顺序

代码和 Table S2 Panel B 明确：

- canonical condition：`D_b > 1e-6`；
- `Nvalid`：IC `427/531=80.4%`，dPL `460/531=86.6%`；
- ratios 是逐 basin 计算后再取 population median；
- `DeltaF_b = F_TGD,b-F_close,b` 也是逐 basin 计算后再汇总；
- ratios 不 clipping；因此允许 `<0` 或 `>1` 的尾部；
- bootstrap 单元为 basin，canonical summary 为 2,000 draws、seed=`20260730`。

### 表 C：Normalized recovery fractions

| Estimation | Nvalid | `F_close` median | 95% CI | `F_TGD` median | 95% CI | `DeltaF` median | 95% CI | `P(F_TGD>F_close)` |
|---|---:|---:|---|---:|---|---:|---|---:|
| IC | 427 | 0.101494 | [0.079791, 0.112551] | 0.545645 | [0.519301, 0.572078] | +0.459951 | [+0.432797, +0.498494] | 91.6% |
| dPL | 460 | 0.102051 | [0.094877, 0.110619] | 0.521162 | [0.504806, 0.537804] | +0.440998 | [+0.412270, +0.467038] | 92.8% |

### 4.2 未截断尾部

| Estimation | `F_close<0` | `0≤F_close≤1` | `F_close>1` | `F_TGD<0` | `0≤F_TGD≤1` | `F_TGD>1` |
|---|---:|---:|---:|---:|---:|---:|
| IC | 31.6% | 67.9% | 0.5% | 8.4% | 87.6% | 4.0% |
| dPL | 24.8% | 75.0% | 0.2% | 6.1% | 93.0% | 0.9% |

### 4.3 结果判断

- Base 重新估计的 fraction 仅约 `0.10`；TGD common-reference fraction 为 IC `0.546`、dPL `0.521`。
- 在当前 CN-generated synthetic reference 和 canonical TGD formulation 下，可以写“恢复约 52%–55% 的 reference evaluation-period outlet KGE gap”。不能写成“解释了 52%–55% 的融雪物理”，也不能把剩余部分等同于 CN 独有物理贡献。
- `F_TGD>F_close` 的 paired positive fraction 为 IC `91.6%`、dPL `92.8%`；这描述结构对照排序，不代表 TGD-specific parameters 有 generating truth。

## 5. Denominator-valid 样本组成

### 表 D：Denominator-valid / invalid 分层

| Estimation | Snow class | Total N | Valid N | Invalid N | Valid rate |
|---|---|---:|---:|---:|---:|
| IC | S1 | 165 | 70 | 95 | 42.4% |
| IC | S2 | 156 | 148 | 8 | 94.9% |
| IC | S3 | 121 | 120 | 1 | 99.2% |
| IC | S4 | 34 | 34 | 0 | 100.0% |
| IC | S5 | 55 | 55 | 0 | 100.0% |
| dPL | S1 | 165 | 100 | 65 | 60.6% |
| dPL | S2 | 156 | 151 | 5 | 96.8% |
| dPL | S3 | 121 | 120 | 1 | 99.2% |
| dPL | S4 | 34 | 34 | 0 | 100.0% |
| dPL | S5 | 55 | 55 | 0 | 100.0% |

### 5.1 结果判断

- IC invalid 总数为 104，其中 95 个来自 S1，即 `91.3%`；dPL invalid 总数为 71，其中 65 个来自 S1，即 `91.5%`。manuscript “约 91% 来自 S1”有 canonical 支持。
- S4/S5 全部 denominator-valid。
- 合法解释是：低积雪条件下 CN 结构产生的 reference outlet gap 较小，normalized ratio 更容易缺少稳定分母。denominator-invalid 不是模型失败，也不应被当作 low-KGE catchment。

## 6. Recovery 随 snow activity 的变化

`r3_gap_recovery_ratio_audit.csv` 已包含 Full、S1–S5、S4+S5；本节使用 evaluation-period，`D_b/G` 为该 strata 的全部流域 median，fraction 为该 strata denominator-valid subset median。

### 表 E：S1–S5 recovery summary

| Estimation | Snow class | N / Nvalid | `D_b` | `G_Base` | `G_TGD` | `F_close` | `F_TGD` | `DeltaF` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IC | S1 | 165 / 70 | −0.003942 | −0.005442 | −0.003168 | −0.096104 | +0.508104 | +0.515189 |
| IC | S2 | 156 / 148 | +0.060699 | −0.000264 | +0.028555 | +0.002248 | +0.498352 | +0.438232 |
| IC | S3 | 121 / 120 | +0.332512 | +0.045434 | +0.182341 | +0.136096 | +0.549094 | +0.424432 |
| IC | S4 | 34 / 34 | +0.602757 | +0.075472 | +0.284141 | +0.163088 | +0.509354 | +0.385782 |
| IC | S5 | 55 / 55 | +1.064673 | +0.178368 | +0.834055 | +0.207517 | +0.830807 | +0.664918 |
| dPL | S1 | 165 / 100 | +0.003709 | −0.003236 | −0.000161 | −0.101474 | +0.443535 | +0.575726 |
| dPL | S2 | 156 / 151 | +0.071545 | +0.003378 | +0.033188 | +0.062974 | +0.498093 | +0.392438 |
| dPL | S3 | 121 / 120 | +0.336344 | +0.039962 | +0.175761 | +0.117876 | +0.531437 | +0.409816 |
| dPL | S4 | 34 / 34 | +0.594143 | +0.084710 | +0.256592 | +0.174334 | +0.485460 | +0.310997 |
| dPL | S5 | 55 / 55 | +1.043954 | +0.170834 | +0.825403 | +0.182578 | +0.812971 | +0.650143 |

### 6.1 结果判断与正文 placement

- `D_b` 从低雪到高雪总体增大，说明显式 CN 缺失在 snowier synthetic catchments 中造成更大的 outlet gap；S1 的 `D_b` 仍可能为负或接近零，因此不适合用 fraction 叙事。
- raw `G_Base` 与 `G_TGD` 在 S3–S5 明显增大，但该梯度主要是因为 outlet gap 同时增大，不能仅凭 raw gain 写成恢复效率增强。
- `F_TGD` 在 IC/dPL 各层大体保持约 `0.44–0.55`，S5 的高值伴随更大 gap 和小样本，不宜写成普适高雪比例。
- `DeltaF` 在所有 valid strata 均为正，但 S4 样本仅 34；S1 ratio 的分母组成尤其不稳定。
- **推荐**：正文保留 Full `D_b/G_Base/G_TGD/F`，S1–S5 完整数值放附录；正文只用一句说明 low-snow denominator limitation 和 high-snow gap increase。

## 7. Denominator-threshold sensitivity

### 表 F：Table S2 Panel B

| Estimation | Threshold | Nvalid | `F_close` | `F_TGD` | `DeltaF` | Positive paired % |
|---|---:|---:|---:|---:|---:|---:|
| IC | `1e-6` canonical | 427 | 0.101 | 0.546 | +0.460 | 91.6% |
| IC | `1e-4` | 427 | 0.101 | 0.546 | +0.460 | 91.6% |
| IC | `1e-3` | 424 | 0.102 | 0.546 | +0.460 | 92.0% |
| IC | `0.01` | 395 | 0.109 | 0.547 | +0.459 | 92.9% |
| IC | `0.02` | 366 | 0.112 | 0.549 | +0.459 | 93.4% |
| IC | `0.05` | 303 | 0.129 | 0.566 | +0.457 | 96.0% |
| IC | `0.10` | 251 | 0.142 | 0.583 | +0.458 | 96.8% |
| dPL | `1e-6` canonical | 460 | 0.102 | 0.521 | +0.441 | 92.8% |
| dPL | `1e-4` | 460 | 0.102 | 0.521 | +0.441 | 92.8% |
| dPL | `1e-3` | 454 | 0.104 | 0.524 | +0.440 | 93.2% |
| dPL | `0.01` | 421 | 0.109 | 0.529 | +0.436 | 94.8% |
| dPL | `0.02` | 389 | 0.111 | 0.536 | +0.432 | 95.9% |
| dPL | `0.05` | 322 | 0.121 | 0.542 | +0.423 | 96.3% |
| dPL | `0.10` | 257 | 0.132 | 0.578 | +0.441 | 96.1% |

### 7.1 结论

阈值提高会改变 Nvalid 和 fraction 的绝对值，但不改变主要 ordering：`F_TGD > F_close` 的 paired contrast 在所有 tested thresholds 中保持正向。正文最多保留一句；完整网格进入 Supplement。

## 8. Alternative generating-field sensitivity

alternative field 定义为：不使用 canonical PCA/Ridge smoothing，而直接使用 catchment-wise calibrated CN–IC parameter field。不能把该 field 当作新的 synthetic truth；它是已有 controlled sensitivity。

### 表 G：Canonical vs unsmoothed direct calibrated CN–IC field

| Generating field | Estimation | Nvalid | `D_b` | `G_Base` | `G_TGD` | `F_close` | `F_TGD` | `DeltaF` | Positive % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical PCA/Ridge | IC | 427 | +0.086733 | +0.002609 | +0.038562 | +0.101494 | +0.545645 | +0.459951 | 91.6% |
| Canonical PCA/Ridge | dPL | 460 | +0.091071 | +0.007325 | +0.035972 | +0.102051 | +0.521162 | +0.440998 | 92.8% |
| Alternative direct CN–IC | IC | 522 | +0.106483 | −0.009999 | +0.000545 | −0.103338 | +0.007226 | +0.195275 | 72.4% |
| Alternative direct CN–IC | dPL | 123 | +0.298312 | −0.016979 | +0.194014 | −0.051221 | +0.583982 | +0.702323 | 91.1% |

来源：canonical 为 `manuscript/results/discussion_audit/r3_gap_recovery_ratio_audit.csv`；alternative 为 `results/final_r3_ic_dpl_audit/alternative_field_r3_summary.csv`（final audit 的 seed-median reaggregation）及 `results/reviewer2_robustness/alt_generating_field/alt_generating_field_summary.json`（alternate summary）。canonical 行的 `D_b/G` 沿用表 B 的 all-531 口径；alternative summary 的 `D_b/G` 按其 source summary 的 denominator-valid 子集口径报告，因而表 G 的横向 raw `D/G` 不作严格 population-level 比较。alternative dPL 的 JSON 四舍五入/汇总值为 `F_close=-0.0531`、`DeltaF=+0.7012`，final audit CSV 为 `−0.051221/+0.702323`；本报告采用后者并保留该版本差异。该 sensitivity summary 未保存 paired bootstrap CI，表 G 不虚构 CI。

### 8.1 结论

- Base < TGD recovery ordering 在 alternative field 下仍保持：`F_TGD` 高于 `F_close`，两种 estimation 的 `DeltaF` 都为正。
- 定量比例并不恒定：canonical `DeltaF` 约 `+0.46/+0.44`，alternative 为 `+0.20/+0.70`。
- denominator-valid population 变化显著，尤其 dPL 从 `460` 降至 `123`；因此 alternative field 只能支持 qualitative ordering robustness，不能支持“52%–55% 与生成参数场无关”。
- alternative field 的正文位置建议为一句 sensitivity caveat 或脚注，完整数字进附录。

## 9. TGD response-shape sensitivity

### 9.1 设置核准

已有 shape sensitivity 固定同一 CN-generated `Q*`、531 流域、同一 warm-up/train/test、同一 canonical generating field；只改变 TGD thermal response 的 `T_ref` 和 `s_T`，其余 TGD 常数保持固定：

- sharp：`T_ref=0°C`，`s_T=1°C`；
- canonical：`T_ref=0°C`，`s_T=2°C`；
- warm-shifted：`T_ref=+2°C`，`s_T=2°C`；
- broad：`T_ref=0°C`，`s_T=4°C`；
- numerical epsilon=`1e-6 d`；
- same `Q*` / forcing / 531 basin set。

### 9.2 IC 对齐问题

`results/reviewer2_robustness/tgd_shape_sensitivity/tgd_shape_sensitivity_basin_metrics.csv` 的 IC 数值不能直接使用：shape audit 证明 sensitivity evaluator 将 forcing 按 `531sub_id.txt` 顺序加载，却将 IC parameters 按 sorted basin ID 堆叠，导致 index 22 以后发生 positional mismatch。raw file 的 IC canonical `DeltaF=-0.1339` 及 raw IC shape values 是 alignment artifact，不是科学结果。

已有 alignment audit 给出了与 canonical R3 一致的 IC aligned medians：canonical `+0.4600`、sharp `+0.3703`、warm-shifted `+0.2575`、broad `−0.0296`。但 corrected aligned basin-level file / bootstrap distribution 未被保存，因此本报告不为 corrected IC summary 虚构 95% CI。

### 表 H：TGD shape sensitivity

| Shape | `T_ref` | `s_T` | Estimation | Nvalid | `DeltaF` median | 95% CI | Positive % |
|---|---:|---:|---|---:|---:|---|---:|
| Sharp | 0°C | 1°C | IC | 427 | +0.3703 aligned audit | **N/A：aligned CI 未保存；raw IC CI 无效** | 83.61% aligned audit |
| Canonical | 0°C | 2°C | IC | 427 | +0.4600 aligned audit | **N/A：aligned CI 未保存；raw IC CI 无效** | 92.04% aligned audit |
| Warm-shifted | +2°C | 2°C | IC | 427 | +0.2575 aligned audit | **N/A：aligned CI 未保存；raw IC CI 无效** | 79.16% aligned audit |
| Broad | 0°C | 4°C | IC | 427 | −0.0296 aligned audit | **N/A：aligned CI 未保存；raw IC CI 无效** | 46.84% aligned audit |
| Sharp | 0°C | 1°C | dPL | 460 | +0.4135 | [+0.3901, +0.4269] | 87.61% |
| Canonical | 0°C | 2°C | dPL | 460 | +0.4410 | [+0.4122, +0.4670] | 93.04% |
| Warm-shifted | +2°C | 2°C | dPL | 460 | +0.2553 | [+0.2174, +0.2915] | 76.30% |
| Broad | 0°C | 4°C | dPL | 460 | −0.0914 | [−0.1492, −0.0470] | 41.52% |

dPL CI 为对已有 shape basin metrics 的轻量 paired-basin bootstrap，2,000 draws、seed=`20260730`；IC corrected summary 只保留 alignment audit 的 median / prevalence。Shape sensitivity raw summary 中 dPL canonical `G_TGD=+0.0507` 与 primary canonical raw gain `+0.0360` 也存在 summary口径差异；shape 表只用 `DeltaF` ordering，primary raw gain 以表 B 为准。

### 9.3 结论与正文边界

- `s_T=1–2°C` 下，已有 aligned audit 支持 IC/dPL 的正向 TGD mitigation ordering；但 IC aligned CI 尚缺正式保存。
- warm-shifted 仍为正但下降到约 `+0.26`；broad `s_T=4°C` 变为非正，表明“约 52%–55%”依赖 canonical response shape。
- 正文可以报告 canonical value，并明确 quantitative attribution 对 generic thermal response shape 敏感；完整 H 表和 IC alignment caveat 进入附录。
- 不能写成“TGD 普遍解释一半融雪结构收益”。

## 10. Regional omission robustness

实际 R3 direct CSV `r3_huc2_loro.csv` 保存的是 7 个 broad-region omissions：Region 01–07；对应移除数量为 `92/95/93/51/68/60/72`，remaining N 为 `439/436/438/480/463/471/459`。Supplement narrative 曾写“18 HUC-2 regions / 8 retained display regions”，与当前 direct CSV 不一致；本表以实际 7-row source 为准。

### 表 I：Seven-region leave-one-region-out

| Estimation | Omitted region | Removed N | Remaining N | Nvalid | `DeltaF` | Positive % |
|---|---|---:|---:|---:|---:|---:|
| IC | Region 01 | 92 | 439 | 335 | +0.4281 | 83.6% |
| IC | Region 02 | 95 | 436 | 373 | +0.4152 | 86.3% |
| IC | Region 03 | 93 | 438 | 335 | +0.4813 | 85.7% |
| IC | Region 04 | 51 | 480 | 384 | +0.4197 | 87.2% |
| IC | Region 05 | 68 | 463 | 399 | +0.4167 | 86.7% |
| IC | Region 06 | 60 | 471 | 381 | +0.4787 | 88.7% |
| IC | Region 07 | 72 | 459 | 355 | +0.5035 | 91.3% |
| dPL | Region 01 | 92 | 439 | 368 | +0.4240 | 91.0% |
| dPL | Region 02 | 95 | 436 | 393 | +0.4172 | 93.4% |
| dPL | Region 03 | 93 | 438 | 367 | +0.4614 | 91.6% |
| dPL | Region 04 | 51 | 480 | 413 | +0.4408 | 93.0% |
| dPL | Region 05 | 68 | 463 | 423 | +0.4360 | 92.7% |
| dPL | Region 06 | 60 | 471 | 408 | +0.4457 | 92.2% |
| dPL | Region 07 | 72 | 459 | 388 | +0.4616 | 95.9% |

### 10.1 结论

- IC omitted-region `DeltaF` 范围为 `+0.4152`–`+0.5035`；dPL 为 `+0.4172`–`+0.4616`。
- 7 个 omission 均保持 positive median 和 majority positive paired fraction；未发现单一区域改变主要 ordering。
- 文件中的 Full-row IC `DeltaF=+0.4405` 与 primary canonical Full `+0.4600` 不一致，且 regional Full raw gain 是 valid-only aggregation；因此不把 regional Full row 覆盖 primary Table C。Regional table 只用于 omitted-region sensitivity。
- 该分析是 regional omission robustness，不是 spatial correction、spatial independence inference 或 spatially corrected uncertainty。

## 11. Aggregate truth-relative shared-parameter error

### 11.1 实际 frozen 定义核准

用户任务中的 `D_theta` 需要按实际 code/schema 修正命名：

\[
E_{param,M,b}=median_{p\in 15}\left|e_{M,p,b}\right|,
\]

其中

\[
e_{M,p,b}=\frac{\hat\theta_{M,p,b}-\theta^*_{p,b}}{upper_p-lower_p}.
\]

因此 Figure 6 / `figure6_summary.json:panel_a_param_distance` 所报告的 aggregate parameter distance 是 **15 个 normalized shared-parameter absolute errors 的 basin-wise median**，不是 15 维 RMS。15 个参数在 per-parameter median 中等权，但没有平方和/平方根；如果正文写 `D_theta`，应在脚注明确它是 `E_param`，避免误称为 Euclidean/RMS distance。

- truth 为 synthetic `theta_star`；
- bounds 来自 frozen generating manifest；
- 仅包含 15 个 shared XAJ；
- CN-refit residual 不是假设的 0，而是实际重新估计后的 residual；
- TGD-specific parameters 没有对应的 generating truth，因此不纳入。

### 表 J1：Aggregate normalized shared-parameter error

| Estimation | Configuration | N | Median `D_theta` / `E_param` | Q25 | Q75 | 95% CI |
|---|---|---:|---:|---:|---:|---|
| IC | Base-refit | 531 | 0.179977 | 0.093844 | 0.288062 | [0.169834, 0.195936] |
| IC | TGD-refit | 531 | 0.162978 | 0.083348 | 0.262660 | [0.148583, 0.178570] |
| IC | CN-refit | 531 | 0.069943 | 0.046901 | 0.114511 | [0.065370, 0.076522] |
| dPL | Base-refit | 531 | 0.093904 | 0.031591 | 0.234865 | [0.071356, 0.112022] |
| dPL | TGD-refit | 531 | 0.055046 | 0.020316 | 0.143618 | [0.049419, 0.064146] |
| dPL | CN-refit | 531 | 0.025774 | 0.014367 | 0.044654 | [0.022820, 0.028948] |

### 11.2 结果判断

- truth-relative aggregate error 的 ordering 为 `CN-refit < TGD-refit < Base-refit`，在 IC/dPL 均成立。
- IC 的 Base/TGD/CN 为约 `0.180/0.163/0.070`；dPL 为约 `0.094/0.055/0.026`，与 manuscript Figure 6 旧值一致。
- 这支持“在该受控 synthetic system 中，TGD-refit 的 shared-parameter estimate 比 Base-refit 更接近 generating parameter field”。不能外推为真实流域参数真值恢复，也不能由此比较 IC/dPL 优劣。
- `D_theta` 的 aggregate error 与 3.3 的 Base–CN parameter compensation slope 是不同 estimand：3.3 是模型结构对照位移；3.4 是相对于 synthetic generating truth 的估计偏差。

## 12. Excess shared-parameter error 及 S1–S5

定义按 frozen `figure6_summary`：

\[
E_{excess,Base}=E_{param,Base}-E_{param,CN},
\quad
E_{excess,TGD}=E_{param,TGD}-E_{param,CN}.
\]

这是 **aggregate per-basin error difference**，不是 Table K 的 component `|e_M-e_CN|`，也不是 `delta_abs_e=|e_M|-|e_CN|` 的 median。

### 表 J2：Full 与 S1–S5 aggregate excess error

| Estimation | Configuration | Snow class | N | Median excess error | 95% CI | Positive % |
|---|---|---|---:|---:|---|---:|
| IC | Base-refit | Full | 531 | +0.097403 | [+0.082740, +0.113086] | 77.8% |
| IC | TGD-refit | Full | 531 | +0.076484 | [+0.062108, +0.092549] | 76.1% |
| IC | Base-refit | S1 | 165 | +0.003374 | [−0.009087, +0.021210] | 50.9% |
| IC | TGD-refit | S1 | 165 | +0.005786 | [−0.002035, +0.019520] | 54.5% |
| IC | Base-refit | S2 | 156 | +0.083328 | [+0.062326, +0.105710] | 80.8% |
| IC | TGD-refit | S2 | 156 | +0.055907 | [+0.041737, +0.068043] | 75.6% |
| IC | Base-refit | S3 | 121 | +0.170381 | [+0.148989, +0.183220] | 95.0% |
| IC | TGD-refit | S3 | 121 | +0.151316 | [+0.138674, +0.166968] | 90.9% |
| IC | Base-refit | S4 | 34 | +0.234055 | [+0.200589, +0.252481] | 97.1% |
| IC | TGD-refit | S4 | 34 | +0.203010 | [+0.142577, +0.268679] | 94.1% |
| IC | Base-refit | S5 | 55 | +0.305411 | [+0.289415, +0.328568] | 100.0% |
| IC | TGD-refit | S5 | 55 | +0.265200 | [+0.226992, +0.293476] | 98.2% |
| dPL | Base-refit | Full | 531 | +0.063635 | [+0.048280, +0.078462] | 96.4% |
| dPL | TGD-refit | Full | 531 | +0.023841 | [+0.019401, +0.032793] | 84.6% |
| dPL | Base-refit | S1 | 165 | +0.008693 | [+0.007098, +0.011611] | 89.7% |
| dPL | TGD-refit | S1 | 165 | +0.003196 | [+0.001976, +0.004251] | 70.9% |
| dPL | Base-refit | S2 | 156 | +0.044266 | [+0.035137, +0.056355] | 98.7% |
| dPL | TGD-refit | S2 | 156 | +0.017297 | [+0.011857, +0.021515] | 80.8% |
| dPL | Base-refit | S3 | 121 | +0.177062 | [+0.166725, +0.190281] | 100.0% |
| dPL | TGD-refit | S3 | 121 | +0.091264 | [+0.081267, +0.098146] | 96.7% |
| dPL | Base-refit | S4 | 34 | +0.235944 | [+0.222133, +0.249359] | 100.0% |
| dPL | TGD-refit | S4 | 34 | +0.186474 | [+0.149089, +0.296889] | 100.0% |
| dPL | Base-refit | S5 | 55 | +0.252059 | [+0.225615, +0.282183] | 100.0% |
| dPL | TGD-refit | S5 | 55 | +0.202258 | [+0.188506, +0.211948] | 100.0% |

### 12.1 结果判断

- aggregate excess error 总体随 snow activity 增大：IC Base `0.003→0.305`、TGD `0.006→0.265`；dPL Base `0.009→0.252`、TGD `0.003→0.202`。
- TGD 的 Full aggregate excess error 小于 Base：IC `0.0765<0.0974`，dPL `0.0238<0.0636`。
- 在 S2–S5，TGD 通常低于 Base；S1 的 IC 差异接近 0 且 CI 跨 0，应避免对 low-snow stratum 过度解释。
- 该结果支持“在当前受控生成系统中，TGD 减少 shared-parameter 相对生成场的偏差”，不支持“恢复真实水文参数”或“内部过程已恢复”。

## 13. 15 个共享参数 component truth error

### 13.1 Table K 的 estimand 区分

Supplement Figure S3 panel (a) 的 renderer 实际使用：

\[
C15_{M,p}=median_b\left|e_{M,p,b}-e_{CN,p,b}\right|,
\]

即按参数、按 basin 的 **CN-adjusted directional truth-error separation**；脚本先取 `abs(delta_e)`。这不是：

- aggregate `E_param_excess = E_param_M-E_param_CN`；
- primary protocol 的 `delta_abs_e=|e_M|-|e_CN|`；
- 3.3 的 Base–CN structure displacement。

为避免“excess absolute error”这一名称掩盖 estimand，表 K 同时列出 Figure S3 的 `C15=median|e_M-e_CN|` 和 protocol primary `delta_abs_e=median(|e_M|-|e_CN|)`；二者均基于 15 个 shared XAJ、normalized `[0,1]` coordinates。dPL 先在每个 basin 内取 3 seed median，再作 basin summary；IC 直接使用 best restart。

95% CI 均为 existing frozen component table 的轻量 basin bootstrap，2,000 draws、seed=`20260730`，未运行模型。

### 表 K：15 参数 component-level truth error

| Parameter | Estimation | Base `C15` median [95% CI] | Base `delta_abs_e` median [95% CI] | TGD `C15` median [95% CI] | TGD `delta_abs_e` median [95% CI] |
|---|---|---|---|---|---|
| `xaj_k` | IC | +0.031484 [+0.028461,+0.035328] | +0.015418 [+0.012398,+0.017757] | +0.019572 [+0.017186,+0.022687] | +0.006766 [+0.004911,+0.008523] |
| `xaj_b` | IC | +0.102660 [+0.086896,+0.132920] | +0.057358 [+0.036528,+0.075842] | +0.079516 [+0.067544,+0.102600] | +0.042825 [+0.031911,+0.061169] |
| `xaj_im` | IC | +0.059915 [+0.051037,+0.072518] | +0.007154 [+0.003392,+0.010931] | +0.068747 [+0.055331,+0.079975] | +0.004425 [+0.001188,+0.009901] |
| `xaj_um` | IC | +0.296713 [+0.261283,+0.334163] | +0.032255 [+0.005127,+0.066299] | +0.295096 [+0.254916,+0.331038] | +0.021662 [+0.004562,+0.051185] |
| `xaj_lm` | IC | +0.253378 [+0.218387,+0.285803] | +0.060713 [+0.035861,+0.083071] | +0.262526 [+0.228214,+0.281992] | +0.015516 [+0.001040,+0.035609] |
| `xaj_dm` | IC | +0.275521 [+0.242590,+0.307821] | +0.048690 [+0.020812,+0.092186] | +0.244486 [+0.216929,+0.272818] | +0.021976 [+0.005392,+0.048690] |
| `xaj_c` | IC | +0.292944 [+0.261995,+0.318775] | +0.057069 [+0.031088,+0.097017] | +0.289933 [+0.250526,+0.319597] | +0.005691 [0.000000,+0.030471] |
| `xaj_sm` | IC | +0.253116 [+0.220112,+0.289888] | +0.006694 [−0.002945,+0.026105] | +0.308198 [+0.250340,+0.342118] | +0.035345 [+0.011004,+0.064724] |
| `xaj_ex` | IC | +0.445400 [+0.392588,+0.505790] | +0.010474 [0.000000,+0.044576] | +0.441838 [+0.384134,+0.487786] | +0.001044 [0.000000,+0.037223] |
| `xaj_ki` | IC | +0.228987 [+0.196550,+0.252740] | +0.079849 [+0.066908,+0.096936] | +0.212930 [+0.178452,+0.250930] | +0.062760 [+0.042298,+0.083692] |
| `xaj_kg` | IC | +0.198540 [+0.170839,+0.221762] | +0.107302 [+0.087088,+0.134464] | +0.187117 [+0.166203,+0.221124] | +0.107642 [+0.085613,+0.138434] |
| `xaj_ci` | IC | +0.239542 [+0.218769,+0.263024] | +0.112361 [+0.085865,+0.142343] | +0.155033 [+0.142826,+0.176053] | +0.032996 [+0.019533,+0.046523] |
| `xaj_cg` | IC | +0.224961 [+0.205578,+0.252868] | +0.135742 [+0.106156,+0.161349] | +0.176688 [+0.148189,+0.205486] | +0.078015 [+0.060354,+0.100161] |
| `xaj_a` | IC | +0.223350 [+0.189616,+0.264419] | +0.043897 [+0.021592,+0.065327] | +0.175675 [+0.149761,+0.215445] | +0.028287 [+0.018016,+0.044526] |
| `xaj_theta` | IC | +0.091555 [+0.077575,+0.104908] | +0.008142 [+0.003123,+0.015971] | +0.100649 [+0.087370,+0.114367] | +0.013538 [+0.009043,+0.019722] |
| `xaj_k` | dPL | +0.017371 [+0.012767,+0.023737] | +0.006659 [+0.004287,+0.009521] | +0.013652 [+0.011232,+0.016853] | +0.001130 [+0.000669,+0.001866] |
| `xaj_b` | dPL | +0.036210 [+0.027970,+0.048278] | +0.030127 [+0.020697,+0.039371] | +0.037922 [+0.028576,+0.048049] | +0.025712 [+0.017872,+0.033488] |
| `xaj_im` | dPL | +0.030239 [+0.024701,+0.038354] | +0.001838 [+0.000575,+0.002950] | +0.088453 [+0.072931,+0.107279] | −0.016395 [−0.030156,−0.009200] |
| `xaj_um` | dPL | +0.115315 [+0.101812,+0.141288] | +0.104820 [+0.086386,+0.127153] | +0.033581 [+0.028855,+0.041016] | +0.022332 [+0.017215,+0.031337] |
| `xaj_lm` | dPL | +0.072083 [+0.061693,+0.088472] | +0.065249 [+0.052125,+0.083589] | +0.029506 [+0.026683,+0.033809] | +0.006380 [+0.003845,+0.008884] |
| `xaj_dm` | dPL | +0.041196 [+0.035276,+0.048233] | +0.032934 [+0.025630,+0.040179] | +0.032403 [+0.025434,+0.037612] | +0.001826 [−0.000330,+0.004592] |
| `xaj_c` | dPL | +0.065801 [+0.056732,+0.081472] | +0.042885 [+0.031563,+0.052456] | +0.036977 [+0.033130,+0.042473] | +0.016252 [+0.011603,+0.020895] |
| `xaj_sm` | dPL | +0.072928 [+0.062133,+0.085013] | +0.061539 [+0.047953,+0.073015] | +0.079391 [+0.070250,+0.093782] | +0.032124 [+0.023727,+0.040637] |
| `xaj_ex` | dPL | +0.173070 [+0.161589,+0.186592] | +0.109346 [+0.095670,+0.123380] | +0.118147 [+0.108864,+0.135300] | +0.073946 [+0.061680,+0.082441] |
| `xaj_ki` | dPL | +0.121428 [+0.105639,+0.147963] | +0.092790 [+0.078465,+0.116443] | +0.053421 [+0.045029,+0.060533] | +0.012780 [+0.008109,+0.017935] |
| `xaj_kg` | dPL | +0.090740 [+0.066831,+0.109914] | +0.040945 [+0.030173,+0.057142] | +0.125514 [+0.102401,+0.147115] | +0.020026 [+0.013111,+0.029645] |
| `xaj_ci` | dPL | +0.156231 [+0.123219,+0.180153] | +0.090282 [+0.065986,+0.111499] | +0.118307 [+0.106713,+0.133854] | +0.045582 [+0.038896,+0.053232] |
| `xaj_cg` | dPL | +0.098559 [+0.079570,+0.116841] | +0.070011 [+0.054639,+0.087470] | +0.079691 [+0.065964,+0.099739] | +0.054322 [+0.041554,+0.072885] |
| `xaj_a` | dPL | +0.033763 [+0.027949,+0.042484] | +0.020589 [+0.016254,+0.026254] | +0.036900 [+0.032604,+0.042080] | +0.017345 [+0.010966,+0.021673] |
| `xaj_theta` | dPL | +0.017405 [+0.014369,+0.020484] | +0.011717 [+0.009733,+0.014712] | +0.030377 [+0.023107,+0.036193] | +0.019176 [+0.014350,+0.026937] |

### 13.2 component-level placement judgment

- Component patterns are heterogeneous; no single parameter is necessary to carry the 3.4.4 conclusion.
- `u_m` has larger positive C15 under Base than TGD in both IC/dPL, but this is component-level CN-adjusted error separation, not 3.3 compensation slope.
- `c_i` and `k_i` show useful examples of reduced TGD component error in some regimes, but signs and magnitudes are not uniformly identical across the two component definitions.
- Recommendation: main text uses aggregate J1/J2 only; Table K goes Supplement. Do not force 2–3 parameters into the main narrative unless Figure S3 is already retained for manuscript continuity.

## 14. Manuscript / Supplement / Canonical 冲突审计

### 表 L：版本与冲突

| Quantity / artifact | Manuscript | Supplement / other audit | Canonical direct value | Final value to use | Reason |
|---|---|---|---|---|---|
| Base-no-refit KGE | IC/dPL≈0.898 | Figure 7/5 summary≈0.898 | 0.898015 | 0.898015 | seed-median source |
| CN-refit KGE | IC≈0.993，dPL≈0.995 | Figure 7/5 summary same | IC 0.992570；dPL 0.995495 | direct summary | correct-structure benchmark |
| `D_b` | IC≈0.087；dPL≈0.091 | Table S2 / final audit same | IC 0.086733；dPL 0.091071 | `r3_gap_recovery_ratio_audit.csv` | all-531 median |
| `G_Base/G_TGD` CI | rounded old Table 2 values | Figure 5 summary uses offset seeds for some raw-gain CIs | direct audit: IC `G_Base [0.000833,0.007729]`、`G_TGD [0.025385,0.049824]`；dPL `[0.003854,0.009876]`、`[0.026788,0.044640]` | direct Table 2 audit source | one CI protocol for main table |
| `F_close/F_TGD/DeltaF` | IC `.101/.546/.460`；dPL `.102/.521/.441` | `p0_reporting/recovery_denominator_tail_audit.md` 的 dPL medians `0.524/0.443` 为 stale/alternate aggregation；registry 也有 alternate CI | Table C direct audit | Table C | ratios basin-wise、unclipped；不使用 stale tail markdown 覆盖 direct CSV |
| dPL valid N | Table 2 `460` | canonical registry also lists a pool count `468` | seed-median Table 2 N=`460` | `460` | dPL seed median before ratio |
| TGD fraction notation | Table 2 `F_TGD` common reference | `derive_estimand_audit.py` also has incremental `F_tgd2` | common-reference `F_TGD=F_TGD_star` | common-reference | matches task and Table 2 |
| Invalid denominator concentration | manuscript约 91% S1 | tail audit IC 91.4%、dPL 91.5% | direct D table 91.3%、91.5% | 91%（IC 91.3，dPL 91.5） | exact direct counts |
| Alternative field | Supplement S3 reports `+0.195/+0.701` | final audit summary same | IC `+0.195275`、dPL `+0.702323` | direct alternative summary | qualitative ordering only |
| TGD shape IC canonical | Supplement raw `−0.1339` | shape audit proves positional-indexing bug；aligned `+0.4600` | raw shape CSV invalid for IC | aligned audit median with caveat；CI缺失 | never use raw IC shape value |
| TGD shape dPL canonical | `+0.441` | shape summary `+0.441` | basin metrics `+0.4410` | direct shape basin metric + bootstrap | valid dPL alignment |
| Regional LORO count | narrative says 18 / 8 retained | plot script and direct CSV have Region 01–07 | 7 omission rows | use 7 direct rows | actual file is authoritative |
| Regional Full IC reference | summary says `+0.460` | regional CSV Full row `+0.4405` | primary Table C `+0.459951` | use primary Table C | regional Full row uses different valid-only aggregation |
| Parameter aggregate error | manuscript `D_theta≈.180/.163/.070` | code calls `E_param=median abs(e)` | J1 direct summary | report as `D_theta/E_param` with footnote | not RMS |
| Figure S3 component error | narrative may say “excess error” | renderer uses `median abs(e-e_CN)`; protocol primary is `delta_abs_e` | Table K reports both | no conflation | distinct estimands |

### 14.1 旧编号与 R3/R4/R5 边界

仓库实际 R3 renderers 是 `manuscript/scripts/r3/plot_figure5.py`（outlet recovery）和 `plot_figure6.py`（parameter/state truth error），对应用户任务中所称的 Figure 7 / Figure 8 主题。本文档不把仓库文件编号强行改写；博士正文应使用新的第三章图号。Figure S3/S4/S5 的 Supplement numbering 曾有 assembly 变化，不能改变本报告的 direct source precedence。

## 15. 3.4 主文图表规划

### 15.1 主表：表 3-X

题目建议：**“合成流域中融雪结构缺口及其出口恢复程度”**。

字段：

| 参数估计方式 | Nvalid | `D_b` | `G_Base` | `G_TGD` | `F_close` | `F_TGD` | `DeltaF` | `P(F_TGD>F_close)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IC | 427 | +0.0867 | +0.0026 | +0.0386 | 0.101 | 0.546 | +0.460 | 91.6% |
| dPL | 460 | +0.0911 | +0.0073 | +0.0360 | 0.102 | 0.521 | +0.441 | 92.8% |

**表注必须写**：`D_b` 和 raw gains 的 median 使用全部 531 流域；fractions 与 paired `DeltaF` 使用 `D_b>1e-6` 的 denominator-valid subset；ratios 逐 basin 计算且不 clipping。

不要把 J1 truth-relative parameter error 放入同一张表。outlet attribution 与 parameter recovery 是不同 estimand，应放图 3-Y 或单独小表。

### 15.2 主图 3-X：结构缺口与出口恢复

建议 4 panel，现有 Figure 5 可重构，不原样复制全部 panels：

- **(a) outlet KGE ladder**：Base-no-refit、Base-refit、TGD-refit、CN-refit，IC/dPL 并列；
- **(b) raw quantities**：`D_b`、`G_Base`、`G_TGD`，明确它们使用 N=531；
- **(c) normalized recovery**：`F_close` vs `F_TGD`，标注 Nvalid；
- **(d) `DeltaF` robustness**：canonical、alternative generating field、shape sensitivity 的核心点，单独标注 shape IC alignment caveat。

该结构足以回答“缺口有多大—参数重估恢复多少—TGD恢复多少—排序是否稳健”。不要将 internal states/fluxes 放入 3.4 主图。

### 15.3 主图 3-Y：已知真值条件下 shared-parameter error

建议 2 panel：

- **(a) aggregate `D_theta/E_param`**：Base/TGD/CN-refit，按 IC/dPL；
- **(b) aggregate excess error S1–S5**：Base/TGD 两条线，CN residual 作为 excess baseline，不放状态/通量。

该结构足以支撑 3.4.4。component-level Table K 放 Supplement，不强行选单个 parameter 做主结论。

## 16. 建议直接进入正文的关键数字

建议只保留以下 16 组：

1. 样本：`N=531`；evaluation period=`1995-10-01–2010-09-30`。
2. IC Base-no-refit KGE：`0.8980`。
3. dPL Base-no-refit KGE：`0.8980`。
4. IC CN-refit KGE：`0.9926`。
5. dPL CN-refit KGE：`0.9955`。
6. IC Base-refit KGE：`0.8988`。
7. dPL Base-refit KGE：`0.9081`。
8. IC TGD-refit KGE：`0.9337`。
9. dPL TGD-refit KGE：`0.9440`。
10. IC `D_b/G_Base/G_TGD`：`+0.0867/+0.0026/+0.0386`。
11. dPL `D_b/G_Base/G_TGD`：`+0.0911/+0.0073/+0.0360`。
12. IC `F_close/F_TGD/DeltaF`：`0.101/0.546/+0.460`，Nvalid=`427`。
13. dPL `F_close/F_TGD/DeltaF`：`0.102/0.521/+0.441`，Nvalid=`460`。
14. paired positive fraction：IC `91.6%`，dPL `92.8%`。
15. canonical shape sensitivity：aligned IC/dPL `DeltaF≈+0.460/+0.441`；broad shape dPL `−0.091`，IC aligned `−0.030`但 CI未保存。
16. aggregate truth error：IC Base/TGD/CN=`0.180/0.163/0.070`；dPL=`0.094/0.055/0.026`；Full excess Base/TGD 为 IC=`+0.097/+0.076`、dPL=`+0.064/+0.024`。

正文建议将第 15 组写成 sensitivity boundary，而不是写成普适定量归因比例。

## 17. 建议进入脚注和附录的数据

### 脚注

- `D_b>1e-6` 的 exact denominator condition；
- ratios 逐 basin 后汇总，未 clipping；
- raw gains 用 531 流域，fractions 用 Nvalid；
- invalid denominator 主要来自 S1 低雪流域，不是模型失败；
- dPL 是 3-seed per-basin median；IC 是 best restart；
- TGD-specific parameters 没有对应 generating truth；
- `D_theta/E_param` 是 median absolute normalized shared-parameter error，不是 RMS；
- CN-refit residual 不预设为 0。

### 附录

1. 表 B 的全部 Q25–Q75、CI 和 positive fractions；
2. 表 C 的未截断 fraction tails；
3. 表 D、表 E 的完整 denominator/S1–S5 recovery；
4. 表 F 完整 threshold grid；
5. 表 G alternative field 完整结果；
6. 表 H 完整 shape sensitivity 和 IC alignment audit；
7. 表 I 7-region LORO；
8. 表 J1/J2 的 full 与 S1–S5 CI；
9. 表 K 全 15 component error；
10. train/evaluation attenuation；
11. bootstrap draws、seed、basin-unit 和 source hashes。

## 18. 需要移动到 3.5 的结果

以下内容即使出现在 Figure 8 / Figure 6 同一张图中，也不纳入 3.4 正文：

- `WU`、`WL`、`WD`、`Wtot`；
- `Qi`、`Qg` 及所有 state/flux NRMSE；
- state/flux excess error；
- recovery–internal-error Spearman；
- partial Spearman；
- seasonal trajectories / monthly state arrays；
- recovery–state association；
- seasonal liquid-water delivery；
- spring wet-up / peak timing；
- 其他 ET、Response 或 real-catchment consistency 分析。

这些结果在本报告中只作为“MOVE TO 3.5”边界，不参与 3.4 统计结论。

## 19. 数据缺口与是否需要补实验

### A. 必须补（若要把对应结果写成正式定量结论）

1. **Shape sensitivity 的 aligned IC CI**：raw IC shape CSV 有已知 positional-indexing bug；现有 aligned audit 有 median / positive fraction，但没有 corrected basin-level CI。若正文只报告 canonical shape sensitivity 的定性边界，可保留 caveat；若要正式比较四种 shape 的 IC CI，应另存 corrected aligned output。
2. **TGD common-reference notation**：正文、Table 2、Supplement 必须统一使用 `F_TGD=G_TGD/D_b`，避免与 incremental `F_tgd2` 混用。
3. **Parameter error nomenclature**：将 `D_theta` 明确写成 `E_param` 或脚注明确不是 RMS；component Table K 与 aggregate J2 必须分开。

### B. 推荐补（只用 existing frozen outputs 的轻量汇总）

1. 将表 A1 的 KGE bootstrap CI 作为独立 audit CSV 保存；
2. 将 Table H 的 aligned IC shape summary / CI 保存为新的 noncanonical audit artifact；
3. 将表 G、H、I 放入 Supplement，并注明 alternative field、shape alignment 和 direct CSV 版本；
4. 将 Table K 作为 Figure S3 panel (a) 的 machine-readable companion；
5. 在 Supplement 统一修正 “18 regions / 8 displayed” 与当前 7-row direct CSV 的说明。

以上均不需要训练或率定。

### C. 不需要补

以下不应为 3.4 新增：

1. 新 synthetic parameter field；
2. 新 TGD variant；
3. 新 IC calibration；
4. 新 dPL training；
5. 新 loss function；
6. state/flux error analysis；
7. seasonal trajectory analysis；
8. recovery–internal association；
9. real-catchment soil-water consistency；
10. cross-host GR4J / SIMHYD；
11. ET / Response 扩展过程。

## 20. 最终判断

1. **当前数据是否足够写博士论文 3.4？** 基本足够。canonical R3 outputs 已覆盖 outlet KGE ladder、all-531 raw gap/gains、denominator-valid fractions、S1–S5 recovery、threshold sensitivity、alternative generating field、regional omission，以及 15 shared XAJ 的 aggregate / component truth-relative error。
2. **是否存在 scientific blocker？** 核心 3.4 结论没有 blocker。若将四种 TGD shape 的 IC CI 作为正式定量主结论，则存在一个局部 artifact blocker：原始 IC shape output 的 basin alignment 错误，当前只有 aligned median / prevalence，缺 corrected CI。
3. **是否需要新模型训练？** 不需要。
4. **是否需要重新参数率定？** 不需要。
5. **是否仅需 frozen outputs 上的轻量统计？** 是。主要需要统一 common-reference `F_TGD`、保存 KGE/shape audit CI、整理附录，不改变实验设计。
6. **拟定三级标题是否保留？** 保留：
   - **3.4.1 合成流域中的结构缺失基准**
   - **3.4.2 参数重新估计的结构缺口恢复**
   - **3.4.3 通用温度条件控制的结构收益恢复**
   - **3.4.4 已知真值条件下的共享参数偏差**

**最终写作主线**：先用 Base-no-refit 与 CN-refit 建立已知 truth synthetic outlet gap；再说明 Base-refit 只恢复约 10% 的 gap，而 canonical TGD common-reference 恢复约 52%–55%，并用 raw gains 与 paired `DeltaF` 共同支撑；随后用 denominator、alternative field 和 shape sensitivity 限定该 quantitative attribution 的适用边界；最后说明在本受控 synthetic truth 下，TGD-refit 的 15 shared-parameter aggregate error 小于 Base-refit，但不把出口恢复等同于内部过程恢复，也不把 synthetic truth 外推为真实流域真值。
