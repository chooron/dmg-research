# Figure Optimization Handoff

## 0. 交接摘要

本 handoff 用于下一阶段的论文图片修改、版式优化、输出规范化和最终归档。目标不是重新训练模型，而是在**不改变已冻结统计结果和科学结论**的前提下，改进图片的可读性、出版质量、图例、面板布局和 provenance。

当前代码整理已经完成并提交：

```text
Commit: ed6729c
Message: refactor(hydrodiag): organize chapter analysis under manuscript
Branch: master
工作区: clean
master 相对 origin/master: ahead 1
```

当前最重要的科学状态：

1. R3 图 5、图 6 和补充图 S5 使用已整理好的 `manuscript/results/R3/` 数据包。
2. R4 Base/CN/IC 的正式土壤水状态一致性分析已经有正式结果和图表。
3. R4 三结构（Base/TGD2/CN）的派生 CSV、Table 4、Figure 7、Figure 8、Figure S6 已经存在，但**不能因为存在状态 NPZ 或图片就宣称 TGD2 三个 seed 训练全部完成**。
4. TGD2 seed 42 的真实训练目前只到 epoch 60，缺少 `COMPLETE` 和最终训练产物；seed 123、2026 已到 epoch 100。
5. 在 seed 42 完成并通过 provenance 校验前，含 TGD2 的图只能视为 interim/draft，不能作为最终论文证据。

---

## 1. 当前目录结构

从以下目录执行命令：

```bash
cd /home/jingxin/code/dmg-research/project/hydrodiag
```

```text
manuscript/
├── r3/                         # R3 运行时分析包
├── r4/                         # R4 运行时分析包
├── scripts/
│   ├── r1/                     # R1 图表、统计与推演脚本
│   ├── r2/                     # R2 图表、稳健性、ablation、TGD2 specificity
│   ├── r3/                     # R3 图表、审计、汇总和 source-data 生成
│   ├── r4/                     # R4 图表、状态导出、三结构统计
│   ├── r5/                     # R5 生产流水线
│   └── shared/                 # 跨章节质量门禁、采样、审计、绘图样式
├── figures/                    # 当前论文主图和部分诊断图
├── plots/figures/              # R1/R2/R3 部分历史输出位置
├── supplement/figures/         # 补充图
├── tables/                     # 当前论文表格及 CSV/Markdown/LaTeX
├── stats/tables/               # 统计汇总和部分表格
├── results/R3/                 # R3 图表使用的冻结中间数据包
└── README.md                   # manuscript 总说明
```

当前 import 约定：

```python
from manuscript.r3.common import ...
from manuscript.r4.common import ...
from manuscript.scripts.shared.r1_plot_style import ...
```

不要再使用旧路径：

```python
from r3 ...
from r4 ...
from scripts ...
```

---

## 2. 代码职责边界

### 2.1 `manuscript/r3/`

R3 运行时和分析逻辑，包括：

- synthetic truth 生成和加载：`truth_generator.py`, `generate_truth.py`
- recorded-forward 校验：`recorded_forward.py`
- Base/no-refit、IC、dPL gate：`run_base_no_refit.py`, `run_gate_531.py`
- oracle 和参数恢复审计：`oracle_identity.py`, `oracle_dpl_audit.py`, `gate_analysis.py`
- misspecification 分析：`misspec_analysis.py`, `misspec_states.py`
- post-hoc 统计与验证：`posthoc_stats.py`, `posthoc_validation.py`
- R3 协议和审计：`protocol_misspec_v1.json`, `docs/kge_audit.md`

R3 图表代码不要重新实现模型过程；图表应读取已生成的 `manuscript/results/R3/` 或 `results/r3_*` 结果。

### 2.2 `manuscript/r4/`

R4 real-basin state consistency 运行时和统计逻辑，包括：

- 数据适配与 Caravan 参考：`input_adapters.py`, `common.py`
- XAJ/CN/TGD2 状态导出：`state_export.py`, `forward_export.py`
- 土壤状态一致性：`soil_analysis.py`
- R4 phase-1 分析：`phase1_dpl_analysis.py`, `phase1_ic_fused_analysis.py`
- 稳健性分析：`robustness_analysis.py`
- 雪参考和 HUC02 审计：`snow_reference.py`, `audit_huc02_from_daymet.py`
- 运行协议与同步清单：`protocol_r4_soil_v1.json`, `sync_manifest.json`

### 2.3 `manuscript/scripts/r3/`

主要入口：

```text
prepare_figure5_data.py          生成 Figure 5 source data
plot_figure5.py                  渲染 Figure 5
prepare_figure6_data.py          生成 Figure 6 source data
plot_figure6.py                  渲染 Figure 6
export_figure6_process_data.py   导出 Figure 6 过程数据
plot_r3_si_components.py         渲染 Figure S5_R3_components
generate_table_r3_main.py        生成 R3 主表
generate_table_r3_si.py         生成 R3 补充表
```

### 2.4 `manuscript/scripts/r4/`

主要入口：

```text
plot_r4_figure4.py               旧/正式 R4 Base-CN/IC 土壤状态图
plot_r4_figure7.py               旧版两 regime Figure 7 渲染器
plot_r4_figure8.py               旧版两 regime Figure 8 渲染器
plot_r4_figure_s6.py              旧版多 seed/IC 补充图渲染器
build_r4_soil_statistics.py      R4 soil statistics 构建
build_complete_three_structure_r4.py
                                  Base/TGD2/CN 三结构统计和 robustness 构建
generate_three_structure_r4_all.py
                                  三结构 Figure 7/8/S6 + Table 4/S6/S7 一体化生成器
generate_table_r4.py             R4 表格生成器
export_all_tgd2_states.py        TGD2 状态导出
```

### 2.5 `manuscript/scripts/shared/`

- `r1_plot_style.py`：共享颜色、marker、字体、spine 和出版级绘图样式。
- `run_model_test_suite.py`：模型质量门禁。
- `audit_native_response_endpoints.py`：原生响应审计。
- `freeze_native_subsurface_scale.py`：冻结 subsurface scale。
- `phase0_sampling.py`：Phase-0 采样。

修改图形风格时优先修改共享样式，而不是在每个图脚本中复制一套颜色和字体。

---

## 3. 当前图片清单

以下是当前工作区已存在的图片及其尺寸。尺寸只说明文件当前状态，不代表已经满足最终期刊版式要求。

| 图 | 当前文件 | 当前尺寸 | 主要生成器 | 状态 |
|---|---|---:|---|---|
| Figure 1 | `manuscript/plots/figures/Figure1_R1_compensation_overview.png` | 当前未归档在工作区图片清单中 | `scripts/r1/plot_r1_figure1.py` | 需要从 R1 结果重渲染/确认 |
| Figure 2 | `manuscript/plots/figures/Figure2_R1_ct_error_snow_regimes.png` | 当前未归档在工作区图片清单中 | `scripts/r1/plot_r1_figure2.py` | 需要从 R1 结果重渲染/确认 |
| Figure 3 | `manuscript/plots/figures/Figure3_R2_final.png` | 当前未归档在工作区图片清单中 | `scripts/r2/plot_r2_figure3_final.py` | 需要从 R2 结果重渲染/确认 |
| Figure 4 | `manuscript/plots/figures/Figure4_R2.png` 或 R4 同编号文件 | 当前需区分章节编号 | `scripts/r2/plot_r2_figure4.py` / `scripts/r4/plot_r4_figure4.py` | 必须避免编号冲突 |
| Figure 4 R4 | `manuscript/figures/figure4_r4_soil_consistency.png` | 4424 × 3344 | `scripts/r4/plot_r4_figure4.py` | Base/CN/IC 正式 R4 图；另有 PDF |
| Figure 5 | `manuscript/figures/Figure5_R3_final.png` | 6948 × 7870 | `scripts/r3/prepare_figure5_data.py` + `plot_figure5.py` | 已有冻结 R3 数据包 |
| Figure 5 duplicate | `manuscript/plots/figures/Figure5_R3_final.png` | 6948 × 7870 | 同上 | 与 `figures/` 存在重复输出 |
| Figure 6 | `manuscript/figures/Figure6_R3_final.png` | 7979 × 6113 | `scripts/r3/prepare_figure6_data.py` + `plot_figure6.py` | 已有冻结 R3 数据包 |
| Figure 6 duplicate | `manuscript/plots/figures/Figure6_R3_final.png` | 7979 × 6113 | 同上 | 与 `figures/` 存在重复输出 |
| Figure 7 R4 | `manuscript/figures/figure7_r4_soil_consistency.png` | 2131 × 2747 | `generate_three_structure_r4_all.py` 或旧 `plot_r4_figure7.py` | 含 TGD2 的结果需标记 interim |
| Figure 8 R4 | `manuscript/figures/figure8_r4_soil_timing.png` | 2172 × 1861 | `generate_three_structure_r4_all.py` 或旧 `plot_r4_figure8.py` | 含 TGD2 的结果需标记 interim |
| Figure S5 R3 | `manuscript/supplement/figures/Fig_S5_R3_components.png` | 5066 × 5534 | `scripts/r3/plot_r3_si_components.py` | R3 补充图 |
| Figure S6 R4 | `manuscript/supplement/figures/Fig_S6_r4_multiseed_replication.png` | 2114 × 973 | `generate_three_structure_r4_all.py` 或 `plot_r4_figure_s6.py` | TGD2/seed 完成状态需核验 |
| Figure S6 duplicate | `manuscript/figures/Fig_S6_r4_multiseed_replication.png` | 2114 × 973 | 同上 | 现有脚本会同时写两个位置 |
| R1 diagnostic | `manuscript/figures/figure_r13_root_cause_diagnostics.png` | 4799 × 3633 | `scripts/plot_r13_root_cause_diagnostics.py` | 诊断图；另有 PDF |
| R1 diagnostic | `manuscript/figures/figure_r14_feasibility_diagnostics.png` | 4355 × 3596 | `scripts/plot_r14_feasibility_diagnostics.py` | 诊断图；另有 PDF |

### 3.1 重要的编号冲突

R2 有 `Figure4_R2`，R4 也有 `figure4_r4_soil_consistency`。修改图片时必须使用章节前缀或完整文件名，不要只凭 `Figure 4` 判断文件。

建议后续统一采用：

```text
Figure1_R1_...
Figure2_R1_...
Figure3_R2_...
Figure4_R4_...
Figure5_R3_...
Figure6_R3_...
Figure7_R4_...
Figure8_R4_...
FigureS5_R3_...
FigureS6_R4_...
```

---

## 4. 图片与数据的对应关系

### 4.1 R3 Figure 5

代码：

```text
manuscript/scripts/r3/prepare_figure5_data.py
manuscript/scripts/r3/plot_figure5.py
```

输入：

```text
manuscript/results/R3/figure5_summary.json
manuscript/results/R3/figure5_basin_table.csv
manuscript/results/R3/figure5_basin_seedmedian.csv
```

输出：

```text
manuscript/figures/Figure5_R3_final.png
manuscript/plots/figures/Figure5_R3_final.png
```

修改规则：

- `figure5_summary.json` 和两个 CSV 是图的冻结 source data，不要在绘图脚本内重新计算统计量。
- 任何 panel 删除、排序、颜色或 annotation 修改，都应只改变 plot 层。
- 如果改变了统计口径，必须先更新 source data 和 summary，再更新图，不能只修改图上的数字。

### 4.2 R3 Figure 6

代码：

```text
manuscript/scripts/r3/prepare_figure6_data.py
manuscript/scripts/r3/plot_figure6.py
manuscript/scripts/r3/export_figure6_process_data.py
```

输入：

```text
manuscript/results/R3/figure6_summary.json
manuscript/results/R3/figure6_basin_table.csv
manuscript/results/R3/figure6_basin_seedmedian.csv
manuscript/results/R3/fig6_seasonal/fig6_seasonal_meta.json
```

输出：

```text
manuscript/figures/Figure6_R3_final.png
manuscript/plots/figures/Figure6_R3_final.png
```

修改时要重点检查：

- train/test 的区分是否仍然清楚；
- Base/TGD/TGD2/CN 的颜色是否和正文其他图一致；
- seasonal panel 的时间顺序、单位和图例是否一致；
- 不要把 standard KGE 与其他变体混在同一图中。R3 的 KGE 约定见 `manuscript/r3/docs/kge_audit.md`。

### 4.3 R3 Figure S5

代码：

```text
manuscript/scripts/r3/plot_r3_si_components.py
```

输出：

```text
manuscript/supplement/figures/Fig_S5_R3_components.png
```

这是 R3 的组件/补充分析图。修改时保持与 Figure 5/6 相同的模型颜色、字体和 panel label 语法。

### 4.4 R4 Figure 4

代码：

```text
manuscript/scripts/r4/plot_r4_figure4.py
```

输入主目录：

```text
results/r4_phase1_soil_official/
```

关键输入：

```text
basin_state_consistency.csv
paired_structural_effects.csv
timing_metrics_basin_year.csv
timing_metrics_basin_summary.csv
snow_burden_quartile_summary.csv
r4_phase1_soil_official_report.json
```

输出：

```text
manuscript/figures/figure4_r4_soil_consistency.png
manuscript/figures/figure4_r4_soil_consistency.pdf
```

正式 R4 研究语义：

- 531 个 CAMELS-US basin；
- test period：1995-10-01 至 2010-09-30；
- 连续 forward：1980-10-01 至 2014-09-30，从零初始状态连续推演后切片；
- `W_total = WU + WL + WD`；
- 外部参考为 Caravan/ERA5-Land soil moisture，不是 ground truth；
- 主要参考为 `SM100 = 0.07 L1 + 0.21 L2 + 0.72 L3`；
- 只比较 standardized/anomaly/timing dynamics，不做 mm 与 m³/m³ 的绝对存储转换。

### 4.5 R4 Figure 7

目前有两条代码路径，不能混用：

#### 旧版/两 regime 路径

```text
manuscript/scripts/r4/plot_r4_figure7.py
```

它主要使用：

```text
results/r4_phase1_soil_official/robustness_swe_decile_shape.csv
results/r4_phase1_soil_official/robustness_process_phase_consistency.csv
results/r4_phase1_soil_official/robustness_controlled_regressions.csv
results/r4_phase1_soil_official/robustness_extreme_swe_trimming.csv
```

默认重点是 canonical dPL seed 42 和 IC fused，seed 123 主要作为补充复制证据。

#### 当前三结构路径

```text
manuscript/scripts/r4/generate_three_structure_r4_all.py
```

它读取：

```text
results/r4_phase1_soil_official/three_structure_swe_decile_shape.csv
results/r4_phase1_soil_official/three_structure_process_phase_consistency.csv
results/r4_phase1_soil_official/three_structure_paired_structural_effects.csv
results/r4_phase1_soil_official/robustness_controlled_regressions.csv
results/r4_phase1_soil_official/robustness_leave_one_region_out.csv
results/r4_phase1_soil_official/robustness_extreme_swe_trimming.csv
```

该脚本同时生成：

```text
manuscript/figures/figure7_r4_soil_consistency.png
manuscript/figures/figure8_r4_soil_timing.png
manuscript/figures/Fig_S6_r4_multiseed_replication.png
manuscript/supplement/figures/Fig_S6_r4_multiseed_replication.png
manuscript/tables/Table4_soil_state_consistency.*
manuscript/tables/TableS6_robustness_checks.*
manuscript/tables/TableS7_timing_sensitivity.*
```

**不要先运行旧版 Figure 7，再运行三结构版，或反过来。两个脚本会覆盖相同的 PNG 文件。**

### 4.6 R4 Figure 8

Figure 8 重点是：

1. 雪积累/消退；
2. standardized soil-water trajectory；
3. spring process zoom；
4. Q3 basin 的 wet-up timing；
5. soil-water peak timing；
6. timing definition sensitivity。

旧版入口：

```text
manuscript/scripts/r4/plot_r4_figure8.py
```

三结构版由：

```text
manuscript/scripts/r4/generate_three_structure_r4_all.py
```

完成。Figure 8 的 illustrative basin/year 选择由已有数据自动选择；任何手工替换都必须保存 selection audit，不能凭视觉挑选有利样本。

### 4.7 R4 Figure S6

代码：

```text
manuscript/scripts/r4/plot_r4_figure_s6.py
```

或三结构一体化脚本：

```text
manuscript/scripts/r4/generate_three_structure_r4_all.py
```

当前 Figure S6 的设计语义是比较：

- dPL seed 42；
- dPL seed 123；
- IC fused；
- Snow-burden decile dependence；
- process-phase fingerprint。

但如果图片加入 TGD2，必须重新审查标题和 caption，明确这是 observation-trained TGD2 interim 结果，直到三 seed 全部有完整训练 provenance。

---

## 5. 数据和结果资产说明

### 5.1 R4 外部参考和环境轴

```text
results/r4_caravan_soil_reference_v1/
└── caravan_soil_ensemble.npz

results/r4_swe_reference_v1/
├── swe_ensemble.npz
├── swe_basin_burden_test.csv
├── swe_annual_metrics.csv
└── manifest.json
```

含义：

- Caravan `SM100` 是主要 external process-state reference；
- `SM289` 是 sensitivity reference；
- Snow-17 ensemble median annual maximum SWE 是主要 snow-burden axis；
- `frac_snow` 只能作为 CAMELS 静态 baseline axis，不能替代 Snow-17 主轴。

### 5.2 R4 forward state arrays

正式 forward arrays 位于 `results/`，不提交到 Git：

```text
results/r4_official_dpl_XAJ_seed42/
results/r4_official_dpl_XAJ_seed123/
results/r4_official_dpl_XAJ_CN_seed42/
results/r4_official_dpl_XAJ_CN_seed123/
results/r4_official_dpl_XAJ_TGD2_seed42/
results/r4_official_dpl_XAJ_TGD2_seed123/
results/r4_official_dpl_XAJ_TGD2_seed2026/
results/r4_ic_fused_XAJ/
results/r4_ic_fused_XAJ_CN/
results/r4_ic_fused_XAJ_TGD2/
```

这些 NPZ 是状态推演/forward 数组，不等于训练完成证明。训练完成必须回到训练目录核验 `COMPLETE`、epoch history、checkpoint、normalized/physical parameters、final summary 和 report。

### 5.3 R4 正式结果目录

```text
results/r4_phase1_soil_official/
```

关键文件：

| 文件 | 用途 |
|---|---|
| `basin_state_consistency.csv` | 531 basin × regime × structure 的基础状态一致性 |
| `paired_structural_effects.csv` | CN/Base paired effect |
| `timing_metrics_basin_year.csv` | basin-year timing 明细 |
| `timing_metrics_basin_summary.csv` | basin-level timing 汇总 |
| `snow_burden_quartile_summary.csv` | Table 4 的 quartile 汇总 |
| `robustness_process_phase_consistency.csv` | Table S4/process phase |
| `robustness_swe_decile_shape.csv` | SWE decile response |
| `robustness_controlled_regressions.csv` | 控制 `Delta KGE` 的 regression |
| `robustness_leave_one_region_out.csv` | 18 HUC region leave-one-out |
| `robustness_extreme_swe_trimming.csv` | top 1%/5% trimming |
| `robustness_timing_sensitivity.csv` | timing definition sensitivity |
| `three_structure_*.csv` | Base/TGD2/CN 派生结果，需经过 TGD2 provenance gate |
| `r4_phase1_soil_official_report.json` | 正式主报告 |
| `r4_robustness_report.json` | robustness 总报告 |

已核验的主要规模包括：

- `basin_state_consistency.csv`：3,187 rows；
- `paired_structural_effects.csv`：1,594 rows；
- `timing_metrics_basin_year.csv`：47,791 rows；
- `timing_metrics_basin_summary.csv`：3,187 rows；
- `robustness_process_phase_consistency.csv`：4,213 rows。

### 5.4 R3 Figure source-data 包

```text
manuscript/results/R3/
├── figure5_summary.json
├── figure5_basin_table.csv
├── figure5_basin_seedmedian.csv
├── figure6_summary.json
├── figure6_basin_table.csv
├── figure6_basin_seedmedian.csv
├── table5_main_summary.csv
├── tableS5_si_statistics.csv
└── fig6_seasonal/fig6_seasonal_meta.json
```

这是图 5/6 的直接输入包。图片优化阶段应该优先读取这里，不要扫描整个 `results/` 猜测输入。

---

## 6. TGD2 完成状态和图片使用边界

### 6.1 训练源目录

```text
results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/
├── seed_42/
├── seed_123/
└── seed_2026/
```

当前状态：

| seed | config epochs | epoch history | 最大 checkpoint | COMPLETE | final summary | 结论 |
|---:|---:|---:|---:|---|---|---|
| 42 | 100 | 60 | 60 | 否 | 否 | 未完成 |
| 123 | 100 | 100 | 100 | 是 | 是 | 完成 |
| 2026 | 100 | 100 | 100 | 是 | 是 | 完成 |

seed 42 当前有 `best_parameters_physical.npz`，但缺少：

```text
COMPLETE
basin_final_summary.csv
best_parameters_normalized.npz
report.md
```

因此：

- 不要把 seed 42 的 state NPZ 视为完成训练；
- 不要在图注中写“三个 TGD2 seeds completed”；
- 不要把 TGD2 画成已确认的最终竞争结构；
- 不要用 Base/CN/TGD2 图反推训练状态；
- `r4_phase1_soil_official_report.json` 当前仍包含 `TGD2_PENDING` 状态。

### 6.2 三结构 builder 的已知风险

`build_complete_three_structure_r4.py` 当前的 regime mapping 中：

- dPL seed 42 使用 Base/CN seed 42 + TGD2 seed 42；
- dPL seed 123 使用 Base/CN seed 123 + TGD2 seed 123；
- dPL seed 2026 使用 TGD2 seed 2026，但 Base/CN 仍映射到 seed 42 的 forward arrays；
- IC fused 使用对应的 IC arrays。

这意味着 dPL seed 2026 当前不是完整 seed-matched Base/CN/TGD2 三结构复制。后续若要将 seed 2026 作为正式 replication，必须明确：

1. 是否有 seed 2026 的 Base/CN arrays；
2. 是否有必要重新导出 seed-matched Base/CN；
3. 是否要在图中只使用 seed 42/123；
4. 三结构表格和 Figure S6 是否需要重新生成。

在这个问题解决前，图片 caption 不应把 dPL seed 2026 写成完整的 independent three-structure replication。

---

## 7. 允许声明与禁止声明

### 7.1 R4 当前可以保留的结论

可以使用以下表述，但必须限定范围：

- 在强 snow-affected catchments（Q3，约 SWE ≥ 200 mm），显式 snow representation 的 CN 相比 Base 对 Caravan `SM100` 的共享土壤水状态一致性更高；
- 差异主要集中在 Phase 2 Active Melt / Spring Recharge；
- CN 可减少 spring recharge onset timing error，并减弱 Base 的 winter false-saturation timing bias；
- Base/CN 结果在 dPL seed 42、seed 123 和 IC fused sensitivity 中具有相似趋势，具体范围以正式表格为准。

### 7.2 禁止或暂时禁止的表述

- 不要写“效果随 SWE 线性增加”；当前证据更适合表述为 high-snow emergence。
- 不要把 Caravan/ERA5-Land soil moisture 称为 ground truth。
- 不要声称 XAJ 的 WU/WL/WD 与 ERA5-Land L1/L2/L3 存在一一对应深度关系。
- 不要声称 CN 唯一优于 TGD2；TGD2 observation-trained canonical checkpoint 状态仍是 `TGD2_PENDING`。
- 不要用状态导出完成来证明训练完成。
- 不要把临时 illustrative basin 当作代表性 basin，除非保留 selection audit。

---

## 8. 推荐的图片优化工作流

### Phase A：只做视觉审查，不运行模型

1. 复制当前图片到 revision 目录，不覆盖正式 PNG/PDF。
2. 记录每张图的：
   - 版面宽度、纵横比；
   - panel 数量和阅读顺序；
   - 字体、字号、数学符号；
   - 颜色、marker、line style；
   - legend 位置和冗余；
   - x/y 单位、tick 密度、科学计数法；
   - caption 需要的数字是否来自 CSV；
   - PNG/PDF 是否同时存在。
3. 先决定论文目标尺寸：single-column 或 double-column，不要以当前像素尺寸直接判断。

### Phase B：冻结数据输入

对每张图建立一个小型 manifest，至少包含：

```json
{
  "figure": "Figure7_R4",
  "script": "manuscript/scripts/r4/generate_three_structure_r4_all.py",
  "git_commit": "ed6729c",
  "input_files": [],
  "input_sha256": {},
  "data_status": "interim|formal",
  "tgd2_status": "TGD2_PENDING|complete",
  "output_files": [],
  "visual_change_only": true
}
```

大型 NPZ、checkpoint 和结果 CSV 不提交 Git；manifest 和最终图文件可以提交。

### Phase C：先改绘图层

优先修改：

- `manuscript/scripts/shared/r1_plot_style.py`；
- 各图的 layout、label、legend、annotation；
- `figsize`、`gridspec`、`subplots_adjust`；
- 输出格式和 dpi；
- source-data 读取和 provenance。

不要在图片优化阶段修改：

- 模型 forward kernel；
- 训练配置；
- parameter bounds；
- KGE 定义；
- basin selection；
- snow quantile 定义；
- bootstrap seed/rounds；
- test period；
- Base/CN/TGD2 的输入映射。

### Phase D：避免一体化脚本重复重算

`generate_three_structure_r4_all.py` 同时做统计、图和表，并且可能读取大量 NPZ。视觉修改不应每次都运行完整统计 pipeline。

建议后续先增加以下能力之一：

1. 增加 `--figures-only`，只读取现有 `three_structure_*.csv`；或
2. 将 Figure 7/8/S6 renderer 拆出到独立模块；或
3. 在临时分支中复制 renderer，固定输入目录和输出目录。

在未拆分前，不要为了改字体而重新跑完整三结构统计。

### Phase E：输出版本隔离

推荐目录：

```text
manuscript/figures/revision_v1/
manuscript/supplement/figures/revision_v1/
results/figure_optimization_v1/
```

每轮修改用新后缀：

```text
Figure7_R4_v1_draft.png
Figure7_R4_v2_draft.png
Figure7_R4_final.png
```

当前正式文件在获得人工确认前不要覆盖。

---

## 9. 低资源运行建议

WSL/本地环境应默认限制线程：

```bash
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

低风险操作：

```bash
uvx ruff format --check --quiet project/hydrodiag
uvx ruff check --select I project/hydrodiag
git diff --check
```

只渲染已有结果的 R4 图片时，优先使用：

```bash
python manuscript/scripts/r4/plot_r4_figure4.py --out-dir manuscript/figures/revision_v1
python manuscript/scripts/r4/plot_r4_figure7.py --out-dir manuscript/figures/revision_v1
python manuscript/scripts/r4/plot_r4_figure8.py --out-dir manuscript/figures/revision_v1
python manuscript/scripts/r4/plot_r4_figure_s6.py --out-dir manuscript/supplement/figures/revision_v1
```

注意：上述命令中的 `plot_r4_figure7.py`、`plot_r4_figure8.py` 是旧版两-regime renderer。若目标是三结构图，必须使用现有 three-structure CSV，并先给 `generate_three_structure_r4_all.py` 增加 figures-only/out-dir 能力，避免重新跑统计。

禁止在没有明确批准时运行：

```bash
python manuscript/scripts/r4/build_complete_three_structure_r4.py
python manuscript/scripts/r4/generate_three_structure_r4_all.py
```

原因：这些步骤可能加载 531 basin 的全量 state arrays，并产生明显 CPU、内存和磁盘压力；同时会有覆盖现有 tables/figures 的风险。

---

## 10. 图片质量验收清单

每张图片至少检查：

### 科学内容

- [ ] panel 顺序和正文叙述一致；
- [ ] 所有数值来自对应 CSV/JSON；
- [ ] 图例没有把 interim TGD2 写成正式结果；
- [ ] Q3 边界、n 值、单位和时间窗口正确；
- [ ] external reference 没有被写成 truth；
- [ ] Base/CN/TGD2/IC regime 名称与数据列一致；
- [ ] seed 42/123/2026 的实际 provenance 已确认。

### 视觉设计

- [ ] 字体可嵌入 PDF；
- [ ] 数学符号和 minus sign 正确；
- [ ] 黑白打印时线型和 marker 仍可区分；
- [ ] 色盲友好；
- [ ] panel label 统一为 `(a)`, `(b)` 等；
- [ ] legend 不遮挡数据；
- [ ] tick、单位、字体大小符合目标期刊；
- [ ] 无过度留白、裁剪或文本溢出；
- [ ] PNG 与 PDF/SVG 的内容一致；
- [ ] 保存前后没有意外改动 source-data。

### Provenance

- [ ] manifest 记录 Git commit；
- [ ] manifest 记录脚本路径；
- [ ] manifest 记录全部输入文件及 sha256；
- [ ] manifest 记录输出文件尺寸和格式；
- [ ] manifest 记录 `formal` 或 `interim`；
- [ ] 大型 checkpoint/NPZ 没有被误加入 Git；
- [ ] `git diff --check` 通过；
- [ ] 代码没有生成新的未声明 `outputs/` 目录。

---

## 11. 下一阶段建议顺序

### 第一步：先完成纯视觉版本

优先处理不依赖新训练的图：

1. Figure 4 R4；
2. Figure 5 R3；
3. Figure 6 R3；
4. Figure S5 R3；
5. R1/R2 已有结果对应的图片。

这些图可以在不触碰 TGD2 seed 42 的情况下进行版式优化。

### 第二步：确认 Figure 7/8/S6 的目标版本

在修改前先决定：

- 保留正式的 Base/CN/IC 两-regime 版本；还是
- 等 TGD2 三 seed 完成后，制作正式 Base/TGD2/CN 三结构版本。

不能同时把旧版和三结构版写入同一个最终文件名。

### 第三步：完成 TGD2 seed 42 provenance

只由训练负责人执行：

1. 核查是否有另一份更新的 seed 42 训练目录；
2. 若没有，继续或重新完成 100 epoch；
3. 确认 `COMPLETE`、epoch 100、checkpoint、normalised/physical parameters、final summary、report；
4. 重新生成 seed 42 state export；
5. 生成带 manifest 的 state bundle；
6. 再运行三结构统计；
7. 最后重新渲染 Figure 7/8/S6 和 Table 4/S6/S7。

### 第四步：最终归档

最终提交应至少包含：

```text
manuscript/figures/最终图片
manuscript/supplement/figures/最终补充图片
manuscript/tables/最终表格
manuscript/figure_manifests/*.json
manuscript/FIGURE_OPTIMIZATION_HANDOFF.md
```

大型训练和 forward 数据继续保留在本地/结果存储，不直接提交 Git。Git 中必须有足够的 manifest 和 README，使其他人可以知道：图片由什么脚本、什么数据、什么 commit 生成。

---

## 12. 本 handoff 的权威性边界

本文件描述的是当前 `master` 工作区 `ed6729c` 的目录和资产状态。以下内容仍需在下一轮工作中重新确认：

- TGD2 seed 42 是否在其他目录已经完成；
- 三结构 seed 2026 是否有 seed-matched Base/CN forward arrays；
- Figure 7/8/S6 最终采用两-regime 还是三结构版本；
- 目标期刊的最终版面尺寸、字体和 PDF/SVG 要求；
- R1/R2 旧结果是否需要从 `results/` 重新生成图片。

在上述问题确认前，任何包含 TGD2 的新图片都应在文件名、manifest 和内部沟通中明确标注为 `interim`，不得作为最终论文图提交。
