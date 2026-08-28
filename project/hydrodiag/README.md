# HydroDiag: Hydrological Model Structure Diagnosis & Process Fidelity Framework

> **Paper Fast-Start & Reproduction Guide**
> 
> 本文档为 HydroDiag（水文模型结构诊断与过程真实性分析）论文实验的权威快速上手与复现指南。包含完整的实验流程、数据来源、模型定义、复现命令（Figure 1–9、Table 1–2、Table S1–S2）以及故障排查指引。

---

## 1. 论文与研究背景 (Overview)

在可微水文模拟（Differentiable Parameter Learning, dPL）与传统独立率定（Independent Calibration, IC）中，高精度的出口径流表现（如高 KGE）常常掩盖了**结构性缺陷（Structural Mis-specification）**和**参数过补偿（Parameter Equifinality & Compensation）**。

HydroDiag 建立了一套严谨的诊断体系，系统揭示：
1. **R0 (Study Domain)**: CAMELS-531 大样本流域在降雪驱动（Snow fraction / Q3 High-snow）上的空间梯度分布。
2. **R1 (Outlet Timing Signatures)**: 缺乏显式融雪模块的模型（Base / TGD）在积雪流域出现严重的水文过程线滞后（Centroid Timing, CT 误差），而引入显式 CemaNeige（CN）融雪模块能有效纠正出口时相偏差。
3. **R2 (Internal Parameter Compensation)**: 当无雪模型强行拟合高积雪流域时，优化器会扭曲基质张力容量（$U_M$）、壤中流出流系数（$K_I$）、自由水消退系数（$C_I$）等物理参数进行时间补偿。
4. **R3 (Controlled Synthetic Truth Recovery)**: 在受控合成实验（Synthetic Truth）中，验证了模型结构缺陷（如缺失雪模块）无法通过纯参数优化恢复真实内部物理状态（产流、壤中流、地下水分配）。
5. **R4 (Physical State & Soil Moisture Timing vs. Reference)**: 对比 ERA5-Land 100cm 土壤水（$\text{SM}_{100}$）与 Snow-17 SWE 观测，验证了显式物理融雪模块能真实还原春季土壤水分补给脉冲，而通用温度滞后（TGD/TGD2）仅能拟合径流而无法还原真实土壤水动态。
6. **R5 (Cross-Model Structural Replication)**: 在三类异构水文宿主架构（XAJ 新安江、GR4J、SIMHYD）上跨模型验证，证明时相偏差纠正规律具有普适性。

---

## 2. 目录架构 (Directory Structure)

```text
project/hydrodiag/
├── README.md                            # [本文件] 论文实验总览与复现指南
├── models/                              # PyTorch 向量化水文模型与算子内核
│   ├── core/                            # XAJ, GR4J, SIMHYD, HBV, CemaNeige 步进算子
│   └── hydrology_model.py               # 模型统一封装与 Torch Compile 接口
├── training/                            # 模型训练与率定入口
│   ├── dpl/                             # 可微参数学习训练循环 (MLP Parameterizer)
│   └── ic/                              # 独立率定 (CMA-ES / 启发式)
├── results/                             # 实验与基准指标结果源数据
│   ├── dpl_camels_531_lite_v2/          # 基础 dPL 训练结果
│   ├── dpl_camels_531_lite_v3/          # v3 演进实验
│   ├── r3_misspec_analysis_v1/          # R3 受控合成实验基准
│   ├── r4_phase1_soil_official/         # R4 土壤水状态分析基准
│   └── reviewer2_robustness/            # 审稿人鲁棒性验证实验
├── manuscript/                          # 论文图表、表格与统计产出
│   ├── figure_manifests/
│   │   └── canonical_assets.json        # 论文全部图表资产注册表与源数据映射
│   ├── figures/                         # 生成的论文主图与附图 (Figure 1–9, Figure S1)
│   ├── stats/tables/                    # 论文主表与附表 (Table 1–2, Table S1–S2; CSV, MD, LaTeX)
│   ├── analysis/                        # 各章节 (R1–R5) 阶段性分析产物
│   └── scripts/                         # 论文图表与统计分析脚本
│       ├── r0/                          # Figure 1: 流域空间与降雪特征分布图
│       ├── r1/                          # Figure 2: 出口径流时相与积雪梯度响应
│       ├── r2/                          # Figure 3 & 4: 参数扭曲与内部状态代偿
│       ├── r3/                          # Figure 5 & 6: 受控合成真相恢复实验
│       ├── r4/                          # Figure 7 & 8, Figure S1: 土壤水与外部参考对比
│       ├── r5/                          # Figure 9: 三宿主模型跨结构泛化验证
│       ├── shared/                      # Table 1, Table 2, Table S1, Table S2 & 样式库
│       └── supplement/                  # Figure S3, S4, S5 附图分析脚本
└── configs/                             # 实验超参数与数据配置
```

---

## 3. 环境与数据准备 (Environment & Data Setup)

### 3.1 Python 环境与依赖
实验推荐在 Linux / WSL 环境下运行（要求 PyTorch >= 2.0 并支持 CUDA / Inductor Compile）：

```bash
# 激活项目虚拟环境
source /home/jingxin/code/dmg-research/.venv/bin/activate

# 验证 PyTorch 与 CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3.2 数据规范
- **CAMELS-531 径流与气象数据**：`data/camels_dataset`（包含降水 $P$、气温 $T$、潜在蒸发 $PET$、观测流量 $Q_{obs}$ 及 35 维属性）。
- **时间段划分**：
  - **预热期 (Warm-up)**：`1980-10-01` 至 `1981-09-30` (365天)。
  - **率定期 (Calibration / Training)**：`1981-10-01` 至 `1995-09-30` (14 年)。
  - **验证期 (Evaluation / Test)**：`1995-10-01` 至 `2010-09-30` (15 年)。
- **单位转换**：$ft^3/s \to mm/day$ 采用 CAMELS 标准公式：
  $$Q_{mm/day} = Q_{cfs} \times \frac{0.0283168 \times 86400 \times 10^3}{\text{Area}_{km^2} \times 10^6}$$

---

## 4. 论文全部图表一键复现指南 (Figure & Table Reproduction)

所有脚本均从仓库根目录执行，环境会自动解析路径。

### 4.1 主图复现 (Main Figures 1–9)

| 编号 | 描述 | 复现执行命令 | 产出路径 |
| :--- | :--- | :--- | :--- |
| **Figure 1** | 研究区域 531 流域降雪比例与地形空间分布 | `python project/hydrodiag/manuscript/scripts/r0/plot_r0_figure1.py` | `manuscript/figures/Figure1_final.png` |
| **Figure 2** | Base / CN / TGD 出口时相误差与积雪梯度响应 | `python project/hydrodiag/manuscript/scripts/r1/plot_r1_figure2.py` | `manuscript/figures/Figure2_R1_final.png` |
| **Figure 3** | 无雪模型参数扭曲与内部蓄水量时间代偿机制 | `python project/hydrodiag/manuscript/scripts/r2/plot_r2_figure3_final.py` | `manuscript/figures/Figure3_R2_final.png` |
| **Figure 4** | 15 个物理参数沿积雪梯度的系统性漂移分析 | `python project/hydrodiag/manuscript/scripts/r2/plot_r2_figure4_canonical.py` | `manuscript/figures/Figure4_R2_final.png` |
| **Figure 5** | 受控合成真相实验：出口径流与时相指标恢复 | `python project/hydrodiag/manuscript/scripts/r3/plot_figure5.py` | `manuscript/figures/Figure5_R3_final.png` |
| **Figure 6** | 受控合成真相实验：内部过程流与参数真实误差恢复 | `python project/hydrodiag/manuscript/scripts/r3/plot_figure6.py` | `manuscript/figures/Figure6_R3_final.png` |
| **Figure 7** | 土壤水动态一致性：两阶段过程相关性对比 | `python project/hydrodiag/manuscript/scripts/r4/plot_r4_figure7.py` | `manuscript/figures/figure7_r4_soil_consistency.png` |
| **Figure 8** | 春季土壤水回补脉冲时效性（对比 ERA5-Land $\text{SM}_{100}$） | `python project/hydrodiag/manuscript/scripts/r4/plot_r4_figure8.py` | `manuscript/figures/figure8_r4_soil_timing.png` |
| **Figure 9** | 跨模型泛化复现：XAJ, GR4J, SIMHYD 融雪纠偏一致性 | `python project/hydrodiag/manuscript/scripts/r5/plot_r5_figure9.py` | `manuscript/figures/Figure9_R5_cross_model_replication.png` |

### 4.2 主表与附表复现 (Tables 1–2, S1–S2)

所有表格脚本会同时生成 **CSV**、**Markdown**、**LaTeX** 三种格式并保存在 `project/hydrodiag/manuscript/stats/tables/`：

```bash
# 生成 Table 1: 结构配置与参数维度说明
python project/hydrodiag/manuscript/scripts/shared/generate_table1_structural_configurations.py

# 生成 Table 2: 受控实验结构缺损恢复率与泛化代偿比
python project/hydrodiag/manuscript/scripts/shared/generate_table2_controlled_recovery.py

# 生成 Table S1: 全部水文模型与模块物理参数边界范围
python project/hydrodiag/manuscript/scripts/shared/generate_table_s1_parameter_bounds.py

# 生成 Table S2: 阈值敏感性与分母定义稳健性检验
python project/hydrodiag/manuscript/scripts/shared/generate_table_s2_sensitivity.py
```

### 4.3 补充材料附图复现 (Supplementary Figures)

```bash
# Figure S1: 多流域土壤水时相验证
python project/hydrodiag/manuscript/scripts/r4/plot_r4_figure_s1_multibasin.py

# Figure S3: 替代生成场稳健性分析
python project/hydrodiag/manuscript/scripts/supplement/plot_alt_generating_field_robustness.py

# Figure S4: TGD 响应敏感性分析
python project/hydrodiag/manuscript/scripts/supplement/plot_tgd_response_sensitivity.py

# Figure S5: HUC2 LORO 区域留一交叉验证
python project/hydrodiag/manuscript/scripts/supplement/plot_huc2_loro_robustness.py
```

---

## 5. 核心模型结构与物理配置 (Model Invariants)

| 模型代号 | 基础宿主模型 | 附加模块 / 机制 | 自由参数数 | 物理作用 |
| :--- | :--- | :--- | :---: | :--- |
| **XAJ-Base** | 新安江三层蒸发模型 | 无雪模块 | 15 | 基准无雪模型，用于识别无雪条件下的出口与内部代偿 |
| **XAJ-CN** | 新安江三层蒸发模型 | CemaNeige (2 参数) | 17 | 显式度日融雪 + 热量储量模块，纠正积雪滞后 |
| **XAJ-TGD2** | 新安江三层蒸发模型 | Temperature Gated Delay | 17 | 仅通过气温门控滞后降水，作为通用温度延迟控制组 |
| **GR4J-Base / CN** | 法国经典 4 参数模型 | ± CemaNeige (2 参数) | 4 / 6 | 验证集总式非线性蓄水池模型上的融雪响应 |
| **SIMHYD-Base / CN** | 澳大利亚 SIMHYD 模型 | ± CemaNeige (2 参数) | 10 / 12 | 验证含不透水面与入渗超产流机制模型上的融雪响应 |

---

## 6. 实验复跑与训练指南 (Running New Experiments)

如需重新率定或训练新模型：

### 6.1 dPL 端到端训练
```bash
python project/hydrodiag/training/dpl/train_dpl.py \
    --model XAJ_CN \
    --epochs 100 \
    --lr 0.005 \
    --device cuda
```

### 6.2 独立率定 (IC CMA-ES)
```bash
python project/hydrodiag/ablation/run_ic_training.py \
    --model XAJ_CN \
    --starts 10 \
    --generations 300 \
    --device cuda
```

---

## 7. 资产清单与审计规范 (Asset Manifests & Audit)

论文所有图表的源数据依赖与生成脚本均严格受控于：
`project/hydrodiag/manuscript/figure_manifests/canonical_assets.json`

在修改或重跑图表脚本后，可运行资产一致性校验：
```bash
python project/hydrodiag/manuscript/scripts/shared/build_results_freeze_audit.py
```
该命令会自动扫描所有数据表与图表产物，确保无缺失或断链。
