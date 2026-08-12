# 36-Model Hydrological Benchmark & Differentiable Parameter Learning (dPL)

> **Agent Quick-Start & Technical Reference Guide**
> 
> 本文档旨在为后续接手的 AI Agent 和研究人员提供关于本项目架构、数据流、水文模型引擎、CMA-ES 并行率定、验证集评估以及可微参数学习 (dPL) 模块的全面技术说明。

---

## 📌 项目概述 (Executive Summary)

本项目是一个基于 **PyTorch CUDA 向量化加速** 的大规模水文模型基准与可微参数学习研究平台。它将经典水文模型框架（MARRMoT / MOPEX 概念水文模型）重构为纯张量算子，在美国 CAMELS 数据集（531 个典型流域）上实现了高效率定与评估。

项目包含两大核心率定/学习模式：
1. **独立流域 GPU 向量化 CMA-ES 求解 (Framework 1)**：在 531 个流域上并行运行 Batched CMA-ES，10 次随机初始化，300 代进化迭代，自动进行显存感知切块（Chunking）与 Torch Compile 编译加速。
2. **端到端可微参数学习 dPL (Framework 2)**：利用神经网络 $g_\phi(\text{Attributes}) \to \boldsymbol{\theta}$ 学习流域物理属性到水文模型参数的映射，并利用 PyTorch 计算图全自动反向传播求解。

---

## 📁 目录与模块映射 (Project Directory & Sitemap)

根目录：`/home/jingxin/code/dmg-research/project/benchmark/`

```
.
├── README.md                            # [本文件] Agent 快速理解与全面指南
├── dmotpy/                              # 冻结的 36 个水文模型与 PyTorch 张量算子引擎
│   ├── framework/                       # 模型基类 (HydrologyModel) 与计算图抽象
│   ├── models/
│   │   ├── core/                        # 36 个水文模型核心方程 (包含 7 参数 SIMHYD & 质量守恒 Wetland)
│   │   └── unithydro/                   # 单元过程线 routing (UH1/UH2 卷积路由)
│   ├── solver/                          # 常微分方程 (ODE) 差分求解器 (如 explicit_euler)
│   └── utils/                           # 物理参数范围与转换工具
├── src/                                 # CMA-ES 优化与数据加载核心库
│   ├── batched_cmaes.py                 # GPU 向量化 Batched CMA-ES 求解器
│   ├── checkpointing.py                 # 原子化 Checkpoint 读写与版本恢复
│   ├── data_selection.py                # CAMELS 数据集切分 (Warmup 5x 重复 + Train 10yr + Test 10yr)
│   ├── model_registry.py                # 36 模型注册表 (参数维度、模型构造器工厂)
│   ├── objective.py                     # KGE (Kling-Gupta Efficiency) 目标函数 (Torch Compile 优化)
│   └── production_config.py             # 验证 production YAML 配置的合法性
├── configs/                             # 冻结的生产环境配置文件
│   └── full_run_10starts_300gen_warm1980_1981x5.yaml  # 10 starts, 300 generations, 1980-1981 5x warmup
├── scripts/                             # 整合统一的生产执行脚本
│   ├── run_36model_benchmark.py         # 36 模型主训练脚本 (包含显存感知降级与断点续跑)
│   ├── evaluate_benchmark_metrics.py    # 训练期与测试期 (验证集) KGE 精度评估脚本
│   └── run_full_benchmark.sh            # SSH 远程一键后台部署脚本
├── results/                             # 验证集评估指标落盘目录
│   ├── full300_kge_overall.json         # 总体 KGE 统计与 MARRMoT 对比
│   ├── full300_kge_model_summary.csv    # 36 模型中位数 KGE 及胜率表
│   └── full300_kge_by_basin.csv         # 531 流域 × 36 模型逐流域 KGE 明细
└── dpl/                                 # 可微参数学习 (Differentiable Parameter Learning) 模块
    ├── __init__.py
    ├── attributes.py                    # 531 流域 27 维物理属性提取与标准化
    ├── nn_parameterizer.py              # 神经网络映射器 (Catchment Attributes -> Hydrological Parameters)
    └── train_dpl.py                     # 端到端 PyTorch Autograd 可微训练循环
```

---

## 📊 数据集与时间轴规范 (Data & Experiment Protocol)

### 1. 数据来源与路径
- 原始数据存放路径：`/home/jingxin/code/dmg-research/data/`
- 流域列表：`data/531sub_id.txt` (531 个 CAMELS 流域 ID)
- 气象与流量数据 Pickle：`data/camels_dataset`
- 面积与索引映射：`data/gage_id.npy`

### 2. 单位转换公式 (Streamflow Unit Conversion)
气象观测中，径流深度转换为 $mm/day$ 的公式为：
$$Q_{mm/day} = Q_{cfs} \times \frac{0.0283168 \times 86400 \times 10^3}{\text{Area}_{km^2} \times 10^6}$$

### 3. 时间段划分与 Pre-warmup 策略
为消除土壤水与蒸发蓄水池初始状态影响，配置了预热 (Warmup) 重复策略：
- **预热期 (Warmup)**：使用 1980-10-01 至 1981-09-30 (365天) 气象数据连续**重复 5 次**（共 1825 天）。
- **训练期 (Training Period)**：1989-01-01 至 1998-12-31 (10 年，共 3652 天)。
- **测试/验证期 (Test Period)**：1999-01-01 至 2008-12-31 (10 年，共 3652 天)。

---

## 🌊 36 个水文模型全景清单 (Complete 36 Models Inventory)

下表列出本项目中冻结支持的全部 36 个概念水文模型及其参数维度：

| 序号 | 模型名称 (`model_name`) | 参数维度 ($D_{param}$) | 汇流 / 特殊机制说明 |
| :---: | :--- | :---: | :--- |
| 1 | `alpine1` | 5 | 融雪与基流双蓄水池 |
| 2 | `alpine2` | 6 | 冰雪融水与地下水补给 |
| 3 | `australia` | 8 | 澳大利亚干旱区水文模型 |
| 4 | `collie1` | 4 | 单蓄水池蒸发与壤中流 |
| 5 | `collie2` | 5 | 双蓄水池渗漏模型 |
| 6 | `collie3` | 6 | 包含深层地下水补给 |
| 7 | `flexb` | 8 | FLEX-B 架构 (包含 Fast/Slow 响应) |
| 8 | `flexi` | 9 | FLEX-I 架构 (包含坡面路由) |
| 9 | `flexis` | 10 | FLEX-IS (带 UH 过程线，显存要求较高) |
| 10 | `gr4j` | 4 | 法国经典 GR4J 模型 (带 2 个 UH 卷积线) |
| 11 | `gsfb` | 9 | 包含饱水/不饱水带演化 |
| 12 | `hbv96` | 9 | 瑞典 HBV-96 经典 9 参数模型 |
| 13 | `hillslope` | 7 | 坡面径流与基流响应 |
| 14 | `hymod` | 5 | HYMOD 5 参数模型 (非线性蓄水容量分布) |
| 15 | `ihacres` | 4 | IHACRES 4 参数线性/非线性响应模型 |
| 16 | `lascam` | 10 | LASCAM 架构 |
| 17 | `modhydrolog` | 9 | MODHYDROLOG 9 参数模型 |
| 18 | `mopex1` | 7 | MOPEX 框架 Unit 1 |
| 19 | `mopex2` | 8 | MOPEX 框架 Unit 2 |
| 20 | `mopex3` | 8 | MOPEX 框架 Unit 3 |
| 21 | `mopex4` | 9 | MOPEX 框架 Unit 4 |
| 22 | `mopex5` | 9 | MOPEX 框架 Unit 5 |
| 23 | `newzealand1` | 5 | 新西兰水文模型 Unit 1 |
| 24 | `newzealand2` | 6 | 新西兰水文模型 Unit 2 |
| 25 | `penman` | 5 | Penman 产流模型 |
| 26 | `plateau` | 7 | 高原蒸发与基流响应 |
| 27 | `sacramento` | 10 | SAC-SMA 萨克拉门托模型 (10 参数简版) |
| 28 | **`simhyd`** | **7** | **最新替换：7 参数、双状态、无 Gamma-UH 本地验证版** |
| 29 | `smar` | 9 | SMAR 9 参数产汇流模型 |
| 30 | `susannah1` | 5 | Susannah 产流模型 1 |
| 31 | `susannah2` | 6 | Susannah 产流模型 2 |
| 32 | `tank` | 8 | 4 级串联 Tank 箱式模型 |
| 33 | `tcm` | 6 | TCM 模型 |
| 34 | `topmodel` | 8 | TOPMODEL (地形指数指数衰减透水性) |
| 35 | `us1` | 6 | US1 水文模型 |
| 36 | **`wetland`** | **5** | **最新替换：Candidate A 严格质量守恒湿地模型** |

---

## 🛠️ 操作指南：训练与评估 (Execution Guide for Agents)

所有命令需先设置工作目录与环境变量：

```bash
cd /home/jingxin/code/dmg-research/project/benchmark
export PYTHONPATH="/home/jingxin/code/dmg-research/project/benchmark:/home/jingxin/code/dmg-research/project/benchmark/src:${PYTHONPATH:-}"
```

### 1. 运行单个模型的 CMA-ES 优化
```bash
python3 scripts/run_36model_benchmark.py --model simhyd --run-id test_simhyd_20260730
```

### 2. 运行全部 36 个模型 (带显存自动降级与断点续跑)
`run_36model_benchmark.py` 内部内置了自动降级逻辑：默认尝试全批次 531 流域 `compile` 模式，若显存不足（如带 UH 的 `flexis` 或 `gr4j`），将自动降低 chunk 尺寸为 256/128/64 重新尝试，确保不挂断。
```bash
python3 scripts/run_36model_benchmark.py --model all --run-id full300_production
```

### 3. 运行验证集 (测试期) 精度评估
训练完成后，运行评估脚本。该脚本基于训练期得到的最佳参数，在 10 年测试期进行验证，输出模型 KGE 中位数、相对 MARRMoT 的胜率等：
```bash
python3 scripts/evaluate_benchmark_metrics.py \
  --checkpoint-root checkpoints/full300_production \
  --config configs/full_run_10starts_300gen_warm1980_1981x5.yaml \
  --output-dir results
```

### 4. 远程 SSH 服务器一键后台脚本
```bash
nohup bash scripts/run_full_benchmark.sh full300_production > benchmark.log 2>&1 &
```

---

## 🧠 核心扩展：可微参数学习 dPL (Differentiable Parameter Learning)

### 1. dPL 理论与公式
传统 CMA-ES 是针对每个流域独立求解参数 $\boldsymbol{\theta}_i \in \mathbb{R}^{D_{param}}$。而 dPL 是利用神经网络映射器 $g_\phi$ 拟合全局参数生成函数：
$$\boldsymbol{\theta}_i = g_\phi(\mathbf{a}_i)$$
其中 $\mathbf{a}_i \in \mathbb{R}^{27}$ 为流域物理属性。

正向传播与损失计算公式：
1. **参数预测**：$\boldsymbol{\theta}_i = \boldsymbol{\theta}_{min} + (\boldsymbol{\theta}_{max} - \boldsymbol{\theta}_{min}) \odot \sigma(\text{MLP}_\phi(\mathbf{a}_i^{norm}))$
2. **流量模拟**：$\hat{Q}_{i, t} = \text{HydroModel}(X_{i, t}; \boldsymbol{\theta}_i)$
3. **Loss 计算**：$\mathcal{L}(\phi) = \frac{1}{B} \sum_{i=1}^B \left( 1 - \text{KGE}(Q_{obs, i}, \hat{Q}_{i}) \right)$

### 2. dPL 属性矩阵构建 (`dpl/attributes.py`)
`CatchmentAttributeBuilder` 负责提取 CAMELS 数据集中的 27 维属性（包含地形 `elev/slope/area`、气候 `p_mean/pet_mean/aridity/seasonality`、土壤 `porosity/conductivity/clay/sand`、植被 `lai/gvf/forest_frac`）并完成 Z-score 规范化。

### 3. dPL 运行示例
```bash
python3 dpl/train_dpl.py --model simhyd --epochs 50 --lr 0.001 --device cuda
```

---

## 🔒 Agent 遵守的核心原则 (Invariants for Future Agents)

1. **不可破坏 Torch Compile 计算图**：水文模型中的条件分支或循环必须使用张量掩码 (Masking) 或 `torch.where` 实现，不可使用 Python 动态控制流，否则会导致 `torch.compile` 频繁 recompilation 或退化。
2. **严格参数冻结验证**：评估测试期 KGE 时，**绝对禁止**将测试期观测流量反馈给优化器或参数选择，必须严格遵循 `best_of_10_checkpoint_train_kge_only` 规则。
3. **原子落盘契约**：任何模型训练完成时，必须生成包含校验 JSON 的 `DONE` 标志文件（`checkpoints/<run_id>/<model>/DONE`），保证中断续跑逻辑安全生效。
4. **不占用显存死锁**：显存安全阈值设置为 `safety = 10.5 GiB`，发现超限须及时触发清空缓存 `torch.cuda.empty_cache()` 并降低 `chunk_size`。

---

## 📈 当前已落盘的基准精度结果 (Current Validation Benchmark)

目前存储在 [results/](file:///home/jingxin/code/dmg-research/project/benchmark/results/) 目录下的评估结果：
- **评估模型总数**：36 / 36 (100% 成功，0 失败)
- **训练集 KGE 中位数**：**0.7299** (MARRMoT 基准: 0.6894)
- **验证集 (测试期) KGE 中位数**：**0.6113** (MARRMoT 基准: 0.5785)
- **精度优势**：相比原始 MARRMoT，KGE 提升 **+0.0328**。
