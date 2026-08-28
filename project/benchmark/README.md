# 36-Model Hydrological Benchmark & Differentiable Parameter Learning (dPL)

> **Agent Quick-Start & Technical Reference Guide**
> 
> 本文档旨在为后续接手的 AI Agent 和研究人员提供关于本项目架构、数据流、36 水文模型引擎、GPU 向量化 Batched CMA-ES 率定、验证集评估、以及可微参数学习 (dPL) 模块的全面技术说明与最终执行状态报告。

---

## 1. 当前基准最终执行状态与结果 (Final Benchmark Status)

在 531 个 CAMELS-US 大样本流域上，已完成全部 **36 个概念水文模型** 在两大范式下的基准率定与评估：

### 1.1 独立率定 (IC Batched CMA-ES) 最终结果
- **执行配置**：10 次随机初始化 (Starts)，300 代进化迭代 (Generations)，531 流域并行向量化求解，FP64 精度 + Torch Compile 算子加速。
- **预热策略**：1980–1981 单水文年气象强迫循环预热 5 次（1825 天）。
- **指标表现**：
  - **36 模型训练期 KGE 中位数**：**0.7299**（MARRMoT 基准: 0.6894）。
  - **36 模型测试期 (验证集) KGE 中位数**：**0.6113**（MARRMoT 基准: 0.5785，相对 MARRMoT 提升 **+0.0328**）。
  - **收敛率**：36 / 36 (100% 成功，0 失败)。

### 1.2 可微参数学习 (dPL Framework) 最终结果
- **网络架构**：`CatchmentParameterizer` MLP（27 维流域物理属性 $\to$ 水文模型物理参数）。
- **优化与事务**：`FiniteOptimizerTransaction` + Adam 优化器，带显存感知切块与自动梯度恢复。
- **协议对齐**：支持 DPL-aligned 1980–1995 训练期与 1995–2010 验证期协议，支持流式 KGE（`StreamingKGE`）无中间状态内存膨胀。

---

## 2. 核心执行与操作指南 (Execution Guide)

所有脚本均从仓库根目录执行，需保证 `PYTHONPATH` 包含根目录及 `project/benchmark`：

```bash
# 激活环境与设置 PYTHONPATH
source /home/jingxin/code/dmg-research/.venv/bin/activate
export PYTHONPATH="/home/jingxin/code/dmg-research:/home/jingxin/code/dmg-research/project/benchmark:/home/jingxin/code/dmg-research/project/benchmark/src:${PYTHONPATH:-}"
```

### 2.1 运行单个模型的 CMA-ES 优化
```bash
python project/benchmark/scripts/run_36model_benchmark.py \
    --model simhyd \
    --run-id full300_production \
    --config project/benchmark/configs/full_run_10starts_300gen_warm1980_1981x5.yaml \
    --device cuda
```

### 2.2 运行全部 36 个模型 (带显存自动降级与断点续跑)
`run_36model_benchmark.py` 内置显存感知切块逻辑（531 $\to$ 256 $\to$ 128 $\to$ 64），遇到显存不足（如带 UH 卷积的 `flexis` 或 `gr4j`）会自动降低 chunk 尺寸重试，确保流水线不挂断。
```bash
python project/benchmark/scripts/run_36model_benchmark.py \
    --model all \
    --run-id full300_production \
    --config project/benchmark/configs/full_run_10starts_300gen_warm1980_1981x5.yaml \
    --device cuda
```

### 2.3 验证集 (测试期) 精度评估
```bash
python project/benchmark/scripts/evaluate_benchmark_metrics.py \
    --checkpoint-root project/benchmark/checkpoints/full300_final_36models \
    --config project/benchmark/configs/full_run_10starts_300gen_warm1980_1981x5.yaml \
    --output-dir project/benchmark/results/full300_evaluated
```

### 2.4 对齐 1995–2010 评估 (IC-Aligned Gen-300)
```bash
python project/benchmark/scripts/evaluate_ic_aligned_gen300.py \
    --ckpt-root project/benchmark/checkpoints/full300_final_36models \
    --out project/benchmark/results/all36_ic_gen300_aligned_final
```

### 2.5 运行 dPL 端到端训练
```bash
python project/benchmark/dpl/train_dpl.py \
    --model simhyd \
    --epochs 50 \
    --lr 0.001 \
    --device cuda
```

---

## 3. 36 个水文模型全景清单 (Complete 36 Models Inventory)

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
| 28 | `simhyd` | 7 | 7 参数、双状态、无 Gamma-UH 验证版 |
| 29 | `smar` | 9 | SMAR 9 参数产汇流模型 |
| 30 | `susannah1` | 5 | Susannah 产流模型 1 |
| 31 | `susannah2` | 6 | Susannah 产流模型 2 |
| 32 | `tank` | 8 | 4 级串联 Tank 箱式模型 |
| 33 | `tcm` | 6 | TCM 模型 |
| 34 | `topmodel` | 8 | TOPMODEL (地形指数指数衰减透水性) |
| 35 | `us1` | 6 | US1 水文模型 |
| 36 | `wetland` | 5 | 严格质量守恒湿地模型 |

---

## 4. 目录与模块映射 (Directory Layout)

```text
project/benchmark/
├── README.md                            # [本文件] 基准执行状态与快速上手指南
├── src/                                 # CMA-ES 优化与数据加载核心库
│   ├── batched_cmaes.py                 # GPU 向量化 Batched CMA-ES 求解器 (FP64 鲁棒协方差更新)
│   ├── checkpointing.py                 # 原子化 Checkpoint 读写与版本恢复
│   ├── checkpoint_guard.py              # Canonical Checkpoint 守卫与校验 (防低代数降级)
│   ├── data_selection.py                # CAMELS 数据集切分与预热数据构建
│   ├── model_registry.py                # 36 水文模型注册表与构造工厂
│   ├── objective.py                     # KGE 目标函数与流式统计量计算 (StreamingKGE)
│   ├── production_config.py             # Production YAML 配置校验器
│   └── streaming_evaluator.py           # 内存受限流式适应度评估器
├── configs/                             # 生产环境配置文件
│   ├── full_run_10starts_300gen_warm1980_1981x5.yaml
│   └── full_run_10starts_300gen_dpl_aligned_1980_1995.yaml
├── scripts/                             # 生产执行脚本
│   ├── run_36model_benchmark.py         # 36 模型主训练脚本
│   ├── evaluate_benchmark_metrics.py    # 验证集评估脚本
│   ├── evaluate_ic_aligned_gen300.py    # 对齐 1995–2010 评估脚本
│   ├── freeze_model_version.py          # 模型版本与代码哈希冻结工具
│   └── run_parallel_models.py           # 多卡多模型并行运行器
├── dpl/                                 # 可微参数学习模块
│   ├── attributes.py                    # 27 维流域物理属性提取与标准化
│   ├── nn_parameterizer.py              # MLP 参数预测网络
│   ├── train_dpl.py                     # dPL 端到端训练循环
│   └── tests/                           # 单元测试 (含断点续跑、流式适应度检验)
└── results/                             # 指标评估与基准落盘产物
```

---

## 5. 核心设计与防退化原则 (Invariants)

1. **统一模型算子契约**：所有 36 水文模型均继承自 `HydrologyModel`，支持 `backend="compile"`，输入张量使用 FP64 保证数值稳定性。
2. **严禁信息泄露**：率定期最佳参数选择（`best_of_10`）必须且仅能依据训练期 KGE，严禁在参数选择阶段使用测试期观测流量。
3. **原子落盘与守卫**：Checkpoint 保存时必须写入 `DONE` 标志并校验 531 流域完整性，`validate_canonical_checkpoint` 会强制拦截不完整或代数不匹配的检查点。
4. **流式低显存评估**：采用 `StreamingKGEState` 维护一阶、二阶充分统计量，避免在大时间跨度与大种群下保存整段预测时序导致显存 OOM。
