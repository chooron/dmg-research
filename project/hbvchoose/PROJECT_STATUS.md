# HBV Formula MoE 项目执行情况详细报告

> **项目路径**: `/home/jingxin/code/dmg-research/project/hbvchoose`
> **最后修改时间**: 2026年6月30日 - 7月2日
> **当前运行状态**: 无运行中进程

---

## 一、项目概要

本项目旨在构建一个 **StaticFormulaRouter** —— 一个基于静态流域属性（CAMELS 数据集）为 HBV 水文模型的四个核心过程（融雪 snow、补给 recharge、实际蒸散发 AET、汇流响应 response）自动选择最优数学公式的神经网络路由器。通过可微分训练实现流域特异化的模型结构选择。

### 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 公式池 | `model/formula_pool.py` | 候选公式查找与适配器 |
| 路由器 | `model/static_formula_router.py` | 逐节点线性投影头 + 默认 anchor bias |
| 集成模型 | `model/hbv_static_router.py` | 路由器与 HBV 模拟集成 |
| 参数映射 | `model/parameter_mapping.py` | 归一化 [0,1] 到物理参数映射 |
| 参考实现 | `model/hbv_static.py` | 原始 HBV 参考实现 |
| 公式实现 | `model/flux/{snow,recharge,aet,response}.py` | 各过程候选公式实现 |

---

## 二、验证结果总览

| 验证项 | 状态 | 测试规模 | 关键结论 |
|--------|------|----------|----------|
| 公式组合前向稳定性基准测试 | ✅ 通过 | 54组合 × 5场景 = 270例 | 0失败, 18警告 |
| 公式组合校准烟雾测试 | ✅ 通过 | 270例 | 0失败, 0 NaN/Inf |
| 公式梯度审计 | ⚠️ 警告 | 12公式 × 4节点 | 虽均标记为 safe_for_training，但全部存在大梯度 |
| 公式尺度审计 | ⚠️ 警告 | 4节点全量比较 | recharge(12严重不匹配), aet(4严重不匹配) |
| 默认HBV等价性检验 | ❌ PENDING | 三路比较 | eager vs formula 散度达0.058 / 86mm |
| 路由器烟雾测试(合成数据) | ⚠️ 通过 | 16盆地, 100步 | loss未下降(6.7e-5→0.046)，选择退化 |
| CAMELS pilot(未标定) | ⚠️ 通过 | 4盆地, 5步 | 路由器仅选S0_R0_E0_Q0，无探索 |
| CAMELS校准pilot | ❌ 失败 | 4盆地 | 校准NSE为NaN, 路由器0步, 选择结果为空 |

---

## 三、问题详细描述

### 问题 1：默认 HBV 等价性 bug（PENDING，严重）

**影响范围**: 所有基于 `HbvFormulaStatic` 的实验

**现象**:
- 三路对比（compiled vs eager vs formula）中，compiled 与 eager 完全一致（max abs diff = 0），证明参考实现自身内部一致
- 但 eager/compiled 与 formula 模式之间存在显著差异：
  - **流量的最大绝对差异**: 0.058 mm/d
  - **土壤湿度（SM）首次散度**: t=20 (evaluation phase), 差异达 **86.01 mm**（原始 244.75 vs formula 158.74）

**实验配置**:
- dtype=float32, warm_up=20, length=60
- HbvFormulaStatic 以 compat_mode=True, apply_routing=True 运行
- 相同的归一化参数(q=0.5)、相同强迫数据、相同初始状态

**可能原因**:
1. `HbvFormulaStatic` 中公式调用的数值实现与原始 `_hbv_step` kernel 存在差异
2. compat_mode 下的参数重映射或状态更新逻辑有误
3. 浮点精度累积误差在迭代中放大

**影响**: 此 bug 意味着所有基于公式版 HBV 的校准和路由选择结果可能与真实 HBV 行为存在系统性偏差，根本上影响实验的有效性。

---

### 问题 2：校准 CAMELS pilot 完全失败（阻塞性）

**现象**:
- Stage 1（默认 HBV 校准）：100 步训练后，4 个盆地中有 1 个（12043000）的全部指标为 NaN
  - basin 14138900: NSE 0.474 → 0.511 (正常)
  - basin 12147600: NSE 0.242 → 0.161 (退化)
  - basin 14138870: NSE 0.241 → 0.663 (正常)
  - basin 12043000: **所有指标 NaN**（NSE, KGE, RMSE 全为 NaN）
- Stage 2（路由器训练）：training_steps.csv 仅含表头，**0 步完成**（说明训练在开始前或第一步就终止了）
- selection_summary.csv 仅含表头，无任何选择数据
- failures.csv 仅含表头，未记录任何失败原因

**可能原因**:
1. basin 12043000 的数据质量问题（缺测、全零、极端值）导致校准参数发散
2. 某盆地的 NaN 通过损失函数传播导致整个训练循环提前终止
3. 脚本缺少单盆地异常处理的保护逻辑

**影响**: 校准pilot 没有产生可用的选择结果，无法评估路由器在标定场景下的表现。

---

### 问题 3：路由器的学习能力严重不足

**3.1 烟雾测试（合成数据）**
- 16 个合成盆地，100 步训练
- **Loss 不降反升**: 初始 0.000067 → 最终 0.046070（基本回到了默认bias水平）
- 最小 loss（0.000066）出现在训练早期，之后持续上升
- 最终默认选择率高达 **84.38%**（1600 次选择中 1172 次选了 S0_R0_E0_Q0）
- 仅探索了 4 种组合：S0_R0_E0_Q0 (1172), S0_R0_E0_Q2 (417), S5_R0_E0_Q0 (6), S5_R0_E0_Q2 (5)
- 各节点的熵值较高（0.84-0.99），说明路由器没有形成明确的偏好，但也没有脱离默认选择

**3.2 CAMELS pilot（非标定）**
- 4 个真实盆地，仅 5 步训练
- 路由器 **100% 选择了默认公式 S0_R0_E0_Q0**（20 次选择，无一次探索）
- 路由器 NSE（-48.1）略差于默认 HBV（-47.0），ΔNSE = -1.14
- 4 个盆地全部标记为 WARNING（NSE worse than default）

**3.3 校准 CAMELS pilot**
- 路由器训练未产生任何有效步骤（见问题 2）

**核心问题分析**:
1. **Anchor bias 过大**: 默认 bias=2.0（烟雾测试），虽然验证"默认偏置生效"，但也意味着路由器需要克服较大的初始偏置才能做出不同选择，导致探索不足
2. **训练步数不足**: CAMELS pilot 仅 5 步，校准 pilot 虽设 50 步但实际 0 步完成，远不足以让路由器学习
3. **合成数据设计问题**: 烟雾测试中 loss 先降后升，暗示合成数据的标签（最优公式分配）可能与路由器的学习目标不匹配，或者优化器设置（学习率、温度参数）不当
4. **梯度噪声**: 所有公式均检测到大梯度（见问题 5），可能干扰路由器的梯度信号

---

### 问题 4：补水（Recharge）节点尺度严重不匹配

**现象**:
- 在 recharge 节点的 24 个比较场景中，检测到 **12 个严重尺度不匹配**（severe mismatch）
- 最大 log10 比值达 **4.4226**（意味着不同公式输出可差 ~26,000 倍）
- 中位 log10 比值为 1.0235（相差约 10 倍）
- 相比之下，snow (max 0.67)、response (max 0.81)、routing (max 0.68) 的尺度差异远小于 recharge

**受影响的公式**:
- `beta_recharge`, `linear_recharge`, `strong_nonlinear_recharge`, `weak_nonlinear_recharge`, `saturation_threshold_recharge` 之间存在系统性输出量级差异

**当前处理**:
- 梯度审计报告已建议 recharge 节点使用 **hard routing only**（已在代码中实现为 straight-through 模式）
- AET 节点也有 4 个严重不匹配，目前代码中未做特殊处理

**影响**:
- 若未来尝试对 recharge/aet 启用 softmax 混合（dense MoE），输出量级差异将导致混合结果被大尺度公式主导
- 当前 hard routing 规避了此问题，但限制了模型的表达能力

---

### 问题 5：所有公式存在大梯度

**现象**:
- 梯度审计覆盖了 4 个节点共 12 个候选公式
- **无 NaN/Inf 梯度**（正向：梯度稳定性可接受）
- 但 **所有 12 个公式均被标记为大梯度**（LargeGrad=True）
- 典型数据：
  - snow S4: 1028 个大梯度记录（最多）
  - aet E4: 444 个大梯度记录
  - recharge R4: 288 个大梯度记录

**梯度统计**:

| 节点 | 记录数 | 大梯度数 | 最大绝对梯度 | 最大缩放梯度 |
|------|--------|----------|-------------|-------------|
| snow | 16,440 | 1,220 | 484.02 | 1e9 |
| recharge | 5,500 | 392 | 261.01 | 1e9 |
| aet | 5,500 | 434 | 20.86 | 1e9 |
| response | 13,930 | 250 | 353.55 | 1e9 |

**最大缩放梯度均为 1e9**：这是一个饱和/裁剪上界值，表明某些参数组合下梯度确实达到了极大的量级。

**影响**:
1. 训练时容易触发梯度爆炸，需要强制 gradient clipping
2. 大梯度可能掩盖路由器的选择信号，使梯度更新更多受数值波动而非公式优劣驱动
3. 某些公式（如 S4、R4）在特定参数区域可能数值不稳定

**当前处理**:
- 报告建议所有节点在训练时使用 gradient clipping 监控
- 实际训练脚本中是否已配置 gradient clipping 尚需确认

---

### 问题 6：AET 节点额外尺度问题

**现象**:
- AET 节点 20 个比较场景中检测到 **4 个严重尺度不匹配**
- 系统性地 `aet_hbv_default` 输出最大，`aet_power_law` 输出最小

**与 recharge 的区别**:
- 严重程度低于 recharge（最大 log10 比值 1.16 vs 4.42）
- 仍有 4 个 moderate 不匹配
- 梯度审计中 AET 的大梯度数量较少（434，但相对记录数比例最高 ~7.9%）

**影响**:
- 若 AET 节点启用 dense mixing，同样面临尺度不匹配问题
- 当前代码中 AET 未被设为 hard routing

---

## 四、项目结构完整性

```
hbvchoose/
├── model/                          # 核心模型代码 (15 .py + 1 .zip)
│   ├── __init__.py
│   ├── hbv_static.py               # 参考HBV实现
│   ├── hbv_formula_static.py       # 公式可替换HBV
│   ├── formula_pool.py             # 候选公式池
│   ├── static_formula_router.py    # 静态路由器核心
│   ├── hbv_static_router.py        # 集成模型
│   ├── parameter_mapping.py        # 参数映射
│   ├── models.zip                  # 模型压缩包
│   └── flux/
│       ├── formula_registry.py     # 公式注册表
│       ├── snow.py                 # 融雪公式
│       ├── recharge.py             # 补给公式
│       ├── aet.py                  # 蒸散发公式
│       ├── response.py             # 汇流响应公式
│       ├── routing.py              # 河道演进公式
│       ├── parameter_ranges.py     # 参数范围定义
│       └── _utils.py               # 工具函数
├── scripts/                        # 实验脚本 (10 .py)
│   ├── enumerate_formula_combinations.py
│   ├── audit_formula_scales.py
│   ├── audit_formula_gradients.py
│   ├── benchmark_formula_combinations.py
│   ├── calibrate_formula_combinations_smoke.py
│   ├── train_static_router_smoke.py
│   ├── check_default_hbv_equivalence.py
│   ├── debug_single_step_equivalence.py
│   ├── train_static_router_camels_pilot.py
│   └── train_static_router_camels_calibrated_pilot.py
├── tests/                          # 单元测试 (8 .py)
│   ├── test_static_formula_router.py
│   ├── test_formula_combination_enumeration.py
│   ├── test_formula_registry_pool.py
│   ├── test_flux_formula_safety.py
│   ├── test_calibration_smoke_script.py
│   ├── test_static_router_camels_pilot.py
│   └── test_static_router_camels_calibrated_pilot.py
└── validation_results/             # 验证结果 (78 .csv + 21 .md)
    ├── formula_combination_benchmark/
    ├── formula_calibration_smoke/
    ├── formula_gradient_audit/
    ├── formula_scale_audit/
    ├── static_router_smoke/
    ├── static_router_camels_pilot/
    └── static_router_camels_calibrated_pilot/
```

### 外部数据依赖
- `/home/jingxin/code/dmg-research/data/camels_dataset/` — CAMELS 流域数据集
- `/home/jingxin/code/dmg-research/data/gage_id.npy` — 测站 ID 列表

### 缺失项
- 无 README 或项目文档
- 无配置文件（yaml/json/toml）
- 无运行日志（无 .log/.out/.err 文件）
- 无 CI/CD 配置

---

## 五、建议的后续步骤（按优先级排序）

### P0 — 阻塞性问题
1. **修复默认 HBV 等价性 bug**
   - 在 `debug_single_step_equivalence.py` 中逐步对比 eager 与 formula 模式下每个时间步每个状态变量的差异
   - 重点排查 SM（土壤湿度）在 t=20 的突变，可能是 recharge 或 aet 公式中状态更新逻辑不一致
   - 验证 compat_mode 参数映射是否正确
   
2. **修复校准 pilot 的 NaN 问题**
   - 单独加载 basin 12043000 的数据，检查是否存在缺测、零值强迫、极端值等问题
   - 在校准脚本中添加 per-basin NaN 检测和跳过机制，避免单个盆地拖垮全局训练
   - 确认 calibration 的损失计算在向量化时是否正确处理了观测值为零的情况

### P1 — 关键改进
3. **调整路由器训练策略**
   - 降低 anchor bias（从 2.0 降至 0.5~1.0），增加初始探索概率
   - 增加训练步数（建议至少 200~500 步）
   - 尝试 temperature annealing（温度从高到低退火）
   - 重新设计合成数据标签，确保路由器有明确的"非默认"学习目标

4. **梯度裁剪确认**
   - 检查所有训练脚本是否已配置 gradient clipping（建议 max_norm=1.0~10.0）
   - 对 S4（融雪）和 R4（补给）公式考虑更严格的裁剪

5. **扩大 CAMELS pilot**
   - 增加盆地数量（从 4 → 50+）
   - 增加评估期时长（从 30 → 365 天）
   - 修复以上问题后再运行

### P2 — 优化改进
6. **recharge 硬路由确认与 aet 尺度处理**
   - 确认 recharge 节点的 hard routing 实现正确且覆盖所有训练脚本
   - 评估 aet 节点是否也需要改为 hard routing

7. **补充文档**
   - 编写项目 README
   - 添加运行脚本的使用说明和参数说明
   - 记录各实验的完整运行命令
