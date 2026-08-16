# TRAINING_LOG — dpl_full_retrain_20260813(36 模型 dPL 健康重训)

> 本日志记录 `results/dpl_full_retrain_20260813` 这份数据的完整训练过程、规则与来源,
> 供后续 Chapter-4 及任何下游工作追溯。数据快照时间:2026-08-16(35/36 模型,最终集)。

---

## 1. 任务概述

在 CAMELS-531 流域上,对 dmotpy 36 个水文模型做**端到端可微参数学习(dPL)健康重训**:
用神经网络 `g_φ(attributes) → θ` 学习"流域属性 → 水文模型参数"的映射,
损失为可微 1−KGE,训练后冻结参数在验证期评估。

- 管线脚本:`project/benchmark/scripts/diagnostics/k_full_retrain.py`(arm=`auto100`)
- 并行调度:`project/benchmark/scripts/diagnostics/dpl_parallel_scheduler.sh`(6 worker 动态填补)
- 代码版本:git commit `b88f88d` / `94a432b`(master);远端运行目录 `/autodl-fs/data/dpl_run_20260814`

## 2. 运行环境(远端 AutoDL 节点)

| 项 | 值 |
| :--- | :--- |
| 节点 | `connect.westb.seetacloud.com`(Seetacloud/AutoDL,端口随重启变化) |
| GPU | NVIDIA RTX 3080 Ti 12 GB(6 进程并行时每进程约 544 MiB,总 ~3.3 GB) |
| Python | `/root/miniconda3/bin/python`(torch 2.8.0+cu128) |
| 数据盘 | `/autodl-fs/data/`(持久化;`/root/autodl-fs` 为符号链接) |
| 结果盘 | `/autodl-fs/data/dmg-research-results/dpl_full_retrain_20260813/` |
| 启动方式 | `PYTHONPATH=/autodl-fs/data/dpl_run_20260814 python scripts/diagnostics/k_full_retrain.py --arm auto100 --model <name>` |

## 3. 训练方法(与 simhyd/vic/tcm 相同的方式)

- **参数化网络**:`CatchmentParameterizer` MLP,hidden [256,256],dropout 0.05,
  **中点初始化**(最后一层 Linear 权重/偏置置零,保证初始 θ 落在参数空间中部);
  输入 Caravan 属性(zscore 归一化,531 流域)。
- **参数映射**:`auto`(arm auto100 专属;对照 arm linear100 用 `linear`)。
- **损失**:可微 KGE(`streaming_kge, eps=0.1`),`loss = 1 − KGE`,训练期窗口内计算。
- **优化器**:AdamW,lr=1e-3,weight_decay=1e-4,梯度裁剪 max_norm=1.0。
- **训练循环**:100 epochs × 169 steps/epoch;每 step 随机采样 100 个流域 × 730 天窗口;
  预热 warmup=365 天,`warmup_grad_mode=detach`(例外:penman 用 `truncate:90`)。
- **编译**:`torch.compile(step_function, fullgraph=True)`,失败即硬报错。
- **特殊模型**:`CALENDAR_MODELS={mopex4, mopex5, vic}` 额外注入日期对齐的第 4 通道 forcing。
- **checkpoint**:每 10 epoch 保存(含 network/optimizer/RNG/invalid 计数),支持断点续训。
- **数据切分**:训练 1980-10-01..1995-09-30 / 验证 1994-10-01..2010-09-30
  (由 `h_training_pilot.load_camels_time_series` 提供)。

### 早停规则(本次运行的规则,2026-08-15 起生效)

```
最少训练 MIN_EPOCHS = 50 个 epoch;
之后若验证期 median KGE 连续 PATIENCE = 10 个 epoch 提升 < PLATEAU_EPS = 0.001,
则以 PLATEAU_STOP 状态提前停止;上限 100 epochs(COMPLETED)。
```

所有参数可用环境变量覆盖:`DPL_MIN_EPOCHS` / `DPL_PATIENCE` / `DPL_EPS` / `DPL_BATCH`。

### 健康检查(summarize_health,写入 health.csv)

- `pass_integrity`:终态且训练/验证无非有限预测
- `pass_learning`:best−epoch1 KGE > 0.05
- `pass_no_dead_parameters`:无永久零梯度参数
- `pass_no_saturation`:最终边界占比 < 0.20
- `pass_convergence_budget`:best epoch 不落在最后 4 epoch 内
- `pass_no_degradation`:best−final KGE ≤ 0.05

## 4. 执行时间线

| 时间(UTC+8) | 事件 |
| :--- | :--- |
| 2026-08-13 21:54 ~ 08-14 00:08 | 首批 3 模型(simhyd/vic/tcm)各 100 epochs **COMPLETED**(旧早停规则) |
| 2026-08-15 21:38 | 调度器启动:**6 worker 并行**,队列 33 模型(自动跳过已完成的 3 个),失败自动重试 ×3 |
| 2026-08-15 22:17~22:41 | **flexb 连续失败 ×3**(见 §6),被调度器放弃(failed.txt) |
| 2026-08-16 04:27 | `SCHEDULER_ALL_DONE`:32 个模型全部完成(均为 PLATEAU_STOP) |
| 2026-08-16 ~09:30 | flexb bug 修复后单独重跑;仅到 epoch 4(KGE 0.372)节点即失联 |
| 2026-08-16 09:30~ | 35/36 结果下载到本地(workspace + 主仓库),MD5 逐字节校验一致 |
| 2026-08-16 | 决定:**flexb 不同步**,35/36 为最终集(README / registry 已标注) |

## 5. 结果汇总(35/36 模型,验证期 median KGE)

- 3 个 COMPLETED(100 ep):simhyd **0.5573**、vic **0.5433**、tcm **0.4764**
- 32 个 PLATEAU_STOP(50–92 ep),Top10:

| model | best KGE | stop_ep | model | best KGE | stop_ep |
| :--- | :--- | :--- | :--- | :--- | :--- |
| hbv96 | 0.7617 | 63 | ihacres | 0.6340 | 64 |
| mopex5 | 0.7342 | 65 | collie2 | 0.6211 | 65 |
| mopex4 | 0.7204 | 61 | tank | 0.6182 | 69 |
| mopex3 | 0.7204 | 62 | newzealand1 | 0.6179 | 63 |
| mopex2 | 0.7184 | 61 | us1 | 0.6103 | 65 |

完整明细见 `auto100/health.csv`(35 行)。

## 6. flexb 事故记录(重要)

- **现象**:flexb 训练必崩,`ValueError: only one element tensors can be converted to Python scalars`。
- **根因**:`k_full_retrain.py` 原有 bug —— 训练循环变量 `_` 被
  `loss,_=NATIVE.compute_differentiable_kge(...)` 重绑为 KGE 张量;
  flexb 触发非有限梯度守卫路径时 `int(_)` 崩溃。
- **修复**:循环变量改名 `step_i`,`loss,_=` 改为 `loss,_kge=`(已提交 git)。
- **结局**:修复后重跑仅到 epoch 4 节点失联;经确认训练轮数少,**放弃同步,最终集 35/36**。
- **影响**:epochs.csv / parameter_gradients.csv 中残留 flexb 1–4 epoch 的行,属正常记录,不代表完成。

## 7. 文件清单

```
dpl_full_retrain_20260813/
├── TRAINING_LOG.md            # 本文件
├── README.md                  # 运行说明摘要
├── auto100/
│   ├── health.csv             # ★ 每模型终态总结(35 行,最佳/最终 KGE、停止轮数、健康门禁)
│   ├── status.csv             # model/arm/status/last_epoch
│   ├── epochs.csv             # 每 epoch 明细(KGE median/mean、loss、边界占比、耗时)
│   ├── parameter_gradients.csv# 每 epoch 每参数零梯度/边界占比
│   ├── contract.json          # 训练契约(arm/模型/超参)
│   ├── epochs_firstrun_backup.csv / health_plateau_backup.csv / status_plateau_backup.csv  # 历史备份
│   └── checkpoints/<model>/epoch_*.pt   # 234 个 checkpoint(35 模型,每 10 ep)
└── run_20260815/              # 调度器档案
    ├── scheduler.sh / master.log / queue.txt / failed.txt / scheduler_outer.log
    └── logs/<model>.log       # 每模型训练日志(33 个)+ flexb_rerun.log
```

**校验**:health/status/epochs/parameter_gradients 四个 CSV 的 MD5 与远端逐字节一致
(快照时点)。checkpoint 抽查可正常加载。

## 8. 复现方式

```bash
# 单模型(从 0 或从 checkpoint 续训,自动跳过 health 已终态模型)
PYTHONPATH=<repo> python project/benchmark/scripts/diagnostics/k_full_retrain.py \
    --arm auto100 --model hbv96

# 全部剩余模型,6 worker 并行动态填补(推荐)
bash project/benchmark/scripts/diagnostics/dpl_parallel_scheduler.sh

# 规则调整示例
DPL_MIN_EPOCHS=60 DPL_PATIENCE=15 DPL_EPS=0.0005 python ... --arm auto100 --model gr4j
```

注意:运行需 CUDA GPU;数据依赖 `data/531sub_id.txt` 与 `data/camels_dataset`
(远端 `/autodl-fs/data/`,本地 symlink 于 `data/`)。

## 9. 关联记录

- Provenance 注册:`project/benchmark/docs/superseded_results_registry.csv`
  (本结果标记为 canonical,Chapter-4 重建的 dPL 列来源)
- IC 侧对应:full300 CMA-ES(见 `docs/full300_cmaes_training_runbook.md`)
- 本日志的 git 跟踪副本:`project/benchmark/docs/training_log_dpl_full_retrain_20260813.md`
