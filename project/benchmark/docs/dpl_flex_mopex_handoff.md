# dPL + dmotpy 训练交接：Flex 与 MOPEX4/5

> 状态：交接文档，下一步进入 Flex 当前代码可重训性检查。
> 本文只记录 dPL 轨道；IC Full300 已单独定稿，不在本文重新讨论。

## 1. 当前结论

### 1.1 Flex 族

`flexb`、`flexi`、`flexis` 三个模型确实存在训练数值问题，不只是总控脚本的汇总写入错误：

- 训练日志在中途出现 `Val KGE=-0.4142135`，随后 `Train Loss=1.0000` 长时间平台；
- CPU 加载 checkpoint 后发现参数张量为 `NaN`；
- 三个模型的 `best.pt` 仍为有限值，但后续 epoch checkpoint 已污染；
- 因此旧结果不能作为“完整成功训练”使用，只能保留最佳有限 checkpoint 作为诊断材料。

| 模型 | 日志中的崩溃阶段 | 首个发现非有限参数的已保存 checkpoint | best epoch |
|---|---:|---:|---:|
| `flexb` | epoch 3–4 | `epoch_05.pt` | 2 |
| `flexi` | epoch 17–18 | `epoch_20.pt` | 15 |
| `flexis` | epoch 2–3 | `epoch_05.pt` | 1 |

### 1.2 MOPEX4/5

当前 dPL run 中没有 `mopex4`、`mopex5` 的训练结果：

- 不属于“训练后写汇总失败”；
- 本次 20260814 dPL pool 实际只运行了 34 个模型；
- `mopex4`、`mopex5` 是未纳入本次 run 的两个缺口，需后续单独补跑。

### 1.3 其他 12 个总控失败模型

以下模型的总控退出码为 1，但训练和验证产物已经存在：

```text
alpine1 alpine2 australia
collie1 collie2 collie3
gr4j
flexb flexi flexis
gsfb hbv96
```

准确列表以 `dpl_failure_audit.csv` 为准。它们统一在写入缺失目录时失败：

```text
results/dpl/_summary/dpl_model_summary.csv
```

错误为 `OSError: Cannot save file into a non-existent directory`，不是 GPU 或模型计算错误。

## 2. 远程数据与审计产物

远程 dPL 根目录：

```text
/autodl-fs/data/dpl_run_20260814/project/benchmark/
```

原始结果：

```text
checkpoints/dpl/
results/dpl/
logs/dpl_pool/
```

本次新增并已执行的 CPU-only 程序：

```text
scripts/rebuild_dpl_summary.py
scripts/audit_dpl_checkpoint_finiteness.py
```

生成的审计文件：

```text
results/dpl/_summary/dpl_model_summary_rebuilt.csv
results/dpl/_summary/dpl_failure_audit.csv
results/dpl/_summary/dpl_audit_report.json
results/dpl/_summary/dpl_checkpoint_finiteness.csv
```

这两个程序不进行训练，不需要 GPU：

- `rebuild_dpl_summary.py`：从逐模型 `summary.json`、checkpoint 和日志重建汇总；
- `audit_dpl_checkpoint_finiteness.py`：用 CPU 检查 checkpoint 中的 NaN/Inf。

## 3. 当前 dmotpy 修复状态

本地仓库已经包含针对 Flex/saturation3 的源代码和测试材料：

```text
dmotpy/models/flux/saturation.py
dmotpy/models/core/flexb.py
dmotpy/models/core/flexi.py
dmotpy/models/core/flexis.py
dmotpy/tests/test_flex_saturation3_parameter_bound_fix.py
dmotpy/scripts/validate_flex_saturation3_bound_fix.py
```

当前代码的明确事实：

- 三个 Flex 的 `beta` 下界为 `0.0`；
- 仓库存在针对 beta=0、近零 beta、输出有限性和梯度有限性的测试；
- `saturation_3` 已经有稳定性重写及审计报告材料；
- 已核验 20260814 dPL run 使用的 `dmotpy` 四个关键文件与当前本地版本 SHA256 完全一致：`saturation.py`、`flexb.py`、`flexi.py`、`flexis.py`；
- 因此 Flex NaN 不能简单归因于“远程还在使用旧版 dmotpy 修复前代码”；
- 但远程 `run_dpl_benchmark_dmg_native.py` 与当前本地版本 SHA256 不同，重训前仍需保留并审查远程 runner 的确切版本；
- 局部算子测试通过不能替代完整 dPL 长序列反向传播验证。

当前更准确的结论是：

> dmotpy 的 Flex 修复代码已经部署到本次 run，但在 dPL 长序列训练中仍出现参数 NaN；问题需要沿着 dPL runner、float32 反向传播、beta 接近零时的梯度和 minibatch 状态继续定位。
## 4. 下一步计划

### Step A：确认代码版本

比较本地与远程 dPL run 中以下文件的 SHA256：

```text
dmotpy/models/flux/saturation.py
dmotpy/models/core/flexb.py
dmotpy/models/core/flexi.py
dmotpy/models/core/flexis.py
dmotpy/models/hydrology_model.py
project/benchmark/dpl/nn_parameterizer.py
project/benchmark/scripts/run_dpl_benchmark_dmg_native.py
```

若远程不是当前版本，先只同步代码，不覆盖原始 checkpoint。

### Step B：CPU/小规模数值预检

在完整训练前，对三个 Flex 分别执行：

1. saturation3 beta=0、1e-12、1e-6、1e-4、5.0 的输出/梯度有限性测试；
2. 单模型、少量流域、短窗口、1–3 epoch 的 dPL forward/backward；
3. 开启 `torch.autograd.detect_anomaly()`；
4. 每个 minibatch 检查：loss、参数、梯度是否 finite；
5. 一旦出现非有限值，记录模型参数、beta 分布、forcing、状态量和对应 minibatch。

### Step C：小规模通过后再补训

只补训以下五个缺口/问题模型：

```text
flexb flexi flexis mopex4 mopex5
```

不需要重跑其余 31 个模型。

训练前必须先创建：

```text
results/dpl/_summary/
```

### Step D：重训验收标准

每个模型必须满足：

- 所有保存 checkpoint 的 parameterizer 参数均 finite；
- optimizer state 不含 NaN/Inf；
- 训练 loss 全程 finite；
- 不出现持续的 `Val KGE=-0.4142135`；
- 最终 checkpoint 和 `best.pt` 均存在且 finite；
- `summary.json`、`epochs.csv`、`basin_metrics.csv` 完整；
- MOPEX4/5 结果进入新的 36 模型汇总；
- Flex 三模型不得只依赖旧的 `best.pt`，必须完成一次无 NaN 的全程训练。

## 5. 禁止事项

- 不要把 20260814 的 Flex 最终 epoch checkpoint 当作有效训练结果；
- 不要把总控 `OSError` 与 Flex 的 NaN 数值错误混为同一个问题；
- 不要在未确认远程 dmotpy 代码版本前直接启动重训；
- 不要覆盖原始 dPL run 目录，新的重训应使用独立 run-id。

## 6. 2026-08-18 远程 smoke test 执行进度

### 当前状态

- **当前没有训练进程运行，GPU 已释放。**
- 远程服务器：`connect.westb.seetacloud.com:20280`，工作目录：`/root/dmg-research`。
- GPU：NVIDIA GeForce RTX 3080 Ti 12GB。
- PyTorch：`2.8.0+cu128`，CUDA 可用，`torch.compile` 可用。

### 本轮实际配置

```text
models: flexb flexi flexis mopex4 mopex5
backend: compile
device: cuda
max_epochs: 100
min_epochs: 50
patience: 10
min_delta: 1e-4
batch_size: 100
rho: 730
warmup: 365
lr: 1e-3
```

### 首轮结果

五个模型曾并行运行约 74 分钟。为避免单个模型失败后继续浪费 GPU，发现 FlexIS 阻塞后已停止其余四个进程。

| 模型 | 观察到的进度 | 状态 |
|---|---:|---|
| `flexb` | epoch 44 | 已停止，checkpoint finite |
| `flexi` | epoch 38 | 已停止，checkpoint finite |
| `flexis` | epoch 34 | **失败：`non-finite gradient norm`** |
| `mopex4` | epoch 69 | 已停止，checkpoint finite |
| `mopex5` | epoch 66 | 已停止，checkpoint finite |

远程 checkpoint 审计结果：上述五个模型已有 checkpoint 中均未发现 NaN/Inf；但本轮不是完整训练结果，不能作为最终结果使用。

### 已同步代码

```text
dmotpy/models/flux/saturation.py
project/benchmark/scripts/run_dpl_benchmark_dmg_native.py
project/benchmark/scripts/launch_parallel_dpl_remote.py
project/benchmark/dpl/nn_parameterizer.py
project/benchmark/dpl/attributes.py
```

Runner 已加入：

- 至少 50 epoch 后才允许 early stopping；
- train loss early stopping；
- 每 minibatch loss、gradient、parameter finite 检查；
- physical parameter boundary 检查；
- `torch.compile` backend；
- float64 安全梯度范数计算与裁剪。

### 日志与 checkpoint

```text
/root/dmg-research/project/benchmark/logs/dpl_parallel/
/root/dmg-research/project/benchmark/checkpoints/dpl_production_20260730/
```

### 下一步

1. 用已同步的 float64-safe gradient clipping 版本重新做 FlexIS 验证；
2. 确认 FlexIS 不再触发 gradient norm 阻塞；
3. 再重新并行启动五模型完整 smoke/retrain；
4. 完成后审计所有 checkpoint、optimizer state、summary 和 validation 产物。

## 7. 2026-08-18 IC/DPL 对比与修复

### 对比结果

比较产物已下载到：

```text
project/benchmark/results/remote_comparison_20260818/
```

Canonical IC Full300 与原始 dPL 汇总的 validation/test Median KGE：

| 模型 | IC Full300 | dPL | dPL - IC |
|---|---:|---:|---:|
| `flexb` | 0.4729 | 0.4038 | -0.0691 |
| `flexi` | 0.5153 | 0.4595 | -0.0558 |
| `flexis` | 0.6153 | 0.2901 | -0.3252 |

34 个有完整两侧结果的非 Flex 模型，其绝对差值中位数为 0.0159、均值为 0.0195；因此 FlexIS 是明确异常，FlexB/FlexI 也有中等差距，但不能把全部差距归因于模型方程。

### 根因判断

- IC 使用 1989–1998 CMA-ES 独立流域优化和 1999–2009 测试；dPL 使用共享属性到参数 MLP、1980–1995 训练和 1995–2010 验证，时间窗与优化容量不同。
- dPL 原 runner 使用 float32、总体方差、`eps=1e-6` 的自定义 KGE；IC 使用 float64、样本方差、`streaming_kge(eps=0.1)`。
- 原始 Flex dPL 日志均出现 loss=1.0 平台；FlexIS 另有 `non-finite gradient norm`。

### 已实施修复

- `project/benchmark/scripts/run_dpl_benchmark_dmg_native.py` 改为调用 IC-compatible `streaming_kge`。
- `project/benchmark/src/objective.py` 对方差、协方差尺度和 KGE 距离加入 `1e-24` differentiable floor，避免长序列 Flex 反向传播在 `sqrt(0)` 处产生非有限梯度。
- 新增 `project/benchmark/dpl/tests/test_dpl_kge_alignment.py`，验证 dPL KGE 与 IC KGE 一致及梯度有限。
- 远程备用连接的数据树已同步上述 runner/objective；三模型实际单 minibatch eager + anomaly 检查均通过：FlexB、FlexI、FlexIS 均 forward/loss/gradient finite。

下一步：使用修复后的 runner，在独立 run-id 下重新训练三种 Flex，并用 IC-compatible 指标重新评估；暂不修改 Flex 方程本身。
