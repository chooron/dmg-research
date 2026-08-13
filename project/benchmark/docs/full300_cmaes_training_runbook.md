# Full300 36-Model CMA-ES 独立率定（IC）远程训练 Runbook

> 目的：完整记录 2026-07-29/30 在远程 GPU 节点上执行的 36 水文模型 CMA-ES 独立率定
> （full300，10-starts / 300-generations）的环境、代码、配置、执行流程、检查点格式与评价管线，
> 以便后续**调整训练期后重新训练**时可复现。

**状态**：已用当前代码对 canonical 检查点做逐流域精确复现验证（MOPEX1–3 test KGE
0.5547/0.7118/0.7047，median_abs_diff=0.000000），当前仓库代码 == 产生 canonical 结果的训练代码。

---

## 1. 总览（Architecture）

- **任务**：36 个 dmotpy 水文模型，在 CAMELS 531 流域上做**逐流域独立的 CMA-ES 率定**（IC）。
- **目标函数**：训练期 KGE（`streaming_kge, eps=0.1`），**仅用训练期**（`selection_period: train_only`）；
  测试期 KGE 在参数冻结后**事后**评估，不参与选择。
- **搜索空间**：无约束 latent → `sigmoid` → 模型内 `linear` descale 到物理参数边界；无硬 clip。
- **选择规则**：每流域在 `starts` 个独立 CMA-ES 起点中，按训练期 KGE 取最优（best-of-N）；
  检查点保存到最终 generation（300）。
- **GPU**：单卡（3080 Ti 12 GiB），basin × start × population 全向量化，`torch.compile`，
  OOM 时逐级退回更小 basin chunk（531→256→128→64）与 eager backend。

## 2. 远程环境与访问（Remote Environment）

| 项 | 值 |
| :--- | :--- |
| 节点 | `connect.westb.seetacloud.com`（Seetacloud / AutoDL） |
| SSH 端口 | 曾用 **53700**（部署脚本），当前（2026-08-12）为 **20280**；重启后可能变化，以控制台为准 |
| 用户 | `root` |
| 密码 | 见 `project/benchmark/scripts/deploy_to_ssh.py` 与 `remote_exec_launch.py` 内嵌值（本仓库已记录） |
| 训练 Python | `/root/miniconda3/bin/python`（conda env，含 torch + torch.compile） |
| 数据盘 | `/autodl-fs`（持久化；`/root` 为系统盘，重启可能丢失） |
| 训练根目录 | `/autodl-fs/data/dmotpy_cmaes36_full300_20260729`（即 `results_root`） |
| 远程代码 | `/root/dmg-research`（部署时同步的仓库副本） |

常用 SSH：`ssh -p <port> root@connect.westb.seetacloud.com`

## 3. 代码位置（Code Layout）

| 位置 | 内容 |
| :--- | :--- |
| **本地部署快照（只读，git 已跟踪）** | `project/hydrodiag/archive/remote_runtime_snapshots/20260730/ssh_53700_cmaes36/deployment/experiments/cmaes_36models/` —— 完整的训练代码（src/、scripts/、configs/、frozen_versions/、references/） |
| 本地运行中间归档 | `project/hydrodiag/results/archive/remote_cmaes36_full300_20260729_160112_interim/experiments/cmaes_36models/` |
| 远程运行代码 | `/autodl-fs/data/dmotpy_cmaes36_full300_20260729/experiments/cmaes_36models/` |
| 本地检查点缓存（本次拉取） | `project/benchmark/checkpoints/full300_final_36models/<model>/chunk_*_gen_300.pt`（36 模型，gitignored） |
| 本地部署脚本 | `project/benchmark/scripts/deploy_to_ssh.py`、`remote_exec_launch.py`、`launch_36models_pool6.py` |
| 本地早期 runner（参考） | `tmp/cmaes_remote/`（runner.py、run_fused_cmaes_531.py 等） |

**注意**：训练源码未拷贝进当前仓库的常规路径（`dmotpy/experiments/cmaes_36models/` 下只有
`downloads/`）。如需重训，以部署快照（或远程）中的源码为准。

## 4. 配置系统（Config System）

`default.yaml`（不变量）→ `dimension_tiers.yaml`（按参数维度分档）→ `pilot.yaml` / `full_run.yaml`
/ `full_run_10starts_300gen_warm1980_1981x5.yaml`（叠加）。

### 4.1 实际使用的 full300 配置（`full_run_10starts_300gen_warm1980_1981x5.yaml`，完整内容）

```yaml
extends: default.yaml
stage: full_production_prepared
resume: true
model_freeze_manifest: frozen_versions/cmaes36_full300_20260729/manifest.json

optimization:
  starts: 10
  generations: 300
  population_by_dimension:
    '1': 8
    '4-6': 12
    '7-10': 16
    '12-15': 20
  active_cma: true
  full_covariance: true
  separable: false
  stdev_init: 0.10

warmup:
  mode: repeat_forcing
  source: {start_time: '1980-10-01', end_time: '1981-09-30'}
  source_days: 365
  repetitions: 5
  total_days: 1825
  objective_includes_warmup: false

checkpoint_every_generations: 5
checkpoint_milestones: [25, 50, 100, 150, 200, 250, 300]
data:
  basin_ids: data/531sub_id.txt
  reference_ids: data/gage_id.npy
  data_path: data/camels_dataset
  train: {start_time: '1989-01-01', end_time: '1998-12-31'}
  test:  {start_time: '1999-01-01', end_time: '2009-12-31'}
```

### 4.2 关键语义（调整训练期前必读）

| 键 | 含义 | 当前值 |
| :--- | :--- | :--- |
| `data.train` | **率定目标期**（KGE 只在此窗口计算） | 1989-01-01..1998-12-31（10 年） |
| `data.test` | 冻结后**报告用测试期**（不参与选择） | 1999-01-01..2009-12-31（11 年） |
| `warmup.source` | warmup 强迫源（一年水年，重复 5 次） | 1980-10-01..1981-09-30 ×5 = 1825 天 |
| `optimization.starts` | 每流域独立起点数 | 10 |
| `optimization.generations` | 最大代数 | 300 |
| `population_by_dimension` | 按参数维度的人口 | 8/12/16/20 |
| `selection_period` | 选择只用训练期 | `train_only`（default.yaml） |

**数据时间轴**：`data/camels_dataset` 覆盖 **1980-10-01 .. 2014-09-30**（12,418 天）。
warmup.source 必须位于训练期之前且不重叠；warmup 不计入目标。

## 5. 实际执行流程（2026-07-29 ~ 07-30，5 阶段）

| 阶段 | 时间 | 脚本 | 说明 |
| :--- | :--- | :--- | :--- |
| 1. Pilot | 07-29 12:05 | `run_36model_pilot_parallel.sh`（→ `run_36model_pilot.py`，pilot.yaml 5-start/30-gen） | 36 模型冒烟+试点；**MOPEX1–3 因未修复的 wrapper 缺陷结果异常**（train median −0.04/+0.23/+0.07） |
| 2. MOPEX fix | 07-29 14:56 | `run_mopex1_5_retrain.sh`（"explicit_delta_t/nearzero wrapper fix"） | 修复 wrapper 后 MOPEX1–5 重率定（gen-30 即达 0.65/0.79/0.78/0.65/0.65） |
| 3. Full300 | 07-29 16:01 | `run_full300_memory_aware.sh full300_20260729_160112` | 正式 10-start/300-gen，24 模型首批完成；MOPEX 等 12 模型 OOM/续跑 |
| 4. Continuation | 07-30 09:31 | `run_full300_remaining_continuation.sh full300_20260729_160112_remaining_20260730_093100` | 14 个剩余模型（含 mopex1–5、gr4j、hbv96、tank、tcm 等）在新 checkpoint 根续跑 |
| 5a. Wetland A | 07-30 09:40 | `run_full300_wetland_candidate_a_after_continuation.sh` | wetland 候选 A 重跑 |
| 5b. Simhyd Ref | 07-30 09:50 | `run_full300_simhyd_reference_after_wetland.sh` | simhyd 参照重跑 |
| 6. 最终集 | 07-30 20:21 | 手工建符号链接 `checkpoints/full300_final_36models/<model>` → 各真实运行目录 | canonical 最终检查点集（本次已拉取到本地） |
| 7. 评价 | 训练后 | `run_full300_evaluation_after_training.sh <RUN_ID>`（→ `evaluate_full300_metrics.py`，train/test 双期） | 产出 `results/full300_<RUN_ID>_evaluation/`（train/test KGE by basin） |

### 关键脚本清单（部署快照 `scripts/`）

| 脚本 | 用途 |
| :--- | :--- |
| `run_36model_pilot.py` | **每个模型的真正执行入口**（run_full/continuation/memory-aware 都调它）；参数 `--model --run-id --chunk --backend --config` |
| `run_full.py` / `resume_run.py` / `run_pilot.py` / `run_smoke.py` | 官方 CLI（run_full 有 gate 校验） |
| `run_full300_memory_aware.sh` | 正式 full300 启动（两 worker 内存感知，低内存双 lane + 高内存单 lane） |
| `run_full300_remaining_continuation.sh` | 续跑剩余模型（新 checkpoint 根，单进程/模型） |
| `evaluate_full300_metrics.py` | 训练后 train/test KGE 评价（只读，事后） |
| `freeze_model_version.py` | 生成 frozen manifest（代码 hash 冻结） |
| `validate_full300_config.py` / `validate_models.py` / `validate_stage_a.py` | 部署前校验 |
| `benchmark_gpu.py` / `inspect_environment.py` | GPU/环境探测 |

**本地等价物**：`scripts/evaluate_benchmark_metrics.py`（canonical train/test 评价）、
`scripts/evaluate_ic_aligned_gen300.py`（aligned 1995–2010 评价，含 gen-300 guard）。

## 6. 检查点与冻结清单（Checkpoint & Manifest）

### 6.1 检查点文件（`checkpoints/<RUN_ID>/<model>/chunk_<k>_gen_<g>.pt`）

- 按 basin chunk 分片（chunk 大小因模型内存而异：531/256/128/64 单文件，或 0/64/128/256/384/448/512 多片）；
- payload 键：`model, generation, solver, basin_ids, resolved_config, data_metadata, history, rng`；
  `solver.state` 含 `mean, C, A, sigma, p_sigma, p_c, best_fitness (n_basin×starts), best_latent (n×starts×dim), generation`；
- **不内嵌**参数名/bounds/model 版本 → 语义依赖加载时 registry + frozen manifest（重要：改 registry 需重冻结）；
- `DONE` 标记文件表示该模型完成；`resolved_config` 内嵌训练/测试期与 warmup 定义（可自校验）。

### 6.2 Frozen manifest（`frozen_versions/cmaes36_full300_20260729/manifest.json`）

- `source_hashes_sha256`：训练代码各文件 hash（含 `dmotpy/models/core/mopex1..5.py`、`flux/*.py`、`mopex_doy_model.py`）；
- `parameter_bounds`、`model_registry`（参数维度）、`resolved_config`、`aggregate_source_hash`；
- 创建于 07-29 07:39（pilot 之前）。**注意**：MOPEX wrapper（`flux/mopex.py`、`mopex_doy_model.py`）
  在 07:39 之后、pilot 之后应用了 delta_t/nearzero 修复，因此 manifest 中这两文件 hash ≠ 当前；
  但 mopex1/2/3 核心文件 hash == 当前。**当前代码 == 产生 canonical full300 结果的代码**（已由精确复现证实）。

## 7. 评价管线（Evaluation）

| 环节 | 实现 | 说明 |
| :--- | :--- | :--- |
| canonical train/test | `evaluate_full300_metrics.py`（远程）/ `evaluate_benchmark_metrics.py`（本地）→ `src/data_selection.evaluate_period` | repeat_forcing 1825d warmup + train/test 窗口；`streaming_kge(eps=0.1)` |
| aligned 1995–2010 | `evaluate_ic_aligned_gen300.py`（本地，本次重建） | 365d warmup（1994-10-01 起）+ 1995-10-01..2010-09-30；`parameter_mapping=linear`；校验 `generation==300` |
| 加载器 | `src/data_selection.frozen_parameters` | best-of-N（按 `best_fitness` 即训练 KGE 取最优 start）；`sigmoid(latent)` → build_model descale |

**已证实的坑（务必避免）**：aligned 分析曾误用 **pilot（gen-30, best-of-5）** 检查点目录
（`remote_runs/20260729_120525`）作为 IC 来源，导致 MOPEX1–3 aligned IC 为负（−0.12/−0.04/−0.16）。
**IC 检查点必须取自 full300（gen-300, best-of-10）**（`full300_final_36models` 或
`full300_20260729_160112(_remaining_...)_*` 运行目录），并在加载时校验 `resolved_config.generation` 与
`best_fitness` 数量级。

## 8. 调整训练期重新训练（Retrain with New Period）

### 8.1 修改点（最小改动）

编辑 `configs/full_run_10starts_300gen_warm1980_1981x5.yaml`（或复制为新文件）：

1. **`data.train.start_time / end_time`** ← 新率定期（KGE 目标只在此窗口）；
2. **`data.test.start_time / end_time`** ← 新报告测试期；
3. **`warmup.source`** ← 训练期之前的可用数据段（`repeat_forcing` 需 `source_days × repetitions ≤ 训练期起点前的数据长度`，且 `warm_right ≤ train_left`）；
4. 若数据时间轴超出 1980-10-01..2014-09-30，需先扩展 `data/camels_dataset`。

### 8.2 重新训练流程（推荐，参照第 5 节）

```bash
# 0) 前置校验（在部署快照目录，repo venv）
python experiments/cmaes_36models/scripts/validate_full300_config.py --config <new_config>.yaml
python experiments/cmaes_36models/scripts/validate_models.py

# 1) 重新冻结 manifest（新 config/代码 hash）
python experiments/cmaes_36models/scripts/freeze_model_version.py --name cmaes36_<run>_<date> --config <new_config>.yaml

# 2) 远程部署（deploy_to_ssh.py 同步代码+config 到 /root/dmg-research）

# 3) 远程启动（在 /autodl-fs/data/dmotpy_<run>/... 下，参照 memory-aware 脚本）
ROOT=/autodl-fs/data/dmotpy_<run>_<date> \
bash scripts/run_full300_memory_aware.sh <RUN_ID>        # 首批
bash scripts/run_full300_remaining_continuation.sh <CONT_RUN_ID>   # 剩余模型续跑
# 完成后建 full300_final_36models 符号链接集

# 4) 训练后评价
bash scripts/run_full300_evaluation_after_training.sh <RUN_ID>
```

### 8.3 重新训练后的本地衔接（下游全部重算）

1. 拉取新检查点 → 组织为 `project/benchmark/checkpoints/<run>/<model>/chunk_*_gen_300.pt`；
2. `scripts/evaluate_ic_aligned_gen300.py` 重算 1995–2010 aligned IC（或新对齐窗口）；
3. 与 dPL（train-loss 选择版）合并 → 重建 `all36` aligned 诊断表与报告
   （脚本：`scripts/diagnostics/reselect_dpl_trainloss_eval.py`、`scripts/diagnostics/rebuild_all36_diagnosis_trainloss.py`、
   `scripts/evaluate_ic_aligned_gen300.py`）。

### 8.4 约束与注意

- **选择泄漏红线**：训练期调整后，aligned 评价窗口（1995–2010）与训练期的重叠关系会变；
  若新训练期覆盖 1995–2010，则 aligned 不再是"外推"——需同步调整 aligned 窗口或单独保留 held-out
  （数据到 2014-09-30，可用 2010-10-01..2014-09-30 作为两臂均未见过的窗口）。
- **registry 一致性**：改动参数数量/顺序/bounds 必须同步更新 registry 并重新冻结 manifest；
  旧检查点与新 registry 不兼容时会静默错位（无自描述校验）。
- **MOPEX wrapper**：重训前确认 `flux/mopex.py`/`mopex_doy_model.py` 为修复后版本（当前即修复版）。

## 9. 本地已有工件（Artifacts, 2026-08-12 状态）

| 工件 | 路径 |
| :--- | :--- |
| canonical 检查点（36 模型） | `project/benchmark/checkpoints/full300_final_36models/` |
| full300 canonical 结果 | `project/benchmark/results/full300_final_36models_evaluation/` |
| aligned IC 修正评估（36 模型） | `project/benchmark/results/all36_ic_gen300_aligned_20260812/` |
| MOPEX IC 兼容性诊断 | `project/benchmark/results/mopex_ic_compatibility_20260812/` |
| dPL train-loss 选择修正诊断 | `project/benchmark/results/all36_dpl_gap_diagnosis_20260812_trainloss/` |
| 旧（已废弃）aligned 诊断 | `project/benchmark/results/all36_dpl_gap_diagnosis_20260812/`（IC 列错误，勿用） |

## 10. 已知问题与备注

1. **aligned 诊断曾误用 pilot 检查点**（见第 7 节）——重训后务必用 `evaluate_ic_aligned_gen300.py`（带 generation 校验）；
2. **MOPEX4/5**：aligned IC（0.24）明显低于 canonical test（0.65），与 phase 参数化问题相关，属另案（phase-fix 追踪中）；
3. **simhyd/wetland** 有 07-30 专用重跑（simhydRef / wetlandA），canonical 以重跑为准；
4. **Seetacloud 节点重启后 SSH 端口会变**，检查点/代码在 `/autodl-fs` 持久化，`/root` 不持久。

---

## 11. Canonical Pipeline（本地，2026-08-13 收敛后）

本地唯一推荐的 Full300 CMA-ES IC 训练/评价/重建链路（全部相对 `project/benchmark/`）：

| 步骤 | 入口 | 说明 |
| :--- | :--- | :--- |
| config | `configs/full_run_10starts_300gen_warm1980_1981x5.yaml` | 唯一 canonical config（改训练期见 §8） |
| validate | `scripts/validate_full300_config.py --config <cfg> --manifest <manifest>` | 校验 config==manifest、源码 hash、warmup/train shape |
| freeze | `scripts/freeze_model_version.py --name <run> --config <cfg>` | 生成 `frozen_versions/<name>/manifest.json` |
| deploy | `scripts/deploy_to_ssh.py` | 同步代码/config 到远程节点 |
| train (all) | `bash scripts/run_full_benchmark.sh <RUN_ID>` 或 `python scripts/run_36model_benchmark.py --model all --run-id <RUN_ID> --config <cfg>` | 36 模型 CMA-ES（memory-aware chunk fallback） |
| continuation | `bash scripts/run_continuation.sh <RUN_ID> [MODEL ...]` | 只训练未完成模型，自动 resume 最近 generation |
| final set | `python scripts/consolidate_final_checkpoints.py --checkpoint-root checkpoints/<final> --create --source <run dirs>` | 生成/校验 final 检查点集（gen-300, 531 流域） |
| canonical eval | `python scripts/evaluate_benchmark_metrics.py --checkpoint-root <final> --config <cfg> --output-dir results/...` | train/test KGE（**gen-300 guard 强制**） |
| aligned eval | `python scripts/evaluate_ic_aligned_gen300.py --ckpt-root <final> --out results/all36_ic_gen300_aligned_<date>` | 1995–2010 aligned IC（**gen-300 guard 强制**） |
| Chapter-4 rebuild | `python scripts/diagnostics/reselect_dpl_trainloss_eval.py` + `python scripts/diagnostics/rebuild_all36_diagnosis_trainloss.py` | dPL train-loss 选择 + 全 36 诊断重建 |

### Hard provenance guard（Phase 4，已落地）

`src/checkpoint_guard.py`（`validate_canonical_checkpoint`）在 canonical/aligned 评价入口强制校验：

1. model 名匹配；
2. `DONE` 标记存在；
3. 只接受 `chunk_*_gen_300.pt`；**gen<300（如 pilot/gen-30）→ 报错并提示"疑似 pilot/intermediate checkpoint"，拒绝静默降级**；
4. 合并所有 chunk 后 basin 覆盖 == 531（无重叠）；
5. latent 维度 == registry 维度（参数 schema）。

已接入：`evaluate_benchmark_metrics.py`、`evaluate_ic_aligned_gen300.py`；`consolidate_final_checkpoints.py` 亦可独立校验整集。
测试：gen-30 拒绝 ✓、partial 拒绝 ✓、no-DONE 拒绝 ✓、gen-300 接受 ✓。

### 清理结果（2026-08-13）

- **删除**（git 可恢复）：`scripts/diagnostics/analyze_dpl_vs_ic_comparison.py`（孤儿、被 aligned 诊断完全替代、可能误导）；
- **DEPRECATED 标记**：`scripts/diagnostics/round13_finalize.py`（旧验证集 KGE 选择规则，选择泄漏）；
- **保留并标记**：`scripts/diagnostics/` 其余 84 个历史一次性诊断（dPL/MOPEX45/simhyd/vic/Flex 轨道，活跃 dPL 工作仍可能使用），见 `scripts/diagnostics/README.md`；
- **新增 canonical 入口**：`scripts/{run_continuation.sh, validate_full300_config.py, freeze_model_version.py, consolidate_final_checkpoints.py, evaluate_ic_aligned_gen300.py}`、`src/checkpoint_guard.py`、`frozen_versions/`、`docs/canonical_pipeline_inventory.csv`、`docs/superseded_results_registry.csv`；
- **superseded 结果**：见 `docs/superseded_results_registry.csv`（all36_dpl_gap_diagnosis_20260812 等 IC 列已废弃，禁止读取）。
