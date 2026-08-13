# Legacy Cleanup Report — benchmark 收敛与 canonical pipeline 冻结

**日期**：2026-08-13
**范围**：`project/benchmark` — Full300 CMA-ES IC 训练/评价代码收敛与 canonical 冻结清理
**依据文档**：`docs/full300_cmaes_training_runbook.md`（唯一权威 Runbook）

---

## 1. 最终 canonical training/evaluation pipeline 由哪些入口组成？

单一链路（全部相对 `project/benchmark/`）：

| 步骤 | 唯一入口 |
| :--- | :--- |
| config | `configs/full_run_10starts_300gen_warm1980_1981x5.yaml`（+ `configs/default.yaml` base） |
| validate | `scripts/validate_full300_config.py --config <cfg> --manifest <manifest>` |
| freeze | `scripts/freeze_model_version.py --name <run> --config <cfg>` |
| deploy | `scripts/deploy_to_ssh.py` |
| train | `scripts/run_36model_benchmark.py --model <m|all> --run-id <RUN_ID> --config <cfg>`（memory-aware chunk fallback；resume 内置） |
| orchestrate | `bash scripts/run_full_benchmark.sh <RUN_ID>`（train-all + eval 一键） |
| continuation | `bash scripts/run_continuation.sh <RUN_ID> [MODEL...]` |
| final set | `scripts/consolidate_final_checkpoints.py --checkpoint-root <final> [--create --source ...]` |
| canonical eval | `scripts/evaluate_benchmark_metrics.py --checkpoint-root <final> --config <cfg> --output-dir results/...` |
| aligned eval | `scripts/evaluate_ic_aligned_gen300.py --ckpt-root <final> --out results/all36_ic_gen300_aligned_<date>` |
| Chapter-4 rebuild | `scripts/diagnostics/reselect_dpl_trainloss_eval.py` + `scripts/diagnostics/rebuild_all36_diagnosis_trainloss.py` |

dPL（Framework 2）入口保持独立轨道（`run_dpl_benchmark_*`、`launch_*`），与 IC canonical 不混淆。

## 2. 哪些 canonical 源码原本只在远程/部署快照，现在如何纳入本地？

| 文件 | 来源 | 本地处理 |
| :--- | :--- | :--- |
| `scripts/validate_full300_config.py` | 部署快照 `experiments/cmaes_36models/scripts/` | 复制并适配路径（ROOT→BENCHMARK_ROOT），验证通过 |
| `scripts/freeze_model_version.py` | 同上 | 复制并适配路径，验证通过 |
| `scripts/run_continuation.sh` | 对应远程 `run_full300_remaining_continuation.sh` | 本地新写（调用本地训练入口） |
| `scripts/consolidate_final_checkpoints.py` | 无（远程为手工符号链接） | 本地新建（可生成/校验 final 集） |
| `frozen_versions/cmaes36_full300_20260729.json` | 部署快照 `frozen_versions/` | 复制为本地 provenance（hash 为 07-29 冻结状态） |
| `src/*`（16 模块） | 快照 `experiments/cmaes_36models/src/` | **14/16 逐字节一致，未复制**（hash/diff 证据）；`data_selection.py`/`model_registry.py` 为本地扩展版（canonical 超集），未替换 |
| `configs/full_run_10starts_300gen_warm1980_1981x5.yaml`、`default.yaml` | 快照 configs/ | **逐字节一致，未复制**（diff 证据） |
| `scripts/run_36model_benchmark.py` | 对应远程 `run_36model_pilot.py` | 本地已有等价生产路径（diff 仅路径/legacy_data），未复制 |

远程 `evaluate_full300_metrics.py` 与本地 `evaluate_benchmark_metrics.py` 等价（本地版已验证逐位复现 canonical），不重复复制。

## 3. 删除了哪些旧训练/评价/诊断入口？为什么安全？

实际删除（`git rm`，git 可恢复）：

| 文件 | 删除理由 |
| :--- | :--- |
| `scripts/diagnostics/analyze_dpl_vs_ic_comparison.py` | 0 引用（孤儿）；旧 IC-vs-dPL 对比分析，已被 aligned 诊断完全替代；名称易与 canonical 混淆；git 历史可恢复 |

未删除但明确隔离/标记的：
- `scripts/diagnostics/` 其余 84 个历史一次性诊断（dPL/MOPEX45/simhyd/vic/Flex 轨道）：**保留**——dPL/mopex45 轨道仍活跃，脚本有 provenance 价值；目录内新增 `README.md` 明确"不属 canonical"。
- `scripts/diagnostics/round13_finalize.py`（旧验证集 KGE 选择）：**DEPRECATED 头标记**（选择泄漏规则，仅供历史追溯）。

## 4. 哪些历史文件保留但被标记 deprecated？为什么？

| 文件 | 标记 | 原因 |
| :--- | :--- | :--- |
| `scripts/diagnostics/round13_finalize.py` | DEPRECATED（头注释） | 旧 dPL 验证集 KGE 选择（1995–2010 选择泄漏），被 `reselect_dpl_trainloss_eval.py`（train-loss 选择）替代 |
| `frozen_versions/cmaes36_full300_20260729.json` | 保留（provenance） | 07-29 冻结状态；其中 flux/mopex.py 等 hash 早于 08-12 MOPEX wrapper 修复，不能对当前代码 re-validate（预期行为） |
| `scripts/diagnostics/*.py`（84 个） | 保留（目录级 README 标记） | dPL/MOPEX45 轨道活跃，provenance 需要 |

## 5. generation=300 / manifest guard 如何防止 pilot checkpoint 再次被误用？

新增 `src/checkpoint_guard.py::validate_canonical_checkpoint()`，接入所有 canonical/aligned 评价入口：

1. 目录名 == model 名；
2. 必须存在 `DONE` 标记；
3. 只接受 `chunk_*_gen_300.pt`；若只有 gen<300（如 pilot/gen-30），**报错并提示"疑似 pilot/intermediate checkpoint；canonical evaluation 要求 gen-300；拒绝静默降级"**；
4. 合并所有 chunk 后 basin 覆盖必须 == 531（无重叠）——拒绝截断/partial 下载；
5. latent 维度 == registry 维度（参数 schema 一致性）。

接入点：`evaluate_benchmark_metrics.py`（canonical train/test）、`evaluate_ic_aligned_gen300.py`（aligned）。
`consolidate_final_checkpoints.py` 可对整个 final 集独立校验（36 模型 × gen-300 × 531 流域）。
**测试结果**：gen-30 拒绝 ✓、128-basin partial 拒绝 ✓、no-DONE 拒绝 ✓、gen-300 接受 ✓。
resume/continuation 允许中间 generation（训练语义），但 final evaluation 一律走 guard（宽松搜索不进入评价）。

## 6. MOPEX1–3 清理后是否仍精确复现 canonical 结果？

**是（逐流域 100% 精确）**，清理与 guard 接入后重跑：

| 模型 | 复现 test median | canonical test median | median_abs | exact_frac |
| :--- | :---: | :---: | :---: | :---: |
| mopex1 | 0.5547 | 0.5547 | 0 | 1.000 |
| mopex2 | 0.7118 | 0.7118 | 5.4e-10 | 1.000 |
| mopex3 | 0.7047 | 0.7047 | 5.6e-10 | 1.000 |

## 7. simhyd/wetland/Flex 等特殊模型是否仍使用正确 final checkpoint？

**是**。`checkpoints/full300_final_36models/` 为 2026-07-30 建立的 canonical final 集（符号链接：simhyd→simhydRef 重跑、wetland→wetlandA 重跑、flexb/flexi→原始运行、flexis 及 mopex 等→continuation）。清理后复现：

| 模型 | 复现 | canonical | 说明 |
| :--- | :---: | :---: | :--- |
| simhyd | 0.6151 | 0.6151 | exact_frac 0.996（重跑版本，非旧本地副本） |
| wetland | 0.5482 | 0.5482 | exact_frac 0.787（max diff ~1e-7，浮点容差） |
| flexis | 0.6153 | 0.6153 | 9-chunk 正确组合 |
| gr4j / hbv96 | 0.6507 / 0.7233 | 0.6507 / 0.7233 | exact |

`consolidate_final_checkpoints.py` 校验通过（36 模型、gen-300、531 流域）。

## 8. 哪些旧 results 已标记 superseded？

见 `docs/superseded_results_registry.csv`，关键项：

| 旧结果 | 替代 | safe |
| :--- | :--- | :---: |
| `results/all36_dpl_gap_diagnosis_20260812/` | `results/all36_ic_gen300_aligned_20260812/` + `results/all36_dpl_gap_diagnosis_20260812_trainloss/` | **no**（IC 列 pilot/gen-30） |
| `results/six_models_aligned_audit_20260812/` | 同上 | **no** |
| `results/ic_vs_dpl_aligned_1995_2010_recomputed.csv` | `all36_aligned_gen300_ic.csv` | **no** |
| `results/dpl_gap_diagnosis_20260812/` | `all36_dpl_gap_diagnosis_20260812_trainloss/` | no（旧选择规则） |
| `results/dpl_round13_20260805/final/*` | train-loss 选择版 | no（选择规则） |
| `results/full300_final_36models_evaluation/` | 无（canonical） | **yes** |
| `results/all36_ic_gen300_aligned_20260812/` | 无（canonical） | **yes** |

未修改任何旧 CSV 内容（不改写伪装新结果），仅登记替代关系。

## 9. 后续如果修改训练时期，唯一正确执行流程是什么？

1. 编辑 `configs/full_run_10starts_300gen_warm1980_1981x5.yaml`（或复制为新 config）：`data.train.start_time/end_time`、`data.test.start_time/end_time`、`warmup.source`（须位于训练期前、`warm_right ≤ train_left`）；
2. `python scripts/validate_full300_config.py --config <new> --manifest <new manifest>`（先冻结：`python scripts/freeze_model_version.py --name <run> --config <new>`）；
3. `bash scripts/deploy_to_ssh.py` 同步；
4. 远程训练（memory-aware）+ continuation（`run_continuation.sh`），或本地 `run_full_benchmark.sh`；
5. `python scripts/consolidate_final_checkpoints.py --checkpoint-root checkpoints/<final> --create --source <run dirs>`；
6. `python scripts/evaluate_benchmark_metrics.py`（canonical）+ `python scripts/evaluate_ic_aligned_gen300.py`（aligned）；
7. `python scripts/diagnostics/reselect_dpl_trainloss_eval.py` + `rebuild_all36_diagnosis_trainloss.py` 重建下游。
8. **注意**：若新训练期覆盖 aligned 窗口，评价窗口需同步调整或使用 held-out（数据至 2014-09-30，可用 2010-10-01..2014-09-30）。

## 10. 当前 working tree 中是否还有与本任务无关、未触碰的用户修改？

**没有**。任务开始时 `git status` 仅含本任务早前新增的 3 个未跟踪文件（`docs/`、`evaluate_ic_aligned_gen300.py`、`rebuild_all36_diagnosis_trainloss.py`、`reselect_dpl_trainloss_eval.py` —— 均属本收敛工作链），无其他用户未提交修改。清理后 `git status` 见下（新增/删除均为本任务产物）。

---

## 附：清理后验证清单（Validation after cleanup）

| 检查 | 结果 |
| :--- | :--- |
| 36 模型 registry 解析 | ✅ `audit_registry()` 通过（validate_full300_config 输出 registry_models=36） |
| canonical config 加载 | ✅ `load_resolved_config` + `validate_full_run_config`（starts=10, generations=300） |
| final checkpoint 集完整 | ✅ `consolidate_final_checkpoints.py`：36 模型 × gen-300 × 531 流域 |
| gen-300 guard 生效 | ✅ gen-30 拒绝（含 pilot 提示）、partial 拒绝、no-DONE 拒绝 |
| manifest 校验 | ✅ `freeze_model_version` + `validate_full300_config` 端到端通过（当前代码自洽） |
| Flex chunked 组合 | ✅ flexis（9 chunk）正确组合，复现 0.6153 |
| simhyd/wetland 用最终检查点 | ✅ simhydRef/wetlandA 版本复现 canonical |
| MOPEX1–3 复现 | ✅ 100% exact |
| simhyd/wetland/flexis/gr4j/hbv96 复现 | ✅ 全部命中 canonical |
| canonical eval 入口 smoke | ✅ evaluate_benchmark_metrics（mopex1 train/test max|diff|=0.0） |
| aligned eval 入口 | ✅ 迁移后 evaluate_ic_aligned_gen300（mopex1-3 与全量运行一致，diff=0） |
| Runbook 路径存在性 | ✅ 全部新路径已验证存在 |

**未执行**：完整 300-generation 训练（本任务明确不重训）；MOPEX4/5 phase-fix（另案）。
