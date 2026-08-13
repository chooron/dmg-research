# scripts/diagnostics — Historical one-off diagnostics

**这些脚本不属于 canonical Full300 CMA-ES IC 训练/评价管线。**

本目录包含 dPL / MOPEX45 / simhyd / vic / Flex 等主题的一次性诊断、审计与临时验证脚本，
仅作历史追溯（大部分结果已落盘在 `results/`，脚本 git 可恢复）。

## Canonical 入口（在上级 `scripts/`，不要与本目录混淆）

| 环节 | 入口 |
| :--- | :--- |
| 训练 | `scripts/run_36model_benchmark.py`（memory-aware；resume 内置） |
| 编排（train-all + eval） | `scripts/run_full_benchmark.sh` |
| 续跑 | `scripts/run_continuation.sh` |
| config 校验 | `scripts/validate_full300_config.py` |
| manifest 冻结 | `scripts/freeze_model_version.py` |
| final checkpoint 收敛/校验 | `scripts/consolidate_final_checkpoints.py` |
| canonical train/test 评价 | `scripts/evaluate_benchmark_metrics.py`（含 gen-300 guard） |
| aligned 1995–2010 评价 | `scripts/evaluate_ic_aligned_gen300.py`（含 gen-300 guard） |
| Chapter-4 下游重建 | `scripts/diagnostics/reselect_dpl_trainloss_eval.py`、`scripts/diagnostics/rebuild_all36_diagnosis_trainloss.py` |

完整说明见 `docs/full300_cmaes_training_runbook.md`。
