# Penman `truncate:90` Correction Note (2026-08-31)

**Scope**: 记录 `truncate:90` 从 active protocol 中移除的事实，及其对历史结果的非影响。
**依据**: `project/benchmark/PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md`（完整证据链）。

## 1. 哪些历史文档/配置曾写 `truncate:90`

以下为**历史记录，原样保留**（不修改、不删除，作为 provenance）：

| 记录 | 位置 | 说明 |
|---|---|---|
| Frozen canonical v2 manifest（36 job 的启动记录） | `results/dpl_canonical_v2_20260831/canonical_manifest.yaml` | penman job 声明 `warmup_grad_mode: truncate:90`；这是该 run 实际启动时的记录 |
| Canonical retrain 产物 | `results/dpl_full_retrain_20260813/auto100/{contract.json, epochs.csv, status.csv, health.csv}`、`README.md`、`TRAINING_LOG.md` | 记录的 `warmup_grad_mode` 列/字段 |
| flexb 续训产物 | `results/dpl_flexb_retrain_20260830/auto100/contract.json` | 同上 |
| 训练日志 | `project/benchmark/docs/training_log_dpl_full_retrain_20260813.md` | "例外:penman 用 truncate:90" |
| Ablation ledger/report | `results/dpl_protocol_ablation_v2_20260831/`（ledger、PHASE_*_REPORT、manifest、runs/*/config.yaml） | 各 run 的 config 记录 |
| 早期审计文档 | `results/dpl_training_rule_audit_20260831.md`、`dmotpy/DMOTPY_CURRENT_STATE_AUDIT_20260831.md`、`project/benchmark/CURRENT_PROJECT_DPL_AUDIT_SYNTHESIS_20260831.md`、`BENCHMARK_CURRENT_STATE_AUDIT_20260831.md`、`DPL_NOGPU_PHASE_FINAL_REPORT_20260831.md` | 如实记录当时状态 |
| 已删除目录的残留引用 | `scripts/diagnostics/warmup_gradient_contract.py` 早期 `MODES` 定义（7 月梯度审计期） | 本次已在该脚本内以注释形式保留说明 |

## 2. 审计证明该字段为 no-op

- `warmup_grad_mode` 作为 config key 传入 `HydrologyModel`，但 `_load_config()` **从不读取它**。
- 全仓库（当前代码 + git 全历史）不存在解析 `truncate:N`、按 N 天 detach 或 chunked backward 的实现。
- 因此 `"truncate:90"` / `"truncate:180"` / `"full"` / `"state_init"` 与 `"detach"` 在数值上**逐位相同**（CPU FP64 实测，`results/penman_truncate_cleanup_20260831/` 基线：truncate:90 与 detach 的 forward q、gradient、loss 的 sha256 完全一致）。

## 3. Penman 实际训练语义（从未改变）

```text
warmup 365d: torch.no_grad() 运行 + 状态 detach
scored 365d: 全程建图（full computational graph）
loss: 1 - KGE(365d scored)
backward: 单次 full backprop
optimizer: 每 batch 一次 AdamW step
```

与其他 35 个模型完全一致的梯度语义。**不存在任何截断**。

## 4. 历史数值结果不受影响

因为 `truncate:90` 从未改变计算，删除该标签**不改变任何数值训练行为**：

- `dpl_full_retrain_20260813`（35/36 模型 canonical dPL 训练集）结果有效；
- `dpl_canonical_v2_20260831` 36-model run（进行中或已完成部分）结果有效；
- `dpl_protocol_ablation_v2_20260831`（Phase H/W2 等）结果有效。
- **无需重跑 Penman / 36 models / W2**。

## 5. W2 Penman degradation 不能归因于 truncation

- W2 2×2 四格（T365/E365, T365/E730, T730/E365, T730/E730）的 `warm_mode` 全为 `"truncate:90"` 且该标签 inert，因此四格差异不可能由 truncate 造成。
- 该 degradation 是 **730d 训练 warmup 长度** 在真实（full backprop）训练动态下的真实效应。
- 任何"under truncate:90 backpropagation"的因果表述均为**错误归因**（如 `CANONICAL_V2_PROTOCOL.md` 2026-08-31 版的旧表述，已于本轮修正）。

## 6. Penman 365d warmup 例外仍由 W2 结果支持

- W2 证据：Penman Δ(T730−T365)@E730 = **−0.0173 median / −0.0370 Q25**（matched update budget 250）→ `WARMUP_HARM`。
- 该例外是 **warmup-length exception**（365d warmup + 365d scored），与梯度截断无关，保留。
- 机制：`UNRESOLVED`（未隔离，不推测）。

## 7. 从何时起 active protocol 删除该标签

- **2026-08-31（本地）**：active code 全部移除 `truncate:90` 特殊分支；`src.model_registry` 新增 fail-fast：`warmup_grad_mode != "detach"` 一律显式 `raise ValueError`；canonical v2 runner 对 job config 中声明的非 `"detach"` 值显式 `raise RuntimeError`。
- 新产生的 run/config 不应再出现 `truncate:90`。
- 本 note 对应的清理报告：`project/benchmark/PENMAN_TRUNCATE90_CLEANUP_AND_SYNC_REPORT_20260831.md`。