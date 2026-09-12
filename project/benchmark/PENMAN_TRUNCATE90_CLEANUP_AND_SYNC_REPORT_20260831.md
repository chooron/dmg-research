# Penman `truncate:90` Cleanup & Sync Report (2026-08-31)

## 1. Executive conclusion

- **`truncate:90` dead config 是否已从 active pipeline 清理？** 是（本地）。所有 active code 中的 `truncate:90` 特殊分支已移除；`warmup_grad_mode` 从"静默忽略"变为 **fail-fast 校验键**（非 `"detach"` 一律显式 `raise`）；active protocol 文档已改写为正确表述。剩余 `truncate:90` 字符串仅存在于回归测试断言（冻结 manifest 历史记录 + fail-fast 行为）、说明性注释与 fail-fast 错误消息中。
- **Canonical 运行是否被干扰？** 否。本轮未启动任何训练，未触碰 `best.pt`/checkpoint/manifest/ledger/log。由于远程主机不可达，远程亦未做任何修改（见 §8）。
- **Remote 是否已 apply 或只是 staged？** **均未执行** —— 远程 SSH 握手被拒（服务器不可达）。已完成本地 staging bundle（`results/penman_truncate_cleanup_20260831/remote_staging_bundle/`），内容与 §15 要求的 pending 目录一致，随时可按 `APPLY_AFTER_CANONICAL.md` 同步。见 §8 的 honest status。

## 2. Active references removed（逐文件）

| 文件 | 类型 | 修改 |
|---|---|---|
| `scripts/canonical_v2/run_canonical_v2_model.py` | ACTIVE_CODE | docstring 移除 `truncate:90` 表述（Penman 改为 warmup-length exception + full backprop）；`warm_mode = "detach"` 恒值；新增 job-config fail-fast（声明非 `detach` 即 `RuntimeError`） |
| `scripts/canonical_v2/run_canonical_v2_queue.py` | ACTIVE_CODE | docstring "Protocol Contract" 改写（35 模型 730/365 full backprop；Penman 365/365 full backprop） |
| `scripts/diagnostics/k_full_retrain.py` | ACTIVE_CODE | `warm_mode = "truncate:90" if penman...` → 恒 `"detach"`（含注释）；`contract.json` 的 `penman_warmup` 字段改写为准确描述 |
| `scripts/ablation/ablation_runner.py` | ACTIVE_CODE | `warm_mode` 特例 → 恒 `"detach"`（含注释） |
| `scripts/ablation/evaluate_2x2_warmup.py` | ACTIVE_CODE | 同上 |
| `scripts/diagnostics/round12_edge_probe.py` | ACTIVE_CODE | 同上 |
| `scripts/diagnostics/e2_boundary_kink.py` | ACTIVE_CODE | `"full"` ×2 → `"detach"`（注释说明 full 从未实现） |
| `scripts/diagnostics/warmup_gradient_contract.py` | ACTIVE_CODE | `MODES=("detach",)`；t5 改为单模式确定性检查；t7/t7_state_init 的全模式别名注释；t8 单模式 |
| `scripts/diagnostics/full_model_fd_warmup_modes.py` | ACTIVE_CODE | docstring 改写；仅跑 `"detach"`，JSON 只写 detach |
| `src/model_registry.py` | ACTIVE_CODE | 新增 `_validate_warmup_grad_mode`（`ValueError` fail-fast；详见 §5） |
| `tests/test_canonical_v2_runner.py` | TEST | manifest 测试重构（冻结记录 + 历史注释）；新增 runner fail-fast 测试；smoke 注释更新 |
| `tests/test_penman_truncate90_cleanup.py` | TEST（新增） | 防回归测试集（§5） |
| `tests/conftest.py` | TEST（新增） | sys.path bootstrap（沿用 dmotpy/conftest.py 惯例） |
| `results/dpl_canonical_v2_20260831/CANONICAL_V2_PROTOCOL.md` | ACTIVE_PROTOCOL_DOC | Penman Exception 行改写（365/365 full backprop；W2 证据表述；`truncate:90`/错误归因删除）；§2 例外清单同步 |

**保留不动（HISTORICAL_ARTIFACT）**：`canonical_manifest.yaml`（冻结启动记录，含 penman `warmup_grad_mode: truncate:90`）、`dpl_full_retrain_20260813/`（epochs/health/status/contract/README/TRAINING_LOG）、`dpl_flexb_retrain_20260830/`、`dpl_protocol_ablation_v2_20260831/`（ledger、PHASE_*、runs/*/config.yaml）、`docs/training_log_dpl_full_retrain_20260813.md`。
**保留不动（AUDIT_REPORT）**：`PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md`、`dpl_training_rule_audit_20260831.md`、`DMOTPY_CURRENT_STATE_AUDIT_20260831.md`、`BENCHMARK_CURRENT_STATE_AUDIT_20260831.md`、`CURRENT_PROJECT_DPL_AUDIT_SYNTHESIS_20260831.md`、`DPL_NOGPU_PHASE_FINAL_REPORT_20260831.md`。

## 3. Historical references retained（为什么不改历史）

历史产物是已完成/进行中 run 的 provenance 记录：frozen manifest 记载了该 run 实际启动时的配置（含 no-op 标签）；epochs/health/status CSV、contract、ledger、训练日志照实记录了当时的字段。按审计原则"保留原文件作为历史 provenance，必要时新增 correction note 而非重写原始结果"，这些文件一律未动；新增 `PENMAN_TRUNCATE90_CORRECTION_NOTE_20260831.md` 说明其含义与影响面（§10 第 7 点要求）。

## 4. Canonical protocol correction

新旧表述对比：

```text
旧（2026-08-31 早版 CANONICAL_V2_PROTOCOL.md）:
  Penman Exception: warmup = 365d, scored = 365d (truncate:90 mode)
  "Phase W2 proved 730d warmup degrades Penman under truncate:90 backpropagation"

新：
  Penman Exception: warmup = 365d, scored = 365d (full backpropagation,
    identical gradient semantics to the 35 standard models)
  "Phase W2 showed a negative response of Penman to extending training warmup
    to 730d (Δ median KGE = −0.0173, Δ Q25 = −0.0370 at matched update budget).
    The mechanism of this response was not isolated. Pre-registered as a
    model-specific warmup-length exception."
```

正确机制表述：**UNRESOLVED**（不推测、不写 TBPTT）。全协议明确：36 模型梯度语义一致 = warmup no_grad + detach，scored period full backprop。

## 5. Silent-config prevention（防回归）

1. `src/model_registry.py::_validate_warmup_grad_mode`：任何非 `"detach"` 的 `warmup_grad_mode`（`truncate:N` / `full` / `state_init` 等）→ `ValueError: Unsupported warmup_grad_mode: ...`，错误消息引用审计报告。所有 `build_model`/`model_config` 路径必经此校验（未来新增模式必须同时实现并测试，否则直接 fail fast）。
2. `run_canonical_v2_model.py`：job config 若声明非 `"detach"` 的 `warmup_grad_mode` → `RuntimeError`（在创建 run_dir / 加载数据之前触发）。
3. 回归测试（`tests/test_penman_truncate90_cleanup.py` + `test_canonical_v2_runner.py`）：
   - fail-fast × 5 模式（model_config）+ × 2（build_model）+ runner job-config；
   - `detach` 是唯一合法值（含默认值）；
   - Penman 例外 = warmup-length only（365/365 + detach）；
   - 冻结 manifest 历史记录保留断言（防误删）；
   - 前向+梯度与清理前基线逐位一致测试。

## 6. Numerical equivalence

- 清理前基线（`pre_cleanup_truncate90.npz` 与 `pre_cleanup_detach.npz`）逐位一致 → 证实历史标签是 no-op。
- 清理后 `detach` 复算与基线 **bit-exact**（loss / q sha256 / grad sha256 全部相同，见 `LOCAL_TEST_REPORT.md` §3）。
- 结论：**cleanup 不改变任何 forward/backward 数值行为**（代码只删标签分支与加校验；模型、训练循环、loss、optimizer 未动）。已完成的/进行中的 canonical v2 结果不因清理失效，无需重跑。

## 7. Local tests

| 检查 | 结果 |
|---|---|
| `py_compile`（全部改动模块+测试） | PASS |
| `git diff --check` | PASS |
| pytest 轻量（14 tests：fail-fast/基线一致性/manifest/runner guard） | **14 passed** (5.9s) |
| pytest CPU smoke（1-epoch gr4j 730/365 + penman 365/365 全流程） | **1 passed** (198.9s) |
| `pytest --collect-only`（全部 26 tests） | 26 collected, 无 import 错误 |
| 数值等价（清理前后 baseline 对比） | **bit-exact**（sha256 一致） |
| 引用清扫验证（rg） | active code 无 `truncate:90` 行为分支；仅测试断言/注释/错误消息 |

## 8. Remote sync state

**`UNREACHABLE — NOT SYNCED`（honest status；不在任务预设的两个枚举内）**：

- 远程 `autodl`（`connect.nmb2.seetacloud.com:33933`）在多次 bounded 尝试中均于 SSH 握手阶段被断开（`kex_exchange_identification: Connection closed by remote host`）——实例大概率处于停机/不可达状态。
- 因此：无法检查 `canonical_v2_status.py` / tmux / pgrep，无法确认 canonical v2 是否仍在运行，**未做任何远程修改**（既未 stage 也未 apply——远程不可达时二者都不可执行，且绝不冒险中断可能运行的 canonical 实验）。
- 已就绪的同步物：`results/penman_truncate_cleanup_20260831/remote_staging_bundle/`（`PATCH.diff`、`changed_files/` 全量副本、`TEST_REPORT.md`、`APPLY_AFTER_CANONICAL.md`、`SOURCE_SHA.txt`、`SOURCE_STATUS.txt`、`SOURCE_DIFF_SHA256.txt`、两个 pre-cleanup 基线 npz、correction note、audit 报告）。
- 恢复连接后按 `APPLY_AFTER_CANONICAL.md` 执行：先保存远程 provenance → 确认 DONE=36/RUNNING=0/QUEUED=0 → 校验并应用 → CPU 测试 → 引用验证；若 canonical 仍在运行则只落到 `/root/dmg-research_pending/penman_truncate_cleanup_20260831/`。

## 9. Canonical v2 provenance protection

- 本地：未触碰冻结 manifest、ledger、任何 `best.pt`/checkpoint/log；
- 远程：**零修改**；`APPLY_AFTER_CANONICAL.md` Step 1 强制在任何 apply 前保存 `REMOTE_SOURCE_SHA/STATUS/DIFF` + canonical run provenance 副本，确保 36-model run 的 source snapshot 可事后证明；
- 冻结 manifest 中的 `warmup_grad_mode: truncate:90` 字段保留为启动记录（correction note §1 明确列出），并新增测试防止它被意外改写。

## 10. Git status

### 本地（`/home/jingxin/code/dmg-research`，HEAD = `3caca37`）

本轮修改（tracked）：
```text
 M project/benchmark/scripts/diagnostics/k_full_retrain.py   （含清理前已存在的用户改动）
 M project/benchmark/src/model_registry.py
```
本轮新增/修改（untracked，原本即未跟踪的区域）：
```text
 M project/benchmark/scripts/canonical_v2/run_canonical_v2_model.py
 M project/benchmark/scripts/canonical_v2/run_canonical_v2_queue.py
 M project/benchmark/scripts/ablation/ablation_runner.py
 M project/benchmark/scripts/ablation/evaluate_2x2_warmup.py
 M project/benchmark/scripts/diagnostics/{round12_edge_probe,e2_boundary_kink,warmup_gradient_contract,full_model_fd_warmup_modes}.py
 M project/benchmark/results/dpl_canonical_v2_20260831/CANONICAL_V2_PROTOCOL.md
?? project/benchmark/PENMAN_TRUNCATE90_CORRECTION_NOTE_20260831.md
?? project/benchmark/tests/conftest.py
?? project/benchmark/tests/test_penman_truncate90_cleanup.py
?? project/benchmark/tests/test_canonical_v2_runner.py（本轮重建）
?? project/benchmark/results/penman_truncate_cleanup_20260831/（artifacts + staging bundle）
```
其余 pre-existing 改动（dmotpy VIC/doy、hydrodiag supplement 等）未触碰。完整快照在 `results/penman_truncate_cleanup_20260831/LOCAL_SOURCE_STATUS.txt`。

### 远程（`/root/dmg-research`）

不可达，未获取；apply 前由 `APPLY_AFTER_CANONICAL.md` Step 1 强制保存。

### 本轮未执行清单

- ❌ 未启动任何训练（本地/远程）；未重跑 Penman / 36 models / W2；
- ❌ 未停止/重启/污染任何 canonical v2 job；
- ❌ 未使用 GPU（全部测试 `CUDA_VISIBLE_DEVICES=""`）；
- ❌ 未修改任何 HISTORICAL_ARTIFACT / AUDIT_REPORT 原始记录。