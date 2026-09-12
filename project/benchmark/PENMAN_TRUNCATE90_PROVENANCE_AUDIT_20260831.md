# Penman `truncate:90` Provenance & Necessity Audit (2026-08-31)

**Audit scope**: read-only provenance audit + CPU-only micro-tests. NO GPU, NO training, NO canonical-protocol modification.
**Evidence grades**: `VERIFIED_FROM_CODE` / `VERIFIED_FROM_GIT_HISTORY` / `VERIFIED_FROM_ARTIFACT` / `SUPPORTED_BY_CPU_TEST` / `INFERENCE` / `UNRESOLVED`.

---

## 1. Executive conclusion

1. **为什么有 `truncate:90`？** —— `truncate:90` 是 2026-07 末 ~ 08-12 期间 dPL gradient-audit 工作区里引入的一个**字符串标签**，随 `16a1d6c`（2026-08-12 "chore: sync remaining workspace changes to master"）首次进入 git。没有任何 commit message、报告或日志记录其引入理由；它**从未被任何代码解析或实现**（见 §3）。它是一个从未落地的 TBPTT 假想模式的残留标签（`HISTORICAL_HEURISTIC_WITHOUT_EVIDENCE`）。
2. **为什么只有 Penman？** —— 没有代码或历史依据支持 "Penman 需要特殊梯度处理"。Penman 是 36 模型中**最轻**的模型之一（4 参数 / 3 状态 / 无 UH 卷积 / 每步 0.26 s 的 fleet 最快梯队），历史上没有任何 Penman 专属 OOM / NaN / 梯度爆炸记录。唯一“特殊”的是标签本身——所有 runner 都把它写成了 `penman and mapping=="auto"` 条件分支。`INFERENCE`：该标签可能源于 7 月底 warmup-gradient 审计中“penman 的 GAM 参数 FD/autograd 对比”测试（`warmup_gradient_contract.py` T6），但没有文档证实。
3. **90d 是否有证据？** —— 无。90 ≈ 365/4（季度），180 ≈ 365/2，是最可能的启发式来源（`INFERENCE`）；没有消融（30/60/90/180 从未比较过）、没有 OOM 事件、没有外部文献要求（见 §4.3）。**更重要**：由于代码从未实现 truncation，90 这个数字在当前 canonical 训练中**没有任何行为效果**。
4. **它现在是否必要？** —— **不必要，且当前是 no-op**。`warmup_grad_mode` 配置键在 `HydrologyModel._load_config` 中根本不被读取；`warm_mode="truncate:90"` 与 `"detach"` 产生完全相同的 forward 与完全相同的梯度（CPU FP64 下 bit-exact，§7）。Penman 的 canonical dPL 训练实际执行的就是与其他 35 个模型完全一致的 "warmup no_grad + detach @365d，scored 365d 全程建图"。

---

## 2. Provenance timeline

| 时间 | commit/事实 | 内容 | 理由/证据 |
|---|---|---|---|
| ~2026-07-31（workspace，未入库） | `dpl_warmup_contract_20260731` / `dpl_gradient_evidence_20260731_epoch` 目录名（被 `warmup_gradient_contract.py` 引用） | 梯度审计期定义 `MODES=("detach","truncate:90","truncate:180","full")` | `VERIFIED_FROM_GIT_HISTORY`（目录已删除，仅脚本内字符串残留）；引入理由缺失 |
| **2026-08-12** | **`16a1d6c`** "chore: sync remaining workspace changes to master" | `truncate:90` 字符串首次进入 git：`k_full_retrain.py`（`warm_mode = "truncate:90" if model=="penman" and arm=="auto100"`）、`warmup_gradient_contract.py`、`round12_edge_probe.py`；同时 `dpl_gradient_evidence.py`/`dpl_third_round_diagnostics.py` 入库 | `VERIFIED_FROM_GIT_HISTORY`；bulk-sync commit，无理由说明 |
| 2026-08-12 | `215852c` "36-model CMA-ES benchmark and dPL framework"（同日） | dPL 框架正式化，`warmup_grad_mode` 作为配置键传入 `build_model` → `HydrologyModel` config；**无读取端** | `VERIFIED_FROM_CODE` |
| 2026-08-13 | `9b0d963` "docs: add training log for dpl_full_retrain_20260813" | 文档首次记载 "例外:penman 用 truncate:90" | 记录事实，未给理由 |
| 2026-08-16 | `b88f88d`（k_full_retrain health retrain） | 修改 early-stop/telemetry，`truncate:90` 行未动 | `VERIFIED_FROM_GIT_HISTORY` |
| 2026-08-31 | `dpl_training_rule_audit_20260831.md`、`CANONICAL_V2_PROTOCOL.md` | 将 "Penman: 365d warmup + 365d scored, truncate:90" 定为 "Pre-registered Exception"，理由写为 "Phase W2 proved 730d warmup degrades Penman under truncate:90 backpropagation" | 事后(post-hoc)叙述，且归因有误（见 §6） |

**结论**：`truncate:90` 从第一天起就是一个**从未实现的标签**。全部历史中不存在任何解析 `truncate:N`、按 N 天 detach、或 chunked backward 的实现（`git log -S'truncate:' --all` 仅命中 `16a1d6c` / `9b0d963`，均为字符串出现）。

---

## 3. Exact code semantics（真实语义）

**关键事实（`VERIFIED_FROM_CODE`）**：`warmup_grad_mode` 从未被消费。

- `project/benchmark/src/model_registry.py:89,142`：`model_config()` / `build_model()` 只是把 `warmup_grad_mode` 字符串塞进 config dict。
- `dmotpy/models/hydrology_model.py:_load_config()`（L116-142）：`attrs` 列表含 `warm_up`、`warm_up_states`、`variables`… **不含 `warmup_grad_mode`**。
- `HydrologyModel._run_model()`（L356-400）与 `_run_warmup()`（L319-330）：唯一路径 = 前 `warm_up` 步在 `torch.no_grad()` 下运行 → `tuple(state.detach() ...)` → 余下步正常建图 → 单次 `loss.backward()`。对全部 36 模型（含 penman）完全相同。

因此 `warm_mode` 的所有取值（`detach` / `truncate:90` / `truncate:180` / `full`）在今天的代码里行为**逐位相同**。

**如果 `truncate:90` 按名字实现，它原本应该做什么**（基于模式命名与 TBPTT 惯例的语义重构，`INFERENCE`，非现状）：

```text
# 现状（所有 mode 相同）——canonical 窗口 730d、warmup 365d、scored 365d：
forward:
  t = 0 .. 364      no_grad() 运行, 状态 detach      # warmup 不建图
  t = 365 .. 729    建图运行, 状态连续传递            # scored 全程建图
loss = 1 - KGE(q[365:730], y[365:730])               # loss 覆盖完整 365d scored
backward: 单次 backward 穿越全部 365 scored 步
optimizer: 每 batch 1 次 step（k_full_retrain.py:169 steps/epoch）
# 90 天处：什么都不发生；状态不重置（水文状态连续）；无每 90d 单独 backward/step；
# 最后 5 天（365 = 4×90 + 5）无特殊处理；90 是按 tensor step（日）计数，非 calendar 概念。

# 若按"命名语义"实现（未实现）：
for chunk in [90, 90, 90, 95]:                     # scored 内切 4 段
    run chunk 建图; chunk 边界处 states = states.detach()
    最终在 365d 输出上算同一 loss, 单次 backward
```

回答审计问题：前向状态**连续**；90 天处**不**重新初始化水文状态（若实现只 detach 计算图）；loss 覆盖完整 365d；**每 90 天不**单独 backward/step；optimizer 每 batch 一次；最后 5 天无特殊处理。90 按日步数计。

---

## 4. Historical evidence

### 4.1 训练异常（OOM / NaN / 梯度失败）
- canonical retrain `dpl_full_retrain_20260813/auto100`：全部 36 模型 2401 条 epoch 记录中 **`train_nonfinite_cumulative` / `validation_nonfinite_cumulative` 全部为 0**；penman 69 epochs 无一次非有限事件（`VERIFIED_FROM_ARTIFACT`：epochs.csv/health.csv）。
- scheduler log：`[2026-08-16 01:39] worker3 start penman ... 02:32 end penman rc=0`，无错误（`VERIFIED_FROM_ARTIFACT`：run_20260815/master.log）。
- 全 results 树 grep `out of memory|CUDA error|NaN`：无任何 penman 相关命中（`VERIFIED_FROM_ARTIFACT`）。
- **full 365d graph 从未在 Penman 上失败过**——相反，2026-08-31 的 Phase H 二期（`H02/H08…`，1825d horizon、1460d scored、batch 100、FP32 compile）8 个代表模型全部 DONE，penman 亦在内（`H08_penman_1825` best inner-val med KGE 0.4851）。即**更长的全图 backprop 在 penman 上无 OOM**（`VERIFIED_FROM_ARTIFACT`：ABLATION_LEDGER.md）。
- penman 是 fleet 中最快模型之一（median 0.262 s/step vs flexis 0.728 s/step），无图内存压力特征（`VERIFIED_FROM_ARTIFACT`：epochs.csv）。

### 4.2 90d 消融
- 无任何 30/60/90/180/full 训练消融的历史产物。`warmup_gradient_contract.py` 定义过这些 mode 字符串，但其结果目录 `results/dpl_warmup_contract_20260731` 不存在，且该脚本只是把 mode 字符串传入 build_model（同样是 no-op），从未产出过真实的 truncate 梯度结果（`VERIFIED_FROM_CODE`）。
- 唯一接近消融的是 `full_model_fd_warmup_modes.py`（detach vs full 的有限差分检查），只对比 detach/full，不涉及 90。

### 4.3 外部来源
- 仓库内无参考文献指向 "Penman + TBPTT-90"（`VERIFIED_FROM_CODE`）。MARRMoT 被引为 Penman 公式来源（`m_04 evap_16` 等注释），但 MARRMoT 是模拟框架，不规定训练超参数。
- 有界外部检索（Crossref + arXiv API，2026-08-31）：**不存在**公开文献主张 "Penman dPL 必须 90d truncated BPTT"；TBPTT 文献均为通用方法学（语音识别、神经控制、MRI 等）。`UNRESOLVED` 级别为：无文献支持，但也没有研究公开否定 90d 的价值——因为问题本身从未被检验过。

整体判定：**`HISTORICAL_HEURISTIC_WITHOUT_EVIDENCE`**（任务 §3D 类型）：无工程证据、无消融、无文献，且——比这更严重——**从未实现**。

---

## 5. Penman vs other models（为何被单独处理）

| 检查项 | 结论 | 等级 |
|---|---|---|
| 更长 state-memory？ | 否：3 个串行状态（S1 表层土壤、S2 deficit、S3 汇流），无长滞后结构 | `VERIFIED_FROM_CODE` (`dmotpy/models/core/penman.py`) |
| 更深的 recurrent 图？ | 否：单步 = saturation/split/evap/clamp 等普通 flux，无迭代求解器、无 UH 卷积（对比 flexi/flexis/hbv96 等带 UH 或更多状态的模型） | `VERIFIED_FROM_CODE` |
| 特殊 routing/storage？ | 否：无 implicit dependence；`STATE_INFO[penman]=3`，参数 4（smax, phi, gam, k1） | `VERIFIED_FROM_CODE` |
| 更易 exploding/vanishing/NaN/OOM？ | 无证据：0 非有限事件；H2 1825d 全图无 OOM；0.26 s/step（最快梯队） | `VERIFIED_FROM_ARTIFACT` |
| 参数饱和/零梯度？ | epoch1: gam 零梯度 3.4%、k1 boundary 14.5%；与 fleet 水平相当（vic boundary 60%），无异常 | `VERIFIED_FROM_ARTIFACT`（parameter_gradients.csv） |
| 参数/状态规模 | 4 参数/3 状态——36 模型中最小一档（与 collie1/alpine1/wetland 同级） | `VERIFIED_FROM_CODE` |

**结论**：没有任何模型层证据支持 Penman 需要特殊梯度截断。其"特殊性"完全来自标签本身。`INFERENCE`：最可能的真实动机是 7 月底 gradient-audit 期间对 penman `gam` 参数做过 FD-vs-autograd 专项（`warmup_gradient_contract.py` t6 注释明确写了 "Penman GAM uses the dPL 365+365 slice"），之后该模型被顺手挂了标签——此推断无文档证实。

---

## 6. Phase W2 relation（Penman W2 degradation 与 truncate 的关系）

事实（`VERIFIED_FROM_ARTIFACT`：PHASE_W2_2X2_WARMUP_DECOMPOSITION.csv、PHASE_W2_UPDATE_BUDGET_AUDIT.csv）：

- H1（T365，50 epochs×5 steps = 250 updates）vs W2（T730，target 250 updates，实际 210）——**update budget 严格匹配**，seed/lr/mapping/clip 一致（H16 vs W208：都 auto、lr=1e-3、clip=1.0、AdamW）。
- 四个格子（T365/E365=0.4796, T365/E730=0.4658, T730/E365=0.4566, T730/E730=0.4484）中 **warm_mode 完全相同**：`evaluate_2x2_warmup.py:126` 对 penman 恒为 `"truncate:90"`（且该标签 inert）。所以 2×2 任何格子的差异都不可能由 truncate 造成。
- 由于 truncate:90 不改变计算图，真正的对照是"365d vs 730d 训练期 warmup（均为 no_grad+detach 边界 + scored 全程建图）"。Penman Δ(T730−T365)@E730 = **−0.0173 median / −0.0370 Q25** 是**真实有效的训练效应**（其它模型 flexb +0.0275、topmodel +0.0324、mopex4 +0.0232 均正），方向性也与其 Phase H 的 horizon 收益（+0.0334，1825d 更优）形成鲜明对比——说明 Penman 对"更长的预热但更短的 graph"组合不敏感/敏感方向相反。

判断：
- W2 degradation **不能归因于 truncate interaction** —— 归因链不成立，因为 truncate 未参与计算。（对照任务 §8 明确要求写出 `ASSOCIATION ONLY, NOT CAUSALLY ATTRIBUTED`；此处更强：连 association 都不成立——两个模式标签相同且 inert。）
- `CANONICAL_V2_PROTOCOL.md` 中 "Phase W2 proved 730d warmup degrades Penman **under truncate:90 backpropagation**" 的表述**事实性错误**：W2 证明的是 "730d training warmup 在真实（全图）backprop 下损害 penman"，与 truncate 无关。
- 基于 W2 把 Penman 例外固定在 **365d warmup / 365d scored** 本身有数据支持（`WARMUP_HARM`），**但该例外与 `truncate:90` 标签解耦**：正确表述应为 "Penman 保持 365d warmup，warmup_grad_mode 与其余 35 模型一致（=detach 实际语义）"。

---

## 7. CPU micro-tests（full vs truncation，`SUPPORTED_BY_CPU_TEST`）

环境：**CPU only（CUDA_VISIBLE_DEVICES=""）**，FP64，eager，4 basins（1022500/1031500/1047000/1052500），730d 窗口（365 warmup + 365 scored），θ=0.5，loss=1−KGE（`compute_differentiable_kge`, eps=0.1）。脚本 `/tmp/truncate_audit_cpu.py`（未入库）。

### 7.1 A/B — 标签等价性（现状语义）
所有标签（detach / truncate:90 / truncate:180 / full）：
- forward：**bit-exact 一致**（max_abs_diff = 0.0）；
- 单步 loss：0.477493235（全标签相同）；
- 梯度：**bit-exact 一致**（L2=4.447993e-01，max|g|=3.307e-01，finite，逐位相等）。

→ 证明 forward physics 与梯度图对标签不敏感：`truncate:90` **当前不改变任何计算**。

### 7.2 C — 模拟"真实 TBPTT"（若实现，会改变什么）
手动驱动复刻 `_run_model`，在 scored 内按 30/60/90/180/365 步 detach 状态（365=现状 full graph）：

| tbptt | cos vs full | ‖g‖ ratio | fwd bit-equal | loss |
|---|---|---|---|---|
| 30 | +0.669 | 2.53 | True | 0.477493235 |
| 60 | +0.784 | 1.77 | True | 0.477493235 |
| 90 | **+0.845** | **1.22** | True | 0.477493235 |
| 180 | +0.969 | 1.00 | True | 0.477493235 |
| 365（=现状） | 1.000 | 1.00 | — | 0.477493235 |

→ 若真正实现 truncate:90，梯度方向/尺度**确实会显著改变**（cos 0.845，norm +22%，θ=0.5 处）。但这是假设性结论——现状没有这个机制。

### 7.3 D — 图内存粗测
4 basin FP64 CPU：full 0.19 s vs tbptt-90 0.21 s，RSS 同为 ~1315 MB 峰值（小尺度下无可观测差异）。真实差异只能在 batch=100 GPU 生产中体现——而生产从未因 365d 全图失败。

---

## 8. Verdict

**`HISTORICAL_WORKAROUND_NOT_VERIFIED`**（且更强：当前实现中为 **dead configuration / no-op**；其引入理由属 `HISTORICAL_HEURISTIC_WITHOUT_EVIDENCE`）。

- 不是 `JUSTIFIED_AND_VERIFIED`：无 OOM/NaN 证据、无消融、无文献、无 commit 理由。
- 不是 `JUSTIFIED_BUT_90D_HEURISTIC`：90 从未作用于训练。
- 也不是单纯 `NO_LONGER_NECESSARY`（该标签本来就从未"起过作用"）——但它当前的净效果为零，删除后训练行为逐位不变。
- 唯一有证据支持的 Penman 例外是 **warmup 长度（365d 而非 730d）**，来自 Phase W2（`WARMUP_HARM` 判定），与 truncate 标签无关。

---

## 9. Canonical protocol implication

建议（按审计范围：不擅改协议，仅给结论）：

1. **不要**为了"保留传统"继续把 `truncate:90` 当作有实质内容的特殊设置——它当前是静默 no-op，所有 36 模型的实际梯度语义已经一致（warmup detach + scored 全图）。
2. 最小风险动作：在 canonical v2 协议文本/契约中**删除或重命名** `warmup_grad_mode: "truncate:90"`（或改为与事实一致的 `"detach"`），并修正 `CANONICAL_V2_PROTOCOL.md` 的归因表述。该改动变更零行为，只消除误导。（`INFERENCE` 建议，决策权在协议 owner。）
3. Penman 的 365d-warmup / 365d-scored 例外**保留**（W2 证据），但应表述为"warmup-length exception"，与 truncate 解耦。
4. 如果未来想真正引入 TBPTT（例如为全 36 模型控制长窗口内存/梯度尺度），需先实现解析与 chunked detach，再按 §10 设计最小消融——不应把未实现的历史标签当作既有机制引用。

---

## 10. If needed: minimal GPU experiment（仅设计，不执行）

若协议 owner 决定把 truncate 从"历史标签"升级为"候选机制"，最小验证（**Penman single-model only**）：

- 固定：warmup=365d, scored=365d, window=730d, batch=100, seed=42, AdamW lr=1e-3 wd=1e-4, mapping=auto, clip=1.0, updates=250（与 H16/W208 一致）。
- 比较臂（仅 3 个，最多加 truncate:60）：真实实现后的 `full`（=现状基线）、`truncate:90`、`truncate:180`。
- 判定标准：不是最高 KGE，而是 (a) 训练/验证有限性，(b) train-loss 与 inner-val median KGE 与基线差异是否超过 ±0.005，(c) 梯度范数/clip fraction 变化。若与基线差异 <0.005 且无稳定性收益 → 确认 truncate 无必要。
- 不扩展到其他模型；不做额外 seed 扫描（除非差异临界）。
- 本轮**未启动**此实验（`NO GPU`）。

---

## Sources（本报告引用的事实来源）

- git: `16a1d6c`, `215852c`, `9b0d963`, `b88f88d`, `7d1132bf`（best_metadata 记录的 ablation 运行 SHA，与 ledger 声称的 `3caca37` 不一致——旁证，不影响结论）
- code: `dmotpy/models/hydrology_model.py`（_load_config/forward/_run_warmup/_run_model）、`dmotpy/models/core/penman.py`、`project/benchmark/src/model_registry.py`、`k_full_retrain.py`、`canonical_v2/run_canonical_v2_model.py`、`ablation/ablation_runner.py`、`ablation/evaluate_2x2_warmup.py`、`diagnostics/warmup_gradient_contract.py`、`diagnostics/round12_edge_probe.py`、`diagnostics/dpl_gradient_evidence.py`
- artifacts: `results/dpl_full_retrain_20260813/auto100/{epochs,health,status,parameter_gradients}.csv`、`run_20260815/master.log`、`results/dpl_protocol_ablation_v2_20260831/{ABLATION_LEDGER.md, PHASE_H_REPORT.md, PHASE_W2_REPORT.md, PHASE_W2_2X2_WARMUP_DECOMPOSITION.csv, PHASE_W2_UPDATE_BUDGET_AUDIT.csv, runs/H16_penman_730, W208_penman_w730_s365, H08_penman_1825}`、`results/dpl_canonical_v2_20260831/CANONICAL_V2_PROTOCOL.md`、`results/dpl_training_rule_audit_20260831.md`、`docs/superseded_results_registry.csv`
- CPU test: `/tmp/truncate_audit_cpu.py`（2026-08-31, torch 2.9.1, CPU, FP64）
- external: Crossref / arXiv API 有界检索（2026-08-31）