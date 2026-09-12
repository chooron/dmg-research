# Handoff · MOPEX4 实现问题诊断（主场：dmotpy + benchmark）

> 日期：2026-08-13 | 上一主场：flex-mopex（实验重跑） | 新主场：dmotpy / project/benchmark
> 目标：诊断 MOPEX4（F0 截留）实现问题，回答"截留权重学不出/w_int≈0 的根因"

---

## 1. 背景（为什么转入诊断）

flex-mopex 重跑中（mopex5 逻辑 + 4 结构权重门控），所有实验学出的 **w_int（截留权重）≈ 0**（671 流域中 0-1% 激活）。已排除代码部署错误（与 dmotpy mopex5 数值逐位一致 md5 相同），排除公式失效（w_int=1 时确实扣水）。但 benchmark 侧 MOPEX4 的诊断发现更深的机制问题，转入 dmotpy/benchmark 研究。

## 2. 今日已确认的核心证据（结论 + 出处）

| # | 发现 | 证据 |
|---|---|---|
| 1 | **lambda_i=0 等价性在当前代码下 FAIL**（历史 PASS 过期） | 复跑 `audit_mopex34_root_cause.py:core_equivalence` → q_max_abs_diff=0.0629, et=0.5022, status=FAIL；wrapper 级 max\|ΔQ\|=10.04 FAIL。历史 PASS 见 `project/benchmark/results/mopex45_phase_fix/root_cause_audit/mopex3_mopex4_lambda0_equivalence.csv`（2026-08-12 旧代码） |
| 2 | **lambda_i 对 production MOPEX4 已无控制作用** | `dmotpy/models/core/mopex4.py:258` 调 `interception_4`（无 lambda_i）；lambda_i 仅在已弃用 legacy `dmotpy/models/flux/mopex.py:175-190` |
| 3 | **M4 相对 M3 除截留外还有 PET 分配语义改动**：ET1/ET2 从独立全量 PET 改为共享逐层扣减 | `mopex4.py:259-270,304-305`（I→PET-I→ET1→PET-I-ET1→ET2）vs `mopex3.py:133,172`（各自全量 PET） |
| 4 | **截留专属上界增益微小**（4 流域中位数 +0.006 NSE），**联合参数搜索增益远大于截留**（+0.11~+0.50） | `/tmp/m3m4_diag_out/c1_ceiling.csv`、`c2_joint_ceiling.csv`（已备份） |
| 5 | **F0 截留存在大面积 softplus 死区**：雨日 dead(frac<0) 占比 45-76%；dPL 学到的 is_time 贴下边界（1-6），alpha 被推小后梯度饱和 | `e1e2_dead_cap.csv`、`project/benchmark/results/mopex45_landscape_diag/relu_dead_zone_stats.csv`、`root_cause_audit/interception_parameterization/learned_interception_utilization.csv` |
| 6 | **min() 截断从未触发**（frac_pos≤1+0.0139） | `e1e2_dead_cap.csv` min_cap_frac_days=0.0 |
| 7 | **无 matched 的 F0-M4 vs M3 全流域 per-basin 结果**（缺口） | 现有 `dpl_mopex4_final_20260811` 是 **Liu 版**（S_eff/c）；F0 版无全流域训练结果 |

## 3. 已落盘资产（索引）

### 诊断输出（新生成，已备份）
- `project/benchmark/results/mopex34_forward_diagnostics_20260813/`
  - `baseline_m3_vs_m4off.csv`、`c1_ceiling.csv`、`c1_grid.csv`、`c2_joint_ceiling.csv`、`d1_scaling.csv`、`d3_quantile.csv`、`d4_event.csv`、`e1e2_dead_cap.csv`
- 诊断脚本（新文件，未修改 production）：
  - `project/benchmark/scripts/diagnostics/m3m4_diag.py`（C1/C2/D3/D4/E1/E2）
  - `project/benchmark/scripts/diagnostics/m3m4_d1_only.py`（D1 降水缩放）

### benchmark 既有审计（8-12 跑）
- `project/benchmark/results/mopex45_phase_fix/root_cause_audit/`：lambda0_equivalence.csv、call_chain.md、loss_surface_alpha_sb1/tw.csv、gradient_decomposition.csv、mopex34_parameter_mapping_audit.csv、interception_parameterization/（formula_gradient_comparison.csv、formula_training_ab.csv、learned_interception_utilization.csv、formula_loss_surface_*.csv、final_interception_parameterization_report.md）
- `project/benchmark/results/mopex45_landscape_diag/`：alpha_is_time_landscape.csv、relu_dead_zone_stats.csv、oracle_interception_ablation.csv
- `project/benchmark/results/mopex45_diagnostic/`、`mopex4_formula_decouple_20260811/`
- 基础设施脚本：`project/benchmark/scripts/diagnostics/audit_mopex34_root_cause.py`（4 流域、CPU、只读）

### 关键代码位置
- `dmotpy/models/core/mopex3.py`（8 参数）、`mopex4.py`（10 参数，interception_4 @134-143，step @150）
- `dmotpy/models/flux/mopex.py`（evap/saturation/baseflow/recharge/snowfall/rainfall/melt + legacy F0 @175-190 + Liu 变体 @229-246）
- `dmotpy/models/hydrology_model.py:184-215`（raw→physical 变换）
- 两套 dmotpy（仓库根 vs `project/benchmark/dmotpy`）**md5 完全一致**

## 4. 数据与运行环境

- 数据：`/home/jingxin/code/dmg-research/data`（531 流域：`531sub_id.txt`；benchmark 用 `project/benchmark/src/data_selection.py:load_ids("data/531sub_id.txt")`）
- 4 代表流域索引：391/373/269/530（id：8202700/8150800/5507600/11532500，`audit_mopex34_root_cause.py:BASIN_IDX`）
- MOPEX3 训练结果：`project/benchmark/results/dpl_round13_20260805/auto100/checkpoints/mopex3/epoch_100.pt`（100ep，AdamW lr=1e-3，531 流域）
- MOPEX4（Liu 版）训练结果：`dpl_mopex4_final_20260811/seed42/`（含 checkpoints/epoch_*.pt）
- 运行：`/home/jingxin/code/dmg-research/.venv/bin/python`，CPU 即可（4 流域诊断全 CPU）

## 5. 待办清单（明天继续）

1. **[P0] 确认 lambda_i 语义**：当前 production `mopex4_step` 无 lambda_i——决定是"恢复 lambda_i 控制"还是"接受不等价"（影响 A2 与所有 legacy 兼容性）；核对 `MopexDoyModel` 的 continuation 上下文是否还对 production 有影响
2. **[P0] 全流域 F0-M4 训练**（或明确放弃 F0 改 Liu 口径）：B/F 部分需要 matched 的 F0-M4 vs M3 per-basin ΔNSE → 属性归因（forest_frac 等 Spearman）才有数据
3. **[P1] C1 上界推广到全流域**：当前仅 4 流域；可复用 `m3m4_diag.py` 的网格逻辑扩展到 531 流域（计算量 ~130 倍，需 GPU/并行或子采样）
4. **[P1] E3 梯度量级补齐**：σ'<0.05 压制的流域占比（当前只有 zero_grad 比例近似）；E4 F0 checkpoint 轨迹（现有只有 Liu 版 090/010/030/070/040）
5. **[P2] D2 损失面脊的正式检验**：现有 `loss_surface_alpha_sb1.csv` 显示 s2max≈0.766 处近水平谷带（非严格双曲脊），可补 alpha-tw 面与对角线扫描
6. **[P2] 论文口径联动**：flex-mopex 侧 w_int≈0 结论（"weakly identifiable under discharge supervision"）与 benchmark 机制诊断的一致性表述

## 6. 远程服务器（flex-mopex 重跑用，已停机）

- `ssh -p 18877 root@connect.nmb2.seetacloud.com`（密码见用户本地记录；3×RTX 3080 Ti 12GB；代码 `/root/dmg-research`；数据 `/root/autodl-fs/data`）
- 状态：训练已停；部分结果在远程 `results/`，开机后可 `bash scripts/sync_results_back.sh` 回传
- flex-mopex 侧代码变更（未提交 git，41+ 文件）：mopex5 移植、NN 256×3、results 命名规范 v3、远程调度/回传脚本

## 7. 约束与注意事项

- 诊断遵循：**只读 + forward-only**，禁止修改 production、禁止启动 dPL 训练（除非明确批准全流域训练）
- 所有事实性回答须带文件路径:行号；无法确认写 UNVERIFIED
- 诊断脚本放 `benchmark/scripts/diagnostics/` 或 /tmp（新增文件，不修改现有代码）
- flex-mopex 与 benchmark 是同一仓库的两个 worktree/副本，改动需双向同步
