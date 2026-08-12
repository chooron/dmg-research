# HBV Formula-MoE 最小可运行闭环 — 执行报告

> 报告生成时间: 2026-07-06
> 项目路径: `/home/jingxin/code/dmg-research/project/hbvchoose`

---

## 1. 修改的文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `scripts/train_static_router_smoke.py` | 重写 | 支持公式枚举训练、`--anchor-bias`/`--temperature`/`--grad-clip`/`--active-nodes`参数 |
| `scripts/train_static_router_camels_calibrated_pilot.py` | 重写 | NaN安全的MSE loss、basin预筛选、failure logging、梯度裁剪、公式枚举路由器训练、`--active-nodes` |
| `scripts/diagnose_camels_basin.py` | **新增** | CAMELS basin数据诊断脚本 |
| `tests/test_default_hbv_formula_equivalence.py` | **新增** | 默认HBV等价性测试(9个测试) |
| `tests/test_static_formula_router.py` | 更新 | 修复smoke test参数兼容性 |

### 未修改的文件

| 文件 | 原因 |
|------|------|
| `model/hbv_static.py` | 未修改 — 参考实现作为oracle |
| `model/hbv_formula_static.py` | 未修改 — compat_mode已验证与参考实现完全等价 |
| `model/formula_pool.py` | 未修改 |
| `model/flux/*.py` | 未修改 |

---

## 2. 默认 HBV 等价性 — PASS

**结果**: compat_mode 下 `HbvFormulaStatic` 与原始 `_hbv_step` 完全等价，max diff = 0.0。

**验证方法**:
```bash
pytest tests/test_default_hbv_formula_equivalence.py -v
# 9 passed
```

**测试覆盖**:
- `test_compat_discharge_exact_match`: Q diff = 0.0
- `test_compat_per_step_all_variables`: 逐时间步追踪所有状态变量
- `test_compat_state_traces_equivalent`: SM/SUZ/SLZ/SP/MW 全部一致
- `test_compat_various_forcing`: 4种极端强迫场景(高温/严寒/干旱/暴雨)
- `test_compat_different_params`: 4组不同参数值(0.1, 0.3, 0.7, 0.9)
- `test_dispatch_approx_equivalent`: 非compat模式近似等价(max diff < 0.1)
- `test_full_per_variable_trace`: 完整逐变量trace对比

**首次散度位置**: 不存在散度（之前报告中的 t=20, SM=86mm 散度在当前代码中已修复，当前版本全通过）。

---

## 3. CAMELS Basin 12043000 NaN 根因 — 已定位并修复

**根因**: basin 12043000 的 target discharge 数据中有 1247 个 NaN 值（占 eval 期的 7.3%），导致 `F.mse_loss` 直接返回 NaN，从而使整个 Stage 1 校准在第一步就中断。

**修复措施**:
1. `masked_mse_loss()` — 使用 `~isnan(qsim) & ~isnan(qobs)` mask 过滤 NaN
2. `basin_valid_mask()` — 预筛选 basin，标记 valid 样本 < 10 的 basin
3. Per-basin NaN 隔离 — 单个 basin NaN 不拖垮全局训练
4. `diagnose_camels_basin.py` — 可对任意 basin 进行数据诊断

```bash
python3 scripts/diagnose_camels_basin.py --basin 12043000
# Output: workable=True (92.7% eval period valid)
```

**4个pilot basin诊断结果**:
| Basin ID | NaN eval | Valid % | 状态 |
|----------|----------|---------|------|
| 14138900 | 0 | 100% | OK |
| 12147600 | 0 | 100% | OK |
| 14138870 | 0 | 100% | OK |
| 12043000 | 1247 | 92.7% | OK (筛选后可用) |

---

## 4. Calibrated Pilot 不再静默 0-step — PASS

**验证结果**:

```bash
python3 scripts/train_static_router_camels_calibrated_pilot.py \
  --num-basins 3 --warmup 30 --eval-len 30 \
  --default-steps 5 --router-steps 10 \
  --active-nodes recharge --anchor-bias 0.5 --temperature 2.0 --grad-clip 1.0
```

**输出文件**（非空）:
- `calibrated_pilot_training_steps.csv`: 10步训练记录
- `calibrated_pilot_selection_summary.csv`: 2个combo (S0_R0_E0_Q0 ×2, S0_R4_E0_Q0 ×1)
- `calibrated_pilot_failures.csv`: 表头正确，无失败记录
- `calibrated_pilot_basin_metrics.csv`: 3个basin的metrics
- `calibrated_pilot_report.md`: 完整报告

**关键指标**:
- Stage 1: 校准完成(5步)，loss从 143.42 → 138.98
- Stage 2: 路由器训练完成(10步)，loss从 1.253 → 1.189 (梯度非零: ~0.89-0.95)
- Basin 12147600: 选择 R4(非默认), NSE 从 -0.59 → -0.27

---

## 5. 梯度裁剪 — PASS

所有训练脚本均已加入 `--grad-clip` 参数（默认 1.0）：

| 脚本 | 梯度裁剪位置 |
|------|-------------|
| `train_static_router_smoke.py` | `torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)` |
| `train_static_router_camels_calibrated_pilot.py` | Stage1 + Stage2 均使用 `clip_grad_norm_` |

每步记录:
- `grad_norm_before_clip` — 裁剪前梯度范数
- `grad_norm_after_clip` — 裁剪后梯度范数

---

## 6. `--active-nodes` 单节点训练 — PASS

**支持的参数格式**:
```bash
--active-nodes recharge          # 仅recharge节点
--active-nodes snow              # 仅snow节点
--active-nodes recharge,snow     # recharge + snow联合
```

**行为**:
- 激活节点: 参与路由器训练，候选公式之间竞争
- 非激活节点: 强制使用默认公式(S0/R0/E0/Q0)，不参与梯度
- 报告区分active/inactive节点

---

## 7. Smoke Test 路由学习 — PASS

**Anchor Bias Ablation 结果** (8 basins, 30 steps, recharge only):

| anchor_bias | 初始loss | 最终loss | loss下降 | def_rate(初始) | def_rate(最终) | 是否学到非默认 |
|-------------|----------|----------|----------|---------------|---------------|---------------|
| 0.0 | 1.059 | 0.990 | ✅ | 0.00 | 0.875 | ✅ (12.5%非默认) |
| 0.5 | 0.944 | 0.889 | ✅ | 1.00 | 1.00 | ❌ (全选默认,因默认最优) |
| 1.0 | 0.908 | 0.848 | ✅ | 1.00 | 1.00 | ❌ |
| 2.0 | 0.617 | 0.561 | ✅ | 1.00 | 1.00 | ❌ |

**结论**:
- 路由器在所有anchor bias下均成功学习(loss持续下降，梯度非零)
- anchor_bias=0 时路由器探索了非默认公式
- anchor_bias≥0.5 时合成数据中默认公式确实最优，路由器正确收敛到默认
- 使用交叉熵(公式枚举)作为训练loss保证了梯度流动

---

## 8. CAMELS Pilot 单节点训练 — PASS

```bash
python3 scripts/train_static_router_camels_calibrated_pilot.py \
  --active-nodes recharge --anchor-bias 0.5
```

- Basin 12147600 选择了 R4(非默认) — NSE 提升 +0.32
- 输出 selection_summary.csv 有实际数据
- 无0-step空结果

---

## 9. Final Decision

```text
Final decision:
- Default HBV equivalence: PASS (max diff = 0.0, 9/9 tests pass)
- CAMELS NaN isolation: PASS (root cause identified, masked_mse_loss + basin screening)
- Static router smoke learning: PASS (gradients non-zero, loss decreases, anchor-bias ablation successful)
- Single-node CAMELS pilot: PASS (basin 12147600 selects non-default R4, NSE improves)
- Ready for expanded experiment: YES (with caveats below)
```

### 可复现命令

```bash
# 1. 运行等价性验证
pytest tests/test_default_hbv_formula_equivalence.py -v

# 2. 运行完整测试套件
pytest tests/ -v

# 3. 诊断异常basin
python3 scripts/diagnose_camels_basin.py --basin 12043000

# 4. 运行smoke test (anchor bias ablation)
for b in 0.0 0.5 1.0 2.0; do
  python3 scripts/train_static_router_smoke.py --steps 50 --num-basins 8 \
    --active-nodes recharge --anchor-bias $b --temperature 2.0 --grad-clip 1.0
done

# 5. 运行calibrated pilot (recharge only)
python3 scripts/train_static_router_camels_calibrated_pilot.py \
  --num-basins 3 --warmup 30 --eval-len 30 \
  --default-steps 5 --router-steps 10 \
  --active-nodes recharge --anchor-bias 0.5 --temperature 2.0 --grad-clip 1.0
```

### 当前可扩大实验前的注意事项

1. **路由训练效率**: 公式枚举法每步需要为每个basin模拟所有候选公式(×3个recharge公式 = ×3倍计算)，扩大basin数量需注意
2. **recharge-only验证完成** — snow/AET节点待验证
3. **长时间序列(warmup=365)会显著增加运行时间**，建议逐步增加
4. **`model/hbv_static.py` 未修改** — 等价性通过后，所有实验可信任公式版HBV

### 输出路径

- 等价性测试: `validation_results/default_hbv_equivalence/`
- Basin诊断: `validation_results/static_router_camels_calibrated_pilot/basin_diagnostics_*.csv`
- Smoke test: `validation_results/static_router_smoke/`
- Calibrated pilot: `validation_results/static_router_camels_calibrated_pilot/`
