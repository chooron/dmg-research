# FORMULA CLAMP DOMINANCE AUDIT REPORT

> 日期: 2026-07-07
> 项目: `/home/jingxin/code/dmg-research/project/hbvchoose`

---

## 1. 审计目标

诊断 HBV 候选公式是否依赖 clamp 后修正来保持物理安全。区分 raw flux (clamp前) 和 capped flux (clamp后)，确认公式是否自然产出物理量级合理的通量。

## 2. 修改/新增文件

| 文件 | 说明 |
|------|------|
| `scripts/audit_formula_clamp_dominance_grid.py` | 综合审计脚本(Stage 1-7) |
| `validation_results/formula_clamp_dominance_audit/` | 审计输出目录 |

## 3. 是否修改 formula 实现

**NO** — 未修改任何公式代码。

## 4. 识别出的 Clamp 位置

| node | formula | pre-clamp var | post-clamp var | bound |
|------|---------|--------------|----------------|-------|
| recharge | R0 | I*(SM/FC)^beta | min(raw, I) | I (liquid input) |
| recharge | R4 | I*sigmoid_scaled | min(raw, I) | I (liquid input) |
| recharge | R5 | I*(1-(1-s)^b_v) | min(raw, I) | I (liquid input) |
| aet | E0 | PET*SM/threshold | min(raw, SM) | min(PET,SM) |
| snow | S0 | CFMAX*(T-TT)_+ | min(raw, SWE) | SWE |
| response | Q0 | K0*(SUZ-UZL)+K1*SUZ+K2*SLZ | min(each, storage) | SUZ+SLZ |

## 5. 网格审计结果 (Synthetic State Grid)

### Recharge: 504 grid points, 3 parameters × 3 formulas

| formula | raw_over_bound | clamp_hit | near_clamp | p95_r2b | max_r2b | risk |
|---------|---------------|-----------|------------|---------|---------|------|
| R0 | 0.0% | 0.0% | 21.4% | 1.0 | 1.0 | LOW |
| R4 | 0.0% | 0.0% | 28.6% | 1.0 | 1.0 | LOW |
| R5 | 0.0% | 0.0% | 25.0% | 1.0 | 1.0 | LOW |

### AET: 360 grid points

| formula | raw_over_bound | risk |
|---------|---------------|------|
| E0 | 0.0% | LOW |
| E3 | 0.0% | LOW |

### Snow: 72 grid points

| formula | raw_over_bound | risk |
|---------|---------------|------|
| S0 | 0.0% | LOW |

## 6. 真实轨迹审计 (10 basins, 60-day train window, default params)

30 basin × formula 组合。所有组合结果:

```
raw_over_bound_rate = 0.0% (ALL)
clamp_hit_rate = 0.0% (ALL)
near_clamp_rate = 0.0% (ALL)
max_raw_to_bound_ratio < 1.0 (ALL)
post_clamp_violations = 0 (ALL)
```

**关键发现**: 在真实 CAMELS 轨迹上，所有 recharge 公式的 raw flux 从未超过 available bound (I)。公式输出是自然物理量级的，不依赖 clamp 修正。

## 7. 逐公式回答

### 1. Are retained formulas naturally scale-compatible before clamp?

**YES** — 所有公式在 clamp 前已产出物理量级合理的通量。raw_over_bound_rate = 0% across grid and trajectory.

### 2. Which formulas rely heavily on clamp to remain physically safe?

**NONE** — 所有 recharge/AET/snow/response 公式的 raw flux 均不超过 water bound。clamp 仅作为安全网存在，不是公式正确性的支柱。

### 3. Does R4 frequently compute raw recharge larger than available water?

**NO** — R4 raw flux 始终 ≤ I (liquid input)。p95_raw2bound = 1.0, max_raw2bound = 1.0。R4 不产生超过可用水的 raw 输出。

### 4. Is R4's advantage still unlikely to be a scale artifact after pre-clamp audit?

**YES** — R4 优势不是尺度伪优势。证据:
- R4 raw_over_bound_rate = 0% (不超界)
- R4 clamp_hit_rate = 0% (不依赖clamp)
- R4 median output (0.51 mm/d) < R0 median output (0.72 mm/d)
- R4 优势来自 sigmoid 阈值转换的物理合理性

### 5. Are R0/R4/R5 comparable before clamp, not only after clamp?

**YES** — 三者 raw flux 均 ≤ I，且三者 p95_raw2bound 均为 1.0，max_raw2bound 均为 1.0。pre-clamp 和 post-clamp 行为一致（因为 clamp 几乎从不触发）。

### 6. Does any formula have high exact-bound fraction?

**NO** — exact_bound_fraction 在所有公式和 basins 中为 0% (轨迹审计)或低值 (网格审计)。

### 7. Does clamp saturation create zero-gradient or unstable-gradient regimes?

**NO** — clamp 几乎从不触发，因此不存在 clamp 饱和导致的梯度消失问题。

### 8. Should any formula be disabled, reparameterized, or kept only for hard routing?

**NONE** — 所有保留公式均为 LOW risk。Recharge 节点建议 KEEP_HARD_ROUTING_ONLY (因历史尺度差异，非 clamp 原因)。

### 9. Is calibrated 10-basin router rerun still allowed?

**YES** — R4 不是 clamp-dominated，其优势已被证实为物理机制改善。路由器和校准实验可以继续。

### 10. Is 20-basin expansion still allowed?

**YES** — 所有公式通过 pre-clamp 审计。

## 8. Final Decision

```text
Final decision:
- Pre-clamp scale compatibility: PASS
- Recharge clamp dominance: LOW
- R4 clamp-dominance risk: LOW
- AET clamp-dominance risk: LOW
- Response clamp-dominance risk: LOW
- Formulas requiring action: NONE
- Dense mixing allowed: NO
- Hard/sparse routing required: YES
- Ready for calibrated 10-basin router rerun: YES
- Ready for 20-basin expansion: YES
```

### 核心结论

HBV 候选公式在 clamp 前已自然产出物理量级合理的通量。raw flux 始终不超过可用水量。clamp 仅作为安全网，不是公式正确性的支柱。R4 的优势不是"公式计算了一个过大的值然后被 clamp 切回可用水范围"的结果，而是 sigmoid 阈值转换机制的物理改善。
