# Figure 3/4 Design Audit — Frozen Results 3.2 / R2

**Scope:** review-only；未修改绘图代码、统计结果、图片或任何 R1/R3/R4/R5 文件。审查基于当前工作树；`plot_r2_figure3_final.py` 与 `plot_r2_figure4.py` 已有未提交修改，视为既有用户工作。

## 1. Executive verdict

- **Figure 3：`MAJOR_REDESIGN`。** 当前布局有可保留的 macro scatter/trajectory 元素，但数据接口仍指向旧命名的 `r2_tgd2_specificity_*` 包，使用 `Base–TGD2` 标签，主证据顺序不符合冻结 hierarchy，且没有直接绘制 paired `delta_excess` / `Delta_beta`。
- **Figure 4：`MINOR_REBALANCE`。** canonical wrapper 的“两 panel：all-15 overview → selected S1–S5 distributions”方向正确；需要修正 regime/model 颜色语义、把 `im` 纳入 secondary anchor、锁定 canonical source，并清理 legacy renderer 的地图/输出歧义。
- **当前 PNG 状态：不可审图。** 仓库中没有找到 `Figure3_R2_final.png`、`Figure4_R2_final.png` 或旧 `Figure4_R2.png`；`canonical_assets.json` 仍将两图标为 `blocked_missing_source`。因此本报告对视觉尺寸/实际点密度的判断仅基于 renderer，不声称已验证 PNG。

## 2. Current implementation inventory

| Figure | Manuscript-facing entrypoint | Current output coded in script | Current data inputs | Status |
|---|---|---|---|---|
| F3 | `project/hydrodiag/manuscript/scripts/r2/plot_r2_figure3_final.py` | `manuscript/figures/Figure3_R2_final.png` | `manuscript/results/R2/r2_tgd2_specificity_basin_level.csv`, `_summary.csv`, `_regressions.csv` | manifest 指定为 canonical，但 data interface 是旧 specificity/TGD2 命名 |
| F4 | `project/hydrodiag/manuscript/scripts/r2/plot_r2_figure4_canonical.py` | `manuscript/figures/Figure4_R2_final.png` | wrapper 实际加载 `r2_snow_gradients_summary.csv`、`r2_parameter_values_canonical.csv`、`r2_paired_shifts_basin_level.csv` | manifest 指定为 canonical；wrapper 依赖 legacy `plot_r2_figure4.py` 的常量/loader/KDE |
| F4 legacy | `project/hydrodiag/manuscript/scripts/r2/plot_r2_figure4.py` | `manuscript/figures/Figure4_R2.png`（当前代码）；docstring/旧 handoff 还写过 `manuscript/plots/figures/` | 同上，另含 GIS/map 代码与 Supplement map 输出 | 不是 manifest-facing writer；不能与 canonical wrapper 混用 |

重复/冲突点：

1. `manuscript/analysis/R2/results/` 是 canonical analysis 输出，`manuscript/results/R2/` 是 manuscript-facing 镜像；核心 summary 文件多数一致，但 `r2_tgd2_specificity_*` 与 `r2_parameter_values_canonical.csv` 并非 byte-identical。下一轮应明确单一 authority，并更新 manifest。
2. `canonical_assets.json` 的 F3 source list 仍是 `r2_tgd2_specificity_*`；F4 source list 还漏列 `r2_parameter_values_canonical.csv`。两图均没有现成 PNG 可追溯。
3. `manuscript/hess_results_R1_R5_reframed_v2.md` 与 `manuscript/stats/tables/TableS4_exact_estimates_f3_f4.*` 仍有旧的 `TGD2` 标签、旧 dPL slope/interpretation，需要在绘图代码修正后同步。

## 3. Panel → source → columns → estimand mapping

### Figure 3 current renderer

| Panel | Current source / columns | Current estimand and diagnosis |
|---|---|---|
| (a) IC scatter | `r2_tgd2_specificity_basin_level.csv`: `paradigm`, `contrast`, `within_pooled`, `between_all`; locally computes `between_all > within_pooled` | Basin-level within-vs-between separation for IC；对应 canonical prevalence definition，但同时把 Base–CN 与 control 等权绘出。应以 Base–CN 为 primary，TGD 只作 attribution control。 |
| (b) dPL scatter | 同上 | dPL supporting separation；同样不是 IC/dPL benchmark。当前视觉上仍容易读成两方法对比。 |
| (c) hero facets | basin file `frac_snow`, `excess`; summary `metric=excess` 的 `median/ci_lower/ci_upper`; regression `dependent_var=excess` 的 slope/CI；局部拟合 intercept 仅用于画线 | Snow-organized excess trajectory。S1–S5 markers 能表现 IC 单调与 dPL S4/S5 plateau，这是当前 F3 最值得保留的部分；但 OLS line/wedge 比经验 trajectory 更醒目，且用独立 CN/TGD curves 代替 paired attribution。 |
| (d) merged decomposition | summary `metric=between_all/within_pooled`，仅 Base–CN，按 IC/dPL 合并 | 解释 excess 的 between/within 来源；是 supporting evidence，且与 (a)/(b) 重复。IC/dPL 用蓝/橙颜色承载，违反 regime 不抢模型颜色语义的规则。建议 compact annotation 或 Supplement。 |
| (e) slope summary | regressions `slope/slope_ci_lower/slope_ci_upper`，IC/dPL × Full531/ExcludeS5 × Base–CN/Base–TGD | 独立 slope 比较，不是 paired `Delta_beta`，也不是 paired `delta_excess`。当前 panel 不能直接支持“intermediate emergence + high-snow persistence”，并给 TGD 过高视觉权重。应替换为 `r2_paired_cn_tgd_delta_excess_summary.csv` 的 strata paired contrast，并用 `r2_tgd2_slope_difference_summary.csv` 只作小型 Full/ExcludeS5 annotation。 |

**F3 当前未使用但 canonical package 明确生成的 macro files：**
`r2_canonical_15D_displacement_summary.csv` / `_basin_level.csv`、`r2_within_structure_summary.csv`、`r2_s1_s5_macro_trajectory.csv`、`r2_canonical_prevalence_summary.csv`、`r2_paired_cn_tgd_delta_excess_summary.csv`、`r2_tgd2_slope_difference_summary.csv`。这不是数值不存在，而是 renderer 尚未迁移到冻结 package interface。

### Figure 4 canonical wrapper

| Panel | Current source / columns | Current estimand and diagnosis |
|---|---|---|
| (a) all shared parameters | `r2_snow_gradients_summary.csv`: `paradigm`, `parameter`, `beta`, `ci95_low`, `ci95_high`; zero line；IC/dPL offset markers | All-15 Base–CN normalized signed shift slope `beta` vs snow fraction。满足 all-15 first、zero-centered reading、IC/dPL parallel展示；当前未显示 median/Spearman 数字，避免了统计过载。 |
| (b) key paired shifts | `r2_paired_shifts_basin_level.csv`: `paradigm`, `parameter`, `snow_regime`, `delta_base_minus_cn`; locally fixed-bandwidth reflected KDE，zero line | `Delta z = z_Base - z_CN` 的 S1–S5 distributions；方向/零中心是正确的。当前只选 `um/ki/ci`，漏掉冻结 hierarchy 中的 secondary `im`；颜色用 IC blue / dPL orange，容易与 Base/CN 模型颜色混淆。 |
| legacy maps (not wrapper) | 同一 paired shifts + shapefiles | Geographic descriptive view，非 Figure 4 主 explanatory claim。应留 Supplement（若仍需要），不应与 canonical two-panel wrapper 并存为主图。 |

## 4. Legacy / incorrect statistic check

| Check | Finding | Severity |
|---|---|---|
| Canonical prevalence | 当前 F3 直接从旧 specificity basin file 计算 `between_all > within_pooled`，未直接读取 canonical prevalence summary；公式本身没有写成固定阈值，但 source provenance 不合格。 | **MUST FIX** |
| Fixed `D > 0.08` | 当前 F3 renderer 没有用 `between_all > 0.08` 作统计（代码中的 `0.08` 仅出现在坐标范围）；历史 audit 明确记录旧 97.36%/100% 来自固定阈值，不能让旧输入包重新成为 authority。 | **MUST FIX provenance** |
| Old prevalence / single restart / seed-42 | 当前 renderer 没有明确 seed-42 选择；但 `r2_tgd2_specificity_*` 是旧命名接口，不能证明其来自最终 canonical package。应改为 canonical summary/trajectory source。 | **MUST FIX provenance** |
| TGD2 label | F3 常量、标题、legend、docstring 均使用 `Base–TGD2`；canonical frozen files/closure report 使用 `Base–TGD` / TGD。Table S4 与 manuscript 也残留 TGD2。 | **MUST FIX** |
| Noncanonical normalization | F4 使用 `delta_base_minus_cn`，并检查 canonical `z∈[0,1]`；signed shift 方向统一为 `Base − CN`，这一点可保留。 | pass |
| Historical S5-driven interpretation | F3 panel (e) 只画 independent slopes；manuscript 当前文字仍写“contrast weakened when S5 removed”，与冻结的 dPL `Delta_beta` Full `+0.0411`、ExcludeS5 `+0.0861` 及 intermediate-emergence interpretation 不一致。 | **MUST FIX** |
| Parameter-significance mining | F4 all-15 panel 本身没有筛选 15 参数；但 `um/ki/ci` 的 row band 和 selected ridges 必须明确为 recurring directional signatures，不能暗示 post-hoc significance。`im` 应作为 secondary anchor。 | rebalance |

## 5. Figure 3 scientific architecture audit

- **(a)/(b)：** 科学问题是“between 是否超过结构内 variability”；属于 supporting identification，不应先于 Base–CN snow trajectory 成为视觉焦点。当前把 TGD 与 CN 同等散点展示，弱化 `whole-space → snow organization → control` 顺序。
- **(c)：** 是 primary macro claim 的正确载体，经验 S1–S5 markers 与 continuous basin relationship 应保留；需要降低 OLS line/CI wedge 权重，突出 dPL S1→S4 上升、S5 plateau，避免线性 slope 被读成严格持续上升。
- **(d)：** supporting decomposition，和 (a)/(b) 重复；不适合继续作为等权主 panel。
- **(e)：** attribution 层级方向正确但 estimand 不对：独立曲线/独立 slopes 不能替代 paired `delta_excess_b`，也没有直接显示 paired `Delta_beta`。应改为 paired control panel。
- **Prevalence：** 值得保留，但作为 compact supporting strip/annotation；只使用 `P_b(between_all_b > within_pooled_b)`，主图优先 Base–CN 的 Full/S1/S5 或完整 S1–S5 trajectory，不能再放旧 threshold 或 legacy prevalence。
- **LOPO：** 不应进入主图核心；ROBUST_MULTIVARIATE 作为 Supplement 或 caption/结果文字一句话即可。

### F3 verdict

**`MAJOR_REDESIGN`**，但不是重做统计：保留现有 (c) 的经验 trajectory 视觉骨架，重排 primary/supporting/attribution 层级，切换 canonical data interface，并删除/下放重复 decomposition。

## 6. Figure 4 scientific architecture audit

- all-15 first 已做到，且能在 5–10 秒内告诉读者哪些 slope 为正、哪些为负、哪些接近零；这是应保留的核心。
- `Base − CN` 方向统一、零线存在、IC/dPL 以上下偏移和 marker 区分，逻辑基本正确。
- 当前 selected distributions 只有 `um/ki/ci`；冻结结果还要求 `im` 作为 secondary negative signature。建议加入 `im`，但不要加粗/染色成“显著参数”。
- 当前没有把 median shift、Spearman、OLS、所有 S1–S5 数字全部塞入主图，这是好事；Spearman、median/IQR、boundary/KDE diagnostics 应留 Table/Supplement。
- 当前 IC/dPL 颜色直接借用了 CN/Base palette；这会让读者误把 constraint regime 当成 model/structure。改为 panel/marker/line-style 区分，signed shift 的方向只由 x=0 与零中心轴读出；若使用发散色，必须以 0 为中心。
- legacy renderer 的 GIS/maps 不属于当前主 explanatory panel；canonical wrapper 的两 panel 结构无需改成 2×2。

### F4 verdict

**`MINOR_REBALANCE`**：保留 canonical two-panel architecture；改 source/labels、加入 `im` secondary anchor、去除 model-color ambiguity，并把地图与 detailed KDE/boundary checks 保持在 Supplement。

## 7. Main text vs Supplement allocation

### 必须留在 Figure 3

1. Base–CN whole-parameter-space excess 的 S1–S5 empirical trajectory，IC/dPL parallel facets；IC near-monotonic、dPL S1–S4 rise + S5 plateau 必须一眼可读。
2. regime-specific within reference 与 canonical prevalence 的 compact supporting view，明确 reporting unit 是 basin。
3. TGD attribution 的 paired `delta_excess_b`（S1–S5）或 paired `Delta_beta`（Full/ExcludeS5）至少一个直接视觉表达；建议两者合并为主 attribution block，而不画两条等权 independent curves。
4. 若 canonical 4A 仍是冻结 Figure 3 specification 的必需项，加入一个窄小的 15-D displacement inset；不得让它盖过 excess trajectory。

### 必须留在 Figure 4

1. All 15 shared parameters 的 Base–CN signed snow-gradient overview，0 为中心，IC/dPL 平行而非 ranking。
2. `um/ki/ci/im` 的 restrained S1–S5 directional distributions，作为 recurring directional signatures，不写 mechanistic substitute/compensation claim。

### 建议移入 Supplement

- LOPO slope range、parameter contribution shares、完整 robustness/audit tables；
- full independent Base–TGD slope curves、完整 `Delta_beta`/`delta_excess` CI table（主图保留 compact summary）；
- redundant between/within decomposition 与重复 prevalence representation；
- detailed Spearman、median/IQR、boundary point-mass、KDE sensitivity diagnostics；
- legacy GIS maps（若仍需展示）以及 ExcludeS5 的重复图形，而不是再与主图争夺层级。

## 8. Minimal recommended architecture for next implementation round

### Figure 3（建议由当前 5-panel 改为 3 个逻辑 block，不做机械 2×2）

| Block | Purpose | Data | Visual form | Manuscript role |
|---|---|---|---|---|
| (a) hero, IC/dPL two aligned facets | whole-space Base–CN response along snow activity | `r2_s1_s5_macro_trajectory.csv` + `r2_macro_regressions.csv`；可叠加 `r2_within_structure_summary.csv` reference | S1–S5 median + 95% CI、轻量 basin cloud；OLS line/CI subdued；dPL S5 plateau visible | primary macro |
| (b) compact prevalence/reference strip | between-vs-within identification，避免重复散点 | `r2_canonical_prevalence_summary.csv` 或 trajectory prevalence；必要时 `r2_canonical_15D_displacement_summary.csv` 作窄 inset | basin prevalence points/line，或 within/between paired markers；Base–CN 主色阶，IC/dPL marker/line 区分 | supporting |
| (c) paired TGD attribution | show where CN exceeds generic TGD and that S5 is plateau persistence | `r2_paired_cn_tgd_delta_excess_summary.csv`；`r2_tgd2_slope_difference_summary.csv` 仅作 Full/ExcludeS5 inset | paired delta across S1–S5 with zero line + two small Delta-beta intervals | attribution |

当前 (c) trajectory 保留；当前 (a)/(b) scatter 降级/合并；当前 (d) 下放；当前 (e) 替换。LOPO 不进主图。

### Figure 4（保留 canonical two-panel）

| Panel | Purpose | Data | Visual form | Manuscript role |
|---|---|---|---|---|
| (a) all-15 overview | identify recurring positive/negative/near-zero directions without selection mining | `r2_snow_gradients_summary.csv`（仅 Base–CN，IC/dPL） | zero-centered forest/dot-and-CI；regime 用 shape/line style，去掉 Base/CN color collision | primary explanatory overview |
| (b) four restrained anchors | show snow-organized distributions for `um/ki/ci/im` | `r2_paired_shifts_basin_level.csv`，`delta_base_minus_cn`，S1–S5 | four aligned ridges/interval summaries；IC/dPL mirrored/line-style；S1–S5 order清楚；不以 highlight 暗示 significance | explanatory detail |

不要把 maps、Spearman、所有 CI、LOPO 或 boundary diagnostics 加回主图；canonical wrapper 的非 2×2 结构应保留。

## 9. Final answer to audit questions

- **当前思路是否需要调整？** 需要。F3 是科学架构与数据接口同时调整；F4 是小幅再平衡，不需要推倒重来。
- **Minor 还是 major？** F3 `MAJOR_REDESIGN`；F4 `MINOR_REBALANCE`；整体不是重新分析，而是 frozen-statistics 对齐。
- **最值得改的 5 项：**
  1. 两图统一切换到 `analysis/R2/results` 冻结 package（或明确同步的 manuscript mirror），并修正 manifest/source list；
  2. 全面把 `Base–TGD2` 改为 canonical `Base–TGD`/`TGD`，同步 caption/table/manuscript 的旧标签与旧 slope interpretation；
  3. F3 把 empirical S1–S5 Base–CN trajectory 提升为 hero，直接加入 paired `delta_excess`/`Delta_beta`，不再用 independent slope curves 充当 attribution；
  4. F3 下放重复的 between/within decomposition 与 LOPO，明确 dPL plateau；
  5. F4 保留 all-15 → selected distributions，加入 `im`，并用 marker/line/panel 区分 IC/dPL，避免借用 Base/TGD/CN 颜色语义。
- **下一步能否直接修改绘图代码？** 可以直接进入实施轮次，无需新增统计分析或重新运行 hydrological simulations；但应先确认本报告的 F3 三 block 方案，以及 canonical 4A displacement 是否作为窄 inset 保留。实施前还应保留当前工作树中已有的 plotting-script 修改，不覆盖用户工作。
