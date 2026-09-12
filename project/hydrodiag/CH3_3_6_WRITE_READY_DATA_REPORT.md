# Chapter 3.6 Write-Ready Data Report

## 1. Executive verdict

**`READY_WITH_QUALIFICATIONS`**。

在收缩后的 3.6 写作目标下，ET outlet、ET/P、12-month ET climatology、ET parameter shifts、Response outlet/BFI、tau–BFI 以及 Snow frozen evidence 已形成可追溯的数据包。Lite–Full gate 已通过，未发现新的 numerical blocker。

资格限定为：ET dry-down、Response recession/low-flow/FDC、tau–recession 和 corrected P5 仍未实现；这些是无法恢复的历史计划，不应被写成零效应。dPL 结果仍是 seed42 的 directional evidence。当前 verdict 不声称完整的 process-level identifiability。

## 2. Scope and evidence boundary

本轮只读取 canonical CSV/JSON/NPZ 和既有 Snow frozen audit，不训练、率定、replay 或增加 signature。新整理的 paired summaries 以 basin 为 inferential unit；新生成的 95% CI 使用 2,000 次 basin bootstrap，seed `20260730`。tau–BFI 的 rho/CI 继承 canonical association bootstrap，并在 `response_tau_bfi_summary.csv` 中记录相同的 basin-pair bootstrap、draws 与 seed。月度 ET 由既有 full-replay `evap` 数组在 evaluation slice `5478:10957` 上汇总，没有再次 forward。

历史 ET dry-down 与 Response recession/FDC protocol 不再作为本轮写作包的阻塞条件，但也不被替代定义。没有把缺失过程指标解释为“无差异”。

## 3. ET outlet evidence

四个 canonical contrast 均为 531 个 basin。IC `D_E−N` 的 median ΔKGE 为 `+0.001565`，95% CI `[+0.000209,+0.002583]`；IC `G_E−N` 为 `+0.000451`，CI `[-0.000237,+0.001431]`。dPL `D_E−N` 为 `−0.000509`，CI `[-0.002158,+0.001127]`；dPL `G_E−N` 为 `+0.000264`，CI `[-0.002124,+0.001574]`。

因此 ET outlet response 可以称为 **small**，但不能称为结构完全等效。IC 与 dPL 的 D_E 方向不一致，G_E 方向一致为小幅正值但 CI 跨零；结论应使用“总体出口响应较弱且部分为 null”。

机器可读结果：`et_outlet_summary.csv`。

## 4. ET/P evidence

结构层面的 median `ET/P` 为：IC N `0.6073`、D_E `0.6056`、G_E `0.6061`；dPL controlled-N `0.6064`、D_E `0.5997`、G_E `0.5979`。IC paired differences 很小：D_E−N `+0.000391`，95% CI `[-0.000451,+0.001214]`；G_E−N `−0.000161`，CI `[-0.000961,+0.000570]`。dPL differences 为 D_E−N `−0.003011`，CI `[-0.004179,-0.001939]`；G_E−N `−0.001548`，CI `[-0.002685,-0.000514]`。

结果表明，长期水量分配在 IC 中较弱，而在 dPL seed42 中出现小幅、方向一致的 ET/P 降低。该结果不等价于“更物理”或“更真实”，也不构成 IC/dPL 优劣排名。

机器可读结果：`et_partition_summary.csv`。

## 5. ET monthly climatology

三种 ET 结构的 annual cycle 总体保持相同季节形状：冬季和早春较低，夏季较高，秋季回落，因此 population climatologies 总体重叠。差异不是全年同号常数：IC 中 D_E/G_E 相对 N 在 January–March 和 October–December 多为负、May–September 多为正；dPL 中冬春多数月份为负，而 G_E 在 June–August 出现正向差异。

这些月份差异说明 ET 结构可以改变已有 ET 输出的季节组织，但不能被写成 dry-down AUC/decay 证据。报告使用完整 12-month annual cycle，不根据最大月份新增筛选或 peak-month significance。

机器可读结果：`et_monthly_climatology.csv`；图件数据：`figure_ready/Fig3_13_ET_monthly.csv`。

## 6. ET parameter-shift evidence

`et_parameter_shift_all.csv` 包含 4 个 ET contrasts × 16 个严格共同参数。文件同时保留 canonical physical signed shift、按 authoritative variant-model parameter span 归一化的 signed shift、absolute shift、basin-bootstrap CI、sign fraction 及 ET boundary audit。

`xaj_k` 是唯一具有直接 potential-ET/reference-evaporation 含义的严格共同参数。其 normalized median shift 在 IC D_E/G_E 中分别为 `−0.00380` 与 `−0.01012`，在 dPL 中分别为 `−0.00947` 与 `−0.01921`，四个 contrast 方向一致。然而 IC 的 exact-boundary prevalence 约为 `18.1–18.5%`，因此不作为正文唯一代表参数。其余参数完整保留，不按最大 effect size 排序选择。

## 7. ET evidence classification

**`E1: OUTLET_WEAK_PARTITION_VISIBLE`（限定性）。**

ET outlet ΔKGE 很小；IC ET/P paired effect 近零且 CIs 跨零，dPL ET/P 为小幅负向；12-month climatology 显示若干季节性组织差异。因此可以写“总体出口响应弱，而可用 ET partition/seasonality 提供补充层证据”，但不能把该证据升级为 dry-down process non-equivalence，也不能使用 missing dry-down protocol 作为筛选依据。

## 8. Response outlet evidence

Response outlet 的 median ΔKGE 为：IC D_R−N `−0.001670`、G_R−N `−0.001251`；dPL D_R−N `−0.001152`、G_R−N `−0.001291`。四个 contrast 的绝对量级均较小；D_R 的 IC CI 不跨零，而其他 outlet contrast 的 CI 至少部分跨零。结论可称为 **small outlet response**，不称为完全等效或结构无影响。

机器可读结果：`response_outlet_bfi_summary.csv`。

## 9. BFI evidence

既有三遍 Lyne–Hollick BFI 被原样读取并重新组成 paired summary。IC D_R−N median ΔBFI 为 `+0.003589`，valid N `507`，95% CI `[+0.001381,+0.005405]`；IC G_R−N 为 `−0.001883`，valid N `527`，CI `[-0.003886,-0.000056]`。dPL D_R−N 为 `−0.002314`，valid N `531`，CI `[-0.004167,-0.000545]`；dPL G_R−N 为 `−0.002688`，valid N `531`，CI `[-0.004829,-0.001426]`。

相对于 outlet ΔKGE，BFI 在各 contrast 内提供了更有方向性的 targeted response evidence，但这不是不同量纲 effect size 的横向排名。D_R 的 BFI 方向从 IC 正值变为 dPL 负值；G_R 在 IC/dPL 均为负值。方向不一致被保留。

## 10. tau–BFI association

既有 tau–BFI association 被整理为 `response_tau_bfi_summary.csv`。Spearman rho 为：IC D_R `−0.113`、IC G_R `−0.140`、dPL D_R `−0.087`、dPL G_R `−0.116`。这些均属于 **weak association**，不能称为 strong parameter footprint、tau explains BFI 或因果关系。
其 CI 来源为 canonical `run_ch3_6_analysis.py` 中对 finite basin pairs 的 2,000 次 bootstrap，seed `20260730`；本轮仅整理并显式记录该 provenance。

IC variant tau0 exact-boundary prevalence 为 D_R `55.2%`、G_R `46.9%`；dPL exact-boundary prevalence 为 `0%`，但 near-boundary prevalence 分别为 `35.2%` 和 `30.7%`。这些 boundary facts 进一步限制了强参数解释。

## 11. Response shared-parameter evidence

`response_shared_parameter_shift_all.csv` 汇总 4 个 response contrasts × 13 个严格共同参数，并保留 physical shift、按 authoritative variant-model span 归一化的 signed shift、absolute shift、bootstrap CI 和 sign fraction。variant-specific `xaj_tau0` 不被混入 shared-parameter 表，而单独保留在 tau–BFI 文件中。

现有结果没有一个无需 cherry-picking、同时具备清楚 response-path含义、低 boundary 风险且足以代表全部 IC/dPL contrasts 的 aggregate shared-parameter metric。因此正文不挑选单一 shared parameter；完整 parameter-wise evidence 放入附录。

## 12. Response evidence classification

总体分类为 **`R4: REGIME_DEPENDENT`**，同时具备 `OUTLET_WEAK_TARGETED_RESPONSE_VISIBLE` 的局部特征。理由是 outlet ΔKGE 小，而 BFI 在各 contrast 中比 outlet 提供更明确的 targeted signal；但 D_R 的 BFI 方向在 IC 与 dPL 之间反转，G_R 虽在两 regime 均为负，tau–BFI 仍只有 weak association。

因此正文应写“response-path contrasts were associated with BFI differences in the available canonical analysis”，并明确 IC/dPL expression is regime-dependent；不得写成统一方向、强因果或 dPL 更物理。

## 13. IC/dPL comparison boundary

IC 使用 canonical multi-start IC 资产；dPL 使用 seed42。两者只用于比较结构干预在不同估计范式下的表达，不做优劣排名。ET 的 D_E 方向在 IC/dPL 不一致；Response 的 D_R BFI 方向在 IC/dPL 反转，说明结构效应表达依赖 regime。所有 dPL 结论必须标记为 `single-seed directional evidence`。

## 14. Snow / ET / Response final matrix

机器可读结果：`cross_process_evidence_matrix_final.csv`。

| Process | Structural intervention | Outlet evidence | Parameter evidence | Targeted process/internal evidence | IC/dPL consistency | Evidence completeness | Main observable layer |
|---|---|---|---|---|---|---|---|
| Snow | Prior frozen R1–R5 contrasts | Prior frozen outlet timing/skill | Prior frozen compensation | Prior frozen active-melt timing/internal state | As recorded in frozen audits | Complete for frozen scope | High-process-activity outlet plus parameter/internal evidence |
| ET | D_E/G_E vs N | Small ΔKGE; partly null | All common shifts; no clean representative | ET/P and monthly climatology available; dry-down unavailable | Partly consistent; dPL seed42 | Write-ready for available layers | Weak outlet/partition plus boundary-qualified parameter evidence |
| Response | D_R/G_R vs N | Small ΔKGE | Shared shifts and tau–BFI | BFI available; recession/FDC unavailable | Regime-dependent BFI directions; dPL seed42 | Write-ready for available layers | BFI/tau–BFI targeted evidence |

该表比较的是“首先在哪一层可观察”，不是 Snow、ET、Response 的 effect-size 排名。

## 15. Figure 3-13 recommendation

使用 3 panels：

1. paired ΔKGE；
2. paired Δ(ET/P)；
3. complete 12-month ET climatology。

不加入 dry-down AUC/decay，也不加入单一 parameter panel。`xaj_k` 虽方向一致，但 IC boundary prevalence 约 18%，不适合作为唯一正文 representative。

## 16. Figure 3-14 recommendation

使用 3 panels：

1. paired ΔKGE；
2. paired ΔBFI；
3. tau–BFI association summary（明确标注 weak association）。

不加入 recession、low-flow/FDC、tau–recession、corrected P5 或 P4 storage–release panel。若需要第 4 panel，使用 IC/dPL contrast summary，而不是构造新的综合评分。

## 17. Table 3-X recommendation

使用 `figure_ready/Table3_X_cross_process_final.csv`，展示 Snow frozen evidence、ET available-layer evidence 和 Response available-layer evidence。表格应将 dry-down/recession/FDC/P5 标为 unavailable/pending boundary，不填入零效应。

## 18. 3.6 正文建议使用的 12–16 组数字

1. ET outlet：四个 contrast median ΔKGE 为 `+0.001565`、`+0.000451`、`−0.000509`、`+0.000264`。
2. ET outlet：对应 95% CI 分别为 `[+0.000209,+0.002583]`、`[−0.000237,+0.001431]`、`[−0.002158,+0.001127]`、`[−0.002124,+0.001574]`。
3. ET/P structure：IC N/D_E/G_E 为 `0.6073/0.6056/0.6061`。
4. ET/P structure：dPL N/D_E/G_E 为 `0.6064/0.5997/0.5979`。
5. ET/P paired：IC D_E−N/G_E−N 为 `+0.000391/−0.000161`。
6. ET/P paired：dPL D_E−N/G_E−N 为 `−0.003011/−0.001548`。
7. Monthly ET：IC 结构差异主要表现为冬春负向、夏季部分正向的 seasonal organization；不报告新的 peak-month statistic。
8. `xaj_k` normalized shifts：IC D_E/G_E 为 `−0.00380/−0.01012`，dPL 为 `−0.00947/−0.01921`，并注明 IC exact-boundary prevalence `18.1–18.5%`。
9. Response outlet：四个 contrast median ΔKGE 为 `−0.001670`、`−0.001251`、`−0.001152`、`−0.001291`。
10. BFI：IC D_R/G_R median ΔBFI 为 `+0.003589/−0.001883`。
11. BFI：dPL D_R/G_R median ΔBFI 为 `−0.002314/−0.002688`。
12. BFI valid N：IC D_R/G_R 为 `507/527`，dPL 两个 contrast 均为 `531`。
13. tau–BFI rho：IC D_R/G_R 为 `−0.113/−0.140`。
14. tau–BFI rho：dPL D_R/G_R 为 `−0.087/−0.116`。
15. Lite–Full provenance：10 model assets × 531 basins，aligned max `|ΔKGE|=4.066e-8`；建议放脚注或附录，不占主结果名额。

## 19. 正文不应使用的结果

不得将以下内容写成已完成结果：

- ET dry-down event count、AUC、decay/persistence 或 `P<0.5` sensitivity；
- recession slope/constant；
- low-flow/FDC primary metric；
- tau–recession association；
- corrected P5 decoupling distance；
- strong tau footprint、tau causally explains BFI；
- complete process-level closure/identifiability；
- dPL 更物理、更优或 seed-stable；
- 缺失 protocol 等价于零效应。

## 20. Footnote / appendix allocation

**正文或脚注：** ET outlet weak/partly null、ET/P seasonal summary、BFI direction/regime dependence、tau–BFI weak association、dPL seed42 limitation，以及 Lite–Full gate 的一句 provenance statement。

**附录：** 12-month full climatology CSV、全部 ET/Response parameter shifts、boundary audits、tau boundary prevalence、basin-level BFI、complete gate basin table、bootstrap metadata、canonical replay validation、protocol gap reports、figure-ready CSV 和 checksums。

## 21. Final claim table

| Potential claim | Verdict | Evidence | Recommended wording |
|---|---|---|---|
| ET outlet differences are small | Supported, qualified | Four canonical paired ΔKGE summaries | “ET structural contrasts produced small, partly null outlet differences.” |
| ET structures change long-term ET partition | Supported for dPL, weak for IC | ET/P structure and paired summaries | “The available dPL contrasts showed small shifts in ET/P, whereas IC shifts were close to zero; no realism ranking is implied.” |
| ET structures change seasonal ET organization | Supported, available-layer only | Full 12-month ET climatology | “The ET structures preserved the annual cycle but altered its seasonal allocation in several months.” |
| ET parameter shifts are detectable | Supported, boundary-qualified | 64-row common-parameter summary and boundary audit | “ET interventions were accompanied by parameter redistribution, with boundary concentration limiting single-parameter interpretation.” |
| Response outlet differences are small/moderate | Supported as small | Four response outlet summaries | “Response-path contrasts produced small outlet differences.” |
| Response-path contrasts affect BFI | Supported, qualified | Existing BFI paired effects and CIs | “Response-path contrasts were associated with BFI differences in the available canonical analysis.” |
| tau is associated with BFI change | Supported as weak association | Four tau–BFI rho/CI results | “The fitted tau parameter showed a weak basin-level association with BFI change; this is not a causal claim.” |
| IC/dPL express the same structure change identically | Not supported | Directional differences in ET/P and D_R BFI | “The direction and expression of structural effects depended partly on estimation regime.” |
| Snow/ET/Response differ in observable diagnostic layer | Supported as a qualified framework | Frozen Snow matrix plus available ET/Response layers | “The available evidence indicates layer-specific observability, with process completeness differing by intervention.” |
| Complete process-level identifiability is established | **Not supported** | Missing frozen dry-down/recession/FDC/P5 definitions | Do not claim complete process-level closure. |

## 22. Final thesis narrative

现有结果显示，不同过程结构差异并不会在同一诊断层面以相同方式出现：Snow 的差异主要由冻结的高过程活跃期出口、参数和内部状态证据支撑；ET 的出口差异较小，但 ET/P 与月尺度季节组织提供了补充信息；Response 的出口差异同样较小，而 BFI 及 tau–BFI 提供了较有针对性的响应路径证据。与此同时，IC 与 dPL 的方向并不完全一致，且 dPL 仅有单一 seed。因此，本节支持“诊断层具有过程依赖性”的限定性结论，但不支持完整 process-level closure 或因果解释。

**Final status: `READY_WITH_QUALIFICATIONS`。**
