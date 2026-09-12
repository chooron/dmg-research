# 3.5 内部水文状态与过程差异数据核准报告

> **范围**：Snow 主案例；只使用已冻结的 R3 synthetic truth-relative state/flux outputs、R3 seasonal arrays，以及 R4 ERA5-Land/Snow-17 external-reference outputs。本文档是数据核准与写作准备，不重新训练、重新率定、增加状态变量、增加 signature 或重建 synthetic truth。
>
> **核心判断**：出口恢复不等于内部水文恢复。synthetic truth evidence 与 real-catchment external-consistency evidence 分层报告；ERA5-Land SM100 与 Snow-17 均不是水文真值。

## 1. Canonical provenance

### 表 A0：Canonical provenance

| Data role | Canonical path | Rows / shape | Main variables | Bootstrap / seed | Status | Notes |
|---|---|---:|---|---|---|---|
| Synthetic state truth | `results/r3_synthetic_truth_v1/x_star.npz` | 7 × (531, 12,418) + IDs | `wu, wl, wd, s, fr, qi, qg` | N/A | FROZEN_CANONICAL | 1980-10-01–2014-09-30；`x_star` 是状态 truth；`final_states.npz` 仅为终端状态，未用于本审计 |
| Synthetic snow truth | `results/r3_synthetic_truth_v1/snow_star.npz` | 7 arrays × (531, 12,418) | `G, eTG, sca, rain, melt, effective_precip` | N/A | FROZEN_CANONICAL | seasonal generating input 使用 `effective_precip` |
| Base/TGD state and flux metrics | `results/r3_misspec_analysis_v1/state_metrics_basin.csv` | 131,688 rows | per-basin `rmse,nrmse,corr,bias`；train/test/DJF/MAM/JJAS | N/A | FROZEN_CANONICAL | Base/TGD recorded-forward state metrics |
| CN-refit state metrics | `results/r3_gate_v1/gate_state_metrics_basin.csv` | 108,855 rows | same state/flux metrics for correct-CN gate；CN residual source for table A | N/A | FROZEN_CANONICAL | `state_excess.csv` pairs Base/TGD rows to this seed-matched CN source |
| Paired state/flux excess | `results/r3_misspec_analysis_v1/state_excess.csv` | 318,600 rows | `e_M,e_CN,delta_E`；`wu,wl,wd,s,fr,qi,qg,wt` | N/A | FROZEN_CANONICAL | `delta_E = E_M - E_CN`；本报告的表 A/B/C 直接基础 |
| Figure S3 component source | `results/r3_misspec_analysis_v1/paired_parameters.csv` + `state_excess.csv` | 63,720 + 318,600 rows | 15 shared parameters；最终六个 state/flux keys | 2,000 basin draws；seed 20260730（renderer） | FROZEN_CANONICAL | final asset `final_assets/figures/Figure_S3/` 明确只保留 `wu,wl,wd,wt,qi,qg` |
| Figure S3 aggregation source | `manuscript/results/R3/figure6_basin_seedmedian.csv` | 1,062 rows | dPL three-seed basin median；`delta_E_*`；`G_Base,G_TGD` | dPL seeds 42/123/2026；basin bootstrap | FROZEN_CANONICAL | IC passthrough；dPL 先按 basin 取 seed median |
| Seasonal trajectory source | `manuscript/results/R3/fig6_seasonal/fig6_seasonal_input.npz`, `fig6_seasonal_state.npz` | each 6 × (133, 12) + IDs | Base/TGD/CN/Truth；`input` 与 `wt` | summary uses 2,000 monthly basin draws | FROZEN_CANONICAL | high `f_snow >= Q75`，water-year Oct–Sep |
| Seasonal summary / Figure S4 | `manuscript/results/R3/figure6_summary.json`; renderer `manuscript/scripts/r3/plot_r3_si_seasonal_trajectories.py` | JSON panels E/F；12 months | effective liquid-water input；signed `Delta Wt` | `n_boot=2000`；`boot_seed=20260730` | FROZEN_CANONICAL | final asset `final_assets/figures/Figure_S4/`；只有 trajectory summary，没有预冻结 winter/spring fraction scalar |
| Recovery–internal association | `manuscript/results/R3/figure6_summary.json:panel_d_associations` + `figure6_basin_seedmedian.csv` | 4 frozen pairs/regime；expanded rows derived below | `G_Base/G_TGD` vs parameter excess/Wt excess | raw/partial values frozen；CI 为本报告 basin bootstrap | FROZEN + LIGHT EXTENSION | frozen panel 只列 parameter 与 Wt；WU/WL/WD/Qi/Qg 为同一 source 上的轻量扩展，不是新模型运行 |
| ERA5-Land external state | `results/r4_caravan_soil_reference_v1/caravan_soil_ensemble.npz` | `SM100,SM289,L1–L4,caravan_swe`: each (531, 12,418) | primary `SM100` | N/A | FROZEN_CANONICAL | SM100 = 0–100 cm composite；单位 m3/m3 |
| Snow-17 external burden | `results/r4_swe_reference_v1/swe_ensemble.npz` + `swe_basin_burden_test.csv` | SWE (531,12,418,10) + 531 burden rows | `swe_median`；annual max burden；positive days | N/A | FROZEN_CANONICAL | burden uses Snow-17 ensemble median；external-only |
| Figure 9 / phase summary source | `results/r4_phase1_soil_official/figure7_panel_a_phase_burden_matrix.csv`; `figure7_external_phase_rows.csv`; `figure7_metadata.json` | 48 summary rows；5,588 phase rows | phase `Delta r` for CN/TGD relative to Base | 2,000 basin draws；seed 20260730 | FROZEN, PROVENANCE-QUALIFIED | repository current figure is R4 Figure 7；thesis 3.5 can renumber it as Figure 3-11/3-9 as appropriate |
| Figure 10 timing source | `results/r4_phase1_soil_official/three_structure_timing_metrics_basin_year.csv`; `three_structure_timing_metrics_basin_summary.csv`; `manuscript/results/discussion_audit/r4_tgd_spring_timing_audit.csv` | 95,580 year rows；5,373 basin rows；15 audit rows | wet-up/peak signed and absolute offsets | basin-level summary; audit CIs | FROZEN, PROVENANCE-QUALIFIED | dPL population median across available seed-level rows (nominally three)；seed-42 TGD has N=434 and remains visualization/sensitivity source |
| Timing sensitivity | `results/r4_phase1_soil_official/robustness_timing_sensitivity.csv` | 12 rows | 7/14/21 d wet-up；full WY/Mar–Aug peak | 6,541 valid basin-years; no CI stored | FROZEN | IC fused + dPL seed 42; not the N=449 basin-median estimand |
| Figure S5 examples/population | `manuscript/supplement/figures/FigureS1_R4_selection_audit.json` + `FigureS1_R4_population_audit.csv` | 6 examples + 531 population rows | external SWE terciles；SM100 contrasts | source annotations | FROZEN STAGED ASSET | final asset is `final_assets/figures/Figure_S5/`; population eligible N=442, not phase common-support N=331 |

**Bootstrap convention used for the lightweight reaggregation in this report**：basin is the resampling unit；2,000 draws；fixed seed 20260730 with deterministic arm/variable offsets where a new table needs separate raw-error or association intervals. Existing frozen CIs are retained where available. No forward model, training, or calibration was run.

## 2. 共同状态和通量定义

### 2.1 Final common set

| Code | Thesis notation | Definition | Final Figure S3 status |
|---|---|---|---|
| `wu` | WU | upper-layer tension-water storage | included |
| `wl` | WL | lower-layer tension-water storage | included |
| `wd` | WD | deep-layer tension-water storage | included |
| `wt` | Wtot | `wu + wl + wd` | included; registered derived state |
| `qi` | Qi | interflow routing flux | included |
| `qg` | Qg | groundwater routing flux | included |

`state_excess.csv` 的 canonical implementation 还保存 `s`（free-water storage）与 `fr`。但 final Figure S3 的 renderer 明确将最终状态集限制为 `wu, wl, wd, wt, qi, qg`；`s` 不是正文共同状态，`fr` 也不是本节共同 state/flux。早期 protocol 将 `s` 列为 primary、`wd` 列为 secondary，但 final manuscript asset 采用显式 `wt`，不把 `s` 重新命名为总张力水，也不自行把 `s` 加回正文。

### 2.2 Truth-relative NRMSE

实际代码 `manuscript/scripts/r3/misspec_states.py:63–70` 使用：

\[
\mathrm{RMSE}_{M,v,b}=\sqrt{\mathrm{mean}_t[(X_{M,v,b,t}-X^*_{v,b,t})^2]},
\quad
\mathrm{NRMSE}_{M,v,b}=\frac{\mathrm{RMSE}_{M,v,b}}{\mathrm{std}_t(X^*_{v,b,t})+10^{-8}}.
\]

因此 normalization 是**同一 basin、同一 period 的 truth standard deviation 加 `1e-8`**，不是 truth range、mean 或跨 basin standard deviation。`wt` 由三层状态逐日相加后再与 `wu+wl+wd` 的 truth 比较。

### 2.3 Excess error

实际代码与 final Figure S3 README 一致：

\[
E_{\mathrm{excess},M,v,b}=\mathrm{NRMSE}_{M,v,b}-\mathrm{NRMSE}_{CN,v,b}.
\]

CN-refit residual 使用 `e_CN` 的实际数值，**没有预设为 0**。Table B 的 positive fraction 是 `P(E_excess > 0)`，不是 CI 排除 0 的比例。

## 3. Truth-relative state/flux error

表 A 报告 test period、每 basin 的 truth-relative NRMSE；IC 使用 selected restart，dPL 先在 basin 内取三 seed median。括号内为 `[Q25,Q75]`，最后三列为 95% basin-bootstrap CI。

### 表 A

| Variable | Estimation | N | Base median [Q25,Q75] | TGD median [Q25,Q75] | CN median [Q25,Q75] | Base 95% CI | TGD 95% CI | CN 95% CI |
|---|---|---:|---|---|---|---|---|---|
| WU | IC | 531 | 0.808 [0.485,1.408] | 0.850 [0.420,1.311] | 0.453 [0.178,0.917] | [0.772,0.854] | [0.797,0.892] | [0.375,0.539] |
| WL | IC | 531 | 1.021 [0.404,2.071] | 0.711 [0.307,1.592] | 0.516 [0.189,1.223] | [0.851,1.137] | [0.615,0.823] | [0.459,0.605] |
| WD | IC | 531 | 4.972 [0.991,87.679] | 3.905 [0.821,78.814] | 2.487 [0.637,46.905] | [3.401,6.761] | [3.008,5.794] | [1.958,3.424] |
| Wtot | IC | 531 | 0.672 [0.301,1.375] | 0.598 [0.256,1.207] | 0.307 [0.141,0.637] | [0.604,0.742] | [0.533,0.651] | [0.276,0.343] |
| Qi | IC | 531 | 0.687 [0.426,1.053] | 0.759 [0.437,0.960] | 0.250 [0.148,0.400] | [0.641,0.758] | [0.708,0.799] | [0.229,0.269] |
| Qg | IC | 531 | 0.728 [0.321,1.321] | 0.600 [0.308,1.140] | 0.155 [0.087,0.259] | [0.649,0.805] | [0.520,0.645] | [0.144,0.170] |
| WU | dPL | 531 | 0.456 [0.201,0.923] | 0.304 [0.137,0.771] | 0.065 [0.035,0.122] | [0.408,0.537] | [0.264,0.347] | [0.060,0.070] |
| WL | dPL | 531 | 0.475 [0.163,1.332] | 0.190 [0.071,0.574] | 0.097 [0.040,0.200] | [0.421,0.535] | [0.163,0.233] | [0.086,0.111] |
| WD | dPL | 531 | 3.092 [0.327,83.633] | 1.700 [0.185,56.374] | 1.524 [0.128,28.058] | [2.078,4.844] | [1.141,2.765] | [1.036,2.770] |
| Wtot | dPL | 531 | 0.449 [0.164,0.794] | 0.362 [0.114,0.696] | 0.278 [0.044,0.776] | [0.405,0.507] | [0.310,0.416] | [0.214,0.358] |
| Qi | dPL | 531 | 0.548 [0.293,1.076] | 0.432 [0.238,0.675] | 0.137 [0.059,0.203] | [0.480,0.607] | [0.397,0.472] | [0.123,0.145] |
| Qg | dPL | 531 | 0.401 [0.171,1.063] | 0.368 [0.165,0.734] | 0.078 [0.048,0.117] | [0.355,0.469] | [0.320,0.417] | [0.071,0.083] |

**核准结论**：CN-refit 在六个共同变量上均为总体最小的 truth-relative median。TGD 总体低于 Base 的结论在 dPL 六变量均成立；IC 则只对 WL、WD、Wtot、Qg 成立，WU 与 Qi 的 TGD median 略高于 Base。也就是说，“TGD 总体优于 Base”不能推广为每个 IC component 都优于 Base。

## 4. Excess internal error

### 表 B

| Variable | Estimation | Base excess median | 95% CI | Positive % | TGD excess median | 95% CI | Positive % |
|---|---|---:|---|---:|---:|---|---:|
| WU | IC | +0.255 | [+0.174,+0.327] | 67.2% | +0.268 | [+0.186,+0.323] | 68.7% |
| WL | IC | +0.274 | [+0.190,+0.393] | 67.8% | +0.172 | [+0.100,+0.242] | 61.6% |
| WD | IC | +0.807 | [+0.555,+1.299] | 66.7% | +0.345 | [+0.183,+0.557] | 60.5% |
| Wtot | IC | +0.270 | [+0.193,+0.351] | 73.1% | +0.210 | [+0.155,+0.257] | 68.2% |
| Qi | IC | +0.418 | [+0.372,+0.474] | 81.9% | +0.447 | [+0.389,+0.488] | 86.1% |
| Qg | IC | +0.497 | [+0.402,+0.573] | 83.2% | +0.384 | [+0.317,+0.443] | 82.9% |
| WU | dPL | +0.379 | [+0.333,+0.466] | 97.9% | +0.217 | [+0.188,+0.278] | 96.2% |
| WL | dPL | +0.345 | [+0.300,+0.398] | 94.0% | +0.076 | [+0.059,+0.100] | 81.2% |
| WD | dPL | +0.529 | [+0.342,+0.765] | 86.4% | +0.019 | [+0.011,+0.030] | 59.3% |
| Wtot | dPL | +0.040 | [+0.029,+0.069] | 67.0% | +0.004 | [−0.001,+0.008] | 53.1% |
| Qi | dPL | +0.386 | [+0.329,+0.436] | 99.2% | +0.284 | [+0.248,+0.316] | 96.4% |
| Qg | dPL | +0.334 | [+0.280,+0.387] | 94.0% | +0.287 | [+0.256,+0.325] | 93.6% |

**正文可用数字**：Wtot 的 excess 是 IC `+0.270`（Base）对 `+0.210`（TGD），dPL `+0.040` 对 `+0.004`；对应 positive fraction 为 IC 73.1%/68.2%、dPL 67.0%/53.1%。Qi 是一个明确的 IC 例外：TGD `+0.447` 高于 Base `+0.418`。这展开了 manuscript 中“several shared states and fluxes”的含义：正 excess 不只存在于 Wtot，也出现在 WL、WD、Qi、Qg 等变量；但不能说所有变量都具有同一 snow 梯度或同一 TGD 改善方向。

## 5. Internal error 的 S1–S5 变化

下表直接使用 `figure6_basin_seedmedian.csv` 的 `delta_E`，按已冻结的 `snow_stratum` 汇总。数值为各 snow stratum 的 basin median；没有为了单调性新增模型分析。

### 表 C

| Variable | Estimation | S1 | S2 | S3 | S4 | S5 |
|---|---|---:|---:|---:|---:|---:|
| WU | Base IC | −0.012 | 0.254 | 0.256 | 1.281 | 1.899 |
| WU | TGD IC | 0.014 | 0.224 | 0.287 | 0.998 | 1.599 |
| WL | Base IC | 0.037 | 0.236 | 0.518 | 0.850 | 3.008 |
| WL | TGD IC | 0.017 | 0.006 | 0.248 | 0.724 | 2.324 |
| WD | Base IC | 0.000 | 0.956 | 85.238 | 3.532 | 4.762 |
| WD | TGD IC | −0.009 | 0.324 | 1.349 | 3.399 | 3.861 |
| Wtot | Base IC | 0.004 | 0.279 | 0.506 | 0.707 | 1.980 |
| Wtot | TGD IC | −0.005 | 0.139 | 0.575 | 0.493 | 0.799 |
| Qi | Base IC | 0.008 | 0.373 | 0.744 | 0.881 | 0.897 |
| Qi | TGD IC | 0.048 | 0.413 | 0.597 | 0.713 | 0.848 |
| Qg | Base IC | 0.033 | 0.393 | 1.144 | 1.196 | 1.251 |
| Qg | TGD IC | 0.033 | 0.285 | 0.712 | 1.171 | 1.384 |
| WU | Base dPL | 0.068 | 0.327 | 0.723 | 1.485 | 2.355 |
| WU | TGD dPL | 0.042 | 0.188 | 0.591 | 1.165 | 2.080 |
| WL | Base dPL | 0.030 | 0.296 | 1.012 | 1.922 | 2.497 |
| WL | TGD dPL | 0.011 | 0.053 | 0.402 | 0.928 | 1.095 |
| WD | Base dPL | 0.049 | 0.521 | **67,638,453.222** | 20.117 | 0.355 |
| WD | TGD dPL | 0.011 | −0.136 | 31.571 | 1.583 | 0.127 |
| Wtot | Base dPL | 0.027 | 0.031 | −0.031 | 0.145 | 0.818 |
| Wtot | TGD dPL | 0.009 | −0.003 | −0.232 | 0.053 | −0.152 |
| Qi | Base dPL | 0.091 | 0.329 | 0.937 | 0.924 | 0.809 |
| Qi | TGD dPL | 0.058 | 0.218 | 0.494 | 0.709 | 0.615 |
| Qg | Base dPL | 0.064 | 0.281 | 0.949 | 0.873 | 1.203 |
| Qg | TGD dPL | 0.048 | 0.256 | 0.526 | 0.904 | 1.269 |

**解释边界**：WU/WL/Wtot/Qi/Qg 多数呈现从低雪到高雪的上升或高雪增强，尤其 dPL 的 WU/WL/Qi；但 IC 的 Qi、Qg 与 dPL 的 Wtot/Qg 等并不形成统一单调模式。WD dPL 在 S3 出现 `6.76×10^7` 的数值退化，应视为 `std(truth)+1e-8` normalization 下的源数据异常/弱识别区，不能作为正文代表性 component，也不能据此构造“所有状态随 snow activity 单调增大”的故事。

## 6. Outlet recovery 与 internal error 的关联

### 表 D

定义保持 3.4 的 common-reference outlet quantities：Base 行使用 `G_Base`，TGD 行使用 `G_TGD`；不是重新计算 `F_close` 或 `F_TGD`。internal quantity 是对应的 `delta_E`，参数行使用 `E_param_excess`。`rho` 为 Spearman；partial rho 控制 ranked `f_snow`。方括号为 95% basin bootstrap CI。

| Structure | Estimation | Internal variable | Recovery metric | Raw rho [95% CI] | Partial rho [95% CI] |
|---|---|---|---|---:|---:|
| Base | IC | parameter | G_Base | +0.601 [+0.535,+0.663] | +0.023 [−0.088,+0.132] |
| Base | IC | WU* | G_Base | +0.325 [+0.242,+0.411] | −0.055 [−0.130,+0.019] |
| Base | IC | WL* | G_Base | +0.346 [+0.267,+0.422] | −0.017 [−0.097,+0.066] |
| Base | IC | WD* | G_Base | +0.378 [+0.306,+0.449] | +0.044 [−0.043,+0.131] |
| Base | IC | Wtot | G_Base | +0.414 [+0.335,+0.488] | −0.065 [−0.138,+0.012] |
| Base | IC | Qi* | G_Base | +0.676 [+0.616,+0.726] | +0.064 [−0.033,+0.159] |
| Base | IC | Qg* | G_Base | +0.638 [+0.588,+0.685] | +0.030 [−0.058,+0.115] |
| TGD | IC | parameter | G_TGD | +0.522 [+0.447,+0.588] | −0.067 [−0.158,+0.037] |
| TGD | IC | WU* | G_TGD | +0.317 [+0.230,+0.397] | −0.119 [−0.191,−0.044] |
| TGD | IC | WL* | G_TGD | +0.265 [+0.175,+0.351] | −0.078 [−0.163,+0.010] |
| TGD | IC | WD* | G_TGD | +0.333 [+0.259,+0.403] | +0.024 [−0.063,+0.103] |
| TGD | IC | Wtot | G_TGD | +0.406 [+0.331,+0.473] | −0.066 [−0.147,+0.017] |
| TGD | IC | Qi* | G_TGD | +0.582 [+0.524,+0.633] | −0.038 [−0.126,+0.054] |
| TGD | IC | Qg* | G_TGD | +0.673 [+0.622,+0.720] | +0.064 [−0.015,+0.146] |
| Base | dPL | parameter | G_Base | +0.812 [+0.776,+0.840] | +0.138 [+0.055,+0.226] |
| Base | dPL | WU* | G_Base | +0.802 [+0.757,+0.841] | −0.127 [−0.231,−0.012] |
| Base | dPL | WL* | G_Base | +0.745 [+0.704,+0.781] | +0.031 [−0.071,+0.136] |
| Base | dPL | WD* | G_Base | +0.419 [+0.339,+0.491] | +0.141 [+0.051,+0.227] |
| Base | dPL | Wtot | G_Base | +0.155 [+0.059,+0.246] | −0.026 [−0.100,+0.052] |
| Base | dPL | Qi* | G_Base | +0.782 [+0.754,+0.807] | +0.034 [−0.037,+0.116] |
| Base | dPL | Qg* | G_Base | +0.788 [+0.753,+0.817] | +0.012 [−0.069,+0.096] |
| TGD | dPL | parameter | G_TGD | +0.685 [+0.624,+0.736] | −0.128 [−0.230,−0.010] |
| TGD | dPL | WU* | G_TGD | +0.817 [+0.769,+0.858] | −0.071 [−0.161,+0.031] |
| TGD | dPL | WL* | G_TGD | +0.567 [+0.493,+0.632] | −0.234 [−0.310,−0.151] |
| TGD | dPL | WD* | G_TGD | +0.069 [−0.019,+0.156] | −0.282 [−0.346,−0.211] |
| TGD | dPL | Wtot | G_TGD | **−0.327 [−0.410,−0.236]** | **−0.406 [−0.474,−0.337]** |
| TGD | dPL | Qi* | G_TGD | +0.710 [+0.655,+0.757] | −0.229 [−0.317,−0.135] |
| TGD | dPL | Qg* | G_TGD | +0.807 [+0.759,+0.847] | +0.002 [−0.117,+0.135] |

`*` 表示从 frozen `figure6_basin_seedmedian.csv` 按同一 frozen formula 做的轻量 expanded row；Figure 6 的 frozen panel D 实际只保存 parameter 与 Wtot 四组 pair。其 frozen values 的交叉核验为：Base parameter IC raw/partial `+0.601/+0.023`、dPL `+0.812/+0.138`；TGD–dPL–Wtot partial `−0.406`，CI 与 manuscript 的 `[-0.476,-0.336]` 近似一致。

## 7. Recovery–internal association 的 snow-activity 调整

raw association 与 `f_snow` 通常同向，尤其 dPL Base parameter `+0.812`、TGD WU `+0.817`、TGD Qg `+0.807`；这说明 recovery 与 internal excess 的共同变化很大程度上沿 snow-activity gradient 组织。控制 ranked `f_snow` 后，Base–Wtot 与 TGD–Wtot 在 IC/dPL 多数接近 0，parameter 关系也明显衰减。

但不能把“多数减弱”写成“所有关系都等于 0”：扩展表中 TGD–dPL 的 WL、WD、Wtot、Qi partial rho 仍为负，其中 Wtot 的 raw `−0.327`、partial `−0.406` 是预先指出的例外。因而正文建议采用：

> outlet recovery 与 internal error 的 raw association 很大程度上与 snow activity 的共同组织有关；控制 snow activity 后，多数关系明显减弱，说明较好的 outlet recovery 并不对应一个简单、统一的 internal-error reduction pathway。TGD–dPL–Wtot 保留负的 partial association，但不作因果解释。

不做 causal mediation，也不将 partial association 解释为物理机制识别。

## 8. 高雪 synthetic liquid-water delivery

### 表 E：trajectory-level 核准与月度锚点

`Figure S4`/R3 source 没有冻结“winter delivery fraction”或“spring delivery fraction”的 scalar estimand；因此不新增这种 ratio。下表使用已保存的月度 basin-median trajectory，给出 **December winter anchor** 与 **April spring anchor**（单位 mm d-1）以及 12 点 median trajectory 的描述性 peak month。CI 是 source summary 的 monthly median CI。

| Estimation | Structure | Winter anchor: Dec median [95% CI] | Spring anchor: Apr median [95% CI] | Peak month (12-point median, descriptive) | Source |
|---|---|---:|---:|---|---|
| IC | Generating CN truth | 0.680 [0.434,1.019] | 6.394 [5.561,6.913] | Apr | `figure6_summary.panel_e_seasonal_input.Truth` |
| IC | Base-refit | 3.502 [3.162,3.897] | 3.161 [2.840,3.352] | Dec | `Base_IC` |
| IC | TGD-refit | 1.299 [1.071,1.479] | 5.920 [5.164,6.474] | Apr | `TGD_IC` |
| IC | CN-refit | 0.681 [0.422,1.017] | 6.419 [5.652,6.904] | Apr | `CN_IC` |
| dPL | Generating CN truth | 0.680 [0.434,1.019] | 6.394 [5.561,6.913] | Apr | `figure6_summary.panel_e_seasonal_input.Truth` |
| dPL | Base-refit | 3.502 [3.077,3.897] | 3.161 [2.840,3.352] | Dec | `Base_dPL` |
| dPL | TGD-refit | 1.321 [1.069,1.515] | 5.926 [5.113,6.474] | Apr | `TGD_dPL` |
| dPL | CN-refit | 0.688 [0.434,0.964] | 6.532 [5.630,7.149] | Apr | `CN_dPL` |

**trajectory interpretation**：generating CN truth/CN-refit 将输入集中到春季（April peak）；Base 在 December 达到月度 median peak，保留更宽的冬季输入；TGD 的 peak month 向 April 移动并明显接近 truth，但仍有 winter residual，未复现 generating trajectory。这里的 scalar 是月度 trajectory anchor，不是新增 effect test；正文不应写成新的 winter/spring fraction inference。

## 9. 高雪 synthetic Wtot seasonal departure

`figure6_summary.panel_f_seasonal_storage_heatmap` 保存的是 signed monthly `Delta Wt = Wt_model - Wt_truth`、Q25/Q75、IQR 与 monthly CI，没有预冻结的 phase-aggregated scalar。下表选择 December/February（winter/accumulation）、April（active melt）、June（post-melt）和 August（dry-down）作为已保存 monthly series 的锚点；括号为 source monthly median 95% CI，单位 mm。

### 表 F

| Estimation | Structure | Winter/accumulation: Dec; Feb | Active melt: Apr | Post-melt: Jun | Dry-down: Aug | Source |
|---|---|---|---:|---:|---:|---|
| IC | Base | +60.718 [46.744,75.459]; +88.860 [68.904,120.479] | +54.263 [30.434,82.200] | +12.379 [4.102,21.913] | +10.646 [1.729,16.139] | `series.Base_IC` |
| IC | TGD | +10.655 [2.641,19.706]; +35.955 [25.633,44.496] | +31.135 [21.347,44.322] | +21.462 [2.803,28.682] | +7.660 [1.461,15.718] | `series.TGD2_IC` |
| IC | CN-refit | −1.421 [−3.539,−0.046]; −1.614 [−3.538,+0.287] | −1.468 [−4.359,2.012] | −1.327 [−3.978,0.617] | −0.238 [−3.850,0.766] | `series.CN_IC` |
| dPL | Base | +36.014 [28.293,44.966]; +53.140 [43.867,75.832] | +24.994 [19.840,36.239] | +1.281 [−2.878,5.623] | +2.438 [−4.029,3.620] | `series.Base_dPL` |
| dPL | TGD | +1.106 [−2.675,4.986]; +22.557 [17.860,27.856] | +16.849 [11.314,24.853] | +6.583 [0.404,14.517] | −0.407 [−4.249,3.402] | `series.TGD2_dPL` |
| dPL | CN-refit | −29.714 [−36.255,−25.805]; −29.856 [−37.165,−26.369] | −29.621 [−35.601,−23.809] | −30.119 [−35.739,−25.174] | −33.651 [−38.554,−27.490] | `series.CN_dPL` |

**关键核准**：

1. IC 中 Base 的主要 departure 在冬季累积至 February，TGD 将 departure 降低并推向 spring，但仍保留正的 spring departure；CN-refit 几乎贴近 generating trajectory。
2. dPL 中 Base 仍有明显 winter/early-spring positive departure，TGD 显著降低；但 frozen `CN_dPL` trajectory 保留约 −30 mm 的全年负 departure，不能写成“CN-refit 在 IC/dPL 两种 estimation 下都最接近 generating seasonal Wtot trajectory”。
3. 因而正文应写“CN 在 IC seasonal trajectory 上接近 truth；dPL 的 aggregate truth-relative error 较低，但 high-snow seasonal trajectory 仍有 systematic offset”，这正是 outlet recovery 与 internal recovery 不等价的证据。不能把 TGD 的较高 outlet recovery自动翻译为 seasonal Wtot 已恢复。

## 10. ERA5-Land / Snow-17 数据与样本口径

- ERA5-Land reference：`SM100 = 0.07 L1 + 0.21 L2 + 0.72 L3`，0–100 cm depth-weighted composite，单位 m3/m3。SM289 仅为 sensitivity reference，本节主结果用 SM100。
- model state：`Wtot = WU + WL + WD`，单位 mm；不做 mm 与 m3/m3 的绝对值比较，也不做 conceptual XAJ layer 与 ERA5 physical layer 的 1-to-1 深度映射。
- daily state series 先去除 calendar-month climatology；phase consistency 用 within-catchment anomaly dynamics，Pearson correlation 对线性标准化不变。7 d diagnostic 使用 centered rolling mean，`min_periods=4`；所有 external eligibility、burden 与 phase mask 都只由 Snow-17/Caravan SWE 定义，不由任何模型 Wtot 定义。
- external phase：accumulation（SWE >= 5 mm，至 annual peak）、active melt（peak 后且 SWE >= 5 mm）、post-melt（melt-out 后至 June）、dry-down（July–September）。basin-level phase eligibility 另有 `median annual max SWE >=20 mm` gate；这与 daily phase threshold 5 mm 不同。
- R4 current phase metadata：available all-phase population 344；panel-A complete-case common support **N=331**，burden counts Low/Middle/High = 110/110/111。

### external burden thresholds

| Axis | Definition | Threshold / bins | N | Note |
|---|---|---|---:|---|
| phase panel common support | Snow-17 burden tertiles after complete-case support | Low 20.010–47.108 mm；Middle 47.108–144.126 mm；High 144.179–1226.950 mm | 110/110/111 = 331 | `figure7_phase_complete_case_bins.csv`；rank-first qcut(3) |
| phase available case | all-phase eligible before finite four-block complete-case restriction | Low 20.010–49.019；Middle 49.509–146.890；High 147.085–1226.950 mm | 115/114/115 = 344 | not to be labelled N=331 |
| timing external high burden | Snow-17 median annual max SWE Q75 | Q75 = **133.388496 mm**；`>=Q75` | 133 | Figure 10 Q3; external subset |
| synthetic high snow | CAMELS `frac_snow` Q75 | Q75 = **0.2176966694**；`>=Q75` | 133 | Figure S4; separate from external SWE Q75 |

## 11. Phase-resolved external-state consistency

### 定义核准

\[
\Delta r_{CN-Base}=\mathrm{corr}(Wtot_{CN},SM100)-\mathrm{corr}(Wtot_{Base},SM100),
\]
\[
\Delta r_{TGD-Base}=\mathrm{corr}(Wtot_{TGD},SM100)-\mathrm{corr}(Wtot_{Base},SM100).
\]

这里的 `corr` 是 phase 内 calendar-month-anomaly correlation；phase table 由当前 Caravan SWE mask 重建。表 G 使用 `figure7_panel_a_phase_burden_matrix.csv` 的 **N=331 complete-case common support** median/CI，并在相同 complete-case rows 上补算 positive fraction。CI 为已有 2,000-draw basin bootstrap。

### 表 G：complete-case N=331

#### IC fused

| Burden | Phase | CN–Base median [95% CI] | Positive % | TGD–Base median [95% CI] | Positive % |
|---|---|---:|---:|---:|---:|
| Low | Accumulation | +0.038 [+0.020,+0.057] | 70.9% | +0.066 [+0.049,+0.082] | 75.5% |
| Middle | Accumulation | +0.086 [+0.058,+0.116] | 76.4% | +0.063 [+0.036,+0.106] | 70.0% |
| High | Accumulation | +0.140 [+0.115,+0.189] | 79.3% | −0.014 [−0.042,+0.015] | 44.1% |
| Low | Active melt | +0.068 [+0.039,+0.101] | 69.1% | +0.041 [+0.023,+0.064] | 70.9% |
| Middle | Active melt | +0.276 [+0.169,+0.390] | 89.1% | −0.049 [−0.095,−0.005] | 38.2% |
| High | Active melt | +0.523 [+0.457,+0.567] | 93.7% | −0.063 [−0.085,−0.044] | 23.4% |
| Low | Post-melt | −0.000 [−0.008,+0.004] | 49.1% | +0.024 [+0.005,+0.043] | 63.6% |
| Middle | Post-melt | −0.005 [−0.014,+0.001] | 42.7% | +0.025 [+0.012,+0.039] | 69.1% |
| High | Post-melt | −0.021 [−0.049,−0.002] | 39.6% | +0.013 [−0.004,+0.031] | 56.8% |
| Low | Dry-down | −0.001 [−0.006,+0.003] | 46.4% | −0.016 [−0.022,+0.002] | 40.9% |
| Middle | Dry-down | −0.001 [−0.008,+0.005] | 48.2% | +0.012 [−0.004,+0.030] | 58.2% |
| High | Dry-down | −0.004 [−0.014,+0.002] | 43.2% | −0.017 [−0.027,−0.009] | 36.9% |

#### dPL seed 42 visualization / complete-case table block

| Burden | Phase | CN–Base median [95% CI] | Positive % | TGD–Base median [95% CI] | Positive % |
|---|---|---:|---:|---:|---:|
| Low | Accumulation | +0.031 [+0.018,+0.045] | 74.5% | +0.011 [+0.006,+0.018] | 66.4% |
| Middle | Accumulation | +0.044 [+0.027,+0.059] | 70.0% | −0.002 [−0.019,+0.008] | 47.3% |
| High | Accumulation | +0.140 [+0.103,+0.166] | 79.3% | +0.007 [−0.024,+0.044] | 52.3% |
| Low | Active melt | +0.057 [+0.026,+0.084] | 67.3% | −0.001 [−0.010,+0.005] | 47.3% |
| Middle | Active melt | +0.240 [+0.158,+0.328] | 83.6% | −0.063 [−0.079,−0.034] | 30.0% |
| High | Active melt | +0.464 [+0.397,+0.527] | 94.6% | +0.153 [+0.108,+0.198] | 76.6% |
| Low | Post-melt | +0.000 [−0.007,+0.008] | 50.0% | +0.009 [+0.005,+0.017] | 75.5% |
| Middle | Post-melt | −0.012 [−0.022,−0.000] | 40.9% | +0.004 [−0.004,+0.009] | 55.5% |
| High | Post-melt | −0.023 [−0.042,+0.005] | 44.1% | −0.023 [−0.039,−0.004] | 39.6% |
| Low | Dry-down | +0.003 [+0.001,+0.005] | 62.7% | +0.003 [+0.002,+0.005] | 67.3% |
| Middle | Dry-down | −0.010 [−0.013,−0.004] | 31.8% | −0.006 [−0.010,−0.003] | 38.2% |
| High | Dry-down | −0.006 [−0.022,+0.003] | 45.9% | −0.009 [−0.016,−0.002] | 37.8% |

### positive-fraction conflict audit

用户任务中预先给出的 active-melt positive fractions `CN–Base ≈84%/81%`、`TGD–Base ≈44%/51%` 实际对应的是 **available-case phase rows**，不是 N=331 complete-case：

- IC active melt：CN finite available N=344，positive **83.4%**；TGD finite available N=341，positive **43.4%**。
- dPL seed42 active melt：CN finite available N=344，positive **81.1%**；TGD finite available N=334，positive **50.9%**。

在 N=331 complete-case 上，positive fractions 分别为 CN IC 69.1/89.1/93.7%（Low/Middle/High），CN dPL 67.3/83.6/94.6%；因此正文不能把 84%/81% 标作 N=331 结果。建议：主图若坚持 N=331，就使用表 G complete-case fractions；若要保留 manuscript 的 84%/81%，必须明确标为 available-case overall active-melt fractions，并同时给出 N=344/341/334。

## 12. Spring wet-up / storage-peak timing

### timing definition

- 表 H 的 timing-eligible basin：每个 water year 同时要求 external annual maximum SWE >= 5 mm、model annual Wtot maximum >= 0.1 mm，并在 test period 至少有 5 个 valid snow years；共同 timing population **N=449**。因此 H 的 eligibility 含一个 model-Wtot validity gate，并非纯 external-only。
- wet-up：processed soil-water series 在 Jan 1–Jun 30 的最大 14-day increase 的 water-year date。
- peak：annual maximum timing；offset = `model timing - ERA5-Land SM100 timing`，负值表示 model earlier。
- dPL population summary：按 basin 对可用 seed-level summaries 取 median（名义上三 seed；TGD seed42 的 15 个 basin 缺失，因此并非每个 basin 都有三个 TGD rows）；Figure 10 curve/visualization 可单独使用 seed 42，但不可与 population median 混用。

### 表 H

表 H 的 median 为 basin-level population median；`[Q25,Q75]` 来自 canonical basin summary 的 cross-basin spread；CI 优先采用 `r4_tgd_spring_timing_audit.csv` 的直接 audit CI。

| Estimation | Diagnostic | N | Base | TGD | CN |
|---|---|---:|---|---|---|
| IC fused | Wet-up signed offset (d) | 449 | +5.0 [Q25 0.0,Q75 52.0]; CI [+2,+12] | +14.0 [0.5,60.0]; CI [+9,+28] | +3.0 [0.0,30.5]; CI [+2,+4] |
| IC fused | Peak signed offset (d) | 449 | −50.5 [−93.0,−7.0]; CI [−62,−43] | −54.0 [−86.5,−30.0]; CI [−59,−49] | −29.0 [−79.0,0.0]; CI [−38,−23] |
| dPL available-seed basin median (nominally 3) | Wet-up signed offset (d) | 449 | +4.0 [0.0,48.0]; CI [+2,+9] | +7.0 [0.0,54.0]; CI [+4,+10] | +2.0 [0.0,11.0]; CI [+1,+3] |
| dPL available-seed basin median (nominally 3) | Peak signed offset (d) | 449 | −25.0 [−70.0,0.0]; CI [−33,−18] | −25.0 [−77.5,+1.0]; CI [−32.0,−18] | −17.0 [−65.0,0.0]; CI [−25,−12] |

dPL seed-42-only TGD has N=434 and peak median −28 d；这不是上表的 dPL available-seed population median（nominally three seeds）value。Peak 与 wet-up ordering 也不一致，不能强写成 `CN > TGD > Base` 的统一排序。

## 13. Timing-definition sensitivity

### 表 I

`robustness_timing_sensitivity.csv` 直接保存 Base/CN 字段；它**不保存 TGD 字段**。下表的 Base–CN 列是该 CSV 的直接值。为完整核对既有 manuscript/legacy Table S7，Base–TGD 列保留 `generate_three_structure_r4_all.py:818–827` 中的硬编码 legacy summary，但标为 **derived/legacy、非当前 machine-readable frozen output**；若正文只接受 canonical machine-readable evidence，应删除该列或仅报告 Base–CN。这里的 dPL 是 Figure 10 sensitivity 的 `dPL_seed42` visualization，IC 是 `IC_fused`；source 保存的是 pooled valid basin-year statistics（每行 `n_valid_basin_years=6541`），不是表 H 的 N=449 basin-median CI。数值为 `Base MAE - Model MAE`，正值表示 model absolute timing error 减少。
| Estimation | Timing definition | Base–CN MAE reduction (d; direct CSV) | Base–TGD MAE reduction (d; derived/legacy†) |
|---|---|---:|---:|
| IC fused | Wet-up 7 d | +14 | +2 |
| IC fused | Wet-up 14 d | +21 | +5 |
| IC fused | Wet-up 21 d | +28 | +6 |
| IC fused | Peak full water year | +19 | +3 |
| IC fused | Peak Mar–Aug | +8 | +2 |
| dPL seed 42 | Wet-up 7 d | +11 | +2 |
| dPL seed 42 | Wet-up 14 d | +18 | +4 |
| dPL seed 42 | Wet-up 21 d | +24 | +5 |
| dPL seed 42 | Peak full water year | +9 | +3 |
| dPL seed 42 | Peak Mar–Aug | +5 | +2 |

**判断**：Base→CN 在 IC/dPL 的五种 sensitivity definition 下均为正。Base→TGD 在下表所列 legacy summary 中也均为正且较小，但这些 TGD 数值不是 `robustness_timing_sensitivity.csv` 的当前 machine-readable frozen fields；它们应作为未冻结的 legacy supporting evidence，不应独立支撑“稳定改善”的正文结论。两种 sensitivity 还分别使用 pooled basin-years/seed-42，而表 H 的 dPL population 使用 available-seed basin medians；因此不能把它们写成同一 population estimand。

## 14. 两个 Q75 高雪子集的定义核准

- **Synthetic seasonal subset（Figure S4）**：CAMELS `frac_snow` upper quartile，threshold `0.21769666937653748`，N=133；source `fig6_seasonal_meta.json`。
- **External timing subset（Figure 10）**：Snow-17 ensemble-median annual maximum SWE upper quartile，threshold `133.38849639892578 mm`，N=133；source `swe_basin_burden_test.csv` / `figure8_r4_selection_audit.json`。
- 两个集合仅因样本数相同而容易混淆；现有 source membership overlap 为 **117**，各自有 16 个 basin 不在另一集合。正文必须分别写 `f_snow Q75` 与 `Snow-17 SWE-burden Q75`。
- Figure 10 固定示例的 external Q3 threshold 同为 133.388496 mm；这不改变 Figure S4 的 synthetic subset definition。

## 15. Example vs population evidence

### Figure 10 fixed example

`manuscript/scripts/r4/figure8_r4_selection_audit.json` 的 selection protocol 是：在 external Q3 basins、external snow-active water years（annual SWE max >=5 mm、至少 300 finite SWE days）内，按 annual SWE peak 与 basin SWE burden 到 joint median 的 standardized distance 选择；tie-break 为 basin ID、water year。没有使用 model Wtot、KGE、Delta-r 或 timing outcome。固定例子为 **basin 09306242, WY 2004**，annual SWE max 388.426 mm，basin burden 359.407 mm。

### Figure S5 fixed examples

`FigureS1_R4_selection_audit.json` 使用 external Snow-17 burden tercile ranking，固定每组两个、outcome-independent。显示 basin/WY 为：

| Group | Basin | Displayed WY |
|---|---|---:|
| Low | 02472000 | 1996 |
| Low | 07195800 | 2009 |
| Middle | 05495000 | 2008 |
| Middle | 03473000 | 1999 |
| High | 12167000 | 1997 |
| High | 08377900 | 2005 |

示例只展示轨迹，不支持 population-level timing 数字。Figure S5 population panel 的 external example/corroboration eligibility 为 N=442（Low/Middle/High = 88/177/177），不同于 phase common-support N=331，也不同于 timing eligible N=449。population 结论必须来自 N=331、N=449 或 N=133 的对应 source，而不是六个例子。

## 16. Manuscript / Supplement / canonical 冲突审计

| Quantity / issue | Manuscript / legacy statement | Direct canonical finding | Writing disposition |
|---|---|---|---|
| State normalization | 早期文字可能暗示 range/未说明 | `RMSE/(truth.std()+1e-8)`，period/basin-specific | 脚注明确 standard-deviation normalization |
| CN residual | 可能被误读为 0 | `e_CN` 实际保存并用于 subtraction；Table A CN residual 非零 | 不得预设 CN=0 |
| Final common state set | 早期 protocol 有 `s` primary、`wd` secondary | final Figure S3 renderer/README 使用 `wu,wl,wd,wt,qi,qg`；`s`/`fr` 不进正文 | 采用 Wtot，不自行恢复 `s` |
| Figure S3/S4 numbering | legacy R3/SI 曾有 Figure S6 或 seasonal Figure S2 | `final_assets/README.md` 冻结为 Figure S3 component、Figure S4 seasonal | final asset registry 优先；正文新图号另编号 |
| Figure S4 caption notes | `manuscript/supplement/FigureS4_caption_notes.md` 仍是 TGD response-shape sensitivity 的旧编号 | final asset `figures/Figure_S4/` 与 README/provenance 明确是 seasonal trajectory | 以 final_assets Figure S4 为 seasonal；response-shape 是 final Figure S2/旧 Figure S4 证据，不并入本节 |
| R4 main figure number | repository current source uses R4 Figure 7/8 | thesis 3.5 plans Figure 3-11/3-12 | 不把仓库编号当博士图号；引用 canonical path |
| Active-melt positive fraction | expected 84%/81% | 84%/81% 是 available-case overall（N=344）；N=331 complete-case fractions 不同 | 正文标清 support；不可把两个 N 混用 |
| Phase population | available all-phase =344 | panel-A complete finite common support =331，110/110/111 | Figure 3-11 phase×burden 主矩阵用 N=331 |
| Timing N | Q3 example/ECDF N=133 与 population timing N=449 混用风险 | 449 是 all timing-eligible；133 是 external SWE Q3 | population H 用 N=449；Q3 只作 Figure 10 display |
| dPL peak `−25` vs `−28` | manuscript population `−25`; Figure seed-42 `−28` | three-seed basin median `−25`；seed42-only TGD N=434 `−28` | 不混用；脚注写 seed handling |
| Timing sensitivity vs Table H | values may appear inconsistent | sensitivity is pooled 6,541 basin-years；H is per-basin median then N=449 population | 并列保留，注明 estimand/aggregation |
| Synthetic high-snow N=133 vs external Q3 N=133 | same N 易被视为同一 subset | f_snow threshold 0.2176966694；SWE threshold 133.388496 mm；overlap 117/133 | 正文写清两个 definition |
| dPL high-snow seasonal Wtot | 期待 CN-refit always nearest truth | frozen `CN_dPL` has persistent −30 mm departure；IC CN near zero | 不写“CN 在 IC/dPL seasonal trajectory 都最接近” |
| R4 TGD provenance | README/protocol still says TGD2_PENDING | three-structure R4 metadata says `interim_tgd2_provenance_qualified`；dPL seed42 has non-finite parameter rows, invalid state rows excluded | real-catchment TGD 只作 qualified external consistency evidence；不称 fully canonical observation-trained TGD |
| ERA5/Snow-17 semantics | 可被简写成真实 soil/SWE | both are model-derived external references | 禁写 CN soil moisture 更真实、ERA5/Snow-17 为 truth |

## 17. 3.5 正文图表规划

### 推荐三级标题

1. **3.5.1 合成流域中的内部状态与通量偏差**
2. **3.5.2 出口恢复与内部水文恢复的关系**
3. **3.5.3 高积雪条件下的季节水量进入与储存差异**
4. **3.5.4 真实流域中的外部状态一致性与春季时序**

不建议恢复原 3.5.3/3.5.4 的进一步拆分：phase-resolved external consistency 与 spring timing 使用同一 ERA5-Land/Snow-17 evidence family，合并可避免重复介绍样本口径；在 3.5.4 内用两个段落/两个 panels 分开 phase 与 timing 即可。

### 图 3-9：受控真值下的内部水文偏差

推荐保留四个信息块：

- (a) Base/TGD excess NRMSE，IC；
- (b) Base/TGD excess NRMSE，dPL；
- (c) recovery vs Wtot/internal error 的 raw 与 f_snow-adjusted association；
- (d) Wtot snow gradient。若正文追求简洁，WD 不放 panel d，因 dPL S3 的数值退化；完整 Table C 进附录。

图注须声明 CN residual subtraction、NRMSE standard-deviation normalization、basin bootstrap，并注明 Table D 的 expanded rows 是 lightweight audit，而不是新的 model result。

### 图 3-10：高雪流域水量进入与储存季节过程

(a) generating CN truth/Base/TGD/CN effective input trajectory；(b) signed Wtot seasonal departure。IC/dPL 上下排列。不要添加未经冻结定义的 winter/spring fraction scalar；在图内用 April/December trajectory anchors 或直接图形描述。应显式保留 dPL CN 的 negative seasonal departure，不强行画成 CN/TGD/Base 的统一中间排序。

### 图 3-11：真实流域外部状态一致性

沿用 R4 phase matrix 逻辑：burden × phase `Delta r`，以 N=331 complete-case 为主；available-case positive fractions 只在图注或附录交代。强调外部一致性而非 truth validation。

### 图 3-12：真实流域春季土壤水时序

正文保留 N=449 population timing summary + timing-definition sensitivity；固定 example 09306242 WY2004 可放一个小 panel，六个 Figure S5 examples 进附录/补充。若版面不足，ECDF/example 全部移补充，只保留 Table H/表格化 summary 与 sensitivity。

## 18. 建议进入正文的关键数字

建议只筛选以下 12–18 组：

1. Wtot excess：IC Base/TGD `0.270/0.210`；dPL `0.040/0.004`。
2. 一个 IC component exception：Qi `0.418/0.447`（Base/TGD）。
3. 一个 dPL path example：WL `0.345/0.076`，或 Qg `0.334/0.287`。
4. CN-refit Wtot truth-relative error：IC `0.307`、dPL `0.278`，明确为 non-zero residual。
5. recovery–parameter raw→partial：IC `+0.601→+0.023`；dPL `+0.812→+0.138`。
6. recovery–Wtot raw→partial：Base IC `+0.414→−0.065`；TGD dPL `−0.327→−0.406` exception。
7. high-snow synthetic input：Base December peak/anchor约 `3.50`，truth/CN April约 `6.39/6.42`（IC）；TGD April `5.92`。
8. high-snow Wtot：IC Base February `+88.9 mm`、TGD April `+31.1 mm`、CN约 `−1.5 mm`；dPL CN seasonal departure约 `−30 mm`需作为限制/重要结果。
9. active-melt CN–Base complete-case median：IC low/mid/high `+0.068/+0.276/+0.523`；dPL `+0.057/+0.240/+0.464`。
10. active-melt complete-case positive fraction与available-case 84/81冲突必须在正文/图注统一：主矩阵 N=331，84/81仅 available-case overall。
11. accumulation high-burden CN–Base约 `+0.140`（IC/dPL complete-case）。
12. post-melt/dry-down CN–Base medians接近 0（绝对量级约 `0.001–0.021`），作为 seasonal negative/reference condition，而不是“没有任何差异”。
13. phase common support N=331；timing eligible N=449。
14. timing wet-up：IC Base/TGD/CN `+5/+14/+3 d`；dPL `+4/+7/+2 d`。
15. timing peak：IC `−50.5/−54/−29 d`；dPL `−25/−25/−17 d`。
16. sensitivity：Base→CN 的 direct CSV gains 为 IC `14–28 d`（wet-up）、`8–19 d`（peak），dPL seed42 为 `11–24 d`、`5–9 d`；Base→TGD 的 `2–6 d`/`2–5 d` 仅来自未冻结的 legacy Table S7 hard-coded summary，不作为 canonical machine-readable 主结果。
17. external Q3 N=133、threshold 133.388496 mm；不要与 synthetic f_snow Q75 N=133 混写。

## 19. 建议进入脚注和附录的数据

**脚注**：

- NRMSE denominator 是 same-basin/same-period truth standard deviation + `1e-8`；CN residual 不是 0。
- `E_excess = NRMSE_model − NRMSE_CN`；TGD recovery 使用 3.4 的 `G_TGD` common-reference input，不重新定义 F。
- recovery partial Spearman 控制 ranked `f_snow`，不作因果 mediation。
- `f_snow` Q75 与 Snow-17 SWE Q75 都是 N=133 但 overlap=117 的不同集合。
- ERA5-Land SM100/Snow-17 是 model-derived external references。
- dPL state/seasonal population 使用 basin-wise three-seed median；seed 42 只用于明确标注的 visualization/sensitivity。

**附录**：

- 表 A 的完整 Q25/Q75/CI；
- 表 B/C 全变量与 S1–S5；特别保留 WD dPL S3 numerical-degeneracy note；
- 表 D 全 recovery–internal pairs与bootstrap CI；
- Figure S4 full 12-month arrays/CI；
- Figure S5 six fixed examples and N=442 population asset；
- Figure 3-11 full phase × burden table，包括 N=331 complete-case 与 available-case cross-check；
- timing full source rows、Q3 ECDF、seed42 vs three-seed median reconciliation；
- 7/14/21 d and full/Mar–Aug timing sensitivity。

## 20. 是否还存在数据缺口

1. **没有 synthetic 3.5 的核心数据缺口**：truth、Base/TGD/CN state/flux、seasonal input/storage arrays、bootstrap-ready basin rows 均存在。
2. **timing population 核心数据齐全，但 sensitivity 有口径缺口**：N=449 wet-up/peak summary、Q25/Q75、CI 与 Base→CN sensitivity 均存在；Base→TGD sensitivity 仅有未冻结的 legacy hard-coded summary，当前 CSV 未保存 TGD 字段。
3. **存在 support-label gap**：N=331 complete-case 与 manuscript 约 84%/81% available-case positive fractions必须统一标注；不能静默择一。
4. **存在 R4 TGD provenance qualification**：TGD2 README/protocol 仍为 pending，当前 three-structure figure7 metadata 明确是 interim source；dPL seed42 非有限 parameter/state 行被排除。qualified descriptive use 可行，fully-canonical TGD claim 不可无条件写。
5. **存在 dPL high-snow seasonal discrepancy**：`CN_dPL` 的约 −30 mm Wtot departure 不是缺失值，而是 frozen array 的 systematic departure；需要正文保留为限制/重要结果，而不是重跑模型来消除它。
6. **WD dPL S3 normalization degeneracy**：不是本次任务应通过新模型训练解决的问题；正文避开该 cell，附录完整报告。
7. **没有 stable frozen winter/spring delivery fraction scalar**：Figure S4 支持 trajectory-level description；若正文需要 fraction，必须另行提出并批准新的 descriptive estimand，当前不建议增加。

## 21. 最终判断

1. **当前数据是否足够写 3.5？** 足够写出 synthetic truth-relative state/flux、outlet recovery ≠ internal recovery、高雪 seasonal pathway，以及 real-catchment external-consistency/timing 的分层结果；但 phase positive fraction 与 support label、TGD provenance 和 dPL seasonal CN offset 必须在正文/图注中明确。
2. **是否存在 scientific blocker？** 没有阻挡“qualified 3.5”写作的核心 scientific blocker。若要求把 R4 TGD2 写成 fully canonical observation-trained external validation，当前存在 provenance blocker；若要求把 CN 在 IC/dPL 两种 high-snow seasonal trajectory 都写成最接近 truth，frozen dPL seasonal output 直接否定该表述。
3. **是否需要新模型训练？** 不需要。
4. **是否需要重新率定？** 不需要。
5. **是否只需要 existing frozen outputs 上的轻量统计？** 是。表 A/B/C/D/G 的补充数值均来自已冻结 CSV/JSON 的 basin-level reaggregation；表 E/F 直接读取冻结 monthly summary/arrays；不运行 model forward。
6. **哪些结果是 synthetic truth evidence？** 表 A/B/C、表 D 的 truth-relative/paired excess 与 recovery association，以及表 E/F 的 generating-CN seasonal input/Wtot departure。
7. **哪些结果只是 real-catchment external consistency evidence？** 表 G 的 ERA5-Land SM100 `Delta r`、Snow-17 burden/phase 组织，以及表 H/I 的 external-reference timing offset/sensitivity。它们不是 truth validation，不能证明 CN soil moisture 更真实，也不能证明 TGD physical error。
8. **推荐 3.5 最终三级标题**：`3.5.1 合成流域中的内部状态与通量偏差`；`3.5.2 出口恢复与内部水文恢复的关系`；`3.5.3 高积雪条件下的季节水量进入与储存差异`；`3.5.4 真实流域中的外部状态一致性与春季时序`。

**最终写作主线**：在同一受控 synthetic truth 下，CN-refit 的共同内部状态/通量总体最接近 generating system，但仍有 non-zero residual；TGD 通常减少 Base 的 internal excess，却并非每个 component 都改善。outlet recovery 与 internal error 的 raw association 多由 snow activity 的共同梯度组织，控制后没有统一 recovery→internal pathway。高雪 seasonal arrays 显示 Base 的 liquid-water delivery 与 Wtot departure 偏向冬季/早春，TGD 向春季移动但保留 residual；dPL CN 的 seasonal Wtot 仍有系统性负 departure。真实流域中，CN–Base 外部 SM100 consistency 的清晰改善集中在 accumulation/active melt 并随 external SWE burden 增强；spring wet-up/peak timing 也改善，但 TGD 不在所有 timing definition 下稳定成为 CN 与 Base 之间的统一中间状态。