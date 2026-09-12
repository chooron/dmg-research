# R2–R3 Final Statistical Verdict

## Scope

本报告只使用 frozen IC/dPL seen-basin artifacts。没有启动 IC/dPL training、GPU training、optimizer、checkpoint 更新、OOB/PUB/PUR、H1 或水文方程修改。正式后处理使用 CPU 向量化、固定最多四个数值线程；本轮规模不足以合理要求 GPU 重写统计定义。

## 1. 是否存在显著高于 null 的 recurring catchment-information dimensions？

**是，但只能作 association-level 结论。** R2 primary cluster `C012 = high_prec_dur; low_prec_dur; gvf_diff` 出现在 `33/36=0.916667` 个模型；1,000 次 basin-label permutation 的 cluster p=`0.000999`，max-statistic p=`0.000999`。Null 保持属性-derived score 与参数 marginal、只破坏 basin correspondence。

## 2. 哪些 information dimensions 最稳定？

Primary 0.70 clustering 下最稳定的是：

- precipitation-duration/vegetation proxy：C012，33/36；
- terrain/soil：C019，30/36；
- vegetation/root：C011，29/36；
- sand/clay：C017，26/36；
- PET/snow：C016，24/36。

这些是统计信息块，不是已证实的独立因果过程。

## 3. recurrence 是否受 attribute clustering threshold 影响？

Broad recurrence 没有消失：0.60、0.70、0.80 三个阈值下最高 all36 recurrence 分别为 `0.916667`、`0.916667`、`0.944444`。但 exact cluster members 和 cluster IDs 会改变，故不能声称 exact descriptor recurrence 完全不受 threshold 影响。

## 4. boundary saturation 是否改变 R2？

**会，且必须明说。** C012 recurrence 从 no-exclusion 的 `0.916667` 降为 boundary fraction `>0.10/>0.20/>0.30` 后的 `0.305556/0.694444/0.777778`。>0.10 是 severe stress test，删除 `243/271=0.896679` 个 IC parameter cells；tie exclusion 相对温和，为 `0.861111/0.888889/0.888889`。因此原始 R2 结论不能表述为 boundary-independent。

## 5. 相同 information 是否真正落在不同 parameter coordinates？

有相当的描述性支持：C012 在 33 个模型中有 stable landing，累计 84 个 stable model-parameter cells；dominant landing names 有 25 个，dominant normalized parameter-index SD=`0.296351`，within-model concentration median=`0.50`。这不只是把不同名字当成证据，但不同模型参数空间不完全可比，故证据应称为 heterogeneous parameter-coordinate expression，而不是 universal functional substitution。

## 6. functional-role alignment 是增强还是削弱解释？

当前结果**削弱强解释**：只有 FLEX family 存在明确的 registry process-role metadata，其余模型大多为 `UNRESOLVED`。角色信息不能被参数名或 bounds 代替，也不能用于事后强化结构依赖 landing 的解释。它只能作为有限 secondary metadata。

## 7. raw dominant attribute disagreement 中多少是 correlated proxy substitution？

Raw top-1 mismatch 占 `81.5498%`；其中 `13.1222%` 属于同一冻结 information cluster 内的 proxy substitution（占全部 model-parameter pairs `10.7011%`）。因此 raw mismatch 不能直接解释为 information disagreement。

## 8. cluster-level agreement 是否明显高于 raw top-1 agreement？

是，但绝对值仍不高：raw top-1 exact agreement=`0.184502`，cluster top-1 agreement=`0.298893`。raw Top-3/Top-5 overlap mean counts=`0.966790/2.121771`；cluster 为 `1.405904/2.745387`。cluster-level 是较宽的信息 estimand，不代表相同参数或相同机制。

## 9. 历史约 74.6% sign agreement 在 IC-stable/nontrivial cells 中是多少？

Raw all-cell all36 为 `7084/9485=0.746863`，与历史约 74.6% 一致。IC-stable/nontrivial（IC `|rho|>=0.20` 且 bootstrap sign probability `>=0.95`）为 raw `1769/1892=0.934989`，cluster `849/902=0.941242`。后者才是主要方向持久性证据；all-cell rate 不能作为稳定关系的 headline。

## 10. 历史 0.658 与 0.733 分别对应什么定义？

- 约 `0.658`：formal legacy label `model_flattened_profile_spearman_median`，即先对每个模型 flatten 全部 parameter×feature relationship cells，再取模型中位数。
- 约 `0.733`：per-model parameter-profile medians 的 model-equal median；当前 formal table 对应约 `0.7326`，本轮 raw model-equal profile 为 `0.732983`。
- 当前 formal 的约 `0.7527` 是 parameter-profile cell-equal median，不能与前两者混用。

这些数字的 observation unit 和 aggregation 不同。

## 11. 正文最终应采用哪个 reproducibility estimand？

冻结的 primary estimand 是：information-cluster relationship profile 的 `median_p` within model，再 `median_m` across models（model-equal），本轮 all36=`0.715789`，exclude-SIMHYD=`0.712782`。Raw、cell-equal、model-flattened 都是 sensitivity。针对 dPL construction objection，IC-only stable-feature profile 为 `0.819643`（all36）和 `0.810714`（exclude-SIMHYD），但它是明确标注的 IC-anchored sensitivity，不能事后伪装成原始 primary。

## 12. IC relationship 自身 bootstrap stability 如何？

对 `|rho|>=0.20` 的 IC cells，bootstrap sign probability median 为 `1.0`（raw 与 cluster）；全体 cells 的 sign-probability median 为 raw `0.988`、cluster `0.981`。全体 35-attribute cells 的 bootstrap CI excludes zero 比例约 `0.5555`，cluster 约 `0.5221`，因为弱关系不应被当成稳定关系。Top-1 stability 已逐 model/parameter 保存，不假定点估计的 dominant attribute 总是稳定。

## 13. observed IC–dPL correspondence 是否高于 permutation null？

是。Unfiltered primary all36=`0.715789`，null mean=`-0.001455`，null 95% interval=`[-0.086109,0.082716]`，empirical p=`0.000999`。IC-anchored sensitivity=`0.819643`，null mean=`-0.007601`，p=`0.000999`；exclude-SIMHYD 也保持 p=`0.000999`。Maximum-model unfiltered p=`0.000999`，但 maximum-model IC-anchored p=`0.369630`，所以不能写 single best anchored model 的普遍结论。

## 14. identifiability 与 cross-estimator persistence 的关联有多强？

Parameter-level pooled Spearman：median restart-u SD 与 profile reproducibility 为 `rho=-0.374376`，bootstrap CI `[-0.473632,-0.270936]`，n=`271`。Restart-u IQR 为 `-0.304304`，restart range 为 `-0.378199`。这表示较差 restart identifiability 与较弱 persistence 相关，但不是 causality。

## 15. identifiability 是否足以解释大部分 disagreement？

不能这样说。High-identifiability 与 low-identifiability subsets 的 cluster profile median 为 `0.804511` 与 `0.612030`，说明关联明显；但 model/parameter residual heterogeneity 仍存在，且 dPL-emergent cells、cluster selection、boundary 和 role metadata 都是独立限制。不能报告“X% disagreement caused by identifiability”。

## 16. pooled conclusions 是否被少数 models/parameters 主导？

Primary cluster profile 的 all36 model-equal leave-one-model-out range 为 `0.712782–0.718797`；IC-stable cluster sign agreement 的 valid LOO range 为 `0.935835–0.945455`。R2 top C012 recurrence 的 all36 LOO range 为 `0.914286–0.942857`，排除 SIMHYD 后仍为高 recurrence。Cell-equal 与 boundary stress 的变化已单独保存，不能被 model-equal headline 隐藏。

## 17. exclude-SIMHYD 后主结论是否改变？

没有实质改变：R2 的 C012 为 all36 `0.916667`、exclude-SIMHYD `0.914286`；R3 unfiltered cluster profile 为 `0.715789`、`0.712782`；IC-anchored 为 `0.819643`、`0.810714`。SIMHYD 仍必须标记为 accepted generation 280，而不是假装 Full300。

## 18. dPL construction artifact 对 R3 的独立证据价值影响多大？

影响很大。dPL-strong cluster cells 中只有 `692/2163=0.319926` 为 IC-supported，约 `68.0074%` 为 IC-weak、IC-absent 或 IC-opposite-sign。Raw dPL-strong IC-supported fraction 为 `0.339963`。因此 dPL-strong/IC-weak 关系不能用于支撑 shared hydrological information；它们仅作为 construction-artifact audit。

## 19. 不友好的 reviewer 会认为 R2 哪些 claim 真正非平凡？

真正非平凡的是：在 IC 独立率定参数上，先用不含参数结果的属性 correlation 定义 information blocks 后，若干 broad blocks 的 model-equal recurrence 显著高于 basin-label null，并且每个模型内部的 parameter landing set 具有异质性。不能把它升级为 causal shared process、boundary-independent mechanism、universal role equivalence 或 geographic transferability。

## 20. 不友好的 reviewer 会认为 R3 哪些 claim 真正非平凡？

真正非平凡的是：IC independently generated 的 association profile 在另一种参数估计方式下仍有 positive model-equal correspondence，并在 IC-only stable-feature sensitivity 下保持且高于 cross-estimator null；稳定方向 agreement 约 0.935–0.941。它检验的是 persistence under a different estimator，不是 dPL 自身的 independent validation。

## 21. 哪些 claim 必须删除或降级？

必须删除或降级：

- association 写成 causality；
- dPL 的 attribute→parameter mapping 写成独立 hydrological validation；
- 把 74.6% all-cell sign rate 写成 stable-cell reproducibility；
- 把 0.658、0.733、0.7527 当成同一个指标；
- 把 raw top-1 mismatch 当成 information disagreement；
- 把 boundary stress 下的主结论写成 boundary-independent；
- 把 opaque group_11..17 写成已验证的 HUC/hydro-climatic geography；
- 把 identifiability association 写成解释了固定比例 disagreement；
- 把 maximum-model IC-anchored null failure 写成普遍稳定性；
- 把 unresolved parameter names 强行映射为 process roles。

## 22. 是否达到正式 manuscript writing 的统计门槛？

**达到窄化表述的门槛，但未达到原始强命题的无条件门槛。**

- R2：`R2_READY_WITH_REFRAMING`。
- R3：`R3_READY_WITH_REFRAMING`。

建议 manuscript 使用：`recurring, null-exceeding IC association dimensions with model-dependent parameter-coordinate expression`，以及 `IC-independent associations show positive cross-estimator persistence, with construction-aware IC-stable sensitivity`。所有 boundary、role、spatial、identifiability 和 dPL-emergent limitations 必须进入正文或 SI，而不能只留在审计文件。

## Final verdicts

- **R2:** `R2_READY_WITH_REFRAMING`
- **R3:** `R3_READY_WITH_REFRAMING`
- **Overall:** 可开始正式 manuscript writing，但只能写上述窄化结论；原始的 causal、universal、boundary-independent、independent-dPL-validation 版本不通过统计审查。
