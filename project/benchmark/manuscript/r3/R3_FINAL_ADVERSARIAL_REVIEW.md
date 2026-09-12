# R3 Final Adversarial Statistical Review

## Question

IC 中独立形成的 catchment--parameter association，换成 canonical dPL 这种属性驱动的参数估计方式后，是否仍在 information level、方向和 parameter profile 上保持，而不是把 dPL 自己的构造性 mapping 当作独立证据？

## Data and provenance

- Relationship matrices: `tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv`。
- IC/dPL 均来自 R2 canonical table 的 normalized-u 参数坐标；IC 是独立 basin-wise fit，dPL 是 canonical v2 seed 42 X-to-theta network。
- raw space 有 35 attributes；information space 使用 R2 预先固定的 0.70 average-linkage cluster PC1。
- 36 models、531 canonical basins/model；SIMHYD generation 280 只作为 all36 成员，exclude-SIMHYD 全程保留。
- 所有正式 basin bootstrap/permutation 均固定 seed，B=1,000；未启动训练、优化、checkpoint 更新或 OOB/PUB/PUR。

## Estimand

Frozen primary is the unfiltered information-cluster parameter-profile estimand:

\[
R_{m,p}=Spearman(\rho^{IC}_{m,p,k},\rho^{dPL}_{m,p,k}),\quad
R_m=median_p(R_{m,p}),\quad R_{overall}=median_m(R_m).
\]

Models contribute equally；参数只在模型内通过 median 聚合。raw-attribute profile、cell-equal、model-flattened 和 IC-anchored profile are sensitivities. To address dPL construction directly, an IC-anchored profile is also computed using only cluster features with IC `|rho|>=0.20` and IC bootstrap sign probability `>=0.95`; dPL does not select that denominator.

## Denominator

- Primary unfiltered cluster profile: 271 model-parameter profiles, 36 model medians。
- exclude-SIMHYD: 264 profiles, 35 model medians。
- IC-anchored cluster profile: 151 profiles/all36 and 145/exclude-SIMHYD，because the IC-only stability gate is intentionally restrictive。
- Sign denominators are explicit in `R3_SIGN_AGREEMENT_AUDIT.csv`；raw all36=9,485，cluster all36=5,420。

## Method

Built the same IC/dPL relationship estimator in raw and cluster spaces, computed 1,000-resample relationship/bootstrap summaries, froze profile aggregation, then applied a joint within-model permutation of dPL basin rows. Dominant controls use deterministic absolute-rho ranking and separate estimator-specific bootstrap top-1 stability. Restart identifiability uses the existing archived ten-start IC uncertainty table；no restart was rerun。

## Result

1. **Profile correspondence:** unfiltered cluster primary all36=`0.715789`，exclude-SIMHYD=`0.712782`。IC-anchored cluster sensitivity=`0.819643`/`0.810714`。raw primary=`0.732983`/`0.726751`。
2. **Null:** unfiltered primary all36 observed=`0.715789` vs permutation null mean=`-0.001455`，95% interval=`[-0.086109, 0.082716]`，empirical p=`0.000999`。IC-anchored sensitivity observed=`0.819643`，null mean=`-0.007601`，p=`0.000999`。exclude-SIMHYD is also p=`0.000999`。The maximum-model unfiltered p is `0.000999`; maximum IC-anchored p=`0.369630`，so the anchored maximum-model claim is not supported even though the model-equal median is above null。
3. **Sign:** historical-like raw all-cell all36 sign agreement=`7084/9485=0.746863`。IC-stable/nontrivial raw=`1769/1892=0.934989`；cluster=`849/902=0.941242`。Thus the 74.6% number is a weak all-cell comparability summary, not the stable-cell headline。
4. **Dominant information:** raw top-1 exact agreement=`0.184502`；cluster top-1=`0.298893`。Raw Top-3 mean overlap count=`0.966790` and Top-5=`2.121771`；cluster Top-3=`1.405904` and Top-5=`2.745387`。Within-cluster correlated-proxy substitution is `13.122%` of raw top-1 mismatches (`10.701%` of all model-parameter pairs)。IC/dPL estimator-specific bootstrap top-1 stability is stored per pair and is not assumed perfect。
5. **dPL construction artifact:** among dPL-strong cluster cells, only `692/(738+20+692+713)=0.319926` are IC-supported；`68.0074%` are IC-weak、IC-absent 或 IC-opposite-sign。Raw dPL-strong IC-supported fraction=`0.339963`。Therefore most dPL-strong cells cannot be used as independent hydrological validation。
6. **Identifiability:** pooled parameter-level Spearman between median restart-u SD and profile reproducibility is `rho=-0.374376`，bootstrap CI `[-0.473632,-0.270936]`，n=271。Restart-u IQR=`-0.304304`，range=`-0.378199`；boundary fraction=`-0.229779`。High-identifiability versus low-identifiability cluster profile medians are `0.804511` versus `0.612030`。This is an association, not a causal decomposition。
7. **Influence:** primary cluster model-equal LOO range=`0.712782–0.718797`；exclude-SIMHYD does not depend on a fictitious SIMHYD-removal row。Sign-stable LOO cluster range=`0.935835–0.945455`。The primary median is not driven by one model。

## Sensitivity

- `exclude_simhyd` does not materially change the unfiltered profile or the IC-anchored median。
- Raw versus cluster spaces differ in absolute value but preserve the broad conclusion that estimator correspondence is positive and above the basin-correspondence null。
- IC-anchored values are higher, but their feature denominator is only 151/145 profiles and must not be confused with the unfiltered primary。
- Raw top-1 mismatch is much larger than cluster top-1 agreement, but proxy substitution explains only a minority of mismatches under the frozen cluster mapping。
- High versus low restart uncertainty shows a substantial association, yet residual profile heterogeneity remains and no percentage of disagreement is assigned to identifiability。

## Adversarial interpretation

一个不友好的 reviewer 可以成立地指出：

1. dPL 强 relationship 中约三分之二没有 IC support；dPL 本身不能是独立 validation。
2. The unfiltered primary includes cells that are dPL-strong but IC-weak/absent; it is therefore a descriptive cross-estimator correspondence summary, not a clean independent replication。
3. IC-anchored profile is stronger and exceeds its null, but it is a post-hoc sensitivity added to answer the construction objection and has a smaller, explicitly selected IC-only denominator。
4. Cluster agreement is not causal information recovery；clusters remain correlated proxy groups。
5. IC restart uncertainty is associated with persistence, but this does not prove identifiability causes disagreement；model/parameter heterogeneity remains。
6. The maximum-model anchored null is not significant，so claims about the single best anchored model must be omitted。

## Verdict

**R3_READY_WITH_REFRAMING**

可进入 manuscript 的窄化表述：IC-independent catchment--parameter associations show positive cross-estimator persistence at the model-equal information-profile level and remain strong on an IC-only stable-feature sensitivity. 必须把 unfiltered `0.715789` 写成 descriptive primary summary，把 `0.819643` 明确写成 IC-anchored sensitivity，并明确约 68% dPL-strong cluster cells are not IC-supported。不得把 dPL mapping 作为独立 hydrological validation，不得宣称 identifiability causally explains a fixed fraction of disagreement。

## Output map

- `tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv`
- `tables/R3_PARAMETER_PROFILE_REPRODUCIBILITY.csv`
- `tables/R3_MODEL_REPRODUCIBILITY.csv`
- `tables/R3_OVERALL_REPRODUCIBILITY.csv`
- `tables/R3_SIGN_AGREEMENT_AUDIT.csv`
- `tables/R3_DOMINANT_ATTRIBUTE_AGREEMENT.csv`
- `tables/R3_DOMINANT_CLUSTER_AGREEMENT.csv`
- `tables/R3_CORRELATED_PROXY_SUBSTITUTION.csv`
- `tables/R3_CROSS_ESTIMATOR_NULL.csv`
- `tables/R3_IDENTIFIABILITY_REPRODUCIBILITY.csv`
- `tables/R3_MODEL_LEVEL_IDENTIFIABILITY.csv`
- `tables/R3_DPL_CONSTRUCTION_ARTIFACT_AUDIT.csv`
- `tables/R3_POOLED_INFLUENCE.csv`
