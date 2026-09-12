# R3 Research and Analysis Summary

## Final verdict

**FREEZE R3 — CORE SUPPORTED WITH QUALIFICATION**

This freeze is a synthesis of the authoritative frozen R3 products and the A–E audits. No source product, script, cache, checkpoint, canonical result, training run, recalibration, simulation, or dPL seed was changed or added.

## 1. Scientific question

R3 asks whether the 20-dimensional catchment-information association profile of a conceptual-model parameter remains corresponding between basin-wise independent calibration (IC) and shared attribute-constrained dPL, despite parameter-value reorganization. The target is information-profile organization, not equality of parameter values, physical identity, or causal function.

R2 parameter-value change and R3 information-profile correspondence are distinct estimands: parameter-value reorganization can coexist with profile correspondence.

## 2. Frozen universe and aggregation levels

- 36 conceptual hydrological models.
- 531 common CAMELS-US basins per model.
- 35 static catchment descriptors and 20 frozen information dimensions.
- 271 model–parameter coordinates and 5,420 parameter–information cells (271 × 20).
- IC: basin-wise independent calibration.
- dPL: canonical seed 42 only, shared attribute-constrained parameter learning.

Association cells and parameter/pair rows are descriptive. Model-equal summaries are the cross-model inferential level unless otherwise stated; pooled values are explicitly labeled.

## 3. Primary finding 1 — same-parameter profile correspondence

For each model and parameter, the 20-dimensional vector of basin-wise Spearman parameter–information associations is compared between IC and dPL. The model-level median across its coordinates is then aggregated equally across models. The frozen same-parameter profile correspondence is:

- `R_paired = 0.7157894737` (36 models; model-equal median).

This is substantial but incomplete correspondence of association profiles. It does not establish parameter identity, preserved physical meaning, functional-role continuity, IC truth, dPL superiority, or causal information transfer.

### Stable-cell sign retention boundary

The IC-stable subset is selected using IC only:

- `abs(rho_IC) >= 0.20`;
- IC bootstrap sign probability `>= 0.95`.

The dPL value is not used to select the denominator. Among `902/5,420` selected cells, same-sign retention is `849/902 = 0.9412416851` (94.12%). This is conditional on the 902-cell stable subset, not a rate over all 5,420 cells. With additional dPL magnitude requirements, same-sign rates are 87.694% (`|rho_dPL|>=0.10`), 76.718% (`>=0.20`), and 57.317% (`>=0.30`).

## 4. Primary finding 2 — same-coordinate specificity

The frozen within-model profile-correspondence matrix compares the IC profile for coordinate `p` with dPL profile `q`. The same-coordinate diagonal is compared with all finite within-model off-diagonal alternatives, and the model-equal diagonal advantage is:

- diagonal median `0.7157894737`;
- off-diagonal median `-0.0218045113`;
- `A_diag = 0.6150375940`;
- valid model contributors `35/36` (one-parameter model has no off-diagonal contrast);
- within-model parameter-label permutation `p = 0.0009990010` over 1,000 fixed-seed permutations.

SIMHYD exclusion retains the result: `A_diag = 0.6105263158`, valid contributors `34/35`, permutation `p = 0.0009990010`.

The narrow supported claim is preferential same-coordinate alignment against within-model alternatives. It is not parameter identity, physical equivalence, or functional-role preservation.

## 5. Negative functional-role extension

After original same-coordinate pairings are removed, source-backed role labels do not add a stable correspondence signal:

- `A_role = -0.1187969925`;
- role-count-preserving permutation `p = 0.8057194281`;
- same-role off-diagonal median `-0.0278195489` (`n=29`);
- cross-role median `0.0052631579` (`n=29`).

The pre-specified HESS-prior flexible-role comparison is also null:

- prior-minus-comparison `D_rho = -0.0003301577`;
- comparable models `n=27`;
- bootstrap 95% CI `[-0.0066882428, 0.0265313077]`;
- sign-flip `p = 0.8498150185`.

This negative control limits the interpretation to **COORDINATE SPECIFICITY ONLY**. It does not claim that hydrological roles are absent; it says that the tested role labels do not provide an additional stable signal beyond the coordinate effect.

## 6. R2→R3 rank linkage and residual specificity

Raw parameter-rank continuity accompanies profile correspondence and must not be dismissed as noise:

- `R_rank`–`R_info`: pooled `0.743830`, model-centered `0.683142`, median within-model `0.678571`;
- `Q_rank`–`Q_info`: pooled `0.863927`, model-centered `0.860931`, median within-model `0.859901`.

The rank-matched hostile test compares each same-coordinate pair with the three nearest within-model off-diagonal alternatives in absolute raw `Q_rank` distance (`k=3`, no caliper):

- coverage `270/271 = 0.996310`;
- residual advantage `A_info|rank = 0.295238`;
- model-clustered 95% CI `[0.254386, 0.429073]`;
- positive in all matched model summaries;
- sign-flip `p = 0.000400`.

Thus rank continuity explains part of profile correspondence, but same-coordinate information specificity remains beyond raw rank continuity under the frozen matching rule. This does not show that dPL removes noisy ranking or preserves informative ranking.

## 7. Performance-conditioned supporting result

Within each model, the 531 basins were ordered by `abs(DeltaKGE)` and divided into equal-size low/middle/high strata (`177` basins each). The supporting associations are:

- high-minus-low profile divergence: `+0.028033`, bootstrap 95% CI `[0.017163, 0.043610]`;
- high-minus-low paired profile correspondence: `-0.141353`, CI `[-0.215038, -0.078947]`;
- balanced equal-size subset null: 36,000 model draws, one-sided `p=0.000999`.

This is observational conditioning. Basin composition may contribute, and the secondary `D_theta` conditioning is mathematically coupled to the parameter realizations used in the profiles. No causal performance-to-information claim is made.

## 8. Evidence classification and final interpretation

**Primary:** same-parameter profile correspondence and same-coordinate specificity; rank-matched residual specificity is the primary R2→R3 protection.

**Supporting:** raw rank linkage, stable-cell conditional sign retention, and `abs(DeltaKGE)`-conditioned profile contrasts.

**Negative/boundary:** broad functional-role continuity and HESS-prior flexible-role sensitivity are not supported; dPL construction and one-seed design limit physical interpretation.

Final R3 claim: same-parameter IC–dPL association profiles show substantial but incomplete correspondence, and same-coordinate specificity remains after accounting for raw parameter-rank continuity. This specificity does not generalize to broad functional-role continuity.

## Short writing forms

### 中文
尽管 IC 与 dPL 的参数实现发生重组，同一参数的信息关联画像仍具有同坐标特异性，但不支持宽泛功能角色连续性或因果解释。

### English JoH Results sentence
> Despite substantial IC–dPL parameter reorganization, same-parameter catchment-information profiles showed substantial but incomplete correspondence, with residual specificity for the same parameter coordinate rather than broad functional-role continuity.

### Boundary sentence
> The dPL evidence is canonical seed 42 only; IC-self and within-paradigm uncertainty are one-sided and incomplete, all findings are association-profile comparisons rather than causal decompositions, and no physical meaning, functional role, parameter identity, IC truth, or dPL superiority is established.
