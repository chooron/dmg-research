# R3 Claim–Evidence Map

## Headline 1 — Same-parameter correspondence

**Claim:** Same-parameter IC–dPL information profiles show substantial but incomplete correspondence.

**Evidence:** `R_paired = 0.7157894737`, model-equal across 36 models and 20 frozen information dimensions. Source: `tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv` and `R3_PARAMETER_IDENTITY_CORRESPONDENCE_AUDIT.md`.

**Boundary:** correspondence of association profiles is not parameter identity, physical meaning, or causal information retention. Stable-cell sign retention (`849/902 = 94.12%`) is conditional on an IC-only stable subset and is not an all-cell rate.

## Headline 2 — Same-coordinate specificity

**Claim:** Information-profile correspondence is preferentially aligned with the same parameter coordinate.

**Evidence:** `A_diag = 0.6150375940`, diagonal median `0.7157894737`, off-diagonal median `-0.0218045113`, 35 valid model contributors; within-model parameter-label permutation `p = 0.0009990010`. SIMHYD exclusion gives `A_diag = 0.6105263158` and `p = 0.0009990010`.

**Boundary:** the result is a model-level descriptive same-coordinate-versus-alternative comparison. It does not prove physical identity or functional semantics.

## Role boundary — No broad functional-role continuity

**Claim:** Same-coordinate specificity does not generalize to broad functional-role continuity.

**Evidence:** `A_role = -0.1187969925`, role-count-preserving permutation `p = 0.8057194281`, same-role off-diagonal median `-0.0278195489`, cross-role median `0.0052631579`; HESS-prior `D_rho = -0.0003301577`, `n=27`, CI `[-0.0066882428, 0.0265313077]`, `p=0.8498150185`.

**Boundary:** the correct placement is **COORDINATE SPECIFICITY ONLY**. Role coverage and source-backed labels are limited and remain a negative control, not a claim that roles are absent.

## Rank robustness

**Claim:** Same-coordinate information specificity remains beyond raw parameter-rank continuity.

**Evidence:** rank-linkage correlations are real (`R_rank`–`R_info`: pooled `0.743830`, model-centered `0.683142`; `Q_rank`–`Q_info`: pooled `0.863927`, model-centered `0.860931`). The rank-matched residual test (`k=3`, no caliper) has coverage `270/271`, `A_info|rank = 0.295238`, model-clustered CI `[0.254386,0.429073]`, positive in all matched model summaries, sign-flip `p=0.000400`.

**Boundary:** raw rank continuity explains part of profile correspondence and is not noise. This audit does not support noise-removal or informative-ranking-preservation claims.

## Supporting bridge — Performance conditioning

**Claim:** Larger outlet-performance differences are associated with greater profile divergence.

**Evidence:** high-minus-low within-model `abs(DeltaKGE)` strata: divergence contrast `+0.028033`, CI `[0.017163,0.043610]`, balanced null `p=0.000999`; paired profile correspondence contrast `-0.141353`, CI `[-0.215038,-0.078947]`.

**Boundary:** this is an observational, composition-sensitive supporting association. Secondary `D_theta` conditioning is mathematically coupled and is not an independent causal test.
