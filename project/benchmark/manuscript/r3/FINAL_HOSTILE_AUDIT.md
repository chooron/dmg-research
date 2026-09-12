# R3 Final Hostile Audit

## Final disposition

**PASS WITH LIMITATION**

`blocker_count = 0`. Agents A–D correctly recover the frozen R3 evidence chain. The core result is acceptable only at the declared association-profile and same-coordinate level.

## Agent dispositions

- A — primary correspondence: **PASS WITH LIMITATION**.
- B — same-coordinate specificity: **PASS WITH LIMITATION**.
- C — functional-role negative controls: **PASS WITH LIMITATION**.
- D — R2→R3 linkage and conditioned support: **PASS WITH LIMITATION**.
- E — hostile review: **PASS WITH LIMITATION**.

## Hostile questions resolved

1. `R_paired=0.715789` is a same-parameter profile correspondence summary and alone cannot prove identity; the separate diagonal/off-diagonal comparison and label null provide the empirical specificity protection.
2. `94.12%` is `849/902` after IC-only stability selection (`abs(rho_IC)>=0.20`, IC sign probability `>=0.95`), not an all-cell retention rate.
3. Raw rank continuity is real (`R_rank–R_info` and `Q_rank–Q_info` correlations are positive); the rank-matched residual remains `A_info|rank=0.295238`, CI `[0.254386,0.429073]`, `p=0.000400`, coverage `270/271`.
4. Same-coordinate specificity does not imply role preservation: `A_role=-0.1187969925`, role permutation `p=0.8057194281`, and HESS-prior `D_rho=-0.0003301577`, `p=0.8498150185` are negative controls.
5. dPL construction contributes to profile geometry, but the within-model label permutation and rank-matched alternatives are empirical comparisons, not physical or causal proof.
6. `abs(DeltaKGE)` conditioning is supporting observational evidence; equal-size strata and a balanced subset null reduce sampling concerns but do not remove composition confounding or establish direction. Secondary `D_theta` is mathematically coupled.

## Boundary

No claim is made that dPL removes noise, preserves informative ranking, recovers physical parameter meaning, preserves functional roles, establishes IC truth, demonstrates dPL superiority, or identifies a causal mechanism. dPL has seed 42 only and symmetric within-paradigm dPL variability is unavailable. No new analysis or rerun is required.

**Final R3 verdict: FREEZE R3 — CORE SUPPORTED WITH QUALIFICATION.**
