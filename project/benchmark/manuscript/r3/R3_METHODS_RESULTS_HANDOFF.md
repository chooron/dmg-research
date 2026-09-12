# R3 Methods–Results Handoff

This handoff is deliberately compact and follows `estimand → computation → uncertainty/test → result → allowed interpretation`.

## 1. Same-parameter information-profile correspondence

- **Estimand:** for each model–parameter coordinate, the correspondence between the 20-dimensional IC and dPL parameter–information Spearman association profiles; model-level medians are then aggregated equally across 36 models (`R_paired`).
- **Computation:** form the frozen IC and canonical dPL profile vectors over 20 information dimensions; compare the same coordinate; median over coordinates within model, then median over models.
- **Uncertainty/test:** frozen deterministic model-equal summary; no cell-level independence claim.
- **Result:** `R_paired = 0.7157894737`.
- **Allowed interpretation:** substantial but incomplete association-profile correspondence. Do not call this parameter identity, physical meaning, role preservation, or causality.

## 2. IC-stable sign retention

- **Estimand:** sign retention among paired information cells selected by IC-only stability.
- **Computation:** select `abs(rho_IC)>=0.20` and IC bootstrap sign probability `>=0.95`; do not use dPL values in selection; compare signs in the selected pairs.
- **Uncertainty/test:** conditional descriptive rate; thresholded dPL magnitude sensitivities are reported as separate rates.
- **Result:** `902/5,420` stable cells and `849/902 = 0.9412416851` same-sign retention; rates are 0.8769401330, 0.7671840355, and 0.5731707317 for dPL absolute-rho thresholds 0.10, 0.20, and 0.30.
- **Allowed interpretation:** frequent directional retention conditional on IC-defined stability. Never report 94.1% as the all-cell rate.

## 3. Same-coordinate specificity

- **Estimand:** within each model, diagonal IC–dPL profile correspondence minus the median finite off-diagonal correspondence; aggregate valid model contrasts equally.
- **Computation:** compare `C[p,p]` with `C[p,q]` for `q != p`; one-parameter models are undefined for this contrast and excluded only there.
- **Uncertainty/test:** 1,000 fixed-seed within-model dPL parameter-label permutations; labels are shuffled while profile geometry and marginals remain fixed. SIMHYD exclusion is a frozen sensitivity.
- **Result:** diagonal median `0.7157894737`, off-diagonal median `-0.0218045113`, `A_diag=0.6150375940`, `p=0.0009990010`; SIMHYD-excluded `A_diag=0.6105263158`, `p=0.0009990010`.
- **Allowed interpretation:** preferential same-coordinate alignment against within-model alternatives. Do not infer physical identity, functional role, or causal transfer.

## 4. Functional-role negative controls

- **Estimand:** after removing original same-coordinate pairings, compare model-level same-role and cross-role off-diagonal profile correspondence; separately test the pre-specified HESS-prior flexible-role contrast.
- **Computation:** source-backed role registry; high-confidence roles primary, high+medium sensitivity; role-count-preserving permutations; no role selected after inspecting outcomes.
- **Uncertainty/test:** 10,000 fixed role-label permutations for `A_role`; model-level bootstrap and sign-flip test for HESS-prior contrast.
- **Result:** `A_role=-0.1187969925`, `p=0.8057194281`; same-role median `-0.0278195489`, cross-role median `0.0052631579`; HESS-prior `D_rho=-0.0003301577`, `n=27`, CI `[-0.0066882428,0.0265313077]`, `p=0.8498150185`.
- **Allowed interpretation:** role labels do not add a stable signal beyond coordinate specificity; final placement is COORDINATE SPECIFICITY ONLY. Do not claim roles are absent.

## 5. R2→R3 rank linkage and residual test

- **Estimand:** first, descriptive association between raw rank continuity (`R_rank`/`Q_rank`) and profile correspondence (`R_info`/`Q_info`); second, same-coordinate profile advantage over raw-rank-matched off-diagonal alternatives (`A_info|rank`).
- **Computation:** compare pooled, model-centered, and within-model rank/profile associations. For the primary residual, match each diagonal pair to the three nearest within-model off-diagonals by absolute `Q_rank` distance (`k=3`, no caliper).
- **Uncertainty/test:** model-clustered bootstrap CI and fixed sign-flip test; matched coverage is `270/271`.
- **Result:** `R_rank`–`R_info` is 0.743830 pooled, 0.683142 model-centered, 0.678571 within-model median; `Q_rank`–`Q_info` is 0.863927, 0.860931, 0.859901. `A_info|rank=0.295238`, CI `[0.254386,0.429073]`, `p=0.000400`.
- **Allowed interpretation:** raw rank continuity accompanies part of R3 correspondence, but residual same-coordinate specificity remains. Do not call rank rearrangement noise removal.

## 6. Performance-conditioned supporting association

- **Estimand:** high-minus-low profile divergence and paired correspondence after ordering basins within each model by `abs(DeltaKGE)` and using equal-size strata.
- **Computation:** 177 basins per low/middle/high stratum; calculate model-level contrasts. The secondary `D_theta` bridge is mathematically coupled and not primary.
- **Uncertainty/test:** 1,000 model bootstrap resamples; balanced equal-size random-subset null with 36,000 model draws and one-sided `p=0.000999`.
- **Result:** divergence `+0.028033`, CI `[0.017163,0.043610]`; paired correspondence `-0.141353`, CI `[-0.215038,-0.078947]`.
- **Allowed interpretation:** performance differences are associated with profile divergence in this observational conditioning. Basin composition and mathematical coupling remain; no causal direction is allowed.

## Global boundary

dPL is canonical seed 42 only, so symmetric within-paradigm dPL variability is unavailable. IC-self and within-paradigm comparisons are limited and do not make IC truth. All results are statistical association/profile comparisons, not causal decompositions; no physical correctness, role preservation, identity recovery, or dPL superiority claim is permitted.
