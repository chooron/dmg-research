# R3-E Cross-Estimator Null Audit

## Question
Is IC--dPL relationship-profile correspondence higher than expected after destroying basin-level parameter correspondence while preserving estimator marginals?

## Data and provenance
IC information-cluster profiles `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv`; dPL normalized parameter rows from the canonical table; frozen cluster threshold 0.70.

## Estimand
The unfiltered primary is median over models of median over parameters of Spearman(IC profile,dPL profile). An IC-anchored sensitivity uses only information-cluster features with IC |rho|>=0.20 and IC bootstrap sign probability>=0.95; dPL does not select its denominator. Both are applied identically to observed and permuted data.

## Denominator
Basin denominator is 531 within each model; model denominator is 36 or 35. dPL rows are jointly permuted within model, preserving parameter marginals and eligible parameter counts; clustering and thresholds are not recomputed.

## Method
Permutation results are cached and report unfiltered/IC-anchored overall and maximum-model estimands, null intervals, empirical p, and effects above null for all36 and exclude_simhyd.

## Result
The primary unfiltered null used 1000 joint dPL basin-label permutations: observed=0.715789, null mean=-0.001455, 95% interval=[-0.086109,0.082716], p=0.000999. The IC-anchored sensitivity was observed=0.819643, null mean=-0.007601, p=0.000999. Max-model and exclude_simhyd rows are also saved.

## Sensitivity
The null does not test every possible estimator dependence and uses fixed-rank Spearman permutation computation. It is a correspondence null, not a causal null. The anchored sensitivity is the more defensible persistence check when dPL construction is the principal adversarial concern.

## Adversarial interpretation
Because dPL is constructed from attributes, an above-null unfiltered result is not independent hydrological validation. Only IC-anchored/supportive cells can carry the narrower persistence interpretation.

## Verdict
R3_E_READY_WITH_ANCHORED_SENSITIVITY

## Execution metadata
- runtime_seconds: `1.023`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
