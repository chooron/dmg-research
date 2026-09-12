# R3-H Pooled Influence Audit

## Question
Are R3 reproducibility, sign, and dominant-information conclusions driven by weighting, SIMHYD, one model, boundary cells, or identifiability strata?

## Data and provenance
Primary profiles `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/tables/R3_PARAMETER_PROFILE_REPRODUCIBILITY.csv`, sign `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/tables/R3_SIGN_AGREEMENT_AUDIT.csv`, dominant tables, boundary and identifiability tables. All inputs are frozen IC/dPL post-processing.

## Estimand
Repeat profile reproducibility, sign agreement, and dominant top-k summaries in raw and cluster spaces under model-equal, cell-equal, all36, exclude_simhyd, leave-one-model-out, boundary-filtered, and high/low-identifiability definitions.

## Denominator
Model denominator is 36/35; profile-cell denominator is the actual common parameter count by model; sign/dominant denominators are explicit in source rows. No representative model is selected.

## Method
Every model is removed in turn. Boundary cutoffs .10/.20/.30 and identifiability split at pooled median restart-u SD are fixed before interpretation.

## Result
Primary cluster-space model-equal reproducibility was 0.715789; leave-one-model-out range was 0.712782--0.718797. Influence table includes raw/cluster, model/cell weighting, all36/exclude_simhyd, LOO, boundary cutoffs, sign, dominant, and identifiability subsets.

## Sensitivity
Aggregation sensitivity can alter a headline without changing the underlying cell table. LOO only tests influence, not exchangeability. High/low identifiability subsets are descriptive and can be confounded with model family or parameter role.

## Adversarial interpretation
A robust R3 claim requires the primary cluster profile correspondence and IC-stable sign signal to remain directionally similar across these sensitivity families and to exceed the cross-estimator null. dPL-emergent cells cannot rescue a failed primary result.

## Verdict
R3_H_READY

## Execution metadata
- runtime_seconds: `33.356`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
