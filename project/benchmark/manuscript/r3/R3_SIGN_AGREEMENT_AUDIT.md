# R3-C Sign Agreement Audit

## Question
Does IC--dPL direction persist after excluding trivial near-zero cells and anchoring eligibility to independently stable IC relationships?

## Data and provenance
Paired relationship matrices `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv`; IC and dPL canonical normalized-u associations on 531 basins; raw and frozen information-cluster spaces.

## Estimand
A_sign_all includes all finite paired cells. A_sign_nontrivial includes cells where either estimator has |rho|>=0.10. A_sign_both_nontrivial requires both. A_sign_IC_stable requires IC |rho|>=0.20 and IC bootstrap sign probability>=0.95, then tests dPL sign; dPL is not used to define the denominator.

## Denominator
Denominator is paired model×parameter×feature cells; all36=36 models and exclude_simhyd=35. Agreement means equal sign, including zero only in A_sign_all. Binomial-style cell bootstrap uses fixed seed and 1000 replicates.

## Method
All four definitions are written for raw and cluster spaces. The historical approximately 74.6% value is the raw all-cell all36 rate under the old finite-cell denominator; the IC-stable/nontrivial rate is the relevant reproducibility evidence and is reported separately.

## Result
Raw all-cell all36 sign agreement was 0.746863; IC-anchored stable/nontrivial agreement was 0.934989. All definitions include denominators, counts, and 1000-resample bootstrap CIs for both raw and information-cluster spaces and both model populations.

## Sensitivity
Rates can change with |rho| threshold and IC stability threshold; those thresholds are fixed and the full denominator is exposed. Cluster agreement is not allowed to overwrite raw agreement.

## Adversarial interpretation
Sign equality is not magnitude agreement, and a dPL sign on an IC-weak cell is construction-linked. A reviewer can still attribute stable sign patterns to correlated attributes or shared data; the dPL construction-artifact and permutation audits remain necessary.

## Verdict
R3_C_READY

## Execution metadata
- runtime_seconds: `0.519`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
