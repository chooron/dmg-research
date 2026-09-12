# R3-A Relationship Matrix Audit

## Question
Do IC and dPL produce comparable relationship matrices in the same raw-attribute and information-cluster spaces?

## Data and provenance
Canonical table `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r2/cache/canonical_parameter_attribute_table.parquet`; canonical Caravan attributes and 531 basin IDs; dPL source is canonical v2 seed 42. IC is independently fitted; dPL is an X-to-theta construction and is not independent evidence by itself.

## Estimand
For each estimator × model × parameter × feature, Spearman rho between normalized parameter u and feature, n, p, and basin-bootstrap CI/sign stability. Feature spaces are raw 35 attributes and frozen threshold-0.70 cluster PC1 scores.

## Denominator
Each cell has n=531 finite basins; parameter count follows the current model registry. No missing cells are silently dropped. Raw and cluster spaces are kept as separate estimands.

## Method
Bootstrap tensors are saved per model and estimator for profile, sign, and dominant-control audits. No checkpoint is retrained or modified.

## Result
Built 29810 relationship rows for 36 models, IC and dPL, in raw 35-attribute and primary 0.70 information-cluster spaces. Both estimators have exact point rho/p and 1000-replicate basin-bootstrap CI/sign summaries.

## Sensitivity
Monotone physical mapping does not alter rank rho, but normalized u makes the estimator coordinate explicit. dPL rows remain construction-linked and are used later only after IC anchoring.

## Adversarial interpretation
A reviewer can call any dPL relationship tautological because dPL was trained from attributes. The R3 artifact audit therefore classifies dPL-strong/IC-weak cells as dPL-emergent and excludes them from persistence claims.

## Verdict
R3_A_READY

## Execution metadata
- runtime_seconds: `58.440`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
