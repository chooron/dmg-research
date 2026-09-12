# R3-G dPL Construction-Artifact Audit

## Question
Are strong dPL attribute--parameter relationships independently supported by IC, or are they inevitable consequences of the dPL construction?

## Data and provenance
Paired relationship matrices `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv` for raw and frozen information-cluster spaces; IC is independently fitted and dPL is an X-to-theta network.

## Estimand
Classify dPL-strong cells at |rho_dPL|>=0.20 as IC-supported (IC |rho|>=0.20, same sign), IC-opposite-sign, IC-weak (0.10≤|rho_IC|<0.20), or IC-absent (<0.10). dPL-emergent includes all non-supported strong cells.

## Denominator
Denominator is every paired model×parameter×feature cell for each space; strong-cell summaries use only dPL-strong cells. all36 and exclude_simhyd can be filtered by model; primary classification does not select dPL-emergent evidence.

## Method
The same thresholds and paired cells are used in both spaces. Bootstrap sign stability is retained as context but does not make dPL independent.

## Result
Among 6437 dPL-strong cells across both spaces, 0.333230 were IC-supported and 0.666770 were not independently supported (IC-weak/absent/opposite-sign). Only IC-supported cells are eligible for cross-estimator persistence claims.

## Sensitivity
The dPL mapping can produce strong relationships by construction. A dPL-strong/IC-weak or absent cell cannot support shared hydrological information; an opposite-sign cell is disagreement, not evidence for dPL. Even IC-supported cells remain association-level evidence.

## Adversarial interpretation
A reviewer can argue that IC and dPL share data/forcing and are not independent in a broad statistical sense. The correct claim is persistence of an IC relationship under a different parameter-estimation procedure, not independent replication of a causal law. Only IC-supported cells enter the R3 persistence interpretation; dPL-emergent rows remain an explicit artifact audit and are not promoted to validation evidence.

## Verdict
R3_G_READY

## Execution metadata
- runtime_seconds: `0.868`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
