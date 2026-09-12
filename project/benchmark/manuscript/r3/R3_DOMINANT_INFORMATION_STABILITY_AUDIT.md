# R3-D Dominant Information Stability Audit

## Question
How much raw dominant-attribute disagreement is genuine versus substitution among correlated proxies, and is each estimator's dominant choice itself stable?

## Data and provenance
Relationship matrix caches `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/cache/ic_relationship_matrices` and `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/cache/dpl_relationship_matrices`; cluster assignments `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r2/tables/R2_INFORMATION_CLUSTERS.csv` at threshold 0.70.

## Estimand
For each model × common parameter, compare IC and dPL top-1 exact choice, Top-3/Top-5 overlap and Jaccard in raw and cluster spaces. Separately estimate IC and dPL bootstrap top-1 stability relative to each point top-1.

## Denominator
Denominator is 271 model-parameter pairs for all36; raw feature count=35, cluster feature count is frozen from threshold 0.70. Proxy substitution is only a raw top-1 mismatch whose two descriptors share a precomputed cluster.

## Method
Stable top-k calculations use deterministic descending absolute-rho order with feature-index tie break. Bootstrap top-1 selection is calculated from the saved 1000-replicate relationship tensors for IC and dPL independently.

## Result
Raw top-1 exact agreement was 0.184502; information-cluster top-1 agreement was 0.298893. 0.107011 of all raw model-parameter pairs were within-cluster proxy substitutions, representing 0.131222 of raw top-1 mismatches. Bootstrap IC/dPL top-1 stability is reported per pair.

## Sensitivity
Raw top-1 can be low even when Top-3 or cluster agreement is high. Conversely, low estimator-specific bootstrap stability means a mismatch is not interpretable as estimator disagreement. Cluster-level agreement is a broader information estimand, not proof of identical parameter expression.

## Adversarial interpretation
A reviewer can argue that clusters were selected from the same attribute matrix and may overstate coherence; they were fixed before parameter analysis, and the raw/cluster/proxy tables remain separate. dPL-emergent cells are not treated as independent hydrological validation.

## Verdict
R3_D_READY

## Execution metadata
- runtime_seconds: `16.639`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
