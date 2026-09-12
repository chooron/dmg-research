# R3 Estimand Dictionary

## Primary

For every model `m` and common parameter `p`, `R_m,p = Spearman(rho_IC[m,p,k], rho_dPL[m,p,k])` over the frozen information clusters `k` at attribute-clustering threshold 0.70. Then `R_m = median_p R_m,p` and `R_overall = median_m R_m`. Models are equal; parameters are equal only within model through the within-model median.

## IC-anchored sensitivity

For each model-parameter profile, retain only information-cluster features with IC `|rho| >= 0.20` and IC bootstrap sign probability `>= 0.95`; calculate the same IC-versus-dPL profile correlation when at least three such features remain. This denominator is selected by IC only and is the narrower persistence check used in the dPL construction-artifact audit.

## Secondary

- Raw-attribute parameter-profile reproducibility: same operation over 35 raw attributes.
- Model-flattened profile: per-model correlation after flattening all parameter × feature cells, then median over models.
- Cell-equal profile: median of all model × parameter profile correlations.
- `all36` retains accepted SIMHYD generation 280; `exclude_simhyd` is sensitivity.

## Historical mapping

- Approximately `0.658`: legacy median of per-model flattened profile correlations (formal label `model_flattened_profile_spearman_median`).
- Approximately `0.733`: legacy/model-equal median of per-model parameter-profile medians (current formal table gives approximately `0.7326` under that aggregation).
- Approximately `0.7527`: cell-equal median of per-parameter profile correlations in the current formal 36-model table.

These numbers are not interchangeable because their observation units and aggregation differ.
