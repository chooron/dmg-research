# R3-B Reproducibility Definition Audit

## Question
What exactly is the primary IC--dPL reproducibility estimand, and what do the historical approximately 0.658 and 0.733 numbers mean?

## Data and provenance
Relationship matrices `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv` built from the canonical R2 table. The primary feature space is the data-defined 0.70 information-cluster space; raw attributes are sensitivity.

## Estimand
Primary R_m,p = Spearman across information-cluster relationship vectors for each common model parameter; R_m = median_p R_m,p; R_overall = median_m R_m. Secondary model-flattened, mean, cell-equal, raw-attribute, and exclude_simhyd estimands are retained.

## Denominator
Primary denominator: 36 model medians, each model contributing equally; each model's denominator is its own registry parameter count. Profile feature denominator is 20 primary clusters (varies only if clustering contract changes, which is prohibited here).

## Method
Feature-bootstrap CIs use 1000 resamples to quantify profile-feature sampling variability. No basin or parameter is silently dropped; missing/nonfinite profile cells fail the matrix builder. Historical mapping: formal current report's ~0.6584 is the median of 36 model-level flattened profile correlations; the ~0.733 value corresponds to the model-equal median of per-model parameter-profile medians (~0.7326 in the current formal table), not the cell-equal 0.7527 parameter-cell median. Exact historical values and changed exploratory definitions are retained in `project/benchmark/results/ic_dpl_seenbasin_formal_20260901/EXPLORATORY_ATLAS_COMPARISON.csv`.

## Result
Unfiltered information-cluster model-equal median reproducibility was 0.715789; IC-anchored profile sensitivity was 0.819643; raw-attribute sensitivity was 0.732983. Per-parameter, per-model profiles, feature-bootstrap CIs, IC-anchored counts, model medians, cell-equal summaries, and all36/exclude_simhyd are saved.

## Sensitivity
The primary is frozen before sign, dominant-control, and null results are interpreted. Raw attribute, model-flattened, cell-equal, all36/exclude_simhyd, and feature-bootstrap sensitivity are not substituted for it.

## Adversarial interpretation
Profile correlation is reproducibility of association patterns, not agreement of parameter estimates or causal effects. A reviewer can challenge feature-bootstrap uncertainty, rank ties, common attribute confounding, dPL construction, and unequal parameter counts; those are addressed in separate audits.

## Verdict
R3_B_PRIMARY_ESTIMAND_FROZEN

## Execution metadata
- runtime_seconds: `116.367`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
