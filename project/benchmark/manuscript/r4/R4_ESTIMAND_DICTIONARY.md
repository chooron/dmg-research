# R4 Estimand Dictionary

- **Question:** whether R2–R3 seen-basin structure-dependent catchment–parameter relationships persist when parameters are generated for held-out basins.
- **Primary models:** alpine2, hbv96, xinanjiang, newzealand2, ihacres, us1, mopex4, hillslope.
- **Relationship metric:** basin-wise Spearman rho with average ranks, inherited from the final seen-basin atlas; no Pearson substitution.
- **Attribute source:** canonical Caravan 35-dimensional matrix, 531 canonical basin IDs. Continuous attributes are primary; categorical-code attributes remain descriptive and are excluded from the primary atlas.
- **Parameter coordinate:** normalized `u` / sigmoid coordinate for IC, seen dPL, and OOB dPL. Physical parameter values are retained in master tables; monotone mapping preserves finite within-cell ranks.
- **Seen sources:** final formal IC–dPL seen-basin atlas (`ic_dpl_seenbasin_formal_20260901`) and its paired parameter table.
- **OOB source:** exact `best.pt` restored from each completed OOB model/fold, evaluated on its held-out basins only; no OOB coefficient was used before case freeze.
- **Classes:** existing descriptive thresholds high `|rho| >= 0.20`, low `|rho| < 0.10`, sign-changing when both sides are nontrivial and signs differ, persistent only when both sides are high, same-sign, and top-rank ≤ 10.
- **Boundary QC:** near-constant SD ≤ 0.01; boundary-concentrated means ≥50% at normalized ≤0.01 or ≥0.99; sensitivity flags these rows rather than silently deleting them from the population.
- **Tie rule:** average ranks for Spearman; existing atlas top-rank rule uses minimum rank for tied absolute-rho values. Effective/tie counts are reported in the boundary/tie sensitivity table.
- **Information clusters:** secondary descriptive graph from the follow-up atlas, continuous attributes connected when absolute Spearman ≥0.70; primary cluster retention uses the 0.70 graph and reports raw-proxy substitution separately.
- **Bootstrap:** basin-level resampling, 5,000 replicates, fixed seed 20260902, percentile 95% CI; generated in CUDA batches when available and only final statistics retained.
- **Aggregation:** pooled OOF rho is computed after concatenating the five held-out folds (each basin appears exactly once per model); fold-wise rho is reported separately; no complex fold significance test.
- **IC/KGE:** `KGE_IC` and `KGE_dPL_seen` are imported from canonical seen-basin paired KGE evidence. IC is not rerun.
- **Interpretation boundary:** descriptive retention/attenuation/reversal only; no physical truth, causality, universal transfer, or model-suitability conclusion.

**Boundary clarification:** near-constant and boundary-concentrated status excludes a row only from frozen representative-case eligibility. The primary population analysis retains all eligible PRIMARY8 relationship cells and carries boundary/tie flags; it does not silently delete rows because their OOB result is weak or unfavorable. The explicit boundary/tie sensitivity table reports the effect of optional QC exclusions.
