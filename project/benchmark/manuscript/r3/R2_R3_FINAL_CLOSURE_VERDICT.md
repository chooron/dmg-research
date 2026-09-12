# R2–R3 Final Closure Verdict

## Provenance and frozen contract

This closure round used only the audited canonical R2 table and existing R3 relationship matrices. The frozen primary contract was unchanged: attribute clustering threshold `0.70`, IC nontrivial threshold `|rho| >= 0.20`, IC-stability anchor `bootstrap sign probability >= 0.95`, model-equal aggregation, `all36` plus `exclude_simhyd`, and fixed `1000`-replicate stochastic procedures where applicable. No IC/dPL training, simulations, checkpoint changes, OOB/PUB/PUR, equation changes, or post-hoc threshold changes were performed.

## A. R2 closure: does landing exceed parameter-name diversity?

**PARTIALLY**

The source-backed parameter registry covers all `271` canonical model-parameter coordinates:

- high-confidence roles: `202`;
- medium-confidence roles: `36`;
- low-confidence roles: `14`;
- unresolved: `19`;
- low/unresolved combined: `33/271 = 0.121771`.

Among the five all36 information clusters meeting the predeclared recurrence rule (`>=0.50`), `241/284 = 0.848592` stable IC cells have high/medium source-backed role assignments. `194/284 = 0.683099` are additionally in role groups represented by at least five models. Exclude-SIMHYD gives `231/274 = 0.843066` and `187/274 = 0.682482`, respectively.

For the primary C012 cluster, relationships occur in six eligible major roles—evapotranspiration, partitioning/infiltration, recession/baseflow, routing, snow, and storage capacity—with represented-model counts from `5` to `15` and multiple dominant parameter names within roles. This is more than a purely nominal comparison of unrelated parameter labels.

However, the defensible five-model MOPEX family does not show excess dominant-coordinate heterogeneity beyond the within-family label-exchange null for C012: observed heterogeneity=`0.600000`, null mean=`0.516800`, permutation p=`0.587413`. Across all `26` family×cluster tests, no heterogeneity p-value was `<0.05`. The two- and three-model families are descriptive only.

Therefore the stronger statement is **partial same-role landing heterogeneity with limited family-controlled confirmation**. It cannot be upgraded to a universal homologous-coordinate law.

**R2 closure output:** `R2_LANDING_SUPPORTED_BUT_ROLE_METADATA_LIMITED`

## B. R3 construction challenge: same-parameter versus generic dPL gradients

**YES**

The within-model cross-parameter matrix contains `4,790` rows across primary information-cluster and raw-attribute spaces. In the primary information-cluster space:

- same-parameter diagonal correspondence: `0.715789`;
- model-level median off-diagonal correspondence: `-0.021805`;
- model-equal diagonal advantage: `0.615038`;
- model-equal median fraction with diagonal greater than off-diagonal: `1.000000`;
- diagonal top-1/top-3 fractions: `0.625000/0.875000`.

The diagonal/top-rank summaries use all 36/35 models; the off-diagonal and diagonal-advantage contrast has valid denominators `35 of 36` for all36 and `34 of 35` for exclude-SIMHYD because `collie1` has one parameter and therefore no off-diagonal comparison.

The parameter-label permutation null preserves all IC profiles, all dPL profiles, dPL attribute gradients, parameter marginals, and within-model profile geometry, while destroying only same-parameter identity. Its mean is `0.001536`, 95% interval `[-0.123308, 0.112801]`, empirical p=`0.000999`.

Exclude-SIMHYD remains positive: diagonal advantage=`0.610526`, null mean=`-0.000157`, p=`0.000999`. Deterministic LOO diagonal-advantage ranges are `0.610526–0.617293` for all36 and `0.606015–0.615038` for exclude-SIMHYD.

Thus generic shared dPL attribute gradients do not explain the original profile correspondence. The phrase **same-parameter-coordinate association profile persists across estimators** is defensible, provided it remains an association statement.

**R3 identity output:** `R3_PARAMETER_IDENTITY_SUPPORTED`

## C. R3 magnitude retention: magnitude as well as sign?

**STRONG**

The primary IC-stable information-cluster denominator is `902` cells (`849/902 = 0.941242` same-sign). Conditional on IC stability, dPL retains the same sign and has:

- `|rho_dPL| >= 0.10`: `791/902 = 0.876940`;
- `|rho_dPL| >= 0.20`: `692/902 = 0.767184`;
- `|rho_dPL| >= 0.30`: `517/902 = 0.573171`.

Model-equal median rates are `0.873397`, `0.769841`, and `0.548589`. Exclude-SIMHYD remains similar: `0.872390`, `0.758701`, and `0.563805`.

Median absolute relationship strength changes from IC=`0.268044` to dPL=`0.343520`; median delta is `+0.054752`, bootstrap 95% CI `[0.041448, 0.067693]`; median ratio is `1.185404`, CI `[1.141273, 1.231778]`. The typical retained relationship is therefore preserved to strengthened, although delta IQR=`0.209873` shows substantial cell heterogeneity.

All36 transition counts are: same-sign strong=`517`, same-sign moderate=`175`, same-sign near-zero=`58`, same-sign attenuated-small=`99`, and opposite-sign=`53`. LOO p20 rates range `0.758701–0.771889`; p30 rates range `0.563805–0.582949`.

Higher-identifiability/lower-restart-SD cells have p20/p30=`0.787625/0.596990`; lower-identifiability/higher-SD cells have `0.726974/0.526316`. The direct restart-SD versus dPL-magnitude association is only rho=`-0.074642`; this is descriptive and does not establish causal identifiability effects.

This result is `P(dPL retained | IC stable)`. It is not the earlier reverse conditional `P(IC-supported | dPL strong)`, which was approximately `0.32`.

**R3 magnitude output:** `R3_MAGNITUDE_RETENTION_STRONG`

## D. What claims survive?

### Strongest defensible R2 claim

In frozen seen-basin IC estimates, several predeclared, attribute-derived information blocks recur across most model structures above a basin-correspondence null. Their stable relationships show heterogeneous parameter-coordinate expression, and much of the recurrent landing is represented by source-backed functional roles. The family-controlled test supports this only partially because homologous-coordinate heterogeneity is not above exchangeability and role metadata are incomplete.

### Strongest defensible R3 claim

IC independently formed association profiles persist in canonical dPL at the same parameter coordinate more strongly than with other dPL parameter coordinates after preserving generic dPL attribute gradients. Among IC-stable relationships, dPL usually retains the same sign and often retains moderate or strong magnitude; the effect is stable to exclude-SIMHYD and LOO analyses.

## E. Claims that remain weakened or unsupported

The following remain unsupported and must not appear as unqualified conclusions:

- shared causal hydrological mechanisms;
- same dominant controls in the strong universal sense;
- universal physical parameter laws;
- boundary-independent recurrence or correspondence;
- universal process-role equivalence across model families;
- causal identifiability explanations or a fixed causal fraction of disagreement;
- independent validation by all dPL-strong cells;
- geographic transferability from opaque repository grouping files;
- treating dPL attribute-to-parameter construction as independent hydrological validation.

The prior raw top-1 agreement (`0.184502`) and cluster top-1 agreement (`0.298893`) also do not justify universal same-control claims. Same-parameter profile persistence is a narrower cross-estimator identity result, not evidence that every estimator selects the same dominant attribute or mechanism.

## F. Final manuscript gate

**`R2_R3_FROZEN_WITH_MANDATORY_REFRAMING`**

R2/R3 can now be frozen for manuscript writing under the following required language:

> recurring, null-exceeding IC association dimensions with partial same-role and model-dependent parameter-coordinate expression; and IC-independent association profiles with same-parameter cross-estimator persistence and substantial, but heterogeneous, magnitude retention.

The final manuscript must report the role-coverage limitation, family-null result, boundary sensitivity, opaque spatial semantics, identifiability limitation, and dPL construction artifact in the main text or supplement. No additional diagnostic is required by this closure round.

## Closure output map

### R2

- `scripts/09_homologous_landing_closure.py`
- `tables/R2_PARAMETER_ROLE_REGISTRY.csv`
- `tables/R2_ROLE_CONTROLLED_LANDING_SUMMARY.csv`
- `tables/R2_ROLE_CONTROLLED_LANDING_CELLS.csv`
- `tables/R2_ROLE_CONTROLLED_RECURRING_CLUSTERS.csv`
- `tables/R2_FAMILY_HOMOLOGOUS_LANDING_SUMMARY.csv`
- `tables/R2_FAMILY_HOMOLOGOUS_PERMUTATION_NULL.csv`
- `R2_HOMOLOGOUS_LANDING_CLOSURE_AUDIT.md`

### R3 identity

- `scripts/08_parameter_identity_correspondence.py`
- `cache/R3_PARAMETER_CROSS_CORRESPONDENCE/`
- `tables/R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv`
- `tables/R3_PARAMETER_IDENTITY_MODEL_SUMMARY.csv`
- `tables/R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv`
- `tables/R3_PARAMETER_LABEL_PERMUTATION_NULL.csv`
- `tables/R3_PARAMETER_IDENTITY_LOO.csv`
- `figures/R3_diagonal_vs_offdiagonal_correspondence.png`
- `figures/R3_diagonal_vs_offdiagonal_correspondence.pdf`
- `R3_PARAMETER_IDENTITY_CORRESPONDENCE_AUDIT.md`

### R3 magnitude

- `scripts/09_magnitude_retention.py`
- `tables/R3_IC_TO_DPL_MAGNITUDE_RETENTION.csv`
- `tables/R3_IC_TO_DPL_MAGNITUDE_RETENTION_SUMMARY.csv`
- `tables/R3_IC_TO_DPL_MAGNITUDE_RETENTION_MODEL_EQUAL.csv`
- `tables/R3_IC_TO_DPL_RETENTION_MODEL.csv`
- `tables/R3_IC_TO_DPL_STRENGTH_TRANSITIONS.csv`
- `tables/R3_IC_TO_DPL_MAGNITUDE_RETENTION_LOO.csv`
- `tables/R3_IC_TO_DPL_RETENTION_IDENTIFIABILITY.csv`
- `R3_IC_TO_DPL_MAGNITUDE_RETENTION_AUDIT.md`
