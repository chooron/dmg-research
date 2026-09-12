# Agent C — Functional-role falsification and negative controls

## Disposition

**PASS WITH LIMITATION.** This is a read-only audit of the frozen functional-role products. No source product, script, cache, checkpoint, canonical result, training run, recalibration, simulation, role taxonomy, or new statistical test was modified or created. The audit supports the frozen R3 placement **COORDINATE SPECIFICITY ONLY**.

## Authoritative products and provenance

The authoritative run is `project/benchmark/results/joh_functional_role_diagnostic_20260905/`. The following products were read:

| Product | Use | SHA256 |
|---|---|---|
| `FUNCTIONAL_ROLE_VERDICT.md` | frozen verdict, estimand and claim boundary | `1637d4f8073920896d9bf35089af6343d44bdbf7a3a76f823642340b7b8cefe5` |
| `HESS_EXTENSION_INTERPRETATION.md` | pre-specified HESS bridge and non-causal limitation | `435a46c45ab9e27d6cb4de0bb251657f63dd317e7cadcd53b799d82583eab3db` |
| `40_PRIMARY_TESTS.json` | machine-readable primary decision and parameter counts | `5939f43150b7f3303d737b44c042ab0bbe141819f2251f70f508f3e7887875e7` |
| `tables/00_PARAMETER_FUNCTIONAL_ROLE_REGISTRY.csv` | source-backed role registry | `3cbcc28e27d595121afadc4f94783d4d6e36ae8b44b3b02f1f5a55c10ba11cbf` |
| `tables/13_ROLE_OFFDIAGONAL_ADVANTAGE_BY_MODEL.csv` | model-level same-role/cross-role off-diagonal summaries | `fe8378b3f54e35e359f356faead7279152fbc80b74a8c8c75aea8a3d3b140768` |
| `tables/14_CORRESPONDENCE_HIERARCHY.csv` | hierarchical same-coordinate, same-role and cross-role summaries | `148f04cfd5662645dfce2dbd4c40457ccf4564272afdbbd44020ecaf3a8018b6` |
| `tables/15_ROLE_LABEL_PERMUTATION.csv` | role-count-preserving permutation null | `9f6352f1447cf7081d292ed959b8bcad504d7efb151adff40045a42286983e2b` |
| `tables/22_HESS_PRIOR_ROLE_TEST.csv` | HESS-prior paired role comparison | `9c68316fb9761d860641952f32321ba9ce356585d69971e38903b53e8e77f4d4` |
| `tables/41_BOOTSTRAP_SUMMARY.csv` | frozen model-level bootstrap intervals | `3fd6c65a06069872e0e59d3cdb9bb57c6f5cf37572fc8bf09d1cbc34786cd338` |
| `tables/42_PERMUTATION_SUMMARY.csv` | frozen role-label null draws | `d0d295ff71a96d1755b3617677cd97a41965ed7d16524b8a68bab2e62e9778d4` |
| `RUN_MANIFEST.json` | run inputs, seeds and provenance | `d108a23fb6ad95828c96d99fc86243c9d9b46ac9777b2782855e43b371890fd4` |

The frozen implementation source is `scripts/functional_role_diagnostic.py` (SHA256 `cd6cbc401671a50a5f58fccc3cd06ff385d83b647a1f0f975421885b1678464c`).

## Study universe and aggregation

The diagnostic covers 36 conceptual models, 531 common basins per model, 20 frozen catchment-information dimensions, and the model-parameter profile products used for the R3 correspondence baseline. The functional registry was source-backed and frozen before outcome calculations. High-confidence roles are primary; high+medium is sensitivity; low and unresolved roles are not forced into a functional interpretation.

The model is the cross-model inferential unit. Parameter pairs, role-pair rows, and association cells are descriptive within-model observations. The diagonal is excluded from the off-diagonal role comparisons. Same-role and cross-role summaries are therefore not treated as independent coordinate replicates, and no naive coordinate-level significance test is used.

The pre-specified HESS-prior flexible role set is:

- `soil_storage_capacity`;
- `runoff_generation_partitioning`;
- `percolation_interflow`;
- `groundwater_baseflow_recession`.

The HESS-prior comparison is against non-snow, non-unresolved roles with matched model-level summaries. It is an observational bridge motivated by the controlled HESS study, not a test of a common causal mechanism.

## Findings

### 1. Coordinate baseline

The historical same-coordinate profile correspondence is `0.7157894737`, and the diagonal advantage is `0.6150375940` (35 valid contrast models). This reproduces the frozen coordinate-specific baseline. It supports parameter-coordinate continuity only; it does not establish physical parameter identity or preserved physical meaning.

### 2. Functional-role negative control

The high-confidence model-equal role-label statistic is:

- `A_role = -0.1187969925`;
- role-count-preserving permutation p-value `= 0.8057194281`;
- 10,000 fixed role-label permutations with role counts preserved;
- positive high-confidence model statistics: `12/36`.

After excluding original same-coordinate pairings, the valid high-confidence model-level summaries in `CORRESPONDENCE_HIERARCHY.csv` give:

- same-role off-diagonal median: `-0.0278195489` (`n=29` valid models);
- cross-role median: `0.0052631579` (`n=29` valid models).

Thus same-role off-diagonal correspondence is not higher than cross-role correspondence. The negative role statistic and the role-count-preserving null do not support broader functional-role continuity beyond the coordinate effect.

### 3. HESS-prior flexible-role negative control

The pre-specified HESS-prior role contrast is:

- prior-minus-comparison `D_rho = -0.0003301577`;
- comparable models `n=27`;
- bootstrap 95% CI: `[-0.0066882428, 0.0265313077]`;
- model sign-flip p-value: `0.8498150185`.

The interval includes zero, the observed contrast is effectively null and slightly negative, and the sign-flip test provides no evidence that the pre-specified flexible roles are more paradigm-sensitive than the comparison roles.

## Interpretation and claim boundary

The functional-role result is a **negative extension**, not evidence that roles are absent from hydrology. It says that, within this frozen observational calibration-paradigm comparison and after removing the original coordinate pairing, source-backed role labels do not provide an additional stable correspondence signal. This limits R3 to **COORDINATE SPECIFICITY ONLY**.

The results do not support any of the following statements:

- physical meaning is preserved;
- functional roles are preserved;
- parameter identity is recovered;
- dPL learns true hydrological functions;
- the HESS compensation mechanism is reproduced across the 36-model ensemble;
- an omitted process causally produces the observed parameter change;
- IC is truth or dPL is physically superior.

The role registry remains partly judgment-dependent, role coverage differs by model, and not every model supplies valid same-role and cross-role pairs. These are limitations of the negative control, not reasons to redefine the role taxonomy or select a more favorable comparison.

## Final claim

The frozen evidence supports same-coordinate information-profile specificity, but not a broader same-role or cross-role functional organization claim. The appropriate R3 interpretation is therefore:

> **Information-profile correspondence is coordinate-specific only; broad functional-role continuity is not supported.**
