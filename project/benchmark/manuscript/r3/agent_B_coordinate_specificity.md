# Agent B — R3 same-coordinate specificity audit

## Scope and claim classification

**Claim classification: `SUPPORTED WITH QUALIFIER`**

The frozen R3 evidence supports same-coordinate information-profile specificity: the information-profile correspondence for the same model parameter is higher than the correspondence for alternative parameters within the same model. This is a descriptive correspondence at the frozen association-profile level. It does **not** establish parameter identity, preserved physical meaning, functional-role preservation, causality, or a physical law.

No source product, manuscript script, or manuscript cache was modified. No training, recalibration, simulation, new dPL seed, or new random test was run.

## Authoritative source products

The authoritative sources are:

| purpose | path | SHA256 |
|---|---|---|
| diagonal/off-diagonal aggregation and observed `A_diag` | `project/benchmark/manuscript/r3/tables/R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv` | `3bd5dddf52acdef7b33717a69e7d6936b15a05c53af6a19fea78d2e03231a032` |
| frozen parameter-label permutation null | `project/benchmark/manuscript/r3/tables/R3_PARAMETER_LABEL_PERMUTATION_NULL.csv` | `916b37b45ce61b65090bd2fc2ededa46eedbf1993ef20cd1937ea4eb706acffb` |
| authoritative R3 identity audit | `project/benchmark/manuscript/r3/R3_PARAMETER_IDENTITY_CORRESPONDENCE_AUDIT.md` | `e7e69ef442bda5039745e6a48c933ccf3ff9924948497f3e0d0232225f93869a` |
| metric and linkage contract | `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_parameter_axis_audit_20260906/agent_D_r2_r3_rank_linkage/r3_metric_contract.md` | `961bf75d0647b15a27d21e8c0da2ade82b2d1fc594dc4bfeef77b462e7f89a17` |

The prompt values agree with authoritative values after rounding: `0.615038` is `0.6150375940`, `0.611` is `0.6105263158`, and `0.000999` is `0.0009990010`. No substantive prompt/source mismatch was found.

## Frozen estimator and comparator

For each model, the frozen IC and dPL information-profile matrices contain one 20-dimensional profile per parameter. Each profile is the vector of Spearman associations between that parameter and the 20 frozen information-cluster dimensions. The frozen profile-correspondence matrix is `C[p,q]`, comparing the IC profile for parameter `p` with the dPL profile for parameter `q`.

- **Same-coordinate comparator:** diagonal values `C[p,p]`.
- **Alternative comparator:** all finite off-diagonal values `C[p,q]` for `q != p` within the same model.
- **Per-model contrast:** finite median of diagonal values minus finite median of off-diagonal values.
- **Model-equal primary aggregation:** median across valid model-level contrasts; models with no defined off-diagonal contrast are excluded from this contrast only. The one-parameter `collie1` model is therefore not a valid off-diagonal contributor.

This is not a pooled cell-level test and does not treat the 4,790 matrix entries as independent replicates.

## Frozen results

### All 36 models

- diagonal model-equal median: `0.7157894737`
- off-diagonal model-equal median: `-0.0218045113`
- diagonal advantage `A_diag`: `0.6150375940`
- valid model contributors for the advantage: `35/36`
- fraction of valid model summaries with diagonal greater than off-diagonal: `1.000000`
- diagonal top-1 / top-3 fractions: `0.625000 / 0.875000`

### SIMHYD exclusion sensitivity

- diagonal advantage: `0.6105263158`
- valid model contributors: `34/35`
- permutation null mean: `-0.0001571429`
- empirical permutation p-value: `0.0009990010`

The exclusion sensitivity retains the same direction and magnitude of the specificity result.

## Permutation null

The frozen null is a **within-model dPL parameter-label permutation**, not a basin-label permutation. For each model, dPL parameter-profile labels are permuted while retaining:

- all IC profiles;
- all dPL profiles;
- dPL attribute gradients;
- parameter marginals;
- within-model profile geometry.

Only the IC–dPL same-parameter label identity is destroyed. The null recomputes the diagonal/off-diagonal median contrast after each permutation, and model-equal summaries take the median across model draws at each replicate. There are `1,000` fixed-seed permutations. The all-36 table uses seed `20260902`; the exclude-SIMHYD table uses seed `20310902`. The empirical upper-tail p-value is `(1 + number of null draws >= observed) / (1 + 1000)`.

For the primary information-cluster space:

- observed `A_diag`: `0.6150375940`
- null mean: `0.0015360902`
- null 2.5–97.5% interval: `[-0.1233082707, 0.1128007519]`
- empirical p-value: `0.0009990010`

The observed advantage is far above the identity-destroying null. This supports preferential same-coordinate alignment rather than generic profile similarity alone.

## Interpretation boundary

The result supports the narrow statement that frozen IC–dPL information-profile correspondence is preferentially aligned with the same parameter coordinate. It does not show that:

- the parameter has the same physical meaning under IC and dPL;
- a physical or conceptual parameter identity is preserved;
- a functional role is preserved across estimators;
- dPL recovers an IC truth;
- dPL is more physical or correct;
- any causal mechanism produces the observed correspondence.

The result must remain distinct from `R_paired`, which measures same-parameter profile correspondence, and from broader functional-role continuity. The latter is a separate negative-control extension and is not rescued by this coordinate-level result.

## Limitations and audit status

- dPL has only canonical seed 42; symmetric dPL within-paradigm variability is unavailable.
- One-parameter models have no off-diagonal contrast and are excluded only where that contrast is undefined.
- `A_diag` is model-equal descriptive aggregation; no row-wise significance is claimed.
- The label permutation preserves generic dPL profile geometry and tests same-coordinate identity alignment, but cannot establish physical semantics.
- The frozen result is read-only and reproducible from the checksum-verified tables above.

**Agent B status: PASS WITH QUALIFIER.**
