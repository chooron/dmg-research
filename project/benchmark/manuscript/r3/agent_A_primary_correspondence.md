# Agent A — R3 Primary Profile Correspondence Audit

## Audit status

**PASS WITH LIMITATION** — authoritative frozen products reproduce the requested primary correspondence and stable-cell retention values. The finding is an association-profile correspondence result, not parameter identity, physical-role preservation, or causality.

No training, recalibration, simulation, new dPL seed, new bootstrap/permutation, source overwrite, script change, or cache restructuring was performed.

## Frozen study universe

| Item | Verified value | Interpretation / source |
|---|---:|---|
| Conceptual models | 36 | all36 frozen population |
| Common basins per model | 531 | common CAMELS-US basin alignment |
| Static descriptors | 35 | raw descriptor sensitivity space |
| Primary information dimensions | 20 | threshold-0.70 information-cluster space |
| Model-parameter coordinates | 271 | total registry coordinates across models |
| Primary parameter-information cells | 5,420 | 271 coordinates × 20 information dimensions |
| dPL realization | seed 42 | canonical dPL only |
| IC realization | basin-wise independent calibration | canonical IC field |

The relationship-matrix source also contains the 35-descriptor sensitivity space: 29,810 data rows correspond to both estimators, both feature spaces, and the complete registered parameter coordinates. The 5,420-cell R3 primary denominator is specifically the 271 × 20 information-cluster block and must not be confused with the larger raw/sensitivity matrix row count.

## Authoritative estimands

For model `m`, parameter coordinate `p`, and frozen information dimension `k`, let `rho_IC[m,p,k]` and `rho_dPL[m,p,k]` be the basin-wise Spearman associations between normalized parameter values and the information dimension. The primary same-parameter profile correspondence is:

```text
R_m,p = Spearman_k(rho_IC[m,p,k], rho_dPL[m,p,k])
R_m   = median_p(R_m,p)
R_paired = median_m(R_m)
```

The primary aggregation is model-equal: each model contributes one median over its own parameter coordinates. It is not a pooled cell-level correlation. The same-parameter result uses the diagonal `(p,p)` pairing; it does not claim that a parameter's physical identity is preserved.

For the IC-stable sign/magnitude audit, eligibility is defined by IC only:

```text
abs(rho_IC) >= 0.20
IC bootstrap sign probability >= 0.95
```

The dPL value is not used to select the denominator.

## Verified primary results

### Same-parameter profile correspondence

- `R_paired = 0.7157894737` (all36, information-cluster, model-equal median).
- This is substantial but incomplete correspondence of the 20-dimensional association profiles.
- The corresponding all36 model-equal diagonal/off-diagonal identity audit reports diagonal median `0.715789`, off-diagonal median `-0.021805`, and diagonal advantage `0.615038`; those are related but distinct summaries.

### IC-stable sign retention

- Stable IC denominator: `902 / 5420` primary information-cluster cells.
- Same-sign retention: `849 / 902 = 0.9412416851` (`94.12%`).
- Same sign plus dPL `abs(rho) >= 0.10`: `791 / 902 = 0.8769401330`.
- Same sign plus dPL `abs(rho) >= 0.20`: `692 / 902 = 0.7671840355`.
- Same sign plus dPL `abs(rho) >= 0.30`: `517 / 902 = 0.5731707317`.

The `94.12%` value is conditional on the 902-cell IC-stable subset. It is **not** sign retention over all 5,420 cells. For context only, the separate raw all-cell sign audit reports `0.746863`; that statistic has a different denominator and is not interchangeable with the IC-stable result.

## Aggregation-level audit

| Evidence | Unit | Correct use |
|---|---|---|
| `Rho_IC`, `Rho_dPL` | model × coordinate × information cell | association inputs |
| `R_m,p` | model × parameter coordinate | 20-dimensional profile correspondence |
| `R_m` | model | median over that model's coordinates |
| `R_paired` | model-equal | primary same-parameter profile result |
| `902/5420` and sign rates | paired information cells | IC-only stable-cell conditional retention |
| raw all-cell sign rate | paired information cells | separate descriptive audit only |

No cell-level rows were treated as independent model replicates, and no pooled statistic was substituted for the model-equal primary estimand.

## Source references and hashes

1. Primary paired relationship matrix: `project/benchmark/manuscript/r3/tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv`  
   SHA256: `8a757c67baaaef7e2124804f061658d0e1c41cf949195d28194aecc997846b68`  
   29,810 data rows plus header; includes information-cluster and raw-descriptor spaces.
2. Diagonal/off-diagonal authoritative summary: `project/benchmark/manuscript/r3/tables/R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv`  
   SHA256: `3bd5dddf52acdef7b33717a69e7d6936b15a05c53af6a19fea78d2e03231a032`.
3. IC-stable sign/magnitude summary: `project/benchmark/manuscript/r3/tables/R3_IC_TO_DPL_MAGNITUDE_RETENTION_SUMMARY.csv`  
   SHA256: `a7e24d2b1f17fecde82bc5efcf6077a4216c51159dca07716eba0a0ed4667aef`.
4. Identity analysis manifest: `project/benchmark/manuscript/r3/cache/R3_PARAMETER_IDENTITY_MANIFEST.json`  
   SHA256: `f5b203417c40b27446bcaa73c1c08c727d8e92c382b5aa33a2a25c5a1dda5d5d`; records all36/exclude-SIMHYD populations, 1,000 label permutations, seed `20260902`, and the two feature spaces.
5. Sign analysis manifest: `project/benchmark/manuscript/r3/cache/R3_SIGN_MANIFEST.json`  
   SHA256: `d6878bbc5a64674de515b3016f6b9d22c3e30556b8f494f1d61beeec26e378cf`.
6. Formal estimand definition: `project/benchmark/manuscript/r3/R3_ESTIMAND_DICTIONARY.md`.
7. Authoritative identity audit: `project/benchmark/manuscript/r3/R3_PARAMETER_IDENTITY_CORRESPONDENCE_AUDIT.md`.
8. Authoritative sign/magnitude audit: `project/benchmark/manuscript/r3/R3_IC_TO_DPL_MAGNITUDE_RETENTION_AUDIT.md`.

## Mismatch log

No mismatch was found between the requested frozen targets and the authoritative source products. The exact source values above are used rather than rounded prompt values.

## Primary finding and limits

**Primary finding:** Same-parameter IC–dPL information profiles show substantial but incomplete correspondence (`R_paired = 0.715789`). IC-stable relationship directions are frequently retained (`849/902 = 94.12%`), with retention rates declining as the required dPL magnitude threshold increases.

**Limits:** The 94.12% estimate is conditional on an IC-defined stable subset. Profile correspondence is an association-pattern result; it does not establish parameter identity, preserved physical meaning, functional-role continuity, causal information transfer, IC truth, or dPL physical superiority. The dPL evidence is canonical seed 42 only, and raw descriptor-space and information-cluster-space statistics remain separate estimands.
