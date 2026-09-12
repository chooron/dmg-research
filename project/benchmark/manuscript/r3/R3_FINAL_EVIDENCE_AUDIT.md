# R3 Final Evidence Audit

**Scope:** JoH 36-model IC–dPL, R3 only. This is an evidence/provenance audit of the current working tree; no training, recalibration, new dPL seed, new experiment, or new scientific story was added.

**Actual output path:** `project/benchmark/manuscript/r3/R3_FINAL_EVIDENCE_AUDIT.md`

## 1. Executive conclusion

R3 is ready for formal F5–F7 figure construction and manuscript drafting **with a narrow interpretation boundary**.

- There is **no remaining numerical, denominator, aggregation, or permutation blocker** for the frozen R3 headline results.
- `R_paired = 0.7157894737` is a **profile-level** estimand: 271 model–parameter 20-D profiles, aggregated as median over parameters within model and then median over 36 equally weighted models.
- `849/902 = 94.124%`, `791/902 = 87.694%`, `692/902 = 76.718%`, and `517/902 = 57.317%` are **cell-level conditional retention** estimates. They are not profile-retention percentages.
- `A_diag = 0.6150375940` is a model-equal **same-coordinate versus off-diagonal correspondence contrast**, valid in 35 of 36 models because `collie1` has one parameter and no off-diagonal contrast. Its label-permutation null is centered at `0.0015360902`, with `p = 0.0009990010`.
- The functional-role extension is negative: `A_role = -0.1187969925`, `p = 0.8057194281`; HESS-prior `D_rho = -0.0003301577`, `p = 0.8498150185`. R3 must therefore remain **coordinate-specific only**.
- `A_diag^IC-self` is **NO-GO for the main text**. Existing restart assets support a one-sided IC-self **RMS parameter-distance** reference, but no existing artifact computes an analogous IC-self `C_{m,p,q}`/`A_diag` profile contrast. The existing RMS benchmark is SI/supporting evidence only and must not be relabeled as `A_diag^IC-self`.
- Recommended figure path: **F5 → F6 → F7**, with F7 as the interpretation-boundary branch using the functional-role negative control and existing identifiability/boundary support. No new calculation is required beyond lightweight figure-data extraction and the symmetric four-way count closure recorded below.

A stale documentation item is recorded rather than silently used: `R3_ESTIMAND_DICTIONARY.md` maps approximately `0.733` to an older/alternative aggregation, while the checksum-verified freeze table and continuity reference identify the canonical information-cluster headline as `0.7157894737`. The `.733` value must not replace the canonical headline.

## 2. Canonical repository and artifact inventory

### 2.1 Repository state

| Item | Current value |
|---|---|
| Repository root | `/home/jingxin/code/dmg-research` |
| Branch | `master` |
| HEAD | `3caca37a4243ae0a95ebe9cc4f22998672ddf464` |
| Working tree | Not clean; broad pre-existing modified and untracked files |
| R3 manuscript directory | `project/benchmark/manuscript/r3/` |
| R3 output directory | `project/benchmark/manuscript/r3/tables/` and `cache/` |

Relevant status observations:

- `project/benchmark/manuscript/r2/`, `r3/`, and `r4/` are untracked directory trees in this working tree; their artifacts are not represented by the current HEAD commit.
- `project/benchmark/src/model_registry.py` is modified and can affect model parameter-count metadata. The frozen R3 artifacts nevertheless carry explicit model/parameter counts and source checksums.
- Benchmark diagnostic scripts are also modified or untracked. No reset, checkout, cleanup, or overwrite was performed.
- The R3 checksum file verifies `22/22` listed files successfully. This gives artifact-level provenance, but not a clean Git-commit provenance.
- The functional-role run manifest names `project/benchmark/manuscript/r2/tables/R2_PARAMETER_ROLE_REGISTRY.csv`, which is absent in the current tree. The generated source-backed role registry and frozen role outputs are present; clean rerun of that script from the current tree is therefore not guaranteed. This is a rerun/provenance caveat, not a numerical blocker for the frozen role result.

### 2.2 R3 canonical scripts

| Purpose | Current candidate script |
|---|---|
| Build IC/dPL relationship matrices | `project/benchmark/manuscript/r3/scripts/00_build_relationship_matrices.py` |
| Profile-level `R_paired` | `project/benchmark/manuscript/r3/scripts/01_parameter_profile_reproducibility.py` |
| Sign/stable-cell audit | `project/benchmark/manuscript/r3/scripts/02_sign_agreement_audit.py` and `09_magnitude_retention.py` |
| Parameter-label correspondence | `project/benchmark/manuscript/r3/scripts/08_parameter_identity_correspondence.py` |
| Shared path/aggregation helpers | `project/benchmark/manuscript/r3/scripts/r3_common.py` |
| Existing identifiability support | `project/benchmark/manuscript/r3/scripts/05_identifiability_reproducibility.py` |

The current R3 source manifest marks some earlier exploratory scripts, including `04_cross_estimator_permutation.py`, `05_identifiability_reproducibility.py`, and `09_magnitude_retention.py`, as superseded paths. The frozen tables, manifests, numerical freeze table, and JoH continuity reference are treated as the authoritative evidence products; the superseded-path flag is retained as a documentation/provenance warning.

### 2.3 Canonical result artifacts

| Evidence | Canonical artifact |
|---|---|
| Full IC/dPL relationship cells | `project/benchmark/manuscript/r3/tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv` |
| Profile values and model summaries | `R3_PARAMETER_PROFILE_REPRODUCIBILITY.csv`, `R3_MODEL_REPRODUCIBILITY.csv`, `R3_OVERALL_REPRODUCIBILITY.csv` |
| Stable-cell retention | `R3_IC_TO_DPL_MAGNITUDE_RETENTION.csv`, `R3_IC_TO_DPL_MAGNITUDE_RETENTION_SUMMARY.csv`, `R3_SIGN_AGREEMENT_AUDIT.csv` |
| `C_{m,p,q}` and `A_diag` | `R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv`, `R3_PARAMETER_IDENTITY_MODEL_SUMMARY.csv`, `R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv` |
| Parameter-label null | `R3_PARAMETER_LABEL_PERMUTATION_NULL.csv` and `cache/R3_PARAMETER_IDENTITY_MANIFEST.json` |
| Numerical freeze | `R3_NUMERIC_FREEZE_TABLE.csv` |
| Cross-product continuity check | `project/benchmark/results/joh_reorganization_diagnostic_20260905/tables/legacy_continuity_reference.csv` |
| Symmetric four-way | `project/benchmark/results/joh_reorganization_diagnostic_20260905/tables/r2_threshold_fourway.csv` and `scripts/joh_reorganization_diagnostic.py` |
| Functional roles | `project/benchmark/results/joh_functional_role_diagnostic_20260905/` |
| Information-dimension registry | `project/benchmark/results/joh_reorganization_diagnostic_20260905/tables/information_dimension_registry.csv` |

The relationship-table SHA256 recorded by the freeze chain is `8a757c67baaaef7e2124804f061658d0e1c41cf949195d28194aecc997846b68`. The diagonal summary, permutation table, retention summary, and role/HESS products are also recorded with SHA256 values in `R3_SOURCE_MANIFEST.json` and the agent audit documents.

### 2.4 Restart assets

| Asset | Path | Audit status |
|---|---|---|
| Complete restart availability gate | `project/benchmark/results/seenbasin_remaining_analysis_20260901/agent_C/C05_RESTART_DATA_AVAILABILITY_GATE.csv` | 36 models × 531 basins × 10 starts; complete latent and fitness archives |
| Restart uncertainty summary | `.../C05_IC_RESTART_PARAMETER_UNCERTAINTY.csv` | Aggregate restart spread, 271 parameter coordinates |
| Existing IC-self RMS benchmark | `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/claim_audit_multiaudit_20260905/agent_B/` | Valid one-sided RMS reference; not an `A_diag` analogue |
| IC-self implementation | `.../agent_B/ic_self_benchmark.py` | Frozen rule: archived IC training KGE within 0.01 of best restart; no dPL multi-seed product |

### 2.5 Functional-role registry

The frozen generated registry is:

`project/benchmark/results/joh_functional_role_diagnostic_20260905/tables/00_PARAMETER_FUNCTIONAL_ROLE_REGISTRY.csv`

The implementation is:

`project/benchmark/results/joh_functional_role_diagnostic_20260905/scripts/functional_role_diagnostic.py`

The taxonomy is source-backed and frozen before outcome calculation. High-confidence roles are primary; high+medium is sensitivity; low/unresolved roles are not forced into the primary role contrast.

## 3. Estimand dictionary

| ID | Estimand | Unit | Population | Eligibility | Denominator | Within-unit calculation | Aggregation hierarchy | Uncertainty/null | Canonical artifact | Generating script |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | Full association space | model × parameter × information dimension | all36 | finite complete IC/dPL rows | `36 × 271 × 20 = 5,420` cells; 531 basins per cell correlation | basin-wise Spearman `rho` between normalized parameter `u` and information dimension | pooled cell space for descriptive counts; model/parameter summaries retained separately | basin bootstrap/sign probability on relationship table | `R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv` | `00_build_relationship_matrices.py` |
| E2 | `R_paired` | model × parameter profile | all36 | all complete 20-D information profiles; all 271 eligible | 271 profiles; 36 model medians | Spearman across the 20 paired `rho_IC` and `rho_dPL` values | median over parameters within each model, then median over 36 models; equal model weight | 1,000 feature-bootstrap CIs per profile; no single aggregate CI | `R3_OVERALL_REPRODUCIBILITY.csv`, `R3_PARAMETER_PROFILE_REPRODUCIBILITY.csv` | `01_parameter_profile_reproducibility.py` |
| E3 | IC-stable subset | paired association cell | all36 | `abs(rho_IC) >= 0.20` and IC bootstrap sign probability `>= 0.95`; dPL never selects eligibility | 902 of 5,420 cells; 36 models; 219 parameter coordinates have ≥1 retained cell; 19/20 information dimensions appear | paired comparison of `rho_IC` and `rho_dPL` for retained cells | pooled-cell rates and separate model-equal medians | 1,000 paired-cell bootstrap for magnitude-threshold rates | `R3_IC_TO_DPL_MAGNITUDE_RETENTION.csv` and summary | `09_magnitude_retention.py` |
| E4 | Retention ladder | retained paired cell | all36 IC-stable subset | E3 eligibility plus same sign and dPL magnitude threshold | 902 for every ladder numerator | same sign; then same sign and `abs(rho_dPL) >= 0.10/.20/.30` | pooled cell numerator/902; model-equal medians are secondary | 1,000 paired-cell bootstrap for threshold rates | `R3_IC_TO_DPL_MAGNITUDE_RETENTION_SUMMARY.csv` | `09_magnitude_retention.py` |
| E5 | Matched-threshold four-way | full association cell | all36 | symmetric `abs(rho_IC) >= t` and `abs(rho_dPL) >= t` | 5,420 cells at each `t` | `both`, `IC-only`, `dPL-only`, `neither` | pooled cell fractions; model-equal medians are secondary | no permutation; descriptive symmetric ladder | `r2_threshold_fourway.csv` | `joh_reorganization_diagnostic.py` |
| E6 | `C_{m,p,q}` | within-model IC parameter `p` × dPL parameter `q` | all36 | complete 20-D profiles; same model and same coordinate registry | `P_m × P_m` per model; 2,395 primary information-space matrix entries (4,790 including raw-attribute sensitivity) | Spearman between IC profile `p` and dPL profile `q` across 20 information dimensions; average rank ties | retain full matrix; diagonal/off-diagonal summaries are separate | label permutation only for E7/E8 | `R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv` | `08_parameter_identity_correspondence.py` |
| E7 | `A_diag` | model-level same-coordinate contrast | all36 | `P_m > 1` for off-diagonal contrast | 35 valid models of 36; `collie1` is single-parameter | `A_m = median_p[C_{m,p,p} - median_{q != p} C_{m,p,q}]`; `A_diag = median_m A_m` | parameter-row contrast → model median → median over valid models | label-permutation null; exclusion sensitivity | `R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv` | `08_parameter_identity_correspondence.py` |
| E8 | Parameter-label permutation null | permuted model-level contrast | all36 and exclude-SIMHYD | within-model permutation of dPL parameter labels/profiles only | 1,000 permutations; 35/34 valid contrast models | recompute E7 after shuffling dPL profile labels; all profiles remain intact | median across models at each draw | upper-tail empirical p `(1 + #null >= observed)/(1000+1)` | `R3_PARAMETER_LABEL_PERMUTATION_NULL.csv` | `08_parameter_identity_correspondence.py` |
| E9 | 35-model `A_m` distribution | model | eligible contrast models | exclude single-parameter `collie1` only for A_m | 35 models | one `A_m` per model | distribution summary; retain model-by-model table | deterministic summary; LOO table available | `r3_diagonal_advantage.csv` / `R3_PARAMETER_IDENTITY_MODEL_SUMMARY.csv` | `08_parameter_identity_correspondence.py` |
| E10 | Diagonal rank/top-k | IC parameter row | all36 or 35-model contrast subset | finite `C` row; ties share strict-greater rank | 271 rows all36; 270 after excluding `collie1` | descending rank `1 + count(C_{p,q} > C_{p,p})`; normalized percentile `(P-rank)/(P-1)` | pooled rows and model-equal top-k summaries | random-label expectation `min(k,P)/P` | `r3_diagonal_rank.csv` | JoH R3 reorganization diagnostic plus `08_parameter_identity_correspondence.py` |
| E11 | Parameter-count sensitivity | model / parameter row | 35 contrast models | same rows as E9/E10 | 35 model summaries; 270 contrast parameter rows | Spearman association of `A_m`, top-1, normalized rank with `P_m` | descriptive sensitivity only; no model-complexity claim | none beyond rank/random baseline | E9/E10 tables | derived from existing E9/E10 artifacts |
| E12 | `A_diag^IC-self` feasibility | putative within-IC profile contrast | all36 restart archive | would require ≥2 independent performance-comparable IC restarts per basin and comparable 20-D profile construction | no canonical `C^{IC-self}` table exists | not computed as a profile correspondence | not applicable | existing RMS reference is not analogous | `agent_B_IC_self_benchmark.md` only for RMS distance | `ic_self_benchmark.py` computes RMS, not `A_diag` |
| E13 | `A_role` | model-level role off-diagonal contrast | high-confidence roles; high+medium sensitivity | original diagonal removed; same-role and cross-role alternatives both present | 36 model-level permutation rows; 29 valid same/cross-role hierarchy models | per parameter: median same-role off-diagonal minus median cross-role correspondence; then model median | median over model-level role contrasts | 10,000 role-count-preserving within-model label permutations | `15_ROLE_LABEL_PERMUTATION.csv`, `14_CORRESPONDENCE_HIERARCHY.csv` | `functional_role_diagnostic.py` |
| E14 | HESS-prior `D_rho` | model-level paired role difference | pre-specified four-role set | comparable models with prior and comparison role summaries | 27 models | prior-role median `D_rho` minus comparison-role median `D_rho` | median over paired model differences | paired bootstrap CI and 10,000 sign flips | `22_HESS_PRIOR_ROLE_TEST.csv` | `functional_role_diagnostic.py` |
| E15 | Existing identifiability support | model × parameter profile | all36 | complete restart uncertainty joined to 271 profiles | 271 parameter profiles | Spearman association of restart spread with profile reproducibility | pooled parameter-level descriptive association | bootstrap CI; no causal decomposition | `R3_IDENTIFIABILITY_ASSOCIATIONS.csv`, `R3_IDENTIFIABILITY_REPRODUCIBILITY.csv` | `05_identifiability_reproducibility.py` |

### E1 model-by-model parameter counts

The 36-model parameter registry sums to 271 coordinates:

| Model | P | Model | P | Model | P |
|---|---:|---|---:|---|---:|
| alpine1 | 4 | alpine2 | 6 | australia | 8 |
| collie1 | 1 | collie2 | 4 | collie3 | 6 |
| flexb | 9 | flexi | 10 | flexis | 12 |
| gr4j | 4 | gsfb | 8 | hbv96 | 15 |
| hillslope | 7 | hymod | 5 | ihacres | 6 |
| modhydrolog | 15 | mopex1 | 5 | mopex2 | 7 |
| mopex3 | 8 | mopex4 | 10 | mopex5 | 12 |
| newzealand1 | 6 | newzealand2 | 8 | penman | 4 |
| plateau | 8 | simhyd | 7 | smar | 8 |
| susannah1 | 6 | susannah2 | 6 | tank | 12 |
| tcm | 6 | topmodel | 7 | us1 | 5 |
| vic | 10 | wetland | 4 | xinanjiang | 12 |

## 4. R3.1 evidence audit

### 4.1 Profile-level correspondence

The canonical profile input is the 20-dimensional information-cluster vector of basin-wise association coefficients:

```text
R_m,p = Spearman_k(rho_IC[m,p,k], rho_dPL[m,p,k]), k = 1,...,20
R_m   = median_p(R_m,p)
R_paired = median_m(R_m)
```

The 20 dimensions are the frozen `absrho_ge_0.70_C001`–`C020` information clusters, represented by their cluster PC1 scores. The profile metric is exact SciPy average-tie Spearman correlation. There is no additional profile standardization; rank transformation is the correlation calculation. Constant/nonfinite profiles would be ineligible when fewer than three finite values or fewer than two unique values remain, but the canonical 271-profile information block is complete and finite, so **271/271 profiles enter**.

Canonical aggregation and values:

| Space/aggregation | Value | Denominator | Status |
|---|---:|---:|---|
| Information cluster, median over model medians of parameter profiles | `0.7157894737` | 36 model medians; 271 profiles | **Canonical `R_paired`** |
| Information cluster, mean over model medians | `0.704803...` | 36 | Sensitivity only |
| Information cluster, median of flattened model profiles | `0.659779...` | 36 | Alternative, not headline |
| Information cluster, cell-equal median | `0.724812...` | 271 profile values | Alternative, not headline |
| Raw 35-attribute, model-equal profile median | `0.732983...` | 36 | Raw-space sensitivity, not R3 headline |
| Raw 35-attribute, cell-equal median | `0.752661...` | 271 | Raw-space alternative |

The current canonical chain therefore supports `0.7157894737 ≈ 0.716`. The approximately `0.733` value belongs to an alternative/raw or historical aggregation in the working-tree documentation and must not be called the 20-D information-profile headline.

The profile table includes 1,000 feature-bootstrap intervals for individual profiles. There is no single bootstrap CI for the final median-of-medians in the canonical headline table; leave-one-model-out influence products are available as a deterministic sensitivity. This is adequate for a descriptive figure, but the main text should not invent an aggregate CI.

### 4.2 Stable-cell retention

The stable subset is selected only from IC:

```text
abs(rho_IC) >= 0.20
and IC bootstrap sign probability >= 0.95
```

The dPL value is not used for eligibility. The exact separation is:

```text
R_paired  -> profile-level estimand
902 cells -> cell-level stable-relation estimand
```

For the primary information-cluster space:

| Retention event | Numerator / denominator | Percentage | 1,000-cell-bootstrap 95% CI |
|---|---:|---:|---:|
| IC-stable subset | `902 / 5,420` selected cells | `16.642%` of all cells | selection count, not a retention CI |
| Same sign | `849 / 902` | `94.124%` | not stored for the unthresholded sign row |
| Same sign and `abs(rho_dPL) >= 0.10` | `791 / 902` | `87.694%` | `[85.698%, 89.911%]` |
| Same sign and `abs(rho_dPL) >= 0.20` | `692 / 902` | `76.718%` | `[73.947%, 79.490%]` |
| Same sign and `abs(rho_dPL) >= 0.30` | `517 / 902` | `57.317%` | `[54.102%, 60.643%]` |

Coverage audit of the retained table:

- all 36 models contribute at least one stable cell;
- model-level stable-cell counts range from 7 to 54;
- 219 of 271 parameter coordinates have at least one stable information cell;
- 19 of 20 information dimensions appear in the stable subset; one dimension has no IC-stable cells;
- this unequal coverage is a property of the IC-only selection rule, not a reason to redefine the denominator.

The percentages are conditional on `IC stable`; none should be described as retention of parameter profiles or as rates over all 5,420 cells.

### 4.3 Matched-threshold four-way

The existing symmetric implementation is:

```python
ic_hit  = abs(rho_ic)  >= threshold
dpl_hit = abs(rho_dpl) >= threshold
```

It classifies every one of the 5,420 full cells into `both`, `IC-only`, `dPL-only`, or `neither`. No bootstrap stability condition is added to this four-way table, and the unit is a cell. The all-cell results are:

| Threshold | both | IC-only | dPL-only | neither | Model-equal median fractions (`both / IC-only / dPL-only / neither`) |
|---:|---:|---:|---:|---:|---|
| 0.10 | `1,987 / 5,420 = 36.661%` | `489 / 5,420 = 9.022%` | `1,643 / 5,420 = 30.314%` | `1,301 / 5,420 = 24.004%` | `37.750% / 8.750% / 29.375% / 23.750%` |
| 0.20 | `712 / 5,420 = 13.137%` | `190 / 5,420 = 3.506%` | `1,451 / 5,420 = 26.771%` | `3,067 / 5,420 = 56.587%` | `15.917% / 3.750% / 25.625% / 56.875%` |
| 0.30 | `256 / 5,420 = 4.723%` | `72 / 5,420 = 1.328%` | `993 / 5,420 = 18.321%` | `4,099 / 5,420 = 75.627%` | `4.000% / 1.500% / 18.375% / 75.250%` |
| 0.40 | `63 / 5,420 = 1.162%` | `22 / 5,420 = 0.406%` | `623 / 5,420 = 11.494%` | `4,712 / 5,420 = 86.937%` | `0.563% / 0.000% / 11.339% / 87.679%` |

The historical-looking `68% dPL-only` value is not the symmetric four-way headline and is superseded if it came from an asymmetric selection rule. It must not be carried into F5.

## 5. R3.2 evidence audit

### 5.1 `C_{m,p,q}`

The canonical implementation constructs, for each model, an `P_m × P_m` matrix:

```text
C[m,p,q] = Spearman_k(rho_IC[m,p,k], rho_dPL[m,q,k]), k = 1,...,20
```

where `p` is the IC parameter coordinate and `q` is the dPL parameter coordinate. The matrix is calculated by ranking each 20-D row with average tie ranks, centering each row rank vector, and taking the normalized dot product. This is equivalent to Spearman correlation across the 20 frozen information dimensions. It does not compare raw parameter values, does not use attribute-space profiles, and does not add a causal interpretation.

- `C` is complete for all 36 models.
- `collie1` contributes a `1 × 1` matrix; it has no off-diagonal alternative.
- Full information-space correspondence data are retained in `R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv` and the per-model NPZ caches.
- Raw-attribute `C` matrices exist as a sensitivity, not as an R3 headline replacement.

### 5.2 `A_diag`

The exact canonical row and model definitions are:

```text
off_m,p = median_{q != p} C[m,p,q]
adv_m,p = C[m,p,p] - off_m,p
A_m     = median_p(adv_m,p)
A_diag  = median_m(A_m) over models with a defined off-diagonal contrast
```

This is a **parameter-level contrast → model-level median → model-equal ensemble median**. It is not a pooled mean over all `C` cells, not a median of one pooled diagonal and one pooled off-diagonal set, and not a rank statistic.

| Quantity | Canonical value | Denominator |
|---|---:|---:|
| Diagonal model-equal median | `0.7157894737` | 36 model summaries |
| Off-diagonal model-equal median | `-0.0218045113` | 35 valid model summaries |
| `A_diag` | `0.6150375940` | 35 valid contrast models / 36 total |
| Fraction of valid model summaries with diagonal > off-diagonal | `1.000000` | 35 |
| Exclude-SIMHYD `A_diag` | `0.6105263158` | 34 valid contrast models / 35 total |

The narrow interpretation is:

> Cross-paradigm correspondence is preferentially aligned with the same model parameter coordinate rather than alternative parameter coordinates.

It does **not** establish that information belongs to, is stored by, or is carried by the same physical parameter; it does not establish parameter identity, uniqueness, causality, or universal hydrological function.

### 5.3 Permutation null

The primary null is a **within-model permutation of dPL parameter labels/profiles**:

- IC profiles remain fixed;
- every dPL profile remains intact;
- dPL parameter marginals and within-model profile geometry remain intact;
- only the IC–dPL parameter-coordinate pairing is broken;
- the full `P_m × P_m` matrix is recomputed for each draw;
- `n = 1,000`, all36 seed `20260902`, exclude-SIMHYD seed `20310902`;
- empirical p is the one-sided upper tail `(1 + count(null >= observed))/(1 + 1000)`.

Primary information-space result:

| Population | Observed `A_diag` | Null mean | Null 2.5–97.5% interval | p |
|---|---:|---:|---:|---:|
| all36 | `0.6150375940` | `0.0015360902` | `[-0.1233082707, 0.1128007519]` | `0.0009990010` |
| exclude-SIMHYD | `0.6105263158` | `-0.0001571429` | `[-0.124455..., 0.112049...]` | `0.0009990010` |

The approximately `0.0015` historical null center therefore reproduces as `0.0015360902`; the minimum empirical p is expected for 0 of 1,000 null draws at or above the observed statistic.

### 5.4 35-model distribution

`collie1` is the only one-parameter model (`P=1`), so it has no defined off-diagonal contrast and is excluded from the `A_m` distribution. The 35 eligible models have:

- `N = 35`;
- `A_m > 0`: `35/35 = 100%`;
- median `0.615038`;
- Q25 `0.450752`;
- Q75 `0.764286`;
- IQR `0.313534`;
- minimum `0.221805`;
- maximum `1.410526`.

The following table is the F6-ready per-model table. It retains parameter count and the existing top-1/top-3 summaries.

| Model | P | `A_m` | top-1 | top-3 |
|---|---:|---:|---:|---:|
| alpine1 | 4 | 0.801504 | 0.750 | 1.000 |
| alpine2 | 6 | 0.914286 | 0.833 | 1.000 |
| australia | 8 | 0.254887 | 0.625 | 0.750 |
| collie2 | 4 | 1.410526 | 0.750 | 0.750 |
| collie3 | 6 | 0.438346 | 0.667 | 0.667 |
| flexb | 9 | 0.418045 | 0.444 | 0.667 |
| flexi | 10 | 0.763158 | 0.500 | 0.800 |
| flexis | 12 | 0.765414 | 0.333 | 0.917 |
| gr4j | 4 | 0.421805 | 1.000 | 1.000 |
| gsfb | 8 | 0.276692 | 0.250 | 0.875 |
| hbv96 | 15 | 0.747368 | 0.600 | 0.933 |
| hillslope | 7 | 0.530827 | 0.571 | 1.000 |
| hymod | 5 | 0.848120 | 0.800 | 1.000 |
| ihacres | 6 | 0.742105 | 1.000 | 1.000 |
| modhydrolog | 15 | 0.759398 | 0.667 | 0.800 |
| mopex1 | 5 | 0.496992 | 0.200 | 1.000 |
| mopex2 | 7 | 0.960150 | 1.000 | 1.000 |
| mopex3 | 8 | 0.454135 | 0.375 | 0.750 |
| mopex4 | 10 | 0.537594 | 0.300 | 0.700 |
| mopex5 | 12 | 0.606015 | 0.583 | 0.667 |
| newzealand1 | 6 | 0.959398 | 0.667 | 0.833 |
| newzealand2 | 8 | 0.447368 | 0.375 | 0.875 |
| penman | 4 | 0.643609 | 0.750 | 1.000 |
| plateau | 8 | 0.456391 | 0.625 | 0.750 |
| simhyd | 7 | 0.619549 | 0.857 | 1.000 |
| smar | 8 | 0.221805 | 0.375 | 0.750 |
| susannah1 | 6 | 0.388722 | 0.833 | 0.833 |
| susannah2 | 6 | 0.615038 | 0.667 | 1.000 |
| tank | 12 | 0.348872 | 0.417 | 0.500 |
| tcm | 6 | 0.909023 | 0.667 | 0.833 |
| topmodel | 7 | 0.466917 | 0.429 | 0.857 |
| us1 | 5 | 0.743609 | 0.600 | 0.800 |
| vic | 10 | 0.538346 | 0.500 | 0.900 |
| wetland | 4 | 1.245865 | 0.750 | 1.000 |
| xinanjiang | 12 | 0.632331 | 0.500 | 0.917 |

### 5.5 Diagonal rank / top-k

The rank is descending and uses the strict-greater rule from the existing implementation:

```text
rank = 1 + count_q(C[p,q] > C[p,p])
```

Ties therefore share the same rank; no arbitrary tie-breaking is introduced. Normalized percentile is `(P-rank)/(P-1)` for `P>1`.

For all 271 parameter rows, including the single-parameter model:

| Event | Count / n | Observed | Random-label expectation |
|---|---:|---:|---:|
| top-1 | `156 / 271` | `57.565%` | `13.284%` |
| top-2 | `193 / 271` | `71.218%` | `26.199%` |
| top-3 | `228 / 271` | `84.133%` | `39.114%` |

For the 270 rows in the 35-model off-diagonal-eligible subset:

| Event | Count / n | Observed | Random-label expectation |
|---|---:|---:|---:|
| top-1 | `155 / 270` | `57.407%` | `12.963%` |
| top-2 | `192 / 270` | `71.111%` | `25.926%` |
| top-3 | `227 / 270` | `84.074%` | `38.889%` |

All contrast-eligible models have `P >= 4`, so top-3 is mechanically comparable across the 35-model contrast population. These results show that the positive `A_diag` is not merely a tiny universal increment: the same coordinate is often first or in the first three, while not winning for every parameter.

### 5.6 Parameter-count sensitivity

Existing artifacts support a bounded sensitivity statement, not a model-complexity story:

- Spearman `A_m` versus `P_m`: `rho = -0.2825`, `p = 0.1001` across 35 models.
- Spearman top-1 fraction versus `P_m`: `rho = -0.5635`, `p = 0.00042`; more alternatives mechanically make top-1 harder, which is why the random-label baseline is required.
- Pooled normalized rank percentile versus `P_m`: `rho = -0.0851`, `p = 0.1634` across 270 rows.
- Model-equal median normalized percentile versus `P_m`: `rho = -0.4268`, `p = 0.0106`; the weighting choice matters and should remain a sensitivity, not a headline.

The `A_m` effect remains positive in all 35 eligible models, but top-k concentration is not parameter-count invariant. F6 should show the model-level distribution and state the count-sensitivity boundary rather than claim that parameter count is irrelevant.

## 6. R3.3 interpretation-boundary audit

### 6.1 IC-self feasibility

#### Predeclared eligibility rule

The existing IC-self benchmark froze the following rule before reading the distance contrast:

1. use archived **IC training KGE** rather than test KGE;
2. within each basin, retain independent IC starts whose training KGE is within `0.01` of the best archived IC restart;
3. use bounds-normalized RMS distance over the model's parameter coordinates;
4. require finite normalized parameter vectors and exact alignment of the canonical IC vector with the best archived restart;
5. treat the result as a one-sided reference to the archived IC realization, not as a symmetric within-paradigm variance estimate;
6. do not call the best restart itself an independent performance replicate; a profile-level `A_diag^IC-self` would additionally require at least two independent eligible starts and a re-computed 20-D association profile for each start.

The rule is outcome-independent and was not adjusted to make the observed contrast larger. At the `0.01` rule, every basin has at least one eligible restart (the best restart is necessarily included), but `1,696/19,116 = 8.87%` of basin-model rows have only one eligible restart. Thus a strict two-independent-start profile rule would not have complete coverage.

#### Coverage table

The restart gate and `ELIGIBILITY_QC.csv` give complete archived start-level latent vectors and fitness metadata for all models. The table below is the model-level coverage audit; `P` is the frozen model parameter count.

| Model | P | Starts/basin | Basin coverage | Parameter coverage | Performance metadata | Eligible for existing IC-self RMS? | Eligible for analogous `A_diag^IC-self`? |
|---|---:|---:|---|---|---|---|---|
| alpine1 | 4 | 10 | 531/531 | 4/4 | archived IC training fitness | yes | no: no C/self-profile artifact |
| alpine2 | 6 | 10 | 531/531 | 6/6 | archived IC training fitness | yes | no: no C/self-profile artifact |
| australia | 8 | 10 | 531/531 | 8/8 | archived IC training fitness | yes | no: no C/self-profile artifact |
| collie1 | 1 | 10 | 531/531 | 1/1 | archived IC training fitness | yes | no: no off-diagonal contrast |
| collie2 | 4 | 10 | 531/531 | 4/4 | archived IC training fitness | yes | no: no C/self-profile artifact |
| collie3 | 6 | 10 | 531/531 | 6/6 | archived IC training fitness | yes | no: no C/self-profile artifact |
| flexb | 9 | 10 | 531/531 | 9/9 | archived IC training fitness | yes | no: no C/self-profile artifact |
| flexi | 10 | 10 | 531/531 | 10/10 | archived IC training fitness | yes | no: no C/self-profile artifact |
| flexis | 12 | 10 | 531/531 | 12/12 | archived IC training fitness | yes | no: no C/self-profile artifact |
| gr4j | 4 | 10 | 531/531 | 4/4 | archived IC training fitness | yes | no: no C/self-profile artifact |
| gsfb | 8 | 10 | 531/531 | 8/8 | archived IC training fitness | yes | no: no C/self-profile artifact |
| hbv96 | 15 | 10 | 531/531 | 15/15 | archived IC training fitness | yes | no: no C/self-profile artifact |
| hillslope | 7 | 10 | 531/531 | 7/7 | archived IC training fitness | yes | no: no C/self-profile artifact |
| hymod | 5 | 10 | 531/531 | 5/5 | archived IC training fitness | yes | no: no C/self-profile artifact |
| ihacres | 6 | 10 | 531/531 | 6/6 | archived IC training fitness | yes | no: no C/self-profile artifact |
| modhydrolog | 15 | 10 | 531/531 | 15/15 | archived IC training fitness | yes | no: no C/self-profile artifact |
| mopex1 | 5 | 10 | 531/531 | 5/5 | archived IC training fitness | yes | no: no C/self-profile artifact |
| mopex2 | 7 | 10 | 531/531 | 7/7 | archived IC training fitness | yes | no: no C/self-profile artifact |
| mopex3 | 8 | 10 | 531/531 | 8/8 | archived IC training fitness | yes | no: no C/self-profile artifact |
| mopex4 | 10 | 10 | 531/531 | 10/10 | archived IC training fitness | yes | no: no C/self-profile artifact |
| mopex5 | 12 | 10 | 531/531 | 12/12 | archived IC training fitness | yes | no: no C/self-profile artifact |
| newzealand1 | 6 | 10 | 531/531 | 6/6 | archived IC training fitness | yes | no: no C/self-profile artifact |
| newzealand2 | 8 | 10 | 531/531 | 8/8 | archived IC training fitness | yes | no: no C/self-profile artifact |
| penman | 4 | 10 | 531/531 | 4/4 | archived IC training fitness | yes | no: no C/self-profile artifact |
| plateau | 8 | 10 | 531/531 | 8/8 | archived IC training fitness | yes | no: no C/self-profile artifact |
| simhyd | 7 | 10 | 531/531 | 7/7 | archived IC training fitness | yes | no: no C/self-profile artifact |
| smar | 8 | 10 | 531/531 | 8/8 | archived IC training fitness | yes | no: no C/self-profile artifact |
| susannah1 | 6 | 10 | 531/531 | 6/6 | archived IC training fitness | yes | no: no C/self-profile artifact |
| susannah2 | 6 | 10 | 531/531 | 6/6 | archived IC training fitness | yes | no: no C/self-profile artifact |
| tank | 12 | 10 | 531/531 | 12/12 | archived IC training fitness | yes | no: no C/self-profile artifact |
| tcm | 6 | 10 | 531/531 | 6/6 | archived IC training fitness | yes | no: no C/self-profile artifact |
| topmodel | 7 | 10 | 531/531 | 7/7 | archived IC training fitness | yes | no: no C/self-profile artifact |
| us1 | 5 | 10 | 531/531 | 5/5 | archived IC training fitness | yes | no: no C/self-profile artifact |
| vic | 10 | 10 | 531/531 | 10/10 | archived IC training fitness | yes | no: no C/self-profile artifact |
| wetland | 4 | 10 | 531/531 | 4/4 | archived IC training fitness | yes | no: no C/self-profile artifact |
| xinanjiang | 12 | 10 | 531/531 | 12/12 | archived IC training fitness | yes | no: no C/self-profile artifact |

#### Go/no-go decision

`agent_B_IC_self_benchmark.md` reports an existing one-sided RMS result at the frozen `0.01` rule:

- `D_cross = 0.384375`;
- IC-self RMS reference `D_self = 0.073444`;
- difference `0.218775`, bootstrap CI `[0.209136, 0.228145]`;
- all 36 model medians are positive.

This is **not** an `A_diag^IC-self` result. It compares parameter-distance magnitudes, not IC–IC 20-D information-profile correspondence and not same-coordinate versus alternative-coordinate correspondence. Therefore:

> **`A_diag^IC-self = NO-GO for main text`**

No new IC-self profile study is initiated. The RMS reference may appear in SI as a clearly labeled, one-sided parameter-distance reference, but it cannot be used to construct an `A_diag` ordering or an “upper bound.”

### 6.2 Functional-role null

The primary role taxonomy is source-backed and frozen before analysis. Role mapping includes snow, soil storage capacity, runoff-generation partitioning, interception/ET, percolation/interflow, groundwater/baseflow recession, routing delay, exchange loss, and other/unresolved categories. Ambiguous/low-confidence roles are not forced into the primary high-confidence result.

The implementation removes the original diagonal before comparing role alternatives. For each eligible parameter, it computes the median same-role off-diagonal correspondence minus the median cross-role correspondence; it then takes model medians and the median across valid models. Role labels are permuted within model while preserving the observed role counts. The model is the inferential unit; role-pair rows are descriptive.

Canonical high-confidence result:

| Quantity | Value | Unit/denominator |
|---|---:|---|
| `A_role` | `-0.1187969925` | median over 36 model-level role contrasts |
| Positive high-confidence model `A_role` | `12/36` | model count |
| Same-role off-diagonal median | `-0.0278195489` | 29 valid model summaries |
| Cross-role off-diagonal median | `0.0052631579` | 29 valid model summaries |
| Role-label permutations | `10,000` | within-model, role counts fixed |
| Role-label p | `0.8057194281` | upper-tail empirical p |

The result is negative/null, not evidence that hydrological roles do not exist. It says the tested role labels do not add an additional stable signal beyond the coordinate effect. The correct boundary sentence is:

> The retained organization is parameterization-specific rather than evidence for universal hydrological-function correspondence.

### 6.3 Existing boundary / identifiability support

Only existing artifacts are used:

- pooled parameter-level restart spread versus profile reproducibility: `rho = -0.374376`, `n = 271`, bootstrap CI `[-0.473632, -0.270936]`;
- existing boundary/restart sensitivity tables distinguish boundary occupancy, restart spread, and profile persistence without selecting the primary denominator on the outcome;
- existing dPL-construction audit distinguishes IC-supported from dPL-emergent cells and warns that dPL attribute associations are not independent hydrological validation.

These are SI/supporting diagnostics. They are associations, not causal decompositions and not substitutes for `A_diag^IC-self`.

### R3/R4 representation separation

| Item | Representation |
|---|---|
| R3 seen-basin canonical representation | 20 frozen `absrho_ge_0.70_C001`–`C020` information dimensions from the 35-attribute clustering contract |
| R4 OOB representation | prespecified continuous-variable subset: 13 continuous information clusters and 32 continuous attributes in the held-out replay |
| Identical? | **NO** |

R4 OOB values must not be presented as exact replication of the R3 20-D headline. R4 has a different representation and an eight-model held-out scope.

## 7. Figure-ready evidence map

### Figure 5 — Retention magnitude

Scientific question: under predefined same-parameter pairing, how much catchment–parameter association structure is retained across calibration paradigms?

| Panel candidate | Source artifact | Unit | n | Headline number | Uncertainty | Status |
|---|---|---|---:|---:|---|---|
| Profile correspondence distribution | `R3_PARAMETER_PROFILE_REPRODUCIBILITY.csv` | model × parameter profile | 271 | `R_paired = 0.715789` | profile-level feature-bootstrap CIs; no aggregate CI | **READY** |
| Model-level profile summaries | `R3_MODEL_REPRODUCIBILITY.csv` | model | 36 | median of model medians | deterministic model distribution/LOO | **READY** |
| Stable-cell retention ladder | `R3_IC_TO_DPL_MAGNITUDE_RETENTION_SUMMARY.csv` | IC-stable cell | 902 | `849/902`, `791/902`, `692/902`, `517/902` | paired-cell bootstrap for magnitude ladder | **READY** |
| Matched four-way inset | `r2_threshold_fourway.csv` | full association cell | 5,420 per threshold | at `t=.20`: 712/190/1,451/3,067 | descriptive symmetric ladder | **READY after lightweight count closure** |

F5 must not contain `A_diag`, off-diagonal contrasts, or functional-role values.

### Figure 6 — Same-coordinate specificity

Scientific question: is retained correspondence preferentially aligned with the same parameter coordinate?

| Panel candidate | Source artifact | Unit | n | Role | Status |
|---|---|---|---:|---|---|
| `C_{m,p,q}` heatmap or diagonal/off-diagonal comparison | `R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv` and NPZ caches | within-model parameter pair | 2,395 information-space C entries (4,790 including raw sensitivity) | hero structural panel | **READY** |
| Observed versus label-null `A_diag` | `R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv`, `R3_PARAMETER_LABEL_PERMUTATION_NULL.csv` | valid model contrast / permutation | 35 / 1,000 | **hero statistical panel** | **READY** |
| 35-model `A_m` distribution | `r3_diagonal_advantage.csv` | model | 35 | median `.615038`; 35/35 positive | supporting | **READY** |
| Diagonal rank/top-k | `r3_diagonal_rank.csv` | parameter row | 271 or 270 contrast rows | top-1 `.5756`, top-3 `.8413` all36 | supporting/inset | **READY** |
| Parameter-count sensitivity | E9/E10 tables | model/parameter | 35/270 | bounded sensitivity | SI or inset | **READY** |

The figure caption must say “preferential same-coordinate alignment”; it must not say “same physical parameter identity.”

### Figure 7 — Interpretation boundary

The canonical choice is **Branch B**:

| Panel candidate | Source artifact | Unit | Status |
|---|---|---|---|
| Same-coordinate reference versus role alternatives | `14_CORRESPONDENCE_HIERARCHY.csv`, `15_ROLE_LABEL_PERMUTATION.csv` | model-level correspondence summaries / 10,000 null draws | **READY** |
| Functional-role boundary | `15_ROLE_LABEL_PERMUTATION.csv`, `22_HESS_PRIOR_ROLE_TEST.csv` | model-level role contrast | **READY** |
| Existing identifiability/boundary support | `R3_IDENTIFIABILITY_ASSOCIATIONS.csv` and boundary/restart tables | 271 profiles / model sensitivities | **READY for SI/support** |
| IC-self `A_diag` reference | no analogous artifact exists | — | **NO-GO; do not draw as `A_diag^IC-self`** |
| Existing IC-self RMS distance reference | `agent_B_IC_self_benchmark.md` | parameter-distance reference | **SI only; not an A-diag panel** |

F7 therefore answers how far coordinate specificity can be interpreted: it reaches same-coordinate organization, but not broad hydrological functional-role correspondence.

## 8. Main-text / SI placement

| Result | Main text / figure | SI | Reason |
|---|---|---|---|
| `R_paired` profile-level correspondence | F5 hero | full 271-profile table | primary retention estimand; profile unit must be visible |
| IC-stable retention ladder | F5 supporting panel/inset | full cell table and coverage | important conditional result; denominator must be explicit |
| Matched-threshold four-way | F5 inset or concise text | full `.10/.20/.30/.40` table | symmetric closure; not a replacement for profile retention |
| `C_{m,p,q}` / `A_diag` | F6 hero | full matrices and parameter rows | central same-coordinate specificity evidence |
| Parameter-label permutation null | F6 hero/null panel | null draws and implementation diagnostics | protects against generic profile geometry |
| 35-model `A_m` distribution | F6 supporting | full model table | shows breadth and prevents pooled-cell overstatement |
| Diagonal rank/top-k | F6 inset/supporting | full rank rows and tie diagnostics | explains whether `A_diag` is a small advantage or frequent top-rank alignment |
| Parameter-count sensitivity | F6 inset or SI | complete sensitivity table | top-k is mechanically affected by `P_m` |
| Functional-role negative control | F7 main panel | full role registry and role-level sensitivities | interpretation boundary; no role-taxonomy rescue |
| HESS-prior `D_rho` | F7 supporting text | full paired model table and bootstrap/permutation draws | negative pre-specified boundary check |
| IC-self RMS reference | not a main-text `A_diag` panel | SI only, explicitly one-sided RMS | existing artifact is not analogous to `A_diag` |
| Boundary/identifiability association | not hero | SI | supporting alternative explanation, not causal mechanism |
| R4 OOB representation | R4 only | R4 SI | not exact R3 replication |

## 9. Numerical consistency table

| Quantity | Historical expected | Canonical recomputed | Exact denominator | Aggregation | Status |
|---|---:|---:|---|---|---|
| Models | 36 | 36 | model registry | all36 | **MATCH** |
| Model–parameter profiles | 271 | 271 | sum of `P_m` | complete profiles | **MATCH** |
| Information dimensions | 20 | 20 | frozen cluster registry | per profile | **MATCH** |
| Full association cells | 5,420 | 5,420 | `271 × 20` | cell | **MATCH** |
| IC-stable cells | 902 | 902 | IC-only gate within 5,420 | cell | **MATCH** |
| `R_paired` | about 0.716 | `0.7157894737` | 271 profiles; 36 model medians | median-p within model, median-m | **MATCH** |
| Same sign | 94.1% | `849/902 = 94.12416851%` | 902 IC-stable cells | pooled conditional rate | **MATCH** |
| Same sign + `abs(rho_dPL)>=.10` | 87.7% | `791/902 = 87.69401330%` | 902 | pooled conditional rate | **MATCH** |
| Same sign + `abs(rho_dPL)>=.20` | 76.7% | `692/902 = 76.71840355%` | 902 | pooled conditional rate | **MATCH** |
| Same sign + `abs(rho_dPL)>=.30` | 57.3% | `517/902 = 57.31707317%` | 902 | pooled conditional rate | **MATCH** |
| `A_diag` | about 0.615 | `0.6150375940` | 35 valid contrast models | median of model medians of row contrasts | **MATCH** |
| Parameter-label null center | about 0.0015 | `0.0015360902` | 1,000 permutations | null mean | **MATCH** |
| Parameter-label p | 0.000999 | `0.0009990010` | 1,000 permutations | upper tail with +1 correction | **MATCH** |
| Exclude-SIMHYD `A_diag` | about 0.611 | `0.6105263158` | 34 valid contrast models / 35 total | same as primary | **MATCH** |
| `A_role` | about -0.119 | `-0.1187969925` | 36 model role contrasts | median model-level role contrast | **MATCH** |
| Role-label p | about 0.806 | `0.8057194281` | 10,000 permutations | upper tail | **MATCH** |
| Same-role off-diagonal median | -0.028 | `-0.0278195489` | 29 valid hierarchy models | model-equal median | **MATCH** |
| Cross-role off-diagonal median | 0.005 | `0.0052631579` | 29 valid hierarchy models | model-equal median | **MATCH** |
| HESS-prior `D_rho` | about -0.00033 | `-0.0003301577` | 27 paired models | median paired difference | **MATCH** |
| HESS-prior p | about 0.850 | `0.8498150185` | 10,000 sign flips | two-sided sign-flip implementation | **MATCH** |

## 10. Final blocker list

> **NO REMAINING R3 NUMERICAL BLOCKER**

The following are warnings/limits, not blockers under the requested classification:

1. `R3_ESTIMAND_DICTIONARY.md` contains stale/alternative `.733` historical mapping. Use the checksum-verified `R3_NUMERIC_FREEZE_TABLE.csv`, `R3_OVERALL_REPRODUCIBILITY.csv`, and `legacy_continuity_reference.csv` for the canonical information-space headline.
2. The role script's named R2 role-registry input is absent from the current working tree, although the generated source-backed role registry, run manifest, hashes, and role outputs are present. Preserve the frozen output; do not silently rerun with a new taxonomy.
3. IC-self profile-level `A_diag` is not available. This is an explicitly recorded **NO-GO optional reference**, not a blocker to F7 Branch B.
4. The working tree is not clean and R3 directories are untracked relative to HEAD. Preserve the artifacts and their hashes; do not reset or overwrite unrelated work.

## 11. Recommended next action

**Path A (qualified):**

```text
F5 -> F6 -> F7 -> R3.1 -> R3.2 -> R3.3
```

Use F7 **Branch B**. Carry forward the exact statistical identities:

- `R_paired` is profile-level;
- `902` and its retention ladder are cell-level conditional evidence;
- `A_diag` is preferential same-coordinate alignment;
- functional-role evidence is a negative interpretation boundary;
- the existing IC-self RMS result is SI-only and is not `A_diag^IC-self`;
- R4's 13-continuous-cluster representation is not an exact replication of R3's 20-D representation.

No new training, calibration, dPL seed, performance-conditioned R3 stratification, functional taxonomy, or physical-mechanism experiment is recommended.
