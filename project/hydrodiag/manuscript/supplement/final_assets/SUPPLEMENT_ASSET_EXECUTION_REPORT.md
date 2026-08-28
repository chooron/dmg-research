# Supplement asset execution report

Generated: 2026-08-27
Project root: `/home/jingxin/code/dmg-research/project/hydrodiag`
Git HEAD at preflight: `322fb932c922a131a67800bcbc6aa7eb704c7605`
Python: `/home/jingxin/code/dmg-research/.venv/bin/python` (Python 3.10.20)

## A. Final assets

| Asset | Final path | Status | Validated |
|---|---|---|---:|
| Table S1 | `tables/Table_S1/Table_S1.csv` and `Table_S1.md` | CREATE_FROM_EXISTING_DATA | YES |
| Table S2 | `tables/Table_S2/Table_S2_panelA.csv`, `Table_S2_panelB.csv`, `Table_S2.md` | REGENERATE_FROM_EXISTING_DATA | YES |
| Table S3 | `tables/Table_S3/Table_S3_panelA.csv`, `Table_S3_panelB.csv`, `Table_S3.md` | CREATE_FROM_EXISTING_DATA | YES |
| Figure S1 | `figures/Figure_S1/Figure_S1.png` | RENUMBER_ONLY | YES |
| Figure S2 | `figures/Figure_S2/Figure_S2.png` | RENUMBER_ONLY | YES |
| Figure S3 | `figures/Figure_S3/Figure_S3.png` | CREATE_FROM_EXISTING_DATA | YES |
| Figure S4 | `figures/Figure_S4/Figure_S4.png` | RENUMBER_ONLY | YES |
| Figure S5 | `figures/Figure_S5/Figure_S5.png` | RENUMBER_ONLY | YES |
## Final PNG inventory

| Figure | Dimensions | Bytes | SHA-256 |
|---|---:|---:|---|
| S1 | 4073 × 1593 | 231177 | `83092c031751b89e3f39df2c14f56a5f17804647412c541cb99f5b5eb2c422b5` |
| S2 | 6793 × 2497 | 790866 | `4541b60113ae196f21d8a3eb76c61d41dbfe1b1b26d69b1ef5ff05dee308f770` |
| S3 | 5984 × 4920 | 524958 | `a5660d784cdf0e8f3c546959d016122debe897415345acbea85860171c5ae1d6` |
| S4 | 6540 × 2552 | 1039434 | `4f78dbedeb828978e4051eb8570984ebd4423cc9019efa40ee539b3b69b57908` |
| S5 | 2063 × 2562 | 832697 | `8e0477c077b14902874baf6cf3ecc9e59d97e838d269a6c6efa14dae4cbdda5b` |

Every final Figure directory contains `README.md`, `provenance.md`, `caption_facts.md`, a PNG, and a plotting/staging script. Every final Table directory contains `README.md`, `provenance.md`, machine-readable CSV output, Markdown output, and a build script.

## B. Reused vs regenerated

### RENUMBER_ONLY

- Figure S1: existing corrected HUC-2 LORO renderer/output, final numbering and source-region mapping retained.
- Figure S2: existing CSV-backed TGD response asset regenerated under final formal `tau_t`/`r_t` labels.
- Figure S4: existing recorded-forward seasonal asset staged under final numbering.
- Figure S5: existing audited R4 external-state PNG staged under final numbering.

### REGENERATE_FROM_EXISTING_DATA

- Table S2: expanded from existing R1 basin-level CT data and R3 denominator sensitivity data.

### CREATE_FROM_EXISTING_DATA

- Table S1: rebuilt from production parameter dictionaries with host/membership/fixed-constant fields.
- Table S3: assembled from frozen reviewer-2 denominator, tail, canonical registry, and alternative-field summaries.
- Figure S3: rendered from frozen R3 `paired_parameters.csv` and `state_excess.csv` using the canonical state/flux keys.

### REUSE_AS_IS

No final asset was copied without either final-layer staging validation or an explicit final renderer invocation; the frozen R4 source PNG itself remains unchanged.

## C. Important numerical locks

- Table S1 calibrated counts: XAJ Base/TGD/CN = 15/17/17; GR4J = 4/6/6; SIMHYD = 10/12/12. TGD `T_ref`, `s_T`, `epsilon` and CN fixed constants are not counted.
- Table S2 Panel B canonical N_valid: 427 IC and 460 dPL; source threshold grid is `1e-6, 1e-4, 1e-3, 0.01, 0.02, 0.05, 0.10`.
- Table S3 canonical test recovery audit: IC N_valid 427, dPL N_valid 460; pooled dPL union N = 468 remains distinct.
- Figure S1: 8 retained HUC_11–HUC_18 omissions per paradigm, displayed as HUC_01–HUC_08. Full R1 references 47.400/46.267 d; R3 +0.4600/+0.4432; R5 90.91%/90.91%.
- Figure S2 shape medians: canonical Delta F -0.1339 IC and +0.4410 dPL; broad variant -0.3657 IC and -0.0914 dPL.
- Figure S3: parameter source uses `delta_abs_e`; state source uses test NRMSE `delta_E`; final keys are `wu`, `wl`, `wd`, `wt`, `qi`, `qg`.
- Figure S4 high-snow N = 133 and October–September water-year axis.
- Figure S5 eligible population N = 442 (Low 88, Middle 177, High 177); six examples are recorded in the selection audit.

## D. Symbol audit

- Figure 2/Methods formal forcing symbols: `P_t`, `T_t`, `E_{p,t}`, and `P_t^*`. Legacy `PET`/`pet` names remain code/data aliases only.
- TGD formal symbols: `S_t^g`, `tau_t`, `r_t`, `T_ref`, and `s_T`; final Figure S2 uses `tau_t` and `r_t` labels.
- CN formal symbols: `f_{solid,t}`, `P_t^s`, `P_t^r`, `G_t`, `eTG_t`, `M_t`, `C_TG`, and `K_f`.
- XAJ formal symbols: `W_{U,t}`, `W_{L,t}`, `W_{D,t}`, `W_t`, `Q_{i,t}`, `Q_{g,t}`, and `Q_t`, mapped to code/data keys `wu`, `wl`, `wd`, `wt`, `qi`, `qg`, and `qsim`/`Q`.
- `s` versus derived `wt` is a documented R3 protocol/display distinction; final Figure S3 uses explicit `wt` rather than relabeling `s`.
- `S_mm`, `S_0`, `O_i`, `O_g`, `R_i`, and `R_g` remain `UNRESOLVED_SYMBOL_MAPPING`; they were not introduced into final labels.

Full audit: `_audit/symbol_registry.md`.

## E. Blocked items

- No `BLOCKED_MISSING_DATA` item.
- No `PANEL_C_BLOCKED_PENDING_AGGREGATION_AUDIT` item; the TGD shape-sensitivity panel used the frozen basin-metric CSV and summary JSON.
- A documented `CONFLICT` remains between the dPL `F_TGD` value in the tail audit (0.5240) and the canonical registry `F_TGD*` value (0.522659); both sources are preserved and assigned to their respective Table S3 panels.
- A small documented dPL `W_t` source-summary difference remains between Figure S3's `state_excess.csv` and main Figure 6's summary; it is not silently corrected.

## F. Resource use and prohibited operations

- Training: **NO**
- CMA-ES: **NO**
- Recalibration: **NO**
- Full test pipeline: **NO**
- Full evaluation pipeline: **NO**
- Full forward rerun: **NO**
- State export: **NO**
- pytest: **NO**
- unittest: **NO**
- Python environment/package changes: **NO**
- Thread limits for plotting/building: `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2`.
- Preflight reported 4 logical CPUs and approximately 7.8 GiB RAM. No memory pressure or WSL instability was observed.
- At most two read-only scouts ran concurrently. Plotting/building was serial.

## G. Master README

`manuscript/supplement/final_assets/README.md`

Machine-readable manifest: `manuscript/supplement/final_assets/supplement_manifest.csv`.

Audit files: `manuscript/supplement/final_assets/_audit/preflight.md`, `supplement_asset_registry.csv`, `symbol_registry.md`, and `numerical_consistency_report.md`.

Visual QA was completed by opening all five final PNGs and checking panel completeness, labels, legends, units, clipping, whitespace, palette, IC/dPL encoding, reference lines, and category connection behavior. Main manuscript text, Results/Discussion text, main Figure 2, Supplement Word, canonical results, and original legacy assets were not modified by this final asset layer.
