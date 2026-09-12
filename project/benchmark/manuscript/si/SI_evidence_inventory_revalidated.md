# Supplementary Information (SI) Revalidated Evidence Inventory

**Study**: 36-Model Individual Calibration (IC) vs. Differentiable Parameter Learning (dPL) Across 531 CAMELS-US Basins  
**Current Code Commit**: `3caca37a7d446903ee52aa791b5c4900a0dcfeea` (with working-tree dynamic DOY calendar updates)  
**Audit Scope**: Multi-agent revalidation covering current `dmotpy` implementation and historical MARRMoT archive comparison.

---

## 1. Inventory Summary

| Section | Total Claims Audited | VERIFIED (Current Code / Archive) | CONFLICT | UNRESOLVED |
| :--- | :---: | :---: | :---: | :---: |
| **S1. Differentiable Model Reconstruction & Verification** | 18 | 18 | 0 | 0 |
| **S2. Data & Parameter-Estimation Configuration** | 22 | 22 | 0 | 0 |
| **S3. Parameter-Space, Information-Organization & OOB** | 28 | 28 | 0 | 0 |
| **Total** | **68** | **68** | **0** | **0** |

*Note on MARRMoT Performance Comparison*: Classified as **Level B** (Same 15-year estimation/evaluation periods, overlapping CAMELS catchments, independently calibrated). The reference comparison is now fully documented with exact empirical statistics ($\rho = 0.708, p = 1.38 \times 10^{-6}$; paired evaluation $\Delta\text{KGE} = +0.027$), resolving the previous placeholder without manufacturing false ODE-equivalence claims.

---

## 2. Section S1 Revalidated Evidence Ledger: Reconstruction & Verification

| Item / Claim | Current Verified Value / Status | Current Code / Archive Path | Identifier / Evidence Notes | Status |
| :--- | :--- | :--- | :--- | :---: |
| **36 Model Names** | Complete 36 canonical models | `dmotpy/models/registry.py` | `STFN_INFO.keys()` | **VERIFIED** |
| **MARRMoT Reference IDs** | Canonical MARRMoT IDs (`m_01` to `m_40`) | `dmotpy/tests/core_model_registry.py` | `REFERENCE_ID_MAP` | **VERIFIED** |
| **Calibrated Parameters Total** | Exactly **271** parameters across 36 models | `dmotpy/models/registry.py` | `sum(NPARAM_INFO.values()) == 271` | **VERIFIED** |
| **State Variables Total** | 110 state stores in registry (96 unique dynamic stores) | `dmotpy/models/registry.py` | `STATE_INFO` | **VERIFIED** |
| **Physical Parameter Bounds** | Min/max bounds matching MARRMoT calibration ranges | `dmotpy/models/core/*.py` | `*_PARAMS_BOUNDS` in each model file | **VERIFIED** |
| **Mass-Conserving Topology** | $\mathbf{S}_{t+1} = \mathbf{S}_t + \Delta t \sum \mathbf{F}_{\text{in}} - \Delta t \sum \mathbf{F}_{\text{out}}$ | `dmotpy/models/core/*.py` | Step functions in all 36 models | **VERIFIED** |
| **Smooth Gating Operators** | Sigmoidal soft gates $\sigma\left(\frac{k}{\tau} (S - S_{\text{thresh}})\right)$ | `dmotpy/models/flux/smooth.py` | `soft_gate_storage_above`, etc. | **VERIFIED** |
| **Smooth Bounding Operators** | `smooth_relu`, `smooth_min`, `smooth_cap_flux` | `dmotpy/models/flux/smooth.py` | Lines 51–86 | **VERIFIED** |
| **Safe Division Epsilon** | Offset `nearzero = 1e-6` | `dmotpy/models/flux/smooth.py` | Line 9 | **VERIFIED** |
| **Unit Hydrograph Routing** | 9 differentiable convolution kernels (`uh_0` to `uh_8`) | `dmotpy/models/unithydro/` | 100.00% mass conservation | **VERIFIED** |
| **Water Balance Closure** | 35/36 models pass strict closure ($< 10^{-3}$ mm/d, typical $\sim 10^{-11}$) | `dmotpy/tests/test_core_water_balance.py` | Re-run passed 35/36 models | **VERIFIED** |
| **VIC Water Balance Warning** | Max residual $0.274$ mm/d (FP64) / $0.406$ mm/d (FP32) on extreme test | `dmotpy/models/core/vic.py` | Hard clipping boundaries | **VERIFIED** |
| **VIC Dynamic DOY Support** | Gregorian Day of Year channel dynamically modulated | `dmotpy/models/core/vic.py`, `dmotpy/data_contract.py` | `CALENDAR_MODELS` includes `vic` | **VERIFIED** |
| **Forward Stability** | 100% (36/36) finite outputs (no `NaN`/`Inf`) | `dmotpy/tests/test_training_regression_smoke.py` | Re-run passed | **VERIFIED** |
| **End-to-End Autograd** | 36/36 models produce valid non-zero loss gradients | `dmotpy/tests/test_model_gradient_end_to_end.py` | Re-run passed | **VERIFIED** |
| **FP64 Gradcheck** | 13 representative models pass finite-difference gradcheck | `dmotpy/tests/test_model_gradcheck_representative.py` | Re-run passed (13/13) | **VERIFIED** |
| **Gradient Sparsity Audit** | 7 models show zero gradients on inactive subvectors under dry forcing | `dmotpy/scripts/standalone_validation_36.py` | `alpine2`, `gr4j`, `hbv96`, `modhydrolog`, `newzealand2`, `plateau`, `smar` | **VERIFIED** |
| **Euler Time-Step Convergence** | 23 nominal first-order ($p \in [0.85, 1.15]$), 3 recover @ $K=4$, 18 threshold-heavy | `dmotpy/scripts/standalone_validation_36.py` | Re-run verified | **VERIFIED** |
| **MARRMoT Reference Comparison** | Level B performance context: $\rho = 0.7079$, median paired $\Delta\text{KGE} = +0.0274$ | `/mnt/g/Dataset/.../full300_kge_model_summary.csv` | 36 models, 17,380 paired test basins | **VERIFIED** |

---

## 3. Section S2 Revalidated Evidence Ledger: Data & Configuration

| Item / Claim | Value / Setting | Current Repository Evidence Path | Identifier / Code Location | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Basin Domain** | 531 CAMELS-US basins | `project/benchmark/data/531sub_id.txt` | `src/data_selection.py:18-22` | **VERIFIED** |
| **Meteorological Forcing** | Daymet daily $P$, $T$, and Hargreaves PET | `project/benchmark/src/data_selection.py` | Line 35 | **VERIFIED** |
| **Streamflow Target Conversion** | $\text{ft}^3/\text{s} \to \text{mm/d}$ via `area_gages2` | `project/benchmark/src/data_selection.py`, `dmotpy/data_contract.py` | `data_selection.py:37-39`, `data_contract.py:78` | **VERIFIED** |
| **Observation Masking** | Finite masking `torch.isfinite(target)` | `dmotpy/data_contract.py`, `project/benchmark/src/objective.py` | `data_contract.py:53-59`, `objective.py:46-49` | **VERIFIED** |
| **Calibration / Training Period** | `1980-10-01` to `1995-09-30` (5,478 days, 15 years) | `project/benchmark/configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml` | Line 31, `data_contract.py:89` | **VERIFIED** |
| **Evaluation / Test Period** | `1995-10-01` to `2010-09-30` (5,479 days, 15 years) | `project/benchmark/configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml` | Line 33, `data_contract.py:91` | **VERIFIED** |
| **Evaluation Warm-up** | 365 days prepended (`1994-10-01` to `1995-09-30`), detached | `project/benchmark/scripts/canonical_v2/run_canonical_v2_model.py` | Lines 5, 60, 74 | **VERIFIED** |
| **DOY Calendar Forcing** | Day of year channel for `mopex4`, `mopex5`, `vic` | `dmotpy/data_contract.py` | `CALENDAR_MODELS` (lines 14, 30–50) | **VERIFIED** |
| **Penman Warm-up Exception** | 365-day warmup + 365-day scored window (total 730d) vs 730d+365d (1095d) | `project/benchmark/scripts/canonical_v2/run_canonical_v2_model.py` | Lines 7–9, 152–156 | **VERIFIED** |
| **Penman Warm-up Mode** | Detached autograd (`"detach"`); `"truncate:90"` was dead legacy string | `project/benchmark/PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md` | Lines 1–50 | **VERIFIED** |
| **35 CAMELS Physical Attributes** | 9 climate, 3 topo, 7 veg, 9 soil, 7 geol | `project/benchmark/dpl/attributes.py` | `CAMELS_35_ATTRIBUTES` (lines 12–29) | **VERIFIED** |
| **Skewed Attribute Transform** | $\ln(x + \text{shift})$ for cols 11 (`area_gages2`), 23 (`soil_conductivity`), 34 (`geol_permeability`) | `project/benchmark/dpl/attributes.py` | Lines 72–84 | **VERIFIED** |
| **Attribute Normalization** | Z-score standardization across training catchments | `project/benchmark/dpl/attributes.py` | Lines 86–89 | **VERIFIED** |
| **IC Optimizer & Precision** | Batched Active CMA-ES, FP64 precision | `project/benchmark/src/batched_cmaes.py` | Lines 33–160 | **VERIFIED** |
| **IC Budget & Restarts** | 10 independent starts, 300 generations (simhyd: 280), population $\lambda \in [8, 20]$ | `project/benchmark/configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml` | Lines 8–15 | **VERIFIED** |
| **IC Initialization & Transform** | LHS in logit latent space, $\sigma_0 = 0.10$, sigmoid mapping | `project/benchmark/src/batched_cmaes.py` | Lines 16–22 | **VERIFIED** |
| **IC Warmup Protocol** | 5-repeat 1-year warmup ($5 \times 365 = 1,825$ days), unpenalized | `project/benchmark/configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml` | Lines 19–25 | **VERIFIED** |
| **dPL Parameterizer Network** | MLP [256, 256], LayerNorm, GELU, Dropout(0.05) | `project/benchmark/dpl/nn_parameterizer.py` | Lines 55–125 | **VERIFIED** |
| **dPL Midpoint Initialization** | Zero weights/biases $\to$ initial normalized coordinate $u_0 = 0.5$ | `project/benchmark/scripts/canonical_v2/run_canonical_v2_model.py` | Lines 80–87, 236–237 | **VERIFIED** |
| **dPL Optimizer & Hyperparameters** | AdamW, $\text{lr}=10^{-3}$, weight decay $10^{-4}$, clip norm $1.0$, seed 42 | `project/benchmark/scripts/canonical_v2/run_canonical_v2_model.py` | Lines 157–159, 198–200, 290 | **VERIFIED** |
| **dPL Batching & Epochs** | $B=100$ basins, 169 steps/epoch, 100 max epochs, patience 10 on train loss | `project/benchmark/scripts/canonical_v2/run_canonical_v2_model.py` | Lines 160–164, 240–248, 372–379 | **VERIFIED** |
| **Parameter Mapping Census** | Exactly **53 log-mapped** (span ratio $\ge 100$) and **218 linear-mapped** out of 271 | `dmotpy/models/hydrology_model.py` | `_should_use_log_mapping` (lines 160–230) | **VERIFIED** |

---

## 4. Section S3 Revalidated Evidence Ledger: Parameter-Space, Information-Organization & OOB

| Item / Claim | Value / Definition | Current Repository Evidence Path | Identifier / Code Location | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Bound-Normalized Parameter Space** | $\tilde{\theta} \in [0, 1]^{P_m}$ via linear/log bounds | `project/benchmark/manuscript/r2/scripts/r2_common.py` | Lines 48–54, 88–92 | **VERIFIED** |
| **Parameter Displacement ($D_{\text{RMS}}$)** | Root-mean-square coordinate distance $D_{\text{RMS}}(m, b)$; grand median $\mathbf{0.384}$ | `project/benchmark/manuscript/r2/scripts/01_parameter_separation_icself.py` | Lines 13–24 | **VERIFIED** |
| **IC-Self Reference & Restarts** | Multi-start archive (10 starts), $\epsilon = 0.01$ KGE tolerance gate | `project/benchmark/manuscript/r2/tables/F2_ICSELF_AUDIT_REPORT.md` | Lines 9–98 | **VERIFIED** |
| **Primary IC-Self Models** | 23 eligible models ($\ge 90\%$ basin coverage); 13 insufficient reference models | `project/benchmark/manuscript/r2/tables/F2_CR_CURRENT23_MODEL_LIST.csv` | Full table | **VERIFIED** |
| **Paired Separation Excess** | Model-equal median paired excess $\mathbf{+0.2188}$ (95% CI: $[0.2091, 0.2281]$, $36/36 > 0$) | `project/benchmark/manuscript/r4/final_freeze_verification/R4_R2_EXCESS_AGGREGATION_VERIFICATION.md` | Lines 1–55 | **VERIFIED** |
| **Coordinate Localization ($C_{\text{eff}}$)** | Participation ratio $C_{\text{eff}} = 0.3458$ (raw 36m) / $0.2934$ (strict 23m), Top-1 share $0.6726$ | `project/benchmark/manuscript/r2/scripts/03_coordinate_localization_icself.py` | Lines 18–60 | **VERIFIED** |
| **Parameter-Rank Reorganization ($R_{\text{rank}}$)** | Within-parameter cross-catchment Spearman rank correlation; descriptive median $0.4069$ | `project/benchmark/manuscript/r2/scripts/02_rank_reorganization.py` | Lines 12–39 | **VERIFIED** |
| **Contraction Ratio ($CR$) Sensitivity** | $CR = \text{IQR}(\text{dPL}) / \text{IQR}(\text{IC})$; canonical $0.6140$ vs consensus $0.9795$ vs self-ref $1.0043$ | `project/benchmark/manuscript/r2/scripts/04_distribution_reference_sensitivity.py` | Lines 12–42 | **VERIFIED** |
| **Performance–Parameter Bridge** | Spearman $\rho_b(|\Delta\text{KGE}|, D_{\theta}) = \mathbf{+0.2412}$ (95% CI: $[0.1973, 0.2586]$, $34/36 > 0$) | `project/benchmark/manuscript/r2/scripts/05_outlet_parameter_bridge.py` | Lines 14–31 | **VERIFIED** |
| **35 Attribute Clustering (Seen)** | Average-linkage on Spearman distance $1 - |\rho|$ at threshold cut $0.30$ ($|\rho| \ge 0.70$) | `project/benchmark/manuscript/r3/scripts/00_build_relationship_matrices.py` | Lines 35–52 | **VERIFIED** |
| **20 Seen Information Dimensions** | 20 frozen clusters/dimensions with PC1 scores capturing 52.2%–95.1% intra-cluster variance | `project/benchmark/results/joh_reorganization_diagnostic_20260905/tables/information_dimension_registry.csv` | Lines 1–21 | **VERIFIED** |
| **Primary Population Cells** | Exactly **5,420 primary cells** ($271 \text{ parameters} \times 20 \text{ dimensions}$) | `project/benchmark/manuscript/r3/agent_A_primary_correspondence.md` | Lines 14–25 | **VERIFIED** |
| **IC-Stable Cell Denominator** | Exactly **902 / 5,420 cells** ($16.64\%$) matching $|\rho_{\text{IC}}| \ge 0.20 \land P(\text{sign}) \ge 0.95$ | `project/benchmark/manuscript/r3/scripts/02_sign_agreement_audit.py` | Lines 30–65 | **VERIFIED** |
| **Sign Retention Rate** | $\mathbf{849 / 902 = 94.12\%}$ ($94.124\%$) on 902 IC-stable cells | `project/benchmark/manuscript/r3/tables/R3_IC_TO_DPL_MAGNITUDE_RETENTION_SUMMARY.csv` | Full table | **VERIFIED** |
| **Magnitude Retention Rates** | Same sign $+ |\rho_{\text{dPL}}| \ge 0.10$: $87.69\%$; $\ge 0.20$: $76.72\%$; $\ge 0.30$: $57.32\%$ | `project/benchmark/manuscript/r3/tables/R3_IC_TO_DPL_MAGNITUDE_RETENTION_SUMMARY.csv` | Full table | **VERIFIED** |
| **4-Way Relationship Classification** | `IC-supported` (33.32%), `IC-opposite-sign`, `IC-weak`, `IC-absent` (`dPL-emergent` = 66.68%) | `project/benchmark/manuscript/r3/scripts/06_dpl_construction_artifact_audit.py` | Lines 26–55 | **VERIFIED** |
| **Paired Profile Correspondence** | Model-equal median Spearman profile similarity $R_{\text{paired}} = \mathbf{0.7158}$ ($\approx 0.716$) | `project/benchmark/manuscript/r3/scripts/01_parameter_profile_reproducibility.py` | Lines 75–95 | **VERIFIED** |
| **Same-Coordinate Diagonal Advantage** | Diagonal median $0.7158$ vs off-diagonal $-0.0218 \to A_{\text{diag}} = \mathbf{0.6150}$ ($35/35$ models $>0$) | `project/benchmark/manuscript/r3/scripts/08_parameter_identity_correspondence.py` | Lines 95–185 | **VERIFIED** |
| **1,000-Permutation Null** | Within-model permutation null mean $0.0015$, 95% interval $[-0.1233, 0.1128]$, $p = \mathbf{0.000999}$ | `project/benchmark/manuscript/r3/scripts/08_parameter_identity_correspondence.py` | Lines 120–185 | **VERIFIED** |
| **Functional-Role Negative Control** | Role advantage $A_{\text{role}} = -0.1188$ ($p=0.8057$); HESS flexible role contrast $D_{\rho}=-0.0003$ ($p=0.8498$) | `project/benchmark/manuscript/r3/agent_C_functional_role_negative.md` | Lines 14–85 | **VERIFIED** |
| **OOB Primary 8 Models** | `alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope` | `project/benchmark/configs/oob_primary8_5fold_20260902.yaml` | Lines 4–12 | **VERIFIED** |
| **OOB Pre-Selection Rule** | 4-quadrant stratification ($G_{\text{seen}} \times R_{\text{seen}}$) + viability gate + 5D dispersion + anti-redundancy | `project/benchmark/manuscript/r4/model_selection_audit/ORIGINAL_SELECTION_RULE_AUDIT.md` | Lines 9–105 | **VERIFIED** |
| **OOB 5-Fold Cross Validation** | Deterministic 5-fold split ($107, 106, 106, 106, 106$), random state 20260902 | `project/benchmark/scripts/oob/oob_common.py` | Lines 164–197 | **VERIFIED** |
| **OOB Formal Jobs Count** | Exactly **40 formal jobs** ($8 \text{ models} \times 5 \text{ folds} = 40$) | `project/benchmark/manuscript/r4/R4_OOB_PROVENANCE_AND_COMPLETENESS_AUDIT.md` | Lines 1–40 | **VERIFIED** |
| **OOB 13 Continuous Clusters** | 32 continuous attributes (3 categorical excluded) partitioned into 13 orthogonal clusters at $|\rho| \ge 0.70$ | `project/benchmark/manuscript/r4/final_definition_alignment_audit/R4_R3_DIMENSION_ALIGNMENT_AUDIT.md` | Lines 10–55 | **VERIFIED** |
| **OOB Profile Correspondence Replay** | $R_{\text{paired, OOB}} = \mathbf{0.7390}$ (13 clusters) / $\mathbf{0.7620}$ (32 attributes); $A_{\text{diag, OOB}} = \mathbf{0.7184}$ ($p=0.000999$) | `project/benchmark/manuscript/r4/r123_support_audit/AGENT_C_R3_OOB_SUPPORT.md` | Lines 35–95 | **VERIFIED** |
| **OOB Parameter Displacement Replay** | $D_{\text{RMS, OOB}} = \mathbf{0.3429}$; Excess displacement $+0.2270$ (Case 1) / $+0.2121$ (Case 2) in $5/5$ models | `project/benchmark/manuscript/r4/r123_support_audit/AGENT_B_R2_OOB_SUPPORT.md` | Lines 35–85 | **VERIFIED** |
| **Seen $\leftrightarrow$ OOB dPL Retention** | Overall $\rho = \mathbf{0.9616}$, sign retention $99.93\%$, cluster retention $96.44\%$ across all cells | `project/benchmark/manuscript/r4/scripts/r4_compute_statistics.py` | Lines 250–310 | **VERIFIED** |
