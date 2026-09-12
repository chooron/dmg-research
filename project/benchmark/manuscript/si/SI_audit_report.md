# Supplementary Information (SI) Comprehensive Audit Report

**Date**: 2026-09-08  
**Scope**: Multi-agent repository audit of `dmotpy` and `project/benchmark` to verify and finalize the Journal of Hydrology (JoH) Supplementary Information (SI) for the 36-model IC vs. dPL study.  
**Audited Directory Root**: `/home/jingxin/code/dmg-research`

---

## 1. Executive Audit Summary

A rigorous, multi-agent audit was conducted across all source code, model definitions, optimization logs, neural network parameterizers, and statistical analysis pipelines. All headline figures, parameter counts, attribute mappings, configuration tables, and statistical estimands have been verified against canonical frozen manifests and production execution scripts.

### Headline Verification Status
- **36 MARRMoT-Derived Differentiable Models**: Verified. All 36 models appear in registry and Table S1.
- **271 Calibrated Parameters**: Verified. Parameter counts per model sum exactly to 271 (`sum(NPARAM_INFO.values()) == 271`).
- **531 CAMELS-US Basins**: Verified (`data/531sub_id.txt`).
- **Data Partitions**: Calibration period `1980-10-01` to `1995-09-30` (5,478 days); Evaluation period `1995-10-01` to `2010-09-30` (5,479 days); 365-day detached evaluation warm-up.
- **Catchment Attributes & Representation**: 35 total CAMELS-US physical attributes (32 continuous, 3 discrete categorical). Seen-basin information space clusters the 35 attributes into 20 orthogonal dimensions. Out-of-Bag (OOB) information space partitions the 32 continuous attributes into 13 continuous clusters.
- **Population & IC-Stable Subset**: Complete information space comprises exactly $271 \times 20 = \mathbf{5,420\text{ primary cells}}$. The IC-stable denominator ($|\rho_{\text{IC}}| \ge 0.20 \land P(\text{sign}) \ge 0.95$) is confirmed at exactly **902 cells** ($16.64\%$).
- **Sign & Magnitude Retention**: dPL preserves the identical correlation sign on **849 of 902 IC-stable cells** ($\mathbf{94.12\%}$).
- **Parameter Bound Mapping**: Exactly **53 parameters** use logarithmic mapping (span ratio $\ge 100$) and **218 parameters** use linear mapping out of 271.
- **Out-of-Bag (OOB) Protocol**: 8 prespecified models (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope`), deterministic 5-fold cross-validation, exactly 40 completed formal jobs, every basin held out once per model.

---

## 2. Deliverable Artifact Inventory

All required SI tables, evidence inventories, and the completed draft have been generated and deposited under `/home/jingxin/code/dmg-research/project/benchmark/manuscript/si/`:

1. `SI_evidence_inventory.md`: Complete audit ledger for all 68 individual claims across Sections S1–S3.
2. `Table_S1_36model_verification.csv`: Authoritative 36-row matrix of model registry IDs, parameters, states, water balance closure, forward stability, gradient audit coverage, and Euler convergence.
3. `Table_S2_attribute_clusters.csv`: Complete crosswalk of all 35 CAMELS attributes, seen-basin 20-D cluster assignments, and OOB 13-cluster assignments.
4. `Table_S3_training_configuration.csv`: Canonical configuration comparison across Individual Calibration (IC), Seen-Basin dPL, and Held-Out (OOB) dPL.
5. `Table_S4_estimand_audit.csv`: Comprehensive registry of all statistical estimands, populations, aggregation hierarchies, and OOB replay status.
6. `JoH_Supplementary_Information_completed.md`: Fully completed three-section manuscript draft with all verified placeholders replaced and unresolved items clearly demarcated.
7. `SI_audit_report.md`: This comprehensive synthesis report.

---

## 3. Discrepancy, Conflict, and Provenance Analysis

### 3.1 Unresolved Placeholder: External MARRMoT Reference Cross-Validation
- **Status**: `[[UNRESOLVED: External MARRMoT cross-validation unexecuted on matched CAMELS-531 setup]]`
- **Audit Findings**: A search across `dmotpy` and `project/benchmark` confirmed that no genuinely matched cross-validation against external reference suites (MATLAB MARRMoT or `pymarrmot`) exists:
  1. `pymarrmot` contains critical code defects where unit hydrographs are stubbed to identity routing, invalidating hydrograph timing comparisons. Paired Two One-Sided Tests (TOST) were explicitly aborted in the test suite.
  2. MATLAB MARRMoT direct parameter injection into differentiable models results in systematic drift across 35/35 models due to differences between continuous adaptive ODE solvers and discrete Euler time-stepping with smooth sigmoidal transitions.
  3. No paired 531-basin simulation runs with identical parameter sets and solvers were ever generated.
- **Action Taken**: In strict accordance with study constraints, no fabricated `Figure_S1_reference_consistency.csv` was manufactured. The status is reported factually, and the SI placeholder is left explicitly marked. Manuscript Figure S1 is confirmed as the raw $36 \times 531$ $\Delta\text{KGE}$ model-basin matrix (`R1_model_basin_delta_kge.csv`).

### 3.2 Penman Warm-up Mode Provenance
- **Historical Note**: Early exploratory notes mentioned a `"truncate:90"` warm-up mode for the Penman model.
- **Forensic Audit**: A dedicated audit (`PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md`) proved that `"truncate:90"` was a dead, unread string from legacy prototypes. The canonical v2 training script strictly validates and executes standard detached warm-up (`"detach"`) with a 365-day warm-up window for `penman`.
- **Verdict**: Canonical Table S3 and SI text report detached autograd (`"detach"`) with a 365-day warm-up horizon.

### 3.3 Apparent Parameter Space Contraction ($CR$)
- **Audit Findings**: The parameter distribution contraction ratio $\text{CR} = \text{IQR}(\text{dPL}) / \text{IQR}(\text{IC})$ is strongly reference-dependent. Comparing dPL against a single arbitrary IC seed yields $\text{CR}_{\text{canonical}} = 0.6140$ (apparent contraction). However, comparing against consensus IC multi-starts yields $\text{CR}_{\text{consensus}} = 0.9795$, and comparing against synthetic IC-self realizations yields $\text{CR}_{\text{selfref}} = 1.0043$.
- **Verdict**: Apparent parameter variance contraction is an artifact of single-seed IC baseline selection rather than an intrinsic property of dPL. The SI text explicitly documents this finding.

### 3.4 Gradient Sparsity vs. Gradient Correctness
- **Audit Findings**: While 36/36 models pass end-to-end autograd loss backpropagation and 13/13 representative models pass FP64 finite-difference gradchecks, 7 models (`alpine2`, `gr4j`, `hbv96`, `modhydrolog`, `newzealand2`, `plateau`, `smar`) have parameter subsets that evaluate to zero gradients under nominal dry/midpoint forcing conditions due to inactive physical processes (e.g., snowmelt degree-day factors in warm catchments).
- **Verdict**: Table S1 accurately classifies these 7 models with `SPARSE_GRAD` status while documenting full autograd test passage.

---

## 4. Final Manuscript Consistency Checklist

| Verification Check | Target Requirement | Audited Outcome | Verification Result |
| :--- | :---: | :---: | :---: |
| 36 Models in Table S1 | Exactly 36 unique models | 36 unique rows | **PASS** |
| Parameter Count Sum | Exactly 271 parameters | Sum = 271 | **PASS** |
| Basin Count | Exactly 531 basins | 531 unique basins | **PASS** |
| Seen Attributes | 35 CAMELS-US attributes | 35 attributes (9 climate, 3 topo, 7 veg, 9 soil, 7 geol) | **PASS** |
| Seen Information Dimensions | 20 orthogonal dimensions | 20 clusters (average-linkage $|\rho| \ge 0.70$) | **PASS** |
| OOB Continuous Attributes | 32 continuous attributes | 32 continuous attributes (3 categorical excluded) | **PASS** |
| OOB Information Clusters | 13 continuous clusters | 13 clusters ($|\rho| \ge 0.70$) | **PASS** |
| Full Information Space | 5,420 cells | $271 \times 20 = 5,420$ cells | **PASS** |
| IC-Stable Denominator | Exactly 902 cells | 902 cells ($|\rho_{\text{IC}}| \ge 0.20 \land P \ge 0.95$) | **PASS** |
| Linear vs. Log Parameter Count | 53 log / 218 linear | 53 log / 218 linear (Auto log threshold: ratio $\ge 100$) | **PASS** |
| OOB Models & Folds | 8 models, 5 folds | 8 models, 5 folds ($107, 106, 106, 106, 106$) | **PASS** |
| OOB Formal Jobs | Exactly 40 jobs | 40 formal runs completed | **PASS** |
| Remaining Placeholders in SI | No unresolved guess | Exactly 1 unresolved placeholder clearly marked | **PASS** |

The Supplementary Information package is complete, authoritative, and ready for manuscript-level integration.
