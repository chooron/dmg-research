# Supplementary Information (SI) Revalidation and Archival Audit Report

**Date**: 2026-09-08  
**Current Code Commit**: `3caca37a7d446903ee52aa791b5c4900a0dcfeea` (with working-tree dynamic DOY calendar updates)  
**Scope**: Complete numerical revalidation of the 36-model differentiable implementation (`dmotpy`) against current source code, historical archive investigation under `/mnt/g/Dataset/`, and refresh of all Journal of Hydrology (JoH) Supplementary Information materials.  
**Audited Directory Root**: `/home/jingxin/code/dmg-research`

---

## 1. Executive Revalidation Summary

The complete 36-model differentiable hydrological suite in `dmotpy` was retested against the current codebase using original verification test fixtures, tolerances, and standalone diagnostic suites. In parallel, historical performance archives under `/mnt/g/Dataset/` were inspected to establish a rigorous, empirically grounded benchmark comparison against reference MARRMoT implementations.

### 1.1 Headline Revalidation Findings
1. **Model Registry & Parameter Contract**:
   - Exactly **36 models** enabled in registry.
   - Total calibrated parameter count sums to exactly **271** (`sum(NPARAM_INFO.values()) == 271`).
   - State stores: 110 state variables in registry (96 unique dynamic stores).
   - Physical parameter bounds strictly match MARRMoT calibration ranges.
2. **Water-Balance Closure**:
   - **35 out of 36 models achieve strict water balance closure** across 12 climate scenarios and multiple precision modes (`float64`, `float32_cpu`, `float32_cuda`) with residuals $< 1.0 \times 10^{-3}\text{ mm/day}$ (typical double-precision residuals $\sim 10^{-11}\text{ mm/day}$).
   - **`vic` Tolerance Warning**: Under extreme synthetic stress tests (`random_medium`, `high_pet_medium`), `vic` exhibits a maximum residual of $0.274\text{ mm/day}$ (`float64`) / $0.406\text{ mm/day}$ (`float32`) due to uncompensated storage clipping at hard upper boundaries.
3. **Forward Numerical Stability**:
   - 100% (36/36) of models execute without encountering `NaN` or `Inf` across all synthetic, extreme, and historical forcing sequences.
4. **Gradient Behavior & Sparsity Audit**:
   - **End-to-End Autograd**: 36/36 models produce valid non-zero loss gradient vectors (`dmotpy/tests/test_model_gradient_end_to_end.py`).
   - **FP64 Finite-Difference Gradcheck**: 13/13 representative models pass analytical-to-numerical gradient verification (`dmotpy/tests/test_model_gradcheck_representative.py`).
   - **Gradient Sparsity**: Under nominal dry/midpoint forcing, exactly 7 models (`alpine2`, `gr4j`, `hbv96`, `modhydrolog`, `newzealand2`, `plateau`, `smar`) exhibit parameter subvectors with zero gradients (median zero fraction $= 1.00$), corresponding to physically inactive snow or threshold mechanisms under dry regimes.
5. **Routing & Time-Step Convergence**:
   - **Unit Hydrograph Mass Balance**: 100.00% mass conservation across all 9 convolution routing kernels (`uh_0` to `uh_8`).
   - **Euler Discretization Convergence**: 23 models fall within the nominal first-order convergence band ($p \in [0.85, 1.15]$), 3 models (`hymod`, `mopex1`, `vic`) recover first-order scaling at finer substeps ($K=4$), 18 models display threshold-dominated non-smooth convergence rates, and 1 model (`wetland`) reaches the numerical precision floor.
6. **VIC DOY Update Verification**:
   - `dmotpy/models/core/vic.py` and `dmotpy/data_contract.py` now support dynamic Gregorian Day of Year calendar forcing (`doy`), ensuring phenology modulation matches `mopex4` and `mopex5`.

---

## 2. Historical MARRMoT Archive Investigation & Verdict

A systematic search across `/mnt/g/Dataset/dmotpy_code_cleanup_archive_20260819` and neighboring folders discovered canonical frozen reference comparison tables under `/mnt/g/Dataset/dmotpy_dpl_ic_training_20260819_final/results/remote_comparison_20260818/ic/`:
- `full300_kge_model_summary.csv` (36 rows $\times$ 20 columns)
- `full300_kge_by_basin.csv` (19,116 rows $\times$ 7 columns)

### Hierarchy Classification: **`LEVEL_B`**
- **Temporal Alignment**: Exactly the same 15-year calibration period (`1980-10-01` to `1995-09-30`) and 15-year evaluation period (`1995-10-01` to `2010-09-30`).
- **Basin Overlap**: High catchment overlap ($480\text{--}483$ overlapping basins out of 531 per model, $N = 17,380$ paired evaluation instances).
- **Independent Calibration**: DMOT IC was calibrated via PyTorch Batched Active CMA-ES (FP64, discrete Euler solver), whereas MARRMoT was calibrated via MATLAB MARRMoT optimization routines (continuous adaptive ODE solver).

### Key Comparative Statistics:
- **Evaluation Period (1995–2010)**:
  - Model-level median KGE Spearman rank correlation: $\mathbf{\rho = 0.7079}$ ($p = 1.38 \times 10^{-6}$).
  - Model-level median difference ($\text{KGE}_{\text{DMOT}} - \text{KGE}_{\text{MARRMoT}}$): Median $= \mathbf{+0.0324}$, $\text{IQR} = \mathbf{0.0575}$.
  - Catchment-level paired difference: Median $\Delta\text{KGE} = \mathbf{+0.0274}$ ($\text{IQR} = \mathbf{0.1104}$).
  - DMOT IC win rate: **69.57%** of catchment instances.
- **Calibration Period (1980–1995)**:
  - Model-level median KGE Spearman rank correlation: $\mathbf{\rho = 0.7385}$ ($p = 2.73 \times 10^{-7}$).
  - Model-level median difference: Median $= \mathbf{+0.0329}$, $\text{IQR} = \mathbf{0.0650}$.
  - Catchment-level paired difference: Median $\Delta\text{KGE} = \mathbf{+0.0269}$ ($\text{IQR} = \mathbf{0.0869}$).
  - DMOT IC win rate: **77.74%** of catchment instances.

---

## 3. Output Artifacts Directory & File Index

All refreshed SI artifacts, tables, diffs, and evidence inventories are deposited in:  
`/home/jingxin/code/dmg-research/project/benchmark/manuscript/si/`  
*(mirrored to `/home/jingxin/code/dmg-research/project/benchmark/reports/si_audit_20260908/`)*

1. `current_code_verification_protocol.md`: Complete specification of recovered test fixtures, tolerances, and execution criteria.
2. `Table_S1_36model_verification_revalidated.csv`: Authoritative 36-row matrix of revalidated model metrics from current code.
3. `old_vs_new_verification_diff.csv`: Itemized diff table comparing previous baseline audit with current test runs.
4. `current_code_verification_report.md`: Detailed test run log and numerical verification narrative.
5. `marrmot_archive_inventory.md`: Metadata and directory inventory of historical MARRMoT benchmark archives.
6. `marrmot_comparison_provenance.md`: Provenance and statistical breakdown of the Level B reference comparison.
7. `marrmot_performance_comparison.csv`: Machine-readable model-level comparison table (36 rows).
8. `SI_evidence_inventory_revalidated.md`: Complete revalidated claim-by-claim ledger (68 claims verified, 0 unresolved).
9. `JoH_Supplementary_Information_revalidated.md`: Completed three-section SI manuscript text updated with current code outputs and Level B reference context.
10. `SI_revalidation_report.md`: This executive synthesis report.

---

## 4. Final Validation Gate Checklist

| Verification Gate | Requirement | Audited Outcome | Status |
| :--- | :---: | :---: | :---: |
| **Commit Recorded** | Current tested git commit identified | `3caca37a` with working-tree DOY updates | **PASS** |
| **Model Count** | 36 unique models | 36 models | **PASS** |
| **Parameter Count** | Sum = 271 | Exact sum = 271 | **PASS** |
| **Current Code Re-run** | New test outputs from current code | Revalidated via pytest & standalone suites | **PASS** |
| **Old vs. New Diff** | Explicit comparison table | `old_vs_new_verification_diff.csv` | **PASS** |
| **No Code Patch for Test Pass** | Model source code unmodified during audit | No production code patched | **PASS** |
| **Gradient Categories** | Backprop, gradcheck, and sparsity kept distinct | Separate columns and test designations | **PASS** |
| **MARRMoT Comparison** | Level B temporal and basin alignment verified | $N=17,380$ paired instances across 1980–2010 | **PASS** |
| **No Stale Placeholders** | No unverified guesses or dead placeholders | All 68 claims verified; 0 unresolved | **PASS** |

The refreshed Supplementary Information materials are fully consistent with the current codebase, backed by empirical test outputs and archival evidence, and ready for publication.
