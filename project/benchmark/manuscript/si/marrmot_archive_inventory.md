# MARRMoT Historical Archive Inventory

**Audit Target**: Historical performance archives on `/mnt/g/Dataset/` and `dmotpy_code_cleanup_archive_20260819`  
**Audit Date**: 2026-09-08  
**Verdict**: **`LEVEL_B`** (Same temporal estimation and evaluation periods across overlapping CAMELS-531 catchments, independently calibrated).

---

## 1. Archive Search Summary

A systematic search across `/mnt/g/Dataset/` and related directories was conducted to locate historical MARRMoT, pymarrmot, and MATLAB MARRMoT benchmark performance records.

### 1.1 Directories Inspected

| Directory Path | Description / Contents | Relevance to Study |
| :--- | :--- | :--- |
| `/mnt/g/Dataset/dmotpy_dpl_ic_training_20260819_final/` | Official frozen 36-model IC and dPL training benchmark results | **Primary Source**: Contains paired basin-level and model-level MARRMoT comparison tables. |
| `/mnt/g/Dataset/dmotpy_benchmark_official_training_results_20260819/` | Raw CMA-ES optimization outputs across 531 basins | Confirms FP64 10-start 300-generation solver execution. |
| `/mnt/g/Dataset/dmotpy_code_cleanup_archive_20260819/` | Code cleanup archive and diagnostic review logs (`review/flex-gap.md`) | Documents optimizer settings, capacity bounds, and solver difference boundaries. |
| `/mnt/g/Dataset/camels_data/` & `/mnt/g/Dataset/CAMELS_US/` | Static catchment attributes and forcing files | Supporting raw data archives. |

---

## 2. Identified Benchmark Files

### 2.1 Primary Comparison Tables

1. **Model-Level Summary Table**:
   - **Path**: `/mnt/g/Dataset/dmotpy_dpl_ic_training_20260819_final/results/remote_comparison_20260818/ic/full300_kge_model_summary.csv`
   - **Dimensions**: 36 rows $\times$ 20 columns.
   - **Content**: Aggregated calibration and evaluation period KGE medians for DMOT IC Full300 and baseline MARRMoT calibrations, paired median differences, win fractions, and checkpoint parity metrics across all 36 models.
   - **Temporal Contract**:
     - Parameter estimation / Training: `1980-10-01` to `1995-09-30` (5,478 days + 5-yr warmup)
     - Evaluation / Testing: `1995-10-01` to `2010-09-30` (5,479 days)

2. **Basin-Level Paired Table**:
   - **Path**: `/mnt/g/Dataset/dmotpy_dpl_ic_training_20260819_final/results/remote_comparison_20260818/ic/full300_kge_by_basin.csv`
   - **Dimensions**: 19,116 rows ($36 \text{ models} \times 531 \text{ basins}$) $\times$ 7 columns.
   - **Columns**: `model`, `basin_id`, `selected_checkpoint_train_kge`, `train_kge`, `test_kge`, `marrmot_train_kge`, `marrmot_test_kge`.
   - **Valid Overlapping Pairs**:
     - Calibration period: 17,388 valid model–basin pairs (480 to 483 basins per model).
     - Evaluation period: 17,380 valid model–basin pairs (480 to 483 basins per model).

3. **Diagnostic Comparison Ledger**:
   - **Path**: `/mnt/g/Dataset/dmotpy_dpl_ic_training_20260819_final/results/flex_ic_dpl_stability_20260818/basin_kge_gap_available.csv`
   - **Dimensions**: 1,594 rows $\times$ 6 columns.
   - **Content**: Multi-model cross-validation records for FLEX and HBV structural variants.

---

## 3. Comparison Hierarchy Classification

- **Level A (Fully Matched)**: *NOT SATISFIED*. While temporal windows and forcing are aligned, the reference MARRMoT calibrations were performed independently using MATLAB MARRMoT optimization routines (continuous adaptive ODE solver), whereas DMOT IC was calibrated using PyTorch Batched Active CMA-ES (discrete Euler solver).
- **Level B (Same Period + Overlapping Basins, Independently Calibrated)**: **`SATISFIED`**.
  - Exactly the same 15-year calibration period (`1980-10-01` to `1995-09-30`) and 15-year evaluation period (`1995-10-01` to `2010-09-30`).
  - High catchment overlap ($480\text{--}483$ overlapping basins out of 531 per model, $N = 17,380$ paired evaluation instances).
  - Evaluated on the complete 36-model portfolio.
- **Level C (Different Basin Samples)**: Superseded by Level B.
- **Level D (No Usable Archive)**: Superseded by Level B.
