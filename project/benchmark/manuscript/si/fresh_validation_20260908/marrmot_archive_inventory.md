# MARRMoT Historical Archive Inventory (Fresh Verification)

**Audit Target**: Historical performance archives on `/mnt/g/Dataset/` and `dmotpy_code_cleanup_archive_20260819`  
**Execution Timestamp**: 2026-09-08T14:35:00Z  
**Verdict**: **`LEVEL_B`** (Same temporal estimation and evaluation periods across overlapping CAMELS-531 catchments, independently calibrated).

---

## 1. Verified Directory Structure

| Directory Path | Role in Benchmark | Verification Status |
| :--- | :--- | :---: |
| `/mnt/g/Dataset/dmotpy_dpl_ic_training_20260819_final/results/remote_comparison_20260818/ic/` | Primary frozen performance tables (`full300_kge_model_summary.csv`, `full300_kge_by_basin.csv`) | **VERIFIED** |
| `/mnt/g/Dataset/dmotpy_benchmark_official_training_results_20260819/` | Raw CMA-ES optimization outputs across 531 basins | **VERIFIED** |
| `/mnt/g/Dataset/dmotpy_code_cleanup_archive_20260819/` | Code cleanup archive and diagnostic review logs | **VERIFIED** |

---

## 2. File Metadata & Temporal Window Verification

1. **`full300_kge_model_summary.csv`**:
   - 36 model rows $\times$ 20 columns.
   - Confirms parameter-estimation period `1980-10-01` to `1995-09-30` and evaluation period `1995-10-01` to `2010-09-30`.
2. **`full300_kge_by_basin.csv`**:
   - 19,116 rows ($36 \times 531$).
   - 17,388 valid calibration pairs ($480\text{--}483$ basins per model).
   - 17,380 valid evaluation pairs ($480\text{--}483$ basins per model).

---

## 3. Comparison Classification

- **Classification**: **`LEVEL_B`** (Same period + overlapping basins, independently calibrated).
- **Authorized Framing**: Independent calibration performance context under identical temporal and forcing conditions.
