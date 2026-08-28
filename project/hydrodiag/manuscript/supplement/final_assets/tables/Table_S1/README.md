# Table S1 — Calibrated parameter definitions, bounds, units, and structural membership

## 1. Scientific role

Table S1 is the reproducibility parameter reference for the model structures used across R1–R5. It belongs in the SI because it documents the complete parameter contract rather than a result estimate.

## 2. What is shown

The CSV lists production parameter identifiers, manuscript symbols, definitions, bounds, units, and membership in the Base, TGD, and CN structures for XAJ, GR4J, and SIMHYD. Fixed TGD gate constants and fixed CN initialization/threshold constants are listed separately and are not counted as calibrated parameters.

## 3. Source data

- `models/parameter_specs.py`
- `manuscript/results/R2/authoritative_15_parameter_specs.csv`
- `manuscript/methods_supplement_production_audit.md`

## 4. Sample definition

There is no catchment sample: this is a parameter-definition table. Production calibrated counts are XAJ Base/TGD/CN = 15/17/17, GR4J = 4/6/6, and SIMHYD = 10/12/12. TGD `T_ref`, `s_T`, and `epsilon` are fixed constants; CN `g_thresh`, `G_0`, and `eTG_0` are fixed implementation constants.

## 5. Metric definitions

Bounds are the lower and upper values in the production parameter specifications. Structural membership means that the parameter is present in that host/structure parameter vector. The TGD module uses `tau_warm` and `Delta tau_cold`; the fixed gate is `T_ref = 0 °C`, `s_T = 2 °C`.

## 6. Aggregation and uncertainty

No statistical aggregation, bootstrap, interval, seed aggregation, or resampling is used. The table is a deterministic extraction of production specifications.

## 7. Generation method

- Script: `manuscript/supplement/final_assets/tables/Table_S1/build_Table_S1.py`
- Command: `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/tables/Table_S1/build_Table_S1.py`
- Input: production specification dictionaries only.
- Outputs: `Table_S1.csv` and `Table_S1.md`.
- **NO MODEL TRAINING. NO RECALIBRATION. NO FULL TEST PIPELINE.**

## 8. Visual encoding

The Markdown version is a plain table. Symbols follow `_audit/symbol_registry.md`; code identifiers and manuscript symbols are both retained.

## 9. Caption-ready factual statements

- Table S1 contains 47 rows: calibrated parameter rows plus fixed constants.
- Calibrated counts are XAJ 15/17/17, GR4J 4/6/6, and SIMHYD 10/12/12 for Base/TGD/CN.
- TGD gate and CN initialization/threshold constants are explicitly marked fixed.

## 10. Interpretation boundary

Table S1 defines the parameterization; it does not demonstrate calibration quality, identifiability, physical truth, or parameter importance.

## 11. Validation

The final CSV was checked against the production dictionaries. Membership counts exactly match the frozen 15/17/17, 4/6/6, and 10/12/12 contracts. No canonical result file was modified.
