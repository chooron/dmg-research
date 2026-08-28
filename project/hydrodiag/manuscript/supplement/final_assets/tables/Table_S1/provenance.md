# Provenance

Final asset: `manuscript/supplement/final_assets/tables/Table_S1/Table_S1.csv` and `Table_S1.md`
Generated: 2026-08-27
Git hash: `322fb932c922a131a67800bcbc6aa7eb704c7605`

## Inputs

- `models/parameter_specs.py`
- `manuscript/results/R2/authoritative_15_parameter_specs.csv`
- `manuscript/methods_supplement_production_audit.md`

## Columns / keys

`Host`, `Parameter`, `Symbol`, `Definition`, `Lower bound`, `Upper bound`, `Unit`, `Base membership`, `TGD membership`, `CN membership`, `Calibrated / fixed`, `Default / fixed value`, `Process`.

## Filters

No result rows were filtered. Host specification dictionaries were read in production order. Fixed TGD and CN constants were appended as traceability rows.

## Transformations

Production identifiers were mapped to manuscript display symbols. Structural membership was derived from the Base, TGD-module, and CN-module specification dictionaries. No scientific values were recomputed.

## Statistical operations

None. This is a deterministic specification extraction.

## Plot script

`manuscript/supplement/final_assets/tables/Table_S1/build_Table_S1.py`

## Command

`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/tables/Table_S1/build_Table_S1.py`

## Output

47 rows: 41 calibrated host/module parameter rows and 6 fixed-constant rows. Image output is not applicable.

## Image size / checksum

Not applicable to this table. `Table_S1.csv` SHA-256: `0eb4f4454a3cde9ffcf773ec5c6d35301a580f89e429f4da6fb48ba9eebcd404`.

## Known caveats

The production specification module is authoritative for bounds and membership. Historical table files under `manuscript/archive/tables_si_legacy/` are not used. TGD `T_ref`/`s_T` and CN initialization/threshold constants are fixed and must not be counted as calibrated parameters.
