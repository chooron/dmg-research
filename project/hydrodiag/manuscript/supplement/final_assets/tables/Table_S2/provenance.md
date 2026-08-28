# Provenance

Final asset: `manuscript/supplement/final_assets/tables/Table_S2/Table_S2.md`, `Table_S2_panelA.csv`, and `Table_S2_panelB.csv`
Generated: 2026-08-27
Git hash: `322fb932c922a131a67800bcbc6aa7eb704c7605`

## Inputs

- `manuscript/cache/r1_rebuild_audit_staged/r1_basin_level_ct.csv`
- `manuscript/results/discussion_audit/r3_denominator_sensitivity_audit.csv`
- `manuscript/scripts/shared/generate_table_s2_sensitivity.py` for comparison with the existing canonical table generator

## Columns / keys

Panel A: `Regime`, `Denominator type`, `Denominator definition`, `KGE threshold`, `Configuration`, `CT threshold (days)`, `Denominator N`, `Large |Delta CT| N`, `Large |Delta CT| fraction`.

Panel B: source paradigm, `D threshold`, `N_valid`, valid-rate fraction, unclipped recovery medians/dispersion fields, and `P(Delta F > 0)`.

## Filters

R1 rows were restricted to `period == test`. R3 rows were restricted to `period == test`. No basin or threshold row was dropped beyond those period filters.

## Transformations

Panel A was expanded from the source basin-level table across the five KGE thresholds and three CT thresholds. A second denominator class was added using the per-paradigm intersection of Base/TGD/CN KGE-pass basins. Panel B column names were normalized without clipping values.

## Statistical operations

Panel A uses deterministic counts and fractions. Panel B copies existing source summary statistics. No model calculation or new bootstrap was run.

## Plot script

`manuscript/supplement/final_assets/tables/Table_S2/build_Table_S2.py`

## Command

`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/tables/Table_S2/build_Table_S2.py`

## Output

Panel A: 180 rows. Panel B: 14 rows. Image output is not applicable.

## Image size / checksum
Not applicable to this table. Panel A CSV SHA-256: `d5d05727223352a2d35f76639bcb9e80475c9e891a69e5db13b0572d643718a4`; Panel B CSV SHA-256: `3d3f3be94c74bd2cdae03f0fd226bc5b244c0f3eb4475705feca2f12dd21edc2`.

## Known caveats

The R1 source contains one row per structure, regime, and basin; common-pass rows use all three structures. Panel B's dPL canonical valid set is the seed-median set (`N=460`), distinct from the pooled union (`N=468`) retained in the source audit.
