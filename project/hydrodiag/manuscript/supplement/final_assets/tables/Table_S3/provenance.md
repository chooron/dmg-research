# Provenance

Final asset: `manuscript/supplement/final_assets/tables/Table_S3/Table_S3.md`, `Table_S3_panelA.csv`, and `Table_S3_panelB.csv`
Generated: 2026-08-27
Git hash: `322fb932c922a131a67800bcbc6aa7eb704c7605`

## Inputs

- `results/reviewer2_robustness/p0_reporting/recovery_denominator_tail_audit.csv`
- `results/reviewer2_robustness/p0_reporting/invalid_denominator_strata_breakdown.csv`
- `results/reviewer2_robustness/summaries/canonical_registry.csv`
- `results/reviewer2_robustness/alt_generating_field/alt_generating_field_summary.json`

## Columns / keys

Panel A: `section`, `paradigm`, `period`, `snow_stratum`, `metric`, N fields, quantiles, unclipped tail proportions, paired Delta-F proportions, and explicit invalidity-denominator fields. Panel B: generating-field construction, paradigm, period, `G_Base`, `G_TGD`, `F_close`, `F_TGD*`, `Delta F`, total/valid N, and positive-Delta-F proportion.

## Filters

Panel A was restricted to `period == test`; all S1–S5 rows in the source breakdown were retained. Panel B retained IC/dPL test rows from the canonical registry and alternative-field JSON.

## Transformations

Source column names were normalized to manuscript-facing names. Overall denominator rows and S1–S5 denominator-validity rows were combined into one machine-readable Panel A. No recovery value was clipped or recomputed.

## Statistical operations

No new statistical operation was performed. Medians, quantiles, tails, and proportions are copied from frozen source summaries. Panel B values are copied from the canonical registry or alternative-field summary.

## Plot script

`manuscript/supplement/final_assets/tables/Table_S3/build_Table_S3.py`

## Command

`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/tables/Table_S3/build_Table_S3.py`

## Output

Panel A: 18 rows. Panel B: 4 rows. Image output is not applicable.

## Image size / checksum

Not applicable to this table. Panel A CSV SHA-256: `f866592c6e7ddbbc73bcc185cbac3f8db7a6e065e4940dee45c08c88d10f7a89`; Panel B CSV SHA-256: `16478bf058e9091dd9291d34ecf654e5c1d072c5cdcb44f6eda81da43ebbb971`.

## Known caveats

The direct basin-wise calibrated CN-IC parameter field is a generating-field construction sensitivity, not a real-catchment truth. The canonical dPL `N=460` is the seed-median valid set; the pooled union `N=468` remains a separate audit quantity. Canonical `F_TGD*` values are sourced from the canonical registry, while the tail audit is retained in Panel A as its own frozen summary.
