# Table S3 — Controlled-recovery distributions and generating-field robustness

## 1. Scientific role

Table S3 records distributional properties of the R3 controlled recovery ratios and tests sensitivity to the generating-field construction. It belongs in the SI because it exposes denominator/tail behavior and a robustness comparison that would interrupt the main aggregate narrative.

## 2. What is shown

Panel A reports overall test-period distributions for `D`, `F_close`, `F_TGD`, and paired `Delta F`, including quantiles, tail fractions, valid N, and both denominator interpretations for S1–S5 invalidity. Panel B compares canonical PCA/SVD-ridge generating-field summaries with the direct basin-wise calibrated CN-IC parameter field for IC and dPL.

## 3. Source data

- `results/reviewer2_robustness/p0_reporting/recovery_denominator_tail_audit.csv`
- `results/reviewer2_robustness/p0_reporting/invalid_denominator_strata_breakdown.csv`
- `results/reviewer2_robustness/summaries/canonical_registry.csv`
- `results/reviewer2_robustness/alt_generating_field/alt_generating_field_summary.json`
- `results/reviewer2_robustness/REVIEWER2_ROBUSTNESS_FINAL_REPORT.md`

## 4. Sample definition

Panel A uses 531 test-period catchments, with `N_valid` defined by `D_b > 10^-6`; canonical valid N is 427 for IC and 460 for dPL. The stratum audit reports S1–S5 totals 165, 156, 121, 34, and 55. Panel B uses 531 total catchments and the field-specific valid sets; direct-field valid N is 522 for IC and 123 for dPL. No seed aggregation is newly performed here; source summaries already encode the canonical IC/dPL conventions.

## 5. Metric definitions

`D_b` is the reference-outlet gap denominator. `F_close = G_Base / D_b`, `F_TGD* = G_TGD / D_b`, and `Delta F = F_TGD* - F_close`. The `D` and recovery rows retain the source's unclipped values. Panel B reports `G_Base`, `G_TGD`, `F_close`, `F_TGD*`, `Delta F`, and `P(Delta F > 0)` for each field construction.

## 6. Aggregation and uncertainty

Panel A copies median, Q25, Q75, P05, P95, and tail proportions from the frozen audit. S1–S5 rows are validity counts rather than recovery quantiles. Panel B copies field-specific medians and proportions from the canonical registry and alternative-field JSON. No new bootstrap or resampling is run.

## 7. Generation method

- Script: `manuscript/supplement/final_assets/tables/Table_S3/build_Table_S3.py`
- Command: `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/tables/Table_S3/build_Table_S3.py`
- Outputs: `Table_S3_panelA.csv`, `Table_S3_panelB.csv`, and `Table_S3.md`.
- **NO MODEL TRAINING. NO RECALIBRATION. NO FULL TEST PIPELINE.**

## 8. Visual encoding

The Markdown table separates overall distribution rows from S1–S5 validity rows and labels the two generating-field constructions explicitly. Symbol mappings follow `_audit/symbol_registry.md`.

## 9. Caption-ready factual statements

- Panel A uses 531 total catchments and reports valid N of 427 (IC) and 460 (dPL) at the canonical denominator cutoff.
- Invalid-denominator shares and within-stratum invalid rates are both reported for S1–S5.
- Recovery tails are unclipped; negative and above-one values remain visible in the machine-readable table.
- Panel B compares the canonical PCA/SVD-ridge field with the direct basin-wise calibrated CN-IC parameter field.

## 10. Interpretation boundary

The direct field is not a true field or real-catchment truth. The ratios are denominator-sensitive diagnostic summaries, not bounded probabilities or universal performance limits. S1 concentration uses all invalid basins as its denominator, while within-stratum invalidity uses the stratum total.

## 11. Validation

The final package contains 18 Panel A rows (8 overall metric rows and 10 S1–S5 validity rows) and 4 Panel B rows. Canonical values were checked against the reviewer-2 registry and the source tail audit; the known dPL `N=460` seed-median versus `N=468` pooled distinction is preserved.
