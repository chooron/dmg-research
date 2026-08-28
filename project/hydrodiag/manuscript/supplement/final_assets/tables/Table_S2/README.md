# Table S2 — R1 timing and R3 denominator threshold sensitivity

## 1. Scientific role

Table S2 documents threshold sensitivity for the outlet diagnostics in R1 and controlled-recovery denominator in R3. It belongs in the SI because it audits robustness of prespecified screens rather than presenting the primary main-text estimand.

## 2. What is shown

Panel A crosses KGE screening thresholds 0.40–0.80 with `|Delta CT|` thresholds 10, 15, and 20 d for Base, TGD, and CN. It reports both configuration-specific KGE-pass denominators and common-pass denominators requiring all three structures to pass. Panel B reports test-period recovery summaries across the actual `D` threshold grid.

## 3. Source data

- `manuscript/cache/r1_rebuild_audit_staged/r1_basin_level_ct.csv`
- `manuscript/results/discussion_audit/r3_denominator_sensitivity_audit.csv`
- Reference implementation: `manuscript/scripts/shared/generate_table_s2_sensitivity.py`

## 4. Sample definition

Panel A uses the 531 catchments in the R1 test-period source, separately for IC-CMA-ES and dPL-MLP. Structure-specific rows use the named structure's own KGE pass set. Common-pass rows use the intersection of Base/TGD/CN pass sets. Panel B uses 531 total catchments and the source-defined valid denominator set; canonical valid N is 427 for IC and 460 for dPL.

## 5. Metric definitions

`Delta CT` is the signed basin-median center-of-timing error. Panel A counts `abs(Delta CT) >= 10, 15, 20 d`. Panel B uses `D_b` as the reference-outlet gap denominator and reports unclipped `F_close`, `F_TGD`, and `Delta F = F_TGD - F_close` summaries.

## 6. Aggregation and uncertainty

Panel A is a deterministic count/fraction audit with no bootstrap interval. Panel B reproduces the existing source medians, IQR fields, P05/P95 fields, and positive-Delta-F proportion. No model seeds, forward simulations, or new resampling were run.

## 7. Generation method

- Script: `manuscript/supplement/final_assets/tables/Table_S2/build_Table_S2.py`
- Command: `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/tables/Table_S2/build_Table_S2.py`
- Outputs: `Table_S2_panelA.csv`, `Table_S2_panelB.csv`, and `Table_S2.md`.
- **NO MODEL TRAINING. NO RECALIBRATION. NO FULL TEST PIPELINE.**

## 8. Visual encoding

The Markdown table labels denominator type explicitly. Formal symbols and code-column mappings are maintained in `_audit/symbol_registry.md`.

## 9. Caption-ready factual statements

- Panel A covers KGE thresholds 0.40, 0.50, 0.60, 0.70, and 0.80 and CT thresholds 10, 15, and 20 d.
- Panel A distinguishes structure-specific from all-structure common-pass denominators.
- Panel B uses the source grid `1e-6, 1e-4, 1e-3, 0.01, 0.02, 0.05, 0.10`.
- Canonical Panel B valid N is 427 for IC and 460 for dPL.

## 10. Interpretation boundary

Threshold stability does not establish spatial independence, causal structure, or a universal recovery bound. Fractions are diagnostic summaries and should not be interpreted as clipped probabilities when source values are explicitly unclipped.

## 11. Validation

The final table was checked for 180 Panel A rows (both denominator types, two regimes, three structures, five KGE thresholds, three CT thresholds) and 14 Panel B rows. The common-pass N at KGE 0.60 is 321 for IC and 331 for dPL, matching the Figure 2 common-pass audit.
