# 14 — Phase 2 Quantitative GO/NO-GO Criteria

## Purpose

Convert the qualitative GO/NO-GO criteria from `08_phase2_stress_pilot_design.md` Section 11 into provisional quantitative thresholds. These thresholds must be reviewed and confirmed before Phase 2 pilot execution.

## Status

These criteria are **provisional** and correspond to gate item 9 in `11_phase2_execution_gate_form.csv` (currently `NOT_REVIEWED`). They must not be treated as confirmed until explicitly reviewed.

## GO Conditions

The pilot passes to GO if ALL of the following are satisfied:

| # | Condition | Threshold | Metric Source |
|---|-----------|-----------|---------------|
| G1 | Strong-A preservation rate | >= 90% | strong_A_preserved column in gate_metrics_by_basin.csv |
| G2 | Sample collapse basin fraction | <= 10% (≤ 2-3 basins in 20-30 basin pilot) | sample_collapse_flag in gate_metrics_by_basin.csv |
| G3 | Mean event removal rate | <= 30% across all training-eligible basins | event_removal_rate in gate_metrics_by_role.csv |
| G4 | Median point removal rate | <= 20% across all training-eligible basins | point_removal_rate in gate_metrics_by_basin.csv |
| G5 | Suspected_B separability | All suspected_B basins have separable_from_boundary = TRUE | suspected_B_handling_summary.csv |
| G6 | Diagnostic boundary identifiability | >= 2 of 3 boundary basins have boundary_distinguishable = TRUE | boundary_case_diagnostics.csv |
| G7 | Sensitivity stability | min_len=7 does not change final_status for any strong_A basin | event_retention_summary.csv (sensitivity_reverses_main = FALSE) |
| G8 | Output completeness | All 6 output files in 10_phase2_output_schema.md exist and are non-empty | File existence check |
| G9 | Status coverage | Every basin has a non-null final_status | gate_metrics_by_basin.csv |
| G10 | Bootstrap stability | <= 3 basins have bootstrap CI width > 0.3 | bootstrap_stability in gate_metrics_by_role.csv |

## NO-GO Conditions

The pilot is NO-GO if ANY of the following occur:

| # | Condition | Trigger |
|---|-----------|---------|
| N1 | Strong-A damage | Any strong_A_preservation basin has strong_A_preserved = FALSE |
| N2 | Broad sample collapse | > 10% of basins flagged with sample_collapse_flag |
| N3 | Unexplained massive removal | >= 3 basins exceed 50% event removal without diagnostic explanation |
| N4 | Suspected_B / boundary conflated | >= 2 suspected_B or boundary basins produce indistinguishable patterns (same failure mode, same removal rate within 5%) |
| N5 | K=1.5 instability | > 3 basins have bootstrap CI width > 0.3 AND different roles show opposing trends |
| N6 | Sensitivity reversal | Min_len=7 reverses the main conclusion: >= 2 strong-A basins change final_status from PASS to FAIL |
| N7 | Attribution failure | Output metrics cannot isolate whether signal change is from K=1.5, min_event_length, or data quality |
| N8 | Excluded basin contamination | Any excluded basin (01047000, 01134500) appears in pilot training output |
| N9 | Missing output | Any required output file from 10_phase2_output_schema.md is missing or empty |
| N10 | Placeholder contamination | Any placeholder basin remains in the inventory during execution |

## Bucket Definitions

### Strong-A preservation check

A basin is classified as strong_A_preserved if:
- At least 90% of pre-gate strong-A events survive the gate
- No strong-A basin is reclassified as WARNING or FAIL due to gate application

### Sample collapse check

A basin is flagged with sample_collapse if:
- event_removal_rate > 0.50 (more than half of events removed)

### Suspected_B separability

A suspected_B basin is separable if:
- Its removal pattern (event_removal_rate, point_removal_rate, phase_space_R2) differs significantly from diagnostic boundary basins
- The failure mode is attributable to basin-specific characteristics, not a systematic gate defect

### Sensitivity reversal

A sensitivity reversal occurs if:
- A basin with final_status = PASS under primary (K=1.5, len=5) changes to final_status = FAIL or WARNING under sensitivity (K=1.5, len=7)

## Review Requirements

Before execution, the following must be confirmed:

```text
1. Threshold values are appropriate for the final basin inventory
2. No threshold creates an automatic GO or automatic NO-GO for the expected basin set
3. Strong-A preservation criteria match the actual strong-A signal definition used in Phase 1
4. Suspected_B separability criteria are consistent with the evidence-packet review
5. Bootstrap stability CI width of 0.3 is validated against Phase 1 bootstrap results
6. Mean event removal rate threshold of 30% is validated against Phase 1 results
```

## Provisional Status

```text
These criteria are PROVISIONAL.
Gate item 9 in 11_phase2_execution_gate_form.csv remains NOT_REVIEWED.
Criteria must be reviewed and confirmed against the final basin inventory before execution.
```
