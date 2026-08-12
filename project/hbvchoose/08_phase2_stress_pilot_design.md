# 08 — Phase 2 Moderate Low-Flow Gate Stress Pilot: Design Document

## 1. Purpose

This document defines the Phase 2 stress-pilot design for a 20–30 basin extraction-quality diagnostic pilot. It encodes the locked Phase 1 adjudication decisions and establishes basin roles, inclusion/exclusion logic, metrics, GO/NO-GO criteria, and the execution gate. This is a design artifact only — no pilot has been executed.

## 2. Locked Phase 1 Decisions

See `04_human_decision_packet.md` and `05_human_decision_form.csv` for full adjudication records.

| Decision | Value |
|----------|-------|
| Primary K | 1.5 |
| Primary min_event_length | 5 |
| Sensitivity min_event_length | 7 |
| Rejected min_event_length | 10 |
| Excluded basins | 01047000, 01134500 |
| Diagnostic boundary basins | 01047000, 01078000 |
| Failure evidence only | 01134500 |
| 01078000 classification | NOT_SUSPECTED_B |
| Phase 2 design | ALLOWED |
| Phase 2 execution | NOT ALLOWED |

## 3. Phase 2 Objective

Run a conservative 20–30 basin extraction-quality diagnostic pilot to stress-test the moderate low-flow gate. The pilot must verify that:

- K=1.5 with min_event_length=5 preserves strong-A structures
- No broad sample collapse occurs across basin roles
- Suspected_B failure evidence remains separable from diagnostic boundary cases
- The sensitivity setting (min_event_length=7) does not reverse the main conclusion
- Output metrics support clear attribution to gate parameters

## 4. Basin Role Categories

| Role | Description | Count (target) |
|------|-------------|----------------|
| strong_A_preservation | Basins with known strong-A signal; must be preserved by gate | 4–8 |
| moderate_quality | General moderate-quality basins for bulk evaluation | 10–15 |
| suspected_B_failure_evidence | Basins where the gate may fail; retained for diagnostic only | 1–2 |
| IC_phase_boundary | Basins near initial-condition phase boundaries | 2–3 |
| hydroclimatic_diversity | Basins covering varied aridity/snow/rainfall regimes | 4–6 |

## 5. Basin Inclusion/Exclusion Rules

```text
INCLUDE in extraction diagnostic if:
  - Passes strict screening (valid_ratio >= 0.90, no NaN/Inf forcing)
  - Has >= 10 years of continuous data
  - Not in excluded_pilot_training list

EXCLUDE from pilot training if:
  - basin_id in [01047000, 01134500]

RETAIN as diagnostic boundary if:
  - basin_id in [01047000, 01078000]
  - Participates in extraction diagnostic only
  - Does NOT contribute to gate parameter tuning

RETAIN as failure evidence only if:
  - basin_id == 01134500
  - Output collected for post-hoc failure analysis
  - Not used for GO/NO-GO decisions
```

## 6. Diagnostic Boundary Handling

Diagnostic boundary basins (01047000, 01078000) are processed through the extraction pipeline identically to other basins, but their output is sequestered in `boundary_case_diagnostics.csv`. They are used to check whether the gate behaves differently near known IC-phase boundaries. They do not influence GO/NO-GO criteria directly.

## 7. Suspected_B Handling

01134500 is the sole suspected_B basin. It is processed to completion but flagged in output. Its results are retained only in `suspected_B_handling_summary.csv`. Any failure observed in this basin is documented but does not block GO unless it reveals a systematic gate defect that also appears in non-suspected basins.

## 8. Main Configuration

```yaml
gate:
  K: 1.5
  min_event_length: 5
  gate_type: moderate_low_flow
  target_basins: 20-30
  screening_mode: strict
  warmup_days: 365
  extraction_window_days: 365
```

## 9. Sensitivity Configuration

```yaml
gate:
  K: 1.5
  min_event_length: 7
  gate_type: moderate_low_flow
  purpose: sensitivity_check
```

## 10. Metrics to Report

| Metric | File |
|--------|------|
| n_events_before, n_events_after | gate_metrics_by_basin.csv |
| event_removal_rate, point_removal_rate | gate_metrics_by_basin.csv |
| sample_collapse_flag | gate_metrics_by_basin.csv |
| phase_space_R2 (before vs after gate) | gate_metrics_by_role.csv |
| bootstrap_stability (CI of removal rate) | gate_metrics_by_role.csv |
| IC_phase_conflict_flag | boundary_case_diagnostics.csv |
| strong_A_preserved (per basin) | gate_metrics_by_basin.csv |
| suspected_B_flag | suspected_B_handling_summary.csv |
| diagnostic_boundary_flag | boundary_case_diagnostics.csv |
| training_eligible | gate_metrics_by_basin.csv |

## 11. GO / NO-GO Criteria

**GO** if:
- strong-A preservation remains high (>= 90% of strong-A events retained)
- No basin exceeds sample_collapse threshold (removal_rate < 50%)
- Suspected_B failure evidence remains separable (does not mimic boundary cases)
- Diagnostic boundary cases remain identifiable (IC_phase_conflict_flag = False on >= 2 boundary basins)
- Sensitivity (min_len=7) does not reverse the main conclusion (same basins flagged)
- Mean event_removal_rate < 30% across all basins

**NO-GO** if:
- Any strong-A basin loses > 20% of events
- >= 3 basins flagged with sample_collapse
- Suspected_B and boundary cases produce indistinguishable output patterns
- K=1.5 behaves unstably (bootstrap CI width > 0.3 on >= 5 basins)
- >= 3 basins exceed 50% event removal
- Output metrics cannot support attribution to gate parameters

## 12. Execution Gate

All items in `11_phase2_execution_gate_form.csv` must be reviewed and marked ACCEPTED before Phase 2 pilot execution is permitted. The final item `Execution approved` must remain NOT_REVIEWED until all preceding items are accepted and the suspected_B evidence-packet review is completed.

## 13. Explicit Non-Actions

```text
No Phase 2 pilot was run.
No 20–30 basin extraction was run.
No model training was run.
No 50–100 basin expansion was run.
No 671 basin run was performed.
```
