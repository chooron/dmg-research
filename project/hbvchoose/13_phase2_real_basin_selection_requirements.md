# 13 — Phase 2 Real Basin Selection Requirements

## Purpose

This document defines the requirements for replacing the 25 placeholder basins in `09_phase2_basin_inventory.csv` with real CAMELS basin IDs. Execution is not permitted until all placeholders are resolved.

## Current State

`09_phase2_basin_inventory.csv` contains 28 rows:
- 3 adjudicated real basins: 01047000, 01134500, 01078000
- 25 placeholder basins: PLACEHOLDER_01 through PLACEHOLDER_25

All placeholder rows have `ready_for_execution = NO` and `is_placeholder = YES`.

## Required Basin Role Categories

Each real basin must be assigned exactly one primary role from:

| Role | Count Target | Description |
|------|-------------|-------------|
| strong_A_preservation | 4–8 | Basins with known strong-A signal; gate must preserve these |
| moderate_quality | 10–15 | General moderate-quality basins for bulk evaluation |
| suspected_B_failure_evidence | 1–2 | Basins where the gate may fail; diagnostic only |
| IC_phase_boundary | 2–3 | Basins near initial-condition phase boundaries |
| hydroclimatic_diversity | 4–6 | Basins covering varied aridity/snow/rainfall regimes |

## Per-Basin Requirements

Each real basin must have:

```text
1. Real basin ID from CAMELS gage_id list
2. Exactly one primary role
3. Clear include_in_extraction_diagnostic (YES/NO)
4. Clear include_in_pilot_training (YES/NO)
5. diagnostic_boundary (YES/NO)
6. failure_evidence (YES/NO)
7. A written reason for selection
8. is_placeholder = NO
9. ready_for_execution = YES (after all requirements met)
```

## Locked Adjudicated Statuses

These are forced by Phase 1 adjudication and must not be changed:

```text
01047000:
  include_in_pilot_training = NO
  diagnostic_boundary = YES
  failure_evidence = NO
  role = IC_phase_boundary

01134500:
  include_in_pilot_training = NO
  diagnostic_boundary = NO
  failure_evidence = YES
  role = suspected_B_failure_evidence

01078000:
  diagnostic_boundary = YES
  failure_evidence = NO
  known_status = NOT_SUSPECTED_B
  role = IC_phase_boundary
  include_in_pilot_training = YES (eligible for extraction diagnostic; may be DIAGNOSTIC_ONLY if design chooses)
```

If 01078000 is designated as DIAGNOSTIC_ONLY for pilot training, the role must remain IC_phase_boundary and the reason must explain why it is diagnostic-only despite being NOT_SUSPECTED_B.

## Screening Criteria for Candidate Basins

Candidate basins must satisfy:

```text
1. Present in CAMELS gage_id.npy
2. valid_target_ratio >= 0.90 after warmup
3. eval_valid_ratio >= 0.90
4. No NaN or Inf in forcing data
5. No Inf in discharge data
6. q_zero_ratio < 0.95
7. Minimum total length >= warmup + train + eval days
8. Not in excluded list (01047000, 01134500)
```

## Hard Constraints

```text
1. Excluded basins (01047000, 01134500) must not be used for training.
2. Diagnostic boundary basins must be flagged in all output CSVs.
3. Suspected_B failure-evidence basins must not contribute to GO/NO-GO decisions.
4. Placeholder rows must be removed or replaced before execution.
```

## Execution Gate

Execution is not allowed until:

```text
- All placeholder basin IDs are replaced by real basin IDs
- Each basin has exactly one primary role
- Each basin has clear inclusion/exclusion status
- Each basin has a written reason
- Training eligibility is explicitly marked
- Diagnostic-only basins are separated from pilot-training basins
- Excluded basins are not used for training
- Basin roles are frozen (gate item 2 = CONFIRMED)
- Basin inventory is reviewed (gate item 1 = CONFIRMED)
```
