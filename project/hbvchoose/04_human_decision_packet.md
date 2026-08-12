# 04 — Human Decision Packet: Moderate Low-Flow Gate

## Purpose

This packet records the Phase 1 moderate low-flow gate diagnostic decisions. All items below have been reviewed and adjudicated. This packet freezes the Phase 2 design gate.

## Decision Items

| # | Item | Status | Detail |
|---|------|--------|--------|
| 1 | Primary gate threshold K=1.5 | ACCEPTED | K=1.5 selected as the primary gate threshold for Phase 2 design |
| 2 | min_event_length=5 | ACCEPTED | Accepted as the primary event-length setting |
| 3 | min_event_length=7 | ACCEPTED_AS_SENSITIVITY | Retained for sensitivity analysis; not used as primary |
| 4 | min_event_length=10 | REJECTED_AS_PRIMARY | Rejected as the primary setting |
| 5 | Basin 01047000 | EXCLUDED_FROM_PILOT_TRAINING | Excluded from Phase 2 pilot training |
| 6 | Basin 01047000 | RETAINED_AS_DIAGNOSTIC_BOUNDARY | Retained as a diagnostic boundary case for post-hoc analysis |
| 7 | Basin 01134500 | EXCLUDED_FROM_PILOT_TRAINING | Excluded from Phase 2 pilot training |
| 8 | Basin 01134500 | RETAINED_AS_FAILURE_EVIDENCE | Retained only as suspected_B failure evidence |
| 9 | Basin 01078000 | RETAINED_AS_DIAGNOSTIC_BOUNDARY | Retained as a diagnostic boundary case |
| 10 | Basin 01078000 | NOT_SUSPECTED_B | Do not classify as suspected_B |
| 11 | Suspected_B review | LIMITED_REVIEW_REQUIRED | YES — limited evidence-packet review required before execution |
| 12 | Phase 2 pilot execution | PHASE2_EXECUTION_BLOCKED | NO — execution remains blocked |

## Locked Gate Summary

- **Phase 2 design**: ALLOWED
- **Phase 2 execution**: NOT ALLOWED
- **Gate criteria**: K=1.5, min_event_length=5 (primary)
- **Sensitivity**: min_event_length=7 retained
- **Excluded basins**: 01047000, 01134500
- **Diagnostic boundary**: 01047000, 01078000
- **Suspected_B evidence only**: 01134500
- **Pending**: Limited suspected_B evidence-packet review
