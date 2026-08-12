# 06 — Human Adjudication Summary: Moderate Low-Flow Gate

## Status

Phase 1 moderate low-flow gate diagnostic is completed.

## Adjudicated Decisions

| # | Decision | Status |
|---|----------|--------|
| 1 | K=1.5 accepted as primary gate threshold | ACCEPTED |
| 2 | min_event_length=5 accepted as primary | ACCEPTED |
| 3 | min_event_length=7 retained as sensitivity | ACCEPTED_AS_SENSITIVITY |
| 4 | min_event_length=10 rejected as primary | REJECTED_AS_PRIMARY |
| 5 | 01047000 excluded from pilot training | EXCLUDED_FROM_PILOT_TRAINING |
| 6 | 01047000 retained as diagnostic boundary | RETAINED_AS_DIAGNOSTIC_BOUNDARY |
| 7 | 01134500 excluded from pilot training | EXCLUDED_FROM_PILOT_TRAINING |
| 8 | 01134500 retained as failure evidence only | RETAINED_AS_FAILURE_EVIDENCE |
| 9 | 01078000 retained as diagnostic boundary | RETAINED_AS_DIAGNOSTIC_BOUNDARY |
| 10 | 01078000 not classified as suspected_B | NOT_SUSPECTED_B |
| 11 | Limited suspected_B review required | LIMITED_REVIEW_REQUIRED |
| 12 | Phase 2 execution remains blocked | PHASE2_EXECUTION_BLOCKED |

## Phase 2 Gate Status

```text
Phase 2 design:    ALLOWED
Phase 2 execution:  NOT ALLOWED
```

K=1.5 and min_event_length=5 are accepted for Phase 2 design. min_event_length=7 is retained as sensitivity. 01047000 and 01134500 are excluded from pilot training. 01047000 and 01078000 are retained as diagnostic boundary cases. 01134500 is retained only as failure evidence. Phase 2 pilot execution remains blocked pending suspected_B evidence-packet review.
