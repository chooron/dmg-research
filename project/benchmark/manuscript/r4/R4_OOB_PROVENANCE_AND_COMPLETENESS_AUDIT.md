# R4 OOB Provenance and Completeness Audit

- **Audit status:** **PASS**
- **Formal OOB root:** `/home/jingxin/code/dmg-research/project/benchmark/results/oob_primary8_5fold_20260902`
- **Jobs:** 40/40 discovered; required status is 40/40 DONE.
- **Models/folds:** 8 models × 5 folds.
- **OOF basin contract:** expected 531 unique held-out basins per model; fold sizes `[107, 106, 106, 106, 106]`.
- **Fold assignment SHA256:** `fe9013bcd7155ea99ac85597e59ea5e417a13897dc08aaa69f2c63ddb6780085`
- **Configuration:** seed 42; canonical dPL v2 values were checked per run; selection metric is train loss.
- **Leakage evidence:** per-run phase/access flags, train-only normalization metadata, partition IDs, and source provenance were checked.
- **Numerical integrity:** checkpoint tensors and held-out KGE rows were checked one run at a time.
- **Retry evidence:** `0` retry line(s) in the downloaded queue log.

## Failures
- none

## Notes
- none

## Stage boundary
This audit opened only job metadata, partition files, checkpoint finiteness, KGE evidence, and provenance. It did not calculate or inspect any OOB attribute–parameter relationship coefficient. Population relationship analysis remains locked until the frozen case manifest below.

## Frozen seen-evidence case manifest
- Path: `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r4/R4_RELATIONSHIP_CASES_FROZEN.csv`
- SHA256: `44dc1426818bf16aac7acb70c94fe022e7cd2671ab658d89a75570f8d7277a14`
- Cases: 4
