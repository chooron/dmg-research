# OOB model-selection handoff

## Current state

The original rule-based primary 8 remains preserved. Under the new MOPEX gate, both legal same-quadrant scenarios produce 8-model performance-valid candidates with one MOPEX member and complete quadrants, but both fail the unchanged parameter-complexity tertile requirement because only one selected model is in the medium-P tertile. No revised primary/fallback list is adopted; the blocker and both candidate tables are recorded in `project/benchmark/results/oob_model_selection_20260901/`.

## Selection boundary

The selected names are recommendations for a future OOB/PUB design, not OOB results. The four hypotheses in `OOB_MODEL_SELECTION_REPORT.md` remain untested. `simhyd` was allowed in the pool, VIC uses the dynamic-DOY canonical result, and no family was invented because trusted family metadata was unavailable.

## Stop condition

Do not start OOB/PUB, multi-seed, PUR, H1, training, optimizer, or checkpoint work until the user confirms the model list and separately approves the OOB protocol.
