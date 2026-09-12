# R2 Consolidation Cleanup Plan

Created before any deletion. This cleanup only affects the working manuscript pipeline under `project/benchmark/manuscript/r2/`; the historical provenance tree under `project/benchmark/results/` is read-only and is not a deletion target.

## Baseline

- Repository HEAD before cleanup: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
- Baseline git status: 146 untracked files under manuscript/r2 and no tracked changes in this scope; full output was recorded in `/tmp/r2_consolidation/git_status_before.txt` and each path is reproduced in `DELETION_MANIFEST.tsv`.
- Inventory snapshot: `/tmp/r2_consolidation/inventory_before.txt` (146 files).

## Target canonical layout

Keep only the final six-module pipeline, its shared helper, validator, runner, README, derived cache groups, and the explicitly required cleanup audit records:

- `README.md`
- `scripts/r2_common.py`
- `scripts/01_parameter_separation_icself.py`
- `scripts/02_rank_reorganization.py`
- `scripts/03_coordinate_localization_icself.py`
- `scripts/04_distribution_reference_sensitivity.py`
- `scripts/05_outlet_parameter_bridge.py`
- `scripts/06_r2_r3_rank_linkage_audit.py`
- `scripts/99_validate_frozen_results.py`
- `scripts/run_all.sh`
- `cache/inputs/`, six module cache groups, and `cache/final/`
- `CLEANUP_PLAN.md`, `DELETION_MANIFEST.tsv`, and `CLEANUP_REPORT.md`

## Deletion policy

Every pre-existing manuscript/r2 file is listed in `DELETION_MANIFEST.tsv` with path, tracked status, reason, superseding canonical module or cache group, source SHA256, and delete decision. The old numbered 00–09 scripts, exploratory clustering/recurrence/landing/spatial/role branches, old tables, figures, pycache, manifests, and root audit drafts are superseded by the final six-module pipeline and are deletion targets after the replacement pipeline passes reproduction validation. No file under `project/benchmark/results/` is included in the deletion manifest.

The explicit cleanup audit records remain at the root even though the three-entry skeleton is otherwise canonical; this preserves the requested pre-deletion and post-deletion audit trail.
