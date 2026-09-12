# R2 Cleanup Report

Cleanup was performed only under `project/benchmark/manuscript/r2/` after the pre-deletion HEAD/status and 146-file inventory were recorded in `CLEANUP_PLAN.md` and `DELETION_MANIFEST.tsv`.

- Deletion manifest targets: 146
- Physically deleted obsolete files: 145
- Replaced-in-place canonical helper: 1 (old `scripts/r2_common.py` source superseded by the final helper)
- Deleted tracked files: 0
- Physically deleted untracked files: 145
- Replaced-in-place untracked old helper files: 1
- Remaining obsolete manifest paths: 0 (excluding the intentional helper replacement)
- Ambiguous files: 0
- Forbidden broad cleanup: not used; `git clean -fdx` was not run.
- Results/provenance retained: `project/benchmark/results/` was not a deletion target and was not modified.

The final manuscript pipeline contains only the six canonical modules, the shared helper, validator, runner, README, derived cache groups, and the explicitly required cleanup audit records. Old exploratory clustering, recurrence, landing/role, boundary/spatial/influence, superseded contraction, and incorrect Agent B aggregation entrypoints were removed from manuscript/r2 only.
