# Manuscript cache

This directory contains worktree-local intermediate data, audit reports, logs, and temporary exports.
It is not a source-of-record for manuscript figures or tables.

- Final figures: `manuscript/figures/`
- Final and explicitly labelled interim tables: `manuscript/tables/`
- Source results consumed by scripts: `manuscript/results/` or `results/`
- Temporary exports and provenance checks: this directory

Do not write generated assets to a parent checkout or to `manuscript/plots/`.
