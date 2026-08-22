# Flex-MOPEX manuscript data analysis

This directory is the manuscript-facing analysis workspace for the current
Flex-MOPEX project. It contains copied analysis code, analysis plans, and later
compact tables/figures generated from the completed CAMELS-671 results.

## Current result source

The completed formal result root is:

```text
results/formal_671_unified_nmul1_tail3/
```

It contains 103 completed runs: 88 main runs and 15 `nmul` sensitivity runs
(`nmul=1/4/8/16/32`).

## Imported code

`manuscript/scripts/` contains the data-analysis Python scripts imported from
`/home/jingxin/code/dmg-research/project/flexmopex/scripts/`, plus analysis
helpers already present in the current workspace. Training launchers and remote
execution shell scripts were intentionally not copied.

The source checkout does not currently contain a Flex-MOPEX
`project/flexmopex/manuscriptv2` directory, and its root `manuscript/` contains
only cache files. The available Flex-MOPEX manuscript-oriented code is therefore
under `project/flexmopex/scripts/`; this repository-consistent location was used
for the import.

## Status

The scripts have been organized and inventoried, but no full analysis pipeline
has been launched yet. Several historical scripts still assume the old
`results/block1_*` or `results/block3_loro` layout and must be adapted or given
explicit `--project-root`/input paths before use with the final 671 result root.
See `RESULTS_ANALYSIS_PLAN.md` for the section-by-section mapping and gates.

## Resource policy

- Do not retrain models as part of manuscript analysis.
- Start with metadata/JSON metrics and compact per-basin tables; do not load all
  time-series arrays into memory.
- Run one analysis job at a time by default.
- Use conservative BLAS/OpenMP settings (`OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) for CPU analysis.
- Use GPU only for an explicitly required inference/forward calculation, with a
  bounded batch size and a smoke test first.
- Every analysis stage should write a small manifest, counts, and output path.
