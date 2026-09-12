# Remaining seen-basin analysis (2026-09-01)

This directory contains the reproducible, non-training continuation of the formal 36-model Canonical dPL v2 / IC seen-basin analysis. It is deliberately split into short lane scripts rather than one monolithic Python program.

## Frozen contract

- 36 canonical models × 531 canonical basins; TEST `1995-10-01..2010-09-30`.
- dPL: canonical v2, seed 42, frozen `runs/<model>/best.pt`.
- IC: canonical aligned artifacts; VIC uses the dynamic-DOY replacement; `simhyd` generation 280 remains a canonical accepted member of the structural 36-model ensemble.
- Main variable: `G_seen = KGE_IC - KGE_dPL = -Delta_KGE`.
- `G_seen` means only a seen-basin shared attribute→parameter mapping flexibility gap. It is not OOB/PUB transferability.
- No optimizer, training, backward pass, checkpoint update, H1, multi-seed, OOB/PUB, or PUR operation is permitted.

## Scripts

- `common.py`: paths, frozen inputs, canonical attributes, archived IC ten-start loading, correlation and bootstrap helpers.
- `agent_a_gap_and_admissibility.py`: model/basin heterogeneity and predictively admissible sets.
- `agent_b_model_place.py`: model–attribute associations and cross-model consistency.
- `agent_c_distance_restart.py`: normalized parameter distance and archived IC restart identifiability.
- `agent_d_reliability_confound.py`: reproducibility bridges, admissible reliability, and confounds.
- `coordinator_sensitivity_and_master.py`: sensitivity, master tables, provenance, key statistics, and final report.
- `coordinator_figures.py`: figure-ready data and eight scientific QC plots.
- `coordinator_final_qc.py`: independent deterministic source reconstruction and contract QC.
- `run_all.py`: bounded serial orchestration of the above scripts.

## Execution

From repository root:

```bash
.venv/bin/python3 project/benchmark/analysis/seenbasin_remaining_20260901/run_all.py
```

The output is written only to `project/benchmark/results/seenbasin_remaining_analysis_20260901/` with `agent_A/`, `agent_B/`, `agent_C/`, `agent_D/`, `qc/`, `figures/`, and `tables/` subdirectories. Existing formal results are read-only inputs and are never overwritten.

## Outputs and handoff

The result directory contains the requested A/B/C/D CSVs, figure-data CSVs and PNGs, master CSVs, `SEENBASIN_REMAINING_ANALYSIS_FINAL_REPORT.md`, `SEENBASIN_ANALYSIS_PROVENANCE.md`, `SEENBASIN_KEY_STATISTICS.json`, and a QC record. `HANDOFF.md` in this analysis directory records the frozen state and the exact next decision boundary.
