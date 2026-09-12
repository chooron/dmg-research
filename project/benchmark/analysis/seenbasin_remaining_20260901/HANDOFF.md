# Seen-basin remaining analysis handoff

## Completed

The frozen seen-basin continuation is complete under `project/benchmark/results/seenbasin_remaining_analysis_20260901/`. Agent A–D analyses, sensitivity checks, master tables, figure-ready CSVs, eight QC figures, provenance, and final report were generated without training or optimizer activity.

## Frozen contract

- Structural main analysis: 36 models × 531 canonical basins = 19,116 paired TEST rows.
- dPL: Canonical v2 seed 42, frozen `runs/<model>/best.pt`.
- IC: canonical aligned artifacts; VIC is the dynamic-DOY replacement; simhyd generation 280 is the accepted canonical exception and remains in the main 36-model ensemble.
- Main gap: `G_seen = KGE_IC - KGE_dPL`; positive means IC advantage/shared mapping cost.
- This is seen-basin descriptive evidence only, not OOB/PUB transferability.

## Key outputs

- Final report: `project/benchmark/results/seenbasin_remaining_analysis_20260901/SEENBASIN_REMAINING_ANALYSIS_FINAL_REPORT.md`
- Machine summary: `SEENBASIN_KEY_STATISTICS.json`
- Master tables: `SEENBASIN_MASTER_MODEL_SUMMARY.csv`, `SEENBASIN_MASTER_BASIN_SUMMARY.csv`, `SEENBASIN_MASTER_PARAMETER_SUMMARY.csv`
- Provenance: `SEENBASIN_ANALYSIS_PROVENANCE.md`
- Agent outputs: `agent_A/`, `agent_B/`, `agent_C/`, `agent_D/`
- QC and figures: `qc/`, `figures/`

## Scientific status

The evidence chain is closed for performance heterogeneity, predictively admissible models (`tau=0.05` primary; `0.02/0.10` sensitivity), model–place patterns, normalized parameter realization distance, archived IC restart identifiability, reliability bridges, confounds, and robustness summaries. Restart coverage is complete because ten-start IC `best_latent` and `best_fitness` payloads are archived for every canonical model/basin.

## Stop boundary

Do not start multi-seed, OOB/PUB, PUR, H1, training, optimizer changes, or manuscript drafting automatically. The next decision belongs to the user after reviewing the final report and QC outputs.
