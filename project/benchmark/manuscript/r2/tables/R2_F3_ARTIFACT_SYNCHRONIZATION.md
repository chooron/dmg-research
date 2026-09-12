# R2 F3 artifact synchronization

## Verdict

```text
F3_ARTIFACT_SYNC = PASS
```

## Active authoritative artifacts

The active F3 manuscript-readiness branch is the exact-common-support branch:

- `tables/F3_LOCALIZATION_EXACT_MATCHED_MODEL.csv`
- `tables/F3_LOCALIZATION_EXACT_MATCHED_SUMMARY.csv`
- `cache/fig3c_adjusted_localization.csv`
- `tables/F3_PREPLOT_AUDIT_REPORT.md`
- `scripts/audit_f3_preplot.py`
- `scripts/plot_r2_figure3_final.py`
- `tables/R2_FIGURE_DATA_MANIFEST.csv`
- `tables/R2_HEADLINE_REPRO_CHECK.csv`
- `tables/R2_BLOCKER_RESOLUTION_REPORT.md`

The exact-common-support values are:

```text
C_eff:     0.34521687865 -> 0.2933820389
Top-1:     0.5711769539  -> 0.6726335350
Top-2:     0.8799996185  -> 0.9417612386
Direction: 22/23 models for each metric
```

The strict rank and localization model sets remain identical at 23/36 models.

## Status changes

- `R2_FIGURE_DATA_MANIFEST.csv`: Figure 3a changed from `BLOCKED` to `READY`; Figure 3c changed from `PARTIAL` to `READY`. The manifest now points to the frozen model-summary and exact-common-support tables used by the final Figure 3 script.
- `R2_HEADLINE_REPRO_CHECK.csv`: the two exact-basin comparison rows now validate the exact-common-support primary values and are `PASS`. The earlier all-basins raw values remain only in explicitly labeled `HISTORICAL/SENSITIVITY ONLY` rows.
- `R2_FIGURE_DATA_AUDIT.md`: marked `SUPERSEDED / HISTORICAL` at the top. Its former BLOCKED/PARTIAL statuses and old comparison are retained as provenance, not as the active gate.
- `tables/R2_FINAL_MANUSCRIPT_AUDIT.md`: marked `HISTORICAL / SUPERSEDED`; the active gate is this blocker-resolution report.

## Historical values retained without promotion

The values `0.346893` and `0.573874` are the 23-model all-available-basins raw summaries. They are not the raw metrics on the exact adjusted-valid basin support and are not used for the final F3 comparison. No historical value was deleted; it is explicitly labeled as sensitivity/history.

## Scientific result and scope

No scientific result, estimand, aggregation rule, model set, or figure role was changed. The synchronization only makes the already-frozen exact-common-support branch the active manuscript-facing branch and labels the superseded comparison appropriately. No new experiment, training, calibration, simulation, or alternative analysis was run.
