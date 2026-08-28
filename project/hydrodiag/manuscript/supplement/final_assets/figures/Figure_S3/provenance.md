# Provenance

Final asset: `manuscript/supplement/final_assets/figures/Figure_S3/Figure_S3.png`
Generated: 2026-08-27
Git hash: `322fb932c922a131a67800bcbc6aa7eb704c7605`

## Inputs

- `results/r3_misspec_analysis_v1/paired_parameters.csv`
- `results/r3_misspec_analysis_v1/state_excess.csv`
- `manuscript/scripts/r3/protocol_misspec_v1.json`
- `manuscript/methods_supplement_production_audit.md`

## Columns / keys

Parameters: `paradigm`, `structure`, `parameter`, `delta_abs_e`, `seed`, and basin ID. States/fluxes: `paradigm`, `structure`, `variable`, `period`, `metric`, `delta_E`, `seed`, and basin ID. Final state keys are `wu`, `wl`, `wd`, `wt`, `qi`, and `qg`.

## Filters

Parameter rows were restricted to the 15 shared XAJ keys. State rows were restricted to `period == test`, `metric == nrmse`, and the six final state/flux keys. IC and dPL were kept as separate regimes.

## Transformations

For parameters, the frozen primary `delta_abs_e = abs(e_M) - abs(e_CN)` was used directly. For states/fluxes, `delta_E` was used directly. dPL seed values were collapsed by basin/structure/component median before the basin-level median and bootstrap interval.

## Statistical operations

Each point is a basin-level median. Error bars are 2,000 percentile bootstrap medians using seed 20260730. No state export, model forward, or recalibration was performed.

## Plot script

`manuscript/supplement/final_assets/figures/Figure_S3/plot_Figure_S3.py`

## Command

`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S3/plot_Figure_S3.py`

## Output

Two-row, two-facet PNG: 15 shared XAJ parameter components and six canonical XAJ state/flux components. Image checksum is recorded in the final execution report.

## Image size / checksum

5984 × 4920 px, RGBA; SHA-256 `a5660d784cdf0e8f3c546959d016122debe897415345acbea85860171c5ae1d6`.

## Known caveats

The production R3 protocol calls `s` a primary common variable and `wt` a registered derived secondary variable. This final asset uses the explicit `wt` key, together with `wu`, `wl`, `wd`, `qi`, and `qg`, to satisfy the frozen XAJ symbol convention and avoid relabeling `s` as total storage. The figure is a component error audit, not a performance ranking.
