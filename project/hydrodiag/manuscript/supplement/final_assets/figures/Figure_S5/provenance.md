# Provenance

Final asset: `manuscript/supplement/final_assets/figures/Figure_S5/Figure_S5.png`
Generated: 2026-08-27
Git hash: `322fb932c922a131a67800bcbc6aa7eb704c7605`

## Inputs

- `manuscript/supplement/figures/FigureS1_R4_selection_audit.json`
- `manuscript/supplement/figures/FigureS1_R4_population_audit.csv`
- `manuscript/supplement/figures/FigureS1_R4_multibasin_validation.png`
- Original renderer: `manuscript/scripts/r4/plot_r4_figure_s1_multibasin.py`
- External arrays recorded by the original R4 renderer: `results/r4_caravan_soil_reference_v1/caravan_soil_ensemble.npz` and `results/r4_swe_reference_v1/swe_ensemble.npz`

## Columns / keys

Selection JSON keys: `selection_rule`, `evaluation_period`, `eligibility_rule`, `selected_example_basins`, and `population_summaries`. Population CSV keys include `basin_id`, `swe_burden_group`, `snowiest_water_year`, `snow_active_days`, `eligible`, `r_Base`, `r_TGD`, `r_CN`, and the two `delta_r` fields.

## Filters

Examples were accepted only from the frozen external Snow-17 SWE burden selection. Population eligibility requires at least 10 snow-active days with SWE ≥ 5 mm in the snowiest water year. No outcome-based selection was applied.

## Transformations

The existing R4 PNG was staged under final Figure S5 naming. No trajectories, anomalies, or population values were recomputed.

## Statistical operations

Population medians and confidence intervals are source annotations. The staging step performs no resampling or model calculation.

## Plot script

`manuscript/supplement/final_assets/figures/Figure_S5/plot_Figure_S5.py`

## Command

`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S5/plot_Figure_S5.py`

## Output

Seven-panel external-state example/population PNG. Image checksum is recorded in the final execution report.

## Image size / checksum

2063 × 2562 px, RGBA; SHA-256 `8e0477c077b14902874baf6cf3ecc9e59d97e838d269a6c6efa14dae4cbdda5b`.

## Known caveats

The final Figure S5 is a renumbered/staged existing R4 asset. The audit contains 442 eligible catchments (88 Low, 177 Middle, 177 High) and six examples. This is external-state corroboration, not truth validation. Original R4 source provenance is preserved.
