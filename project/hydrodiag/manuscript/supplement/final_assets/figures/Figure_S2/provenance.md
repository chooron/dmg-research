# Provenance

Final asset: `manuscript/supplement/final_assets/figures/Figure_S2/Figure_S2.png`
Generated: 2026-08-27
Git hash: `322fb932c922a131a67800bcbc6aa7eb704c7605`

## Inputs

- `results/reviewer2_robustness/tgd_response/tgd_response_data.csv`
- `results/reviewer2_robustness/tgd_response/tgd_response_summary.md`
- `results/reviewer2_robustness/tgd_shape_sensitivity/tgd_shape_sensitivity_basin_metrics.csv`
- `results/reviewer2_robustness/tgd_shape_sensitivity/tgd_shape_sensitivity_summary.json`
- `manuscript/methods_supplement_production_audit.md`

## Columns / keys

Response data: `temperature_c`, six `tau_*` columns, and their six `retention_*` columns. Shape data: `variant`, `t_ref`, `s_t`, `delta_F_ic`, and `delta_F_dpl`, with the corresponding frozen basin metrics.

## Filters

All 351 response temperatures and all four named shape variants were retained. Panel (c) uses the existing finite basin-level metric values; no additional basin filter or seed selection was introduced.

## Transformations

Response column labels were mapped to formal display symbols `tau_t` and `r_t`. Panel (c) displays empirical median and Q25–Q75 values for `Delta F`; no KGE or `G` series were added.

## Statistical operations

Panels (a)–(b) are direct curves. Panel (c) computes only the existing CSV-level median and Q25/Q75 summaries through the renderer. No model or experiment computation was performed.

## Plot script

`manuscript/supplement/final_assets/figures/Figure_S2/plot_Figure_S2.py`

## Command

`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S2/plot_Figure_S2.py --out manuscript/supplement/final_assets/figures/Figure_S2/Figure_S2.png`

## Output

Three-panel PNG: response residence time, daily retention, and fixed-shape recovery sensitivity. Image checksum is recorded in the final execution report.

## Image size / checksum

6793 × 2497 px, RGBA; SHA-256 `4541b60113ae196f21d8a3eb76c61d41dbfe1b1b26d69b1ef5ff05dee308f770`.

## Known caveats

Some source column names retain `tau(T)`/`retention` wording for legacy traceability. The final panel labels use the formal time-indexed `tau_t` and `r_t`. The broad shape variant is an empirical degradation, not a bound.
