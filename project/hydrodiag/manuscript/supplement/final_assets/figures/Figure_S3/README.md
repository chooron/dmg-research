# Figure S3 — Component-wise truth-relative parameter, state, and flux errors

## 1. Scientific role

Figure S3 expands the R3 controlled-experiment aggregate recovery diagnostics into component-wise errors. It belongs in the SI because it shows whether parameter/state distortion is distributed across components rather than adding another main-text aggregate ranking.

## 2. What is shown

Panel (a) contains IC and dPL facets for the 15 shared XAJ parameters. Each point is the excess absolute truth-relative parameter error of Base or TGD relative to the CN refit. Panel (b) contains the same facets for six canonical XAJ state/flux keys: upper, lower, and deep tension storage, total tension storage, interflow, and groundwater routing.

## 3. Source data

- `results/r3_misspec_analysis_v1/paired_parameters.csv`
- `results/r3_misspec_analysis_v1/state_excess.csv`
- `manuscript/scripts/r3/protocol_misspec_v1.json`
- Equations: `manuscript/methods_supplement_production_audit.md`, Sections 3.6 and 3.2

## 4. Sample definition

The parameter source covers 531 catchments in each IC structure comparison and three dPL seeds before basin-level aggregation. State values use `period == test` and `metric == nrmse`, with 531 catchments per IC structure and the three-seed dPL source collapsed to one basin value. The final state set is `wu`, `wl`, `wd`, `wt`, `qi`, and `qg`; the legacy `s` key is not used as a substitute for registered total storage `wt`.

## 5. Metric definitions

For parameters, `Delta|e| = |e_M| - |e_CN|`, where `e` is the normalized truth-relative parameter error and `M` is Base or TGD. For states/fluxes, `Delta E = NRMSE_M - NRMSE_CN`; positive values indicate greater deviation than the CN refit. `wt` is `wu + wl + wd`.

## 6. Aggregation and uncertainty

IC uses its selected restart; dPL uses the median across seeds per basin before the basin-level median. Points are basin-level medians. Error bars are 95% percentile bootstrap intervals from 2,000 basin resamples using seed 20260730. Parameter and state comparisons remain within estimation regime.

## 7. Generation method

- Script: `manuscript/supplement/final_assets/figures/Figure_S3/plot_Figure_S3.py`
- Command: `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S3/plot_Figure_S3.py`
- Input: frozen R3 basin-level CSVs and protocol definitions.
- Output: `Figure_S3.png`.
- **NO MODEL TRAINING. NO RECALIBRATION. NO FULL TEST PIPELINE.**

## 8. Visual encoding

Base is orange and TGD is green/teal. IC uses filled circles and dPL open triangles; no new hue is assigned to estimation regime. Zero reference lines are grey and intervals are horizontal error bars.

## 9. Caption-ready factual statements

- Panel (a) shows all 15 shared XAJ parameters.
- Panel (b) shows `W_{U,t}`, `W_{L,t}`, `W_{D,t}`, `W_t`, `Q_{i,t}`, and `Q_{g,t}`.
- Positive excess values indicate more truth-relative error than the CN refit.
- Intervals are 95% basin-level percentile bootstrap intervals from 2,000 draws.

## 10. Interpretation boundary

This figure is a controlled synthetic truth-relative error decomposition. It is not a model-performance ranking, real-catchment truth validation, or claim that every component is independently identifiable.

## 11. Validation

The renderer verified all 15 parameter keys and all six requested canonical state/flux keys before plotting. The output was visually inspected for complete facets, visible zero lines, legible symbols, and non-clipped intervals. No state export or forward simulation was run.
