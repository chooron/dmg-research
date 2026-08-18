# R4 — Real-Basin Shared Soil-Water State Consistency Pipeline

R4 tests whether missing snow processes and parameter compensation in hydrological
models (Base vs CN vs TGD2) translate into divergent downstream shared soil-water
state trajectories (`W_total = wu + wl + wd`) in real catchments, and whether
explicit snow physics yields more consistent internal soil moisture against external
references along an environmental snow-burden gradient.

## Scientific and Technical Setup

- **Canonical Basins**: 531 CAMELS-US catchments (`data/531sub_id.txt`).
- **Evaluation Period**: Test period 1995-10-01 .. 2010-09-30 (5,479 contiguous daily steps).
- **Primary External State Reference**: Caravan v1.1 CAMELS-US ERA5-Land `SM100`
  (0–100 cm depth-weighted composite: `0.07*L1 + 0.21*L2 + 0.72*L3` [m³/m³]).
- **Sensitivity Reference**: `SM289` (0–289 cm composite: `(0.07*L1 + 0.21*L2 + 0.72*L3 + 1.89*L4)/2.89`).
- **Model Shared State**: Total tension water storage $W_{\text{total}} = WU + WL + WD$ [mm].
- **Core Principles**:
  - NO absolute value comparison (mm vs m³/m³ is physically non-convertible).
  - NO 1-to-1 depth horizon mapping between conceptual XAJ layers and physical ERA5-Land soil layers.
  - Evaluate standardized dynamics (Pearson & Spearman correlation, 7-day smoothing, calendar-month anomalies, z-score NRMSE) and seasonal timing diagnostics.
  - ERA5-Land soil moisture is an *external process-state consistency reference*, NOT ground-truth soil moisture.

## Output Tagging & Separation

- `OFFICIAL_DPL_OBSERVATION_TRAINED`: Observation-trained canonical dPL Base/CN (`dpl_camels_531_lite_v2` seeds 42, 123).
- `IC_FUSED_5x200_SENSITIVITY`: Observation-trained fused IC Base/CN (5 starts × 200 generations).
- `DEV_ONLY` / `SYNTHETIC_TRAINED`: Produced from R3 synthetic-$q^*$ runs (`r3_gate_*`, `r3_misspec_*`); smoke tests only.
- `TGD2_PENDING`: Canonical TGD2 observation-trained checkpoints are currently pending; legacy TGD (`tgd_a/tgd_k_slow`) is strictly excluded.

## Module Map

| Module | Purpose |
|---|---|
| `common.py` | Canonical paths, R1/R2 periods, bundle loader, model registry |
| `protocol_r4_soil_v1.json` | Frozen machine-readable formal R4 protocol specification |
| `extract_caravan_soil.py` | Extracts 531 basins from Caravan v1.1 CAMELS-US NetCDF into compact cache (`results/r4_caravan_soil_reference_v1/`) |
| `state_export.py` | Continuous full-axis recorded forward (production kernels, per-day states, identity validation) |
| `input_adapters.py` | IC canonical/fused + dPL parameter loaders with fail-loud checks |
| `forward_export.py` | Post-hoc forward execution and npz/CSV array export |
| `snow_reference.py` | CAMELS-US Snow-17 SWE reference reader and annual burden metrics |
| `soil_analysis.py` | Formal state consistency, timing diagnostics, and snow-burden regressions |
| `robustness_analysis.py` | 4 formal robustness modules (performance controls, LORO, trimming, deciles, process phases, timing sensitivity) |
| `smoke_test.py` | `DEV_ONLY` pipeline smoke test verifying forward identity and adapter contracts |

## Reproduction Commands

```bash
# 1. Pipeline verification (DEV_ONLY smoke test)
python -m r4.smoke_test --device cuda

# 2. Extract Caravan soil moisture reference cache (531 basins x 5479 test days)
python -m r4.extract_caravan_soil

# 3. Run formal R4 soil-state consistency analysis (dPL Base/CN + IC fused sensitivity)
python -m r4.soil_analysis

# 4. Run full suite of R4 robustness checks
python -m r4.robustness_analysis
```

## Generated Artifacts

All formal outputs reside under `results/r4_phase1_soil_official/`:

- `basin_state_consistency.csv`: Basin-level state consistency metrics (531 basins × 2 models × 3 regimes).
- `paired_structural_effects.csv`: Paired $\Delta C(\text{CN} - \text{Base})$ per basin across all metrics.
- `timing_metrics_basin_year.csv`: Water-year level peak and spring wet-up timing metrics.
- `timing_metrics_basin_summary.csv`: Basin-level median timing errors and IQRs.
- `snow_burden_quartile_summary.csv`: Quantile breakdown table (Q0..Q3 by Snow-17 SWE burden).
- `robustness_performance_subsets.csv`: Similar-discharge performance subset results ($|\Delta\text{KGE}| \le 0.02, \le 0.05$).
- `robustness_controlled_regressions.csv`: OLS regressions controlling for $\Delta\text{KGE}$ with 2,000-replicate bootstrap 95% CIs.
- `robustness_leave_one_region_out.csv`: Leave-one-region-out cross-validation across 18 HUC regions.
- `robustness_extreme_swe_trimming.csv`: Extreme SWE trimming (top 1% and top 5% removed).
- `robustness_swe_decile_shape.csv`: SWE deciles D01..D10 response shape analysis.
- `robustness_process_phase_consistency.csv`: 4-phase process-conditioned state consistency.
- `robustness_timing_sensitivity.csv`: Timing definition sensitivity across 7d/14d/21d wet-up and annual/spring-summer peak windows.
- `r4_phase1_soil_official_report.json` & `r4_robustness_report.json`: Master JSON reports.
