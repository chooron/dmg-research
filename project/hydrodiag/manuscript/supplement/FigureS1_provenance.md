# Figure S1 provenance

- **Scientific question:** Does the R4 external-state consistency pattern appear in representative catchments spanning low, middle, and high external SWE burden, with a population anchor panel?
- **Input files:**
  - `results/r4_caravan_soil_reference_v1/caravan_soil_ensemble.npz`
  - `results/r4_swe_reference_v1/swe_ensemble.npz`
  - `manuscript/supplement/figures/FigureS1_R4_population_audit.csv`
  - `manuscript/supplement/figures/FigureS1_R4_selection_audit.json`
- **Input keys:** Caravan `basin_ids`, `dates`, `test_slice_start`, `test_slice_stop`, `SM100`; R4 model-array inputs `wu`, `wl`, `wd`; Snow-17 `swe_median`; audit columns `swe_burden_group`, `eligible`, `delta_r_CN_Base`, and `delta_r_TGD_Base`.
- **Selection/aggregation:** Six outcome-independent example catchments, two per external Snow-17 SWE burden tercile. Panel g uses eligible rows in the audit CSV, standardized anomaly correlation contrasts, and the recorded population medians with 95% intervals.
- **N:** Six examples. Panel-g eligible total is **442** (`Low=88`, `Middle=177`, `High=177`). The prior image label `N=443` was stale.
- **IC/dPL handling:** Figure S1 uses the existing R4 dPL seed-42 replay/official arrays; it is not an IC/dPL comparison figure.
- **Interval definition:** Population panel uses the `ci_95` fields recorded in the selection audit (existing Figure S1 workflow); example trajectories are selected seasonal traces.
- **Plot script:** `manuscript/scripts/r4/plot_r4_figure_s1_multibasin.py`. A one-line dynamic eligible-count correction was made so the title reads `N=442`; layout and data selections were unchanged.
- **Output:** `manuscript/supplement/figures/FigureS1_R4_multibasin_validation.png`.
- **Canonical values checked:** The audit JSON records Low CN–Base median −0.0116, Middle +0.1303, and High +0.3451; the output retains these values and the six IDs in the selection JSON.
- **Limitation:** The current production audit contains an older example-ID/caption listing; the selection JSON and audit CSV used by the plotting script are the authoritative current inputs.
