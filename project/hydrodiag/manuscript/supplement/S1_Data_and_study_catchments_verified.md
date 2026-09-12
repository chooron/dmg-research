# Text S1. Data and study catchments

## S1.1 Data sources and preprocessing

The CAMELS-US source dataset is described by Addor et al. (2017; DOI: 10.5194/hess-21-5293-2017). The active 531-basin project path reads a pickle tuple containing daily forcing, observed discharge, and 35 static attributes, together with basin IDs, dates, and forcing names from `camels_forcing_v2.pkl`. Runtime inspection verified 671 source basins and 531 study basins. The selected forcing array has shape (531, 12418, 3) and the target array has shape (531, 12418). The forcing order is precipitation (P), mean air temperature (T), and potential evapotranspiration (PET), with project input units of mm d-1, degC, and mm d-1, respectively. Observed discharge is read as ft3 s-1 and converted to basin-average runoff depth in mm d-1 using `area_gages2` in km2.

The source date axis runs from 1980-10-01 to 2014-09-30 with 12418 contiguous daily records; leap days are retained. PET is used as the active project's precomputed forcing field; no named generation method is assigned in this S1. PET provenance statistics are provided in `results/s1_pet_audit.json` and `results/s1_pet_distribution.csv`.

**Table S1.1.** The generated data manifest, array inventory, forcing summary, and PET distribution are provided in `results/s1_data_manifest.json`, `results/s1_data_arrays.csv`, `results/s1_forcing_summary.csv`, and `results/s1_pet_distribution.csv`.

## S1.2 Snow water equivalent reference and catchment aggregation

The external SWE product is treated only as a gridded process-state consistency reference. Raster metadata are recorded in `results/s1_swe_product_manifest.json`; no basin aggregation or SWE data statistics are part of this S1 audit. It is not called truth or ground truth.

**Table S1.2.** The official raster-product metadata and the out-of-scope status of basin aggregation are recorded in `results/s1_swe_product_manifest.json` and `results/s1_swe_aggregation_protocol.json`.

## S1.3 Catchment selection

The active study list is `data/531sub_id.txt`, which contains 531 unique eight-digit basin IDs in a fixed order. The runtime source collection contains 671 IDs. The 531 list is the direct project source-of-truth list. The proposed filtering cascade is not required for this S1 audit and is not used to redefine the study set. The selection cascade and set differences are in `results/s1_catchment_selection_cascade.csv` and `results/s1_catchment_selection_set_differences.csv`.

The completion scan counted a model cell as complete only when a readable result file contained exactly the selected 531 IDs in aligned form. The resulting matrix is shown below; it is an audit of coverage, not a model-performance result.

| route | host | structure | n_complete_531_files | status |
| --- | --- | --- | --- | --- |
| IC-XNES | XAJ | Base | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| IC-XNES | XAJ | GD/TGD surrogate | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| IC-XNES | XAJ | CN | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| IC-XNES | GR4J | Base | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| IC-XNES | GR4J | GD/TGD surrogate | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| IC-XNES | GR4J | CN | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| IC-XNES | SIMHYD | Base | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| IC-XNES | SIMHYD | GD/TGD surrogate | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| IC-XNES | SIMHYD | CN | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| dPL-MLP | XAJ | Base | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| dPL-MLP | XAJ | GD/TGD surrogate | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| dPL-MLP | XAJ | CN | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| dPL-MLP | GR4J | Base | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| dPL-MLP | GR4J | GD/TGD surrogate | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| dPL-MLP | GR4J | CN | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| dPL-MLP | SIMHYD | Base | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| dPL-MLP | SIMHYD | GD/TGD surrogate | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |
| dPL-MLP | SIMHYD | CN | 0 | UNRESOLVED_NO_COMPLETION_MANIFEST |

**Table S1.3.** The generated selection manifest, selection flags, cascade, set differences, completion matrix, and failed-run list are the machine-readable versions of this audit (`results/s1_catchment_manifest.csv`, `results/s1_catchment_selection_flags.csv`, `results/s1_catchment_selection_cascade.csv`, `results/s1_catchment_selection_set_differences.csv`, `results/s1_design_completion_matrix.csv`, and `results/s1_failed_or_missing_runs.csv`).

## S1.4 Streamflow completeness and missing-data handling

The conversion is Q(ft3 s-1) times 0.028316846592 m3 ft-3, 86400 s d-1, and 1000 mm m-1, divided by area (km2) times 10^6 m2 km-2. The implementation factor is 2.44657554555; independent elementwise checks are recorded in `results/s1_streamflow_conversion_audit.json`. Finite zero discharge is retained as zero. Nonfinite and negative raw discharge values are masked, not zero-filled. The exact valid-day counts and completeness by basin and fixed stratum are in `results/s1_streamflow_completeness_basin.csv` and `results/s1_streamflow_completeness_by_stratum.csv`.

**Table S1.4.** Streamflow conversion, missing-value counts, per-basin completeness, low-completeness basins, and stratum summaries are generated in `results/s1_streamflow_conversion_audit.json`, `results/s1_streamflow_completeness_basin.csv`, `results/s1_low_completeness_catchments.csv`, and `results/s1_streamflow_completeness_by_stratum.csv`.

## S1.5 Catchment attributes

The active network input contract contains 35 attributes in the exact order listed in `results/s1_attribute_manifest.csv`. Three categorical code fields (`dom_land_cover`, `geol_1st_class`, and `geol_2nd_class`) were excluded from the Spearman correlation figure; they are not interpreted as continuous measurements. Figure S1.1 was generated from pairwise-complete ranks among the remaining 32 fields.

The dPL route uses column-median imputation, IQR scaling with a standard-deviation fallback, and clipping to [-5, 5], based on the selected 531 basins. The active IC adapter returns raw attributes. [[UNRESOLVED: definitions and units for all 35 fields were not loaded from a CAMELS metadata file by the active project path.]] The preprocessing evidence and leakage assessment are in `results/s1_attribute_preprocessing.json`.

**Table S1.5.** The 35-field order, descriptive statistics, missingness, pairwise Spearman coefficients, effective sample sizes, and high-correlation pairs are in `results/s1_attribute_manifest.csv`, `results/s1_attribute_descriptive_statistics.csv`, `results/s1_attribute_missingness.csv`, `results/s1_attribute_spearman.csv`, `results/s1_attribute_pairwise_n.csv`, and `results/s1_attribute_high_correlations.csv`.

## S1.6 Process-activity index and stratification

The active contract identifies `frac_snow` at attribute index 3. It was extracted directly from `attributes[:, 3]` in `/home/jingxin/code/dmg-research/data/camels_dataset` after 531 ID alignment, without recomputation or standardization. Fixed strata were applied to the unstandardized field using S1 [0, 0.05), S2 [0.05, 0.15), S3 [0.15, 0.30), S4 [0.30, 0.50), and S5 [0.50, 1.00]. The exact counts are:

| stratum | interval | lower | upper | n_basins | P25 | P50 | P75 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | [0, 0.05) | 0 | 0.05 | 165 | 0.00276202 | 0.0124691 | 0.0310455 |
| S2 | [0.05, 0.15) | 0.05 | 0.15 | 156 | 0.0673389 | 0.0844551 | 0.106279 |
| S3 | [0.15, 0.30) | 0.15 | 0.3 | 121 | 0.176741 | 0.202089 | 0.23532 |
| S4 | [0.30, 0.50) | 0.3 | 0.5 | 34 | 0.333413 | 0.365243 | 0.42968 |
| S5 | [0.50, 1.00] | 0.5 | 1 | 55 | 0.632982 | 0.682273 | 0.71723 |

Boundary unit tests and a quintile sensitivity cross-tabulation are in `results/s1_frac_snow_boundary_tests.csv` and `results/s1_fixed_vs_quintile_crosstab.csv`. The semantic definition is inherited from the named CAMELS attribute; this project does not recompute it from daily forcing.

## S1.7 Extended catchment characteristics by activity stratum

For this S1 description only, annual P, PET, and observed runoff were aggregated over calendar years contained in the active calibration/test union, with a 90% valid-day requirement applied jointly to the three daily series. Runoff coefficient is annual runoff divided by annual precipitation, and the aridity index is annual PET divided by annual precipitation. This rule is documented separately from the experiment configurations and has a transparent sensitivity table in `results/s1_annual_aggregation_sensitivity.csv`.

The generated stratum table is:

| stratum | aridity_index | calibration_completeness | catchment_area_km2 | frac_snow | mean_annual_pet_mm | mean_annual_precip_mm | mean_annual_runoff_mm | mean_elevation_m | mean_slope_pct | runoff_coefficient | test_completeness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All | 0.929479 | 1 | 309.75 | 0.0964519 | 1007.56 | 1153.03 | 416.361 | 437.01 | 28.9567 | 0.369154 | 1 |
| S1 | 0.988574 | 1 | 400.53 | 0.0124691 | 1334.96 | 1307.29 | 380.431 | 250.6 | 9.66599 | 0.302457 | 1 |
| S2 | 0.991741 | 1 | 340.52 | 0.0844551 | 1055.68 | 1102.2 | 354.02 | 349.44 | 11.799 | 0.328111 | 1 |
| S3 | 0.767809 | 1 | 246.33 | 0.202089 | 890.152 | 1179.78 | 588.906 | 543.37 | 38.2676 | 0.490228 | 1 |
| S4 | 0.875292 | 1 | 211.645 | 0.365243 | 794.582 | 998.22 | 615.803 | 1210.93 | 107.711 | 0.567207 | 1 |
| S5 | 0.816268 | 1 | 150.95 | 0.682273 | 732.574 | 909.851 | 503.872 | 2448.68 | 112.779 | 0.580408 | 1 |

All machine-readable values are in `results/s1_hydroclimatic_characteristics_long.csv` and `results/s1_hydroclimatic_characteristics_by_stratum.csv`. Figure S1.1 is `figures/Fig_S1_1_attribute_correlation.png` and its vector PDF counterpart. No model-performance result is reported in this Text S1.

**Table S1.6.** Fixed-stratum counts, quintile sensitivity, the fixed-versus-quintile cross-tabulation, and extended catchment characteristics are generated in `results/s1_fixed_strata_summary.csv`, `results/s1_quintile_strata_summary.csv`, `results/s1_fixed_vs_quintile_crosstab.csv`, `results/s1_hydroclimatic_characteristics_long.csv`, and `results/s1_hydroclimatic_characteristics_by_stratum.csv`.
