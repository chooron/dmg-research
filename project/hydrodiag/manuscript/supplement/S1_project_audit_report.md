# S1 项目审计报告

## 执行摘要

本次审计从运行时读取 671 个源流域和 531 个研究流域。源数据时间轴为 1980-10-01 至 2014-09-30，共 12418 个连续日步；当前 531 foundation 配置实际为 1980-10-01 至 1981-09-30 warm-up、1981-10-01 至 1995-09-30 calibration、1995-10-01 至 2010-09-30 test。审计范围只包含当前 531 流域主路径。

## Source-of-truth map

```text
camels_forcing_v2.pkl + camels_dataset + 531sub_id.txt
  -> ablation/ic_core/data_adapter.py::load_531_bundle
     -> active 531 IC-XNES foundation data contract
  -> training/dpl/run_dpl_model.py::load_data
     -> active dPL CAMELS-531 result directories
```

## 核心结论

| 审计项 | 状态 | 结论 |
|---|---|---|
| 源集合 | VERIFIED | 671 basins |
| 研究集合 | VERIFIED | 531 unique eight-digit IDs |
| 强迫顺序 | VERIFIED | P, T, PET |
| PET 输入 | INFERRED | active project uses a precomputed PET field; formula tracing is outside S1 scope |
| 时间协议 | VERIFIED/UNRESOLVED | active 531 routes share the foundation dates; IC test metric is not enabled in preflight |
| 流量换算 | VERIFIED | implementation factor 2.44657554555 mm day-1 per (ft3 s-1 km-2) |
| 531 selection | INFERRED | data/531sub_id.txt is the project source-of-truth list |
| Static attributes | VERIFIED/CONFLICT | 35-name order verified; dPL normalization differs from IC raw contract |
| frac_snow attribute | VERIFIED | extracted from camels_dataset attributes[:,3] using the active 35-field order |
| SWE | OUT OF SCOPE | retained only as an external gridded process-state consistency reference |
| all design cells | UNRESOLVED | completion counted only from aligned active 531 result files |

## Evidence and generated outputs

Evidence is recorded in `S1_verified_facts.json`; data fingerprints and shapes are in `results/s1_data_manifest.json`; each statistics table records the generating path via the audit log and entry-point script. The generated files are listed in `results/s1_manuscript_change_log.md`.

## Fixed strata

| stratum | interval | lower | upper | n_basins | P25 | P50 | P75 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | [0, 0.05) | 0 | 0.05 | 165 | 0.00276202 | 0.0124691 | 0.0310455 |
| S2 | [0.05, 0.15) | 0.05 | 0.15 | 156 | 0.0673389 | 0.0844551 | 0.106279 |
| S3 | [0.15, 0.30) | 0.15 | 0.3 | 121 | 0.176741 | 0.202089 | 0.23532 |
| S4 | [0.30, 0.50) | 0.3 | 0.5 | 34 | 0.333413 | 0.365243 | 0.42968 |
| S5 | [0.50, 1.00] | 0.5 | 1 | 55 | 0.632982 | 0.682273 | 0.71723 |

## Hydrological climate description

The primary S1 aggregation rule uses calendar years contained in the active calibration/test union and a 90% valid-day threshold. It is explicitly a descriptive S1 rule, not an original model hyperparameter. The full long and wide tables are in `s1_hydroclimatic_characteristics_long.csv` and `s1_hydroclimatic_characteristics_by_stratum.csv`.

## Scope closures


## Claims prohibited from manuscript

- Do not call the external SWE product a truth or ground truth.
- Do not state a PET method without a verified generating script or metadata.
- Do not state that the 531 list was reconstructed from the two proposed filters.
- Do not state that every host, structure, and estimation route completed all 531 basins.
