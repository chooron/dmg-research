# Flex-MOPEX current project audit before manuscript v2

> Audit date: 2026-07-08
> Auditor: read-only agent
> Git branch: master (clean before/after audit, same diff)

---

## 1. Project root overview

| Item | Value |
| --- | --- |
| Target path | `/home/jingxin/code/dmg-research/project/flexmopex` |
| Is git repo | Yes (parent repo `dmg-research`) |
| Branch | `master` |
| Recent commits | `22d8ebb update ungauge training`, `b7acc94 update ungauge training`, `3f5b127 update project`, `2a14d25 upload readme`, `fb8904d submit new project` |

Top-level contents:

```
conf/               - Hydra config (multiple alpha/version variants)
models/             - Core model implementations (9 .py files)
scripts/            - Run scripts (.sh) and collection scripts (.py)
manuscript/         - Full paper draft, figures, tables, plots, supplement
results/            - Major experiment outputs (block1/block3, binary_pilot, lopo_smoke)
logs/               - Training logs organized by experiment
output/             - Legacy v1 learned_weight_mopex outputs
test/               - Unit tests (3 files)
run_model.py        - Main training entry point
run_batch_fixed_weights.py   - Batch runner for fixed-weight variants
run_batch_loro.py            - Batch runner for LORO experiments
model_builder.py             - Model factory
local_model_handler.py       - Data loading
```
- `manuscript/`, `results/`, `logs/`, `scripts/`, `.tmp/`, etc. 均为 unstaged untracked files（从未提交到 git，仅存在于工作区）。
- 存在 `analysis/` 目录曾跟踪但现在被删除（staged as D）。

---

## 2. Manuscript inventory

### 2.1 Paper sections (`manuscript/paper/`)

| File | Size | Lines | Role |
| --- | --- | --- | --- |
| `abstract.md` | 1.8K | 7 | Abstract |
| `introduction.md` | 8.5K | 15 | Introduction (cites Ye et al. 2012, CAMELS, differentiable modelling) |
| `method.md` | 12K | 108 | Methods (Flex-MOPEX, CFlex/DFlex, experiments) |
| `results.md` | 14K | 83 | Full results with figure/table placeholders |
| `discussion.md` | 17K | 51 | Discussion (process hierarchy, Ye comparison, limitations) |
| `conclusions.md` | 4.4K | 13 | Conclusions |
| `supplement.md` | 51K | 583 | Comprehensive supplement |
| `methods_information_package.md` | 34K | 381 | Detailed methods/implementation appendix |
| `doi.md` | 0.9K | 35 | DOI/dataset references |

**Status**: Draft manuscript exists with all sections written, figures referenced, and table placeholders. Written in English, markdown format. No .docx or .pdf compiled version found.

### 2.2 Supplement sections (`manuscript/supplement/`)

| File | Lines | Content |
| --- | --- | --- |
| `textS7_nmul_ablation.md` | 19 | nmul ablation supplement text |
| `textS8_parameter_space_readout.md` | 43 | Parameter space readout supplement |
| `textS9_evaluation_metrics.md` | 49 | Evaluation metrics supplement |
| `textS10_statistical_testing.md` | 22 | Statistical testing details |
| `textS10b_existing_supplement_backfill.md` | 138 | Existing supplement backfill |
| `textS11_appendix_and_availability.md` | 44 | Appendix & data availability |

### 2.3 Figure captions

| File | Lines | Role |
| --- | --- | --- |
| `manuscript/figure_caption_notes.md` | 150 | Full figure caption drafts (main + appendix, 14 main + 6 appendix) |
| `manuscript/md/detailed_figure_captions_en.md` | ~400 | Detailed extended English captions |

---

## 3. Main figures inventory

### 3.1 Main figures (`manuscript/figures/main/`)

| File | Size | Date | Figure |
| --- | --- | --- | --- |
| `figure1_nse.png` | 867K | 2026-06-30 | Fig 1 - Predictive performance |
| `figure2_alpha_effect_nse.png` | 1.2M | 2026-06-30 | Fig 2 - Alpha path / tradeoff |
| `figure3_spatial_weights_v2.png` | 7.7M | 2026-06-23 | Fig 3 - CONUS weight maps |
| `figure4_alpha_weight_response_stacked.png` | 965K | 2026-06-30 | Fig 4 - Alpha response / seed stability |
| `figure5_weight_attribute_relationships_heatmap.png` | 904K | 2026-06-30 | Fig 5 - Weight-attribute heatmap |
| `figure6_weight_attribute_relationships.png` | 601K | 2026-06-30 | Fig 6 - Binned attribute response |
| `figure7_gradient_beeswarm.png` | 2.3M | 2026-06-23 | Fig 7 - Gradient beeswarm |
| `figure8_weight_parameter_relationships.png` | 410K | 2026-06-30 | Fig 8 - Weight-parameter matrix |
| `plot_fig09_bridge_alpha0p005.png` | 2.6M | 2026-06-23 | Fig 9 - Bridge (alpha=0.005) |
| `plot_fig09_bridge_alpha0p01.png` | 2.4M | 2026-06-23 | Fig 9 - Bridge (alpha=0.01) |
| `plot_fig09_bridge_alpha0p03.png` | 2.3M | 2026-06-23 | Fig 9 - Bridge (alpha=0.03) |
| `plot_fig10_weight_parameter_regime_gradients.png` | 4.0M | 2026-06-23 | Fig 10 - Parameter regime gradients |
| `plot_fig11_parameter_organization.png` | 1.1M | 2026-06-30 | Fig 11 - Parameter org / readout |
| `plot_fig12_loro_performance.png` | 1.5M | 2026-06-30 | Fig 12 - LORO predictive |
| `plot_fig13_structural_coordinate_transfer.png` | 856K | 2026-06-30 | Fig 13 - LORO coordinate transfer |
| `plot_fig14_structural_decision_transfer.png` | 628K | 2026-06-30 | Fig 14 - LORO categorical |
| `figure16_weight_lopo_validation.png` | 1.3M | 2026-06-30 | Fig 16 - LOPO validation |
| `Figure11_parameter_space_readout.png` | 1.1M | 2026-06-30 | Alt Fig 11 |

**Summary**: 14 main figures (Fig 1-14) + 1 extra (Fig 16) all rendered as PNG.

### 3.2 Appendix figures (`manuscript/figures/appendix/`)

| File | Size | Date | Figure |
| --- | --- | --- | --- |
| `figureS1_kge.png` | 978K | 2026-06-30 | Fig S1 - KGE performance |
| `figureS2_alpha_effect_kge.png` | 403K | 2026-06-30 | Fig S2 - Alpha-KGE path |
| `figA3_metric_tradeoff.png` | 550K | 2026-06-30 | Fig A3 - Multi-metric tradeoff |
| `figA4_seed_robustness.png` | 456K | 2026-06-30 | Fig A4 - Seed robustness |
| `figA5_nmul_ablation.png` | 506K | 2026-06-30 | Fig A5 - nmul ablation |
| `figA6_threshold_sensitivity.png` | 597K | 2026-06-30 | Fig A6 - Threshold sensitivity |
| `figA7_parameter_readout_stability.png` | 682K | 2026-06-30 | Fig A7 - Readout stability |
| `figA8_interception_diagnostic.png` | 599K | 2026-06-30 | Fig A8 - Interception diagnostic |
| `figureS_full_weight_parameter_relationships.png` | 579K | 2026-06-30 | Full weight-parameter appendix |
| `figAs.zip` | 2.8M | 2026-06-22 | Zip of appendix figs |

**Summary**: 9 appendix/supplement figures.

### 3.3 Legacy figure cache (`manuscript/figures/tmp/`)

Contains `figure7_gradient_cache/` (4 .pkl files), `figure3_spatial_weights_cache.pkl`, and `reorg_backup_20260605T095126/` with older versions and SVGs.

---

## 4. Tables inventory (`manuscript/tables/`)

### Main tables

| Table | CSV | MD | TEX | Content |
| --- | --- | --- | --- | --- |
| Table 1 | `table1_performance_complexity_summary.csv` | Yes | Yes | Basic/Full/CFlex/DFlex NSE/KGE/Complexity |
| Table 1b | `table1b_panelB_process_extension_weights.csv` | Yes | Yes | Process extension weights by alpha |
| Table 2 | `table2_process_coordinate_evidence_synthesis.csv` | Yes | Yes | Process-level evidence synthesis |
| Table 3 | `table3_loro_transferability_summary.csv` | Yes | Yes | LORO transfer by region |
| Fig 11 tables | 6 CSV files | -- | -- | Parameter org details |

### Supplement tables

| Table | Content |
| --- | --- |
| S1a/b | Multi-metric performance (train+test / test-only) |
| S2 | Alpha tradeoff path |
| S3 | Seed robustness |
| S4a/b | Hydroclimatic controls |
| S5a/b/c | Parameter space readout |
| S6 | LORO regional performance |
| S7 | Continuous coordinate transfer |
| S8a/b | Categorical decision transfer |
| S9a/b | nmul ablation |
| S10 | Threshold sensitivity |

**Total**: 4 main tables + 12 supplement table files. All have CSV + MD + TEX triples.

### Table manifest

`table_manifest.csv` (21 rows) tracks every table with its upstream data sources.

---

## 5. Experimental results inventory

### 5.1 Training outputs (raw)

| Experiment | Location | Contents |
| --- | --- | --- |
| **block1_main** | `results/block1_main/` | base/flex/full, alpha=0.0/0.005/0.01/0.03, 5 seeds each |
| **block1_alpha_path** | `results/block1_alpha_path/` | flex alpha=0.0-0.1 sweep, seeds 42/123/456 |
| **block1_full_lopo** | `results/block1_full_lopo/` | 4 LOPO variants, 3 seeds each |
| **block1_nmul_ablation** | `results/block1_nmul_ablation/` | nmul=1/8/16/32 at alpha=0.01, seed=42 |
| **block3_loro** | `results/block3_loro/` | base/flex/full x 7 regions x 2-3 seeds |
| **binary_pilot** | `results/binary_pilot/` | Full alpha sweep (0.0-0.1) x 3 seeds, both CFlex+DFlex |
| **lopo_smoke** | `results/lopo_smoke/` | Preflight tests |

Per-run output structure: `model/`, `sim/` (streamflow.npy, w_*.npy, z_*.npy, streamflow_obs.npy), `metrics.json`, `metrics_agg.json`.

**Metrics**: 219 `metrics_agg.json` files, 225 `streamflow.npy` files, 1256 `.pt` checkpoint files.

### 5.2 Analysis outputs (`.tmp/analysis/`)

| Directory | Content |
| --- | --- |
| `flex_mopex_binary_flex_pilot` | Binary pilot analysis |
| `flex_mopex_block1_spatial_seed_diagnostics` | Spatial autocorrelation (Moran's I), seed stability, weight analysis (30+ CSV files, 12 figures) |
| `flex_mopex_loro_corrected_statistics` | LORO corrected statistics |
| `flex_mopex_loro_structure_diagnostics` | LORO structure diagnostics (includes figures/) |

### 5.3 Figure backing data (`manuscript/figures/csv/`)

70+ CSV/JSON files providing the exact data behind every main and appendix figure. Well-organized with metadata.json files.

---

## 6. Scripts inventory

### 6.1 Plotting scripts (`manuscript/plots/`)

| Script | Figure |
| --- | --- |
| `plot_fig01_predictive_performance.py` | Fig 1 |
| `plot_fig02_alpha_generalization.py` | Fig 2 |
| `plot_fig03_spatial_weights.py` | Fig 3 (CONUS maps) |
| `plot_fig04_alpha_weight_response.py` / `_stacked.py` | Fig 4 |
| `plot_fig05_weight_attribute_heatmap.py` | Fig 5 (correlation heatmap) |
| `plot_fig06_weight_attribute_relationships.py` | Fig 6 (binned response) |
| `plot_fig07_gradient_beeswarm.py` | Fig 7 (gradient attribution) |
| `plot_fig08_weight_parameter_relationships.py` | Fig 8 |
| `plot_fig09_alpha_specific_alluvial_bridge_REVISED.py` | Fig 9 (bridge network) |
| `plot_fig10_weight_parameter_regime_gradients.py` | Fig 10 |
| `plot_fig11_parameter_organization.py` | Fig 11 (parameter readout) |
| `plot_fig12_loro_performance.py` | Fig 12 (LORO predictive) |
| `plot_fig13_structural_coordinate_transfer.py` | Fig 13 |
| `plot_fig14_structural_decision_transfer.py` | Fig 14 |
| `plot_fig16_weight_lopo_validation.py` | Fig 16 (LOPO) |
| `common.py`, `make_tableS_attribute_weight_summary.py` | Utilities |

### 6.2 Appendix plotting scripts (`manuscript/plot_appendix/`)

| Script | Figure |
| --- | --- |
| `plot_figA3_metric_tradeoff.py` | Fig A3 |
| `plot_figA4_seed_robustness.py` | Fig A4 |
| `plot_figA5_nmul_ablation.py` | Fig A5 |
| `plot_figA6_threshold_sensitivity.py` | Fig A6 |
| `plot_figA7_readout_stability.py` | Fig A7 |
| `plot_figA8_interception_diagnostic.py` | Fig A8 |
| `build_appendix_figures.py` | Batch builder |
| `appendix_build_common.py`, `appendix_plot_utils.py` | Utilities |

### 6.3 Run/launch scripts (`scripts/`)

- `run_block1_main.sh` - Block1 main experiment
- `run_block1_alpha_path.sh` - Alpha path sweep
- `run_block1_full_lopo.sh` - LOPO experiment
- `run_block1_nmul_ablation.sh` - nmul ablation
- `run_block3_loro_batch.sh`, `run_block3_loro_matrix_nohup.sh` - LORO batch
- `run_binary_pilot.sh`, `run_binary_full_pilot.sh`, `run_binary_extra_alpha.sh` - Binary pilot
- `run_parallel.sh` - Parallel runner
- `run_remote_*.sh` - Remote execution variants (3 scripts)
- `launch_detached.sh`, `launch_single.sh`, `run_extra_detached.sh` - Detached launch
- `run_analysis.sh` - Analysis pipeline
- `build_paper_tables.py` - Table generation
- `collect_full_lopo_ablation.py` - LOPO data collection
- `collect_nmul_ablation.py` - nmul data collection

### 6.4 Model code (`models/`)

| File | Role |
| --- | --- |
| `base_mopex.py` | Base MOPEX model |
| `fixed_weight_mopex.py` | Fixed-weight variant |
| `learned_weight_mopex.py` | CFlex continuous weights |
| `binary_weight_mopex.py` | DFlex binary gates |
| `binary_structure_net.py` | Binary structure network |
| `mopex_core.py` | Core MOPEX ODE implementation |
| `parameter_nets.py` | Parameter network |
| `static_mopex.py` | Static-parameter MOPEX |
| `pub_sampler.py`, `pub_trainer.py` | Training infrastructure |
| `nse_aic_batch_loss.py`, `nse_dyn_aic_batch_loss.py`, `nse_l0_batch_loss.py` | Loss functions |

### 6.5 Analysis scripts (`.tmp/analysis/`)

- `flex_mopex_block1_spatial_seed_diagnostics/run_analysis.py` - Spatial autocorrelation + seed diagnostics (69K, Moran's I computation)

---

## 7. Reusable evidence: what is FOUND

All evidence items below are **FOUND** with high confidence, backed by CSV/JSON data and rendered figures/tables.

### 7.1 Performance-complexity (Items 1-2)

| Evidence | Status | Supporting paths | Notes |
| --- | --- | --- | --- |
| **Basic/Full/CFlex/DFlex basin-level NSE/KGE** | **FOUND** | `table1_performance_complexity_summary.csv`, `results/block1_main/`, `manuscript/figures/csv/figure1_*.csv`, 219 metrics_agg.json files | Full multi-seed per-basin data exists |
| **Lambda path (alpha=0.005, 0.01, 0.03)** | **FOUND** | `results/block1_alpha_path/flex/`, `results/block1_main/flex/`, `tableS2_alpha_tradeoff_summary.csv`, `figure2_alpha_summary_stats.csv` | Full sweep 0.0-0.1 with 3-5 seeds |

### 7.2 Process extension weights (Item 3)

| Evidence | Status | Supporting paths |
| --- | --- | --- |
| **4-process weight table (snow, subsurface, phenology, interception)** | **FOUND** | `table1b_panelB_process_extension_weights.csv`, w_*.npy in all results dirs |

### 7.3 CONUS maps & attributes (Items 4-5)

| Evidence | Status | Supporting paths | Notes |
| --- | --- | --- | --- |
| **CONUS lat/lon or basin attribute table** | **FOUND** | `manuscript/figures/csv/figure6_cflex_attribute_relationships.csv`, references to `basin_attributes.csv` in parameterize project | Attribute data lives in `../parameterize/outputs/analysis/stability_stats/tables/basin_attributes.csv` |
| **Attribute correlations / hydroclimatic controls** | **FOUND** | `tableS4_hydroclimatic_control_summary.csv`, `figure6_cflex_attribute_relationships.csv`, `figure7_gradient_beeswarm.png` data | Full attribute-weight correlation matrix per process per alpha |

### 7.4 LOPO (Item 6)

| Evidence | Status | Supporting paths |
| --- | --- | --- |
| **LOPO leave-one-process-out** | **FOUND** | `results/block1_full_lopo/analysis/lopo_basin_level.csv`, `lopo_seed_level.csv`, `lopo_process_summary.md` |

### 7.5 LORO (Item 7)

| Evidence | Status | Supporting paths | Notes |
| --- | --- | --- | --- |
| **LORO regional transfer** | **FOUND** | `results/block3_loro/`, `manuscript/figures/csv/plot_fig12_loro_performance_*.csv`, `plot_fig13_*.csv`, `plot_fig14_*.csv`, `loro_structural_seed123_456_combined.csv`, Table 3, Tables S6-S8 | 7 regions, base/flex/full, 2 seeds each; predictive + structural transfer |

### 7.6 Seed stability (Item 8)

| Evidence | Status | Supporting paths |
| --- | --- | --- |
| **Seed stability results** | **FOUND** | `tableS3_seed_robustness_summary.csv`, `figA4_seed_robustness_data.csv`, `.tmp/analysis/flex_mopex_block1_spatial_seed_diagnostics/seed_*.csv` (10+ files) |

### 7.7 nmul ablation (Item 9)

| Evidence | Status | Supporting paths |
| --- | --- | --- |
| **nmul ablation** | **FOUND** | `results/block1_nmul_ablation/` (nmul=1/8/16/32), `figA5_nmul_ablation_data.csv`, `tableS9_nmul_ablation_summary.csv` |

### 7.8 Parameter space readout (Item 10)

| Evidence | Status | Supporting paths |
| --- | --- | --- |
| **Parameter space readout** | **FOUND** | `Figure11_` CSVs, `plot_fig11_*.csv`, `figA7_parameter_readout_stability_data.csv`, Tables S5a/b/c |

### 7.9 Moran's I / spatial autocorrelation (Item 11)

| Evidence | Status | Supporting paths | Data |
| --- | --- | --- | --- |
| **Moran's I** | **FOUND** | `.tmp/analysis/flex_mopex_block1_spatial_seed_diagnostics/morans_i_*.csv` (3 files), `fig_morans_i_*.png` (3 figures) | w_snow Moran's I=0.787 at alpha=0.01 (k=8 k-NN, p<0.001 FDR-corrected) |

### 7.10 Ye et al. 2012 comparison (Item 12)

| Evidence | Status | Supporting paths | Notes |
| --- | --- | --- | --- |
| **Ye et al. 2012 reference material** | **FOUND** | `manuscript/paper/introduction.md` (paragraph 2), `discussion.md` (Sect. 3, 4, 9), `supplement.md` (Lines 145-327: parameter mapping tables), `textS11_appendix_and_availability.md` (Line 26: citation) | Extensive textual comparison; no separate Ye-specific data table or quantitative benchmark |

---

## 8. Missing or uncertain evidence

| Item | Status | Notes |
| --- | --- | --- |
| **DFlex seed=1024, 789** | **MISSING** from block1_main | DFlex in block1_main has only seeds 42/123/456 (3 seeds) vs CFlex with 5. Table 1 shows this. |
| **block3_loro flex_region0 seed=42** | **DELETED** (git shows D) | Model/sim files deleted but seed=123 and 456 remain |
| **Ye et al. quantitative benchmark** | **PARTIAL** | Textual comparison is rich (discussion.md paragraph 3-9), parameter tables (supplement.md), but no side-by-side quantitative NSE/process table |
| **No .docx or .pdf manuscript** | **MISSING** | Only .md files exist for the manuscript |
| **No formal equivalency test data** | **UNCLEAR** | Performance data exists but no specific equivalence-test setup |
| **Attribute table location** | **EXTERNAL** | basin_attributes.csv lives in `../parameterize/`, outside flexmopex |

---

## 9. Readiness for equivalency testing

**YES** - Sufficient data exists:
- 671-basin NSE/KGE distributions for all model variants (Basic, Full, CFlex alpha=0.005/0.01/0.03, DFlex)
- Per-basin metrics.json files for 219 run configurations
- All weights (w_snow, w_sub, w_phen, w_int) per basin
- Seed-level data for variability estimation (3-5 seeds per config)
- LORO transfer data for held-out regions
- LOPO process ablation data

Required work: none of this data needs regeneration; it is already in structured CSV/JSON.

---

## 10. Readiness for Ye et al. 2012 comparison

**PARTIALLY** - Qualitative comparison exists abundantly in discussion.md and supplement.md. The manuscript already frames the work against Ye et al. (2012)'s downward modelling approach. However:
- No quantitative side-by-side table comparing e.g. "Ye's active process count by region" vs "Flex-MOPEX continuous weight by region"
- A quantitative benchmark could be constructed from existing data (region-level weight summaries already exist in LORO data)

---

## 11. Potential contamination concerns

| Concern | Severity | Detail |
| --- | --- | --- |
| **Uncommitted untracked files** | **LOW** | `manuscript/`, `logs/`, `results/block1_*`, `.tmp/` are all unstaged but present. No data corruption risk. |
| **Deleted analysis/ files** | **LOW** | Old v1 analysis scripts deleted from index but available in `.tmp/analysis_history/analysis/` |
| **Modified config** | **LOW** | `conf/config_dmopex_v1.yaml` has unstaged changes |
| **Parent repo noise** | **N/A** | Many ../../ deletions unrelated to flexmopex |

No file corruption, no unauthorized modification during this audit.

---

## 12. Git status comparison

| Phase | Status |
| --- | --- |
| **Before audit** | Identical to Section 12 output |
| **After audit** | **IDENTICAL** - no new modifications introduced |

---

## 13. Declaration

**本轮审计未修改任何已有文件。** 所有操作均为只读（find/ls/glob/grep/read）。审计报告是本轮唯一新建的文件 (`audit_current_state.md`)，位于项目根目录。未运行任何训练、绘图或迁移脚本。
