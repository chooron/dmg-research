# R1 execution report

## Scope and estimand

This package implements only R1: **aggregate outlet performance cannot resolve internal equivalence**. The primary estimand is

`delta_KGE = KGE_dPL - KGE_IC`

so positive values indicate higher dPL outlet KGE and negative values indicate higher IC outlet KGE. Pooled model–basin rows are descriptive; the primary ensemble observation unit is the model (`N=36`). No model ranking, OOB/PUR, suitability, recommendation, R2, or R3 analysis is included.

## A. Input provenance

- Formal paired source: `/home/jingxin/code/dmg-research/project/benchmark/results/ic_dpl_seenbasin_formal_20260901/02_BASIN_PAIRED_KGE_LONG.csv` (`02_BASIN_PAIRED_KGE_LONG.csv`). It was independently split into IC and dPL sides and outer-joined on `(model, basin_id)` by `01_build_fulltest_pair_table.py`.
- IC parameters: `/home/jingxin/code/dmg-research/project/benchmark/results/ic_dpl_aligned_full300_20260819_final/checkpoints/ic_dpl_aligned_full300_20260819/<model>/chunk_*_gen_*.pt`; the forward pass reads each full CMA-ES checkpoint's stored `solver.state.best_latent`/`best_fitness` for the ten starts per basin. The extracted `best_training` files remain a byte-equal lightweight derivative and are not the temporal forward source.
- dPL parameters: `/home/jingxin/code/dmg-research/project/benchmark/results/dpl_canonical_v2_20260831/runs/<model>/best.pt`; canonical v2 metadata records seed 42 and train-loss checkpoint selection. The corresponding per-basin KGE source is each model's `basin_test_kge.csv`.
- Data contract used for the temporal forward: `/home/jingxin/code/dmg-research/data`. `caravan_671_attributes.npy` SHA-256 is `686366653e5cbcac00ac24ecb20b710b4940ccf73fe1957f27f7da1969dfd825`; this matches the recovered canonical remote/local data-contract record. The audited core data files and checksums are recorded in `R1_input_audit.csv`.
- Structural population: 36 models × 531 canonical CAMELS-US basins = 19,116 full-test paired rows. Every model has 531 IC rows, 531 dPL rows, and 531 joined rows.
- Test period: `1995-10-01..2010-09-30`. Canonical dPL evaluation warm-up: 365 days. A/B uses post-warm-up outputs only.
- Temporal A/B definition reused from `/home/jingxin/code/dmg-research/project/benchmark/scripts/diagnostics/r1_r2_nontraining_complete.py`: A=`1995-10-01..2003-03-31`; B=`2003-04-01..2010-09-30`. The prior complete-R1 and quick-survey records agree; no new split was designed.
- Device/forward: NVIDIA GeForce RTX 3060, one model at a time, eager backend; forcing/targets float32, network/hydrology float64. The temporal run used `forward_only=True`, constructed no optimizer, called no backward pass, and wrote no checkpoint.

## B. Audit

- Model coverage: **36 / 36** inventory rows PASS; inventory status counts: `{'PASS': 36}`.
- Basin pairing: all 36 models have 531 unique IC IDs, 531 unique dPL IDs, and 531 common IDs. All joins use explicit `basin_id`, never array position.
- KGE validity: full-test and temporal model–basin tables contain no NaN/Inf values and no duplicate `(model, basin_id[, partition])` keys. Full-test rows=19,116; temporal model–basin rows=38,232; temporal model summary rows=36.
- Special cases: `vic` uses the current dynamic-DOY IC result; `simhyd` generation 280 is accepted in the primary 36-model structural ensemble and is excluded only in sensitivity; `flexb` uses the current formal paired result without a historical special marker. Dynamic-DOY evidence directory: `/home/jingxin/code/dmg-research/project/benchmark/results/ic_vic_full300_dynamic_doy_20260901`.
- Canonical v2 checkpoint audit: all 36 `best.pt` files and hash sidecars pass; each has 531 dPL test rows. The recovered forensic record supports conditional canonical-v2 validity after matching the data contract.
- Audit status counts: `{'PASS': 127, 'WARN': 3}`. The following warnings are retained rather than hidden:
- `formal_manifest_root_flag`: old_manifest_exists=False; current_exists=True. stale root-level flag; per-model files are audited above
- `formal_manifest_root_flag`: old_manifest_exists=False; current_exists=True. stale root-level flag; per-model files are audited above
- `canonical_v2_source_sha`: manifest=3caca37a; checkpoint metadata=7d1132bf; forensic report resolves conditional validity. do not claim byte-identical tracked source; use recovered data contract and checkpoint SHA evidence

## C. Primary numerical results

### Full-test aggregate performance

- IC model-level ensemble median KGE (median across 36 basin medians): **0.621513**.
- dPL model-level ensemble median KGE: **0.609271**.
- Median model-level ΔKGE: **-0.010415**; model-level ΔKGE range: **-0.073383..0.009987**.
- dPL median KGE exceeded IC in **12/36** models and was lower in **24/36**; ties: 0.
- Across the 36 model-level median ΔKGE values, Q10/Q25/Q75/Q90 were **-0.032925 / -0.016294 / -0.006211 / 0.000746**.
- Pooled 19,116 model–basin ΔKGE Q25/Q75 were **-0.060898 / 0.026905**; pooled rows are not independent inferential replicates.

### Temporal persistence

- Model-level temporal A/B Spearman correlation of median ΔKGE: **ρ=0.638095**, with secondary Pearson **0.271020** (N=36 models).
- Same-sign model fraction: **33/36 = 91.7%**. Exact-zero models are neutral and excluded from this denominator; neutral count=0.
- Median model-level ΔKGE in A versus B: **-0.009808** versus **-0.006031**.

## D. Sensitivity

- **Exclude `simhyd` (35 models):** IC ensemble median 0.621425 vs full 0.621513; dPL 0.607820 vs 0.609271; median model-level ΔKGE -0.010776 vs -0.010415.
- Excluding `simhyd` changes temporal ρ from **0.638095** to **0.677871** and same-sign fraction from **91.7%** to **94.3%**. This does not change the R1 headline.
- **Mean instead of median basin aggregation:** at the ensemble model-level median, IC=0.555005 and dPL=0.535508, with median ΔKGE=-0.008880. The primary basin-median values are IC=0.621513, dPL=0.609271, ΔKGE=-0.010415; mean-minus-median ΔKGE=0.001535. The conclusion is not dependent on using basin medians.
- Existing low-complexity confound checks:
- `parameter_count` ↔ model-level median ΔKGE: Pearson=-0.027819, Spearman=0.108446 (N=36; descriptive only).
- `baseline_IC_median` ↔ model-level median ΔKGE: Pearson=-0.028641, Spearman=-0.065894 (N=36; descriptive only).

## E. Figures

- `Fig1_R1_main_final`: the five-panel PNG-only final figure package (aggregate paired performance, model-level ΔKGE distributions, basin-level median effect with cross-model spread, decomposition of ΔKGE variation, and model-level temporal persistence).
- `20_prepare_fig1_data.py` and `26_build_r1_supp_tables.py` prepare the existing-matrix derivative tables, map metadata, and Supplementary Table S1 (S1A, S1B, and Markdown); `32_plot_fig1_final.py` writes only the assembled 600-dpi PNG, and `run_r1_figures.sh` additionally refreshes the build report. No standalone panels or PDFs are generated by the revised package.
- No supplementary figure is regenerated in this revision; existing compact Table S1/S2 support is retained without adding new analysis. No ranking, suitability, attribute, OOB/PUR, R2, or R3 analysis is included.

- The historical R1 forward analysis and its prior draft artifacts are not overwritten by the revised figure renderer.

## Validation and reproducibility

Executed successfully:

- `00_audit_inputs.py` → 36/36 inventory PASS; 127 PASS and 3 declared WARN audit rows.
- `01_build_fulltest_pair_table.py` → explicit outer ID join, 19,116 rows.
- `02_temporal_ab_forward.py` → 36-model GPU forward; 38,232 rows; per-model caches record source/implementation fingerprints and self-hashes, and the subsequent full rerun reused all 36 after integrity validation.
- `03_summarize_r1.py`, `08_run_sensitivity.py`, and the revised PNG-only figure runner → PASS.
- `py_compile` over all R1 scripts and final coverage/finite/parquet/PNG-DPI checks → PASS.

Required source tables, cache metadata, and scripts are under `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r1`. No files outside `project/benchmark/manuscript/r1/` were modified by this R1 execution.

## F. R1 verdict

**PASS:** aggregate performance similarity coexists with structure-dependent model–basin heterogeneity, and the model-level pattern is temporally reproducible.
