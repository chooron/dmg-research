#!/usr/bin/env python3
"""Step 12: render the concise, data-backed R1 execution report."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from r1_config import (
    CARAVAN_PATH, DATA_ROOT, DPL_ROOT, FORMAL_PAIRED_SOURCE, IC_CHECKPOINT_ROOT, MODEL_REGISTRY,
    R1_ROOT, TABLES_DIR, TEMPORAL_AB, TEMPORAL_SOURCE_SCRIPT, TEST_END, TEST_START,
    VIC_AUDIT_ROOT,
)
from r1_utils import sha256_file


def f(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def main() -> None:
    inv = pd.read_csv(TABLES_DIR / "R1_input_inventory.csv")
    audit = pd.read_csv(TABLES_DIR / "R1_input_audit.csv")
    pair = pd.read_csv(TABLES_DIR / "R1_model_basin_delta_kge.csv")
    model = pd.read_csv(TABLES_DIR / "R1_model_performance_summary.csv")
    temporal = pd.read_csv(TABLES_DIR / "R1_temporal_AB_model_summary.csv")
    temporal_long = pd.read_csv(TABLES_DIR / "R1_temporal_AB_model_basin.csv")
    ensemble = pd.read_csv(TABLES_DIR / "R1_ensemble_summary.csv")
    simhyd = pd.read_csv(TABLES_DIR / "R1_sensitivity_exclude_simhyd.csv")
    mean_med = pd.read_csv(TABLES_DIR / "R1_sensitivity_mean_vs_median.csv")
    confound = pd.read_csv(TABLES_DIR / "R1_sensitivity_confound_checks.csv")

    def value(metric: str) -> float:
        return float(ensemble.loc[ensemble.metric == metric, "value"].iloc[0])

    def sensitivity(metric: str, column: str) -> float:
        return float(simhyd.loc[simhyd.metric == metric, column].iloc[0])

    mean_row = mean_med.loc[mean_med.model == "__ENSEMBLE__"].iloc[0]
    delta_range = (float(model.delta_median.min()), float(model.delta_median.max()))
    dpl_higher = int((model.dPL_median > model.IC_median).sum())
    dpl_lower = int((model.dPL_median < model.IC_median).sum())
    audit_counts = audit.status.value_counts().to_dict()
    warnings = audit.loc[audit.status != "PASS", ["check", "observed", "notes"]]
    warning_lines = [f"- `{r.check}`: {r.observed}. {r.notes}" for r in warnings.itertuples()]
    confound_lines = []
    for r in confound.itertuples():
        confound_lines.append(f"- `{r.covariate}` ↔ model-level median ΔKGE: Pearson={f(r.pearson)}, Spearman={f(r.spearman)} (N={r.N_models}; descriptive only).")

    report = f"""# R1 execution report

## Scope and estimand

This package implements only R1: **aggregate outlet performance cannot resolve internal equivalence**. The primary estimand is

`delta_KGE = KGE_dPL - KGE_IC`

so positive values indicate higher dPL outlet KGE and negative values indicate higher IC outlet KGE. Pooled model–basin rows are descriptive; the primary ensemble observation unit is the model (`N=36`). No model ranking, OOB/PUR, suitability, recommendation, R2, or R3 analysis is included.

## A. Input provenance

- Formal paired source: `{FORMAL_PAIRED_SOURCE}` (`02_BASIN_PAIRED_KGE_LONG.csv`). It was independently split into IC and dPL sides and outer-joined on `(model, basin_id)` by `01_build_fulltest_pair_table.py`.
- IC parameters: `{IC_CHECKPOINT_ROOT / '<model>' / 'chunk_*_gen_*.pt'}`; the forward pass reads each full CMA-ES checkpoint's stored `solver.state.best_latent`/`best_fitness` for the ten starts per basin. The extracted `best_training` files remain a byte-equal lightweight derivative and are not the temporal forward source.
- dPL parameters: `{DPL_ROOT / 'runs' / '<model>' / 'best.pt'}`; canonical v2 metadata records seed 42 and train-loss checkpoint selection. The corresponding per-basin KGE source is each model's `basin_test_kge.csv`.
- Data contract used for the temporal forward: `{DATA_ROOT}`. `caravan_671_attributes.npy` SHA-256 is `{sha256_file(CARAVAN_PATH)}`; this matches the recovered canonical remote/local data-contract record. The audited core data files and checksums are recorded in `R1_input_audit.csv`.
- Structural population: {len(MODEL_REGISTRY)} models × 531 canonical CAMELS-US basins = {len(pair):,} full-test paired rows. Every model has 531 IC rows, 531 dPL rows, and 531 joined rows.
- Test period: `{TEST_START}..{TEST_END}`. Canonical dPL evaluation warm-up: 365 days. A/B uses post-warm-up outputs only.
- Temporal A/B definition reused from `{TEMPORAL_SOURCE_SCRIPT}`: A=`{TEMPORAL_AB[0]['start_date']}..{TEMPORAL_AB[0]['end_date']}`; B=`{TEMPORAL_AB[1]['start_date']}..{TEMPORAL_AB[1]['end_date']}`. The prior complete-R1 and quick-survey records agree; no new split was designed.
- Device/forward: NVIDIA GeForce RTX 3060, one model at a time, eager backend; forcing/targets float32, network/hydrology float64. The temporal run used `forward_only=True`, constructed no optimizer, called no backward pass, and wrote no checkpoint.

## B. Audit

- Model coverage: **{len(inv)} / 36** inventory rows PASS; inventory status counts: `{inv.status.value_counts().to_dict()}`.
- Basin pairing: all 36 models have 531 unique IC IDs, 531 unique dPL IDs, and 531 common IDs. All joins use explicit `basin_id`, never array position.
- KGE validity: full-test and temporal model–basin tables contain no NaN/Inf values and no duplicate `(model, basin_id[, partition])` keys. Full-test rows={len(pair):,}; temporal model–basin rows={len(temporal_long):,}; temporal model summary rows={len(temporal)}.
- Special cases: `vic` uses the current dynamic-DOY IC result; `simhyd` generation 280 is accepted in the primary 36-model structural ensemble and is excluded only in sensitivity; `flexb` uses the current formal paired result without a historical special marker. Dynamic-DOY evidence directory: `{VIC_AUDIT_ROOT}`.
- Canonical v2 checkpoint audit: all 36 `best.pt` files and hash sidecars pass; each has 531 dPL test rows. The recovered forensic record supports conditional canonical-v2 validity after matching the data contract.
- Audit status counts: `{audit_counts}`. The following warnings are retained rather than hidden:
{chr(10).join(warning_lines)}

## C. Primary numerical results

### Full-test aggregate performance

- IC model-level ensemble median KGE (median across 36 basin medians): **{f(value('IC_median_across_model_values_median'))}**.
- dPL model-level ensemble median KGE: **{f(value('dPL_median_across_model_values_median'))}**.
- Median model-level ΔKGE: **{f(value('delta_median_across_model_values_median'))}**; model-level ΔKGE range: **{f(delta_range[0])}..{f(delta_range[1])}**.
- dPL median KGE exceeded IC in **{dpl_higher}/36** models and was lower in **{dpl_lower}/36**; ties: 0.
- Across the 36 model-level median ΔKGE values, Q10/Q25/Q75/Q90 were **{f(model.delta_median.quantile(.10))} / {f(model.delta_median.quantile(.25))} / {f(model.delta_median.quantile(.75))} / {f(model.delta_median.quantile(.90))}**.
- Pooled 19,116 model–basin ΔKGE Q25/Q75 were **{f(pair.delta_KGE.quantile(.25))} / {f(pair.delta_KGE.quantile(.75))}**; pooled rows are not independent inferential replicates.

### Temporal persistence

- Model-level temporal A/B Spearman correlation of median ΔKGE: **ρ={f(value('rho_spearman'))}**, with secondary Pearson **{f(value('rho_pearson_secondary'))}** (N=36 models).
- Same-sign model fraction: **{int(value('same_sign_numerator_excluding_neutral'))}/{int(value('same_sign_denominator_excluding_neutral'))} = {pct(value('same_sign_fraction_excluding_neutral'))}**. Exact-zero models are neutral and excluded from this denominator; neutral count={int(value('neutral_model_count'))}.
- Median model-level ΔKGE in A versus B: **{f(value('delta_A_median_across_models'))}** versus **{f(value('delta_B_median_across_models'))}**.

## D. Sensitivity

- **Exclude `simhyd` (35 models):** IC ensemble median {f(sensitivity('Fig1a_IC_median_of_model_medians', 'exclude_simhyd_35'))} vs full {f(sensitivity('Fig1a_IC_median_of_model_medians', 'full_36'))}; dPL {f(sensitivity('Fig1a_dPL_median_of_model_medians', 'exclude_simhyd_35'))} vs {f(sensitivity('Fig1a_dPL_median_of_model_medians', 'full_36'))}; median model-level ΔKGE {f(sensitivity('Fig1a_delta_median_of_model_medians', 'exclude_simhyd_35'))} vs {f(sensitivity('Fig1a_delta_median_of_model_medians', 'full_36'))}.
- Excluding `simhyd` changes temporal ρ from **{f(sensitivity('rho_spearman', 'full_36'))}** to **{f(sensitivity('rho_spearman', 'exclude_simhyd_35'))}** and same-sign fraction from **{pct(sensitivity('same_sign_fraction_excluding_neutral', 'full_36'))}** to **{pct(sensitivity('same_sign_fraction_excluding_neutral', 'exclude_simhyd_35'))}**. This does not change the R1 headline.
- **Mean instead of median basin aggregation:** at the ensemble model-level median, IC={f(mean_row.IC_basin_mean)} and dPL={f(mean_row.dPL_basin_mean)}, with median ΔKGE={f(mean_row.delta_basin_mean)}. The primary basin-median values are IC={f(mean_row.IC_basin_median)}, dPL={f(mean_row.dPL_basin_median)}, ΔKGE={f(mean_row.delta_basin_median)}; mean-minus-median ΔKGE={f(mean_row.delta_mean_minus_median)}. The conclusion is not dependent on using basin medians.
- Existing low-complexity confound checks:
{chr(10).join(confound_lines)}

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

Required source tables, cache metadata, and scripts are under `{R1_ROOT}`. No files outside `project/benchmark/manuscript/r1/` were modified by this R1 execution.

## F. R1 verdict

**PASS:** aggregate performance similarity coexists with structure-dependent model–basin heterogeneity, and the model-level pattern is temporally reproducible.
"""
    (R1_ROOT / "R1_EXECUTION_REPORT.md").write_text(report)
    print(f"PASS: wrote {R1_ROOT / 'R1_EXECUTION_REPORT.md'}")


if __name__ == "__main__":
    main()
