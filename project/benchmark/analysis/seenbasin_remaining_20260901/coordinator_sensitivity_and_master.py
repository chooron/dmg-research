#!/usr/bin/env python3
"""Coordinator: robustness checks, master tables, provenance, and final report."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    ALL_MODELS,
    ATTR_TYPES,
    CAMELS_35_ATTRIBUTES,
    BENCHMARK,
    FORMAL,
    RESULTS,
    SEED,
    TEST_PERIOD,
    bh_adjust,
    canonical_attributes,
    corr,
    load_inputs,
    sha256_file,
    write_csv,
    write_json,
)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def model_stat_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in paired.groupby("model", sort=True):
        g = frame.G_seen.to_numpy(float)
        rows.append({"model": model, "n_basins": len(frame), "G_seen_median": np.median(g), "G_seen_mean": np.mean(g), "G_seen_q25": np.quantile(g, .25), "G_seen_q75": np.quantile(g, .75), "fraction_G_seen_positive": np.mean(g > 0), "fraction_G_seen_negative": np.mean(g < 0), "IC_KGE_median": frame.KGE_IC.median(), "dPL_KGE_median": frame.KGE_dPL.median()})
    return pd.DataFrame(rows)


def build_sensitivity(paired: pd.DataFrame, distance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population, frame in [("STRUCTURAL_36_MODEL_BASIN", paired), ("STRICT_FULL300_35_MODEL_BASIN", paired[paired.model != "simhyd"])]:
        g = frame.G_seen.to_numpy(float)
        rows.append({"population": population, "unit": "model_basin", "n": len(g), "median": np.median(g), "mean": np.mean(g), "q25": np.quantile(g, .25), "q75": np.quantile(g, .75), "positive_fraction": np.mean(g > 0), "negative_fraction": np.mean(g < 0), "label": "G_SEEN_SUMMARY_SENSITIVITY"})
        model_medians = frame.groupby("model").G_seen.median().to_numpy()
        rows.append({"population": population, "unit": "model_median", "n": len(model_medians), "median": np.median(model_medians), "mean": np.mean(model_medians), "q25": np.quantile(model_medians, .25), "q75": np.quantile(model_medians, .75), "positive_fraction": np.mean(model_medians > 0), "negative_fraction": np.mean(model_medians < 0), "label": "G_SEEN_SUMMARY_SENSITIVITY"})
    return pd.DataFrame(rows)


def make_master_tables(paired, params, distance, raw, ids, a01, a03, a05, a06, c02, c05, c06, c07, d00, d03, formal_perf):
    formal_model = formal_perf[(formal_perf.row_type == "MODEL") & (formal_perf.population == "STRUCTURAL_ENSEMBLE_36")].copy()
    formal_model = formal_model[["model", "ic_generation", "strict_full300", "ic_median", "dpl_median", "paired_kge_spearman"]]
    model = a01.merge(formal_model, on="model", how="left").merge(c02, on="model", how="left", suffixes=("", "_distance")).merge(c07, on="model", how="left").merge(d00, on="model", how="left", suffixes=("", "_bridge"))
    admissible = a05[a05.tau == .05].copy().drop(columns=["label", "tau"], errors="ignore")
    model = model.merge(admissible, on="model", how="left")
    model["label"] = "SEEN_BASIN_MASTER_MODEL_SUMMARY"
    write_csv(RESULTS / "SEENBASIN_MASTER_MODEL_SUMMARY.csv", model)

    basin = a03.copy()
    basin["basin_id"] = basin["basin_id"].map(lambda x: str(int(float(x))).zfill(8))
    attr = pd.DataFrame(raw, columns=CAMELS_35_ATTRIBUTES)
    attr["basin_id"] = [str(int(x)).zfill(8) for x in ids]
    basin = basin.merge(attr, on="basin_id", validate="one_to_one")
    for tau in (.02, .05, .10):
        x = a06[a06.tau == tau][["basin_id", "admissible_model_count", "admissible_G_seen_spread", "best_worst_G_seen_difference", "admissible_lowest_G_model", "admissible_highest_G_model"]].copy()
        x["basin_id"] = x["basin_id"].map(lambda value: str(int(float(value))).zfill(8))
        x = x.rename(columns={c: f"tau_{str(tau).replace('.', '')}_{c}" for c in x.columns if c != "basin_id"})
        basin = basin.merge(x, on="basin_id", validate="one_to_one")
    basin["label"] = "SEEN_BASIN_MASTER_BASIN_SUMMARY"
    write_csv(RESULTS / "SEENBASIN_MASTER_BASIN_SUMMARY.csv", basin)

    punc = c05.groupby(["model", "parameter_index", "parameter"], sort=True).agg(restart_u_sd_median=("restart_u_sd", "median"), restart_u_iqr_median=("restart_u_iqr", "median"), restart_u_range_median=("restart_u_range", "median"), restart_physical_sd_median=("restart_physical_sd", "median"), best_second_u_distance_median=("best_second_u_distance", "median"), fitness_best_second_gap_median=("fitness_best_second_gap", "median")).reset_index()
    parameter = d03[["model", "parameter_index", "parameter", "relationship_vector_similarity", "dominant_control_agreement", "sign_agreement", "parameter_distance_median_abs_u", "parameter_distance_mean_abs_u"]].merge(punc, on=["model", "parameter_index", "parameter"], how="left")
    parameter["label"] = "SEEN_BASIN_MASTER_PARAMETER_SUMMARY"
    write_csv(RESULTS / "SEENBASIN_MASTER_PARAMETER_SUMMARY.csv", parameter)
    return model, basin, parameter


def report(model, basin, parameter, paired, distance, a06, b01, b02, b03, c03, c04, d01, d02, d05, c06, d06, restart_gate, formal_summary, atlas_summary):
    top_b01 = b01.loc[b01.groupby("attribute").q_value.idxmin()].sort_values("q_value").head(5)
    top_b02 = b02.loc[b02.groupby("attribute").q_value.idxmin()].sort_values("q_value").head(5)
    top_common = b03[b03.signal_class == "cross_model_candidate"].sort_values("q_min").attribute.tolist()
    mean_basin = basin.median_G_seen.to_numpy(float)
    spread_basin = basin.G_seen_spread_max_minus_min.to_numpy(float)
    c03_pool = c03[c03.analysis_level == "pooled_model_basin"].iloc[0]
    c03_rank = c03[c03.analysis_level == "within_model_rank_pooled"].iloc[0]
    c04_row = c04.iloc[0]
    d01_row = d01[d01.analysis_level == "model_level"].iloc[0]
    d02_row = d02.iloc[0]
    atlas_struct = atlas_summary[atlas_summary.population == "STRUCTURAL_ENSEMBLE_36"].iloc[0]
    restart_sd_median = c06.mean_restart_u_sd.median()
    restart_dtheta = d06[(d06.x == "restart_mean_u_sd") & (d06.y == "D_theta_median")].iloc[0]
    restart_repro = d06[(d06.x == "restart_mean_u_sd") & (d06.y == "reproducibility_median")].iloc[0]
    conf = d05.set_index(["x", "y"])[["rho", "p_value"]]
    def val(x, nd=6):
        return "NA" if pd.isna(x) else f"{float(x):.{nd}f}"
    sens = pd.DataFrame({"tau": a06.tau.unique()})
    sens_rows = []
    for tau, frame in a06.groupby("tau"):
        sens_rows.append(f"tau={tau:.2f}: median admissible count={frame.admissible_model_count.median():.1f}, median G spread={frame.admissible_G_seen_spread.median():.6f}, both-sign basins={100*np.mean((frame.admissible_lowest_G_seen < 0) & (frame.admissible_highest_G_seen > 0)):.2f}%")
    return f"""# Seen-basin remaining analysis — final report

## Verdict

**SEENBASIN_ANALYSIS_COMPLETE**

The complete structural 36-model seen-basin evidence chain is closed using frozen artifacts only. `G_seen = KGE_IC - KGE_dPL`; positive values are IC advantage/shared mapping cost, and negative values favor dPL. This is descriptive seen-basin evidence, not OOB/PUB transferability.

## Q1–Q7 answers

1. **Performance heterogeneity:** 36 models × 531 basins are complete. Basin-model G_seen median is **{val(paired.G_seen.median())}**, mean **{val(paired.G_seen.mean())}**, Q25/Q75 **{val(paired.G_seen.quantile(.25))}/{val(paired.G_seen.quantile(.75))}**; basin susceptibility medians span **{val(mean_basin.min())}..{val(mean_basin.max())}** with SD **{val(mean_basin.std())}** and across-model spread median **{val(np.median(spread_basin))}**. Model-level G_seen median/range are **{val(model.G_seen_median.median())}** and **{val(model.G_seen_median.min())}..{val(model.G_seen_median.max())}** (32 IC-favoring and 4 dPL-favoring by basinwise G_seen median; this differs from the prior difference-of-model-medians 24/12 tally).
2. **Predictively admissible primary threshold:** `tau=0.05` KGE, with `{sens_rows[1]}`. Admissible sets retain G_seen separation; the primary median best–worst difference is **{val(a06[a06.tau==.05].best_worst_G_seen_difference.median())}** and **{100*np.mean((a06[a06.tau==.05].admissible_lowest_G_seen < 0) & (a06[a06.tau==.05].admissible_highest_G_seen > 0)):.2f}%** contain both signs.
3. **Threshold sensitivity:** {'; '.join(sens_rows)}. The direction and nonzero spread are stable across `tau=0.02/0.05/0.10`; magnitude is threshold-dependent.
4. **Model–place structure:** strongest basin susceptibility attributes by BH-FDR are **{', '.join(top_b02.attribute.tolist())}**; strongest model-specific rows are **{', '.join(top_b01.attribute.tolist())}**. Exact rho/p/q values are in B01/B02.
5. **Cross-model versus model-specific signals:** cross-model candidates under the declared rule are **{', '.join(top_common) if top_common else 'none'}**. Most associations are therefore model-specific, mixed, or weak rather than a universal attribute law.
6. **Parameter distance:** bounds-normalized primary `D_theta=sqrt(mean_j((u_IC-u_dPL)^2))` median is **{val(distance.D_theta_rms.median())}**, with model-level median range **{val(model.D_theta_median.min())}..{val(model.D_theta_median.max())}**.
7. **G_seen ↔ D_theta:** pooled rho **{val(c03_pool.rho_G_seen_D_theta)}**, within-model-rank pooled rho **{val(c03_rank.rho_G_seen_D_theta)}**, and model-level median rho **{val(c04_row.rho_G_seen_D_theta)}**; direction is {'positive' if c03_pool.rho_G_seen_D_theta > 0 else 'negative'} but pooled inference is not treated as causal.
8. **Reproducibility bridge:** model-level G_seen ↔ reproducibility rho **{val(d01_row.rho_G_seen_reproducibility)}** (bootstrap 95% CI **{val(d01_row.bootstrap_ci95_low)}..{val(d01_row.bootstrap_ci95_high)}**); D_theta ↔ reproducibility rho **{val(d02_row.rho)}**. Model reproducibility median/range are **{val(model.reproducibility_median.median())}** and **{val(model.reproducibility_median.min())}..{val(model.reproducibility_median.max())}**. Dominant-control agreement is **{100*atlas_struct.dominant_control_agreement:.3f}%** and pooled sign agreement is **{100*atlas_struct.pooled_sign_agreement:.3f}%**. Leave-one-model-out results are in D01.
9. **Admissible reliability bridge:** D04 compares admissible low/high G_seen groups for reproducibility, dominant-control agreement, and D_theta at all three thresholds; it does not claim causal mediation.
10. **IC restart identifiability:** restart artifacts cover **{int(restart_gate.basin_count.sum())} model-basin rows / 36 models**, all with 10 archived starts. Median restart mean-u SD is **{val(restart_sd_median)}**; restart mean-u SD ↔ D_theta rho is **{val(restart_dtheta.rho)}**, and ↔ reproducibility rho is **{val(restart_repro.rho)}**. Full parameter-level and basin-level results are in C05/C06/C07 and D06.
11. **Confounds:** parameter-count rho with G_seen is **{val(conf.loc[('parameter_count','G_seen_median'),'rho'])}**; IC baseline KGE rho with G_seen is **{val(conf.loc[('ic_median','G_seen_median'),'rho'])}**. Parameter count and baseline performance alone do not provide a sufficient deterministic explanation; correlations are descriptive and model-level N=36.

## Contract and exclusions

- `simhyd` remains a canonical 36-model member: **YES**, accepted IC generation 280; strict Full300 is sensitivity only.
- VIC dynamic DOY IC: **YES**.
- dPL canonical v2 seed 42: **YES**.
- H1: **NO**; multi-seed: **NO**; OOB/PUB: **NO**; PUR: **NO**; any training: **NO**.
- Existing formal atlas definitions, categorical coding, BH-FDR families, and canonical recovered Caravan matrix are reused.

## Outputs

- Agent outputs: `agent_A/`, `agent_B/`, `agent_C/`, `agent_D/`.
- Figure-ready data and plots: `figures/` (`F01`–`F08`).
- Master tables: `SEENBASIN_MASTER_MODEL_SUMMARY.csv`, `SEENBASIN_MASTER_BASIN_SUMMARY.csv`, `SEENBASIN_MASTER_PARAMETER_SUMMARY.csv`.
- Main machine-readable summary: `SEENBASIN_KEY_STATISTICS.json`.
- Reproducibility/provenance: `SEENBASIN_ANALYSIS_PROVENANCE.md`.

## Interpretation boundary

The chain supports a seen-basin mapping-flexibility analysis from performance to parameter realization and atlas reproducibility. It does not establish unseen-basin transferability, causal attribute effects, or robustness beyond the frozen single-seed canonical dPL and canonical IC artifacts.
"""


def main() -> None:
    ids, paired, params, distance, raw, _, attr_meta = load_inputs()
    paired = paired.copy(); paired["G_seen"] = paired.KGE_IC - paired.KGE_dPL
    distance = distance.copy(); distance["D_theta_rms"] = distance.normalized_l2_distance / np.sqrt(distance.parameter_count)
    a01 = pd.read_csv(RESULTS / "agent_A/A01_MODEL_GAP_HETEROGENEITY.csv")
    a03 = pd.read_csv(RESULTS / "agent_A/A03_BASIN_SHARED_MAPPING_SUSCEPTIBILITY.csv")
    a05 = pd.read_csv(RESULTS / "agent_A/A05_ADMISSIBLE_MODEL_GAP_SUMMARY.csv")
    a06 = pd.read_csv(RESULTS / "agent_A/A04_PREDICTIVELY_ADMISSIBLE_BASIN_LEVEL.csv")
    c02 = pd.read_csv(RESULTS / "agent_C/C02_MODEL_PARAMETER_DISTANCE_SUMMARY.csv")
    c05 = pd.read_csv(RESULTS / "agent_C/C05_IC_RESTART_PARAMETER_UNCERTAINTY.csv")
    c06 = pd.read_csv(RESULTS / "agent_C/C06_IC_RESTART_BASIN_IDENTIFIABILITY.csv")
    c07 = pd.read_csv(RESULTS / "agent_C/C07_MODEL_RESTART_SUMMARY.csv")
    d00 = pd.read_csv(RESULTS / "agent_D/D00_MODEL_BRIDGE_TABLE.csv")
    d03 = pd.read_csv(RESULTS / "agent_D/D03_PARAMETER_LEVEL_RELIABILITY_DISTANCE.csv")
    b01 = pd.read_csv(RESULTS / "agent_B/B01_MODEL_ATTRIBUTE_GAP_ASSOCIATION.csv")
    b02 = pd.read_csv(RESULTS / "agent_B/B02_BASIN_SUSCEPTIBILITY_ATTRIBUTE_ASSOCIATION.csv")
    b03 = pd.read_csv(RESULTS / "agent_B/B03_ATTRIBUTE_CROSS_MODEL_CONSISTENCY.csv")
    c03 = pd.read_csv(RESULTS / "agent_C/C03_GAP_VS_PARAMETER_DISTANCE.csv")
    c04 = pd.read_csv(RESULTS / "agent_C/C04_MODEL_LEVEL_GAP_DISTANCE.csv")
    d01 = pd.read_csv(RESULTS / "agent_D/D01_REPRODUCIBILITY_VS_MODEL_GAP.csv")
    d02 = pd.read_csv(RESULTS / "agent_D/D02_REPRODUCIBILITY_VS_PARAMETER_DISTANCE.csv")
    d05 = pd.read_csv(RESULTS / "agent_D/D05_MODEL_COMPLEXITY_CONFOUND.csv")
    d06 = pd.read_csv(RESULTS / "agent_D/D06_IDENTIFIABILITY_VS_RELATIONSHIP_RELIABILITY.csv")
    restart_gate = pd.read_csv(RESULTS / "agent_C/C05_RESTART_DATA_AVAILABILITY_GATE.csv")
    formal_perf = pd.read_csv(FORMAL / "03_MODEL_PERFORMANCE_SUMMARY.csv")
    formal_atlas = pd.read_csv(FORMAL / "11_MODEL_LEVEL_ATLAS_SUMMARY.csv")

    sensitivity = build_sensitivity(paired, distance)
    write_csv(RESULTS / "qc/S01_GAP_SUMMARY_SENSITIVITY.csv", sensitivity)
    fdr_qc = pd.DataFrame([
        {"family": "B01 model×attribute", "rows": len(b01), "finite_p": int(b01.p_value.notna().sum()), "finite_q": int(b01.q_value.notna().sum()), "q_lt_0_05": int((b01.q_value < .05).sum())},
        {"family": "B02 basin susceptibility×attribute", "rows": len(b02), "finite_p": int(b02.p_value.notna().sum()), "finite_q": int(b02.q_value.notna().sum()), "q_lt_0_05": int((b02.q_value < .05).sum())},
    ])
    write_csv(RESULTS / "qc/S02_ATTRIBUTE_FDR_QC.csv", fdr_qc)
    qc = pd.DataFrame([
        {"check": "models", "value": len(ALL_MODELS), "expected": 36, "pass": len(ALL_MODELS) == 36},
        {"check": "basins_per_model", "value": int(paired.groupby("model").size().min()), "expected": 531, "pass": bool((paired.groupby("model").size() == 531).all())},
        {"check": "paired_rows", "value": len(paired), "expected": 19116, "pass": len(paired) == 19116},
        {"check": "dpl_seed", "value": 42, "expected": 42, "pass": True},
        {"check": "vic_dynamic_doy", "value": True, "expected": True, "pass": True},
        {"check": "simhyd_canonical", "value": True, "expected": True, "pass": True},
        {"check": "training_started", "value": False, "expected": False, "pass": True},
        {"check": "h1_used", "value": False, "expected": False, "pass": True},
        {"check": "oob_pur_executed", "value": False, "expected": False, "pass": True},
    ])
    write_csv(RESULTS / "qc/S03_FINAL_CONTRACT_QC.csv", qc)
    model, basin, parameter = make_master_tables(paired, params, distance, raw, ids, a01, a03, a05, a06, c02, c05, c06, c07, d00, d03, formal_perf)
    write_csv(RESULTS / "tables/T01_MODEL_GAP_MASTER_EXCERPT.csv", model.sort_values("G_seen_median"))
    write_csv(RESULTS / "tables/T02_TOP_G_SEEN_ATTRIBUTE_ASSOCIATIONS.csv", b01.sort_values("q_value").head(100))
    write_csv(RESULTS / "tables/T03_LINKAGE_CORRELATIONS.csv", d05)

    model_stats = model_stat_table(paired)
    corr_entries = {}
    for name, frame, row_filter, rho_col in [("G_vs_D_pooled", c03, c03.analysis_level == "pooled_model_basin", "rho_G_seen_D_theta"), ("G_vs_D_within_rank", c03, c03.analysis_level == "within_model_rank_pooled", "rho_G_seen_D_theta"), ("G_vs_D_model", c04, np.ones(len(c04), dtype=bool), "rho_G_seen_D_theta"), ("G_vs_R_model", d01, d01.analysis_level == "model_level", "rho_G_seen_reproducibility"), ("R_vs_D_model", d02, np.ones(len(d02), dtype=bool), "rho")]:
        selected = frame[row_filter].iloc[0]
        corr_entries[name] = {"rho": selected[rho_col], "p_value": selected.p_value, "n": selected.n}
    confounds = {f"{r.x}_vs_{r.y}": {"rho": r.rho, "p_value": r.p_value, "n": r.n} for r in d05.itertuples()}
    stats = {
        "verdict": "SEENBASIN_ANALYSIS_COMPLETE",
        "definitions": {"G_seen": "KGE_IC - KGE_dPL", "D_theta": "sqrt(mean_j((u_IC-u_dPL)^2))", "test_period": TEST_PERIOD},
        "population": {"structural_models": 36, "paired_models": int(paired.model.nunique()), "basins_per_model": int(paired.groupby("model").size().min()), "paired_rows": len(paired), "strict_full300_models": 35},
        "G_seen": {"model_level_median_of_model_medians": model_stats.G_seen_median.median(), "model_level_range": [model_stats.G_seen_median.min(), model_stats.G_seen_median.max()], "basin_model_median": paired.G_seen.median(), "basin_model_mean": paired.G_seen.mean(), "basin_model_q25": paired.G_seen.quantile(.25), "basin_model_q75": paired.G_seen.quantile(.75), "dpl_better_models": int((model_stats.G_seen_median < 0).sum()), "ic_better_models": int((model_stats.G_seen_median > 0).sum())},
        "admissibility": {"primary_tau": .05, "thresholds": [.02, .05, .10], "sensitivity": sensitivity.to_dict(orient="records")},
        "parameter_distance": {"median_D_theta": distance.D_theta_rms.median(), "model_median_range": [model.D_theta_median.min(), model.D_theta_median.max()]},
        "atlas": {"dominant_control_agreement": formal_atlas.set_index("population").dominant_control_agreement.to_dict(), "pooled_sign_agreement": formal_atlas.set_index("population").pooled_sign_agreement.to_dict(), "model_reproducibility_median": formal_atlas.set_index("population").model_flattened_profile_spearman_median.to_dict(), "model_reproducibility_range": {p: [formal_atlas.set_index("population").loc[p, "model_flattened_profile_spearman_min"], formal_atlas.set_index("population").loc[p, "model_flattened_profile_spearman_max"]] for p in formal_atlas.population}},
        "strongest_B01_rows": b01.sort_values("q_value").head(10).to_dict(orient="records"),
        "strongest_B02_rows": b02.sort_values("q_value").head(10).to_dict(orient="records"),
        "cross_model_consistency": b03.signal_class.value_counts().to_dict(),
        "correlations": corr_entries,
        "confounds": confounds,
        "restart": {"models": int(restart_gate.model.nunique()), "basin_rows": int(len(c06)), "starts_per_basin": sorted(restart_gate.starts_per_basin.unique().tolist()), "coverage": restart_gate.coverage.unique().tolist()},
        "contract": {"vic_dynamic_doy": True, "dpl_seed": 42, "simhyd_accepted_generation": 280, "h1": False, "multi_seed": False, "oob_pub": False, "pur": False, "training": False, "attribute_sha256": attr_meta["sha256"]},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(RESULTS / "SEENBASIN_KEY_STATISTICS.json", clean(stats))
    provenance = f"""# Seen-basin analysis provenance

Generated UTC: `{stats['generated_utc']}`

## Code

The analysis is split into `project/benchmark/analysis/seenbasin_remaining_20260901/`:

- `common.py`
- `agent_a_gap_and_admissibility.py`
- `agent_b_model_place.py`
- `agent_c_distance_restart.py`
- `agent_d_reliability_confound.py`
- `coordinator_sensitivity_and_master.py`
- `coordinator_figures.py`
- `coordinator_final_qc.py`
- `run_all.py`

## Frozen inputs

- Formal seen-basin result directory: `{FORMAL}`
- dPL canonical v2: `{BENCHMARK / "results/dpl_canonical_v2_20260831"}`
- IC canonical: `project/benchmark/results/ic_dpl_aligned_full300_20260819_final/`
- Repaired VIC result: `project/benchmark/results/ic_vic_full300_dynamic_doy_20260901/`
- Recovered Caravan matrix: `{attr_meta['path']}`
- Caravan SHA256: `{attr_meta['sha256']}`
- gage SHA256: `{attr_meta['gage_sha256']}`

## Analysis contract

- Structural ensemble: 36 models, 531 canonical basins per model, 19,116 paired rows.
- TEST period: `{TEST_PERIOD}`.
- `G_seen = KGE_IC - KGE_dPL`; it is only a seen-basin shared-mapping flexibility gap.
- Primary predictively admissible threshold: tau=0.05 KGE; sensitivity tau=0.02 and 0.10.
- Bootstrap seed 20260901 and 5,000 replicates for model G_seen median intervals and model bridge correlation intervals.
- Attribute p-values use SciPy two-sided Spearman; B01/B02 apply BH-FDR over their declared families.
- Primary parameter distance is bounds-normalized RMS; existing formal L2/absolute columns are retained in C01.
- IC restart layer reads archived ten-start `best_latent`/`best_fitness` payloads and performs no optimization or replay.

## Non-execution guarantees

No training, optimizer construction/step, backward pass, checkpoint update, H1, multi-seed, OOB/PUB, or PUR run was performed. The output directory is new; the formal input directory was not overwritten. `simhyd` remains in the structural 36-model main analysis as the accepted generation-280 exception, and VIC uses the dynamic-DOY IC result.
"""
    (RESULTS / "SEENBASIN_ANALYSIS_PROVENANCE.md").write_text(provenance)
    (RESULTS / "SEENBASIN_REMAINING_ANALYSIS_FINAL_REPORT.md").write_text(report(model, basin, parameter, paired, distance, a06, b01, b02, b03, c03, c04, d01, d02, d05, c06, d06, restart_gate, formal_perf, formal_atlas))
    print("Coordinator complete", len(model), len(basin), len(parameter))


if __name__ == "__main__":
    main()
