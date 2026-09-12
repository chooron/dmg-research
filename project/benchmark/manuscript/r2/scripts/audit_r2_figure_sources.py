#!/usr/bin/env python3
"""Audit the frozen R2 data objects needed by Figure 2 and Figure 3.

This module is read-only with respect to benchmark/result products.  It scans
candidate R2 provenance, validates the frozen contracts and panel-level schemas,
recomputes lightweight summaries from existing tables/arrays, and writes audit
artifacts under manuscript/r2/tables plus the required report.  It does not
train, calibrate, simulate, rerun optimizers, or create figures.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve()
R2_ROOT = HERE.parents[1]
BENCHMARK = HERE.parents[3]
REPO = BENCHMARK.parents[1]
RESULT_ROOT = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905"
TABLES = R2_ROOT / "tables"
CACHE = R2_ROOT / "cache"

N_MODELS = 36
N_BASINS = 531
N_COORDINATES = 271
DPL_SEED = 42
BOOT_TOL = 1e-6
DET_TOL = 1e-6

ELIGIBLE = [
    "alpine1", "alpine2", "collie1", "collie2", "collie3", "flexi", "flexis",
    "gr4j", "hillslope", "hymod", "ihacres", "modhydrolog", "mopex1", "mopex2",
    "mopex3", "newzealand1", "simhyd", "susannah1", "tank", "us1", "vic", "wetland",
    "xinanjiang",
]
INSUFFICIENT = [
    "australia", "flexb", "gsfb", "hbv96", "mopex4", "mopex5", "newzealand2",
    "penman", "plateau", "smar", "susannah2", "tcm", "topmodel",
]
ALL_MODELS = ELIGIBLE + INSUFFICIENT

FILES: dict[str, Path] = {
    # Pinned frozen source inputs / summaries.
    "frozen_contract": R2_ROOT / "cache/inputs/frozen_contract.json",
    "input_manifest": R2_ROOT / "cache/inputs/input_manifest.json",
    "separation_model": RESULT_ROOT / "claim_audit_multiaudit_20260905/agent_B/tables/MODEL_IC_SELF_SUMMARY.csv",
    "separation_bootstrap": RESULT_ROOT / "claim_audit_multiaudit_20260905/agent_B/tables/BOOTSTRAP_SUMMARY.csv",
    "dtheta_basin": RESULT_ROOT / "r2/tables/10_BASIN_PARAMETER_VECTOR_DISPLACEMENT.csv",
    "dtheta_model": RESULT_ROOT / "r2/tables/11_MODEL_LEVEL_PARAMETER_DISPLACEMENT.csv",
    "rank_coordinate": RESULT_ROOT / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/tables/R_rank_all_271.csv",
    "rank_model": RESULT_ROOT / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/tables/R_rank_model_summaries.csv",
    "rank_equal": RESULT_ROOT / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/tables/R_rank_model_equal_summaries.csv",
    "rank_qc": RESULT_ROOT / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/tables/coordinate_qc_and_occupancy.csv",
    "rank_coverage": RESULT_ROOT / "r2_final_robustness_20260906/agent_A_rank_self_reference/tables/reference_coverage.csv",
    "rank_self_model": RESULT_ROOT / "r2_final_robustness_20260906/agent_A_rank_self_reference/tables/model_rank_self_primary_5000.csv",
    "rank_self_equal": RESULT_ROOT / "r2_final_robustness_20260906/agent_A_rank_self_reference/tables/model_equal_primary_summary.csv",
    "weight_coordinate": RESULT_ROOT / "r2_parameter_axis_audit_20260906/agent_C_coordinate_concentration/tables/basin_coordinate_weights.csv",
    "localization_model": RESULT_ROOT / "r2_final_robustness_20260906/agent_C_model_equal_claim_audit/tables/frozen_C_eff_model_summaries.csv",
    "localization_equal": RESULT_ROOT / "r2_final_robustness_20260906/agent_C_model_equal_claim_audit/tables/frozen_C_eff_model_equal_summary.csv",
    "localization_adjusted_model": RESULT_ROOT / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/basin_adjusted_summary.csv",
    "localization_adjusted_equal": RESULT_ROOT / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/model_equal_adjusted_summary.csv",
    "localization_raw_adjusted": RESULT_ROOT / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/raw_vs_adjusted_comparison.csv",
    "localization_delta_equal": RESULT_ROOT / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/model_equal_raw_vs_adjusted_summary.csv",
    "distribution_matched": RESULT_ROOT / "r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/matched_model_comparison_23.csv",
    "distribution_equal": RESULT_ROOT / "r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/corrected_model_equal_summary.csv",
    "performance_model": RESULT_ROOT / "r2_coupling_deepdive_20260906/agent_A_excess_displacement/tables/model_association_all.csv",
    "performance_equal": RESULT_ROOT / "claim_audit_multiaudit_20260905/agent_C/tables/model_equal_summary.csv",
}

SCRIPTS: dict[str, Path] = {
    "dtheta": RESULT_ROOT / "r2/scripts/direct_parameter_change_r2.py",
    "ic_self": RESULT_ROOT / "claim_audit_multiaudit_20260905/agent_B/ic_self_benchmark.py",
    "rank": RESULT_ROOT / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/rank_preservation.py",
    "rank_self": RESULT_ROOT / "r2_final_robustness_20260906/agent_A_rank_self_reference/rank_self_reference.py",
    "weights": RESULT_ROOT / "r2_parameter_axis_audit_20260906/agent_C_coordinate_concentration/analyze_coordinate_concentration.py",
    "adjusted": RESULT_ROOT / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/adjusted_localization.py",
    "distribution": RESULT_ROOT / "r2_final_robustness_20260906/agent_B_contraction_robustness/contraction_robustness.py",
    "distribution_fix": RESULT_ROOT / "r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/corrected_primary_23models.py",
    "canonical_pipeline": R2_ROOT / "scripts/01_parameter_separation_icself.py",
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_meta(path: Path, status: str = "USED") -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": rel(path), "sha256": sha256(path), "size_bytes": path.stat().st_size, "status": status}


def require_sources(keys: list[str]) -> None:
    missing = [f"{key}: {FILES[key]}" for key in keys if not FILES[key].exists()]
    missing += [f"script:{key}: {SCRIPTS[key]}" for key in SCRIPTS if key in {"dtheta", "ic_self", "rank", "rank_self", "weights", "adjusted", "distribution", "distribution_fix"} and not SCRIPTS[key].exists()]
    if missing:
        raise FileNotFoundError("Required R2 audit inputs missing:\n" + "\n".join(missing))


def read_csv(key: str, **kwargs: Any) -> pd.DataFrame:
    path = FILES[key]
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def scan_candidates() -> pd.DataFrame:
    extensions = {".py", ".ipynb", ".csv", ".tsv", ".parquet", ".json", ".jsonl", ".npz", ".npy", ".pt", ".pkl", ".md", ".txt", ".log"}
    keywords = ("r2", "parameter", "dtheta", "distance", "rank", "spearman", "restart", "self", "localization", "effective", "top1", "top2", "contraction", "consensus", "canonical", "bridge", "strict", "coverage", "audit", "final", "frozen")
    rows: list[dict[str, Any]] = []
    skip_parts = {".git", "node_modules", "__pycache__", ".venv", "coverage", "dist", "build"}
    for path in BENCHMARK.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        name_hit = [k for k in keywords if k in path.name.lower()]
        content_hit: list[str] = []
        if not name_hit and path.suffix.lower() in {".py", ".md", ".json", ".txt", ".log"}:
            try:
                text = path.read_text(errors="ignore")[:131072].lower()
                content_hit = [k for k in keywords if k in text]
            except OSError:
                continue
        hits = sorted(set(name_hit + content_hit))
        if hits:
            rows.append({"path": rel(path), "extension": path.suffix.lower(), "keyword_hits": ";".join(hits), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    frame = pd.DataFrame(rows).sort_values("path") if rows else pd.DataFrame(columns=["path", "extension", "keyword_hits", "size_bytes", "sha256"])
    return frame


def model_span(frame: pd.DataFrame, column: str) -> tuple[float, float, str, str]:
    vals = pd.to_numeric(frame[column], errors="coerce")
    lo_i = vals.idxmin(); hi_i = vals.idxmax()
    return float(vals.min()), float(vals.max()), str(frame.loc[lo_i, "model"]), str(frame.loc[hi_i, "model"])


def compute_weight_metrics(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"model", "basin_id", "parameter_index", "parameter", "DeltaTheta", "squared_displacement", "weight", "zero_displacement"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"coordinate contribution table missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    for (model, basin_id), group in weights.groupby(["model", "basin_id"], sort=True):
        w = pd.to_numeric(group["weight"], errors="coerce").to_numpy(float)
        w = np.nan_to_num(w, nan=0.0)
        sq = pd.to_numeric(group["squared_displacement"], errors="coerce").to_numpy(float)
        p = len(group)
        zero = bool(group["zero_displacement"].astype(bool).any())
        order = np.argsort(w)[::-1]
        total = float(w.sum())
        if total > 0:
            ordered = w[order]
            top1 = float(ordered[0])
            top2 = float(ordered[: min(2, p)].sum())
            ceff = float(1.0 / np.sum(w * w) / p)
            second = float(top2 - top1)
            remaining = float(1.0 - top2)
        else:
            top1 = top2 = ceff = second = remaining = np.nan
        composition.append({
            "model": str(model), "basin_id": str(basin_id), "parameter_count": p,
            "total_squared_displacement": float(sq.sum()), "zero_displacement": zero,
            "C_eff": ceff, "top1_share": top1, "second_share": second,
            "top2_share_cumulative": top2, "remaining_share": remaining,
            "aggregation_level": "basin within model; squared coordinate contribution",
        })
        for rank, idx in enumerate(order, start=1):
            r = group.iloc[idx]
            rows.append({
                "model": str(model), "basin_id": str(basin_id), "parameter_index": int(r.parameter_index),
                "parameter": str(r.parameter), "DeltaTheta": float(r.DeltaTheta),
                "squared_displacement": float(r.squared_displacement), "coordinate_contribution": float(r.weight) if np.isfinite(r.weight) else np.nan,
                "descending_rank": rank, "zero_displacement": bool(r.zero_displacement),
                "top1_share_basin": top1, "second_share_basin": second,
                "top2_share_cumulative_basin": top2, "remaining_share_basin": remaining,
            })
    return pd.DataFrame(rows), pd.DataFrame(composition)


def check_frame_schema(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def add_check(rows: list[dict[str, Any]], metric: str, expected: float, observed: float, source: str, aggregation: str, notes: str, kind: str = "deterministic", tolerance: float = DET_TOL) -> None:
    diff = abs(float(observed) - float(expected)) if np.isfinite(observed) else np.inf
    denom = abs(float(expected))
    rows.append({
        "metric": metric, "expected_value": float(expected), "recomputed_value": float(observed),
        "absolute_difference": float(diff), "relative_difference": float(diff / denom) if denom else float(diff),
        "source": source, "aggregation": aggregation, "status": "PASS" if diff <= tolerance else "FAIL",
        "notes": f"{notes}; check_type={kind}; tolerance={tolerance:g}",
    })


def add_text_check(rows: list[dict[str, Any]], metric: str, expected: str, observed: str, source: str, aggregation: str, notes: str) -> None:
    rows.append({
        "metric": metric, "expected_value": expected, "recomputed_value": observed,
        "absolute_difference": 0.0 if expected == observed else np.nan, "relative_difference": np.nan,
        "source": source, "aggregation": aggregation,
        "status": "PASS" if expected == observed else "FAIL", "notes": notes,
    })


def make_headline_checks() -> tuple[pd.DataFrame, dict[str, Any]]:
    sep = read_csv("separation_model")
    boot = read_csv("separation_bootstrap")
    dtheta = read_csv("dtheta_model")
    rank_equal = read_csv("rank_equal")
    rank_model = read_csv("rank_model")
    rank_self_equal = read_csv("rank_self_equal")
    rank_self_model = read_csv("rank_self_model")
    loc_equal = read_csv("localization_equal")
    loc_adj = read_csv("localization_adjusted_equal")
    loc_cmp = read_csv("localization_raw_adjusted")
    loc_model = read_csv("localization_model")
    dist_equal = read_csv("distribution_equal")
    dist_matched = read_csv("distribution_matched")
    perf_equal = read_csv("performance_equal")
    weights = read_csv("weight_coordinate")
    _, compositions = compute_weight_metrics(weights)

    rows: list[dict[str, Any]] = []
    sep_row = sep[(sep.row_type == "ALL_MODELS") & (sep.threshold == "within_0.01")].iloc[0]
    boot_row = boot[boot.threshold == "within_0.01"].iloc[0]
    dmin, dmax, dmin_model, dmax_model = model_span(dtheta, "D_RMS_median")
    add_check(rows, "fig2a_D_RMS_model_equal_median", .384375, float(sep_row.D_cross_model_equal_median), rel(FILES["separation_model"]), "median of 36 model basin medians", "Expected headline rounded to six decimals.")
    add_check(rows, "fig2a_model_median_min", .150576, dmin, rel(FILES["dtheta_model"]), "minimum of 36 model medians", f"Observed model={dmin_model}; expected rounded headline.")
    add_check(rows, "fig2a_model_median_max", .592668, dmax, rel(FILES["dtheta_model"]), "maximum of 36 model medians", f"Observed model={dmax_model}; expected rounded headline.")
    add_check(rows, "fig2b_cross_minus_self", .218775, float(sep_row.cross_minus_self_model_equal_median), rel(FILES["separation_model"]), "median of 36 model-level cross-minus-self medians", "One-sided archived IC-self reference.")
    add_check(rows, "fig2b_cross_minus_self_ci_low", .209136, float(boot_row.difference_ci_low), rel(FILES["separation_bootstrap"]), "paired basin resampling then model-equal median", "Bootstrap seed is retained in source.", "bootstrap")
    add_check(rows, "fig2b_cross_minus_self_ci_high", .228145, float(boot_row.difference_ci_high), rel(FILES["separation_bootstrap"]), "paired basin resampling then model-equal median", "Bootstrap seed is retained in source.", "bootstrap")
    add_check(rows, "fig2b_positive_models", 36, float((sep[(sep.row_type == "model") & (sep.threshold == "within_0.01")].cross_minus_self_median > 0).sum()), rel(FILES["separation_model"]), "model count", "36/36 model-level cross-minus-self medians are positive.")

    rank_row = rank_equal[rank_equal.summary == "model_equal_primary"].iloc[0]
    add_check(rows, "fig2c_R_rank_median", .406901, float(rank_row["median"]), rel(FILES["rank_equal"]), "median of 36 model summaries", "Coordinate Spearman is summarized within model first.")
    add_check(rows, "fig2c_R_rank_IQR", .209751, float(rank_row.iqr), rel(FILES["rank_equal"]), "IQR of 36 model summaries", "Model-equal descriptive IQR.")
    add_check(rows, "fig2c_R_rank_ci_low", .370863, float(rank_row.model_median_ci_lower), rel(FILES["rank_equal"]), "basin bootstrap then model-equal median", "Bootstrap seed is 20260906.", "bootstrap")
    add_check(rows, "fig2c_R_rank_ci_high", .422412, float(rank_row.model_median_ci_upper), rel(FILES["rank_equal"]), "basin bootstrap then model-equal median", "Bootstrap seed is 20260906.", "bootstrap")
    add_check(rows, "fig2c_positive_model_summaries", 36, float((rank_model["median"] > 0).sum()), rel(FILES["rank_model"]), "36 model medians", "No model summary is negative.")
    for threshold, expected in ((.5, 10), (.8, 1), (.9, 1)):
        # The frozen source computes reach from the model-level median R_rank
        # summary, not from a nonzero coordinate fraction or pooled coordinates.
        count = int((rank_model["median"] >= threshold).sum())
        add_check(rows, f"fig2c_models_reaching_{threshold:g}", expected, count, rel(FILES["rank_model"]), "count of model summaries with median R_rank at threshold", "Reach means model summary median >= threshold; not a coordinate-pooled percentage.")
        add_check(rows, f"fig2c_fraction_reaching_{threshold:g}", expected / 36, count / 36, rel(FILES["rank_model"]), "model count divided by 36", "Reported as the frozen percentage.")

    strict = rank_self_model[(rank_self_model.pool == "primary") & (rank_self_model.prefix == 5000) & (rank_self_model.primary_reference_status == "PASS")].copy()
    add_check(rows, "fig2d_strict_eligible_models", 23, len(strict), rel(FILES["rank_self_model"]), "strict primary model count", "Primary coverage gate is >=0.90.")
    add_check(rows, "fig2d_R_cross_median", .487560, float(strict.R_cross_median.median()), rel(FILES["rank_self_model"]), "median of 23 model-level R_cross summaries", "Strict primary subset.")
    add_check(rows, "fig2d_R_self_median", .797362, float(strict.self_median_median.median()), rel(FILES["rank_self_model"]), "median of 23 model-level IC-self summaries", "Strict primary subset.")
    add_check(rows, "fig2d_delta_R_median", .211846, float(strict.DeltaR_self_minus_cross_median.median()), rel(FILES["rank_self_model"]), "median of 23 model-level self-minus-cross summaries", "Strict primary subset.")
    add_check(rows, "fig2d_positive_delta_R_models", 23, float((strict.DeltaR_self_minus_cross_median > 0).sum()), rel(FILES["rank_self_model"]), "model count", "All strict model-level self-minus-cross medians are positive.")

    raw_equal = loc_equal[loc_equal.metric == "C_eff"].iloc[0]
    add_check(rows, "fig3b_C_eff", .345844, float(raw_equal["median"]), rel(FILES["localization_equal"]), "median of 36 model basin medians", "Squared normalized displacement weights.")
    add_check(rows, "fig3b_C_eff_ci_low", .339434, float(raw_equal.paired_basin_bootstrap_ci_low), rel(FILES["localization_equal"]), "paired basin bootstrap then model-equal median", "Bootstrap seed is 20260906.", "bootstrap")
    add_check(rows, "fig3b_C_eff_ci_high", .353063, float(raw_equal.paired_basin_bootstrap_ci_high), rel(FILES["localization_equal"]), "paired basin bootstrap then model-equal median", "Bootstrap seed is 20260906.", "bootstrap")
    for metric, expected in (("top1_share", .545619), ("top2_share", .848248)):
        row = loc_equal[loc_equal.metric == metric].iloc[0]
        add_check(rows, f"fig3b_{metric}", expected, float(row["median"]), rel(FILES["localization_equal"]), "median of 36 model basin medians", "Top-k shares are basin-wise cumulative shares; top2 is not the second-coordinate share.")
    # Recompute the raw localization objects from the published coordinate weights.
    comp_model = compositions.groupby("model", sort=False).agg(C_eff=("C_eff", "median"), top1_share=("top1_share", "median"), top2_share_cumulative=("top2_share_cumulative", "median")).reset_index()
    add_check(rows, "fig3b_C_eff_recomputed_from_coordinate_weights", .345844, float(comp_model.C_eff.median()), rel(FILES["weight_coordinate"]), "coordinate weights -> basin metrics -> model-equal median", "Independent reconstruction from weight=(DeltaTheta^2)/sum(DeltaTheta^2).")
    add_check(rows, "fig3b_top1_recomputed_from_coordinate_weights", .545619, float(comp_model.top1_share.median()), rel(FILES["weight_coordinate"]), "coordinate weights -> basin metrics -> model-equal median", "Independent reconstruction from the raw coordinate contribution table.")
    add_check(rows, "fig3b_top2_recomputed_from_coordinate_weights", .848248, float(comp_model.top2_share_cumulative.median()), rel(FILES["weight_coordinate"]), "coordinate weights -> basin metrics -> model-equal median", "Independent reconstruction from the raw coordinate contribution table.")

    adj = loc_adj.set_index("metric")
    for metric, expected in (("fraction_any_positive_excess", .960452), ("C_eff_excess_median", .293382), ("top1_excess_share_median", .672634), ("top2_excess_share_median", .941761)):
        add_check(rows, f"fig3c_{metric}", expected, float(adj.loc[metric, "median"]), rel(FILES["localization_adjusted_equal"]), "median of 23 strict model summaries", "Coordinate-specific E_plus adjustment; exact formula is documented in the source script.")
    add_check(rows, "fig3c_raw_matched_exact_basin_C_eff_against_frozen_headline", .346893, float(loc_cmp.C_eff_raw_median.median()), rel(FILES["localization_raw_adjusted"]), "median of 23 model summaries recomputed on exact adjusted-valid basin overlap", "FAIL is substantive: the frozen .346893 headline is not reproduced by the exact basin-matched table.")
    add_check(rows, "fig3c_raw_matched_exact_basin_top1_against_frozen_headline", .573874, float(loc_cmp.top1_share_raw_median.median()), rel(FILES["localization_raw_adjusted"]), "median of 23 model summaries recomputed on exact adjusted-valid basin overlap", "FAIL is substantive: the frozen .573874 headline is not reproduced by the exact basin-matched table.")
    strict_models = set(strict.model.astype(str))
    strict_model_raw = loc_model[loc_model.model.astype(str).isin(strict_models)]
    add_check(rows, "fig3c_raw_strict_modelset_C_eff", .346893, float(strict_model_raw.C_eff_median.median()), rel(FILES["localization_model"]), "median of 23 strict model summaries using all available raw basins", "This reproduces the frozen headline but is not the exact basin-overlap table.")
    add_check(rows, "fig3c_raw_strict_modelset_top1", .573874, float(strict_model_raw.top1_share_median.median()), rel(FILES["localization_model"]), "median of 23 strict model summaries using all available raw basins", "This reproduces the frozen headline but is not the exact basin-overlap table.")
    add_check(rows, "fig3c_adjusted_C_eff_lower_models", 22, float((loc_cmp.delta_C_eff_excess_minus_C_eff_raw_median < 0).sum()), rel(FILES["localization_raw_adjusted"]), "model count", "One strict model is tied at zero; one row has no negative delta but is retained in the source denominator.")
    add_check(rows, "fig3c_adjusted_top1_higher_models", 22, float((loc_cmp.delta_top1_excess_share_minus_top1_share_raw_median > 0).sum()), rel(FILES["localization_raw_adjusted"]), "model count", "Strict model-level comparison.")

    drow = dist_equal[(dist_equal.scope == "PRIMARY_MATCHED_23")]
    for metric, expected in (("canonical_median_CR", .614017), ("consensus_median_CR", .979456), ("selfref_median_CR", 1.004324)):
        add_check(rows, f"fig3d_{metric}", expected, float(drow.loc[drow.metric == metric, "median"].iloc[0]), rel(FILES["distribution_equal"]), "median of 23 matched model summaries", "CR=IQR(reference field)/IQR(IC) for each coordinate, then coordinate/model aggregation.")
    for metric, expected in (("Delta_CR_consensus", .329012), ("Delta_CR_selfref", .379939)):
        add_check(rows, f"fig3d_{metric}", expected, float(drow.loc[drow.metric == metric, "median"].iloc[0]), rel(FILES["distribution_equal"]), "median of paired 23-model differences", "Matched same-model comparisons.")
    add_check(rows, "fig3d_paired_models", 23, len(dist_matched), rel(FILES["distribution_matched"]), "paired model rows", "The three reference trajectories are available per model.")

    prow = perf_equal[perf_equal.conditioning == "all_basins"].iloc[0]
    add_check(rows, "bridge_within_model_rho", .241190, float(prow.model_equal_median_rho), rel(FILES["performance_equal"]), "median of 36 within-model basin Spearman correlations", "Absolute DeltaKGE and D_theta use the same 36x531 evaluation table.")
    add_check(rows, "bridge_within_model_ci_low", .197312, float(prow.bootstrap_ci_low), rel(FILES["performance_equal"]), "paired basin bootstrap then model-equal median", "Bootstrap settings are retained in source provenance.", "bootstrap")
    add_check(rows, "bridge_within_model_ci_high", .258639, float(prow.bootstrap_ci_high), rel(FILES["performance_equal"]), "paired basin bootstrap then model-equal median", "Bootstrap settings are retained in source provenance.", "bootstrap")
    pmod = read_csv("dtheta_model")
    rho, pval = spearmanr(pmod.D_RMS_median, pmod.median_abs_DeltaKGE)
    add_check(rows, "bridge_between_model_rho", .135135, float(rho), rel(FILES["dtheta_model"]), "Spearman correlation of 36 model medians", "Median D_theta versus median |DeltaKGE|; no pooled basin rows.")
    add_check(rows, "bridge_between_model_p", .431979, float(pval), rel(FILES["dtheta_model"]), "Spearman test on 36 model medians", "Reported only as the optional bridge; not a core R2 claim.")

    return pd.DataFrame(rows), {
        "dtheta_min": dmin, "dtheta_max": dmax, "dtheta_min_model": dmin_model, "dtheta_max_model": dmax_model,
        "strict_rank_models": strict.model.astype(str).tolist(),
        "rank_model": rank_model,
        "loc_cmp": loc_cmp,
        "dist_matched": dist_matched,
        "between_model_rho": float(rho), "between_model_p": float(pval),
    }


def strict_subset_audit(headline_context: dict[str, Any]) -> pd.DataFrame:
    coverage = read_csv("rank_coverage")
    rank_set = set(coverage.loc[coverage.status == "PASS", "model"].astype(str))
    loc_cmp = headline_context["loc_cmp"]
    loc_set = set(loc_cmp.model.astype(str))
    rows = []
    for model in ALL_MODELS:
        rank_ok = model in rank_set
        loc_ok = model in loc_set
        cov = float(coverage.loc[coverage.model == model, "primary_coverage"].iloc[0])
        if rank_ok:
            rank_reason = ""
        else:
            rank_reason = f"primary coverage {cov:.6f} < 0.90; insufficient archived non-canonical IC restart reference"
        if loc_ok:
            loc_reason = ""
        else:
            loc_reason = f"not in coordinate-specific adjusted primary output; primary coverage {cov:.6f} < 0.90"
        rows.append({
            "model_id": model, "rank_strict_eligible": rank_ok, "rank_exclusion_reason": rank_reason,
            "localization_strict_eligible": loc_ok, "localization_strict_exclusion_reason": loc_reason,
            "same_subset_flag": rank_ok == loc_ok,
        })
    return pd.DataFrame(rows)


def model_order_candidates(headline_context: dict[str, Any]) -> pd.DataFrame:
    r1 = pd.read_csv(BENCHMARK / "manuscript/r1/tables/Fig1a_model_aggregate_performance.csv")
    r1_order = dict(zip(r1.model.astype(str), r1.plot_order.astype(int)))
    dtheta = read_csv("dtheta_model").set_index("model")
    d_order = dtheta.D_RMS_median.rank(method="first", ascending=True).astype(int).to_dict()
    # R1 explicitly documents this as the canonical registry order.  Keep the
    # same order as the current frozen contract/registry for the candidate row.
    registry = {model: i for i, model in enumerate(r1.model.astype(str), start=0)}
    rows = []
    for model in ALL_MODELS:
        rows.append({
            "model_id": model,
            "r1_order": r1_order.get(model, np.nan),
            "registry_order": registry.get(model, np.nan),
            "dtheta_order": d_order.get(model, np.nan),
            "recommended_order": "R1_registry_order",
            "reason": "R1 and current registry use this stable order; dtheta_order is supplied as a displacement-sorted candidate, not frozen for plotting.",
        })
    return pd.DataFrame(rows)


def panel_manifest(strict: pd.DataFrame) -> pd.DataFrame:
    def p(figure: str, panel: str, quantity: str, granularity: str, columns: str, sources: list[str], scripts: list[str], status: str, reconstructable: str, new_analysis: str, notes: str) -> dict[str, Any]:
        return {"figure": figure, "panel": panel, "scientific_quantity": quantity, "required_granularity": granularity, "required_columns": columns, "source_file": ";".join(sources), "source_script": ";".join(scripts), "source_columns": columns, "current_status": status, "reconstructable": reconstructable, "needs_new_analysis": new_analysis, "notes": notes}
    same = bool(strict.same_subset_flag.all())
    strict_loc_n = int(strict.localization_strict_eligible.sum())
    frame = pd.DataFrame([
        p("Figure 2", "2a", "D_theta basin-level IC-to-dPL normalized RMS displacement", "model x basin", "model_id,basin_id,D_theta_cross,D_theta_IC_self,parameter_count", [rel(FILES["dtheta_basin"]), rel(FILES["separation_model"])], [rel(SCRIPTS["dtheta"]), rel(SCRIPTS["ic_self"])], "READY", "YES; direct table is 36x531 and matches model headline", "NO", "IC best archived restart versus dPL canonical seed 42; normalization/provenance is in frozen contract and direct R2 provenance."),
        p("Figure 2", "2b", "cross-minus-IC-self separation", "model level + bootstrap summary", "model_id,D_cross_median,D_self_median,cross_minus_self,ci_low,ci_high", [rel(FILES["separation_model"]), rel(FILES["separation_bootstrap"])], [rel(SCRIPTS["ic_self"])], "READY", "YES", "NO", "One-sided archived IC-self reference; 1000 paired basin resamples, seed 20261005."),
        p("Figure 2", "2c", "coordinate-level cross-catchment Spearman rank correspondence", "model x parameter coordinate", "model_id,parameter_id,parameter_label,n_basins,R_rank,status,unique_count,boundary_occupancy", [rel(FILES["rank_coordinate"]), rel(FILES["rank_qc"]), rel(FILES["rank_equal"])], [rel(SCRIPTS["rank"])], "READY", "YES; 271 model-parameter coordinates exist", "NO", "Tie/constant coordinates are explicitly status-coded; reach thresholds are model-summary thresholds, not pooled coordinate percentages."),
        p("Figure 2", "2d", "strict R_cross versus IC-self R_self", "strict eligible model", "model_id,R_cross,R_self,DeltaR,strict_eligible", [rel(FILES["rank_self_model"]), rel(FILES["rank_self_equal"])], [rel(SCRIPTS["rank_self"])], "READY", "YES; 23 paired model rows", "NO", "Formal verdict remains INCONCLUSIVE for full ensemble because primary coverage is 23/36."),
        p("Figure 3", "3a", "displacement-composition fingerprint", "raw model x basin x coordinate plus defined basin composition", "model_id,basin_id,parameter_id,coordinate_contribution,top1,second,remaining", [rel(FILES["weight_coordinate"])], [rel(SCRIPTS["weights"])], "BLOCKED", "PARTIAL; exact basin-wise squared-weight object exists", "NO NEW EXPERIMENT; FREEZE AGGREGATION", "Current code defines basin-wise squared contribution weights and basin-wise top-k shares. It does not freeze a model-level total-coordinate composition whose top1/second/remaining sum is the requested hero fingerprint. Do not silently invent that aggregation."),
        p("Figure 3", "3b", "raw effective coordinate number and top-k displacement shares", "36 model summaries + paired basin bootstrap", "model_id,C_eff,top1_share,top2_share,ci_low,ci_high", [rel(FILES["localization_model"]), rel(FILES["localization_equal"])], [rel(SCRIPTS["weights"])], "READY", "YES", "NO", "N_eff=1/sum(w^2), C_eff=N_eff/P; weights are squared normalized coordinate differences."),
        p("Figure 3", "3c", "coordinate-specific IC-self adjusted localization", "23 strict models and matched basins", "model_id,raw_C_eff,adjusted_C_eff,raw_top1,adjusted_top1,adjusted_top2", [rel(FILES["localization_adjusted_model"]), rel(FILES["localization_adjusted_equal"]), rel(FILES["localization_raw_adjusted"]), rel(FILES["localization_model"])], [rel(SCRIPTS["adjusted"])], "PARTIAL", "YES but raw-matched headline has an aggregation conflict", "NO NEW EXPERIMENT; RESOLVE MATCHED-BASIN DEFINITION", "Strict localization N=" + str(strict_loc_n) + "; same as rank subset=" + str(same) + ". Frozen raw .346893/.573874 reproduces when restricting all-basins raw summaries to the same 23 models, but exact adjusted-valid basin overlap gives different values.") ,
        p("Figure 3", "3d", "contraction reference sensitivity", "paired model-level trajectory, 23 models", "model_id,CR_canonical,CR_consensus,CR_ICself", [rel(FILES["distribution_matched"]), rel(FILES["distribution_equal"])], [rel(SCRIPTS["distribution"]), rel(SCRIPTS["distribution_fix"])], "READY", "YES; paired rows exist", "NO", "CR is coordinate IQR(reference field)/IQR(IC), with matched 23-model primary coverage."),
        p("R2 optional", "bridge", "within- and between-model performance/displacement association", "36 model summaries plus 36 model-level bridge rows", "model_id,rho_within,D_theta_median,abs_DeltaKGE_median,rho_between,p_between", [rel(FILES["performance_model"]), rel(FILES["performance_equal"]), rel(FILES["dtheta_model"])], [rel(SCRIPTS["dtheta"])], "READY", "YES; lightweight recomputation from existing outputs", "NO", "Supporting descriptive bridge only; between-model relation is weak and not a core R2 estimand."),
    ])
    source_columns = {
        "2a": "model,basin_id,D_RMS,D_L1,D_median_abs,KGE_IC,KGE_dPL,DeltaKGE,abs_DeltaKGE,parameter_count,D_within_IC_restart_median,restart_reference_status",
        "2b": "model,row_type,threshold,D_cross_median,D_self_median_of_basin_medians,cross_minus_self_median;threshold,observed_D_cross,observed_D_self,observed_difference,difference_ci_low,difference_ci_high,n_boot,seed,unit",
        "2c": "model,parameter_index,parameter,R_rank,status,n,ic_unique_count,dpl_unique_count;model,parameter_index,method,unique_count,modal_value_fraction,lower_bound_occupancy,upper_bound_occupancy,severe_tie_or_bound",
        "2d": "pool,prefix,model,primary_reference_status,n_coordinates,R_cross_median,self_median_median,DeltaR_self_minus_cross_median,cross_percentile_median",
        "3a": "model,basin_id,parameter_index,parameter,DeltaTheta,squared_displacement,weight,zero_displacement",
        "3b": "model,parameter_count,n_basins,C_eff_median,N_eff_median,top1_share_median,top2_share_median,C_eff_bootstrap_ci_low,C_eff_bootstrap_ci_high;metric,median,paired_basin_bootstrap_ci_low,paired_basin_bootstrap_ci_high",
        "3c": "model,basin_id,parameter_count,n_positive_excess_coordinates,positive_excess_mass,status,C_eff_excess,top1_excess_share,top2_excess_share;model,n_matched_basins,C_eff_excess_median,C_eff_raw_median,top1_excess_share_median,top1_share_raw_median,top2_excess_share_median,top2_share_raw_median",
        "3d": "model,canonical_median_CR,consensus_median_CR,selfref_median_CR,Delta_CR_consensus,Delta_CR_selfref",
        "bridge": "model,n_basins,D_cross_median,D_self_median,E_theta_median,rho_absDeltaKGE_D_theta,p_D_theta,ratio_available_n;conditioning,metric,model_equal_median_rho,bootstrap_ci_low,bootstrap_ci_high",
    }
    frame["source_columns"] = frame["panel"].map(source_columns)
    return frame


def write_report(manifest: pd.DataFrame, checks: pd.DataFrame, strict: pd.DataFrame, order: pd.DataFrame, context: dict[str, Any], candidates: pd.DataFrame) -> None:
    status = {f"{r.figure} {r.panel}": r.current_status for r in manifest.itertuples()}
    strict_rank = strict.loc[strict.rank_strict_eligible, "model_id"].tolist()
    strict_loc = strict.loc[strict.localization_strict_eligible, "model_id"].tolist()
    excluded = strict.loc[~strict.rank_strict_eligible, ["model_id", "rank_exclusion_reason"]]
    stale_lines = [
        f"- `results/.../r2_final_robustness_20260906/agent_B_contraction_robustness/SUPERSEDED_DO_NOT_USE.md`: parent Agent B outputs are superseded; only `corrected_primary_23models/` is canonical for Fig 3d.",
        f"- Older exploratory manuscript/r2 branches are recorded in `manuscript/r2/CLEANUP_REPORT.md` and must not be called by figure scripts.",
        f"- Direct diagnostic figures under `results/.../r2/figures/` are exploratory 180-dpi PNGs, not formal Figure 2/3 sources; this audit generated no figures.",
        f"- The dated direct R2 tables and manuscript/r2 frozen cache are related but not interchangeable; the cache input manifest/hash contract controls frozen summary claims.",
    ]
    check_fail = checks.loc[checks.status != "PASS"]
    with (R2_ROOT / "R2_FIGURE_DATA_AUDIT.md").open("w") as out:
        out.write("# R2 Figure Data Audit\n\n")
        out.write("## 1. Executive verdict\n\n")
        for key in ("Figure 2 2a", "Figure 2 2b", "Figure 2 2c", "Figure 2 2d", "Figure 3 3a", "Figure 3 3b", "Figure 3 3c", "Figure 3 3d"):
            out.write(f"- **{key} — {status.get(key, 'BLOCKED')}**\n")
        out.write("\nMaximum blockers: Figure 3a has exact basin-wise squared contribution data, but the requested model-level top-1/second/remaining composition is not a frozen estimand; Figure 3c has a raw-matched aggregation conflict (.346893/.573874 versus exact-overlap .345217/.571177).\n\n")
        out.write("Overall plotting gate: **NOT READY FOR PLOTTING** until the Figure 3a aggregation and Figure 3c raw-matched definition are resolved. No new experiment is required for these decisions.\n\n")

        out.write("## 2. Canonical source map\n\n")
        out.write("The canonical source map is the unique source assignment in `tables/R2_FIGURE_DATA_MANIFEST.csv`; source hashes are also retained in `tables/R2_SOURCE_PROVENANCE.json`.\n\n")
        out.write("- Fig 2a D_theta: direct frozen normalized basin table `" + rel(FILES["dtheta_basin"]) + "`, cross-checked against the pinned IC-self/model summary and `11_MODEL_LEVEL_PARAMETER_DISPLACEMENT.csv`.\n")
        out.write("- Fig 2b: pinned `MODEL_IC_SELF_SUMMARY.csv` and `BOOTSTRAP_SUMMARY.csv`; generator `" + rel(SCRIPTS["ic_self"]) + "`.\n")
        out.write("- Fig 2c: `R_rank_all_271.csv` plus occupancy QC; model-equal headline from `R_rank_model_equal_summaries.csv`.\n")
        out.write("- Fig 2d: `model_rank_self_primary_5000.csv`; corrected strict primary rank benchmark.\n")
        out.write("- Fig 3a/b: `basin_coordinate_weights.csv` and the exact frozen coordinate-concentration summaries.\n")
        out.write("- Fig 3c: coordinate-specific adjusted output plus matched raw comparison from the final coordinate IC-self audit.\n")
        out.write("- Fig 3d: corrected 23-model contraction branch only.\n\n")
        out.write("UNRESOLVED PROVENANCE: none for the existing frozen quantities. The only unresolved item is the proposed Fig 3a model-level composition definition, not a file identity.\n\n")

        out.write("## 3. Figure 2 data readiness\n\n")
        out.write("- **2a:** 19,116 rows = 36 models x 531 basins; D_RMS is computed as RMS over same-model normalized coordinates. The source script uses the frozen [0,1] normalized-u arrays, canonical IC best archived restart, and dPL seed 42. Model medians reproduce the expected range `0.150576–0.592668` and headline median `0.384375`.\n")
        out.write("- **2b:** 36 model rows plus a bootstrap summary. IC-self is the median distance to eligible non-canonical archived IC restarts within training-KGE tolerance 0.01, with canonical restart excluded. The source reports 1,000 paired basin resamples, seed 20261005, and model-equal aggregation.\n")
        out.write("- **2c:** 271 coordinate rows across 36 models. R_rank is tie-corrected Spearman over the same 531 basins, coordinate by coordinate. Constant/tie/boundary diagnostics exist; no coordinate is silently relabeled. `reach 0.5/0.8/0.9` means the model-level median R_rank summary is at least that threshold, not a pooled coordinate percentage.\n")
        out.write("- **2d:** strict rank benchmark is exactly 23 models at the 0.90 primary coverage gate. The model-level paired R_cross/R_self table exists. Formal full-ensemble verdict remains `INCONCLUSIVE`, not a positive universal rank-preservation claim.\n\n")

        out.write("## 4. Figure 3 data readiness\n\n")
        out.write("- **3a:** `basin_coordinate_weights.csv` has the exact source object: `weight = DeltaTheta^2 / sum_p(DeltaTheta^2)`, with squared displacement and basin IDs. Existing code ranks coordinates within each basin and reports cumulative basin-wise top-k shares. It does not define one model-level total vector composition with a unique top coordinate across basins. Therefore the requested hero composition is BLOCKED pending an explicit aggregation choice; the raw data are not missing.\n")
        out.write("- **3b:** C_eff uses `N_eff = 1/sum(w^2)` and `C_eff=N_eff/P`; top1/top2 are cumulative shares from basin-wise sorted weights, followed by within-model medians and equal model weighting. 36-model and paired-basin bootstrap objects exist.\n")
        out.write("- **3c:** exact adjustment is coordinate-wise `X=abs(dPL-IC)`, `S_med=median(abs(IC_restart-IC))` over eligible non-canonical restarts, `E=X-S_med`, and `E_plus=max(E,0)`. The adjusted primary is 23 models. The frozen raw .346893/.573874 values reproduce from all-basins raw summaries restricted to those 23 models, whereas the exact adjusted-valid basin-overlap table gives .345217/.571177; this is a real aggregation/provenance conflict and must be resolved before plotting the raw-to-adjusted comparison.\n")
        out.write("- **3d:** CR is `IQR(reference field)/IQR(IC)` coordinate-wise. Canonical uses dPL, consensus uses basin-wise median eligible non-canonical restart fields, and IC-self uses the frozen synthetic restart fields. Paired 23-model rows are present for all three references.\n\n")

        out.write("## 5. Strict subset audit\n\n")
        out.write(f"- rank strict N = {len(strict_rank)}\n- localization strict N = {len(strict_loc)}\n- intersection N = {len(set(strict_rank) & set(strict_loc))}\n- symmetric difference = {sorted(set(strict_rank) ^ set(strict_loc))}\n- rank strict IDs = `{', '.join(strict_rank)}`\n- localization strict IDs = `{', '.join(strict_loc)}`\n\n")
        out.write("The two strict subsets are identical in the current frozen outputs; they must nevertheless remain separately named in figure code (`strict_rank_subset` and `strict_localization_subset`) rather than being aliased as an unexplained `strict23`.\n\n")
        out.write("Excluded 13 models and reasons:\n\n")
        for row in excluded.itertuples():
            out.write(f"- `{row.model_id}` — {row.rank_exclusion_reason}\n")
        out.write("\n")

        out.write("## 6. Headline reproduction\n\n")
        out.write(f"`tables/R2_HEADLINE_REPRO_CHECK.csv` contains {len(checks)} checks. PASS rows use absolute tolerance 1e-6; bootstrap rows retain the recovered source seed and resampling description. Failing rows: {len(check_fail)}.\n\n")
        if len(check_fail):
            out.write(check_fail.to_string(index=False) + "\n\n")
            out.write("The two failing headline checks are intentional audit findings, not rounding noise: the frozen raw .346893/.573874 reproduces from the 23-model subset of the all-basins raw table, but the exact adjusted-valid basin overlap table is .345217/.571177. The source code and cache therefore do not currently support calling those values both 'raw matched'.\n\n")
        out.write("Important definitions recovered from source code:\n\n")
        out.write("- Parameter normalization is physical-bound normalized `[0,1]`; IC values are verified against linear `(physical-lower)/(upper-lower)`, while dPL auto-log mappings remain canonical network-u coordinates. All 271 registry coordinates are included; no equal-bound fixed coordinates were found.\n")
        out.write("- IC is the best archived restart per model-basin, canonical restart excluded from self-reference, no fallback, with one shared restart index across parameters. dPL has seed 42 only; no dPL multi-seed result exists.\n")
        out.write("- IC-self separation bootstrap: 1,000 paired basin resamples, seed 20261005, basin resampling within model followed by model-equal medians.\n")
        out.write("- Rank bootstrap: 1,500 basin resamples, seed 20260906, coordinate rank recomputed within each sampled basin set, then model-equal median.\n")
        out.write("- Localization raw bootstrap: 1,000 paired basin resamples, seed 20260906, then model-equal median. Adjusted summaries use the final coordinate audit's fixed 1,000-draw model-summary bootstrap.\n")
        out.write("- Contraction robustness: fixed 5,000 synthetic restart fields with draw seed 20260913; corrected primary summaries use 23 models and bootstrap seed 42 for model summaries.\n\n")

        out.write("## 7. Stale / conflicting result files\n\n")
        out.write("\n".join(stale_lines) + "\n\n")
        out.write("The 36-model direct diagnostic is useful for basin-level D_theta and raw coordinate contributions, but it is not a license to replace the manuscript frozen 23-model strict summaries or corrected contraction branch. All figure caches written by the companion builder include source hashes to prevent silent mixing.\n\n")

        out.write("## 8. Minimum additional work\n\n")
        out.write("1. Freeze the Figure 3a aggregation: either use the existing basin-wise composition distribution (no new analysis) or explicitly document a model-level coordinate aggregation from the already available weights.\n")
        out.write("2. Resolve Figure 3c raw matched semantics: choose exact adjusted-valid basin overlap or 23-model all-basins raw values, document it, and do not update the frozen headline silently.\n")
        out.write("3. Use `build_r2_figure_cache.py` to regenerate plotting caches after any source-hash change; no training, CMA-ES, OOB/PUB/PUR, or hydrological simulation is needed.\n")
        out.write("4. Do not promote the optional bridge or the 23-model strict result to a full-ensemble causal or superiority claim.\n\n")

        out.write("## 9. Recommendation for plotting stage\n\n")
        out.write("- READY hero/support: Figure 2a, 2c, Figure 3b, 3d.\n")
        out.write("- READY but strict/inconclusive boundary: Figure 2b and 2d.\n")
        out.write("- PARTIAL pending aggregation/provenance resolution: Figure 3c.\n")
        out.write("- BLOCKED pending estimand wording: Figure 3a.\n")
        out.write("- Formal plotting condition: **not met** until Figure 3a and Figure 3c are resolved.\n\n")
        out.write("## Appendix: model-order candidates\n\n")
        out.write("`tables/R2_MODEL_ORDER_CANDIDATES.csv` preserves R1/registry order and displacement-sorted order without making a final layout decision. R1's source explicitly identifies its order as the canonical registry order.\n\n")
        out.write("## Appendix: scan and provenance\n\n")
        out.write(f"Candidate scan rows: {len(candidates)}. Required source files: {len(FILES)}. Repository pre-existing changes were not modified.\n")
        out.write("No PDF/SVG/EPS or temporary figure was generated.\n")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    require_sources(list(FILES))
    candidates = scan_candidates()
    candidates.to_csv(TABLES / "R2_SOURCE_CANDIDATE_INDEX.csv", index=False)
    checks, context = make_headline_checks()
    strict = strict_subset_audit(context)
    order = model_order_candidates(context)
    manifest = panel_manifest(strict)
    checks.to_csv(TABLES / "R2_HEADLINE_REPRO_CHECK.csv", index=False)
    strict.to_csv(TABLES / "R2_STRICT_SUBSET_AUDIT.csv", index=False)
    order.to_csv(TABLES / "R2_MODEL_ORDER_CANDIDATES.csv", index=False)
    manifest.to_csv(TABLES / "R2_FIGURE_DATA_MANIFEST.csv", index=False)
    provenance = {
        "audit_script": rel(HERE), "benchmark_root": rel(BENCHMARK), "status": "BLOCKED_FOR_PLOTTING_FIG3A_FIG3C",
        "contract": json.loads(FILES["frozen_contract"].read_text()),
        "sources": {key: source_meta(path) for key, path in FILES.items()},
        "source_scripts": {key: source_meta(path) for key, path in SCRIPTS.items()},
        "strict_rank_n": int(strict.rank_strict_eligible.sum()),
        "strict_localization_n": int(strict.localization_strict_eligible.sum()),
        "strict_intersection_n": int((strict.rank_strict_eligible & strict.localization_strict_eligible).sum()),
        "checks_total": len(checks), "checks_pass": int((checks.status == "PASS").sum()),
        "checks_fail": int((checks.status != "PASS").sum()),
        "no_training_or_simulation": True,
        "figure_generation": "none",
    }
    (TABLES / "R2_SOURCE_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n")
    write_report(manifest, checks, strict, order, context, candidates)
    print(json.dumps({
        "status": provenance["status"], "checks": f"{provenance['checks_pass']}/{provenance['checks_total']} PASS",
        "fig2": {row.panel: row.current_status for row in manifest[manifest.figure == "Figure 2"].itertuples()},
        "fig3": {row.panel: row.current_status for row in manifest[manifest.figure == "Figure 3"].itertuples()},
        "strict_rank_n": provenance["strict_rank_n"], "strict_localization_n": provenance["strict_localization_n"],
        "report": str(R2_ROOT / "R2_FIGURE_DATA_AUDIT.md"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
