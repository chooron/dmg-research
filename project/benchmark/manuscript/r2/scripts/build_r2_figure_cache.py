#!/usr/bin/env python3
"""Build read-only, provenance-carrying Figure 2/3 plotting caches.

The builder only reshapes existing frozen tables and reconstructs raw
coordinate-weight objects from existing audit output.  It does not define new
inferential estimands, train models, run CMA-ES, run hydrological simulations,
or create figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
R2_ROOT = HERE.parents[1]
TABLES = R2_ROOT / "tables"
CACHE = R2_ROOT / "cache"

# Import the audit module so cache paths, source identities, and the exact
# basin-wise squared-weight reconstruction remain single-sourced.
sys.path.insert(0, str(HERE.parent))
import audit_r2_figure_sources as audit  # noqa: E402


def metadata(frame: pd.DataFrame, source_keys: list[str], source_script_keys: list[str], aggregation: str) -> pd.DataFrame:
    out = frame.copy()
    out["source_file"] = ";".join(audit.rel(audit.FILES[k]) for k in source_keys)
    out["source_sha256"] = ";".join(audit.sha256(audit.FILES[k]) for k in source_keys)
    out["source_script"] = ";".join(audit.rel(audit.SCRIPTS[k]) for k in source_script_keys)
    out["aggregation_level"] = aggregation
    return out


def write_csv(name: str, frame: pd.DataFrame) -> Path:
    path = CACHE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")
    return path


def write_parquet(name: str, frame: pd.DataFrame) -> Path:
    path = CACHE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def build_fig2a() -> Path:
    basin = audit.read_csv("dtheta_basin")
    audit.check_frame_schema(basin, {"model", "basin_id", "D_RMS", "parameter_count", "D_within_IC_restart_median", "restart_reference_status"}, "Fig2a basin displacement")
    out = basin.rename(columns={
        "model": "model_id", "D_RMS": "D_theta_cross", "D_within_IC_restart_median": "D_theta_IC_self",
    })[["model_id", "basin_id", "D_theta_cross", "D_theta_IC_self", "parameter_count", "restart_reference_status"]].copy()
    out["n_basins_per_model"] = out.groupby("model_id")["basin_id"].transform("nunique")
    return write_parquet("fig2a_parameter_displacement.parquet", metadata(out, ["dtheta_basin", "dtheta_model"], ["dtheta"], "basin within model"))


def build_fig2b() -> Path:
    model = audit.read_csv("separation_model")
    model = model[(model.row_type == "model") & (model.threshold == "within_0.01")].copy()
    out = model.rename(columns={
        "model": "model_id", "D_cross_median": "D_cross_median", "D_self_median_of_basin_medians": "D_self_median",
    })[["model_id", "D_cross_median", "D_self_median", "cross_minus_self_median", "n_insufficient"]]
    out["scope"] = "model"
    boot = audit.read_csv("separation_bootstrap")
    b = boot[boot.threshold == "within_0.01"].iloc[0]
    summary = pd.DataFrame([{
        "model_id": "ALL36", "scope": "all36_summary", "D_cross_median": float(b.observed_D_cross),
        "D_self_median": float(b.observed_D_self), "cross_minus_self_median": float(b.observed_difference),
        "n_insufficient": np.nan, "ci_low": float(b.difference_ci_low), "ci_high": float(b.difference_ci_high),
        "n_boot": int(b.n_boot), "bootstrap_seed": int(b.seed), "bootstrap_unit": str(b.unit),
    }])
    out["ci_low"] = np.nan; out["ci_high"] = np.nan; out["n_boot"] = np.nan; out["bootstrap_seed"] = np.nan; out["bootstrap_unit"] = ""
    return write_csv("fig2b_cross_minus_self.csv", metadata(pd.concat([out, summary], ignore_index=True), ["separation_model", "separation_bootstrap"], ["ic_self"], "model equal after within-model basin summaries"))


def build_fig2c() -> Path:
    rank = audit.read_csv("rank_coordinate")
    audit.check_frame_schema(rank, {"model", "parameter_index", "parameter", "R_rank", "status", "n", "ic_unique_count", "dpl_unique_count"}, "Fig2c coordinate rank")
    qc = audit.read_csv("rank_qc")
    qc = qc[[c for c in qc.columns if c in {"model", "parameter_index", "method", "unique_count", "modal_value_fraction", "lower_bound_occupancy", "upper_bound_occupancy", "severe_tie_or_bound"}]].copy()
    wide = qc.pivot_table(index=["model", "parameter_index"], columns="method", values=[c for c in qc.columns if c not in {"model", "parameter_index", "method"}], aggfunc="first")
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    wide = wide.reset_index()
    out = rank.rename(columns={"model": "model_id", "parameter_index": "parameter_id", "parameter": "parameter_label", "n": "n_basins"}).merge(
        wide.rename(columns={"model": "model_id", "parameter_index": "parameter_id"}), on=["model_id", "parameter_id"], how="left", validate="one_to_one"
    )
    # tie_fraction was not defined by the source code; preserve that fact rather
    # than inventing a tie-count convention.  Boundary occupancy is retained
    # exactly as the source's lower/upper occupancy fields.
    out["tie_fraction"] = np.nan
    out["tie_fraction_status"] = "not defined by canonical source"
    out["eligible_flag"] = out["status"].eq("finite")
    return write_parquet("fig2c_rank_correspondence.parquet", metadata(out, ["rank_coordinate", "rank_qc", "rank_equal"], ["rank"], "coordinate within model over aligned basins"))


def build_fig2d() -> Path:
    rank = audit.read_csv("rank_self_model")
    out = rank[(rank.pool == "primary") & (rank.prefix == 5000)].copy()
    out = out.rename(columns={"model": "model_id", "R_cross_median": "R_cross", "self_median_median": "R_self", "DeltaR_self_minus_cross_median": "DeltaR_self_minus_cross"})
    out["strict_eligible"] = out["primary_reference_status"].eq("PASS")
    out = out[out.strict_eligible].copy()
    cols = ["model_id", "R_cross", "R_self", "DeltaR_self_minus_cross", "fraction_cross_below_self_q05", "fraction_cross_below_self_median", "primary_reference_status", "strict_eligible"]
    return write_csv("fig2d_strict_rank_benchmark.csv", metadata(out[cols], ["rank_self_model", "rank_self_equal"], ["rank_self"], "strict eligible model"))


def build_fig3a() -> tuple[Path, Path]:
    weights = audit.read_csv("weight_coordinate")
    coordinate, composition = audit.compute_weight_metrics(weights)
    coordinate = metadata(coordinate, ["weight_coordinate"], ["weights"], "coordinate contribution within basin and model")
    composition = metadata(composition, ["weight_coordinate"], ["weights"], "basin-wise top-k composition within model")
    return write_parquet("fig3a_coordinate_weights.parquet", coordinate), write_parquet("fig3a_coordinate_composition.parquet", composition)


def build_fig3b() -> Path:
    model = audit.read_csv("localization_model")
    model = model.rename(columns={"model": "model_id"})
    keep = ["model_id", "parameter_count", "n_basins", "zero_displacement_basins", "C_eff_median", "N_eff_median", "top1_share_median", "top2_share_median", "top3_share_median", "C_eff_bootstrap_ci_low", "C_eff_bootstrap_ci_high", "top1_share_bootstrap_ci_low", "top1_share_bootstrap_ci_high", "top2_share_bootstrap_ci_low", "top2_share_bootstrap_ci_high"]
    out = model[[c for c in keep if c in model.columns]].copy()
    out["scope"] = "model"
    equal = audit.read_csv("localization_equal")
    summaries = []
    for metric in ("C_eff", "N_eff", "top1_share", "top2_share", "top3_share"):
        row = equal[equal.metric == metric].iloc[0]
        summaries.append({"model_id": "ALL36", "scope": "model_equal", "parameter_count": np.nan, "n_basins": 531, "zero_displacement_basins": np.nan, "C_eff_median": np.nan, "N_eff_median": np.nan, "top1_share_median": np.nan, "top2_share_median": np.nan, "top3_share_median": np.nan, "C_eff_bootstrap_ci_low": np.nan, "C_eff_bootstrap_ci_high": np.nan, "top1_share_bootstrap_ci_low": np.nan, "top1_share_bootstrap_ci_high": np.nan, "top2_share_bootstrap_ci_low": np.nan, "top2_share_bootstrap_ci_high": np.nan, "summary_metric": metric, "summary_value": float(row['median']), "summary_ci_low": float(row.paired_basin_bootstrap_ci_low), "summary_ci_high": float(row.paired_basin_bootstrap_ci_high)})
    out["summary_metric"] = ""; out["summary_value"] = np.nan; out["summary_ci_low"] = np.nan; out["summary_ci_high"] = np.nan
    frame = pd.concat([out, pd.DataFrame(summaries)], ignore_index=True)
    return write_csv("fig3b_localization_metrics.csv", metadata(frame, ["localization_model", "localization_equal"], ["weights"], "model summaries plus model-equal headline rows"))


def build_fig3c() -> Path:
    adjusted_basin = audit.read_csv("localization_adjusted_model")
    adjusted_basin["model"] = adjusted_basin.model.astype(str)
    rows = []
    for model, group in adjusted_basin.groupby("model", sort=True):
        passed = group[group.status == "PASS"]
        rows.append({
            "model_id": model, "n_basins": len(group), "n_valid_excess_basins": len(passed),
            "fraction_any_positive_excess": float((group.status == "PASS").mean()),
            "adjusted_C_eff": float(passed.C_eff_excess.median()) if len(passed) else np.nan,
            "adjusted_top1": float(passed.top1_excess_share.median()) if len(passed) else np.nan,
            "adjusted_top2": float(passed.top2_excess_share.median()) if len(passed) else np.nan,
        })
    out = pd.DataFrame(rows)
    raw = audit.read_csv("localization_raw_adjusted").rename(columns={"model": "model_id", "C_eff_excess_median": "adjusted_C_eff_from_matched", "C_eff_raw_median": "raw_matched_C_eff", "top1_excess_share_median": "adjusted_top1_from_matched", "top1_share_raw_median": "raw_matched_top1", "top2_excess_share_median": "adjusted_top2_from_matched", "top2_share_raw_median": "raw_matched_top2"})
    out = out.merge(raw[["model_id", "n_matched_basins", "adjusted_C_eff_from_matched", "raw_matched_C_eff", "adjusted_top1_from_matched", "raw_matched_top1", "adjusted_top2_from_matched", "raw_matched_top2", "delta_C_eff_excess_minus_C_eff_raw_median", "delta_top1_excess_share_minus_top1_share_raw_median", "delta_top2_excess_share_minus_top2_share_raw_median"]], on="model_id", how="inner", validate="one_to_one")
    out["strict_eligible"] = True
    equal = audit.read_csv("localization_adjusted_equal")
    summary_rows = []
    for _, row in equal.iterrows():
        summary_rows.append({"model_id": "STRICT23", "n_basins": np.nan, "n_valid_excess_basins": np.nan, "fraction_any_positive_excess": np.nan, "adjusted_C_eff": np.nan, "adjusted_top1": np.nan, "adjusted_top2": np.nan, "n_matched_basins": np.nan, "adjusted_C_eff_from_matched": np.nan, "raw_matched_C_eff": np.nan, "adjusted_top1_from_matched": np.nan, "raw_matched_top1": np.nan, "adjusted_top2_from_matched": np.nan, "raw_matched_top2": np.nan, "delta_C_eff_excess_minus_C_eff_raw_median": np.nan, "delta_top1_excess_share_minus_top1_share_raw_median": np.nan, "delta_top2_excess_share_minus_top2_share_raw_median": np.nan, "strict_eligible": True, "summary_metric": str(row.metric), "summary_value": float(row['median']), "summary_ci_low": float(row.bootstrap_ci_low), "summary_ci_high": float(row.bootstrap_ci_high)})
    out["summary_metric"] = ""; out["summary_value"] = np.nan; out["summary_ci_low"] = np.nan; out["summary_ci_high"] = np.nan
    return write_csv("fig3c_adjusted_localization.csv", metadata(pd.concat([out, pd.DataFrame(summary_rows)], ignore_index=True), ["localization_adjusted_model", "localization_adjusted_equal", "localization_raw_adjusted"], ["adjusted"], "strict model with matched basin comparison"))


def build_fig3d() -> Path:
    out = audit.read_csv("distribution_matched")
    out = out.rename(columns={"model": "model_id", "canonical_median_CR": "CR_canonical", "consensus_median_CR": "CR_consensus", "selfref_median_CR": "CR_ICself"})
    out["strict_eligible"] = True
    return write_csv("fig3d_contraction_reference.csv", metadata(out, ["distribution_matched", "distribution_equal"], ["distribution_fix"], "paired strict model"))


def build_bridge() -> Path:
    within = audit.read_csv("performance_model")
    dtheta = audit.read_csv("dtheta_model")[["model", "D_RMS_median", "median_abs_DeltaKGE"]].copy()
    out = within.merge(dtheta, on="model", how="inner", validate="one_to_one")
    out = out.rename(columns={"model": "model_id", "rho_absDeltaKGE_D_theta": "rho_within", "D_RMS_median": "D_theta_median", "median_abs_DeltaKGE": "abs_DeltaKGE_median"})
    rho, p = spearmanr(out.D_theta_median, out.abs_DeltaKGE_median)
    out["scope"] = "model"
    summary = pd.DataFrame([{"model_id": "ALL36", "scope": "between_model_summary", "rho_within": np.nan, "D_theta_median": np.nan, "abs_DeltaKGE_median": np.nan, "rho_between": float(rho), "p_between": float(p)}])
    out["rho_between"] = np.nan; out["p_between"] = np.nan
    return write_csv("r2_bridge_optional.csv", metadata(pd.concat([out, summary], ignore_index=True), ["performance_model", "performance_equal", "dtheta_model"], ["dtheta"], "36 within-model rows plus one between-model summary"))


def main() -> None:
    manifest_path = TABLES / "R2_FIGURE_DATA_MANIFEST.csv"
    provenance_path = TABLES / "R2_SOURCE_PROVENANCE.json"
    if not manifest_path.exists() or not provenance_path.exists():
        raise RuntimeError("Run audit_r2_figure_sources.py first; plotting caches require the audited source manifest.")
    provenance = json.loads(provenance_path.read_text())
    checks = pd.read_csv(TABLES / "R2_HEADLINE_REPRO_CHECK.csv")
    failed_metrics = set(checks.loc[checks.status != "PASS", "metric"].astype(str))
    allowed_conflicts = {"fig3c_raw_matched_exact_basin_C_eff_against_frozen_headline", "fig3c_raw_matched_exact_basin_top1_against_frozen_headline"}
    unexpected = failed_metrics - allowed_conflicts
    if unexpected:
        raise RuntimeError(f"Refusing to build cache: unexpected failed headline checks {sorted(unexpected)}")
    if provenance.get("strict_rank_n") != 23 or provenance.get("strict_localization_n") != 23:
        raise RuntimeError("Refusing to build cache: strict subset gate is not 23/23")
    outputs: list[Path] = []
    outputs.append(build_fig2a())
    outputs.append(build_fig2b())
    outputs.append(build_fig2c())
    outputs.append(build_fig2d())
    outputs.extend(build_fig3a())
    outputs.append(build_fig3b())
    outputs.append(build_fig3c())
    outputs.append(build_fig3d())
    outputs.append(build_bridge())
    cache_manifest = {
        "status": "PASS_WITH_FIG3A_AND_FIG3C_BLOCKERS",
        "audit_manifest": audit.rel(manifest_path),
        "source_provenance": audit.rel(provenance_path),
        "no_new_statistical_definition": True,
        "fig3a_note": "fig3a_coordinate_composition.parquet is the existing basin-wise squared-weight composition; model-level hero aggregation remains unresolved in the audit.",
        "files": [{"path": audit.rel(path), "sha256": audit.sha256(path), "size_bytes": path.stat().st_size} for path in outputs],
    }
    (CACHE / "R2_FIGURE_CACHE_MANIFEST.json").write_text(json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": cache_manifest["status"], "files_written": len(outputs),
        "cache_dir": str(CACHE), "figure3a": "raw basin-wise object only; hero aggregation unresolved",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
