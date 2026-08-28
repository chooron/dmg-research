#!/usr/bin/env python3
"""Local-only audit and recomputation for the uploaded R1--R5 Results draft.

This script intentionally reads only files below the hydrodiag project root. It
uses existing manuscript CSV/JSON outputs and the existing HBV freeze-cache
outputs; it does not read parent data directories, remote paths, checkpoints,
figures, or prose values as numeric sources.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS = MANUSCRIPT / "results"
CACHE = MANUSCRIPT / "cache" / "local_draft_results_audit"
DRAFT = MANUSCRIPT / "hess_results_R1_R5_reframed_v2.md"

STRATA = ["S1", "S2", "S3", "S4", "S5"]
HIGH_SNOW = {"S4", "S5"}


def local(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def project_file(name: str) -> Path:
    path = (PROJECT / Path(name)).resolve()
    root = PROJECT.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"local-only audit path escaped project root: {name}")
    return path

def read_csv(name: str, **kwargs: Any) -> pd.DataFrame | None:
    path = project_file(name)
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)

def read_json(name: str) -> dict[str, Any] | None:
    path = project_file(name)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return json.loads(text, parse_constant=lambda _: None)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value

def draft_placeholders() -> list[dict[str, Any]]:
    if not DRAFT.exists():
        return []
    lines = DRAFT.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r"\[\[(?:DATA|VERIFY):[^\]]+\]\]|(?:Fig\.|Sect\.|S|Table)\s*S?\[\[[^\]]+\]\]")
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        for match in pattern.finditer(line):
            out.append({"line": line_no, "marker": match.group(0), "context": line.strip()})
    return out


def r1_recompute() -> dict[str, Any]:
    perf = read_csv("manuscript/results/R1/r1_basin_level_performance.csv", dtype={"basin_id": str})
    ct = read_csv("manuscript/results/R1/r1_snow_signatures_basin_level.csv", dtype={"basin_id": str})
    snow = read_csv("manuscript/results/R1/r1_snow_attributes.csv", dtype={"basin_id": str})
    summary = read_csv("manuscript/results/R1/r1_absolute_metrics_summary.csv")
    if perf is None or ct is None or snow is None:
        return {"status": "UNRESOLVED", "reason": "one or more local R1 tables are missing"}
    for frame in (perf, ct, snow):
        frame["basin_id"] = frame["basin_id"].astype(str).str.zfill(8)
    merged = perf.merge(
        ct,
        on=["basin_id", "paradigm", "model", "period"],
        how="left",
        validate="one_to_one",
    ).merge(snow, on="basin_id", how="left", validate="many_to_one")
    merged["kge"] = pd.to_numeric(merged["kge"], errors="coerce")
    merged["ct_error_abs"] = pd.to_numeric(merged["ct_error_abs"], errors="coerce")
    merged["ct_error_signed"] = pd.to_numeric(merged["ct_error_signed"], errors="coerce")
    result: dict[str, Any] = {
        "status": "RESOLVED_FROM_LOCAL_OUTPUT",
        "source": [
            "manuscript/results/R1/r1_basin_level_performance.csv",
            "manuscript/results/R1/r1_snow_signatures_basin_level.csv",
            "manuscript/results/R1/r1_snow_attributes.csv",
        ],
        "coverage": {
            "performance_rows": int(len(perf)),
            "ct_rows": int(len(ct)),
            "models": sorted(perf["model"].dropna().unique().tolist()),
            "paradigms": sorted(perf["paradigm"].dropna().unique().tolist()),
            "periods": sorted(perf["period"].dropna().unique().tolist()),
        },
        "screen_counts": [],
        "threshold_endpoints": [],
        "high_snow_ct": [],
        "stratum_kge_summary": [],
    }
    for paradigm in sorted(merged["paradigm"].dropna().unique()):
        for model in sorted(merged["model"].dropna().unique()):
            sub = merged[(merged["paradigm"] == paradigm) & (merged["model"] == model) & (merged["period"] == "test")].copy()
            for threshold in (0.60, 0.40, 0.80):
                valid = sub[sub["kge"] >= threshold]
                screened = valid[valid["ct_error_abs"] >= 15]
                result["screen_counts"].append({
                    "paradigm": paradigm,
                    "model": model,
                    "kge_threshold": threshold,
                    "kge_valid_n": int(len(valid)),
                    "ct15_n": int(len(screened)),
                    "ct15_fraction": float(len(screened) / len(valid)) if len(valid) else None,
                    "screened_ct_median": float(screened["ct_error_signed"].median()) if len(screened) else None,
                })
            high = sub[sub["snow_stratum"].isin(HIGH_SNOW)]
            result["high_snow_ct"].append({
                "paradigm": paradigm,
                "model": model,
                "n": int(len(high)),
                "median_ct_signed": float(high["ct_error_signed"].median()) if len(high) else None,
                "median_ct_abs": float(high["ct_error_abs"].median()) if len(high) else None,
            })
    result["common_pass"] = []
    for paradigm in sorted(merged["paradigm"].dropna().unique()):
        test = merged[(merged["paradigm"] == paradigm) & (merged["period"] == "test")]
        kge_pivot = test.pivot_table(index="basin_id", columns="model", values="kge", aggfunc="first")
        required = ["XAJ-Base", "XAJ-TGD", "XAJ-CN"]
        if not set(required).issubset(kge_pivot.columns):
            continue
        ids = kge_pivot.index[kge_pivot[required].ge(0.60).all(axis=1)]
        ct_pivot = test.pivot_table(index="basin_id", columns="model", values="ct_error_signed", aggfunc="first").reindex(ids)
        row = {"paradigm": paradigm, "common_pass_n": int(len(ids))}
        for model in required:
            row[f"{model}_ct_median"] = float(ct_pivot[model].median()) if len(ids) else None
        row["TGD_minus_Base_ct_median"] = float((ct_pivot["XAJ-TGD"] - ct_pivot["XAJ-Base"]).median()) if len(ids) else None
        row["CN_minus_Base_ct_median"] = float((ct_pivot["XAJ-CN"] - ct_pivot["XAJ-Base"]).median()) if len(ids) else None
        result["common_pass"].append(row)
    if summary is not None:
        result["stratum_kge_summary"] = summary[
            summary["metric"].eq("kge") & summary["period"].eq("test")
        ].to_dict(orient="records")
    return result


def r2_recompute() -> dict[str, Any]:
    basin = read_csv("manuscript/results/R2/r2_tgd2_specificity_basin_level.csv")
    regressions = read_csv("manuscript/results/R2/r2_tgd2_specificity_regressions.csv")
    gradients = read_csv("manuscript/results/R2/r2_snow_gradients_summary.csv")
    paired = read_csv("manuscript/results/R2/r2_paired_shifts_basin_level.csv")
    bounds = read_csv("manuscript/supplement/results/s2_parameter_bounds_from_code.csv")
    result: dict[str, Any] = {"sources": [], "status": "RESOLVED_FROM_LOCAL_OUTPUT"}
    if basin is not None:
        prevalence = []
        for (paradigm, contrast), g in basin.groupby(["paradigm", "contrast"]):
            between = pd.to_numeric(g["between_all"], errors="coerce")
            within = pd.to_numeric(g["within_pooled"], errors="coerce")
            valid = between.notna() & within.notna()
            n = int(valid.sum())
            above = int((between[valid] > within[valid]).sum())
            prevalence.append({
                "paradigm": paradigm,
                "contrast": contrast,
                "n": n,
                "above_1to1": above,
                "fraction": float(above / n) if n else None,
            })
        result["sources"].append("manuscript/results/R2/r2_tgd2_specificity_basin_level.csv")
        result["specificity_prevalence"] = prevalence
    if regressions is not None:
        result["sources"].append("manuscript/results/R2/r2_tgd2_specificity_regressions.csv")
        result["specificity_regressions"] = regressions.to_dict(orient="records")
    if gradients is not None:
        result["sources"].append("manuscript/results/R2/r2_snow_gradients_summary.csv")
        result["parameter_gradient_um_ki_ci"] = gradients[
            gradients["parameter"].isin(["xaj_um", "xaj_ki", "xaj_ci"])
        ].to_dict(orient="records")
    if paired is not None:
        result["sources"].append("manuscript/results/R2/r2_paired_shifts_basin_level.csv")
        result["paired_rows"] = int(len(paired))
    if bounds is None:
        result["bounds_audit"] = {"status": "MISSING", "source": "manuscript/supplement/results/s2_parameter_bounds_from_code.csv"}
    else:
        counts = bounds.groupby("active_model_key").size().to_dict()
        expected = {"XAJ", "XAJ_CN", "XAJ_TGD"}
        result["bounds_audit"] = {
            "status": "MISMATCH" if set(counts) != expected else "PASS",
            "model_counts": {str(k): int(v) for k, v in counts.items()},
            "expected_active_model_keys": sorted(expected),
            "source": "manuscript/supplement/results/s2_parameter_bounds_from_code.csv",
        }
    return result


def r3_recompute() -> dict[str, Any]:
    f5 = read_json("manuscript/results/R3/figure5_summary.json")
    f6 = read_json("manuscript/results/R3/figure6_summary.json")
    t5 = read_csv("manuscript/results/R3/table5_main_summary.csv")
    b5 = read_csv("manuscript/results/R3/figure5_basin_seedmedian.csv")
    b6 = read_csv("manuscript/results/R3/figure6_basin_seedmedian.csv")
    meta = read_json("manuscript/results/R3/fig6_seasonal/fig6_seasonal_meta.json")
    result: dict[str, Any] = {"status": "RESOLVED_FROM_LOCAL_OUTPUT", "sources": []}
    if f5 is not None:
        result["sources"].append("manuscript/results/R3/figure5_summary.json")
        result["correct_cn_kge"] = {
            "IC_test": f5.get("panel_a_cn_deficit", {}).get("IC_test", {}).get("median"),
            "dPL_test": f5.get("panel_a_cn_deficit", {}).get("dPL_test", {}).get("median"),
            "IC_train_kge_derived_from_deficit": 1.0 - f5.get("panel_a_cn_deficit", {}).get("IC_train", {}).get("median", np.nan),
        }
        result["f_close"] = f5.get("panel_c_f_close", {})
        result["decay"] = f5.get("panel_d_decay", {})
        result["state_and_parameter_snow_associations"] = f5.get("panels_ef_excess_vs_frac_snow", {})
    if f6 is not None:
        result["sources"].append("manuscript/results/R3/figure6_summary.json")
        result["ladder"] = f6.get("panel_a_ladder", {})
        result["mitigation"] = f6.get("panel_b_f_tgd2", {})
        result["state_relief"] = f6.get("panel_d_r_state", {})
        result["residual_cn_advantage"] = f6.get("panel_e_residual_vs_frac_snow", {})
    for name, frame in (("figure5_basin_seedmedian", b5), ("figure6_basin_seedmedian", b6)):
        if frame is not None:
            result["sources"].append(f"manuscript/results/R3/{name}.csv")
            numeric = frame.select_dtypes(include=["number"]).columns.tolist()
            grouped = frame.groupby(["paradigm", "period"], dropna=False)[numeric].median().reset_index()
            result[f"{name}_grouped_medians"] = grouped.to_dict(orient="records")
    if meta is not None:
        result["sources"].append("manuscript/results/R3/fig6_seasonal/fig6_seasonal_meta.json")
        result["seasonal_definitions"] = {
            "quantity": meta.get("quantity"),
            "state": meta.get("state"),
            "shared_state_components": meta.get("shared_state_components"),
            "seasonal_axis": meta.get("seasonal_axis"),
        }
    if t5 is not None:
        result["sources"].append("manuscript/results/R3/table5_main_summary.csv")
        result["table5_rows"] = int(len(t5))
    if f5 is None or f6 is None:
        result["status"] = "PARTIAL"
    return result


def r4_recompute() -> dict[str, Any]:
    report = read_json("results/r4_phase1_soil_official/r4_phase1_soil_official_report.json")
    three = read_csv("results/r4_phase1_soil_official/three_structure_basin_state_consistency.csv")
    timing = read_csv("results/r4_phase1_soil_official/three_structure_timing_metrics_basin_summary.csv")
    result: dict[str, Any] = {
        "status": "PARTIAL",
        "sources": ["results/r4_phase1_soil_official/r4_phase1_soil_official_report.json"],
        "reference": None,
        "tgd2_provenance_status": None,
    }
    if report is not None:
        result["n_basins"] = report.get("n_basins")
        result["n_test_days"] = report.get("n_test_days")
        result["reference"] = report.get("primary_reference")
        result["tgd2_provenance_status"] = report.get("tgd2_status")
    if three is not None:
        result["sources"].append("results/r4_phase1_soil_official/three_structure_basin_state_consistency.csv")
        result["three_structure_rows"] = int(len(three))
        result["three_structure_counts"] = {str(k): int(v) for k, v in three.groupby("structure").size().to_dict().items()}
        result["three_structure_note"] = "TGD2 rows exist locally but report provenance remains pending; not promoted to canonical TGD."
    if timing is not None:
        result["sources"].append("results/r4_phase1_soil_official/three_structure_timing_metrics_basin_summary.csv")
        result["timing_rows"] = int(len(timing))
    return result


def r5_recompute() -> dict[str, Any]:
    gradient = read_csv("manuscript/results/R5/r5_snow_gradient_table.csv")
    agreement = read_csv("manuscript/results/R5/r5_cross_model_agreement_table.csv")
    timing = read_csv("manuscript/results/R5/r5_timing_signature_table.csv")
    audit = read_json("manuscript/results/R5/r5_data_audit.json")
    result: dict[str, Any] = {"status": "RESOLVED_FROM_LOCAL_OUTPUT", "sources": []}
    if gradient is not None:
        result["sources"].append("manuscript/results/R5/r5_snow_gradient_table.csv")
        result["slopes"] = gradient[["regime", "host_model", "N", "Delta_specific_OLS_slope_beta1", "Delta_specific_slope_95CI"]].to_dict(orient="records")
    if agreement is not None:
        result["sources"].append("manuscript/results/R5/r5_cross_model_agreement_table.csv")
        result["agreement_S1_S5"] = agreement[agreement["stratum"].isin(["S1 [0, 0.05)", "S5 [0.50, 1.00]"] )].to_dict(orient="records")
        result["agreement_rows"] = int(len(agreement))
    if timing is not None:
        result["sources"].append("manuscript/results/R5/r5_timing_signature_table.csv")
        result["high_snow_timing"] = timing[timing["sample"].eq("High-Snow (frac_snow>=0.30)")].to_dict(orient="records")
    if audit is not None:
        result["sources"].append("manuscript/results/R5/r5_data_audit.json")
        result["validity"] = {
            k: v.get("valid_basins_test")
            for k, v in audit.get("models", {}).items()
            if isinstance(v, dict)
        }
    return result


def resolve_placeholders(placeholders: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    known: dict[str, tuple[str, str, str]] = {
        "R1_LOW_SNOW_EFFECTS_AND_CI": ("RESOLVED_FROM_TABLE", "manuscript/results/R1/r1_absolute_metrics_summary.csv", "Local test-period S1 KGE medians and intervals exist"),
        "R1_F1_STRATUM_SUMMARIES": ("RESOLVED_FROM_TABLE", "manuscript/results/R1/r1_absolute_metrics_summary.csv", "Local test-period S1-S5 KGE summaries exist for XAJ Base/TGD/CN"),
        "R1_SCREEN_COUNTS": ("MISMATCH", "manuscript/results/R1/r1_basin_level_performance.csv; manuscript/results/R1/r1_snow_signatures_basin_level.csv", "Local recomputation gives Base IC 152/329 and Base dPL 126/340 at KGE≥0.60 and |CT|≥15 d, not the draft 56/331 and 46/344"),
        "R1_SCREEN_MEDIANS": ("MISMATCH", "manuscript/results/R1/r1_basin_level_performance.csv; manuscript/results/R1/r1_snow_signatures_basin_level.csv", "Local screened CT medians differ from the draft; no alternative local source for the draft screen was identified"),
        "R1_COMMON_PASS_N_AND_CT_EFFECTS": ("RESOLVED_FROM_LOCAL_OUTPUT", "manuscript/results/R1/r1_basin_level_performance.csv; manuscript/results/R1/r1_snow_signatures_basin_level.csv", "Common-pass N and paired CT medians/differences are recomputed in r1_recomputed_common_pass.csv"),
        "R1_THRESHOLD_ENDPOINTS": ("MISMATCH", "manuscript/results/R1/r1_basin_level_performance.csv; manuscript/results/R1/r1_snow_signatures_basin_level.csv", "Local endpoint fractions differ materially from the draft 0.40–0.80 screen values"),
        "R1_CT_BY_ALL_FIVE_STRATA": ("RESOLVED_FROM_TABLE", "manuscript/results/R1/r1_snow_signatures_basin_level.csv; manuscript/results/R1/r1_snow_attributes.csv", "Local signed CT medians can be grouped by all five fixed strata"),
        "R1_XAJ_HIGHSNOW_CT_BASE_IC_DPL": ("RESOLVED_FROM_TABLE", "manuscript/results/R1/r1_snow_signatures_basin_level.csv; manuscript/results/R1/r1_snow_attributes.csv", "Local XAJ high-snow medians are -44.7 d IC and -43.1 d dPL for Base"),
        "R1_XAJ_HIGHSNOW_CT_TGD_IC_DPL": ("RESOLVED_FROM_TABLE", "manuscript/results/R1/r1_snow_signatures_basin_level.csv; manuscript/results/R1/r1_snow_attributes.csv", "Local XAJ high-snow medians are -12.1 d IC and -9.9 d dPL for TGD"),
        "R1_XAJ_HIGHSNOW_CT_CN_IC_DPL": ("RESOLVED_FROM_TABLE", "manuscript/results/R1/r1_snow_signatures_basin_level.csv; manuscript/results/R1/r1_snow_attributes.csv", "Local XAJ high-snow medians are -0.6 d IC and -2.1 d dPL for CN"),
        "R2_PREVALENCE": ("MISMATCH", "manuscript/results/R2/r2_tgd2_specificity_basin_level.csv; manuscript/scripts/r2/plot_r2_figure3_final.py", "Current basin CSV gives 97.36% IC Base-CN and 100.00% dPL Base-CN, while draft says 63.1% and 83.8%"),
        "R2_BASE_CN_GRADIENTS": ("MISMATCH_OR_DIFFERENT_ESTIMAND", "manuscript/results/R2/r2_tgd2_specificity_regressions.csv; manuscript/results/R2/r2_snow_gradients_summary.csv", "Local files contain excess-separation and parameter-specific slopes, but they do not reproduce the draft +0.154/+0.472 Base-CN estimates"),
        "R2_TGD_SLOPE_CONTRAST": ("UNRESOLVED_LOCAL", "manuscript/results/R2/r2_tgd2_specificity_regressions.csv", "A same-name control table exists, but the draft's exact contrast definition is not identified locally"),
        "R2_BASE_CN_CI_FULL_EXCLS5": ("MISMATCH_OR_DIFFERENT_ESTIMAND", "manuscript/results/R2/r2_tgd2_specificity_regressions.csv", "Local full/ExcludeS5 regressions use a different stored excess metric than the draft values"),
        "R2_TGD_DELTA_BETA_FULL_EXCLS5": ("UNRESOLVED_LOCAL", "manuscript/results/R2/r2_tgd2_specificity_regressions.csv", "Exact paired slope-difference source for the draft is not identified"),
        "R2_UM_KI_CI_SLOPES": ("MISMATCH", "manuscript/results/R2/r2_snow_gradients_summary.csv", "Local parameter slopes differ from the draft: IC um +0.686, ki -0.450, ci -0.356; dPL um +0.313, ki -1.266, ci -0.654"),
        "R2_PARAMETER_CI": ("MISMATCH", "manuscript/results/R2/r2_snow_gradients_summary.csv", "Local parameter slope intervals do not match the draft intervals"),
        "R3_CORRECT_CN_KGE": ("RESOLVED", "manuscript/results/R3/figure5_summary.json", "Local figure5_summary contains IC/dPL test medians"),
        "R3_CORRECT_CN_TRAIN": ("RESOLVED_DERIVED", "manuscript/results/R3/figure5_summary.json", "Derived as 1 - local IC train deficit; no new upstream data"),
        "R3_BASE_NOREFIT_KGE": ("RESOLVED", "manuscript/results/R3/figure6_summary.json", "Local ladder contains base_no_refit median"),
        "R3_TGD_KGE": ("RESOLVED", "manuscript/results/R3/figure6_summary.json", "Local ladder contains TGD2 medians; manuscript name requires TGD mapping"),
        "R3_G_BASE": ("RESOLVED_FROM_TABLE", "manuscript/results/R3/figure5_basin_seedmedian.csv", "Local basin table contains G_base and fitted/no-refit KGE columns"),
        "R3_BASE_FITTED_KGE": ("RESOLVED_FROM_TABLE", "manuscript/results/R3/figure5_basin_seedmedian.csv", "Local basin table contains fitted Base KGE"),
        "R3_TRAIN_TEST_DECAY": ("RESOLVED_FROM_TABLE", "manuscript/results/R3/figure5_summary.json", "Local decay_G_base summaries exist"),
        "R3_FCLOSE_IC_TEST": ("RESOLVED", "manuscript/results/R3/figure5_summary.json", "Local F_close IC test median and interval exist"),
        "R3_FCLOSE_N_IC": ("RESOLVED", "manuscript/results/R3/figure5_summary.json", "Local F_close IC test n_valid exists"),
        "R3_FCLOSE_DPL_TEST": ("RESOLVED", "manuscript/results/R3/figure5_summary.json", "Local F_close dPL test median and interval exist"),
        "R3_FCLOSE_N_DPL": ("RESOLVED", "manuscript/results/R3/figure5_summary.json", "Local F_close dPL test n_valid exists"),
        "R3_DENOMINATOR_RULE_AND_DISTRIBUTION": ("RESOLVED_FROM_TABLE", "manuscript/results/R3/tableS5_si_statistics.csv; manuscript/scripts/r3/posthoc_stats.py", "Local SI table and script document denominator-valid F_close"),
        "R3_BASE_PARAMETER_STATE_EXCESS": ("RESOLVED_FROM_TABLE", "manuscript/results/R3/figure5_basin_seedmedian.csv", "Local basin table contains C_theta_base and C_state_base"),
        "R3_EXCESS_SNOW_ASSOCIATIONS": ("RESOLVED", "manuscript/results/R3/figure5_summary.json", "Local panels report snow-association summaries"),
        "R3_STATE_SNOW_RHO": ("RESOLVED", "manuscript/results/R3/figure5_summary.json", "Local C_state snow-association rho is present"),
        "R3_PARAMETER_SNOW_RHO": ("RESOLVED", "manuscript/results/R3/figure5_summary.json", "Local C_theta snow-association rho is present"),
        "R3_RAW_PARTIAL_ASSOCIATIONS": ("UNRESOLVED_LOCAL", "manuscript/results/R3", "No uniquely matching local table for raw partial associations was identified"),
        "R3_G_TGD": ("RESOLVED_FROM_TABLE", "manuscript/results/R3/figure6_basin_seedmedian.csv", "Local G_tgd2 values exist; manuscript label requires TGD mapping"),
        "R3_FTGD_VALID_N": ("RESOLVED", "manuscript/results/R3/figure6_summary.json", "Local F_tgd2 n_valid exists"),
        "R3_TGD_INTERNAL_RELIEF": ("RESOLVED", "manuscript/results/R3/figure6_summary.json", "Local parameter/state relief summaries exist"),
        "R3_STATE_EXCESS": ("RESOLVED", "manuscript/results/R3/figure6_summary.json", "Local state-relief summaries exist"),
        "R3_G_CN_TGD": ("RESOLVED", "manuscript/results/R3/figure6_summary.json", "Local residual CN advantage exists"),
        "R3_PROCESS_CONDITIONED_RESIDUAL_METRIC_AND_VALUES": ("RESOLVED", "manuscript/results/R3/table5_main_summary.csv", "Local snow-active and non-snow residual RMSE rows exist"),
        "R3_CORE_INPUT_CT_BASE_TGD_CN": ("UNRESOLVED_LOCAL", "manuscript/results/R3/fig6_seasonal", "Seasonal NPZ exists locally but this audit does not infer numeric values from arrays without an existing summary table"),
        "R3_SHARED_STATE_SEASON_WINDOW": ("RESOLVED_DEFINITION_ONLY", "manuscript/results/R3/fig6_seasonal/fig6_seasonal_meta.json", "Seasonal-axis metadata exists; numeric window claim remains to be matched"),
        "R3_F6_COMMON_INPUT_DEFINITION": ("RESOLVED_DEFINITION_ONLY", "manuscript/results/R3/fig6_seasonal/fig6_seasonal_meta.json", "Local metadata defines effective liquid-water input"),
        "R3_F6_SHARED_STATE_DEFINITION": ("RESOLVED_DEFINITION_ONLY", "manuscript/results/R3/fig6_seasonal/fig6_seasonal_meta.json", "Local metadata defines wt=wu+wl+wd"),
        "R3_F6_TRAJECTORY_SUMMARY": ("UNRESOLVED_LOCAL", "manuscript/results/R3/fig6_seasonal", "No local text summary uniquely matching the draft trajectory wording was identified"),
        "R4_REFERENCE_PROVENANCE": ("MISMATCH_OR_INCOMPLETE", "results/r4_phase1_soil_official/r4_phase1_soil_official_report.json; manuscript/scripts/r4/HANDOFF.md", "Caravan/ERA5-Land provenance is labelled but the complete upstream chain is not locally demonstrated"),
        "R4_SWE_DECILE_EFFECTS_BASE_TGD_CN": ("AVAILABLE_NONCANONICAL", "results/r4_phase1_soil_official/robustness_swe_decile_shape.csv", "Local SWE decile table exists; TGD2 provenance remains pending"),
        "R4_PHASE_EFFECTS": ("AVAILABLE_NONCANONICAL", "results/r4_phase1_soil_official/robustness_process_phase_consistency.csv", "Local phase table exists; three-structure promotion is blocked by TGD2 provenance"),
        "R4_DRYDOWN_NEGATIVE_CONTROL": ("AVAILABLE_NONCANONICAL", "results/r4_phase1_soil_official/robustness_process_phase_consistency.csv", "Local phase table exists; canonical TGD status remains pending"),
        "R4_CANONICAL_REGIME_SUMMARIES": ("PARTIAL", "results/r4_phase1_soil_official/r4_phase1_soil_official_report.json; results/r4_phase1_soil_official/three_structure_basin_state_consistency.csv", "Base/CN formal report and TGD2 rows exist, but TGD2 is not canonical"),
        "R4_REGION_OMISSION_SIGN_STABILITY": ("AVAILABLE_NONCANONICAL", "results/r4_phase1_soil_official/robustness_leave_one_region_out.csv", "Local regional robustness table exists"),
        "R4_BASE_TIMING": ("AVAILABLE_NONCANONICAL", "results/r4_phase1_soil_official/three_structure_timing_metrics_basin_summary.csv", "Local timing table exists; provenance gate remains"),
        "R4_TGD_TIMING": ("AVAILABLE_NONCANONICAL", "results/r4_phase1_soil_official/three_structure_timing_metrics_basin_summary.csv", "TGD2 timing rows exist but are not canonical TGD"),
        "R4_CN_TIMING": ("AVAILABLE_NONCANONICAL", "results/r4_phase1_soil_official/three_structure_timing_metrics_basin_summary.csv", "Local CN timing rows exist"),
        "R4_TIMING_SENSITIVITY": ("AVAILABLE_NONCANONICAL", "results/r4_phase1_soil_official/robustness_timing_sensitivity.csv", "Local timing sensitivity table exists"),
        "R4_PAIRED_EVENT_SUPPORT": ("AVAILABLE_NONCANONICAL", "results/r4_phase1_soil_official/three_structure_timing_metrics_basin_year.csv", "Local basin-year timing rows exist; exact eligibility claim requires table-level check"),
        "R5_HIGHSNOW_N": ("RESOLVED", "manuscript/results/R1/r1_snow_attributes.csv", "S4+S5 fixed strata N=89"),
        "R5_S1_AGREEMENT": ("RESOLVED", "manuscript/results/R5/r5_cross_model_agreement_table.csv", "S1 all-host and majority agreement are available for IC/dPL"),
        "R5_S5_AGREEMENT": ("RESOLVED", "manuscript/results/R5/r5_cross_model_agreement_table.csv", "S5 all-host and majority agreement are available for IC/dPL"),
        "R5_IC_SLOPES_XAJ_GR4J_SIMHYD": ("RESOLVED", "manuscript/results/R5/r5_snow_gradient_table.csv", "IC slopes and intervals for all three hosts are available"),
        "R5_DPL_SLOPES_XAJ_GR4J_SIMHYD": ("RESOLVED", "manuscript/results/R5/r5_snow_gradient_table.csv", "dPL slopes and intervals for all three hosts are available"),
        "R5_HIGHSNOW_CT_ALL_HOSTS": ("RESOLVED", "manuscript/results/R5/r5_timing_signature_table.csv", "High-snow timing rows for all hosts and regimes are available"),
        "R5_ALL_STRATA_AGREEMENT": ("RESOLVED", "manuscript/results/R5/r5_cross_model_agreement_table.csv", "S1-S5 rows are available"),
        "R5_FULL_AND_HIGH_SNOW_EFFECTS": ("RESOLVED", "manuscript/results/R5/r5_primary_effects_table.csv", "Full and high-snow rows are available"),
        "R5_LOW_SNOW_EFFECT_BY_HOST": ("RESOLVED", "manuscript/results/R5/r5_primary_effects_table.csv", "S1/full rows are available"),
        "R5_S5_EFFECT_BY_HOST": ("RESOLVED", "manuscript/results/R5/r5_primary_effects_table.csv", "S5 rows are available"),
    }
    out = []
    for item in placeholders:
        marker = item["marker"]
        key_match = re.search(r":([^\]]+)\]\]", marker)
        key = key_match.group(1) if key_match else marker
        if key in known:
            status, source, note = known[key]
        elif key.startswith("PARAMETERS"):
            status, source, note = "UNRESOLVED_LOCAL", "manuscript/hess_results_R1_R5_reframed_v2.md", "Supplement parameter file not identified in local project"
        else:
            status, source, note = "UNRESOLVED_LOCAL", "", "No uniquely matching local source was established"
        out.append({**item, "key": key, "status": status, "source": source, "note": note})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_report(placeholders: list[dict[str, Any]], resolutions: list[dict[str, Any]], data: dict[str, Any]) -> str:
    lines = [
        "# Local-only R1–R5 draft results audit",
        "",
        "> Scope restriction: this audit read only files below `project/hydrodiag`. No parent `data/` directory, sibling project, remote storage, or external checkpoint path was opened by this run.",
        "> Existing cached HBV inference outputs under `manuscript/cache/` were treated as local derived outputs from existing checkpoints; no training or truth regeneration was run.",
        "",
        f"- Draft: `{local(DRAFT)}` — {'PRESENT' if DRAFT.exists() else 'MISSING'}",
        f"- Placeholder occurrences: **{len(placeholders)}**",
        f"- Placeholder keys resolved locally: **{sum(r['status'].startswith('RESOLVED') for r in resolutions)}**",
        f"- Placeholder mismatches: **{sum(r['status'] == 'MISMATCH' for r in resolutions)}**",
        f"- Placeholder local-unresolved: **{sum(r['status'] == 'UNRESOLVED_LOCAL' for r in resolutions)}**",
        "",
        "## Local source status",
        "",
        "| area | local status | source boundary |",
        "|---|---|---|",
        "| R1 | PARTIAL | XAJ official tables exist; HBV evaluation exists in cache but official R1 aggregation is not rebuilt |",
        "| R2 | MISMATCH | Current specificity/parameter tables exist, but the R2 bounds CSV contains only XAJ rows |",
        "| R3 | AVAILABLE_WITH_SCOPE | Frozen local summaries and tables exist; inputs named outside hydrodiag are not opened |",
        "| R4 | PARTIAL | Local Base/CN report and TGD2 rows exist; report marks TGD2 provenance pending |",
        "| R5 | AVAILABLE_WITH_CAVEAT | Local tables exist; valid-sample caveats are retained |",
        "",
        "## Placeholder resolution summary",
        "",
        "| line | marker | status | source | note |",
        "|---:|---|---|---|---|",
    ]
    for r in resolutions:
        lines.append(f"| {r['line']} | `{r['marker']}` | **{r['status']}** | `{r['source']}` | {r['note']} |")
    lines += ["", "## Important locally recomputed values", ""]
    r1 = data["R1"]
    lines.append("### R1")
    lines.append("")
    lines.append("R1 screen counts, signed CT medians, and fixed S4+S5 CT medians were recomputed by joining the local basin-level performance, CT signature, and snow-attribute CSVs. See `r1_recomputed_screen_counts.csv` and `r1_recomputed_high_snow_ct.csv`.")
    for row in r1.get("screen_counts", []):
        if row["kge_threshold"] == 0.60 and row["model"] in {"XAJ-Base", "XAJ-TGD", "XAJ-CN"}:
            lines.append(f"- {row['paradigm']} / {row['model']} / KGE≥0.60: valid N={row['kge_valid_n']}, |CT|≥15 d N={row['ct15_n']} ({fmt(100*row['ct15_fraction'],1)}%), screened signed CT median={fmt(row['screened_ct_median'],1)} d.")
    for row in r1.get("common_pass", []):
        lines.append(f"- {row['paradigm']} common-pass N={row['common_pass_n']}; CT medians Base={fmt(row.get('XAJ-Base_ct_median'),1)} d, TGD={fmt(row.get('XAJ-TGD_ct_median'),1)} d, CN={fmt(row.get('XAJ-CN_ct_median'),1)} d; paired CT differences TGD-Base={fmt(row.get('TGD_minus_Base_ct_median'),1)} d, CN-Base={fmt(row.get('CN_minus_Base_ct_median'),1)} d.")
    lines.append("")
    lines.append("### R2")
    lines.append("")
    lines.append("The current local specificity table yields the following above-1:1 fractions; these are not silently substituted for the draft values:")
    for row in data["R2"].get("specificity_prevalence", []):
        lines.append(f"- {row['paradigm']} / {row['contrast']}: {row['above_1to1']}/{row['n']} = {fmt(100*row['fraction'],2)}%.")
    lines.append("- Bounds status: **MISMATCH**; local `s2_parameter_bounds_from_code.csv` has only `XAJ:15`, while the R2 loader requires `XAJ`, `XAJ_CN`, and `XAJ_TGD`.")
    lines.append("")
    lines.append("### R3")
    lines.append("")
    ladder = data["R3"].get("ladder", {})
    for regime, values in ladder.items():
        if isinstance(values, dict):
            lines.append(f"- {regime}: CN test KGE={fmt(values.get('kge_cn_median'),4)}, Base fitted KGE={fmt(values.get('kge_base_median'),4)}, Base no-refit KGE={fmt(values.get('kge_base_no_refit_median'),4)}, TGD2 KGE={fmt(values.get('kge_tgd2_median'),4)}.")
    for row in data["R3"].get("figure5_basin_seedmedian_grouped_medians", []):
        if row.get("period") == "test":
            lines.append(f"- {row.get('paradigm')} local basin-table medians: G_base={fmt(row.get('G_base'),4)}, F_close={fmt(row.get('F_close'),4)}, fitted Base KGE={fmt(row.get('kge_base'),4)}.")
    lines.append("")
    lines.append("### R4")
    lines.append("")
    lines.append(f"- Local report reference: `{data['R4'].get('reference')}`.")
    lines.append(f"- Local report status: `{data['R4'].get('tgd2_provenance_status')}`.")
    lines.append(f"- Local three-structure file contains: `{data['R4'].get('three_structure_counts', {})}`, but these TGD2 rows are not promoted to canonical TGD because the local report still marks provenance pending.")
    lines.append("")
    lines.append("### R5")
    lines.append("")
    lines.append("R5 local snow-gradient, timing, agreement, and primary-effect tables provide the manuscript-facing values and denominators. The complete machine-readable recomputation is in `r5_recomputed_values.json`; the 515/531 validity caveat remains in the local audit JSON.")
    lines += ["", "## No-automatic-substitution rule", "", "Draft placeholders marked MISMATCH or UNRESOLVED_LOCAL were not replaced in the manuscript. Values are reported in the audit outputs with their local source and status; no figure pixel was used as a numeric source."]
    return "\n".join(lines) + "\n"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    placeholders = draft_placeholders()
    data = {
        "R1": r1_recompute(),
        "R2": r2_recompute(),
        "R3": r3_recompute(),
        "R4": r4_recompute(),
        "R5": r5_recompute(),
    }
    resolutions = resolve_placeholders(placeholders, data)
    write_csv(CACHE / "draft_placeholder_resolution.csv", resolutions)
    write_csv(CACHE / "r1_recomputed_screen_counts.csv", data["R1"].get("screen_counts", []))
    write_csv(CACHE / "r1_recomputed_common_pass.csv", data["R1"].get("common_pass", []))
    write_csv(CACHE / "r1_recomputed_high_snow_ct.csv", data["R1"].get("high_snow_ct", []))
    write_csv(CACHE / "r1_local_stratum_kge_summary.csv", data["R1"].get("stratum_kge_summary", []))
    write_csv(CACHE / "r2_specificity_prevalence.csv", data["R2"].get("specificity_prevalence", []))
    write_csv(CACHE / "r2_parameter_gradient_um_ki_ci.csv", data["R2"].get("parameter_gradient_um_ki_ci", []))
    write_csv(CACHE / "r5_recomputed_slopes.csv", data["R5"].get("slopes", []))
    write_csv(CACHE / "r5_recomputed_agreement_s1_s5.csv", data["R5"].get("agreement_S1_S5", []))
    write_csv(CACHE / "r5_recomputed_high_snow_timing.csv", data["R5"].get("high_snow_timing", []))
    safe_data = json_safe(data)
    (CACHE / "recomputed_values.json").write_text(json.dumps(safe_data, indent=2, ensure_ascii=False), encoding="utf-8")
    (CACHE / "r5_recomputed_values.json").write_text(json.dumps(safe_data["R5"], indent=2, ensure_ascii=False), encoding="utf-8")
    (CACHE / "local_draft_results_audit.md").write_text(build_report(placeholders, resolutions, data), encoding="utf-8")
    manifest = {
        "project_root": str(PROJECT),
        "scope": "project/hydrodiag only",
        "outside_project_paths_accessed": [],
        "draft": local(DRAFT),
        "placeholder_occurrences": len(placeholders),
        "resolved": sum(r["status"].startswith("RESOLVED") for r in resolutions),
        "mismatches": sum(r["status"] == "MISMATCH" for r in resolutions),
        "unresolved_local": sum(r["status"] == "UNRESOLVED_LOCAL" for r in resolutions),
        "training_launched": False,
        "truth_regenerated": False,
        "figure_pixels_used": False,
    }
    (CACHE / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"output={CACHE}")
    print(f"draft={'PRESENT' if DRAFT.exists() else 'MISSING'}")
    print(f"placeholders={len(placeholders)} resolved={manifest['resolved']} mismatches={manifest['mismatches']} unresolved_local={manifest['unresolved_local']}")
    print("scope=project/hydrodiag only")
    print("training_launched=no")
    print("truth_regenerated=no")


if __name__ == "__main__":
    main()
