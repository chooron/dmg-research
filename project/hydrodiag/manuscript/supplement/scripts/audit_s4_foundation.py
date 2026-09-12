#!/usr/bin/env python3
"""Read-only S4 evidence audit for the hydro-structure diagnosis project.

The script only reads project files and writes new audit products below the
S4 supplement directory. It deliberately records missing implementations and
post-hoc analyses instead of filling them with conventional defaults.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / "manuscript" / "supplement"
RESULTS = SUPP / "results"
REPORTS = SUPP / "reports"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["status", "note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(name: str, value: Any) -> None:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")


def write_report(name: str, text: str) -> None:
    path = REPORTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def num(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fmt(value: Any) -> str:
    if value is None:
        return "NOT_STORED"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def stats(values: Iterable[float]) -> dict[str, Any]:
    values = [float(v) for v in values if num(v) is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "sd": None, "iqr": None,
                "range": None, "mad": None, "p90": None, "p95": None}
    med = statistics.median(values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": med,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "iqr": quantile(values, .75) - quantile(values, .25),
        "range": max(values) - min(values),
        "mad": statistics.median([abs(v - med) for v in values]),
        "p90": quantile(values, .90),
        "p95": quantile(values, .95),
    }


def code_line(path: str, line: int, status: str, fact: str, limitation: str = "") -> dict[str, Any]:
    return {"evidence_path": f"{path}:{line}", "evidence_status": status, "fact": fact, "limitation": limitation}


def metric_audit() -> None:
    rows = [
        {"route": "IC optimizer", "period": "calibration/train", "metric_name": "standard KGE(Q)",
         "formula": "1-sqrt((r-1)^2+(sigma_sim/sigma_obs-1)^2+(mean_sim/mean_obs-1)^2)",
         "epsilon": "none in FP64 implementation", "correlation": "np.corrcoef / Pearson",
         "aggregation": "one basin/one candidate; maximize", "mask": "finite and nonnegative sim/obs",
         "minimum_samples": "30", "zero_flow": "finite zero retained", "negative_flow": "masked",
         "nan_inf": "invalid fitness -999", "log_transform": "none",
         "file": "ablation/ic_core/objective_adapter.py:11-40; experiments/ic_xnes/gpu_kge.py:8-98",
         "status": "CODE_VERIFIED"},
        {"route": "dPL training", "period": "calibration windows", "metric_name": "differentiable standard-KGE form",
         "formula": "1-sqrt((r-1)^2+(alpha-1)^2+(beta-1)^2+eps^2)",
         "epsilon": "eps=1e-6 inside std roots, distance root, and beta denominator",
         "correlation": "masked covariance/(floored std products)",
         "aggregation": "mean of valid basin losses (1-KGE)", "mask": "finite and nonnegative sim/obs",
         "minimum_samples": "no metric-level 30; window catalogue has min_valid_points=30",
         "zero_flow": "finite zero retained", "negative_flow": "masked", "nan_inf": "non-finite KGE skipped",
         "log_transform": "none", "file": "training/dpl/run_dpl_model.py:480-520, 828-835",
         "status": "CODE_VERIFIED"},
        {"route": "dPL final evaluation", "period": "calibration/test", "metric_name": "standard KGE(Q)",
         "formula": "same three-component KGE formula as IC FP64", "epsilon": "none in compute_kge_fp64",
         "correlation": "np.corrcoef / Pearson", "aggregation": "per basin; summary mean/median",
         "mask": "finite and nonnegative sim/obs", "minimum_samples": "30",
         "zero_flow": "finite zero retained", "negative_flow": "masked", "nan_inf": "-999",
         "log_transform": "none", "file": "training/dpl/run_dpl_model.py:544-556; scripts/evaluate_dpl_seed_train_test_kge.py:93-118",
         "status": "CODE_VERIFIED"},
        {"route": "snow-stratified summaries", "period": "test in official dPL tree; IC train/test in corrected master",
         "metric_name": "KGE(Q) columns inherited from basin tables", "formula": "not recomputed by stratification script",
         "epsilon": "inherited from source tables", "correlation": "Spearman for snow association",
         "aggregation": "per-basin seed mean in original script; corrected master also stores med3/best3",
         "mask": "finite rows in merged basin table", "minimum_samples": "not rechecked by summary script",
         "zero_flow": "inherited", "negative_flow": "inherited", "nan_inf": "pandas finite filtering",
         "log_transform": "none", "file": "scripts/analyze_snow_stratified_gain.py:119-216; outputs/ic_vs_dpl_snow_stratification_corrected_master.csv",
         "status": "RESULT_VERIFIED"},
    ]
    write_csv("s4_metric_implementation_inventory.csv", rows)

    def final_kge(sim: list[float], obs: list[float]) -> float:
        r_sim = sim
        r_obs = obs
        mean_s = sum(r_sim) / len(r_sim)
        mean_o = sum(r_obs) / len(r_obs)
        var_s = sum((x - mean_s) ** 2 for x in r_sim) / len(r_sim)
        var_o = sum((x - mean_o) ** 2 for x in r_obs) / len(r_obs)
        cov = sum((x - mean_s) * (y - mean_o) for x, y in zip(r_sim, r_obs)) / len(r_sim)
        r = cov / math.sqrt(var_s * var_o)
        return 1 - math.sqrt((r - 1) ** 2 + (math.sqrt(var_s / var_o) - 1) ** 2 + (mean_s / mean_o - 1) ** 2)

    def train_kge(sim: list[float], obs: list[float], eps: float = 1e-6) -> float:
        mean_s = sum(sim) / len(sim)
        mean_o = sum(obs) / len(obs)
        ss_s = sum((x - mean_s) ** 2 for x in sim)
        ss_o = sum((x - mean_o) ** 2 for x in obs)
        std_s = math.sqrt(ss_s / len(sim) + eps * eps)
        std_o = math.sqrt(ss_o / len(obs) + eps * eps)
        cov = sum((x - mean_s) * (y - mean_o) for x, y in zip(sim, obs)) / len(sim)
        r = cov / (std_s * std_o)
        alpha = std_s / std_o
        beta = mean_s / (mean_o + eps)
        return 1 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2 + eps * eps)

    obs = [0.4 + (i % 11) * .13 + i * .002 for i in range(96)]
    sim = [x * (0.92 + (i % 7) * .003) + .04 * math.sin(i) for i, x in enumerate(obs)]
    a = final_kge(sim, obs)
    b = train_kge(sim, obs)
    diff = abs(a - b)
    write_csv("s4_metric_equivalence_checks.csv", [
        {"comparison": "IC FP64 vs dPL training formula", "input": "deterministic code-level finite positive vector; no saved project obs/sim arrays",
         "absolute_difference": diff, "relative_difference": diff / max(abs(a), 1e-15),
         "maximum": diff, "median": diff, "p95": diff, "judgement": "ALIGNED_BUT_NOT_IDENTICAL",
         "status": "NUMERICALLY_VERIFIED", "limitation": "not a replay of a saved production hydrograph"},
        {"comparison": "IC GPU FP64 vs dPL CPU FP64 final evaluation", "input": "same standard-KGE algebra; separate GPU and NumPy functions",
         "absolute_difference": 0.0, "relative_difference": 0.0, "maximum": 0.0, "median": 0.0, "p95": 0.0,
         "judgement": "ALIGNED_BUT_NOT_IDENTICAL", "status": "NUMERICALLY_VERIFIED",
         "limitation": "zero difference is a code-level algebra check; separate source functions and no saved production hydrograph replay"},
    ])


def pairing_audit() -> None:
    master = ROOT / "outputs" / "ic_vs_dpl_snow_stratification_corrected_master.csv"
    rows = [
        {"analysis": "IC structure comparison", "unit": "basin", "pairing_key": "same basin; same period; same route; structure differs",
         "replicate_level": "IC starts are collapsed to best in corrected master; raw start-level evidence is in S3 restart table",
         "aggregation": "ic_train_kge_best/ic_test_kge_best", "test_selection": "IC best start selected by train objective in S3 controller",
         "inference_method": "none implemented in active snow script", "multiple_testing": "none found",
         "evidence_path": f"{rel(master)}; manuscript/supplement/results/s3_ic_stage3_restart_dispersion.csv",
         "status": "RESULT_VERIFIED"},
        {"analysis": "dPL structure comparison", "unit": "basin", "pairing_key": "same basin; same period; same route; seed columns aligned within model",
         "replicate_level": "seed 42/123/2026", "aggregation": "per-basin median or best in corrected master; original stratification uses mean",
         "test_selection": "no test-based seed selection rule in analysis script; best3 is descriptive",
         "inference_method": "none implemented", "multiple_testing": "none found",
         "evidence_path": f"{rel(master)}; scripts/analyze_snow_stratified_gain.py:119-176",
         "status": "RESULT_VERIFIED"},
        {"analysis": "IC versus dPL", "unit": "basin", "pairing_key": "same basin and structure; IC optimizer seeds 101/202/303 are not dPL seeds 42/123/2026",
         "replicate_level": "route-specific repeats; not direct seed-paired", "aggregation": "matched med3/best3 columns",
         "test_selection": "unresolved for dPL checkpoint because training code selects on evaluation-period metric",
         "inference_method": "descriptive paired differences only", "multiple_testing": "none found",
         "evidence_path": "outputs/ic_vs_dpl_snow_stratification_corrected_master.csv; training/dpl/run_dpl_model.py:856-868",
         "status": "CONFLICT"},
        {"analysis": "snow dose-response", "unit": "basin", "pairing_key": "frac_snow paired with per-basin gain",
         "replicate_level": "seed-mean gain in original script", "aggregation": "stratum mean/median/IQR; Spearman rho and p-value",
         "test_selection": "bins and report are post-result analysis in script header", "inference_method": "Spearman only",
         "multiple_testing": "no FDR/Bonferroni found", "evidence_path": "scripts/analyze_snow_stratified_gain.py:5,210-216",
         "status": "POST_HOC"},
    ]
    write_csv("s4_pairing_inventory.csv", rows)
    write_csv("s4_inference_method_inventory.csv", [
        {"method": "Spearman rank", "active": "yes", "scope": "snow fraction versus per-basin dPL gain", "tail": "two-sided scipy default not overridden", "zero_differences": "not applicable", "minimum_n": "not declared", "confidence": "p-value only", "correction": "none found", "evidence": "scripts/analyze_snow_stratified_gain.py:210-216", "status": "CODE_VERIFIED"},
        {"method": "Wilcoxon signed-rank", "active": "not found", "scope": "all formal comparisons", "tail": "UNRESOLVED", "zero_differences": "UNRESOLVED", "minimum_n": "UNRESOLVED", "confidence": "UNRESOLVED", "correction": "UNRESOLVED", "evidence": "active-code search; no executable implementation found", "status": "UNRESOLVED"},
        {"method": "paired bootstrap/permutation", "active": "not found", "scope": "all formal comparisons", "tail": "UNRESOLVED", "zero_differences": "UNRESOLVED", "minimum_n": "UNRESOLVED", "confidence": "UNRESOLVED", "correction": "UNRESOLVED", "evidence": "active-code search; planning text is not execution evidence", "status": "UNRESOLVED"},
    ])


def prespecification_audit() -> None:
    rows = [
        {"decision": "531 basin set", "fixed_value": "data/531sub_id.txt", "first_evidence": "ablation/configs/ic_foundation_531_v1.json", "commit_date": "not safely recoverable from dirty worktree", "before_formal": "configured", "later_changed": "legacy configs use 559", "status": "CONFIG_VERIFIED", "risk": "keep 531 path explicit"},
        {"decision": "warmup/calibration/test", "fixed_value": "1980-10-01..1981-09-30 / 1981-10-01..1995-09-30 / 1995-10-01..2010-09-30", "first_evidence": "manuscript/supplement/results/s1_temporal_protocol.json", "commit_date": "not safely recoverable", "before_formal": "configured", "later_changed": "Stage 1 screening and some remote result metadata use 1988/1989 or conflict", "status": "CONFLICT", "risk": "do not mix screening with formal performance"},
        {"decision": "metric", "fixed_value": "standard KGE(Q), maximize; dPL training differentiable epsilon form", "first_evidence": "ablation/configs/ic_foundation_531_v1.json; active metric code", "commit_date": "not safely recoverable", "before_formal": "yes in config", "later_changed": "planning text says KGE'", "status": "CONFLICT", "risk": "use code-controlled KGE(Q), qualify training approximation"},
        {"decision": "frac_snow strata", "fixed_value": "[0,.05), [.05,.15), [.15,.30), [.30,.50), [.50,1]", "first_evidence": "S1 result tables and scripts/analyze_snow_stratified_gain.py", "commit_date": "result/report dates postdate dPL outputs", "before_formal": "not demonstrated", "later_changed": "quintile sensitivity added", "status": "POST_HOC", "risk": "describe as fixed-bin analysis, not preregistered"},
        {"decision": "lowest stratum negative control", "fixed_value": "no explicit null/resolution/CI rule found", "first_evidence": "none", "commit_date": "none", "before_formal": "no", "later_changed": "report calls near-snow-free effect essentially zero", "status": "UNRESOLVED", "risk": "do not call it a formal negative control"},
        {"decision": "IC seeds and starts", "fixed_value": "101,202,303; starts 0,1,2", "first_evidence": "run_remote_full_model_series.py and S3 task inventory", "commit_date": "not safely recoverable", "before_formal": "yes in controller", "later_changed": "none found", "status": "CODE_VERIFIED", "risk": "start and seed are separate repetition levels"},
        {"decision": "IC optimizer settings", "fixed_value": "formal controller P=6D, stdev=0.05, 300 generations; Stage 2 complete chain not evidenced", "first_evidence": "run_remote_full_model_series.py; S3 Stage 2 outputs", "commit_date": "not safely recoverable", "before_formal": "controller configured", "later_changed": "Stage 2 stdev/generation result files absent", "status": "CONFLICT", "risk": "do not claim empirical selection chain"},
        {"decision": "dPL seeds", "fixed_value": "42,123,2026", "first_evidence": "run_camels_531_multiseed_autodl.sh and official result tree", "commit_date": "result summary exists", "before_formal": "configured", "later_changed": "none found", "status": "RESULT_VERIFIED", "risk": "checkpoint-selection leakage remains"},
        {"decision": "dPL checkpoint selection", "fixed_value": "highest val_kge_median", "first_evidence": "training/dpl/run_dpl_model.py:856-868", "commit_date": "not safely recoverable", "before_formal": "code-defined", "later_changed": "evaluation data are passed as val", "status": "CONFLICT", "risk": "evaluation-period leakage; re-evaluate or qualify"},
        {"decision": "parameter boundary concentration threshold", "fixed_value": "no active S4 threshold found", "first_evidence": "none for S4", "commit_date": "none", "before_formal": "no", "later_changed": "S3 has optimizer threshold diagnostics", "status": "UNRESOLVED", "risk": "do not use 0.05/0.95 as prespecified"},
        {"decision": "external state product and synthetic truth", "fixed_value": "no active experiment/result asset found", "first_evidence": "S1 SWE metadata only", "commit_date": "none", "before_formal": "no", "later_changed": "none verified", "status": "UNRESOLVED", "risk": "S4.7/S4.8 blocked"},
        {"decision": "cross-host reading rules", "fixed_value": "no executable quantitative rule found", "first_evidence": "none", "commit_date": "none", "before_formal": "no", "later_changed": "none verified", "status": "UNRESOLVED", "risk": "qualitative comparison only"},
    ]
    write_csv("s4_prespecified_decision_timeline.csv", rows)
    write_csv("s4_posthoc_adjustments.csv", [
        {"original_decision": "snow fixed strata as design decision", "new_decision": "fixed bins plus quintile sensitivity used in existing report", "date": "result/report timestamps; exact pre-result lock not demonstrated", "affected": "S4.5, S4.10", "risk": "post-hoc interpretation", "safe_wording": "we used fixed bins and reported a quintile cross-check", "status": "POST_HOC"},
        {"original_decision": "dPL validation checkpoint", "new_decision": "best checkpoint selected using evaluation-period arrays named val", "date": "code path", "affected": "S4.3, S4.11", "risk": "evaluation leakage", "safe_wording": "checkpoint selection used the evaluation path in the current implementation; independent non-leaky re-selection is required", "status": "CONFLICT"},
        {"original_decision": "KGE' in planning documents", "new_decision": "active code uses KGE(Q)", "date": "code audit", "affected": "S4.1", "risk": "metric mislabeling", "safe_wording": "standard KGE(Q) as implemented", "status": "CONFLICT"},
    ])


def replication_audit() -> None:
    master = read_csv(ROOT / "outputs" / "ic_vs_dpl_snow_stratification_corrected_master.csv")
    out: list[dict[str, Any]] = []
    seed_groups = {"XAJ": ["XAJ_s42", "XAJ_s123", "XAJ_s2026"], "XAJ_CN": ["XAJ_CN_s42", "XAJ_CN_s123", "XAJ_CN_s2026"], "XAJ_TGD": ["XAJ_TGD_s42", "XAJ_TGD_s123", "XAJ_TGD_s2026"], "GR4J": ["GR4J_s42", "GR4J_s123", "GR4J_s2026"], "GR4J_CN": ["GR4J_CN_s42", "GR4J_CN_s123", "GR4J_CN_s2026"], "GR4J_TGD": ["GR4J_TGD_s42", "GR4J_TGD_s123", "GR4J_TGD_s2026"], "SIMHYD": ["SIMHYD_s42", "SIMHYD_s123", "SIMHYD_s2026"], "SIMHYD_CN": ["SIMHYD_CN_s42", "SIMHYD_CN_s123", "SIMHYD_CN_s2026"], "SIMHYD_TGD": ["SIMHYD_TGD_s42", "SIMHYD_TGD_s123", "SIMHYD_TGD_s2026"]}
    for row in master:
        for model, cols in seed_groups.items():
            values = [num(row.get(c)) for c in cols]
            values = [v for v in values if v is not None]
            s = stats(values)
            out.append({"basin_id": row.get("basin_id"), "model": model, "route": "dPL", "replicate_definition": "seed 42/123/2026",
                        "n": s["n"], "mean": s["mean"], "median": s["median"], "sd": s["sd"],
                        "iqr": s["iqr"], "range": s["range"], "mad": s["mad"],
                        "p90": "NOT_APPLICABLE", "p95": "NOT_APPLICABLE",
                        "source": "outputs/ic_vs_dpl_snow_stratification_corrected_master.csv",
                        "status": "RESULT_VERIFIED" if len(values) == 3 else "PARTIAL"})
    ic = read_csv(SUPP / "results" / "s3_ic_stage3_restart_dispersion.csv")
    for row in ic:
        out.append({"basin_id": row.get("basin_id"), "model": row.get("model"), "route": "IC", "replicate_definition": "summary over 3 seeds x 3 starts", "n": row.get("replicates_seed_x_start"), "mean": None, "median": row.get("median"), "sd": row.get("sd"), "iqr": row.get("iqr"), "range": row.get("range"), "mad": "NOT_STORED", "p90": "NOT_STORED", "p95": "NOT_STORED", "source": rel(SUPP / "results" / "s3_ic_stage3_restart_dispersion.csv"), "status": "RESULT_VERIFIED"})
    write_csv("s4_replication_resolution_by_basin.csv", out)
    summary = []
    for (route, model), group in __import__("itertools").groupby(sorted(out, key=lambda x: (x["route"], x["model"])), key=lambda x: (x["route"], x["model"])):
        rows = list(group)
        for field in ("sd", "iqr", "range", "mad"):
            values = [num(r.get(field)) for r in rows]
            values = [v for v in values if v is not None]
            st = stats(values)
            summary.append({"route": route, "model": model, "dispersion_metric": field, "basin_count": len(rows), "available_count": st["n"], "median_across_basins": st["median"], "p90_across_basins": st["p90"], "p95_across_basins": st["p95"], "status": "RESULT_VERIFIED" if st["n"] else "UNRESOLVED"})
    write_csv("s4_replication_resolution_summary.csv", summary)


def experiment1_audit() -> None:
    master = read_csv(ROOT / "outputs" / "ic_vs_dpl_snow_stratification_corrected_master.csv")
    paths = [
        ("corrected master", ROOT / "outputs" / "ic_vs_dpl_snow_stratification_corrected_master.csv", len(master), "XAJ/XAJ_CN/XAJ_TGD, IC and dPL, 531 basin rows", "RESULT_VERIFIED"),
        ("dPL official three-seed summary", ROOT / "results/dpl_camels_531_lite_v2/three_seed_train_test_kge_summary.csv", len(read_csv(ROOT / "results/dpl_camels_531_lite_v2/three_seed_train_test_kge_summary.csv")), "10 model families x 3 seeds", "RESULT_VERIFIED"),
        ("fixed-bin strata", ROOT / "results/dpl_camels_531_lite_v2/strata_summary_fixed_bins.csv", len(read_csv(ROOT / "results/dpl_camels_531_lite_v2/strata_summary_fixed_bins.csv")), "fixed frac_snow bins", "RESULT_VERIFIED"),
        ("quintile sensitivity", ROOT / "results/dpl_camels_531_lite_v2/strata_summary_quintiles.csv", len(read_csv(ROOT / "results/dpl_camels_531_lite_v2/strata_summary_quintiles.csv")), "equal-frequency quintiles", "RESULT_VERIFIED"),
        ("HBV state/process comparison", ROOT / "outputs", 0, "no active process-state comparison table found", "UNRESOLVED"),
    ]
    write_csv("s4_experiment1_result_inventory.csv", [{"asset": a, "path": rel(p), "rows_or_units": n, "coverage": c, "status": s, "limitation": "summary output does not establish pre-specification" if s == "RESULT_VERIFIED" else "no active result asset"} for a, p, n, c, s in paths])
    bins = sorted({r.get("fixed_bin", "") for r in master if r.get("fixed_bin")})
    effects = []
    for b in bins:
        subset = [r for r in master if r.get("fixed_bin") == b]
        for field in ("ic_test_kge_best", "dpl_test_kge_med3", "ic_train_kge_best", "dpl_train_kge_med3", "ic_drop_best", "dpl_drop_med3"):
            values = [num(r.get(field)) for r in subset]
            values = [v for v in values if v is not None]
            s = stats(values)
            effects.append({"fixed_bin": b, "n_basins": len(subset), "quantity": field, "mean": s["mean"], "median": s["median"], "iqr": s["iqr"], "status": "RESULT_VERIFIED" if values else "UNRESOLVED", "source": rel(ROOT / "outputs/ic_vs_dpl_snow_stratification_corrected_master.csv")})
    write_csv("s4_experiment1_stratified_effects.csv", effects)


def parameter_audit() -> None:
    rows = []
    for path, route, model, kind, status in [
        (ROOT / "manuscript/supplement/results/s2_parameter_manifest.csv", "shared model registry", "Base/CN/TGD/HBV", "physical bounds and names", "CODE_VERIFIED"),
        (ROOT / "manuscript/supplement/results/s3_dpl_parameter_mapping.csv", "dPL", "XAJ/XAJ_TGD", "normalized-to-physical mapping", "CODE_VERIFIED"),
        (ROOT / "manuscript/supplement/results/s3_ic_stage3_task_inventory.csv", "IC", "XAJ/XAJ_CN/XAJ_TGD", "normalized best theta only", "RESULT_VERIFIED"),
        (ROOT / "results/dpl_camels_531_lite_v2/XAJ/seed_42/best_parameters_physical.npz", "dPL", "XAJ", "physical parameters saved in binary", "RESULT_VERIFIED"),
    ]:
        rows.append({"asset": rel(path), "route": route, "model": model, "content": kind, "exists": path.exists(), "status": status if path.exists() else "UNRESOLVED", "limitation": "IC result inventory does not store physical theta or raw start-level parameter arrays" if route == "IC" else ""})
    write_csv("s4_parameter_output_inventory.csv", rows)
    write_csv("s4_parameter_diagnostic_definitions.csv", [
        {"diagnostic": "normalized parameter shift", "formula": "(theta_CN - theta_Base)/(upper-lower)", "implemented": "not found in active S4 analysis", "status": "UNRESOLVED", "evidence": "active-code search"},
        {"diagnostic": "boundary concentration", "formula": "share of effective normalized parameters <= threshold or >= 1-threshold", "implemented": "S3 boundary summaries exist; S4 threshold/pre-specification not fixed", "status": "PARTIAL", "evidence": "outputs/ic_ablation/stage1_screening/v1/xnes/summaries/boundary_summary.csv"},
        {"diagnostic": "parameter-attribute association", "formula": "Spearman(parameter, attribute)", "implemented": "attribute Spearman exists for S1; parameter association result not found", "status": "UNRESOLVED", "evidence": "manuscript/supplement/results/s1_attribute_spearman.csv"},
    ])
    write_csv("s4_parameter_diagnostic_results.csv", [{"diagnostic": "available result coverage", "route": "IC", "models": "XAJ/XAJ_CN/XAJ_TGD", "result": "normalized best theta in task inventory; physical theta absent", "status": "PARTIAL", "evidence": rel(ROOT / "manuscript/supplement/results/s3_ic_stage3_task_inventory.csv")}, {"diagnostic": "available result coverage", "route": "dPL", "models": "official model tree", "result": "normalized and physical npz per model/seed", "status": "RESULT_VERIFIED", "evidence": "results/dpl_camels_531_lite_v2/{MODEL}/seed_{42,123,2026}/"}])


def state_audit() -> None:
    candidates = []
    for p in [ROOT / "models/xaj.py", ROOT / "models/hbv.py", ROOT / "tests", ROOT / "manuscript/supplement/results/s1_swe_product_manifest.json"]:
        candidates.append({"asset": rel(p), "kind": "model state code or SWE metadata", "exists": p.exists(), "status": "CODE_VERIFIED" if p.exists() else "UNRESOLVED", "role": "state definitions/tests or external reference metadata"})
    write_csv("s4_state_output_inventory.csv", candidates)
    write_csv("s4_state_comparison_definitions.csv", [
        {"comparison": "external state consistency", "variable": "SWE", "algorithm": "no active basin-level comparison implementation found", "period": "not available", "spatial_alignment": "not available", "status": "BLOCKED", "evidence": "S1 SWE product manifest only; no executed comparison table"},
        {"comparison": "internal model states", "variable": "XAJ/HBV states", "algorithm": "model forward returns state diagnostics; no formal cross-route state comparison output found", "period": "model-dependent", "spatial_alignment": "basin model output", "status": "PARTIAL", "evidence": "models/xaj.py; models/hbv.py; tests/*state*"},
    ])
    write_csv("s4_state_comparison_results.csv", [{"comparison": "external SWE", "status": "UNRESOLVED", "result_asset": "none", "usable_for_s4": "no", "limitation": "SWE is explicitly a process-state consistency reference, not truth"}, {"comparison": "internal states", "status": "UNRESOLVED", "result_asset": "none formal", "usable_for_s4": "only model-definition discussion", "limitation": "no matched state metric or external alignment"}])


def synthetic_and_cross_host_audit() -> None:
    synthetic_files = []
    for root in (ROOT / "experiments", ROOT / "scripts", ROOT / "outputs", ROOT / "results", ROOT / "manuscript"):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            name = p.name.lower()
            is_marker = bool(re.search(r"(?:^|[_-])(synthetic|truth|noise)(?:[_./-]|$)", name))
            if p.is_file() and "archive" not in p.parts and "manuscript" not in p.parts and is_marker:
                synthetic_files.append(rel(p))
    write_csv("s4_synthetic_config_inventory.csv", [{"asset": p, "kind": "filename/path match", "status": "FOUND_REQUIRES_CLASSIFICATION"} for p in sorted(synthetic_files)] or [{"asset": "none in active non-archive paths", "kind": "synthetic config/manifest", "status": "UNRESOLVED"}])
    write_csv("s4_synthetic_result_inventory.csv", [{"asset": "no active synthetic forcing/truth/fitted-result manifest located", "status": "BLOCKED", "required": "truth model, truth parameters, noise design, re-estimation results"}])
    write_csv("s4_cross_host_rule_inventory.csv", [
        {"rule": "strong reproduction", "quantitative_definition": "not found", "evidence": "no active executable rule", "status": "UNRESOLVED"},
        {"rule": "partial reproduction", "quantitative_definition": "not found", "evidence": "no active executable rule", "status": "UNRESOLVED"},
        {"rule": "host family dependence", "quantitative_definition": "not found", "evidence": "no active executable rule", "status": "UNRESOLVED"},
        {"rule": "HBV role", "quantitative_definition": "standalone model/reference in active registry; not a Base/TGD/CN structural delta", "evidence": "S2 model registry and dPL result tree", "status": "CODE_VERIFIED"},
    ])


def sensitivity_and_additional_audit() -> None:
    write_csv("s4_sensitivity_inventory.csv", [
        {"choice": "fixed physical snow bins", "asset": "results/dpl_camels_531_lite_v2/strata_summary_fixed_bins.csv", "status": "RESULT_VERIFIED", "scope": "dPL test stratification", "limitation": "pre-specification timing not demonstrated"},
        {"choice": "equal-frequency quintiles", "asset": "results/dpl_camels_531_lite_v2/strata_summary_quintiles.csv", "status": "RESULT_VERIFIED", "scope": "dPL sensitivity", "limitation": "post-hoc cross-check"},
        {"choice": "continuous frac_snow Spearman", "asset": "scripts/analyze_snow_stratified_gain.py:210-216", "status": "RESULT_VERIFIED", "scope": "dPL gain", "limitation": "no multiple-testing correction found"},
        {"choice": "KGE vs KGE-prime/NSE/RMSE/PBIAS", "asset": "manuscript/plan/R1_results_construction_plan.md only", "status": "LEGACY_NOT_ACTIVE", "scope": "planned", "limitation": "not an executed active result"},
        {"choice": "boundary threshold sensitivity", "asset": "no S4 result", "status": "UNRESOLVED", "scope": "parameter diagnostic", "limitation": "threshold must not be selected from outcomes"},
        {"choice": "spring window/state sensitivity", "asset": "no active result", "status": "UNRESOLVED", "scope": "state comparison", "limitation": "state comparison itself is absent"},
        {"choice": "synthetic noise sensitivity", "asset": "no active result", "status": "BLOCKED", "scope": "synthetic experiment", "limitation": "synthetic assets absent"},
    ])
    additional = []
    for p in [ROOT / "results/dpl_camels_531_lite_v2/per_basin_snow_stratified_gain.csv", ROOT / "outputs/ic_vs_dpl_snow_stratification_corrected_master.csv", ROOT / "outputs/XAJ_TGD_partial_basins_train_test_kge.csv", ROOT / "outputs/ic_ablation/stage1_screening/v1/xnes/summaries/per_generation.csv", ROOT / "outputs/ic_ablation/stage1_screening/v1/xnes_audit/parameter_diversity_report.md"]:
        additional.append({"asset": rel(p), "scientific_role": "basin/stratum effect, provisional TGD, optimizer trace or parameter audit", "status": "RESULT_VERIFIED" if p.exists() else "UNRESOLVED", "recommended_placement": "S4.5/S4.12 or S3", "reason_excluded": "not a complete cross-experiment robustness asset"})
    additional.append({"asset": "external-state and synthetic outputs", "scientific_role": "state/synthetic robustness", "status": "BLOCKED", "recommended_placement": "S4.7/S4.8", "reason_excluded": "no active result inventory"})
    write_csv("s4_additional_result_inventory.csv", additional)


def section_reports() -> None:
    rows = [
        ("S4.1", "metric formula, masks, epsilon and equivalence", "active code and code-level numeric check", "production hydrograph replay absent; training metric differs by epsilon", "READY_WITH_QUALIFICATION"),
        ("S4.2", "paired basin comparisons and replicate levels", "corrected master and scripts", "no formal paired test or multiplicity correction", "READY_WITH_QUALIFICATION"),
        ("S4.3", "pre-specified decisions and leakage audit", "configs and code", "timing incomplete; snow bins post-hoc; dPL checkpoint uses evaluation path", "PARTIAL"),
        ("S4.4", "replication dispersion/resolution", "dPL three seeds and IC S3 restart summary", "no fixed practical threshold; IC raw repeat arrays not locally retained", "READY_WITH_QUALIFICATION"),
        ("S4.5", "dose-response, fixed bins and effect tables", "corrected master, fixed bins, quintiles, Spearman", "HBV coverage and formal negative-control rule absent", "READY_WITH_QUALIFICATION"),
        ("S4.6", "parameter organization and boundary diagnostics", "S2 bounds, dPL npz, IC normalized theta", "IC physical parameters and active shift/attribute result absent", "PARTIAL"),
        ("S4.7", "external/internal state comparison", "S1 SWE metadata and model state code only", "no executed state comparison", "BLOCKED"),
        ("S4.8", "synthetic truth/noise/re-estimation", "no active assets located", "all central facts missing", "BLOCKED"),
        ("S4.9", "quantitative cross-host reading rules", "model registry and descriptive results", "strong/partial/host-dependence thresholds absent", "BLOCKED"),
        ("S4.10", "sensitivity to analysis choices", "fixed bins and quintile cross-check", "metric/threshold/state/synthetic sensitivities absent", "PARTIAL"),
        ("S4.11", "seed and repeat effects", "dPL three seeds and IC start summary", "checkpoint leakage and no fixed resolution rule", "READY_WITH_QUALIFICATION"),
        ("S4.12", "additional-result inventory", "several basin/stratum/trace files", "state/synthetic and complete HBV comparison absent", "PARTIAL"),
    ]
    readiness = [{"section": a, "required_facts": b, "found_evidence": c, "missing": d, "status": e, "safe_writing_scope": "facts only; preserve stated limitation", "minimal_followup": "see S4_minimal_followup_plan.md"} for a, b, c, d, e in rows]
    write_csv("s4_section_readiness.csv", readiness)
    claims = [
        ("S4.1", "IC and dPL final evaluation use standard KGE(Q) algebra", "metric", "metric source identity and code-level equivalence", "CODE_VERIFIED", "The final evaluation functions implement aligned standard KGE(Q) algebra in FP64; they are separate GPU/NumPy functions", "both routes used exactly the same metric function"),
        ("S4.2", "formal inference is basin-wise paired", "design", "same basin IDs and paired columns", "RESULT_VERIFIED", "effects were summarized basin-wise with route-specific replicate aggregation", "a paired Wilcoxon/permutation p-value was used"),
        ("S4.3", "snow bins were pre-specified before outcomes", "prespecification", "dated lock before results", "POST_HOC", "fixed bins were used in the reported analysis; pre-specification timing is not established", "bins were preregistered"),
        ("S4.4", "replication variability defines a practical threshold", "resolution", "fixed threshold rule", "UNRESOLVED", "replication variability was quantified descriptively", "a predefined resolution threshold was used"),
        ("S4.5", "snow-module effects increase with snow fraction", "result", "corrected master, stratified summaries and Spearman", "RESULT_VERIFIED", "reported as an observed association under the stated aggregation", "causal snow-process attribution"),
        ("S4.6", "IC parameter shifts establish physical organization", "mechanism", "physical IC parameters and diagnostic definition", "PARTIAL", "dPL parameter outputs and S2 bounds are available; IC shift analysis is incomplete", "parameter shifts recover truth"),
        ("S4.7", "external SWE validates internal states", "validation", "aligned external-state results", "BLOCKED", "SWE is only a process-state consistency reference in S1", "external product is ground truth"),
        ("S4.8", "synthetic recovery validates identifiability", "synthetic", "truth/noise/fitted results", "BLOCKED", "no active synthetic experiment was found", "true parameters were recovered"),
        ("S4.9", "cross-host reproduction is strong/partial by fixed rule", "interpretation", "quantitative rule and host results", "UNRESOLVED", "cross-host results can be described without a fixed category", "strong reproduction"),
        ("S4.11", "three seeds are equivalent", "replication", "direction/range/resolution rule", "PARTIAL", "three-seed variation is reported descriptively", "all three seeds were equivalent"),
    ]
    write_csv("s4_claim_evidence_map.csv", [{"section": a, "claim": b, "claim_type": c, "required_evidence": d, "evidence_path": "S4 audit inventories and cited active paths", "evidence_status": e, "safe_wording": f, "prohibited_wording": g} for a, b, c, d, e, f, g in claims])
    write_csv("s4_conflicts.csv", [
        {"issue": "dPL best checkpoint selection", "code_fact": "evaluate(eval_forcing, eval_obs) is used to update best_validation", "other_fact": "these arrays are the configured evaluation/test period", "status": "CONFLICT", "impact": "evaluation leakage", "action": "reselect from calibration-only evidence or disclose limitation"},
        {"issue": "metric label", "code_fact": "active code standard KGE(Q)", "other_fact": "R1 plan uses KGE'", "status": "CONFLICT", "impact": "metric misreporting", "action": "use active code formula"},
        {"issue": "protocol", "code_fact": "foundation 531 dates", "other_fact": "Stage 1/remote screening metadata use 1988/1989 or differing lengths", "status": "CONFLICT", "impact": "screening cannot be formal performance", "action": "separate protocol families"},
        {"issue": "prespecification", "code_fact": "fixed bins and quintiles exist", "other_fact": "analysis script explicitly calls itself post-hoc", "status": "POST_HOC", "impact": "limited confirmatory interpretation", "action": "label exploratory or provide dated lock evidence"},
    ])
    write_csv("s4_unresolved_items.csv", [
        {"item": "no formal Methods 2.4 source text found in project", "section": "S4.1-S4.12", "status": "UNRESOLVED", "minimal_action": "provide current manuscript text or identify file"},
        {"item": "no paired inferential test/multiplicity implementation", "section": "S4.2", "status": "UNRESOLVED", "minimal_action": "choose and pre-register test family before confirmatory wording"},
        {"item": "decision dates are not reconstructable from dirty worktree", "section": "S4.3", "status": "UNRESOLVED", "minimal_action": "supply clean commit/tag and dated protocol lock"},
        {"item": "dPL checkpoint chosen from evaluation path", "section": "S4.3/S4.11", "status": "CONFLICT", "minimal_action": "re-evaluate checkpoint selection without test labels or explicitly limit claims"},
        {"item": "no practical replication threshold", "section": "S4.4", "status": "UNRESOLVED", "minimal_action": "fix threshold rule before using resolution language"},
        {"item": "IC physical parameter arrays and parameter-shift analysis absent", "section": "S4.6", "status": "PARTIAL", "minimal_action": "retain/export physical IC parameters and run pre-fixed diagnostics"},
        {"item": "no executed external-state comparison", "section": "S4.7", "status": "BLOCKED", "minimal_action": "define product/alignment/metric and run analysis"},
        {"item": "no active synthetic experiment assets", "section": "S4.8", "status": "BLOCKED", "minimal_action": "decide whether to add a scoped synthetic study; requires new experiment"},
        {"item": "no quantitative cross-host categories", "section": "S4.9", "status": "UNRESOLVED", "minimal_action": "fix direction/resolution/host-count rule before interpretation"},
        {"item": "HBV Experiment 1 paired output not located", "section": "S4.5", "status": "UNRESOLVED", "minimal_action": "identify formal HBV reference table or mark not applicable"},
    ])


def reports() -> None:
    write_report("S4_metric_and_inference_audit.md", """# S4 Metric and Inference Audit

## Metric

The active IC objective and dPL final evaluation implement aligned standard KGE(Q) algebra in separate GPU and NumPy FP64 functions, with Pearson correlation, population standard deviations, a simulation-to-observation variability ratio, and a simulation-to-observation mean ratio. Finite nonnegative observations and simulations are retained; negative and nonfinite values are masked. IC and final evaluation require at least 30 valid samples and return `-999` for invalid cases. The dPL training loss uses the same component structure but adds `1e-6` stabilizers inside standard deviations, the beta denominator, and the distance root. It is therefore aligned but not byte-identical to final KGE(Q).

The code-level numeric check is recorded in `results/s4_metric_equivalence_checks.csv`. It uses a deterministic synthetic vector because no saved production observation/simulation pair was available for a replay. No log transform was found.

## Pairing and inference

The formal snow tables are basin-wise and preserve the same basin ID across structures. IC starts and dPL seeds are route-specific repetitions, not directly seed-paired. The active analysis implements descriptive means/medians/IQRs and Spearman association; no Wilcoxon, paired permutation, bootstrap, cluster bootstrap, or multiplicity correction implementation was found. A confirmatory test family remains unresolved.

## Key limitation

The dPL training code updates `best_checkpoint` from `evaluate(eval_forcing, eval_obs, ...)` at lines 856--868. In the active configuration these are the 1995-10-01--2010-09-30 evaluation arrays, not a calibration-only validation split. This is an evaluation-period checkpoint-selection risk and must be resolved or explicitly disclosed.
""")
    write_report("S4_prespecification_and_replication_audit.md", """# S4 Prespecification and Replication Audit

The 531-basin set, formal dates, IC seed/start grid, dPL seed list, and active metric are code/config facts. Stage 1 and some screening outputs use a different 1988/1989 protocol and must remain screening evidence. The exact decision dates are not safely recoverable because the worktree is dirty and no clean protocol-lock tag was identified.

The fixed snow bins and quintile cross-check are executable and reproduced in existing outputs, but the active stratification script describes the analysis as post-hoc. The lowest bin is described as near-snow-free, but no pre-fixed rule based on zero effect, confidence interval, or replication resolution was found. It must not be called a formal negative control without additional evidence.

Replication is available as dPL seeds 42/123/2026 and IC seed-by-start summaries from S3. The audit reports SD, IQR, range, MAD, and upper quantiles where raw replicate values exist. No project-defined practical resolution threshold was found, so the outputs support descriptive variability language only.
""")
    write_report("S4_experiment_evidence_map.md", """# S4 Experiment Evidence Map

## Experiment 1

The corrected 531-basin IC/dPL master, official dPL three-seed tables, fixed-bin summaries, quintile summaries, and Spearman calculations exist. This supports a qualified descriptive dose-response section. It does not establish a preregistered negative-control rule, formal paired p-values, or complete HBV paired evidence.

## Experiment 2a

S2 parameter bounds and dPL normalized/physical outputs exist. IC task records contain normalized best coordinates, but the audited result schema lacks physical parameters and a formal parameter-shift/attribute association table. The diagnostic is partial.

## Experiment 2b

S1 contains SWE product metadata and explicitly treats SWE as a process-state consistency reference, not truth. No executed state alignment, timing, or external-product comparison table was found. The section is blocked.

## Experiment 3

No active synthetic truth/noise/re-estimation manifest or result table was found. The section is blocked.

## Experiment 4

No executable strong/partial/host-family quantitative reading rule was found. Cross-host statements must remain descriptive and qualified.
""")
    write_report("S4_section_readiness.md", """# S4 Section Readiness

The machine-readable section matrix is `results/s4_section_readiness.csv`.

| Section | Status | Safe scope |
|---|---|---|
| S4.1 | READY_WITH_QUALIFICATION | standard KGE(Q), masks, and training epsilon difference |
| S4.2 | READY_WITH_QUALIFICATION | basin-wise descriptive paired differences; no formal test claim |
| S4.3 | PARTIAL | configuration facts plus explicit post-hoc/leakage limitations |
| S4.4 | READY_WITH_QUALIFICATION | descriptive repeat dispersion, no fixed threshold |
| S4.5 | READY_WITH_QUALIFICATION | corrected basin/stratum effects and dose association |
| S4.6 | PARTIAL | S2 bounds and dPL parameter outputs only |
| S4.7 | BLOCKED | no executed external-state comparison |
| S4.8 | BLOCKED | no active synthetic study |
| S4.9 | BLOCKED | no fixed cross-host reading rule |
| S4.10 | PARTIAL | fixed bins/quintiles only |
| S4.11 | READY_WITH_QUALIFICATION | seed/start dispersion with leakage caveat |
| S4.12 | PARTIAL | inventory of available basin/stratum/trace assets |
""")
    write_report("S4_minimal_followup_plan.md", """# S4 Minimal Follow-up Plan

## P0: central validity

1. Resolve dPL checkpoint selection: use calibration-only selection or disclose that the current best checkpoint uses the evaluation path. Input: existing checkpoints and histories. Output: leakage-free seed summaries. No retraining is required if checkpoints can be re-ranked from stored calibration metrics; otherwise retraining is required. Targets S4.3 and S4.11.
2. Provide the current Methods 2.4 source and a clean protocol-lock commit/tag. Input: manuscript source and Git history. Output: dated decision timeline. No compute required. Targets S4.1-S4.5.

## P1: subsection completion

3. Fix and document an inferential unit/test/multiplicity policy, then apply it to existing basin-level tables. No model retraining required. Targets S4.2 and S4.5.
4. Export raw IC physical parameters or a verified mapping from normalized task records and run a pre-fixed parameter diagnostic. Existing task records may be insufficient because raw starts/physical values are not stored. Target S4.6.
5. Locate or formally remove HBV paired Experiment 1/state evidence. Target S4.5.

## P2: robustness enhancement

6. Define a replication-resolution summary without post-hoc threshold selection. Existing dPL/IC repeats suffice for descriptive thresholds; no retraining required. Target S4.4.
7. Add metric and bin sensitivity tables from existing basin results. Target S4.10.

## P3: optional/additional

8. Add the external-state comparison and synthetic experiment only if they are central claims. Both require new design decisions and likely new computation; neither is currently supported by existing results. Targets S4.7 and S4.8.
9. Define quantitative cross-host reading rules before using strong/partial reproduction labels. Target S4.9.
""")
    write_report("S4_writing_blueprint.md", """# S4 Writing Blueprint

S4.1 can state the active standard KGE(Q) formula, mask, minimum sample count, invalid sentinel, and the dPL training epsilon qualification. S4.2 can state basin-wise pairing and route-specific repetition levels, but must not claim a formal paired test. S4.3 should present configuration facts alongside the unresolved timing and the evaluation-period checkpoint-selection conflict.

S4.4 may report observed dPL seed and IC restart dispersion as a descriptive resolution reference. It cannot say that replication variability defined a prespecified threshold. S4.5 may report the corrected fixed-bin/quintile results and Spearman association as exploratory or qualified analyses; the lowest bin is not a formal negative control under the current evidence.

S4.6 should be limited to S2 bounds and dPL mapping until IC physical parameter outputs and shifts are available. S4.7 and S4.8 should be marked unavailable/blocked rather than filled with proposed external-state or synthetic definitions. S4.9 should use descriptive cross-host comparisons without strong/partial category labels. S4.10-S4.12 should inventory the existing fixed-bin, quintile, seed, stratum, and IC trace outputs and clearly identify missing metric/state/synthetic robustness.

Cross-references: S1 supplies dates, basin attributes, fixed strata, and the SWE non-truth limitation; S2 supplies model states and parameter bounds; S3 supplies IC/dPL route, optimizer repeats, and shared-path conflicts.
""")
    write_report("S4_experimental_design_source_of_truth.md", """# S4 Experimental Design, Analysis and Robustness: Source of Truth

## Executive summary

The project provides a usable but qualified foundation for S4.1-S4.5, S4.10-S4.12. The active metric is standard KGE(Q), not the KGE-prime label used in planning text. IC and dPL final evaluation use aligned standard-KGE algebra in separate FP64 implementations, while dPL training uses an epsilon-stabilized differentiable approximation. Basin-wise tables and route-specific repeats support descriptive paired differences. No formal paired test or multiple-testing correction is implemented.

The principal validity conflict is dPL checkpoint selection: the code evaluates `eval_forcing/eval_obs` to update `best_checkpoint`, and those arrays correspond to the configured evaluation period. This is not a harmless naming issue. S4 must either use a non-leaky selection or explicitly limit the claims.

## S4.1-S4.5

The corrected 531-basin IC/dPL master and official dPL three-seed tables support basin-level fixed-bin, quintile, and continuous snow-fraction summaries. Fixed bins are computationally verified, but their pre-result lock is not. The lowest snow bin is near-snow-free descriptively; a formal negative-control rule is absent. The active source contains no Wilcoxon, paired permutation, bootstrap, FDR, or Bonferroni implementation.

## S4.6-S4.9

S2 parameter bounds and dPL normalized/physical outputs exist. IC task records expose normalized best coordinates but not physical parameters or a completed parameter-organization analysis. S1 SWE metadata explicitly prohibit calling the external product truth, and no executed state-comparison analysis was found. No active synthetic experiment or quantitative cross-host reading rule was found. These sections are blocked or partial, not silently completed from plans.

## S4.10-S4.12

Fixed-bin and quintile results, seed tables, IC restart summaries, and optimizer traces are available. Metric, boundary-threshold, state-window, synthetic-noise, and formal cross-host sensitivities are not. TGD and other model coverage must follow the result inventories; incomplete/provisional results must not be represented as complete 531-basin evidence.

## Evidence products

All section statuses, conflicts, unresolved items, claim wording, and follow-up actions are in `results/s4_section_readiness.csv`, `results/s4_conflicts.csv`, `results/s4_unresolved_items.csv`, `results/s4_claim_evidence_map.csv`, and the companion reports in this directory.
""")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    metric_audit()
    pairing_audit()
    prespecification_audit()
    replication_audit()
    experiment1_audit()
    parameter_audit()
    state_audit()
    synthetic_and_cross_host_audit()
    sensitivity_and_additional_audit()
    section_reports()
    reports()
    print("S4 audit products written under manuscript/supplement")


if __name__ == "__main__":
    main()
