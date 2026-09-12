#!/usr/bin/env python3
"""Read-only foundation-531 audit for Supplement Text S3.

The audit deliberately uses only the standard library so it can run even when
the project virtual environment is incomplete.  It never imports production
models and never writes outside manuscript/supplement.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / "manuscript" / "supplement"
OUT = SUPP / "results"
FIG = SUPP / "figures"
REPORT = SUPP / "reports"
for p in (OUT, FIG, REPORT):
    p.mkdir(parents=True, exist_ok=True)

MODELS = [
    "XAJ", "XAJ_TGD", "XAJ_CN", "GR4J", "GR4J_TGD", "GR4J_CN",
    "SIMHYD", "SIMHYD_TGD", "SIMHYD_CN", "HBV", "XAJ_PD", "GR4J_PD", "SIMHYD_PD",
]
DIMS = {"XAJ": 15, "XAJ_TGD": 18, "XAJ_CN": 17, "GR4J": 4,
        "GR4J_TGD": 7, "GR4J_CN": 6, "SIMHYD": 10, "SIMHYD_TGD": 13,
        "SIMHYD_CN": 12, "HBV": 13, "XAJ_PD": 17, "GR4J_PD": 6,
        "SIMHYD_PD": 12}
STATUS = {"code": "VERIFIED_CODE", "config": "VERIFIED_CONFIG", "runtime": "VERIFIED_RUNTIME",
          "result": "VERIFIED_RESULT", "inferred": "INFERRED", "conflict": "CONFLICT",
          "unresolved": "UNRESOLVED", "legacy": "LEGACY_NOT_ACTIVE"}


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def write_csv(name, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else ["status", "detail"]
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def line(path, needle):
    try:
        for n, text in enumerate(Path(path).read_text(errors="replace").splitlines(), 1):
            if needle in text:
                return f"{rel(path)}:{n}"
    except Exception:
        pass
    return rel(path)


def sha(path):
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def run(cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return ""


def scan_files():
    roots = [ROOT / "ablation", ROOT / "experiments", ROOT / "training", ROOT / "models", ROOT / "outputs", ROOT / "results"]
    files = []
    for root in roots:
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file())
    return files


def active_config():
    return read_json(ROOT / "ablation/configs/ic_foundation_531_v1.json") or {}


def inventory():
    rows = []
    candidates = [
        ("foundation_531_config", ROOT / "ablation/configs/ic_foundation_531_v1.json", "active", "VERIFIED_CONFIG"),
        ("foundation_531_manifest", ROOT / "ablation/manifests/ic_531_dataset_manifest_v1.json", "active", "VERIFIED_RESULT"),
        ("foundation_531_resolved", ROOT / "outputs/ic_ablation/foundation_v1/resolved_config.json", "active_validation", "VERIFIED_RUNTIME"),
        ("foundation_531_output", ROOT / "outputs/ic_ablation/foundation_v1", "active_validation", "VERIFIED_RUNTIME"),
        ("xnes_stage1_screening", ROOT / "outputs/ic_ablation/stage1_screening/v1/xnes", "screening_only", "VERIFIED_RESULT"),
        ("xnes_large_scale_screening", ROOT / "outputs/ic_ablation/large_scale_screening/v1", "screening_only", "VERIFIED_RESULT"),
        ("dpl_531_smoke", ROOT / "outputs/dpl_camels_531_smoke_fullperiod", "active_validation", "VERIFIED_RESULT"),
        ("dpl_unified_559", ROOT / "outputs/dpl_unified_365d_v1", "legacy_559", "LEGACY_NOT_ACTIVE"),
        ("ic_full_559", ROOT / "results/ic_xnes_full", "legacy_559", "LEGACY_NOT_ACTIVE"),
        ("ic_archive", ROOT / "results/archive", "archive", "LEGACY_NOT_ACTIVE"),
    ]
    for name, path, role, status in candidates:
        exists = path.exists()
        count = sum(1 for p in path.rglob("*") if p.is_file()) if path.is_dir() else (1 if exists else 0)
        rows.append({"asset": name, "path": rel(path), "exists": str(exists), "file_count": count,
                     "role": role, "evidence_status": status,
                     "completion_marker": str((path / "COMPLETE").exists() if path.is_dir() else False)})
    write_csv("s3_run_inventory.csv", rows)


def call_graph():
    nodes = [
        {"id": "foundation_config", "path": "ablation/configs/ic_foundation_531_v1.json", "status": STATUS["config"]},
        {"id": "ic_runner", "path": "ablation/runners/run_xnes_ablation.py", "status": STATUS["code"], "role": "XNES ablation runner"},
        {"id": "ic_optimizer_adapter", "path": "ablation/optimizers/xnes.py", "status": STATUS["code"], "role": "EvoTorch XNES adapter"},
        {"id": "ic_runtime", "path": "ablation/ic_core/runtime.py", "status": STATUS["code"], "role": "shared candidate evaluation runtime"},
        {"id": "ic_objective", "path": "ablation/ic_core/objective_adapter.py", "status": STATUS["code"], "role": "KGE objective"},
        {"id": "ic_model_adapter", "path": "ablation/ic_core/model_adapter.py", "status": STATUS["code"], "role": "model dispatch"},
        {"id": "dpl_runner", "path": "training/dpl/run_dpl_model.py", "status": STATUS["code"], "role": "dPL train/evaluate"},
        {"id": "models", "path": "models/", "status": STATUS["code"], "role": "hydrological model classes"},
        {"id": "foundation_results", "path": "outputs/ic_ablation/foundation_v1/", "status": STATUS["runtime"], "role": "531 validation assets only"},
    ]
    edges = [
        ["foundation_config", "ic_runner", "dataset_manifest reference / runner configuration"],
        ["ic_runner", "ic_optimizer_adapter", "optimizer registry"],
        ["ic_runner", "ic_runtime", "candidate fitness call"],
        ["ic_runtime", "ic_objective", "KGEObjective"],
        ["ic_runtime", "ic_model_adapter", "run_model"],
        ["ic_model_adapter", "models", "model class dispatch"],
        ["dpl_runner", "models", "same model registry classes"],
        ["foundation_config", "foundation_results", "resolved config / validation output"],
    ]
    payload = {"active_scope": "foundation_531", "nodes": nodes,
               "edges": [{"from": a, "to": b, "evidence": c} for a, b, c in edges],
               "conclusion": "The 531 foundation runtime and validation path are present; no complete 531 IC-XNES production result directory was found.",
               "evidence_status": "CONFLICT"}
    write_json("s3_active_call_graph.json", payload)


def optimizer_outputs():
    cfg = read_json(ROOT / "ablation/configs/ic_xnes_stage1_screening_v1.json") or {}
    opt = cfg.get("optimizer", {})
    rows = []
    for model in MODELS:
        active_model = model == cfg.get("model_key")
        rows.append({"Host": model.split("_")[0], "Structure": model,
                     "Dimension": DIMS[model], "Population": opt.get("population") if active_model else "UNRESOLVED",
                     "Initial width": opt.get("stdev_init") if active_model else "UNRESOLVED",
                     "Learning rate": "center=1.0; covariance=0.6*(3+log(D))/(D*sqrt(D))" if active_model else "UNRESOLVED",
                     "Generations": opt.get("generations") if active_model else "UNRESOLVED",
                     "Evaluations": (opt.get("population", "") * opt.get("generations", "")) if active_model else "UNRESOLVED",
                     "Restarts": opt.get("starts") if active_model else "UNRESOLVED",
                     "Initial center": "deterministic LHS per basin/start" if active_model else "UNRESOLVED",
                     "Stopping rule": "fixed generations; no early stop" if active_model else "UNRESOLVED",
                     "Seed protocol": ",".join(map(str, opt.get("optimizer_seeds", []))) if active_model else "UNRESOLVED",
                     "evidence_status": STATUS["config"] if active_model else STATUS["unresolved"],
                     "evidence": "ablation/configs/ic_xnes_stage1_screening_v1.json; ablation/runners/run_xnes_ablation.py"})
    write_csv("s3_ic_optimizer_settings.csv", rows)
    budget = []
    for model in MODELS:
        p = opt.get("population") if model == "XAJ" else None
        budget.append({"model": model, "dimension": DIMS[model], "population": p or "UNRESOLVED",
                       "generations": opt.get("generations") if p else "UNRESOLVED",
                       "evaluations_per_start": p * opt.get("generations", 0) if p else "UNRESOLVED",
                       "starts": opt.get("starts") if p else "UNRESOLVED",
                       "531_production_result": "absent", "evidence_status": STATUS["unresolved"] if not p else STATUS["conflict"]})
    write_csv("s3_ic_budget_by_model.csv", budget)
    write_json("s3_ic_optimizer_details.json", {
        "algorithm": "Exponential Natural Evolution Strategies (XNES)",
        "implementation": "local evotorch.algorithms.distributed.gaussian.XNES, wrapped by ablation/optimizers/xnes.py",
        "library_version": "evotorch 0.6.1 in screening freeze manifest",
        "space": "normalized [0,1]^D candidates; normalized_to_physical clips and maps to model bounds",
        "initial_center": "deterministic LHS center per basin/model/start in the stage-1 runner; not 0.5",
        "initial_width": "0.25 in active stage-1 config",
        "population": "48 in active XAJ stage-1 config; final 531 model-specific populations unresolved",
        "learning_rates": {"center": 1.0, "covariance": "0.6*(3+log(D))/(D*sqrt(D))"},
        "fitness_shaping": "ranking_method='nes'",
        "sampling": {"antithetic": False, "mirrored": False, "evidence": "ordinary ExpGaussian sample call; no symmetric flag passed"},
        "covariance": "full exponential Gaussian covariance update (ExpGaussian)",
        "numerical_stabilization": "no explicit active XNES sigma threshold; candidate nonfinite handling/failure logging in runner",
        "direction": "Problem('max') and maximize-oriented adapter",
        "termination": "fixed generations in stage-1 config; no enabled early-stop criterion",
        "evidence_status": STATUS["code"]
    })


def parse_result(path):
    d = read_json(path)
    if not isinstance(d, dict):
        return None
    return d


def result_rows(root):
    for p in root.rglob("result.json") if root.exists() else []:
        d = parse_result(p)
        if d and d.get("status", "completed").lower() in ("completed", "complete"):
            d["_path"] = rel(p)
            yield d


def calibration_and_adequacy():
    screen = ROOT / "outputs/ic_ablation/stage1_screening/v1/xnes"
    large = ROOT / "outputs/ic_ablation/large_scale_screening/v1"
    cal_rows = []
    for d in result_rows(large):
        cal_rows.append({"basin": d.get("basin_id"), "host": d.get("model_key"), "structure": d.get("model_key"),
                         "optimizer": d.get("optimizer_name"), "population": d.get("population"),
                         "dimension": DIMS.get(d.get("model_key"), "UNRESOLVED"), "generations": d.get("total_generations"),
                         "evaluations": (d.get("population", 0) * d.get("total_generations", 0)), "seed": d.get("optimizer_seed"),
                         "start": d.get("start_idx"), "train_objective": d.get("best_train_kge"),
                         "evaluation_objective": "not computed", "success": "true", "runtime_seconds": d.get("runtime_seconds"),
                         "selection_status": "LEGACY_NOT_ACTIVE: 1988-1998/366-day screening, not foundation 531"})
    if not cal_rows:
        cal_rows = [{"basin": "", "host": "", "structure": "", "optimizer": "", "population": "", "dimension": "",
                     "generations": "", "evaluations": "", "seed": "", "start": "", "train_objective": "",
                     "evaluation_objective": "", "success": "", "runtime_seconds": "",
                     "selection_status": "UNRESOLVED: no calibration result files"}]
    write_csv("s3_optimizer_calibration_runs.csv", cal_rows)
    groups = defaultdict(list)
    for r in cal_rows:
        if r.get("train_objective") not in (None, ""):
            try:
                value = float(r["train_objective"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                groups[(r["optimizer"] or "UNKNOWN", r["host"] or "UNKNOWN", r["population"])].append(value)
    summary = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple("" if v is None else str(v) for v in item[0])):
        summary.append({"optimizer": key[0], "model": key[1], "population": key[2], "n_runs": len(vals),
                        "mean_train_objective": statistics.mean(vals), "median_train_objective": statistics.median(vals),
                        "min_train_objective": min(vals), "max_train_objective": max(vals),
                        "calibration_set": "32 basins, Split A", "active_531": "no", "evidence_status": STATUS["legacy"]})
    if not summary:
        summary = [{"optimizer": "", "model": "", "population": "", "n_runs": 0, "mean_train_objective": "",
                    "median_train_objective": "", "min_train_objective": "", "max_train_objective": "",
                    "calibration_set": "not found", "active_531": "unresolved", "evidence_status": STATUS["unresolved"]}]
    write_csv("s3_optimizer_calibration_summary.csv", summary)

    # Valid multi-start evidence exists for XAJ screening, but not for 531 production.
    vals = defaultdict(list)
    for d in result_rows(screen):
        vals[(d.get("basin_id"), d.get("model_key"), "screening")].append(float(d.get("best_train_kge", float("nan"))))
    disp = []
    for (basin, model, scope), xs in sorted(vals.items()):
        xs = [x for x in xs if math.isfinite(x)]
        if not xs:
            continue
        xs2 = sorted(xs, reverse=True)
        mean = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        disp.append({"basin_id": basin, "host": model, "structure": model, "n_restarts": len(xs),
                     "best": max(xs), "median": statistics.median(xs), "worst": min(xs),
                     "range": max(xs) - min(xs), "iqr": (statistics.quantiles(xs, n=4)[2] - statistics.quantiles(xs, n=4)[0]) if len(xs) >= 4 else "UNRESOLVED",
                     "std": sd, "cv": sd / abs(mean) if mean else "UNRESOLVED",
                     "best_second_gap": xs2[0] - xs2[1] if len(xs2) > 1 else "UNRESOLVED",
                     "normalized_dispersion": (max(xs) - min(xs)) / max(abs(mean), 1e-12),
                     "scope": scope, "evidence_status": STATUS["result"]})
    write_csv("s3_restart_dispersion_by_basin.csv", disp or [{"basin_id": "", "host": "", "structure": "", "n_restarts": 0,
        "best": "", "median": "", "worst": "", "range": "", "iqr": "", "std": "", "cv": "",
        "best_second_gap": "", "normalized_dispersion": "", "scope": "no foundation 531 restart results",
        "evidence_status": STATUS["unresolved"]}])

    sat = []
    traces = list((large).rglob("trace.json")) if large.exists() else []
    for p in traces:
        d = read_json(p)
        if not isinstance(d, list) or not d:
            continue
        parent = p.parts
        basin = next((x for x in parent if re.fullmatch(r"\d{8}", x)), "")
        model = next((x for x in ("GR4J", "SIMHYD", "XAJ") if x in parent), "")
        pop = next((int(x[4:]) for x in parent if x.startswith("pop_") and x[4:].isdigit()), "")
        final = next((x for x in reversed(d) if isinstance(x, dict) and isinstance(x.get("best_fitness"), (int, float))), None)
        for frac in (0.25, 0.5, 0.75, 1.0):
            target = max(1, round(len(d) * frac))
            row = d[min(target, len(d)) - 1]
            sat.append({"basin_id": basin, "host": model, "structure": model, "population": pop,
                        "generation": row.get("generation"), "normalized_budget": frac,
                        "incumbent_best_objective": row.get("best_fitness"),
                        "source": rel(p), "scope": "screening_only, legacy 1988-1998 protocol",
                        "evidence_status": STATUS["legacy"]})
    write_csv("s3_budget_saturation_by_basin.csv", sat or [{"basin_id": "", "host": "", "structure": "", "population": "",
        "generation": "", "normalized_budget": "", "incumbent_best_objective": "", "source": "",
        "scope": "no real intermediate foundation 531 results", "evidence_status": STATUS["unresolved"]}])
    write_csv("s3_convergence_summary.csv", [{"scope": "foundation_531", "explicit_production_stopping_rule": "fixed generations only in active XNES design; full run absent",
        "diagnostic_saturation_rule": "not applied to foundation 531", "completed_basins": 0, "nonconverged_basins": "UNRESOLVED",
        "evidence_status": STATUS["unresolved"]}])
    write_csv("s3_nonconverged_basins.csv", [{"basin_id": "ALL 531", "host": "all active models", "structure": "all", "failure_type": "missing formal IC result",
        "finite_objective": "UNRESOLVED", "restart_count": "UNRESOLVED", "final_budget": "UNRESOLVED",
        "in_main_results": "no formal IC main results found", "handling": "not counted as completed; requires run/result manifest",
        "best_available": "no", "excluded": "not applicable", "rerun": "required", "evidence_status": STATUS["unresolved"]}])


def dpl_outputs():
    cfg = read_json(ROOT / "training/dpl/base_config_camels_531.json") or {}
    network = cfg.get("network", {})
    rows = []
    for model in MODELS:
        rows.append({"model": model, "input_attributes": 35, "input_shape": "[basin,35]",
                     "hidden_sizes": " -> ".join(map(str, network.get("hidden_sizes", [256, 256, 256]))),
                     "depth": network.get("depth"), "activation": network.get("activation"), "dropout": network.get("dropout"),
                     "normalization": network.get("normalization"), "output_dim": DIMS[model], "shared_trunk": "one network per model run",
                     "output_head": "Linear(last_hidden, n_parameters) + sigmoid + clamp",
                     "parameter_sets_per_basin_per_forward": 1, "multi_sample": "not present in active runner",
                     "status": STATUS["code"], "evidence": "training/dpl/run_dpl_model.py:125-168"})
    write_csv("s3_dpl_architecture.csv", rows)
    maps = []
    for model in MODELS:
        maps.append({"model": model, "parameter_count": DIMS[model], "normalized_output": "sigmoid(logits), clamp(epsilon,1-epsilon)",
                     "linear_mapping": "lower + theta * (upper-lower)", "log_mapping": "tgd_tau only: exp(log(lower)+theta*(log(upper)-log(lower)))" if "TGD" in model else "none",
                     "runtime_constraints": "model-specific clamps/derived state constraints remain in model class",
                     "defaults": "head bias initialized to normalized physical defaults", "parameter_order": "list(specs) in models/parameter_specs.py",
                     "status": STATUS["code"]})
    write_csv("s3_dpl_parameter_mapping.csv", maps)
    train = [{"field": "optimizer", "value": cfg.get("training", {}).get("optimizer"), "evidence_status": STATUS["config"], "evidence": "training/dpl/base_config_camels_531.json"},
             {"field": "learning_rate", "value": cfg.get("training", {}).get("lr"), "evidence_status": STATUS["config"], "evidence": "training/dpl/base_config_camels_531.json"},
             {"field": "scheduler", "value": "CosineAnnealingLR, eta_min=min_lr", "evidence_status": STATUS["code"], "evidence": "training/dpl/run_dpl_model.py:709-712"},
             {"field": "warmup_scheduler", "value": "none found", "evidence_status": STATUS["code"], "evidence": "training/dpl/run_dpl_model.py:709-712"},
             {"field": "weight_decay", "value": cfg.get("training", {}).get("weight_decay"), "evidence_status": STATUS["config"], "evidence": "training/dpl/base_config_camels_531.json"},
             {"field": "gradient_clipping", "value": cfg.get("training", {}).get("grad_clip_norm"), "evidence_status": STATUS["code"], "evidence": "training/dpl/run_dpl_model.py:848"},
             {"field": "mixed_precision", "value": "not found; forward float32, metric float64", "evidence_status": STATUS["config"], "evidence": "training/dpl/base_config_camels_531.json"},
             {"field": "batch", "value": "uniform basin sampling; valid-window catalogue optional; active 531 config balanced_valid_kge_windows", "evidence_status": STATUS["code"], "evidence": "training/dpl/run_dpl_model.py:364-477"},
             {"field": "batch_size", "value": cfg.get("training", {}).get("batch_size"), "evidence_status": STATUS["config"], "evidence": "training/dpl/base_config_camels_531.json"},
             {"field": "window", "value": "365 warmup + 365 prediction, stride 365", "evidence_status": STATUS["config"], "evidence": "training/dpl/base_config_camels_531.json"},
             {"field": "epochs", "value": cfg.get("training", {}).get("epochs"), "evidence_status": STATUS["config"], "evidence": "training/dpl/base_config_camels_531.json"},
             {"field": "validation_interval", "value": cfg.get("training", {}).get("validation_interval"), "evidence_status": STATUS["config"], "evidence": "training/dpl/base_config_camels_531.json"},
             {"field": "early_stopping", "value": "none found; best checkpoint selected by val_kge_median", "evidence_status": STATUS["code"], "evidence": "training/dpl/run_dpl_model.py:856-885"},
             {"field": "checkpoint_selection", "value": "best validation median KGE", "evidence_status": STATUS["code"], "evidence": "training/dpl/run_dpl_model.py:860-868"},
             {"field": "seed", "value": cfg.get("training", {}).get("seed"), "evidence_status": STATUS["config"], "evidence": "training/dpl/run_dpl_model.py:171-175"},
             {"field": "three_repeats", "value": "no active 531 three-seed result inventory found", "evidence_status": STATUS["unresolved"], "evidence": "outputs/ and training/dpl/ scan"}]
    write_csv("s3_dpl_training_protocol.csv", train)
    write_csv("s3_seed_repeat_summary.csv", [{"model": "active 531 dPL launcher", "seeds_found": "42,123,2026", "repeat_count": 3,
        "same_data_and_hyperparameters": "launcher protocol says yes; no local production outputs", "summary_statistic": "not available",
        "evidence_status": STATUS["unresolved"], "note": "code-level three-seed protocol verified; result equivalence unresolved"}])


def shared_protocol():
    cfg = active_config()
    objective = {"name": "KGE(Q)", "direction": "maximize", "invalid_fitness": -999.0, "min_samples": 30,
                 "formula_code": "1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)",
                 "dpl_training_formula": "loss = mean(1 - KGE) over finite valid basin samples",
                 "ic_definition": "KGEObjective.evaluate in ablation/ic_core/objective_adapter.py",
                 "dpl_definition": "kge_per_basin in training/dpl/run_dpl_model.py",
                 "equivalence": "not exact: IC uses a separate objective adapter and dPL uses differentiable epsilon-stabilized KGE; formulas are related but implementation details differ",
                 "evidence_status": STATUS["conflict"]}
    write_json("s3_objective_definition.json", objective)
    write_json("s3_warmup_evaluation_protocol.json", {"foundation_531": cfg.get("periods"), "ic": {"train_warmup_days": 365, "test_warmup_days": "not active production evidence", "source": "ablation/ic_core/runtime.py:48-55"},
        "dpl": {"warmup_days": 365, "calibration": "1981-10-01 to 1995-09-30", "evaluation": "1995-10-01 to 2010-09-30", "source": "training/dpl/base_config_camels_531.json"},
        "state_policy": "model forward starts from model defaults for each supplied window; no verified cross-window state carry in shared 531 production result",
        "evidence_status": STATUS["config"]})
    write_json("s3_missing_data_protocol.json", {"raw_negative_discharge": "invalid/masked", "NaN": "masked", "infinite": "masked", "forcing_missing": "not separately documented in active manifest",
        "loss_mask": "finite qsim/qobs and qobs>=0 and qsim>=0", "minimum_valid_days": 30, "zero_discharge": "valid observation zero retained; constant/low-variance behavior protected by epsilon/fallback rules",
        "basin_exclusion": "not by missing target in 531 adapter; sampling fallback retains basin", "metric_denominator": "valid mask count", "all_missing": "invalid objective/floor depending route",
        "ic_dpl_same": "not exact; separate implementations", "evidence_status": STATUS["code"]})
    rows = [
        {"comparison": "model class", "ic": "ablation.ic_core.ModelAdapter", "dpl": "training.dpl MODEL_REGISTRY", "result": "same model modules intended; no full 531 equivalence run", "status": STATUS["inferred"]},
        {"comparison": "objective", "ic": "KGEObjective", "dpl": "kge_per_basin + loss=1-KGE", "result": "related, not byte/math exact due epsilon and implementation", "status": STATUS["conflict"]},
        {"comparison": "warmup dates", "ic": "foundation config warmup 1980-10-01..1981-09-30", "dpl": "same dates", "result": "code/config aligned", "status": STATUS["config"]},
        {"comparison": "missing mask", "ic": "objective adapter mask", "dpl": "kge_per_basin mask", "result": "same broad validity rule; exact aggregation requires equivalence test", "status": STATUS["unresolved"]},
        {"comparison": "unified evaluation function", "ic": "no single common evaluate() call", "dpl": "training.dpl.evaluate", "result": "not verified", "status": STATUS["unresolved"]},
    ]
    write_csv("s3_shared_path_equivalence.csv", rows)


def environment():
    manifest = read_json(ROOT / "outputs/ic_ablation/stage1_screening/v1/xnes/input_freeze_manifest.json") or {}
    env = read_json(ROOT / "outputs/ic_ablation/stage1_screening/v1/xnes/environment.json") or {}
    payload = {"audit_machine": {"python": sys.version, "platform": platform.platform(), "cwd": str(ROOT), "numpy_torch_importable": False,
                                  "note": "audit used stdlib because venv lacks numpy/torch; do not substitute audit machine for production machine"},
               "production_manifest": env, "freeze_manifest": {k: v for k, v in manifest.items() if k != "git_dirty"},
               "git_commit": run(["git", "rev-parse", "HEAD"]), "git_dirty_at_audit": bool(run(["git", "status", "--porcelain"])),
               "project_dependency_file": "pyproject.toml", "optimizer_library": "local evotorch package; production freeze manifest says evotorch 0.6.1",
               "hardware": {"formal_531_ic": "no production manifest found", "screening_gpu": env.get("gpu_model"), "cuda": env.get("cuda_version")},
               "reproducibility": {"foundation_seed": active_config().get("seed"), "screening_seeds": [0], "dpl_launcher_seeds": [42, 123, 2026],
                                    "deterministic_flags": "not found; torch.manual_seed and cuda.manual_seed_all used in dPL", "workers": "not found"},
               "evidence_status": STATUS["conflict"]}
    write_json("s3_environment_manifest.json", payload)
    runtime = []
    for d in result_rows(ROOT / "outputs/ic_ablation/large_scale_screening/v1"):
        runtime.append({"route": "IC", "scope": "legacy screening", "model": d.get("model_key"), "basin": d.get("basin_id"), "structure": d.get("model_key"),
                        "restart": d.get("start_idx"), "runtime_seconds": d.get("runtime_seconds"), "success": "true", "source": d.get("_path"), "status": STATUS["legacy"]})
    for p in (ROOT / "outputs/dpl_camels_531_smoke_fullperiod").glob("*/epoch_history.csv"):
        rows = list(csv.DictReader(p.open()))
        runtime.append({"route": "dPL", "scope": "531 smoke", "model": p.parent.name, "basin": 2, "structure": p.parent.name,
                        "restart": "", "runtime_seconds": rows[-1].get("elapsed_s") if rows else "", "success": "true", "source": rel(p), "status": STATUS["result"]})
    write_csv("s3_runtime_summary.csv", runtime or [{"route": "", "scope": "formal 531 missing", "model": "", "basin": "", "structure": "", "restart": "", "runtime_seconds": "", "success": "", "source": "", "status": STATUS["unresolved"]}])
    inv = [{"item": "source code", "path": "ablation/, training/dpl/, models/", "available": "yes", "status": STATUS["code"]},
           {"item": "531 config", "path": "ablation/configs/ic_foundation_531_v1.json", "available": "yes", "status": STATUS["config"]},
           {"item": "531 manifest", "path": "outputs/ic_ablation/foundation_v1/ic_531_dataset_manifest_resolved.json", "available": "yes", "status": STATUS["result"]},
           {"item": "formal 531 IC results", "path": "outputs/ic_ablation/foundation_v1/", "available": "no", "status": STATUS["unresolved"]},
        {"item": "formal 531 dPL results", "path": "remote launcher output dpl_camels_531_multiseed_v1", "available": "no local directory", "status": STATUS["unresolved"]},
           {"item": "screening results", "path": "outputs/ic_ablation/large_scale_screening/v1/", "available": "yes, legacy protocol", "status": STATUS["legacy"]},
           {"item": "license/release/Zenodo", "path": "not found in project scan", "available": "no", "status": STATUS["unresolved"]}]
    write_csv("s3_reproducibility_inventory.csv", inv)


def conflicts():
    rows = [
        {"item": "active IC production results", "status": STATUS["unresolved"], "evidence": "foundation_v1 contains validation/smoke only; no formal 531 IC result manifest", "impact": "cannot report 531 restart convergence, saturation, runtime, or adequacy"},
        {"item": "XNES model-specific settings", "status": STATUS["unresolved"], "evidence": "active stage1 config specifies XAJ only; model-specific final configs absent", "impact": "Table S3.1 incomplete for 12 rows"},
        {"item": "optimizer calibration", "status": STATUS["conflict"], "evidence": "large_scale_screening scans populations/models but uses 1988-1998 and 366-day warmup; not foundation 531", "impact": "cannot call it active calibration"},
        {"item": "objective equivalence", "status": STATUS["conflict"], "evidence": "IC KGEObjective and dPL kge_per_basin are separate implementations", "impact": "claim must be qualified"},
        {"item": "three dPL seeds", "status": STATUS["unresolved"], "evidence": "launcher protocol specifies 42, 123, 2026; local production result inventory absent", "impact": "cannot claim seed equivalence"},
        {"item": "hardware/runtime reproducibility", "status": STATUS["conflict"], "evidence": "screening manifest has GPU, formal 531 production manifest absent", "impact": "runtime cannot be fully reproduced from local evidence"},
        {"item": "legacy 559 assets", "status": STATUS["legacy"], "evidence": "results/ic_xnes_full and outputs/dpl_unified_365d_v1 use 559 IDs/older dates", "impact": "must not be used as current 531 settings"},
    ]
    write_csv("s3_conflicts.csv", rows)
    write_csv("s3_unresolved_items.csv", [r for r in rows if r["status"] == STATUS["unresolved"]] + [{"item": "minimum actions", "status": STATUS["unresolved"], "evidence": "run formal 531 IC and dPL with manifests/checkpoints/logs; add exact equivalence test; record three-seed outputs", "impact": "required before unqualified S3 claims"}])


def pdf_placeholder(path, title, body):
    text = (title + "\n" + body).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 50 760 Td ({text[:180]}) Tj ET"
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", f"<< /Length {len(stream.encode())} >>\nstream\n{stream}\nendstream"]
    out = "%PDF-1.4\n"; offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out.encode())); out += f"{i} 0 obj\n{obj}\nendobj\n"
    xref = len(out.encode()); out += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets[1:])
    out += f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"; path.write_bytes(out.encode())


def figures():
    # A valid, intentionally conservative diagnostic image: trace data exist
    # only for non-active screening, so the figure says so instead of implying
    # foundation-531 coverage.
    ppm = FIG / "_s3_audit_placeholder.ppm"
    w, h = 900, 420
    lines = ["S3 diagnostic evidence", "Foundation 531 formal optimizer calibration: unavailable", "Available traces are legacy/screening only", "See s3_optimizer_calibration_runs.csv and s3_budget_saturation_by_basin.csv"]
    pix = bytearray([245, 247, 250] * (w * h))
    for i, text in enumerate(lines):
        y0 = 50 + i * 52
        for y in range(y0, min(y0 + 22, h)):
            for x in range(45, min(45 + len(text) * 7, w - 5)):
                if (x + y) % 11 < 7:
                    k = (y * w + x) * 3; pix[k:k+3] = bytes((30, 45, 65))
    ppm.write_bytes(f"P6\n{w} {h}\n255\n".encode() + pix)
    for name in ("Fig_S3_1_optimizer_calibration", "Fig_S3_2_estimation_adequacy"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(ppm), str(FIG / (name + ".png"))], check=False)
        pdf_placeholder(FIG / (name + ".pdf"), name, "No active foundation-531 calibration/adequacy result is available.")
    ppm.unlink(missing_ok=True)


def reports():
    table_rows = list(csv.DictReader((OUT / "s3_ic_optimizer_settings.csv").open()))
    table = "| Host | Structure | Dimension | Population | Initial width | Learning rate | Generations | Evaluations | Restarts | Initial center | Stopping rule | Seed protocol |\n|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|\n"
    table += "\n".join(f"| {r['Host']} | {r['Structure']} | {r['Dimension']} | {r['Population']} | {r['Initial width']} | {r['Learning rate']} | {r['Generations']} | {r['Evaluations']} | {r['Restarts']} | {r['Initial center']} | {r['Stopping rule']} | {r['Seed protocol']} |" for r in table_rows)
    (REPORT / "S3_table_and_figure_data.md").write_text("# S3 table and figure data\n\n## Table S3.1\n\n" + table + "\n\n## Figure S3.1\n\nNo active foundation-531 optimizer scan was found. Existing large-scale scans are retained in the CSV with `LEGACY_NOT_ACTIVE` status.\n\n**Caption draft.** Optimizer calibration evidence available in the repository. Foundation-531 calibration is unresolved; legacy screening scans use a different 1988-1998 date protocol and are shown only as provenance.\n\n## Figure S3.2\n\nNo formal foundation-531 restart or budget trace was found. Screening traces are not promoted to 531 production evidence.\n\n**Caption draft.** Estimation adequacy diagnostics. Foundation-531 restart dispersion, budget saturation, and convergence coverage are unavailable from the current result inventory; screening traces are retained for audit traceability only.\n\n## Objective\n\n`KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)`; dPL minimizes the mean of `1-KGE` over valid sampled basin windows. The IC implementation has no epsilon in the GPU KGE formula; dPL uses epsilon-stabilized differentiable KGE.\n\n## Protocol\n\nFoundation 531 dates are warmup 1980-10-01 to 1981-09-30, calibration 1981-10-01 to 1995-09-30, evaluation 1995-10-01 to 2010-09-30. Both routes are configured around 365-day warm-up/prediction windows, but exact shared evaluation equivalence remains unresolved.\n")
    (REPORT / "S3_parameter_estimation_source_of_truth.md").write_text("""# S3 Parameter Estimation: source of truth

## Executive summary

The active foundation-531 data contract is verified: 531 basins, 35 static attributes, explicit 1980-10-01 to 2010-09-30 dates, KGE(Q), and 365-day warm-up windows. The active IC design is EvoTorch XNES in normalized coordinates with clipping, deterministic LHS centers, three starts, 400 generations, population 48 for the XAJ screening configuration, and fixed-budget termination. A complete formal 531 IC-XNES result inventory was not found. Therefore restart dispersion, budget saturation, all-basin convergence, and the compensation-upper-bound claim are unresolved.

The active dPL code is a static attribute-to-parameter MLP with 35 inputs, three 256-unit SiLU/LayerNorm blocks, dropout 0.05 between hidden blocks, a linear output head, sigmoid/clamp outputs, AdamW, learning rate 1e-3, cosine annealing to 1e-4, batch size 128, 100 epochs, and gradient clipping 1.0. The active launcher protocol specifies seeds 42, 123, and 2026. Local 531 outputs are smoke results, not the full production run.

## S3.1 Independent route: optimizer and settings

The active XNES adapter calls local `evotorch.algorithms.XNES` through `ablation/optimizers/xnes.py`; candidate coordinates are normalized and clipped to [0,1] before `normalized_to_physical`. The stage-1 XAJ configuration records population 48, width 0.25, 400 generations, three starts, seed 0, and no test metric. It does not expose a user learning rate; XNES uses its implementation defaults. Initial centers are deterministic LHS points per basin/start, not normalized 0.5. See `s3_ic_optimizer_settings.csv` for rows whose model-specific production evidence is absent.

## S3.2 Hyperparameter calibration

`large_scale_screening/v1` contains real population/model scans, but its manifest uses the 1988 warm-up and 1989-1998 training protocol with 366 warm-up days. It is therefore a screening/calibration asset, not active foundation-531 calibration. No active 531 calibration set covering all hosts/structures was located.

## S3.3 Estimation adequacy

No formal 531 restart result or intermediate generation manifest was located. Screening traces are retained as legacy evidence only. It is not supported to state that independent estimation reached a global optimum, recovered the true parameter set, or established a compensation upper bound.

## S3.4 Constrained route: from attributes to parameters

The dPL runner consumes 35 attributes in the order defined by `ablation/ic_core/data_adapter.py`, robust median/IQR-normalizes them, imputes nonfinite attributes by column median, and clips normalized attributes to ±5. Each model has one parameter head with output dimension equal to its parameter specification. Physical mapping is linear for ordinary parameters and log-linear for `tgd_tau`; the head bias is initialized from physical defaults.

## S3.5 Constrained route: training protocol

The active 531 config specifies AdamW, 1e-3 initial learning rate, weight decay 1e-4, cosine annealing to 1e-4, batch 128, 100 epochs, validation every 10 epochs, and gradient clipping 1.0. The `balanced_valid_kge_windows` sampler samples basins uniformly and filters within-basin windows by at least 30 valid observations and observed standard deviation 0.05. The active launcher protocol specifies seeds 42, 123, and 2026, but no local three-seed production result inventory was found.

## S3.6 Shared objective, warm-up and evaluation path

Both routes target KGE(Q) and use the same broad validity rule, but the IC objective adapter and dPL differentiable KGE are separate implementations. They differ in epsilon stabilization and aggregation context. Code supports the same foundation dates and model classes, but a full parameter/state/forcing equivalence test was not found. Use a qualified statement, not “exactly the same objective/evaluation path.”

## S3.7 Computational environment and reproducibility

The screening freeze manifest records EvoTorch 0.6.1, PyTorch 2.9.1+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 3080 Ti Laptop GPU. A formal foundation-531 production freeze manifest, hardware record, and complete runtime log were not found. The local audit machine lacked importable NumPy/PyTorch in its selected environments; this is not evidence about the production machine.

## Conflicts and unresolved items

See `s3_conflicts.csv` and `s3_unresolved_items.csv`. In particular, 559 assets under `results/ic_xnes_full` and `outputs/dpl_unified_365d_v1` are legacy and must not be cited as foundation-531 settings.

## Evidence index

| Fact | Evidence |
|---|---|
| 531 basin list, 35 attributes, periods, KGE(Q), clipping | `ablation/configs/ic_foundation_531_v1.json`; `outputs/ic_ablation/foundation_v1/ic_531_dataset_manifest_resolved.json` |
| IC XNES adapter and maximize/ranking path | `ablation/optimizers/xnes.py:55-90`; `ablation/ic_core/runtime.py:79-117` |
| normalized-to-physical mapping and log `tgd_tau` | `ablation/ic_core/parameter_adapter.py:56-80` |
| XNES population/width/generations/starts/seed | `ablation/configs/ic_xnes_stage1_screening_v1.json`; `outputs/ic_ablation/stage1_screening/v1/xnes/dry_run_plan.json` |
| optimizer learning-rate/covariance defaults | `evotorch/algorithms/distributed/gaussian.py:1369-1405` |
| screening calibration traces | `outputs/ic_ablation/large_scale_screening/v1/**/result.json`; `outputs/ic_ablation/large_scale_screening/v1/**/trace.json` |
| dPL network construction | `training/dpl/run_dpl_model.py:125-168` |
| dPL attribute preprocessing and windows | `training/dpl/run_dpl_model.py:298-477` |
| dPL optimizer, checkpoint and training loop | `training/dpl/run_dpl_model.py:685-885`; `training/dpl/base_config_camels_531.json` |
| dPL three-seed launcher protocol | `training/dpl/run_camels_531_multiseed_autodl.sh:25-26,104-117` |
| IC objective | `ablation/ic_core/objective_adapter.py:11-40`; `experiments/ic_xnes/gpu_kge.py:58-98` |
| dPL objective | `training/dpl/run_dpl_model.py:480-520` |
| environment and screening runtime | `outputs/ic_ablation/stage1_screening/v1/xnes/environment.json`; `input_freeze_manifest.json` |

## Safe manuscript claims

Claim the verified code/configuration protocol, the XNES implementation and normalized mapping, the dPL architecture/training configuration, and the existence of screening evidence with its different date protocol. State explicitly that formal 531 IC adequacy, three-seed result equivalence, and complete shared-path equivalence remain to be documented.

## Prohibited claims

Do not claim global optimality, recovery of true parameters, a proven compensation upper bound, all-basin convergence, exact objective/evaluation equivalence, equivalent three-seed dPL results, or fully reproducible production runtime.
""")
    readiness = []
    for sec, ready, missing, action, risk in [
        ("S3.1", "XNES adapter, normalized coordinates, XAJ screening settings", "final model-by-structure 531 settings", "archive formal 531 resolved configs", "high"),
        ("S3.2", "legacy screening scan inventory", "active 531 calibration set and selection rule", "run/locate foundation-compatible calibration", "high"),
        ("S3.3", "no unsupported adequacy claim", "531 restart/budget traces", "retain all completed restart results and generation traces", "high"),
        ("S3.4", "35 attributes, MLP and mapping code", "checkpoint metadata for formal run", "archive production checkpoints", "medium"),
        ("S3.5", "active training config and sampler code", "three-seed summaries and production logs", "archive seed-specific outputs", "high"),
        ("S3.6", "date and broad mask alignment", "exact equivalence test", "run deterministic IC/dPL same-input comparison", "high"),
        ("S3.7", "screening environment freeze", "formal 531 hardware/runtime manifest", "attach production manifest and logs", "high"),
    ]:
        readiness.append({"Section": sec, "Ready facts": ready, "Missing facts": missing, "Required action": action, "Manuscript risk": risk})
    with (REPORT / "S3_manuscript_readiness.md").open("w") as f:
        f.write("# S3 manuscript readiness\n\n| Section | Ready facts | Missing facts | Required action | Manuscript risk |\n|---|---|---|---|---|\n")
        f.write("\n".join("| {Section} | {Ready facts} | {Missing facts} | {Required action} | {Manuscript risk} |".format(**r) for r in readiness))
        f.write("\n")
    (REPORT / "S3_claims_and_limitations.md").write_text("""# S3 claims and limitations

## Fully supported

- Foundation 531 data contract: 531 basins, 35 attributes, explicit periods, units, and manifest fingerprints.
- XNES is the active optimizer implementation in the active stage-1 design; coordinates are normalized and clipped before physical mapping.
- The dPL architecture, mapping, optimizer, schedule, sampler, and launcher seed protocol (42, 123, 2026) are code/config verified.

## Supported with qualification

- “Independent estimation used multiple restarts” applies to the XAJ screening design, not a verified full 531 production inventory.
- “The routes target KGE(Q)” is supported; “exactly the same objective” is not, because implementations differ.
- “The routes use the foundation date protocol” is configured, while complete state/mask/evaluation equivalence remains untested.
- “dPL uses one parameter vector per basin per forward” is code-supported for the runner, but the full production checkpoint inventory is absent.

## Prohibited

- XNES reached the global optimum.
- Independent estimation recovered the true parameter set.
- Independent estimation is the compensation upper bound.
- Both routes used exactly the same objective.
- Both routes used exactly the same warm-up/evaluation path.
- All basins converged.
- Three seeds were equivalent.
- Runtime is fully reproducible.

Minimum supplement action: add the formal foundation-531 IC manifest/checkpoints/traces, full dPL production checkpoints and three-seed logs, and a deterministic same-input IC/dPL equivalence test before upgrading any qualified statement.
""")


def main():
    inventory(); call_graph(); optimizer_outputs(); calibration_and_adequacy(); dpl_outputs(); shared_protocol(); environment(); conflicts(); figures(); reports()
    print("S3 audit outputs written to", SUPP)


if __name__ == "__main__":
    main()
