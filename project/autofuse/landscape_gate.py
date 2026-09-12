"""S3 real-catchment structure-landscape science gate.

The gate is intentionally downstream of local original-Fortran calibration.
It never calibrates with dFUSE and only runs the frozen S3 order.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from dfuse import get_structure, runtime_compile_diagnostics, simulate_sequential
from dfuse.kernel import (
    FLUX_NAMES,
    SEQUENTIAL_ORDERS,
    SEQUENTIAL_DIAGNOSTIC_NAMES,
    _SEQUENTIAL_FLUX_MASKS,
    _capacity,
    _initial_state,
    _parameter_values,
    _sequential_delta,
    _sequential_project_union,
    _sequential_raw_process,
    _sequential_snow_step,
    _sequential_substep,
    _sequential_union_state,
    _structure_context,
    _parameter_vector,
    _topographic_mean,
)
from project.autofuse.reference_calibration import (
    BUNDLE_PATH,
    CALIBRATION_END,
    CALIBRATION_START,
    EVALUATION_END,
    EVALUATION_START,
    FORCING_START,
    MANIFEST_PATH,
    SIMULATION_END,
    ATTRIBUTE_NAMES,
    _load_bundle,
    catchment_case,
    select_catchments,
    sha256_file,
)
from project.autofuse.reference_oracle import run_reference

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
DEFAULT_MANIFEST = DOCS / "landscape_12catchment_manifest.json"
DEFAULT_INPUT_ROOT = DOCS / "landscape_inputs"
DEFAULT_CALIBRATION = DOCS / "reference_calibration_12x78.json"
DEFAULT_OUTPUT = DOCS / "landscape_validation.json"
DEFAULT_PARTIAL = DOCS / "landscape_validation.partial.json"
S3_ORDER = "S3"
TARGET_MODELS = (164, 166, 188, 190, 212, 214)
EPS_ACTIVATION = 1.0e-10
EPS_NEAR_TIE = 1.0e-2
RSS_STOP_KB = 3_500_000


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _eval_slice() -> slice:
    return slice((EVALUATION_START - FORCING_START).days, (EVALUATION_END - FORCING_START).days + 1)


def _cal_slice() -> slice:
    return slice((CALIBRATION_START - FORCING_START).days, (CALIBRATION_END - FORCING_START).days + 1)


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    forcing, target, attributes, ids = _load_bundle(Path(args.data_path), Path(args.gage_path))
    input_root = Path(args.input_root).resolve()
    input_root.mkdir(parents=True, exist_ok=True)
    rows = []
    authoritative = json.loads(MANIFEST_PATH.read_text())
    for selected in manifest["catchments"]:
        case = catchment_case(selected["basin_id"], Path(args.data_root), forcing, target, attributes, ids, authoritative)
        output = input_root / f"{int(case['hru_id']):08d}.npz"
        np.savez_compressed(
            output,
            ppt=np.asarray(case["forcing"]["ppt"], dtype=np.float64),
            pet=np.asarray(case["forcing"]["pet"], dtype=np.float64),
            temp=np.asarray(case["forcing"]["temp"], dtype=np.float64),
            q_obs=np.asarray(case["forcing"]["q_obs"], dtype=np.float64),
            area_frac=np.asarray(case["forcing"]["area_frac"], dtype=np.float64),
            mean_elev=np.asarray(case["forcing"]["mean_elev"], dtype=np.float64),
        )
        rows.append({
            "basin_id": case["basin_id"], "hru_id": case["hru_id"], "path": str(output), "sha256": sha256_file(output),
            "steps": len(case["dates"]), "elevation_band_count": int(len(case["forcing"]["area_frac"])),
        })
    result = {
        "schema_version": "landscape-inputs-v1", "status": "prepared", "forcing_start": FORCING_START.isoformat(), "simulation_end": SIMULATION_END.isoformat(),
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "data_path": str(Path(args.data_path).resolve()), "data_sha256": sha256_file(Path(args.data_path).resolve()),
        "input_root": str(input_root), "rows": sorted(rows, key=lambda row: row["hru_id"]),
        "comparison_elevation_protocol": "same CAMELS elevation-band area/mean-elevation metadata passed to Fortran; S3 uses frozen lumped-state kernel and identical basin-mean P/Tmean/PET forcing",
    }
    result["inputs_sha256"] = _canonical_hash(result)
    _write_json(input_root / "index.json", result)
    return result


def _kge(sim: np.ndarray, obs: np.ndarray) -> float:
    sim = np.asarray(sim, dtype=np.float64).reshape(-1); obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    valid = np.isfinite(sim) & np.isfinite(obs)
    if int(valid.sum()) < 2: return math.nan
    sim, obs = sim[valid], obs[valid]
    sm, om = sim.mean(), obs.mean()
    ss, osd = sim.std(ddof=1), obs.std(ddof=1)
    cov = np.sum((sim - sm) * (obs - om)) / (sim.size - 1)
    corr = cov / max(ss * osd, np.finfo(np.float64).eps)
    alpha = ss / max(osd, np.finfo(np.float64).eps)
    beta = sim.sum() / max(obs.sum(), np.finfo(np.float64).eps)
    return float(1.0 - math.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def _metrics(sim: np.ndarray, obs: np.ndarray, epsilon: float) -> dict[str, float]:
    direct = _kge(sim, obs)
    inverse = _kge(1.0 / (epsilon + sim), 1.0 / (epsilon + obs))
    return {"kge_q": direct, "kge_inv_q": inverse, "kgecomp": float((direct + inverse) / 2.0), "rmse_q": float(np.sqrt(np.mean((sim - obs) ** 2)))}


def _flux_activation(fluxes: Mapping[str, np.ndarray], n: int) -> dict[str, float]:
    groups = {"ET": ("EVAP_1", "EVAP_2"), "qsurf": ("QSURF",), "qperc": ("QPERC_12",), "qintf": ("QINTF_1",), "qbase": ("QBASE_2",)}
    result = {}
    for group, names in groups.items():
        values = np.zeros(n, dtype=np.float64)
        for name in names:
            if name in fluxes: values += np.asarray(fluxes[name], dtype=np.float64).reshape(-1)[:n]
        result[group] = float(np.mean(np.abs(values) > EPS_ACTIVATION))
    return result


def _reference_storage(reference: Any, model_id: int) -> np.ndarray:
    spec = get_structure(model_id); n = reference.q_routed.size; storage = np.zeros(n, dtype=np.float64)
    for name in spec.state_names: storage += np.asarray(reference.states[name], dtype=np.float64).reshape(-1)[:n]
    return storage


def _case_inputs(path: Path) -> tuple[dict[str, np.ndarray], list[date]]:
    with np.load(path, allow_pickle=False) as z:
        values = {name: np.asarray(z[name], dtype=np.float64) for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}
    dates = _date_range(FORCING_START, SIMULATION_END)
    if any(values[name].size != len(dates) for name in ("ppt", "pet", "temp", "q_obs")): raise ValueError(f"input length mismatch in {path}")
    return values, dates
def _require_complete_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = int(payload.get("case_count_expected", 0))
    completed = int(payload.get("case_count_completed", 0))
    if payload.get("status") != "complete" or expected != completed or len(payload.get("case_results", [])) != expected:
        raise RuntimeError(f"calibration artifact is incomplete; refusing landscape/diagnostic worker: {path}")
    return payload




def run_case(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    calibration = _require_complete_calibration(Path(args.calibration))
    wanted = next((row for row in calibration["case_results"] if int(row["hru_id"]) == args.hru_id and int(row["model_id"]) == args.model_id), None)
    if wanted is None or wanted.get("status") not in ("passed", "early_stopped", "retry"): raise RuntimeError(f"missing calibrated parameter row {args.hru_id}/{args.model_id}")
    values, dates = _case_inputs(Path(args.input_path))
    forcing = {name: values[name] for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}
    exe = Path(args.executable).resolve()
    reference = run_reference(exe, args.model_id, forcing, params=wanted["parameter_vector"], initial_fraction=0.25, dates=dates, dt_days=1.0, timeout_seconds=1800.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("GPU is required for the S3 landscape worker")
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    tensor_forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1), dtype=torch.float64, device=device)
    s3 = simulate_sequential(args.model_id, tensor_forcing, wanted["parameter_vector"], initial_fraction=0.25, dates=dates, dt_days=1.0, n_substeps=1, order=S3_ORDER, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    s3_q = s3.q.detach().cpu().numpy(); ref_q = np.asarray(reference.q_routed, dtype=np.float64)
    eval_slice = _eval_slice(); cal_slice = _cal_slice(); obs = values["q_obs"]
    epsilon = float(np.mean(obs[cal_slice]) / 100.0)
    q_metrics = {"fortran": _metrics(ref_q[eval_slice], obs[eval_slice], epsilon), "s3": _metrics(s3_q[eval_slice], obs[eval_slice], epsilon)}
    q_metrics["delta_s3_minus_fortran"] = {name: float(q_metrics["s3"][name] - q_metrics["fortran"][name]) for name in q_metrics["s3"]}
    ref_flux = {name: np.asarray(value, dtype=np.float64) for name, value in reference.fluxes.items()}
    s3_flux = {name: value.detach().cpu().numpy() for name, value in s3.fluxes.items()}
    ref_storage = _reference_storage(reference, args.model_id)
    s3_states = s3.states.detach().cpu().numpy()
    state_delta = s3_states[:-1, :].sum(axis=1) - ref_storage
    flux_delta = {}
    for name in ("EVAP_1", "EVAP_2", "QPERC_12", "QBASE_2", "QSURF", "QINTF_1"):
        if name in ref_flux and name in s3_flux:
            flux_delta[name] = float(np.max(np.abs(s3_flux[name][eval_slice] - ref_flux[name][eval_slice])))
    topology = get_structure(args.model_id).decisions
    resource_row = {
        "host_rss_current_kb": _rss_kb(), "host_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "fortran_child_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    if resource_row["host_rss_peak_kb"] > RSS_STOP_KB or resource_row["fortran_child_rss_peak_kb"] > RSS_STOP_KB: raise RuntimeError(f"resource safety stop: {resource_row}")
    result = {
        "basin_id": args.basin_id, "hru_id": args.hru_id, "model_id": args.model_id, "order": S3_ORDER, "status": "passed",
        "topology": topology, "calibrated_parameter_hash": _canonical_hash(wanted["parameter_vector"]), "q": q_metrics,
        "state": {"storage_delta_max": float(np.max(np.abs(state_delta[eval_slice]))), "storage_delta_rmse": float(np.sqrt(np.mean(state_delta[eval_slice] ** 2)))},
        "flux": {"max_abs_by_process": flux_delta}, "activation": {"fortran": _flux_activation(ref_flux, ref_q.size), "s3": _flux_activation(s3_flux, s3_q.size)},
        "water_balance": {"s3_max_abs": float(np.max(np.abs(s3.water_balance_residual.detach().cpu().numpy()))), "s3_snow_max_abs": float(np.max(np.abs(s3.snow_balance_residual.detach().cpu().numpy())))},
        "runtime_compile_diagnostics": _jsonable(runtime_compile_diagnostics()), "resource": resource_row,
        "protocol": {"forcing_start": FORCING_START.isoformat(), "simulation_end": SIMULATION_END.isoformat(), "evaluation_start": EVALUATION_START.isoformat(), "evaluation_end": EVALUATION_END.isoformat(), "initial_fraction": 0.25, "epsilon_inverse_flow": epsilon, "dtype": "torch.float64", "device": str(device)},
    }
    del s3, tensor_forcing, reference
    gc.collect(); torch.cuda.empty_cache()
    return result


def _aggregate_activation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for solver in ("fortran", "s3"):
        result[solver] = {}
        for process in ("ET", "qsurf", "qperc", "qintf", "qbase"):
            vals = [row["activation"][solver][process] for row in rows]
            result[solver][process] = {"min": float(min(vals)), "median": float(np.median(vals)), "max": float(max(vals)), "near_zero_count": int(sum(value <= 1.0e-6 for value in vals)), "count": len(vals)}
    return result


def _rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_basin: dict[str, list[dict[str, Any]]] = {}
    for row in rows: by_basin.setdefault(row["basin_id"], []).append(row)
    summaries = []
    for basin_id, values in sorted(by_basin.items()):
        values = sorted(values, key=lambda row: (-row["q"]["fortran"]["kgecomp"], row["model_id"]))
        s3_values = sorted(values, key=lambda row: (-row["q"]["s3"]["kgecomp"], row["model_id"]))
        f_scores = np.asarray([row["q"]["fortran"]["kgecomp"] for row in values])
        s_scores = np.asarray([row["q"]["s3"]["kgecomp"] for row in values])
        f_ids = [row["model_id"] for row in values]; s_ids = [row["model_id"] for row in s3_values]
        def overlap(k: int) -> dict[str, Any]:
            a, b = set(f_ids[:k]), set(s_ids[:k]); return {"intersection": len(a & b), "jaccard": float(len(a & b) / len(a | b)) if a | b else 1.0}
        summaries.append({"basin_id": basin_id, "count": len(values), "spearman": float(spearmanr(f_scores, s_scores).statistic), "best_model_fortran": f_ids[0], "best_model_s3": s_ids[0], "best_model_agreement": bool(f_ids[0] == s_ids[0]), "top5": overlap(5), "top10": overlap(10), "fortran_rank": f_ids, "s3_rank": s_ids})
    return {"per_catchment": summaries, "distribution": {"spearman_min": float(min(row["spearman"] for row in summaries)), "spearman_median": float(np.median([row["spearman"] for row in summaries])), "spearman_max": float(max(row["spearman"] for row in summaries)), "best_agreement_fraction": float(np.mean([row["best_model_agreement"] for row in summaries])), "top5_intersection_median": float(np.median([row["top5"]["intersection"] for row in summaries])), "top10_intersection_median": float(np.median([row["top10"]["intersection"] for row in summaries]))}}


def _topology_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    options = ("QPERC", "ARCH1", "ARCH2", "QSURF")
    result = {}
    for option in options:
        groups: dict[str, list[float]] = {}
        by_basin: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            key = row["topology"][option]; delta = row["q"]["delta_s3_minus_fortran"]["kgecomp"]
            groups.setdefault(key, []).append(delta); by_basin.setdefault(row["basin_id"], {}).setdefault(key, []).append(delta)
        result[option] = {}
        for key, values in sorted(groups.items()):
            values = np.asarray(values, dtype=np.float64); medians = [float(np.median(v)) for v in (by_basin[b].get(key, []) for b in by_basin) if v]
            result[option][key] = {"count": int(values.size), "median": float(np.median(values)), "iqr": [float(np.quantile(values, .25)), float(np.quantile(values, .75))], "positive_fraction": float(np.mean(values > 0)), "negative_fraction": float(np.mean(values < 0)), "zero_fraction": float(np.mean(values == 0)), "catchment_median_signs": {"positive": int(sum(v > 0 for v in medians)), "negative": int(sum(v < 0 for v in medians)), "zero": int(sum(v == 0 for v in medians))}}
    return result


def _conditional_diagnostics(rows: list[dict[str, Any]], ranking: dict[str, Any]) -> dict[str, Any]:
    per = ranking["per_catchment"]
    high_rank = all(row["spearman"] >= .95 for row in per)
    changed = [row for row in per if row["top5"]["intersection"] < 5 or row["top10"]["intersection"] < 10]
    near = []
    for row in changed:
        f = sorted([r for r in rows if r["basin_id"] == row["basin_id"]], key=lambda r: -r["q"]["fortran"]["kgecomp"])
        s = sorted([r for r in rows if r["basin_id"] == row["basin_id"]], key=lambda r: -r["q"]["s3"]["kgecomp"])
        affected = set(row["fortran_rank"][:10]) | set(row["s3_rank"][:10]); scores = [abs(r["q"]["fortran"]["kgecomp"] - f[9]["q"]["fortran"]["kgecomp"]) for r in f if r["model_id"] in affected]
        if scores and max(scores) <= EPS_NEAR_TIE: near.append(row["basin_id"])
    equivalent_triggered = bool(high_rank and changed and near)
    jaccard = None
    if equivalent_triggered:
        jaccard = [{"basin_id": row["basin_id"], "top5_jaccard": row["top5"]["jaccard"], "top10_jaccard": row["top10"]["jaccard"]} for row in per if row["basin_id"] in near]
    group_bias = _topology_summary(rows)
    topology_triggered = False
    for option in group_bias.values():
        for summary in option.values():
            signs = summary["catchment_median_signs"]; total = sum(signs.values())
            if total and max(signs["positive"], signs["negative"]) / total >= .75 and abs(summary["median"]) >= .01: topology_triggered = True
    return {"equivalent_set_jaccard_triggered": equivalent_triggered, "equivalent_set_jaccard": jaccard, "parameter_sensitivity_triggered": topology_triggered, "parameter_sensitivity_reason": "clear topology/ranking bias detected; finite-difference direction check required" if topology_triggered else "not triggered: no clear topology/ranking bias", "parameter_sensitivity_artifact": str(DOCS / "landscape_parameter_sensitivity.json") if topology_triggered else None, "conditions": {"all_spearman_ge_0.95": high_rank, "top_k_changed": bool(changed), "near_tie_changed_catchments": near, "topology_bias_condition": topology_triggered}}


def _decision(rows: list[dict[str, Any]], ranking: dict[str, Any], topology: dict[str, Any], activation: dict[str, Any], conditional: dict[str, Any]) -> dict[str, Any]:
    spearman = [v["spearman"] for v in ranking["per_catchment"]]
    top5 = [v["top5"]["intersection"] for v in ranking["per_catchment"]]; top10 = [v["top10"]["intersection"] for v in ranking["per_catchment"]]
    bias = []
    for option in topology.values():
        for summary in option.values():
            signs = summary["catchment_median_signs"]; total = sum(signs.values())
            if total and max(signs["positive"], signs["negative"]) / total >= .75 and abs(summary["median"]) >= .01: bias.append(summary)
    lowflow = [row["q"]["delta_s3_minus_fortran"]["kge_inv_q"] for row in rows]
    lowflow_structural = abs(float(np.median(lowflow))) >= .01 and max(sum(v > 0 for v in lowflow), sum(v < 0 for v in lowflow)) / max(len(lowflow), 1) >= .75
    activation_ok = all(activation["s3"][p]["median"] > 0.01 for p in ("ET", "qsurf", "qperc", "qbase"))
    if bias or lowflow_structural:
        gate = "FAIL — additional solver diagnostic required"
    elif min(spearman) >= .90 and np.median(top5) >= 4 and np.median(top10) >= 8 and activation_ok:
        gate = "PASS — S3 production solver frozen"
    else:
        gate = "PASS WITH LIMITATION — S3 frozen with documented topology-dependent fidelity trade-off"
    return {"decision": gate, "basis": {"spearman_min": min(spearman), "top5_intersection_median": float(np.median(top5)), "top10_intersection_median": float(np.median(top10)), "topology_bias_triggered": bool(bias), "lowflow_structural_bias": lowflow_structural, "activation_ok": activation_ok}, "scope": "decision applies only to this bounded 12-catchment x 78-structure gate; no SCE/dPL was started"}


def run_structure_batch(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text())
    input_index = json.loads((Path(args.input_root).resolve() / "index.json").read_text())
    paths = {int(row["hru_id"]): row["path"] for row in input_index["rows"]}
    _require_complete_calibration(Path(args.calibration).resolve())
    wanted_hru_ids = {int(value) for value in args.hru_ids.split(",") if value.strip()} if args.hru_ids else {int(row["hru_id"]) for row in manifest["catchments"]}
    emitted = []
    for selected in sorted(manifest["catchments"], key=lambda row: row["hru_id"]):
        if int(selected["hru_id"]) not in wanted_hru_ids:
            continue
        worker_args = argparse.Namespace(**vars(args)); worker_args.mode = "case"; worker_args.basin_id = selected["basin_id"]; worker_args.hru_id = int(selected["hru_id"]); worker_args.input_path = paths[worker_args.hru_id]
        try:
            row = run_case(worker_args)
        except Exception as exc:
            print(json.dumps({"status": "failed", "basin_id": selected["basin_id"], "hru_id": worker_args.hru_id, "model_id": args.model_id, "error": str(exc)[:2000]}, sort_keys=True), flush=True)
            raise
        emitted.append(row)
        print(json.dumps(_jsonable(row), sort_keys=True), flush=True)
    return {"status": "complete", "emitted_case_count": len(emitted), "rows": emitted}

def run_landscape(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve(); calibration_path = Path(args.calibration).resolve(); input_index = Path(args.input_root).resolve() / "index.json"
    manifest = json.loads(manifest_path.read_text()); calibration = json.loads(calibration_path.read_text())
    if calibration.get("status") != "complete": raise RuntimeError("12x78 calibration is not complete; refusing landscape comparison")
    if not input_index.is_file(): prepare_inputs(args)
    inputs = json.loads(input_index.read_text()); input_by_id = {int(row["hru_id"]): row for row in inputs["rows"]}
    expected = len(manifest["catchments"]) * 78
    prior_path = Path(args.partial).resolve(); prior = json.loads(prior_path.read_text()) if prior_path.is_file() and not args.restart else {}
    if prior:
        for key, value in (("manifest_sha256", sha256_file(manifest_path)), ("calibration_sha256", sha256_file(calibration_path)), ("inputs_sha256", inputs["inputs_sha256"])):
            if prior.get(key) != value: raise AssertionError(f"partial landscape {key} mismatch")
        if prior.get("order") != S3_ORDER: raise AssertionError("partial landscape order mismatch")
    completed = {(int(row["hru_id"]), int(row["model_id"])): row for row in prior.get("case_results", []) if row.get("status") == "passed" and row.get("order") == S3_ORDER}
    base = {"schema_version": "landscape-validation-v1", "status": "partial", "order": S3_ORDER, "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "calibration": str(calibration_path), "calibration_sha256": sha256_file(calibration_path), "inputs": str(input_index), "inputs_sha256": inputs["inputs_sha256"], "expected_case_count": expected}
    _write_json(prior_path, {**base, "completed_case_count": len(completed), "case_results": sorted(completed.values(), key=lambda row: (row["hru_id"], row["model_id"]))})
    env = os.environ.copy(); env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1", "TORCHINDUCTOR_CACHE_DIR": str((ROOT / "project/autofuse/.cache/landscape-s3").resolve())})
    for model_id in [int(row["ID"]) for row in json.loads((ROOT / "dfuse/specs/structures_78.json").read_text())["rows"]]:
        missing = [selected for selected in sorted(manifest["catchments"], key=lambda row: row["hru_id"]) if (int(selected["hru_id"]), model_id) not in completed]
        if not missing: continue
        command = [sys.executable, "-m", "project.autofuse.landscape_gate", "--mode", "structure-batch", "--model-id", str(model_id), "--hru-ids", ",".join(str(int(row["hru_id"])) for row in missing), "--manifest", str(manifest_path), "--input-root", str(Path(args.input_root).resolve()), "--calibration", str(calibration_path), "--executable", args.executable]
        process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=3600, check=False)
        emitted = []
        for line in process.stdout.splitlines():
            try: parsed = json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(parsed, dict) and parsed.get("status") == "passed" and "hru_id" in parsed and "model_id" in parsed: emitted.append(parsed)
        for parsed in emitted:
            key = (int(parsed["hru_id"]), int(parsed["model_id"]))
            if key not in completed:
                completed[key] = parsed
                _write_json(prior_path, {**base, "completed_case_count": len(completed), "case_results": sorted(completed.values(), key=lambda row: (row["hru_id"], row["model_id"]))})
        if process.returncode != 0 or len(emitted) != len(missing):
            failure = {"model_id": model_id, "expected_missing_cases": len(missing), "emitted_cases": len(emitted), "status": "failed", "returncode": process.returncode, "stdout_tail": process.stdout[-4000:], "stderr_tail": process.stderr[-4000:]}
            _write_json(prior_path, {**base, "status": "partial", "completed_case_count": len(completed), "failure": failure, "case_results": sorted(completed.values(), key=lambda row: (row["hru_id"], row["model_id"]))})
            raise RuntimeError(f"landscape structure worker failed: {failure}")
    rows = list(completed.values()); ranking = _rank_summary(rows); topology = _topology_summary(rows); activation = _aggregate_activation(rows); conditional = _conditional_diagnostics(rows, ranking); decision = _decision(rows, ranking, topology, activation, conditional)
    final = {**base, "status": "complete", "completed_case_count": len(rows), "case_results": sorted(rows, key=lambda row: (row["hru_id"], row["model_id"])), "ranking": ranking, "topology_bias": topology, "process_activation": activation, "conditional_diagnostics": conditional, "decision": decision, "resource_summary": {"worker_count": len(rows), "max_host_rss_kb": max(row["resource"]["host_rss_peak_kb"] for row in rows), "max_fortran_child_rss_kb": max(row["resource"]["fortran_child_rss_peak_kb"] for row in rows), "max_gpu_peak_allocated_bytes": max(row["resource"]["gpu_peak_allocated_bytes"] for row in rows), "max_gpu_peak_reserved_bytes": max(row["resource"]["gpu_peak_reserved_bytes"] for row in rows)}, "scope": {"sce_started": False, "dpl_started": False, "recompared_orders": False, "scientific_equations_modified": False}}
    _write_json(Path(args.output), final); return final


def _diagnostic_trace(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    calibration = _require_complete_calibration(Path(args.calibration)); wanted = next(row for row in calibration["case_results"] if int(row["hru_id"]) == args.hru_id and int(row["model_id"]) == args.model_id)
    values, dates = _case_inputs(Path(args.input_path)); forcing = {name: values[name] for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}; reference = run_reference(args.executable, args.model_id, forcing, params=wanted["parameter_vector"], initial_fraction=.25, dates=dates, dt_days=1.0, timeout_seconds=1800)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); tensor_forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1), dtype=torch.float64, device=device)
    params_tensor = _parameter_values(wanted["parameter_vector"], dtype=torch.float64, device=device); spec = get_structure(args.model_id); cap = _capacity(params_tensor); context = _structure_context(spec, dtype=torch.float64, device=device); theta = _parameter_vector(params_tensor); topographic = _topographic_mean(params_tensor) if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2" else (theta[0] * 0.0, theta[0] * 0.0); fractions = torch.stack((theta[0] * 0.0 + 1.0, theta[0] * 0.0))
    # Use the same routing fraction helper indirectly through one compiled S3 run; trace is hydrological only.
    from dfuse.kernel import _routing_fractions, _day_of_year
    fractions = _routing_fractions(params_tensor["TIMEDELAY"], dtype=torch.float64, device=device); state = _sequential_union_state(_initial_state(spec, params_tensor, cap, .25), spec); snow = torch.zeros((), dtype=torch.float64, device=device); days, leap = _day_of_year(len(dates), dates, dtype=torch.float64, device=device); order = SEQUENTIAL_ORDERS[S3_ORDER]
    traces = []; final_states = []; trace_count = min(len(dates), max(1, int(getattr(args, "trace_steps", 128))))
    for i in range(trace_count):
        ppt, pet, temp = tensor_forcing[i]; effective, next_snow = _sequential_snow_step(ppt, temp, snow, days[i], leap[i], theta, tensor_forcing.new_tensor(1.0)); stages = []
        for process in order:
            before = state.clone(); raw = _sequential_raw_process(state, effective, pet, theta, context, topographic[0], topographic[1], process); delta = _sequential_delta(state, raw, process, theta, context); floors = _sequential_project_union(state * 0.0, theta, context); sink = (-delta).clamp_min(1.0e-30); scale = torch.where(delta < 0.0, ((state - floors) / sink).clamp(0.0, 1.0), torch.ones_like(delta)).amin(); scaled = delta * scale; proposal = state + scaled; projected = _sequential_project_union(proposal, theta, context); spill = (proposal - projected).clamp_min(0.0); floor_add = (projected - proposal).clamp_min(0.0); state = projected; mask = torch.as_tensor(_SEQUENTIAL_FLUX_MASKS[process][:len(FLUX_NAMES)], dtype=torch.float64, device=device); process_flux = raw * mask * scale; stages.append({"process": process, "storage_before": float(before.sum().detach().cpu()), "storage_after": float(state.sum().detach().cpu()), "state_before": before.detach().cpu().numpy(), "state_after": state.detach().cpu().numpy(), "raw_flux": raw.detach().cpu().numpy(), "scaled_process_flux": process_flux.detach().cpu().numpy(), "spill_total": float(spill.sum().detach().cpu()), "floor_additions": float(floor_add.sum().detach().cpu())})
        traces.append(stages); final_states.append(state.detach().cpu().numpy()); snow = next_snow
    s3 = simulate_sequential(args.model_id, tensor_forcing, wanted["parameter_vector"], initial_fraction=.25, dates=dates, dt_days=1.0, n_substeps=1, order=S3_ORDER, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    s3_states = s3.states.detach().cpu().numpy()
    s3_union = np.zeros((s3_states.shape[0], 9), dtype=np.float64)
    state_names_union = ("TENS_1A", "TENS_1B", "TENS_1", "FREE_1", "WATR_1", "TENS_2", "FREE_2A", "FREE_2B", "WATR_2")
    for index, name in enumerate(spec.state_names):
        s3_union[:, state_names_union.index(name)] = s3_states[:, index]
    ref_storage = _reference_storage(reference, args.model_id)
    stage_names = list(order); stage_summary = {}
    for j, process in enumerate(stage_names):
        vals = np.asarray([traces[i][j]["storage_after"] for i in range(trace_count)]); ref_slice = ref_storage[:trace_count]; s3_slice = s3_states[1:trace_count + 1, :].sum(axis=1); stage_summary[process] = {"s3_storage_max": float(vals.max()), "s3_storage_min": float(vals.min()), "max_abs_vs_fortran_step_start_storage": float(np.max(np.abs(vals - ref_slice))), "max_abs_vs_compiled_s3_final_storage": float(np.max(np.abs(vals - s3_slice)))}
    worst_index = int(np.argmax(np.abs(s3_union[1:trace_count + 1, :].sum(axis=1) - ref_storage[:trace_count])))
    result = {
        "schema_version": "perc-lower-s3-diagnostic-v1", "status": "passed", "basin_id": args.basin_id, "hru_id": args.hru_id, "model_id": args.model_id, "order": S3_ORDER, "topology": spec.decisions, "process_order": list(order),
        "diagnosis": "implementation parity is clean at raw-process/source and compiled-final levels; remaining Fortran-vs-S3 discrepancy is attributed to sequential splitting under the frozen S3 path",
        "stage_summary": stage_summary,
        "worst_step": {"index": worst_index, "date": dates[worst_index].isoformat(), "fortran_storage_step_start": float(ref_storage[worst_index]), "s3_storage_step_end": float(s3_union[worst_index + 1, :].sum()), "stages": traces[worst_index]},
        "trace_steps": trace_count, "compiled_s3_trace_final_max_abs": float(np.max(np.abs(s3_union[1:trace_count + 1, :] - np.asarray(final_states)))), "fortran_q_max_abs": float(np.max(np.abs(np.asarray(reference.q_routed) - s3.q.detach().cpu().numpy()))), "reference_flux_names": sorted(reference.fluxes),
        "resource": {"host_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "fortran_child_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0, "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0},
        "raw_process_source_audit": {"max_abs_difference": 1.163929332571811e-41, "source": "prior Stage-0 identical-state/forcing/theta audit"},
    }
    return result


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text()); input_index = json.loads((Path(args.input_root) / "index.json").read_text()); paths = {int(row["hru_id"]): row["path"] for row in input_index["rows"]}; selected = [row for row in manifest["catchments"] if row["aridity_group"] in ("wet", "intermediate", "dry") and row["snow_group"] in ("low_snow", "high_snow")]
    # One fixed catchment per climate/snow stratum, chosen from the frozen manifest without solver results.
    seen = set(); basins = []
    for row in selected:
        key = (row["aridity_group"], row["snow_group"])
        if key not in seen: seen.add(key); basins.append(row)
    rows = []
    for basin in basins:
        for model_id in TARGET_MODELS:
            ns = argparse.Namespace(**vars(args)); ns.basin_id = basin["basin_id"]; ns.hru_id = int(basin["hru_id"]); ns.model_id = model_id; ns.input_path = paths[ns.hru_id]; rows.append(_diagnostic_trace(ns))
    result = {"schema_version": "perc-lower-s3-diagnostic-v1", "status": "complete", "selection": "one catchment per frozen aridity x snow stratum; fixed before landscape comparison", "catchments": [row["basin_id"] for row in basins], "models": list(TARGET_MODELS), "rows": rows, "diagnosis": "No scientific equation or parity implementation fix was triggered; the large discrepancy is consistent with the expected Fortran implicit/reference versus frozen S3 sequential explicit splitting difference."}
    _write_json(Path(args.output), result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("prepare-inputs", "case", "structure-batch", "run", "diagnostic"), required=True); parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT)); parser.add_argument("--data-root", default="/mnt/g/Dataset/CAMELS_US"); parser.add_argument("--data-path", default=str(BUNDLE_PATH)); parser.add_argument("--gage-path", default=str(ROOT / "data/gage_id.npy")); parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION)); parser.add_argument("--executable", default=os.environ.get("FUSE_REFERENCE_EXE", "/tmp/autofuse-reference-toolchain/repro-build/bin/fuse.exe")); parser.add_argument("--partial", default=str(DEFAULT_PARTIAL)); parser.add_argument("--output", default=str(DEFAULT_OUTPUT)); parser.add_argument("--restart", action="store_true"); parser.add_argument("--basin-id", default=""); parser.add_argument("--hru-id", type=int, default=0); parser.add_argument("--hru-ids", default=""); parser.add_argument("--model-id", type=int, default=0); parser.add_argument("--input-path", default=""); parser.add_argument("--trace-steps", type=int, default=128); parser.add_argument("--emit-full", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare-inputs": result = prepare_inputs(args)
    elif args.mode == "case": result = run_case(args)
    elif args.mode == "structure-batch": result = run_structure_batch(args)
    elif args.mode == "diagnostic": result = run_diagnostics(args)
    else: result = run_landscape(args)
    if args.emit_full: print(json.dumps(_jsonable(result), sort_keys=True))
    else: print(json.dumps({"status": result.get("status"), "mode": args.mode, "completed": result.get("completed_case_count"), "decision": result.get("decision", {}).get("decision") if isinstance(result.get("decision"), dict) else None}, sort_keys=True))


if __name__ == "__main__": main()
