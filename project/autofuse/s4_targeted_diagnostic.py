"""Small real-catchment S4 diagnostic.

This is deliberately not a landscape gate replacement.  It freezes the four
catchments and twelve structures in the two JSON selection artifacts, reuses
completed S3 ranking/metrics, and reruns S3 only for the missing raw arrays
needed for direct Q/state/flux fidelity.  S4 is the existing explicit runtime
order with percolation before baseflow and ET.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import statistics
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import spearmanr

from dfuse import get_structure, reset_compile_diagnostics, runtime_compile_diagnostics, simulate_sequential
from dfuse.kernel import SEQUENTIAL_ORDERS
from dfuse.spec import PARAMETER_NAMES, STATE_NAMES, default_parameters
from project.autofuse.landscape_gate import (
    _cal_slice,
    _case_inputs,
    _canonical_hash,
    _date_range,
    _eval_slice,
    _flux_activation,
    _metrics,
    sha256_file,
)
from project.autofuse.reference_calibration import (
    BUNDLE_PATH,
    EVALUATION_END,
    EVALUATION_START,
    FORCING_START,
    MANIFEST_PATH,
    SIMULATION_END,
)
from project.autofuse.reference_oracle import run_reference

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
DEFAULT_SELECTION_CATCHMENTS = DOCS / "s4_diagnostic_catchments.json"
DEFAULT_SELECTION_STRUCTURES = DOCS / "s4_diagnostic_structures.json"
DEFAULT_MANIFEST = DOCS / "landscape_12catchment_manifest.json"
DEFAULT_INPUT_ROOT = DOCS / "landscape_inputs"
DEFAULT_S3 = DOCS / "landscape_validation.json"
DEFAULT_CALIBRATION = DOCS / "reference_calibration_12x78.json"
DEFAULT_CACHE = ROOT / "project/autofuse/.cache/runtime-validation-78"
DEFAULT_OUTPUT = DOCS / "s4_targeted_diagnostic.json"
DEFAULT_PARTIAL = DOCS / "s4_targeted_diagnostic.partial.json"
DEFAULT_EXE = "/tmp/autofuse-reference-toolchain/repro-build/bin/fuse.exe"
S3_ORDER = "S3"
S4_ORDER_NAME = "S4"
S4_ORDER = SEQUENTIAL_ORDERS[S4_ORDER_NAME]
MOTHER_MODELS = (2, 108, 178, 210)
TOLERANCE = 1.0e-12
REFERENCE_TIMEOUT_SECONDS = 600.0
RSS_STOP_KB = 3_500_000
GPU_RESERVED_FRACTION_STOP = 0.85
EPS_ACTIVATION = 1.0e-10


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
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


def _set_resource_limits() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _cache_info(path: Path) -> dict[str, int]:
    files = 0
    size = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
                size += item.stat().st_size
    return {"files": files, "bytes": size}


def _date_list() -> list[date]:
    return _date_range(FORCING_START, SIMULATION_END)


def _load_frozen(selection_catchments: Path, selection_structures: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catchment_payload = json.loads(selection_catchments.read_text())
    structure_payload = json.loads(selection_structures.read_text())
    if catchment_payload.get("status") != "frozen_before_s4_results" or structure_payload.get("status") != "frozen_before_s4_results":
        raise RuntimeError("S4 selection artifacts are not marked frozen_before_s4_results")
    catchment_source = Path(catchment_payload["selection_basis"]["source_artifact"])
    if not catchment_source.is_absolute():
        catchment_source = ROOT / catchment_source
    structure_source = Path(structure_payload["selection_basis"]["source_catalogue"])
    if not structure_source.is_absolute():
        structure_source = ROOT / structure_source
    if catchment_payload["selection_basis"].get("source_sha256") != sha256_file(catchment_source):
        raise RuntimeError("S4 catchment selection source hash mismatch")
    if structure_payload["selection_basis"].get("source_catalogue_sha256") != sha256_file(structure_source):
        raise RuntimeError("S4 structure selection source hash mismatch")
    catchments = catchment_payload.get("catchments", [])
    structures = structure_payload.get("structures", [])
    if len(catchments) != 4 or len(structures) < 12 or len(structures) > 16:
        raise RuntimeError("S4 diagnostic selection must contain four catchments and 12-16 structures")
    if len({int(row["hru_id"]) for row in catchments}) != len(catchments):
        raise RuntimeError("S4 catchment selection contains duplicate HRUs")
    if len({int(row["model_id"]) for row in structures}) != len(structures):
        raise RuntimeError("S4 structure selection contains duplicate model IDs")
    required = {164, 166, 178, 188, 190, 210, 212, 214}
    if not required.issubset({int(row["model_id"]) for row in structures}):
        raise RuntimeError("S4 structure selection does not contain all required perc_lower models")
    return catchments, structures


def _load_s3(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or int(payload.get("completed_case_count", 0)) != 936:
        raise RuntimeError("completed S3 landscape artifact is required; refusing partial S3 source")
    rows = payload.get("case_results", [])
    result = {(int(row["hru_id"]), int(row["model_id"])): row for row in rows}
    if len(result) != 936:
        raise RuntimeError("completed S3 landscape artifact does not contain 936 unique cases")
    return result


def _load_inputs(input_root: Path) -> dict[int, Path]:
    index = json.loads((input_root / "index.json").read_text())
    return {int(row["hru_id"]): Path(row["path"]) for row in index["rows"]}


def _parameter_caps(params: Mapping[str, float]) -> dict[str, float]:
    max1 = float(params["MAXWATR_1"])
    max2 = float(params["MAXWATR_2"])
    fracten = float(params["FRACTEN"])
    frchzne = float(params.get("FRCHZNE", 0.0))
    fprimqb = float(params.get("FPRIMQB", 0.0))
    return {
        "TENS_1A": frchzne * fracten * max1,
        "TENS_1B": (1.0 - frchzne) * fracten * max1,
        "TENS_1": fracten * max1,
        "FREE_1": (1.0 - fracten) * max1,
        "WATR_1": max1,
        "TENS_2": fracten * max2,
        "FREE_2A": fprimqb * (1.0 - fracten) * max2,
        "FREE_2B": (1.0 - fprimqb) * (1.0 - fracten) * max2,
        "WATR_2": max2,
    }


def _state_quality(states: Mapping[str, np.ndarray], state_names: tuple[str, ...], params: Mapping[str, float], model_id: int) -> dict[str, Any]:
    caps = _parameter_caps(params)
    values = np.stack([np.asarray(states[name], dtype=np.float64) for name in state_names], axis=1)
    max_violation = 0.0
    for index, name in enumerate(state_names):
        if name == "WATR_2" and get_structure(model_id).decisions["ARCH2"] != "fixedsiz_2":
            continue
        max_violation = max(max_violation, float(np.max(values[:, index] - caps[name])))
    return {
        "finite": bool(np.isfinite(values).all()),
        "min_state": float(np.min(values)),
        "max_capacity_violation": max(0.0, max_violation),
    }


def _reference_water_balance(reference: Any, model_id: int) -> float:
    spec = get_structure(model_id)
    storage = np.stack([np.asarray(reference.states[name], dtype=np.float64) for name in spec.state_names], axis=1).sum(axis=1)
    flux = {name: np.asarray(value, dtype=np.float64) for name, value in reference.fluxes.items()}
    external = flux["EFF_PPT"] - flux["EVAP_1"] - flux["EVAP_2"] - flux["QSURF"] - flux["QINTF_1"] - flux["OFLOW_1"] - flux["OFLOW_2"] - flux["QBASE_2"]
    residual = np.diff(storage) - external[:-1]
    return float(np.max(np.abs(residual)))


def _fidelity(reference: Any, simulation: Any, model_id: int, params: Mapping[str, float], eval_slice: slice) -> dict[str, Any]:
    spec = get_structure(model_id)
    sim_q = simulation.q.detach().cpu().numpy().astype(np.float64)
    ref_q = np.asarray(reference.q_routed, dtype=np.float64)
    q_diff = sim_q[eval_slice] - ref_q[eval_slice]
    sim_states = simulation.states.detach().cpu().numpy().astype(np.float64)[:-1]
    ref_states = np.stack([np.asarray(reference.states[name], dtype=np.float64) for name in spec.state_names], axis=1)
    state_diff = sim_states[eval_slice] - ref_states[eval_slice]
    flux_by_process: dict[str, float] = {}
    sim_flux = {name: value.detach().cpu().numpy().astype(np.float64) for name, value in simulation.fluxes.items()}
    for name, values in reference.fluxes.items():
        if name in sim_flux:
            flux_by_process[name] = float(np.max(np.abs(sim_flux[name][eval_slice] - np.asarray(values, dtype=np.float64)[eval_slice])))
    return {
        "q_max_abs": float(np.max(np.abs(q_diff))),
        "q_rmse": float(np.sqrt(np.mean(q_diff**2))),
        "state_max_abs": float(np.max(np.abs(state_diff))),
        "state_rmse": float(np.sqrt(np.mean(state_diff**2))),
        "flux_max_abs_by_name": flux_by_process,
        "flux_max_abs": max(flux_by_process.values(), default=0.0),
        "water_balance_max_abs": float(np.max(np.abs(simulation.water_balance_residual.detach().cpu().numpy()[eval_slice]))),
        "reference_reconstructed_water_balance_max_abs": _reference_water_balance(reference, model_id),
        "finite_and_capacity": _state_quality({name: sim_states[:, i] for i, name in enumerate(spec.state_names)}, spec.state_names, params, model_id),
    }


def _simulation(model_id: int, order: str, values: Mapping[str, np.ndarray], params: Mapping[str, float], dates: list[date], device: torch.device) -> tuple[Any, float]:
    forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1), dtype=torch.float64, device=device)
    started = time.perf_counter()
    result = simulate_sequential(
        model_id,
        forcing,
        params,
        initial_fraction=0.25,
        dates=dates,
        dt_days=1.0,
        n_substeps=1,
        order=order,
        compile_step=True,
        compile_backend="inductor",
        compile_fullgraph=True,
    )
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - started


def _solver_summary(reference: Any, simulation: Any, model_id: int, params: Mapping[str, float], values: Mapping[str, np.ndarray], eval_slice: slice, elapsed: float, solver: str) -> dict[str, Any]:
    q = simulation.q.detach().cpu().numpy().astype(np.float64) if simulation is not None else np.asarray(reference.q_routed, dtype=np.float64)
    fluxes = {name: value.detach().cpu().numpy().astype(np.float64) for name, value in simulation.fluxes.items()} if simulation is not None else {name: np.asarray(value, dtype=np.float64) for name, value in reference.fluxes.items()}
    if solver == "fortran":
        q_metrics = _metrics(np.asarray(reference.q_routed)[eval_slice], values["q_obs"][eval_slice], float(np.mean(values["q_obs"][_cal_slice()]) / 100.0))
        quality = {
            "finite": bool(np.isfinite(np.asarray(reference.q_routed)).all() and all(np.isfinite(v).all() for v in reference.states.values()) and all(np.isfinite(v).all() for v in reference.fluxes.values())),
            "min_state": float(min(np.min(reference.states[name]) for name in get_structure(model_id).state_names)),
            "max_capacity_violation": None,
        }
    else:
        q_metrics = _metrics(q[eval_slice], values["q_obs"][eval_slice], float(np.mean(values["q_obs"][_cal_slice()]) / 100.0))
        quality = _state_quality({name: simulation.states.detach().cpu().numpy().astype(np.float64)[:-1, i] for i, name in enumerate(get_structure(model_id).state_names)}, get_structure(model_id).state_names, params, model_id)
    return {
        "solver": solver,
        "q_metrics": q_metrics,
        "activation": _flux_activation(fluxes, q.size),
        "elapsed_seconds": elapsed,
        "finite_capacity": quality,
    }


def _case_run(basin: Mapping[str, Any], structure: Mapping[str, Any], input_path: Path, calibration: Mapping[tuple[int, int], Mapping[str, Any]], s3_rows: Mapping[tuple[int, int], Mapping[str, Any]], executable: Path, dates: list[date], device: torch.device) -> dict[str, Any]:
    hru_id = int(basin["hru_id"])
    model_id = int(structure["model_id"])
    values, _ = _case_inputs(input_path)
    wanted = calibration[(hru_id, model_id)]
    params = wanted["parameter_vector"]
    old_s3 = s3_rows[(hru_id, model_id)]
    if wanted.get("status") not in ("passed", "early_stopped"):
        raise RuntimeError(f"unsupported calibration status for {hru_id}/{model_id}: {wanted.get('status')}")
    if old_s3.get("calibrated_parameter_hash") != _canonical_hash(params):
        raise RuntimeError(f"S3 theta hash mismatch for {hru_id}/{model_id}")
    forcing = {name: values[name] for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}
    reference_started = time.perf_counter()
    reference = run_reference(executable, model_id, forcing, params=params, initial_fraction=0.25, dates=dates, dt_days=1.0, timeout_seconds=REFERENCE_TIMEOUT_SECONDS)
    reference_elapsed = time.perf_counter() - reference_started
    s3, s3_elapsed = _simulation(model_id, S3_ORDER, values, params, dates, device)
    s4, s4_elapsed = _simulation(model_id, S4_ORDER_NAME, values, params, dates, device)
    eval_slice = _eval_slice()
    epsilon = float(np.mean(values["q_obs"][_cal_slice()]) / 100.0)
    fortran_summary = _solver_summary(reference, None, model_id, params, values, eval_slice, reference_elapsed, "fortran")
    s3_summary = _solver_summary(reference, s3, model_id, params, values, eval_slice, s3_elapsed, "s3")
    s4_summary = _solver_summary(reference, s4, model_id, params, values, eval_slice, s4_elapsed, "s4")
    s3_artifact_q = old_s3["q"]["s3"]
    s3_artifact_activation = old_s3["activation"]["s3"]
    s3_reuse_check = {
        "source": str(DEFAULT_S3.resolve()),
        "reason_for_small_s3_rerun": "completed S3 artifact stores metrics but not raw Q/state/flux arrays required for direct fidelity metrics",
        "artifact_q_metrics": s3_artifact_q,
        "rerun_q_metrics": s3_summary["q_metrics"],
        "q_kgecomp_abs_difference": abs(float(s3_summary["q_metrics"]["kgecomp"]) - float(s3_artifact_q["kgecomp"])),
        "artifact_activation": s3_artifact_activation,
        "rerun_activation": s3_summary["activation"],
        "activation_max_abs_difference": max((abs(float(s3_summary["activation"][name]) - float(s3_artifact_activation[name])) for name in s3_artifact_activation), default=0.0),
        "ranking_source": "completed S3 artifact metrics; raw rerun used only for direct Q/state/flux fidelity",
    }
    if s3_reuse_check["q_kgecomp_abs_difference"] > 1.0e-10 or s3_reuse_check["activation_max_abs_difference"] > 1.0e-10:
        raise RuntimeError(f"selected S3 rerun does not match completed artifact for {hru_id}/{model_id}: {s3_reuse_check}")
    return {
        "basin_id": basin["basin_id"],
        "hru_id": hru_id,
        "model_id": model_id,
        "topology": dict(get_structure(model_id).decisions),
        "calibrated_parameter_hash": _canonical_hash(params),
        "calibration_status": wanted["status"],
        "fortran": fortran_summary,
        "s3": s3_summary,
        "s4": s4_summary,
        "fidelity_s3_vs_fortran": _fidelity(reference, s3, model_id, params, eval_slice),
        "fidelity_s4_vs_fortran": _fidelity(reference, s4, model_id, params, eval_slice),
        "delta_kgecomp": {
            "s3_minus_fortran": float(s3_summary["q_metrics"]["kgecomp"] - fortran_summary["q_metrics"]["kgecomp"]),
            "s4_minus_fortran": float(s4_summary["q_metrics"]["kgecomp"] - fortran_summary["q_metrics"]["kgecomp"]),
        },
        "delta_kge_inv_q": {
            "s3_minus_fortran": float(s3_summary["q_metrics"]["kge_inv_q"] - fortran_summary["q_metrics"]["kge_inv_q"]),
            "s4_minus_fortran": float(s4_summary["q_metrics"]["kge_inv_q"] - fortran_summary["q_metrics"]["kge_inv_q"]),
        },
        "s3_reuse_check": s3_reuse_check,
        "reference_process": {"stdout_tail": reference.metadata.get("stdout_tail", ""), "stderr_tail": reference.metadata.get("stderr_tail", "")},
        "resource_after_case": {
            "host_rss_kb": _rss_kb(),
            "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "fortran_child_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
            "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(array)), "median": float(np.median(array)), "max": float(np.max(array)), "mean": float(np.mean(array))}


def _ranking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_basin: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_basin.setdefault(row["basin_id"], []).append(row)
    per = []
    for basin_id, basin_rows in sorted(by_basin.items()):
        def rank(solver: str) -> list[int]:
            return [r["model_id"] for r in sorted(basin_rows, key=lambda r: (-r[solver]["q_metrics"]["kgecomp"], r["model_id"]))]
        f = rank("fortran")
        s3 = rank("s3")
        s4 = rank("s4")
        f_scores = np.asarray([next(r for r in basin_rows if r["model_id"] == model)["fortran"]["q_metrics"]["kgecomp"] for model in f])
        s3_scores = np.asarray([next(r for r in basin_rows if r["model_id"] == model)["s3"]["q_metrics"]["kgecomp"] for model in f])
        s4_scores = np.asarray([next(r for r in basin_rows if r["model_id"] == model)["s4"]["q_metrics"]["kgecomp"] for model in f])
        def overlap(left: list[int], right: list[int], k: int) -> dict[str, Any]:
            a, b = set(left[:k]), set(right[:k])
            return {"intersection": len(a & b), "jaccard": float(len(a & b) / len(a | b)) if a | b else 1.0}
        per.append({
            "basin_id": basin_id,
            "count": len(basin_rows),
            "spearman_s3": float(spearmanr(f_scores, s3_scores).statistic),
            "spearman_s4": float(spearmanr(f_scores, s4_scores).statistic),
            "fortran_rank": f,
            "s3_rank": s3,
            "s4_rank": s4,
            "top3_s3": overlap(f, s3, 3),
            "top3_s4": overlap(f, s4, 3),
            "top5_s3": overlap(f, s3, 5),
            "top5_s4": overlap(f, s4, 5),
            "best_model_fortran": f[0],
            "best_model_s3": s3[0],
            "best_model_s4": s4[0],
            "best_agreement_s3": f[0] == s3[0],
            "best_agreement_s4": f[0] == s4[0],
        })
    return {"per_catchment": per, "summary": {
        "spearman_s3": _summary([r["spearman_s3"] for r in per]),
        "spearman_s4": _summary([r["spearman_s4"] for r in per]),
        "top3_intersection_s3": _summary([r["top3_s3"]["intersection"] for r in per]),
        "top3_intersection_s4": _summary([r["top3_s4"]["intersection"] for r in per]),
        "top5_intersection_s3": _summary([r["top5_s3"]["intersection"] for r in per]),
        "top5_intersection_s4": _summary([r["top5_s4"]["intersection"] for r in per]),
        "best_agreement_s3_count": int(sum(r["best_agreement_s3"] for r in per)),
        "best_agreement_s4_count": int(sum(r["best_agreement_s4"] for r in per)),
        "s4_spearman_improved_count": int(sum(r["spearman_s4"] > r["spearman_s3"] for r in per)),
        "s4_top3_improved_count": int(sum(r["top3_s4"]["intersection"] > r["top3_s3"]["intersection"] for r in per)),
        "s4_top5_improved_count": int(sum(r["top5_s4"]["intersection"] > r["top5_s3"]["intersection"] for r in per)),
    }}


def _activation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for solver in ("fortran", "s3", "s4"):
        result[solver] = {}
        for process in ("ET", "qsurf", "qperc", "qintf", "qbase"):
            values = [float(row[solver]["activation"][process]) for row in rows]
            result[solver][process] = {**_summary(values), "near_zero_count": int(sum(value <= 1.0e-6 for value in values)), "count": len(values)}
    result["qperc_gap_to_fortran"] = {
        "s3": _summary([abs(row["s3"]["activation"]["qperc"] - row["fortran"]["activation"]["qperc"]) for row in rows]),
        "s4": _summary([abs(row["s4"]["activation"]["qperc"] - row["fortran"]["activation"]["qperc"]) for row in rows]),
    }
    return result


def _topology(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for option in ("QPERC", "ARCH1", "ARCH2", "QSURF"):
        groups: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            key = row["topology"][option]
            groups.setdefault(key, {"s3": [], "s4": []})
            groups[key]["s3"].append(row["delta_kgecomp"]["s3_minus_fortran"])
            groups[key]["s4"].append(row["delta_kgecomp"]["s4_minus_fortran"])
        result[option] = {}
        for key, solver_values in sorted(groups.items()):
            result[option][key] = {}
            for solver, values in solver_values.items():
                result[option][key][solver] = {**_summary(values), "iqr": [float(np.quantile(values, .25)), float(np.quantile(values, .75))], "negative_fraction": float(np.mean(np.asarray(values) < 0.0)), "positive_fraction": float(np.mean(np.asarray(values) > 0.0))}
            result[option][key]["absolute_median_reduction_s4_vs_s3"] = 1.0 - abs(result[option][key]["s4"]["median"]) / max(abs(result[option][key]["s3"]["median"]), 1.0e-15)
    return result


def _low_flow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {solver: _summary([row["delta_kge_inv_q"][f"{solver}_minus_fortran"] for row in rows]) for solver in ("s3", "s4")}


def _fidelity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for solver in ("s3", "s4"):
        keys = ("q_max_abs", "q_rmse", "state_max_abs", "state_rmse", "flux_max_abs", "water_balance_max_abs", "reference_reconstructed_water_balance_max_abs")
        result[solver] = {key: _summary([row[f"fidelity_{solver}_vs_fortran"][key] for row in rows]) for key in keys}
        result[solver]["finite_count"] = int(sum(row[solver]["finite_capacity"]["finite"] for row in rows))
        result[solver]["capacity_clean_count"] = int(sum(row[solver]["finite_capacity"]["max_capacity_violation"] <= TOLERANCE for row in rows))
    return result


def _compile_audit() -> dict[str, Any]:
    diagnostics = runtime_compile_diagnostics()
    result = {}
    for order_name, order in ((S3_ORDER, SEQUENTIAL_ORDERS[S3_ORDER]), (S4_ORDER_NAME, S4_ORDER)):
        records = [record for record in diagnostics.get("records", {}).values() if record.get("graph_signature", {}).get("sequential_order") == list(order)]
        result[order_name] = {
            "record_count": len(records),
            "compile_attempts": sum(int(record.get("compile_attempts", 0)) for record in records),
            "compile_successes": sum(int(record.get("compile_successes", 0)) for record in records),
            "fallbacks": sum(int(record.get("fallbacks", 0)) for record in records),
            "graph_breaks": sum(int(record.get("graph_breaks", 0)) for record in records),
            "recompilations": sum(int(record.get("recompilations", 0)) for record in records),
            "autograd_recompilations": sum(int(record.get("autograd_recompilations", 0)) for record in records),
            "records": records,
        }
    return result


def _timed(fn: Any, device: torch.device, repeats: int = 2) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        fn()
        _sync(device)
        values.append(time.perf_counter() - started)
    return {"min_seconds": min(values), "median_seconds": statistics.median(values), "max_seconds": max(values)}


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _gradient_run(model_id: int, order: str, forcing: torch.Tensor, params: Mapping[str, float], compile_step: bool, dates: list[date]) -> torch.Tensor:
    tensors = {name: torch.tensor(value, dtype=torch.float64, device=forcing.device, requires_grad=True) for name, value in params.items()}
    result = simulate_sequential(model_id, forcing, tensors, initial_fraction=0.25, dates=dates, dt_days=1.0, n_substeps=1, order=order, compile_step=compile_step, compile_backend="inductor", compile_fullgraph=True)
    return result.q.sum()


def _runtime_benchmark(first_basin: Mapping[str, Any], input_path: Path, calibration: Mapping[tuple[int, int], Mapping[str, Any]], device: torch.device, dates: list[date]) -> dict[str, Any]:
    values, _ = _case_inputs(input_path)
    forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1)[:64], dtype=torch.float64, device=device)
    rows = []
    for model_id in MOTHER_MODELS:
        params = calibration[(int(first_basin["hru_id"]), model_id)]["parameter_vector"]
        model_row = {"model_id": model_id, "steps": 64}
        for order in (S3_ORDER, S4_ORDER_NAME):
            simulate_sequential(model_id, forcing, params, initial_fraction=0.25, dates=dates[:64], dt_days=1.0, n_substeps=1, order=order, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
            eager_forward = _timed(lambda: simulate_sequential(model_id, forcing, params, initial_fraction=0.25, dates=dates[:64], dt_days=1.0, n_substeps=1, order=order, compile_step=False), device)
            compiled_forward = _timed(lambda: simulate_sequential(model_id, forcing, params, initial_fraction=0.25, dates=dates[:64], dt_days=1.0, n_substeps=1, order=order, compile_step=True, compile_backend="inductor", compile_fullgraph=True), device)
            eager_backward = _timed(lambda: _gradient_run(model_id, order, forcing, params, False, dates[:64]).backward(), device, repeats=1)
            compiled_backward = _timed(lambda: _gradient_run(model_id, order, forcing, params, True, dates[:64]).backward(), device, repeats=1)
            model_row[order] = {
                "forward": {"eager": eager_forward, "compiled": compiled_forward, "compiled_speedup": eager_forward["median_seconds"] / compiled_forward["median_seconds"]},
                "backward": {"eager": eager_backward, "compiled": compiled_backward, "compiled_speedup": eager_backward["median_seconds"] / compiled_backward["median_seconds"]},
            }
        rows.append(model_row)
    return {"catchment": first_basin["basin_id"], "hru_id": int(first_basin["hru_id"]), "steps": 64, "rows": rows}


def _decision(rows: list[dict[str, Any]], activation: Mapping[str, Any], topology: Mapping[str, Any], ranking: Mapping[str, Any]) -> dict[str, Any]:
    target_groups = []
    for option in ("QPERC", "ARCH1", "ARCH2", "QSURF"):
        for key, summary in topology[option].items():
            if key in {"perc_lower", "tension2_1", "fixedsiz_2", "unlimfrc_2"} or option == "QSURF":
                target_groups.append(summary)
    topology_improved = sum(summary["absolute_median_reduction_s4_vs_s3"] >= 0.25 for summary in target_groups) >= max(1, math.ceil(0.6 * len(target_groups)))
    qperc_gap_s3 = activation["qperc_gap_to_fortran"]["s3"]["median"]
    qperc_gap_s4 = activation["qperc_gap_to_fortran"]["s4"]["median"]
    qperc_recovered = qperc_gap_s4 <= 0.75 * qperc_gap_s3
    rank = ranking["summary"]
    ranking_improved = rank["s4_spearman_improved_count"] >= 3 and (rank["s4_top3_improved_count"] + rank["s4_top5_improved_count"]) >= 2
    new_systematic_bias = any(summary["s4"]["median"] < -0.01 and summary["s4"]["negative_fraction"] >= 0.75 and summary["absolute_median_reduction_s4_vs_s3"] < 0.0 for summary in target_groups)
    criteria = {"qperc_activation_gap_reduced_25pct": qperc_recovered, "topology_bias_reduced_in_majority": topology_improved, "ranking_improved_in_majority": ranking_improved, "no_new_systematic_target_bias": not new_systematic_bias}
    successful = all(criteria.values())
    return {"criteria": criteria, "successful": successful, "recommendation": "S4 targeted diagnostic successful — proceed to 12×78 S4 gate" if successful else "S4 insufficient — storage-block sequential explicit preferred", "thresholds": {"qperc_gap_reduction": "at least 25% median absolute gap reduction", "topology": "at least 60% of targeted groups reduce absolute median bias by at least 25%", "ranking": "Spearman improves in at least 3/4 catchments and at least two Top-k improvements", "new_bias": "no targeted group with negative median <= -0.01, negative fraction >= 0.75, and worsening"}}


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    _set_resource_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("S4 targeted diagnostic requires CUDA; refusing CPU fallback")
    device = torch.device("cuda")
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    catchments, structures = _load_frozen(Path(args.selection_catchments), Path(args.selection_structures))
    s3_rows = _load_s3(Path(args.s3_artifact))
    calibration_payload = json.loads(Path(args.calibration).read_text())
    calibration = {(int(row["hru_id"]), int(row["model_id"])): row for row in calibration_payload["case_results"]}
    if len(calibration) != 936:
        raise RuntimeError("incomplete calibration archive")
    inputs = _load_inputs(Path(args.input_root))
    dates = _date_list()
    reset_compile_diagnostics()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    initial_rss = _rss_kb()
    stopped_early = False
    for basin in catchments:
        for structure in structures:
            try:
                row = _case_run(basin, structure, inputs[int(basin["hru_id"])], calibration, s3_rows, Path(args.executable).resolve(), dates, device)
                rows.append(row)
            except Exception as exc:
                failures.append({"basin_id": basin["basin_id"], "hru_id": int(basin["hru_id"]), "model_id": int(structure["model_id"]), "error": f"{type(exc).__name__}: {str(exc)[:2000]}"})
                stopped_early = True
            partial = {
                "schema_version": "s4-targeted-diagnostic-v1",
                "status": "partial",
                "selection_catchments": str(Path(args.selection_catchments).resolve()),
                "selection_structures": str(Path(args.selection_structures).resolve()),
                "order_s3": list(SEQUENTIAL_ORDERS[S3_ORDER]),
                "order_s4": list(S4_ORDER),
                "rows": rows,
                "failures": failures,
                "stopped_early": stopped_early,
                "resource": {"initial_host_rss_kb": initial_rss, "host_rss_kb": _rss_kb(), "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "fortran_child_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss), "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device))},
                "cache": _cache_info(cache_dir),
            }
            _write_json(Path(args.partial), partial)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            total_gpu = int(torch.cuda.get_device_properties(device).total_memory)
            if _rss_kb() >= RSS_STOP_KB or int(torch.cuda.memory_reserved(device)) >= int(total_gpu * GPU_RESERVED_FRACTION_STOP):
                failures.append({"kind": "resource_safety_stop", "host_rss_kb": _rss_kb(), "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device))})
                stopped_early = True
            if stopped_early:
                break
        if stopped_early:
            break
    complete = len(rows) == len(catchments) * len(structures) and not failures
    if not complete:
        payload = {
            "schema_version": "s4-targeted-diagnostic-v1",
            "status": "stopped",
            "rows": rows,
            "failures": failures,
            "selection": {"catchments": catchments, "structures": structures},
            "scope": {"s3_full_gate_rerun": False, "calibration_rerun": False, "formal_training_started": False, "sce_started": False, "dpl_started": False, "scientific_equations_modified": False},
        }
        _write_json(Path(args.output), payload)
        return payload
    ranking = _ranking(rows)
    activation = _activation(rows)
    topology = _topology(rows)
    low_flow = _low_flow(rows)
    fidelity = _fidelity_summary(rows)
    runtime_benchmark = _runtime_benchmark(catchments[0], inputs[int(catchments[0]["hru_id"])], calibration, device, dates)
    compile_audit = _compile_audit()
    decision = _decision(rows, activation, topology, ranking)
    compile_clean = all(
        compile_audit[order]["record_count"] >= len(structures)
        and compile_audit[order]["fallbacks"] == 0
        and compile_audit[order]["graph_breaks"] == 0
        and compile_audit[order]["recompilations"] == 0
        and compile_audit[order]["autograd_recompilations"] == 0
        for order in (S3_ORDER, S4_ORDER_NAME)
    )
    payload = {
        "schema_version": "s4-targeted-diagnostic-v1",
        "status": "complete",
        "selection": {
            "catchments_artifact": str(Path(args.selection_catchments).resolve()),
            "catchments_sha256": sha256_file(Path(args.selection_catchments)),
            "structures_artifact": str(Path(args.selection_structures).resolve()),
            "structures_sha256": sha256_file(Path(args.selection_structures)),
            "catchments": catchments,
            "structures": structures,
        },
        "protocol": {
            "fortran_vs_s3_vs_s4": True,
            "same_forcing": True,
            "same_qobs": True,
            "same_evaluation_period": [EVALUATION_START.isoformat(), EVALUATION_END.isoformat()],
            "forcing_period": [FORCING_START.isoformat(), SIMULATION_END.isoformat()],
            "initial_fraction": 0.25,
            "fixed_theta": True,
            "calibration_archive": str(Path(args.calibration).resolve()),
            "calibration_sha256": sha256_file(Path(args.calibration)),
            "reference_executable": str(Path(args.executable).resolve()),
            "reference_executable_sha256": sha256_file(Path(args.executable).resolve()),
            "s3_order": list(SEQUENTIAL_ORDERS[S3_ORDER]),
            "s4_order": list(S4_ORDER),
            "s3_existing_landscape_reused_for_ranking": True,
            "s3_raw_arrays_rerun_only_for_selected_subset": True,
            "no_calibration": True,
            "no_full_12x78_s3_rerun": True,
        },
        "rows": rows,
        "activation": activation,
        "topology_delta_kgecomp": topology,
        "ranking": ranking,
        "low_flow_delta_kge_inv_q": low_flow,
        "fidelity": fidelity,
        "runtime_benchmark": runtime_benchmark,
        "compile_audit": compile_audit,
        "compile_clean": compile_clean,
        "resource_summary": {
            "host_rss_initial_kb": initial_rss,
            "host_rss_end_kb": _rss_kb(),
            "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "fortran_child_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
            "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "gpu_peak_allocated_bytes": max((row["resource_after_case"]["gpu_peak_allocated_bytes"] for row in rows), default=0),
            "gpu_peak_reserved_bytes": max((row["resource_after_case"]["gpu_peak_reserved_bytes"] for row in rows), default=0),
            "cache_dir": str(cache_dir),
            "cache_final": _cache_info(cache_dir),
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "rss_stop_kb": RSS_STOP_KB,
            "gpu_reserved_fraction_stop": GPU_RESERVED_FRACTION_STOP,
        },
        "decision": {**decision, "compile_clean": compile_clean, "compile_requirement_satisfied": compile_clean},
        "scope": {"calibration_rerun": False, "s3_full_gate_rerun": False, "formal_training_started": False, "sce_started": False, "dpl_started": False, "scientific_equations_modified": False, "fortran_reference_modified": False},
    }
    _write_json(Path(args.output), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-catchments", default=str(DEFAULT_SELECTION_CATCHMENTS))
    parser.add_argument("--selection-structures", default=str(DEFAULT_SELECTION_STRUCTURES))
    parser.add_argument("--s3-artifact", default=str(DEFAULT_S3))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--executable", default=os.environ.get("FUSE_REFERENCE_EXE", DEFAULT_EXE))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--partial", default=str(DEFAULT_PARTIAL))
    args = parser.parse_args()
    payload = run_diagnostic(args)
    print(json.dumps({"status": payload["status"], "rows": len(payload.get("rows", [])), "failures": len(payload.get("failures", [])), "recommendation": payload.get("decision", {}).get("recommendation")}, sort_keys=True))


if __name__ == "__main__":
    main()
