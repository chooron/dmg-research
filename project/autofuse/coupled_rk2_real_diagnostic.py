"""Frozen real-catchment validation for the coupled RK2/Fortran FIX_STATES path.

This module deliberately keeps the existing S3 landscape gate untouched.  It
reuses the frozen S3 artifact for S3 ranking/activation values, runs the pinned
Fortran oracle only where raw Q/state/flux arrays are required, and evaluates
Torch-FUSE with the already validated coupled RHS + fixed Heun kernel.
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
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from dfuse import (
    get_structure,
    reset_compile_diagnostics,
    runtime_compile_diagnostics,
    simulate_coupled_rk2,
)
from dfuse.spec import FLUX_NAMES
from project.autofuse.landscape_gate import (
    _cal_slice,
    _case_inputs,
    _canonical_hash,
    _date_range,
    _eval_slice,
    _flux_activation,
    _metrics,
)
from project.autofuse.reference_calibration import (
    CALIBRATION_END,
    CALIBRATION_START,
    EVALUATION_END,
    EVALUATION_START,
    FORCING_START,
    MANIFEST_PATH,
    SIMULATION_END,
)
from project.autofuse.reference_oracle import run_reference

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
CATALOGUE = ROOT / "dfuse/specs/structures_78.json" if (ROOT / "dfuse/specs/structures_78.json").is_file() else ROOT / "project/autofuse/dfuse/specs/structures_78.json"
DEFAULT_SELECTION_CATCHMENTS = DOCS / "s4_diagnostic_catchments.json"
DEFAULT_SELECTION_STRUCTURES = DOCS / "s4_diagnostic_structures.json"
DEFAULT_MANIFEST = DOCS / "landscape_12catchment_manifest.json"
DEFAULT_INPUT_ROOT = DOCS / "landscape_inputs"
DEFAULT_S3 = DOCS / "landscape_validation.json"
DEFAULT_CALIBRATION = DOCS / "reference_calibration_12x78.json"
DEFAULT_EXE = "/tmp/autofuse-reference-toolchain/repro-build/bin/fuse.exe"
DEFAULT_CACHE = ROOT / "project/autofuse/.cache/coupled-rk2-real"
DEFAULT_STAGE1_OUTPUT = DOCS / "coupled_rk2_real_diagnostic_4x12.json"
DEFAULT_STAGE1_PARTIAL = DOCS / "coupled_rk2_real_diagnostic_4x12.partial.json"
DEFAULT_STAGE4_OUTPUT = DOCS / "coupled_rk2_landscape_gate_12x78.json"
DEFAULT_STAGE4_PARTIAL = DOCS / "coupled_rk2_landscape_gate_12x78.partial.json"
DEFAULT_UNSUPPORTED = DOCS / "coupled_rk2_unsupported_path_audit.json"

FROZEN_CATCHMENTS = (
    "USA_09447800",
    "USA_14138900",
    "USA_07167500",
    "USA_13240000",
)
FROZEN_STRUCTURES = (2, 6, 8, 14, 164, 166, 178, 188, 190, 210, 212, 214)
PROCESS_NAMES = ("ET", "qsurf", "qperc", "qintf", "qbase")
TARGET_TOPOLOGIES = {
    "QPERC": {"perc_lower"},
    "ARCH1": {"tension2_1"},
    "ARCH2": {"fixedsiz_2", "unlimfrc_2"},
    "QSURF": {"arno_x_vic", "prms_varnt", "tmdl_param"},
}
RSS_STOP_KB = 3_500_000
GPU_RESERVED_FRACTION_STOP = 0.85
REFERENCE_TIMEOUT_SECONDS = 1800.0
EPS_ACTIVATION = 1.0e-10
CAPACITY_TOLERANCE = 1.0e-12
WATER_BALANCE_TOLERANCE = 1.0e-8


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": math.nan, "median": math.nan, "max": math.nan, "mean": math.nan}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _error_stats(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    return {
        "max_abs": float(np.max(absolute)) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(absolute * absolute))) if absolute.size else 0.0,
        "median_abs": float(np.median(absolute)) if absolute.size else 0.0,
    }


def _date_list() -> list[date]:
    return _date_range(FORCING_START, SIMULATION_END)


def _set_runtime_environment(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_frozen_selection(catchment_path: Path, structure_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catchment_payload = json.loads(catchment_path.read_text())
    structure_payload = json.loads(structure_path.read_text())
    if catchment_payload.get("status") != "frozen_before_s4_results":
        raise RuntimeError("frozen catchment selection is not marked frozen_before_s4_results")
    if structure_payload.get("status") != "frozen_before_s4_results":
        raise RuntimeError("frozen structure selection is not marked frozen_before_s4_results")
    catchment_source = _resolve(Path(catchment_payload["selection_basis"]["source_artifact"]))
    structure_source = _resolve(Path(structure_payload["selection_basis"]["source_catalogue"]))
    if catchment_payload["selection_basis"].get("source_sha256") != sha256_file(catchment_source):
        raise RuntimeError("frozen catchment selection source hash mismatch")
    if structure_payload["selection_basis"].get("source_catalogue_sha256") != sha256_file(structure_source):
        raise RuntimeError("frozen structure selection source hash mismatch")
    catchments = catchment_payload.get("catchments", [])
    structures = structure_payload.get("structures", [])
    if tuple(row["basin_id"] for row in catchments) != FROZEN_CATCHMENTS:
        raise RuntimeError("the required four catchments differ from the frozen S4 selection")
    if tuple(int(row["model_id"]) for row in structures) != FROZEN_STRUCTURES:
        raise RuntimeError("the required twelve structures differ from the frozen S4 selection")
    return catchments, structures


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload.get("catchments", [])
    by_id = {row["basin_id"]: row for row in rows}
    if len(rows) != 12 or len(by_id) != 12 or len({int(row["hru_id"]) for row in rows}) != 12:
        raise RuntimeError("the frozen 12-catchment manifest is incomplete or non-unique")
    return payload


def _load_inputs(input_root: Path) -> tuple[dict[int, Path], dict[str, Any]]:
    index_path = input_root / "index.json"
    payload = json.loads(index_path.read_text())
    paths: dict[int, Path] = {}
    for row in payload.get("rows", []):
        path = _resolve(Path(row["path"]))
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"landscape input hash mismatch: {path}")
        paths[int(row["hru_id"])] = path
    if len(paths) != 12:
        raise RuntimeError("the prepared landscape inputs do not contain twelve catchments")
    return paths, payload


def _load_calibration(path: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    expected = int(payload.get("case_count_expected", 0))
    rows = payload.get("case_results", [])
    if payload.get("status") != "complete" or expected != 936 or len(rows) != 936:
        raise RuntimeError("the fixed 12x78 calibration archive is incomplete")
    result = {(int(row["hru_id"]), int(row["model_id"])): row for row in rows}
    if len(result) != 936:
        raise RuntimeError("the fixed calibration archive contains duplicate cases")
    return result, payload


def _load_s3(path: Path, manifest: Path, calibration: Path, input_index: Mapping[str, Any]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or int(payload.get("completed_case_count", 0)) != 936:
        raise RuntimeError("completed S3 landscape artifact is required")
    expected = {
        "manifest_sha256": sha256_file(manifest),
        "calibration_sha256": sha256_file(calibration),
        "inputs_sha256": input_index["inputs_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"S3 artifact {key} does not match the frozen source")
    rows = {(int(row["hru_id"]), int(row["model_id"])): row for row in payload.get("case_results", [])}
    if len(rows) != 936:
        raise RuntimeError("completed S3 artifact does not contain 936 unique cases")
    return rows, payload


def _write_unsupported_audit(output: Path, reference_audit: Path) -> dict[str, Any]:
    catalogue = json.loads(CATALOGUE.read_text())
    rows = catalogue.get("rows", [])
    unsupported = [int(row["ID"]) for row in rows if row.get("ARCH2") == "topmdexp_2"]
    payload = {
        "schema_version": "coupled-rk2-unsupported-path-audit-v1",
        "status": "complete",
        "source_catalogue": str(CATALOGUE),
        "source_catalogue_sha256": sha256_file(CATALOGUE),
        "catalogue_count": len(rows),
        "topmdexp_2_structure_ids": unsupported,
        "topmdexp_2_structure_count": len(unsupported),
        "triggered_in_frozen_78_catalogue": bool(unsupported),
        "source_contract": {
            "audit_artifact": str(reference_audit),
            "audit_artifact_sha256": sha256_file(reference_audit),
            "handling": "no silent bypass; any source STOP path remains unsupported",
        },
        "result": "not encountered: frozen 78-structure catalogue contains no ARCH2=topmdexp_2 rows" if not unsupported else "present: stop before final gate and record every triggered path",
        "scope": {"equations_modified": False, "catalogue_modified": False, "calibration_modified": False},
    }
    _write_json(output, payload)
    return payload


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


def _quality(q: np.ndarray, instantaneous: np.ndarray, fluxes: Mapping[str, np.ndarray], states: np.ndarray, state_names: Sequence[str], params: Mapping[str, float], model_id: int, residual: np.ndarray | None) -> dict[str, Any]:
    caps = _parameter_caps(params)
    finite_values = [q, instantaneous, states, *(np.asarray(value) for value in fluxes.values())]
    if residual is not None:
        finite_values.append(residual)
    finite = bool(all(np.isfinite(value).all() for value in finite_values))
    negative_count = int(np.sum(states < 0.0))
    capacity_count = 0
    max_excess = 0.0
    for index, name in enumerate(state_names):
        if name == "WATR_2" and get_structure(model_id).decisions["ARCH2"] != "fixedsiz_2":
            continue
        excess = np.asarray(states[:, index], dtype=np.float64) - caps[name]
        capacity_count += int(np.sum(excess > CAPACITY_TOLERANCE))
        max_excess = max(max_excess, float(np.max(excess)))
    return {
        "finite": finite,
        "negative_state_count": negative_count,
        "min_state": float(np.min(states)),
        "capacity_violation_count": capacity_count,
        "max_capacity_violation": max(0.0, max_excess),
    }


def _reference_water_balance(reference: Any, model_id: int) -> np.ndarray:
    spec = get_structure(model_id)
    storage = np.stack([np.asarray(reference.states[name], dtype=np.float64) for name in spec.state_names], axis=1).sum(axis=1)
    flux = {name: np.asarray(value, dtype=np.float64) for name, value in reference.fluxes.items()}
    external = flux["EFF_PPT"] - flux["EVAP_1"] - flux["EVAP_2"] - flux["QSURF"] - flux["QINTF_1"] - flux["OFLOW_1"] - flux["OFLOW_2"] - flux["QBASE_2"]
    return np.diff(storage) - external[:-1]


def _fidelity(reference: Any, simulation: Any, model_id: int, params: Mapping[str, float], eval_slice: slice) -> dict[str, Any]:
    spec = get_structure(model_id)
    ref_q = np.asarray(reference.q_routed, dtype=np.float64)
    sim_q = simulation.q.detach().cpu().numpy().astype(np.float64)
    ref_states = np.stack([np.asarray(reference.states[name], dtype=np.float64) for name in spec.state_names], axis=1)
    sim_states = simulation.states.detach().cpu().numpy().astype(np.float64)[:-1]
    q_error = sim_q[eval_slice] - ref_q[eval_slice]
    state_error = sim_states[eval_slice] - ref_states[eval_slice]
    storage_error = state_error.sum(axis=1)
    ref_flux = {name: np.asarray(value, dtype=np.float64) for name, value in reference.fluxes.items()}
    sim_flux = {name: value.detach().cpu().numpy().astype(np.float64) for name, value in simulation.fluxes.items()}
    flux_by_name = {name: _error_stats(sim_flux[name][eval_slice] - values[eval_slice]) for name, values in ref_flux.items() if name in sim_flux}
    all_flux_error = np.concatenate([sim_flux[name][eval_slice] - values[eval_slice] for name, values in ref_flux.items() if name in sim_flux])
    ref_wb = _reference_water_balance(reference, model_id)
    sim_wb = simulation.water_balance_residual.detach().cpu().numpy().astype(np.float64)
    return {
        "q": _error_stats(q_error),
        "storage": _error_stats(storage_error),
        "state": _error_stats(state_error),
        "flux": {"overall": _error_stats(all_flux_error), "by_name": flux_by_name},
        "water_balance": {
            "coupled_full_max_abs": float(np.max(np.abs(sim_wb))),
            "coupled_evaluation_max_abs": float(np.max(np.abs(sim_wb[eval_slice]))),
            "fortran_reconstructed_max_abs": float(np.max(np.abs(ref_wb))),
        },
        "quality": {
            "coupled": _quality(sim_q, simulation.q_instantaneous.detach().cpu().numpy(), sim_flux, sim_states, spec.state_names, params, model_id, sim_wb),
            "fortran": _quality(ref_q, np.asarray(reference.q_instantaneous), ref_flux, ref_states, spec.state_names, params, model_id, ref_wb),
        },
    }


def _stage_activation(simulation: Any, name: str) -> float:
    values = simulation.sequential_diagnostics[name].detach().cpu().numpy().astype(np.float64)
    return float(np.mean(np.abs(values) > EPS_ACTIVATION))


def _run_case(basin: Mapping[str, Any], model_id: int, input_path: Path, calibration: Mapping[tuple[int, int], Mapping[str, Any]], s3_rows: Mapping[tuple[int, int], Mapping[str, Any]], executable: Path, dates: list[date], device: torch.device) -> dict[str, Any]:
    hru_id = int(basin["hru_id"])
    values, _ = _case_inputs(input_path)
    wanted = calibration[(hru_id, model_id)]
    if wanted.get("status") not in ("passed", "early_stopped"):
        raise RuntimeError(f"unsupported calibration status for {hru_id}/{model_id}: {wanted.get('status')}")
    params = wanted["parameter_vector"]
    old_s3 = s3_rows[(hru_id, model_id)]
    theta_hash = _canonical_hash(params)
    if old_s3.get("calibrated_parameter_hash") != theta_hash:
        raise RuntimeError(f"S3 theta hash mismatch for {hru_id}/{model_id}")
    forcing = {name: values[name] for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}
    reference_started = time.perf_counter()
    reference = run_reference(executable, model_id, forcing, params=params, initial_fraction=0.25, dates=dates, dt_days=1.0, timeout_seconds=REFERENCE_TIMEOUT_SECONDS)
    reference_elapsed = time.perf_counter() - reference_started
    tensor_forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1), dtype=torch.float64, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    coupled_started = time.perf_counter()
    coupled = simulate_coupled_rk2(model_id, tensor_forcing, params, initial_fraction=0.25, dates=dates, dt_days=1.0, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    torch.cuda.synchronize(device)
    coupled_elapsed = time.perf_counter() - coupled_started
    eval_slice = _eval_slice()
    epsilon = float(np.mean(values["q_obs"][_cal_slice()]) / 100.0)
    ref_q = np.asarray(reference.q_routed, dtype=np.float64)
    coupled_q = coupled.q.detach().cpu().numpy().astype(np.float64)
    fortran_metrics = _metrics(ref_q[eval_slice], values["q_obs"][eval_slice], epsilon)
    coupled_metrics = _metrics(coupled_q[eval_slice], values["q_obs"][eval_slice], epsilon)
    s3_metrics = dict(old_s3["q"]["s3"])
    old_fortran_metrics = old_s3["q"]["fortran"]
    fortran_reuse = {name: abs(float(fortran_metrics[name]) - float(old_fortran_metrics[name])) for name in ("kge_q", "kge_inv_q", "kgecomp")}
    ref_flux = {name: np.asarray(value, dtype=np.float64) for name, value in reference.fluxes.items()}
    coupled_flux = {name: value.detach().cpu().numpy().astype(np.float64) for name, value in coupled.fluxes.items()}
    fidelity = _fidelity(reference, coupled, model_id, params, eval_slice)
    activation = {
        "fortran": _flux_activation(ref_flux, ref_q.size),
        "s3": dict(old_s3["activation"]["s3"]),
        "coupled_rk2": _flux_activation(coupled_flux, coupled_q.size),
    }
    stage_activation = {name: _stage_activation(coupled, name) for name in ("stage1_qperc", "stage2_qperc", "stage1_qsurf", "stage2_qsurf", "stage1_evap_total", "stage2_evap_total")}
    old_s3_flux_max = max((float(value) for value in old_s3["flux"]["max_abs_by_process"].values()), default=0.0)
    resource_row = {
        "host_rss_kb": _rss_kb(),
        "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "fortran_child_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "reference_elapsed_seconds": reference_elapsed,
        "coupled_elapsed_seconds": coupled_elapsed,
    }
    if resource_row["host_peak_rss_kb"] > RSS_STOP_KB or resource_row["fortran_child_rss_peak_kb"] > RSS_STOP_KB:
        raise RuntimeError(f"resource safety stop: {resource_row}")
    return {
        "basin_id": basin["basin_id"],
        "hru_id": hru_id,
        "model_id": model_id,
        "topology": dict(get_structure(model_id).decisions),
        "calibrated_parameter_hash": theta_hash,
        "calibration_status": wanted["status"],
        "q": {
            "fortran": fortran_metrics,
            "s3": s3_metrics,
            "coupled_rk2": coupled_metrics,
            "delta_kgecomp": {
                "s3_minus_fortran": float(s3_metrics["kgecomp"] - fortran_metrics["kgecomp"]),
                "coupled_minus_fortran": float(coupled_metrics["kgecomp"] - fortran_metrics["kgecomp"]),
            },
            "delta_kge_q": {
                "s3_minus_fortran": float(s3_metrics["kge_q"] - fortran_metrics["kge_q"]),
                "coupled_minus_fortran": float(coupled_metrics["kge_q"] - fortran_metrics["kge_q"]),
            },
            "delta_kge_inv_q": {
                "s3_minus_fortran": float(s3_metrics["kge_inv_q"] - fortran_metrics["kge_inv_q"]),
                "coupled_minus_fortran": float(coupled_metrics["kge_inv_q"] - fortran_metrics["kge_inv_q"]),
            },
        },
        "activation": activation,
        "coupled_stage_activation": stage_activation,
        "fidelity_coupled_rk2_vs_fortran": fidelity,
        "s3_artifact_reuse": {
            "raw_arrays_rerun": False,
            "q_metrics_reused": True,
            "activation_reused": True,
            "state_storage_delta_max": float(old_s3["state"]["storage_delta_max"]),
            "state_storage_delta_rmse": float(old_s3["state"]["storage_delta_rmse"]),
            "flux_max_abs": old_s3_flux_max,
            "fortran_metric_reuse_abs_difference": fortran_reuse,
            "fortran_metric_reuse_pass": bool(max(fortran_reuse.values()) <= 1.0e-8),
        },
        "water_balance": fidelity["water_balance"],
        "resource": resource_row,
        "protocol": {
            "forcing_start": FORCING_START.isoformat(),
            "simulation_end": SIMULATION_END.isoformat(),
            "calibration_start": CALIBRATION_START.isoformat(),
            "calibration_end": CALIBRATION_END.isoformat(),
            "evaluation_start": EVALUATION_START.isoformat(),
            "evaluation_end": EVALUATION_END.isoformat(),
            "initial_fraction": 0.25,
            "dt_days": 1.0,
            "epsilon_inverse_flow": epsilon,
            "dtype": "torch.float64",
            "device": str(device),
        },
    }


def _rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["basin_id"], []).append(row)
    per = []
    for basin_id, basin_rows in sorted(grouped.items()):
        def rank(solver: str) -> list[int]:
            return [int(row["model_id"]) for row in sorted(basin_rows, key=lambda item: (-float(item["q"][solver]["kgecomp"]), int(item["model_id"]))) ]
        f_rank = rank("fortran")
        s3_rank = rank("s3")
        coupled_rank = rank("coupled_rk2")
        by_id = {int(row["model_id"]): row for row in basin_rows}
        f_scores = np.asarray([by_id[mid]["q"]["fortran"]["kgecomp"] for mid in f_rank], dtype=np.float64)
        s3_scores = np.asarray([by_id[mid]["q"]["s3"]["kgecomp"] for mid in f_rank], dtype=np.float64)
        coupled_scores = np.asarray([by_id[mid]["q"]["coupled_rk2"]["kgecomp"] for mid in f_rank], dtype=np.float64)
        def overlap(left: list[int], right: list[int], k: int) -> dict[str, Any]:
            a, b = set(left[:k]), set(right[:k])
            return {"intersection": len(a & b), "jaccard": float(len(a & b) / len(a | b)) if a | b else 1.0}
        per.append({
            "basin_id": basin_id,
            "count": len(basin_rows),
            "spearman_s3": float(spearmanr(f_scores, s3_scores).statistic),
            "spearman_coupled_rk2": float(spearmanr(f_scores, coupled_scores).statistic),
            "fortran_rank": f_rank,
            "s3_rank": s3_rank,
            "coupled_rk2_rank": coupled_rank,
            "top3_s3": overlap(f_rank, s3_rank, 3),
            "top3_coupled_rk2": overlap(f_rank, coupled_rank, 3),
            "top5_s3": overlap(f_rank, s3_rank, 5),
            "top5_coupled_rk2": overlap(f_rank, coupled_rank, 5),
            "top10_s3": overlap(f_rank, s3_rank, 10),
            "top10_coupled_rk2": overlap(f_rank, coupled_rank, 10),
            "best_model_fortran": f_rank[0],
            "best_model_s3": s3_rank[0],
            "best_model_coupled_rk2": coupled_rank[0],
            "best_agreement_s3": bool(f_rank[0] == s3_rank[0]),
            "best_agreement_coupled_rk2": bool(f_rank[0] == coupled_rank[0]),
        })
    def dist(key: str) -> dict[str, Any]:
        return _summary([float(row[key]) for row in per])
    summary = {
        "spearman_s3": dist("spearman_s3"),
        "spearman_coupled_rk2": dist("spearman_coupled_rk2"),
        "spearman_delta_coupled_minus_s3": _summary([row["spearman_coupled_rk2"] - row["spearman_s3"] for row in per]),
        "top3_intersection_s3": _summary([row["top3_s3"]["intersection"] for row in per]),
        "top3_intersection_coupled_rk2": _summary([row["top3_coupled_rk2"]["intersection"] for row in per]),
        "top5_intersection_s3": _summary([row["top5_s3"]["intersection"] for row in per]),
        "top5_intersection_coupled_rk2": _summary([row["top5_coupled_rk2"]["intersection"] for row in per]),
        "top10_intersection_s3": _summary([row["top10_s3"]["intersection"] for row in per]),
        "top10_intersection_coupled_rk2": _summary([row["top10_coupled_rk2"]["intersection"] for row in per]),
        "best_agreement_s3_count": int(sum(row["best_agreement_s3"] for row in per)),
        "best_agreement_coupled_rk2_count": int(sum(row["best_agreement_coupled_rk2"] for row in per)),
    }
    return {"per_catchment": per, "summary": summary}


def _activation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for solver in ("fortran", "s3", "coupled_rk2"):
        result[solver] = {}
        for process in PROCESS_NAMES:
            result[solver][process] = _summary([float(row["activation"][solver][process]) for row in rows])
    result["qperc_abs_gap_to_fortran"] = {
        solver: _summary([abs(float(row["activation"][solver]["qperc"]) - float(row["activation"]["fortran"]["qperc"])) for row in rows])
        for solver in ("s3", "coupled_rk2")
    }
    result["stage_activation_coupled_rk2"] = {name: _summary([float(row["coupled_stage_activation"][name]) for row in rows]) for name in ("stage1_qperc", "stage2_qperc", "stage1_qsurf", "stage2_qsurf", "stage1_evap_total", "stage2_evap_total")}
    return result


def _topology_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for option in ("QPERC", "ARCH1", "ARCH2", "QSURF"):
        groups: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            key = row["topology"][option]
            groups.setdefault(key, {"s3": [], "coupled_rk2": []})
            groups[key]["s3"].append(float(row["q"]["delta_kgecomp"]["s3_minus_fortran"]))
            groups[key]["coupled_rk2"].append(float(row["q"]["delta_kgecomp"]["coupled_minus_fortran"]))
        result[option] = {}
        for key, values in sorted(groups.items()):
            result[option][key] = {}
            for solver, data in values.items():
                result[option][key][solver] = {**_summary(data), "iqr": [float(np.quantile(data, .25)), float(np.quantile(data, .75))], "negative_fraction": float(np.mean(np.asarray(data) < 0.0)), "positive_fraction": float(np.mean(np.asarray(data) > 0.0))}
            old_abs = abs(float(result[option][key]["s3"]["median"]))
            new_abs = abs(float(result[option][key]["coupled_rk2"]["median"]))
            result[option][key]["absolute_median_reduction_coupled_vs_s3"] = (1.0 - new_abs / old_abs) if old_abs > 0.0 else (0.0 if new_abs == 0.0 else -math.inf)
    return result


def _low_flow_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for solver in ("s3", "coupled_rk2"):
        delta_key = "s3_minus_fortran" if solver == "s3" else "coupled_minus_fortran"
        result[solver] = {
            "kge_q": _summary([float(row["q"]["delta_kge_q"][delta_key]) for row in rows]),
            "kge_inv_q": _summary([float(row["q"]["delta_kge_inv_q"][delta_key]) for row in rows]),
        }
    return result


def _fidelity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coupled = {}
    for path in (("q",), ("storage",), ("state",), ("flux", "overall")):
        key = "_".join(path)
        coupled[key] = {metric: _summary([float(row["fidelity_coupled_rk2_vs_fortran"][path[0]][metric] if len(path) == 1 else row["fidelity_coupled_rk2_vs_fortran"][path[0]][path[1]][metric]) for row in rows]) for metric in ("max_abs", "rmse", "median_abs")}
    coupled["q_kgecomp_abs_delta"] = _summary([abs(float(row["q"]["delta_kgecomp"]["coupled_minus_fortran"])) for row in rows])
    coupled["water_balance_full_max_abs"] = _summary([float(row["water_balance"]["coupled_full_max_abs"]) for row in rows])
    coupled["water_balance_evaluation_max_abs"] = _summary([float(row["water_balance"]["coupled_evaluation_max_abs"]) for row in rows])
    coupled["fortran_reconstructed_water_balance_max_abs"] = _summary([float(row["water_balance"]["fortran_reconstructed_max_abs"]) for row in rows])
    coupled["finite_count"] = int(sum(bool(row["fidelity_coupled_rk2_vs_fortran"]["quality"]["coupled"]["finite"]) for row in rows))
    coupled["capacity_clean_count"] = int(sum(int(row["fidelity_coupled_rk2_vs_fortran"]["quality"]["coupled"]["capacity_violation_count"]) == 0 for row in rows))
    coupled["negative_state_case_count"] = int(sum(int(row["fidelity_coupled_rk2_vs_fortran"]["quality"]["coupled"]["negative_state_count"]) > 0 for row in rows))
    s3 = {
        "q_kgecomp_abs_delta": _summary([abs(float(row["q"]["delta_kgecomp"]["s3_minus_fortran"])) for row in rows]),
        "state_storage_delta_max": _summary([float(row["s3_artifact_reuse"]["state_storage_delta_max"]) for row in rows]),
        "state_storage_delta_rmse": _summary([float(row["s3_artifact_reuse"]["state_storage_delta_rmse"]) for row in rows]),
        "flux_max_abs": _summary([float(row["s3_artifact_reuse"]["flux_max_abs"]) for row in rows]),
    }
    return {"coupled_rk2_vs_fortran": coupled, "s3_reused_artifact_comparators": s3}


def _target_topology_entries(topology: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for option, keys in TARGET_TOPOLOGIES.items():
        for key in sorted(keys):
            if key in topology.get(option, {}):
                entries.append(topology[option][key])
    return entries


def _decision(stage: str, rows: list[dict[str, Any]], ranking: Mapping[str, Any], activation: Mapping[str, Any], topology: Mapping[str, Any], low_flow: Mapping[str, Any], fidelity: Mapping[str, Any], compile_clean: bool) -> dict[str, Any]:
    per = ranking["per_catchment"]
    rank_strict = sum(row["spearman_coupled_rk2"] > row["spearman_s3"] + 1.0e-12 for row in per)
    rank_non_decrease = all(row["spearman_coupled_rk2"] >= row["spearman_s3"] - 1.0e-12 for row in per)
    top3_strict = sum(row["top3_coupled_rk2"]["intersection"] > row["top3_s3"]["intersection"] for row in per)
    top5_strict = sum(row["top5_coupled_rk2"]["intersection"] > row["top5_s3"]["intersection"] for row in per)
    top10_strict = sum(row["top10_coupled_rk2"]["intersection"] > row["top10_s3"]["intersection"] for row in per)
    rank = ranking["summary"]
    qperc_gap_s3 = float(activation["qperc_abs_gap_to_fortran"]["s3"]["median"])
    qperc_gap_coupled = float(activation["qperc_abs_gap_to_fortran"]["coupled_rk2"]["median"])
    qperc_fortran_median = float(activation["fortran"]["qperc"]["median"])
    qperc_coupled_median = float(activation["coupled_rk2"]["qperc"]["median"])
    qperc_gap_recovered = qperc_gap_coupled < qperc_gap_s3 and abs(qperc_coupled_median - qperc_fortran_median) <= abs(float(activation["s3"]["qperc"]["median"]) - qperc_fortran_median)
    qperc_close = qperc_coupled_median >= 0.9 * qperc_fortran_median
    qperc = qperc_gap_recovered and qperc_close
    target = _target_topology_entries(topology)
    topology_reduced = sum(float(entry["absolute_median_reduction_coupled_vs_s3"]) > 0.0 for entry in target)
    coupled_bias = []
    for entry in target:
        old = entry["s3"]
        new = entry["coupled_rk2"]
        old_systematic = float(old["median"]) <= -0.01 and float(old["negative_fraction"]) >= 0.75
        new_systematic = float(new["median"]) <= -0.01 and float(new["negative_fraction"]) >= 0.75
        if new_systematic and (not old_systematic or abs(float(new["median"])) >= abs(float(old["median"]))):
            coupled_bias.append(entry)
    topology_ok = bool(target) and topology_reduced == len(target) and not coupled_bias
    direct = fidelity["coupled_rk2_vs_fortran"]
    s3 = fidelity["s3_reused_artifact_comparators"]
    numerical = (
        direct["finite_count"] == len(rows)
        and direct["capacity_clean_count"] == len(rows)
        and direct["negative_state_case_count"] == 0
        and float(direct["water_balance_full_max_abs"]["max"]) <= WATER_BALANCE_TOLERANCE
        and float(direct["q_kgecomp_abs_delta"]["median"]) <= float(s3["q_kgecomp_abs_delta"]["median"])
        and float(direct["storage"]["median_abs"]["median"]) <= float(s3["state_storage_delta_max"]["median"])
        and float(direct["flux_overall"]["median_abs"]["median"]) <= float(s3["flux_max_abs"]["median"])
    )
    criteria = {
        "compile_clean": compile_clean,
        "per_catchment_spearman_non_decrease": rank_non_decrease,
        "per_catchment_spearman_strict_improvement_count": rank_strict,
        "top3_non_decreasing_and_improving": bool(all(row["top3_coupled_rk2"]["intersection"] >= row["top3_s3"]["intersection"] for row in per) and top3_strict > 0),
        "top5_non_decreasing_and_improving": bool(all(row["top5_coupled_rk2"]["intersection"] >= row["top5_s3"]["intersection"] for row in per) and top5_strict > 0),
        "best_model_agreement_non_decrease": rank["best_agreement_coupled_rk2_count"] >= rank["best_agreement_s3_count"],
        "qperc_activation_gap_reduced": qperc_gap_recovered,
        "qperc_activation_close_to_fortran": qperc_close,
        "qperc_activation_recovered": qperc,
        "sensitive_topology_bias_reduced_without_new_systematic_bias": topology_ok,
        "direct_q_state_flux_wb_finite_capacity_clean": numerical,
        "low_flow_abs_bias_reduced": abs(float(low_flow["coupled_rk2"]["kge_inv_q"]["median"])) <= abs(float(low_flow["s3"]["kge_inv_q"]["median"])),
    }
    if stage == "4x12":
        passed = bool(all(criteria.values()) and rank_strict >= max(1, math.ceil(len(per) * 0.75)))
        final_decision = "4×12 passed — 12×78 final landscape gate required" if passed else "4×12 failed — coupled-RK2 real-catchment fidelity unresolved"
    else:
        ranking_ok = (
            float(rank["spearman_coupled_rk2"]["min"]) >= float(rank["spearman_s3"]["min"]) - 1.0e-12
            and float(rank["spearman_coupled_rk2"]["median"]) >= float(rank["spearman_s3"]["median"]) - 1.0e-12
            and float(rank["top5_intersection_coupled_rk2"]["median"]) >= float(rank["top5_intersection_s3"]["median"])
            and float(rank["top10_intersection_coupled_rk2"]["median"]) >= float(rank["top10_intersection_s3"]["median"])
            and rank["best_agreement_coupled_rk2_count"] >= rank["best_agreement_s3_count"]
        )
        criteria["12x78_ranking_landscape_non_decrease"] = ranking_ok
        criteria["low_flow_abs_bias_reduced"] = abs(float(low_flow["coupled_rk2"]["kge_inv_q"]["median"])) <= abs(float(low_flow["s3"]["kge_inv_q"]["median"]))
        passed = bool(all(criteria.values()))
        final_decision = "Torch-FUSE v1 numerical kernel validated — ready for formal SCE/dPL experiments" if passed else "12×78 failed — Torch-FUSE numerical kernel not yet validated"
    return {
        "passed": passed,
        "final_decision": final_decision,
        "criteria": criteria,
        "observed_improvement": {
            "spearman_strict_improvement_count": rank_strict,
            "top3_strict_improvement_count": top3_strict,
            "top5_strict_improvement_count": top5_strict,
            "top10_strict_improvement_count": top10_strict,
            "qperc_gap_s3_median": qperc_gap_s3,
            "qperc_gap_coupled_rk2_median": qperc_gap_coupled,
            "qperc_fortran_median": qperc_fortran_median,
            "qperc_coupled_rk2_median": qperc_coupled_median,
            "target_topology_groups": len(target),
            "target_topology_groups_reduced": topology_reduced,
        },
        "threshold_policy": {
            "not_single_metric": True,
            "all_required_criteria_must_pass": True,
            "spearman_strict_count_for_4x12": f"at least ceil(0.75 * {len(per)})",
            "water_balance_abs_tolerance": WATER_BALANCE_TOLERANCE,
            "qperc_close_policy": "coupled median activation must be at least 90% of the Fortran median, while also reducing the S3 absolute activation gap",
            "topology_policy": "every requested target group must reduce absolute median Delta_KGEcomp and no new systematic negative group may appear",
            "s3_direct_fidelity_note": "S3 raw arrays were not rerun; state/flux comparators are the completed artifact's stored max/RMSE fields, while coupled Q/state/flux uses fresh raw-array fidelity",
        },
    }


def _compile_audit() -> dict[str, Any]:
    diagnostics = runtime_compile_diagnostics()
    records = [record for record in diagnostics.get("records", {}).values() if record.get("graph_signature", {}).get("execution_mode") == "coupled_rk2"]
    return {
        "record_count": len(records),
        "compile_attempts": sum(int(record.get("compile_attempts", 0)) for record in records),
        "compile_successes": sum(int(record.get("compile_successes", 0)) for record in records),
        "fallbacks": sum(int(record.get("fallbacks", 0)) for record in records),
        "graph_breaks": sum(int(record.get("graph_breaks", 0)) for record in records),
        "recompilations": sum(int(record.get("recompilations", 0)) for record in records),
        "autograd_recompilations": sum(int(record.get("autograd_recompilations", 0)) for record in records),
        "records": records,
    }


def _compile_smoke(model_ids: Sequence[int], basin: Mapping[str, Any], input_path: Path, calibration: Mapping[tuple[int, int], Mapping[str, Any]], dates: list[date], device: torch.device) -> dict[str, Any]:
    values, _ = _case_inputs(input_path)
    forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1)[:1], dtype=torch.float64, device=device)
    for model_id in model_ids:
        simulate_coupled_rk2(model_id, forcing, calibration[(int(basin["hru_id"]), int(model_id))]["parameter_vector"], initial_fraction=0.25, dates=dates[:1], dt_days=1.0, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
        torch.cuda.synchronize(device)
    del forcing
    torch.cuda.empty_cache()
    return _compile_audit()

def _resource_summary(rows: list[dict[str, Any]], initial_rss: int, cache_dir: Path) -> dict[str, Any]:
    return {
        "host_rss_initial_kb": initial_rss,
        "host_rss_end_kb": _rss_kb(),
        "host_peak_rss_kb": max((int(row["resource"]["host_peak_rss_kb"]) for row in rows), default=_rss_kb()),
        "fortran_child_rss_peak_kb": max((int(row["resource"]["fortran_child_rss_peak_kb"]) for row in rows), default=0),
        "gpu_peak_allocated_bytes": max((int(row["resource"]["gpu_peak_allocated_bytes"]) for row in rows), default=0),
        "gpu_peak_reserved_bytes": max((int(row["resource"]["gpu_peak_reserved_bytes"]) for row in rows), default=0),
        "wall_clock_seconds": float(sum(float(row["resource"]["reference_elapsed_seconds"]) + float(row["resource"]["coupled_elapsed_seconds"]) for row in rows)),
        "cache_dir": str(cache_dir),
        "cache_final": _cache_info(cache_dir),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "rss_stop_kb": RSS_STOP_KB,
        "gpu_reserved_fraction_stop": GPU_RESERVED_FRACTION_STOP,
    }


def _selection_for_stage(stage: str, catchment_selection: Path, structure_selection: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    if stage == "4x12":
        catchments, structures = _load_frozen_selection(catchment_selection, structure_selection)
        return catchments, [int(row["model_id"]) for row in structures], {"catchments_artifact": str(catchment_selection), "structures_artifact": str(structure_selection), "catchments_sha256": sha256_file(catchment_selection), "structures_sha256": sha256_file(structure_selection)}
    catchments = sorted(manifest["catchments"], key=lambda row: int(row["hru_id"]))
    catalogue = json.loads(CATALOGUE.read_text())["rows"]
    model_ids = [int(row["ID"]) for row in catalogue]
    if len(catchments) != 12 or len({row["basin_id"] for row in catchments}) != 12 or len({int(row["hru_id"]) for row in catchments}) != 12:
        raise RuntimeError("12x78 gate requires exactly twelve unique manifest catchments")
    if len(model_ids) != 78 or len(set(model_ids)) != 78:
        raise RuntimeError("12x78 gate requires exactly 78 unique catalogue structures")
    return catchments, model_ids, {"catchments_manifest": str(DEFAULT_MANIFEST), "structures_catalogue": str(CATALOGUE), "manifest_source_sha256": sha256_file(DEFAULT_MANIFEST), "structures_catalogue_sha256": sha256_file(CATALOGUE)}


def run(stage: str, args: argparse.Namespace) -> dict[str, Any]:
    _set_runtime_environment(_resolve(Path(args.cache_dir)))
    if not torch.cuda.is_available():
        raise RuntimeError("coupled RK2 real-catchment diagnostic requires CUDA; refusing CPU fallback")
    if stage == "12x78":
        stage1_path = _resolve(Path(args.stage1_artifact))
        stage1 = json.loads(stage1_path.read_text())
        if stage1.get("decision", {}).get("final_decision") != "4×12 passed — 12×78 final landscape gate required":
            raise RuntimeError("12x78 gate is conditional on a passing 4x12 artifact")
    cache_dir = _resolve(Path(args.cache_dir))
    manifest_path = _resolve(Path(args.manifest))
    calibration_path = _resolve(Path(args.calibration))
    s3_path = _resolve(Path(args.s3_artifact))
    input_root = _resolve(Path(args.input_root))
    executable = _resolve(Path(args.executable))
    catchment_selection = _resolve(Path(args.selection_catchments))
    structure_selection = _resolve(Path(args.selection_structures))
    manifest = _load_manifest(manifest_path)
    input_paths, input_index = _load_inputs(input_root)
    calibration, calibration_payload = _load_calibration(calibration_path)
    s3_rows, s3_payload = _load_s3(s3_path, manifest_path, calibration_path, input_index)
    catchments, model_ids, selection_meta = _selection_for_stage(stage, catchment_selection, structure_selection, manifest)
    manifest_by_basin = {row["basin_id"]: row for row in manifest["catchments"]}
    expected_ids = set(manifest_by_basin)
    if not {row["basin_id"] for row in catchments}.issubset(expected_ids):
        raise RuntimeError("diagnostic catchment is absent from the frozen 12-catchment manifest")
    if any(int(row["hru_id"]) != int(manifest_by_basin[row["basin_id"]]["hru_id"]) for row in catchments):
        raise RuntimeError("diagnostic catchment basin_id/hru_id mapping differs from the frozen manifest")
    if any(int(row["hru_id"]) not in input_paths for row in catchments):
        raise RuntimeError("diagnostic catchment is absent from prepared inputs")
    executable_hash = sha256_file(executable)
    if calibration_payload.get("executable_sha256") != executable_hash:
        raise RuntimeError("reference executable hash differs from the frozen calibration archive")
    if stage == "4x12":
        frozen_catchment_payload = json.loads(catchment_selection.read_text())
        frozen_s3_source = _resolve(Path(frozen_catchment_payload["selection_basis"]["source_artifact"]))
        if s3_path != frozen_s3_source or sha256_file(s3_path) != frozen_catchment_payload["selection_basis"].get("source_sha256"):
            raise RuntimeError("S3 artifact is not the exact frozen catchment-selection source artifact")
    else:
        frozen_structure_payload = json.loads(structure_selection.read_text())
        if frozen_structure_payload.get("status") != "frozen_before_s4_results" or frozen_structure_payload["selection_basis"].get("source_catalogue_sha256") != sha256_file(CATALOGUE):
            raise RuntimeError("12x78 catalogue differs from the frozen structure-selection source")
    valid_keys = {(int(row["hru_id"]), int(model_id)) for row in catchments for model_id in model_ids}
    if any((int(row["hru_id"]), model_id) not in calibration or (int(row["hru_id"]), model_id) not in s3_rows for row in catchments for model_id in model_ids):
        raise RuntimeError("diagnostic selection is absent from frozen theta/S3 artifacts")
    _write_unsupported_audit(_resolve(Path(args.unsupported_output)), _resolve(Path(args.reference_audit)))
    dates = _date_list()
    device = torch.device("cuda")
    reset_compile_diagnostics()
    output_path = _resolve(Path(args.output))
    partial_path = _resolve(Path(args.partial))
    selection_hash = _canonical_hash({"catchments": [row["basin_id"] for row in catchments], "structures": model_ids})
    base = {
        "schema_version": "coupled-rk2-real-diagnostic-v1" if stage == "4x12" else "coupled-rk2-landscape-gate-v1",
        "stage": stage,
        "status": "partial",
        "selection": selection_meta,
        "selection_hash": selection_hash,
        "protocol": {
            "same_forcing_qobs_theta": True,
            "forcing_period": [FORCING_START.isoformat(), SIMULATION_END.isoformat()],
            "calibration_period": [CALIBRATION_START.isoformat(), CALIBRATION_END.isoformat()],
            "evaluation_period": [EVALUATION_START.isoformat(), EVALUATION_END.isoformat()],
            "initial_fraction": 0.25,
            "dt_days": 1.0,
            "fixed_theta": True,
            "no_recalibration": True,
            "s3_metrics_reused": True,
            "s3_raw_arrays_rerun": False,
            "fortran_raw_arrays_rerun": True,
            "coupled_rhs": "structure-specific coupled RHS",
            "solver": "fixed-step explicit RK2/Heun",
            "fix_states": "Fortran-compatible pinned semantics",
            "no_substep_sweep": True,
            "no_process_order_search": True,
            "no_real_diagnostic_before_stage8": False,
            "no_formal_sce": True,
            "no_dpl": True,
            "no_training": True,
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "calibration": {"path": str(calibration_path), "sha256": sha256_file(calibration_path), "case_count": len(calibration_payload["case_results"])},
        "s3_artifact": {"path": str(s3_path), "sha256": sha256_file(s3_path), "status": s3_payload.get("status")},
        "reference": {"executable": str(executable), "sha256": sha256_file(executable), "source_commit": "e6e23a4fc4ff4019bcab55f14537ea43b9525967", "library_dir": os.environ.get("FUSE_REFERENCE_LIB_DIR")},
        "rows_expected": len(catchments) * len(model_ids),
    }
    if stage == "12x78":
        expected_stage1_selection_hash = _canonical_hash({"catchments": list(FROZEN_CATCHMENTS), "structures": list(FROZEN_STRUCTURES)})
        if stage1.get("status") != "complete" or len(stage1.get("rows", [])) != 48 or stage1.get("selection_hash") != expected_stage1_selection_hash:
            raise RuntimeError("stage1 artifact is not the complete frozen 4x12 result")
        for provenance_key in ("manifest", "calibration", "s3_artifact", "reference"):
            if stage1.get(provenance_key) != base.get(provenance_key):
                raise RuntimeError(f"stage1 artifact {provenance_key} provenance differs from the current final-gate sources")
    prior_rows: dict[tuple[int, int], dict[str, Any]] = {}
    if partial_path.is_file() and not args.restart:
        prior = json.loads(partial_path.read_text())
        for provenance_key in ("schema_version", "stage", "selection", "selection_hash", "protocol", "manifest", "calibration", "s3_artifact", "reference", "rows_expected"):
            if prior.get(provenance_key) != base.get(provenance_key):
                raise RuntimeError(f"partial diagnostic {provenance_key} provenance mismatch")
        prior_list = prior.get("rows", [])
        prior_rows = {(int(row["hru_id"]), int(row["model_id"])): row for row in prior_list}
        if len(prior_rows) != len(prior_list) or not set(prior_rows).issubset(valid_keys):
            raise RuntimeError("partial diagnostic contains duplicate or out-of-selection rows")
    rows = list(prior_rows.values())
    failures: list[dict[str, Any]] = []
    initial_rss = _rss_kb()
    started = time.perf_counter()
    stopped_early = False
    for basin in catchments:
        for model_id in model_ids:
            key = (int(basin["hru_id"]), int(model_id))
            if key in prior_rows:
                continue
            try:
                row = _run_case(basin, model_id, input_paths[int(basin["hru_id"])], calibration, s3_rows, executable, dates, device)
                rows.append(row)
                print(json.dumps({"status": "passed", "stage": stage, "basin_id": basin["basin_id"], "model_id": model_id, "completed": len(rows), "expected": base["rows_expected"]}, sort_keys=True), flush=True)
            except Exception as exc:
                failures.append({"basin_id": basin["basin_id"], "hru_id": int(basin["hru_id"]), "model_id": int(model_id), "error": f"{type(exc).__name__}: {str(exc)[:2000]}"})
                stopped_early = True
            partial = {**base, "rows": rows, "failures": failures, "stopped_early": stopped_early, "resource": _resource_summary(rows, initial_rss, cache_dir), "elapsed_seconds": time.perf_counter() - started}
            _write_json(partial_path, partial)
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
    rows.sort(key=lambda row: (int(row["hru_id"]), int(row["model_id"])))
    complete = len(rows) == base["rows_expected"] and not failures
    if not complete:
        payload = {**base, "status": "stopped", "rows": rows, "failures": failures, "stopped_early": stopped_early, "resource": _resource_summary(rows, initial_rss, cache_dir), "elapsed_seconds": time.perf_counter() - started, "decision": {"passed": False, "final_decision": "4×12 failed — coupled-RK2 real-catchment fidelity unresolved" if stage == "4x12" else "12×78 failed — Torch-FUSE numerical kernel not yet validated"}}
        _write_json(output_path, payload)
        return payload
    ranking = _rank_summary(rows)
    activation = _activation_summary(rows)
    topology = _topology_summary(rows)
    low_flow = _low_flow_summary(rows)
    fidelity = _fidelity_summary(rows)
    compile_audit = _compile_audit()
    if compile_audit["record_count"] < len(model_ids):
        compile_audit = _compile_smoke(model_ids, catchments[0], input_paths[int(catchments[0]["hru_id"])], calibration, dates, device)
    compile_clean = bool(compile_audit["record_count"] >= len(model_ids) and compile_audit["compile_attempts"] == compile_audit["compile_successes"] and compile_audit["fallbacks"] == compile_audit["graph_breaks"] == compile_audit["recompilations"] == compile_audit["autograd_recompilations"] == 0)
    decision = _decision(stage, rows, ranking, activation, topology, low_flow, fidelity, compile_clean)
    payload = {
        **base,
        "status": "complete",
        "rows": rows,
        "ranking": ranking,
        "process_activation": activation,
        "topology_bias_delta_kgecomp": topology,
        "low_flow_delta": low_flow,
        "fidelity": fidelity,
        "compile_audit": compile_audit,
        "compile_clean": compile_clean,
        "decision": decision,
        "resource": {**_resource_summary(rows, initial_rss, cache_dir), "wall_clock_total_seconds": time.perf_counter() - started},
        "scope": {"calibration_rerun": False, "s3_full_gate_rerun": False, "formal_sce_started": False, "dpl_started": False, "training_started": False, "scientific_equations_modified": False, "fortran_reference_modified": False},
    }
    _write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("4x12", "12x78"), default="4x12")
    parser.add_argument("--selection-catchments", default=str(DEFAULT_SELECTION_CATCHMENTS))
    parser.add_argument("--selection-structures", default=str(DEFAULT_SELECTION_STRUCTURES))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--s3-artifact", default=str(DEFAULT_S3))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--executable", default=os.environ.get("FUSE_REFERENCE_EXE", DEFAULT_EXE))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default="")
    parser.add_argument("--partial", default="")
    parser.add_argument("--stage1-artifact", default=str(DEFAULT_STAGE1_OUTPUT))
    parser.add_argument("--unsupported-output", default=str(DEFAULT_UNSUPPORTED))
    parser.add_argument("--reference-audit", default=str(DOCS / "fortran_fix_states_dependency_audit.json"))
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if not args.output:
        args.output = str(DEFAULT_STAGE1_OUTPUT if args.stage == "4x12" else DEFAULT_STAGE4_OUTPUT)
    if not args.partial:
        args.partial = str(DEFAULT_STAGE1_PARTIAL if args.stage == "4x12" else DEFAULT_STAGE4_PARTIAL)
    payload = run(args.stage, args)
    print(json.dumps({"status": payload["status"], "stage": args.stage, "rows": len(payload.get("rows", [])), "failures": len(payload.get("failures", [])), "final_decision": payload.get("decision", {}).get("final_decision")}, sort_keys=True))


if __name__ == "__main__":
    main()
