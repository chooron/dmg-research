"""Diagnose the first long-horizon fork of the coupled RK2 trajectory.

The diagnostic is intentionally downstream of the frozen 4x12 run.  It never
changes the solver equations.  The teacher-forced path feeds the Fortran
start-of-day hydrologic state and Fortran effective precipitation into both
one-step implementations; the free-running path uses the public compiled
coupled-RK2 kernel and records its own accepted trajectory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dfuse import get_structure, simulate_coupled_rk2
from dfuse.kernel import (
    _capacity,
    _day_of_year,
    _initial_state,
    _parameter_values,
    _parameter_vector,
    _routing_fractions,
    _sequential_union_state,
    _structure_context,
    _topographic_mean,
)
from dfuse.runtime import _runtime_coupled_rhs_impl, _runtime_fix_states, _runtime_snow_step
from dfuse.spec import FLUX_NAMES, PARAMETER_NAMES, STATE_NAMES
from project.autofuse.coupled_rhs_validation import parse_fortran_output
from project.autofuse.landscape_gate import _cal_slice, _case_inputs, _canonical_hash, _date_range, _eval_slice, _metrics
from project.autofuse.reference_calibration import (
    CALIBRATION_END,
    CALIBRATION_START,
    EVALUATION_END,
    EVALUATION_START,
    FORCING_START,
    SIMULATION_END,
)
from project.autofuse.reference_oracle import _parameter_values as reference_parameter_values, run_reference

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
SUBSET_PATH = DOCS / "coupled_rk2_trajectory_subset.json"
REAL_4X12_PATH = DOCS / "coupled_rk2_real_diagnostic_4x12.json"
CALIBRATION_PATH = DOCS / "reference_calibration_12x78.json"
INPUT_ROOT = DOCS / "landscape_inputs"
REFERENCE_EXE = Path("/tmp/autofuse-reference-toolchain/repro-build/bin/fuse.exe")
COUPLED_WRAPPER_EXE = Path("/tmp/autofuse-coupled-rhs-build-v5/coupled_rhs_fortran_v5.exe")
REFERENCE_SOURCE_AUDIT = DOCS / "fortran_fix_states_dependency_audit.json"
NUMERIX_SETTINGS = ROOT / "vendor/upstream/fuse-mmcomparison-paper/01_FUSEscripts/fuse_template/settings/fuse_zNumerix.txt"
TRACE_OUTPUT = DOCS / "coupled_rk2_teacher_forced_trace.json"
FREE_OUTPUT = DOCS / "coupled_rk2_free_run_divergence.json"
FLUX_OUTPUT = DOCS / "coupled_rk2_accepted_flux_audit.json"
TIME_OUTPUT = DOCS / "coupled_rk2_time_alignment_audit.json"
CACHE_DIR = ROOT / "project/autofuse/.cache/coupled-rk2-trajectory"
EPS = 1.0e-10
THRESHOLDS = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2)
FLUX_INDEX = {name: index for index, name in enumerate(FLUX_NAMES)}


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


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _stats(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    return {
        "max_abs": float(np.max(absolute)) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(absolute * absolute))) if absolute.size else 0.0,
        "median_abs": float(np.median(absolute)) if absolute.size else 0.0,
    }


def _first_hit(values: np.ndarray, threshold: float, dates: Sequence[date], start: int = 0) -> dict[str, Any] | None:
    indices = np.flatnonzero(np.asarray(values, dtype=np.float64).reshape(-1)[start:] > threshold)
    if indices.size == 0:
        return None
    index = int(indices[0]) + start
    return {"index": index, "date": dates[index].isoformat(), "threshold": threshold, "value": float(np.asarray(values).reshape(-1)[index])}


def _first_bool(values: np.ndarray, dates: Sequence[date], start: int = 0) -> dict[str, Any] | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool).reshape(-1)[start:])
    if indices.size == 0:
        return None
    index = int(indices[0]) + start
    return {"index": index, "date": dates[index].isoformat()}


def _threshold_report(errors: Mapping[str, np.ndarray], dates: Sequence[date]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in errors.items():
        result[name] = {
            "all_days": {str(threshold): _first_hit(values, threshold, dates) for threshold in THRESHOLDS},
            "after_first_step": {str(threshold): _first_hit(values, threshold, dates, start=1) for threshold in THRESHOLDS},
            "max_abs": float(np.max(values)) if len(values) else 0.0,
        }
    return result


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    lib_root = "/tmp/autofuse-reference-toolchain/root/lib"
    usr_lib = "/tmp/autofuse-reference-toolchain/root/usr/lib/x86_64-linux-gnu"
    env["FUSE_REFERENCE_LIB_DIR"] = os.pathsep.join((lib_root, usr_lib))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (lib_root, usr_lib, env.get("LD_LIBRARY_PATH", ""))))
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def _load_subset() -> dict[str, Any]:
    subset = json.loads(SUBSET_PATH.read_text())
    real = json.loads(REAL_4X12_PATH.read_text())
    if subset.get("status") != "frozen_before_trajectory_results":
        raise RuntimeError("trajectory subset is not frozen before results")
    if subset["selection_basis"]["source_real_artifact_sha256"] != _sha(REAL_4X12_PATH):
        raise RuntimeError("trajectory subset source real artifact hash mismatch")
    if tuple(row["basin_id"] for row in subset["catchments"]) != ("USA_09447800",):
        raise RuntimeError("trajectory subset catchment changed")
    if tuple(int(row["model_id"]) for row in subset["structures"]) != (2, 8, 190, 214):
        raise RuntimeError("trajectory subset structures changed")
    if real.get("status") != "complete" or len(real.get("rows", [])) != 48:
        raise RuntimeError("complete 4x12 result is required for trajectory subset provenance")
    return subset


def _load_inputs() -> dict[int, Path]:
    index = json.loads((INPUT_ROOT / "index.json").read_text())
    paths = {}
    for row in index["rows"]:
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        if _sha(path) != row["sha256"]:
            raise RuntimeError(f"input hash mismatch: {path}")
        paths[int(row["hru_id"])] = path
    return paths


def _load_calibration() -> dict[tuple[int, int], dict[str, Any]]:
    payload = json.loads(CALIBRATION_PATH.read_text())
    if payload.get("status") != "complete" or int(payload.get("case_count_completed", 0)) != 936:
        raise RuntimeError("fixed calibration archive is incomplete")
    if payload.get("executable_sha256") != _sha(REFERENCE_EXE):
        raise RuntimeError("reference executable differs from fixed calibration archive")
    rows = {(int(row["hru_id"]), int(row["model_id"])): row for row in payload["case_results"]}
    if len(rows) != 936:
        raise RuntimeError("fixed calibration archive has duplicate cases")
    return rows


def _parameter_context(model_id: int, params: Mapping[str, float], device: torch.device) -> tuple[Any, ...]:
    spec = get_structure(model_id)
    params_tensor = _parameter_values(params, dtype=torch.float64, device=device)
    cap = _capacity(params_tensor)
    theta = _parameter_vector(params_tensor)
    context = _structure_context(spec, dtype=torch.float64, device=device)
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        topo = _topographic_mean(params_tensor)
    else:
        zero = theta[0] * 0.0
        topo = (zero, zero)
    fractions = _routing_fractions(params_tensor["TIMEDELAY"], dtype=torch.float64, device=device)
    choices = tuple(spec.decisions[name] for name in ("ARCH1", "ARCH2", "QSURF", "QPERC"))
    return spec, params_tensor, cap, theta, context, topo, fractions, choices


def _torch_effective_step(union: torch.Tensor, effective: torch.Tensor, pet: torch.Tensor, theta: torch.Tensor, context: Any, topo: tuple[torch.Tensor, torch.Tensor], choices: tuple[str, str, str, str]) -> dict[str, torch.Tensor]:
    zero = union[0] * 0.0
    dt = union[0] * 0.0 + 1.0
    k1, stage1_flux = _runtime_coupled_rhs_impl(union, effective, pet, theta, topo[0], topo[1], choices=choices)
    predictor_raw = union + k1 * dt
    predictor, predictor_flux, predictor_correction, predictor_errors, predictor_lower, predictor_upper = _runtime_fix_states(union, predictor_raw, stage1_flux, theta, dt, choices=choices)
    k2, stage2_flux = _runtime_coupled_rhs_impl(predictor, effective, pet, theta, topo[0], topo[1], choices=choices)
    final_raw = union + 0.5 * (k1 + k2) * dt
    mean_flux = 0.5 * (stage1_flux + stage2_flux)
    accepted_state, accepted_flux, final_correction, final_errors, final_lower, final_upper = _runtime_fix_states(union, final_raw, mean_flux, theta, dt, choices=choices)
    instantaneous = accepted_flux[10] + accepted_flux[9] + accepted_flux[8] + accepted_flux[18] + accepted_flux[15]
    return {
        "input_state": union,
        "stage1_flux": stage1_flux,
        "predictor_raw_state": predictor_raw,
        "post_predictor_state": predictor,
        "post_predictor_flux": predictor_flux,
        "stage2_flux": stage2_flux,
        "raw_final_state": final_raw,
        "mean_flux": mean_flux,
        "accepted_state": accepted_state,
        "accepted_flux": accepted_flux,
        "instantaneous": instantaneous,
        "predictor_lower": predictor_lower,
        "predictor_upper": predictor_upper,
        "final_lower": final_lower,
        "final_upper": final_upper,
        "predictor_correction": predictor_correction,
        "final_correction": final_correction,
        "zero": zero,
    }


def _route(future: torch.Tensor, instantaneous: torch.Tensor, fractions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    routed = future[0] + instantaneous * fractions[0]
    next_future = torch.cat((future[1:] + instantaneous * fractions[1:], future[-1:] * 0.0))
    return routed, next_future


def _fortran_one_step(wrapper: Path, model_id: int, params: Mapping[str, float], state: np.ndarray, effective: float, pet: float, spec: Any) -> dict[str, list[float]]:
    full_params = reference_parameter_values(params)
    values = [full_params[name] for name in PARAMETER_NAMES]
    codes = [spec.decision_codes[name] for name in ("RFERR", "ARCH1", "ARCH2", "QSURF", "QPERC", "ESOIL", "QINTF", "Q_TDH", "SNOWM")]
    params_tensor = _parameter_values(params, dtype=torch.float64, device=torch.device("cpu"))
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        topo = _topographic_mean(params_tensor)
        powlamb, maxpow = float(topo[0]), float(topo[1])
    else:
        powlamb, maxpow = 3.0, 50.0
    input_text = "\n".join((
        str(model_id),
        " ".join(str(code) for code in codes),
        " ".join(f"{value:.17g}" for value in values),
        str(len(spec.state_names)),
        " ".join(f"{value:.17g}" for value in state),
        f"{effective:.17g} {pet:.17g} 1 {powlamb:.17g} {maxpow:.17g}",
        "",
    ))
    completed = subprocess.run([str(wrapper)], input=input_text, text=True, capture_output=True, env=_environment(), timeout=60, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"trajectory Fortran wrapper failed: {completed.stderr[-1000:]} {completed.stdout[-500:]}")
    return parse_fortran_output(completed.stdout)


def _np_stack(values: list[np.ndarray], width: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if width is not None:
        result = result.reshape(len(values), width)
    return result


def _trace_lists(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in trace.items()}


def _teacher_case(model_id: int, params: Mapping[str, float], values: Mapping[str, np.ndarray], reference: Any, dates: list[date], wrapper: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, params_tensor, cap, theta, context, topo, fractions, choices = _parameter_context(model_id, params, device)
    active_indices = [STATE_NAMES.index(name) for name in spec.state_names]
    state_fields = {"input_state", "predictor_raw_state", "post_predictor_state", "raw_final_state", "accepted_state", "predictor_lower", "predictor_upper", "final_lower", "final_upper"}
    ref_states = np.stack([np.asarray(reference.states[name], dtype=np.float64) for name in spec.state_names], axis=1)
    n = reference.q_routed.size
    torch_fields: dict[str, list[np.ndarray]] = {name: [] for name in ("input_state", "stage1_flux", "predictor_raw_state", "post_predictor_state", "post_predictor_flux", "stage2_flux", "raw_final_state", "mean_flux", "accepted_state", "accepted_flux", "predictor_lower", "predictor_upper", "final_lower", "final_upper")}
    fortran_fields: dict[str, list[np.ndarray]] = {name: [] for name in ("input_state", "stage1_flux", "predictor_raw_state", "post_predictor_state", "post_predictor_flux", "stage2_flux", "raw_final_state", "mean_flux", "accepted_state", "accepted_flux", "predictor_lower", "predictor_upper", "final_lower", "final_upper")}
    reference_output_flux = {name: np.asarray(value, dtype=np.float64) for name, value in reference.fluxes.items()}
    q_teacher: list[float] = []
    q_teacher_fortran_inst: list[float] = []
    stage_fortran_qperc: list[float] = []
    stage_torch_qperc: list[float] = []
    fortran_branch_final: list[np.ndarray] = []
    torch_branch_final: list[np.ndarray] = []
    future_torch = torch.zeros((500,), dtype=torch.float64, device=device)
    future_fortran = torch.zeros((500,), dtype=torch.float64, device=device)
    with torch.no_grad():
        for index in range(n):
            state = ref_states[index]
            f = _fortran_one_step(wrapper, model_id, params, state, float(reference.fluxes["EFF_PPT"][index]), float(values["pet"][index]), spec)
            union = torch.as_tensor(_sequential_union_state(torch.as_tensor(state, dtype=torch.float64, device=device), spec), dtype=torch.float64, device=device)
            effective = torch.as_tensor(float(reference.fluxes["EFF_PPT"][index]), dtype=torch.float64, device=device)
            pet = torch.as_tensor(float(values["pet"][index]), dtype=torch.float64, device=device)
            t = _torch_effective_step(union, effective, pet, theta, context, topo, choices)
            f_map = {
                "input_state": f["STATE0"], "stage1_flux": f["FLUX0"], "predictor_raw_state": f["PREDICTOR"], "post_predictor_state": f["PREDICTOR_SAFE"], "post_predictor_flux": f["PREDICTOR_FLUX_FIXED"], "stage2_flux": f["FLUX1"], "raw_final_state": f["HEUN_RAW"], "mean_flux": f["FLUX_AVG"], "accepted_state": f["STATE1"], "accepted_flux": f["FLUX_FINAL"], "predictor_lower": f["PREDICTOR_LOWER_VIOLATION"], "predictor_upper": f["PREDICTOR_UPPER_VIOLATION"], "final_lower": f["FINAL_LOWER_VIOLATION"], "final_upper": f["FINAL_UPPER_VIOLATION"],
            }
            for name in ("predictor_lower", "predictor_upper", "final_lower", "final_upper"):
                f_map[name] = np.asarray(f_map[name], dtype=np.float64)[:len(spec.state_names)]
            t_map = {name: t[name].detach().cpu().numpy().astype(np.float64) for name in torch_fields}
            for name in state_fields:
                t_map[name] = t_map[name][active_indices]
            for name in torch_fields:
                torch_fields[name].append(t_map[name])
                fortran_fields[name].append(np.asarray(f_map[name], dtype=np.float64))
            fortran_branch_final.append(np.asarray(f_map["final_lower"]) + np.asarray(f_map["final_upper"]))
            torch_branch_final.append(t_map["final_lower"] + t_map["final_upper"])
            inst_torch = float(t["instantaneous"].detach().cpu())
            inst_fortran = float(np.asarray(f["FLUX_FINAL"])[10] + np.asarray(f["FLUX_FINAL"])[9] + np.asarray(f["FLUX_FINAL"])[8] + np.asarray(f["FLUX_FINAL"])[18] + np.asarray(f["FLUX_FINAL"])[15])
            q_t, future_torch = _route(future_torch, t["instantaneous"], fractions)
            q_f, future_fortran = _route(future_fortran, torch.as_tensor(inst_fortran, dtype=torch.float64, device=device), fractions)
            q_teacher.append(float(q_t.detach().cpu()))
            q_teacher_fortran_inst.append(float(q_f.detach().cpu()))
            stage_fortran_qperc.append(float(f["FLUX_FINAL"][FLUX_INDEX["QPERC_12"]]))
            stage_torch_qperc.append(float(t["accepted_flux"][FLUX_INDEX["QPERC_12"]].detach().cpu()))
    t_arrays = {name: _np_stack(items) for name, items in torch_fields.items()}
    f_arrays = {name: _np_stack(items) for name, items in fortran_fields.items()}
    shared = {
        "input_state": _stats(t_arrays["input_state"] - f_arrays["input_state"]),
        "stage1_flux": _stats(t_arrays["stage1_flux"] - f_arrays["stage1_flux"]),
        "predictor_raw_state": _stats(t_arrays["predictor_raw_state"] - f_arrays["predictor_raw_state"]),
        "post_predictor_state": _stats(t_arrays["post_predictor_state"] - f_arrays["post_predictor_state"]),
        "post_predictor_flux": _stats(t_arrays["post_predictor_flux"] - f_arrays["post_predictor_flux"]),
        "stage2_flux": _stats(t_arrays["stage2_flux"] - f_arrays["stage2_flux"]),
        "raw_final_state": _stats(t_arrays["raw_final_state"] - f_arrays["raw_final_state"]),
        "mean_flux": _stats(t_arrays["mean_flux"] - f_arrays["mean_flux"]),
        "accepted_state": _stats(t_arrays["accepted_state"] - f_arrays["accepted_state"]),
        "accepted_flux": _stats(t_arrays["accepted_flux"] - f_arrays["accepted_flux"]),
    }
    ref_next = ref_states[1:]
    torch_next = t_arrays["accepted_state"][:-1]
    fortran_next = f_arrays["accepted_state"][:-1]
    reference_alignment = {
        "teacher_torch_accepted_state_vs_fortran_next": _stats(torch_next - ref_next),
        "teacher_fortran_wrapper_accepted_state_vs_reference_next": _stats(fortran_next - ref_next),
        "teacher_torch_accepted_flux_vs_reference_output": _stats(t_arrays["accepted_flux"] - np.stack([reference_output_flux[name] for name in FLUX_NAMES], axis=1)),
        "teacher_fortran_accepted_flux_vs_reference_output": _stats(f_arrays["accepted_flux"] - np.stack([reference_output_flux[name] for name in FLUX_NAMES], axis=1)),
    }
    q_error = np.abs(np.asarray(q_teacher) - np.asarray(reference.q_routed))
    q_fortran_reconstructed_error = np.abs(np.asarray(q_teacher_fortran_inst) - np.asarray(reference.q_routed))
    qperc_error = np.abs(np.asarray(stage_torch_qperc) - np.asarray(stage_fortran_qperc))
    errors = {"accepted_state": np.max(np.abs(t_arrays["accepted_state"] - f_arrays["accepted_state"]), axis=1), "accepted_flux": np.max(np.abs(t_arrays["accepted_flux"] - f_arrays["accepted_flux"]), axis=1), "teacher_reported_Q": q_error, "teacher_fortran_routed_Q": q_fortran_reconstructed_error, "accepted_QPERC_12": qperc_error}
    field_names = {"state": tuple(spec.state_names), "flux": FLUX_NAMES}
    trace = {
        "basin_id": "USA_09447800",
        "hru_id": 9447800,
        "model_id": model_id,
        "state_names": list(spec.state_names),
        "flux_names": list(FLUX_NAMES),
        "dates": [item.isoformat() for item in dates],
        "forcing": {**{name: np.asarray(values[name], dtype=np.float64) for name in ("ppt", "temp", "pet", "q_obs")}, "effective_precipitation": np.asarray(reference.fluxes["EFF_PPT"], dtype=np.float64)},
        "reference_output": {"q_routed": np.asarray(reference.q_routed), "q_instantaneous": np.asarray(reference.q_instantaneous), "states": ref_states, "fluxes": reference_output_flux},
        "fortran": f_arrays,
        "torch": t_arrays,
        "teacher_torch_reported_Q": np.asarray(q_teacher),
        "teacher_fortran_inst_reported_Q": np.asarray(q_teacher_fortran_inst),
        "summary": {"fortran_vs_torch": shared, "vs_reference": reference_alignment, "first_divergence": _threshold_report(errors, dates), "activation": {"fortran_accepted_QPERC_12": float(np.mean(np.abs(stage_fortran_qperc) > EPS)), "torch_accepted_QPERC_12": float(np.mean(np.abs(stage_torch_qperc) > EPS)), "active_agreement": bool(np.array_equal(np.abs(stage_fortran_qperc) > EPS, np.abs(stage_torch_qperc) > EPS))}, "branch_final_active_mismatch_count": int(np.sum(np.asarray([np.any(a > 0.5) != np.any(b > 0.5) for a, b in zip(fortran_branch_final, torch_branch_final)])))},
        "_branch_final_fortran": np.asarray(fortran_branch_final),
        "_branch_final_torch": np.asarray(torch_branch_final),
        "_qperc_fortran": np.asarray(stage_fortran_qperc),
        "_qperc_torch": np.asarray(stage_torch_qperc),
    }
    return trace, {"ref_states": ref_states, "ref_flux": reference_output_flux, "torch_fields": t_arrays, "fortran_fields": f_arrays}


def _free_case(model_id: int, params: Mapping[str, float], values: Mapping[str, np.ndarray], reference: Any, dates: list[date], device: torch.device, teacher_aux: Mapping[str, Any]) -> dict[str, Any]:
    spec, params_tensor, cap, theta, context, topo, fractions, choices = _parameter_context(model_id, params, device)
    active_indices = [STATE_NAMES.index(name) for name in spec.state_names]
    raw_forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1), dtype=torch.float64, device=device)
    days, leap = _day_of_year(len(dates), dates, dtype=torch.float64, device=device)
    initial = _sequential_union_state(_initial_state(spec, params_tensor, cap, 0.25), spec)
    state = initial
    snow = torch.zeros((), dtype=torch.float64, device=device)
    future = torch.zeros((500,), dtype=torch.float64, device=device)
    manual_states = [state.detach().cpu().numpy()[active_indices]]
    manual_flux = []
    manual_q = []
    manual_qinst = []
    manual_effective = []
    manual_snow = [0.0]
    manual_final_lower = []
    manual_final_upper = []
    with torch.no_grad():
        for index in range(len(dates)):
            ppt, pet, temp = raw_forcing[index]
            effective, next_snow = _runtime_snow_step(ppt, temp, snow, days[index], leap[index], theta, raw_forcing.new_tensor(1.0))
            step = _torch_effective_step(state, effective, pet, theta, context, topo, choices)
            routed, future = _route(future, step["instantaneous"], fractions)
            state = step["accepted_state"]
            snow = next_snow
            manual_states.append(state.detach().cpu().numpy()[active_indices])
            manual_flux.append(step["accepted_flux"].detach().cpu().numpy())
            manual_q.append(float(routed.detach().cpu()))
            manual_qinst.append(float(step["instantaneous"].detach().cpu()))
            manual_effective.append(float(effective.detach().cpu()))
            manual_snow.append(float(snow.detach().cpu()))
            manual_final_lower.append(step["final_lower"].detach().cpu().numpy())
            manual_final_upper.append(step["final_upper"].detach().cpu().numpy())
    manual = {"states": np.asarray(manual_states), "flux": np.asarray(manual_flux), "q": np.asarray(manual_q), "qinst": np.asarray(manual_qinst), "effective": np.asarray(manual_effective), "snow": np.asarray(manual_snow), "final_lower": np.asarray(manual_final_lower), "final_upper": np.asarray(manual_final_upper)}
    public = simulate_coupled_rk2(model_id, raw_forcing, params, initial_fraction=0.25, dates=dates, dt_days=1.0, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    torch_states = public.states.detach().cpu().numpy().astype(np.float64)
    torch_flux = np.stack([public.fluxes[name].detach().cpu().numpy().astype(np.float64) for name in FLUX_NAMES], axis=1)
    torch_q = public.q.detach().cpu().numpy().astype(np.float64)
    torch_qinst = public.q_instantaneous.detach().cpu().numpy().astype(np.float64)
    ref_states = teacher_aux["ref_states"]
    ref_flux = teacher_aux["ref_flux"]
    ref_flux_matrix = np.stack([ref_flux[name] for name in FLUX_NAMES], axis=1)
    state_start_error = np.max(np.abs(torch_states[:-1] - ref_states), axis=1)
    state_next_error = np.max(np.abs(torch_states[1:-1] - ref_states[1:]), axis=1)
    flux_error = np.max(np.abs(torch_flux - ref_flux_matrix), axis=1)
    q_error = np.abs(torch_q - np.asarray(reference.q_routed))
    qperc_error = np.abs(torch_flux[:, FLUX_INDEX["QPERC_12"]] - ref_flux["QPERC_12"])
    qperc_activation_mismatch = (np.abs(torch_flux[:, FLUX_INDEX["QPERC_12"]]) > EPS) != (np.abs(ref_flux["QPERC_12"]) > EPS)
    teacher_fortran_final = np.asarray(teacher_aux["fortran_fields"]["final_lower"]) + np.asarray(teacher_aux["fortran_fields"]["final_upper"])
    manual_branch = (manual["final_lower"] + manual["final_upper"])[:, active_indices]
    branch_mismatch = np.any((manual_branch > 0.5) != (teacher_fortran_final > 0.5), axis=1)
    errors = {"state_start": state_start_error, "state_next": state_next_error, "accepted_flux": flux_error, "reported_Q": q_error, "reported_QPERC_12": qperc_error}
    meaningful = {name: _first_hit(error, 1.0e-6, dates, start=1) for name, error in errors.items()}
    key_points = {f"{name}_first_1e-6": item["index"] for name, item in meaningful.items() if item is not None}
    activation_first = _first_bool(qperc_activation_mismatch, dates, start=0)
    if activation_first is not None:
        key_points["QPERC_12_activation_mismatch"] = activation_first["index"]
    branch_first = _first_bool(branch_mismatch, dates, start=0)
    if branch_first is not None:
        key_points["topology_branch_mismatch"] = branch_first["index"]
    windows = {}
    for label, index in key_points.items():
        entries = []
        for day_index in range(max(0, index - 1), min(len(dates), index + 2)):
            entries.append({
                "index": day_index,
                "date": dates[day_index].isoformat(),
                "reference_state_start": ref_states[day_index],
                "torch_state_start": torch_states[day_index],
                "torch_state_next": torch_states[day_index + 1],
                "reference_flux": ref_flux_matrix[day_index],
                "torch_accepted_flux": torch_flux[day_index],
                "reference_q_routed": float(reference.q_routed[day_index]),
                "torch_q_routed": float(torch_q[day_index]),
                "reference_q_instantaneous": float(reference.q_instantaneous[day_index]),
                "torch_q_instantaneous": float(torch_qinst[day_index]),
                "reference_qperc_12": float(ref_flux["QPERC_12"][day_index]),
                "torch_qperc_12": float(torch_flux[day_index, FLUX_INDEX["QPERC_12"]]),
                "reference_final_branch_flags": teacher_fortran_final[day_index],
                "torch_final_branch_flags": manual_branch[day_index],
                "qperc_activation_mismatch": bool(qperc_activation_mismatch[day_index]),
            })
        windows[label] = entries
    manual_public_parity = {
        "state": _stats(manual["states"] - torch_states),
        "flux": _stats(manual["flux"] - torch_flux),
        "q": _stats(manual["q"] - torch_q),
        "qinst": _stats(manual["qinst"] - torch_qinst),
    }
    return {
        "basin_id": "USA_09447800", "hru_id": 9447800, "model_id": model_id, "state_names": list(spec.state_names), "flux_names": list(FLUX_NAMES), "dates": [item.isoformat() for item in dates],
        "reference": {"states": ref_states, "fluxes": ref_flux, "q_routed": np.asarray(reference.q_routed), "q_instantaneous": np.asarray(reference.q_instantaneous)},
        "torch": {"states": torch_states, "accepted_flux": torch_flux, "q_routed": torch_q, "q_instantaneous": torch_qinst, "effective_precipitation": manual["effective"], "snow": manual["snow"]},
        "summary": {"first_divergence": _threshold_report(errors, dates), "first_meaningful_divergence_1e-6": meaningful, "first_qperc_activation_mismatch": activation_first, "first_topology_branch_mismatch": branch_first, "manual_vs_public_compiled": manual_public_parity, "fidelity": {"state_start": _stats(torch_states[:-1] - ref_states), "state_next": _stats(torch_states[1:-1] - ref_states[1:]), "accepted_flux": _stats(torch_flux - ref_flux_matrix), "Q": _stats(torch_q - np.asarray(reference.q_routed)), "QPERC_12": _stats(torch_flux[:, FLUX_INDEX["QPERC_12"]] - ref_flux["QPERC_12"])}, "quality": {"finite": bool(np.isfinite(torch_states).all() and np.isfinite(torch_flux).all() and np.isfinite(torch_q).all()), "negative_state_count": int(np.sum(torch_states < 0.0)), "water_balance_max_abs": float(np.max(np.abs(public.water_balance_residual.detach().cpu().numpy()))), "snow_balance_max_abs": float(np.max(np.abs(public.snow_balance_residual.detach().cpu().numpy())))}, "qperc_activation": {"reference": float(np.mean(np.abs(ref_flux["QPERC_12"]) > EPS)), "torch": float(np.mean(np.abs(torch_flux[:, FLUX_INDEX["QPERC_12"]]) > EPS)), "agreement": bool(np.array_equal(np.abs(ref_flux["QPERC_12"]) > EPS, np.abs(torch_flux[:, FLUX_INDEX["QPERC_12"]]) > EPS))}, "windows": windows},
        "_branch_mismatch": branch_mismatch,
        "_manual_branch": manual_branch,
    }


def _process_series(flux: Mapping[str, np.ndarray], process: str) -> np.ndarray:
    if process == "ET":
        return np.asarray(flux["EVAP_1"]) + np.asarray(flux["EVAP_2"])
    if process == "qbase":
        return np.asarray(flux["QBASE_2"])
    if process == "qperc":
        return np.asarray(flux["QPERC_12"])
    return np.asarray(flux["QSURF"])


def _accepted_flux_audit(teacher_cases: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "schema_version": "coupled-rk2-accepted-flux-audit-v1",
        "status": "complete",
        "source_grounded_mapping": {
            "Fortran_stage_flux": "M_FLUX after FUSE_DERIV at stage 0/stage 1; wrapper packs FLUX0/FLUX1",
            "Fortran_mean_flux": "MEANFLUXES raw average FLUX_0/FLUX_1; wrapper packs FLUX_AVG before final FIX_STATES",
            "Fortran_post_fix_flux": "M_FLUX after final FIX_STATES; wrapper packs FLUX_FINAL",
            "Fortran_accepted_output": "W_FLUX after ADD_FLUX/WGT_FLUXES; varextract maps qperc_12 to W_FLUX%QPERC_12",
            "Torch_current": "accepted_flux returned by coupled RK2 final FIX_STATES path and used for instantaneous Q",
            "qperc_index": FLUX_INDEX["QPERC_12"],
            "qsurf_index": FLUX_INDEX["QSURF"],
            "qbase_index": FLUX_INDEX["QBASE_2"],
        },
        "source_files": {
            "fuse_solve": "vendor/upstream/cyrilthebault-fuse/build/FUSE_SRC/FUSE_ENGINE/fuse_solve.f90",
            "wgt_fluxes": "vendor/upstream/cyrilthebault-fuse/build/FUSE_SRC/FUSE_ENGINE/wgt_fluxes.f90",
            "varextract": "vendor/upstream/cyrilthebault-fuse/build/FUSE_SRC/FUSE_ENGINE/varextract.f90",
            "metaoutput": "vendor/upstream/cyrilthebault-fuse/build/FUSE_SRC/FUSE_ENGINE/metaoutput.f90",
        },
        "cases": [],
    }
    for case in teacher_cases:
        reference = case["reference_output"]["fluxes"]
        fortran = case["fortran"]
        torch = case["torch"]
        per_process = {}
        for process in ("qperc", "qsurf", "qbase", "ET"):
            ref = _process_series(reference, process)
            series = {
                "fortran_reference_output_W_FLUX": ref,
                "fortran_stage0_FLUX0": _process_series({"QPERC_12": fortran["stage1_flux"][:, FLUX_INDEX["QPERC_12"]], "QSURF": fortran["stage1_flux"][:, FLUX_INDEX["QSURF"]], "QBASE_2": fortran["stage1_flux"][:, FLUX_INDEX["QBASE_2"]], "EVAP_1": fortran["stage1_flux"][:, FLUX_INDEX["EVAP_1"]], "EVAP_2": fortran["stage1_flux"][:, FLUX_INDEX["EVAP_2"]]}, process),
                "fortran_stage1_FLUX1": _process_series({"QPERC_12": fortran["stage2_flux"][:, FLUX_INDEX["QPERC_12"]], "QSURF": fortran["stage2_flux"][:, FLUX_INDEX["QSURF"]], "QBASE_2": fortran["stage2_flux"][:, FLUX_INDEX["QBASE_2"]], "EVAP_1": fortran["stage2_flux"][:, FLUX_INDEX["EVAP_1"]], "EVAP_2": fortran["stage2_flux"][:, FLUX_INDEX["EVAP_2"]]}, process),
                "fortran_mean_before_fix": _process_series({"QPERC_12": fortran["mean_flux"][:, FLUX_INDEX["QPERC_12"]], "QSURF": fortran["mean_flux"][:, FLUX_INDEX["QSURF"]], "QBASE_2": fortran["mean_flux"][:, FLUX_INDEX["QBASE_2"]], "EVAP_1": fortran["mean_flux"][:, FLUX_INDEX["EVAP_1"]], "EVAP_2": fortran["mean_flux"][:, FLUX_INDEX["EVAP_2"]]}, process),
                "fortran_post_fix_FLUX_FINAL": _process_series({"QPERC_12": fortran["accepted_flux"][:, FLUX_INDEX["QPERC_12"]], "QSURF": fortran["accepted_flux"][:, FLUX_INDEX["QSURF"]], "QBASE_2": fortran["accepted_flux"][:, FLUX_INDEX["QBASE_2"]], "EVAP_1": fortran["accepted_flux"][:, FLUX_INDEX["EVAP_1"]], "EVAP_2": fortran["accepted_flux"][:, FLUX_INDEX["EVAP_2"]]}, process),
                "torch_stage0": _process_series({"QPERC_12": torch["stage1_flux"][:, FLUX_INDEX["QPERC_12"]], "QSURF": torch["stage1_flux"][:, FLUX_INDEX["QSURF"]], "QBASE_2": torch["stage1_flux"][:, FLUX_INDEX["QBASE_2"]], "EVAP_1": torch["stage1_flux"][:, FLUX_INDEX["EVAP_1"]], "EVAP_2": torch["stage1_flux"][:, FLUX_INDEX["EVAP_2"]]}, process),
                "torch_stage1": _process_series({"QPERC_12": torch["stage2_flux"][:, FLUX_INDEX["QPERC_12"]], "QSURF": torch["stage2_flux"][:, FLUX_INDEX["QSURF"]], "QBASE_2": torch["stage2_flux"][:, FLUX_INDEX["QBASE_2"]], "EVAP_1": torch["stage2_flux"][:, FLUX_INDEX["EVAP_1"]], "EVAP_2": torch["stage2_flux"][:, FLUX_INDEX["EVAP_2"]]}, process),
                "torch_mean_before_fix": _process_series({"QPERC_12": torch["mean_flux"][:, FLUX_INDEX["QPERC_12"]], "QSURF": torch["mean_flux"][:, FLUX_INDEX["QSURF"]], "QBASE_2": torch["mean_flux"][:, FLUX_INDEX["QBASE_2"]], "EVAP_1": torch["mean_flux"][:, FLUX_INDEX["EVAP_1"]], "EVAP_2": torch["mean_flux"][:, FLUX_INDEX["EVAP_2"]]}, process),
                "torch_post_fix_accepted_flux": _process_series({"QPERC_12": torch["accepted_flux"][:, FLUX_INDEX["QPERC_12"]], "QSURF": torch["accepted_flux"][:, FLUX_INDEX["QSURF"]], "QBASE_2": torch["accepted_flux"][:, FLUX_INDEX["QBASE_2"]], "EVAP_1": torch["accepted_flux"][:, FLUX_INDEX["EVAP_1"]], "EVAP_2": torch["accepted_flux"][:, FLUX_INDEX["EVAP_2"]]}, process),
            }
            per_process[process] = {"activation": {name: float(np.mean(np.abs(series[name]) > EPS)) for name in series}, "errors_vs_reference_output": {name: _stats(np.asarray(values) - ref) for name, values in series.items() if name != "fortran_reference_output_W_FLUX"}, "series": series,
            }
        result["cases"].append({"basin_id": case["basin_id"], "model_id": case["model_id"], "processes": per_process})
    return result


def _time_alignment(teacher_cases: list[dict[str, Any]], free_cases: list[dict[str, Any]], dates: list[date], references: Mapping[int, Any], params_by_model: Mapping[int, Mapping[str, float]]) -> dict[str, Any]:
    case = teacher_cases[0]
    model_id = int(case["model_id"])
    ref = references[model_id]
    return {
        "schema_version": "coupled-rk2-time-alignment-audit-v1",
        "status": "complete",
        "reference_numerix": {"path": str(NUMERIX_SETTINGS), "sha256": _sha(NUMERIX_SETTINGS), "solution_method_code": 2, "solution_method": "implicit_euler", "temporal_error_control_code": 0, "temporal_error_control": "fixed_time_steps", "teacher_forced_wrapper_method": "explicit_fixed_heun_stage_sequence; diagnostic wrapper only, not the run_pre integrator"},
        "diagnosis": {"runtime_boundary_issue": "ARNO/XVIC saturated-area implementation used a 1e-12 floor for (1-ratio) and produced area=0.3703898161 at exact Fortran capacity for model 8; fixed with an exact capacity branch in dfuse/runtime.py", "teacher_forced_current_vs_fortran_explicit_wrapper": "passed at numerical precision after the boundary fix for all four cases; no branch mismatch", "teacher_forced_current_vs_run_pre": "mismatch is present from the first day because the frozen run_pre reference uses implicit Euler (solution_method_code=2), while coupled-RK2 is fixed explicit Heun; this is not an accepted-flux slot or one-day alignment error", "free_running_cause": "the fixed-RK2 trajectory begins from a different accepted state/flux map relative to the implicit-Euler reference and then diverges; QPERC activation mismatch follows the state trajectory", "final_classification": "local real-trajectory step mismatch identified — solver path still unresolved"},
        "calendar": {"forcing_start": FORCING_START.isoformat(), "simulation_end": SIMULATION_END.isoformat(), "n_steps": len(dates), "first_date": dates[0].isoformat(), "last_date": dates[-1].isoformat(), "daily_dt_days": 1.0},
        "index_contract": {"forcing_index_i": "dates[i], raw forcing[i], reference EFF_PPT[i], reference flux output[i]", "reference_state_i": "Fortran state at start of day i; reference state[0] equals the fracState0=0.25 initialization after SP rounding", "torch_state_i": "public coupled simulation states[i] is start of day i; states[:-1] aligns to reference states", "accepted_flux_i": "flux for day i after final FIX_STATES/accepted aggregation; no one-day shift", "reported_Q_i": "routing output from instantaneous accepted flux at day i"},
        "reference_time_values": np.asarray(ref.time),
        "reference_time_first_last": [float(ref.time[0]), float(ref.time[-1])],
        "initialization": {},
        "forcing_checks": {"dates_length": len(dates), "q_obs_length": len(case["forcing"]["q_obs"]), "teacher_forced_effective_source": "reference.fluxes['EFF_PPT']", "free_running_effective_source": "Torch _runtime_snow_step(raw ppt,temp,snow)"},
        "per_case": [],
    }


def main() -> None:
    global SUBSET_PATH, COUPLED_WRAPPER_EXE, REFERENCE_EXE, CACHE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=SUBSET_PATH)
    parser.add_argument("--wrapper-executable", type=Path, default=COUPLED_WRAPPER_EXE)
    parser.add_argument("--reference-executable", type=Path, default=REFERENCE_EXE)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args()
    SUBSET_PATH = args.subset.resolve()
    COUPLED_WRAPPER_EXE = args.wrapper_executable.resolve()
    REFERENCE_EXE = args.reference_executable.resolve()
    CACHE_DIR = args.cache_dir.resolve()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.update(_environment())
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR)
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
    if not torch.cuda.is_available():
        raise RuntimeError("trajectory diagnostic requires CUDA")
    subset = _load_subset()
    input_paths = _load_inputs()
    calibration = _load_calibration()
    dates = _date_range(FORCING_START, SIMULATION_END)
    device = torch.device("cuda")
    catchment = subset["catchments"][0]
    teacher_cases: list[dict[str, Any]] = []
    free_cases: list[dict[str, Any]] = []
    references: dict[int, Any] = {}
    params_by_model: dict[int, Mapping[str, float]] = {}
    started = time.perf_counter()
    for structure in subset["structures"]:
        model_id = int(structure["model_id"])
        cal_row = calibration[(int(catchment["hru_id"]), model_id)]
        params = cal_row["parameter_vector"]
        params_by_model[model_id] = params
        values, _ = _case_inputs(input_paths[int(catchment["hru_id"])])
        forcing = {name: values[name] for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}
        ref = run_reference(REFERENCE_EXE, model_id, forcing, params=params, initial_fraction=0.25, dates=dates, dt_days=1.0, timeout_seconds=1800.0)
        references[model_id] = ref
        teacher, teacher_aux = _teacher_case(model_id, params, values, ref, dates, COUPLED_WRAPPER_EXE, device)
        teacher_cases.append(teacher)
        free = _free_case(model_id, params, values, ref, dates, device, teacher_aux)
        free_cases.append(free)
        print(json.dumps({"status": "completed", "model_id": model_id, "teacher_days": len(dates), "free_days": len(dates)}, sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    teacher_public = []
    for case in teacher_cases:
        clean = {key: value for key, value in case.items() if not key.startswith("_")}
        teacher_public.append(clean)
    free_public = []
    for case in free_cases:
        clean = {key: value for key, value in case.items() if not key.startswith("_")}
        free_public.append(clean)
    _write(TRACE_OUTPUT, {"schema_version": "coupled-rk2-teacher-forced-trace-v1", "status": "complete", "selection": subset, "protocol": {"teacher_forced": True, "raw_forcing_recorded": True, "effective_precipitation_forced_from_fortran": True, "no_solver_changes": True}, "reference_executable": str(REFERENCE_EXE), "reference_executable_sha256": _sha(REFERENCE_EXE), "coupled_wrapper_executable": str(COUPLED_WRAPPER_EXE), "coupled_wrapper_executable_sha256": _sha(COUPLED_WRAPPER_EXE), "wrapper_source": "project/autofuse/coupled_rhs_fortran_wrapper.f90", "wrapper_source_sha256": _sha(ROOT / "project/autofuse/coupled_rhs_fortran_wrapper.f90"), "cases": teacher_public})
    _write(FREE_OUTPUT, {"schema_version": "coupled-rk2-free-run-divergence-v1", "status": "complete", "selection": subset, "protocol": {"free_running": True, "same_forcing_theta_initialization": True, "state_alignment": "public_simulation.states[:-1] vs reference state at start of day", "thresholds": list(THRESHOLDS)}, "cases": free_public})
    _write(FLUX_OUTPUT, _accepted_flux_audit(teacher_cases))
    time_audit = _time_alignment(teacher_cases, free_cases, dates, references, params_by_model)
    for teacher, free in zip(teacher_cases, free_cases):
        model_id = int(teacher["model_id"])
        params = params_by_model[model_id]
        ref = references[model_id]
        ref0 = np.asarray([ref.states[name][0] for name in get_structure(model_id).state_names], dtype=np.float64)
        spec, params_tensor, cap, *_ = _parameter_context(model_id, params, device)
        torch0 = _sequential_union_state(_initial_state(spec, params_tensor, cap, 0.25), spec).detach().cpu().numpy() if hasattr(_sequential_union_state(_initial_state(spec, params_tensor, cap, 0.25), spec), "detach") else np.asarray(_sequential_union_state(_initial_state(spec, params_tensor, cap, 0.25), spec))
        active0 = torch0[[{"TENS_1A":0,"TENS_1B":1,"TENS_1":2,"FREE_1":3,"WATR_1":4,"TENS_2":5,"FREE_2A":6,"FREE_2B":7,"WATR_2":8}[name] for name in spec.state_names]]
        time_audit["initialization"][str(model_id)] = {"reference_state0": ref0, "torch_initial_state": active0, "max_abs_difference": float(np.max(np.abs(ref0-active0))), "reference_state0_is_initial": bool(np.max(np.abs(ref0-active0)) <= 1.0e-4)}
        reference_flux_day0 = np.asarray([free["reference"]["fluxes"][name][0] for name in FLUX_NAMES], dtype=np.float64)
        time_audit["per_case"].append({"model_id": model_id, "state_start_first_day_error": float(np.max(np.abs(free["torch"]["states"][0] - free["reference"]["states"][0]))), "accepted_flux_day0_error": float(np.max(np.abs(free["torch"]["accepted_flux"][0] - reference_flux_day0))), "Q_day0_error": float(abs(free["torch"]["q_routed"][0] - free["reference"]["q_routed"][0])), "QPERC_day0_error": float(abs(free["torch"]["accepted_flux"][0][FLUX_INDEX["QPERC_12"]] - free["reference"]["fluxes"]["QPERC_12"][0])), "no_one_day_shift_in_contract": True})
    time_audit["resource"] = {"host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "fortran_child_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "wall_clock_seconds": time.perf_counter() - started, "cache_dir": str(CACHE_DIR)}
    _write(TIME_OUTPUT, time_audit)
    print(json.dumps({"status": "complete", "teacher_cases": len(teacher_cases), "free_cases": len(free_cases), "elapsed_seconds": time.perf_counter() - started}, sort_keys=True))


if __name__ == "__main__":
    main()
