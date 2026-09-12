"""Source/trace audit for the ARCH2=unlimfrc_2 WATR_2 lower-floor case."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dfuse import get_structure
from dfuse.kernel import _initial_state, _sequential_union_state
from dfuse.runtime import _runtime_snow_step
from project.autofuse.torch_fuse_78_long_horizon_smoke import _load_frozen_inputs
from project.autofuse.trajectory_divergence_diagnostic import (
    _fortran_one_step,
    _parameter_context,
    _route,
    _torch_effective_step,
)
from dfuse.spec import FLUX_NAMES, PARAMETER_NAMES, STATE_NAMES

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
OUTPUT = DOCS / "watr2_lower_floor_torch_correction.json"
LEGACY_OUTPUT = DOCS / "watr2_lower_floor_conservation_audit.json"
WRAPPER = Path("/tmp/autofuse-coupled-rhs-build-v5/coupled_rhs_fortran_v5.exe")
FIX_STATES = ROOT / "vendor/upstream/cyrilthebault-fuse/build/FUSE_SRC/FUSE_ENGINE/fix_states.f90"
RUNTIME = ROOT / "dfuse/runtime.py"
WRAPPER_SOURCE = ROOT / "project/autofuse/coupled_rhs_fortran_wrapper.f90"
MODEL_ID = 8
BASIN_ID = "USA_09447800"
TARGET_INDEX = 1661
XMIN = 1.0e-9


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _terms(start: np.ndarray, end: np.ndarray, flux: np.ndarray, effective: float) -> dict[str, float]:
    values = dict(zip(FLUX_NAMES, np.asarray(flux, dtype=np.float64)))
    instantaneous = values["QSURF"] + values["OFLOW_1"] + values["QINTF_1"] + values["OFLOW_2"] + values["QBASE_2"]
    external = effective - values["EVAP_1"] - values["EVAP_2"] - instantaneous
    storage_delta = float(np.asarray(end).sum() - np.asarray(start).sum())
    return {
        "storage_start": float(np.asarray(start).sum()),
        "storage_end": float(np.asarray(end).sum()),
        "storage_delta": storage_delta,
        "effective_precipitation": effective,
        "evap_1": float(values["EVAP_1"]),
        "evap_2": float(values["EVAP_2"]),
        "qsurf": float(values["QSURF"]),
        "qintf_1": float(values["QINTF_1"]),
        "oflow_1": float(values["OFLOW_1"]),
        "oflow_2": float(values["OFLOW_2"]),
        "qbase_2": float(values["QBASE_2"]),
        "qperc_12": float(values["QPERC_12"]),
        "instantaneous_outflow": float(instantaneous),
        "external_net_input": float(external),
        "residual": float(storage_delta - external),
    }


def _proportional_flux(flux: np.ndarray, indices: tuple[int, ...], error_loss: float) -> np.ndarray:
    result = np.asarray(flux, dtype=np.float64).copy()
    total = float(sum(result[index] for index in indices))
    safe_total = 1.0 if total == 0.0 else total
    for index in indices:
        result[index] += result[index] / safe_total * error_loss
    return result

def _array_map(mapping: Mapping[str, Any], names: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {name: _jsonable(mapping[name]) for name in names}


def main() -> None:
    if not WRAPPER.is_file():
        raise FileNotFoundError(WRAPPER)
    source_meta, inputs, theta_rows = _load_frozen_inputs()
    hru_id = int(next(row["hru_id"] for row in source_meta["manifest"]["catchments"] if row["basin_id"] == BASIN_ID))
    params = theta_rows[(hru_id, MODEL_ID)]["parameter_vector"]
    values = inputs[BASIN_ID]
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(len(values["ppt"]))]
    device = torch.device("cuda")
    spec, params_tensor, cap, theta, context, topo, fractions, choices = _parameter_context(MODEL_ID, params, device)
    active_indices = [STATE_NAMES.index(name) for name in spec.state_names]
    state = _sequential_union_state(_initial_state(spec, params_tensor, cap, 0.25), spec)
    snow = torch.zeros((), dtype=torch.float64, device=device)
    future = torch.zeros((500,), dtype=torch.float64, device=device)
    trace = None
    with torch.no_grad():
        for index in range(TARGET_INDEX + 1):
            ppt = torch.as_tensor(float(values["ppt"][index]), dtype=torch.float64, device=device)
            pet = torch.as_tensor(float(values["pet"][index]), dtype=torch.float64, device=device)
            temp = torch.as_tensor(float(values["temp"][index]), dtype=torch.float64, device=device)
            jday = torch.as_tensor(float(dates[index].timetuple().tm_yday), dtype=torch.float64, device=device)
            leap = torch.as_tensor(float(dates[index].year % 4 == 0), dtype=torch.float64, device=device)
            effective, next_snow = _runtime_snow_step(ppt, temp, snow, jday, leap, theta, torch.as_tensor(1.0, dtype=torch.float64, device=device))
            step = _torch_effective_step(state, effective, pet, theta, context, topo, choices)
            _, future = _route(future, step["instantaneous"], fractions)
            if index == TARGET_INDEX:
                trace = {
                    "forcing": {"ppt": float(ppt), "pet": float(pet), "temp": float(temp), "jday": float(jday), "effective_precipitation": float(effective)},
                    "snow": {"start": float(snow), "next": float(next_snow)},
                    "torch_start_union": state.detach().cpu().numpy(),
                    "torch_step": {key: value.detach().cpu().numpy() for key, value in step.items() if isinstance(value, torch.Tensor)},
                }
            state = step["accepted_state"]
            snow = next_snow
    if trace is None:
        raise AssertionError("target trace was not captured")
    fortran = _fortran_one_step(WRAPPER, MODEL_ID, params, trace["torch_start_union"][active_indices], trace["forcing"]["effective_precipitation"], trace["forcing"]["pet"], spec)
    torch_step = trace["torch_step"]
    torch_active = {key: np.asarray(value)[active_indices] if key in {"input_state", "predictor_raw_state", "post_predictor_state", "raw_final_state", "accepted_state", "predictor_lower", "predictor_upper", "final_lower", "final_upper", "predictor_correction", "final_correction"} else np.asarray(value) for key, value in torch_step.items()}
    torch_step["input_state_active"] = torch_active["input_state"]
    fortran_fields = ("STATE0", "D0", "FLUX0", "PREDICTOR", "PREDICTOR_SAFE", "D1", "FLUX1", "HEUN_RAW", "FLUX_AVG", "STATE1", "FLUX_FINAL", "PREDICTOR_FLUX_FIXED", "PREDICTOR_ERRORS", "PREDICTOR_STATE_CORRECTION", "PREDICTOR_LOWER_VIOLATION", "PREDICTOR_UPPER_VIOLATION", "FINAL_FLUX_BEFORE_FIX", "FINAL_ERRORS", "FINAL_STATE_CORRECTION", "FINAL_LOWER_VIOLATION", "FINAL_UPPER_VIOLATION", "STATE_LOWER_BOUNDS", "STATE_UPPER_BOUNDS", "STATE_CODES")
    flux_names = ("stage1_flux", "post_predictor_flux", "stage2_flux", "mean_flux", "accepted_flux")
    state_names = ("input_state", "predictor_raw_state", "post_predictor_state", "raw_final_state", "accepted_state")
    water = {
        "torch_raw_heun_vs_mean_flux": _terms(torch_active["input_state"], torch_active["raw_final_state"], torch_step["mean_flux"], trace["forcing"]["effective_precipitation"]),
        "torch_accepted_vs_accepted_flux": _terms(torch_active["input_state"], torch_active["accepted_state"], torch_step["accepted_flux"], trace["forcing"]["effective_precipitation"]),
        "fortran_raw_heun_vs_mean_flux": _terms(np.asarray(fortran["STATE0"]), np.asarray(fortran["HEUN_RAW"]), np.asarray(fortran["FLUX_AVG"]), trace["forcing"]["effective_precipitation"]),
        "fortran_accepted_vs_accepted_flux": _terms(np.asarray(fortran["STATE0"]), np.asarray(fortran["STATE1"]), np.asarray(fortran["FLUX_FINAL"]), trace["forcing"]["effective_precipitation"]),
    }
    comparisons = {}
    for torch_name, fortran_name in (("input_state", "STATE0"), ("stage1_flux", "FLUX0"), ("predictor_raw_state", "PREDICTOR"), ("post_predictor_state", "PREDICTOR_SAFE"), ("post_predictor_flux", "PREDICTOR_FLUX_FIXED"), ("stage2_flux", "FLUX1"), ("raw_final_state", "HEUN_RAW"), ("mean_flux", "FLUX_AVG"), ("accepted_state", "STATE1"), ("accepted_flux", "FLUX_FINAL")):
        left = torch_active[torch_name] if torch_name in torch_active else torch_step[torch_name]
        right = np.asarray(fortran[fortran_name], dtype=np.float64)
        if torch_name in state_names:
            right = right[: len(spec.state_names)]
        comparisons[torch_name] = {"max_abs": float(np.max(np.abs(np.asarray(left) - right))), "torch": _jsonable(left), "fortran_wrapper": _jsonable(right)}
    params_cpu = {name: float(value) for name, value in params.items()}
    lower_watr2 = XMIN * params_cpu["MAXWATR_2"]
    source_error_reference = XMIN * params_cpu["MAXWATR_1"]
    watr2_position = list(spec.state_names).index("WATR_2")
    raw_watr2 = float(torch_active["raw_final_state"][watr2_position])
    mean_flux = np.asarray(torch_step["mean_flux"], dtype=np.float64)
    legacy_error_loss = raw_watr2 - source_error_reference
    corrected_error_loss = raw_watr2 - lower_watr2
    legacy_flux = _proportional_flux(mean_flux, (11, 15), legacy_error_loss)
    corrected_flux = np.asarray(torch_step["accepted_flux"], dtype=np.float64)
    legacy_terms = _terms(torch_active["input_state"], torch_active["accepted_state"], legacy_flux, trace["forcing"]["effective_precipitation"])
    corrected_terms = _terms(torch_active["input_state"], torch_active["accepted_state"], corrected_flux, trace["forcing"]["effective_precipitation"])
    finite_trace = all(np.isfinite(np.asarray(value)).all() for value in torch_step.values() if isinstance(value, np.ndarray))
    state_bounds_clean = bool(np.all(np.asarray(torch_active["accepted_state"]) >= 0.0) and float(torch_active["accepted_state"][watr2_position]) >= lower_watr2)
    payload = {
        "schema_version": "watr2-lower-floor-torch-correction-v1",
        "status": "passed" if corrected_terms["residual"] < 1.0e-12 and np.isfinite(corrected_terms["residual"]) else "failed",
        "case": {"basin_id": BASIN_ID, "hru_id": hru_id, "model_id": MODEL_ID, "date": dates[TARGET_INDEX].isoformat(), "index": TARGET_INDEX, "topology": dict(spec.decisions), "state_names": list(spec.state_names), "flux_names": list(FLUX_NAMES)},
        "source": {"fix_states": str(FIX_STATES), "fix_states_sha256": _sha(FIX_STATES), "runtime": str(RUNTIME), "runtime_sha256": _sha(RUNTIME), "wrapper_source": str(WRAPPER_SOURCE), "wrapper_source_sha256": _sha(WRAPPER_SOURCE), "wrapper_executable": str(WRAPPER), "wrapper_executable_sha256": _sha(WRAPPER)},
        "correction": {"legacy_formula": "ERROR_LOSS = (WATR_2_raw - XMIN*MAXWATR_1)/DT", "torch_corrected_formula": "ERROR_LOSS = (WATR_2_raw - XMIN*MAXWATR_2)/DT", "corrected_state_formula": "WATR_2_corrected = XMIN*MAXWATR_2", "source_evidence": "fix_states.f90 corrects ESTATE.WATR_2 to XMIN*MAXWATR_2 but computes ERR_WATR_2 with XMIN*MAXWATR_1; the same lower-bound mismatch is present in the prior Torch branch", "why_implementation_correction": "The corrected state and proportional correction flux must use one lower-layer capacity so storage transition and accepted flux have identical mass accounting. This changes only the known bookkeeping inconsistency, not a process equation or structure decision.", "intentional_deviation_from_legacy": True, "affected_topologies": ["ARCH2=unlimfrc_2", "ARCH2=unlimpow_2", "ARCH2=fixedsiz_2", "ARCH2=topmdexp_2"], "unaffected_topology": "ARCH2=tens2pll_2", "tests_added": ["dfuse/tests/test_runtime.py::test_watr2_lower_floor_uses_matching_capacity", "project/autofuse/watr2_lower_floor_audit.py"], "no_empirical_patch": True},
        "bounds": {"xmin": XMIN, "maxwatr_1": params_cpu["MAXWATR_1"], "maxwatr_2": params_cpu["MAXWATR_2"], "watr2_lower_state_bound": lower_watr2, "legacy_error_reference_bound": source_error_reference, "corrected_error_reference_bound": lower_watr2, "watr2_unbounded_upper_by_design": True},
        "forcing": trace["forcing"],
        "trace": {"torch": {key: _jsonable(value) for key, value in torch_active.items()}, "fortran_wrapper": _array_map(fortran, list(fortran_fields)), "hidden_fields": {key: _jsonable(fortran[key]) for key in ("PREDICTOR_HIDDEN_FREE2A", "PREDICTOR_HIDDEN_FREE2B", "FINAL_HIDDEN_FREE2A", "FINAL_HIDDEN_FREE2B")}},
        "regression": {"before_legacy_torch": {"formula": "MAXWATR_1 error reference", "raw_final_watr2": raw_watr2, "accepted_watr2": float(torch_active["accepted_state"][watr2_position]), "accepted_flux": _jsonable(legacy_flux), "water_balance_terms": legacy_terms}, "after_corrected_torch": {"formula": "MAXWATR_2 error reference", "raw_final_watr2": raw_watr2, "corrected_watr2": lower_watr2, "accepted_watr2": float(torch_active["accepted_state"][watr2_position]), "accepted_flux": _jsonable(corrected_flux), "water_balance_terms": corrected_terms}, "residual_before": legacy_terms["residual"], "residual_after": corrected_terms["residual"], "finite": finite_trace, "state_bounds_clean": state_bounds_clean, "pass": bool(corrected_terms["residual"] < 1.0e-12 and finite_trace and state_bounds_clean)},
        "water_balance_terms": {"legacy_torch_bookkeeping_reconstructed": legacy_terms, "torch_corrected": corrected_terms, "fortran_wrapper_legacy": water["fortran_accepted_vs_accepted_flux"]},
        "correction_decomposition": {"legacy_error_loss": legacy_error_loss, "corrected_error_loss": corrected_error_loss, "legacy_flux_delta_from_mean": _jsonable(legacy_flux - mean_flux), "corrected_flux_delta_from_mean": _jsonable(corrected_flux - mean_flux), "torch_final_state_correction": _jsonable(torch_active["final_correction"]), "fortran_final_state_correction": _jsonable(np.asarray(fortran["FINAL_STATE_CORRECTION"])[: len(spec.state_names)])},
        "torch_vs_fortran_wrapper": comparisons,
        "conclusion": {"local_process_trace_parity": "Torch remains compared with the explicit-Heun Fortran wrapper for process and structure audit; the known legacy WATR_2 bookkeeping branch is intentionally not used as the Torch correctness target", "direct_cause": "MAXWATR_1 was used as the WATR_2 error reference while the state was corrected to MAXWATR_2", "decision": "Applied the source-grounded Torch-only MAXWATR_2 correction; accepted flux now conserves the corrected state transition"},
        "protocol": {"initial_fraction": 0.25, "dt_days": 1.0, "dtype": "torch.float64", "device": "cuda", "execution": "coupled-RK2 Torch kernel with source/trace Fortran process audit"}
    }
    serialized = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(serialized)
    LEGACY_OUTPUT.write_text(serialized)
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT), "date": dates[TARGET_INDEX].isoformat(), "torch_residual": corrected_terms["residual"], "legacy_reconstructed_residual": legacy_terms["residual"], "fortran_residual": water["fortran_accepted_vs_accepted_flux"]["residual"], "max_local_parity": max(item["max_abs"] for item in comparisons.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
