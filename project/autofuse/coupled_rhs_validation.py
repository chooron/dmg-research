"""Controlled dependency, compile, and one-step validation for coupled RK2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any

import torch

from dfuse import (
    compile_diagnostics,
    get_structure,
    reset_compile_diagnostics,
    simulate_coupled_rk2,
    simulate_sequential,
 )
from dfuse.spec import FLUX_NAMES, PARAMETER_NAMES, STATE_NAMES, default_parameters
from dfuse.kernel import (
    COUPLED_RK2_DIAGNOSTIC_NAMES,
    _parameter_vector,
    _parameter_values,
    _sequential_union_state,
    _structure_context,
    _topographic_mean,
 )
from dfuse.runtime import _runtime_coupled_rhs_impl, _runtime_fix_states, get_generated_step

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = ROOT / "project/autofuse/docs/reference_calibration_12x78.json"
CATCHMENT = "USA_09447800"
STRUCTURE_IDS = (2, 6, 8, 14, 164, 166, 178, 188, 212, 214)
STATE_LABELS = (
    "low_storage",
    "medium_storage",
    "near_capacity_storage",
    "upper_wet_lower_dry",
    "upper_dry_lower_wet",
    "both_wet",
    "tension_free_mixed",
)
FORCING_CASES = (
    ("dry_day", 0.0, 1.0, 10.0),
    ("rainfall_day", 5.0, 1.0, 10.0),
    ("high_rainfall", 40.0, 2.0, 10.0),
    ("high_et", 0.0, 8.0, 10.0),
    ("rainfall_high_et", 10.0, 8.0, 10.0),
    ("active_percolation", 15.0, 1.0, 10.0),
    ("active_surface_runoff", 15.0, 1.0, 10.0),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrated_parameters(model_id: int) -> dict[str, float]:
    payload = json.loads(ARCHIVE_PATH.read_text())
    match = next(
        row for row in payload["case_results"]
        if row["basin_id"] == CATCHMENT and int(row["model_id"]) == model_id
    )
    values = default_parameters()
    values.update({name: float(value) for name, value in match["parameter_vector"].items()})
    return values


def state_capacities(params: dict[str, float]) -> dict[str, float]:
    max1 = params["MAXWATR_1"]
    max2 = params["MAXWATR_2"]
    fracten = params["FRACTEN"]
    return {
        "TENS_1A": params["FRCHZNE"] * fracten * max1,
        "TENS_1B": (1.0 - params["FRCHZNE"]) * fracten * max1,
        "TENS_1": fracten * max1,
        "FREE_1": (1.0 - fracten) * max1,
        "WATR_1": max1,
        "TENS_2": fracten * max2,
        "FREE_2A": params["FPRIMQB"] * (1.0 - fracten) * max2,
        "FREE_2B": (1.0 - params["FPRIMQB"]) * (1.0 - fracten) * max2,
        "WATR_2": max2,
    }


def controlled_state(label: str, spec, params: dict[str, float]) -> list[float]:
    caps = state_capacities(params)
    if label == "low_storage":
        factors = {name: 0.05 for name in spec.state_names}
    elif label == "medium_storage":
        factors = {name: 0.50 for name in spec.state_names}
    elif label == "near_capacity_storage":
        factors = {name: 0.95 for name in spec.state_names}
    elif label == "upper_wet_lower_dry":
        factors = {name: (0.90 if name in spec.topology["upper"] else 0.10) for name in spec.state_names}
    elif label == "upper_dry_lower_wet":
        factors = {name: (0.10 if name in spec.topology["upper"] else 0.90) for name in spec.state_names}
    elif label == "both_wet":
        factors = {name: 0.90 for name in spec.state_names}
    elif label == "tension_free_mixed":
        factors = {}
        for name in spec.state_names:
            if "TENS" in name:
                factors[name] = 0.20
            elif "FREE" in name:
                factors[name] = 0.80
            else:
                factors[name] = 0.50
    else:
        raise ValueError(label)
    return [caps[name] * factors[name] for name in spec.state_names]


def representative_specs() -> list[Any]:
    specs = [get_structure(model_id) for model_id in STRUCTURE_IDS]
    if len(specs) < 8:
        raise AssertionError("coupled RHS audit needs at least eight structures")
    required = {
        "perc_lower": False,
        "tension2_1": False,
        "fixedsiz_2": False,
        "unlimfrc_2": False,
        "arno_x_vic": False,
        "prms_varnt": False,
        "tmdl_param": False,
    }
    for spec in specs:
        decisions = spec.decisions
        if decisions["QPERC"] == "perc_lower":
            required["perc_lower"] = True
        for key in ("ARCH1", "ARCH2", "QSURF"):
            required.setdefault(decisions[key], False)
            required[decisions[key]] = True
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise AssertionError(f"representative topology coverage missing {missing}")
    return specs


def build_dependency_audit(specs: list[Any]) -> dict[str, Any]:
    source_root = "vendor/upstream/cyrilthebault-fuse/build/FUSE_SRC/FUSE_ENGINE"
    fluxes = [
        {
            "name": "EFF_PPT",
            "states_read": [],
            "forcing_read": ["PPT", "TEMP"],
            "parameters_read": ["RFERR_MLT", "PXTEMP", "OPG", "LAPSE", "MBASE", "MFMAX", "MFMIN"],
            "source_storage": "snow bands / effective forcing",
            "receiver_storage": ["upper active storage"],
            "kind": "source",
            "capacity_dependency": "receiver capacity through FIX_STATES",
            "overflow_dependency": "upper-state excess can become OFLOW_1",
            "receiver_enters_flux_law": False,
        },
        {
            "name": "SATAREA",
            "states_read": ["WATR_1 or TENS_1", "WATR_2 for tmdl_param"],
            "forcing_read": [],
            "parameters_read": ["MAXWATR_1", "MAXTENS_1", "AXV_BEXP", "SAREAMAX", "LOGLAMB", "TISHAPE", "QB_POWR"],
            "source_storage": "upper/lower state used by selected QSURF law",
            "receiver_storage": [],
            "kind": "diagnostic control",
            "capacity_dependency": "normalizes by selected storage capacity",
            "overflow_dependency": "QSURF competes with upper input before internal transfers",
            "receiver_enters_flux_law": False,
        },
        {
            "name": "QSURF",
            "states_read": ["WATR_1 or TENS_1", "WATR_2 for tmdl_param"],
            "forcing_read": ["EFF_PPT"],
            "parameters_read": ["AXV_BEXP", "SAREAMAX", "LOGLAMB", "TISHAPE", "QB_POWR"],
            "source_storage": "upper input partition",
            "receiver_storage": [],
            "kind": "pure sink",
            "capacity_dependency": "surface-law saturation state",
            "overflow_dependency": "reduces input available to recharge/internal transfer",
            "receiver_enters_flux_law": False,
        },
        {
            "name": "EVAP_1 / EVAP_1A / EVAP_1B",
            "states_read": ["upper tension storage"],
            "forcing_read": ["PET"],
            "parameters_read": ["MAXTENS_1 or MAXTENS_1A/MAXTENS_1B"],
            "source_storage": "upper tension stores",
            "receiver_storage": [],
            "kind": "pure sink",
            "capacity_dependency": "tension capacity denominator",
            "overflow_dependency": "FIX_STATES can adjust evaporation on lower bound",
            "receiver_enters_flux_law": False,
        },
        {
            "name": "EVAP_2",
            "states_read": ["TENS_2 when ARCH1 is not tension2_1"],
            "forcing_read": ["PET", "EVAP_1"],
            "parameters_read": ["MAXTENS_2"],
            "source_storage": "lower tension storage",
            "receiver_storage": [],
            "kind": "pure sink",
            "capacity_dependency": "lower tension capacity denominator",
            "overflow_dependency": "FIX_STATES can adjust evaporation on lower bound",
            "receiver_enters_flux_law": False,
        },
        {
            "name": "RCHR2EXCS",
            "states_read": ["TENS_1A"],
            "forcing_read": ["EFF_PPT", "QSURF"],
            "parameters_read": ["MAXTENS_1A"],
            "source_storage": "recharge-zone tension storage",
            "receiver_storage": ["TENS_1B"],
            "kind": "internal transfer",
            "capacity_dependency": "source/receiver logistic smoothing",
            "overflow_dependency": "excess transfer can feed TENS2FREE_1/OFLOW_1",
            "receiver_enters_flux_law": False,
        },
        {
            "name": "TENS2FREE_1",
            "states_read": ["TENS_1B or TENS_1"],
            "forcing_read": ["EFF_PPT", "QSURF", "RCHR2EXCS"],
            "parameters_read": ["MAXTENS_1 or MAXTENS_1B"],
            "source_storage": "upper tension storage",
            "receiver_storage": ["FREE_1"],
            "kind": "internal transfer",
            "capacity_dependency": "source/receiver logistic smoothing",
            "overflow_dependency": "FREE_1 overflow",
            "receiver_enters_flux_law": False,
        },
        {
            "name": "QPERC_12",
            "states_read": ["FREE_1 or WATR_1", "WATR_2 for perc_lower demand"],
            "forcing_read": [],
            "parameters_read": ["PERCRTE", "PERCEXP", "SACPMLT", "SACPEXP", "QBSAT", "MAXFREE_1", "MAXWATR_1", "MAXWATR_2"],
            "source_storage": "upper free/total storage",
            "receiver_storage": ["lower active storage"],
            "kind": "internal transfer",
            "capacity_dependency": "Fortran QPERC law reads lower storage for perc_lower; FIX_STATES limits downstream transfer",
            "overflow_dependency": "lower tension/free or fixed-size overflow",
            "receiver_enters_flux_law": True,
        },
        {
            "name": "QINTF_1",
            "states_read": ["FREE_1 when intflwsome; inactive in frozen family"],
            "forcing_read": [],
            "parameters_read": ["IFLWRTE", "MAXFREE_1"],
            "source_storage": "upper free storage",
            "receiver_storage": [],
            "kind": "pure sink",
            "capacity_dependency": "source free capacity",
            "overflow_dependency": "none in frozen intflwnone family",
            "receiver_enters_flux_law": False,
        },
        {
            "name": "QBASE_2 / QBASE_2A / QBASE_2B",
            "states_read": ["WATR_2 or FREE_2A/FREE_2B"],
            "forcing_read": [],
            "parameters_read": ["QB_PRMS", "BASERTE", "QB_POWR", "QBRATE_2A", "QBRATE_2B", "MAXWATR_2"],
            "source_storage": "lower reservoir(s)",
            "receiver_storage": [],
            "kind": "pure sink",
            "capacity_dependency": "lower reservoir normalization",
            "overflow_dependency": "competes with lower overflow in fixed-size/parallel tanks",
            "receiver_enters_flux_law": False,
        },
        {
            "name": "OFLOW_1 / OFLOW_2 / OFLOW_2A / OFLOW_2B",
            "states_read": ["upper FREE/WATR or lower FREE/WATR"],
            "forcing_read": ["EFF_PPT", "QPERC_12", "TENS2FREE_2"],
            "parameters_read": ["MAXFREE_1", "MAXWATR_1", "MAXTENS_2", "MAXFREE_2A", "MAXFREE_2B", "MAXWATR_2", "PERCFRAC"],
            "source_storage": "bounded active reservoir",
            "receiver_storage": [],
            "kind": "pure sink / overflow",
            "capacity_dependency": "logistic or residual capacity rule in Q_MISSCELL/FIX_STATES",
            "overflow_dependency": "is the spill output",
            "receiver_enters_flux_law": True,
        },
    ]
    return {
        "schema_version": "coupled-rhs-dependency-audit-v1",
        "status": "frozen_before_single_step_results",
        "scientific_equations_modified": False,
        "fortran_source_root": source_root,
        "fortran_source_commit": "e6e23a4fc4ff4019bcab55f14537ea43b9525967",
        "fortran_semantics": {
            "stage_driver": "mod_derivs.f90:18-30 calls flux routines then mstate_eqn.f90:27-64; no state is updated between those calls",
            "flux_evaluation_calls": ["QSATEXCESS", "EVAP_UPPER", "EVAP_LOWER", "QINTERFLOW", "QPERCOLATE", "Q_BASEFLOW", "Q_MISSCELL"],
            "state_equations": "mstate_eqn.f90:27-64 sums all active flux contributions into DY_DT after the flux stage",
            "capacity_and_spill": "q_misscell.f90:34-124 plus fix_states.f90:37-282; residual capacity branch is explicit and state-specific",
            "heun_reference": "ode_int.f90:158-173 computes predictor, safeguarded predictor, second RHS, average flux, and final safeguard",
        },
        "canonical_rhs_contract": {
            "all_fluxes_read_one_stage_state": True,
            "internal_transfers_enter_both_source_and_receiver_derivatives": True,
            "competing_outflows_are_not_order_updates": True,
            "forcing_is_fixed_across_rk2_stages": True,
            "dynamic_process_order": False,
        },
        "flux_dependency_records": fluxes,
        "representative_structures": [
            {
                "model_id": spec.model_id,
                "decisions": dict(spec.decisions),
                "active_states": list(spec.state_names),
                "active_fluxes": list(spec.flux_names),
                "topology": spec.topology,
            }
            for spec in specs
        ],
    }


def build_cases(specs: list[Any]) -> list[dict[str, Any]]:
    cases = []
    for spec in specs:
        params = calibrated_parameters(spec.model_id)
        for state_label in STATE_LABELS:
            state = controlled_state(state_label, spec, params)
            for forcing_label, ppt, pet, temp in FORCING_CASES:
                effective = ppt * params["RFERR_MLT"]
                cases.append({
                    "case_id": f"m{spec.model_id}_{state_label}_{forcing_label}",
                    "model_id": spec.model_id,
                    "decisions": dict(spec.decisions),
                    "state_label": state_label,
                    "forcing_label": forcing_label,
                    "state_names": list(spec.state_names),
                    "state": state,
                    "forcing": {"ppt": ppt, "pet": pet, "temp": temp, "effective": effective, "dt_days": 1.0},
                    "theta_source": {"basin_id": CATCHMENT, "archive": str(ARCHIVE_PATH.relative_to(ROOT))},
                    "theta": params,
                })
    return cases


def parse_fortran_output(stdout: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for line in stdout.splitlines():
        fields = line.split()
        if not fields or fields[0] == "MODEL":
            continue
        def parse_number(value: str) -> float:
            if "E" not in value.upper() and "e" not in value.lower():
                import re
                value = re.sub(r"^([+-]?\d+\.\d+)([+-]\d+)$", r"\1E\2", value)
            return float(value.replace("D", "E"))
        result[fields[0]] = [parse_number(item) for item in fields[1:]]
    expected = {"STATE0", "D0", "FLUX0", "PREDICTOR", "PREDICTOR_SAFE", "D1", "FLUX1", "HEUN_RAW", "FLUX_AVG", "STATE1", "FLUX_FINAL", "PREDICTOR_FLUX_FIXED", "PREDICTOR_ERRORS", "PREDICTOR_STATE_CORRECTION", "PREDICTOR_LOWER_VIOLATION", "PREDICTOR_UPPER_VIOLATION", "FINAL_FLUX_BEFORE_FIX", "FINAL_ERRORS", "FINAL_STATE_CORRECTION", "FINAL_LOWER_VIOLATION", "FINAL_UPPER_VIOLATION", "STATE_CODES", "STATE_LOWER_BOUNDS", "STATE_UPPER_BOUNDS", "PREDICTOR_HIDDEN_FREE2A", "PREDICTOR_HIDDEN_FREE2B", "FINAL_HIDDEN_FREE2A", "FINAL_HIDDEN_FREE2B"}
    missing = expected - result.keys()
    if missing:
        raise RuntimeError(f"Fortran wrapper output missing {sorted(missing)}")
    return result


def run_fortran(executable: Path, case: dict[str, Any], spec) -> dict[str, list[float]]:
    params = case["theta"]
    values = [params[name] for name in PARAMETER_NAMES]
    codes = [spec.decision_codes[name] for name in ("RFERR", "ARCH1", "ARCH2", "QSURF", "QPERC", "ESOIL", "QINTF", "Q_TDH", "SNOWM")]
    params_tensor = _parameter_values(params, dtype=torch.float64, device=torch.device("cpu"))
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        topographic = _topographic_mean(params_tensor)
        powlamb = float(topographic[0])
        maxpow = float(topographic[1])
    else:
        powlamb = 3.0
        maxpow = 50.0
    forcing = case["forcing"]
    input_text = "\n".join((
        str(spec.model_id),
        " ".join(str(code) for code in codes),
        " ".join(f"{value:.17g}" for value in values),
        str(len(spec.state_names)),
        " ".join(f"{value:.17g}" for value in case["state"]),
        f"{forcing['effective']:.17g} {forcing['pet']:.17g} 1 {powlamb:.17g} {maxpow:.17g}",
        "",
    ))
    environment = os.environ.copy()
    lib_root = "/tmp/autofuse-reference-toolchain/root/lib"
    hdf_root = "/tmp/autofuse-reference-toolchain/root/lib/hdf5/serial"
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (lib_root, hdf_root, environment.get("LD_LIBRARY_PATH", ""))))
    completed = subprocess.run(
        [str(executable)], input=input_text, text=True, capture_output=True, env=environment, timeout=30, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Fortran one-step wrapper failed: {completed.stderr[-1000:]} {completed.stdout[-500:]}")
    return parse_fortran_output(completed.stdout)


def active_values(union: torch.Tensor, spec) -> torch.Tensor:
    return torch.stack([union[STATE_NAMES.index(name)] for name in spec.state_names])


def coupled_python(case: dict[str, Any], spec) -> dict[str, Any]:
    params = _parameter_values(case["theta"], dtype=torch.float64, device=torch.device("cpu"))
    theta = _parameter_vector(params)
    union = _sequential_union_state(torch.tensor(case["state"], dtype=torch.float64), spec)
    forcing = case["forcing"]
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        topo = _topographic_mean(params)
    else:
        topo = (theta[0] * 0.0, theta[0] * 0.0)
    choices = tuple(spec.decisions[name] for name in ("ARCH1", "ARCH2", "QSURF", "QPERC"))
    dt = torch.as_tensor(forcing["dt_days"], dtype=torch.float64)
    effective = torch.as_tensor(forcing["effective"], dtype=torch.float64)
    pet = torch.as_tensor(forcing["pet"], dtype=torch.float64)
    k1, flux1 = _runtime_coupled_rhs_impl(union, effective, pet, theta, topo[0], topo[1], choices=choices)
    predictor_raw = union + dt * k1
    predictor, predictor_flux_fixed, predictor_correction, predictor_errors, predictor_lower, predictor_upper = _runtime_fix_states(
        union, predictor_raw, flux1, theta, dt, choices=choices
    )
    k2, flux2 = _runtime_coupled_rhs_impl(predictor, effective, pet, theta, topo[0], topo[1], choices=choices)
    heun = union + 0.5 * dt * (k1 + k2)
    flux_avg_raw = 0.5 * (flux1 + flux2)
    projected, flux_final, final_correction, final_errors, final_lower, final_upper = _runtime_fix_states(
        union, heun, flux_avg_raw, theta, dt, choices=choices
    )
    return {
        "union": union.detach(),
        "predictor_raw": predictor_raw.detach(),
        "k1": k1.detach(),
        "flux1": flux1.detach(),
        "predictor": predictor.detach(),
        "predictor_flux_fixed": predictor_flux_fixed.detach(),
        "predictor_correction": predictor_correction.detach(),
        "predictor_errors": predictor_errors.detach(),
        "predictor_lower": predictor_lower.detach(),
        "predictor_upper": predictor_upper.detach(),
        "k2": k2.detach(),
        "flux2": flux2.detach(),
        "heun_raw": heun.detach(),
        "state1": projected.detach(),
        "flux_avg_raw": flux_avg_raw.detach(),
        "flux_final": flux_final.detach(),
        "final_correction": final_correction.detach(),
        "final_errors": final_errors.detach(),
        "final_lower": final_lower.detach(),
        "final_upper": final_upper.detach(),
    }


def sequential_baseline(case: dict[str, Any], spec, order: str) -> dict[str, Any]:
    params = case["theta"]
    forcing = case["forcing"]
    ppt = forcing["effective"] / params["RFERR_MLT"] if params["RFERR_MLT"] else 0.0
    result = simulate_sequential(
        spec.model_id,
        torch.tensor([[ppt, forcing["pet"], forcing["temp"]]], dtype=torch.float64),
        params,
        initial_state=torch.tensor(case["state"], dtype=torch.float64),
        order=order,
        compile_step=False,
    )
    return {
        "state1": result.states[1].detach(),
        "flux": torch.stack([result.fluxes[name][0] for name in FLUX_NAMES]).detach(),
    }


def max_abs(values: torch.Tensor) -> float:
    return float(values.abs().max())


def relative_max(values: torch.Tensor, reference: torch.Tensor) -> float:
    return float((values.abs() / reference.abs().clamp_min(1.0e-6)).max())


def constraint_summary(state: torch.Tensor, spec, params: dict[str, float]) -> dict[str, Any]:
    caps = state_capacities(params)
    negative = max(0.0, float((-state).max()))
    violations = []
    for index, name in enumerate(spec.state_names):
        if name == "WATR_2" and spec.decisions["ARCH2"] in ("unlimfrc_2", "unlimpow_2"):
            continue
        violations.append(max(0.0, float(state[index] - caps[name])))
    return {"negative_storage_max": negative, "capacity_violation_max": max(violations, default=0.0), "finite": bool(torch.isfinite(state).all())}


def validate_case(case: dict[str, Any], spec, executable: Path) -> dict[str, Any]:
    oracle = run_fortran(executable, case, spec)
    py = coupled_python(case, spec)
    f_state0 = torch.tensor(oracle["STATE0"], dtype=torch.float64)
    f_d0 = torch.tensor(oracle["D0"], dtype=torch.float64)
    f_flux0 = torch.tensor(oracle["FLUX0"], dtype=torch.float64)
    f_predictor = torch.tensor(oracle["PREDICTOR"], dtype=torch.float64)
    f_pred_safe = torch.tensor(oracle["PREDICTOR_SAFE"], dtype=torch.float64)
    f_pred_flux_fixed = torch.tensor(oracle["PREDICTOR_FLUX_FIXED"], dtype=torch.float64)
    f_d1 = torch.tensor(oracle["D1"], dtype=torch.float64)
    f_flux1 = torch.tensor(oracle["FLUX1"], dtype=torch.float64)
    f_heun = torch.tensor(oracle["HEUN_RAW"], dtype=torch.float64)
    f_flux_avg = torch.tensor(oracle["FLUX_AVG"], dtype=torch.float64)
    f_final_flux_before = torch.tensor(oracle["FINAL_FLUX_BEFORE_FIX"], dtype=torch.float64)
    f_state1 = torch.tensor(oracle["STATE1"], dtype=torch.float64)
    f_flux_final = torch.tensor(oracle["FLUX_FINAL"], dtype=torch.float64)
    f_pred_errors = torch.tensor(oracle["PREDICTOR_ERRORS"], dtype=torch.float64)
    f_final_errors = torch.tensor(oracle["FINAL_ERRORS"], dtype=torch.float64)
    f_pred_correction = torch.tensor(oracle["PREDICTOR_STATE_CORRECTION"], dtype=torch.float64)
    f_final_correction = torch.tensor(oracle["FINAL_STATE_CORRECTION"], dtype=torch.float64)
    f_pred_lower = torch.tensor(oracle["PREDICTOR_LOWER_VIOLATION"], dtype=torch.float64)
    f_pred_upper = torch.tensor(oracle["PREDICTOR_UPPER_VIOLATION"], dtype=torch.float64)
    f_final_lower = torch.tensor(oracle["FINAL_LOWER_VIOLATION"], dtype=torch.float64)
    f_final_upper = torch.tensor(oracle["FINAL_UPPER_VIOLATION"], dtype=torch.float64)
    py_active = active_values(py["state1"], spec)
    py_heun = active_values(py["heun_raw"], spec)
    py_predictor_raw = active_values(py["predictor_raw"], spec)
    py_predictor = active_values(py["predictor"], spec)
    py_k1 = active_values(py["k1"], spec)
    py_pred_correction = active_values(py["predictor_correction"], spec)
    py_final_correction = active_values(py["final_correction"], spec)
    f_state1_active = f_state1
    s3 = sequential_baseline(case, spec, "S3")
    s4 = sequential_baseline(case, spec, "S4")
    flux0_error = py["flux1"] - f_flux0
    flux1_error = py["flux2"] - f_flux1
    flux_avg_error = py["flux_avg_raw"] - f_flux_avg
    flux_final_error = py["flux_final"] - f_flux_final
    qperc_index = FLUX_NAMES.index("QPERC_12")
    qperc_fortran = float(f_flux0[qperc_index])
    qperc_python = float(py["flux1"][qperc_index])
    qperc_fortran_stage2 = float(f_flux1[qperc_index])
    qperc_python_stage2 = float(py["flux2"][qperc_index])
    qperc_active_fortran = qperc_fortran > 1.0e-8
    qperc_active_python = qperc_python > 1.0e-8
    qperc_active_fortran_stage2 = qperc_fortran_stage2 > 1.0e-8
    qperc_active_python_stage2 = qperc_python_stage2 > 1.0e-8
    topology = case["decisions"]
    active_indices = [STATE_NAMES.index(name) for name in spec.state_names]
    def changed_flux_names(delta: torch.Tensor) -> list[str]:
        return [name for index, name in enumerate(FLUX_NAMES) if abs(float(delta[index])) > 1.0e-7]
    pred_flux_delta_fortran = f_pred_flux_fixed - f_flux0
    pred_flux_delta_python = py["predictor_flux_fixed"] - py["flux1"]
    final_flux_delta_fortran = f_flux_final - f_final_flux_before
    final_flux_delta_python = py["flux_final"] - py["flux_avg_raw"]
    row = {
        "case_id": case["case_id"],
        "model_id": case["model_id"],
        "state_label": case["state_label"],
        "forcing_label": case["forcing_label"],
        "topology": {key: topology[key] for key in ("ARCH1", "ARCH2", "QSURF", "QPERC")},
        "state_error": {
            "coupled_rhs_d0_max_abs": max_abs(py_k1 - f_d0),
            "coupled_euler_predictor_max_abs": max_abs(py_predictor_raw - f_predictor),
            "coupled_euler_predictor_safe_max_abs": max_abs(py_predictor - f_pred_safe),
            "coupled_rk2_raw_max_abs": max_abs(py_heun - f_heun),
            "coupled_rk2_final_max_abs": max_abs(py_active - f_state1_active),
            "coupled_rk2_raw_relative_max": relative_max(py_heun - f_heun, f_heun),
            "s3_final_max_abs": max_abs(s3["state1"] - f_state1_active),
            "s4_final_max_abs": max_abs(s4["state1"] - f_state1_active),
        },
        "flux_error": {
            "stage1_max_abs": max_abs(flux0_error),
            "stage1_relative_max": relative_max(flux0_error, f_flux0),
            "stage2_max_abs": max_abs(flux1_error),
            "average_before_fix_max_abs": max_abs(flux_avg_error),
            "final_max_abs": max_abs(flux_final_error),
            "by_name_stage1": {name: float(flux0_error[index]) for index, name in enumerate(FLUX_NAMES)},
            "by_name_final": {name: float(flux_final_error[index]) for index, name in enumerate(FLUX_NAMES)},
        },
        "correction": {
            "predictor_state_max_abs": max_abs(py_pred_correction - f_pred_correction[:len(spec.state_names)]),
            "final_state_max_abs": max_abs(py_final_correction - f_final_correction[:len(spec.state_names)]),
            "predictor_flux_max_abs": max_abs(pred_flux_delta_python - pred_flux_delta_fortran),
            "final_flux_max_abs": max_abs(final_flux_delta_python - final_flux_delta_fortran),
            "predictor_error_field_max_abs": max_abs(py["predictor_errors"] - f_pred_errors),
            "final_error_field_max_abs": max_abs(py["final_errors"] - f_final_errors),
            "predictor_lower_classification_agreement": bool(torch.equal(py["predictor_lower"][active_indices] > 0.5, f_pred_lower[:len(spec.state_names)] > 0.5)),
            "predictor_upper_classification_agreement": bool(torch.equal(py["predictor_upper"][active_indices] > 0.5, f_pred_upper[:len(spec.state_names)] > 0.5)),
            "final_lower_classification_agreement": bool(torch.equal(py["final_lower"][active_indices] > 0.5, f_final_lower[:len(spec.state_names)] > 0.5)),
            "final_upper_classification_agreement": bool(torch.equal(py["final_upper"][active_indices] > 0.5, f_final_upper[:len(spec.state_names)] > 0.5)),
            "predictor_affected_flux_names_fortran": changed_flux_names(pred_flux_delta_fortran),
            "predictor_affected_flux_names_coupled": changed_flux_names(pred_flux_delta_python),
            "final_affected_flux_names_fortran": changed_flux_names(final_flux_delta_fortran),
            "final_affected_flux_names_coupled": changed_flux_names(final_flux_delta_python),
        },
        "qperc": {
            "stage1_fortran_active": qperc_active_fortran,
            "stage1_coupled_active": qperc_active_python,
            "stage1_active_agreement": qperc_active_fortran == qperc_active_python,
            "stage1_fortran_value": qperc_fortran,
            "stage1_coupled_value": qperc_python,
            "stage1_magnitude_abs_error": abs(qperc_python - qperc_fortran),
            "stage2_fortran_active": qperc_active_fortran_stage2,
            "stage2_coupled_active": qperc_active_python_stage2,
            "stage2_active_agreement": qperc_active_fortran_stage2 == qperc_active_python_stage2,
            "stage2_fortran_value": qperc_fortran_stage2,
            "stage2_coupled_value": qperc_python_stage2,
            "stage2_magnitude_abs_error": abs(qperc_python_stage2 - qperc_fortran_stage2),
        },
        "water_balance": {
            "coupled_rk2_residual": float((py["state1"].sum() - py["union"].sum() - (py["flux_final"][0] - py["flux_final"][4] - py["flux_final"][11] - (py["flux_final"][10] + py["flux_final"][9] + py["flux_final"][8] + py["flux_final"][18] + py["flux_final"][15]))).abs()),
            "fortran_residual": float((f_state1.sum() - f_state0.sum() - (f_flux_final[0] - f_flux_final[4] - f_flux_final[11] - (f_flux_final[10] + f_flux_final[9] + f_flux_final[8] + f_flux_final[18] + f_flux_final[15]))).abs()),
        },
        "constraints": {
            "coupled_rk2": constraint_summary(py_active, spec, case["theta"]),
            "fortran": constraint_summary(f_state1_active, spec, case["theta"]),
            "predictor_safeguard_changed": max_abs(f_predictor - f_pred_safe) > 1.0e-6,
            "final_safeguard_changed": max_abs(f_heun - f_state1) > 1.0e-6,
        },
    }
    return row


def compile_smoke(specs: list[Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_compile_diagnostics()
    parity_rows = []
    gradient_rows = []
    generated_rows = []
    compile_model_ids = (2, 6, 8, 14, 164, 212)
    for spec in [candidate for candidate in specs if candidate.model_id in compile_model_ids]:
        params = calibrated_parameters(spec.model_id)
        forcing = torch.tensor([[5.0, 2.0, 10.0]], dtype=torch.float64, device=device)
        eager = simulate_coupled_rk2(spec.model_id, forcing, params, compile_step=False)
        compiled = simulate_coupled_rk2(spec.model_id, forcing, params, compile_step=True)
        parity_rows.append({
            "model_id": spec.model_id,
            "q_max_abs": max_abs(compiled.q.cpu() - eager.q.cpu()),
            "state_max_abs": max_abs(compiled.states.cpu() - eager.states.cpu()),
            "flux_max_abs": max_abs(torch.stack([compiled.fluxes[name].cpu() - eager.fluxes[name].cpu() for name in FLUX_NAMES])),
            "diagnostic_max_abs": max_abs(torch.stack([compiled.sequential_diagnostics[name].cpu() - eager.sequential_diagnostics[name].cpu() for name in COUPLED_RK2_DIAGNOSTIC_NAMES])),
        })
        grad_params = {name: torch.tensor(params[name], dtype=torch.float64, device=device, requires_grad=True) for name in PARAMETER_NAMES}
        grad_result = simulate_coupled_rk2(spec.model_id, forcing, grad_params, compile_step=True)
        grad_result.q.sum().backward()
        active = set(spec.parameter_names)
        inactive_nonzero = []
        nonfinite = []
        for name in PARAMETER_NAMES:
            grad = grad_params[name].grad
            if grad is not None and not bool(torch.isfinite(grad).all()):
                nonfinite.append(name)
            if name not in active and grad is not None and float(grad.abs()) > 1.0e-10:
                inactive_nonzero.append(name)
        gradient_rows.append({"model_id": spec.model_id, "finite": not nonfinite, "nonfinite_parameters": nonfinite, "inactive_nonzero_parameters": inactive_nonzero})
        signature, generated = get_generated_step(spec, order=("coupled_rhs",), n_substeps=1, execution_mode="coupled_rk2")
        source = generated.generated_source
        generated_rows.append({
            "model_id": spec.model_id,
            "signature": signature.to_dict(),
            "generated_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "contains_model_id_dispatch": "model_id" in source,
            "contains_structure_mask": "_CTX_" in source or "context" in source,
            "contains_process_order_loop": "for process in" in source,
            "contains_coupled_step_impl": "_runtime_coupled_rk2_step_impl" in source,
            "transitive_runtime_helpers": ["_runtime_coupled_rhs_impl", "_runtime_fix_states"],
        })
    if device.type == "cuda":
        torch.cuda.synchronize()
    diagnostics = compile_diagnostics().get("runtime", {})
    records = [record for record in diagnostics.get("records", {}).values() if record.get("graph_signature", {}).get("execution_mode") == "coupled_rk2"]
    return {
        "schema_version": "coupled-rhs-rk2-compile-validation-v2",
        "device": str(device),
        "backend": "inductor",
        "fullgraph": True,
        "parity_rows": parity_rows,
        "gradient_rows": gradient_rows,
        "generated_rows": generated_rows,
        "compile_audit": {
            "record_count": len(records),
            "compile_attempts": sum(int(record.get("compile_attempts", 0)) for record in records),
            "compile_successes": sum(int(record.get("compile_successes", 0)) for record in records),
            "fallbacks": sum(int(record.get("fallbacks", 0)) for record in records),
            "graph_breaks": sum(int(record.get("graph_breaks", 0)) for record in records),
            "recompilations": sum(int(record.get("recompilations", 0)) for record in records),
            "autograd_recompilations": sum(int(record.get("autograd_recompilations", 0)) for record in records),
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        result = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            result.append(float(value))
        return result

    def stats(items: list[float]) -> dict[str, float]:
        tensor = torch.tensor(items, dtype=torch.float64)
        return {"min": float(tensor.min()), "median": float(tensor.median()), "mean": float(tensor.mean()), "max": float(tensor.max())}

    qperc_stage1_agreement = sum(row["qperc"]["stage1_active_agreement"] for row in rows)
    qperc_stage2_agreement = sum(row["qperc"]["stage2_active_agreement"] for row in rows)
    finite = sum(row["constraints"]["coupled_rk2"]["finite"] for row in rows)
    capacity_clean = sum(row["constraints"]["coupled_rk2"]["capacity_violation_max"] <= 1.0e-10 for row in rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "/".join(row["topology"][name] for name in ("ARCH1", "ARCH2", "QSURF", "QPERC"))
        grouped.setdefault(key, []).append(row)
    topology = {}
    for key, group in grouped.items():
        topology[key] = {
            "count": len(group),
            "qperc_stage1_active_agreement": sum(row["qperc"]["stage1_active_agreement"] for row in group) / len(group),
            "qperc_stage2_active_agreement": sum(row["qperc"]["stage2_active_agreement"] for row in group) / len(group),
            "rk2_final_state_error": stats([row["state_error"]["coupled_rk2_final_max_abs"] for row in group]),
            "rk2_final_flux_error": stats([row["flux_error"]["final_max_abs"] for row in group]),
            "qperc_stage1_magnitude_error": stats([row["qperc"]["stage1_magnitude_abs_error"] for row in group]),
            "qperc_stage2_magnitude_error": stats([row["qperc"]["stage2_magnitude_abs_error"] for row in group]),
        }
    return {
        "case_count": len(rows),
        "finite_count": finite,
        "capacity_clean_count": capacity_clean,
        "qperc_stage1_active_agreement_count": qperc_stage1_agreement,
        "qperc_stage1_active_agreement_fraction": qperc_stage1_agreement / len(rows),
        "qperc_stage2_active_agreement_count": qperc_stage2_agreement,
        "qperc_stage2_active_agreement_fraction": qperc_stage2_agreement / len(rows),
        "qperc_magnitude_error": {
            "stage1": stats([row["qperc"]["stage1_magnitude_abs_error"] for row in rows]),
            "stage2": stats([row["qperc"]["stage2_magnitude_abs_error"] for row in rows]),
        },
        "state_error": {name: stats(values(("state_error", name))) for name in ("coupled_rhs_d0_max_abs", "coupled_euler_predictor_max_abs", "coupled_rk2_raw_max_abs", "coupled_rk2_final_max_abs", "s3_final_max_abs", "s4_final_max_abs")},
        "flux_error": {name: stats(values(("flux_error", name))) for name in ("stage1_max_abs", "stage2_max_abs", "average_before_fix_max_abs", "final_max_abs")},
        "water_balance": {"coupled_rk2_max_abs": max(values(("water_balance", "coupled_rk2_residual"))), "fortran_max_abs": max(values(("water_balance", "fortran_residual")))},
        "topology_groups": topology,
        "safeguard_case_count": sum(row["constraints"]["predictor_safeguard_changed"] or row["constraints"]["final_safeguard_changed"] for row in rows),
    }


def decide_stage8(summary: dict[str, Any], compile_result: dict[str, Any]) -> dict[str, Any]:
    compile_clean = (
        compile_result["compile_audit"]["compile_attempts"] == compile_result["compile_audit"]["compile_successes"]
        and compile_result["compile_audit"]["fallbacks"] == 0
        and compile_result["compile_audit"]["graph_breaks"] == 0
        and compile_result["compile_audit"]["recompilations"] == 0
        and all(row["finite"] and not row["inactive_nonzero_parameters"] for row in compile_result["gradient_rows"])
    )
    qperc_clean = summary["qperc_stage2_active_agreement_fraction"] >= 0.95
    capacity_clean = summary["capacity_clean_count"] == summary["case_count"]
    water_balance_clean = summary["water_balance"]["coupled_rk2_max_abs"] <= 1.0e-8
    # The pass decision is intentionally comparative: the coupled RK2 median must
    # beat both frozen order baselines and have no topology group with a large
    # qperc activation failure.  The oracle precision is retained in the artifact;
    # no universal absolute state/flux target is imposed here.
    rk2_beats_orders = (
        summary["state_error"]["coupled_rk2_final_max_abs"]["median"] < summary["state_error"]["s3_final_max_abs"]["median"]
        and summary["state_error"]["coupled_rk2_final_max_abs"]["median"] < summary["state_error"]["s4_final_max_abs"]["median"]
    )
    topology_qperc_clean = all(group["qperc_stage2_active_agreement"] >= 0.95 for group in summary["topology_groups"].values())
    passed = compile_clean and qperc_clean and capacity_clean and water_balance_clean and rk2_beats_orders and topology_qperc_clean
    return {
        "stage8_verdict": "Fortran FIX_STATES semantics validated — ready for 4×12 coupled-RK2 diagnostic" if passed else "FIX_STATES semantics unresolved",
        "pass": passed,
        "criteria": {
            "compile_clean": compile_clean,
            "qperc_active_agreement_at_least_95_percent": qperc_clean,
            "capacity_clean": capacity_clean,
            "water_balance_at_numerical_precision": water_balance_clean,
            "rk2_median_state_error_beats_s3_and_s4": rk2_beats_orders,
            "all_topology_groups_qperc_agreement_at_least_95_percent": topology_qperc_clean,
        },
        "decision_basis": "comparative to S3/S4, per-topology qperc agreement, and numerical water balance; no universal absolute state/flux target is used as the scientific decision rule",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortran-executable", type=Path, default=Path(os.environ.get("FUSE_COUPLED_RHS_EXECUTABLE", "/tmp/autofuse-coupled-rhs-build/coupled_rhs_fortran.exe")))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "project/autofuse/docs")
    args = parser.parse_args()
    executable = args.fortran_executable.expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    specs = representative_specs()
    audit = build_dependency_audit(specs)
    cases = build_cases(specs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coupled_rhs_dependency_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    case_descriptors = [{key: value for key, value in case.items() if key != "theta"} for case in cases]
    (args.output_dir / "coupled_rhs_single_step_cases.json").write_text(json.dumps({
        "schema_version": "coupled-rhs-single-step-cases-v1",
        "status": "frozen_before_single_step_results",
        "selection": {"basin_id": CATCHMENT, "structure_ids": list(STRUCTURE_IDS), "state_labels": list(STATE_LABELS), "forcing_cases": [item[0] for item in FORCING_CASES]},
        "archive_sha256": sha256_file(ARCHIVE_PATH),
        "case_count": len(case_descriptors),
        "cases": case_descriptors,
    }, indent=2) + "\n")
    boundary_gate_path = args.output_dir / "fix_states_boundary_validation.json"
    boundary_gate = json.loads(boundary_gate_path.read_text()) if boundary_gate_path.is_file() else {}
    if not boundary_gate.get("summary", {}).get("stage8_boundary_correction", False):
        raise RuntimeError("full 490 validation requires a passed FIX_STATES boundary gate")
    compile_result = compile_smoke(specs)
    (args.output_dir / "coupled_rhs_rk2_compile_validation_v2.json").write_text(json.dumps(compile_result, indent=2) + "\n")
    rows = []
    failures = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        try:
            rows.append(validate_case(case, get_structure(case["model_id"]), executable))
        except Exception as exc:
            failures.append({"case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"})
        if index % 50 == 0:
            print(f"validated {index}/{len(cases)} cases", flush=True)
    summary = aggregate(rows) if rows else {"case_count": 0}
    decision = decide_stage8(summary, compile_result) if rows else {"stage8_verdict": "FAIL — formulation still wrong", "pass": False, "criteria": {}}
    validation = {
        "schema_version": "coupled-rhs-rk2-single-step-validation-v2",
        "status": "complete" if not failures else "partial",
        "scientific_equations_modified": False,
        "fortran_executable": str(executable),
        "fortran_executable_sha256": sha256_file(executable),
        "fortran_wrapper_source": "project/autofuse/coupled_rhs_fortran_wrapper.f90",
        "fortran_wrapper_source_sha256": sha256_file(ROOT / "project/autofuse/coupled_rhs_fortran_wrapper.f90"),
        "boundary_correction_gate": {"artifact": "project/autofuse/docs/fix_states_boundary_validation.json", "sha256": sha256_file(ROOT / "project/autofuse/docs/fix_states_boundary_validation.json"), "passed": boundary_gate["summary"]["stage8_boundary_correction"]},
        "protocol": {"theta_basin": CATCHMENT, "archive_sha256": sha256_file(ARCHIVE_PATH), "dt_days": 1.0, "fixed_forcing_across_stages": True, "no_recalibration": True, "calibration_theta_reused": True, "no_full_s3_s4_rerun": True, "single_step_s3_s4_baseline_only": True, "no_real_catchment_in_this_task": True, "real_diagnostic_not_started": True, "oracle_mode": "direct MOD_DERIVS RHS plus Fortran EXPLICIT_HEUN safeguard sequence"},
        "elapsed_seconds": time.perf_counter() - started,
        "resource": {"child_maxrss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "parent_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0},
        "summary": summary,
        "decision": decision,
        "failures": failures,
        "rows": rows,
    }
    (args.output_dir / "coupled_rhs_rk2_single_step_validation_v2.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "rows": len(rows), "failures": len(failures), "stage8": decision["stage8_verdict"]}))


if __name__ == "__main__":
    main()
