"""Freeze and validate a small Fortran FIX_STATES boundary set."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch

from dfuse.spec import FLUX_NAMES, STATE_NAMES
from project.autofuse.coupled_rhs_validation import (
    build_cases,
    representative_specs,
    run_fortran,
    sha256_file,
    coupled_python,
)

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "project/autofuse/docs/coupled_rhs_single_step_cases.json"
V1_PATH = ROOT / "project/autofuse/docs/coupled_rhs_rk2_single_step_validation.json"
BOUNDARY_CASES_PATH = ROOT / "project/autofuse/docs/fix_states_boundary_cases.json"
BOUNDARY_VALIDATION_PATH = ROOT / "project/autofuse/docs/fix_states_boundary_validation.json"

# Selection is deliberately driven by the previous Fortran safeguard flags and
# coverage dimensions, never by the old PyTorch error magnitude.
TRIGGER_FORCINGS = ("high_rainfall", "high_et", "rainfall_high_et", "active_percolation")
TRIGGER_STATES = ("near_capacity_storage", "low_storage", "upper_wet_lower_dry", "upper_dry_lower_wet")


def _load_triggered_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_cases = build_cases(representative_specs())
    by_id = {case["case_id"]: case for case in all_cases}
    prior = json.loads(V1_PATH.read_text())
    triggered = {
        row["case_id"]
        for row in prior["rows"]
        if row["constraints"]["predictor_safeguard_changed"] or row["constraints"]["final_safeguard_changed"]
    }
    candidates = [by_id[case_id] for case_id in sorted(triggered) if case_id in by_id]
    if len(candidates) < 30:
        raise AssertionError(f"only {len(candidates)} previously triggered cases available")
    return candidates, prior


def _topology_key(case: dict[str, Any]) -> str:
    return "/".join(case["decisions"][name] for name in ("ARCH1", "ARCH2", "QSURF", "QPERC"))


def freeze_boundary_cases() -> dict[str, Any]:
    candidates, prior = _load_triggered_cases()
    by_key = {(case["model_id"], case["forcing_label"], case["state_label"]): case for case in candidates}
    selected: dict[str, dict[str, Any]] = {}

    # First guarantee each selected structure sees rainfall, ET/deficit, and
    # lower-layer/percolation stress in a near-capacity or low-storage state.
    for model_id in sorted({case["model_id"] for case in candidates}):
        for forcing in TRIGGER_FORCINGS:
            preferred = [
                case for case in candidates
                if case["model_id"] == model_id and case["forcing_label"] == forcing
            ]
            preferred.sort(key=lambda case: (TRIGGER_STATES.index(case["state_label"]) if case["state_label"] in TRIGGER_STATES else 99, case["case_id"]))
            if preferred:
                selected[preferred[0]["case_id"]] = preferred[0]

    dimensions = {
        "models": {str(case["model_id"]) for case in selected.values()},
        "state_labels": {case["state_label"] for case in selected.values()},
        "forcings": {case["forcing_label"] for case in selected.values()},
        "topologies": {_topology_key(case) for case in selected.values()},
    }
    # Fill deterministically to 48, maximizing unrepresented coverage while
    # retaining only cases that already triggered the Fortran safeguard.
    while len(selected) < min(48, len(candidates)):
        remaining = [case for case in candidates if case["case_id"] not in selected]
        if not remaining:
            break
        scored = []
        for case in remaining:
            score = sum(
                value not in dimensions[name]
                for name, value in (
                    ("models", str(case["model_id"])),
                    ("state_labels", case["state_label"]),
                    ("forcings", case["forcing_label"]),
                    ("topologies", _topology_key(case)),
                )
            )
            scored.append((score, case["case_id"], case))
        _, _, chosen = max(scored, key=lambda item: (item[0], -len(item[1]), item[1]))
        selected[chosen["case_id"]] = chosen
        dimensions["models"].add(str(chosen["model_id"]))
        dimensions["state_labels"].add(chosen["state_label"])
        dimensions["forcings"].add(chosen["forcing_label"])
        dimensions["topologies"].add(_topology_key(chosen))

    cases = [selected[case_id] for case_id in sorted(selected)]
    if not 30 <= len(cases) <= 48:
        raise AssertionError(f"boundary selection has {len(cases)} cases")
    result = {
        "schema_version": "fix-states-boundary-cases-v1",
        "status": "frozen_before_boundary_validation",
        "selection_basis": {
            "source_cases": str(CASES_PATH.relative_to(ROOT)),
            "source_validation": str(V1_PATH.relative_to(ROOT)),
            "source_trigger": "prior Fortran predictor/final safeguard changed flag",
            "not_used": "previous PyTorch error magnitude",
            "prior_validation_sha256": sha256_file(V1_PATH),
        },
        "coverage": {
            "case_count": len(cases),
            "models": sorted(dimensions["models"]),
            "state_labels": sorted(dimensions["state_labels"]),
            "forcing_labels": sorted(dimensions["forcings"]),
            "topologies": sorted(dimensions["topologies"]),
            "required_forcings": list(TRIGGER_FORCINGS),
            "required_state_labels": list(TRIGGER_STATES),
        },
        "cases": cases,
    }
    BOUNDARY_CASES_PATH.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _active(value: torch.Tensor, spec) -> torch.Tensor:
    return torch.stack([value[STATE_NAMES.index(name)] for name in spec.state_names])


def _max_abs(value: torch.Tensor) -> float:
    return float(value.abs().max())

def _scaled_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    scale = torch.maximum(torch.maximum(left.abs(), right.abs()), torch.ones_like(left))
    return float((left - right).abs().div(scale).max())


def _changed_names(delta: torch.Tensor) -> list[str]:
    return [name for index, name in enumerate(FLUX_NAMES) if abs(float(delta[index])) > 1.0e-7]


def _water_balance(state: torch.Tensor, previous: torch.Tensor, flux: torch.Tensor, dt: float) -> float:
    instantaneous = flux[10] + flux[9] + flux[8] + flux[18] + flux[15]
    return float((state.sum() - previous.sum() - (flux[0] - flux[4] - flux[11] - instantaneous) * dt).abs())


def validate_boundary_case(case: dict[str, Any], spec, executable: Path) -> dict[str, Any]:
    oracle = run_fortran(executable, case, spec)
    py = coupled_python(case, spec)
    active_indices = [STATE_NAMES.index(name) for name in spec.state_names]
    f = {name: torch.tensor(oracle[name], dtype=torch.float64) for name in (
        "STATE0", "D0", "FLUX0", "PREDICTOR", "PREDICTOR_SAFE", "PREDICTOR_FLUX_FIXED",
        "HEUN_RAW", "FLUX_AVG", "FINAL_FLUX_BEFORE_FIX", "STATE1", "FLUX_FINAL",
        "PREDICTOR_ERRORS", "FINAL_ERRORS", "PREDICTOR_STATE_CORRECTION", "FINAL_STATE_CORRECTION",
        "PREDICTOR_LOWER_VIOLATION", "PREDICTOR_UPPER_VIOLATION", "FINAL_LOWER_VIOLATION", "FINAL_UPPER_VIOLATION",
        "PREDICTOR_HIDDEN_FREE2A", "PREDICTOR_HIDDEN_FREE2B", "FINAL_HIDDEN_FREE2A", "FINAL_HIDDEN_FREE2B",
        "STATE_LOWER_BOUNDS", "STATE_UPPER_BOUNDS",
    )}
    dt = float(case["forcing"]["dt_days"])
    pre = {
        "d0_state_max_abs": _max_abs(_active(py["k1"], spec) - f["D0"]),
        "d0_state_scaled_max": _scaled_max_abs(_active(py["k1"], spec), f["D0"]),
        "stage1_flux_max_abs": _max_abs(py["flux1"] - f["FLUX0"]),
        "stage1_flux_scaled_max": _scaled_max_abs(py["flux1"], f["FLUX0"]),
        "raw_predictor_state_max_abs": _max_abs(_active(py["predictor_raw"], spec) - f["PREDICTOR"]),
        "raw_predictor_state_scaled_max": _scaled_max_abs(_active(py["predictor_raw"], spec), f["PREDICTOR"]),
        "raw_heun_state_max_abs": _max_abs(_active(py["heun_raw"], spec) - f["HEUN_RAW"]),
        "raw_heun_state_scaled_max": _scaled_max_abs(_active(py["heun_raw"], spec), f["HEUN_RAW"]),
        "raw_average_flux_max_abs": _max_abs(py["flux_avg_raw"] - f["FLUX_AVG"]),
        "raw_average_flux_scaled_max": _scaled_max_abs(py["flux_avg_raw"], f["FLUX_AVG"]),
    }
    pred_flux_delta_f = f["PREDICTOR_FLUX_FIXED"] - f["FLUX0"]
    pred_flux_delta_p = py["predictor_flux_fixed"] - py["flux1"]
    final_flux_delta_f = f["FLUX_FINAL"] - f["FINAL_FLUX_BEFORE_FIX"]
    final_flux_delta_p = py["flux_final"] - py["flux_avg_raw"]
    post = {
        "predictor_state_max_abs": _max_abs(_active(py["predictor"], spec) - f["PREDICTOR_SAFE"]),
        "predictor_state_scaled_max": _scaled_max_abs(_active(py["predictor"], spec), f["PREDICTOR_SAFE"]),
        "final_state_max_abs": _max_abs(_active(py["state1"], spec) - f["STATE1"]),
        "final_state_scaled_max": _scaled_max_abs(_active(py["state1"], spec), f["STATE1"]),
        "predictor_flux_max_abs": _max_abs(pred_flux_delta_p - pred_flux_delta_f),
        "predictor_flux_scaled_max": _scaled_max_abs(pred_flux_delta_p, pred_flux_delta_f),
        "final_flux_max_abs": _max_abs(final_flux_delta_p - final_flux_delta_f),
        "final_flux_scaled_max": _scaled_max_abs(final_flux_delta_p, final_flux_delta_f),
        "predictor_correction_amount_max_abs": _max_abs(_active(py["predictor_correction"], spec) - f["PREDICTOR_STATE_CORRECTION"][:len(spec.state_names)]),
        "predictor_correction_amount_scaled_max": _scaled_max_abs(_active(py["predictor_correction"], spec), f["PREDICTOR_STATE_CORRECTION"][:len(spec.state_names)]),
        "final_correction_amount_max_abs": _max_abs(_active(py["final_correction"], spec) - f["FINAL_STATE_CORRECTION"][:len(spec.state_names)]),
        "final_correction_amount_scaled_max": _scaled_max_abs(_active(py["final_correction"], spec), f["FINAL_STATE_CORRECTION"][:len(spec.state_names)]),
        "predictor_error_field_max_abs": _max_abs(py["predictor_errors"] - f["PREDICTOR_ERRORS"]),
        "predictor_error_field_scaled_max": _scaled_max_abs(py["predictor_errors"], f["PREDICTOR_ERRORS"]),
        "final_error_field_max_abs": _max_abs(py["final_errors"] - f["FINAL_ERRORS"]),
        "final_error_field_scaled_max": _scaled_max_abs(py["final_errors"], f["FINAL_ERRORS"]),
    }
    predictor_active_raw = _active(py["predictor_raw"], spec)
    final_active_raw = _active(py["heun_raw"], spec)
    predictor_lower_actual = predictor_active_raw < f["STATE_LOWER_BOUNDS"][:len(spec.state_names)]
    predictor_upper_actual = predictor_active_raw > f["STATE_UPPER_BOUNDS"][:len(spec.state_names)]
    final_lower_actual = final_active_raw < f["STATE_LOWER_BOUNDS"][:len(spec.state_names)]
    final_upper_actual = final_active_raw > f["STATE_UPPER_BOUNDS"][:len(spec.state_names)]
    classification = {
        "predictor_lower": bool(torch.equal(predictor_lower_actual, f["PREDICTOR_LOWER_VIOLATION"][:len(spec.state_names)] > 0.5)),
        "predictor_upper": bool(torch.equal(predictor_upper_actual, f["PREDICTOR_UPPER_VIOLATION"][:len(spec.state_names)] > 0.5)),
        "final_lower": bool(torch.equal(final_lower_actual, f["FINAL_LOWER_VIOLATION"][:len(spec.state_names)] > 0.5)),
        "final_upper": bool(torch.equal(final_upper_actual, f["FINAL_UPPER_VIOLATION"][:len(spec.state_names)] > 0.5)),
    }
    affected = {
        "predictor_fortran": _changed_names(pred_flux_delta_f),
        "predictor_coupled": _changed_names(pred_flux_delta_p),
        "final_fortran": _changed_names(final_flux_delta_f),
        "final_coupled": _changed_names(final_flux_delta_p),
    }
    violation_types = {
        "predictor": {name: {"lower": bool(f["PREDICTOR_LOWER_VIOLATION"][offset] > 0.5), "upper": bool(f["PREDICTOR_UPPER_VIOLATION"][offset] > 0.5)} for offset, name in enumerate(spec.state_names)},
        "final": {name: {"lower": bool(f["FINAL_LOWER_VIOLATION"][offset] > 0.5), "upper": bool(f["FINAL_UPPER_VIOLATION"][offset] > 0.5)} for offset, name in enumerate(spec.state_names)},
    }
    water_balance = {
        "fortran_residual": _water_balance(f["STATE1"], f["STATE0"], f["FLUX_FINAL"], dt),
        "coupled_residual": _water_balance(py["state1"], py["union"], py["flux_final"], dt),
    }
    float32_scaled_tolerance = 256.0 * torch.finfo(torch.float32).eps
    pre_scaled = [value for key, value in pre.items() if key.endswith("_scaled_max")]
    post_scaled = [value for key, value in post.items() if key.endswith("_scaled_max")]
    passed = (
        max(pre_scaled) <= float32_scaled_tolerance
        and max(post_scaled) <= float32_scaled_tolerance
        and all(classification.values())
        and affected["predictor_fortran"] == affected["predictor_coupled"]
        and affected["final_fortran"] == affected["final_coupled"]
        and water_balance["coupled_residual"] <= 1.0e-8
    )
    return {
        "oracle": {
            "state_before": f["STATE0"].tolist(),
            "raw_predictor_state": f["PREDICTOR"].tolist(),
            "predictor_state_after_fix": f["PREDICTOR_SAFE"].tolist(),
            "raw_stage1_fluxes": f["FLUX0"].tolist(),
            "corrected_predictor_fluxes": f["PREDICTOR_FLUX_FIXED"].tolist(),
            "raw_heun_state": f["HEUN_RAW"].tolist(),
            "averaged_fluxes_before_final_fix": f["FINAL_FLUX_BEFORE_FIX"].tolist(),
            "final_state_after_fix": f["STATE1"].tolist(),
            "corrected_final_fluxes": f["FLUX_FINAL"].tolist(),
            "predictor_error_fields": f["PREDICTOR_ERRORS"].tolist(),
            "final_error_fields": f["FINAL_ERRORS"].tolist(),
            "predictor_hidden_free2a": float(f["PREDICTOR_HIDDEN_FREE2A"][0]),
            "predictor_hidden_free2b": float(f["PREDICTOR_HIDDEN_FREE2B"][0]),
            "final_hidden_free2a": float(f["FINAL_HIDDEN_FREE2A"][0]),
            "final_hidden_free2b": float(f["FINAL_HIDDEN_FREE2B"][0]),
        },
        "coupled": {
            "predictor_state_after_fix": _active(py["predictor"], spec).tolist(),
            "corrected_predictor_fluxes": py["predictor_flux_fixed"].tolist(),
            "final_state_after_fix": _active(py["state1"], spec).tolist(),
            "corrected_final_fluxes": py["flux_final"].tolist(),
        },
        "case_id": case["case_id"],
        "model_id": case["model_id"],
        "state_label": case["state_label"],
        "forcing_label": case["forcing_label"],
        "state_names": list(spec.state_names),
        "topology": case["decisions"],
        "pre_correction": pre,
        "post_correction": post,
        "classification": classification,
        "violation_types": violation_types,
        "affected_flux_names": affected,
        "water_balance": water_balance,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--fortran-executable", type=Path, required=False, default=Path("/tmp/autofuse-coupled-rhs-build-v2/coupled_rhs_fortran_v2.exe"))
    args = parser.parse_args()
    frozen = freeze_boundary_cases()
    if args.freeze_only:
        print(json.dumps({"frozen_cases": frozen["coverage"]["case_count"], "path": str(BOUNDARY_CASES_PATH)}))
        return
    executable = args.fortran_executable.expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    specs = {spec.model_id: spec for spec in representative_specs()}
    started = time.perf_counter()
    rows = []
    failures = []
    for case in frozen["cases"]:
        try:
            rows.append(validate_boundary_case(case, specs[case["model_id"]], executable))
        except Exception as exc:
            failures.append({"case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"})
    all_passed = bool(rows) and not failures and all(row["passed"] for row in rows)
    validation = {
        "schema_version": "fix-states-boundary-validation-v1",
        "status": "complete" if not failures else "partial",
        "fortran_executable": str(executable),
        "fortran_executable_sha256": sha256_file(executable),
        "fortran_wrapper_source": "project/autofuse/coupled_rhs_fortran_wrapper.f90",
        "protocol": {"dt_days": 1.0, "serial": True, "no_recalibration": True, "no_real_catchment": True, "scaled_error_tolerance_factor_float32_ulp": 256, "water_balance_abs_tolerance": 1.0e-8},
        "selection": {"path": str(BOUNDARY_CASES_PATH.relative_to(ROOT)), "sha256": sha256_file(BOUNDARY_CASES_PATH)},
        "resource": {"child_maxrss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "parent_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "elapsed_seconds": time.perf_counter() - started,
        "summary": {"case_count": len(frozen["cases"]), "rows": len(rows), "failures": len(failures), "passed_cases": sum(row["passed"] for row in rows), "stage8_boundary_correction": all_passed},
        "rows": rows,
        "failures": failures,
    }
    BOUNDARY_VALIDATION_PATH.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation["summary"]))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
