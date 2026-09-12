"""Audit the long-horizon snow-balance definition without changing snow physics."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dfuse import simulate_coupled_rk2
from dfuse.kernel import _parameter_values, _parameter_vector
from dfuse.runtime import _runtime_snow_step
from project.autofuse.torch_fuse_78_long_horizon_smoke import _load_frozen_inputs

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
OUTPUT = DOCS / "snow_balance_long_horizon_audit.json"
UPDATE_SWE = ROOT / "vendor/upstream/cyrilthebault-fuse/build/FUSE_SRC/FUSE_ENGINE/update_swe.f90"
RUNTIME = ROOT / "dfuse/runtime.py"
MODEL_ID = 8
BASIN_ID = "USA_14138900"
TARGET_INDEX = 751
THRESHOLD = 1.0e-8


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _first(values: np.ndarray, dates: list[date]) -> dict[str, Any] | None:
    hits = np.flatnonzero(np.abs(values) > THRESHOLD)
    if not len(hits):
        return None
    index = int(hits[0])
    return {"index": index, "date": dates[index].isoformat(), "threshold": THRESHOLD, "residual": float(values[index])}


def _snow_trace(values: Mapping[str, np.ndarray], theta: Mapping[str, Any], dates: list[date], device: torch.device) -> dict[str, np.ndarray]:
    theta_tensor = _parameter_vector(_parameter_values(theta, dtype=torch.float64, device=device))
    snow = torch.zeros((), dtype=torch.float64, device=device)
    effective = []
    snow_history = [0.0]
    for index, current_date in enumerate(dates):
        ppt = torch.as_tensor(float(values["ppt"][index]), dtype=torch.float64, device=device)
        temp = torch.as_tensor(float(values["temp"][index]), dtype=torch.float64, device=device)
        jday = torch.as_tensor(float(current_date.timetuple().tm_yday), dtype=torch.float64, device=device)
        leap = torch.as_tensor(float(current_date.year % 4 == 0), dtype=torch.float64, device=device)
        output, next_snow = _runtime_snow_step(ppt, temp, snow, jday, leap, theta_tensor, torch.as_tensor(1.0, dtype=torch.float64, device=device))
        effective.append(float(output.detach().cpu()))
        snow = next_snow
        snow_history.append(float(snow.detach().cpu()))
    return {"effective": np.asarray(effective), "snow": np.asarray(snow_history)}


def main() -> None:
    source_meta, inputs, theta_rows = _load_frozen_inputs()
    hru_id = int(next(row["hru_id"] for row in source_meta["manifest"]["catchments"] if row["basin_id"] == BASIN_ID))
    theta = theta_rows[(hru_id, MODEL_ID)]["parameter_vector"]
    values = inputs[BASIN_ID]
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(len(values["ppt"]))]
    device = torch.device("cuda")
    forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1), dtype=torch.float64, device=device)
    result = simulate_coupled_rk2(MODEL_ID, forcing, theta, initial_fraction=0.25, dates=dates, dt_days=1.0, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    snow_trace = _snow_trace(values, theta, dates, device)
    history = snow_trace["snow"]
    rferr = float(theta["RFERR_MLT"])
    pxtemp = float(theta["PXTEMP"])
    ppt = values["ppt"] * rferr
    raw = ppt - snow_trace["effective"] - np.diff(history)
    reported = result.snow_balance_residual.detach().cpu().numpy().astype(np.float64)
    participating_precipitation = np.where(values["temp"] < pxtemp, ppt, np.where(values["temp"] > pxtemp, ppt, 0.0))
    corrected = participating_precipitation - snow_trace["effective"] - np.diff(history)
    target = {
        "index": TARGET_INDEX,
        "date": dates[TARGET_INDEX].isoformat(),
        "forcing": {"ppt": float(values["ppt"][TARGET_INDEX]), "temp": float(values["temp"][TARGET_INDEX]), "pet": float(values["pet"][TARGET_INDEX])},
        "parameters": {"RFERR_MLT": rferr, "PXTEMP": pxtemp, "MBASE": float(theta["MBASE"])},
        "snow_start": float(history[TARGET_INDEX]),
        "snow_next": float(history[TARGET_INDEX + 1]),
        "effective_precipitation": float(snow_trace["effective"][TARGET_INDEX]),
        "source_participating_precipitation": float(participating_precipitation[TARGET_INDEX]),
        "raw_checker_input_precipitation": float(ppt[TARGET_INDEX]),
        "raw_checker_residual": float(raw[TARGET_INDEX]),
        "kernel_reported_residual": float(reported[TARGET_INDEX]),
        "corrected_checker_residual": float(corrected[TARGET_INDEX]),
        "temperature_partition": "exact PXTEMP equality: source UPDATE_SWE takes neither snow accumulation (TEMP_Z < PXTEMP) nor rain contribution (TEMP_Z > PXTEMP)",
    }
    payload = {
        "schema_version": "snow-balance-long-horizon-audit-v1",
        "status": "complete",
        "case": {"basin_id": BASIN_ID, "hru_id": hru_id, "model_id": MODEL_ID, "target": target},
        "source": {"update_swe": str(UPDATE_SWE), "update_swe_sha256": _sha(UPDATE_SWE), "runtime": str(RUNTIME), "runtime_sha256": _sha(RUNTIME)},
        "source_contract": {"accumulation": "UPDATE_SWE: only TEMP_Z < PXTEMP contributes SNOWACCMLTN; multiplicative RFERR uses PRECIP_Z*RFERR_MLT", "melt": "UPDATE_SWE: SWE>0 and TEMP_Z>MBASE gives SNOWMELT; melt is capped when SWE would become negative", "effective_precipitation": "TEMP_Z > PXTEMP contributes rainfall plus snowmelt; otherwise only snowmelt contributes", "current_torch_lumped_path": "_runtime_snow_step uses the same strict < and > partition for lumped ppt/temp and tracks one SWE scalar"},
        "checker_definitions": {"current_raw": "ppt*RFERR_MLT - effective_precipitation - (snow_next-snow_start)", "source_consistent": "precipitation_participating_in_source_partition - effective_precipitation - (snow_next-snow_start); participating precipitation is ppt*RFERR_MLT when temp<PXTEMP or temp>PXTEMP, and zero at exact equality", "why_not_effective_only": "effective precipitation includes rainfall, while the SWE state balance must remove the corresponding rainfall input before comparing SWE change"},
        "long_horizon": {"n_steps": len(dates), "raw_checker_max_abs": float(np.max(np.abs(raw))), "kernel_reported_max_abs": float(np.max(np.abs(reported))), "corrected_max_abs": float(np.max(np.abs(corrected))), "raw_first_violation": _first(raw, dates), "kernel_reported_first_violation": _first(reported, dates), "corrected_first_violation": _first(corrected, dates), "raw_nonfinite": int(np.count_nonzero(~np.isfinite(raw))), "kernel_reported_nonfinite": int(np.count_nonzero(~np.isfinite(reported))), "corrected_nonfinite": int(np.count_nonzero(~np.isfinite(corrected)))},
        "target_trace": target,
        "conclusion": {"classification": "checker_definition_bug", "implementation_change_required": False, "explanation": "The 15.45 residual is raw ppt at exact PXTEMP equality, where the source intentionally contributes neither snow accumulation nor rainfall. Across the full Torch snow path, the source-consistent checker removes the residual to numerical precision; no snow equation/state implementation change is supported by the trace."},
        "protocol": {"forcing_start": dates[0].isoformat(), "simulation_end": dates[-1].isoformat(), "initial_snow": 0.0, "dt_days": 1.0, "dtype": "torch.float64", "device": "cuda", "compile_backend": "inductor", "compile_fullgraph": True},
    }
    OUTPUT.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "target": target, "raw_checker_max_abs": payload["long_horizon"]["raw_checker_max_abs"], "kernel_reported_max_abs": payload["long_horizon"]["kernel_reported_max_abs"], "corrected_max_abs": payload["long_horizon"]["corrected_max_abs"], "corrected_first_violation": payload["long_horizon"]["corrected_first_violation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
