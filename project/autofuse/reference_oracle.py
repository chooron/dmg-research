"""Callable oracle wrapper for the pinned Fortran FUSE executable.

The wrapper owns only temporary NetCDF/file-manager inputs and parses the
reference output.  It never edits ``vendor/upstream`` and deliberately keeps
reference values out of tests and source fixtures.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.io import netcdf_file

from dfuse.spec import PARAMETER_NAMES, SOLVER_CONFIG, get_structure


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "vendor/upstream/fuse-mmcomparison-paper"
TEMPLATE_ROOT = PAPER_ROOT / "01_FUSEscripts/fuse_template"
SETTINGS_ROOT = TEMPLATE_ROOT / "settings"
DECISION_ROOT = SETTINGS_ROOT / "fuse_zDecisions"

_PARAMETER_CATALOG = json.loads((ROOT / "dfuse/specs/parameter_catalog.json").read_text())["parameters"]
_PARAMETER_DEFAULTS = {item["name"]: float(item["default"]) for item in _PARAMETER_CATALOG}

_REFERENCE_STATE_VARS = {
    "TENS_1A": "tens_1a",
    "TENS_1B": "tens_1b",
    "TENS_1": "tens_1",
    "FREE_1": "free_1",
    "WATR_1": "watr_1",
    "TENS_2": "tens_2",
    "FREE_2A": "free_2a",
    "FREE_2B": "free_2b",
    "WATR_2": "watr_2",
}
_REFERENCE_FLUX_VARS = {
    "EFF_PPT": "eff_ppt",
    "SATAREA": "satarea",
    "EVAP_1A": "evap_1a",
    "EVAP_1B": "evap_1b",
    "EVAP_1": "evap_1",
    "EVAP_2": "evap_2",
    "RCHR2EXCS": "rchr2excs",
    "TENS2FREE_1": "tens2free_1",
    "TENS2FREE_2": "tens2free_2",
    "QPERC_12": "qperc_12",
    "QINTF_1": "qintf_1",
    "OFLOW_1": "oflow_1",
    "QSURF": "qsurf",
    "QBASE_2A": "qbase_2a",
    "QBASE_2B": "qbase_2b",
    "QBASE_2": "qbase_2",
    "OFLOW_2A": "oflow_2a",
    "OFLOW_2B": "oflow_2b",
    "OFLOW_2": "oflow_2",
}


@dataclass(frozen=True)
class ReferenceResult:
    """Reference time series returned by one isolated FUSE invocation."""

    model_id: int
    time: np.ndarray
    q_instantaneous: np.ndarray
    q_routed: np.ndarray
    states: dict[str, np.ndarray]
    fluxes: dict[str, np.ndarray]
    initial_state: dict[str, float]
    metadata: dict[str, object]


def _scalar(value: object) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"reference parameters must be scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _series(value: object, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"forcing series {name!r} is empty")
    return array


def _forcing_series(forcing: Mapping[str, object] | object) -> tuple[dict[str, np.ndarray], bool]:
    if isinstance(forcing, Mapping):
        def pick(*names: str) -> object:
            for name in names:
                if name in forcing:
                    return forcing[name]
            raise KeyError(f"forcing is missing one of {names}")

        values = {
            "ppt": _series(pick("ppt", "pr", "PPT"), "ppt"),
            "temp": _series(pick("temp", "temperature", "TEMP"), "temp"),
            "pet": _series(pick("pet", "PET"), "pet"),
        }
        if any(name in forcing for name in ("q_obs", "q", "Q")):
            values["q_obs"] = _series(pick("q_obs", "q", "Q"), "q_obs")
        for name in ("area_frac", "mean_elev"):
            if name in forcing:
                values[name] = _series(forcing[name], name)
        return values, True

    if hasattr(forcing, "detach"):
        forcing = forcing.detach().cpu().numpy()
    array = np.asarray(forcing, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("forcing tensor must have shape [time, 3] (ppt, pet, temp)")
    return {
        "ppt": _series(array[:, 0], "ppt"),
        "pet": _series(array[:, 1], "pet"),
        "temp": _series(array[:, 2], "temp"),
    }, False


def _dates(dates: Sequence[date | datetime] | None, n: int, dt_days: float) -> list[date]:
    if dates is None:
        start = date(2000, 1, 1)
        return [(start + timedelta(days=i * dt_days)) for i in range(n)]
    result = [item.date() if isinstance(item, datetime) else item for item in dates]
    if len(result) != n:
        raise ValueError(f"dates has {len(result)} entries but forcing has {n} steps")
    return result


def _parameter_values(params: Mapping[str, object] | object | None) -> dict[str, float]:
    result = dict(_PARAMETER_DEFAULTS)
    if params is None:
        return result
    if isinstance(params, Mapping):
        for name in PARAMETER_NAMES:
            if name in params:
                result[name] = _scalar(params[name])
            elif name.lower() in params:
                result[name] = _scalar(params[name.lower()])
        return result
    if hasattr(params, "detach"):
        params = params.detach().cpu().numpy()
    array = np.asarray(params).reshape(-1)
    if array.size != len(PARAMETER_NAMES):
        raise ValueError(f"parameter vector must have {len(PARAMETER_NAMES)} coordinates")
    return {name: float(array[i]) for i, name in enumerate(PARAMETER_NAMES)}


def _initial_state(model_id: int, params: Mapping[str, float], fraction: float) -> dict[str, float]:
    if not np.isclose(fraction, 0.25, rtol=0.0, atol=1.0e-12):
        raise ValueError("the frozen reference executable hard-codes fracState0=0.25")
    spec = get_structure(model_id)
    maxwatr1 = params["MAXWATR_1"]
    maxwatr2 = params["MAXWATR_2"]
    fracten = params["FRACTEN"]
    max_tens1 = fracten * maxwatr1
    max_tens2 = fracten * maxwatr2
    max_free1 = (1.0 - fracten) * maxwatr1
    max_free2 = (1.0 - fracten) * maxwatr2
    capacities = {
        "TENS_1A": params["FRCHZNE"] * max_tens1,
        "TENS_1B": (1.0 - params["FRCHZNE"]) * max_tens1,
        "TENS_1": max_tens1,
        "FREE_1": max_free1,
        "WATR_1": maxwatr1,
        "TENS_2": max_tens2,
        "FREE_2A": params["FPRIMQB"] * max_free2,
        "FREE_2B": (1.0 - params["FPRIMQB"]) * max_free2,
        "WATR_2": maxwatr2,
    }
    return {name: capacities[name] * fraction for name in spec.state_names}


def _write_forcing(path: Path, values: Mapping[str, np.ndarray], dates: Sequence[date], dt_days: float) -> None:
    n = len(values["ppt"])
    if any(values[name].size != n for name in ("pet", "temp")):
        raise ValueError("ppt, pet, and temp forcing lengths must match")
    q_obs = values.get("q_obs", np.zeros(n, dtype=np.float32))
    if q_obs.size != n:
        raise ValueError("q_obs forcing length must match")
    area = values.get("area_frac", np.asarray([1.0], dtype=np.float32))
    elev = values.get("mean_elev", np.asarray([0.0], dtype=np.float32))
    if area.size != elev.size or area.size == 0:
        raise ValueError("area_frac and mean_elev must have the same non-zero length")
    if not np.isclose(float(area.sum()), 1.0, rtol=0.0, atol=1.0e-5):
        raise ValueError("area_frac must sum to one for the Fortran reference")

    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("longitude", 1)
        nc.createDimension("latitude", 1)
        nc.createDimension("time", n)
        lon = nc.createVariable("longitude", "f4", ("longitude",))
        lat = nc.createVariable("latitude", "f4", ("latitude",))
        time = nc.createVariable("time", "f4", ("time",))
        dims = ("time", "latitude", "longitude")
        variables = {name: nc.createVariable(name, "f4", dims) for name in ("pr", "temp", "pet", "q_obs")}
        lon[:] = [0.0]
        lat[:] = [45.0]
        time[:] = np.arange(n, dtype=np.float32) * dt_days
        time.units = f"days since {dates[0].isoformat()} 00:00:00"
        variables["pr"][:] = values["ppt"].reshape(n, 1, 1)
        variables["temp"][:] = values["temp"].reshape(n, 1, 1)
        variables["pet"][:] = values["pet"].reshape(n, 1, 1)
        variables["q_obs"][:] = q_obs.reshape(n, 1, 1)


def _write_elevation_bands(path: Path, values: Mapping[str, np.ndarray]) -> None:
    area = values.get("area_frac", np.asarray([1.0], dtype=np.float32))
    elev = values.get("mean_elev", np.asarray([0.0], dtype=np.float32))
    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("longitude", 1)
        nc.createDimension("latitude", 1)
        nc.createDimension("elevation_band", area.size)
        dims = ("elevation_band", "latitude", "longitude")
        area_var = nc.createVariable("area_frac", "f4", dims)
        elev_var = nc.createVariable("mean_elev", "f4", dims)
        area_var[:] = area.reshape(area.size, 1, 1)
        elev_var[:] = elev.reshape(elev.size, 1, 1)


def _write_parameters(path: Path, params: Mapping[str, float]) -> None:
    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("par", 1)
        for name in PARAMETER_NAMES:
            variable = nc.createVariable(name, "f4", ("par",))
            variable[:] = np.asarray([params[name]], dtype=np.float32)


def _write_input_info(path: Path, dt_days: float) -> None:
    text = (SETTINGS_ROOT / "input_info.txt").read_text()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith("<deltim>"):
            lines[index] = f"<deltim>          {dt_days:.10g}                                  ! time step (days)"
    path.write_text("\n".join(lines) + "\n")


def _write_file_manager(path: Path, case: Path, model_id: int, dates: Sequence[date], dt_days: float) -> None:
    settings = str(case / "settings") + os.sep
    input_dir = str(case / "input") + os.sep
    output_dir = str(case / "output") + os.sep
    start = dates[0].isoformat()
    end = dates[-1].isoformat()
    path.write_text(
        "\n".join(
            [
                "FUSE_FILEMANAGER_V1.5",
                "! generated by project/autofuse/reference_oracle.py",
                f"'{settings}'     ! SETNGS_PATH",
                f"'{input_dir}'        ! INPUT_PATH",
                f"'{output_dir}'       ! OUTPUT_PATH",
                "! *** suffixes for input files",
                "'_input.nc'                      ! suffix_forcing",
                "'_elev_bands.nc'                 ! suffix_elev_bands",
                "! *** settings files",
                "'input_info.txt'                 ! FORCING INFO",
                "'fuse_zConstraints_snow.txt'     ! CONSTRAINTS",
                "'fuse_zNumerix.txt'              ! MOD_NUMERIX",
                f"'fuse_zDecisions_{model_id}.txt' ! M_DECISIONS",
                "! *** output files",
                f"'{model_id}'                            ! FMODEL_ID",
                "'FALSE'                          ! Q_ONLY",
                "! *** dates",
                f"'{start}'                     ! date_start_sim",
                f"'{end}'                       ! date_end_sim",
                f"'{start}'                     ! date_start_eval",
                f"'{end}'                       ! date_end_eval",
                "'-9999'                          ! numtim_sub",
                "! *** evaluation metrics and transformation",
                "'RMSE'                           ! METRIC",
                "'1'                              ! TRANSFO",
                "! *** SCE parameters",
                "'20'                             ! MAXN",
                "'3'                              ! KSTOP",
                "'0.001'                          ! PCENTO",
            ]
        )
        + "\n"
    )


def _read_series(nc: object, name: str, n: int) -> np.ndarray:
    variables = getattr(nc, "variables")
    if name not in variables:
        raise RuntimeError(f"reference output is missing variable {name!r}")
    data = np.array(variables[name].data, dtype=np.float64, copy=True).reshape(-1)
    if data.size != n:
        raise RuntimeError(f"reference variable {name!r} has {data.size} values, expected {n}")
    return data


def run_reference(
    executable: str | Path,
    model_id: int,
    forcing: Mapping[str, object] | object,
    params: Mapping[str, object] | object | None = None,
    *,
    initial_state: Mapping[str, object] | object | None = None,
    initial_fraction: float = 0.25,
    dates: Sequence[date | datetime] | None = None,
    dt_days: float = 1.0,
    timeout_seconds: float = 120.0,
) -> ReferenceResult:
    """Run pinned FUSE in ``run_pre`` mode and return copied output arrays.

    The v1.0 executable has a compiled ``fracState0=0.25`` rather than an
    input-file initial-state interface.  ``initial_state`` is therefore
    accepted as an audit assertion and must equal the frozen initialization;
    arbitrary initial states require a separately compiled upstream variant.
    """
    executable_path = Path(executable).expanduser().resolve()
    spec = get_structure(model_id)
    values, is_mapping = _forcing_series(forcing)
    n = values["ppt"].size
    if any(values[name].size != n for name in ("pet", "temp")):
        raise ValueError("ppt, pet, and temp forcing lengths must match")
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    params_values = _parameter_values(params)
    expected_initial = _initial_state(model_id, params_values, initial_fraction)
    if initial_state is not None:
        if isinstance(initial_state, Mapping):
            supplied = {name: _scalar(initial_state[name]) for name in spec.state_names}
        else:
            if hasattr(initial_state, "detach"):
                initial_state = initial_state.detach().cpu().numpy()
            array = np.asarray(initial_state).reshape(-1)
            if array.size != len(spec.state_names):
                raise ValueError(f"initial_state must have {len(spec.state_names)} active coordinates")
            supplied = {name: float(array[i]) for i, name in enumerate(spec.state_names)}
        if not all(np.isclose(supplied[name], expected_initial[name], rtol=1.0e-6, atol=1.0e-6) for name in spec.state_names):
            raise ValueError("reference initial_state must equal the frozen Fortran fracState0=0.25 state")

    if not executable_path.is_file():
        raise FileNotFoundError(executable_path)

    run_dates = _dates(dates, n, dt_days)
    decision_file = DECISION_ROOT / f"fuse_zDecisions_{model_id}.txt"
    if not decision_file.is_file():
        raise FileNotFoundError(f"paper decision file not found for model {model_id}: {decision_file}")

    with tempfile.TemporaryDirectory(prefix="dfuse-reference-") as temporary:
        case = Path(temporary)
        (case / "settings").mkdir()
        (case / "input").mkdir()
        (case / "output").mkdir()
        for name in ("fuse_zConstraints_snow.txt", "fuse_zNumerix.txt"):
            (case / "settings" / name).write_text((SETTINGS_ROOT / name).read_text())
        (case / "settings" / decision_file.name).write_text(decision_file.read_text())
        _write_input_info(case / "settings" / "input_info.txt", dt_days)
        _write_file_manager(case / "fm_catch.txt", case, model_id, run_dates, dt_days)
        _write_forcing(case / "input" / "oracle_input.nc", values, run_dates, dt_days)
        _write_elevation_bands(case / "input" / "oracle_elev_bands.nc", values)
        parameter_file = case / "output" / "oracle_params.nc"
        _write_parameters(parameter_file, params_values)
        parameter_list = case / "params.txt"
        parameter_list.write_text(parameter_file.name + "\n")

        environment = os.environ.copy()
        library_dir = environment.get("FUSE_REFERENCE_LIB_DIR")
        if library_dir:
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                [library_dir, environment.get("LD_LIBRARY_PATH", "")]
            ).rstrip(os.pathsep)
        command = [str(executable_path), str(case / "fm_catch.txt"), "oracle", "run_pre", str(parameter_list)]
        completed = subprocess.run(
            command,
            cwd=case,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        output_path = case / "output" / f"oracle_{model_id}_runs_pre.nc"
        if completed.returncode != 0:
            raise RuntimeError(f"reference FUSE failed ({completed.returncode}): {completed.stderr[-4000:]}\n{completed.stdout[-2000:]}")
        if not output_path.is_file():
            raise RuntimeError(f"reference FUSE returned success but did not create {output_path}: {completed.stdout[-4000:]}")
        with netcdf_file(str(output_path), "r", mmap=False) as output:
            time = np.array(output.variables["time"].data, dtype=np.float64, copy=True).reshape(-1)
            q_instantaneous = _read_series(output, "q_instnt", n)
            q_routed = _read_series(output, "q_routed", n)
            states = {key: _read_series(output, name, n) for key, name in _REFERENCE_STATE_VARS.items() if key in spec.state_names}
            fluxes = {key: _read_series(output, name, n) for key, name in _REFERENCE_FLUX_VARS.items()}
            if "swe_tot" in output.variables:
                states["SWE_TOT"] = _read_series(output, "swe_tot", n)
            for variable_name in output.variables:
                match = re.fullmatch(r"swe_z(\d+)", variable_name)
                if match:
                    states[f"SWE_Z{match.group(1)}"] = _read_series(output, variable_name, n)
                match = re.fullmatch(r"snwacml_z(\d+)", variable_name)
                if match:
                    fluxes[f"SNWACML_Z{match.group(1)}"] = _read_series(output, variable_name, n)
                match = re.fullmatch(r"snwmelt_z(\d+)", variable_name)
                if match:
                    fluxes[f"SNWMELT_Z{match.group(1)}"] = _read_series(output, variable_name, n)

    metadata = {
        "executable": str(executable_path),
        "executable_sha256": hashlib.sha256(executable_path.read_bytes()).hexdigest(),
        "model_id": model_id,
        "source_commit": "e6e23a4fc4ff4019bcab55f14537ea43b9525967",
        "forcing_kind": "mapping" if is_mapping else "tensor",
        "n_steps": n,
        "dt_days": dt_days,
        "initial_fraction": initial_fraction,
        "solver": dict(SOLVER_CONFIG),
        "reference_invocation": "run_pre; parameter NetCDF; frozen file-manager settings",
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }
    return ReferenceResult(
        model_id=model_id,
        time=time,
        q_instantaneous=q_instantaneous,
        q_routed=q_routed,
        states=states,
        fluxes=fluxes,
        initial_state=expected_initial,
        metadata=metadata,
    )
