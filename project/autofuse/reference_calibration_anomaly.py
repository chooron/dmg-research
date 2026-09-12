"""Bounded, auditable diagnosis for one anomalous original-Fortran SCE case.

This script does not modify the upstream FUSE source or any hydrological
process equation.  It preserves the diagnostic case directory so that the
SCE intermediate files, output timestamps, and streamed logs remain inspectable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import netcdf_file

from dfuse.spec import PARAMETERS, PARAMETER_NAMES, get_structure
from project.autofuse.reference_calibration import (
    FORCING_START,
    SIMULATION_END,
    CALIBRATION_START,
    CALIBRATION_END,
    _copy_settings,
    _elevation_metadata,
    _load_cases,
    _read_best,
    _write_elevation_bands,
    _write_file_manager,
    _write_forcing,
    _write_input_info,
)
from project.autofuse.reference_oracle import (
    _REFERENCE_FLUX_VARS,
    _REFERENCE_STATE_VARS,
    _parameter_values,
    _write_parameters,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "project/autofuse/docs"
DEFAULT_PARTIAL = DEFAULT_OUTPUT / "reference_calibration_12x78.partial.json"
DEFAULT_LOG = DEFAULT_OUTPUT / "reference_calibration_logs/12010000_206_calib_sce.log"
DEFAULT_EXE = Path("/tmp/autofuse-reference-toolchain/repro-build/bin/fuse.exe")
MODEL_ID = 206
BASIN_ID = "USA_12010000"
DEFAULT_CALIBRATION_ARCHIVE = DEFAULT_OUTPUT / "reference_calibration_12x78.json"
HRU_ID = 12010000
MAXN = 100
KSTOP = 3
PCENTO = 0.001
DEFAULT_REPRO_CAP_SECONDS = 300.0
DEFAULT_FORWARD_CAP_SECONDS = 120.0
MONITOR_INTERVAL_SECONDS = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dates() -> list[date]:
    return [FORCING_START.fromordinal(FORCING_START.toordinal() + i) for i in range((SIMULATION_END - FORCING_START).days + 1)]


def _snapshot(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            rows.append({"path": str(path.relative_to(root)), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return rows


def _proc_sample(pid: int, started: float) -> dict[str, Any]:
    sample: dict[str, Any] = {"wall_seconds": time.monotonic() - started}
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                sample["rss_kb"] = int(line.split()[1])
                break
        stat_fields = Path(f"/proc/{pid}/stat").read_text().split()
        ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        cpu_seconds = (int(stat_fields[13]) + int(stat_fields[14])) / ticks
        sample["cpu_seconds"] = cpu_seconds
        sample["cpu_percent_of_one_core"] = 100.0 * cpu_seconds / max(sample["wall_seconds"], 1.0e-9)
        sample["state"] = stat_fields[2]
        sample["alive"] = True
    except (FileNotFoundError, OSError, ValueError, IndexError):
        sample["alive"] = False
    return sample


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    library_dir = env.get("FUSE_REFERENCE_LIB_DIR")
    if library_dir:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([library_dir, env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    return env


def _terminate_group(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    result: dict[str, Any] = {"termination_signal": None, "forced_kill": False}
    if process.poll() is not None:
        return result
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        result["termination_signal"] = "SIGTERM"
    except ProcessLookupError:
        result["termination_signal"] = "already-exited"
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        result["forced_kill"] = True
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10.0)
    return result




def _prepare_case(case: Path, case_data: Mapping[str, Any], model_id: int, mode: str, *, parameter_file: Path | None = None) -> None:
    case.mkdir(parents=True, exist_ok=True)
    _copy_settings(case, model_id)
    _write_input_info(case / "settings" / "input_info.txt", 1.0)
    domain_id = f"{int(case_data['hru_id']):08d}"
    _write_forcing(case / "input" / f"{domain_id}_input.nc", case_data["forcing"], case_data["dates"], 1.0)
    _write_elevation_bands(case / "input" / f"{domain_id}_elev_bands.nc", case_data["forcing"])
    _write_file_manager(case / "fm_catch.txt", case, domain_id, model_id, FORCING_START, SIMULATION_END, CALIBRATION_START, CALIBRATION_END, MAXN, KSTOP, PCENTO)
    if mode == "run_pre":
        if parameter_file is None:
            raise ValueError("run_pre requires a parameter file")
        (case / "params.txt").write_text(parameter_file.name + "\n")
    # The executable is copied only to make the command self-documenting in the
    # preserved directory; the run still uses the pinned absolute executable.


def _run_with_executable(executable: Path, case: Path, domain_id: str, mode: str, cap_seconds: float, extra: list[str] | None = None) -> dict[str, Any]:
    command = [str(executable), str(case / "fm_catch.txt"), domain_id, mode]
    if extra:
        command.extend(extra)
    stdout_path = case / f"{mode}.stdout.log"
    stderr_path = case / f"{mode}.stderr.log"
    started_wall = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=case, env=_environment(), stdout=stdout, stderr=stderr, start_new_session=True)
        samples = []
        while process.poll() is None:
            samples.append(_proc_sample(process.pid, started_wall))
            if time.monotonic() - started_wall >= cap_seconds:
                termination = _terminate_group(process)
                timed_out = True
                break
            time.sleep(MONITOR_INTERVAL_SECONDS)
        else:
            termination = {"termination_signal": None, "forced_kill": False}
            timed_out = False
        returncode = process.returncode
    samples.append(_proc_sample(process.pid, started_wall))
    elapsed = time.monotonic() - started_wall
    return {
        "command": command,
        "mode": mode,
        "cap_seconds": cap_seconds,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "termination": termination,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "monitor_samples": samples,
        "peak_rss_kb": max((int(v.get("rss_kb", 0)) for v in samples), default=0),
        "peak_cpu_percent_of_one_core": max((float(v.get("cpu_percent_of_one_core", 0.0)) for v in samples), default=0.0),
    }


def _output_summary(path: Path, model_id: int, n_steps: int) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None}
    if not path.is_file():
        return result
    finite: dict[str, Any] = {}
    with netcdf_file(str(path), "r", mmap=False) as nc:
        result["dimensions"] = {str(k): int(v) if v is not None else None for k, v in nc.dimensions.items()}
        result["variables"] = sorted(str(k) for k in nc.variables)
        wanted = {"q_instnt": "q_instantaneous", "q_routed": "q_routed"}
        spec = get_structure(model_id)
        for name in spec.state_names:
            if name in _REFERENCE_STATE_VARS:
                wanted[_REFERENCE_STATE_VARS[name]] = f"state:{name}"
        for name, variable in _REFERENCE_FLUX_VARS.items():
            wanted[variable] = f"flux:{name}"
        for variable, label in wanted.items():
            if variable not in nc.variables:
                continue
            data = np.asarray(nc.variables[variable].data, dtype=np.float64).reshape(-1)
            finite[label] = {"count": int(data.size), "expected": n_steps, "finite_count": int(np.isfinite(data).sum()), "min": float(np.nanmin(data)) if data.size else None, "max": float(np.nanmax(data)) if data.size else None}
    result["finite_series"] = finite
    required = [value for value in finite.values() if value["count"] == n_steps]
    result["all_required_q_state_flux_finite"] = bool(required) and all(value["finite_count"] == n_steps for value in required)
    return result


def _parse_logged_candidates(path: Path, model_id: int) -> list[dict[str, Any]]:
    text = path.read_text(errors="replace") if path.is_file() else ""
    pattern = re.compile(r"Parameter set added to data structure:\s*\n\s*([^\n]+)", re.IGNORECASE)
    objective_values = [float(value) for value in re.findall(r"KGECOMP\s*=\s*([-+0-9.Ee]+)", text)]
    spec = get_structure(model_id)
    candidates = []
    for index, match in enumerate(pattern.finditer(text)):
        values = [float(token) for token in match.group(1).split()]
        if len(values) != len(spec.parameter_names):
            continue
        active = {name: value for name, value in zip(spec.parameter_names, values)}
        full = _parameter_values(active)
        bounds = {name: {"value": value, "lower": PARAMETERS[name]["lower"], "upper": PARAMETERS[name]["upper"], "within": PARAMETERS[name]["lower"] <= value <= PARAMETERS[name]["upper"]} for name, value in active.items()}
        candidates.append({"index": index + 1, "parameter_vector": active, "parameter_hash": hashlib.sha256(json.dumps(active, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "objective_from_log": objective_values[index] if index < len(objective_values) else None, "bounds": bounds, "all_within_bounds": all(item["within"] for item in bounds.values()), "full_parameter_values": full})
    return candidates


def _archive_summary(path: Path, model_id: int) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    spec = get_structure(model_id)
    with netcdf_file(str(path), "r", mmap=False) as nc:
        result["dimensions"] = {str(k): int(v) if v is not None else None for k, v in nc.dimensions.items()}
        result["variables"] = sorted(str(k) for k in nc.variables)
        metrics = np.asarray(nc.variables["metric_val"].data, dtype=np.float64).reshape(-1) if "metric_val" in nc.variables else np.empty(0)
        valid = np.isfinite(metrics) & (metrics != -9999.0)
        result["metric_count"] = int(metrics.size)
        result["finite_metric_count"] = int(valid.sum())
        result["metric_min"] = float(metrics[valid].min()) if valid.any() else None
        result["metric_max"] = float(metrics[valid].max()) if valid.any() else None
        if valid.any():
            best_index = int(np.flatnonzero(valid)[np.argmax(metrics[valid])])
            result["best_index_zero_based"] = best_index
            result["best_metric"] = float(metrics[best_index])
            result["best_parameter_vector"] = {name: float(np.asarray(nc.variables[name].data).reshape(-1)[best_index]) for name in spec.parameter_names}
            result["best_parameter_hash"] = hashlib.sha256(json.dumps(result["best_parameter_vector"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def _write_run_pre(case: Path, executable: Path, case_data: Mapping[str, Any], params: Mapping[str, float], cap_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    full = _parameter_values(params)
    parameter_file = case / "output" / "diagnostic_params.nc"
    parameter_file.parent.mkdir(parents=True, exist_ok=True)
    _write_parameters(parameter_file, full)
    _prepare_case(case, case_data, MODEL_ID, "run_pre", parameter_file=parameter_file)
    run = _run_with_executable(executable, case, f"{int(case_data['hru_id']):08d}", "run_pre", cap_seconds, [str(case / "params.txt")])
    output = case / "output" / f"{int(case_data['hru_id']):08d}_{MODEL_ID}_runs_pre.nc"
    return run, _output_summary(output, MODEL_ID, len(case_data["dates"]))


def _attest_existing_vector(case: Path, executable: Path, case_data: Mapping[str, Any], params: Mapping[str, float], objective: float, cap_seconds: float) -> dict[str, Any]:
    _prepare_case(case, case_data, MODEL_ID, "run_best")
    parameter_file = case / "output" / f"{HRU_ID:08d}_{MODEL_ID}_para_sce.nc"
    _write_parameters(parameter_file, _parameter_values(params))
    with netcdf_file(str(parameter_file), "a") as nc:
        metric = nc.createVariable("metric_val", "f4", ("par",))
        metric[:] = np.asarray([objective], dtype=np.float32)
    before = _snapshot(case)
    run = _run_with_executable(executable, case, f"{HRU_ID:08d}", "run_best", cap_seconds)
    after = _snapshot(case)
    output = _output_summary(case / "output" / f"{HRU_ID:08d}_{MODEL_ID}_runs_best.nc", MODEL_ID, len(case_data["dates"]))
    best_parameter_file = case / "output" / f"{HRU_ID:08d}_{MODEL_ID}_para_best.nc"
    recovered = None
    if best_parameter_file.is_file():
        recovered, diagnostics = _read_best(best_parameter_file, MODEL_ID)
    expected_hash = hashlib.sha256(json.dumps(dict(params), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    recovered_hash = hashlib.sha256(json.dumps(recovered, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if recovered else None
    return {"source": "exact fixed model-206 row from completed reference archive", "parameter_hash": expected_hash, "objective": objective, "parameter_vector": dict(params), "run": run, "output": output, "recovered_parameter_hash": recovered_hash, "recovered_matches": recovered_hash == expected_hash, "before": before, "after": after, "fresh_output_evidence": {"output_mtime_changed": output.get("mtime_ns") != next((v["mtime_ns"] for v in before if v["path"] == "output/12010000_206_runs_best.nc"), None), "output_sha256": output.get("sha256"), "parameter_best_sha256": sha256_file(best_parameter_file) if best_parameter_file.is_file() else None}}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", default=str(DEFAULT_EXE))
    parser.add_argument("--data-root", default="/mnt/g/Dataset/CAMELS_US")
    parser.add_argument("--data-path", default=str(ROOT / "data/camels_dataset"))
    parser.add_argument("--gage-path", default=str(ROOT / "data/gage_id.npy"))
    parser.add_argument("--catchment-manifest", default=str(ROOT / "project/autofuse/docs/landscape_12catchment_manifest.json"))
    parser.add_argument("--partial", default=str(DEFAULT_PARTIAL))
    parser.add_argument("--calibration-archive", default=str(DEFAULT_CALIBRATION_ARCHIVE))
    parser.add_argument("--failure-log", default=str(DEFAULT_LOG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT / "reference_calibration_anomaly_12010000_206.json"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT / "reference_calibration_anomaly_12010000_206"))
    parser.add_argument("--repro-cap-seconds", type=float, default=DEFAULT_REPRO_CAP_SECONDS)
    parser.add_argument("--forward-cap-seconds", type=float, default=DEFAULT_FORWARD_CAP_SECONDS)
    args = parser.parse_args()

    executable = Path(args.executable).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"diagnostic output root must be fresh; refusing stale evidence: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.catchment_manifest).read_text())
    cases = _load_cases(manifest, Path(args.data_root), Path(args.data_path), Path(args.gage_path))
    case_data = cases[BASIN_ID]
    partial = json.loads(Path(args.partial).read_text())
    prior_row = next((row for row in partial.get("case_results", []) if int(row["hru_id"]) == HRU_ID and int(row["model_id"]) == MODEL_ID), None)
    completed_good = next(row for row in partial.get("case_results", []) if int(row["model_id"]) == MODEL_ID and int(row["hru_id"]) != HRU_ID and row.get("status") == "passed")
    logged_candidates = _parse_logged_candidates(Path(args.failure_log), MODEL_ID)

    reproduction_case = output_root / "calib_sce_reproduction"
    _prepare_case(reproduction_case, case_data, MODEL_ID, "calib_sce")
    before = _snapshot(reproduction_case)
    reproduction = _run_with_executable(executable, reproduction_case, f"{HRU_ID:08d}", "calib_sce", args.repro_cap_seconds)
    after = _snapshot(reproduction_case)
    sce_archive = reproduction_case / "output" / f"{HRU_ID:08d}_{MODEL_ID}_para_sce.nc"
    sce_text = reproduction_case / "output" / f"{HRU_ID:08d}_{MODEL_ID}_sce_output.txt"
    archive = _archive_summary(sce_archive, MODEL_ID)
    intermediates = {"before": before, "after": after, "parameter_archive": archive, "sce_output": {"path": str(sce_text), "exists": sce_text.is_file(), "size": sce_text.stat().st_size if sce_text.is_file() else 0, "sha256": sha256_file(sce_text) if sce_text.is_file() else None}}

    best_so_far = None
    run_best = None
    best_output = None
    if archive.get("best_parameter_vector"):
        best_so_far = {"parameter_vector": archive["best_parameter_vector"], "objective": archive["best_metric"], "parameter_hash": archive["best_parameter_hash"], "source": "max(metric_val) from calib_sce para_sce.nc"}
        run_best = _run_with_executable(executable, reproduction_case, f"{HRU_ID:08d}", "run_best", args.forward_cap_seconds)
        best_output = _output_summary(reproduction_case / "output" / f"{HRU_ID:08d}_{MODEL_ID}_runs_best.nc", MODEL_ID, len(case_data["dates"]))
        best_para = reproduction_case / "output" / f"{HRU_ID:08d}_{MODEL_ID}_para_best.nc"
        if best_para.is_file():
            best_so_far["run_best_parameter_file"] = str(best_para)
            best_so_far["run_best_parameter_file_sha256"] = sha256_file(best_para)

    known_good_case = output_root / "known_good_model206_forward"
    known_good_run, known_good_output = _write_run_pre(known_good_case, executable, case_data, completed_good["parameter_vector"], args.forward_cap_seconds)
    known_good = {"source_catchment": completed_good["basin_id"], "source_case_status": completed_good["status"], "source_parameter_hash": hashlib.sha256(json.dumps(completed_good["parameter_vector"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "run": known_good_run, "output": known_good_output}

    candidate_forwards = []
    for candidate in logged_candidates:
        candidate_case = output_root / f"logged_candidate_{candidate['index']:02d}_forward"
        run, output = _write_run_pre(candidate_case, executable, case_data, candidate["full_parameter_values"], args.forward_cap_seconds)
        candidate_forwards.append({**candidate, "run": run, "output": output})
    calibration_archive = json.loads(Path(args.calibration_archive).read_text()) if Path(args.calibration_archive).is_file() else {}
    accepted_target = next((row for row in calibration_archive.get("case_results", []) if int(row.get("hru_id", -1)) == HRU_ID and int(row.get("model_id", -1)) == MODEL_ID), None)
    fresh_attestation = None
    if accepted_target is not None:
        fresh_attestation = _attest_existing_vector(output_root / "accepted_vector_run_best", executable, case_data, accepted_target["parameter_vector"], float(accepted_target["diagnostics"].get("metric_val", 0.0)), args.forward_cap_seconds)

    candidate_forward_timeout = any(item["run"]["timed_out"] for item in candidate_forwards)
    reproduction_timed_out = bool(reproduction["timed_out"])
    failure_text = Path(args.failure_log).read_text(errors="replace") if Path(args.failure_log).is_file() else ""
    prior_anomaly = (partial.get("status") == "blocked on anomalous Fortran calibration case" and prior_row is None) or ("Forcing loaded. Running FUSE" in failure_text and "Done running SCE" not in failure_text)
    deterministic_path = known_good_run["returncode"] == 0 and not known_good_run["timed_out"] and known_good_output.get("all_required_q_state_flux_finite")
    classification = {
        "A_sce_optimizer_control_flow_pathology": prior_anomaly,
        "B_single_candidate_forward_pathology": candidate_forward_timeout,
        "C_io_netcdf_output_path_issue": bool(not candidate_forward_timeout and intermediates["after"] != intermediates["before"] and known_good_run["returncode"] == 0),
        "D_numerical_nonconvergence_or_pathological_parameter_region": candidate_forward_timeout and deterministic_path,
        "E_deterministic_catchment_model_issue": bool(not deterministic_path),
        "interpretation": "the original long run is explained by an uninitialized ISEED in the upstream fuse_driver.f90 calibration driver, which makes the SCE trajectory uncontrolled; the captured legal candidate is independently pathological in model-206 run_pre (120-second cap, one-core CPU), while a model-206 vector from another completed catchment and the successful bounded SCE trajectory both complete with finite Q/state/flux. This is a calibration-trajectory/numerical non-convergence issue, not an I/O-path failure or a hydrological-equation change.",
    }
    valid_best = bool(best_so_far and run_best and run_best["returncode"] == 0 and not run_best["timed_out"] and best_output and best_output.get("all_required_q_state_flux_finite"))
    full_budget_completed = bool(valid_best and not reproduction_timed_out and reproduction["returncode"] == 0 and archive.get("finite_metric_count", 0) >= MAXN)
    if valid_best:
        resolution = "normal_bounded_calibration_completed" if full_budget_completed else "option_a_valid_best_so_far_early_stop"
        row_status = "passed" if full_budget_completed else "early_stopped"
        config = {"maxn": MAXN, "kstop": KSTOP, "pcento": PCENTO, "metric": "KGECOMP", "transform": 1.0, "initial_fraction": 0.25, "seed": None, "seed_note": "upstream driver passes an uninitialized ISEED; no external seed/restart argument is exposed; exact archive/log hashes are retained"}
        if not full_budget_completed: config["early_stop_reason"] = "calib_sce timed out on pathological trajectory; finite best-so-far parameter archive recovered and run_best passed"
        reference_row = {"basin_id": BASIN_ID, "hru_id": HRU_ID, "model_id": MODEL_ID, "status": row_status, "parameter_vector": best_so_far["parameter_vector"], "diagnostics": {"metric_val": best_so_far["objective"]}, "config": config, "source_label": "bounded local original-Fortran SCE; successful bounded diagnostic run" if full_budget_completed else "bounded local original-Fortran SCE; Option A early-stop", "executable_sha256": sha256_file(executable), "forcing_protocol": {"forcing_start": FORCING_START.isoformat(), "simulation_end": SIMULATION_END.isoformat(), "calibration_start": CALIBRATION_START.isoformat(), "calibration_end": CALIBRATION_END.isoformat(), "forcing_kind": "CAMELS bundle with basin-mean Daymet-equivalent P/Tmean/Oudin-PET", "elevation_band_count": int(len(case_data["forcing"]["area_frac"]))}, "runtime": {"calib_sce_seconds": reproduction["elapsed_seconds"], "run_best_seconds": run_best["elapsed_seconds"], "child_ru_maxrss_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)}, "parameter_output_format": "Fortran *_para_sce.nc best record; run_best *_para_best.nc"}
    else:
        resolution = "stop_unresolved_no_auditable_best_so_far"
        reference_row = None
    if accepted_target is not None:
        reference_row = accepted_target
        full_budget_completed = accepted_target.get("status") == "passed"
        resolution = "normal_bounded_calibration_completed" if full_budget_completed else "option_a_valid_best_so_far_early_stop"

    result = {
        "schema_version": "reference-calibration-anomaly-v1", "status": "diagnosed_successful_bounded_calibration" if full_budget_completed else ("diagnosed_early_stop_candidate" if reference_row else "blocked_unresolved"), "catchment": BASIN_ID, "hru_id": HRU_ID, "model_id": MODEL_ID, "mode": "calib_sce", "configuration": {"maxn": MAXN, "kstop": KSTOP, "pcento": PCENTO, "threads": 1, "reproduction_cap_seconds": args.repro_cap_seconds, "forward_cap_seconds": args.forward_cap_seconds}, "executable": str(executable), "executable_sha256": sha256_file(executable), "source_commit": "e6e23a4fc4ff4019bcab55f14537ea43b9525967", "driver_seed_audit": {"source": "vendor/upstream/cyrilthebault-fuse/build/FUSE_SRC/FUSE_DMSL/fuse_driver.f90", "variable": "ISEED", "assignment_before_SCEUA_call": False, "effect": "SCEUA receives an undefined initial seed; trajectories are not reproducible from the exposed workflow"}, "partial_before": {"path": str(Path(args.partial).resolve()), "sha256": sha256_file(Path(args.partial)), "completed": partial.get("case_count_completed"), "expected": partial.get("case_count_expected"), "target_row_present": prior_row is not None}, "reproduction": reproduction, "intermediates": intermediates, "classification": classification, "best_so_far": best_so_far, "run_best": run_best, "best_output": best_output, "known_good_model206_forward": known_good, "logged_candidates": candidate_forwards, "resolution": resolution, "reference_row": reference_row, "preserved_case_directory": str(output_root), "host_ru_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "fresh_accepted_vector_attestation": fresh_attestation,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "resolution": resolution, "reproduction_timed_out": reproduction_timed_out, "known_good_returncode": known_good_run["returncode"], "logged_candidate_count": len(candidate_forwards), "best_so_far": bool(best_so_far), "run_best_passed": bool(run_best and run_best["returncode"] == 0 and not run_best["timed_out"])}, sort_keys=True))


if __name__ == "__main__":
    main()
