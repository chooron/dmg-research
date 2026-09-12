"""Torch-only Gate 1 for the frozen coupled-RK2/FIX_STATES kernel.

This is a bounded readiness smoke, not a landscape or calibration launcher.  The
parent runs one isolated worker per structure, serially, so Inductor's native
compiler state cannot accumulate across the 78 signatures.  Each worker builds
and fullgraph-compiles one coupled-RK2 step, then runs both frozen catchments for
the complete 1987-01-01..2009-12-31 forcing window.
"""
from __future__ import annotations

import argparse
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

from dfuse import (
    GraphSignature,
    get_structure,
    reset_compile_diagnostics,
    runtime_compile_diagnostics,
    simulate_coupled_rk2,
)
from dfuse.kernel import _capacity, _parameter_values
from dfuse.runtime import get_generated_step
from dfuse.spec import PARAMETER_NAMES, STATE_NAMES, enumerate_structures

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
CATALOGUE = ROOT / "dfuse/specs/structures_78.json"
MANIFEST = DOCS / "landscape_12catchment_manifest.json"
INPUT_INDEX = DOCS / "landscape_inputs/index.json"
CALIBRATION = DOCS / "reference_calibration_12x78.json"
DEFAULT_OUTPUT = DOCS / "torch_fuse_78_long_horizon_smoke.json"
DEFAULT_PARTIAL = DOCS / "torch_fuse_78_long_horizon_smoke.partial.json"
DEFAULT_CACHE = ROOT / "project/autofuse/.cache/torch-fuse-78-long-horizon"
WORKER_DIR_NAME = "torch-fuse-78-workers"
CATCHMENT_IDS = ("USA_09447800", "USA_14138900")
GRADIENT_MODEL_IDS = (2, 6, 10, 14, 20, 26, 50, 84, 90, 96, 146, 214)
FORCING_START = date(1987, 1, 1)
SIMULATION_END = date(2009, 12, 31)
DT_DAYS = 1.0
INITIAL_FRACTION = 0.25
WATER_BALANCE_TOLERANCE = 1.0e-8
CAPACITY_TOLERANCE = 1.0e-12
RSS_STOP_KB = 3_500_000
COMPILE_TIMEOUT_SECONDS = 1800

_STATE_CAPACITY = {
    "TENS_1A": "MAXTENS_1A",
    "TENS_1B": "MAXTENS_1B",
    "TENS_1": "MAXTENS_1",
    "FREE_1": "MAXFREE_1",
    "WATR_1": "MAXWATR_1",
    "TENS_2": "MAXTENS_2",
    "FREE_2A": "MAXFREE_2A",
    "FREE_2B": "MAXFREE_2B",
    "WATR_2": "MAXWATR_2",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dates() -> list[date]:
    return [FORCING_START + timedelta(days=i) for i in range((SIMULATION_END - FORCING_START).days + 1)]


def _cache_info(path: Path) -> dict[str, int]:
    files = 0
    size = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
                size += item.stat().st_size
    return {"files": files, "bytes": size}


def _set_environment(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
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


def _rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _load_frozen_inputs() -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[tuple[int, int], dict[str, Any]]]:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("status") != "frozen before solver comparison":
        raise RuntimeError("landscape manifest is not the frozen pre-solver-comparison manifest")
    manifest_rows = {row["basin_id"]: row for row in manifest.get("catchments", [])}
    if set(CATCHMENT_IDS) - set(manifest_rows):
        raise RuntimeError("required frozen catchment is absent from the 12-catchment manifest")

    input_index = json.loads(INPUT_INDEX.read_text())
    if input_index.get("status") != "prepared":
        raise RuntimeError("prepared landscape input index is not available")
    input_rows = {row["basin_id"]: row for row in input_index.get("rows", [])}
    inputs: dict[str, dict[str, np.ndarray]] = {}
    for basin_id in CATCHMENT_IDS:
        row = input_rows.get(basin_id)
        if row is None:
            raise RuntimeError(f"prepared input is missing {basin_id}")
        path = ROOT / row["path"] if not Path(row["path"]).is_absolute() else Path(row["path"])
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"prepared input hash mismatch for {basin_id}: {path}")
        with np.load(path, allow_pickle=False) as archive:
            loaded = {name: np.asarray(archive[name], dtype=np.float64) for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}
        inputs[basin_id] = loaded
        expected_steps = len(_dates())
        if any(loaded[name].size != expected_steps for name in ("ppt", "pet", "temp", "q_obs")):
            raise RuntimeError(f"forcing length mismatch for {basin_id}")
        if not all(np.isfinite(loaded[name]).all() for name in ("ppt", "pet", "temp", "q_obs")):
            raise RuntimeError(f"non-finite frozen forcing/Qobs for {basin_id}")

    theta_cache = os.environ.get("TORCH_FUSE_THETA_CACHE")
    calibration = json.loads(Path(theta_cache).read_text()) if theta_cache else json.loads(CALIBRATION.read_text())
    if theta_cache and calibration.get("source_sha256") != _sha256(CALIBRATION):
        raise RuntimeError("compact theta cache does not match the frozen calibration archive")
    expected_theta_rows = 936 if not theta_cache else len(CATCHMENT_IDS) * 78
    if calibration.get("status") != "complete" or int(calibration.get("case_count_expected", 0)) != expected_theta_rows or int(calibration.get("case_count_completed", 0)) != expected_theta_rows:
        raise RuntimeError("fixed reference theta archive is incomplete")
    theta_rows = {(int(row["hru_id"]), int(row["model_id"])): row for row in calibration.get("case_results", [])}
    if len(theta_rows) != expected_theta_rows:
        raise RuntimeError("fixed reference theta archive has duplicate or missing rows")
    selected_hru = {int(manifest_rows[basin_id]["hru_id"]) for basin_id in CATCHMENT_IDS}
    for hru_id in selected_hru:
        for spec in enumerate_structures():
            row = theta_rows.get((hru_id, spec.model_id))
            if row is None or row.get("status") not in ("passed", "early_stopped"):
                raise RuntimeError(f"missing legal theta for hru/model {hru_id}/{spec.model_id}")
            if set(row.get("parameter_vector", {})) - set(PARAMETER_NAMES):
                raise RuntimeError(f"theta archive has unknown parameter for hru/model {hru_id}/{spec.model_id}")
    return {"manifest": manifest, "input_index": input_index, "calibration": calibration}, inputs, theta_rows


def _topology_matrix() -> dict[str, list[str]]:
    specs = enumerate_structures()
    return {
        "ARCH1": sorted({spec.decisions["ARCH1"] for spec in specs}),
        "ARCH2": sorted({spec.decisions["ARCH2"] for spec in specs}),
        "QPERC": sorted({spec.decisions["QPERC"] for spec in specs}),
        "QSURF": sorted({spec.decisions["QSURF"] for spec in specs}),
    }


def _quality(result: Any, model_id: int, params: Mapping[str, Any], spec: Any) -> dict[str, Any]:
    state = result.states.detach().cpu().numpy().astype(np.float64)
    q = result.q.detach().cpu().numpy().astype(np.float64)
    q_instantaneous = result.q_instantaneous.detach().cpu().numpy().astype(np.float64)
    fluxes = {name: value.detach().cpu().numpy().astype(np.float64) for name, value in result.fluxes.items()}
    water_balance = result.water_balance_residual.detach().cpu().numpy().astype(np.float64)
    snow_balance = result.snow_balance_residual.detach().cpu().numpy().astype(np.float64)
    arrays = [q, q_instantaneous, state, water_balance, snow_balance, *fluxes.values()]
    nonfinite_count = int(sum(np.count_nonzero(~np.isfinite(array)) for array in arrays))
    parameter_values = _parameter_values(params, dtype=torch.float64, device=torch.device("cpu"))
    capacities = {name: float(value.detach()) for name, value in _capacity(parameter_values).items()}
    negative_count = int(np.count_nonzero(state < 0.0))
    violations = {}
    bounded_state_names = []
    unbounded_state_names = []
    for index, name in enumerate(spec.state_names):
        if name == "WATR_2" and spec.decisions["ARCH2"] != "fixedsiz_2":
            unbounded_state_names.append(name)
            continue
        bounded_state_names.append(name)
        maximum = capacities[_STATE_CAPACITY[name]]
        excess = state[:, index] - maximum
        count = int(np.count_nonzero(excess > CAPACITY_TOLERANCE))
        if count:
            violations[name] = {"count": count, "max_excess": float(np.max(excess))}
    dates = _dates()
    def first_residual_detail(values: np.ndarray, threshold: float, kind: str) -> dict[str, Any] | None:
        hits = np.flatnonzero(np.abs(values) > threshold)
        if not len(hits):
            return None
        index = int(hits[0])
        detail: dict[str, Any] = {"kind": kind, "threshold": threshold, "index": index, "date": dates[index].isoformat(), "residual": float(values[index]), "state_names": list(spec.state_names), "state_start": state[index].tolist(), "state_next": state[index + 1].tolist()}
        if kind == "water_balance":
            detail["involved_fluxes"] = {name: float(fluxes[name][index]) for name in ("EFF_PPT", "EVAP_1", "EVAP_2", "QSURF", "QINTF_1", "OFLOW_1", "OFLOW_2", "QBASE_2", "QPERC_12")}
        else:
            detail["snow_start"] = float(result.snow[index].detach().cpu())
            detail["snow_next"] = float(result.snow[index + 1].detach().cpu())
        return detail
    first_water_balance_violation = first_residual_detail(water_balance, WATER_BALANCE_TOLERANCE, "water_balance")
    first_snow_balance_violation = first_residual_detail(snow_balance, WATER_BALANCE_TOLERANCE, "snow_balance")
    return {
        "model_id": model_id,
        "completed_full_period": bool(q.shape[0] == len(_dates()) and state.shape[0] == len(_dates()) + 1),
        "output_shapes": {"q": list(q.shape), "q_instantaneous": list(q_instantaneous.shape), "states": list(state.shape), "fluxes": {name: list(value.shape) for name, value in fluxes.items()}},
        "finite": nonfinite_count == 0,
        "nonfinite_count": nonfinite_count,
        "negative_active_state_count": negative_count,
        "min_active_state": float(np.min(state)),
        "capacity_violation_count": int(sum(item["count"] for item in violations.values())),
        "capacity_violations": violations,
        "bounded_state_names": bounded_state_names,
        "unbounded_by_design_state_names": unbounded_state_names,
        "water_balance_max_abs": float(np.max(np.abs(water_balance))),
        "snow_balance_max_abs": float(np.max(np.abs(snow_balance))),
        "first_water_balance_violation": first_water_balance_violation,
        "first_snow_balance_violation": first_snow_balance_violation,
        "quality_pass": bool(
            q.shape[0] == len(_dates())
            and state.shape[0] == len(_dates()) + 1
            and nonfinite_count == 0
            and negative_count == 0
            and not violations
            and float(np.max(np.abs(water_balance))) <= WATER_BALANCE_TOLERANCE
            and float(np.max(np.abs(snow_balance))) <= WATER_BALANCE_TOLERANCE
        ),
    }


def _compile_audit() -> dict[str, Any]:
    diagnostics = runtime_compile_diagnostics()
    records = list(diagnostics.get("records", {}).values())
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


def _worker(cache_dir: Path, output: Path, model_id: int) -> dict[str, Any]:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("78-structure Torch smoke requires CUDA; refusing CPU fallback")
    _set_environment(cache_dir)
    reset_compile_diagnostics()
    source_meta, inputs, theta_rows = _load_frozen_inputs()
    specs = {spec.model_id: spec for spec in enumerate_structures()}
    spec = specs[model_id]
    signature = GraphSignature.from_structure(spec, sequential_order=("coupled_rhs",), n_substeps=1, execution_mode="coupled_rk2")
    generated_signature, generated = get_generated_step(spec, order=("coupled_rhs",), n_substeps=1, execution_mode="coupled_rk2")
    if generated_signature != signature:
        raise AssertionError(f"generated signature mismatch for model {model_id}")
    dates = _dates()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    compile_probe = None
    runs = []
    for basin_id in CATCHMENT_IDS:
        hru_id = int(source_meta["manifest"]["catchments"][[row["basin_id"] for row in source_meta["manifest"]["catchments"]].index(basin_id)]["hru_id"])
        theta = theta_rows[(hru_id, model_id)]["parameter_vector"]
        values = inputs[basin_id]
        tensor_forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1), dtype=torch.float64, device=device)
        if compile_probe is None:
            probe_result = simulate_coupled_rk2(model_id, tensor_forcing[:1], theta, initial_fraction=INITIAL_FRACTION, dates=dates[:1], dt_days=DT_DAYS, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
            compile_probe = {"finite": bool(torch.isfinite(probe_result.q).all().item()), "completed": True}
            del probe_result
        run_started = time.perf_counter()
        result = simulate_coupled_rk2(model_id, tensor_forcing, theta, initial_fraction=INITIAL_FRACTION, dates=dates, dt_days=DT_DAYS, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
        torch.cuda.synchronize(device)
        quality = _quality(result, model_id, theta, spec)
        quality.update({"basin_id": basin_id, "hru_id": hru_id, "elapsed_seconds": time.perf_counter() - run_started})
        runs.append(quality)
        del result, tensor_forcing
        torch.cuda.empty_cache()
    compile_audit = _compile_audit()
    compile_pass = bool(
        compile_audit["record_count"] == 1
        and compile_audit["compile_attempts"] == 1
        and compile_audit["compile_successes"] == 1
        and compile_audit["fallbacks"] == 0
        and compile_audit["graph_breaks"] == 0
        and compile_audit["recompilations"] == 0
        and compile_audit["autograd_recompilations"] == 0
    )
    quality_pass = bool(len(runs) == len(CATCHMENT_IDS) and all(row["quality_pass"] for row in runs))
    resource_row = {
        "host_rss_current_kb": _rss_kb(),
        "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cache": _cache_info(cache_dir),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if resource_row["host_peak_rss_kb"] > RSS_STOP_KB:
        raise RuntimeError(f"resource safety stop: {resource_row}")
    payload = {
        "status": "passed" if compile_pass and quality_pass else "failed",
        "model_id": model_id,
        "topology": dict(spec.decisions),
        "graph_signature": signature.to_dict(),
        "graph_signature_digest": signature.digest,
        "generated_source_sha256": _canonical_hash(getattr(generated, "generated_source", "")),
        "compile_probe": compile_probe,
        "compile_audit": compile_audit,
        "runs": runs,
        "compile_pass": compile_pass,
        "quality_pass": quality_pass,
        "resource": resource_row,
        "protocol": {"forcing_start": FORCING_START.isoformat(), "simulation_end": SIMULATION_END.isoformat(), "dt_days": DT_DAYS, "initial_fraction": INITIAL_FRACTION, "dtype": "torch.float64", "device": str(device), "compile_backend": "inductor", "compile_fullgraph": True},
    }
    _write_json(output, payload)
    print(json.dumps({"model_id": model_id, "status": payload["status"], "compile_pass": compile_pass, "quality_pass": quality_pass}, sort_keys=True), flush=True)
    return payload


def _selection_metadata(specs: list[Any]) -> dict[str, Any]:
    return {
        "catchments": list(CATCHMENT_IDS),
        "structure_ids": [spec.model_id for spec in specs],
        "structure_count": len(specs),
        "run_count_expected": len(CATCHMENT_IDS) * len(specs),
        "gradient_model_ids": list(GRADIENT_MODEL_IDS),
        "gradient_selection_rule": "frozen before execution: cover every ARCH1, ARCH2, QPERC, and QSURF branch plus their requested sensitive combinations; no result-dependent selection",
        "topology_values": _topology_matrix(),
    }


def _worker_command(cache_dir: Path, output: Path, model_id: int) -> list[str]:
    return [sys.executable, "-m", "project.autofuse.torch_fuse_78_long_horizon_smoke", "--mode", "worker", "--model-id", str(model_id), "--cache-dir", str(cache_dir), "--worker-output", str(output)]


def _run_parent(output: Path, partial: Path, cache_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("78-structure Torch smoke requires CUDA; refusing CPU fallback")
    _set_environment(cache_dir)
    metadata, _, theta_rows = _load_frozen_inputs()
    theta_cache_path = cache_dir / "theta-cache.json"
    selected_hru = {int(row["hru_id"]) for row in metadata["manifest"]["catchments"] if row["basin_id"] in CATCHMENT_IDS}
    selected_theta_rows = [row for (hru_id, _), row in theta_rows.items() if hru_id in selected_hru]
    _write_json(theta_cache_path, {"status": "complete", "case_count_expected": len(selected_theta_rows), "case_count_completed": len(selected_theta_rows), "source_sha256": _sha256(CALIBRATION), "case_results": sorted(selected_theta_rows, key=lambda row: (int(row["hru_id"]), int(row["model_id"])))})
    specs = list(enumerate_structures())
    if len(specs) != 78 or len({spec.model_id for spec in specs}) != 78:
        raise RuntimeError("strict 78-structure catalogue check failed")
    selection = _selection_metadata(specs)
    worker_dir = cache_dir / WORKER_DIR_NAME
    worker_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reports = []
    failures = []
    for index, spec in enumerate(specs, start=1):
        worker_output = worker_dir / f"{index:03d}-{spec.model_id}.json"
        command = _worker_command(cache_dir, worker_output, spec.model_id)
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
        env["TORCH_FUSE_THETA_CACHE"] = str(theta_cache_path.resolve())
        child = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=COMPILE_TIMEOUT_SECONDS, check=False)
        if child.returncode != 0 or not worker_output.is_file():
            failure = {"model_id": spec.model_id, "topology": dict(spec.decisions), "returncode": child.returncode, "stderr_tail": child.stderr[-4000:], "stdout_tail": child.stdout[-2000:]}
            failures.append(failure)
            partial_payload = {"schema_version": "torch-fuse-78-long-horizon-smoke-v1", "status": "failed", "selection": selection, "completed_structure_count": len(reports), "reports": reports, "failures": failures, "stopped_after_failure": True, "failure": failure}
            _write_json(partial, partial_payload)
            _write_json(output, partial_payload)
            return partial_payload
        report = json.loads(worker_output.read_text())
        reports.append(report)
        if report.get("status") != "passed":
            failure = {"model_id": spec.model_id, "topology": dict(spec.decisions), "error_type": "gate_failure", "report_status": report.get("status"), "compile_audit": report.get("compile_audit"), "runs": report.get("runs"), "failure_detail": next((run.get("first_water_balance_violation") for run in report.get("runs", []) if run.get("first_water_balance_violation") is not None), None)}
            failures.append(failure)
            partial_payload = {"schema_version": "torch-fuse-78-long-horizon-smoke-v1", "status": "failed", "selection": selection, "completed_structure_count": len(reports), "reports": reports, "failures": failures, "stopped_after_failure": True, "failure": failure}
            _write_json(partial, partial_payload)
            _write_json(output, partial_payload)
            return partial_payload
        worker_output.unlink()
        partial_payload = {"schema_version": "torch-fuse-78-long-horizon-smoke-v1", "status": "partial", "selection": selection, "completed_structure_count": len(reports), "reports": reports, "failures": failures}
        _write_json(partial, partial_payload)
        print(json.dumps({"structure_index": index, "model_id": spec.model_id, "status": "passed"}, sort_keys=True), flush=True)
    compile_pass = all(report.get("compile_pass") for report in reports) and len(reports) == 78
    run_rows = [run for report in reports for run in report.get("runs", [])]
    long_run_pass = len(run_rows) == 156 and all(run.get("quality_pass") for run in run_rows)
    finite_count = sum(bool(run.get("finite")) for run in run_rows)
    negative_case_count = sum(int(run.get("negative_active_state_count", 0)) > 0 for run in run_rows)
    capacity_clean_count = sum(int(run.get("capacity_violation_count", 0)) == 0 for run in run_rows)
    gradient_coverage = {"selected_model_ids": list(GRADIENT_MODEL_IDS), "status": "pending", "selection_frozen_before_execution": True}
    payload = {
        "schema_version": "torch-fuse-78-long-horizon-smoke-v1",
        "status": "passed" if compile_pass and long_run_pass else "failed",
        "gate": "78-structure Torch-FUSE long-horizon smoke",
        "selection": selection,
        "source": {
            "catalogue": str(CATALOGUE), "catalogue_sha256": _sha256(CATALOGUE),
            "catchment_manifest": str(MANIFEST), "catchment_manifest_sha256": _sha256(MANIFEST),
            "input_index": str(INPUT_INDEX), "input_index_sha256": _sha256(INPUT_INDEX), "inputs_sha256": metadata["input_index"].get("inputs_sha256"),
            "calibration_archive": str(CALIBRATION), "calibration_archive_sha256": _sha256(CALIBRATION),
        },
        "protocol": {"forcing_start": FORCING_START.isoformat(), "simulation_end": SIMULATION_END.isoformat(), "n_steps": len(_dates()), "dt_days": DT_DAYS, "initial_fraction": INITIAL_FRACTION, "device": "cuda", "dtype": "torch.float64", "cpu_threads": 1, "cpu_interop_threads": 1, "compile_backend": "inductor", "compile_fullgraph": True, "execution_mode": "coupled_rk2", "persistent_cache": str(cache_dir.resolve())},
        "build_compile_coverage": {"build_success_count": len(reports), "compile_attempts": sum(report["compile_audit"]["compile_attempts"] for report in reports), "compile_successes": sum(report["compile_audit"]["compile_successes"] for report in reports), "fallbacks": sum(report["compile_audit"]["fallbacks"] for report in reports), "graph_breaks": sum(report["compile_audit"]["graph_breaks"] for report in reports), "recompilations": sum(report["compile_audit"]["recompilations"] for report in reports), "autograd_recompilations": sum(report["compile_audit"]["autograd_recompilations"] for report in reports), "all_fullgraph_clean": compile_pass},
        "long_horizon": {"expected_runs": 156, "completed_runs": len(run_rows), "finite_run_count": finite_count, "negative_state_case_count": negative_case_count, "capacity_clean_count": capacity_clean_count, "max_water_balance_abs": max((float(run["water_balance_max_abs"]) for run in run_rows), default=math.inf), "max_snow_balance_abs": max((float(run["snow_balance_max_abs"]) for run in run_rows), default=math.inf), "all_outputs_finite": finite_count == 156, "all_quality_checks_pass": long_run_pass},
        "gradient_smoke": gradient_coverage,
        "reports": reports,
        "failures": failures,
        "resource": {"host_peak_rss_kb": max((int(report["resource"]["host_peak_rss_kb"]) for report in reports), default=0), "gpu_peak_allocated_bytes": max((int(report["resource"]["gpu_peak_allocated_bytes"]) for report in reports), default=0), "gpu_peak_reserved_bytes": max((int(report["resource"]["gpu_peak_reserved_bytes"]) for report in reports), default=0), "wall_clock_seconds": time.perf_counter() - started, "cache_final": _cache_info(cache_dir), "rss_stop_kb": RSS_STOP_KB},
        "gate_decision": {"build_compile_pass": compile_pass, "long_horizon_pass": long_run_pass, "gradient_smoke_pass": False, "passed": False, "reason": "gradient smoke is executed in the next serial phase before final Gate 1 decision"},
    }
    _write_json(output, payload)
    return payload


def _make_gradient_params(theta: Mapping[str, Any], *, device: torch.device) -> dict[str, torch.Tensor]:
    defaults = {name: float(theta.get(name, 0.0)) for name in PARAMETER_NAMES}
    return {name: torch.tensor(value, dtype=torch.float64, device=device, requires_grad=True) for name, value in defaults.items()}


def _gradient_probe(model_id: int, basin_id: str, cache_dir: Path, inputs: Mapping[str, np.ndarray], theta: Mapping[str, Any], hru_id: int) -> dict[str, Any]:
    _set_environment(cache_dir)
    reset_compile_diagnostics()
    device = torch.device("cuda")
    values = inputs[basin_id]
    dates = _dates()[:8]
    forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1)[:8], dtype=torch.float64, device=device)
    spec = get_structure(model_id)
    eager_params = _make_gradient_params(theta, device=device)
    compiled_params = _make_gradient_params(theta, device=device)
    eager = simulate_coupled_rk2(model_id, forcing, eager_params, initial_fraction=INITIAL_FRACTION, dates=dates, dt_days=DT_DAYS, compile_step=False)
    compiled = simulate_coupled_rk2(model_id, forcing, compiled_params, initial_fraction=INITIAL_FRACTION, dates=dates, dt_days=DT_DAYS, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    eager_loss = eager.q.sum()
    compiled_loss = compiled.q.sum()
    eager_grads = torch.autograd.grad(eager_loss, tuple(eager_params.values()), allow_unused=True)
    compiled_grads = torch.autograd.grad(compiled_loss, tuple(compiled_params.values()), allow_unused=True)
    active = set(spec.parameter_names)
    active_diffs = []
    inactive_values = []
    finite = True
    for name, eager_grad, compiled_grad in zip(PARAMETER_NAMES, eager_grads, compiled_grads):
        for grad in (eager_grad, compiled_grad):
            if grad is not None:
                finite = finite and bool(torch.isfinite(grad).all().item())
        if name in active:
            left = torch.zeros((), device=device, dtype=torch.float64) if eager_grad is None else eager_grad
            right = torch.zeros((), device=device, dtype=torch.float64) if compiled_grad is None else compiled_grad
            active_diffs.append(float((left - right).abs().max().detach().cpu()))
        elif compiled_grad is not None:
            inactive_values.append(float(compiled_grad.abs().max().detach().cpu()))
    q_diff = float((eager.q.detach() - compiled.q.detach()).abs().max().cpu())
    state_diff = float((eager.states.detach() - compiled.states.detach()).abs().max().cpu())
    flux_diff = max((float((eager.fluxes[name].detach() - compiled.fluxes[name].detach()).abs().max().cpu()) for name in eager.fluxes), default=0.0)
    audit = _compile_audit()
    result = {
        "model_id": model_id, "basin_id": basin_id, "hru_id": hru_id, "topology": dict(spec.decisions), "status": "passed",
        "forward_finite": bool(torch.isfinite(eager.q).all().item() and torch.isfinite(compiled.q).all().item()),
        "backward_finite": finite,
        "active_gradient_max_abs": max(active_diffs, default=0.0),
        "inactive_gradient_max_abs": max(inactive_values, default=0.0),
        "inactive_gradients_zero_or_none": max(inactive_values, default=0.0) == 0.0,
        "compiled_eager_parity": {"q_max_abs": q_diff, "state_max_abs": state_diff, "flux_max_abs": flux_diff},
        "compile_audit": audit,
    }
    result["status"] = "passed" if result["forward_finite"] and result["backward_finite"] and result["inactive_gradients_zero_or_none"] and max(q_diff, state_diff, flux_diff) <= 1.0e-12 and audit["fallbacks"] == 0 and audit["graph_breaks"] == 0 and audit["recompilations"] == 0 else "failed"
    del eager, compiled, forcing
    torch.cuda.empty_cache()
    return result


def _run_gradients(output: Path, cache_dir: Path, prior: dict[str, Any]) -> dict[str, Any]:
    metadata, inputs, theta_rows = _load_frozen_inputs()
    manifest_rows = {row["basin_id"]: row for row in metadata["manifest"]["catchments"]}
    results = []
    for model_id in GRADIENT_MODEL_IDS:
        hru_id = int(manifest_rows[CATCHMENT_IDS[0]]["hru_id"])
        theta = theta_rows[(hru_id, model_id)]["parameter_vector"]
        result = _gradient_probe(model_id, CATCHMENT_IDS[0], cache_dir, inputs, theta, hru_id)
        results.append(result)
        print(json.dumps({"gradient_model_id": model_id, "status": result["status"]}, sort_keys=True), flush=True)
        if result["status"] != "passed":
            break
    passed = len(results) == len(GRADIENT_MODEL_IDS) and all(row["status"] == "passed" for row in results)
    prior["gradient_smoke"] = {"selected_model_ids": list(GRADIENT_MODEL_IDS), "selection_frozen_before_execution": True, "results": results, "passed": passed}
    prior["gate_decision"] = {"build_compile_pass": bool(prior["build_compile_coverage"]["all_fullgraph_clean"]), "long_horizon_pass": bool(prior["long_horizon"]["all_quality_checks_pass"]), "gradient_smoke_pass": passed, "passed": bool(prior["build_compile_coverage"]["all_fullgraph_clean"] and prior["long_horizon"]["all_quality_checks_pass"] and passed)}
    prior["status"] = "passed" if prior["gate_decision"]["passed"] else "failed"
    prior["resource"]["cache_final"] = _cache_info(cache_dir)
    _write_json(output, prior)
    return prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "worker", "gradients"), default="run")
    parser.add_argument("--model-id", type=int)
    parser.add_argument("--worker-output", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--partial", default=str(DEFAULT_PARTIAL))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    if args.mode == "worker":
        if args.model_id is None or args.worker_output is None:
            raise ValueError("worker mode requires --model-id and --worker-output")
        _worker(cache_dir, Path(args.worker_output).resolve(), args.model_id)
        return
    if args.mode == "gradients":
        prior = json.loads(Path(args.output).read_text())
        _run_gradients(Path(args.output).resolve(), cache_dir, prior)
        return
    _run_parent(Path(args.output).resolve(), Path(args.partial).resolve(), cache_dir)


if __name__ == "__main__":
    main()
