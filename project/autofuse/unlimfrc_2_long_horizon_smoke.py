"""Serial-structure, GPU-batched long-horizon smoke for ARCH2=unlimfrc_2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfuse import (
    batched_compile_diagnostics,
    enumerate_structures,
    reset_batched_compile_diagnostics,
    simulate_coupled_rk2_batched,
)
from project.autofuse.torch_fuse_78_long_horizon_smoke import _load_frozen_inputs

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
OUTPUT = DOCS / "unlimfrc_2_long_horizon_smoke.json"
PARTIAL = DOCS / "unlimfrc_2_long_horizon_smoke.partial.json"
CACHE = ROOT / "project/autofuse/.cache/torch-fuse-unlimfrc-2"
WORKER_DIR = CACHE / "workers"
BASINS = ("USA_09447800", "USA_14138900")
START = date(1987, 1, 1)
END = date(2009, 12, 31)
DT_DAYS = 1.0
INITIAL_FRACTION = 0.25
MONITOR_CHUNK_SIZE = 64
WATER_TOLERANCE = 1.0e-8
SNOW_TOLERANCE = 1.0e-8
RSS_STOP_KB = 3_500_000
WORKER_TIMEOUT_SECONDS = 1800


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dates() -> list[date]:
    return [START + timedelta(days=i) for i in range((END - START).days + 1)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _cache_info(path: Path) -> dict[str, int]:
    files = 0
    size = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
                size += item.stat().st_size
    return {"files": files, "bytes": size}


def _rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _set_environment() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE.resolve())
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


def _compile_pass(record: dict[str, Any]) -> bool:
    return bool(
        record.get("compile_attempts") == 1
        and record.get("compile_successes") == 1
        and record.get("fallbacks") == 0
        and record.get("graph_breaks") == 0
        and record.get("recompilations") == 0
    )


def _forcing_batch(inputs: dict[str, dict[str, np.ndarray]], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack([np.stack((inputs[basin]["ppt"], inputs[basin]["pet"], inputs[basin]["temp"]), axis=1) for basin in BASINS]), dtype=torch.float64, device=device)


def _structure_row(spec: Any, result: Any, elapsed: float, compile_record: dict[str, Any], device: torch.device) -> dict[str, Any]:
    q_finite = bool(torch.isfinite(result.q).all().item())
    states_finite = bool(torch.isfinite(result.states).all().item())
    flux_finite = bool(all(torch.isfinite(value).all().item() for value in result.fluxes.values()))
    monitoring = result.monitoring
    per_basin = []
    for item in monitoring["per_basin"]:
        per_basin.append({
            "basin_id": item["basin_id"],
            "first_failure_index": item["first_failure_index"],
            "first_failure_category": item["first_failure_category"],
            "failure_trace": item.get("failure_trace"),
        })
    full = bool(monitoring["completed_full_period"] and monitoring["actual_steps"] == len(_dates()))
    finite = bool(q_finite and states_finite and flux_finite and all(value == 0 for value in monitoring["nonfinite_count"]))
    negative = int(sum(monitoring["negative_active_state_count"]))
    capacity = int(sum(monitoring["capacity_violation_count"]))
    failure_free = all(item["first_failure_index"] is None for item in per_basin)
    quality_pass = bool(
        full
        and finite
        and negative == 0
        and capacity == 0
        and failure_free
        and max(monitoring["water_balance_max_abs"], default=float("inf")) <= WATER_TOLERANCE
        and max(monitoring["snow_balance_max_abs"], default=float("inf")) <= SNOW_TOLERANCE
    )
    return {
        "model_id": spec.model_id,
        "topology": dict(spec.decisions),
        "status": "passed" if quality_pass and _compile_pass(compile_record) else "failed",
        "compile": {key: compile_record.get(key) for key in ("compile_attempts", "compile_successes", "fallbacks", "graph_breaks", "recompilations", "forward_unique_graphs", "cold_compile_seconds")},
        "compile_pass": _compile_pass(compile_record),
        "completed_full_period": full,
        "logical_run_count": len(per_basin),
        "finite": finite,
        "q_finite": q_finite,
        "states_finite": states_finite,
        "flux_finite": flux_finite,
        "negative_active_state_count": negative,
        "capacity_violation_count": capacity,
        "water_balance_max_abs": max(monitoring["water_balance_max_abs"], default=float("inf")),
        "snow_balance_max_abs": max(monitoring["snow_balance_max_abs"], default=float("inf")),
        "monitoring": {"chunk_size": monitoring["monitor_chunk_size"], "actual_steps": monitoring["actual_steps"], "stopped_on_failure": monitoring["stopped_on_failure"], "per_basin": per_basin, "nonfinite_count": monitoring["nonfinite_count"], "negative_active_state_count": monitoring["negative_active_state_count"], "capacity_violation_count": monitoring["capacity_violation_count"], "water_balance_max_abs": monitoring["water_balance_max_abs"], "snow_balance_max_abs": monitoring["snow_balance_max_abs"]},
        "quality_pass": quality_pass,
        "elapsed_seconds": elapsed,
        "gpu_device": str(device),
    }


def _worker(model_id: int, worker_output: Path) -> None:
    _set_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("unlimfrc_2 worker requires CUDA; refusing CPU fallback")
    metadata, inputs, theta_rows = _load_frozen_inputs()
    manifest = {row["basin_id"]: row for row in metadata["manifest"]["catchments"]}
    spec = next((item for item in enumerate_structures() if item.model_id == model_id), None)
    if spec is None or spec.decisions["ARCH2"] != "unlimfrc_2":
        raise RuntimeError(f"model {model_id} is not a catalogue ARCH2=unlimfrc_2 structure")
    dates = _dates()
    device = torch.device("cuda")
    forcing = _forcing_batch(inputs, device)
    params = [theta_rows[(int(manifest[basin]["hru_id"]), model_id)]["parameter_vector"] for basin in BASINS]
    reset_batched_compile_diagnostics()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = simulate_coupled_rk2_batched(model_id, forcing, params, basin_ids=BASINS, dates=dates, initial_fraction=INITIAL_FRACTION, dt_days=DT_DAYS, compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=MONITOR_CHUNK_SIZE)
    torch.cuda.synchronize(device)
    records = list(batched_compile_diagnostics()["records"].values())
    if len(records) != 1:
        raise RuntimeError(f"expected one batched compile record for model {model_id}, got {len(records)}")
    row = _structure_row(spec, result, time.perf_counter() - started, records[0], device)
    row["resource"] = {"host_rss_kb": _rss_kb(), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    _atomic_write(worker_output, row)
    print(json.dumps({"model_id": model_id, "status": row["status"], "water_balance_max_abs": row["water_balance_max_abs"], "snow_balance_max_abs": row["snow_balance_max_abs"]}, sort_keys=True), flush=True)
    del result
    torch.cuda.empty_cache()


def _selection(selected: list[Any]) -> dict[str, Any]:
    return {"structure_count": len(selected), "model_ids": [item.model_id for item in selected], "catchments": list(BASINS), "logical_run_count_expected": len(selected) * len(BASINS), "structure_order": "catalogue order, serial"}


def _failed_payload(selection: dict[str, Any], reports: list[dict[str, Any]], failure: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "unlimfrc-2-long-horizon-smoke-v1", "status": "failed", "gate": "all ARCH2=unlimfrc_2 structures, two-catchment GPU-batched long-horizon smoke", "selection": selection, "completed_structure_count": len(reports), "reports": reports, "failures": [failure], "stopped_after_failure": True, "gate_decision": {"all_unlimfrc_2_pass": False, "passed": False, "next_stage_allowed": False}}


def _parent() -> None:
    _set_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("unlimfrc_2 smoke requires CUDA; refusing CPU fallback")
    selected = [spec for spec in enumerate_structures() if spec.decisions["ARCH2"] == "unlimfrc_2"]
    if len(selected) != 21:
        raise RuntimeError(f"expected 21 ARCH2=unlimfrc_2 structures, found {len(selected)}")
    selection = _selection(selected)
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, spec in enumerate(selected, start=1):
        worker_output = WORKER_DIR / f"{index:03d}-{spec.model_id}.json"
        command = [sys.executable, "-m", "project.autofuse.unlimfrc_2_long_horizon_smoke", "--mode", "worker", "--model-id", str(spec.model_id), "--worker-output", str(worker_output)]
        environment = os.environ.copy()
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE.resolve())
        child = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=WORKER_TIMEOUT_SECONDS, check=False)
        if child.returncode != 0 or not worker_output.is_file():
            failure = {"model_id": spec.model_id, "topology": dict(spec.decisions), "error_type": "worker_failure", "returncode": child.returncode, "stderr_tail": child.stderr[-4000:], "stdout_tail": child.stdout[-2000:]}
            payload = _failed_payload(selection, reports, failure)
            _atomic_write(PARTIAL, payload)
            _atomic_write(OUTPUT, payload)
            print(json.dumps({"model_id": spec.model_id, "status": "failed", "error_type": "worker_failure"}, sort_keys=True), flush=True)
            return
        report = json.loads(worker_output.read_text())
        reports.append(report)
        worker_output.unlink()
        if report.get("status") != "passed":
            failure = {"model_id": spec.model_id, "topology": dict(spec.decisions), "error_type": "gate_failure", "report": report}
            payload = _failed_payload(selection, reports, failure)
            _atomic_write(PARTIAL, payload)
            _atomic_write(OUTPUT, payload)
            print(json.dumps({"model_id": spec.model_id, "status": "failed", "error_type": "gate_failure"}, sort_keys=True), flush=True)
            return
        partial = {"schema_version": "unlimfrc-2-long-horizon-smoke-v1", "status": "partial", "selection": selection, "completed_structure_count": len(reports), "reports": reports, "failures": [], "stopped_after_failure": False}
        _atomic_write(PARTIAL, partial)
        print(json.dumps({"structure_index": index, "model_id": spec.model_id, "status": "passed", "water_balance_max_abs": report["water_balance_max_abs"], "snow_balance_max_abs": report["snow_balance_max_abs"]}, sort_keys=True), flush=True)
        if report["resource"]["host_rss_kb"] > RSS_STOP_KB:
            raise RuntimeError(f"resource safety stop after model {spec.model_id}: rss={report['resource']['host_rss_kb']} KB")
    compile_pass = len(reports) == len(selected) and all(row["compile_pass"] for row in reports)
    long_pass = len(reports) == len(selected) and all(row["quality_pass"] for row in reports)
    payload = {
        "schema_version": "unlimfrc-2-long-horizon-smoke-v1",
        "status": "passed" if compile_pass and long_pass else "failed",
        "gate": "all ARCH2=unlimfrc_2 structures, two-catchment GPU-batched long-horizon smoke",
        "selection": selection,
        "source": {"catalogue": str(ROOT / "dfuse/specs/structures_78.json"), "catalogue_sha256": _sha(ROOT / "dfuse/specs/structures_78.json"), "manifest": str(ROOT / "project/autofuse/docs/landscape_12catchment_manifest.json"), "input_index": str(ROOT / "project/autofuse/docs/landscape_inputs/index.json"), "calibration_archive": str(ROOT / "project/autofuse/docs/reference_calibration_12x78.json"), "batched_source": str(ROOT / "dfuse/batched.py"), "batched_source_sha256": _sha(ROOT / "dfuse/batched.py")},
        "protocol": {"forcing_start": START.isoformat(), "simulation_end": END.isoformat(), "n_steps": len(_dates()), "dt_days": DT_DAYS, "initial_fraction": INITIAL_FRACTION, "dtype": "torch.float64", "device": "cuda", "cpu_threads": 1, "cpu_interop_threads": 1, "batch_size": len(BASINS), "monitor_chunk_size": MONITOR_CHUNK_SIZE, "structure_execution": "one isolated worker process at a time", "basin_execution": "GPU leading-dimension batch", "full_daily_trace_persisted": False, "monitor_overhead": "not separately isolated; chunked GPU checks are included in elapsed time"},
        "coverage": {"catalogue_unlimfrc_2_count": 21, "selected_count": len(selected), "unsupported_branch_count": 0, "all_selected_arch2_unlimfrc_2": True},
        "compile": {"build_success_count": len(reports), "fullgraph_compile_success_count": sum(row["compile_pass"] for row in reports), "compile_attempts": sum(int(row["compile"].get("compile_attempts") or 0) for row in reports), "compile_successes": sum(int(row["compile"].get("compile_successes") or 0) for row in reports), "fallbacks": sum(int(row["compile"].get("fallbacks") or 0) for row in reports), "graph_breaks": sum(int(row["compile"].get("graph_breaks") or 0) for row in reports), "recompilations": sum(int(row["compile"].get("recompilations") or 0) for row in reports), "all_fullgraph_clean": compile_pass},
        "long_horizon": {"expected_structures": len(selected), "completed_structures": len(reports), "expected_logical_runs": len(selected) * len(BASINS), "completed_logical_runs": sum(row["logical_run_count"] for row in reports), "all_full_period": all(row["completed_full_period"] for row in reports), "all_finite": all(row["finite"] for row in reports), "negative_active_state_total": sum(row["negative_active_state_count"] for row in reports), "capacity_violation_total": sum(row["capacity_violation_count"] for row in reports), "max_water_balance_abs": max((row["water_balance_max_abs"] for row in reports), default=float("inf")), "max_snow_balance_abs": max((row["snow_balance_max_abs"] for row in reports), default=float("inf")), "all_monitor_first_failure_none": all(all(item["first_failure_index"] is None for item in row["monitoring"]["per_basin"]) for row in reports), "all_quality_checks_pass": long_pass},
        "reports": reports,
        "resource": {"host_peak_rss_kb": max((row["resource"]["host_rss_kb"] for row in reports), default=0), "gpu_peak_allocated_bytes": max((row["resource"]["gpu_peak_allocated_bytes"] for row in reports), default=0), "gpu_peak_reserved_bytes": max((row["resource"]["gpu_peak_reserved_bytes"] for row in reports), default=0), "wall_clock_seconds": time.perf_counter() - started, "cache": _cache_info(CACHE), "rss_stop_kb": RSS_STOP_KB},
        "gate_decision": {"all_unlimfrc_2_pass": bool(compile_pass and long_pass), "passed": bool(compile_pass and long_pass), "next_stage_allowed": bool(compile_pass and long_pass)},
    }
    _atomic_write(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "structures": len(reports), "logical_runs": payload["long_horizon"]["completed_logical_runs"], "max_water_balance_abs": payload["long_horizon"]["max_water_balance_abs"], "max_snow_balance_abs": payload["long_horizon"]["max_snow_balance_abs"]}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parent", "worker"), default="parent")
    parser.add_argument("--model-id", type=int)
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.mode == "worker":
        if args.model_id is None or args.worker_output is None:
            raise ValueError("worker mode requires --model-id and --worker-output")
        _worker(args.model_id, Path(args.worker_output).resolve())
    else:
        _parent()


if __name__ == "__main__":
    main()
