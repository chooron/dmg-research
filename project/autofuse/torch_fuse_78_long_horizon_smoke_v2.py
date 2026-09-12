"""Full 78-structure Torch-FUSE v1 long-horizon gate with GPU basin batching."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from dfuse import enumerate_structures
from project.autofuse.torch_fuse_78_long_horizon_smoke import _load_frozen_inputs
from project.autofuse.unlimfrc_2_long_horizon_smoke import (
    BASINS,
    END,
    INITIAL_FRACTION,
    MONITOR_CHUNK_SIZE,
    START,
    WATER_TOLERANCE,
    SNOW_TOLERANCE,
    _atomic_write,
    _cache_info,
    _compile_pass,
    _forcing_batch,
    _rss_kb,
    _sha,
    _structure_row,
    _dates,
)
from dfuse import batched_compile_diagnostics, reset_batched_compile_diagnostics, simulate_coupled_rk2_batched

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
OUTPUT = DOCS / "torch_fuse_78_long_horizon_smoke_v2.json"
PARTIAL = DOCS / "torch_fuse_78_long_horizon_smoke_v2.partial.json"
CACHE = ROOT / "project/autofuse/.cache/torch-fuse-78-v2"
WORKER_DIR = CACHE / "workers"
WORKER_TIMEOUT_SECONDS = 1800
GRADIENT_SELECTION = (2, 10, 20, 26, 50, 146)


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


def _selection(specs: list[object]) -> dict[str, object]:
    return {"structure_count": len(specs), "model_ids": [spec.model_id for spec in specs], "catchments": list(BASINS), "logical_run_count_expected": len(specs) * len(BASINS), "structure_order": "catalogue order, serial"}


def _worker(model_id: int, worker_output: Path) -> None:
    _set_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("78-structure worker requires CUDA; refusing CPU fallback")
    metadata, inputs, theta_rows = _load_frozen_inputs()
    manifest = {row["basin_id"]: row for row in metadata["manifest"]["catchments"]}
    spec = next((item for item in enumerate_structures() if item.model_id == model_id), None)
    if spec is None:
        raise RuntimeError(f"model {model_id} is absent from the frozen catalogue")
    dates = _dates()
    device = torch.device("cuda")
    forcing = _forcing_batch(inputs, device)
    params = [theta_rows[(int(manifest[basin]["hru_id"]), model_id)]["parameter_vector"] for basin in BASINS]
    reset_batched_compile_diagnostics()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = simulate_coupled_rk2_batched(model_id, forcing, params, basin_ids=BASINS, dates=dates, initial_fraction=INITIAL_FRACTION, dt_days=1.0, compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=MONITOR_CHUNK_SIZE)
    torch.cuda.synchronize(device)
    records = list(batched_compile_diagnostics()["records"].values())
    if len(records) != 1:
        raise RuntimeError(f"expected one compile record for model {model_id}, got {len(records)}")
    row = _structure_row(spec, result, time.perf_counter() - started, records[0], device)
    row["resource"] = {"host_rss_kb": _rss_kb(), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    _atomic_write(worker_output, row)
    print(json.dumps({"model_id": model_id, "status": row["status"], "water_balance_max_abs": row["water_balance_max_abs"], "snow_balance_max_abs": row["snow_balance_max_abs"]}, sort_keys=True), flush=True)
    del result
    torch.cuda.empty_cache()


def _failure_payload(selection: dict[str, object], reports: list[dict[str, object]], failure: dict[str, object]) -> dict[str, object]:
    return {"schema_version": "torch-fuse-78-long-horizon-smoke-v2", "status": "failed", "gate": "78 structures x 2 catchments GPU-batched coupled-RK2 long-horizon smoke", "selection": selection, "completed_structure_count": len(reports), "reports": reports, "failures": [failure], "stopped_after_failure": True, "gate_decision": {"passed": False, "next_stage_allowed": False}}


def _parent(*, fresh: bool = False) -> None:
    _set_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("78-structure smoke requires CUDA; refusing CPU fallback")
    specs = list(enumerate_structures())
    if len(specs) != 78 or len({spec.model_id for spec in specs}) != 78:
        raise RuntimeError("strict 78-structure catalogue check failed")
    selection = _selection(specs)
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    resumed_from_structure_count = 0
    if not fresh and PARTIAL.is_file():
        try:
            prior = json.loads(PARTIAL.read_text())
            prior_ids = prior.get("selection", {}).get("model_ids", [])
            selected_ids = [spec.model_id for spec in specs]
            prior_reports = prior.get("reports", [])
            prior_report_ids = [report.get("model_id") for report in prior_reports]
            if prior.get("status") == "partial" and prior_ids == selected_ids and prior_report_ids == selected_ids[: len(prior_reports)]:
                reports = prior_reports
                resumed_from_structure_count = len(reports)
        except (OSError, json.JSONDecodeError, TypeError):
            reports = []
    started = time.perf_counter()
    for index, spec in enumerate(specs[resumed_from_structure_count:], start=resumed_from_structure_count + 1):
        worker_output = WORKER_DIR / f"{index:03d}-{spec.model_id}.json"
        command = [sys.executable, "-m", "project.autofuse.torch_fuse_78_long_horizon_smoke_v2", "--mode", "worker", "--model-id", str(spec.model_id), "--worker-output", str(worker_output)]
        environment = os.environ.copy()
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE.resolve())
        child = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=WORKER_TIMEOUT_SECONDS, check=False)
        if child.returncode != 0 or not worker_output.is_file():
            failure = {"model_id": spec.model_id, "topology": dict(spec.decisions), "error_type": "worker_failure", "returncode": child.returncode, "stderr_tail": child.stderr[-4000:], "stdout_tail": child.stdout[-2000:]}
            payload = _failure_payload(selection, reports, failure)
            _atomic_write(PARTIAL, payload)
            _atomic_write(OUTPUT, payload)
            print(json.dumps({"model_id": spec.model_id, "status": "failed", "error_type": "worker_failure"}, sort_keys=True), flush=True)
            return
        report = json.loads(worker_output.read_text())
        worker_output.unlink()
        reports.append(report)
        print(json.dumps({"structure_index": index, "model_id": spec.model_id, "status": report["status"], "water_balance_max_abs": report["water_balance_max_abs"], "snow_balance_max_abs": report["snow_balance_max_abs"]}, sort_keys=True), flush=True)
        if report.get("status") != "passed":
            failure = {"model_id": spec.model_id, "topology": dict(spec.decisions), "error_type": "gate_failure", "report": report}
            payload = _failure_payload(selection, reports, failure)
            _atomic_write(PARTIAL, payload)
            _atomic_write(OUTPUT, payload)
            return
        partial = {"schema_version": "torch-fuse-78-long-horizon-smoke-v2", "status": "partial", "selection": selection, "completed_structure_count": len(reports), "reports": reports, "failures": [], "stopped_after_failure": False}
        _atomic_write(PARTIAL, partial)
        if report["resource"]["host_rss_kb"] > 3_500_000:
            raise RuntimeError(f"resource safety stop after model {spec.model_id}: rss={report['resource']['host_rss_kb']} KB")
    compile_pass = len(reports) == 78 and all(report["compile_pass"] for report in reports)
    long_pass = len(reports) == 78 and all(report["quality_pass"] for report in reports)
    payload = {
        "schema_version": "torch-fuse-78-long-horizon-smoke-v2",
        "status": "passed" if compile_pass and long_pass else "failed",
        "gate": "78 structures x 2 catchments GPU-batched coupled-RK2 long-horizon smoke",
        "selection": selection,
        "source": {"catalogue": str(ROOT / "dfuse/specs/structures_78.json"), "catalogue_sha256": _sha(ROOT / "dfuse/specs/structures_78.json"), "manifest": str(ROOT / "project/autofuse/docs/landscape_12catchment_manifest.json"), "input_index": str(ROOT / "project/autofuse/docs/landscape_inputs/index.json"), "calibration_archive": str(ROOT / "project/autofuse/docs/reference_calibration_12x78.json"), "kernel": str(ROOT / "dfuse/kernel.py"), "kernel_sha256": _sha(ROOT / "dfuse/kernel.py"), "runtime": str(ROOT / "dfuse/runtime.py"), "runtime_sha256": _sha(ROOT / "dfuse/runtime.py"), "batched_source": str(ROOT / "dfuse/batched.py"), "batched_source_sha256": _sha(ROOT / "dfuse/batched.py")},
        "resume": {"resumed_from_structure_count": resumed_from_structure_count, "source_partial": str(PARTIAL) if resumed_from_structure_count else None},
        "protocol": {"forcing_start": START.isoformat(), "simulation_end": END.isoformat(), "n_steps": len(_dates()), "dt_days": 1.0, "initial_fraction": INITIAL_FRACTION, "dtype": "torch.float64", "device": "cuda", "cpu_threads": 1, "cpu_interop_threads": 1, "batch_size": len(BASINS), "monitor_chunk_size": MONITOR_CHUNK_SIZE, "structure_execution": "one isolated worker process at a time", "basin_execution": "GPU leading-dimension batch", "full_daily_trace_persisted": False, "partial_write": "atomic os.replace", "monitor_overhead": "not separately isolated; chunked GPU checks are included in elapsed time"},
        "coverage": {"catalogue_structure_count": 78, "build_success_count": len(reports), "unsupported_branch_count": 0, "hidden_fallback_count": 0},
        "compile": {"build_success_count": len(reports), "fullgraph_compile_success_count": sum(report["compile_pass"] for report in reports), "compile_attempts": sum(int(report["compile"].get("compile_attempts") or 0) for report in reports), "compile_successes": sum(int(report["compile"].get("compile_successes") or 0) for report in reports), "fallbacks": sum(int(report["compile"].get("fallbacks") or 0) for report in reports), "graph_breaks": sum(int(report["compile"].get("graph_breaks") or 0) for report in reports), "recompilations": sum(int(report["compile"].get("recompilations") or 0) for report in reports), "all_fullgraph_clean": compile_pass},
        "long_horizon": {"expected_structures": 78, "completed_structures": len(reports), "expected_logical_runs": 156, "completed_logical_runs": sum(report["logical_run_count"] for report in reports), "all_full_period": all(report["completed_full_period"] for report in reports), "all_finite": all(report["finite"] for report in reports), "negative_active_state_total": sum(report["negative_active_state_count"] for report in reports), "capacity_violation_total": sum(report["capacity_violation_count"] for report in reports), "max_water_balance_abs": max((report["water_balance_max_abs"] for report in reports), default=float("inf")), "max_snow_balance_abs": max((report["snow_balance_max_abs"] for report in reports), default=float("inf")), "all_monitor_first_failure_none": all(all(item["first_failure_index"] is None for item in report["monitoring"]["per_basin"]) for report in reports), "all_quality_checks_pass": long_pass},
        "gradient_smoke": {"status": "pending", "selection_frozen_before_execution": True, "selected_model_ids": list(GRADIENT_SELECTION), "selection_rule": "one catalogue structure covering each requested process/topology family; no result-dependent selection"},
        "reports": reports,
        "resource": {"host_peak_rss_kb": max((report["resource"]["host_rss_kb"] for report in reports), default=0), "gpu_peak_allocated_bytes": max((report["resource"]["gpu_peak_allocated_bytes"] for report in reports), default=0), "gpu_peak_reserved_bytes": max((report["resource"]["gpu_peak_reserved_bytes"] for report in reports), default=0), "wall_clock_seconds": time.perf_counter() - started, "cache": _cache_info(CACHE), "rss_stop_kb": 3_500_000},
        "gate_decision": {"build_compile_pass": compile_pass, "long_horizon_pass": long_pass, "gradient_smoke_pass": False, "passed": False, "next_stage_allowed": False, "reason": "gradient smoke is intentionally a separate post-gate stage"},
    }
    _atomic_write(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "structures": len(reports), "logical_runs": payload["long_horizon"]["completed_logical_runs"], "max_water_balance_abs": payload["long_horizon"]["max_water_balance_abs"], "max_snow_balance_abs": payload["long_horizon"]["max_snow_balance_abs"]}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parent", "worker"), default="parent")
    parser.add_argument("--model-id", type=int)
    parser.add_argument("--worker-output")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.mode == "worker":
        if args.model_id is None or args.worker_output is None:
            raise ValueError("worker mode requires --model-id and --worker-output")
        _worker(args.model_id, Path(args.worker_output).resolve())
    else:
        _parent(fresh=args.fresh)


if __name__ == "__main__":
    main()
