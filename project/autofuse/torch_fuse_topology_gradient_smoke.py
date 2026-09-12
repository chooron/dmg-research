"""B=2 topology gradient smoke after the full long-horizon gate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfuse import (
    PARAMETER_NAMES,
    batched_compile_diagnostics,
    enumerate_structures,
    reset_batched_compile_diagnostics,
    simulate_coupled_rk2,
    simulate_coupled_rk2_batched,
)
from project.autofuse.torch_fuse_78_long_horizon_smoke import _load_frozen_inputs
from project.autofuse.unlimfrc_2_long_horizon_smoke import (
    BASINS,
    _atomic_write,
    _cache_info,
    _forcing_batch,
    _rss_kb,
    _sha,
    _dates,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
OUTPUT = DOCS / "torch_fuse_topology_gradient_smoke.json"
PARTIAL = DOCS / "torch_fuse_topology_gradient_smoke.partial.json"
CACHE = ROOT / "project/autofuse/.cache/torch-fuse-topology-gradients"
WORKER_DIR = CACHE / "workers"
SELECTED_MODELS = (2, 10, 20, 26, 50, 146)
SMOKE_DAYS = 8
TOLERANCE = 1.0e-12


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


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu())


def _finite_output(result: Any) -> bool:
    return bool(torch.isfinite(result.q).all().item() and torch.isfinite(result.states).all().item() and all(torch.isfinite(value).all().item() for value in result.fluxes.values()))


def _output_parity(left: Any, right: Any, basin_index: int) -> dict[str, float]:
    values = {
        "q_max_abs": _max_diff(left.q, right.q[basin_index]),
        "q_instantaneous_max_abs": _max_diff(left.q_instantaneous, right.q_instantaneous[basin_index]),
        "states_max_abs": _max_diff(left.states, right.states[basin_index]),
        "water_balance_max_abs": _max_diff(left.water_balance_residual, right.water_balance_residual[basin_index]),
        "snow_balance_max_abs": _max_diff(left.snow_balance_residual, right.snow_balance_residual[basin_index]),
    }
    values["flux_max_abs"] = max(_max_diff(left.fluxes[name], right.fluxes[name][basin_index]) for name in left.fluxes)
    return values


def _worker(model_id: int, worker_output: Path) -> None:
    _set_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("topology gradient smoke requires CUDA; refusing CPU fallback")
    metadata, inputs, theta_rows = _load_frozen_inputs()
    manifest = {row["basin_id"]: row for row in metadata["manifest"]["catchments"]}
    spec = next(item for item in enumerate_structures() if item.model_id == model_id)
    device = torch.device("cuda")
    forcing = _forcing_batch(inputs, device)[:, :SMOKE_DAYS]
    dates = _dates()[:SMOKE_DAYS]
    parameter_maps = [theta_rows[(int(manifest[basin]["hru_id"]), model_id)]["parameter_vector"] for basin in BASINS]
    theta_values = np.asarray([[float(params.get(name, 0.0)) for name in PARAMETER_NAMES] for params in parameter_maps], dtype=np.float64)
    active_positions = [PARAMETER_NAMES.index(name) for name in spec.parameter_names]
    inactive_positions = [index for index in range(len(PARAMETER_NAMES)) if index not in active_positions]

    with torch.no_grad():
        eager_batched = simulate_coupled_rk2_batched(model_id, forcing, parameter_maps, basin_ids=BASINS, dates=dates, compile_step=False, monitor_chunk_size=SMOKE_DAYS)
        serial_results = [simulate_coupled_rk2(model_id, forcing[index], parameter_maps[index], dates=dates, compile_step=False) for index in range(len(BASINS))]
    serial_parity = [_output_parity(serial_results[index], eager_batched, index) for index in range(len(BASINS))]
    serial_parity_max = max((max(row.values()) for row in serial_parity), default=0.0)

    eager_theta = torch.tensor(theta_values, dtype=torch.float64, device=device, requires_grad=True)
    eager_gradient_result = simulate_coupled_rk2_batched(model_id, forcing, eager_theta, basin_ids=BASINS, dates=dates, compile_step=False, monitor_chunk_size=SMOKE_DAYS)
    eager_gradient = torch.autograd.grad(eager_gradient_result.q.sum(), (eager_theta,), allow_unused=True)[0]
    if eager_gradient is None:
        raise RuntimeError(f"no eager gradient returned for model {model_id}")

    reset_batched_compile_diagnostics()
    compiled_theta = torch.tensor(theta_values, dtype=torch.float64, device=device, requires_grad=True)
    compiled_result = simulate_coupled_rk2_batched(model_id, forcing, compiled_theta, basin_ids=BASINS, dates=dates, compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=SMOKE_DAYS)
    compiled_gradient = torch.autograd.grad(compiled_result.q.sum(), (compiled_theta,), allow_unused=True)[0]
    torch.cuda.synchronize(device)
    if compiled_gradient is None:
        raise RuntimeError(f"no compiled gradient returned for model {model_id}")
    records = list(batched_compile_diagnostics()["records"].values())
    if len(records) != 1:
        raise RuntimeError(f"expected one batched compile record for model {model_id}, got {len(records)}")
    record = records[0]
    compiled_eager_parity = {
        "q_max_abs": _max_diff(eager_gradient_result.q, compiled_result.q),
        "q_instantaneous_max_abs": _max_diff(eager_gradient_result.q_instantaneous, compiled_result.q_instantaneous),
        "states_max_abs": _max_diff(eager_gradient_result.states, compiled_result.states),
        "water_balance_max_abs": _max_diff(eager_gradient_result.water_balance_residual, compiled_result.water_balance_residual),
        "snow_balance_max_abs": _max_diff(eager_gradient_result.snow_balance_residual, compiled_result.snow_balance_residual),
        "flux_max_abs": max(_max_diff(eager_gradient_result.fluxes[name], compiled_result.fluxes[name]) for name in eager_gradient_result.fluxes),
        "active_gradient_max_abs": float((eager_gradient[:, active_positions] - compiled_gradient[:, active_positions]).abs().max().detach().cpu()) if active_positions else 0.0,
    }
    inactive_values = compiled_gradient[:, inactive_positions] if inactive_positions else compiled_gradient.new_zeros((len(BASINS), 0))
    active_values = compiled_gradient[:, active_positions] if active_positions else compiled_gradient.new_zeros((len(BASINS), 0))
    active_finite = bool(torch.isfinite(active_values).all().item())
    inactive_max_abs = float(inactive_values.abs().max().detach().cpu()) if inactive_positions else 0.0
    compile_pass = bool(record.get("compile_attempts") == 1 and record.get("compile_successes") == 1 and record.get("fallbacks") == 0 and record.get("graph_breaks") == 0 and record.get("recompilations") == 0)
    forward_finite = bool(_finite_output(eager_batched) and _finite_output(compiled_result))
    monitoring_pass = all(item["first_failure_index"] is None for item in compiled_result.monitoring["per_basin"])
    passed = bool(forward_finite and serial_parity_max <= TOLERANCE and active_finite and inactive_max_abs == 0.0 and max(compiled_eager_parity.values()) <= TOLERANCE and compile_pass and monitoring_pass)
    row = {
        "model_id": model_id,
        "topology": dict(spec.decisions),
        "active_parameter_names": list(spec.parameter_names),
        "forward_finite": forward_finite,
        "backward_finite": bool(torch.isfinite(eager_gradient).all().item() and torch.isfinite(compiled_gradient).all().item()),
        "active_gradient_finite": active_finite,
        "inactive_gradient_max_abs": inactive_max_abs,
        "inactive_gradients_zero_or_none": inactive_max_abs == 0.0,
        "serial_vs_batched_parity": {"basins": serial_parity, "max_abs": serial_parity_max},
        "eager_vs_compiled_parity": compiled_eager_parity,
        "monitoring": compiled_result.monitoring,
        "compile": {key: record.get(key) for key in ("compile_attempts", "compile_successes", "fallbacks", "graph_breaks", "recompilations", "forward_unique_graphs", "cold_compile_seconds")},
        "compile_pass": compile_pass,
        "passed": passed,
        "resource": {"host_rss_kb": _rss_kb(), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))},
    }
    _atomic_write(worker_output, row)
    print(json.dumps({"model_id": model_id, "status": "passed" if passed else "failed", "serial_parity_max": serial_parity_max, "inactive_gradient_max_abs": inactive_max_abs}, sort_keys=True), flush=True)


def _parent() -> None:
    _set_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("topology gradient smoke requires CUDA; refusing CPU fallback")
    specs = list(enumerate_structures())
    selected = [next(spec for spec in specs if spec.model_id == model_id) for model_id in SELECTED_MODELS]
    if len(selected) != len(SELECTED_MODELS):
        raise RuntimeError("frozen gradient selection is incomplete")
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, spec in enumerate(selected, start=1):
        worker_output = WORKER_DIR / f"{index:02d}-{spec.model_id}.json"
        command = [sys.executable, "-m", "project.autofuse.torch_fuse_topology_gradient_smoke", "--mode", "worker", "--model-id", str(spec.model_id), "--worker-output", str(worker_output)]
        environment = os.environ.copy()
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE.resolve())
        child = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=600, check=False)
        if child.returncode != 0 or not worker_output.is_file():
            failure = {"model_id": spec.model_id, "topology": dict(spec.decisions), "error_type": "worker_failure", "returncode": child.returncode, "stderr_tail": child.stderr[-4000:], "stdout_tail": child.stdout[-2000:]}
            payload = {"schema_version": "torch-fuse-topology-gradient-smoke-v1", "status": "failed", "selection": {"model_ids": list(SELECTED_MODELS), "catchments": list(BASINS)}, "completed_model_count": len(reports), "reports": reports, "failure": failure}
            _atomic_write(PARTIAL, payload)
            _atomic_write(OUTPUT, payload)
            return
        report = json.loads(worker_output.read_text())
        worker_output.unlink()
        reports.append(report)
        print(json.dumps({"model_id": spec.model_id, "status": report["passed"]}, sort_keys=True), flush=True)
        if not report.get("passed"):
            failure = {"model_id": spec.model_id, "topology": dict(spec.decisions), "error_type": "gradient_gate_failure", "report": report}
            payload = {"schema_version": "torch-fuse-topology-gradient-smoke-v1", "status": "failed", "selection": {"model_ids": list(SELECTED_MODELS), "catchments": list(BASINS)}, "completed_model_count": len(reports), "reports": reports, "failure": failure}
            _atomic_write(PARTIAL, payload)
            _atomic_write(OUTPUT, payload)
            return
        _atomic_write(PARTIAL, {"schema_version": "torch-fuse-topology-gradient-smoke-v1", "status": "partial", "selection": {"model_ids": list(SELECTED_MODELS), "catchments": list(BASINS)}, "completed_model_count": len(reports), "reports": reports})
    compile_pass = len(reports) == len(SELECTED_MODELS) and all(report["compile_pass"] for report in reports)
    gradient_pass = len(reports) == len(SELECTED_MODELS) and all(report["passed"] for report in reports)
    payload = {
        "schema_version": "torch-fuse-topology-gradient-smoke-v1",
        "status": "passed" if compile_pass and gradient_pass else "failed",
        "gate": "B=2 topology gradient smoke",
        "selection": {"model_ids": list(SELECTED_MODELS), "catchments": list(BASINS), "days": SMOKE_DAYS, "selection_frozen_before_execution": True, "selection_rule": "cover perc_lower, tension2_1, fixedsiz_2, unlimfrc_2, arno_x_vic, prms_varnt, and tmdl_param"},
        "source": {"catalogue": str(ROOT / "dfuse/specs/structures_78.json"), "catalogue_sha256": _sha(ROOT / "dfuse/specs/structures_78.json"), "runtime": str(ROOT / "dfuse/runtime.py"), "runtime_sha256": _sha(ROOT / "dfuse/runtime.py"), "batched_source": str(ROOT / "dfuse/batched.py"), "batched_source_sha256": _sha(ROOT / "dfuse/batched.py")},
        "protocol": {"batch_size": len(BASINS), "dtype": "torch.float64", "device": "cuda", "compile_backend": "inductor", "compile_fullgraph": True, "serial_structure_execution": True, "parameter_gradient_input": "[basin, PARAMETER_NAMES] tensor"},
        "reports": reports,
        "resource": {"host_peak_rss_kb": max((report["resource"]["host_rss_kb"] for report in reports), default=0), "gpu_peak_allocated_bytes": max((report["resource"]["gpu_peak_allocated_bytes"] for report in reports), default=0), "gpu_peak_reserved_bytes": max((report["resource"]["gpu_peak_reserved_bytes"] for report in reports), default=0), "wall_clock_seconds": time.perf_counter() - started, "cache": _cache_info(CACHE)},
        "gate_decision": {"compile_pass": compile_pass, "gradient_pass": gradient_pass, "passed": bool(compile_pass and gradient_pass)},
    }
    _atomic_write(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "models": len(reports), "compile_pass": compile_pass, "gradient_pass": gradient_pass}, sort_keys=True), flush=True)


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
