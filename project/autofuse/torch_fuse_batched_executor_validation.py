"""Correctness and small throughput gate for the basin-batched coupled RK2 executor."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfuse import (
    batched_compile_diagnostics,
    reset_batched_compile_diagnostics,
    simulate_coupled_rk2,
    simulate_coupled_rk2_batched,
)
from dfuse.spec import FLUX_NAMES
from project.autofuse.torch_fuse_78_long_horizon_smoke import _load_frozen_inputs

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
OUTPUT = DOCS / "torch_fuse_batched_executor_validation.json"
CACHE = ROOT / "project/autofuse/.cache/torch-fuse-batched"
BASINS = ("USA_09447800", "USA_14138900")
MODELS = (2, 8)
START = date(1987, 1, 1)
END = date(2009, 12, 31)
WATER_TOLERANCE = 1.0e-8
SNOW_TOLERANCE = 1.0e-8


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dates() -> list[date]:
    return [START + timedelta(days=i) for i in range((END - START).days + 1)]


def _first_failure(serial: Any) -> tuple[int | None, str | None]:
    finite = bool(torch.isfinite(serial.q).all() and torch.isfinite(serial.states).all() and all(torch.isfinite(value).all() for value in serial.fluxes.values()))
    if not finite:
        return 0, "finite"
    if bool((serial.states < 0.0).any()):
        return 0, "negative_state"
    water = serial.water_balance_residual.detach().abs().cpu().numpy()
    hits = np.flatnonzero(water > WATER_TOLERANCE)
    if len(hits):
        return int(hits[0]), "water_balance"
    snow = serial.snow_balance_residual.detach().abs().cpu().numpy()
    hits = np.flatnonzero(snow > SNOW_TOLERANCE)
    if len(hits):
        return int(hits[0]), "snow_balance"
    return None, None


def _parity(serial: Any, batched: Any, basin_index: int) -> dict[str, float]:
    values = {
        "q_max_abs": float((serial.q.detach() - batched.q[basin_index].detach()).abs().max().cpu()),
        "q_instantaneous_max_abs": float((serial.q_instantaneous.detach() - batched.q_instantaneous[basin_index].detach()).abs().max().cpu()),
        "states_max_abs": float((serial.states.detach() - batched.states[basin_index].detach()).abs().max().cpu()),
        "water_balance_max_abs": float((serial.water_balance_residual.detach() - batched.water_balance_residual[basin_index].detach()).abs().max().cpu()),
        "snow_balance_max_abs": float((serial.snow_balance_residual.detach() - batched.snow_balance_residual[basin_index].detach()).abs().max().cpu()),
    }
    values["flux_max_abs"] = max(float((serial.fluxes[name].detach() - batched.fluxes[name][basin_index].detach()).abs().max().cpu()) for name in FLUX_NAMES)
    return values


def _benchmark(model_id: int, forcing: torch.Tensor, params: list[dict[str, Any]], basin_ids: tuple[str, ...], dates: list[date], device: torch.device) -> dict[str, Any]:
    rows = []
    for batch_size in (1, 2):
        reset_batched_compile_diagnostics()
        torch.cuda.reset_peak_memory_stats(device)
        selected_forcing = forcing[:batch_size]
        selected_params = params[:batch_size]
        selected_ids = basin_ids[:batch_size]
        started = time.perf_counter()
        with torch.no_grad():
            result = simulate_coupled_rk2_batched(model_id, selected_forcing, selected_params, basin_ids=selected_ids, dates=dates, compile_step=True, monitor_chunk_size=len(dates))
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        compile_records = list(batched_compile_diagnostics()["records"].values())
        record = compile_records[0] if compile_records else {}
        rows.append({
            "batch_size": batch_size,
            "basin_ids": list(selected_ids),
            "days": len(dates),
            "elapsed_seconds": elapsed,
            "days_per_second": len(dates) / elapsed,
            "basin_days_per_second": batch_size * len(dates) / elapsed,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "compile": {key: record.get(key) for key in ("compile_attempts", "compile_successes", "fallbacks", "graph_breaks", "recompilations", "forward_unique_graphs", "cold_compile_seconds")},
            "monitoring": result.monitoring,
            "safe": bool(record.get("fallbacks", 1) == 0 and record.get("graph_breaks", 1) == 0 and record.get("recompilations", 1) == 0),
        })
        del result
        torch.cuda.empty_cache()
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    safe_rows = [row for row in rows if row["safe"] and row["gpu_peak_reserved_bytes"] < 0.85 * total_memory]
    best = max(safe_rows, key=lambda row: row["basin_days_per_second"]) if safe_rows else None
    return {"model_id": model_id, "tested_batch_sizes": rows, "gpu_total_memory_bytes": total_memory, "safe_default_basin_batch_size": int(best["batch_size"]) if best else None}


def main() -> None:
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
    if not torch.cuda.is_available():
        raise RuntimeError("batched validation requires CUDA; refusing CPU fallback")
    source_meta, inputs, theta_rows = _load_frozen_inputs()
    manifest = {row["basin_id"]: row for row in source_meta["manifest"]["catchments"]}
    dates = _dates()
    device = torch.device("cuda")
    forcing = torch.as_tensor(np.stack([np.stack((inputs[basin]["ppt"], inputs[basin]["pet"], inputs[basin]["temp"]), axis=1) for basin in BASINS]), dtype=torch.float64, device=device)
    params = [theta_rows[(int(manifest[basin]["hru_id"]), MODELS[0])]["parameter_vector"] for basin in BASINS]
    parity_rows = []
    all_parity_pass = True
    compile_rows = []
    conservation_rows = []
    for model_id in MODELS:
        model_params = [theta_rows[(int(manifest[basin]["hru_id"]), model_id)]["parameter_vector"] for basin in BASINS]
        serial = []
        with torch.no_grad():
            for basin_index in range(len(BASINS)):
                serial.append(simulate_coupled_rk2(model_id, forcing[basin_index], model_params[basin_index], dates=dates, initial_fraction=0.25, dt_days=1.0, compile_step=True, compile_backend="inductor", compile_fullgraph=True))
            reset_batched_compile_diagnostics()
            batched = simulate_coupled_rk2_batched(model_id, forcing, model_params, basin_ids=BASINS, dates=dates, initial_fraction=0.25, dt_days=1.0, compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=len(dates))
            torch.cuda.synchronize(device)
        model_parity = []
        for basin_index, basin_id in enumerate(BASINS):
            values = _parity(serial[basin_index], batched, basin_index)
            first_index, first_category = _first_failure(serial[basin_index])
            monitored = batched.monitoring["per_basin"][basin_index]
            parity_pass = max(values.values()) <= 1.0e-12 and monitored["first_failure_index"] == first_index and monitored["first_failure_category"] == first_category
            all_parity_pass = all_parity_pass and parity_pass
            model_parity.append({"basin_id": basin_id, "parity": values, "serial_first_failure": {"index": first_index, "category": first_category}, "batched_monitoring": monitored, "passed": parity_pass})
            conservation_rows.append({"model_id": model_id, "basin_id": basin_id, "water_balance_max_abs": float(serial[basin_index].water_balance_residual.abs().max().cpu()), "snow_balance_max_abs": float(serial[basin_index].snow_balance_residual.abs().max().cpu())})
        compile_records = list(batched_compile_diagnostics()["records"].values())
        compile_record = compile_records[0] if compile_records else {}
        compile_pass = bool(compile_record.get("compile_attempts") == 1 and compile_record.get("compile_successes") == 1 and compile_record.get("fallbacks") == 0 and compile_record.get("graph_breaks") == 0 and compile_record.get("recompilations") == 0)
        compile_rows.append({"model_id": model_id, "audit": {key: compile_record.get(key) for key in ("compile_attempts", "compile_successes", "fallbacks", "graph_breaks", "recompilations", "forward_unique_graphs", "cold_compile_seconds")}, "passed": compile_pass})
        parity_rows.extend({"model_id": model_id, **row} for row in model_parity)
        del serial, batched
        torch.cuda.empty_cache()
    with torch.no_grad():
        monitoring_probe = simulate_coupled_rk2_batched(MODELS[1], forcing, [theta_rows[(int(manifest[basin]["hru_id"]), MODELS[1])]["parameter_vector"] for basin in BASINS], basin_ids=BASINS, dates=dates, initial_fraction=0.25, dt_days=1.0, compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=64)
        torch.cuda.synchronize(device)
    monitoring_probe_summary = monitoring_probe.monitoring
    del monitoring_probe
    torch.cuda.empty_cache()

    benchmark = _benchmark(MODELS[0], forcing, params, BASINS, dates, device)
    payload = {
        "schema_version": "torch-fuse-batched-executor-validation-v1",
        "status": "passed" if all_parity_pass and all(row["passed"] for row in compile_rows) else "failed",
        "execution": {"executor": "simulate_coupled_rk2_batched", "batch_dimension": "leading basin dimension", "structure_policy": "one structure-specialized GraphSignature per batch", "timestep_policy": "one Python time loop; no catchment loop inside timestep", "implementation": "torch.vmap over generated scalar coupled-RK2 step, then torch.compile(fullgraph=True)", "monitoring": "GPU first-failure masks and extrema, host synchronization only at configurable chunk boundaries"},
        "source": {"catalogue": str(ROOT / "dfuse/specs/structures_78.json"), "catalogue_sha256": _sha(ROOT / "dfuse/specs/structures_78.json"), "input_index": str(ROOT / "project/autofuse/docs/landscape_inputs/index.json"), "calibration_archive": str(ROOT / "project/autofuse/docs/reference_calibration_12x78.json"), "batched_source": str(ROOT / "dfuse/batched.py"), "batched_source_sha256": _sha(ROOT / "dfuse/batched.py"), "runtime_source": str(ROOT / "dfuse/runtime.py"), "runtime_source_sha256": _sha(ROOT / "dfuse/runtime.py")},
        "protocol": {"basins": list(BASINS), "structures": list(MODELS), "forcing_start": START.isoformat(), "simulation_end": END.isoformat(), "n_steps": len(dates), "dtype": "torch.float64", "device": str(device), "cpu_threads": 1, "cpu_interop_threads": 1, "monitor_chunk_size_for_parity": len(dates)},
        "parity": {"rows": parity_rows, "max_abs_by_field": {field: max((row["parity"][field] for row in parity_rows), default=float("inf")) for field in ("q_max_abs", "q_instantaneous_max_abs", "states_max_abs", "flux_max_abs", "water_balance_max_abs", "snow_balance_max_abs")}, "passed": all_parity_pass},
        "compile": {"rows": compile_rows, "passed": all(row["passed"] for row in compile_rows)},
        "conservation_observed": {"rows": conservation_rows, "note": "model 8 USA_09447800 retains the pre-existing WATR_2 lower-floor residual; this is intentionally reported and not hidden by batched parity"},
        "monitoring_chunk_probe": {"chunk_size": 64, "model_id": MODELS[1], "summary": monitoring_probe_summary, "purpose": "verify per-basin first-failure detection and chunk-boundary stop without daily host synchronization"},
        "throughput_benchmark": benchmark,
        "resource": {"host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "gpu_peak_allocated_bytes": max((row["gpu_peak_allocated_bytes"] for row in benchmark["tested_batch_sizes"]), default=0), "gpu_peak_reserved_bytes": max((row["gpu_peak_reserved_bytes"] for row in benchmark["tested_batch_sizes"]), default=0), "cache_dir": str(CACHE.resolve())},
        "gate_decision": {"batched_parity_passed": all_parity_pass, "batched_compile_passed": all(row["passed"] for row in compile_rows), "executor_validation_passed": all_parity_pass and all(row["passed"] for row in compile_rows), "science_conservation_gate_passed": False if any(row["water_balance_max_abs"] > WATER_TOLERANCE or row["snow_balance_max_abs"] > SNOW_TOLERANCE for row in conservation_rows) else True},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "parity": all_parity_pass, "compile": payload["compile"]["passed"], "safe_default_basin_batch_size": benchmark["safe_default_basin_batch_size"]}, sort_keys=True))


if __name__ == "__main__":
    main()
