"""Minimal GPU/compiler smoke for the frozen S4 sequential explicit order.

S4 is already part of the frozen explicit order vocabulary; this harness only
selects it and exercises the existing runtime generator.  It does not alter
process equations, the structure catalogue, or the reference calibration.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from dfuse import get_structure, reset_compile_diagnostics, runtime_compile_diagnostics
from dfuse.kernel import FLUX_NAMES, PARAMETER_NAMES, SEQUENTIAL_ORDERS, STATE_NAMES, _make_sequential_step
from dfuse.runtime import get_compiled_step, get_generated_step
from project.autofuse.runtime_validation import (
    INPUT_SHAPES,
    _default_forcing,
    _measure,
    _measure_backward,
    _prepare,
    _run,
    _sync,
)
from dfuse.spec import default_parameters

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = Path(__file__).with_name(".cache") / "runtime-validation-78"
DEFAULT_OUTPUT = Path(__file__).with_name("docs") / "s4_runtime_smoke.json"
MOTHER_MODELS = (2, 108, 178, 210)
S4_ORDER = SEQUENTIAL_ORDERS["S4"]
N_SUBSTEPS = 1
TOLERANCE = 1.0e-12
RSS_STOP_KB = 3_500_000


def _set_resource_limits() -> None:
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


def _cache_info(path: Path) -> dict[str, int]:
    files = 0
    size = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
                size += item.stat().st_size
    return {"files": files, "bytes": size}


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu())


def _finite(result: Mapping[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in result.values())


def _record_summary(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    records = list(diagnostics.get("records", {}).values())
    if not records:
        raise AssertionError("S4 compiler produced no runtime audit record")
    return {
        "record_count": len(records),
        "compile_attempts": sum(int(row.get("compile_attempts", 0)) for row in records),
        "compile_successes": sum(int(row.get("compile_successes", 0)) for row in records),
        "fallbacks": sum(int(row.get("fallbacks", 0)) for row in records),
        "calls": sum(int(row.get("calls", 0)) for row in records),
        "graph_breaks": sum(int(row.get("graph_breaks", 0)) for row in records),
        "recompilations": sum(int(row.get("recompilations", 0)) for row in records),
        "autograd_recompilations": sum(int(row.get("autograd_recompilations", 0)) for row in records),
        "forward_unique_graphs": [int(row.get("forward_unique_graphs", 0)) for row in records],
        "autograd_unique_graphs": [int(row.get("autograd_unique_graphs", 0)) for row in records],
        "dynamo_counters": [row.get("dynamo_counters", {}) for row in records],
    }


def _gradient_loss(step: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]], model_id: int, forcing: torch.Tensor) -> torch.Tensor:
    params = {
        name: torch.tensor(value, dtype=forcing.dtype, device=forcing.device, requires_grad=True)
        for name, value in default_parameters().items()
    }
    result = _run(step, _prepare(model_id, forcing, params))
    return result["q"].sum()


def _model_smoke(model_id: int, device: torch.device, cache_dir: Path) -> dict[str, Any]:
    reset_compile_diagnostics()
    torch.cuda.reset_peak_memory_stats(device)
    forcing = _default_forcing(16, device)
    prepared = _prepare(model_id, forcing)
    current_step = _make_sequential_step(S4_ORDER, N_SUBSTEPS)
    signature, generated = get_generated_step(prepared["spec"], order=S4_ORDER, n_substeps=N_SUBSTEPS)
    _, compiled = get_compiled_step(
        prepared["spec"],
        order=S4_ORDER,
        n_substeps=N_SUBSTEPS,
        device=device,
        dtype=forcing.dtype,
        input_shapes=INPUT_SHAPES,
        backend="inductor",
        fullgraph=True,
    )

    current = _run(current_step, prepared, current=True)
    eager = _run(generated, _prepare(model_id, forcing))
    compiled_result = _run(compiled, _prepare(model_id, forcing))
    for offset in (0.1, -0.2, 0.3):
        _run(compiled, _prepare(model_id, forcing + offset))

    gradient_loss = _gradient_loss(compiled, model_id, forcing[:4])
    gradient_loss.backward()
    gradient_finite = bool(torch.isfinite(gradient_loss).all())
    diagnostics = runtime_compile_diagnostics()
    compile_summary = _record_summary(diagnostics)
    parity = {
        "current_vs_generated_packed_max_abs": _max_abs(current["packed"], eager["packed"]),
        "current_vs_generated_q_max_abs": _max_abs(current["q"], eager["q"]),
        "current_vs_generated_diagnostics_max_abs": _max_abs(current["diagnostics"], eager["diagnostics"]),
        "generated_vs_compiled_packed_max_abs": _max_abs(eager["packed"], compiled_result["packed"]),
        "generated_vs_compiled_q_max_abs": _max_abs(eager["q"], compiled_result["q"]),
        "generated_vs_compiled_diagnostics_max_abs": _max_abs(eager["diagnostics"], compiled_result["diagnostics"]),
        "finite": _finite(current) and _finite(eager) and _finite(compiled_result) and gradient_finite,
    }

    benchmark_forcing = _default_forcing(64, device)
    eager_prepared = _prepare(model_id, benchmark_forcing)
    compiled_prepared = _prepare(model_id, benchmark_forcing)
    with torch.no_grad():
        _run(generated, eager_prepared)
        _run(compiled, compiled_prepared)
    _sync(device)
    with torch.no_grad():
        eager_forward = _measure(lambda: _run(generated, _prepare(model_id, benchmark_forcing)), device)
        compiled_forward = _measure(lambda: _run(compiled, _prepare(model_id, benchmark_forcing)), device)
    eager_backward = _measure_backward(lambda: _gradient_loss(generated, model_id, benchmark_forcing), device)
    eager_peak = int(torch.cuda.max_memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    compiled_backward = _measure_backward(lambda: _gradient_loss(compiled, model_id, benchmark_forcing), device)
    compiled_peak = int(torch.cuda.max_memory_allocated(device))
    _sync(device)
    resource_row = {
        "host_rss_kb": _rss_kb(),
        "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    passed = (
        parity["finite"]
        and max(value for key, value in parity.items() if key.endswith("max_abs")) <= TOLERANCE
        and compile_summary["record_count"] == 1
        and compile_summary["compile_attempts"] == 1
        and compile_summary["compile_successes"] == 1
        and compile_summary["fallbacks"] == 0
        and compile_summary["graph_breaks"] == 0
        and compile_summary["recompilations"] == 0
        and compile_summary["autograd_recompilations"] == 0
        and resource_row["host_peak_rss_kb"] < RSS_STOP_KB
    )
    return {
        "model_id": model_id,
        "decisions": dict(get_structure(model_id).decisions),
        "graph_signature": signature.to_dict(),
        "signature_digest": signature.digest,
        "parity": parity,
        "compile_audit": compile_summary,
        "benchmark": {
            "steps": 64,
            "forward": {"generated_eager": eager_forward, "compiled": compiled_forward, "speedup": eager_forward["median_seconds"] / compiled_forward["median_seconds"]},
            "backward": {"generated_eager": eager_backward, "compiled": compiled_backward, "speedup": eager_backward["median_seconds"] / compiled_backward["median_seconds"]},
            "peak_gpu_allocated_bytes": {"generated_eager": eager_peak, "compiled": compiled_peak},
        },
        "resource": resource_row,
        "status": "passed" if passed else "failed",
    }


def run_smoke(output: Path, cache_dir: Path) -> dict[str, Any]:
    _set_resource_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("S4 smoke requires CUDA; refusing CPU fallback")
    device = torch.device("cuda")
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    initial_rss = _rss_kb()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stopped_early = False
    for model_id in MOTHER_MODELS:
        try:
            row = _model_smoke(model_id, device, cache_dir)
            rows.append(row)
            if row["status"] != "passed":
                failures.append({"model_id": model_id, "error": "S4 parity or compile audit failed", "row": row})
                stopped_early = True
        except Exception as exc:
            failures.append({"model_id": model_id, "error": f"{type(exc).__name__}: {str(exc)[:1000]}"})
            stopped_early = True
        partial = {
            "schema_version": "s4-runtime-smoke-v1",
            "status": "partial",
            "order": list(S4_ORDER),
            "mother_models": list(MOTHER_MODELS),
            "rows": rows,
            "failures": failures,
            "stopped_early": stopped_early,
            "resource": {"initial_host_rss_kb": initial_rss, "current_host_rss_kb": _rss_kb()},
            "cache_dir": str(cache_dir),
            "cache": _cache_info(cache_dir),
        }
        output.with_suffix(".partial.json").write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
        gc.collect()
        torch.cuda.empty_cache()
        if stopped_early:
            break
    payload = {
        "schema_version": "s4-runtime-smoke-v1",
        "status": "complete" if len(rows) == len(MOTHER_MODELS) and not failures else "stopped",
        "order": list(S4_ORDER),
        "n_substeps": N_SUBSTEPS,
        "mother_models": list(MOTHER_MODELS),
        "rows": rows,
        "failures": failures,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "dtype": "torch.float64",
            "device": str(device),
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "cache_dir": str(cache_dir),
            "formal_training_started": False,
            "sce_started": False,
            "dpl_started": False,
        },
        "resource_summary": {
            "initial_host_rss_kb": initial_rss,
            "host_rss_kb": _rss_kb(),
            "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "gpu_memory": {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": max((row["resource"]["gpu_peak_allocated_bytes"] for row in rows), default=0),
                "peak_reserved_bytes": max((row["resource"]["gpu_peak_reserved_bytes"] for row in rows), default=0),
            },
            "rss_stop_kb": RSS_STOP_KB,
        },
        "cache_final": _cache_info(cache_dir),
        "graph_break_total": sum(row["compile_audit"]["graph_breaks"] for row in rows),
        "recompilation_total": sum(row["compile_audit"]["recompilations"] for row in rows),
        "verdict": "S4 generated/compiled parity smoke passed" if len(rows) == len(MOTHER_MODELS) and not failures else "S4 runtime smoke failed or stopped",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    payload = run_smoke(args.output, args.cache_dir)
    print(json.dumps({"status": payload["status"], "models": len(payload["rows"]), "graph_breaks": payload["graph_break_total"], "recompilations": payload["recompilation_total"], "verdict": payload["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
