"""Bounded GPU-first validation for runtime-generated FUSE steps.

This script intentionally validates only the four mother structures first.  It
never starts SCE/dPL or a 78/1248-structure sweep.  The persistent-cache
cross-process probe is kept as a separate invocation so each process has a
fresh Python/Dynamo registry.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from dfuse import get_structure, reset_compile_diagnostics, runtime_compile_diagnostics
from dfuse.kernel import (
    _capacity,
    _day_of_year,
    _initial_state,
    _make_sequential_step,
    _parameter_values,
    _parameter_vector,
    _routing_fractions,
    _sequential_project_union,
    _sequential_union_state,
    _structure_context,
    _topographic_mean,
)
from dfuse.spec import FLUX_NAMES, PARAMETER_NAMES, STATE_NAMES, default_parameters
from dfuse.runtime import get_compiled_step, get_generated_step


MOTHER_MODELS = (2, 108, 178, 210)
EXTRA_SMOKE_MODELS = (6, 10, 194, 216)
ORDER = "S1"
N_SUBSTEPS = 1
INPUT_SHAPES = ((len(STATE_NAMES) + 1 + 500,), (6,), (len(PARAMETER_NAMES),), (), (), (500,))


def _set_resource_limits() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _default_forcing(steps: int, device: torch.device) -> torch.Tensor:
    base = torch.tensor(
        [[5.0, 2.0, 10.0], [0.0, 2.0, -2.0], [4.0, 2.0, 10.0], [3.0, 1.5, 8.0]],
        dtype=torch.float64,
        device=device,
    )
    return base.repeat((steps + base.shape[0] - 1) // base.shape[0], 1)[:steps]


def _prepare(model_id: int, forcing: torch.Tensor, params: Mapping[str, object] | None = None) -> dict[str, Any]:
    spec = get_structure(model_id)
    params_tensor = _parameter_values(params, dtype=forcing.dtype, device=forcing.device)
    cap = _capacity(params_tensor)
    theta = _parameter_vector(params_tensor)
    context = _structure_context(spec, dtype=forcing.dtype, device=forcing.device)
    active = _initial_state(spec, params_tensor, cap, 0.25)
    state = _sequential_project_union(_sequential_union_state(active, spec), theta, context)
    days, leap = _day_of_year(forcing.shape[0], None, dtype=forcing.dtype, device=forcing.device)
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        topographic = _topographic_mean(params_tensor)
    else:
        topographic = (theta[0] * 0.0, theta[0] * 0.0)
    fractions = _routing_fractions(params_tensor["TIMEDELAY"], dtype=forcing.dtype, device=forcing.device)
    step_forcing = torch.cat(
        (
            forcing,
            days.unsqueeze(1),
            leap.to(dtype=forcing.dtype).unsqueeze(1),
            torch.ones((forcing.shape[0], 1), dtype=forcing.dtype, device=forcing.device),
        ),
        dim=1,
    )
    return {
        "spec": spec,
        "params": params_tensor,
        "theta": theta,
        "context": context,
        "packed": torch.cat((state, torch.zeros((501,), dtype=forcing.dtype, device=forcing.device))),
        "forcing": step_forcing,
        "topographic": topographic,
        "fractions": fractions,
    }


def _run(step: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]], prepared: Mapping[str, Any], *, current: bool = False) -> dict[str, torch.Tensor]:
    packed = prepared["packed"]
    q_values = []
    diagnostics = []
    current_step = step
    for row in prepared["forcing"]:
        if current:
            packed, q, diag = current_step(
                packed,
                row,
                prepared["theta"],
                prepared["context"],
                prepared["topographic"][0],
                prepared["topographic"][1],
                prepared["fractions"],
            )
        else:
            packed, q, diag = current_step(
                packed,
                row,
                prepared["theta"],
                prepared["topographic"][0],
                prepared["topographic"][1],
                prepared["fractions"],
            )
        q_values.append(q)
        diagnostics.append(diag)
    return {"packed": packed, "q": torch.stack(q_values), "diagnostics": torch.stack(diagnostics)}


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu())


def _eager_parity(model_id: int, forcing: torch.Tensor) -> dict[str, Any]:
    prepared = _prepare(model_id, forcing)
    order_tuple = ("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow")
    current_step = _make_sequential_step(order_tuple, N_SUBSTEPS)
    signature, generated = get_generated_step(prepared["spec"], order=order_tuple, n_substeps=N_SUBSTEPS)
    current = _run(current_step, prepared, current=True)
    generated_result = _run(generated, prepared)
    active_positions = [STATE_NAMES.index(name) for name in prepared["spec"].state_names]
    gradient = _gradient_probe(model_id, forcing[:2])
    return {
        "model_id": model_id,
        "graph_signature": signature.to_dict(),
        "signature_digest": signature.digest,
        "code_object_id": id(generated.__code__),
        "code_object_name": generated.__qualname__,
        "packed_max_abs": _max_diff(current["packed"], generated_result["packed"]),
        "q_max_abs": _max_diff(current["q"], generated_result["q"]),
        "state_max_abs": _max_diff(current["packed"][active_positions], generated_result["packed"][active_positions]),
        "diagnostics_max_abs": _max_diff(current["diagnostics"], generated_result["diagnostics"]),
        "water_balance_max_abs": _max_diff(
            current["diagnostics"][:, len(FLUX_NAMES)], generated_result["diagnostics"][:, len(FLUX_NAMES)]
        ),
        "finite": bool(
            torch.isfinite(generated_result["packed"]).all()
            and torch.isfinite(generated_result["q"]).all()
            and torch.isfinite(generated_result["diagnostics"]).all()
        ),
        "gradient": gradient,
    }


def _gradient_probe(model_id: int, forcing: torch.Tensor) -> dict[str, Any]:
    defaults = default_parameters()
    order_tuple = ("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow")
    eager_params = {
        name: torch.tensor(value, dtype=forcing.dtype, device=forcing.device, requires_grad=True)
        for name, value in defaults.items()
    }
    generated_params = {
        name: torch.tensor(value, dtype=forcing.dtype, device=forcing.device, requires_grad=True)
        for name, value in defaults.items()
    }
    eager_prepared = _prepare(model_id, forcing, eager_params)
    generated_prepared = _prepare(model_id, forcing, generated_params)
    eager = _run(_make_sequential_step(order_tuple, N_SUBSTEPS), eager_prepared, current=True)
    _, generated_step = get_generated_step(get_structure(model_id), order=order_tuple, n_substeps=N_SUBSTEPS)
    generated = _run(generated_step, generated_prepared)
    eager_grads = torch.autograd.grad(eager["q"].sum(), tuple(eager_params.values()), allow_unused=True)
    generated_grads = torch.autograd.grad(generated["q"].sum(), tuple(generated_params.values()), allow_unused=True)
    active = set(get_structure(model_id).parameter_names)
    active_errors = []
    inactive_values = []
    finite = True
    for name, left, right in zip(eager_params, eager_grads, generated_grads):
        if left is not None:
            finite = finite and bool(torch.isfinite(left).all())
        if right is not None:
            finite = finite and bool(torch.isfinite(right).all())
        if name in active:
            left_value = torch.zeros((), device=forcing.device, dtype=forcing.dtype) if left is None else left
            right_value = torch.zeros((), device=forcing.device, dtype=forcing.dtype) if right is None else right
            error = float((left_value - right_value).abs().detach().cpu())
            finite = finite and math.isfinite(error)
            active_errors.append(error)
        elif right is not None:
            value = float(right.abs().max().detach().cpu())
            finite = finite and math.isfinite(value)
            inactive_values.append(value)
    return {
        "model_id": model_id,
        "active_gradient_max_abs": max(active_errors, default=0.0),
        "inactive_gradient_max_abs": max(inactive_values, default=0.0),
        "inactive_gradients_zero_or_none": max(inactive_values, default=0.0) == 0.0,
        "finite": finite,
    }

def _compiled_gradient_probe(model_id: int, forcing: torch.Tensor, generated, compiled) -> dict[str, Any]:
    defaults = default_parameters()
    generated_params = {
        name: torch.tensor(value, dtype=forcing.dtype, device=forcing.device, requires_grad=True)
        for name, value in defaults.items()
    }
    compiled_params = {
        name: torch.tensor(value, dtype=forcing.dtype, device=forcing.device, requires_grad=True)
        for name, value in defaults.items()
    }
    generated_result = _run(generated, _prepare(model_id, forcing, generated_params))
    compiled_result = _run(compiled, _prepare(model_id, forcing, compiled_params))
    generated_grads = torch.autograd.grad(generated_result["q"].sum(), tuple(generated_params.values()), allow_unused=True)
    compiled_grads = torch.autograd.grad(compiled_result["q"].sum(), tuple(compiled_params.values()), allow_unused=True)
    active = set(get_structure(model_id).parameter_names)
    active_errors = []
    inactive_values = []
    finite = True
    for name, generated_grad, compiled_grad in zip(generated_params, generated_grads, compiled_grads):
        if generated_grad is not None:
            finite = finite and bool(torch.isfinite(generated_grad).all())
        if compiled_grad is not None:
            finite = finite and bool(torch.isfinite(compiled_grad).all())
        if name in active:
            generated_value = torch.zeros((), device=forcing.device, dtype=forcing.dtype) if generated_grad is None else generated_grad
            compiled_value = torch.zeros((), device=forcing.device, dtype=forcing.dtype) if compiled_grad is None else compiled_grad
            error = float((generated_value - compiled_value).abs().detach().cpu())
            finite = finite and math.isfinite(error)
            active_errors.append(error)
        elif compiled_grad is not None:
            value = float(compiled_grad.abs().max().detach().cpu())
            finite = finite and math.isfinite(value)
            inactive_values.append(value)
    return {
        "model_id": model_id,
        "active_gradient_max_abs": max(active_errors, default=0.0),
        "inactive_gradient_max_abs": max(inactive_values, default=0.0),
        "inactive_gradients_zero_or_none": max(inactive_values, default=0.0) == 0.0,
        "finite": finite,
    }


def _compiled_parity(model_id: int, forcing: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = _prepare(model_id, forcing)
    order_tuple = ("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow")
    signature, generated = get_generated_step(prepared["spec"], order=order_tuple, n_substeps=N_SUBSTEPS)
    _, compiled = get_compiled_step(
        prepared["spec"],
        order=order_tuple,
        n_substeps=N_SUBSTEPS,
        device=forcing.device,
        dtype=forcing.dtype,
        input_shapes=INPUT_SHAPES,
        backend="inductor",
        fullgraph=True,
    )
    eager = _run(generated, prepared)
    compiled_result = _run(compiled, prepared)
    parity = {
        "model_id": model_id,
        "signature_digest": signature.digest,
        "q_max_abs": _max_diff(eager["q"], compiled_result["q"]),
        "packed_max_abs": _max_diff(eager["packed"], compiled_result["packed"]),
        "diagnostics_max_abs": _max_diff(eager["diagnostics"], compiled_result["diagnostics"]),
        "finite": bool(torch.isfinite(compiled_result["packed"]).all() and torch.isfinite(compiled_result["q"]).all()),
    }
    gradient = _compiled_gradient_probe(model_id, forcing[:2], generated, compiled)
    parity["gradient"] = gradient
    return parity, runtime_compile_diagnostics()


def _reuse_audit(model_id: int, forcing: torch.Tensor) -> dict[str, Any]:
    spec = get_structure(model_id)
    before = runtime_compile_diagnostics()
    for offset in (0.0, 0.1, -0.2, 0.3):
        changed_forcing = forcing + offset
        changed_params = default_parameters()
        parameter_name = spec.parameter_names[min(1, len(spec.parameter_names) - 1)]
        changed_params[parameter_name] += offset * 0.01
        prepared = _prepare(model_id, changed_forcing, changed_params)
        _, compiled = get_compiled_step(
            spec,
            order=("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow"),
            n_substeps=N_SUBSTEPS,
            device=forcing.device,
            dtype=forcing.dtype,
            input_shapes=INPUT_SHAPES,
        )
        _run(compiled, prepared)
    # A changed initial basin state has the same fixed shape and therefore the
    # same compiled entry; it is deliberately not a new graph signature.
    prepared = _prepare(model_id, forcing)
    prepared["packed"] = prepared["packed"].clone()
    prepared["packed"][0] = prepared["packed"][0] + 0.01
    _, compiled = get_compiled_step(
        spec,
        order=("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow"),
        n_substeps=N_SUBSTEPS,
        device=forcing.device,
        dtype=forcing.dtype,
        input_shapes=INPUT_SHAPES,
    )
    _run(compiled, prepared)
    after = runtime_compile_diagnostics()
    records = [record for record in after["records"].values() if record["graph_signature"] == spec_signature(spec)]
    record = records[0]
    return {
        "model_id": model_id,
        "registry_size_before": before["compiled_registry_size"],
        "registry_size_after": after["compiled_registry_size"],
        "compile_attempts": record["compile_attempts"],
        "compile_successes": record["compile_successes"],
        "calls": record["calls"],
        "unique_graphs": record["unique_graphs"],
        "graph_breaks": record["graph_breaks"],
        "recompilations": record["recompilations"],
        "guard_failure_reasons": record["guard_failure_reasons"],
    }


def spec_signature(spec) -> dict[str, Any]:
    _, step = get_generated_step(spec, order=("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow"), n_substeps=N_SUBSTEPS)
    return step.graph_signature.to_dict()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn: Callable[[], Any], device: torch.device, repeats: int = 2) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        fn()
        _sync(device)
        values.append(time.perf_counter() - started)
    return {"min_seconds": min(values), "median_seconds": statistics.median(values), "max_seconds": max(values)}

def _measure_backward(fn: Callable[[], torch.Tensor], device: torch.device, repeats: int = 2) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        loss = fn()
        _sync(device)
        started = time.perf_counter()
        loss.backward()
        _sync(device)
        values.append(time.perf_counter() - started)
        del loss
    return {"min_seconds": min(values), "median_seconds": statistics.median(values), "max_seconds": max(values)}


def _series(step, prepared: Mapping[str, Any], steps: int):
    packed = prepared["packed"]
    q_total = packed[0] * 0.0
    for row in prepared["forcing"][:steps]:
        packed, q, _ = step(packed, row, prepared["theta"], prepared["topographic"][0], prepared["topographic"][1], prepared["fractions"])
        q_total = q_total + q
    return packed, q_total


def _benchmark_one(model_id: int, device: torch.device) -> dict[str, Any]:
    forcing = _default_forcing(128, device)
    prepared = _prepare(model_id, forcing)
    spec = prepared["spec"]
    order_tuple = ("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow")
    _, eager = get_generated_step(spec, order=order_tuple, n_substeps=N_SUBSTEPS)
    _, compiled = get_compiled_step(spec, order=order_tuple, n_substeps=N_SUBSTEPS, device=device, dtype=forcing.dtype, input_shapes=INPUT_SHAPES)
    _series(compiled, prepared, 128)
    _sync(device)
    result: dict[str, Any] = {"model_id": model_id, "device": torch.cuda.get_device_name(device), "dtype": str(forcing.dtype), "steps": {}}
    for steps in (32, 64, 128):
        with torch.no_grad():
            eager_t = _measure(lambda: _series(eager, prepared, steps), device)
            compiled_t = _measure(lambda: _series(compiled, prepared, steps), device)
        result["steps"][str(steps)] = {
            "forward": {"eager": eager_t, "compiled": compiled_t, "speedup": eager_t["median_seconds"] / compiled_t["median_seconds"]},
        }

        def make_loss(step):
            raw = {name: torch.tensor(value, dtype=forcing.dtype, device=device, requires_grad=True) for name, value in default_parameters().items()}
            local = _prepare(model_id, forcing[:steps], raw)
            _, total = _series(step, local, steps)
            return total

        def total_run(step):
            loss = make_loss(step)
            loss.backward()
            return loss

        # Warm both forward/backward paths before measuring steady state.
        total_run(eager)
        total_run(compiled)
        _sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        eager_bwd = _measure_backward(lambda: make_loss(eager), device)
        eager_peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        compiled_bwd = _measure_backward(lambda: make_loss(compiled), device)
        compiled_peak = torch.cuda.max_memory_allocated(device)
        eager_total = _measure(lambda: total_run(eager), device)
        compiled_total = _measure(lambda: total_run(compiled), device)
        result["steps"][str(steps)]["backward"] = {
            "eager": eager_bwd, "compiled": compiled_bwd, "speedup": eager_bwd["median_seconds"] / compiled_bwd["median_seconds"],
        }
        result["steps"][str(steps)]["forward_backward"] = {
            "eager": eager_total, "compiled": compiled_total, "speedup": eager_total["median_seconds"] / compiled_total["median_seconds"],
        }
        result["steps"][str(steps)]["peak_gpu_memory_bytes"] = {"eager": eager_peak, "compiled": compiled_peak}
    return result


def _environment(cache_dir: str | None) -> dict[str, Any]:
    import triton
    import resource
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "triton": triton.__version__,
        "dtype": "torch.float64",
        "cache_dir": cache_dir or os.environ.get("TORCHINDUCTOR_CACHE_DIR", "default torch cache"),
        "torch_compiler_save_cache_artifacts": hasattr(torch.compiler, "save_cache_artifacts"),
        "torch_compiler_load_cache_artifacts": hasattr(torch.compiler, "load_cache_artifacts"),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "host_peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "formal_training_started": False,
        "sce_started": False,
        "dpl_started": False,
    }


def run_validation(output: Path, *, include_extra_smoke: bool = False) -> dict[str, Any]:
    _set_resource_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-first validation requires CUDA; refusing CPU fallback")
    device = torch.device("cuda")
    forcing = _default_forcing(4, device)
    reset_compile_diagnostics()
    eager_rows = [_eager_parity(model_id, forcing) for model_id in MOTHER_MODELS]
    if not all(
        row["finite"]
        and row["gradient"]["finite"]
        and row["gradient"]["inactive_gradients_zero_or_none"]
        and row["q_max_abs"] <= 1.0e-12
        and row["packed_max_abs"] <= 1.0e-12
        and row["diagnostics_max_abs"] <= 1.0e-12
        for row in eager_rows
    ):
        raise AssertionError("generated eager step does not match current sequential implementation")

    compiled_rows = []
    reuse_rows = []
    for model_id in MOTHER_MODELS:
        parity, _ = _compiled_parity(model_id, forcing)
        if (
            not parity["finite"]
            or not parity["gradient"]["finite"]
            or not parity["gradient"]["inactive_gradients_zero_or_none"]
            or max(parity["q_max_abs"], parity["packed_max_abs"], parity["diagnostics_max_abs"], parity["gradient"]["active_gradient_max_abs"]) > 1.0e-12
        ):
            raise AssertionError(f"compiled parity failed for model {model_id}: {parity}")
        compiled_rows.append(parity)
        reuse_rows.append(_reuse_audit(model_id, forcing))

    extra_rows = []
    if include_extra_smoke:
        # The extra smoke test is reached only after all four mothers passed.
        for model_id in EXTRA_SMOKE_MODELS:
            parity, _ = _compiled_parity(model_id, forcing[:2])
            if (
                not parity["finite"]
                or not parity["gradient"]["finite"]
                or not parity["gradient"]["inactive_gradients_zero_or_none"]
                or max(parity["q_max_abs"], parity["packed_max_abs"], parity["diagnostics_max_abs"], parity["gradient"]["active_gradient_max_abs"]) > 1.0e-12
            ):
                raise AssertionError(f"extra smoke compile/parity failed for model {model_id}: {parity}")
            extra_rows.append(parity)
    cross_structure = {
        "unique_mother_signature_count": len({row["signature_digest"] for row in eager_rows}),
        "unique_mother_code_object_count": len({row["code_object_id"] for row in eager_rows}),
        "model_id_absent_from_signature": all("model_id" not in row["graph_signature"] for row in eager_rows),
        "signature_digests": {str(row["model_id"]): row["signature_digest"] for row in eager_rows},
    }
    benchmark_compile_audit = runtime_compile_diagnostics()
    benchmark = [_benchmark_one(model_id, device) for model_id in MOTHER_MODELS]
    payload = {
        "schema": "dfuse-runtime-step-validation-v1",
        "scope": {"mother_models": list(MOTHER_MODELS), "extra_smoke_models": list(EXTRA_SMOKE_MODELS if include_extra_smoke else ()), "full_78_run": False, "full_1248_run": False},
        "environment": _environment(os.environ.get("TORCHINDUCTOR_CACHE_DIR")),
        "order": list(("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow")),
        "n_substeps": N_SUBSTEPS,
        "generated_eager_vs_current": eager_rows,
        "cross_structure_isolation": cross_structure,
        "compiled_vs_generated_eager": compiled_rows,
        "reuse_audit": reuse_rows,
        "extra_smoke": extra_rows,
        "runtime_compile_diagnostics": benchmark_compile_audit,
        "benchmark": benchmark,
        "persistent_cache_cross_process": {
            "process_1": "project/autofuse/docs/runtime_cache_process1.json",
            "process_2": "project/autofuse/docs/runtime_cache_process2.json",
            "cache_probe_mode": "cache-probe",
        },
        "architecture_verdict": "runtime step builder + precompile architecture viable",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def cache_probe(output: Path, cache_dir: str, process: int) -> None:
    _set_resource_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("cache probe requires CUDA")
    device = torch.device("cuda")
    forcing = _default_forcing(1, device)
    reset_compile_diagnostics()
    before = _cache_info(Path(cache_dir))
    started = time.perf_counter()
    result = __import__("dfuse").simulate_sequential(2, forcing, order=ORDER, n_substeps=N_SUBSTEPS, compile_step=True)
    _sync(device)
    elapsed = time.perf_counter() - started
    diag = runtime_compile_diagnostics()
    record = next(iter(diag["records"].values()))
    payload = {
        "cache_dir": str(Path(cache_dir).resolve()),
        "process": process,
        "elapsed_seconds": elapsed,
        "cache_before": before,
        "cache_after": _cache_info(Path(cache_dir)),
        "cold_compile_seconds": record["cold_compile_seconds"],
        "dynamo_counters": record["dynamo_counters"],
        "unique_graphs": record["unique_graphs"],
        "q": float(result.q.detach().cpu()[0]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _cache_info(path: Path) -> dict[str, int]:
    files = 0
    size = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
                size += item.stat().st_size
    return {"files": files, "bytes": size}


def compare_cache_probes(output: Path, process1: Path, process2: Path) -> dict[str, Any]:
    first = json.loads(process1.read_text())
    second = json.loads(process2.read_text())
    fx_hits = int(second.get("dynamo_counters", {}).get("inductor", {}).get("fxgraph_cache_hit", 0))
    autograd_hits = int(second.get("dynamo_counters", {}).get("aot_autograd", {}).get("autograd_cache_hit", 0))
    same_cache_dir = first.get("cache_dir") == second.get("cache_dir")
    cache_stable = same_cache_dir and first["cache_after"] == second["cache_before"] == second["cache_after"]
    q_stable = abs(float(first["q"]) - float(second["q"])) <= 1.0e-12
    latency_reduced = float(second["elapsed_seconds"]) < 0.75 * float(first["elapsed_seconds"])
    payload = {
        "process_1": first,
        "process_2": second,
        "fxgraph_cache_hit": fx_hits,
        "autograd_cache_hit": autograd_hits,
        "cache_artifacts_stable": cache_stable,
        "same_cache_dir": same_cache_dir,
        "output_stable": q_stable,
        "warm_latency_at_most_75pct_of_cold": latency_reduced,
        "persistent_cache_reuse_verified": bool(cache_stable and q_stable and fx_hits >= 1 and autograd_hits >= 1 and latency_reduced),
    }
    if not payload["persistent_cache_reuse_verified"]:
        raise AssertionError(f"persistent cache reuse was not verified: {payload}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"persistent_cache_reuse_verified": True, "fxgraph_cache_hit": fx_hits, "autograd_cache_hit": autograd_hits}, sort_keys=True))
    return payload

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "cache-probe", "cache-compare"), default="validate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""))
    parser.add_argument("--process", type=int, default=0)
    parser.add_argument("--extra-smoke", action="store_true")
    parser.add_argument("--process1", type=Path)
    parser.add_argument("--process2", type=Path)
    args = parser.parse_args()
    if args.cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.cache_dir
    if args.mode == "cache-probe":
        if not args.cache_dir:
            raise SystemExit("--cache-dir is required for cache-probe")
        cache_probe(args.output, args.cache_dir, args.process)
        return
    if args.mode == "cache-compare":
        if args.process1 is None or args.process2 is None:
            raise SystemExit("--process1 and --process2 are required for cache-compare")
        compare_cache_probes(args.output, args.process1, args.process2)
        return
    payload = run_validation(args.output, include_extra_smoke=args.extra_smoke)
    print(json.dumps({"mothers": len(payload["generated_eager_vs_current"]), "compiled": len(payload["compiled_vs_generated_eager"]), "extra": len(payload["extra_smoke"]), "verdict": payload["architecture_verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
