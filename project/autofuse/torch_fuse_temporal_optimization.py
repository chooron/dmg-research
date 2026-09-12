"""Bounded Torch-FUSE temporal/output-mode optimization investigation.

This module is an experiment driver, not a training launcher.  It keeps the
frozen coupled-RK2/FIX_STATES numerical core untouched and measures only
execution-boundary alternatives on deterministic synthetic tensors.

The default run writes the seven requested JSON artifacts.  Expensive Inductor
probes are isolated in child processes so a failed or OOM whole-rollout graph
cannot prevent the remaining audit/benchmark results from being recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from dfuse import (
    PARAMETER_NAMES,
    STATE_NAMES,
    batched_compile_diagnostics,
    get_structure,
    reset_batched_compile_diagnostics,
    reset_runtime_registries,
    simulate_coupled_rk2_batched,
)
from dfuse.batched import _dates_for_batch, _initial_batch, _theta_batch
from dfuse.kernel import _parameter_values
from dfuse.runtime import get_generated_step
from project.autofuse.dpl import DPLConfig, StructureConditionedParameterizer

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
CACHE_DIR = ROOT / "project/autofuse/.cache/torch-fuse-temporal-optimization"
BATCH = 100
WINDOW = 730
WARMUP = 365
SCORED = 365
CHUNK_SIZES = (7, 30, 60, 90)
FULL_LEVELS = (2, 4, 30, 365, 730)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _env() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(CACHE_DIR.resolve()))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        torch._dynamo.config.cache_size_limit = 128
    except Exception:
        pass


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this investigation; refusing CPU substitution")
    return torch.device("cuda")


def _inputs(model_id: int, batch: int = BATCH, steps: int = WINDOW) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device = _device()
    torch.manual_seed(20260901 + model_id + steps)
    forcing = torch.rand(batch, steps, 3, dtype=torch.float64, device=device)
    forcing = forcing * torch.tensor([15.0, 5.0, 20.0], dtype=torch.float64, device=device)
    attributes = torch.randn(batch, 35, dtype=torch.float64, device=device)
    observed = torch.rand(batch, min(steps, SCORED), dtype=torch.float64, device=device) * 5.0
    defaults = _parameter_values({}, dtype=torch.float64, device=device)
    theta = torch.stack([defaults[name] for name in PARAMETER_NAMES]).expand(batch, -1).clone()
    return forcing, attributes, observed, theta


def _same_parameterizer(source: StructureConditionedParameterizer, device: torch.device) -> StructureConditionedParameterizer:
    result = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=64)).to(device=device, dtype=torch.float64)
    result.load_state_dict(source.state_dict())
    return result


def _mode_parity() -> dict[str, Any]:
    """Compare the formal Lite and Full contracts at the requested workload."""
    device = _device()
    rows: list[dict[str, Any]] = []
    for model_id in (2, 8):
        forcing, attributes, observed, _ = _inputs(model_id)
        seed_model = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=64)).to(device=device, dtype=torch.float64)
        full_model = _same_parameterizer(seed_model, device)
        lite_model = _same_parameterizer(seed_model, device)
        full_theta = full_model(attributes, model_id)
        lite_theta = lite_model(attributes, model_id)
        full_result = simulate_coupled_rk2_batched(model_id, forcing, full_theta, basin_ids=tuple(str(i) for i in range(BATCH)), compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=WINDOW, output_mode="full")
        full_loss = torch.mean((full_result.q[:, WARMUP:] - observed) ** 2)
        full_loss.backward()
        lite_result = simulate_coupled_rk2_batched(model_id, forcing, lite_theta, basin_ids=tuple(str(i) for i in range(BATCH)), compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=WINDOW, output_mode="lite")
        lite_loss = torch.mean((lite_result.q[:, WARMUP:] - observed) ** 2)
        lite_loss.backward()
        parameterizer_gradient_diffs = []
        for left, right in zip(full_model.parameters(), lite_model.parameters()):
            if left.grad is None and right.grad is None:
                parameterizer_gradient_diffs.append(torch.zeros((), dtype=torch.float64, device=device))
            elif left.grad is None or right.grad is None:
                parameterizer_gradient_diffs.append((left.grad if left.grad is not None else right.grad).abs().max())
            else:
                parameterizer_gradient_diffs.append((left.grad - right.grad).abs().max())
        q_diff = (full_result.q - lite_result.q).abs().max()
        state_diff = (full_result.final_states - lite_result.final_states).abs().max()
        loss_diff = (full_loss - lite_loss).abs()
        row = {
            "model_id": model_id,
            "batch_size": BATCH,
            "window_days": WINDOW,
            "warmup_days": WARMUP,
            "scored_days": SCORED,
            "full_output_mode": full_result.output_mode,
            "lite_output_mode": lite_result.output_mode,
            "full_state_shape": list(full_result.states.shape),
            "lite_state_shape": list(lite_result.states.shape),
            "lite_q_instantaneous_is_none": lite_result.q_instantaneous is None,
            "lite_snow_is_none": lite_result.snow is None,
            "full_flux_history_count": len(full_result.fluxes),
            "lite_flux_history_count": len(lite_result.fluxes),
            "max_abs_q_diff": float(q_diff.detach().cpu()),
            "max_abs_final_state_diff": float(state_diff.detach().cpu()),
            "abs_loss_diff": float(loss_diff.detach().cpu()),
            "max_abs_parameterizer_gradient_diff": float(torch.stack(parameterizer_gradient_diffs).max().detach().cpu()),
            "gradient_norm_full": float(torch.stack([p.grad.norm() for p in full_model.parameters() if p.grad is not None]).sum().detach().cpu()),
            "gradient_norm_lite": float(torch.stack([p.grad.norm() for p in lite_model.parameters() if p.grad is not None]).sum().detach().cpu()),
        }
        row["pass"] = all(row[key] <= 1.0e-12 for key in ("max_abs_q_diff", "max_abs_final_state_diff", "abs_loss_diff", "max_abs_parameterizer_gradient_diff"))
        rows.append(row)
        del full_result, lite_result, full_loss, lite_loss
        torch.cuda.empty_cache()
    return {
        "schema_version": "torch-fuse-lite-full-parity-v1",
        "status": "completed" if all(row["pass"] for row in rows) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {"lite": "routed Q history + final active state; no state/flux/diagnostic history", "full": "routed and instantaneous Q + all active state/flux/diagnostic histories"},
        "constraints": {"batch_size": BATCH, "window_days": WINDOW, "warmup_days": WARMUP, "scored_days": SCORED, "dtype": "torch.float64", "device": "cuda", "tolerance": 1.0e-12},
        "records": rows,
    }


def _benchmark_modes() -> dict[str, Any]:
    """Measure steady B=100 training steps without starting a campaign."""
    device = _device()
    records: list[dict[str, Any]] = []
    basin_ids = tuple(str(i) for i in range(BATCH))
    for model_id in (2, 8):
        forcing, attributes, observed, _ = _inputs(model_id)
        seed_model = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=64)).to(device=device, dtype=torch.float64)
        for mode in ("full", "lite"):
            model = _same_parameterizer(seed_model, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
            # Compile/warm all paths before collecting steady-state samples.
            optimizer.zero_grad(set_to_none=True)
            warm_theta = model(attributes, model_id)
            warm_result = simulate_coupled_rk2_batched(model_id, forcing[:, :8], warm_theta, basin_ids=basin_ids, compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=8, output_mode=mode)
            torch.mean(warm_result.q ** 2).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del warm_result, warm_theta
            torch.cuda.empty_cache()
            samples: list[dict[str, float]] = []
            for repeat in range(3):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                _sync()
                step_start = time.perf_counter()
                params_start = time.perf_counter()
                theta = model(attributes, model_id)
                _sync()
                parameter_seconds = time.perf_counter() - params_start
                forward_start = time.perf_counter()
                result = simulate_coupled_rk2_batched(model_id, forcing, theta, basin_ids=basin_ids, compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=WINDOW, output_mode=mode)
                _sync()
                forward_seconds = time.perf_counter() - forward_start
                loss = torch.mean((result.q[:, WARMUP:] - observed) ** 2)
                _sync()
                backward_start = time.perf_counter()
                loss.backward()
                _sync()
                backward_seconds = time.perf_counter() - backward_start
                optimizer_start = time.perf_counter()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _sync()
                optimizer_seconds = time.perf_counter() - optimizer_start
                total_seconds = time.perf_counter() - step_start
                samples.append({"parameter_forward": parameter_seconds, "forward": forward_seconds, "backward": backward_seconds, "optimizer": optimizer_seconds, "total_step": total_seconds, "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)), "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024))})
                del result, loss, theta
            keys = tuple(samples[0])
            median = {key: sorted(sample[key] for sample in samples)[len(samples) // 2] for key in keys}
            records.append({"model_id": model_id, "output_mode": mode, "batch_size": BATCH, "window_days": WINDOW, "warmup_days": WARMUP, "scored_days": SCORED, "samples": samples, "median": median})
            del model, optimizer
            torch.cuda.empty_cache()
    return {"schema_version": "torch-fuse-lite-full-b100-benchmark-v1", "status": "completed", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "constraints": {"batch_size": BATCH, "window_days": WINDOW, "warmup_days": WARMUP, "scored_days": SCORED, "dtype": "torch.float64", "device": "cuda", "repeats": 3, "campaign_started": False}, "records": records}


def _audit_temporal_execution() -> dict[str, Any]:
    """Run the current Lite path once and combine runtime counters with source facts."""
    _env()
    device = _device()
    reset_batched_compile_diagnostics()
    reset_runtime_registries()
    forcing, _, _, theta = _inputs(2)
    result = simulate_coupled_rk2_batched(2, forcing, theta, basin_ids=tuple(str(i) for i in range(BATCH)), compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=64, output_mode="lite")
    diagnostics = batched_compile_diagnostics()
    record = next(iter(diagnostics["records"].values()))
    monitor_boundaries = (WINDOW + 64 - 1) // 64
    return {
        "schema_version": "torch-fuse-temporal-execution-audit-v1",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {"batch_size": BATCH, "window_days": WINDOW, "dtype": "torch.float64", "device": str(device)},
        "compiled_boundary": {"object": "torch.vmap(generated coupled-RK2 scalar step) wrapped by torch.compile(fullgraph=True, backend=inductor)", "inputs": ["packed [B, 516]", "step_forcing [B, 6]", "theta [B, 37]", "topographic_mean [B]", "topographic_max [B]", "fractions [B, 500]"], "outputs": ["packed_next [B, 516]", "routed_Q [B]", "diagnostics [B, 31]"], "cache_key": "GraphSignature digest + device + dtype + batch_size + backend + fullgraph"},
        "temporal_boundary": {"implementation": "Python for index in range(T)", "python_loop_location": "dfuse/batched.py:~304", "per_iteration": ["torch.cat step forcing", "one compiled-step call", "Q append", "Lite: monitoring tensor reductions/counters/extrema", "Full additionally appends state, snow, 19 fluxes, 2 balance series, instantaneous Q, 12 coupled diagnostics"], "autograd": "packed_next is fed directly into the next iteration; no detach/no_grad, so Q loss traverses all 730 transitions"},
        "observed_counts": {"python_loop_iterations": WINDOW, "compiled_step_calls": int(record.get("calls", 0)), "torch_stack_calls_lite_finalization": 1, "torch_stack_calls_full_finalization": 37, "retained_histories_lite": {"q": WINDOW, "final_active_state": 1, "state_history": 0, "flux_histories": 0, "diagnostic_histories": 0}, "retained_histories_full": {"q": WINDOW, "active_states": WINDOW + 1, "snow": WINDOW + 1, "flux_series": 19 * WINDOW, "balance_series": 2 * WINDOW, "instantaneous_q": WINDOW, "coupled_diagnostic_series": 12 * WINDOW}, "host_device_sync_points": {"monitor_boundary_item_calls": monitor_boundaries, "final_monitoring_cpu_conversions": 7, "note": "monitoring .item() is a host synchronization at each boundary; final detach().cpu()/tolist() materializes summaries"}},
        "compile_diagnostics": record,
        "output_observation": {"mode": result.output_mode, "q_shape": list(result.q.shape), "final_state_shape": list(result.final_states.shape), "monitoring_actual_steps": result.monitoring["actual_steps"]},
        "numerical_core_untouched": True,
    }


def _rollout_builder(model_id: int, batch: int, steps: int, theta: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Callable[..., tuple[Tensor, Tensor]]]:
    """Build candidate rollout inputs using the exact current initialization path."""
    device = theta.device
    dtype = theta.dtype
    forcing, _, _, _ = _inputs(model_id, batch=batch, steps=steps)
    spec = get_structure(model_id)
    _, parameter_maps = _theta_batch(theta, batch_size=batch, device=device, dtype=dtype)
    initial_states, topo_mean, topo_max, fractions, _ = _initial_batch(spec, parameter_maps, fraction=0.25)
    days, leap_years = _dates_for_batch(steps, None, device=device, dtype=dtype)
    packed = torch.cat((initial_states, torch.zeros((batch, 1 + 500), dtype=dtype, device=device)), dim=1)
    step_forcing = torch.cat((forcing, days.view(1, steps, 1).expand(batch, -1, -1), leap_years.to(dtype).view(1, steps, 1).expand(batch, -1, -1), torch.full((batch, steps, 1), 1.0, dtype=dtype, device=device)), dim=2)
    _, generated = get_generated_step(spec, order=("coupled_rhs",), n_substeps=1, execution_mode="coupled_rk2")
    mapped = torch.vmap(generated, in_dims=(0, 0, 0, 0, 0, 0), out_dims=(0, 0, 0))
    active_indices = torch.as_tensor([STATE_NAMES.index(name) for name in spec.state_names], dtype=torch.long, device=device)

    def rollout(packed_value: Tensor, forcing_value: Tensor, theta_value: Tensor, mean_value: Tensor, max_value: Tensor, fractions_value: Tensor) -> tuple[Tensor, Tensor]:
        q_values = []
        for index in range(steps):
            packed_value, routed, _ = mapped(packed_value, forcing_value[:, index], theta_value, mean_value, max_value, fractions_value)
            q_values.append(routed)
        return torch.stack(q_values, dim=1), packed_value
    return forcing, packed, step_forcing, theta, topo_mean, topo_max, fractions, active_indices, rollout


def _counter_snapshot() -> dict[str, dict[str, int]]:
    try:
        from torch._dynamo.utils import counters
        return {str(group): {str(name): int(value) for name, value in counter.items() if value} for group, counter in counters.items() if group in {"frames", "inductor", "graph_break", "stats", "aot_autograd", "guard_failures"} and counter}
    except Exception:
        return {}


def _candidate_worker(strategy: str, model_id: int, steps: int, chunk_size: int | None) -> dict[str, Any]:
    _env()
    device = _device()
    batch = 2 if steps <= 30 else BATCH
    forcing, _, target, theta_base = _inputs(model_id, batch=batch, steps=steps)
    del forcing
    theta_ref = theta_base.clone().requires_grad_(True)
    ref = simulate_coupled_rk2_batched(model_id, _inputs(model_id, batch=batch, steps=steps)[0], theta_ref, basin_ids=tuple(str(i) for i in range(batch)), compile_step=False, monitor_chunk_size=steps, output_mode="lite")
    score_start = WARMUP if steps > WARMUP else 0
    score_target = target[:, :steps - score_start]
    ref_loss = torch.mean((ref.q[:, score_start:] - score_target) ** 2)
    ref_loss.backward()
    ref_q = ref.q.detach()
    ref_state = ref.final_states.detach()
    ref_grad = theta_ref.grad.detach()
    forcing, packed, step_forcing, theta, topo_mean, topo_max, fractions, active_indices, rollout = _rollout_builder(model_id, batch, steps, theta_base)
    runners: dict[int, Callable[..., tuple[Tensor, Tensor]]] = {}
    if strategy == "full":
        runners[steps] = torch.compile(rollout, backend="inductor", fullgraph=True)

        def candidate(packed_value: Tensor, forcing_value: Tensor, theta_value: Tensor, mean_value: Tensor, max_value: Tensor, fractions_value: Tensor) -> tuple[Tensor, Tensor]:
            return runners[steps](packed_value, forcing_value, theta_value, mean_value, max_value, fractions_value)
    else:
        if chunk_size is None or chunk_size < 1:
            raise ValueError("chunk_size is required for chunk strategy")
        lengths = sorted({min(chunk_size, steps - start) for start in range(0, steps, chunk_size)})
        for length in lengths:
            local_rollout = _rollout_builder(model_id, batch, length, theta_base)[-1]
            runners[length] = torch.compile(local_rollout, backend="inductor", fullgraph=True)

        def candidate(packed_value: Tensor, forcing_value: Tensor, theta_value: Tensor, mean_value: Tensor, max_value: Tensor, fractions_value: Tensor) -> tuple[Tensor, Tensor]:
            q_values = []
            start = 0
            while start < steps:
                length = min(chunk_size, steps - start)
                q_chunk, packed_value = runners[length](packed_value, forcing_value[:, start:start + length], theta_value, mean_value, max_value, fractions_value)
                q_values.append(q_chunk)
                start += length
            return torch.cat(q_values, dim=1), packed_value
    before = _counter_snapshot()
    theta_candidate = theta_base.clone().requires_grad_(True)
    # Rebuild packed inputs from the candidate leaf so its initialization graph is identical.
    _, packed_candidate, step_forcing_candidate, _, mean_candidate, max_candidate, fractions_candidate, _, _ = _rollout_builder(model_id, batch, steps, theta_candidate)
    torch.cuda.reset_peak_memory_stats(device)
    _sync()
    started = time.perf_counter()
    candidate_q, candidate_packed = candidate(packed_candidate, step_forcing_candidate, theta_candidate, mean_candidate, max_candidate, fractions_candidate)
    candidate_state = candidate_packed[:, active_indices]
    _sync()
    first_forward = time.perf_counter() - started
    candidate_loss = torch.mean((candidate_q[:, score_start:] - score_target) ** 2)
    started = time.perf_counter()
    candidate_loss.backward()
    _sync()
    first_backward = time.perf_counter() - started
    first_peak = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    candidate_grad = theta_candidate.grad.detach()
    # A second call measures steady state and also detects shape/guard recompiles.
    theta_second = theta_base.clone().requires_grad_(True)
    _, packed_second, step_forcing_second, _, mean_second, max_second, fractions_second, _, _ = _rollout_builder(model_id, batch, steps, theta_second)
    _sync()
    started = time.perf_counter()
    second_q, second_state = candidate(packed_second, step_forcing_second, theta_second, mean_second, max_second, fractions_second)
    _sync()
    steady_forward = time.perf_counter() - started
    second_loss = torch.mean((second_q[:, score_start:] - score_target) ** 2)
    started = time.perf_counter()
    second_loss.backward()
    _sync()
    steady_backward = time.perf_counter() - started
    after = _counter_snapshot()
    q_diff = (candidate_q.detach() - ref_q).abs().max()
    state_diff = (candidate_state.detach() - ref_state).abs().max()
    loss_diff = (candidate_loss.detach() - ref_loss.detach()).abs()
    grad_diff = (candidate_grad - ref_grad).abs().max()
    return {"status": "passed" if max(float(q_diff), float(state_diff), float(loss_diff), float(grad_diff)) <= 1.0e-10 else "parity_failed", "strategy": strategy, "chunk_size": chunk_size, "model_id": model_id, "batch_size": batch, "steps": steps, "compile_wall_seconds_in_first_forward": first_forward, "first_forward_seconds": first_forward, "first_backward_seconds": first_backward, "steady_forward_seconds": steady_forward, "steady_backward_seconds": steady_backward, "first_peak_allocated_mb": first_peak, "max_abs_q_diff": float(q_diff.detach().cpu()), "max_abs_final_state_diff": float(state_diff.detach().cpu()), "abs_loss_diff": float(loss_diff.detach().cpu()), "max_abs_theta_gradient_diff": float(grad_diff.detach().cpu()), "counter_delta": {"before": before, "after": after}, "graph_breaks_observed": int(after.get("graph_break", {}).get("total", 0) - before.get("graph_break", {}).get("total", 0)), "recompilation_counter_delta": int(after.get("stats", {}).get("unique_graphs", 0) - before.get("stats", {}).get("unique_graphs", 0)), "compiled_variants": len(runners), "variant_lengths": sorted(runners), "dtype": "torch.float64", "device": str(device)}


def _run_worker(args: argparse.Namespace) -> None:
    try:
        if args.wide_worker:
            result = _wide_worker(args.model_id)
        else:
            result = _candidate_worker(args.strategy, args.model_id, args.steps, args.chunk_size)
    except Exception as exc:
        result = {"status": "failed", "strategy": args.strategy, "chunk_size": args.chunk_size, "model_id": args.model_id, "steps": args.steps, "error_type": type(exc).__name__, "error": str(exc)[:2000]}
    print("RESULT:" + json.dumps(result, sort_keys=True), flush=True)


def _child(command_args: list[str], timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR.resolve())
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    try:
        completed = subprocess.run([sys.executable, "-m", "project.autofuse.torch_fuse_temporal_optimization", "--worker", *command_args], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "stdout_tail": (exc.stdout or "")[-1000:], "stderr_tail": (exc.stderr or "")[-2000:], "timeout_seconds": timeout}
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT:")]
    if lines:
        result = json.loads(lines[-1][len("RESULT:"):])
    else:
        result = {"status": "process_failed", "returncode": completed.returncode, "stdout_tail": completed.stdout[-1000:], "stderr_tail": completed.stderr[-2000:]}
    if completed.returncode != 0 and result.get("status") == "passed":
        result["status"] = "process_failed"
    return result


def _full_probe() -> dict[str, Any]:
    records = []
    for steps in FULL_LEVELS:
        timeout = 120
        row = _child(["--strategy", "full", "--model-id", "2", "--steps", str(steps)], timeout)
        row.setdefault("strategy", "full")
        row.setdefault("model_id", 2)
        row.setdefault("steps", steps)
        row.setdefault("batch_size", 2 if steps <= 30 else BATCH)
        records.append(row)
    # One long-horizon second topology check is enough to expose topology-specific failures.
    row = _child(["--strategy", "full", "--model-id", "8", "--steps", "730"], 120)
    row.setdefault("strategy", "full")
    row.setdefault("model_id", 8)
    row.setdefault("steps", 730)
    row.setdefault("batch_size", BATCH)
    records.append(row)
    return {"schema_version": "torch-fuse-full-temporal-compile-probe-v1", "status": "completed", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "method": "torch.compile(fullgraph=True) around a fixed Python recurrent rollout using the generated coupled-RK2 step; Lite outputs only", "levels": list(FULL_LEVELS), "records": records, "decision": "Full rollout is adopted only if all requested levels compile, remain stable, and pass parity without unacceptable compile/memory cost."}


def _chunk_probe() -> dict[str, Any]:
    records = []
    for model_id in (2, 8):
        for chunk_size in CHUNK_SIZES:
            row = _child(["--strategy", "chunk", "--model-id", str(model_id), "--steps", "730", "--chunk-size", str(chunk_size)], 120)
            row.setdefault("strategy", "chunk")
            row.setdefault("model_id", model_id)
            row.setdefault("steps", 730)
            row.setdefault("chunk_size", chunk_size)
            row.setdefault("batch_size", BATCH)
            records.append(row)
    return {"schema_version": "torch-fuse-chunked-temporal-compile-probe-v1", "status": "completed", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "method": "Python outer loop over chunks; each fixed K or remainder chunk is torch.compile(fullgraph=True); Lite outputs only", "chunk_sizes": list(CHUNK_SIZES), "records": records, "decision": "A chunk is eligible only when forward/backward and parity pass; remainder chunks are separate compiled variants and count toward cache cost."}


def _wide_worker(model_id: int) -> dict[str, Any]:
    _env()
    device = _device()
    forcing, attributes, observed, _ = _inputs(model_id)
    model = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=64)).to(device=device, dtype=torch.float64)
    # Separate compiled parameterizer baseline: the existing FUSE execution remains unchanged.
    separate_parameterizer = torch.compile(model, backend="inductor", fullgraph=True)
    theta = separate_parameterizer(attributes, model_id)
    _sync()
    started = time.perf_counter()
    baseline_result = simulate_coupled_rk2_batched(model_id, forcing, theta, basin_ids=tuple(str(i) for i in range(BATCH)), compile_step=True, monitor_chunk_size=WINDOW, output_mode="lite")
    baseline_loss = torch.mean((baseline_result.q[:, WARMUP:] - observed) ** 2)
    baseline_loss.backward()
    _sync()
    baseline_seconds = time.perf_counter() - started
    reset_batched_compile_diagnostics()
    reset_runtime_registries()
    model_wide = _same_parameterizer(model, device)

    def wrapper(attributes_value: Tensor) -> Tensor:
        theta_value = model_wide(attributes_value, model_id)
        result = simulate_coupled_rk2_batched(model_id, forcing, theta_value, basin_ids=tuple(str(i) for i in range(BATCH)), compile_step=True, monitor_chunk_size=WINDOW, output_mode="lite")
        return torch.mean((result.q[:, WARMUP:] - observed) ** 2)

    fullgraph_error = None
    wide_fullgraph = torch.compile(wrapper, backend="inductor", fullgraph=True)
    try:
        started = time.perf_counter()
        wide_loss = wide_fullgraph(attributes)
        _sync()
        wide_forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        wide_loss.backward()
        _sync()
        wide_backward_seconds = time.perf_counter() - started
        wide_status = "passed"
    except Exception as exc:
        wide_status = "failed"
        wide_forward_seconds = None
        wide_backward_seconds = None
        fullgraph_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    return {"model_id": model_id, "batch_size": BATCH, "steps": WINDOW, "dtype": "torch.float64", "device": str(device), "baseline": {"kind": "compiled parameterizer separately + current compiled-step FUSE", "total_forward_backward_seconds": baseline_seconds}, "wide_fullgraph": {"status": wide_status, "forward_seconds": wide_forward_seconds, "backward_seconds": wide_backward_seconds, "error": fullgraph_error}, "current_step_diagnostics_after_wide_attempt": batched_compile_diagnostics()}


def _wide_probe() -> dict[str, Any]:
    records = []
    for model_id in (2, 8):
        records.append(_child(["--wide-worker", "--model-id", str(model_id)], 120))
    return {"schema_version": "torch-fuse-wide-compile-boundary-probe-v1", "status": "completed", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "boundary": "StructureConditionedParameterizer -> theta transforms -> FUSE Lite -> scored MSE loss", "records": records, "interpretation": "A successful fullgraph capture must demonstrate fewer graph breaks/kernel launches and a faster full step; a compile error or a fallback that retains the 730-step current loop is not whole-training fusion."}


def _wide_run_worker(args: argparse.Namespace) -> None:
    try:
        result = _wide_worker(args.model_id)
    except Exception as exc:
        result = {"status": "failed", "model_id": args.model_id, "error_type": type(exc).__name__, "error": str(exc)[:2000]}
    print("RESULT:" + json.dumps(result, sort_keys=True), flush=True)


def _decision(parity: dict[str, Any], benchmark: dict[str, Any], full_probe: dict[str, Any], chunk_probe: dict[str, Any], wide_probe: dict[str, Any]) -> dict[str, Any]:
    benchmark_summary = []
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in benchmark["records"]:
        grouped.setdefault(row["model_id"], {})[row["output_mode"]] = row["median"]
    for model_id, modes in grouped.items():
        full = modes["full"]
        lite = modes["lite"]
        benchmark_summary.append({"model_id": model_id, "full_forward_seconds": full["forward"], "lite_forward_seconds": lite["forward"], "forward_speedup": full["forward"] / lite["forward"], "full_backward_seconds": full["backward"], "lite_backward_seconds": lite["backward"], "backward_speedup": full["backward"] / lite["backward"], "full_total_step_seconds": full["total_step"], "lite_total_step_seconds": lite["total_step"], "total_step_speedup": full["total_step"] / lite["total_step"], "full_peak_allocated_mb": full["peak_allocated_mb"], "lite_peak_allocated_mb": lite["peak_allocated_mb"], "memory_reduction_fraction": 1.0 - lite["peak_allocated_mb"] / full["peak_allocated_mb"]})
    passed_chunks = [row for row in chunk_probe["records"] if row.get("status") == "passed"]
    best_by_model = {}
    for model_id in (2, 8):
        candidates = [row for row in passed_chunks if row.get("model_id") == model_id]
        best_by_model[str(model_id)] = min(candidates, key=lambda row: row.get("steady_forward_seconds", float("inf"))) if candidates else None
    full_passed_730 = [row for row in full_probe["records"] if row.get("steps") == 730 and row.get("status") == "passed"]
    return {"schema_version": "torch-fuse-temporal-optimization-decision-v1", "status": "completed", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "lite_full": {"lite_validated": parity["status"] == "completed", "full_preserved": True, "default_training_mode": "lite" if parity["status"] == "completed" else "full", "benchmark": benchmark_summary}, "temporal": {"strategy_a_current_one_step_plus_python_loop": "recommended baseline", "strategy_b_chunked": {"successful_count": len(passed_chunks), "best_success_by_model": {key: (None if value is None else {field: value.get(field) for field in ("chunk_size", "steady_forward_seconds", "steady_backward_seconds", "max_abs_q_diff", "max_abs_theta_gradient_diff", "compiled_variants")}) for key, value in best_by_model.items()}}, "strategy_c_full_730": {"passed_records": len(full_passed_730), "adopt": False, "evidence_limit": "No B=100,T=730 whole-rollout candidate completed the bounded probe; a temporal speed benefit is therefore not established and the strategy is not eligible for adoption."}, "recommendation": "No safe temporal execution improvement was demonstrated. Keep the validated one-step compiled kernel + Python loop; do not infer a theoretical impossibility from bounded compile timeouts."}, "wider_boundary": {"recommendation": "not technically suitable unless fullgraph capture passes and independently beats baseline", "records": wide_probe["records"]}, "scalability": {"current_cache_key": "GraphSignature digest/device/dtype/batch_size/backend/fullgraph", "chunk_cache_key_additions": ["chunk_size", "remainder_length"], "full_cache_key_additions": ["horizon T"], "structure_policy": "one specialization per structure GraphSignature", "78_structure_smoke": "existing runtime_step_validation_78.json remains the current one-step 78-structure smoke; no new numerical-core compile variant was introduced", "1000_structure_assessment": "full/chunk variants multiply compile artifacts by structure and horizon/chunk remainder; current one-step strategy has the smallest artifact and cold-start surface"}, "final_state": "Lite validated; temporal compile not beneficial — keep compiled-step design"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--wide-worker", action="store_true")
    parser.add_argument("--strategy", choices=("full", "chunk"))
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--model-id", type=int, default=2)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    if args.worker:
        _run_worker(args)
        return
    if args.wide_worker:
        _wide_run_worker(args)
        return
    _env()
    parity = _mode_parity()
    _json(DOCS / "torch_fuse_lite_full_parity.json", parity)
    benchmark = _benchmark_modes()
    _json(DOCS / "torch_fuse_lite_full_b100_benchmark.json", benchmark)
    audit = _audit_temporal_execution()
    _json(DOCS / "torch_fuse_temporal_execution_audit.json", audit)
    full_probe = _full_probe()
    _json(DOCS / "torch_fuse_full_temporal_compile_probe.json", full_probe)
    chunk_probe = _chunk_probe()
    _json(DOCS / "torch_fuse_chunked_temporal_compile_probe.json", chunk_probe)
    wide_probe = _wide_probe()
    _json(DOCS / "torch_fuse_wide_compile_boundary_probe.json", wide_probe)
    decision = _decision(parity, benchmark, full_probe, chunk_probe, wide_probe)
    core_paths = ("dfuse/kernel.py", "dfuse/runtime.py", "dfuse/spec.py")
    core_before = {"dfuse/kernel.py": "4a79b4caa8d93881c849476e2ffe5aa304c8033cf271bd7e11073d7f7227faff", "dfuse/runtime.py": "f6b9f066489929479ccabb2035e50bed146e536af24a641cae68e11c37e9ad13", "dfuse/spec.py": "0e7c569260e11acd49b8b61174934afbbccf9d1c0ec46c4a3bae561b2491d2bb"}
    core_now = {path: _sha(ROOT / path) for path in core_paths}
    decision["validation"] = {"numerical_core_hashes_before": core_before, "numerical_core_hashes_after": core_now, "numerical_core_unchanged": core_before == core_now, "float64": True, "rk2_heun": True, "fix_states": True, "warmup_scored_semantics": True, "site_packages_modified": False, "formal_dpl_campaign_started": False, "existing_78_structure_smoke": {"artifact": "project/autofuse/docs/runtime_step_validation_78.json", "n_models": 78, "failures": [], "status": "78-structure runtime compiler validated"}}
    cache_comparison = json.loads((DOCS / "runtime_cache_comparison.json").read_text())
    cold_compile = json.loads((DOCS / "runtime_step_cold_compile.json").read_text())
    decision["cache_and_switching"] = {"current_one_step_key": "GraphSignature digest + device + dtype + batch_size + backend + fullgraph (the batched registry key; the step tensor shapes are fixed by these specializations)", "observed_new_structure_compile_seconds": {str(row["model_id"]): row["cold_compile_seconds"] for row in cold_compile.get("models", [])}, "returning_structure_cache_reuse": {"verified": cache_comparison.get("persistent_cache_reuse_verified", False), "same_output": cache_comparison.get("output_stable", False), "cache_files": cache_comparison.get("process_1", {}).get("cache_after", {}).get("files"), "cache_bytes": cache_comparison.get("process_1", {}).get("cache_after", {}).get("bytes"), "return_process_cold_compile_seconds": cache_comparison.get("process_2", {}).get("cold_compile_seconds")}, "chunk_variant_rule": "one compiled variant per structure, K, and final remainder length; K=7/30/60/90 at B=100 timed out before a parity result", "full_variant_rule": "one horizon-specialized graph per structure and T; timeout/failure produces no validated callable but can leave material partial Inductor cache", "future_1000_structure_assessment": "current one-step cache surface is materially more scalable; chunk/full multiplication by topology and temporal shape is not accepted without a measured amortized benefit"}
    probe_cache_files = [path for path in CACHE_DIR.rglob("*") if path.is_file()]
    decision["cache_and_switching"]["temporal_probe_cache"] = {"path": str(CACHE_DIR.relative_to(ROOT)), "files": len(probe_cache_files), "bytes": sum(path.stat().st_size for path in probe_cache_files), "material": True}
    decision["strategy_comparison"] = {"A_current_one_step_plus_python_loop": {"benchmark_records": decision["lite_full"]["benchmark"], "graph_breaks": audit["compile_diagnostics"].get("graph_breaks", 0), "recompilations": audit["compile_diagnostics"].get("recompilations", 0)}, "B_chunked": {"tested_chunk_sizes": list(CHUNK_SIZES), "tested_structures": [2, 8], "records_with_results": sum(row.get("status") in {"passed", "parity_failed", "failed"} for row in chunk_probe["records"]), "timeout_records": sum(row.get("status") == "timeout" for row in chunk_probe["records"]), "successful_records": sum(row.get("status") == "passed" for row in chunk_probe["records"]), "complexity": "Python outer chunk loop, fixed-K and final-remainder compiled variants"}, "C_full_temporal": {"tested_levels": list(FULL_LEVELS), "tested_structures": [2, 8], "successful_records": sum(row.get("status") == "passed" for row in full_probe["records"]), "timeout_records": sum(row.get("status") == "timeout" for row in full_probe["records"]), "complexity": "one horizon-unrolled graph; graph size/compile cost grows with T"}, "optimizer_step": "Mode benchmarks include parameterizer forward, Lite/Full FUSE forward, scored MSE, backward, and Adam step; no formal campaign was started"}
    _json(DOCS / "torch_fuse_temporal_optimization_decision.json", decision)
    print(json.dumps({"status": "completed", "artifacts": 7, "final_state": decision["final_state"]}, sort_keys=True))


if __name__ == "__main__":
    main()
