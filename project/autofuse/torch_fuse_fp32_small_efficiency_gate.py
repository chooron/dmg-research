"""Minimal real dPL FP32-vs-FP64 efficiency gate.

This benchmark intentionally measures only the frozen B=100,T=730 Lite training
step for structures 2 and 8.  It reuses the existing frozen synthetic dPL input
and parameterizer probe, and never computes precision or long-horizon metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dfuse import batched_compile_diagnostics, simulate_coupled_rk2_batched
from project.autofuse.torch_fuse_fp32_precision_efficiency_audit import (
    BATCH,
    SCORED,
    WARMUP,
    WINDOW,
    _base_training_inputs,
    _device,
    _dtype_name,
    _make_shared_model_state,
    _model_from_state,
    _reset_compilers,
    _set_environment,
    _sync,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "project/autofuse/docs/torch_fuse_fp32_small_efficiency_gate.json"
STRUCTURES = (2, 8)
REPEATS = 3
LEARNING_RATE = 1.0e-3
INITIAL_FRACTION = 0.25


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _write(value: Mapping[str, Any]) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch_hash(base: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for name, value in zip(("forcing", "attributes", "observed", "initial_theta"), base):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_gradients(model: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def _compile_record() -> dict[str, Any]:
    records = list(batched_compile_diagnostics().get("records", {}).values())
    if not records:
        return {"records": [], "compile_success": False, "compile_time_seconds": None}
    cold_compile = [value for record in records for value in record.get("cold_compile_seconds", [])]
    return {
        "records": records,
        "compile_success": bool(
            len(records) == 1
            and sum(int(record.get("compile_attempts", 0)) for record in records) == 1
            and sum(int(record.get("compile_successes", 0)) for record in records) == 1
            and sum(int(record.get("fallbacks", 0)) for record in records) == 0
        ),
        "compile_time_seconds": float(cold_compile[0]) if cold_compile else None,
    }


def _run_dtype(
    model_id: int,
    base: tuple[torch.Tensor, ...],
    dtype: torch.dtype,
    state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    device = _device()
    cache = _set_environment(dtype, f"small-gate-structure-{model_id}")
    _reset_compilers()
    forcing, attributes, observed, _ = (value.to(dtype=dtype) for value in base)
    model = _model_from_state(state, dtype, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    basin_ids = tuple(str(index) for index in range(BATCH))

    # Compile and warm up the exact measured T=730 Lite workload.  This full
    # training step is intentionally excluded from the three steady-state samples.
    optimizer.zero_grad(set_to_none=True)
    _sync()
    warm_started = time.perf_counter()
    warm_theta = model(attributes, model_id)
    warm_result = simulate_coupled_rk2_batched(
        model_id,
        forcing,
        warm_theta,
        basin_ids=basin_ids,
        compile_step=True,
        compile_backend="inductor",
        compile_fullgraph=True,
        monitor_chunk_size=WINDOW,
        output_mode="lite",
    )
    warm_loss = torch.mean((warm_result.q[:, WARMUP:] - observed) ** 2)
    warm_loss.backward()
    optimizer.step()
    _sync()
    warmup_wall = time.perf_counter() - warm_started
    compile = _compile_record()
    warmup_sanity = {
        "loss_finite": bool(torch.isfinite(warm_loss).item()),
        "gradients_finite": _finite_gradients(model),
        "no_nan_inf": bool(torch.isfinite(warm_result.q).all().item() and torch.isfinite(warm_loss).item()),
        "physics_q_dtype": _dtype_name(warm_result.q.dtype),
        "theta_dtype": _dtype_name(warm_theta.dtype),
        "loss_dtype": _dtype_name(warm_loss.dtype),
        "path_dtype_correct": bool(warm_result.q.dtype == dtype and warm_theta.dtype == dtype and warm_loss.dtype == dtype),
    }
    del warm_result, warm_loss, warm_theta
    optimizer.zero_grad(set_to_none=True)
    _sync()
    torch.cuda.empty_cache()

    samples: list[dict[str, Any]] = []
    for repeat in range(1, REPEATS + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        _sync()
        step_started = time.perf_counter()

        forward_started = time.perf_counter()
        theta = model(attributes, model_id)
        result = simulate_coupled_rk2_batched(
            model_id,
            forcing,
            theta,
            basin_ids=basin_ids,
            compile_step=True,
            compile_backend="inductor",
            compile_fullgraph=True,
            monitor_chunk_size=WINDOW,
            output_mode="lite",
        )
        _sync()
        forward_seconds = time.perf_counter() - forward_started

        loss_started = time.perf_counter()
        loss = torch.mean((result.q[:, WARMUP:] - observed) ** 2)
        _sync()
        loss_seconds = time.perf_counter() - loss_started

        backward_started = time.perf_counter()
        loss.backward()
        _sync()
        backward_seconds = time.perf_counter() - backward_started
        gradients_finite = _finite_gradients(model)

        optimizer_started = time.perf_counter()
        optimizer.step()
        _sync()
        optimizer_seconds = time.perf_counter() - optimizer_started
        total_seconds = time.perf_counter() - step_started

        samples.append(
            {
                "repeat": repeat,
                "forward_seconds": forward_seconds,
                "loss_seconds": loss_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "total_step_seconds": total_seconds,
                "steps_per_hour": 3600.0 / total_seconds,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "loss_finite": bool(torch.isfinite(loss).item()),
                "gradients_finite": gradients_finite,
                "no_nan_inf": bool(torch.isfinite(result.q).all().item() and torch.isfinite(loss).item() and gradients_finite),
                "path_dtype_correct": bool(
                    result.q.dtype == dtype
                    and theta.dtype == dtype
                    and loss.dtype == dtype
                    and all(parameter.dtype == dtype for parameter in model.parameters())
                ),
            }
        )
        del result, loss, theta
        torch.cuda.empty_cache()

    def median(name: str) -> float:
        return float(np.median([sample[name] for sample in samples]))

    median_total = median("total_step_seconds")
    return {
        "dtype": _dtype_name(dtype),
        "compile_cache": str(cache.relative_to(ROOT)),
        "compile_time_seconds": compile["compile_time_seconds"],
        "compile_diagnostics": compile,
        "warmup_wall_seconds": warmup_wall,
        "steady_state_repeats": REPEATS,
        "forward_median_seconds": median("forward_seconds"),
        "loss_median_seconds": median("loss_seconds"),
        "backward_median_seconds": median("backward_seconds"),
        "optimizer_median_seconds": median("optimizer_seconds"),
        "total_step_median_seconds": median_total,
        "steps_per_hour": 3600.0 / median_total,
        "peak_allocated_bytes": max(sample["peak_allocated_bytes"] for sample in samples),
        "peak_reserved_bytes": max(sample["peak_reserved_bytes"] for sample in samples),
        "peak_allocated_mb": max(sample["peak_allocated_bytes"] for sample in samples) / (1024 * 1024),
        "peak_reserved_mb": max(sample["peak_reserved_bytes"] for sample in samples) / (1024 * 1024),
        "samples": samples,
        "sanity_checks": {
            "warmup": warmup_sanity,
            "loss_finite": all(sample["loss_finite"] for sample in samples),
            "gradients_finite": all(sample["gradients_finite"] for sample in samples),
            "no_nan_inf": all(sample["no_nan_inf"] for sample in samples),
            "path_dtype_correct": all(sample["path_dtype_correct"] for sample in samples) and warmup_sanity["path_dtype_correct"],
        },
        "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _run_structure(model_id: int, batch_hash: str, base: tuple[torch.Tensor, ...], state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {"model_id": model_id, "status": "completed", "batch_hash": batch_hash}
    try:
        result["fp64"] = _run_dtype(model_id, base, torch.float64, state)
        result["fp32"] = _run_dtype(model_id, base, torch.float32, state)
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["error"] = str(exc)[-2000:]
        return result
    except Exception as exc:
        result["status"] = "runtime_failure"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[-2000:]
        return result

    fp64 = result["fp64"]
    fp32 = result["fp32"]
    for dtype_result in (fp64, fp32):
        if not dtype_result["compile_diagnostics"]["compile_success"]:
            result["status"] = "compile_failure"
            return result
        sanity = dtype_result["sanity_checks"]
        if not all(sanity[key] for key in ("loss_finite", "gradients_finite", "no_nan_inf", "path_dtype_correct")) or not all(sanity["warmup"].values()):
            result["status"] = "sanity_failure"
            return result

    def ratio(name: str) -> float:
        return fp64[name] / fp32[name]

    allocated_reduction = fp64["peak_allocated_bytes"] - fp32["peak_allocated_bytes"]
    reserved_reduction = fp64["peak_reserved_bytes"] - fp32["peak_reserved_bytes"]
    result["comparison"] = {
        "forward_speedup": ratio("forward_median_seconds"),
        "backward_speedup": ratio("backward_median_seconds"),
        "step_speedup": ratio("total_step_median_seconds"),
        "allocated_memory_reduction": {
            "bytes": allocated_reduction,
            "mb": allocated_reduction / (1024 * 1024),
            "fraction": allocated_reduction / fp64["peak_allocated_bytes"],
        },
        "reserved_memory_reduction": {
            "bytes": reserved_reduction,
            "mb": reserved_reduction / (1024 * 1024),
            "fraction": reserved_reduction / fp64["peak_reserved_bytes"],
        },
        "fp64_steps_per_hour": fp64["steps_per_hour"],
        "fp32_steps_per_hour": fp32["steps_per_hour"],
    }
    return result


def _classify(step_speedup: float) -> str:
    if step_speedup <= 1.05:
        return "FP32 speed benefit is small"
    if step_speedup <= 1.25:
        return "FP32 speed benefit is moderate"
    return "FP32 speed benefit is large enough to justify further study"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU substitution")
    device = _device()
    torch.manual_seed(20261130)
    structure_records = []
    for model_id in STRUCTURES:
        base = _base_training_inputs(model_id)
        state = _make_shared_model_state(model_id, device)
        structure_records.append(_run_structure(model_id, _batch_hash(base), base, state))
        del state, base
        torch.cuda.empty_cache()

    completed = [record for record in structure_records if record["status"] == "completed"]
    speedups = [record["comparison"]["step_speedup"] for record in completed]
    if speedups:
        median_speedup = float(np.median(speedups))
        mean_speedup = float(np.mean(speedups))
        classification = _classify(median_speedup)
        decision = (
            "FP32 performance benefit insufficient — keep FP64"
            if median_speedup <= 1.05
            else "FP32 performance benefit meaningful — run one short optimizer trajectory before any adoption"
        )
    else:
        median_speedup = math.nan
        mean_speedup = math.nan
        classification = "not measurable"
        decision = "FP32 performance benefit insufficient — keep FP64"

    payload = {
        "schema_version": "torch-fuse-fp32-small-efficiency-gate-v1",
        "status": "completed" if len(completed) == len(STRUCTURES) else "partial",
        "generated_at_utc": _now(),
        "reference": {
            "git_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "execution_layer_reference_id": "torch-fuse-execution-layer-v1-2-v1",
            "dtype_reference": "float64",
            "temporal_execution": "one-step compiled kernel + Python loop",
            "training_mode": "Lite",
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
        },
        "protocol": {
            "structures": list(STRUCTURES),
            "batch_size": BATCH,
            "window_days": WINDOW,
            "warmup_days": WARMUP,
            "scored_days": SCORED,
            "mode": "Lite",
            "steady_state_repeats": REPEATS,
            "optimizer": "Adam lr=1e-3",
            "compile_and_cuda_warmup_excluded_from_steady_state": True,
            "cuda_synchronized_timing": True,
            "same_frozen_input_generator": True,
            "initial_fraction": INITIAL_FRACTION,
            "only_dtype_changed": True,
            "excluded": ["long_horizon_precision", "78_structure_smoke", "optimizer_trajectory", "temporal_compile", "chunked_compile", "AMP", "TF32", "FP16", "BF16", "TBPTT", "formal_dPL_training"],
        },
        "batch_hashes": {
            str(record["model_id"]): {
                "sha256": record["batch_hash"],
                "seed": 20260901 + int(record["model_id"]) + WINDOW,
                "model_initialization_seed": 20261001 + int(record["model_id"]),
                "hash_basis": "same FP64 generated forcing/attributes/observed/default initial-theta tensors; each dtype is a cast of this fixed batch",
            }
            for record in structure_records
        },
        "structures": structure_records,
        "aggregate": {
            "mean_step_speedup": mean_speedup,
            "median_step_speedup": median_speedup,
            "benefit_classification": classification,
            "decision": decision,
            "short_optimizer_trajectory_worthwhile": bool(speedups and median_speedup > 1.05),
        },
        "scope_note": "Efficiency-only gate; no new precision, long-horizon, 78-structure, or trajectory claims are made.",
    }
    _write(payload)
    print(json.dumps({"artifact": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": _sha256_file(ARTIFACT), "status": payload["status"], "median_step_speedup": median_speedup, "classification": classification, "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
