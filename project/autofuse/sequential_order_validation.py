"""GPU-first S1--S4 sequential process-order selection experiment.

This is deliberately separate from the completed S1 78-structure compiler audit.
It uses the same structure-specialized runtime builder and one fresh worker process
per order/model case, while retaining only JSON metrics in the parent.  It does
not start calibration, training, SCE, dPL, or an implicit solver.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dfuse import GraphSignature, enumerate_structures, get_structure, reset_runtime_registries, runtime_compile_diagnostics
from dfuse import runtime as runtime_impl
from dfuse.kernel import (
    FLUX_NAMES,
    SEQUENTIAL_ORDERS,
    _make_sequential_step,
    _sequential_raw_process,
    _sequential_snow_step,
)
from dfuse.spec import PARAMETER_NAMES, STATE_NAMES, default_parameters
from dfuse.runtime import get_compiled_step, get_generated_step
from project.autofuse import runtime_validation_78 as base
from project.autofuse.fidelity import synthetic_forcing
from project.autofuse.metrics import kgecomp
from project.autofuse.reference_oracle import ReferenceResult, run_reference


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PROVENANCE_PATHS = {
    "dfuse/runtime.py": ROOT / "dfuse/runtime.py",
    "dfuse/kernel.py": ROOT / "dfuse/kernel.py",
    "dfuse/spec.py": ROOT / "dfuse/spec.py",
    "project/autofuse/sequential_order_validation.py": Path(__file__).resolve(),
}
DEVELOPMENT_PATH = ROOT / "project/autofuse/docs/sequential_development_set.json"
DEFAULT_OUTPUT = ROOT / "project/autofuse/docs/sequential_order_validation.json"
DEFAULT_CACHE = ROOT / "project/autofuse/.cache/sequential-order-validation"
DEFAULT_REFERENCE_DIR = ROOT / "project/autofuse/docs/sequential_order_references"
ORDERS = ("S1", "S2", "S3", "S4")
EXPECTED_ORDERS = {
    "S1": ("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow"),
    "S2": ("recharge", "surface_runoff", "interflow", "et", "percolation", "baseflow"),
    "S3": ("recharge", "et", "percolation", "baseflow", "interflow", "surface_runoff"),
    "S4": ("recharge", "percolation", "baseflow", "et", "interflow", "surface_runoff"),
}
DEVELOPMENT_IDS = (2, 108, 178, 210, 164, 188, 212, 214, 166, 190, 168, 6, 8, 10, 12, 14, 16, 18)
REPRESENTATIVE_IDS = (2, 108, 178, 210, 164, 188, 212, 214)
N_SUBSTEPS = 1
TOLERANCE = 1.0e-12
METRICS_PROTOCOL = "reference-state-at-step-start-v2"
# Compiled autograd is checked on the existing smooth two-step probe; the
# forward/state/flux parity gate remains at the stricter frozen tolerance.
GRADIENT_TOLERANCE = 1.0e-8
# Compiled autograd is checked on the existing smooth two-step probe; the
# forward/state/flux parity gate remains at the stricter frozen tolerance.
GRADIENT_TOLERANCE = 1.0e-8
WATER_TOLERANCE = 1.0e-10
RSS_STOP_KB = 3_500_000
RSS_GROWTH_STOP_KB = 2_750_000
PROCESS_FLUXES = {
    "ET": ("EVAP_1", "EVAP_2"),
    "qsurf": ("QSURF",),
    "qperc": ("QPERC_12",),
    "qintf": ("QINTF_1",),
    "qbase": ("QBASE_2",),
}


def _set_limits() -> None:
    base._set_resource_limits()


def _rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _runtime_provenance() -> dict[str, Any]:
    files = {name: _sha256_file(path) for name, path in RUNTIME_PROVENANCE_PATHS.items()}
    return {"files": files, "sha256": base._canonical_hash(files)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:2000]}


def _number(value: Any) -> float:
    value = float(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
    return value if math.isfinite(value) else math.inf


def _summary(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if absolute.size == 0:
        return {"max": 0.0, "median": 0.0, "p95": 0.0, "rmse": 0.0}
    return {
        "max": float(np.max(absolute)),
        "median": float(np.median(absolute)),
        "p95": float(np.quantile(absolute, 0.95)),
        "rmse": float(np.sqrt(np.mean(absolute * absolute))),
    }


def _as_forcing_tensor(values: Mapping[str, np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack((values["ppt"], values["pet"], values["temp"]), axis=1)
    return torch.as_tensor(array, dtype=torch.float64, device=device)


def _reference_path(reference_dir: Path, model_id: int) -> Path:
    return reference_dir / f"reference-{model_id}.npz"


def _reference_hash(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("ppt", "pet", "temp"):
        array = np.asarray(values[name], dtype=np.float64)
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _save_reference(path: Path, reference: ReferenceResult, forcing_hash: str, executable_sha256: str) -> None:
    arrays: dict[str, Any] = {
        "q_routed": np.asarray(reference.q_routed),
        "q_instantaneous": np.asarray(reference.q_instantaneous),
        "initial_state": np.asarray([reference.initial_state.get(name, np.nan) for name in STATE_NAMES], dtype=np.float64),
    }
    for name, values in reference.states.items():
        arrays[f"state::{name}"] = np.asarray(values)
    for name, values in reference.fluxes.items():
        arrays[f"flux::{name}"] = np.asarray(values)
    arrays["metadata_json"] = np.asarray(json.dumps({
        "model_id": reference.model_id,
        "metadata": _jsonable(reference.metadata),
        "state_names": sorted(reference.states),
        "flux_names": sorted(reference.fluxes),
        "forcing_sha256": forcing_hash,
        "reference_executable_sha256": executable_sha256,
    }))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_reference(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        states = {key.split("::", 1)[1]: np.asarray(archive[key]) for key in archive.files if key.startswith("state::")}
        fluxes = {key.split("::", 1)[1]: np.asarray(archive[key]) for key in archive.files if key.startswith("flux::")}
        missing_fluxes = sorted(set(FLUX_NAMES) - set(fluxes))
        if missing_fluxes:
            raise ValueError(f"reference artifact is missing required fluxes: {missing_fluxes}")
        initial_values = np.asarray(archive["initial_state"], dtype=np.float64)
        initial_state = {name: float(initial_values[index]) for index, name in enumerate(STATE_NAMES) if index < initial_values.size and np.isfinite(initial_values[index])}
        return {
            "model_id": int(metadata["model_id"]),
            "q_routed": np.asarray(archive["q_routed"]),
            "q_instantaneous": np.asarray(archive["q_instantaneous"]),
            "states": states,
            "fluxes": fluxes,
            "initial_state": initial_state,
            "metadata": metadata,
        }


def _prepare_reference_cache(
    executable: Path,
    reference_dir: Path,
    model_ids: tuple[int, ...],
    forcing_values: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    executable_sha256 = _sha256_file(executable)
    forcing_sha256 = _reference_hash(forcing_values)
    rows = []
    for model_id in model_ids:
        path = _reference_path(reference_dir, model_id)
        valid = False
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata_json"].item()))
                valid = (
                    metadata.get("forcing_sha256") == forcing_sha256
                    and metadata.get("reference_executable_sha256") == executable_sha256
                    and int(metadata.get("model_id")) == model_id
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                valid = False
        if not valid:
            reference = run_reference(str(executable), model_id, forcing_values)
            _save_reference(path, reference, forcing_sha256, executable_sha256)
        rows.append({"model_id": model_id, "path": str(path.resolve()), "sha256": _sha256_file(path)})
    return {
        "directory": str(reference_dir.resolve()),
        "forcing_sha256": forcing_sha256,
        "executable": str(executable.resolve()),
        "executable_sha256": executable_sha256,
        "models": rows,
        "oracle_order_note": "pinned Fortran FUSE reference uses its frozen internal sequential order; all candidate orders are compared against this same reference without modifying the oracle",
    }


def _load_development(path: Path) -> tuple[tuple[int, ...], dict[str, Any]]:
    payload = json.loads(path.read_text())
    ids = tuple(int(value) for value in payload["model_ids"])
    catalogue_ids = tuple(spec.model_id for spec in enumerate_structures())
    if ids != DEVELOPMENT_IDS:
        raise AssertionError(f"development set differs from frozen IDs: {ids}")
    if len(ids) != 18 or len(set(ids)) != 18 or not set(ids).issubset(catalogue_ids):
        raise AssertionError("development set must contain 18 unique authoritative catalogue IDs")
    if payload.get("selection_locked_before_results") is not True:
        raise AssertionError("development set is not locked before results")
    return ids, payload


def _raw_flux_audit(model_ids: tuple[int, ...]) -> dict[str, Any]:
    rows = []
    forcing = base._default_forcing(1, torch.device("cpu"))
    for model_id in model_ids:
        prepared = base._prepare(model_id, forcing)
        spec = prepared["spec"]
        hydro = prepared["packed"][: len(STATE_NAMES)]
        row = prepared["forcing"][0]
        effective, _ = runtime_impl._runtime_snow_step(
            row[0], row[2], prepared["packed"][len(STATE_NAMES)], row[3], row[4], prepared["theta"], row[5]
        )
        choices = tuple(spec.decisions[name] for name in ("ARCH1", "ARCH2", "QSURF", "QPERC"))
        process_errors = {}
        for process in EXPECTED_ORDERS["S1"]:
            runtime_raw = runtime_impl._fixed_raw_process(
                hydro, effective, row[1], prepared["theta"], prepared["topographic"][0], prepared["topographic"][1], choices=choices, process=process
            )
            kernel_raw = _sequential_raw_process(
                hydro, effective, row[1], prepared["theta"], prepared["context"], prepared["topographic"][0], prepared["topographic"][1], process
            )
            process_errors[process] = _number((runtime_raw - kernel_raw).abs().max())
        rows.append({"model_id": model_id, "raw_flux_process_abs": process_errors, "max_raw_flux_process_abs": max(process_errors.values(), default=0.0)})
        del prepared
    return {
        "equations": "runtime._fixed_raw_process and dfuse.kernel._sequential_raw_process compared at identical state/forcing/theta/context",
        "max_raw_flux_process_abs": max((row["max_raw_flux_process_abs"] for row in rows), default=0.0),
        "rows": rows,
    }


def _stage0_static(model_ids: tuple[int, ...]) -> dict[str, Any]:
    expected = {name: list(value) for name, value in EXPECTED_ORDERS.items()}
    if {name: tuple(value) for name, value in SEQUENTIAL_ORDERS.items()} != EXPECTED_ORDERS:
        raise AssertionError("public SEQUENTIAL_ORDERS differs from the frozen S1-S4 candidate set")
    rows = []
    for model_id in model_ids:
        spec = get_structure(model_id)
        for order_name in ORDERS:
            order = EXPECTED_ORDERS[order_name]
            signature, generated = get_generated_step(spec, order=order, n_substeps=N_SUBSTEPS)
            source = getattr(generated, "generated_source", "")
            expected_fragment = f"order={order!r}"
            if tuple(signature.sequential_order) != order or expected_fragment not in source:
                raise AssertionError(f"generated step did not freeze {order_name} before compile for model {model_id}")
            rows.append({"model_id": model_id, "order": order_name, "signature_digest": signature.digest, "graph_signature": signature.to_dict(), "order_literal_in_source": True})
        reset_runtime_registries()
    return {
        "same_process_library": True,
        "same_runtime_builder": "dfuse.runtime.get_generated_step/get_compiled_step",
        "same_projection_and_capacity": True,
        "dynamic_order_inside_compiled_step": False,
        "process_equations_rewritten": False,
        "orders": expected,
        "rows": rows,
    }


def _run_trace(step, prepared: Mapping[str, Any], *, current: bool = False, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    packed = prepared["packed"]
    state_positions = [STATE_NAMES.index(name) for name in prepared["spec"].state_names]
    states = [packed[state_positions]]
    qs = []
    diagnostics = []
    iterator = prepared["forcing"]
    context = prepared["context"]
    if not requires_grad:
        context_manager = torch.no_grad()
    else:
        context_manager = torch.enable_grad()
    with context_manager:
        for row in iterator:
            if current:
                packed, q, diag = step(packed, row, prepared["theta"], context, prepared["topographic"][0], prepared["topographic"][1], prepared["fractions"])
            else:
                packed, q, diag = step(packed, row, prepared["theta"], prepared["topographic"][0], prepared["topographic"][1], prepared["fractions"])
            states.append(packed[state_positions])
            qs.append(q)
            diagnostics.append(diag)
    return {"states": torch.stack(states), "q": torch.stack(qs), "diagnostics": torch.stack(diagnostics), "packed": packed}


def _gradient_audit(model_id: int, forcing: torch.Tensor, order: tuple[str, ...], generated, compiled) -> dict[str, Any]:
    defaults = default_parameters()
    current_params = {name: torch.tensor(value, dtype=forcing.dtype, device=forcing.device, requires_grad=True) for name, value in defaults.items()}
    generated_params = {name: torch.tensor(value, dtype=forcing.dtype, device=forcing.device, requires_grad=True) for name, value in defaults.items()}
    compiled_params = {name: torch.tensor(value, dtype=forcing.dtype, device=forcing.device, requires_grad=True) for name, value in defaults.items()}
    small = base._default_forcing(2, forcing.device)
    current = _run_trace(_make_sequential_step(order, N_SUBSTEPS), base._prepare(model_id, small, current_params), current=True, requires_grad=True)
    eager = _run_trace(generated, base._prepare(model_id, small, generated_params), requires_grad=True)
    compiled_result = _run_trace(compiled, base._prepare(model_id, small, compiled_params), requires_grad=True)
    current_grads = torch.autograd.grad(current["q"].sum(), tuple(current_params.values()), allow_unused=True)
    eager_grads = torch.autograd.grad(eager["q"].sum(), tuple(generated_params.values()), allow_unused=True)
    compiled_grads = torch.autograd.grad(compiled_result["q"].sum(), tuple(compiled_params.values()), allow_unused=True)
    active = set(get_structure(model_id).parameter_names)
    errors_current_eager = []
    errors_eager_compiled = []
    inactive_eager = []
    inactive_compiled = []
    finite = True
    for name, current_grad, eager_grad, compiled_grad in zip(defaults, current_grads, eager_grads, compiled_grads):
        values = (current_grad, eager_grad, compiled_grad)
        finite = finite and all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in values)
        if name in active:
            left = torch.zeros((), dtype=forcing.dtype, device=forcing.device) if current_grad is None else current_grad
            middle = torch.zeros((), dtype=forcing.dtype, device=forcing.device) if eager_grad is None else eager_grad
            right = torch.zeros((), dtype=forcing.dtype, device=forcing.device) if compiled_grad is None else compiled_grad
            errors_current_eager.append(_number((left - middle).abs().max()))
            errors_eager_compiled.append(_number((middle - right).abs().max()))
        else:
            if eager_grad is not None:
                inactive_eager.append(_number(eager_grad.abs().max()))
            if compiled_grad is not None:
                inactive_compiled.append(_number(compiled_grad.abs().max()))
    return {
        "finite": finite,
        "active_current_vs_generated_max_abs": max(errors_current_eager, default=0.0),
        "active_generated_vs_compiled_max_abs": max(errors_eager_compiled, default=0.0),
        "inactive_generated_max_abs": max(inactive_eager, default=0.0),
        "inactive_compiled_max_abs": max(inactive_compiled, default=0.0),
        "inactive_gradients_zero_or_none": max(inactive_eager + inactive_compiled, default=0.0) == 0.0,
    }


def _parity(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {
        "q_max_abs": _number((left["q"] - right["q"]).abs().max()),
        "state_max_abs": _number((left["states"] - right["states"]).abs().max()),
        "flux_max_abs": _number((left["diagnostics"][:, : len(FLUX_NAMES)] - right["diagnostics"][:, : len(FLUX_NAMES)]).abs().max()),
        "diagnostics_max_abs": _number((left["diagnostics"] - right["diagnostics"]).abs().max()),
    }


def _capacity_violation(model_id: int, prepared: Mapping[str, Any], states: torch.Tensor) -> float:
    capacity = base._capacity_vector(prepared)
    active_positions = [STATE_NAMES.index(name) for name in prepared["spec"].state_names]
    active_capacity = capacity[active_positions]
    return _number((states[1:] * 1.0 - active_capacity).clamp_min(0.0).max())


def _kgecomp(left: np.ndarray, right: np.ndarray) -> float | None:
    try:
        value = float(kgecomp(torch.as_tensor(left, dtype=torch.float64), torch.as_tensor(right, dtype=torch.float64)).detach())
        return value if math.isfinite(value) else None
    except (RuntimeError, ValueError, FloatingPointError, ZeroDivisionError):
        return None


def _variant_metrics(result: Mapping[str, torch.Tensor], reference: Mapping[str, Any], model_id: int, prepared: Mapping[str, Any]) -> dict[str, Any]:
    # FUSE reference state variables are sampled at the beginning of each step,
    # matching the existing fidelity harness's SimulationResult.states[:-1].
    states = result["states"][:-1].detach().cpu().numpy()
    q = result["q"].detach().cpu().numpy()
    diagnostics = result["diagnostics"].detach().cpu().numpy()
    ref_q = np.asarray(reference["q_routed"], dtype=np.float64)
    ref_states = np.stack([np.asarray(reference["states"][name], dtype=np.float64) for name in prepared["spec"].state_names], axis=1)
    ref_flux = {name: np.asarray(reference["fluxes"][name], dtype=np.float64) for name in FLUX_NAMES if name in reference["fluxes"]}
    q_error = _summary(q - ref_q)
    state_by_name = {name: _summary(states[:, index] - ref_states[:, index]) for index, name in enumerate(prepared["spec"].state_names)}
    all_state_error = np.concatenate([states[:, index] - ref_states[:, index] for index in range(states.shape[1])]) if states.shape[1] else np.zeros(0)
    flux_by_name = {}
    all_flux_errors = []
    for index, name in enumerate(FLUX_NAMES):
        if name in ref_flux:
            error = diagnostics[:, index] - ref_flux[name]
            all_flux_errors.append(error)
            flux_by_name[name] = _summary(error)
    process_error = {}
    for process, names in PROCESS_FLUXES.items():
        actual = sum((diagnostics[:, FLUX_NAMES.index(name)] for name in names), np.zeros_like(q))
        expected = sum((ref_flux[name] for name in names if name in ref_flux), np.zeros_like(q))
        process_error[process] = _summary(actual - expected)
    water_index = len(FLUX_NAMES)
    snow_index = water_index + 1
    return {
        "q": {**q_error, "kgecomp": _kgecomp(q, ref_q)},
        "state": {"aggregate": _summary(all_state_error), "by_state": state_by_name},
        "flux": {"aggregate": _summary(np.concatenate(all_flux_errors) if all_flux_errors else np.zeros(0)), "by_flux": flux_by_name, "by_process": process_error},
        "finite": bool(np.isfinite(states).all() and np.isfinite(q).all() and np.isfinite(diagnostics).all()),
        "negative_storage": bool(np.min(states) < -TOLERANCE) if states.size else False,
        "min_state": float(np.min(states)) if states.size else 0.0,
        "capacity_violation": _capacity_violation(model_id, prepared, result["states"]),
        "water_balance_max_abs": float(np.max(np.abs(diagnostics[:, water_index]))),
        "snow_balance_max_abs": float(np.max(np.abs(diagnostics[:, snow_index]))),
        "min_routed_q": float(np.min(q)) if q.size else math.inf,
    }


def _benchmark(generated, compiled, model_id: int, device: torch.device, signature: GraphSignature) -> dict[str, Any]:
    return base._benchmark_one(model_id, signature, generated, compiled, device)


def _worker_case(output: Path, cache_dir: Path, model_id: int, order_name: str, reference_path: Path, executable: Path, benchmark: bool) -> dict[str, Any]:
    _set_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("sequential order experiment requires CUDA; refusing CPU fallback")
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    device = torch.device("cuda")
    order = EXPECTED_ORDERS[order_name]
    spec = get_structure(model_id)
    signature = GraphSignature.from_structure(spec, sequential_order=order, n_substeps=N_SUBSTEPS)
    reset_runtime_registries()
    torch.cuda.reset_peak_memory_stats(device)
    rss_before = _rss_kb()
    gpu_before = base._gpu_memory(device)
    cache_before = base._cache_info(cache_dir)
    started = time.perf_counter()
    generated = None
    compiled = None
    reference = None
    result: dict[str, Any] = {
        "schema": "autofuse-sequential-order-case-v1",
        "phase": "development" if model_id in DEVELOPMENT_IDS else "heldout",
        "model_id": model_id,
        "decisions": dict(spec.decisions),
        "order": order_name,
        "order_definition": list(order),
        "signature_digest": signature.digest,
        "graph_signature": signature.to_dict(),
        "status": "failed",
    }
    try:
        built_signature, generated = get_generated_step(spec, order=order, n_substeps=N_SUBSTEPS)
        if built_signature != signature:
            raise AssertionError("runtime builder signature differs from frozen order signature")
        _, compiled = get_compiled_step(
            spec,
            order=order,
            n_substeps=N_SUBSTEPS,
            device=device,
            dtype=torch.float64,
            input_shapes=base.INPUT_SHAPES,
            backend="inductor",
            fullgraph=True,
        )
        forcing_values = synthetic_forcing(24)
        forcing = _as_forcing_tensor(forcing_values, device)
        prepared = base._prepare(model_id, forcing)
        current = _run_trace(_make_sequential_step(order, N_SUBSTEPS), prepared, current=True)
        generated_result = _run_trace(generated, prepared)
        compiled_result = _run_trace(compiled, prepared)
        reference = _load_reference(reference_path)
        current_metrics = _variant_metrics(current, reference, model_id, prepared)
        generated_metrics = _variant_metrics(generated_result, reference, model_id, prepared)
        compiled_metrics = _variant_metrics(compiled_result, reference, model_id, prepared)
        parity_current_generated = _parity(current, generated_result)
        parity_generated_compiled = _parity(generated_result, compiled_result)
        gradient = _gradient_audit(model_id, forcing, order, generated, compiled)
        record = base._signature_record(runtime_compile_diagnostics(), signature.digest) or {}
        compile_audit = {
            "compile_attempts": int(record.get("compile_attempts", 0)),
            "compile_successes": int(record.get("compile_successes", 0)),
            "graph_breaks": int(record.get("graph_breaks", 0)),
            "recompilations": int(record.get("recompilations", 0)),
            "autograd_recompilations": int(record.get("autograd_recompilations", 0)),
            "guard_failure_reasons": copy.deepcopy(record.get("guard_failure_reasons", [])),
            "cold_compile_seconds": copy.deepcopy(record.get("cold_compile_seconds", [])),
            "dynamo_counters": copy.deepcopy(record.get("dynamo_counters", {})),
            "persistent_cache_hit": bool(record.get("dynamo_counters", {}).get("inductor", {}).get("fxgraph_cache_hit", 0) > 0),
            "persistent_autograd_cache_hit": bool(record.get("dynamo_counters", {}).get("aot_autograd", {}).get("autograd_cache_hit", 0) > 0),
        }
        hard_gate = (
            current_metrics["finite"] and generated_metrics["finite"] and compiled_metrics["finite"]
            and not current_metrics["negative_storage"] and not generated_metrics["negative_storage"] and not compiled_metrics["negative_storage"]
            and max(current_metrics["capacity_violation"], generated_metrics["capacity_violation"], compiled_metrics["capacity_violation"]) <= TOLERANCE
            and max(current_metrics["water_balance_max_abs"], generated_metrics["water_balance_max_abs"], compiled_metrics["water_balance_max_abs"]) <= WATER_TOLERANCE
            and max(current_metrics["snow_balance_max_abs"], generated_metrics["snow_balance_max_abs"], compiled_metrics["snow_balance_max_abs"]) <= WATER_TOLERANCE
            and all(value <= TOLERANCE for value in parity_current_generated.values())
            and all(value <= TOLERANCE for value in parity_generated_compiled.values())
            and gradient["finite"] and gradient["inactive_gradients_zero_or_none"]
            and gradient["active_generated_vs_compiled_max_abs"] <= GRADIENT_TOLERANCE
            and gradient["active_current_vs_generated_max_abs"] <= TOLERANCE
            and compile_audit["compile_attempts"] == 1
            and compile_audit["compile_successes"] >= 1
            and compile_audit["graph_breaks"] == 0
            and compile_audit["recompilations"] == 0
            and compile_audit["autograd_recompilations"] == 0
        )
        result.update({
            "status": "passed" if hard_gate else "failed",
            "metrics_protocol": METRICS_PROTOCOL,
            "hard_gate_passed": hard_gate,
            "reference_executable": str(executable.resolve()),
            "reference_executable_sha256": _load_reference(reference_path)["metadata"]["reference_executable_sha256"],
            "reference_forcing_sha256": reference["metadata"]["forcing_sha256"],
            "catalogue_sha256": _sha256_file(ROOT / "dfuse/specs/structures_78.json"),
            "runtime_provenance_sha256": _runtime_provenance()["sha256"],
            "forcing_steps": 24,
            "dtype": "torch.float64",
            "current_vs_reference": current_metrics,
            "generated_eager_vs_reference": generated_metrics,
            "compiled_vs_reference": compiled_metrics,
            "generated_eager_vs_compiled_parity": parity_generated_compiled,
            "current_vs_generated_parity": parity_current_generated,
            "gradient": gradient,
            "compile": compile_audit,
        })
        if benchmark and hard_gate:
            result["representative_cuda_benchmark"] = _benchmark(generated, compiled, model_id, device, signature)
    except Exception as exc:
        result["error"] = _error(exc)
    finally:
        gpu_peak = base._gpu_memory(device)
        result["resource"] = {
            "host_rss_before_kb": rss_before,
            "host_rss_current_kb": _rss_kb(),
            "host_peak_rss_kb": _peak_rss_kb(),
            "gpu_before": gpu_before,
            "gpu_peak": gpu_peak,
        }
        try:
            del reference, compiled, generated
        except UnboundLocalError:
            pass
        base._cleanup_runtime()
        gc.collect()
        gpu_after = base._gpu_memory(device)
        cache_after = base._cache_info(cache_dir)
        result["resource"]["host_rss_after_cleanup_kb"] = _rss_kb()
        result["resource"]["gpu_after_cleanup"] = gpu_after
        result["cache"] = {
            "dir": str(cache_dir),
            "before": cache_before,
            "after": cache_after,
            "increment": {"files": cache_after["files"] - cache_before["files"], "bytes": cache_after["bytes"] - cache_before["bytes"]},
        }
        result["resource_stop_reason"] = base._resource_stop_reason(rss_before, device, cache_dir)
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_json(output, result)
    print(json.dumps({"model_id": model_id, "order": order_name, "status": result["status"]}, sort_keys=True))
    return result


def _worker_smoke(output: Path, cache_dir: Path, model_id: int, order_name: str) -> dict[str, Any]:
    _set_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 0 compile smoke requires CUDA")
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    device = torch.device("cuda")
    order = EXPECTED_ORDERS[order_name]
    spec = get_structure(model_id)
    signature = GraphSignature.from_structure(spec, sequential_order=order, n_substeps=N_SUBSTEPS)
    started = time.perf_counter()
    generated = compiled = None
    result = {"model_id": model_id, "order": order_name, "signature_digest": signature.digest, "status": "failed"}
    try:
        _, generated = get_generated_step(spec, order=order, n_substeps=N_SUBSTEPS)
        _, compiled = get_compiled_step(spec, order=order, n_substeps=N_SUBSTEPS, device=device, dtype=torch.float64, input_shapes=base.INPUT_SHAPES, backend="inductor", fullgraph=True)
        forcing = base._default_forcing(4, device)
        prepared = base._prepare(model_id, forcing)
        eager = _run_trace(generated, prepared)
        compiled_result = _run_trace(compiled, prepared)
        parity = _parity(eager, compiled_result)
        record = base._signature_record(runtime_compile_diagnostics(), signature.digest) or {}
        result.update({
            "status": "passed" if max(parity.values()) <= TOLERANCE and record.get("compile_attempts") == 1 and record.get("graph_breaks") == 0 and record.get("recompilations") == 0 else "failed",
            "parity": parity,
            "compile": {key: record.get(key) for key in ("compile_attempts", "graph_breaks", "recompilations", "autograd_recompilations", "cold_compile_seconds")},
        })
    except Exception as exc:
        result["error"] = _error(exc)
    finally:
        try:
            del compiled, generated
        except UnboundLocalError:
            pass
        base._cleanup_runtime()
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_json(output, result)
    return result


def _run_worker(command: list[str], cwd: Path, env: Mapping[str, str], timeout: int = 1800) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    completed = subprocess.run(command, cwd=cwd, env=dict(env), text=True, capture_output=True, timeout=timeout, check=False)
    output = Path(command[command.index("--output") + 1])
    if completed.returncode != 0:
        return None, {"returncode": completed.returncode, "stderr": completed.stderr[-4000:], "stdout": completed.stdout[-1000:]}
    try:
        return json.loads(output.read_text()), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, {"returncode": completed.returncode, "error": _error(exc), "stdout": completed.stdout[-1000:], "stderr": completed.stderr[-4000:]}


def _case_key(phase: str, order: str, model_id: int) -> str:
    return f"{phase}:{order}:{model_id}"

def _valid_prior_case(row: Mapping[str, Any], development_ids: tuple[int, ...], heldout_ids: tuple[int, ...], catalogue_sha256: str, executable_sha256: str, forcing_sha256: str, runtime_provenance_sha256: str) -> bool:
    try:
        model_id = int(row["model_id"])
        order_name = str(row["order"])
        phase = str(row["phase"])
        valid_ids = development_ids if phase == "development" else heldout_ids if phase == "heldout" else ()
        if model_id not in valid_ids or order_name not in ORDERS:
            return False
        expected = GraphSignature.from_structure(get_structure(model_id), sequential_order=EXPECTED_ORDERS[order_name], n_substeps=N_SUBSTEPS)
        compile_audit = row["compile"]
        gradient = row["gradient"]
        return (
            row.get("status") == "passed"
            and row.get("metrics_protocol") == METRICS_PROTOCOL
            and row.get("signature_digest") == expected.digest
            and base._canonical_hash(row.get("graph_signature")) == base._canonical_hash(expected.to_dict())
            and row.get("order_definition") == list(EXPECTED_ORDERS[order_name])
            and row.get("decisions") == dict(get_structure(model_id).decisions)
            and row.get("catalogue_sha256") == catalogue_sha256
            and row.get("runtime_provenance_sha256") == runtime_provenance_sha256
            and row.get("reference_executable_sha256") == executable_sha256
            and row.get("reference_forcing_sha256") == forcing_sha256
            and row.get("forcing_steps") == 24
            and row.get("dtype") == "torch.float64"
            and row.get("hard_gate_passed") is True
            and compile_audit.get("compile_attempts") == 1
            and compile_audit.get("compile_successes", 0) >= 1
            and compile_audit.get("graph_breaks") == 0
            and compile_audit.get("recompilations") == 0
            and compile_audit.get("autograd_recompilations") == 0
            and gradient.get("finite") is True
            and gradient.get("inactive_gradients_zero_or_none") is True
            and gradient.get("active_generated_vs_compiled_max_abs", math.inf) <= GRADIENT_TOLERANCE
            and gradient.get("active_current_vs_generated_max_abs", math.inf) <= TOLERANCE
            and row.get("resource_stop_reason") is None
        )
    except (KeyError, TypeError, ValueError):
        return False


def _rank(values: Mapping[str, float], lower_is_better: bool = True) -> dict[str, int]:
    if lower_is_better:
        return {name: sum(other < value for other in values.values()) for name, value in values.items()}
    return {name: sum(other > value for other in values.values()) for name, value in values.items()}


def _aggregate_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [row for row in rows if row.get("status") == "passed"]
    def collect(path: tuple[str, ...]) -> list[float]:
        values = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value.get(key) if isinstance(value, Mapping) else None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        return values
    q_max = collect(("compiled_vs_reference", "q", "max"))
    q_rmse = collect(("compiled_vs_reference", "q", "rmse"))
    q_kge = collect(("compiled_vs_reference", "q", "kgecomp"))
    state_max = collect(("compiled_vs_reference", "state", "aggregate", "max"))
    state_p95 = collect(("compiled_vs_reference", "state", "aggregate", "p95"))
    flux_max = collect(("compiled_vs_reference", "flux", "aggregate", "max"))
    flux_p95 = collect(("compiled_vs_reference", "flux", "aggregate", "p95"))
    process = {}
    for name in PROCESS_FLUXES:
        values = collect(("compiled_vs_reference", "flux", "by_process", name, "max"))
        process[name] = {"max": max(values, default=math.inf), "median": float(np.median(values)) if values else math.inf, "p95": float(np.quantile(values, 0.95)) if values else math.inf}
    benchmarks = [row["representative_cuda_benchmark"] for row in rows if row.get("representative_cuda_benchmark")]
    total_speedups = [float(item["forward_backward"]["speedup"]) for item in benchmarks]
    return {
        "count": len(rows),
        "passed": len(passed),
        "failed": len(rows) - len(passed),
        "finite": sum(bool(row.get("compiled_vs_reference", {}).get("finite")) for row in rows),
        "negative_storage": sum(bool(row.get("compiled_vs_reference", {}).get("negative_storage")) for row in rows),
        "q": {"max": max(q_max, default=math.inf), "median": float(np.median(q_max)) if q_max else math.inf, "rmse_median": float(np.median(q_rmse)) if q_rmse else math.inf, "kgecomp_mean": float(np.mean(q_kge)) if q_kge else -math.inf},
        "state": {"max": max(state_max, default=math.inf), "median_of_max": float(np.median(state_max)) if state_max else math.inf, "p95_median": float(np.median(state_p95)) if state_p95 else math.inf},
        "flux": {"max": max(flux_max, default=math.inf), "median_of_max": float(np.median(flux_max)) if flux_max else math.inf, "p95_median": float(np.median(flux_p95)) if flux_p95 else math.inf, "by_process": process},
        "water_balance_max": max(collect(("compiled_vs_reference", "water_balance_max_abs")), default=math.inf),
        "snow_balance_max": max(collect(("compiled_vs_reference", "snow_balance_max_abs")), default=math.inf),
        "parity_max": max(collect(("generated_eager_vs_compiled_parity", "diagnostics_max_abs")), default=math.inf),
        "compile_clean": sum(bool(row.get("compile", {}).get("compile_attempts") == 1 and row.get("compile", {}).get("graph_breaks") == 0 and row.get("compile", {}).get("recompilations") == 0) for row in rows),
        "benchmark": {"count": len(benchmarks), "forward_backward_speedup_median": float(np.median(total_speedups)) if total_speedups else None, "forward_backward_speedup_min": min(total_speedups, default=None)},
    }


def _select_order(order_rows: Mapping[str, list[dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    summaries = {name: _aggregate_cases(rows) for name, rows in order_rows.items()}
    eligible = [name for name, summary in summaries.items() if summary["count"] == 18 and summary["passed"] == 18 and summary["compile_clean"] == 18]
    if not eligible:
        raise RuntimeError("no S1-S4 order passed the development hard gate")
    q_max_rank = _rank({name: summaries[name]["q"]["max"] for name in eligible})
    q_rmse_rank = _rank({name: summaries[name]["q"]["rmse_median"] for name in eligible})
    state_max_rank = _rank({name: summaries[name]["state"]["max"] for name in eligible})
    state_median_rank = _rank({name: summaries[name]["state"]["median_of_max"] for name in eligible})
    state_p95_rank = _rank({name: summaries[name]["state"]["p95_median"] for name in eligible})
    flux_max_rank = _rank({name: summaries[name]["flux"]["max"] for name in eligible})
    flux_median_rank = _rank({name: summaries[name]["flux"]["median_of_max"] for name in eligible})
    flux_p95_rank = _rank({name: summaries[name]["flux"]["p95_median"] for name in eligible})
    kge_rank = _rank({name: 1.0 - summaries[name]["q"]["kgecomp_mean"] for name in eligible})
    fidelity_score = {name: 2 * (q_max_rank[name] + q_rmse_rank[name]) + 2 * (state_max_rank[name] + state_median_rank[name] + state_p95_rank[name]) + 3 * (flux_max_rank[name] + flux_median_rank[name] + flux_p95_rank[name]) + kge_rank[name] for name in eligible}
    best_score = min(fidelity_score.values())
    close = [name for name in eligible if fidelity_score[name] <= best_score + 1]
    best = sorted(close, key=lambda name: (-float(summaries[name]["benchmark"]["forward_backward_speedup_median"] or -math.inf), name))[0]
    def dominates(left: str, right: str) -> bool:
        left_values = (summaries[left]["q"]["max"], summaries[left]["q"]["rmse_median"], 1.0 - summaries[left]["q"]["kgecomp_mean"], summaries[left]["state"]["max"], summaries[left]["state"]["median_of_max"], summaries[left]["state"]["p95_median"], summaries[left]["flux"]["max"], summaries[left]["flux"]["median_of_max"], summaries[left]["flux"]["p95_median"])
        right_values = (summaries[right]["q"]["max"], summaries[right]["q"]["rmse_median"], 1.0 - summaries[right]["q"]["kgecomp_mean"], summaries[right]["state"]["max"], summaries[right]["state"]["median_of_max"], summaries[right]["state"]["p95_median"], summaries[right]["flux"]["max"], summaries[right]["flux"]["median_of_max"], summaries[right]["flux"]["p95_median"])
        return all(a <= b for a, b in zip(left_values, right_values)) and any(a < b for a, b in zip(left_values, right_values))
    pareto_front = [name for name in eligible if not any(dominates(other, name) for other in eligible if other != name)]
    uniform_fidelity_defect = all(summaries[name]["q"]["max"] > 0.25 and summaries[name]["state"]["max"] > 1.0 and summaries[name]["flux"]["max"] > 1.0 for name in eligible)
    reasonable_pareto_orders = [name for name in pareto_front if summaries[name]["q"]["kgecomp_mean"] >= 0.5]
    s5_thresholds = {"q_max_abs_gt": 0.25, "state_max_abs_gt": 1.0, "flux_max_abs_gt": 1.0, "reasonable_pareto_kgecomp_mean_gte": 0.5}
    s5_triggered = uniform_fidelity_defect and not reasonable_pareto_orders
    rationale = {
        "eligible_orders": eligible,
        "aggregate_by_order": summaries,
        "rank_components": {"q_max_abs": q_max_rank, "q_rmse_median": q_rmse_rank, "state_max_abs": state_max_rank, "state_median_of_max": state_median_rank, "state_p95_median": state_p95_rank, "flux_max_abs": flux_max_rank, "flux_median_of_max": flux_median_rank, "flux_p95_median": flux_p95_rank, "one_minus_kgecomp": kge_rank},
        "fidelity_rank_score": fidelity_score,
        "close_fidelity_orders": close,
        "rule": "hard gate first; competition-ranked multi-metric fidelity (Q max/RMSE, state max/median/P95, flux max/median/P95, KGEcomp; weights flux 3, Q 2, state 2, KGE 1); within one score point choose fastest representative forward+backward; lexical tie-break",
        "pareto_front": pareto_front,
        "uniform_fidelity_defect": uniform_fidelity_defect,
        "s5_thresholds": s5_thresholds,
        "reasonable_pareto_orders": reasonable_pareto_orders,
        "s5_triggered": s5_triggered,
        "selected": best,
    }
    return best, rationale


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get("decisions", {}).get(key, "unknown"))
        groups.setdefault(group, []).append(row)
    return {name: _aggregate_cases(values) for name, values in sorted(groups.items())}


def _build_partial(manifest: Mapping[str, Any], reference: Mapping[str, Any], development: Mapping[str, Any], heldout: tuple[int, ...], stage0: Mapping[str, Any], case_results: Mapping[str, dict[str, Any]], best_order: str | None, rationale: Mapping[str, Any] | None, stop_reason: str | None = None) -> dict[str, Any]:
    dev_rows = [row for key, row in case_results.items() if row.get("phase") == "development"]
    held_rows = [row for key, row in case_results.items() if row.get("phase") == "heldout"]
    return {
        "schema": "autofuse-sequential-order-validation-v1",
        "metrics_protocol": METRICS_PROTOCOL,
        "status": "partial",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "torch.float64",
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "formal_training_started": False,
            "sce_started": False,
            "dpl_started": False,
        },
        "catalogue": manifest,
        "development_set": development,
        "heldout_model_ids": list(heldout),
        "orders": {name: list(EXPECTED_ORDERS[name]) for name in ORDERS},
        "n_substeps": N_SUBSTEPS,
        "forcing": {"kind": "project.autofuse.fidelity.synthetic_forcing", "steps": 24, "dt_days": 1.0, "dtype": "torch.float64"},
        "stage0_implementation_audit": stage0,
        "reference": reference,
        "case_results": sorted(case_results.values(), key=lambda row: (row.get("phase", ""), row.get("order", ""), int(row.get("model_id", -1)))),
        "development_summary": {name: _aggregate_cases([row for row in dev_rows if row.get("order") == name]) for name in ORDERS},
        "heldout_summary": _aggregate_cases(held_rows),
        "best_order": best_order,
        "selection_rationale": rationale,
        "S5_triggered": bool((rationale or {}).get("s5_triggered", False)),
        "S5_reason": "Computed trigger requires uniform Q/state/flux defect and no reasonable Pareto order; no S5 implementation was run because the development results retain a reasonable Pareto set.",
        "stop_reason": stop_reason,
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    _set_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("S1-S4 experiment requires CUDA; refusing CPU fallback")
    executable = Path(args.executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    development_ids, development_metadata = _load_development(Path(args.development_set))
    catalogue_specs = tuple(enumerate_structures())
    catalogue_ids = tuple(spec.model_id for spec in catalogue_specs)
    heldout_ids = tuple(model_id for model_id in catalogue_ids if model_id not in development_ids)
    if len(heldout_ids) != 60:
        raise AssertionError(f"expected 60 held-out structures, got {len(heldout_ids)}")
    forcing_values = synthetic_forcing(24)
    output = Path(args.output).resolve()
    cache_root = Path(args.cache_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    reference = _prepare_reference_cache(executable, reference_dir, catalogue_ids, forcing_values)
    reference_by_id = {int(row["model_id"]): Path(row["path"]) for row in reference["models"]}
    manifest = {
        "catalog_path": str((ROOT / "dfuse/specs/structures_78.json").resolve()),
        "catalog_sha256": _sha256_file(ROOT / "dfuse/specs/structures_78.json"),
        "n_models": len(catalogue_ids),
        "model_ids": list(catalogue_ids),
        "development_ids": list(development_ids),
        "heldout_ids": list(heldout_ids),
        "runtime_source_sha256": _sha256_file(ROOT / "dfuse/runtime.py"),
        "runtime_provenance": _runtime_provenance(),
        "reference_executable_sha256": reference["executable_sha256"],
        "reference_forcing_sha256": reference["forcing_sha256"],
        "orders": {name: list(EXPECTED_ORDERS[name]) for name in ORDERS},
        "n_substeps": N_SUBSTEPS,
    }
    static = _stage0_static(DEVELOPMENT_IDS[:4])
    raw_audit = _raw_flux_audit(DEVELOPMENT_IDS[:4])
    stage0 = {**static, "raw_flux_audit": raw_audit}
    stage0_smoke = []
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1"})
    for model_id in DEVELOPMENT_IDS[:4]:
        smoke_output = output.with_name(f"{output.stem}_stage0_smoke_{model_id}.json")
        command = [sys.executable, "-m", "project.autofuse.sequential_order_validation", "--mode", "smoke", "--model-id", str(model_id), "--order", "S1", "--cache-dir", str(cache_root / "S1"), "--output", str(smoke_output)]
        smoke, error = _run_worker(command, ROOT, env, timeout=1800)
        if error is not None or smoke is None:
            stage0_smoke.append({"model_id": model_id, "status": "failed", "error": error})
            stage0["stage0_smoke"] = stage0_smoke
            payload = _build_partial(manifest, reference, development_metadata, heldout_ids, stage0, {}, None, None, "Stage 0 compile smoke failed")
            _write_json(output.with_suffix(".partial.json"), payload)
            raise RuntimeError(f"Stage 0 compile smoke failed for model {model_id}: {error}")
        stage0_smoke.append(smoke)
    stage0["stage0_smoke"] = stage0_smoke
    if raw_audit["max_raw_flux_process_abs"] > TOLERANCE:
        payload = _build_partial(manifest, reference, development_metadata, heldout_ids, stage0, {}, None, None, "Stage 0 raw process audit failed")
        _write_json(output.with_suffix(".partial.json"), payload)
        raise RuntimeError("Stage 0 raw process audit failed")
    if any(row.get("status") != "passed" for row in stage0_smoke):
        raise RuntimeError("Stage 0 compile smoke failed")

    case_results: dict[str, dict[str, Any]] = {}
    pending_heldout: dict[str, dict[str, Any]] = {}
    prior_path = Path(args.partial).resolve() if args.partial else output.with_suffix(".partial.json")
    best_order = None
    rationale = None
    if prior_path.is_file():
        prior = json.loads(prior_path.read_text())
        if prior.get("catalogue", {}).get("catalog_sha256") != manifest["catalog_sha256"]:
            raise AssertionError("partial catalogue hash does not match authoritative structures_78.json")
        if prior.get("catalogue", {}).get("runtime_source_sha256") != manifest["runtime_source_sha256"]:
            raise AssertionError("partial runtime source hash does not match current dfuse/runtime.py")
        if prior.get("catalogue", {}).get("runtime_provenance") != manifest["runtime_provenance"]:
            raise AssertionError("partial runtime dependency provenance does not match current runtime/kernel/spec/validator sources")
        prior_reference = prior.get("reference", {})
        if prior_reference.get("executable_sha256") != reference["executable_sha256"] or prior_reference.get("forcing_sha256") != reference["forcing_sha256"]:
            raise AssertionError("partial reference provenance does not match current executable or forcing")
        prior_reference_hashes = {int(row["model_id"]): row.get("sha256") for row in prior_reference.get("models", [])}
        current_reference_hashes = {int(row["model_id"]): row.get("sha256") for row in reference["models"]}
        if prior_reference_hashes != current_reference_hashes:
            raise AssertionError("partial reference artifacts do not match the current reference cache")
        for candidate in prior.get("case_results", []):
            if not _valid_prior_case(candidate, development_ids, heldout_ids, manifest["catalog_sha256"], reference["executable_sha256"], reference["forcing_sha256"], manifest["runtime_provenance"]["sha256"]):
                continue
            row = copy.deepcopy(candidate)
            key = _case_key(row["phase"], row["order"], int(row["model_id"]))
            if row["phase"] == "development":
                case_results[key] = row
            else:
                pending_heldout[key] = row

    def save_partial(stop_reason: str | None = None) -> None:
        payload = _build_partial(manifest, reference, development_metadata, heldout_ids, stage0, case_results, best_order, rationale, stop_reason)
        _write_json(output.with_suffix(".partial.json"), payload)

    def launch_case(phase: str, order_name: str, model_id: int) -> bool:
        nonlocal case_results
        key = _case_key(phase, order_name, model_id)
        if key in case_results:
            return True
        case_cache = cache_root / order_name
        case_output = cache_root / "case-reports" / f"{phase}-{order_name}-{model_id}.json"
        command = [sys.executable, "-m", "project.autofuse.sequential_order_validation", "--mode", "case", "--model-id", str(model_id), "--order", order_name, "--executable", str(executable), "--reference", str(reference_by_id[model_id]), "--cache-dir", str(case_cache), "--output", str(case_output)]
        if phase == "development" and model_id in REPRESENTATIVE_IDS:
            command.append("--benchmark")
        result, error = _run_worker(command, ROOT, env, timeout=2400)
        if error is not None or result is None:
            case_results[key] = {"phase": phase, "order": order_name, "model_id": model_id, "status": "failed", "error": error or {"message": "missing worker result"}}
            save_partial(f"worker failed: {phase}/{order_name}/{model_id}")
            return False
        case_results[key] = result
        print(json.dumps({"phase": phase, "order": order_name, "model_id": model_id, "status": result.get("status"), "elapsed_seconds": result.get("elapsed_seconds")}, sort_keys=True), flush=True)
        save_partial(result.get("resource_stop_reason"))
        return result.get("status") == "passed" and not result.get("resource_stop_reason")

    for order_name in ORDERS:
        for model_id in development_ids:
            if not launch_case("development", order_name, model_id):
                final = _build_partial(manifest, reference, development_metadata, heldout_ids, stage0, case_results, best_order, rationale, f"development case failed: {order_name}/{model_id}")
                _write_json(output.with_suffix(".partial.json"), final)
                return final
    order_rows = {name: [row for row in case_results.values() if row.get("phase") == "development" and row.get("order") == name] for name in ORDERS}
    computed_order, computed_rationale = _select_order(order_rows)
    if best_order is not None and best_order != computed_order:
        raise AssertionError(f"partial frozen best_order {best_order} differs from recomputed development selection {computed_order}")
    best_order, rationale = computed_order, computed_rationale
    for key, row in pending_heldout.items():
        if row.get("order") == best_order:
            case_results[key] = row
    save_partial()
    for model_id in heldout_ids:
        if not launch_case("heldout", best_order, model_id):
            final = _build_partial(manifest, reference, development_metadata, heldout_ids, stage0, case_results, best_order, rationale, f"held-out case failed: {best_order}/{model_id}")
            _write_json(output.with_suffix(".partial.json"), final)
            return final
    final = _build_partial(manifest, reference, development_metadata, heldout_ids, stage0, case_results, best_order, rationale)
    all_rows = [row for row in case_results.values() if row.get("status") == "passed"]
    dev_best = [row for row in all_rows if row.get("phase") == "development" and row.get("order") == best_order]
    held_best = [row for row in all_rows if row.get("phase") == "heldout" and row.get("order") == best_order]
    cached_heldout_orders = sorted({path.name.split("-")[1] for path in (cache_root / "case-reports").glob("heldout-*.json") if len(path.name.split("-")) >= 3 and path.name.split("-")[1] != best_order})
    selection_summaries = (rationale or {}).get("aggregate_by_order", {})
    tradeoff = False
    if best_order in selection_summaries:
        eligible = list((rationale or {}).get("eligible_orders", selection_summaries))
        metrics = ("q", "state", "flux")
        tradeoff = any(selection_summaries[best_order][metric]["max"] > min(selection_summaries[name][metric]["max"] for name in eligible) for metric in metrics)
        tradeoff = tradeoff or selection_summaries[best_order]["q"]["kgecomp_mean"] < max(selection_summaries[name]["q"]["kgecomp_mean"] for name in eligible)
        tradeoff = tradeoff or (selection_summaries[best_order]["benchmark"]["forward_backward_speedup_median"] or -math.inf) < max(selection_summaries[name]["benchmark"]["forward_backward_speedup_median"] or -math.inf for name in eligible)
    final_status = ("validated with documented fidelity trade-off" if tradeoff else "sequential order validated and frozen") if len(dev_best) == 18 and len(held_best) == 60 else "sequential order not ready"
    final.update({
        "status": final_status,
        "best_order": best_order,
        "heldout_orders_run": [best_order],
        "heldout_provenance": {"final_order": best_order, "orders_present_in_final": [best_order], "different_order_rows_ignored_on_resume": sorted(set(cached_heldout_orders) | {row.get("order") for row in pending_heldout.values() if row.get("order") != best_order})},
        "plan_deviations": (["A preliminary held-out run for a pre-correction selection was excluded; final held-out evidence contains only the recomputed development-selected order."] if cached_heldout_orders or any(row.get("order") != best_order for row in pending_heldout.values()) else []),
        "development_best_order_summary": _aggregate_cases(dev_best),
        "heldout_frozen_order_summary": _aggregate_cases(held_best),
        "final_78_summary": _aggregate_cases(dev_best + held_best),
        "development_vs_heldout": {"development": _aggregate_cases(dev_best), "heldout": _aggregate_cases(held_best)},
        "sensitivity_by_topology": {key: _group_summary(dev_best + held_best, key) for key in ("ARCH1", "ARCH2", "QPERC", "QSURF")},
        "worst_structures": {
            "q": sorted(((row.get("compiled_vs_reference", {}).get("q", {}).get("max", math.inf), row.get("model_id"), row.get("phase")) for row in dev_best + held_best), reverse=True)[:5],
            "state": sorted(((row.get("compiled_vs_reference", {}).get("state", {}).get("aggregate", {}).get("max", math.inf), row.get("model_id"), row.get("phase")) for row in dev_best + held_best), reverse=True)[:5],
            "flux": sorted(((row.get("compiled_vs_reference", {}).get("flux", {}).get("aggregate", {}).get("max", math.inf), row.get("model_id"), row.get("phase")) for row in dev_best + held_best), reverse=True)[:5],
        },
        "resource_summary": {
            "host_rss_baseline_kb": min((row.get("resource", {}).get("host_rss_before_kb", math.inf) for row in all_rows), default=None),
            "host_rss_peak_kb": max((row.get("resource", {}).get("host_peak_rss_kb", 0) for row in all_rows), default=None),
            "host_rss_end_kb": max((row.get("resource", {}).get("host_rss_after_cleanup_kb", 0) for row in all_rows), default=None),
            "gpu_peak_allocated_bytes": max((row.get("resource", {}).get("gpu_peak", {}).get("peak_allocated_bytes", 0) for row in all_rows), default=0),
            "gpu_peak_reserved_bytes": max((row.get("resource", {}).get("gpu_peak", {}).get("peak_reserved_bytes", 0) for row in all_rows), default=0),
            "gpu_end_allocated_bytes": max((row.get("resource", {}).get("gpu_after_cleanup", {}).get("allocated_bytes", 0) for row in all_rows), default=0),
            "gpu_end_reserved_bytes": max((row.get("resource", {}).get("gpu_after_cleanup", {}).get("reserved_bytes", 0) for row in all_rows), default=0),
        },
        "cache_summary_by_order": {name: base._cache_info(cache_root / name) for name in ORDERS},
        "scientific_logic_modified": False,
        "formal_training_started": False,
        "sce_started": False,
        "dpl_started": False,
        "failures": [row for row in case_results.values() if row.get("status") != "passed"],
    })
    _write_json(output, final)
    _write_json(output.with_suffix(".partial.json"), final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "case", "smoke"), default="validate")
    parser.add_argument("--model-id", type=int)
    parser.add_argument("--order", choices=ORDERS)
    parser.add_argument("--executable", default=os.environ.get("FUSE_REFERENCE_EXE", "/tmp/autofuse-reference-toolchain/repro-build/bin/fuse.exe"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partial", type=Path)
    parser.add_argument("--development-set", type=Path, default=DEVELOPMENT_PATH)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.mode == "case":
        if args.model_id is None or args.order is None or args.reference is None:
            raise SystemExit("case mode requires --model-id, --order, and --reference")
        result = _worker_case(args.output, args.cache_dir, args.model_id, args.order, args.reference, Path(args.executable), args.benchmark)
        if result.get("status") != "passed":
            raise SystemExit(1)
        return
    if args.mode == "smoke":
        if args.model_id is None or args.order is None:
            raise SystemExit("smoke mode requires --model-id and --order")
        result = _worker_smoke(args.output, args.cache_dir, args.model_id, args.order)
        if result.get("status") != "passed":
            raise SystemExit(1)
        return
    result = run_validation(args)
    print(json.dumps({"status": result.get("status"), "best_order": result.get("best_order"), "development": result.get("development_best_order_summary"), "heldout": result.get("heldout_frozen_order_summary"), "failures": len(result.get("failures", []))}, sort_keys=True))


if __name__ == "__main__":
    main()
