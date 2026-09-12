"""Full 78-structure GPU-first runtime-step validation.

This harness keeps the existing four-model validator unchanged.  It performs a
metadata-only catalog pass first, then compiles one canonical signature at a
time and releases the generated/compiled references before continuing.  It
only exercises the fixed S1 explicit sequential Euler macro-step; it never
starts training, SCE, dPL, or an implicit solver.
"""

from __future__ import annotations

import argparse
import gc
import copy
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from dfuse import (
    GraphSignature,
    enumerate_structures,
    get_structure,
    reset_runtime_registries,
    runtime_compile_diagnostics,
    validate_catalog,
)
from dfuse.kernel import (
    FLUX_NAMES,
    PARAMETER_NAMES,
    STATE_NAMES,
    SEQUENTIAL_ORDERS,
    _make_sequential_step,
    _theta,
)
from dfuse.spec import default_parameters
from dfuse.runtime import get_compiled_step, get_generated_step
from project.autofuse.runtime_validation import (
    INPUT_SHAPES,
    N_SUBSTEPS,
    _compiled_gradient_probe,
    _default_forcing,
    _gradient_probe,
    _measure,
    _measure_backward,
    _prepare,
    _run,
    _series,
    _sync,
)

ORDER_NAME = "S1"
ORDER = SEQUENTIAL_ORDERS[ORDER_NAME]
TOLERANCE = 1.0e-12
DEFAULT_CACHE_DIR = Path(__file__).with_name(".cache") / "runtime-validation-78"
DEFAULT_OUTPUT = Path(__file__).with_name("docs") / "runtime_step_validation_78.json"
MANIFEST_OUTPUT = Path(__file__).with_name("docs") / "runtime_step_validation_78_manifest.json"
REPRESENTATIVE_MODELS = (2, 108, 178, 210, 212, 214, 164, 188, 166, 190)
CROSS_PROCESS_MODELS = (2, 108, 178, 210, 212, 214)

# These are safety stops, not performance targets.  They keep a compiler leak
# from exhausting the host or device while preserving all completed records.
RSS_STOP_KB = 3_500_000
RSS_GROWTH_STOP_KB = 2_750_000
DISK_FREE_STOP_BYTES = 2 * 1024**3


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


def _peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _gpu_memory(device: torch.device) -> dict[str, int]:
    _sync(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _cache_info(path: Path) -> dict[str, int]:
    files = 0
    size = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
                size += item.stat().st_size
    return {"files": files, "bytes": size}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _error_text(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:1000]}


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    value = float((left.detach() - right.detach()).abs().max().cpu())
    return value if math.isfinite(value) else math.inf


def _finite(*values: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def _capacity_vector(prepared: Mapping[str, Any]) -> torch.Tensor:
    theta = prepared["theta"]
    fracten = _theta(theta, "FRACTEN")
    maxwatr1 = _theta(theta, "MAXWATR_1")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    values = torch.stack(
        (
            _theta(theta, "FRCHZNE") * fracten * maxwatr1,
            (1.0 - _theta(theta, "FRCHZNE")) * fracten * maxwatr1,
            fracten * maxwatr1,
            (1.0 - fracten) * maxwatr1,
            maxwatr1,
            fracten * maxwatr2,
            _theta(theta, "FPRIMQB") * (1.0 - fracten) * maxwatr2,
            (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - fracten) * maxwatr2,
            maxwatr2,
        )
    )
    # The sequential projection deliberately leaves unlimfrc/unlimpow's
    # lower reservoir un-capped; fixedsiz_2 has the explicit upper cap.
    if prepared["spec"].decisions["ARCH2"] != "fixedsiz_2":
        values = torch.cat((values[:8], torch.full_like(values[8:], math.inf)))
    return values


def _run_checked(step, prepared: Mapping[str, Any], *, current: bool = False) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    packed = prepared["packed"]
    q_values = []
    diagnostics = []
    max_negative = 0.0
    max_capacity = 0.0
    min_q = math.inf
    finite = True
    capacity = _capacity_vector(prepared)
    for row in prepared["forcing"]:
        if current:
            packed, q, diag = step(
                packed,
                row,
                prepared["theta"],
                prepared["context"],
                prepared["topographic"][0],
                prepared["topographic"][1],
                prepared["fractions"],
            )
        else:
            packed, q, diag = step(
                packed,
                row,
                prepared["theta"],
                prepared["topographic"][0],
                prepared["topographic"][1],
                prepared["fractions"],
            )
        q_values.append(q)
        diagnostics.append(diag)
        negative = float((-packed).clamp_min(0.0).max().detach().cpu())
        capacity_violation = float((packed[: len(STATE_NAMES)] - capacity).clamp_min(0.0).max().detach().cpu())
        min_q = min(min_q, float(q.detach().min().cpu()))
        max_negative = max(max_negative, negative)
        max_capacity = max(max_capacity, capacity_violation)
        finite = finite and _finite(packed, q, diag)
    result = {"packed": packed, "q": torch.stack(q_values), "diagnostics": torch.stack(diagnostics)}
    return result, {
        "finite": finite,
        "max_negative_state_violation": max_negative,
        "max_capacity_violation": max_capacity,
        "min_routed_q": min_q,
    }


def _model_audit(
    model_id: int,
    generated,
    compiled,
    forcing: torch.Tensor,
) -> dict[str, Any]:
    prepared = _prepare(model_id, forcing)
    current_step = _make_sequential_step(ORDER, N_SUBSTEPS)
    current, current_invariants = _run_checked(current_step, prepared, current=True)
    generated_result, generated_invariants = _run_checked(generated, prepared)
    compiled_result, compiled_invariants = _run_checked(compiled, prepared)
    active_positions = [STATE_NAMES.index(name) for name in prepared["spec"].state_names]
    eager_gradient = _gradient_probe(model_id, forcing[:2])
    compiled_gradient = _compiled_gradient_probe(model_id, forcing[:2], generated, compiled)
    water_index = len(FLUX_NAMES)
    snow_index = water_index + 1
    parity_values = {
        "generated_eager_packed_max_abs": _max_abs(current["packed"], generated_result["packed"]),
        "generated_eager_state_max_abs": _max_abs(current["packed"][active_positions], generated_result["packed"][active_positions]),
        "generated_eager_q_max_abs": _max_abs(current["q"], generated_result["q"]),
        "generated_eager_diagnostics_max_abs": _max_abs(current["diagnostics"], generated_result["diagnostics"]),
        "generated_eager_water_balance_max_abs": _max_abs(current["diagnostics"][:, water_index], generated_result["diagnostics"][:, water_index]),
        "compiled_packed_max_abs": _max_abs(generated_result["packed"], compiled_result["packed"]),
        "compiled_q_max_abs": _max_abs(generated_result["q"], compiled_result["q"]),
        "compiled_diagnostics_max_abs": _max_abs(generated_result["diagnostics"], compiled_result["diagnostics"]),
        "compiled_water_balance_max_abs": _max_abs(generated_result["diagnostics"][:, water_index], compiled_result["diagnostics"][:, water_index]),
    }
    water_balance_abs = {
        "current_max_abs": float(current["diagnostics"][:, water_index].abs().max().detach().cpu()),
        "generated_max_abs": float(generated_result["diagnostics"][:, water_index].abs().max().detach().cpu()),
        "compiled_max_abs": float(compiled_result["diagnostics"][:, water_index].abs().max().detach().cpu()),
        "snow_current_max_abs": float(current["diagnostics"][:, snow_index].abs().max().detach().cpu()),
        "snow_generated_max_abs": float(generated_result["diagnostics"][:, snow_index].abs().max().detach().cpu()),
        "snow_compiled_max_abs": float(compiled_result["diagnostics"][:, snow_index].abs().max().detach().cpu()),
    }
    invariant_values = {
        "current": current_invariants,
        "generated": generated_invariants,
        "compiled": compiled_invariants,
    }
    all_errors = tuple(value for value in parity_values.values())
    all_balances = tuple(water_balance_abs.values())
    all_state_invariants = tuple(
        value
        for invariant in invariant_values.values()
        for key, value in invariant.items()
        if key != "finite" and key != "min_routed_q"
    )
    min_routed_q_values = tuple(invariant["min_routed_q"] for invariant in invariant_values.values())
    passed = (
        _finite(current["packed"], current["q"], current["diagnostics"], generated_result["packed"], generated_result["q"], generated_result["diagnostics"], compiled_result["packed"], compiled_result["q"], compiled_result["diagnostics"])
        and current_invariants["finite"]
        and generated_invariants["finite"]
        and compiled_invariants["finite"]
        and eager_gradient["finite"]
        and compiled_gradient["finite"]
        and eager_gradient["inactive_gradients_zero_or_none"]
        and compiled_gradient["inactive_gradients_zero_or_none"]
        and all(math.isfinite(value) and value <= TOLERANCE for value in all_errors)
        and all(math.isfinite(value) and value <= TOLERANCE for value in all_balances)
        and all(math.isfinite(value) and value <= TOLERANCE for value in all_state_invariants)
        and all(math.isfinite(value) and value >= -TOLERANCE for value in min_routed_q_values)
        and all(value >= -TOLERANCE for value in (current_invariants["min_routed_q"], generated_invariants["min_routed_q"], compiled_invariants["min_routed_q"]))
        and eager_gradient["active_gradient_max_abs"] <= TOLERANCE
        and compiled_gradient["active_gradient_max_abs"] <= TOLERANCE
    )
    return {
        "model_id": model_id,
        "signature_digest": generated.graph_signature.digest,
        "status": "passed" if passed else "failed",
        "code_object_id": id(generated.__code__),
        "code_object_name": generated.__qualname__,
        "parity": parity_values,
        "water_balance": water_balance_abs,
        "invariants": invariant_values,
        "eager_gradient": eager_gradient,
        "compiled_gradient": compiled_gradient,
    }


def _benchmark_one(model_id: int, signature: GraphSignature, generated, compiled, device: torch.device) -> dict[str, Any]:
    forcing = _default_forcing(128, device)
    prepared = _prepare(model_id, forcing)
    with torch.no_grad():
        _series(generated, prepared, 128)
        _series(compiled, prepared, 128)
    _sync(device)
    with torch.no_grad():
        eager_forward = _measure(lambda: _series(generated, prepared, 128), device)
        compiled_forward = _measure(lambda: _series(compiled, prepared, 128), device)

    def make_loss(step):
        params = {
            name: torch.tensor(value, dtype=forcing.dtype, device=device, requires_grad=True)
            for name, value in default_parameters().items()
        }
        local = _prepare(model_id, forcing, params)
        _, total = _series(step, local, 128)
        return total

    def total_run(step):
        loss = make_loss(step)
        loss.backward()
        return loss

    total_run(generated)
    total_run(compiled)
    _sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    eager_backward = _measure_backward(lambda: make_loss(generated), device)
    eager_memory = _gpu_memory(device)
    torch.cuda.reset_peak_memory_stats(device)
    compiled_backward = _measure_backward(lambda: make_loss(compiled), device)
    compiled_memory = _gpu_memory(device)
    eager_total = _measure(lambda: total_run(generated), device)
    compiled_total = _measure(lambda: total_run(compiled), device)
    return {
        "model_id": model_id,
        "signature_digest": signature.digest,
        "decisions": dict(signature.decisions),
        "steps": 128,
        "forward": {"generated_eager": eager_forward, "compiled": compiled_forward, "speedup": eager_forward["median_seconds"] / compiled_forward["median_seconds"]},
        "backward": {"generated_eager": eager_backward, "compiled": compiled_backward, "speedup": eager_backward["median_seconds"] / compiled_backward["median_seconds"]},
        "forward_backward": {"generated_eager": eager_total, "compiled": compiled_total, "speedup": eager_total["median_seconds"] / compiled_total["median_seconds"]},
        "peak_gpu_memory": {"generated_eager": eager_memory, "compiled": compiled_memory},
    }


def _build_manifest() -> tuple[dict[str, Any], list[tuple[GraphSignature, list[Any]]]]:
    catalog_path = Path(__file__).parents[2] / "dfuse" / "specs" / "structures_78.json"
    specs = enumerate_structures()
    groups: dict[GraphSignature, list[Any]] = {}
    rows = []
    digest_payloads: dict[str, set[str]] = {}
    for spec in specs:
        signature = GraphSignature.from_structure(spec, sequential_order=ORDER, n_substeps=N_SUBSTEPS)
        groups.setdefault(signature, []).append(spec)
        payload = signature.to_dict()
        digest_payloads.setdefault(signature.digest, set()).add(_canonical_hash(payload))
        rows.append(
            {
                "model_id": spec.model_id,
                "decision_vector": list(spec.decision_vector),
                "decisions": dict(spec.decisions),
                "signature_digest": signature.digest,
                "graph_signature": payload,
            }
        )
    model_to_signature = {str(row["model_id"]): row["signature_digest"] for row in rows}
    signature_to_models: dict[str, list[int]] = {}
    for signature, members in groups.items():
        signature_to_models[signature.digest] = [spec.model_id for spec in members]
    collisions = {digest: sorted(payloads) for digest, payloads in digest_payloads.items() if len(payloads) > 1}
    manifest_body = {"rows": rows, "order": list(ORDER), "n_substeps": N_SUBSTEPS}
    manifest = {
        "catalog_path": str(catalog_path),
        "catalog_sha256": _sha256_file(catalog_path),
        "catalog_validation": validate_catalog(),
        "order_name": ORDER_NAME,
        "order": list(ORDER),
        "n_substeps": N_SUBSTEPS,
        "n_models": len(specs),
        "n_unique_signatures": len(groups),
        "n_shared_signatures": sum(len(members) > 1 for members in groups.values()),
        "model_to_signature": model_to_signature,
        "signature_to_models": signature_to_models,
        "signature_hash_collisions": collisions,
        "unexpected_signature_collision": bool(collisions),
        "models": rows,
        "manifest_sha256": _canonical_hash(manifest_body),
    }
    return manifest, list(groups.items())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _clear_inductor_memory() -> None:
    """Release loaded in-memory compiler modules without touching disk caches."""
    try:
        import torch._inductor.codecache as codecache
        for class_name in ("PyCodeCache", "CppCodeCache", "CppPythonBindingsCodeCache", "CudaKernelParamCache"):
            cache_class = getattr(codecache, class_name, None)
            if cache_class is None:
                continue
            for attribute in ("modules", "modules_no_attr", "linemaps", "cache"):
                value = getattr(cache_class, attribute, None)
                if hasattr(value, "clear"):
                    value.clear()
    except Exception:
        pass


def _cleanup_runtime() -> None:
    reset_runtime_registries()
    try:
        torch.compiler.reset()
    except Exception:
        try:
            torch._dynamo.reset()
        except Exception:
            pass
    _clear_inductor_memory()
    try:
        import ctypes
        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _resource_stop_reason(initial_rss_kb: int, device: torch.device, cache_dir: Path) -> str | None:
    current_rss = _rss_kb()
    if current_rss >= RSS_STOP_KB or current_rss - initial_rss_kb >= RSS_GROWTH_STOP_KB:
        return f"host RSS safety stop: current={current_rss}KB initial={initial_rss_kb}KB"
    if shutil.disk_usage(cache_dir).free < DISK_FREE_STOP_BYTES:
        return "persistent cache filesystem has less than 2GiB free"
    total_gpu = int(torch.cuda.get_device_properties(device).total_memory)
    if torch.cuda.memory_reserved(device) >= int(total_gpu * 0.85):
        return "GPU reserved memory remained above 85% after cleanup"
    return None


def _signature_record(diagnostics: Mapping[str, Any], digest: str) -> dict[str, Any] | None:
    for key, record in diagnostics.get("records", {}).items():
        if key.startswith(digest + "|"):
            return record
    return None


def _partial_payload(
    manifest: Mapping[str, Any],
    signature_audits: list[dict[str, Any]],
    model_audits: list[dict[str, Any]],
    resource_growth: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    stopped_early: bool,
    stop_reason: str | None,
    cache_dir: Path,
) -> dict[str, Any]:
    return {
        "schema": "dfuse-runtime-step-validation-78-v1",
        "status": "partial",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": "torch.float64",
            "cache_dir": str(cache_dir.resolve()),
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "formal_training_started": False,
            "sce_started": False,
            "dpl_started": False,
        },
        "manifest": manifest,
        "signature_audits": signature_audits,
        "model_audits": model_audits,
        "resource_growth": resource_growth,
        "failures": failures,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "cache_final": _cache_info(cache_dir),
    }


def _cache_probe(output: Path, cache_dir: Path, model_id: int) -> dict[str, Any]:
    _set_resource_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("cache probe requires CUDA")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
    device = torch.device("cuda")
    forcing = _default_forcing(2, device)
    reset_runtime_registries()
    before = _cache_info(cache_dir)
    started = time.perf_counter()
    spec = get_structure(model_id)
    signature, generated = get_generated_step(spec, order=ORDER, n_substeps=N_SUBSTEPS)
    _, compiled = get_compiled_step(spec, order=ORDER, n_substeps=N_SUBSTEPS, device=device, dtype=forcing.dtype, input_shapes=INPUT_SHAPES, backend="inductor", fullgraph=True)
    prepared = _prepare(model_id, forcing)
    result = _run(compiled, prepared)
    _compiled_gradient_probe(model_id, forcing, generated, compiled)
    _sync(device)
    elapsed = time.perf_counter() - started
    diagnostics = runtime_compile_diagnostics()
    record = _signature_record(diagnostics, signature.digest)
    payload = {
        "schema": "dfuse-runtime-step-cache-probe-v1",
        "model_id": model_id,
        "signature_digest": signature.digest,
        "cache_dir": str(cache_dir.resolve()),
        "elapsed_seconds": elapsed,
        "q": float(result["q"].detach().cpu()[0]),
        "cache_before": before,
        "cache_after": _cache_info(cache_dir),
        "resource": {"host_rss_kb": _rss_kb(), "host_peak_rss_kb": _peak_rss_kb(), "gpu": _gpu_memory(device)},
        "dynamo_counters": record["dynamo_counters"] if record else {},
        "compile_record": record,
    }
    _write_json(output, payload)
    print(json.dumps({"model_id": model_id, "signature_digest": signature.digest, "elapsed_seconds": elapsed}, sort_keys=True))
    return payload


def _run_cross_process_audit(output: Path, cache_dir: Path, model_rows: Mapping[int, Mapping[str, Any]], signature_audits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    probes = []
    failures = []
    root = Path(__file__).parents[2]
    for model_id in CROSS_PROCESS_MODELS:
        probe_path = output.with_name(f"{output.stem}_cache_probe_{model_id}.json")
        command = [
            sys.executable,
            "-m",
            "project.autofuse.runtime_validation_78",
            "--mode",
            "cache-probe",
            "--cache-dir",
            str(cache_dir),
            "--model-id",
            str(model_id),
            "--output",
            str(probe_path),
        ]
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
        try:
            completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=1200, check=False)
            if completed.returncode != 0:
                failures.append({"model_id": model_id, "returncode": completed.returncode, "stderr": completed.stderr[-2000:]})
                continue
            probe = json.loads(probe_path.read_text())
            digest = probe["signature_digest"]
            record = probe.get("compile_record") or {}
            counters = probe.get("dynamo_counters", {})
            inductor = counters.get("inductor", {})
            aot = counters.get("aot_autograd", {})
            reference = model_rows[model_id]
            cold = float(signature_audits[digest].get("cold_compile_seconds", [math.inf])[0])
            q_diff = abs(float(probe["q"]) - float(reference["reference_q"]))
            audit = {
                "model_id": model_id,
                "signature_digest": digest,
                "probe_output": str(probe_path),
                "fxgraph_cache_hit": int(inductor.get("fxgraph_cache_hit", 0)),
                "aot_autograd_cache_hit": int(aot.get("autograd_cache_hit", 0)),
                "cache_stable": probe["cache_before"] == probe["cache_after"],
                "output_max_abs": q_diff,
                "clean_cold_compile_seconds": cold,
                "warm_process_elapsed_seconds": float(probe["elapsed_seconds"]),
                "warm_vs_clean_cold_ratio": float(probe["elapsed_seconds"]) / cold if math.isfinite(cold) and cold > 0 else math.inf,
                "compile_record": record,
            }
            audit["passed"] = bool(
                audit["fxgraph_cache_hit"] >= 1
                and audit["aot_autograd_cache_hit"] >= 1
                and audit["cache_stable"]
                and q_diff <= TOLERANCE
                and audit["warm_vs_clean_cold_ratio"] < 0.75
            )
            probes.append(audit)
        except Exception as exc:
            failures.append({"model_id": model_id, "error": _error_text(exc)})
    payload = {"models": list(CROSS_PROCESS_MODELS), "probes": probes, "failures": failures, "all_passed": len(probes) == len(CROSS_PROCESS_MODELS) and not failures and all(row["passed"] for row in probes)}
    _write_json(output.with_name(output.stem + "_cross_process.json"), payload)
    return payload


def run_validation(output: Path, cache_dir: Path, *, run_cross_process: bool = True) -> dict[str, Any]:
    _set_resource_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("full 78 validation requires CUDA; refusing CPU fallback")
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    device = torch.device("cuda")
    initial_rss = _rss_kb()
    torch.cuda.reset_peak_memory_stats(device)
    initial_gpu = _gpu_memory(device)
    reset_runtime_registries()

    # Metadata pass: this is the only operation before compilation and writes a
    # durable manifest even if a later signature fails or a safety stop fires.
    manifest, groups = _build_manifest()
    _write_json(MANIFEST_OUTPUT, manifest)
    metadata_diagnostics = runtime_compile_diagnostics()
    metadata_compile_calls = int(metadata_diagnostics["compile_calls"])
    signature_audits: list[dict[str, Any]] = []
    model_audits: list[dict[str, Any]] = []
    resource_growth: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    representative_benchmarks: list[dict[str, Any]] = []
    stopped_early = False
    stop_reason = None

    for index, (signature, members) in enumerate(groups, start=1):
        if stopped_early:
            break
        reset_runtime_registries()
        torch.cuda.reset_peak_memory_stats(device)
        cache_before = _cache_info(cache_dir)
        rss_before = _rss_kb()
        gpu_before = _gpu_memory(device)
        generated = None
        compiled = None
        signature_record: dict[str, Any] = {
            "signature_index": index,
            "signature_digest": signature.digest,
            "model_ids": [spec.model_id for spec in members],
            "graph_signature": signature.to_dict(),
            "status": "failed",
            "cache_before": cache_before,
            "resource_before": {"host_rss_kb": rss_before, "host_peak_rss_kb": _peak_rss_kb(), "gpu": gpu_before},
        }
        benchmark_models: list[int] = []
        try:
            signature_built, generated = get_generated_step(members[0], order=ORDER, n_substeps=N_SUBSTEPS)
            if signature_built != signature:
                raise AssertionError("canonical signature changed between metadata and runtime build")
            signature_record["code_object_id"] = id(generated.__code__)
            signature_record["code_object_name"] = generated.__qualname__
            signature_record["generated_source_sha256"] = _canonical_hash(getattr(generated, "generated_source", ""))
            _, compiled = get_compiled_step(
                members[0],
                order=ORDER,
                n_substeps=N_SUBSTEPS,
                device=device,
                dtype=torch.float64,
                input_shapes=INPUT_SHAPES,
                backend="inductor",
                fullgraph=True,
            )
            for spec in members:
                try:
                    row = _model_audit(spec.model_id, generated, compiled, _default_forcing(4, device))
                    row["graph_signature"] = signature.to_dict()
                    model_audits.append(row)
                    if spec.model_id in REPRESENTATIVE_MODELS and row["status"] == "passed":
                        benchmark_models.append(spec.model_id)
                except Exception as exc:
                    model_audits.append({"model_id": spec.model_id, "signature_digest": signature.digest, "status": "failed", "error": _error_text(exc)})
                    failures.append({"kind": "model_validation", "model_id": spec.model_id, "signature_digest": signature.digest, "error": _error_text(exc)})
            record = _signature_record(runtime_compile_diagnostics(), signature.digest)
            if record is not None:
                signature_record.update(copy.deepcopy(record))
            signature_record["status"] = "passed" if all(row.get("status") == "passed" for row in model_audits if row.get("signature_digest") == signature.digest) else "failed"
            if signature_record["status"] != "passed":
                failures.append({"kind": "signature_validation", "signature_digest": signature.digest, "model_ids": signature_record["model_ids"]})
            else:
                for benchmark_model_id in benchmark_models:
                    representative_benchmarks.append(_benchmark_one(benchmark_model_id, signature, generated, compiled, device))
        except Exception as exc:
            signature_record["error"] = _error_text(exc)
            failures.append({"kind": "signature_compile_or_build", "signature_digest": signature.digest, "model_ids": signature_record["model_ids"], "error": _error_text(exc)})
            for spec in members:
                model_audits.append({"model_id": spec.model_id, "signature_digest": signature.digest, "status": "not_run", "error": signature_record["error"]})
            record = _signature_record(runtime_compile_diagnostics(), signature.digest)
            if record is not None:
                signature_record.update(copy.deepcopy(record))
        finally:
            gpu_peak = _gpu_memory(device)
            signature_record["resource_peak"] = {"host_rss_kb": _rss_kb(), "host_peak_rss_kb": _peak_rss_kb(), "gpu": gpu_peak}
            del compiled
            del generated
            _cleanup_runtime()
            gpu_after = _gpu_memory(device)
            cache_after = _cache_info(cache_dir)
            signature_record["cache_after"] = cache_after
            signature_record["cache_increment"] = {"files": cache_after["files"] - cache_before["files"], "bytes": cache_after["bytes"] - cache_before["bytes"]}
            signature_record["resource_after_cleanup"] = {"host_rss_kb": _rss_kb(), "host_peak_rss_kb": _peak_rss_kb(), "gpu": gpu_after}
            signature_audits.append(signature_record)
            resource_growth.append(
                {
                    "signature_index": index,
                    "signature_digest": signature.digest,
                    "host_rss_before_kb": rss_before,
                    "host_rss_after_cleanup_kb": _rss_kb(),
                    "host_peak_rss_kb": _peak_rss_kb(),
                    "gpu_allocated_before_bytes": gpu_before["allocated_bytes"],
                    "gpu_reserved_before_bytes": gpu_before["reserved_bytes"],
                    "gpu_peak_allocated_bytes": gpu_peak["peak_allocated_bytes"],
                    "gpu_peak_reserved_bytes": gpu_peak["peak_reserved_bytes"],
                    "gpu_allocated_after_cleanup_bytes": gpu_after["allocated_bytes"],
                    "gpu_reserved_after_cleanup_bytes": gpu_after["reserved_bytes"],
                    "cache_files": cache_after["files"],
                    "cache_bytes": cache_after["bytes"],
                    "cache_increment_bytes": cache_after["bytes"] - cache_before["bytes"],
                }
            )
            partial = _partial_payload(manifest, signature_audits, model_audits, resource_growth, failures, stopped_early, stop_reason, cache_dir)
            _write_json(output.with_suffix(".partial.json"), partial)
            stop_reason = _resource_stop_reason(initial_rss, device, cache_dir)
            if stop_reason:
                stopped_early = True

    reference_qs: dict[int, float] = {}
    # Cross-process probes use a fresh eager generated reference.  This pass
    # does not compile and is released before spawning probe processes.
    for model_id in CROSS_PROCESS_MODELS:
        if any(row.get("model_id") == model_id and row.get("status") == "passed" for row in model_audits):
            prepared = _prepare(model_id, _default_forcing(1, device))
            _, reference_step = get_generated_step(get_structure(model_id), order=ORDER, n_substeps=N_SUBSTEPS)
            reference = _run(reference_step, prepared)
            reference_qs[model_id] = float(reference["q"].detach().cpu()[0])
            del reference_step, reference, prepared
    _cleanup_runtime()

    model_rows_by_id = {model_id: {"reference_q": value} for model_id, value in reference_qs.items()}
    signature_rows_by_digest = {row["signature_digest"]: row for row in signature_audits}
    cross_process = {"skipped": not run_cross_process, "models": [], "probes": [], "failures": [], "all_passed": False}
    if run_cross_process and not stopped_early:
        cross_process = _run_cross_process_audit(output, cache_dir, model_rows_by_id, signature_rows_by_digest)
    if run_cross_process and not cross_process["all_passed"]:
        failures.append({"kind": "cross_process_cache", "details": cross_process})

    compiled_success = [row for row in signature_audits if row.get("compile_successes", 0) >= 1 and row.get("status") == "passed"]
    model_pass = [row for row in model_audits if row.get("status") == "passed"]
    all_compile_audits_clean = all(
        row.get("compile_attempts") == 1 and row.get("graph_breaks") == 0 and row.get("recompilations") == 0 and row.get("autograd_recompilations") == 0
        for row in compiled_success
    )
    if (
        len(enumerate_structures()) == 78
        and len(model_audits) == 78
        and len(model_pass) == 78
        and len(signature_audits) == len(groups)
        and len(compiled_success) == len(groups)
        and all_compile_audits_clean
        and not failures
        and not stopped_early
    ):
        final_status = "78-structure runtime compiler validated"
    elif model_pass or compiled_success:
        final_status = "validated with limited exceptions"
    else:
        final_status = "not ready"

    payload = _partial_payload(manifest, signature_audits, model_audits, resource_growth, failures, stopped_early, stop_reason, cache_dir)
    payload.update(
        {
            "status": final_status,
            "metadata_pass": {"compile_calls_before": metadata_compile_calls, "generated_builds_before": metadata_diagnostics["generated_builds"], "metadata_triggered_compilation": metadata_compile_calls != 0},
            "n_models": 78,
            "n_unique_signatures": len(groups),
            "n_runtime_build_success": sum(len(row["model_ids"]) for row in signature_audits if "code_object_id" in row),
            "n_unique_signatures_compiled_successfully": len(compiled_success),
            "n_models_parity_pass": len(model_pass),
            "persistent_cache_audit": {"cache_dir": str(cache_dir), "final": _cache_info(cache_dir), "total_artifacts": _cache_info(cache_dir)["files"], "total_bytes": _cache_info(cache_dir)["bytes"]},
            "resource_summary": {
                "host_rss_baseline_kb": initial_rss,
                "host_rss_peak_kb": max([initial_rss, _peak_rss_kb()] + [int(row["host_peak_rss_kb"]) for row in resource_growth]),
                "host_rss_end_kb": _rss_kb(),
                "gpu_baseline": initial_gpu,
                "gpu_end": _gpu_memory(device),
                "resource_stop_thresholds": {"rss_stop_kb": RSS_STOP_KB, "rss_growth_stop_kb": RSS_GROWTH_STOP_KB, "disk_free_stop_bytes": DISK_FREE_STOP_BYTES},
            },
            "representative_models": list(REPRESENTATIVE_MODELS),
            "representative_cuda_benchmark": representative_benchmarks,
            "persistent_cache_cross_process": cross_process,
            "failures": failures,
            "scientific_logic_modified": False,
            "formal_training_started": False,
            "sce_started": False,
            "dpl_started": False,
            "final_status": final_status,
        }
    )
    _write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "cache-probe"), default="validate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-id", type=int)
    parser.add_argument("--skip-cross-process", action="store_true")
    args = parser.parse_args()
    if args.mode == "cache-probe":
        if args.model_id is None:
            raise SystemExit("--model-id is required for cache-probe")
        _cache_probe(args.output, args.cache_dir, args.model_id)
        return
    payload = run_validation(args.output, args.cache_dir, run_cross_process=not args.skip_cross_process)
    print(json.dumps({
        "status": payload["final_status"],
        "n_models": payload["n_models"],
        "n_unique_signatures": payload["n_unique_signatures"],
        "compiled_signatures": payload["n_unique_signatures_compiled_successfully"],
        "model_parity_pass": payload["n_models_parity_pass"],
        "failures": len(payload["failures"]),
        "cache": payload["persistent_cache_audit"]["final"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
