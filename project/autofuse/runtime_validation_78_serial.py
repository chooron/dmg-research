"""Resource-safe, serial full-catalog runtime-step validation.

The parent performs the 78-row metadata pass and launches exactly one fresh
worker process per canonical signature.  Workers are never concurrent: each
builds, compiles, validates, persists its cache artifacts, records resources,
and exits before the next signature starts.  This is deliberate process
isolation for PyTorch/Inductor native compiler state; it prevents an observed
in-process allocator/cache RSS accumulation from violating the hard resource
limit while preserving the required one-signature-at-a-time protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import shutil
import time
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from dfuse import GraphSignature, enumerate_structures, get_structure, reset_runtime_registries, runtime_compile_diagnostics
from dfuse.kernel import SEQUENTIAL_ORDERS
from dfuse.runtime import get_compiled_step, get_generated_step
from project.autofuse import runtime_validation_78 as base

ORDER_NAME = base.ORDER_NAME
ORDER = SEQUENTIAL_ORDERS[ORDER_NAME]
N_SUBSTEPS = base.N_SUBSTEPS
INPUT_SHAPES = base.INPUT_SHAPES
TOLERANCE = base.TOLERANCE
DEFAULT_CACHE_DIR = Path(__file__).with_name(".cache") / "runtime-validation-78"
DEFAULT_OUTPUT = Path(__file__).with_name("docs") / "runtime_step_validation_78.json"
REPRESENTATIVE_MODELS = base.REPRESENTATIVE_MODELS
CROSS_PROCESS_MODELS = base.CROSS_PROCESS_MODELS


def _worker(output: Path, cache_dir: Path, model_id: int, benchmark: bool) -> dict[str, Any]:
    base._set_resource_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("signature worker requires CUDA")
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    device = torch.device("cuda")
    spec = get_structure(model_id)
    signature = GraphSignature.from_structure(spec, sequential_order=ORDER, n_substeps=N_SUBSTEPS)
    reset_runtime_registries()
    torch.cuda.reset_peak_memory_stats(device)
    rss_before = base._rss_kb()
    gpu_before = base._gpu_memory(device)
    cache_before = base._cache_info(cache_dir)
    started = time.perf_counter()
    generated = None
    compiled = None
    model_row: dict[str, Any] = {"model_id": model_id, "signature_digest": signature.digest, "status": "not_run"}
    signature_row: dict[str, Any] = {
        "signature_digest": signature.digest,
        "model_ids": [model_id],
        "graph_signature": signature.to_dict(),
        "status": "failed",
        "cache_before": cache_before,
        "resource_before": {"host_rss_kb": rss_before, "host_peak_rss_kb": base._peak_rss_kb(), "gpu": gpu_before},
    }
    failures: list[dict[str, Any]] = []
    benchmark_row = None
    reference_q = None
    try:
        built_signature, generated = get_generated_step(spec, order=ORDER, n_substeps=N_SUBSTEPS)
        if built_signature != signature:
            raise AssertionError("worker signature differs from metadata manifest")
        signature_row.update(
            {
                "code_object_id": id(generated.__code__),
                "code_object_name": generated.__qualname__,
                "generated_source_sha256": base._canonical_hash(getattr(generated, "generated_source", "")),
            }
        )
        _, compiled = get_compiled_step(
            spec,
            order=ORDER,
            n_substeps=N_SUBSTEPS,
            device=device,
            dtype=torch.float64,
            input_shapes=INPUT_SHAPES,
            backend="inductor",
            fullgraph=True,
        )
        forcing = base._default_forcing(4, device)
        model_row = base._model_audit(model_id, generated, compiled, forcing)
        model_row["graph_signature"] = signature.to_dict()
        reference = base._run(generated, base._prepare(model_id, base._default_forcing(1, device)))
        reference_q = float(reference["q"].detach().cpu()[0])
        del reference
        record = base._signature_record(runtime_compile_diagnostics(), signature.digest)
        if record is not None:
            # Snapshot before representative benchmarking so intentional
            # no-grad/grad benchmark paths cannot look like guard recompiles.
            signature_row.update(copy.deepcopy(record))
        signature_row["status"] = "passed" if model_row["status"] == "passed" else "failed"
        if signature_row["status"] != "passed":
            failures.append({"kind": "model_validation", "model_id": model_id, "signature_digest": signature.digest})
        if benchmark and model_row["status"] == "passed":
            benchmark_row = base._benchmark_one(model_id, signature, generated, compiled, device)
    except Exception as exc:
        error = base._error_text(exc)
        signature_row["error"] = error
        failures.append({"kind": "signature_build_compile_or_validation", "model_id": model_id, "signature_digest": signature.digest, "error": error})
        model_row = {"model_id": model_id, "signature_digest": signature.digest, "status": "failed", "error": error}
    finally:
        gpu_peak = base._gpu_memory(device)
        signature_row["resource_peak"] = {"host_rss_kb": base._rss_kb(), "host_peak_rss_kb": base._peak_rss_kb(), "gpu": gpu_peak}
        del compiled
        del generated
        base._cleanup_runtime()
        gpu_after = base._gpu_memory(device)
        cache_after = base._cache_info(cache_dir)
        signature_row["cache_after"] = cache_after
        signature_row["cache_increment"] = {"files": cache_after["files"] - cache_before["files"], "bytes": cache_after["bytes"] - cache_before["bytes"]}
        signature_row["resource_after_cleanup"] = {"host_rss_kb": base._rss_kb(), "host_peak_rss_kb": base._peak_rss_kb(), "gpu": gpu_after}
    resource_row = {
        "signature_digest": signature.digest,
        "host_rss_before_kb": rss_before,
        "host_rss_after_cleanup_kb": signature_row["resource_after_cleanup"]["host_rss_kb"],
        "host_peak_rss_kb": signature_row["resource_peak"]["host_peak_rss_kb"],
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
    stop_reason = base._resource_stop_reason(rss_before, device, cache_dir)
    payload = {
        "schema": "dfuse-runtime-step-signature-worker-v1",
        "elapsed_seconds": time.perf_counter() - started,
        "model_id": model_id,
        "signature_digest": signature.digest,
        "signature_audit": signature_row,
        "model_audit": model_row,
        "resource_growth": resource_row,
        "resource_stop_reason": stop_reason,
        "benchmark": benchmark_row,
        "reference_q": reference_q,
        "failures": failures,
    }
    base._write_json(output, payload)
    print(json.dumps({"model_id": model_id, "signature_digest": signature.digest, "status": signature_row["status"]}, sort_keys=True))
    return payload


def _aggregate_metrics(model_rows: list[dict[str, Any]], signature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    parity_values = []
    water_values = []
    eager_gradient_values = []
    compiled_gradient_values = []
    for row in model_rows:
        parity = row.get("parity", {})
        parity_values.extend(value for value in parity.values() if isinstance(value, (int, float)))
        water_values.extend(value for value in row.get("water_balance", {}).values() if isinstance(value, (int, float)))
        eager_gradient_values.append(row.get("eager_gradient", {}).get("active_gradient_max_abs", math.inf))
        compiled_gradient_values.append(row.get("compiled_gradient", {}).get("active_gradient_max_abs", math.inf))
    cold = [float((row.get("initial_cold_compile_seconds") or row["cold_compile_seconds"])[0]) for row in signature_rows if row.get("cold_compile_seconds") or row.get("initial_cold_compile_seconds")]
    ordered = sorted(cold)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1) if ordered else 0
    return {
        "parity_max_abs": max(parity_values, default=math.inf),
        "water_balance_max_abs": max(water_values, default=math.inf),
        "eager_active_gradient_max_abs": max(eager_gradient_values, default=math.inf),
        "compiled_active_gradient_max_abs": max(compiled_gradient_values, default=math.inf),
        "cold_compile_seconds": {
            "count": len(cold),
            "min": min(cold, default=math.inf),
            "median": statistics.median(cold) if cold else math.inf,
            "p95": ordered[p95_index] if ordered else math.inf,
            "max": max(cold, default=math.inf),
        },
    }


def _run_affected_recheck(partial_path: Path, output: Path, cache_dir: Path) -> dict[str, Any]:
    prior = json.loads(partial_path.read_text())
    completed_ids = {int(row["model_id"]) for row in prior.get("model_audits", []) if row.get("status") == "passed"}
    affected_ids = [
        model_id
        for model_id in (int(row["model_id"]) for row in prior.get("model_audits", []))
        if model_id in completed_ids and get_structure(model_id).decisions["ARCH1"] in ("tension1_1", "tension2_1")
    ]
    cache_dir = cache_dir.resolve()
    worker_dir = cache_dir / "repair-reports"
    worker_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parents[2]
    results = []
    failures = []
    for model_id in affected_ids:
        worker_path = worker_dir / f"repair-{model_id}.json"
        command = [
            sys.executable,
            "-m",
            "project.autofuse.runtime_validation_78_serial",
            "--mode",
            "worker",
            "--model-id",
            str(model_id),
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(worker_path),
        ]
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        try:
            completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=1800, check=False)
            if completed.returncode != 0 or not worker_path.exists():
                failures.append({"model_id": model_id, "returncode": completed.returncode, "stderr": completed.stderr[-4000:]})
                continue
            report = json.loads(worker_path.read_text())
            model_row = report["model_audit"]
            signature_row = report["signature_audit"]
            resource_row = report["resource_growth"]
            passed = bool(
                model_row.get("status") == "passed"
                and signature_row.get("compile_attempts") == 1
                and signature_row.get("graph_breaks") == 0
                and signature_row.get("recompilations") == 0
                and model_row.get("parity", {}).get("generated_eager_packed_max_abs", math.inf) <= TOLERANCE
                and model_row.get("parity", {}).get("compiled_packed_max_abs", math.inf) <= TOLERANCE
            )
            results.append({"model_id": model_id, "signature_audit": signature_row, "model_audit": model_row, "resource_growth": resource_row, "passed": passed})
            if not passed:
                failures.append({"model_id": model_id, "error": "OFLOW_1 repair parity recheck failed"})
        except Exception as exc:
            failures.append({"model_id": model_id, "error": base._error_text(exc)})
        finally:
            if worker_path.exists():
                worker_path.unlink()
    runtime_path = Path(__file__).parents[2] / "dfuse" / "runtime.py"
    partial_sha256 = base._sha256_file(partial_path)
    payload = {
        "schema": "dfuse-runtime-step-oflow-recheck-v1",
        "source_partial": str(partial_path.resolve()),
        "source_partial_sha256": partial_sha256,
        "manifest_sha256": prior.get("manifest", {}).get("manifest_sha256"),
        "runtime_source_sha256": base._sha256_file(runtime_path),
        "source_partial_completed_model_ids": sorted(completed_ids),
        "affected_completed_model_ids": affected_ids,
        "results": results,
        "failures": failures,
        "all_passed": bool(len(results) == len(affected_ids) and not failures and all(row["passed"] for row in results)),
    }
    base._write_json(output, payload)
    return payload

def _run_cross_process_audit(output: Path, cache_dir: Path) -> dict[str, Any]:
    """Run two fresh workers per representative signature on a clean mini-cache."""
    cache_dir = cache_dir.resolve()
    root = Path(__file__).parents[2]
    reports_dir = cache_dir.parent / "runtime-cross-process-reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    probes = []
    failures = []
    for model_id in CROSS_PROCESS_MODELS:
        mini_cache = cache_dir.parent / f"runtime-cross-process-cache-{model_id}"
        if mini_cache.exists():
            shutil.rmtree(mini_cache)
        mini_cache.mkdir(parents=True, exist_ok=True)
        reports = []
        for process in (1, 2):
            report_path = reports_dir / f"{output.stem}-{model_id}-{process}.json"
            command = [
                sys.executable,
                "-m",
                "project.autofuse.runtime_validation_78_serial",
                "--mode",
                "worker",
                "--model-id",
                str(model_id),
                "--cache-dir",
                str(mini_cache),
                "--output",
                str(report_path),
            ]
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = str(mini_cache)
            try:
                completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=1800, check=False)
                if completed.returncode != 0 or not report_path.exists():
                    raise RuntimeError(f"worker returncode={completed.returncode}: {completed.stderr[-2000:]}")
                reports.append(json.loads(report_path.read_text()))
            except Exception as exc:
                failures.append({"model_id": model_id, "process": process, "error": base._error_text(exc)})
            finally:
                if report_path.exists():
                    report_path.unlink()
        if len(reports) == 2:
            first, second = reports
            second_counter = second["signature_audit"].get("dynamo_counters", {})
            second_inductor = second_counter.get("inductor", {})
            second_aot = second_counter.get("aot_autograd", {})
            cold = float(first["elapsed_seconds"])
            warm = float(second["elapsed_seconds"])
            q_diff = abs(float(first["reference_q"]) - float(second["reference_q"]))
            first_cache = first["signature_audit"].get("cache_after")
            second_before_cache = second["signature_audit"].get("cache_before")
            second_cache = second["signature_audit"].get("cache_after")
            cache_growth_bytes = int(second_cache["bytes"] - second_before_cache["bytes"])
            cache_growth_files = int(second_cache["files"] - second_before_cache["files"])
            row = {
                "model_id": model_id,
                "signature_digest": second["signature_digest"],
                "cache_dir": str(mini_cache),
                "process_1": {"elapsed_seconds": cold, "cache": first_cache, "compile_record": first["signature_audit"]},
                "process_2": {"elapsed_seconds": warm, "cache": second_cache, "compile_record": second["signature_audit"]},
                "cache_artifacts_stable": first_cache == second_before_cache and cache_growth_bytes <= 1024 * 1024 and cache_growth_files <= 4,
                "cache_growth_bytes": cache_growth_bytes,
                "cache_growth_files": cache_growth_files,
                "fxgraph_cache_hit": int(second_inductor.get("fxgraph_cache_hit", 0)),
                "aot_autograd_cache_hit": int(second_aot.get("autograd_cache_hit", 0)),
                "output_max_abs": q_diff,
                "warm_vs_cold_ratio": warm / cold if cold > 0 else math.inf,
            }
            row["passed"] = bool(row["cache_artifacts_stable"] and row["fxgraph_cache_hit"] >= 1 and row["aot_autograd_cache_hit"] >= 1 and q_diff <= TOLERANCE and row["warm_vs_cold_ratio"] < 0.75)
            probes.append(row)
        if mini_cache.exists():
            shutil.rmtree(mini_cache)
    payload = {"models": list(CROSS_PROCESS_MODELS), "probes": probes, "failures": failures, "all_passed": len(probes) == len(CROSS_PROCESS_MODELS) and not failures and all(row["passed"] for row in probes)}
    base._write_json(output.with_name(output.stem + "_cross_process.json"), payload)
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    return payload

def run_validation(output: Path, cache_dir: Path, *, run_cross_process: bool = True, partial_path: Path | None = None, recheck_path: Path | None = None) -> dict[str, Any]:
    base._set_resource_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("full 78 validation requires CUDA; refusing CPU fallback")
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    parent_rss_initial = base._rss_kb()
    device = torch.device("cuda")
    parent_gpu_initial = base._gpu_memory(device)
    reset_runtime_registries()

    manifest, groups = base._build_manifest()
    manifest_path = output.with_name(output.stem + "_manifest.json")
    base._write_json(manifest_path, manifest)
    metadata_diag = runtime_compile_diagnostics()
    signature_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    benchmarks: list[dict[str, Any]] = []
    if partial_path is not None:
        prior = json.loads(partial_path.read_text())
        current_runtime_sha256 = base._sha256_file(Path(__file__).parents[2] / "dfuse" / "runtime.py")
        if prior.get("runtime_source_sha256") != current_runtime_sha256:
            raise AssertionError("resume partial runtime source hash does not match current dfuse/runtime.py")
        expected_models = manifest["model_to_signature"]
        prior_model_rows = prior.get("model_audits", [])
        prior_model_ids = [int(row["model_id"]) for row in prior_model_rows]
        if len(prior_model_ids) != len(set(prior_model_ids)) or any(str(model_id) not in expected_models for model_id in prior_model_ids):
            raise AssertionError("resume partial contains an unknown or duplicate model ID")
        if any(expected_models[str(row["model_id"])] != row.get("signature_digest") for row in prior_model_rows):
            raise AssertionError("resume partial model-to-signature mapping does not match manifest")
        expected_signatures = manifest["signature_to_models"]
        prior_signature_rows = prior.get("signature_audits", [])
        if len({row["signature_digest"] for row in prior_signature_rows}) != len(prior_signature_rows):
            raise AssertionError("resume partial contains duplicate signature audits")
        expected_signature_payloads = {row["signature_digest"]: row["graph_signature"] for row in manifest["models"]}
        if any(
            row["signature_digest"] not in expected_signatures
            or sorted(row.get("model_ids", [])) != sorted(expected_signatures[row["signature_digest"]])
            or base._canonical_hash(row.get("graph_signature")) != base._canonical_hash(expected_signature_payloads[row["signature_digest"]])
            for row in prior_signature_rows
        ):
            raise AssertionError("resume partial signature digest or payload does not match manifest")
        if prior.get("manifest", {}).get("manifest_sha256") != manifest.get("manifest_sha256") :
            raise AssertionError("resume partial manifest does not match current authoritative catalog")
        signature_rows = copy.deepcopy(prior.get("signature_audits", []))
        model_rows = copy.deepcopy(prior.get("model_audits", []))
        resource_rows = copy.deepcopy(prior.get("resource_growth", []))
        failures = copy.deepcopy(prior.get("failures", []))
        benchmarks = copy.deepcopy(prior.get("representative_cuda_benchmark", []))
    if recheck_path is not None and partial_path is None:
        raise AssertionError("--recheck requires --partial")
    if recheck_path is not None:
        repair = json.loads(recheck_path.read_text())
        if not repair.get("all_passed") :
            raise AssertionError(f"OFLOW_1 repair recheck failed: {repair.get('failures')}")
        current_runtime_sha256 = base._sha256_file(Path(__file__).parents[2] / "dfuse" / "runtime.py")
        if repair.get("runtime_source_sha256") != current_runtime_sha256:
            raise AssertionError("OFLOW_1 recheck runtime source hash does not match current dfuse/runtime.py")
        source_partial_path = Path(repair.get("source_partial", "")).resolve()
        if not source_partial_path.is_file():
            raise AssertionError("OFLOW_1 recheck source partial artifact is missing")
        source_partial = json.loads(source_partial_path.read_text())
        if repair.get("source_partial_sha256") != base._sha256_file(source_partial_path):
            raise AssertionError("OFLOW_1 recheck source partial hash does not match")
        if source_partial.get("manifest", {}).get("manifest_sha256") != manifest.get("manifest_sha256") or source_partial.get("manifest_sha256") not in (None, manifest.get("manifest_sha256")):
            raise AssertionError("OFLOW_1 source partial manifest hash does not match current catalog")
        if source_partial.get("runtime_source_sha256") != current_runtime_sha256:
            raise AssertionError("OFLOW_1 source partial runtime source hash does not match current dfuse/runtime.py")
        source_model_rows = source_partial.get("model_audits", [])
        if any(str(row.get("model_id")) not in expected_models or expected_models[str(row["model_id"])] != row.get("signature_digest") or base._canonical_hash(row.get("graph_signature")) != base._canonical_hash(expected_signature_payloads[row.get("signature_digest")]) for row in source_model_rows):
            raise AssertionError("OFLOW_1 source partial model mapping does not match manifest")
        source_signature_rows = source_partial.get("signature_audits", [])
        if any(
            row.get("signature_digest") not in expected_signatures
            or sorted(row.get("model_ids", [])) != sorted(expected_signatures[row["signature_digest"]])
            or base._canonical_hash(row.get("graph_signature")) != base._canonical_hash(expected_signature_payloads[row["signature_digest"]])
            for row in source_signature_rows
        ):
            raise AssertionError("OFLOW_1 source partial signature mapping or payload does not match manifest")
        if repair.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise AssertionError("OFLOW_1 recheck manifest hash does not match current catalog")
        current_completed = {int(row["model_id"]) for row in prior.get("model_audits", []) if row.get("status") == "passed"}
        source_completed_list = [int(row["model_id"]) for row in source_partial.get("model_audits", []) if row.get("status") == "passed"]
        source_completed_ids = set(source_completed_list)
        declared_source_ids = [int(model_id) for model_id in repair.get("source_partial_completed_model_ids", [])]
        if len(source_completed_list) != len(source_completed_ids) or len(declared_source_ids) != len(set(declared_source_ids)) or set(declared_source_ids) != source_completed_ids:
            raise AssertionError("OFLOW_1 recheck source completion set is not an exact match to its source partial")
        if not source_completed_ids.issubset(current_completed):
            raise AssertionError("OFLOW_1 recheck source completion set is not present in the resume partial")
        expected_affected = sorted(model_id for model_id in source_completed_ids if get_structure(model_id).decisions["ARCH1"] in ("tension1_1", "tension2_1"))
        actual_affected = sorted(int(model_id) for model_id in repair.get("affected_completed_model_ids", []))
        if actual_affected != expected_affected or sorted(int(result["model_id"]) for result in repair.get("results", [])) != expected_affected:
            raise AssertionError("OFLOW_1 recheck model set does not match exact affected source set")
        for result in repair.get("results", []):
            model_id = int(result["model_id"])
            expected_digest = manifest["model_to_signature"].get(str(model_id))
            expected_payload = expected_signature_payloads[expected_digest]
            result_signature = result.get("signature_audit", {})
            result_model = result.get("model_audit", {})
            if (
                expected_digest != result_signature.get("signature_digest")
                or expected_digest != result_model.get("signature_digest")
                or sorted(result_signature.get("model_ids", [])) != sorted(expected_signatures[expected_digest])
                or base._canonical_hash(result_signature.get("graph_signature")) != base._canonical_hash(expected_payload)
                or base._canonical_hash(result_model.get("graph_signature")) != base._canonical_hash(expected_payload)
            ):
                raise AssertionError("OFLOW_1 recheck result signature payload does not match manifest")
        for result in repair["results"]:
            digest = result["signature_audit"]["signature_digest"]
            old_signature = next(row for row in signature_rows if row["signature_digest"] == digest)
            replacement_signature = copy.deepcopy(result["signature_audit"])
            replacement_signature["signature_index"] = old_signature["signature_index"]
            replacement_signature["initial_cold_compile_seconds"] = old_signature.get("cold_compile_seconds")
            replacement_signature["repair_recheck"] = {"source": str(recheck_path), "passed": result["passed"], "worker_cold_compile_seconds": result["signature_audit"].get("cold_compile_seconds")}
            signature_rows[signature_rows.index(old_signature)] = replacement_signature
            old_model = next(row for row in model_rows if row["model_id"] == result["model_id"])
            model_rows[model_rows.index(old_model)] = result["model_audit"]
            old_resource = next(row for row in resource_rows if row["signature_digest"] == digest)
            replacement_resource = copy.deepcopy(result["resource_growth"])
            replacement_resource["signature_index"] = old_resource["signature_index"]
            replacement_resource["repair_recheck"] = True
            resource_rows[resource_rows.index(old_resource)] = replacement_resource
    completed_digests = {row["signature_digest"] for row in signature_rows if row.get("status") == "passed" and row.get("compile_successes", 0) >= 1}
    groups_to_run = [(index, signature, members) for index, (signature, members) in enumerate(groups, start=1) if signature.digest not in completed_digests]
    stopped_early = False
    stop_reason = None
    worker_dir = cache_dir / "worker-reports"
    worker_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parents[2]

    for index, signature, members in groups_to_run:
        if stopped_early:
            break
        # Catalog currently has one model per signature, but this loop keeps
        # sharing semantics explicit: one worker is launched per signature.
        model_id = members[0].model_id
        worker_path = worker_dir / f"{index:03d}-{signature.digest}.json"
        command = [
            sys.executable,
            "-m",
            "project.autofuse.runtime_validation_78_serial",
            "--mode",
            "worker",
            "--model-id",
            str(model_id),
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(worker_path),
        ]
        if model_id in REPRESENTATIVE_MODELS:
            command.append("--benchmark")
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        try:
            completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=1800, check=False)
            if completed.returncode != 0 or not worker_path.exists():
                error = {"returncode": completed.returncode, "stderr": completed.stderr[-4000:]}
                failures.append({"kind": "worker_process", "signature_index": index, "signature_digest": signature.digest, "model_ids": [model_id], "error": error})
                signature_rows.append({"signature_index": index, "signature_digest": signature.digest, "model_ids": [model_id], "graph_signature": signature.to_dict(), "status": "failed", "error": error})
                model_rows.append({"model_id": model_id, "signature_digest": signature.digest, "status": "not_run", "error": error})
            else:
                report = json.loads(worker_path.read_text())
                worker_signature = report["signature_audit"]
                worker_signature["signature_index"] = index
                signature_rows.append(worker_signature)
                model_rows.append(report["model_audit"])
                resource = report["resource_growth"]
                resource["signature_index"] = index
                resource_rows.append(resource)
                if report.get("benchmark") is not None:
                    benchmarks.append(report["benchmark"])
                for failure in report.get("failures", []):
                    failures.append(failure)
                if report.get("resource_stop_reason"):
                    stopped_early = True
                    stop_reason = report["resource_stop_reason"]
        except Exception as exc:
            error = base._error_text(exc)
            failures.append({"kind": "worker_process", "signature_index": index, "signature_digest": signature.digest, "model_ids": [model_id], "error": error})
            signature_rows.append({"signature_index": index, "signature_digest": signature.digest, "model_ids": [model_id], "graph_signature": signature.to_dict(), "status": "failed", "error": error})
            model_rows.append({"model_id": model_id, "signature_digest": signature.digest, "status": "not_run", "error": error})
        finally:
            if worker_path.exists():
                worker_path.unlink()
            parent_stop = base._resource_stop_reason(parent_rss_initial, device, cache_dir)
            if parent_stop and not stopped_early:
                stopped_early = True
                stop_reason = parent_stop
            partial = base._partial_payload(manifest, signature_rows, model_rows, resource_rows, failures, stopped_early, stop_reason, cache_dir)
            partial["runtime_source_sha256"] = base._sha256_file(Path(__file__).parents[2] / "dfuse" / "runtime.py")
            partial["manifest_sha256"] = manifest["manifest_sha256"]
            partial["representative_cuda_benchmark"] = benchmarks
            base._write_json(output.with_suffix(".partial.json"), partial)

    cross_process = {"skipped": not run_cross_process, "models": [], "probes": [], "failures": [], "all_passed": False}
    if run_cross_process and not stopped_early:
        cross_process = _run_cross_process_audit(output, cache_dir)
        if not cross_process["all_passed"]:
            failures.append({"kind": "cross_process_cache", "details": cross_process})

    compiled_success = [row for row in signature_rows if row.get("compile_successes", 0) >= 1 and row.get("status") == "passed"]
    model_pass = [row for row in model_rows if row.get("status") == "passed"]
    clean_audits = all(
        row.get("compile_attempts") == 1
        and row.get("graph_breaks") == 0
        and row.get("recompilations") == 0
        and row.get("autograd_recompilations") == 0
        for row in compiled_success
    )
    if (
        len(enumerate_structures()) == 78
        and len(signature_rows) == len(groups)
        and len(model_rows) == 78
        and len(model_pass) == 78
        and len(compiled_success) == len(groups)
        and clean_audits
        and not failures
        and not stopped_early
    ):
        final_status = "78-structure runtime compiler validated"
    elif model_pass or compiled_success:
        final_status = "validated with limited exceptions"
    else:
        final_status = "not ready"

    parent_gpu_end = base._gpu_memory(device)
    worker_peaks = [row.get("gpu_peak_allocated_bytes", 0) for row in resource_rows]
    worker_reserved_peaks = [row.get("gpu_peak_reserved_bytes", 0) for row in resource_rows]
    worker_host_peaks = [row.get("host_peak_rss_kb", 0) for row in resource_rows]
    cache_final = base._cache_info(cache_dir)
    payload = base._partial_payload(manifest, signature_rows, model_rows, resource_rows, failures, stopped_early, stop_reason, cache_dir)
    payload.update(
        {
            "status": final_status,
            "execution_mode": "serial-signature-worker-process-isolation",
            "runtime_source_sha256": base._sha256_file(Path(__file__).parents[2] / "dfuse" / "runtime.py"),
            "metadata_pass": {
                "compile_calls_before": int(metadata_diag["compile_calls"]),
                "generated_builds_before": int(metadata_diag["generated_builds"]),
                "metadata_triggered_compilation": int(metadata_diag["compile_calls"]) != 0,
            },
            "n_models": 78,
            "n_unique_signatures": len(groups),
            "n_runtime_build_success": sum(1 for row in signature_rows if "code_object_id" in row),
            "n_unique_signatures_compiled_successfully": len(compiled_success),
            "n_models_parity_pass": len(model_pass),
            "persistent_cache_audit": {"cache_dir": str(cache_dir), "final": cache_final, "total_artifacts": cache_final["files"], "total_bytes": cache_final["bytes"]},
            "resource_summary": {
                "host_rss_baseline_kb": parent_rss_initial,
                "host_rss_peak_kb": max([parent_rss_initial, base._peak_rss_kb(), *worker_host_peaks]),
                "host_rss_end_kb": base._rss_kb(),
                "gpu_baseline": parent_gpu_initial,
                "gpu_peak_allocated_bytes": max(worker_peaks, default=parent_gpu_initial["peak_allocated_bytes"]),
                "gpu_peak_reserved_bytes": max(worker_reserved_peaks, default=parent_gpu_initial["peak_reserved_bytes"]),
                "gpu_end": parent_gpu_end,
                "resource_stop_thresholds": {"rss_stop_kb": base.RSS_STOP_KB, "rss_growth_stop_kb": base.RSS_GROWTH_STOP_KB, "disk_free_stop_bytes": base.DISK_FREE_STOP_BYTES},
            },
            "representative_models": list(REPRESENTATIVE_MODELS),
            "representative_cuda_benchmark": benchmarks,
            "persistent_cache_cross_process": cross_process,
            "aggregate_metrics": _aggregate_metrics(model_rows, signature_rows),
            "failures": failures,
            "scientific_logic_modified": False,
            "formal_training_started": False,
            "sce_started": False,
            "dpl_started": False,
            "final_status": final_status,
        }
    )
    base._write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "worker", "affected-recheck"), default="validate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partial", type=Path)
    parser.add_argument("--recheck", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-id", type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--skip-cross-process", action="store_true")
    args = parser.parse_args()
    if args.mode == "affected-recheck":
        if args.partial is None:
            raise SystemExit("--partial is required for affected-recheck")
        result = _run_affected_recheck(args.partial, args.output, args.cache_dir)
        print(json.dumps({"mode": args.mode, "models": len(result["affected_completed_model_ids"]), "all_passed": result["all_passed"]}, sort_keys=True))
        return
    if args.mode == "worker":
        if args.model_id is None:
            raise SystemExit("--model-id is required for worker mode")
        _worker(args.output, args.cache_dir, args.model_id, args.benchmark)
        return
    payload = run_validation(args.output, args.cache_dir, run_cross_process=not args.skip_cross_process, partial_path=args.partial, recheck_path=args.recheck)
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
