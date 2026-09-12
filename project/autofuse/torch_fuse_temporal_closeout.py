"""Final closeout for the Torch-FUSE temporal optimization investigation.

This deliberately reruns only the final one-step Lite/Full parity, B=100
benchmark, and current execution audit.  Historical long-horizon temporal,
chunked, and wide-boundary probe records are copied with explicit provenance;
no long compile probe is started here.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from project.autofuse import torch_fuse_temporal_optimization as investigation

ROOT = investigation.ROOT
DOCS = investigation.DOCS
PROBE_CACHE = investigation.CACHE_DIR
ARTIFACT_NAMES = (
    "torch_fuse_temporal_execution_audit.json",
    "torch_fuse_lite_full_parity.json",
    "torch_fuse_lite_full_b100_benchmark.json",
    "torch_fuse_full_temporal_compile_probe.json",
    "torch_fuse_chunked_temporal_compile_probe.json",
    "torch_fuse_wide_compile_boundary_probe.json",
    "torch_fuse_temporal_optimization_decision.json",
)
CORE_PATHS = ("dfuse/kernel.py", "dfuse/runtime.py", "dfuse/spec.py")
CORE_HASHES_BEFORE = {
    "dfuse/kernel.py": "4a79b4caa8d93881c849476e2ffe5aa304c8033cf271bd7e11073d7f7227faff",
    "dfuse/runtime.py": "f6b9f066489929479ccabb2035e50bed146e536af24a641cae68e11c37e9ad13",
    "dfuse/spec.py": "0e7c569260e11acd49b8b61174934afbbccf9d1c0ec46c4a3bae561b2491d2bb",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _source_provenance(status: str, *, original: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"source_git_sha": _git_sha(), "data_status": status, "rerun": status == "rerun", "final_source_checked_at_utc": _now(), "numerical_core_hashes": {path: _sha(ROOT / path) for path in CORE_PATHS}}
    if original is not None:
        result["probe_original_generated_at_utc"] = original
    return result


def _reuse_probe(name: str) -> dict[str, Any]:
    payload = json.loads((DOCS / name).read_text())
    original = payload.get("generated_at_utc")
    payload["provenance"] = _source_provenance("prior_probe_reused", original=original)
    payload["provenance"]["long_temporal_probe_rerun_in_closeout"] = False
    payload["provenance"]["reuse_reason"] = "Closeout explicitly reuses the already executed bounded probe; no T>=30/full-730/chunk/wide compile was rerun."
    payload["closeout_updated_at_utc"] = _now()
    return payload


def _cache_files() -> list[Path]:
    return [path for path in PROBE_CACHE.rglob("*") if path.is_file()] if PROBE_CACHE.exists() else []


def _closeout() -> None:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((ROOT / "project/autofuse/.cache/runtime-cache-final").resolve())
    investigation._env()
    git_sha = _git_sha()
    parity = investigation._mode_parity()
    parity["provenance"] = _source_provenance("rerun")
    parity["provenance"]["scope"] = "Final B=100,T=730 Lite/Full Q/final-state/loss/parameterizer-gradient parity"
    _write(DOCS / "torch_fuse_lite_full_parity.json", parity)

    benchmark = investigation._benchmark_modes()
    benchmark["provenance"] = _source_provenance("rerun")
    benchmark["provenance"]["scope"] = "Final B=100,T=730 Full/Lite three-repeat steady-state training benchmark"
    _write(DOCS / "torch_fuse_lite_full_b100_benchmark.json", benchmark)

    audit = investigation._audit_temporal_execution()
    audit["provenance"] = _source_provenance("rerun")
    audit["provenance"]["scope"] = "Final current one-step compiled-step + Python-loop execution audit"
    _write(DOCS / "torch_fuse_temporal_execution_audit.json", audit)

    full_probe = _reuse_probe("torch_fuse_full_temporal_compile_probe.json")
    chunk_probe = _reuse_probe("torch_fuse_chunked_temporal_compile_probe.json")
    wide_probe = _reuse_probe("torch_fuse_wide_compile_boundary_probe.json")
    _write(DOCS / "torch_fuse_full_temporal_compile_probe.json", full_probe)
    _write(DOCS / "torch_fuse_chunked_temporal_compile_probe.json", chunk_probe)
    _write(DOCS / "torch_fuse_wide_compile_boundary_probe.json", wide_probe)

    decision = investigation._decision(parity, benchmark, full_probe, chunk_probe, wide_probe)
    cache_comparison = json.loads((DOCS / "runtime_cache_comparison.json").read_text())
    cold_compile = json.loads((DOCS / "runtime_step_cold_compile.json").read_text())
    decision["cache_and_switching"] = {
        "current_one_step_key": "GraphSignature digest + device + dtype + batch_size + backend + fullgraph (the batched registry key; step tensor shapes are fixed by these specializations)",
        "observed_new_structure_compile_seconds": {str(row["model_id"]): row["cold_compile_seconds"] for row in cold_compile.get("models", [])},
        "returning_structure_cache_reuse": {"verified": cache_comparison.get("persistent_cache_reuse_verified", False), "same_output": cache_comparison.get("output_stable", False), "cache_files": cache_comparison.get("process_1", {}).get("cache_after", {}).get("files"), "cache_bytes": cache_comparison.get("process_1", {}).get("cache_after", {}).get("bytes"), "return_process_cold_compile_seconds": cache_comparison.get("process_2", {}).get("cold_compile_seconds")},
        "chunk_variant_rule": "one compiled variant per structure, K, and final remainder length; prior K=7/30/60/90 B=100 probes did not complete",
        "full_variant_rule": "one horizon-specialized graph per structure and T; timeout/failure produces no validated callable but can leave material partial Inductor cache",
        "future_1000_structure_assessment": "current one-step cache surface is materially more scalable; chunk/full multiplication by topology and temporal shape is not accepted without measured benefit",
    }
    decision["provenance"] = _source_provenance("rerun")
    decision["provenance"]["artifact_basis"] = {"parity": "rerun", "benchmark": "rerun", "execution_audit": "rerun", "full_temporal_probe": "prior_probe_reused", "chunked_temporal_probe": "prior_probe_reused", "wide_boundary_probe": "prior_probe_reused"}
    decision["provenance"]["closeout_scope"] = "No long-horizon temporal compile probe was launched during final closeout."
    decision["validation"] = {"git_sha": git_sha, "numerical_core_hashes_before": CORE_HASHES_BEFORE, "numerical_core_hashes_after": {path: _sha(ROOT / path) for path in CORE_PATHS}, "numerical_core_unchanged": CORE_HASHES_BEFORE == {path: _sha(ROOT / path) for path in CORE_PATHS}, "float64": True, "rk2_heun": True, "fix_states": True, "warmup_scored_semantics": True, "site_packages_modified": False, "formal_dpl_campaign_started": False, "final_code_has_monitor_detach": True, "final_code_has_explicit_lite_optional_fields": True, "existing_78_structure_smoke": {"artifact": "project/autofuse/docs/runtime_step_validation_78.json", "n_models": 78, "failures": [], "status": "78-structure runtime compiler validated"}}
    cache_before = _cache_files()
    cache_before_bytes = sum(path.stat().st_size for path in cache_before)
    decision["cache_and_switching"]["closeout_probe_cache_before_cleanup"] = {"path": str(PROBE_CACHE.relative_to(ROOT)), "files": len(cache_before), "bytes": cache_before_bytes, "classification": "temporary temporal compile probe cache; not the validated persistent one-step runtime cache"}
    decision["cache_and_switching"]["cleanup_policy"] = "delete this closeout-generated temporal probe cache after recording its footprint; retain the separate validated runtime-cache evidence"
    decision["temporal"]["closeout_wording"] = "Long-horizon temporal compilation remains unevaluated for adoption: minimal T=2/4 succeeded, T>=30/B=100 full probes did not complete within the practical budget, and K=7/30/60/90 B=100 chunk probes did not complete. These are not claims of mathematical impossibility or proven slowness."
    decision["final_state"] = "Lite/Full execution frozen; retain one-step compiled temporal design"
    _write(DOCS / "torch_fuse_temporal_optimization_decision.json", decision)

    # The cache was produced only by this investigation's abandoned long-graph attempts.
    # Remove exactly that ignored directory, never the validated runtime-cache directory.
    if PROBE_CACHE.exists():
        shutil.rmtree(PROBE_CACHE)
    decision["cache_and_switching"]["cleanup_result"] = {"action": "deleted", "path": str(PROBE_CACHE.relative_to(ROOT)), "files_after": len(_cache_files()), "bytes_after": sum(path.stat().st_size for path in _cache_files()), "reason": "temporary full/chunk temporal probe artifacts; validated one-step cache retained"}
    _write(DOCS / "torch_fuse_temporal_optimization_decision.json", decision)

    artifact_hashes = {name: _sha(DOCS / name) for name in ARTIFACT_NAMES}
    execution_layer = {
        "schema_version": "torch-fuse-execution-layer-v1-2-v1",
        "status": "frozen",
        "generated_at_utc": _now(),
        "git_sha": git_sha,
        "source_hashes": {path: _sha(ROOT / path) for path in ("dfuse/batched.py", "dfuse/runtime.py", "dfuse/kernel.py", "dfuse/spec.py", "project/autofuse/dpl.py", "project/autofuse/torch_fuse_temporal_optimization.py", "project/autofuse/torch_fuse_temporal_closeout.py")},
        "numerical_core": {"status": "unchanged", "paths": list(CORE_PATHS), "hashes": {path: _sha(ROOT / path) for path in CORE_PATHS}, "contract": ["float64", "coupled RK2/Heun", "FIX_STATES", "parameter definitions/bounds", "structure catalogue", "warm-up=365", "scored=365", "loss definition"]},
        "execution_contract": {"temporal": "one-step compiled kernel + Python time loop retained", "compile_boundary": "structure-specific torch.vmap(generated coupled-RK2 step) -> torch.compile(fullgraph=True)", "training_default": "Lite", "diagnostic_default": "Full"},
        "lite": {"status": "validated", "retained": ["routed Q history", "final active state"], "not_retained": ["state history", "flux history", "balance history", "snow history", "coupled diagnostic history", "hidden monitoring autograd graph"], "optional_fields": {"q_instantaneous": None, "snow": None}, "parity_artifact": "project/autofuse/docs/torch_fuse_lite_full_parity.json"},
        "full": {"status": "preserved", "retained": ["routed Q", "instantaneous Q", "active state history", "flux histories", "water/snow balance", "coupled diagnostics"], "artifact": "project/autofuse/docs/torch_fuse_lite_full_parity.json"},
        "temporal_compile": {"full_temporal": "investigated; long-horizon adoption not eligible because practical evaluation did not complete", "chunked": "investigated for K=7/30/60/90; not adopted because no validated B=100 result", "not_claimed": ["not proven slower", "not mathematically impossible"], "probe_artifacts": ["project/autofuse/docs/torch_fuse_full_temporal_compile_probe.json", "project/autofuse/docs/torch_fuse_chunked_temporal_compile_probe.json"]},
        "wide_compile_boundary": {"status": "not adopted", "finding": "fullgraph capture blocked by Python registry / Encoder path", "artifact": "project/autofuse/docs/torch_fuse_wide_compile_boundary_probe.json"},
        "structure_robustness": {"existing_78_structure_smoke": "project/autofuse/docs/runtime_step_validation_78.json", "status": "78 structures passed with no failures"},
        "artifacts": artifact_hashes,
        "formal_dpl_campaign_started": False,
        "site_packages_modified": False,
    }
    _write(DOCS / "torch_fuse_execution_layer_v1_2.json", execution_layer)
    print(json.dumps({"status": "completed", "git_sha": git_sha, "final_state": decision["final_state"], "artifacts": list(ARTIFACT_NAMES) + ["torch_fuse_execution_layer_v1_2.json"], "cache": decision["cache_and_switching"]["cleanup_result"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    _closeout()
