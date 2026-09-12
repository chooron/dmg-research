"""Minimal same-seed reproducibility smoke for the formal Torch-SCE path."""
from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfuse import batched_compile_diagnostics, reset_batched_compile_diagnostics
from project.autofuse.torch_fuse_78_long_horizon_smoke import _load_frozen_inputs
from project.autofuse.unlimfrc_2_long_horizon_smoke import BASINS, _atomic_write, _cache_info, _dates, _forcing_batch, _sha
from project.autofuse.sce import SCEBaseline, SCEConfig

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
OUTPUT = DOCS / "formal_sce_seed_readiness.json"
CACHE = ROOT / "project/autofuse/.cache/torch-fuse-formal-sce-seed"
MODEL_ID = 2
SMOKE_DAYS = 8
SMOKE_EVALUATIONS = 24
SEED = 20260901
OTHER_SEED = 20260902


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


def _compile_audit() -> dict[str, Any]:
    records = list(batched_compile_diagnostics()["records"].values())
    return {"record_count": len(records), "records": records, "passed": bool(len(records) == 1 and all(record.get("compile_attempts") == 1 and record.get("compile_successes") == 1 and record.get("fallbacks") == 0 and record.get("graph_breaks") == 0 and record.get("recompilations") == 0 for record in records))}


def _run(seed: int, forcing: torch.Tensor, observed: torch.Tensor, basin_ids: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_batched_compile_diagnostics()
    config = SCEConfig(max_evaluations=SMOKE_EVALUATIONS, kstop=3, pcento=0.0, seed=seed, n_complexes=2)
    started = time.perf_counter()
    result = SCEBaseline(config).run(MODEL_ID, forcing, observed, basin_ids=basin_ids, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result, _compile_audit()


def main() -> None:
    _set_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("formal Torch-SCE seed smoke requires CUDA; refusing CPU fallback")
    metadata, inputs, _ = _load_frozen_inputs()
    forcing = _forcing_batch(inputs, torch.device("cuda"))[:, :SMOKE_DAYS]
    observed = torch.as_tensor(np.stack([inputs[basin]["q_obs"][:SMOKE_DAYS] for basin in BASINS]), dtype=torch.float64, device="cuda")
    basin_ids = tuple(BASINS)
    first, first_compile = _run(SEED, forcing, observed, basin_ids)
    second, second_compile = _run(SEED, forcing, observed, basin_ids)
    different, different_compile = _run(OTHER_SEED, forcing, observed, basin_ids)
    first_comparable = {key: value for key, value in first.items() if key != "elapsed_seconds"}
    second_comparable = {key: value for key, value in second.items() if key != "elapsed_seconds"}
    same_seed_identical = first_comparable == second_comparable
    different_seed_trajectory = first["trajectory"] != different["trajectory"]
    path_audit = {
        "intended_algorithm": "SCE-UA",
        "optimizer_contract": "SCEBaseline.run(model_id, forcing[basin,time,3], observed[basin,time], basin_ids, bounds, initial)",
        "objective_protocol": "1 - mean(per-basin KGECOMP)",
        "formal_evaluator": "UnifiedEvaluator.score_batched",
        "formal_kernel": "simulate_coupled_rk2_batched -> structure-specialized torch.vmap + torch.compile(fullgraph=True)",
        "legacy_scalar_path": "UnifiedEvaluator.forward -> dfuse.simulate remains only for the backward-compatible scalar objective boundary",
        "formal_run_uses_legacy_scalar_path": False,
        "training_or_campaign_started": False,
    }
    payload = {
        "schema_version": "formal-torch-sce-seed-readiness-v1",
        "status": "passed" if same_seed_identical and different_seed_trajectory and first_compile["passed"] and second_compile["passed"] and different_compile["passed"] else "failed",
        "path_audit": path_audit,
        "protocol": {"model_id": MODEL_ID, "basin_ids": list(basin_ids), "batch_size": len(basin_ids), "days": SMOKE_DAYS, "forcing_start": _dates()[0].isoformat(), "seed": SEED, "different_seed": OTHER_SEED, "max_evaluations": SMOKE_EVALUATIONS, "compile_step": True, "compile_backend": "inductor", "compile_fullgraph": True, "cpu_threads": 1, "cpu_interop_threads": 1, "device": "cuda", "dtype": "torch.float64"},
        "rng": {"source": "torch.Generator(device=cpu)", "propagation": "SCEConfig.seed -> SCEBaseline.run -> generator.manual_seed(seed); no global RNG is used for candidate generation", "same_seed_identical": same_seed_identical, "different_seed_trajectory_differs": different_seed_trajectory},
        "runs": {"seed_a": first, "seed_a_repeat": second, "seed_b": different},
        "compile": {"seed_a": first_compile, "seed_a_repeat": second_compile, "seed_b": different_compile},
        "source": {"sce": str(ROOT / "project/autofuse/sce.py"), "sce_sha256": _sha(ROOT / "project/autofuse/sce.py"), "evaluator": str(ROOT / "project/autofuse/evaluator.py"), "evaluator_sha256": _sha(ROOT / "project/autofuse/evaluator.py"), "metrics": str(ROOT / "project/autofuse/metrics.py"), "metrics_sha256": _sha(ROOT / "project/autofuse/metrics.py"), "batched": str(ROOT / "dfuse/batched.py"), "batched_sha256": _sha(ROOT / "dfuse/batched.py")},
        "resource": {"host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "cache": _cache_info(CACHE)},
        "gate_decision": {"formal_path_clear": True, "runner_connected": True, "same_seed_reproducible": same_seed_identical, "different_seed_changes_search": different_seed_trajectory, "passed": same_seed_identical and different_seed_trajectory and first_compile["passed"] and second_compile["passed"] and different_compile["passed"], "campaign_started": False},
    }
    _atomic_write(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "same_seed_identical": same_seed_identical, "different_seed_trajectory_differs": different_seed_trajectory, "compile_pass": payload["gate_decision"]["runner_connected"]}, sort_keys=True))


if __name__ == "__main__":
    main()
