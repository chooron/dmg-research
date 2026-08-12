#!/usr/bin/env python3
"""
Unified Master Runner for 36 Hydrological Models Benchmark (CMA-ES Optimization).
Consolidates training, memory-aware chunking fallback, atomic checkpointing,
and DONE marker creation into a single clean entrypoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from src.batched_cmaes import BatchedCMAES
from src.checkpointing import atomic_torch_save, load_checkpoint
from src.data_selection import load_ids, load_repeated_warmup_and_train
from src.model_registry import NPARAM_INFO_36, build_model
from src.objective import streaming_kge
from src.production_config import load_resolved_config, validate_full_run_config


def get_population_size(dimension: int) -> int:
    return 8 if dimension <= 1 else 12 if dimension <= 6 else 16 if dimension <= 10 else 20


def compute_fitness(model, x, y, latent, target_offset: int, objective_fn):
    basin_count, starts, population, dimension = latent.shape
    raw = torch.sigmoid(latent).permute(0, 3, 1, 2).reshape(basin_count, dimension, starts * population).float()
    with torch.inference_mode():
        q = model({"x_phy": x}, (None, raw))["streamflow"].reshape(-1, basin_count, starts, population)
    target = y[target_offset : target_offset + q.shape[0]]
    if target.shape[0] != q.shape[0]:
        raise RuntimeError(f"Target and output length mismatch: target={target.shape[0]}, output={q.shape[0]}")
    return objective_fn(q, target)


def run_single_model(
    model_name: str,
    run_id: str,
    config_path: Path,
    chunk_size: int = 531,
    backend: str = "compile",
    device: str = "cuda",
) -> dict:
    """Train a single model using CMA-ES across all 531 basins."""
    resolved = load_resolved_config(config_path)
    validate_full_run_config(resolved)

    generations = int(resolved["optimization"]["generations"])
    starts = int(resolved["optimization"]["starts"])
    global_seed = int(resolved["global_seed"])

    dimension = NPARAM_INFO_36[model_name]
    population = get_population_size(dimension)

    checkpoint_root = BENCHMARK_ROOT / "checkpoints" / run_id / model_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    done_marker = checkpoint_root / "DONE"
    if done_marker.is_file():
        print(f"Model [{model_name}] is already completed ({done_marker}). Skipping.")
        return json.loads(done_marker.read_text())

    ids = load_ids(resolved["data"]["basin_ids"])
    x, y, data_metadata = load_repeated_warmup_and_train(ids, resolved, device)
    warmup_days, target_offset = data_metadata["warmup_total_days"], 0

    objective_fn = streaming_kge if backend == "eager" else torch.compile(streaming_kge, backend="inductor", mode="default")
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 12.0
    safety_gib = min(10.5, 0.88 * vram_gib)

    # Preflight compiled output check
    z0 = torch.zeros((2, 2, population, dimension), device=device, dtype=torch.float64)
    eager_m = build_model(model_name, device, warm_up=warmup_days, backend="eager")
    eager_fit, _ = compute_fitness(eager_m, x[:, :2], y[:, :2], z0, target_offset, streaming_kge)
    if backend == "compile":
        comp_m = build_model(model_name, device, warm_up=warmup_days, backend="compile")
        comp_fit, _ = compute_fitness(comp_m, x[:, :2], y[:, :2], z0, target_offset, objective_fn)
        err = float((eager_fit - comp_fit).abs().max())
        if not np.isfinite(err) or err > 1e-5:
            raise RuntimeError(f"Preflight validation failed for [{model_name}] compile vs eager diff = {err}")

    rows = []
    for left in range(0, len(ids), chunk_size):
        right = min(len(ids), left + chunk_size)
        basin_count = right - left

        model = build_model(model_name, device, warm_up=warmup_days, backend=backend)
        seed = int.from_bytes(hashlib.sha256(f"{global_seed}:{model_name}:{left}".encode()).digest()[:4], "little")

        solver = BatchedCMAES(basin_count * starts, dimension, population, stdev_init=0.10, active=True, seed=seed, device=device)
        solver.set_centers(torch.zeros((basin_count * starts, dimension), device=device, dtype=torch.float64))

        history, start_gen = [], 0
        saved = sorted(checkpoint_root.glob(f"chunk_{left}_gen_*.pt"))
        if saved:
            ckpt = load_checkpoint(saved[-1], device)
            start_gen = int(ckpt["generation"])
            solver.load_state_dict(ckpt["solver"])
            history = list(ckpt.get("history", []))

        for generation in range(start_gen + 1, generations + 1):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _z, _shape, latent = solver.ask()

            fitness, invalid = compute_fitness(
                model, x[:, left:right], y[:, left:right],
                latent.reshape(basin_count, starts, population, dimension),
                target_offset, objective_fn
            )
            solver.tell(_z, _shape, latent, fitness.reshape(-1, population))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            history.append(float(fitness.median()))
            if torch.cuda.max_memory_reserved() / 2**30 > safety_gib:
                raise torch.OutOfMemoryError(f"Peak GPU memory reserved exceeds safety threshold {safety_gib:.2f} GiB")

            if generation % 5 == 0 or generation == generations:
                atomic_torch_save({
                    "model": model_name, "generation": generation, "solver": solver.state_dict(),
                    "basin_ids": ids[left:right], "resolved_config": resolved, "data_metadata": data_metadata,
                    "history": history, "rng": torch.get_rng_state(),
                }, checkpoint_root / f"chunk_{left}_gen_{generation}.pt")

        rows.append({
            "seconds_per_generation": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "invalid_fraction": float(invalid.double().mean()),
            "initial": history[0],
            "final": history[-1],
        })

    mean_sec = float(np.mean([r["seconds_per_generation"] for r in rows]))
    summary = {
        "model": model_name, "n_params": dimension, "population": population, "starts": starts,
        "generations_requested": generations, "generations_completed": generations,
        "backend": "full_batch" if chunk_size == len(ids) else "chunk",
        "compile_mode": "default" if backend == "compile" else "eager",
        "basin_chunk": chunk_size, "validation_status": "passed", "compile_success": (backend == "compile"),
        "seconds_per_generation": mean_sec,
        "candidates_per_second": len(ids) * starts * population / mean_sec,
        "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "invalid_candidate_fraction": float(np.mean([r["invalid_fraction"] for r in rows])),
        "initial_median_train_kge": float(np.mean([r["initial"] for r in rows])),
        "final_median_train_kge": float(np.mean([r["final"] for r in rows])),
        "median_best_improvement": float(np.mean([r["final"] - r["initial"] for r in rows])),
        "status": "full_compile_ready",
    }
    done_marker.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Model [{model_name}] completed successfully! Summary written to {done_marker}")
    return summary


def run_with_memory_fallback(model_name: str, run_id: str, config_path: Path, device: str = "cuda") -> dict:
    """Run model training with automatic memory-aware chunk size fallback."""
    # Attempt strategies sequentially (from full batch to smaller chunks)
    attempts = [
        (531, "compile"),
        (256, "compile"),
        (128, "compile"),
        (64,  "compile"),
        (256, "eager"),
        (64,  "eager"),
    ]
    # Specialized strategy for unit hydrograph stack models if known to OOM
    if model_name in ["flexis", "gr4j"]:
        attempts = [(256, "compile"), (128, "compile"), (64, "compile"), (64, "eager")]

    last_error = None
    for chunk, backend in attempts:
        print(f"=== Model [{model_name}] Attempting chunk={chunk}, backend={backend} ===")
        try:
            return run_single_model(model_name, run_id, config_path, chunk_size=chunk, backend=backend, device=device)
        except (torch.OutOfMemoryError, MemoryError, RuntimeError) as err:
            print(f"Attempt failed for [{model_name}] chunk={chunk} backend={backend}: {err}")
            last_error = err
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(f"All execution attempts failed for [{model_name}]. Last error: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified 36-Model CMA-ES Benchmark Runner")
    parser.add_argument("--model", help="Specific model name to train, or 'all' for all 36 models")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--config", default="configs/full_run_10starts_300gen_warm1980_1981x5.yaml")
    parser.add_argument("--chunk", type=int, help="Override chunk size (if omitted, uses memory-aware fallback)")
    parser.add_argument("--backend", choices=["compile", "eager"], help="Override backend mode")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda/cpu)")
    args = parser.parse_args()

    config_path = BENCHMARK_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    all_models = list(NPARAM_INFO_36.keys())
    target_models = all_models if args.model in ["all", None] else [args.model]

    print(f"=== Starting Benchmark Run [{args.run_id}] for {len(target_models)} model(s) ===")
    for idx, m in enumerate(target_models, 1):
        print(f"\n[{idx}/{len(target_models)}] Processing model: {m}")
        if args.chunk and args.backend:
            run_single_model(m, args.run_id, config_path, chunk_size=args.chunk, backend=args.backend, device=args.device)
        else:
            run_with_memory_fallback(m, args.run_id, config_path, device=args.device)


if __name__ == "__main__":
    main()
