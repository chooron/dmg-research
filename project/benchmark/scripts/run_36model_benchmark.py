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
import re
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
from src.streaming_evaluator import compute_streaming_fitness
from src.production_config import load_resolved_config, validate_full_run_config


def get_population_size(dimension: int) -> int:
    return 8 if dimension <= 1 else 12 if dimension <= 6 else 16 if dimension <= 10 else 20


def compute_fitness(model, x, y, latent, target_offset: int, objective_fn):
    """Compatibility full-series reference path with explicit FP64 inputs."""
    model.to(dtype=torch.float64)
    model.compute_dtype = torch.float64
    x = x if x.dtype == torch.float64 else x.to(torch.float64)
    y = y if y.dtype == torch.float64 else y.to(torch.float64)
    latent = latent if latent.dtype == torch.float64 else latent.to(torch.float64)
    basin_count, starts, population, dimension = latent.shape
    raw = torch.sigmoid(latent).permute(0, 3, 1, 2).reshape(
        basin_count, dimension, starts * population
    ).to(torch.float64)
    with torch.inference_mode():
        q = model({"x_phy": x}, (None, raw))["streamflow"].reshape(-1, basin_count, starts, population)
    target = y[target_offset : target_offset + q.shape[0]]
    if target.shape[0] != q.shape[0]:
        raise RuntimeError(f"Target and output length mismatch: target={target.shape[0]}, output={q.shape[0]}")
    return objective_fn(q, target)

def compute_model_fitness(
    model_name: str,
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    latent: torch.Tensor,
    warmup_days: int,
    target_offset: int,
    objective_fn,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_streaming_fitness(
        model, x, y, latent, warmup_days=warmup_days
    )

def checkpoint_chunk_start(path: Path) -> int:
    match = re.match(r"chunk_(\d+)_gen_\d+\.pt$", path.name)
    if match is None:
        raise ValueError(f"checkpoint filename has no chunk start: {path}")
    return int(match.group(1))


def checkpoint_generation(path: Path) -> int:
    match = re.search(r"_gen_(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"checkpoint filename has no generation: {path}")
    return int(match.group(1))


def existing_checkpoint_chunk_size(checkpoint_root: Path) -> int | None:
    paths = list(checkpoint_root.glob("chunk_*_gen_*.pt"))
    if not paths:
        return None
    starts = sorted({checkpoint_chunk_start(path) for path in paths})
    if len(starts) > 1:
        return starts[1] - starts[0]
    checkpoint = load_checkpoint(min(paths, key=checkpoint_generation), "cpu")
    basin_ids = checkpoint.get("basin_ids")
    return len(basin_ids) if basin_ids is not None else None


def run_single_model(
    model_name: str,
    run_id: str,
    config_path: Path,
    chunk_size: int = 531,
    backend: str = "compile",
    device: str = "cuda",
) -> dict:
    """Train a single model using CMA-ES across all 531 basins."""
    if backend != "compile":
        raise ValueError("This DPL-aligned IC run requires backend=compile for every model.")
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

    objective_fn = torch.compile(streaming_kge, backend="inductor", mode="default")
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 12.0
    safety_gib = min(10.5, 0.88 * vram_gib)

    # Compile-only preflight on two basins using the same streaming family hooks.
    z0 = torch.zeros((2, 2, population, dimension), device=device, dtype=torch.float64)
    comp_m = build_model(
        model_name, device, warm_up=warmup_days, backend="compile", dtype=torch.float64
    )
    comp_fit, probe_invalid = compute_streaming_fitness(
        comp_m, x[:, :2], y[:, :2], z0, warmup_days=warmup_days
    )
    if bool(probe_invalid.any()) or not bool(torch.isfinite(comp_fit).all()):
        raise RuntimeError(f"Compile preflight produced invalid fitness for [{model_name}]")

    rows = []
    for left in range(0, len(ids), chunk_size):
        right = min(len(ids), left + chunk_size)
        basin_count = right - left

        model = build_model(
            model_name, device, warm_up=warmup_days, backend=backend, dtype=torch.float64
        )
        seed = int.from_bytes(hashlib.sha256(f"{global_seed}:{model_name}:{left}".encode()).digest()[:4], "little")

        solver = BatchedCMAES(basin_count * starts, dimension, population, stdev_init=0.10, active=True, seed=seed, device=device)
        solver.set_centers(torch.zeros((basin_count * starts, dimension), device=device, dtype=torch.float64))

        history, start_gen = [], 0
        saved = sorted(
            checkpoint_root.glob(f"chunk_{left}_gen_*.pt"),
            key=checkpoint_generation,
        )
        if saved:
            ckpt = load_checkpoint(saved[-1], device)
            checkpoint_basin_ids = ckpt.get("basin_ids")
            if checkpoint_basin_ids is not None and len(checkpoint_basin_ids) != basin_count:
                raise RuntimeError(
                    f"checkpoint {saved[-1]} has {len(checkpoint_basin_ids)} basins; "
                    f"requested chunk has {basin_count}"
                )
            expected_basin_ids = tuple(int(basin_id) for basin_id in ids[left:right])
            actual_basin_ids = tuple(int(basin_id) for basin_id in (checkpoint_basin_ids if checkpoint_basin_ids is not None else ()))
            if actual_basin_ids != expected_basin_ids:
                raise RuntimeError(
                    f"checkpoint {saved[-1]} basin IDs do not match requested chunk "
                    f"{expected_basin_ids[:3]}..."
                )
            start_gen = int(ckpt["generation"])
            solver.load_state_dict(ckpt["solver"])
            history = list(ckpt.get("history", []))
        if start_gen >= generations:
            if not history:
                raise RuntimeError(
                    f"{saved[-1]} is at generation {start_gen} but has no history"
                )
            # A final-generation checkpoint may exist without DONE after interruption.
            # Preserve its solver state and make resume idempotent without another step.
            rows.append({
                "seconds_per_generation": 0.0,
                "baseline_allocated_gib": 0.0,
                "peak_allocated_gib": 0.0,
                "incremental_peak_allocated_gib": 0.0,
                "baseline_reserved_gib": 0.0,
                "peak_reserved_gib": 0.0,
                "incremental_peak_reserved_gib": 0.0,
                "invalid_fraction": 0.0,
                "initial": history[0],
                "final": history[-1],
            })
            continue

        for generation in range(start_gen + 1, generations + 1):
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated = torch.cuda.memory_allocated()
            baseline_reserved = torch.cuda.memory_reserved()
            t0 = time.perf_counter()
            _z, _shape, latent = solver.ask()

            fitness, invalid = compute_model_fitness(
                model_name,
                model,
                x[:, left:right],
                y[:, left:right],
                latent.reshape(basin_count, starts, population, dimension),
                warmup_days,
                target_offset,
                objective_fn,
            )
            solver.tell(_z, _shape, latent, fitness.reshape(-1, population))
            torch.cuda.synchronize()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            elapsed = time.perf_counter() - t0

            history.append(float(fitness.median()))
            if peak_reserved / 2**30 > safety_gib:
                raise torch.OutOfMemoryError(f"Peak GPU memory reserved exceeds safety threshold {safety_gib:.2f} GiB")

            if generation % 5 == 0 or generation == generations:
                atomic_torch_save({
                    "model": model_name, "generation": generation, "solver": solver.state_dict(),
                    "basin_ids": ids[left:right], "resolved_config": resolved, "data_metadata": data_metadata,
                    "history": history, "rng": torch.get_rng_state(),
                }, checkpoint_root / f"chunk_{left}_gen_{generation}.pt")

        rows.append({
            "seconds_per_generation": elapsed,
            "baseline_allocated_gib": baseline_allocated / 2**30,
            "peak_allocated_gib": peak_allocated / 2**30,
            "incremental_peak_allocated_gib": max(0, peak_allocated - baseline_allocated) / 2**30,
            "baseline_reserved_gib": baseline_reserved / 2**30,
            "peak_reserved_gib": peak_reserved / 2**30,
            "incremental_peak_reserved_gib": max(0, peak_reserved - baseline_reserved) / 2**30,
            "invalid_fraction": float(invalid.double().mean()),
            "initial": history[0],
            "final": history[-1],
        })

    mean_sec = float(np.mean([r["seconds_per_generation"] for r in rows]))
    summary = {
        "model": model_name, "n_params": dimension, "population": population, "starts": starts,
        "generations_requested": generations, "generations_completed": generations,
        "backend": "full_batch" if chunk_size == len(ids) else "chunk",
        "compile_mode": "default",
        "basin_chunk": chunk_size, "validation_status": "passed", "compile_success": True,
        "seconds_per_generation": mean_sec,
        "candidates_per_second": (
            len(ids) * starts * population / mean_sec if mean_sec > 0 else 0.0
        ),
        "baseline_allocated_gib": max(r["baseline_allocated_gib"] for r in rows),
        "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
        "incremental_peak_allocated_gib": max(r["incremental_peak_allocated_gib"] for r in rows),
        "baseline_reserved_gib": max(r["baseline_reserved_gib"] for r in rows),
        "peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
        "incremental_peak_reserved_gib": max(r["incremental_peak_reserved_gib"] for r in rows),
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
    # Every attempt stays on torch.compile; eager fallback would violate the
    # benchmark's compile-only contract. OOM is handled by smaller chunks only.
    attempts = [
        (531, "compile"),
        (256, "compile"),
        (128, "compile"),
        (64,  "compile"),
    ]
    # Specialized strategy for unit hydrograph stack models if known to OOM
    if model_name in ["flexis", "gr4j"]:
        # Existing Flex checkpoints use 128-basin chunks; keep resume boundaries stable.
        attempts = [(128, "compile"), (64, "compile")]
    checkpoint_root = BENCHMARK_ROOT / "checkpoints" / run_id / model_name
    resume_chunk = existing_checkpoint_chunk_size(checkpoint_root)
    if resume_chunk is not None:
        attempts = [(resume_chunk, "compile")] + [
            attempt for attempt in attempts if attempt[0] != resume_chunk
        ]

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
    parser.add_argument("--backend", choices=["compile"], default="compile", help="Compile backend is mandatory for this run")
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
