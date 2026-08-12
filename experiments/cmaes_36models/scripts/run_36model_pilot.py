#!/usr/bin/env python3
"""One isolated CMA-ES model process; supports legacy pilot and frozen production configs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/cmaes_36models")]

from src.batched_cmaes import BatchedCMAES
from src.checkpointing import atomic_torch_save, load_checkpoint
from src.data_selection import load_ids, load_repeated_warmup_and_train
from src.model_registry import NPARAM_INFO_36, build_model
from src.objective import streaming_kge
from src.production_config import load_resolved_config, validate_full_run_config


def pop(d: int) -> int:
    return 8 if d <= 1 else 12 if d <= 6 else 16 if d <= 10 else 20


def legacy_data(device: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int, dict[str, int]]:
    """The historical pilot data path, retained only for old pilot reproducibility."""
    with open(ROOT / "data/camels_dataset", "rb") as handle:
        forcings, target, attributes = pickle.load(handle)
    ids = np.asarray(ast.literal_eval((ROOT / "data/531sub_id.txt").read_text()), dtype=np.int64)
    reference = np.load(ROOT / "data/gage_id.npy")
    index = np.array([np.where(reference == basin)[0][0] for basin in ids])
    dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
    left, right = dates.get_loc("1989-01-01"), dates.get_loc("1998-12-31") + 1
    x = forcings[index, left:right, :3]
    doy = dates[left:right].dayofyear.to_numpy()
    x = np.concatenate((x, np.broadcast_to(doy[None, :, None], (len(index), len(doy), 1))), axis=2)
    y = target[index, left:right, 0].copy() * (0.0283168 * 86400 * 1e3 / (attributes[index, 11] * 1e6))[:, None]
    return (
        torch.as_tensor(x.transpose(1, 0, 2), device=device, dtype=torch.float32),
        torch.as_tensor(y.T, device=device, dtype=torch.float32),
        ids,
        365,
        {"warmup_total_days": 365, "train_days": right - left, "input_days": right - left},
    )


def score(model, x, y, latent, target_offset: int, fn):
    basin_count, starts, population, dimension = latent.shape
    raw = torch.sigmoid(latent).permute(0, 3, 1, 2).reshape(basin_count, dimension, starts * population).float()
    with torch.inference_mode():
        q = model({"x_phy": x}, (None, raw))["streamflow"].reshape(-1, basin_count, starts, population)
    target = y[target_offset : target_offset + q.shape[0]]
    if target.shape[0] != q.shape[0]:
        raise RuntimeError(f"target/output length mismatch: target={target.shape[0]} output={q.shape[0]}")
    return fn(q, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--chunk", type=int, default=531)
    parser.add_argument("--backend", default="compile")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--starts", type=int)
    parser.add_argument("--config", help="Resolved production YAML; omitted only for legacy pilot replay")
    args = parser.parse_args()

    resolved = load_resolved_config(args.config) if args.config else None
    if resolved:
        validate_full_run_config(resolved)
        generations = args.generations if args.generations is not None else int(resolved["optimization"]["generations"])
        starts = args.starts if args.starts is not None else int(resolved["optimization"]["starts"])
        if generations != int(resolved["optimization"]["generations"]) or starts != int(resolved["optimization"]["starts"]):
            raise ValueError("frozen production settings cannot be overridden from the command line")
    else:
        generations, starts = args.generations or 30, args.starts or 5

    device = "cuda"
    dimension, population = NPARAM_INFO_36[args.model], pop(NPARAM_INFO_36[args.model])
    base = ROOT / "experiments/cmaes_36models"
    checkpoint_root = base / "checkpoints" / args.run_id / args.model
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if resolved:
        ids = load_ids(resolved["data"]["basin_ids"])
        x, y, data_metadata = load_repeated_warmup_and_train(ids, resolved, device)
        warmup_days, target_offset = data_metadata["warmup_total_days"], 0
        global_seed = int(resolved["global_seed"])
    else:
        x, y, ids, warmup_days, data_metadata = legacy_data(device)
        target_offset, global_seed = warmup_days, 20260729

    objective_fn = streaming_kge if args.backend == "eager" else torch.compile(streaming_kge, backend="inductor", mode="default")
    safety = min(10.5, 0.88 * torch.cuda.get_device_properties(0).total_memory / 2**30)

    # The numerical core is the only compiled portion. This preflight uses a
    # static two-basin shape and compares eager with compiled train KGE.
    z0 = torch.zeros((2, 2, population, dimension), device=device, dtype=torch.float64)
    eager = build_model(args.model, device, warm_up=warmup_days, backend="eager")
    eager_fitness, _ = score(eager, x[:, :2], y[:, :2], z0, target_offset, streaming_kge)
    if args.backend == "compile":
        compiled = build_model(args.model, device, warm_up=warmup_days, backend="compile")
        compiled_fitness, _ = score(compiled, x[:, :2], y[:, :2], z0, target_offset, objective_fn)
        error = float((eager_fitness - compiled_fitness).abs().max())
        if not np.isfinite(error) or error > 1e-5:
            raise RuntimeError(f"validation_failed objective difference {error}")

    rows = []
    for left in range(0, len(ids), args.chunk):
        right, basin_count = min(len(ids), left + args.chunk), min(len(ids), left + args.chunk) - left
        model = build_model(args.model, device, warm_up=warmup_days, backend=args.backend)
        seed = int.from_bytes(hashlib.sha256(f"{global_seed}:{args.model}:{left}".encode()).digest()[:4], "little")
        solver = BatchedCMAES(basin_count * starts, dimension, population, stdev_init=0.10, active=True, seed=seed, device=device)
        solver.set_centers(torch.zeros((basin_count * starts, dimension), device=device, dtype=torch.float64))
        history, start_generation = [], 0
        saved = sorted(checkpoint_root.glob(f"chunk_{left}_gen_*.pt"))
        if saved:
            checkpoint = load_checkpoint(saved[-1], device)
            start_generation = int(checkpoint["generation"])
            solver.load_state_dict(checkpoint["solver"])
            history = list(checkpoint.get("history", []))
        for generation in range(start_generation + 1, generations + 1):
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            _z, _shape, latent = solver.ask()
            fitness, invalid = score(model, x[:, left:right], y[:, left:right], latent.reshape(basin_count, starts, population, dimension), target_offset, objective_fn)
            solver.tell(_z, _shape, latent, fitness.reshape(-1, population))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            history.append(float(fitness.median()))
            if torch.cuda.max_memory_reserved() / 2**30 > safety:
                raise MemoryError(f"peak reserved exceeds safety limit {safety:.2f} GiB")
            if generation % 5 == 0:
                atomic_torch_save({
                    "model": args.model, "generation": generation, "solver": solver.state_dict(),
                    "basin_ids": ids[left:right], "resolved_config": resolved, "data_metadata": data_metadata,
                    "history": history, "rng": torch.get_rng_state(),
                }, checkpoint_root / f"chunk_{left}_gen_{generation}.pt")
        rows.append({"seconds_per_generation": elapsed, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "invalid_fraction": float(invalid.double().mean()), "initial": history[0], "final": history[-1]})

    mean_seconds = float(np.mean([row["seconds_per_generation"] for row in rows]))
    result = {
        "model": args.model, "n_params": dimension, "population": population, "starts": starts,
        "generations_requested": generations, "generations_completed": generations,
        "backend": "full_batch" if args.chunk == len(ids) else "chunk", "compile_mode": "default" if args.backend == "compile" else "eager",
        "basin_chunk": args.chunk, "validation_status": "passed", "compile_success": args.backend == "compile",
        "seconds_per_generation": mean_seconds, "candidates_per_second": len(ids) * starts * population / mean_seconds,
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in rows), "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "invalid_candidate_fraction": float(np.mean([row["invalid_fraction"] for row in rows])),
        "initial_median_train_kge": float(np.mean([row["initial"] for row in rows])),
        "final_median_train_kge": float(np.mean([row["final"] for row in rows])),
        "median_best_improvement": float(np.mean([row["final"] - row["initial"] for row in rows])),
        "resolved_config": str(resolved.get("_resolved_from")) if resolved else None, "data_metadata": data_metadata,
        "status": "full_compile_ready",
    }
    (checkpoint_root / "DONE").write_text(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "runtime_failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}), flush=True)
        raise
