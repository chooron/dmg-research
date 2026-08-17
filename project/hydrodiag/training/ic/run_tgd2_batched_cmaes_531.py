#!/usr/bin/env python3
"""Calibrate 531-basin lite hydrological models with batched CMA-ES."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# The last, smaller basin chunk has a different batch/storage layout.  The
# compiled fullgraph kernels legitimately specialize for these layouts; keep
# Dynamo from aborting before the optimizer reaches its first generation.
import torch._dynamo as _dynamo

_dynamo.config.recompile_limit = max(_dynamo.config.recompile_limit, 256)
_dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 256)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from ablation.ic_core.data_adapter import load_531_bundle, manifest_for_bundle
from ablation.ic_core.parameter_adapter import get_parameter_spec, normalized_to_physical
from ablation.ic_core.runtime import ICObjectiveRuntime
from models.parameter_specs import TGD2_STRUCTURE_VERSION
from training.ic.batched_cmaes import BatchedCMAES


MODEL_DIMENSIONS = {"GR4J_TGD2": 6, "SIMHYD_TGD2": 12, "XAJ_TGD2": 17}
# Kept separate so the original TGD2 protocol contract remains explicit.
ADDITIONAL_MODEL_DIMENSIONS = {
    "N": 17, "D_E": 16, "G_E": 17, "D_R": 15, "G_R": 16,
    "XAJ": 15, "XAJ_CN": 17,
    "GR4J": 4,
    "GR4J_CN": 6,
    "SIMHYD": 10,
    "SIMHYD_CN": 12,
}
MODEL_STRUCTURE_VERSIONS = {
    "N": "phase0_xaj_controlled_n_cemaneige_v1",
    "D_E": "phase0_xaj_d_e_cemaneige_v1",
    "G_E": "phase0_xaj_g_e_cemaneige_v1",
    "D_R": "phase0_xaj_d_r_cemaneige_v1",
    "G_R": "phase0_xaj_g_r_cemaneige_v1",
    "XAJ": "xaj_base_v1",
    "XAJ_CN": "cemaneige_v1",
    "GR4J": "gr4j_base_v1",
    "GR4J_CN": "cemaneige_v1",
    "SIMHYD": "simhyd_base_v1",
    "SIMHYD_CN": "cemaneige_v1",
    "GR4J_TGD2": TGD2_STRUCTURE_VERSION,
    "XAJ_TGD2": TGD2_STRUCTURE_VERSION,
    "SIMHYD_TGD2": TGD2_STRUCTURE_VERSION,
}
DEFAULT_STARTS = 10
DEFAULT_GENERATIONS = 300
STOP = False


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def population_for_dimension(dimension: int) -> int:
    """Scale the XAJ_TGD2 population (25 at 17D), with a stable lower floor."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return max(12, round(25 * dimension / 17))


def supported_models() -> tuple[str, ...]:
    return tuple(MODEL_DIMENSIONS | ADDITIONAL_MODEL_DIMENSIONS)


def structure_version_for(model: str) -> str:
    return MODEL_STRUCTURE_VERSIONS[model]


def seed_for(model: str, basin_id: str, start: int) -> int:
    if model in {"N", "D_E", "G_E", "D_R", "G_R"}:
        protocol = "PHASE0-IC-CMAES-v1"
    else:
        protocol = "TGD2-CMAES-v1" if model in MODEL_DIMENSIONS else "ABLAT-CMAES-v1"
    value = f"{protocol}:{model}:{basin_id}:{start}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**31 - 1)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def handle_stop(_signal: int, _frame: object) -> None:
    global STOP
    STOP = True


class UnitRandom:
    def __init__(self, model: str, basin_ids: tuple[str, ...], starts: int, population: int,
                 dimension: int, states: list[dict] | None = None) -> None:
        self.population, self.dimension = population, dimension
        self.generators = [np.random.default_rng(seed_for(model, basin, start))
                           for basin in basin_ids for start in range(starts)]
        if states is not None:
            if len(states) != len(self.generators):
                raise ValueError("RNG checkpoint unit count mismatch")
            for generator, state in zip(self.generators, states):
                generator.bit_generator.state = state

    def sample(self) -> torch.Tensor:
        values = [item.standard_normal((self.population, self.dimension)) for item in self.generators]
        return torch.from_numpy(np.stack(values))

    def state_dict(self) -> list[dict]:
        return [item.bit_generator.state for item in self.generators]


def evaluate_all(runtime: ICObjectiveRuntime, candidates: torch.Tensor, basin_indices: list[int],
                 starts: int, chunk_basins: int, split: str) -> torch.Tensor:
    units, population, _dimension = candidates.shape
    if units != len(basin_indices) * starts:
        raise ValueError("candidate unit count mismatch")
    values = torch.empty((units, population), dtype=torch.float64, device=runtime.device)
    for left in range(0, len(basin_indices), chunk_basins):
        right = min(len(basin_indices), left + chunk_basins)
        count = right - left
        lo, hi = left * starts, right * starts
        theta = candidates[lo:hi].reshape(count, starts * population, -1)
        score, _diagnostics = runtime.evaluate_candidates_tensor(
            theta, basin_indices=basin_indices[left:right], split=split
        )
        values[lo:hi] = score.reshape(count, starts, population).reshape(-1, population)
    return values


def checkpoint(path: Path, *, model: str, solver: BatchedCMAES, random: UnitRandom,
               basin_ids: tuple[str, ...], starts: int, population: int, generations: int,
               chunk_basins: int, history: list[dict], started: float) -> None:
    atomic_torch(path, {
        "protocol": "batched_cmaes_phase0_or_531_v1",
        "structure_version": structure_version_for(model),
        "model": model,
        "basin_ids": basin_ids,
        "starts": starts,
        "population": population,
        "generations": generations,
        "chunk_basins": chunk_basins,
        "solver": solver.state_dict(),
        "rng_states": random.state_dict(),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "updated_at": utcnow(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=supported_models(), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS)
    parser.add_argument("--population", type=int)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--chunk-basins", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--basin-ids", help="Comma-separated canonical basin IDs for smoke runs.")
    parser.add_argument("--target-npz", type=Path,
                        help="Optional NPZ (key 'target_mm_day', [531, full-time]) replacing the observed "
                             "calibration target with a synthetic series (R3).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.starts, args.generations, args.chunk_basins, args.checkpoint_interval) < 1:
        raise ValueError("starts, generations, chunk-basins, and checkpoint-interval must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    specs = get_parameter_spec(args.model)
    dimension = len(specs)
    expected_dimension = (MODEL_DIMENSIONS | ADDITIONAL_MODEL_DIMENSIONS)[args.model]
    if dimension != expected_dimension:
        raise RuntimeError(f"{args.model} dimension changed unexpectedly: {dimension}")
    population = args.population or population_for_dimension(dimension)
    if population < 4:
        raise ValueError("population must be at least four")
    output = args.output or PROJECT / "results" / f"{args.model.lower()}_cmaes_531_batched_v1"

    config = json.loads((PROJECT / "ablation/configs/ic_foundation_531_v1.json").read_text())
    config.update({"device": str(device), "model_variant": "lite", "tgd_structure_version": TGD2_STRUCTURE_VERSION})
    config.setdefault("batching", {})["cache_device_data"] = True
    bundle = load_531_bundle(config)
    target_override = None
    if args.target_npz is not None:
        from dataclasses import replace as _replace

        saved = np.load(args.target_npz)
        if "target_mm_day" not in saved:
            raise ValueError("--target-npz must contain key 'target_mm_day'")
        synthetic = np.asarray(saved["target_mm_day"], dtype=np.float64)
        if synthetic.shape != bundle.target_mm_day.shape:
            raise ValueError(
                f"--target-npz shape {synthetic.shape} != bundle target {bundle.target_mm_day.shape}"
            )
        if not np.isfinite(synthetic).all() or (synthetic < 0).any():
            raise ValueError("--target-npz must be finite and non-negative")
        bundle = _replace(
            bundle,
            target_mm_day=synthetic,
            valid_target_mask=np.ones(synthetic.shape, dtype=bool),
            target_unit_ic="mm/day (synthetic Q*)",
        )
        target_override = {
            "path": str(args.target_npz),
            "shape": list(synthetic.shape),
            "unit": "mm/day (synthetic Q*)",
            "notes": "R3 synthetic-truth target; observed discharge untouched",
        }
        # Synthetic-truth protocol: fix CN forcing-derived quantities to the
        # canonical full-record values so the objective path reproduces the
        # generating truth (no split/window redefinition of g_thresh).
        config["canonical_cn_psol_annual"] = True
    basin_indices = list(range(len(bundle.basin_ids)))
    basin_ids = bundle.basin_ids
    if args.basin_ids:
        requested = tuple(item.strip().zfill(8) for item in args.basin_ids.split(",") if item.strip())
        index = {basin: position for position, basin in enumerate(bundle.basin_ids)}
        missing = sorted(set(requested) - set(index))
        if missing:
            raise ValueError(f"unknown basin IDs: {missing}")
        basin_indices = [index[basin] for basin in requested]
        basin_ids = requested

    runtime = ICObjectiveRuntime(bundle, config, args.model, model_variant="lite")
    units = len(basin_ids) * args.starts
    checkpoint_path = output / "checkpoints" / f"{args.model.lower()}_batched.pt"
    manifest = {
        "created_at": utcnow(),
        "dataset": manifest_for_bundle(bundle, config),
        "protocol": {
            "model": args.model,
            "structure_version": structure_version_for(args.model),
            "independent_starts": args.starts,
            "population_size": population,
            "population_rule": "max(12, round(25 * dimension / 17))",
            "max_generations": args.generations,
            "seed": "sha256(TGD2-CMAES-v1:model:basin_id:start) for TGD2; sha256(ABLAT-CMAES-v1:model:basin_id:start) otherwise",
            "objective": "KGE(Q), maximize, train only",
            "target_override": target_override,
            "canonical_cn_psol_annual": bool(config.get("canonical_cn_psol_annual", False)),
            "warmup": "1980-10-01..1981-09-30",
            "train": "1981-10-01..1995-09-30",
            "validation": "1995-10-01..2010-09-30",
            "model_variant": "lite",
        },
    }
    atomic_json(output / "manifest.json", manifest)
    solver = BatchedCMAES(units, dimension, population, device=device)
    history: list[dict] = []
    started = time.perf_counter()
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = (args.model, structure_version_for(args.model), basin_ids, args.starts, population, args.generations)
        actual = (saved.get("model"), saved.get("structure_version"), tuple(saved.get("basin_ids", ())),
                  saved.get("starts"), saved.get("population"), saved.get("generations"))
        if actual != expected:
            raise RuntimeError("checkpoint protocol or basin list mismatch")
        solver.load_state_dict(saved["solver"])
        random = UnitRandom(args.model, basin_ids, args.starts, population, dimension, saved["rng_states"])
        history = list(saved.get("history", []))
    else:
        centers = np.stack([np.random.default_rng(seed_for(args.model, basin, start)).uniform(.01, .99, dimension)
                            for basin in basin_ids for start in range(args.starts)])
        solver.set_centers(torch.from_numpy(centers).to(device))
        random = UnitRandom(args.model, basin_ids, args.starts, population, dimension)

    try:
        while solver.state.generation < args.generations:
            if STOP:
                checkpoint(checkpoint_path, model=args.model, solver=solver, random=random, basin_ids=basin_ids,
                           starts=args.starts, population=population, generations=args.generations,
                           chunk_basins=args.chunk_basins, history=history, started=started)
                return
            began = time.perf_counter()
            z, candidates = solver.ask(random.sample())
            fitness = evaluate_all(runtime, candidates, basin_indices, args.starts, args.chunk_basins, "train")
            solver.tell(z, candidates, fitness)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            history.append({
                "generation": solver.state.generation,
                "median_best_kge": float(solver.state.best_fitness.median()),
                "median_generation_kge": float(fitness.max(dim=1).values.median()),
                "seconds": time.perf_counter() - began,
                "candidate_evaluations": units * population,
            })
            if solver.state.generation % args.checkpoint_interval == 0 or solver.state.generation == args.generations:
                checkpoint(checkpoint_path, model=args.model, solver=solver, random=random, basin_ids=basin_ids,
                           starts=args.starts, population=population, generations=args.generations,
                           chunk_basins=args.chunk_basins, history=history, started=started)
                print(json.dumps(history[-1]), flush=True)
    except BaseException:
        checkpoint(checkpoint_path, model=args.model, solver=solver, random=random, basin_ids=basin_ids,
                   starts=args.starts, population=population, generations=args.generations,
                   chunk_basins=args.chunk_basins, history=history, started=started)
        raise

    best = solver.state.best_candidate.detach()
    selected = best[:, None, :].expand(-1, population, -1)
    train = evaluate_all(runtime, selected, basin_indices, args.starts, args.chunk_basins, "train")[:, 0]
    validation = evaluate_all(runtime, selected, basin_indices, args.starts, args.chunk_basins, "test")[:, 0]
    physical = normalized_to_physical(args.model, best, clip=True).detach().cpu().numpy()
    best_values = solver.state.best_fitness.detach().cpu().numpy()
    for basin_position, basin in enumerate(basin_ids):
        for start in range(args.starts):
            unit = basin_position * args.starts + start
            record = {
                "status": "complete", "model": args.model, "structure_version": structure_version_for(args.model),
                "basin_id": basin, "start": start, "seed": seed_for(args.model, basin, start),
                "parameter_names": list(specs), "theta_normalized": best[unit].detach().cpu().tolist(),
                "parameters": physical[unit].tolist(), "objective_definition": "KGE(Q), maximize, train only",
                "train_metrics": {"kge": float(train[unit])}, "test_metrics": {"kge": float(validation[unit])},
                "best_train_objective": float(best_values[unit]), "generations": args.generations,
                "population": population, "candidate_evaluations": args.generations * population,
                "checkpoint": str(checkpoint_path), "completed_at": utcnow(),
            }
            atomic_json(output / "raw" / args.model.lower() / f"{basin}_start{start:02d}.json", record)
    atomic_torch(output / "summaries" / "per_start_train_best_history.pt", {
        "basin_ids": basin_ids, "starts": args.starts, "history": history,
    })
    atomic_json(output / "DONE.json", {
        "status": "complete", "records": units, "candidate_evaluations": units * population * args.generations,
        "elapsed_seconds": time.perf_counter() - started, "chunk_basins": args.chunk_basins,
    })


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    main()
