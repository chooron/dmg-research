from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any
import torch

from .batched_cmaes import BatchedCMAES, lhs_latent, stable_hash
from .checkpointing import atomic_torch_save
from .convergence import cross_start_certified, plateau
from .model_adapter import BatchedModelAdapter


def git_revision(root: Path) -> str:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception: return "unknown"


class SolverCoordinator:
    """Batched model evaluation plus independent CMA-ES state management."""

    def __init__(self, adapter: BatchedModelAdapter, basin_ids: list[int], settings: dict[str, Any], *, starts: int, population: int, generations: int, seed: int) -> None:
        self.adapter, self.basin_ids, self.settings = adapter, list(map(int, basin_ids)), settings
        self.starts, self.population, self.generations, self.seed = starts, population, generations, seed
        self.basin_count, self.dimension = len(basin_ids), adapter.spec.dimension
        self.solver = BatchedCMAES(self.basin_count * starts, self.dimension, population,
                                   stdev_init=settings["stdev_init"], active=settings["active"], seed=seed, device=adapter.device)
        centers = torch.stack([lhs_latent(1, self.dimension, stable_hash(seed, adapter.spec.name, basin, start), adapter.device)[0]
                               for basin in self.basin_ids for start in range(starts)])
        self.solver.set_centers(centers)
        self.history: list[torch.Tensor] = []
        self.invalid_history: list[torch.Tensor] = []

    def checkpoint_payload(self) -> dict[str, Any]:
        return {"schema": 1, "model": self.adapter.spec.name, "basin_ids": self.basin_ids, "starts": self.starts,
                "population": self.population, "generations_target": self.generations, "solver": self.solver.state_dict(),
                "history": torch.stack(self.history).cpu() if self.history else torch.empty(0),
                "invalid_history": torch.stack(self.invalid_history).cpu() if self.invalid_history else torch.empty(0),
                "torch_rng": torch.get_rng_state().cpu(), "cuda_rng": [x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                "resolved_config": self.settings}

    def restore(self, payload: dict[str, Any]) -> None:
        self.solver.load_state_dict(payload["solver"])
        self.history = [x.to(self.adapter.device) for x in payload.get("history", [])]
        self.invalid_history = [x.to(self.adapter.device) for x in payload.get("invalid_history", [])]
        torch.set_rng_state(payload["torch_rng"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng"): torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng"]])

    def run(self, checkpoint_path: Path | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        while self.solver.state.generation < self.generations:
            z, y, x = self.solver.ask()
            latent = x.reshape(self.basin_count, self.starts, self.population, self.dimension)
            ev = self.adapter.evaluate(latent)
            fitness = ev.kge.reshape(-1, self.population)
            self.solver.tell(z, y, x, fitness)
            self.history.append(self.solver.state.best_fitness.reshape(self.basin_count, self.starts).amax(1).detach().clone())
            self.invalid_history.append(ev.invalid.reshape(self.basin_count, -1).to(torch.float64).mean(1))
            if checkpoint_path and self.solver.state.generation % int(self.settings["checkpoint_every_generations"]) == 0:
                atomic_torch_save(self.checkpoint_payload(), checkpoint_path)
        if checkpoint_path: atomic_torch_save(self.checkpoint_payload(), checkpoint_path)
        history = torch.stack(self.history)
        best_by_start = self.solver.state.best_fitness.reshape(self.basin_count, self.starts)
        convergence = self.settings["convergence"]
        return {"best_by_start": best_by_start, "best_latent": self.solver.state.best_latent.reshape(self.basin_count, self.starts, self.dimension),
                "plateau": plateau(history, improvement_window=convergence["improvement_window"], improvement_threshold=convergence["improvement_threshold"], range_window=convergence["objective_range_window"], range_threshold=convergence["objective_range_threshold"]), "cross_start": cross_start_certified(best_by_start, convergence["cross_start_gap"]),
                "invalid_fraction": torch.stack(self.invalid_history).mean(0), "history": history,
                "elapsed_seconds": time.perf_counter() - started, "generation": self.solver.state.generation}
