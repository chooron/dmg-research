"""Minimal formal SCE-UA runner over the frozen Torch-FUSE batched kernel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from dfuse import PARAMETER_NAMES, get_structure
from dfuse.spec import default_parameters, parameter_metadata

from .evaluator import EvaluationOutput, UnifiedEvaluator
from .protocol import ExperimentProtocol


@dataclass(frozen=True)
class SCEConfig:
    max_evaluations: int = 10_000
    kstop: int = 3
    pcento: float = 0.001
    seed: int = 20260901
    n_complexes: int = 2

    @classmethod
    def from_protocol(cls, protocol: ExperimentProtocol) -> "SCEConfig":
        return cls(protocol.sce_max_evaluations, protocol.sce_kstop, protocol.sce_pcento, protocol.default_seed)


class SCEBaseline:
    """SCE-UA implementation using the shared formal Torch evaluator.

    The optimizer is intentionally small and bounded for readiness smokes.  It
    follows the SCE-UA complex/simplex update contract, while all hydrological
    evaluation is delegated to ``UnifiedEvaluator.score_batched``.
    """

    def __init__(self, config: SCEConfig | None = None, evaluator: UnifiedEvaluator | None = None):
        self.config = config or SCEConfig()
        self.evaluator = evaluator or UnifiedEvaluator()

    def objective(
        self,
        model_id: int,
        forcing: Tensor,
        observed: Tensor,
        params: Mapping[str, object] | Tensor,
    ) -> Tensor:
        """Return scalar ``1-KGEcomp`` from the legacy scalar evaluator boundary."""
        scored: EvaluationOutput = self.evaluator.score(model_id, forcing, observed, params)
        return 1.0 - scored.kge_comp

    def objective_batched(
        self,
        model_id: int,
        forcing: Tensor,
        observed: Tensor,
        params: Mapping[str, object] | Tensor | Sequence[Mapping[str, object]],
        *,
        basin_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Return the mean basin objective through the formal batched path."""
        scored = self.evaluator.score_batched(model_id, forcing, observed, params, basin_ids=basin_ids, **kwargs)
        return 1.0 - scored.kge_comp.mean()

    def plan(self, model_ids: list[int] | None = None) -> dict[str, object]:
        return {
            "method": "SCE-UA",
            "model_ids": model_ids,
            "max_evaluations": self.config.max_evaluations,
            "kstop": self.config.kstop,
            "pcento": self.config.pcento,
            "seed": self.config.seed,
            "n_complexes": self.config.n_complexes,
            "training_started": False,
            "shared_forward": "simulate_coupled_rk2_batched",
            "evaluator": "UnifiedEvaluator.score_batched",
            "kernel": "coupled-RK2/FIX_STATES Torch runtime",
        }

    @staticmethod
    def _bounds(model_id: int, bounds: Mapping[str, Sequence[float]] | None) -> tuple[tuple[str, ...], Tensor, Tensor]:
        names = tuple(get_structure(model_id).parameter_names)
        lower = []
        upper = []
        for name in names:
            if bounds is not None and name in bounds:
                values = tuple(float(value) for value in bounds[name])
                if len(values) != 2 or values[0] >= values[1]:
                    raise ValueError(f"invalid bounds for parameter {name}")
                low, high = values
            else:
                metadata = parameter_metadata(name)
                low, high = float(metadata["lower"]), float(metadata["upper"])
            lower.append(low)
            upper.append(high)
        return names, torch.tensor(lower, dtype=torch.float64), torch.tensor(upper, dtype=torch.float64)

    @staticmethod
    def _initial_candidate(model_id: int, names: tuple[str, ...], lower: Tensor, upper: Tensor, initial: Mapping[str, object] | Tensor | None) -> Tensor:
        defaults = default_parameters()
        if initial is None:
            values = [float(defaults[name]) for name in names]
        elif isinstance(initial, Tensor):
            flat = initial.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            if flat.numel() == len(names):
                values = flat.tolist()
            elif flat.numel() == len(PARAMETER_NAMES):
                positions = [PARAMETER_NAMES.index(name) for name in names]
                values = flat[positions].tolist()
            else:
                raise ValueError(f"initial must have {len(names)} active or {len(PARAMETER_NAMES)} union parameters")
        else:
            values = [float(initial.get(name, defaults[name])) for name in names]
        candidate = torch.tensor(values, dtype=torch.float64)
        return torch.maximum(torch.minimum(candidate, upper), lower)

    def run(
        self,
        model_id: int,
        forcing: Tensor,
        observed: Tensor,
        *,
        basin_ids: Sequence[str] | None = None,
        bounds: Mapping[str, Sequence[float]] | None = None,
        initial: Mapping[str, object] | Tensor | None = None,
        inverse_epsilon: float | Tensor | None = None,
        compile_step: bool = True,
        compile_backend: str = "inductor",
        compile_fullgraph: bool = True,
        candidate_batch_size: int = 1,
    ) -> dict[str, object]:
        """Run a bounded SCE-UA search; no campaign is launched implicitly."""
        if forcing.ndim != 3 or observed.ndim != 2 or forcing.shape[:2] != observed.shape:
            raise ValueError("formal SCE requires forcing [basin, time, 3] and observed [basin, time]")
        if forcing.shape[0] < 1:
            raise ValueError("formal SCE requires at least one basin")
        if self.config.max_evaluations < 2:
            raise ValueError("SCE max_evaluations must be at least 2")
        if self.config.n_complexes < 1:
            raise ValueError("SCE n_complexes must be positive")
        names, lower, upper = self._bounds(model_id, bounds)
        initial_candidate = self._initial_candidate(model_id, names, lower, upper, initial)
        n_parameters = len(names)
        population_size = min(self.config.max_evaluations, 2 * n_parameters + 1)
        if population_size < 2:
            raise ValueError("SCE population requires at least two evaluations")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.seed))
        population = lower + (upper - lower) * torch.rand((population_size, n_parameters), generator=generator, dtype=torch.float64)
        population[0] = initial_candidate
        defaults = default_parameters()
        default_vector = torch.tensor([float(defaults[name]) for name in PARAMETER_NAMES], dtype=torch.float64)
        active_positions = [PARAMETER_NAMES.index(name) for name in names]
        basin_tuple = tuple(str(value) for value in basin_ids) if basin_ids is not None else tuple(str(index) for index in range(int(forcing.shape[0])))
        trajectory: list[dict[str, object]] = []
        scores: list[float] = []

        def evaluate_batch(candidates: Tensor, operation: str) -> list[float]:
            c_count = int(candidates.shape[0])
            full = default_vector.unsqueeze(0).expand(c_count, -1).clone()
            full[:, active_positions] = candidates
            device_matrix = full.to(device=forcing.device, dtype=forcing.dtype)
            with torch.no_grad():
                if int(forcing.shape[0]) == 1 and c_count > 1:
                    batched_forcing = forcing.expand(c_count, -1, -1)
                    batched_observed = observed.expand(c_count, -1)
                    batched_basin_tuple = tuple(f"{basin_tuple[0]}_c{i}" for i in range(c_count))
                    scored = self.evaluator.score_batched(
                        model_id,
                        batched_forcing,
                        batched_observed,
                        device_matrix,
                        basin_ids=batched_basin_tuple,
                        inverse_epsilon=inverse_epsilon,
                        compile_step=compile_step,
                        compile_backend=compile_backend,
                        compile_fullgraph=compile_fullgraph,
                        monitor_chunk_size=max(1, min(64, int(forcing.shape[1]))),
                        output_mode="q_only",
                    )
                    scores_tensor = 1.0 - scored.kge_comp
                    scores_list = [float(s) if torch.isfinite(s) else float("inf") for s in scores_tensor.detach().cpu()]
                else:
                    scores_list = []
                    for i in range(c_count):
                        single_dev = device_matrix[i:i+1]
                        scored = self.evaluator.score_batched(
                            model_id,
                            forcing,
                            observed,
                            single_dev,
                            basin_ids=basin_tuple,
                            inverse_epsilon=inverse_epsilon,
                            compile_step=compile_step,
                            compile_backend=compile_backend,
                            compile_fullgraph=compile_fullgraph,
                            monitor_chunk_size=max(1, min(64, int(forcing.shape[1]))),
                            output_mode="q_only",
                        )
                        val = float(1.0 - scored.kge_comp.mean()) if torch.isfinite(scored.kge_comp).all() else float("inf")
                        scores_list.append(val)
            for i in range(c_count):
                trajectory.append({"evaluation": len(trajectory) + 1, "operation": operation, "score": scores_list[i], "active_parameters": candidates[i].detach().cpu().tolist()})
            return scores_list

        def evaluate(candidate: Tensor, operation: str) -> float:
            return evaluate_batch(candidate.unsqueeze(0), operation)[0]

        if candidate_batch_size > 1 and population_size > 1:
            for start in range(0, population_size, candidate_batch_size):
                chunk = population[start : min(start + candidate_batch_size, population_size)]
                scores.extend(evaluate_batch(chunk, "initial_population"))
        else:
            for index in range(population_size):
                scores.append(evaluate(population[index], "initial_population"))
        stop_reason = "max_evaluations"
        best_history: list[float] = []
        while len(trajectory) < self.config.max_evaluations:
            ordered = sorted(range(population_size), key=lambda index: scores[index])
            current_best = scores[ordered[0]]
            best_history.append(current_best)
            if len(best_history) >= self.config.kstop:
                previous = best_history[-self.config.kstop]
                relative_change = abs(previous - current_best) / max(abs(previous), 1.0e-12)
                if relative_change <= self.config.pcento:
                    stop_reason = "kstop_pcento"
                    break
            for complex_index in range(self.config.n_complexes):
                members = ordered[complex_index:: self.config.n_complexes]
                if len(members) < 2:
                    continue
                weights = torch.arange(len(members), 0, -1, dtype=torch.float64)
                sample_count = min(n_parameters + 1, len(members))
                simplex_positions = torch.multinomial(weights / weights.sum(), sample_count, replacement=False, generator=generator).tolist()
                simplex = [members[position] for position in simplex_positions]
                simplex.sort(key=lambda index: scores[index])
                worst_index = simplex[-1]
                best_simplex = torch.stack([population[index] for index in simplex[:-1]])
                centroid = best_simplex.mean(dim=0)
                reflected = torch.maximum(torch.minimum(centroid + (centroid - population[worst_index]), upper), lower)
                score = evaluate(reflected, "reflection")
                if len(trajectory) >= self.config.max_evaluations:
                    population[worst_index], scores[worst_index] = reflected, score
                    break
                if score >= scores[worst_index]:
                    contracted = torch.maximum(torch.minimum(centroid + 0.5 * (population[worst_index] - centroid), upper), lower)
                    score = evaluate(contracted, "contraction")
                    candidate = contracted
                    if score >= scores[worst_index]:
                        candidate = lower + (upper - lower) * torch.rand(n_parameters, generator=generator, dtype=torch.float64)
                        score = evaluate(candidate, "random_replacement")
                else:
                    candidate = reflected
                population[worst_index], scores[worst_index] = candidate, score
                if len(trajectory) >= self.config.max_evaluations:
                    break
            if len(trajectory) >= self.config.max_evaluations:
                break
        ordered = sorted(range(population_size), key=lambda index: scores[index])
        best_index = ordered[0]
        best_vector = default_vector.clone()
        best_vector[active_positions] = population[best_index]
        return {
            "method": "SCE-UA",
            "model_id": model_id,
            "seed": int(self.config.seed),
            "config": {"max_evaluations": self.config.max_evaluations, "kstop": self.config.kstop, "pcento": self.config.pcento, "n_complexes": self.config.n_complexes, "population_size": population_size},
            "parameter_names": list(names),
            "best_score": scores[best_index],
            "best_parameters": {name: float(best_vector[position]) for position, name in enumerate(PARAMETER_NAMES)},
            "evaluation_count": len(trajectory),
            "stop_reason": stop_reason,
            "trajectory": trajectory,
            "execution": {"evaluator": "UnifiedEvaluator.score_batched", "kernel": "simulate_coupled_rk2_batched", "basin_ids": list(basin_tuple), "batch_size": int(forcing.shape[0]), "compile_step": compile_step, "compile_backend": compile_backend, "compile_fullgraph": compile_fullgraph, "rng": "torch.Generator(device=cpu).manual_seed(config.seed)"},
        }
