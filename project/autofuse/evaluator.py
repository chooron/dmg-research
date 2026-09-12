"""Unified experiment evaluator: SCE and dPL must call this same kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from dfuse import BatchedSimulationResult, SimulationResult, simulate_coupled_rk2_batched, simulate

from .metrics import kgecomp, kgecomp_batched


@dataclass
class EvaluationOutput:
    model_id: int
    result: SimulationResult | BatchedSimulationResult
    kge_comp: Tensor | None


class UnifiedEvaluator:
    """Thin experiment-layer adapter around ``dfuse.simulate``."""

    def forward(self, model_id: int, forcing: Tensor, params: Mapping[str, object] | Tensor | None = None, **kwargs) -> SimulationResult:
        return simulate(model_id, forcing, params, **kwargs)
    @staticmethod
    def _batched_params(params: Mapping[str, object] | Tensor | Sequence[Mapping[str, object]] | None, batch_size: int) -> Sequence[Mapping[str, object]] | Tensor:
        if params is None:
            return [{} for _ in range(batch_size)]
        if isinstance(params, Tensor):
            if params.ndim == 1:
                return params.unsqueeze(0).expand(batch_size, -1)
            if params.ndim == 2 and params.shape[0] == 1 and batch_size > 1:
                return params.expand(batch_size, -1)
            return params
        if isinstance(params, Mapping):
            return [params for _ in range(batch_size)]
        return params

    def forward_batched(
        self,
        model_id: int,
        forcing: Tensor,
        params: Mapping[str, object] | Tensor | Sequence[Mapping[str, object]] | None = None,
        *,
        basin_ids: Sequence[str] | None = None,
        **kwargs,
    ) -> BatchedSimulationResult:
        if forcing.ndim != 3:
            raise ValueError("batched forcing must have shape [basin, time, 3]")
        batched_params = self._batched_params(params, int(forcing.shape[0]))
        return simulate_coupled_rk2_batched(model_id, forcing, batched_params, basin_ids=basin_ids, **kwargs)


    def score(
        self,
        model_id: int,
        forcing: Tensor,
        observed: Tensor,
        params: Mapping[str, object] | Tensor | None = None,
        *,
        inverse_epsilon: float | Tensor | None = None,
        **kwargs,
    ) -> EvaluationOutput:
        result = self.forward(model_id, forcing, params, **kwargs)
        observed = observed.to(device=result.q.device, dtype=result.q.dtype)
        if result.q.shape[0] < observed.shape[0]:
            observed_slice = observed[:result.q.shape[0]]
        else:
            observed_slice = observed
        return EvaluationOutput(result.model_id, result, kgecomp(result.q, observed_slice, epsilon=inverse_epsilon))
    def score_batched(
        self,
        model_id: int,
        forcing: Tensor,
        observed: Tensor,
        params: Mapping[str, object] | Tensor | Sequence[Mapping[str, object]] | None = None,
        *,
        inverse_epsilon: float | Tensor | None = None,
        basin_ids: Sequence[str] | None = None,
        **kwargs,
    ) -> EvaluationOutput:
        result = self.forward_batched(model_id, forcing, params, basin_ids=basin_ids, **kwargs)
        observed = observed.to(device=result.q.device, dtype=result.q.dtype)
        if observed.shape[0] != result.q.shape[0]:
            raise ValueError(f"observed batch size {observed.shape[0]} does not match batched q {result.q.shape[0]}")
        observed_slice = observed[:, :result.q.shape[1]] if result.q.shape[1] < observed.shape[1] else observed
        kge = kgecomp_batched(result.q, observed_slice, epsilon=inverse_epsilon)
        if result.monitoring.get("stopped_on_failure"):
            for idx, p_info in enumerate(result.monitoring.get("per_basin", [])):
                if p_info.get("first_failure_index") is not None:
                    kge[idx] = float("nan")
        return EvaluationOutput(result.model_id, result, kge)



def evaluate_one(
    model_id: int,
    forcing: Tensor,
    observed: Tensor | None = None,
    params: Mapping[str, object] | Tensor | None = None,
    **kwargs,
) -> EvaluationOutput:
    evaluator = UnifiedEvaluator()
    if observed is None:
        result = evaluator.forward(model_id, forcing, params, **kwargs)
        return EvaluationOutput(model_id, result, None)
    return evaluator.score(model_id, forcing, observed, params, **kwargs)
