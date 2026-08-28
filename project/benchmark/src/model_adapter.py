from __future__ import annotations

from dataclasses import dataclass
import torch

from .model_registry import ModelSpec, build_model, get_spec
from .objective import streaming_kge
from .parameter_transform import LatentBoundTransform


@dataclass
class Evaluation:
    kge: torch.Tensor
    invalid: torch.Tensor


class BatchedModelAdapter:
    """One-model GPU evaluator; no Python loop over basin/start/population candidates."""

    def __init__(self, model_name: str, forcing: torch.Tensor, observation: torch.Tensor, *,
                 warm_up: int = 365, device: str | torch.device = "cuda", backend: str = "eager",
                 objective_compile_mode: str = "eager",
                 invalid_penalty: float = -1_000_000.0) -> None:
        self.device = torch.device(device)
        self.spec: ModelSpec = get_spec(model_name, self.device)
        self.model = build_model(model_name, self.device, warm_up=warm_up, backend=backend, dtype=torch.float64)
        self.forcing = forcing.to(self.device, dtype=torch.float64, non_blocking=True)
        self.observation = observation.to(self.device, dtype=torch.float64, non_blocking=True)
        self.warm_up = min(int(warm_up), int(self.forcing.shape[0]))
        self.transform = LatentBoundTransform(self.spec.bounds)
        self.invalid_penalty = float(invalid_penalty)
        # Compile the pure reduction separately. Do not compile this adapter or the
        # solver coordinator: those own Python dictionaries, checkpointing and state.
        self.objective_compile_mode = objective_compile_mode
        self.kge_kernel = streaming_kge if objective_compile_mode == "eager" else torch.compile(
            streaming_kge, backend="inductor", mode=objective_compile_mode, fullgraph=False
        )

    @property
    def basin_count(self) -> int:
        return int(self.forcing.shape[1])

    def evaluate(self, latent: torch.Tensor) -> Evaluation:
        """latent [B,S,P,D] -> KGE [B,S,P]."""
        if latent.ndim != 4 or latent.shape[0] != self.basin_count or latent.shape[-1] != self.spec.dimension:
            raise ValueError(f"expected [B,S,P,{self.spec.dimension}], got {tuple(latent.shape)}")
        b, starts, pop, dim = latent.shape
        normalized = self.transform.latent_to_normalized(latent).to(torch.float64)
        raw = normalized.permute(0, 3, 1, 2).reshape(b, dim, starts * pop)
        with torch.inference_mode():
            q = self.model({"x_phy": self.forcing}, (None, raw))["streamflow"]
        t = q.shape[0]
        q = q.reshape(t, b, starts, pop)
        obs = self.observation[self.warm_up : self.warm_up + t]
        if obs.ndim == 3:
            obs = obs[..., 0]
        score, invalid = self.kge_kernel(q, obs, invalid_penalty=self.invalid_penalty)
        return Evaluation(score.to(torch.float64), invalid)
