from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from project.flexmopex.models import mopex_core
from project.flexmopex.models.base_mopex import BaseMopex


class LearnedWeightMopex(BaseMopex):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "LearnedWeightMopex"
        self.structure_tau = float(self.config.get("structure_tau", 1.0))
        self.learnable_param_count += len(self.weight_names) * 2
        self.step_fn = self._compile_step(mopex_core.mopex_step)

    def _structure_weights(self, raw_weights: torch.Tensor) -> torch.Tensor:
        logits = raw_weights.view(raw_weights.shape[0], len(self.weight_names), 2)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        if self.training:
            probs = F.gumbel_softmax(
                logits, tau=self.structure_tau, hard=False, dim=-1
            )
        else:
            probs = F.softmax(logits, dim=-1)
        return probs[..., 1]

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        mopex_params = self._descale_mopex_params(parameters["params"])
        routing_params = self._descale_routing_params(parameters["gamma_uh"])
        weights_on = self._structure_weights(parameters["weights"])
        P, T, PET, doy, n_steps, n_grid = self._prepare_forcings(x_dict)
        Q_mopex = self._run_weighted_loop(
            P, T, PET, doy, mopex_params, weights_on, n_steps, n_grid
        )
        Qrouted = self._apply_routing(Q_mopex.mean(-1), routing_params)
        result = {"streamflow": Qrouted}
        result.update(self._weight_outputs(weights_on, Q_mopex.shape[0]))
        return result
