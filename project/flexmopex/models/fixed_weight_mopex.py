from __future__ import annotations

from typing import Any

import torch

from project.flexmopex.models import mopex_core
from project.flexmopex.models.base_mopex import BaseMopex


class FixedWeightMopex(BaseMopex):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "FixedWeightMopex"
        fixed_weights = self.config["fixed_weights"]
        values = [float(fixed_weights[name]) for name in self.weight_names]
        self.register_buffer(
            "fixed_weight_values",
            torch.tensor(values, dtype=torch.float32, device=self.device),
        )
        self.step_fn = self._compile_step(mopex_core.mopex_step)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        mopex_params = self._descale_mopex_params(parameters["params"])
        routing_params = self._descale_routing_params(parameters["gamma_uh"])
        P, T, PET, doy, n_steps, n_grid = self._prepare_forcings(x_dict)
        weights_on = self.fixed_weight_values.unsqueeze(0).expand(n_grid, -1)
        Q_mopex = self._run_weighted_loop(
            P, T, PET, doy, mopex_params, weights_on, n_steps, n_grid
        )
        Qrouted = self._apply_routing(Q_mopex.mean(-1), routing_params)
        result = {"streamflow": Qrouted}
        result.update(self._weight_outputs(weights_on, Q_mopex.shape[0]))
        return result
