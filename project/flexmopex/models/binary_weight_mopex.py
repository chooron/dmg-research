from __future__ import annotations

import math
from typing import Any

import torch

from project.flexmopex.models import mopex_core
from project.flexmopex.models.base_mopex import BaseMopex

# Process order matches WEIGHT_NAMES in base_mopex: [w_phen, w_int, w_snow, w_sub]
_PROCESS_ORDER = ["phen", "int", "snow", "sub"]


class BinaryWeightMopex(BaseMopex):
    """Hard-Concrete (L0) gated MOPEX model.

    During training, gates are relaxed via the Hard-Concrete distribution.
    During eval, deterministic binary gates are computed from p_nonzero >= 0.5,
    where p_nonzero = sigmoid(log_alpha - temperature * log(-gamma / zeta)).
    This keeps the L0 penalty and the eval binarization consistent.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "BinaryWeightMopex"
        self.temperature = float(self.config.get("concrete_temperature", 0.5))
        self.gamma = float(self.config.get("concrete_gamma", -0.1))
        self.zeta = float(self.config.get("concrete_zeta", 1.1))
        # 4 log_alpha logits (one per process), not 8
        self.learnable_param_count += len(self.weight_names)
        self.step_fn = self._compile_step(mopex_core.mopex_step)
        # Pre-compute the p_nonzero offset: temperature * log(-gamma / zeta)
        self._pnz_offset = self.temperature * math.log(-self.gamma / self.zeta)

    def _p_nonzero(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Expected non-zero probability under Hard-Concrete: sigmoid(log_alpha - offset)."""
        return torch.sigmoid(log_alpha - self._pnz_offset)

    def _structure_weights(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Return gates: relaxed during training, hard binary during eval.

        Args:
            log_alpha: (n_grid, 4) gate logits

        Returns:
            (n_grid, 4) gates in [0, 1]
        """
        if self.training:
            u = torch.zeros_like(log_alpha).uniform_().clamp(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid(
                (u.log() - (1.0 - u).log() + log_alpha) / self.temperature
            )
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            return s_bar.clamp(0.0, 1.0)
        else:
            p_nz = self._p_nonzero(log_alpha)
            return (p_nz >= 0.5).float()

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        mopex_params = self._descale_mopex_params(parameters["params"])
        routing_params = self._descale_routing_params(parameters["gamma_uh"])
        log_alpha = parameters["log_alpha"]  # (n_grid, 4)

        gates = self._structure_weights(log_alpha)  # (n_grid, 4)
        p_nz = self._p_nonzero(log_alpha)           # (n_grid, 4) — always soft

        P, T, PET, doy, n_steps, n_grid = self._prepare_forcings(x_dict)
        Q_mopex = self._run_weighted_loop(
            P, T, PET, doy, mopex_params, gates, n_steps, n_grid
        )
        Qrouted = self._apply_routing(Q_mopex.mean(-1), routing_params)

        n_train = Q_mopex.shape[0]
        result: dict[str, torch.Tensor] = {"streamflow": Qrouted}

        # Save gates (z_*) and activation probabilities (p_*) for each process
        for i, proc in enumerate(_PROCESS_ORDER):
            gate_i = gates[:, i].view(1, -1, 1).expand(n_train, -1, -1)
            p_i = p_nz[:, i].view(1, -1, 1).expand(n_train, -1, -1)
            result[f"z_{proc}"] = gate_i
            result[f"p_{proc}"] = p_i

        # Store log_alpha as a model attribute for the loss function to read.
        # It is NOT included in the output dict to avoid shape conflicts during
        # trainer batching (log_alpha has no time dimension).
        self._last_log_alpha = log_alpha

        return result
