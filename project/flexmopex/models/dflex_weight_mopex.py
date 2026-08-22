"""Deterministic hard-gate Flex-MOPEX variant with CF/BCE supervision.

This module is deliberately separate from ``BinaryWeightMopex``.  DFlex-CF/BCE
uses the canonical Candidate E/S0 simulator and the canonical two-logit
structural head, but replaces the simulator's continuous process weights with
``1[p_struct > 0.5]``.  There is no Hard-Concrete sampling and no expected-L0
term in this path.
"""
from __future__ import annotations

from typing import Any

import torch

from project.flexmopex.models.base_mopex import WEIGHT_NAMES
from project.flexmopex.models.learned_weight_mopex_candidates import LearnedWeightMopexE


class DFlexWeightMopexCF(LearnedWeightMopexE):
    """Candidate E/S0 MOPEX with deterministic 0/1 process gates.

    The structural support score is the same two-logit contrast used by CFlex:
    ``p_struct = sigmoid(z_ON - z_OFF)``.  The score remains differentiable and
    is exposed for the CF/BCE trainer, while the simulator receives only the
    detached-valued hard gate ``(p_struct > 0.5).float()``.
    """

    is_dflex = True
    gate_threshold = 0.5

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "DFlexWeightMopexCF"
        self._last_p_struct: torch.Tensor | None = None
        self._last_hard_gates: torch.Tensor | None = None

    def _structure_support_and_gate(
        self, raw_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = raw_weights.view(raw_weights.shape[0], len(WEIGHT_NAMES), 2)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        p_struct = torch.sigmoid(logits[..., 1] - logits[..., 0])
        hard_gates = (p_struct > self.gate_threshold).to(p_struct.dtype)

        if self.removed_processes:
            for index, name in enumerate(self.weight_names):
                if name in self.removed_processes:
                    hard_gates[:, index] = 0.0

        return p_struct, hard_gates

    def _structure_weights(self, raw_weights: torch.Tensor) -> torch.Tensor:
        p_struct, hard_gates = self._structure_support_and_gate(raw_weights)
        self._last_p_struct = p_struct
        self._last_hard_gates = hard_gates
        return hard_gates

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        result = super().forward(x_dict, parameters)
        if self._last_p_struct is None or self._last_hard_gates is None:
            raise RuntimeError("DFlex forward did not produce structural support scores.")

        n_train = result["streamflow"].shape[0]
        p_struct = self._last_p_struct.view(1, -1, len(WEIGHT_NAMES)).expand(
            n_train, -1, -1
        )
        hard_gates = self._last_hard_gates.view(1, -1, len(WEIGHT_NAMES)).expand(
            n_train, -1, -1
        )

        # Keep the regular w_* outputs as the simulator gates for compatibility
        # with existing loss/analysis code, and expose unambiguous DFlex names.
        result["p_struct"] = p_struct
        result["hard_gates"] = hard_gates
        result["z_struct"] = hard_gates
        for index, name in enumerate(self.weight_names):
            result[f"p_{name.removeprefix('w_')}"] = p_struct[:, :, index : index + 1]
            result[f"z_{name.removeprefix('w_')}"] = hard_gates[:, :, index : index + 1]
        return result
