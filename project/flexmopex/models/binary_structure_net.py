from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from project.flexmopex.models.parameter_nets import _BaseParameterNet


class BinaryStructureNet(_BaseParameterNet):
    """Parameter net for BinaryWeightMopex.

    Outputs 4 gate logits (log_alpha) instead of 8 Gumbel-softmax logit pairs.
    The log_alpha head is initialized with near-zero weights and a positive bias
    so that initial p_nonzero is high — Binary-Flex starts near Full structure
    and is compressed by the L0 penalty during training.
    """

    process_order = ["w_phen", "w_int", "w_snow", "w_sub"]

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        nmul: int = 1,
        device: str | torch.device = "cpu",
        init_bias: float = 2.0,
    ) -> None:
        self._init_bias = init_bias
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nmul=nmul,
            output_sizes={"params": 12 * nmul, "log_alpha": 4, "gamma_uh": 2},
            device=device,
        )

    def _initialize_weights(self) -> None:
        # Base class initializes all heads with std=0.001, bias=0.0
        super()._initialize_weights()
        # Override log_alpha head: keep near-zero weights, set positive bias
        log_alpha_head = self.heads["log_alpha"]
        nn.init.normal_(log_alpha_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(log_alpha_head.bias, self._init_bias)

    @classmethod
    def build_by_config(
        cls,
        config: dict[str, Any],
        device: str | torch.device = "cpu",
    ) -> "BinaryStructureNet":
        gate_cfg = config.get("binary_gate", {})
        return cls(
            input_dim=config["nx2"],
            nmul=config["nmul"],
            hidden_dim=config["hidden_size"],
            dropout=config["dr"],
            device=device,
            init_bias=float(gate_cfg.get("init_bias", 2.0)),
        )
