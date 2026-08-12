"""
Neural Network Parameterizer for Differentiable Parameter Learning (dPL).
Maps normalized physical catchment attributes to physical hydrological model parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CatchmentParameterizer(nn.Module):
    """
    MLP mapping Catchment Attributes -> Hydrological Model Parameters theta.
    Uses Sigmoid + MinMax scaling to constrain outputs within physical parameter bounds.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: list[int] = [64, 64],
        param_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        dropout: float = 0.0,
        initial_theta: torch.Tensor | None = None,
    ):
        super().__init__()
        layers = []
        curr_dim = in_features
        for hdim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hdim))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = hdim
        layers.append(nn.Linear(curr_dim, out_features))

        self.net = nn.Sequential(*layers)
        self.param_bounds = param_bounds
        if initial_theta is not None:
            self.initialize_output_bias_from_theta(initial_theta)

    def initialize_output_bias_from_theta(
        self,
        theta: torch.Tensor,
        *,
        clamp_eps: float = 1e-5,
    ) -> torch.Tensor:
        """Set the final-layer bias so a zero output logit maps to ``theta``."""
        layer = self.net[-1]
        if not isinstance(layer, nn.Linear):
            raise TypeError("parameterizer.net[-1] must be an nn.Linear output layer")
        values = torch.as_tensor(theta, dtype=layer.bias.dtype, device=layer.bias.device).reshape(-1)
        if values.numel() != layer.out_features:
            raise ValueError(f"expected {layer.out_features} theta values, got {values.numel()}")
        logits = torch.logit(values.clamp(clamp_eps, 1.0 - clamp_eps))
        with torch.no_grad():
            layer.bias.copy_(logits)
        return logits.detach().clone()

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attributes: Tensor of shape (Batch_size, in_features)

        Returns:
            theta: Tensor of shape (Batch_size, out_features) constrained to physical bounds.
        """
        raw_output = self.net(attributes)
        scaled_output = torch.sigmoid(raw_output)

        if self.param_bounds is not None:
            min_b, max_b = self.param_bounds
            scaled_output = min_b + (max_b - min_b) * scaled_output

        return scaled_output
