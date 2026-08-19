"""Neural-network parameterizers used by the differentiable benchmark."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class CatchmentParameterizer(nn.Module):
    """Map normalized catchment attributes to normalized/physical parameters.

    The default ``legacy`` architecture intentionally retains the historical
    ``net`` module and its state-dict keys.  ``process_heads`` is opt-in and
    uses a shared nonlinear encoder followed by an independent nonlinear head
    for each process group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: list[int] = [64, 64],
        param_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        dropout: float = 0.0,
        initial_theta: torch.Tensor | None = None,
        *,
        architecture: str = "legacy",
        parameter_names: list[str] | tuple[str, ...] | None = None,
        parameter_groups: Mapping[str, list[str] | tuple[str, ...]] | None = None,
        head_hidden_dims: list[int] | None = None,
        output_transform: str = "sigmoid",
        saturation_floor: float = 0.01,
        saturation_regularizer_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if architecture not in {"legacy", "process_heads", "multi_head", "residual_process", "residual_selective"}:
            raise ValueError(
                "architecture must be 'legacy', 'process_heads', 'residual_process', or 'residual_selective'"
            )
        if output_transform not in {"sigmoid", "softsign", "arctan", "identity", "linear"}:
            raise ValueError("output_transform must be sigmoid, softsign, arctan, identity, or linear")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if saturation_floor <= 0 or saturation_floor >= 0.25:
            raise ValueError("saturation_floor must be in (0, 0.25)")
        if saturation_regularizer_weight < 0 or not torch.isfinite(torch.tensor(saturation_regularizer_weight)):
            raise ValueError("saturation_regularizer_weight must be finite and non-negative")

        self.architecture = "process_heads" if architecture == "multi_head" else architecture
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_dims = [int(x) for x in hidden_dims]
        self.head_hidden_dims = [int(x) for x in (head_hidden_dims or [self.hidden_dims[-1]])]
        if param_bounds is None:
            self.param_bounds = None
        else:
            lower, upper = (torch.as_tensor(value).reshape(-1) for value in param_bounds)
            if lower.numel() != out_features or upper.numel() != out_features:
                raise ValueError("param_bounds must contain one lower/upper value per output")
            if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all() and (lower < upper).all()):
                raise ValueError("param_bounds must be finite and satisfy lower < upper")
            self.param_bounds = (lower, upper)
        self.output_transform = output_transform
        self.saturation_floor = float(saturation_floor)
        self.saturation_regularizer_weight = float(saturation_regularizer_weight)
        self.parameter_names = tuple(parameter_names or [f"parameter_{i}" for i in range(out_features)])
        if len(self.parameter_names) != out_features or len(set(self.parameter_names)) != out_features:
            raise ValueError("parameter_names must contain one unique entry per output")

        if self.architecture == "legacy":
            self.net = self._make_mlp(in_features, self.hidden_dims, out_features, dropout)
            self.parameter_groups = OrderedDict([("all", tuple(self.parameter_names))])
        elif self.architecture == "process_heads":
            if parameter_groups is None:
                raise ValueError("process_heads architecture requires parameter_groups")
            self.parameter_groups = self._validate_groups(parameter_groups)
            missing = set(self.parameter_names) - {name for names in self.parameter_groups.values() for name in names}
            if missing:
                raise ValueError(f"parameter_groups do not cover outputs: {sorted(missing)}")
            self.encoder = self._make_mlp(in_features, self.hidden_dims, self.hidden_dims[-1], dropout)
            self.heads = nn.ModuleDict({
                group: self._make_mlp(self.hidden_dims[-1], self.head_hidden_dims, len(names), dropout)
                for group, names in self.parameter_groups.items()
            })
            self._parameter_indices = {
                group: torch.tensor([self.parameter_names.index(name) for name in names], dtype=torch.long)
                for group, names in self.parameter_groups.items()
            }
        else:
            if parameter_groups is None:
                raise ValueError(f"{self.architecture} architecture requires parameter_groups")
            self.parameter_groups = self._validate_groups(parameter_groups)
            missing = set(self.parameter_names) - {name for names in self.parameter_groups.values() for name in names}
            if missing:
                raise ValueError(f"parameter_groups do not cover outputs: {sorted(missing)}")
            self.net = self._make_mlp(in_features, self.hidden_dims, out_features, dropout)
            adapter_groups = self.parameter_groups
            if self.architecture == "residual_selective":
                adapter_groups = OrderedDict(
                    (group, names)
                    for group, names in self.parameter_groups.items()
                    if group in {"routing", "interception"}
                )
            self.residual_adapter_groups = adapter_groups
            self.residual_adapters = nn.ModuleDict({
                group: self._make_residual_adapter(self.hidden_dims[-1], len(names))
                for group, names in adapter_groups.items()
            })
            self._parameter_indices = {
                group: torch.tensor([self.parameter_names.index(name) for name in names], dtype=torch.long)
                for group, names in self.parameter_groups.items()
            }

        if initial_theta is not None:
            self.initialize_output_bias_from_theta(initial_theta)
    @staticmethod
    def _make_mlp(in_features: int, hidden_dims: list[int], out_features: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        curr_dim = in_features
        for hdim in hidden_dims:
            layers.extend((nn.Linear(curr_dim, hdim), nn.LayerNorm(hdim), nn.GELU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = hdim
        layers.append(nn.Linear(curr_dim, out_features))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_residual_adapter(in_features: int, out_features: int) -> nn.Sequential:
        """Build a light residual adapter whose final layer starts exactly at zero."""
        hidden = min(64, int(in_features))
        adapter = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_features),
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def _validate_groups(self, groups: Mapping[str, list[str] | tuple[str, ...]]) -> OrderedDict[str, tuple[str, ...]]:
        result: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        seen: set[str] = set()
        for group, names in groups.items():
            names_tuple = tuple(names)
            if not names_tuple:
                continue
            if len(set(names_tuple)) != len(names_tuple):
                raise ValueError(f"duplicate parameter in group {group}")
            if any(name not in self.parameter_names for name in names_tuple):
                raise ValueError(f"unknown parameter in group {group}")
            overlap = seen.intersection(names_tuple)
            if overlap:
                raise ValueError(f"parameters assigned to multiple groups: {sorted(overlap)}")
            result[str(group)] = names_tuple
            seen.update(names_tuple)
        if not result:
            raise ValueError("parameter_groups must contain at least one non-empty group")
        return result

    def _output_layers(self) -> list[nn.Linear]:
        if self.architecture in {"legacy", "residual_process", "residual_selective"}:
            layers = [self.net[-1]]  # type: ignore[list-item]
            if self.architecture in {"residual_process", "residual_selective"}:
                layers.extend(adapter[-1] for adapter in self.residual_adapters.values())
            return layers
        return [head[-1] for head in self.heads.values()]  # type: ignore[list-item]


    def initialize_output_bias_from_theta(
        self,
        theta: torch.Tensor,
        *,
        clamp_eps: float = 1e-5,
    ) -> torch.Tensor:
        """Initialize output logits so sigmoid output starts at ``theta``."""
        values = torch.as_tensor(theta, dtype=next(self.parameters()).dtype, device=next(self.parameters()).device).reshape(-1)
        if values.numel() != self.out_features:
            raise ValueError(f"expected {self.out_features} theta values, got {values.numel()}")
        if self.output_transform != "sigmoid":
            raise ValueError("initial_theta bias initialization requires sigmoid output_transform")
        logits = torch.logit(values.clamp(clamp_eps, 1.0 - clamp_eps))
        with torch.no_grad():
            if self.architecture in {"legacy", "residual_process", "residual_selective"}:
                self.net[-1].bias.copy_(logits)
            else:
                for group, names in self.parameter_groups.items():
                    indices = [self.parameter_names.index(name) for name in names]
                    self.heads[group][-1].bias.copy_(logits[indices])
        return logits.detach().clone()

    def _raw_output(self, attributes: torch.Tensor) -> torch.Tensor:
        if self.architecture == "legacy":
            return self.net(attributes)
        if self.architecture in {"residual_process", "residual_selective"}:
            base_output = self.net(attributes)
            embedding = self.net[:-1](attributes)
            delta = torch.zeros_like(base_output)
            for group, adapter in self.residual_adapters.items():
                indices = self._parameter_indices[group].to(device=delta.device)
                delta[:, indices] = adapter(embedding)
            return base_output + delta
        embedding = self.encoder(attributes)
        output = torch.empty(
            (attributes.shape[0], self.out_features), dtype=embedding.dtype, device=embedding.device
        )
        for group, names in self.parameter_groups.items():
            group_output = self.heads[group](embedding)
            output[:, self._parameter_indices[group].to(device=output.device)] = group_output
        return output

    def _apply_transform(self, raw_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.output_transform == "sigmoid":
            normalized = torch.sigmoid(raw_output)
            # Compute sigma'(z) without the ``1 - sigma(z)`` cancellation at
            # large positive logits.  This keeps the diagnostic penalty
            # differentiable for finite extreme latents while leaving the
            # forward mapping mathematically unchanged.
            tail = torch.exp(-raw_output.abs())
            jacobian = tail / (1.0 + tail).square()
        elif self.output_transform == "softsign":
            denominator = 1.0 + raw_output.abs()
            normalized = 0.5 * (raw_output / denominator + 1.0)
            jacobian = 0.5 / denominator.square()
        elif self.output_transform == "arctan":
            normalized = 0.5 + torch.atan(raw_output) / torch.pi
            jacobian = 0.5 / (torch.pi * (1.0 + raw_output.square()))
        else:
            normalized = raw_output
            jacobian = torch.ones_like(raw_output)
        return normalized, jacobian

    def mapping_diagnostics(
        self,
        raw_latent: torch.Tensor,
        normalized_output: torch.Tensor | None = None,
        jacobian: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Return raw latent, normalized Jacobian, position, and summaries.

        The Jacobian is normalized by the physical bound span when bounds are
        supplied: the output here is the local derivative of the normalized
        mapping, not a derivative in physical units.
        """
        if normalized_output is None or jacobian is None:
            normalized_output, jacobian = self._apply_transform(raw_latent)
        position = normalized_output
        if self.param_bounds is not None:
            lower, upper = self.param_bounds
            span = (upper - lower).to(device=normalized_output.device, dtype=normalized_output.dtype)
            lower = lower.to(device=normalized_output.device, dtype=normalized_output.dtype)
            physical = lower + span * normalized_output
            position = (physical - lower) / span
        return {
            "raw_latent": raw_latent,
            "normalized_output": normalized_output,
            # transform_jacobian is the raw-latent -> parameterizer-output
            # derivative; callers may add downstream physical mapping factors
            # to total_mapping_jacobian for telemetry.
            "transform_jacobian": jacobian,
            "normalized_jacobian": jacobian,
            "physical_normalized_position": position,
            "parameter_names": self.parameter_names,
            "transform": self.output_transform
        }

    def saturation_regularizer_from_diagnostics(self, diagnostics: Mapping[str, Any]) -> torch.Tensor:
        """Continuously penalize only transform saturation below the Jacobian floor."""
        jacobian = diagnostics.get("transform_jacobian", diagnostics["normalized_jacobian"])
        if self.output_transform in {"identity", "linear"}:
            return diagnostics["raw_latent"].sum() * 0.0
        deficit = F.relu(self.saturation_floor - jacobian)
        return deficit.square().mean()

    def saturation_regularizer(self, raw_latent: torch.Tensor) -> torch.Tensor:
        """Compute the optional, transform-aware saturation penalty."""
        diagnostics = self.mapping_diagnostics(raw_latent)
        return self.saturation_regularizer_from_diagnostics(diagnostics)

    def regularization_loss(self, attributes: torch.Tensor) -> torch.Tensor:
        """Convenience method for callers that do not need the diagnostics."""
        raw = self._raw_output(attributes)
        return self.saturation_regularizer(raw) * self.saturation_regularizer_weight

    def summarize_mapping_diagnostics(self, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
        """Small CPU-friendly telemetry summary; no per-sample log is emitted."""
        jacobian = diagnostics.get("total_mapping_jacobian", diagnostics["normalized_jacobian"]).detach()
        position = diagnostics["physical_normalized_position"].detach()
        raw = diagnostics["raw_latent"].detach()
        if jacobian.ndim != 2:
            raise ValueError("mapping diagnostics must have shape [batch, parameter]")
        quantile = lambda value, q: torch.quantile(value, q, dim=0).cpu().tolist()
        return {
            "transform": diagnostics["transform"],
            "parameter_names": list(self.parameter_names),
            "jacobian_p05": quantile(jacobian, 0.05),
            "jacobian_median": quantile(jacobian, 0.50),
            "jacobian_p95": quantile(jacobian, 0.95),
            "fraction_below_saturation_floor": (jacobian < self.saturation_floor).to(torch.float32).mean(dim=0).cpu().tolist(),
            "physical_lower_1pct": (position < 0.01).to(torch.float32).mean(dim=0).cpu().tolist(),
            "physical_lower_5pct": (position < 0.05).to(torch.float32).mean(dim=0).cpu().tolist(),
            "physical_upper_5pct": (position > 0.95).to(torch.float32).mean(dim=0).cpu().tolist(),
            "physical_upper_1pct": (position > 0.99).to(torch.float32).mean(dim=0).cpu().tolist(),
            "latent_abs_median": torch.quantile(raw.abs(), 0.5, dim=0).cpu().tolist(),
            "latent_abs_p95": torch.quantile(raw.abs(), 0.95, dim=0).cpu().tolist(),
        }

    def forward(
        self,
        attributes: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Return constrained outputs; diagnostics are an opt-in auxiliary path."""
        raw_output = self._raw_output(attributes)
        normalized_output, jacobian = self._apply_transform(raw_output)
        scaled_output = normalized_output
        if self.param_bounds is not None:
            min_b, max_b = self.param_bounds
            min_b = min_b.to(device=scaled_output.device, dtype=scaled_output.dtype)
            max_b = max_b.to(device=scaled_output.device, dtype=scaled_output.dtype)
            scaled_output = min_b + (max_b - min_b) * scaled_output
        if return_diagnostics:
            diagnostics = self.mapping_diagnostics(raw_output, normalized_output, jacobian)
            return scaled_output, diagnostics
        return scaled_output
