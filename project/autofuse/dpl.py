"""Structure-conditioned dPL scaffolding without a training launcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from dfuse.spec import DECISION_ORDER, PARAMETER_NAMES, get_structure, parameter_metadata
from project.autofuse.parameter_contract import get_parameter_contract


@dataclass(frozen=True)
class DPLConfig:
    attribute_dim: int = 35
    hidden_dim: int = 64
    seed: int = 20260901
    learning_rate: float = 1.0e-3
    training_started: bool = False


class StructureConditionedParameterizer(nn.Module):
    """Map catchment attributes + decision codes to bounded union parameters.

    Uses a shared conditioning trunk with 37 independent scalar coordinate heads.
    On any forward step for structure s, only active parameter heads are evaluated,
    guaranteeing that inactive heads have .grad is None and no optimizer momentum drift.
    """

    def __init__(self, config: DPLConfig | None = None):
        super().__init__()
        self.config = config or DPLConfig()
        self.trunk = nn.Sequential(
            nn.Linear(self.config.attribute_dim + len(DECISION_ORDER), self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(self.config.hidden_dim, 1) for _ in range(len(PARAMETER_NAMES))
        ])

        lowers, uppers = [], []
        for name in PARAMETER_NAMES:
            meta = parameter_metadata(name)
            low, up = float(meta["lower"]), float(meta["upper"])
            if low > up:
                low, up = up, low
            lowers.append(low)
            uppers.append(up)

        lower = torch.tensor(lowers, dtype=torch.float64)
        upper = torch.tensor(uppers, dtype=torch.float64)
        defaults = torch.tensor([float(parameter_metadata(name)["default"]) for name in PARAMETER_NAMES], dtype=torch.float64)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.register_buffer("defaults", defaults)

    def _decision_features(self, model_id: int, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        codes = torch.tensor(get_structure(model_id).decision_code_vector, device=device, dtype=dtype) / 1000.0
        return codes.expand(batch, -1)

    def forward(self, attributes: Tensor, model_id: int) -> Tensor:
        if attributes.ndim != 2 or attributes.shape[1] != self.config.attribute_dim:
            raise ValueError(f"attributes must have shape [batch, {self.config.attribute_dim}]")
        model_device = next(self.trunk.parameters()).device
        model_dtype = next(self.trunk.parameters()).dtype
        attributes = attributes.to(device=model_device, dtype=model_dtype)
        batch_size = attributes.shape[0]
        dtype = model_dtype
        device = model_device

        # 1. Shared conditioning & trunk forward
        decision = self._decision_features(model_id, batch_size, device=device, dtype=dtype)
        network_input = torch.cat((attributes, decision), dim=1)
        trunk_dtype = next(self.trunk.parameters()).dtype
        hidden = self.trunk(network_input.to(dtype=trunk_dtype))

        # 2. Query authoritative structure contract
        contract = get_parameter_contract(model_id)
        active_indices_set = set(contract.active_indices)

        # 3. Assemble simulator-facing [batch, 37] tensor
        lower = self.lower.to(device=device, dtype=dtype)
        upper = self.upper.to(device=device, dtype=dtype)
        defaults = self.defaults.to(device=device, dtype=dtype)

        outputs = []
        for i in range(len(PARAMETER_NAMES)):
            if i in active_indices_set:
                raw_i = self.heads[i](hidden).squeeze(-1)
                val_i = lower[i] + torch.sigmoid(raw_i) * (upper[i] - lower[i])
                outputs.append(val_i)
            else:
                # Inactive coordinate: insert graph-free constant default value.
                # self.heads[i] is NOT executed and NOT part of the autograd graph.
                def_i = defaults[i].expand(batch_size)
                outputs.append(def_i)

        return torch.stack(outputs, dim=1)

    def parameter_mapping(self, attributes: Tensor, model_id: int) -> dict[str, Tensor]:
        values = self.forward(attributes, model_id)
        return {name: values[:, i] for i, name in enumerate(PARAMETER_NAMES)}


class PureSharedParameterizer(nn.Module):
    """Pure shared dPL parameterizer mapping catchment attributes X_c -> canonical parameter super-vector.

    Hard constraints:
    - No structure ID / decision codes input to parameterizer;
    - Shared trunk + 37 independent scalar coordinate heads;
    - Same catchment attributes produce identical canonical parameters across all structures;
    - Inactive parameters are masked to constant defaults and do not receive gradients;
    - Exposes pre-transform coordinate representations for gradient conflict probing.
    """

    def __init__(self, config: DPLConfig | None = None):
        super().__init__()
        self.config = config or DPLConfig()
        self.trunk = nn.Sequential(
            nn.Linear(self.config.attribute_dim, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(self.config.hidden_dim, 1) for _ in range(len(PARAMETER_NAMES))
        ])

        lowers, uppers = [], []
        for name in PARAMETER_NAMES:
            meta = parameter_metadata(name)
            low, up = float(meta["lower"]), float(meta["upper"])
            if low > up:
                low, up = up, low
            lowers.append(low)
            uppers.append(up)

        lower = torch.tensor(lowers, dtype=torch.float64)
        upper = torch.tensor(uppers, dtype=torch.float64)
        defaults = torch.tensor([float(parameter_metadata(name)["default"]) for name in PARAMETER_NAMES], dtype=torch.float64)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.register_buffer("defaults", defaults)

    def forward_canonical(self, attributes: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (u_raw, theta_physical) for all 37 canonical parameters before active masking."""
        if attributes.ndim != 2 or attributes.shape[1] != self.config.attribute_dim:
            raise ValueError(f"attributes must have shape [batch, {self.config.attribute_dim}]")
        model_device = next(self.trunk.parameters()).device
        model_dtype = next(self.trunk.parameters()).dtype
        attributes = attributes.to(device=model_device, dtype=model_dtype)

        hidden = self.trunk(attributes)
        lower = self.lower.to(device=model_device, dtype=model_dtype)
        upper = self.upper.to(device=model_device, dtype=model_dtype)

        u_list = []
        theta_list = []
        for i in range(len(PARAMETER_NAMES)):
            u_i = self.heads[i](hidden).squeeze(-1)
            theta_i = lower[i] + torch.sigmoid(u_i) * (upper[i] - lower[i])
            u_list.append(u_i)
            theta_list.append(theta_i)

        return torch.stack(u_list, dim=1), torch.stack(theta_list, dim=1)

    def forward(self, attributes: Tensor, model_id: int | Tensor) -> Tensor:
        """Forward for structure s: active heads connected to graph, inactive masked to constant default."""
        if isinstance(model_id, Tensor):
            model_id = int(model_id.item() if model_id.numel() == 1 else model_id[0].item())

        if attributes.ndim != 2 or attributes.shape[1] != self.config.attribute_dim:
            raise ValueError(f"attributes must have shape [batch, {self.config.attribute_dim}]")
        model_device = next(self.trunk.parameters()).device
        model_dtype = next(self.trunk.parameters()).dtype
        attributes = attributes.to(device=model_device, dtype=model_dtype)
        batch_size = attributes.shape[0]

        hidden = self.trunk(attributes)
        contract = get_parameter_contract(model_id)
        active_indices_set = set(contract.active_indices)

        lower = self.lower.to(device=model_device, dtype=model_dtype)
        upper = self.upper.to(device=model_device, dtype=model_dtype)
        defaults = self.defaults.to(device=model_device, dtype=model_dtype)

        outputs = []
        for i in range(len(PARAMETER_NAMES)):
            if i in active_indices_set:
                raw_i = self.heads[i](hidden).squeeze(-1)
                val_i = lower[i] + torch.sigmoid(raw_i) * (upper[i] - lower[i])
                outputs.append(val_i)
            else:
                def_i = defaults[i].expand(batch_size)
                outputs.append(def_i)

        return torch.stack(outputs, dim=1)

def dpl_plan(config: DPLConfig | None = None) -> dict[str, object]:
    cfg = config or DPLConfig()
    return {
        "method": "structure-conditioned-dPL",
        "attribute_dim": cfg.attribute_dim,
        "parameter_union_size": len(PARAMETER_NAMES),
        "structure_conditioning": list(DECISION_ORDER),
        "shared_forward": "dfuse.simulate",
        "training_started": cfg.training_started,
        "seed": cfg.seed,
    }
