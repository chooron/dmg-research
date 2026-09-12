"""Fixed-Dimensional FUSE Parameter Interface for Differentiable Learning."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from dfuse import PARAMETER_NAMES, StructureSpec, get_structure
from dfuse.spec import default_parameters, parameter_metadata
from project.autofuse.parameter_contract import FUSEParameterContract, get_parameter_contract


class FUSEParameterInterface(nn.Module):
    """Clean, differentiable fixed-dimensional output interface for FUSE dPL.

    The neural network predicts a fixed 37-dimensional tensor corresponding to
    the union of all FUSE parameters. For any given structure s, this interface:
    - Binds parameters within legal physical bounds [lower, upper];
    - Exposes active vs inactive coordinates explicitly;
    - Produces simulator-ready parameter tensors or mappings;
    - Guarantees that loss gradients backpropagate strictly to active coordinates.
    """

    def __init__(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        lowers = []
        uppers = []
        for name in PARAMETER_NAMES:
            meta = parameter_metadata(name)
            low, up = float(meta["lower"]), float(meta["upper"])
            if low > up:
                low, up = up, low
            lowers.append(low)
            uppers.append(up)
        lower = torch.tensor(lowers, dtype=dtype, device=self.device)
        upper = torch.tensor(uppers, dtype=dtype, device=self.device)
        defaults = torch.tensor([parameter_metadata(name)["default"] for name in PARAMETER_NAMES], dtype=dtype, device=self.device)

        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.register_buffer("defaults", defaults)
        self.union_size = len(PARAMETER_NAMES)

    def bounds_transform(self, raw_logits: Tensor) -> Tensor:
        """Map unconstrained raw network logits to bounded physical parameters [lower, upper]."""
        if raw_logits.shape[-1] != self.union_size:
            raise ValueError(f"raw_logits last dimension must be {self.union_size}, got {raw_logits.shape[-1]}")
        lower = self.lower.to(device=raw_logits.device, dtype=raw_logits.dtype)
        upper = self.upper.to(device=raw_logits.device, dtype=raw_logits.dtype)
        return lower + torch.sigmoid(raw_logits) * (upper - lower)

    def active_mask_tensor(self, model_id: int | StructureSpec, *, device: torch.device | None = None) -> Tensor:
        """Return a 1D boolean tensor of shape [37] indicating active coordinates for structure s."""
        contract = get_parameter_contract(model_id)
        mask = [name in contract.active_parameters for name in PARAMETER_NAMES]
        dev = device if device is not None else self.device
        return torch.tensor(mask, dtype=torch.bool, device=dev)

    def mask_parameters(
        self,
        bounded_params: Tensor,
        model_id: int | StructureSpec,
        *,
        inactive_fill: str = "defaults",
    ) -> Tensor:
        """Mask inactive coordinates so they do not receive spurious parameter values.

        Parameters
        ----------
        bounded_params : Tensor
            Tensor of shape [batch, 37] or [37] within legal bounds.
        model_id : int | StructureSpec
            FUSE structure identifier.
        inactive_fill : str
            'defaults' to fill inactive parameters with canonical defaults,
            'zero' to fill with 0.0, or 'identity' to leave bounded predictions.

        Returns
        -------
        Tensor
            Masked parameter tensor of identical shape, differentiable with respect to active coordinates.
        """
        mask = self.active_mask_tensor(model_id, device=bounded_params.device)
        if bounded_params.ndim == 2:
            mask = mask.unsqueeze(0).expand_as(bounded_params)

        if inactive_fill == "defaults":
            fill = self.defaults.to(device=bounded_params.device, dtype=bounded_params.dtype)
            if bounded_params.ndim == 2:
                fill = fill.unsqueeze(0).expand_as(bounded_params)
            return torch.where(mask, bounded_params, fill)
        elif inactive_fill == "zero":
            zero = torch.zeros_like(bounded_params)
            return torch.where(mask, bounded_params, zero)
        elif inactive_fill == "identity":
            return bounded_params
        else:
            raise ValueError(f"unknown inactive_fill mode: {inactive_fill}")

    def extract_active_tensor(self, bounded_params: Tensor, model_id: int | StructureSpec) -> Tensor:
        """Extract only the structure-active parameter slice [batch, n_active]."""
        contract = get_parameter_contract(model_id)
        active_indices = torch.tensor(contract.active_indices, device=bounded_params.device, dtype=torch.long)
        if bounded_params.ndim == 1:
            return bounded_params[active_indices]
        return bounded_params[:, active_indices]

    def to_simulator_dict(self, bounded_params: Tensor, model_id: int | StructureSpec) -> dict[str, Tensor]:
        """Convert a [batch, 37] bounded parameter tensor into a named parameter dictionary."""
        contract = get_parameter_contract(model_id)
        masked = self.mask_parameters(bounded_params, model_id, inactive_fill="defaults")
        if masked.ndim == 1:
            return {name: masked[i] for i, name in enumerate(PARAMETER_NAMES)}
        return {name: masked[:, i] for i, name in enumerate(PARAMETER_NAMES)}
