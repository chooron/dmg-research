"""Small production-shaped Trainer used by the dMoT remediation gate.

It intentionally uses the real HydrologyModel, parameter mapping, optimizer,
mask and checkpoint implementation while keeping the gate independent of the
external dMG neural-network package.  This is also useful for model-by-model
CPU/CUDA preflight before a larger parameter-network run.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from losses import NseBatchLoss
from models import HydrologyModel

from .checkpoint import load_training_checkpoint, save_training_checkpoint


class ControlledHydroModel(torch.nn.Module):
    def __init__(self, model_name: str, n_basins: int, *, device: torch.device, dtype: torch.dtype, warm_up: int = 30, uh: bool = False) -> None:
        super().__init__()
        config: dict[str, Any] = {"model_name": model_name, "warm_up": warm_up, "backend": "eager"}
        if uh:
            config["uh_enabled"] = True
            config["uh_mode"] = "intermediate" if model_name in {"flexb", "flexi", "flexis", "gr4j"} else "endpoint"
        self.phy_model = HydrologyModel(config, device=device)
        self.phy_model.to(device=device, dtype=dtype)
        generator = torch.Generator(device="cpu").manual_seed(20260717)
        initial = 0.25 + 0.45 * torch.rand(
            (n_basins, len(self.phy_model.parameter_bounds)), generator=generator, dtype=dtype
        )
        self.raw_parameters = torch.nn.Parameter(initial.to(device))
        self.model_name = model_name
        self.config = {"model_name": model_name, "warm_up": warm_up}

    @property
    def physical_parameters(self) -> dict[str, torch.Tensor]:
        return self.phy_model._descale_params(self.raw_parameters)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.phy_model(x, (None, self.raw_parameters))

    def export_hydrological_state(self) -> dict[str, Any]:
        return {}

    def export_uh_state(self) -> dict[str, Any]:
        return {name: value.detach().cpu() for name, value in self.named_buffers()}

    def export_warmup_state(self) -> dict[str, Any]:
        return {"warm_up": int(self.phy_model.warm_up)}


class ControlledTrainer:
    def __init__(self, model: ControlledHydroModel, *, device: torch.device, loss: torch.nn.Module | None = None, checkpoint_dir: str | Path | None = None) -> None:
        self.model = model
        self.device = device
        self.loss_func = loss or NseBatchLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=2)
        self.global_step = 0
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def step(self, x: dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(x)["streamflow"]
        if not torch.isfinite(output).all():
            raise FloatingPointError("controlled Trainer output contains NaN or Inf")
        n = min(output.shape[0], target.shape[0])
        output = output[-n:]
        target = target[-n:]
        mask = mask[-n:]
        loss = self.loss_func(output, target, mask=mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("controlled Trainer loss contains NaN or Inf")
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.requires_grad]
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("controlled Trainer found a missing gradient")
        if any(not torch.isfinite(gradient).all() for gradient in gradients if gradient is not None):
            raise FloatingPointError("controlled Trainer found a non-finite gradient")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), 1.0).detach().cpu())
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        if any(not torch.isfinite(value).all() for value in self.model.parameters()):
            raise FloatingPointError("controlled Trainer parameter became NaN or Inf")
        update = max(float((value.detach() - before[name]).abs().max().cpu()) for name, value in self.model.named_parameters())
        update_by_parameter = {
            name: (value.detach() - before[name]).abs().detach().cpu()
            for name, value in self.model.named_parameters()
        }
        return {"loss": float(loss.detach().cpu()), "gradient_norm": grad_norm, "parameter_update_max": update, "parameter_update_by_parameter": update_by_parameter, "output": output.detach()}

    def save(self, epoch: int, *, hydrological_states: Any = None, uh_states: Any = None) -> Path:
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not configured")
        return save_training_checkpoint(
            self.checkpoint_dir,
            model=self.model,
            epoch=epoch,
            global_step=self.global_step,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config={"model_name": self.model.model_name, "dataset_manifest_hash": "diagnostic"},
            hydrological_states=hydrological_states,
            uh_states=uh_states,
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        payload = load_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.global_step = int(payload["global_step"])
        return payload


class ReplayStateManager:
    """Reference state manager for exact chunk/resume validation.

    It stores the forcing history and emitted position, then replays the
    causal model to reconstruct the exact state at a chunk boundary.  This is
    intentionally a correctness reference, not a claim of production
    efficiency; it makes UH tail/state loss impossible to hide during gating.
    """

    def __init__(self) -> None:
        self.forcing_history: torch.Tensor | None = None
        self.emitted_steps = 0

    def run(self, model: ControlledHydroModel, forcing: dict[str, torch.Tensor]) -> torch.Tensor:
        x = forcing["x_phy"]
        if self.forcing_history is None:
            combined = x
        else:
            combined = torch.cat((self.forcing_history, x), dim=0)
        output = model({**forcing, "x_phy": combined})["streamflow"]
        warm_up = int(model.phy_model.warm_up)
        start = max(self.emitted_steps, warm_up)
        result = output[max(start - warm_up, 0):]
        self.forcing_history = combined.detach().clone()
        self.emitted_steps = int(combined.shape[0])
        return result

    def state_dict(self) -> dict[str, Any]:
        return {"forcing_history": self.forcing_history, "emitted_steps": self.emitted_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.forcing_history = state.get("forcing_history")
        self.emitted_steps = int(state.get("emitted_steps", 0))
