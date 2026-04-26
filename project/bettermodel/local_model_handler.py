from __future__ import annotations

import os
from typing import Any, Optional

import torch

from dmg.core.utils import save_model as save_model_state

from project.bettermodel.model_builder import (
    build_nn_model,
    build_phy_model,
    get_phy_model_names,
)


class LocalDplModel(torch.nn.Module):
    """Local differentiable model wrapper for bettermodel custom components."""

    def __init__(
        self,
        *,
        phy_model: torch.nn.Module,
        nn_model: torch.nn.Module,
        config: dict[str, Any],
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = "Local BetterModel DPL"
        self.phy_model = phy_model
        self.nn_model = nn_model
        self.config = config
        self.device = torch.device(device or config.get("device", "cpu"))
        self.static_feature_count = len(config.get("nn", {}).get("attributes", []))
        self.phy_model.to(self.device)
        self.nn_model.to(self.device)

    def forward(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "c_nn_norm" not in data_dict and "xc_nn_norm" in data_dict and self.static_feature_count > 0:
            data_dict = dict(data_dict)
            data_dict["c_nn_norm"] = data_dict["xc_nn_norm"][0, :, -self.static_feature_count :]
        elif "c_nn_norm" not in data_dict and "c_nn" in data_dict:
            data_dict = dict(data_dict)
            data_dict["c_nn_norm"] = data_dict["c_nn"]
        try:
            parameters = self.nn_model(data_dict)
        except TypeError:
            parameters = self.nn_model(data_dict["xc_nn_norm"], data_dict["c_nn_norm"])
        return self.phy_model(data_dict, parameters)


class LocalModelHandler(torch.nn.Module):
    """Minimal ModelHandler-compatible wrapper using local bettermodel components."""

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.name = "BetterModel Local Handler"
        self.verbose = verbose
        self.device = device or config["device"]
        self.model_path = config["model_dir"]
        self.target_names = config["train"]["target"]
        self.model_dict: dict[str, LocalDplModel] = {}
        self.output_dict: dict[str, dict[str, torch.Tensor]] = {}

        self.models = get_phy_model_names(config)
        self.loss_func = None
        self.loss_dict = dict.fromkeys(self.models, 0.0)

        if config["mode"] == "train":
            load_epoch = int(config["train"].get("start_epoch", 0))
        elif config["mode"] in {"test", "sim"}:
            load_epoch = int(config["test"].get("test_epoch", 0))
        else:
            load_epoch = int(config.get("load_epoch", 0))

        self._init_models(load_epoch)

    def _build_phy_model(self, phy_model_name: str) -> torch.nn.Module:
        return build_phy_model(self.config, phy_model_name, device=self.device)

    def _build_nn_model(self, phy_model: torch.nn.Module) -> torch.nn.Module:
        return build_nn_model(self.config, phy_model, device=self.device)

    def _init_models(self, load_epoch: int) -> None:
        for name in self.models:
            phy_model = self._build_phy_model(name)
            nn_model = self._build_nn_model(phy_model)
            model = LocalDplModel(
                phy_model=phy_model,
                nn_model=nn_model,
                config=self.config["model"],
                device=self.device,
            )
            self.model_dict[name] = model

        if load_epoch > 0:
            self.load_model(load_epoch)

    def train(self, mode: bool = True) -> "LocalModelHandler":
        super().train(mode)
        for model in self.model_dict.values():
            model.train(mode)
        return self

    def get_parameters(self) -> list[torch.Tensor]:
        parameters: list[torch.Tensor] = []
        for model in self.model_dict.values():
            parameters.extend(list(model.parameters()))
        return parameters

    def forward(
        self,
        dataset_dict: dict[str, torch.Tensor],
        eval: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        self.output_dict = {}
        for name, model in self.model_dict.items():
            if eval:
                model.eval()
                with torch.no_grad():
                    self.output_dict[name] = model(dataset_dict)
            else:
                model.train()
                self.output_dict[name] = model(dataset_dict)
        return self.output_dict

    def calc_loss(
        self,
        dataset_dict: dict[str, torch.Tensor],
        loss_func: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        if self.loss_func is None and loss_func is None:
            raise ValueError("No loss function defined.")
        criterion = loss_func or self.loss_func
        assert criterion is not None

        loss_combined: Optional[torch.Tensor] = None
        for name, output in self.output_dict.items():
            if self.target_names[0] not in output:
                raise ValueError(f"Target variable '{self.target_names[0]}' not in model outputs.")
            model_output = output[self.target_names[0]]
            model_output, target = self._trim(model_output, dataset_dict["target"])
            loss = criterion(
                model_output.squeeze(),
                target.squeeze(),
                sample_ids=dataset_dict["batch_sample"],
            )
            loss_combined = loss if loss_combined is None else loss_combined + loss
            self.loss_dict[name] += float(loss.item())
        if loss_combined is None:
            raise RuntimeError("No model outputs were available for loss computation.")
        return loss_combined

    def save_model(self, epoch: int) -> None:
        for name, model in self.model_dict.items():
            save_model_state(self.config["model_dir"], model, name, epoch)

    def load_model(self, epoch: int = 0) -> None:
        for name, model in self.model_dict.items():
            path = os.path.join(self.model_path, f"{name.lower()}_ep{epoch}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found for model {name}.")
            model.load_state_dict(
                torch.load(path, weights_only=True, map_location=self.device),
                strict=False,
            )

    def _trim(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output.ndim == 3 and output.shape[-1] == 1:
            output = output[:, :, 0]
        if target.ndim == 3 and target.shape[-1] == 1:
            target = target[:, :, 0]

        output_steps = output.shape[0]
        target_steps = target.shape[0]

        if target_steps > output_steps:
            target = target[-output_steps:, ...]
        elif output_steps > target_steps:
            output = output[-target_steps:, ...]
        return output, target
