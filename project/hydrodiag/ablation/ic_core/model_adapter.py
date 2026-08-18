from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from models import (
    GR4J,
    HBV,
    SIMHYD,
    XAJ,
    GR4JLite,
    GR4JWithCemaNeige,
    GR4JWithCemaNeigeLite,
    GR4JWithPrecipitationDelay,
    GR4JWithPrecipitationDelayLite,
    GR4JWithTGD2,
    GR4JWithTGD2Lite,
    HBVLite,
    SIMHYDLite,
    SIMHYDWithCemaNeige,
    SIMHYDWithCemaNeigeLite,
    SIMHYDWithPrecipitationDelay,
    SIMHYDWithPrecipitationDelayLite,
    SIMHYDWithTGD2,
    SIMHYDWithTGD2Lite,
    XAJ2SWithCemaNeige,
    XAJ2SWithCemaNeigeLite,
    XAJControlledNWithCemaNeige,
    XAJControlledNWithCemaNeigeLite,
    XAJDEWithCemaNeige,
    XAJDEWithCemaNeigeLite,
    XAJDRWithCemaNeige,
    XAJDRWithCemaNeigeLite,
    XAJGEWithCemaNeige,
    XAJGEWithCemaNeigeLite,
    XAJGRWithCemaNeige,
    XAJGRWithCemaNeigeLite,
    XAJLite,
    XAJRWPEWithCemaNeige,
    XAJRWPEWithCemaNeigeLite,
    XAJWithCemaNeige,
    XAJWithCemaNeigeLite,
    XAJWithPrecipitationDelay,
    XAJWithPrecipitationDelayLite,
    XAJWithTGD2,
    XAJWithTGD2Lite,
)

from .parameter_adapter import get_parameter_spec

MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "XAJ": XAJ,
    "N": XAJControlledNWithCemaNeige,
    "D_E": XAJDEWithCemaNeige,
    "G_E": XAJGEWithCemaNeige,
    "D_R": XAJDRWithCemaNeige,
    "G_R": XAJGRWithCemaNeige,
    "XAJ_CN": XAJWithCemaNeige,
    "XAJ_2S": XAJ2SWithCemaNeige,
    "XAJ_RWPE": XAJRWPEWithCemaNeige,
    "XAJ_PD": XAJWithPrecipitationDelay,
    "XAJ_TGD2": XAJWithTGD2,
    "GR4J": GR4J,
    "GR4J_CN": GR4JWithCemaNeige,
    "GR4J_PD": GR4JWithPrecipitationDelay,
    "GR4J_TGD2": GR4JWithTGD2,
    "SIMHYD": SIMHYD,
    "SIMHYD_CN": SIMHYDWithCemaNeige,
    "SIMHYD_PD": SIMHYDWithPrecipitationDelay,
    "SIMHYD_TGD2": SIMHYDWithTGD2,
    "HBV": HBV,
}


LITE_MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "XAJ": XAJLite,
    "N": XAJControlledNWithCemaNeigeLite,
    "D_E": XAJDEWithCemaNeigeLite,
    "G_E": XAJGEWithCemaNeigeLite,
    "D_R": XAJDRWithCemaNeigeLite,
    "G_R": XAJGRWithCemaNeigeLite,
    "XAJ_CN": XAJWithCemaNeigeLite,
    "XAJ_2S": XAJ2SWithCemaNeigeLite,
    "XAJ_RWPE": XAJRWPEWithCemaNeigeLite,
    "XAJ_PD": XAJWithPrecipitationDelayLite,
    "XAJ_TGD2": XAJWithTGD2Lite,
    "GR4J": GR4JLite,
    "GR4J_CN": GR4JWithCemaNeigeLite,
    "GR4J_PD": GR4JWithPrecipitationDelayLite,
    "GR4J_TGD2": GR4JWithTGD2Lite,
    "SIMHYD": SIMHYDLite,
    "SIMHYD_CN": SIMHYDWithCemaNeigeLite,
    "SIMHYD_PD": SIMHYDWithPrecipitationDelayLite,
    "SIMHYD_TGD2": SIMHYDWithTGD2Lite,
    "HBV": HBVLite,
}


def model_variant_inventory() -> list[dict[str, str | bool]]:
    """Return the explicit model-to-variant mapping used by the ablation layer."""
    return [
        {
            "model_key": key,
            "full_class": MODEL_CLASSES[key].__name__,
            "lite_class": LITE_MODEL_CLASSES[key].__name__,
            "native_lite": True,
            "lite_semantics": "same equations, compact streamflow-only output when return_states=False",
        }
        for key in MODEL_CLASSES
    ]


def _forcing_dict(
    forcing: torch.Tensor, forcing_names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    if forcing.ndim == 2:
        forcing = forcing.unsqueeze(0)
    if forcing.ndim != 3:
        raise ValueError(
            f"forcing must have shape [T,F] or [B,T,F], got {tuple(forcing.shape)}"
        )
    if len(forcing_names) != forcing.shape[-1]:
        raise ValueError("forcing_names do not match forcing feature dimension")
    indexes = {name: forcing_names.index(name) for name in forcing_names}
    required = {"P", "T", "PET"}
    if set(indexes) != required:
        raise ValueError(f"expected forcing names P,T,PET, got {forcing_names}")
    return {
        "precip": forcing[:, :, indexes["P"]],
        "temp": forcing[:, :, indexes["T"]],
        "pet": forcing[:, :, indexes["PET"]],
    }


class ModelAdapter:
    def __init__(
        self,
        model_key: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        variant: str = "full",
    ):
        if model_key not in MODEL_CLASSES:
            raise KeyError(f"Unknown IC model key: {model_key}")
        if variant not in {"full", "lite"}:
            raise ValueError(f"Unknown model variant: {variant}")
        self.model_key = model_key
        self.variant = variant
        self.device = torch.device(device)
        self.dtype = dtype
        model_class = (
            MODEL_CLASSES[model_key]
            if variant == "full"
            else LITE_MODEL_CLASSES[model_key]
        )
        self.model_class = model_class
        self.model = model_class().to(device=self.device, dtype=dtype).eval()
        self.parameter_names = tuple(get_parameter_spec(model_key))

    def run_model(
        self,
        forcing: torch.Tensor,
        physical_parameters: torch.Tensor,
        *,
        forcing_names: tuple[str, ...] = ("P", "T", "PET"),
        temp_mean_train: torch.Tensor | None = None,
        temp_std_train: torch.Tensor | None = None,
        cn_psol_annual: torch.Tensor | None = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if physical_parameters.ndim == 1:
            physical_parameters = physical_parameters.unsqueeze(0)
        if physical_parameters.ndim != 2 or physical_parameters.shape[-1] != len(
            self.parameter_names
        ):
            raise ValueError("physical_parameters must have shape [N,D]")
        forcing = forcing.to(device=self.device, dtype=self.dtype)
        if forcing.ndim == 2:
            forcing = forcing.unsqueeze(0)
        n_forcing = forcing.shape[0]
        n_params = physical_parameters.shape[0]
        if n_forcing == 1 and n_params > 1:
            forcing = forcing.expand(n_params, -1, -1)
        elif n_forcing != n_params:
            if n_params % n_forcing != 0:
                raise ValueError(
                    f"forcing batch {n_forcing} cannot align with parameter batch {n_params}"
                )
            forcing = forcing.repeat_interleave(n_params // n_forcing, dim=0)
        model_forcing = _forcing_dict(forcing, forcing_names)
        if temp_mean_train is not None:
            temp_mean_train = temp_mean_train.to(
                device=self.device, dtype=self.dtype
            ).reshape(-1)
            if temp_mean_train.shape[0] == 1 and n_params > 1:
                temp_mean_train = temp_mean_train.expand(n_params)
            elif temp_mean_train.shape[0] != n_params:
                temp_mean_train = temp_mean_train.repeat_interleave(
                    n_params // temp_mean_train.shape[0]
                )
            model_forcing["temp_mean_train"] = temp_mean_train
        if temp_std_train is not None:
            temp_std_train = temp_std_train.to(
                device=self.device, dtype=self.dtype
            ).reshape(-1)
            if temp_std_train.shape[0] == 1 and n_params > 1:
                temp_std_train = temp_std_train.expand(n_params)
            elif temp_std_train.shape[0] != n_params:
                temp_std_train = temp_std_train.repeat_interleave(
                    n_params // temp_std_train.shape[0]
                )
            model_forcing["temp_std_train"] = temp_std_train
        if cn_psol_annual is not None:
            cn_psol_annual = cn_psol_annual.to(
                device=self.device, dtype=self.dtype
            ).reshape(-1)
            if cn_psol_annual.shape[0] == 1 and n_params > 1:
                cn_psol_annual = cn_psol_annual.expand(n_params)
            elif cn_psol_annual.shape[0] != n_params:
                cn_psol_annual = cn_psol_annual.repeat_interleave(
                    n_params // cn_psol_annual.shape[0]
                )
            model_forcing["cn_psol_annual"] = cn_psol_annual
        params = {
            name: physical_parameters[:, index].to(device=self.device, dtype=self.dtype)
            for index, name in enumerate(self.parameter_names)
        }
        with torch.no_grad():
            return self.model(
                forcings=model_forcing, params=params, return_states=return_states
            )
