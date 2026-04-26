from __future__ import annotations

import copy
from importlib import import_module
from typing import Any

import torch


PHY_MODEL_MODULES: dict[str, str] = {
    "Exphydro": "project.bettermodel.implements.phy_models.exphydro",
    "Hbv": "project.bettermodel.implements.phy_models.hbv",
    "Hbv_2": "project.bettermodel.implements.phy_models.hbv_2",
    "Hbv_2f": "project.bettermodel.implements.phy_models.hbv_2f",
    "Hbv_2a": "project.bettermodel.implements.phy_models.hbv_2a",
    "HbvMoeV1": "project.bettermodel.implements.phy_models.hbv_moe_v1",
    "Hmets": "project.bettermodel.implements.phy_models.hmets",
    "Hmets_2": "project.bettermodel.implements.phy_models.hmets_2",
    "Identity": "project.bettermodel.implements.phy_models.identity",
    "Unify": "project.bettermodel.implements.phy_models.unify",
    "UnifyV2": "project.bettermodel.implements.phy_models.unify_v2",
    "hmets": "project.bettermodel.implements.phy_models.hmets_old",
}

NN_MODEL_MODULES: dict[str, str] = {
    "AnnModel": "project.bettermodel.implements.neural_networks.ann",
    "GruMlpModel": "project.bettermodel.implements.neural_networks.gru_mlp",
    "Hope": "project.bettermodel.implements.neural_networks.hope_v1",
    "HopeMlpV1": "project.bettermodel.implements.neural_networks.hope_mlp_v1",
    "HopeMlpV2": "project.bettermodel.implements.neural_networks.hope_mlp_v2",
    "HopeMlpV3": "project.bettermodel.implements.neural_networks.hope_mlp_v3",
    "HopeV1": "project.bettermodel.implements.neural_networks.hope_v1",
    "LstmMlpModel": "project.bettermodel.implements.neural_networks.lstm_mlp",
    "S4DBaseline": "project.bettermodel.implements.neural_networks.ablation",
    "S4DLN": "project.bettermodel.implements.neural_networks.ablation",
    "S4DLNSoftsign": "project.bettermodel.implements.neural_networks.ablation",
    "S4DSoftsign": "project.bettermodel.implements.neural_networks.ablation",
    "S5DConvOnly": "project.bettermodel.implements.neural_networks.ablation",
    "S5DFull": "project.bettermodel.implements.neural_networks.ablation",
    "LstmStaticModel": "project.bettermodel.implements.neural_networks.lstm_static",
    "TSMixerMlpModel": "project.bettermodel.implements.neural_networks.tsmixer_mlp",
    "TcnMlpModel": "project.bettermodel.implements.neural_networks.tcn_mlp_v2",
    "TcnMlpV1Model": "project.bettermodel.implements.neural_networks.tcn_mlp_v1",
    "VanillaTransformerMlpModel": "project.bettermodel.implements.neural_networks.vanilla_transformer_mlp",
}


def get_phy_model_names(config: dict[str, Any]) -> list[str]:
    phy_names = config["model"].get("phy", {}).get("name", [])
    if not phy_names:
        raise ValueError("LocalModelHandler requires config['model']['phy']['name'].")
    return phy_names if isinstance(phy_names, list) else [phy_names]


def _resolve_model_class(
    component_name: str,
    *,
    registry: dict[str, str],
    kind: str,
) -> type[torch.nn.Module]:
    module_path = registry.get(component_name)
    if module_path is None:
        available = ", ".join(sorted(registry))
        raise ImportError(f"Unknown bettermodel {kind} '{component_name}'. Available: {available}")

    module = import_module(module_path)
    component_class = getattr(module, component_name, None)
    if component_class is None:
        raise ImportError(f"Module '{module_path}' does not define '{component_name}'.")
    return component_class


def build_phy_config(config: dict[str, Any]) -> dict[str, Any]:
    phy_config = copy.deepcopy(config["model"]["phy"])
    phy_config.setdefault("variables", phy_config.get("forcings", []))
    return phy_config


def build_nn_config(
    config: dict[str, Any],
    phy_model: torch.nn.Module,
) -> dict[str, Any]:
    nn_config = copy.deepcopy(config["model"]["nn"])
    forcings = nn_config.get("forcings", [])
    attributes = nn_config.get("attributes", [])
    sequence_input_size = len(forcings) + len(attributes)
    nn_config.setdefault("nx", sequence_input_size)
    nn_config.setdefault("nx1", sequence_input_size)
    nn_config.setdefault("nx2", len(attributes))
    nn_config.setdefault("ny1", getattr(phy_model, "learnable_param_count1", 0))
    nn_config.setdefault("ny2", getattr(phy_model, "learnable_param_count2", 0))
    nn_config.setdefault(
        "ny",
        getattr(phy_model, "learnable_param_count", nn_config["ny1"] + nn_config["ny2"]),
    )
    nn_config.setdefault(
        "hidden_size",
        nn_config.get("hope_hidden_size", nn_config.get("mlp_hidden_size")),
    )
    nn_config.setdefault(
        "dropout",
        nn_config.get("hope_dropout", nn_config.get("mlp_dropout", 0.0)),
    )
    nn_config.setdefault("nmul", config["model"]["phy"].get("nmul", 1))
    return nn_config


def build_phy_model(
    config: dict[str, Any],
    phy_model_name: str,
    *,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    phy_class = _resolve_model_class(
        phy_model_name,
        registry=PHY_MODEL_MODULES,
        kind="physics model",
    )
    return phy_class(build_phy_config(config), device=device)


def build_nn_model(
    config: dict[str, Any],
    phy_model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    nn_name = config["model"]["nn"]["name"]
    nn_class = _resolve_model_class(
        nn_name,
        registry=NN_MODEL_MODULES,
        kind="neural model",
    )
    nn_config = build_nn_config(config, phy_model)
    if hasattr(nn_class, "build_by_config"):
        return nn_class.build_by_config(nn_config, device)
    raise NotImplementedError(
        f"Local NN model '{nn_name}' does not implement build_by_config().",
    )
