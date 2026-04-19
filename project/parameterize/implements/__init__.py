"""Parameterize-local implementation boundary for the paper stack."""

from __future__ import annotations

from typing import Any

from .deterministic_param_trainer import DeterministicParamTrainer
from .distributional_param_trainer import DistributionalParamTrainer
from .hbv_static import HbvStatic
from .mc_dropout_param_trainer import McDropoutParamTrainer
from .mc_mlp import McMlpModel
from .my_dpl_model import MyDplModel
from .my_trainer import MyTrainer
from .param_models import DeterministicParamModel, DistributionalParamModel


def build_paper_dpl(config: dict[str, Any]) -> MyDplModel:
    """Build the parameterize-local DPL model for the paper stack."""
    phy_cfg = config["model"]["phy"]
    nn_cfg = config["model"]["nn"]
    phy_model = HbvStatic(config=phy_cfg, device=config["device"])
    nx = len(nn_cfg["forcings"]) + len(nn_cfg["attributes"])
    ny = phy_model.learnable_param_count
    nn_name = str(nn_cfg.get("name", ""))

    if nn_name == "DeterministicParamModel":
        nn_model = DeterministicParamModel(nn_cfg, nx=nx, ny=ny)
    elif nn_name in {"McMlpModel", "McMlp", "mc_mlp"}:
        nn_model = McMlpModel(nn_cfg, nx=nx, ny=ny)
    elif nn_name == "DistributionalParamModel":
        nn_model = DistributionalParamModel(nn_cfg, nx=nx, ny=ny)
    else:
        raise ValueError(f"Unsupported paper nn model '{nn_name}'.")

    return MyDplModel(
        phy_model=phy_model,
        nn_model=nn_model,
        config=config,
        device=config["device"],
    )


def build_paper_trainer(config: dict[str, Any], **kwargs: Any) -> MyTrainer:
    """Dispatch the paper trainer from ``paper.variant``."""
    variant = str(config.get("paper", {}).get("variant", "")).lower()
    trainer_map = {
        "deterministic": DeterministicParamTrainer,
        "mc_dropout": McDropoutParamTrainer,
        "distributional": DistributionalParamTrainer,
    }
    trainer_cls = trainer_map.get(variant)
    if trainer_cls is None:
        raise ValueError(f"Unsupported paper.variant '{variant}'.")
    return trainer_cls(config=config, **kwargs)


__all__ = [
    "DeterministicParamModel",
    "DeterministicParamTrainer",
    "DistributionalParamModel",
    "DistributionalParamTrainer",
    "HbvStatic",
    "McDropoutParamTrainer",
    "McMlpModel",
    "MyDplModel",
    "MyTrainer",
    "build_paper_dpl",
    "build_paper_trainer",
]
