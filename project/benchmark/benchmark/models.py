"""Thin dmotpy model construction helpers."""

from __future__ import annotations

import torch

from dmotpy.models import HydrologyModel
from dmotpy.models.registry import PARAM_INFO


def available_model_ids() -> list[str]:
    return sorted(PARAM_INFO.keys())


def build_hydrology_model(config: dict, model_id: str, device: str) -> HydrologyModel:
    model_cfg = config.get("model", {})
    return HydrologyModel(
        config={
            "model_name": model_id,
            "warm_up": int(model_cfg.get("warm_up", 365)),
            "warm_up_states": bool(model_cfg.get("warm_up_states", True)),
            "variables": list(model_cfg.get("forcings", ["prcp", "tmean", "pet"])),
            "nearzero": float(model_cfg.get("nearzero", 1e-5)),
            "backend": model_cfg.get("backend", "eager"),
        },
        device=torch.device(device),
        backend=model_cfg.get("backend", "eager"),
    )
