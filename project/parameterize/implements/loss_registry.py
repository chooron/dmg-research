"""Parameterize-local loss resolver."""

from __future__ import annotations

from typing import Any

from dmg.models.criterion.kge_batch_loss import KgeBatchLoss

from .losses import HybridNseBatchLoss, LogNseBatchLoss, NseBatchLoss

_LOSS_REGISTRY = {
    "KgeBatchLoss": KgeBatchLoss,
    "NseBatchLoss": NseBatchLoss,
    "LogNseBatchLoss": LogNseBatchLoss,
    "HybridNseBatchLoss": HybridNseBatchLoss,
}


def resolve_loss_class(loss_name: str):
    loss_cls = _LOSS_REGISTRY.get(loss_name)
    if loss_cls is None:
        raise ValueError(f"Unknown loss '{loss_name}'. Available: {list(_LOSS_REGISTRY)}")
    return loss_cls


def build_loss_function(config: dict[str, Any], **kwargs: Any):
    loss_name = config["train"]["loss_function"]["name"]
    return resolve_loss_class(loss_name)(config, device=config["device"], **kwargs)
