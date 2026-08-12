from __future__ import annotations
from .base import OptimizerAdapter

_REGISTRY: dict[str, type[OptimizerAdapter]] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_optimizer_class(name: str) -> type[OptimizerAdapter]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown optimizer: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]

def list_optimizers() -> list[str]:
    return list(_REGISTRY.keys())

from . import xnes  # noqa: F401, E402
from . import cmaes, snes, cem, pgpe, genetic_algorithm  # noqa: F401, E402
