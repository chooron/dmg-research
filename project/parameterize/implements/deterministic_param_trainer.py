"""Deterministic paper trainer."""

from __future__ import annotations

from .my_trainer import MyTrainer


class DeterministicParamTrainer(MyTrainer):
    """Hydro-loss trainer for the deterministic parameter model."""

