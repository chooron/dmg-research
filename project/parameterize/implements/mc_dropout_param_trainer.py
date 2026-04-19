"""MC-dropout paper trainer."""

from __future__ import annotations

from .my_trainer import MyTrainer


class McDropoutParamTrainer(MyTrainer):
    """Hydro-loss trainer that keeps MC-dropout evaluation policy enabled."""

