"""Physical model interfaces for dmot (old-models)."""

from .registry import INIT_INFO, NPARAM_INFO, NUMBER_INFO, PARAM_INFO, STATE_INFO, STFN_INFO
from .hydrology_model import HydrologyModel, _maybe_compile

__all__ = [
    "INIT_INFO",
    "NPARAM_INFO",
    "NUMBER_INFO",
    "PARAM_INFO",
    "STATE_INFO",
    "STFN_INFO",
    "HydrologyModel",
    "_maybe_compile",
]
