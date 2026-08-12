"""Compatibility exports for older imports.

Prefer importing the specialized modules directly:
- endpoint_uh_model.py
- intermediate_uh_model.py
- gr4j_uh_model.py
- mopex_doy_model.py
"""

from .endpoint_uh_model import ENDPOINT_UH_SCHEMES, EndpointUHModel
from .gr4j_uh_model import GR4JUHModel
from .intermediate_uh_model import INTERMEDIATE_UH_CONFIG, IntermediateUHModel
from .mopex_doy_model import MopexDoyModel
from .tcm_model import TCMModel

__all__ = [
    "ENDPOINT_UH_SCHEMES",
    "INTERMEDIATE_UH_CONFIG",
    "EndpointUHModel",
    "IntermediateUHModel",
    "GR4JUHModel",
    "MopexDoyModel",
    "TCMModel",
]
