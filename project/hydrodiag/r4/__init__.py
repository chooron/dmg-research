"""R4 — real-basin snow-state consistency pipeline (development stage).

R4 compares the internal snow states (``snow_pack``/G, ``melt``, ``sca``,
solid precipitation) of R1/R2 observation-trained Base / CN / TGD2 models
against external CAMELS-US Snow-17/SAC-SMA SWE references on real basins.

Pipeline stage tags (must never be mixed):

- ``DEV_ONLY`` / ``SYNTHETIC_TRAINED``: produced from R3 synthetic-q* trained
  checkpoints; pipeline smoke tests only. Never a formal R4 scientific result.
- ``OFFICIAL_OBSERVATION_TRAINED``: produced from R1/R2 observation-trained
  parameters/checkpoints after provenance verification. This is the only tag
  allowed for formal R4 analysis outputs.

See ``r4/README.md`` for the full protocol.
"""

from __future__ import annotations

__version__ = "0.1.0"

DEV_ONLY = "DEV_ONLY"
SYNTHETIC_TRAINED = "SYNTHETIC_TRAINED"
DEV_ONLY_SYNTHETIC_TRAINED = "DEV_ONLY_SYNTHETIC_TRAINED"
OFFICIAL_OBSERVATION_TRAINED = "OFFICIAL_OBSERVATION_TRAINED"
OFFICIAL_DPL_OBSERVATION_TRAINED = "OFFICIAL_DPL_OBSERVATION_TRAINED"
IC_FUSED_5x200_SENSITIVITY = "IC_FUSED_5x200_SENSITIVITY"
