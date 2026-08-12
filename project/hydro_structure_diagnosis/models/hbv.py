from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import BaseHydrologicalModel
from .parameter_specs import HBV_PARAM_SPECS
from .utils import validate_forcings, validate_params


def _hbv_step(
    precip_t: torch.Tensor,
    temp_t: torch.Tensor,
    pet_t: torch.Tensor,
    SNOWPACK: torch.Tensor,
    MELTWATER: torch.Tensor,
    SM: torch.Tensor,
    SUZ: torch.Tensor,
    SLZ: torch.Tensor,
    parTT: torch.Tensor,
    parCFMAX: torch.Tensor,
    parCFR: torch.Tensor,
    parCWH: torch.Tensor,
    parFC: torch.Tensor,
    parBETA: torch.Tensor,
    parLP: torch.Tensor,
    parPERC: torch.Tensor,
    parUZL: torch.Tensor,
    parK0: torch.Tensor,
    parK1: torch.Tensor,
    parK2: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    RAIN = precip_t * (temp_t >= parTT).float()
    SNOW = precip_t * (temp_t < parTT).float()
    SNOWPACK = SNOWPACK + SNOW
    melt = torch.clamp(parCFMAX * (temp_t - parTT), min=0.0)
    melt = torch.min(melt, SNOWPACK)
    MELTWATER = MELTWATER + melt
    SNOWPACK = SNOWPACK - melt
    refreezing = torch.clamp(parCFR * parCFMAX * (parTT - temp_t), min=0.0)
    refreezing = torch.min(refreezing, MELTWATER)
    SNOWPACK = SNOWPACK + refreezing
    MELTWATER = MELTWATER - refreezing
    tosoil = torch.clamp(MELTWATER - parCWH * SNOWPACK, min=0.0)
    MELTWATER = MELTWATER - tosoil

    soil_wetness = torch.clamp((SM / parFC) ** parBETA, 0.0, 1.0)
    recharge = (RAIN + tosoil) * soil_wetness
    SM = SM + RAIN + tosoil - recharge
    excess = torch.clamp(SM - parFC, min=0.0)
    SM = SM - excess
    evapfactor = torch.clamp(SM / (parLP * parFC), 0.0, 1.0)
    ETact = torch.min(SM, pet_t * evapfactor)
    SM = torch.clamp(SM - ETact, min=nearzero)

    SUZ = SUZ + recharge + excess
    perc = torch.min(SUZ, parPERC)
    SUZ = SUZ - perc
    Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
    SUZ = SUZ - Q0
    Q1 = parK1 * SUZ
    SUZ = SUZ - Q1
    SLZ = SLZ + perc
    Q2 = parK2 * SLZ
    SLZ = SLZ - Q2

    q_t = Q0 + Q1 + Q2
    return q_t, SNOWPACK, MELTWATER, SM, SUZ, SLZ


class HBV(BaseHydrologicalModel):
    """HBV 1.0 model with snow module included.

    Implements the HBV model as described in:
    - Bergstrom, S. (1995). The HBV model.
    - Seibert, J. (2005). HBV-light version 2.

    Refs for PyTorch translation:
    - Feng et al. (2022), hydroDL2
    """

    _STATE_DIM = 5

    def __init__(self, nearzero: float = 1e-8, compact_output: bool = False):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self._step = torch.compile(_hbv_step, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return HBV_PARAM_SPECS

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        precip, pet, temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)

        SNOWPACK, MELTWATER, SM, SUZ, SLZ = self._init_states(batch, device, dtype, initial_states)

        parTT = params["parTT"]
        parCFMAX = params["parCFMAX"]
        parCFR = params["parCFR"]
        parCWH = params["parCWH"]
        parFC = params["parFC"]
        parBETA = params["parBETA"]
        parLP = params["parLP"]
        parPERC = params["parPERC"]
        parUZL = params["parUZL"]
        parK0 = params["parK0"]
        parK1 = params["parK1"]
        parK2 = params["parK2"]

        qsim, (SNOWPACK, MELTWATER, SM, SUZ, SLZ) = self._step_loop(
            precip, pet, temp, nsteps, batch,
            SNOWPACK, MELTWATER, SM, SUZ, SLZ,
            parTT, parCFMAX, parCFR, parCWH,
            parFC, parBETA, parLP, parPERC, parUZL,
            parK0, parK1, parK2, device,
        )

        if self.compact_output and not return_states:
            return qsim, {}

        aux = {
            "SWE": SNOWPACK,
            "soil_moisture": SM,
            "upper_zone": SUZ,
            "lower_zone": SLZ,
        }
        if return_states:
            aux["final_states"] = {
                "SNOWPACK": SNOWPACK,
                "MELTWATER": MELTWATER,
                "SM": SM,
                "SUZ": SUZ,
                "SLZ": SLZ,
            }

        return qsim, aux

    def _init_states(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        initial_states: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, ...]:
        if initial_states is not None:
            return (
                initial_states.get("SNOWPACK", torch.zeros(batch, device=device, dtype=dtype)),
                initial_states.get("MELTWATER", torch.zeros(batch, device=device, dtype=dtype)),
                initial_states.get("SM", torch.full((batch,), 0.5, device=device, dtype=dtype)),
                initial_states.get("SUZ", torch.zeros(batch, device=device, dtype=dtype)),
                initial_states.get("SLZ", torch.zeros(batch, device=device, dtype=dtype)),
            )
        return (
            torch.zeros(batch, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=dtype),
            torch.full((batch,), 0.5, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=dtype),
        )

    def _step_loop(
        self,
        precip: torch.Tensor,
        pet: torch.Tensor,
        temp: torch.Tensor,
        nsteps: int,
        batch: int,
        SNOWPACK: torch.Tensor,
        MELTWATER: torch.Tensor,
        SM: torch.Tensor,
        SUZ: torch.Tensor,
        SLZ: torch.Tensor,
        parTT: torch.Tensor,
        parCFMAX: torch.Tensor,
        parCFR: torch.Tensor,
        parCWH: torch.Tensor,
        parFC: torch.Tensor,
        parBETA: torch.Tensor,
        parLP: torch.Tensor,
        parPERC: torch.Tensor,
        parUZL: torch.Tensor,
        parK0: torch.Tensor,
        parK1: torch.Tensor,
        parK2: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        qsim = torch.zeros(batch, nsteps, device=device, dtype=precip.dtype)
        nz = self.nearzero

        for t in range(nsteps):
            qsim[:, t], SNOWPACK, MELTWATER, SM, SUZ, SLZ = self._step(
                precip[:, t],
                temp[:, t],
                pet[:, t],
                SNOWPACK,
                MELTWATER,
                SM,
                SUZ,
                SLZ,
                parTT,
                parCFMAX,
                parCFR,
                parCWH,
                parFC,
                parBETA,
                parLP,
                parPERC,
                parUZL,
                parK0,
                parK1,
                parK2,
                nz,
            )

        return qsim, (SNOWPACK, MELTWATER, SM, SUZ, SLZ)


class HBVLite(HBV):
    """HBV training path returning only streamflow."""

    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)
