"""Seven-parameter SIMHYD runoff generation without unit-hydrograph routing.

This is the two-store formulation from
``project/hydro_structure_diagnosis/models/simhyd.py`` with its optional PET
multiplier fixed to one and its Gamma unit hydrograph removed.  It therefore
keeps the fixed 7-parameter calibration contract while avoiding an additional
routing model and its two parameters.
"""
from __future__ import annotations

from typing import Tuple

import torch


SIMHYD_PARAMS_BOUNDS = {
    "insc": [0.0, 5.0],
    "coeff": [0.0, 600.0],
    "sq": [0.0, 15.0],
    "smsc": [1.0, 2000.0],
    "sub": [0.0, 1.0],
    "crak": [0.0, 1.0],
    "k": [0.0, 1.0],
}

SIMHYD_PARAMS_DESC = {
    "insc": "Same-day interception and evaporation capacity [mm]",
    "coeff": "Maximum infiltration capacity [mm]",
    "sq": "Infiltration capacity exponent [-]",
    "smsc": "Maximum soil moisture capacity [mm]",
    "sub": "Interflow proportionality constant [-]",
    "crak": "Recharge proportionality constant [-]",
    "k": "Groundwater recession coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create the reference variant's soil and groundwater stores.

    The production path repeats five water years as warm-up, so these neutral
    initial states are equilibrated before any calibrated target is scored.
    """
    soil = torch.full((n_grid, nmul), nearzero, device=device)
    groundwater = torch.zeros((n_grid, nmul), device=device)
    return soil, groundwater


def simhyd_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    insc: torch.Tensor,
    coeff: torch.Tensor,
    sq: torch.Tensor,
    smsc: torch.Tensor,
    sub: torch.Tensor,
    crak: torch.Tensor,
    k: torch.Tensor,
    soil: torch.Tensor,
    groundwater: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One mass-conserving daily step of the no-UH 7p SIMHYD variant.

    ``T`` is accepted for the common model interface but is not part of this
    rainfall/PET model.  The equations follow the diagnosis implementation:
    interception is same-day evaporation capped by precipitation/PET/capacity;
    soil overflow recharges groundwater; and baseflow is released from the
    step-initial groundwater store.  No Gamma unit hydrograph is applied.
    """
    del T
    precip = torch.clamp(P, min=0.0)
    pet = torch.clamp(PET, min=0.0)
    insc_safe = torch.clamp(insc, min=nearzero)
    smsc_safe = torch.clamp(smsc, min=nearzero)
    coeff_safe = torch.clamp(coeff, min=nearzero)

    interception = torch.minimum(torch.minimum(insc_safe, pet), precip)
    rainfall_excess = precip - interception
    pet_remaining = pet - interception

    soil_ratio = torch.clamp(soil / (smsc_safe + nearzero), min=0.0, max=1.0)
    infiltration_capacity = coeff_safe * torch.exp(-torch.clamp(sq, min=0.0) * soil_ratio)
    infiltration = torch.minimum(infiltration_capacity, rainfall_excess)
    direct_runoff = rainfall_excess - infiltration

    interflow = torch.clamp(sub, min=0.0, max=1.0) * soil_ratio * infiltration
    recharge = torch.clamp(crak, min=0.0, max=1.0) * soil_ratio * (infiltration - interflow)
    soil_fill = infiltration - interflow - recharge

    soil_available = torch.clamp(soil + soil_fill, min=nearzero)
    soil_evaporation = torch.minimum(10.0 * soil_ratio, pet_remaining)
    soil_evaporation = torch.minimum(soil_evaporation, soil_available)
    soil_after_evaporation = soil_available - soil_evaporation
    soil_overflow = torch.clamp(soil_after_evaporation - smsc_safe, min=0.0)
    soil_new = soil_after_evaporation - soil_overflow

    recharge_total = recharge + soil_overflow
    baseflow = torch.clamp(k, min=0.0, max=1.0) * groundwater
    groundwater_new = groundwater + recharge_total - baseflow

    streamflow = direct_runoff + interflow + baseflow
    evaporation = interception + soil_evaporation
    return streamflow, evaporation, soil_new, groundwater_new
