"""Seven-parameter SIMHYD with the canonical 3-store structure (m_18-like).

Restored from the historical 3-store implementation (``fb8904d``) that
participated in the pymarrmot cross-check, following the healthy dMoT
process-order conventions:

- S1 interception store: rainfall is partitioned into interception excess
  (throughfall) first, then interception evaporation is drawn from the
  same-day intermediate state ``S1_tmp = S1 + P - EXC``.
- S2 soil-moisture store: infiltration / interflow / recharge / saturation
  excess are evaluated as a sequential incoming-water partition of the
  throughfall, then soil ET is drawn from the intermediate state
  ``S2_tmp = S2 + SMF - GWF`` (residual-state depletion, ET after slow
  fluxes -- same convention as mopex2/xinanjiang/alpine1).
- S3 groundwater store: baseflow is drawn from the same-day intermediate
  state ``S3_tmp = S3 + REC + GWF`` (downstream stores receive same-day
  upstream transfer before their own fluxes).

Numerical-stability adaptations (inherited from the shared flux helpers,
none change the hydrologic equations):

- ``infiltration_1`` clamps the exponent to [-30, 0] and guards ``smsc``;
- ``interflow_1`` / ``recharge_1`` / ``evap_2`` guard the ``S/Smax`` ratio.

No proportional flux rescaling is used; competing outflows from a store are
resolved by the model-specific priority above, each flux capped against the
current residual state.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from ..flux.evap import evap_1, evap_2
from ..flux.interception import interception_1
from ..flux.infiltration import infiltration_1
from ..flux.interflow import interflow_1
from ..flux.recharge import recharge_1
from ..flux.saturation import saturation_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_18_simhyd_7p_3s)
SIMHYD_PARAMS_BOUNDS = {
    "insc": [0.0, 5.0],  # Maximum interception capacity [mm]
    "coeff": [0.0, 600.0],  # Maximum infiltration loss parameter [mm]
    "sq": [0.0, 15.0],  # Infiltration loss exponent [-]
    "smsc": [1.0, 2000.0],  # Maximum soil moisture capacity [mm]
    "sub": [0.0, 1.0],  # Proportionality constant for interflow [-]
    "crak": [0.0, 1.0],  # Proportionality constant for recharge [-]
    "k": [0.0, 1.0],  # Slow flow time scale [d-1]
}

# Parameter description dictionary
SIMHYD_PARAMS_DESC = {
    "insc": "Maximum interception capacity [mm]",
    "coeff": "Maximum infiltration loss parameter [mm]",
    "sq": "Infiltration loss exponent [-]",
    "smsc": "Maximum soil moisture capacity [mm]",
    "sub": "Proportionality constant for interflow [-]",
    "crak": "Proportionality constant for recharge [-]",
    "k": "Slow flow time scale [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for SimHyd model.
    S1: Interception store
    S2: Soil moisture store
    S3: Groundwater store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3


def simhyd_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching SIMHYD_PARAMS_BOUNDS keys
    insc: torch.Tensor,
    coeff: torch.Tensor,
    sq: torch.Tensor,
    smsc: torch.Tensor,
    sub: torch.Tensor,
    crak: torch.Tensor,
    k: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    SimHyd model single-step calculation (canonical 3-store structure).

    Model reference:
    Chiew, F. H. S., Peel, M. C., & Western, A. W. (2002). Application and
    testing of the simple rainfall-runoff model SIMHYD.
    """

    # --- 1. Interception Process (S1) ---
    # flux_EXC: Excess rainfall after interception (throughfall / input
    # partition of P by the interception capacity gate).
    flux_EXC = interception_1(P, S1, insc, nearzero=nearzero)
    zeros = torch.zeros_like(flux_EXC)
    flux_EXC = torch.clamp(flux_EXC, min=zeros, max=P)

    # State update for interception evaporation (same-day intermediate state:
    # S1 + P - EXC is the water actually present in the interception store).
    S1_tmp = S1 + P - flux_EXC
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_Ei: Evaporation from interception store (storage depletion from
    # the intermediate state, capped against it).
    flux_Ei = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_Ei = torch.minimum(flux_Ei, S1_tmp - nearzero)
    flux_Ei = F.relu(flux_Ei)

    # Final S1 update
    S1_new = S1_tmp - flux_Ei
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Soil Moisture Process (S2) ---
    # Step 2.1: Surface processes (Infiltration and Runoff)
    # flux_INF: Infiltration into the soil (incoming-water partition of the
    # throughfall, exponentially declining with relative soil moisture).
    flux_INF = infiltration_1(coeff, sq, S2, smsc, flux_EXC, nearzero=nearzero)
    flux_INF = torch.minimum(flux_INF, flux_EXC)

    # flux_SRUN: Surface runoff (saturation excess before infiltration;
    # residual partition of the throughfall).
    flux_SRUN = F.relu(flux_EXC - flux_INF)

    # Step 2.2: Internal soil moisture split
    # flux_INT: Interflow from infiltrated water (incoming-water partition).
    flux_INT = interflow_1(sub, S2, smsc, flux_INF, nearzero=nearzero)
    flux_INT = torch.minimum(flux_INT, flux_INF)

    # flux_REC: Groundwater recharge (incoming-water partition of the
    # remaining infiltrated water).
    flux_rem_inf = F.relu(flux_INF - flux_INT)
    flux_REC = recharge_1(crak, S2, smsc, flux_rem_inf, nearzero=nearzero)
    flux_REC = torch.minimum(flux_REC, flux_rem_inf)

    # flux_SMF: Soil moisture filling flux (residual partition).
    flux_SMF = F.relu(flux_rem_inf - flux_REC)

    # flux_GWF: Saturation excess from soil moisture store to groundwater
    # (overflow of the incoming SMF driven by soil saturation).
    flux_GWF = saturation_1(flux_SMF, S2, smsc, nearzero=nearzero)
    flux_GWF = torch.clamp(flux_GWF, min=zeros, max=flux_SMF)

    # Step 2.3: State update and Evapotranspiration (ET is a storage
    # depletion drawn from the same-day intermediate state, after the
    # incoming-water partition -- same convention as mopex2/xinanjiang).
    S2_tmp = S2 + flux_SMF - flux_GWF
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Remaining PET after interception ET
    pet_rem = F.relu(PET - flux_Ei)

    # flux_Et: Transpiration from soil
    # MATLAB: evap_2(10, S2, smsc, Ep) - p1=10 is used as a constant
    p1_const = torch.tensor(10.0, device=P.device)
    flux_Et = evap_2(p1_const, S2_tmp, smsc, pet_rem, nearzero=nearzero)
    flux_Et = torch.minimum(flux_Et, S2_tmp - nearzero)
    flux_Et = torch.minimum(flux_Et, pet_rem)
    flux_Et = F.relu(flux_Et)

    # Final S2 update
    S2_new = S2_tmp - flux_Et
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Groundwater Process (S3) ---
    # Inflow to S3: groundwater recharge (REC) and saturation overflow (GWF)
    # arrive same-day; baseflow then depletes the intermediate state.
    inflow_S3 = flux_REC + flux_GWF

    S3_tmp = S3 + inflow_S3
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)

    # flux_BAS: Baseflow from groundwater (storage depletion).
    flux_BAS = baseflow_1(k, S3_tmp, nearzero=nearzero)
    flux_BAS = torch.minimum(flux_BAS, S3_tmp - nearzero)
    flux_BAS = F.relu(flux_BAS)

    # Final S3 update
    S3_new = S3_tmp - flux_BAS
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. Output Aggregation ---
    # Qsim = Surface Runoff + Interflow + Baseflow
    # Ea = Interception ET + Soil Transpiration
    Qsim = flux_SRUN + flux_INT + flux_BAS
    Ea = flux_Ei + flux_Et

    return Qsim, Ea, S1_new, S2_new, S3_new
