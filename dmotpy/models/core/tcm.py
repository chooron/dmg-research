from typing import Tuple

import torch
import torch.nn.functional as F

from ..flux.baseflow import baseflow_1, baseflow_tcm as baseflow_6
from ..flux.effective import effective_1
from ..flux.evap import evap_1, evap_16
from ..flux.saturation import saturation_1, saturation_9
from ..flux.split import split_1

# Parameter range dictionary (based on MARRMoT m_25_tcm_6p_4s)
# Note: fa is the fraction of mean(P) that forms abstraction rate.
# The actual abstraction rate ca = fa * mean(P) must be pre-computed
# from the catchment's mean precipitation before calling tcm_step.
TCM_PARAMS_BOUNDS = {
    "phi": [0.0, 1.0],  # Fraction preferential recharge [-]
    "rc": [1.0, 2000.0],  # Maximum soil moisture depth [mm]
    "gam": [0.0, 1.0],  # Fraction of Ep reduction with depth [-]
    "k1": [0.0, 1.0],  # Runoff coefficient [d-1]
    "fa": [0.0, 1.0],  # Fraction of mean(P) that forms abstraction rate [-]
    "k2": [0.0, 1.0],  # Runoff coefficient [mm-1 d-1]
}

# Parameter description dictionary
TCM_PARAMS_DESC = {
    "phi": "Fraction preferential recharge [-]",
    "rc": "Maximum soil moisture depth [mm]",
    "gam": "Fraction of Ep reduction with depth [-]",
    "k1": "Runoff coefficient [d-1]",
    "fa": "Fraction of mean(P) that forms abstraction rate [-]",
    "k2": "Runoff coefficient [mm-1 d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for TCM model.
    S1: Upper soil moisture store
    S2: Soil moisture deficit store (0 = fully saturated)
    S3: Fast routing reservoir
    S4: Slow routing reservoir
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4


def tcm_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching TCM_PARAMS_BOUNDS keys
    phi: torch.Tensor,
    rc: torch.Tensor,
    gam: torch.Tensor,
    k1: torch.Tensor,
    fa: torch.Tensor,
    k2: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    nearzero: float = 1e-6,
    *,
    mean_P: torch.Tensor,
    return_diagnostics: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Thames Catchment Model (TCM) single-step calculation.

    MATLAB reference: m_25_tcm_6p_4s
    - fa is fraction of mean(P) forming abstraction rate: ca = fa * mean(P)
    - mean_P should be pre-computed from the entire precipitation time series
    - S2 is a deficit store (StoreSigns = -1): increases with ET, decreases with qex1
    - flux_qex2 uses saturation_9: passes qex1 through when S2 deficit is near zero

    Model reference:
    Moore, R. J., & Bell, V. A. (2001). Comparison of rainfall-runoff models
    for flood forecasting. Part 1: Literature review of models.
    """
    # Abstraction rate [mm/d] = fa * mean(P), matching MATLAB init()
    ca = fa * mean_P

    # --- 1. Pre-process ---
    flux_pn = effective_1(P, PET, nearzero=nearzero)
    zeros = torch.zeros_like(P)
    flux_pn = torch.clamp(flux_pn, min=zeros, max=P)
    flux_en = P - flux_pn  # Interception Loss

    flux_pby = split_1(phi, flux_pn, nearzero=nearzero)
    flux_pin = flux_pn - flux_pby

    # --- 2. Upper Store (S1) ---
    # In MATLAB ODE: dS1 = flux_pin - flux_ea - flux_qex1
    # Sequential: add flux_pin first, then compute saturation excess
    S1 = S1 + flux_pin

    # Saturation overflow: saturation_1(flux_pin, S1, rc)
    # When S1 approaches rc, excess water flows out
    flux_qex1 = saturation_1(flux_pin, S1, rc, nearzero=nearzero)
    flux_qex1 = torch.minimum(flux_qex1, S1)
    S1 = S1 - flux_qex1

    # Evap from S1
    flux_ea = evap_1(S1, PET, nearzero=nearzero)
    flux_ea = torch.minimum(flux_ea, S1)
    S1 = S1 - flux_ea
    S1_new = torch.clamp(S1, min=nearzero)

    # --- 3. Deficit Store (S2) ---
    # evap_16: gam * Ep, active when S1 > 0.01 (smooth threshold)
    # Uses full PET per MATLAB: flux_et = evap_16(gam, Inf, S1, 0.01, Ep, dt)
    inf_tensor = torch.full_like(S1_new, float("inf"))
    flux_et = evap_16(
        gam,
        inf_tensor,
        S1_new,
        torch.tensor(0.01, device=P.device),
        PET,
        nearzero=nearzero,
    )

    # S2 is a deficit store: ET deepens deficit, qex1 fills it
    # dS2 = flux_et + flux_qex2 - flux_qex1  (MATLAB ODE)
    # Sequential: first compute qex2 from current S2, then update
    # flux_qex2 = saturation_9(flux_qex1, S2, 0.01):
    #   passes qex1 through when S2 deficit is near zero (saturated)
    flux_qex2 = saturation_9(
        flux_qex1, S2, torch.tensor(0.01, device=P.device), nearzero=nearzero
    )
    S2_raw = S2 + flux_et + flux_qex2 - flux_qex1
    # If discrete recharge over-fills the deficit store, route the excess forward
    # instead of losing it in the nearzero clamp.
    flux_qex2 = flux_qex2 + torch.relu(nearzero - S2_raw)
    S2_new = torch.clamp(S2 + flux_et + flux_qex2 - flux_qex1, min=nearzero)

    # --- 4. Fast Routing (S3) ---
    inflow_S3 = flux_qex2 + flux_pby
    S3 = S3 + inflow_S3

    flux_quz = baseflow_1(k1, S3, nearzero=nearzero)
    flux_quz = torch.minimum(flux_quz, S3)
    S3 = S3 - flux_quz
    S3_new = torch.clamp(S3, min=nearzero)

    # --- 5. Slow Routing (S4) ---
    S4 = S4 + flux_quz

    # Abstraction loss: ca = fa * mean(P)
    flux_a = torch.minimum(ca, S4)
    S4 = S4 - flux_a

    # Baseflow: baseflow_6(k2, 0, S4) — quadratic, threshold=0
    flux_q = baseflow_6(k2, torch.tensor(0.0, device=P.device), S4, nearzero=nearzero)
    flux_q = torch.minimum(flux_q, S4)
    S4 = S4 - flux_q
    S4_new = torch.clamp(S4, min=nearzero)

    # --- 6. Output ---
    Qsim = flux_q
    # Ea = interception loss + S1 evap + deep ET; abstraction is a separate sink
    Ea = flux_en + flux_ea + flux_et

    if return_diagnostics:
        diagnostics = {
            "external_losses": flux_a,
            "runoff_prerouting": Qsim,
            "actual_et": Ea,
            "state_names": ("S1", "S2", "S3", "S4"),
        }
        return Qsim, Ea, S1_new, S2_new, S3_new, S4_new, diagnostics

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new
