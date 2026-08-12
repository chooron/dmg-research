import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_ihacres_deficit as evap_linear_deficit
from ..flux.saturation import saturation_5
from ..flux.split import split_1

# Parameter range dictionary (MARRMoT m_05 IHACRES: 6 parameters, 1 state)
IHACRES_PARAMS_BOUNDS = {
    "lp": [1.0, 2000.0],  # Wilting point [mm]
    "d": [1.0, 2000.0],  # Threshold for flow generation [mm]
    "p": [0.0, 10.0],  # Flow response non-linearity [-]
    "alpha": [0.0, 1.0],  # Fast/slow flow division [-]
    "tau_q": [1.0, 5.0],  # Fast flow routing delay [d]
    "tau_s": [1.0, 30.0],  # Slow flow routing delay [d]
}

# Parameter description dictionary
IHACRES_PARAMS_DESC = {
    "lp": "Wilting point [mm]",
    "d": "Threshold for flow generation [mm]",
    "p": "Flow response non-linearity [-]",
    "alpha": "Fast/slow flow division [-]",
    "tau_q": "Fast flow routing delay [d]",
    "tau_s": "Slow flow routing delay [d]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """
    Create initial state for IHACRES model.
    Note: S1 is a deficit store.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def ihacres_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching IHACRES_PARAMS_BOUNDS keys
    lp: torch.Tensor,
    d: torch.Tensor,
    p: torch.Tensor,
    alpha: torch.Tensor,
    tau_q: torch.Tensor,
    tau_s: torch.Tensor,
    # State variable (Deficit store)
    S1: torch.Tensor,
    nearzero: float = 1e-6,
    *,
    return_routing_fluxes: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    IHACRES model single-step calculation.

    Model references:
    Croke, B. F. W., & Jakeman, A. J. (2004). A catchment moisture deficit
    module for the IHACRES rainfall-runoff model. Environmental Modelling and
    Software, 19(1), 1-5.
    """

    # 1. 蒸发 (让亏缺变大)
    flux_ea = evap_linear_deficit(S1, lp, PET, nearzero=nearzero)
    flux_ea = F.relu(flux_ea)

    # 2. 参数计算出的产流 (Effective Rainfall)
    # 这是基于 d, p 参数估算的产流
    flux_u_calc = saturation_5(S1, d, p, P, nearzero=nearzero)
    flux_u_calc = torch.clamp(flux_u_calc, min=torch.zeros_like(P), max=P)

    # 3. 核心修正：先算账，别急着 Clamp
    # 理论上的新亏缺 = 旧亏缺 - 进水 + 出水
    # S_temp = S1 - P + flux_ea + flux_u_calc
    S1_temp = S1 - P + flux_ea + flux_u_calc
    
    # 4. 捕捉溢出 (Overflow / Saturation Excess)
    # 如果 S1_temp < 0，说明水溢出来了。溢出量 = -S1_temp
    flux_overflow = F.relu(-S1_temp)
    
    # 5. 将溢出转化为径流
    # 总径流 = 参数算的径流 + 溢出的径流
    flux_u_total = flux_u_calc + flux_overflow
    
    # 6. 更新状态
    # 如果有溢出，S1_new 必然是 0（饱和）；如果没有，就是 S1_temp
    S1_new = torch.clamp(S1_temp, min=nearzero)

    # 7. 汇流分配 (Split)
    flux_uq = split_1(alpha, flux_u_total, nearzero=nearzero)
    flux_us = split_1(1.0 - alpha, flux_u_total, nearzero=nearzero)

    Qsim = flux_uq + flux_us
    Ea = flux_ea

    if return_routing_fluxes:
        return Qsim, Ea, S1_new, (flux_uq, flux_us)
    return Qsim, Ea, S1_new
