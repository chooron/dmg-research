import torch
from typing import Tuple
from ..flux.evap import evap_6, evap_5
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_9
from ..flux.baseflow import baseflow_1

# 参数取值范围字典 (基于 MARRMoT m_04_newzealand1_6p_1s)
NEWZEALAND1_PARAMS_BOUNDS = {
    "s1max": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "sfc_frac": [
        0.05,
        0.95,
    ],  # Field capacity as fraction of maximum soil moisture [-]
    "m": [0.05, 0.95],  # Fraction forest [-]
    "a": [0.0, 1.0],  # Subsurface runoff coefficient [d-1]
    "b": [1.0, 5.0],  # Runoff non-linearity [-]
    "tcbf": [0.0, 1.0],  # Baseflow runoff coefficient [d-1]
}

# 参数描述字典
NEWZEALAND1_PARAMS_DESC = {
    "s1max": "Maximum soil moisture storage [mm]",
    "sfc_frac": "Field capacity as fraction of maximum soil moisture [-]",
    "m": "Fraction forest [-]",
    "a": "Subsurface runoff coefficient [d-1]",
    "b": "Runoff non-linearity [-]",
    "tcbf": "Baseflow runoff coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """
    创建 New Zealand v1 模型的初始状态.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def newzealand1_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数顺序与 NEWZEALAND1_PARAMS_BOUNDS 的键顺序一致
    s1max: torch.Tensor,
    sfc_frac: torch.Tensor,
    m: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tcbf: torch.Tensor,
    # 状态变量
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    New Zealand model v1 单步计算函数.

    模型引用:
    Atkinson, S. E., Woods, R. A., & Sivapalan, M. (2002). Climate and
    landscape controls on water balance model complexity over changing
    timescales. Water Resources Research, 38(12), 17-50.
    """

    # MARRMoT m_04 defines all five flux rates from the state at the start
    # of the time step, then integrates dS1 once.  Do not successively
    # update S1 while evaluating the individual fluxes: that is an operator
    # split scheme and lets today's precipitation alter same-step ET and
    # subsurface/baseflow, which is not the model equation.

    # 1. Fluxes evaluated at the common, beginning-of-step state S1.
    flux_qse = saturation_1(P, S1, s1max, nearzero=nearzero)
    flux_qse = torch.minimum(torch.clamp(flux_qse, min=0.0), P)
    flux_veg = evap_6(m, sfc_frac, S1, s1max, PET, nearzero=nearzero)
    flux_ebs = evap_5(m, S1, s1max, PET, nearzero=nearzero)
    sfc_threshold = sfc_frac * s1max
    flux_qss = interflow_9(S1, a, sfc_threshold, b, nearzero=nearzero)
    flux_qbf = baseflow_1(tcbf, S1, nearzero=nearzero)

    # A simultaneous explicit Euler step can only lose the water available
    # during this daily step.  Scale all outgoing rates together only in the
    # exceptional depletion case; for ordinary states this factor is exactly
    # one and the MARRMoT flux equations above are unchanged.
    flux_ea_total = torch.clamp(flux_veg + flux_ebs, min=0.0)
    flux_qss = torch.clamp(flux_qss, min=0.0)
    flux_qbf = torch.clamp(flux_qbf, min=0.0)
    available = torch.clamp(S1 + P - nearzero, min=0.0)
    outgoing = flux_ea_total + flux_qse + flux_qss + flux_qbf
    depletion_scale = torch.minimum(
        torch.ones_like(outgoing), available / (outgoing + nearzero)
    )
    flux_ea_total = flux_ea_total * depletion_scale
    flux_qse = flux_qse * depletion_scale
    flux_qss = flux_qss * depletion_scale
    flux_qbf = flux_qbf * depletion_scale
    S1_new = torch.clamp(S1 + P - flux_ea_total - flux_qse - flux_qss - flux_qbf, min=nearzero)

    # 6. 变量聚合与返回
    # Qsim = qse (地表) + qss (壤中) + qbf (底流)
    Qsim = flux_qse + flux_qss + flux_qbf
    Ea = flux_ea_total

    return Qsim, Ea, S1_new
