import torch
from typing import Tuple
from ..flux.interception import interception_2
from ..flux.evap import evap_1
from ..flux.saturation import saturation_2
from ..flux.baseflow import baseflow_1
from ..flux.excess import excess_1

# 参数取值范围字典 (基于 MARRMoT m_02_wetland_4p_1s)
WETLAND_PARAMS_BOUNDS = {
    "dw": [0.0, 5.0],  # Interception capacity [mm]
    "betaw": [0.0, 10.0],  # Soil moisture distribution parameter [-]
    "swmax": [1.0, 2000.0],  # Maximum soil moisture depth [mm]
    "kw": [0.0, 1.0],  # Base flow time parameter [d-1]
}

# 参数描述字典
WETLAND_PARAMS_DESC = {
    "dw": "Interception capacity [mm]",
    "betaw": "Soil moisture distribution parameter [-]",
    "swmax": "Maximum soil moisture depth [mm]",
    "kw": "Base flow time parameter [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """
    创建 Wetland 模型的初始状态.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def wetland_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数顺序与 WETLAND_PARAMS_BOUNDS 的键顺序一致
    dw: torch.Tensor,
    betaw: torch.Tensor,
    swmax: torch.Tensor,
    kw: torch.Tensor,
    # 状态变量
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wetland model (FLEX-Topo) 单步计算函数.

    模型引用:
    Savenije, H. H. G. (2010). "Topography driven conceptual modelling
    (FLEX-Topo)." Hydrology and Earth System Sciences, 14(12), 2681-2692.
    """

    # Fixed-priority daily explicit Euler split:
    # interception excess -> saturation excess -> storage-capacity overflow
    # -> ET -> linear baseflow.  Each later process sees the water left by the
    # preceding process; no proportional multi-flux limiter is used.

    # 1. Rainfall partitioning
    flux_pe = interception_2(P, dw, nearzero=nearzero)
    flux_ei = P - flux_pe

    # 2. Saturation excess is evaluated from the beginning-of-step storage.
    # saturation_2 has a tiny negative dry-end round-off tail because it uses
    # (term + nearzero)**betaw, so enforce the physical [0, flux_pe] envelope.
    flux_qwsof = saturation_2(S1, swmax, betaw, flux_pe, nearzero=nearzero)
    flux_qwsof = torch.clamp(
        flux_qwsof,
        min=torch.zeros_like(flux_qwsof),
        max=flux_pe,
    )

    # 3. Add the retained input, then make any discrete capacity exceedance an
    # explicit saturation-excess runoff flux.  This guarantees S1 <= swmax
    # without an unaccounted state clamp.
    S1_curr = S1 + flux_pe - flux_qwsof
    flux_qwsof = flux_qwsof + excess_1(S1_curr, swmax, nearzero=nearzero)
    S1_curr = S1 + flux_pe - flux_qwsof

    # 4. ET from the post-input, post-overflow storage. evap_1 already enforces
    # 0 <= flux_ew <= S1_curr for non-negative PET, so no second limiter is
    # needed.
    flux_ew = evap_1(S1_curr, PET, nearzero=nearzero)
    S1_curr = S1_curr - flux_ew

    # 5. Linear baseflow from the ET-reduced storage. With kw in [0, 1],
    # baseflow_1(kw, S1_curr) is already bounded by S1_curr.
    flux_qwgw = baseflow_1(kw, S1_curr, nearzero=nearzero)

    # 6. Exact residual storage: do not floor it at nearzero, which would add
    # unaccounted water during dry periods.
    S1_new = S1_curr - flux_qwgw

    # 5. 变量聚合与返回
    # Ea = ei (拦截蒸发) + ew (土壤蒸发)
    # Qsim = qwsof (地表产流) + qwgw (地底产流/底流)
    Ea = flux_ei + flux_ew
    Qsim = flux_qwsof + flux_qwgw

    return Qsim, Ea, S1_new
