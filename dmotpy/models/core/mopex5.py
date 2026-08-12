import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from ..flux.mopex import (
    mopex_baseflow_1 as baseflow_1,
    mopex_evap_7 as evap_7,
    mopex_melt_1 as melt_1,
    mopex_rainfall_1 as rainfall_1,
    mopex_recharge_3 as recharge_3,
    mopex_saturation_1 as saturation_1,
    mopex_snowfall_1 as snowfall_1,
)
# The original seasonal interception kernel is shared with restored MOPEX4
# (alpha/is_time, context-free hot path) so MOPEX5 nests the exact MOPEX4
# interception semantics.
from .mopex4 import interception_4

# ================================================================
# 1. Parameter Configuration
# 在 MOPEX-4 基础上新增 tmin、trange 两个物候参数
# ================================================================

MOPEX5_PARAMS_BOUNDS = {
    "tcrit":   [-3.0, 3.0],    # Snowfall & snowmelt temperature threshold [°C]
    "ddf":     [0.0,  20.0],   # Degree-day factor [mm/°C/d]
    "s2max":   [1.0,  2000.0], # Maximum soil moisture storage [mm]
    "tw":      [0.0,  1.0],    # Groundwater leakage rate [d⁻¹]
    "alpha":   [0.0,  1.0],    # Intercepted fraction of rainfall [-]
    "is_time": [1.0,  365.0],  # Timing of maximum interception [d]
    "tmin":    [-10.0, 0.0],   # GSI minimum temperature (ET stops below) [°C]
    "trange":  [1.0,  20.0],   # GSI temperature range (ET ramps over) [°C]
    "tu":      [0.0,  1.0],    # Slow flow routing rate [d⁻¹]
    "se":      [0.05, 0.95],   # Root zone ET capacity fraction [-]
    "s3max":   [1.0,  2000.0], # Root zone (subsurface) storage capacity [mm]
    "tc":      [0.0,  1.0],    # Mean residence rate [d⁻¹]
}

MOPEX5_PARAMS_DESC = {
    "tcrit":   "Temperature threshold for snow/rain partitioning and melt [°C]",
    "ddf":     "Degree-day factor [mm/°C/d]",
    "s2max":   "Maximum soil moisture storage [mm]",
    "tw":      "Groundwater leakage rate [d⁻¹]",
    "alpha":   "Mean interception fraction [-]",
    "is_time": "Day-of-year of maximum interception [d]",
    "tmin":    "GSI lower temperature threshold; ET=0 when T <= tmin [°C]",
    "trange":  "GSI temperature range; ET=Ep when T >= tmin+trange [°C]",
    "tu":      "Slow flow routing rate [d⁻¹]",
    "se":      "Root zone ET capacity fraction [-], ET2 Smax = se * s3max",
    "s3max":   "Root zone (subsurface) storage capacity [mm]",
    "tc":      "Mean residence rate [d⁻¹]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (Sn, S1, S2, Sc1, Sc2).
    与 MOPEX-4 完全相同的状态变量布局：
        Sn  ↔ MATLAB S1 (snow)
        S1  ↔ MATLAB S2 (soil),       容量参数 Sb1 = s2max
        S2  ↔ MATLAB S3 (subsurface), 容量参数 Sb2 = s3max
        Sc1 ↔ MATLAB S4 (fast route)
        Sc2 ↔ MATLAB S5 (slow route)
    """
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
    )


# ================================================================
# 2. Phenology — original GSI (Growing Season Index) PET adjustment
# ================================================================

def phenology_effective_pet(
    T: torch.Tensor,
    tmin: torch.Tensor,
    trange: torch.Tensor,
    PET: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """Original MOPEX5 phenology PET demand (MARRMoT `phenology_1` with the
    production lambda_p = 1.0).

        gsi    = clamp((T - tmin) / trange, 0, 1)
        PET_epc = gsi * PET

    The GSI factor lies in [0, 1], so the phenology-adjusted daily demand
    satisfies ``0 <= PET_epc <= PET`` by construction.  This single quantity
    is the shared daily evaporative demand consumed in order by interception,
    ET1 and ET2 (same corrected budget semantics as restored MOPEX4).
    """
    safe_trange = torch.clamp(trange, min=nearzero)
    gsi = torch.clamp((T - tmin) / safe_trange, 0.0, 1.0)
    return gsi * PET


# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex5_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters — order matches MOPEX5_PARAMS_BOUNDS keys
    tcrit: torch.Tensor,
    ddf: torch.Tensor,
    Sb1: torch.Tensor,       # s2max
    tw: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmin: torch.Tensor,
    trange: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,        # se
    Sb2: torch.Tensor,       # s3max
    tc: torch.Tensor,
    # States
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
    *,
    doy: torch.Tensor = None,
    phase_cos: torch.Tensor = None,
    phase_sin: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MOPEX-5 离散单步计算 — original formulas with corrected process order.

    MOPEX5 keeps the MOPEX4 core hydrology and adds the phenology PET
    extension: the daily shared evaporative demand is the phenology-adjusted
    ``PET_epc = GSI(T) * PET`` (original ``phenology_1`` formula, production
    lambda_p = 1.0; ``0 <= PET_epc <= PET`` by construction).

    Corrected process semantics (aligned with restored MOPEX4; the OLD MOPEX5
    added gross Pr+qn to S1, ran ET1 before interception, capped interception
    by the post-ET1 storage instead of the shared PET demand, fed gross
    Pr+qn to saturation_1, and let ET2 consume the full PET_epc again):

        I       = min(I_pot, Pr, PET_epc)          # interception-first
        Pr_net  = Pr - I                           # only liquid rainfall
        soil    = Pr_net + qn                      # net + snowmelt
        ET1     = evap_7(S1, Sb1, PET_epc - I)
        q1f     = saturation_1(soil, S1, Sb1)      # net event input
        ET2     = evap_7(S2, se*s3max, PET_epc - I - ET1)

    so ``I + ET1 + ET2 <= PET_epc <= PET`` holds by construction.
    ``phase_cos``/``phase_sin`` are accepted for call compatibility and unused.

    MATLAB ODE 对应关系（原始公式不变）：
        dS1 = ps   - qn                                  (Sn：积雪)
        dS2 = pr   + qn - et1 - i - q1f - qw            (S1：土壤)
        dS3 = qw   - et2 - q2f - q2u                    (S2：地下)
        dS4 = q1f  + q2f - qf                            (Sc1：快速流)
        dS5 = q2u  - qs                                  (Sc2：慢速流)
    """
    del phase_cos, phase_sin  # unused in the fixed hot path

    # ── Guards ────────────────────────────────────────────────────
    Sn  = F.relu(Sn)
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Phenology Module — the single shared daily evaporative demand
    # MATLAB: flux_epc = phenology_1(T, tmin, tmin+trange, Ep)
    # ============================================================
    PET_epc = phenology_effective_pet(T, tmin, trange, PET, nearzero)

    # ============================================================
    # Snow Bucket (Sn = MATLAB S1)
    # ============================================================
    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    # 守恒：flux_ps + flux_pr = P

    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)

    Sn      = Sn + flux_ps
    Sn_new  = Sn - flux_qn               # flux_qn ≤ Sn，非负保证

    # ============================================================
    # Soil Bucket (S1 = MATLAB S2)
    # 顺序（corrected）：截留 I（从液态雨扣除）→ 净雨 Pr_net + qn 进入土壤
    #     → 蒸发 et1 → 饱和径流 q1f → 下渗 qw
    # ============================================================

    # Step 1: interception-first water path (original alpha/is_time kernel,
    # sourced exclusively from the post-partition liquid rainfall and capped
    # by the shared phenology-adjusted PET demand).
    i_pot = interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero)
    flux_i = torch.minimum(i_pot, PET_epc)         # exact hard budget limiter
    flux_pr_net = flux_pr - flux_i
    pet_after_i = PET_epc - flux_i

    soil_input = flux_pr_net + flux_qn
    S1 = S1 + soil_input

    # Step 2: ET1 consumes the remaining shared demand after interception.
    flux_et1 = evap_7(S1, Sb1, pet_after_i, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S1)
    S1 = S1 - flux_et1
    pet_remaining = pet_after_i - flux_et1

    # Step 3: 饱和径流（corrected：event input = net rainfall + snowmelt）
    flux_q1f = saturation_1(soil_input, S1, Sb1, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f, S1)
    S1 = S1 - flux_q1f

    # Step 4：下渗
    flux_qw = recharge_3(tw, S1)
    S1_new  = S1 - flux_qw               # S1_new ≥ 0 保证

    # ============================================================
    # Subsurface Bucket (S2 = MATLAB S3)
    # 顺序：加入下渗 → 地下溢流 → 基流 → 蒸发（剩余共享 budget）
    # ============================================================

    S2 = S2 + flux_qw

    # Step 1：地下溢流
    flux_q2f = saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f

    # Step 2：基流
    flux_q2u = baseflow_1(tu, S2)
    S2 = S2 - flux_q2u

    # Step 3：蒸发（剩余共享 budget after I and ET1）
    se_abs   = Se * Sb2
    flux_et2 = evap_7(S2, se_abs, pet_remaining, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new   = S2 - flux_et2             # S2_new ≥ 0 保证

    # ============================================================
    # Routing Buckets
    # ============================================================
    Sc1      = Sc1 + flux_q1f + flux_q2f
    flux_qf  = baseflow_1(tc, Sc1)
    Sc1_new  = Sc1 - flux_qf

    Sc2      = Sc2 + flux_q2u
    flux_qs  = baseflow_1(tc, Sc2)
    Sc2_new  = Sc2 - flux_qs

    # ============================================================
    # Output
    # ET_total 含截留蒸发，与 CAMELS ET 口径一致
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new
