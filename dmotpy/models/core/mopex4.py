import torch
import torch.nn.functional as F
from typing import Tuple

from ..flux.mopex import (
    mopex_baseflow_1 as baseflow_1,
    mopex_evap_7 as evap_7,
    mopex_melt_1 as melt_1,
    mopex_rainfall_1 as rainfall_1,
    mopex_recharge_3 as recharge_3,
    mopex_saturation_1 as saturation_1,
    mopex_snowfall_1 as snowfall_1,
)

# ================================================================
# 1. Parameter Configuration
# MOPEX4 基于 MARRMoT 原版结构。interception 恢复为仓库可追溯的原始
# 季节余弦参数化（alpha + (1-alpha)*cos），与 MARRMoT m_32_mopex4 的
# i_alpha / i_s 语义一致；其余参数语义与 MARRMoT 一致。
# 过程顺序（interception-first、共享日 PET budget、net soil input）
# 保持当前已修正并验证过的语义。
# ================================================================

MOPEX4_PARAMS_BOUNDS = {
    "tcrit":   [-3.0, 3.0],    # Snowfall & snowmelt temperature threshold [°C]
    "ddf":     [0.0,  20.0],   # Degree-day factor [mm/°C/d]
    "s2max":   [1.0,  2000.0], # Maximum soil moisture storage [mm]      → Sb1
    "tw":      [0.0,  1.0],    # Groundwater leakage rate [d⁻¹]
    "alpha":   [0.0,  1.0],    # Intercepted fraction of rainfall [-]    → i_alpha
    "is_time": [1.0,  365.0],  # Timing of maximum interception [d]      → i_s
    "tu":      [0.0,  1.0],    # Slow flow routing rate [d⁻¹]
    "se":      [0.05, 0.95],   # Root zone ET capacity as fraction of s3max [-]
    "s3max":   [1.0,  2000.0], # Root zone (subsurface) storage capacity → Sb2
    "tc":      [0.0,  1.0],    # Mean residence rate [d⁻¹]
}

MOPEX4_PARAMS_DESC = {
    "tcrit":   "Temperature threshold for snow/rain partitioning and melt [°C]",
    "ddf":     "Degree-day factor [mm/°C/d]",
    "s2max":   "Maximum soil moisture storage [mm]                       (= Sb1)",
    "tw":      "Groundwater leakage rate [d⁻¹], flux = tw * S_soil",
    "alpha":   "Mean interception fraction [-], seasonal cosine modulation",
    "is_time": "Day-of-year of maximum interception [d]                  (= i_s)",
    "tu":      "Slow flow routing rate [d⁻¹], flux = tu * S_sub",
    "se":      "Root zone ET capacity fraction [-], ET2 Smax = se * s3max",
    "s3max":   "Root zone (subsurface) storage capacity [mm]             (= Sb2)",
    "tc":      "Mean residence rate [d⁻¹], flux = tc * S",
}

MOPEX4_LIU_INTERCEPTION_NAMES = ("S_eff", "c")
MOPEX4_LEGACY_INTERCEPTION_NAMES = ("alpha", "is_time")


def validate_mopex4_parameter_schema(
    parameter_names: tuple[str, ...] | list[str], *, legacy_f0: bool = False
) -> None:
    """Reject accidental reuse of the wrong MOPEX4 interception schema.

    A raw vector has the same length in both formulations (10 parameters), so
    callers that persist parameter names must persist and validate the schema
    explicitly.  ``legacy_f0=True`` requests the original ``(alpha, is_time)``
    schema (the current restored MOPEX4); ``legacy_f0=False`` requests the
    Liu ``(S_eff, c)`` schema (retained for loading pre-restore checkpoints).
    """
    names = tuple(parameter_names)
    expected_slots = MOPEX4_LEGACY_INTERCEPTION_NAMES if legacy_f0 else MOPEX4_LIU_INTERCEPTION_NAMES
    actual_slots = (names[4], names[5]) if len(names) > 5 else ()
    if actual_slots != expected_slots:
        mode = "original alpha/is_time" if legacy_f0 else "Liu S_eff/c"
        raise ValueError(
            f"MOPEX4 parameter schema mismatch: expected {mode} slots "
            f"{expected_slots}, got {actual_slots}. Explicit schema routing is required."
        )


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (Sn, S1, S2, Sc1, Sc2).
    对应 MATLAB: S1=snow, S2=soil, S3=subsurface, S4=fast route, S5=slow route
    Python 接口映射：
        Sn  ↔ MATLAB S1 (snow)
        S1  ↔ MATLAB S2 (soil),       参数 Sb1 = s2max
        S2  ↔ MATLAB S3 (subsurface), 参数 Sb2 = s3max
        Sc1 ↔ MATLAB S4 (fast route)
        Sc2 ↔ MATLAB S5 (slow route)
    """
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sn
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S1 (soil)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S2 (subsurface)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc1
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc2
    )


# ================================================================
# 2. Interception Flux Function — original seasonal formulation
# ================================================================

def interception_4(
    flux_pr: torch.Tensor,
    doy: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmax: float = 365.25,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """Original MOPEX4 interception kernel (restored from git HEAD / MARRMoT).

    MATLAB 原版：
        out = max(0, p1 + (1-p1)*cos(2π*(t*dt - p2)/tmax)) * In
        其中 p1=i_alpha, p2=i_s, t*dt≈doy, In=flux_pr

    物理含义：
        截留比例随季节余弦变化，峰值在 doy=is_time 处（LAI 最大时截留最多）。
        alpha=0 时全年无截留；alpha=1 时全年截留全部降雨。
        比例下限截断为 0（max(0,...)），防止出现负截留（即降雨增益）。

    梯度处理：
        MATLAB 的 max(0,...) 在比例为零时梯度截断。
        替换为 F.softplus(x, beta) 的缩放近似，保留平滑下限，
        但 beta 取较大值（50）使其在实践中等价于 relu 而梯度连续。
        alpha 和 is_time 均通过余弦函数全程可导。

    参数：
        flux_pr  - 到达冠层的液态降雨通量 [mm/d]（snow/rain partition 之后）
        doy      - 当前儒略日 [d]，shape 与 flux_pr 一致
        alpha    - 平均截留比例 [-]，∈ [0,1]
        is_time  - 截留峰值时刻 [d]，∈ [1, 365]
        tmax     - 季节周期长度 [d]，默认 365.25
    """
    del nearzero  # preserved for call compatibility; unused in the original kernel
    rad          = 2.0 * torch.pi * (doy - is_time) / tmax
    interc_frac  = alpha + (1.0 - alpha) * torch.cos(rad)

    # 梯度处理：softplus(x * beta) / beta ≈ relu(x)，但在 x=0 处光滑可导
    # beta=50 时与 relu 的最大偏差 < 0.014，实践中可忽略
    interc_frac_pos = F.softplus(interc_frac * 50.0) / 50.0   # ≥ 0，光滑

    # 截留量同时受降雨量约束
    flux_i = torch.minimum(interc_frac_pos * flux_pr, flux_pr)
    return flux_i


# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex4_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters — order matches MOPEX4_PARAMS_BOUNDS keys
    tcrit: torch.Tensor,     # tcrit
    ddf: torch.Tensor,       # ddf
    Sb1: torch.Tensor,       # s2max → 土壤水库容量
    tw: torch.Tensor,        # tw
    alpha: torch.Tensor,     # alpha → i_alpha (mean interception fraction)
    is_time: torch.Tensor,   # is_time → i_s (interception peak day-of-year)
    tu: torch.Tensor,        # tu
    Se: torch.Tensor,        # se
    Sb2: torch.Tensor,       # s3max → 地下水库容量
    tc: torch.Tensor,        # tc
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
    MOPEX-4 离散单步计算 — original interception formula with corrected
    process order.

    Interception uses the original seasonal cosine parameterization,

        frac   = alpha + (1 - alpha) * cos(2π*(doy - is_time)/tmax)
        I_pot  = softplus(50*frac)/50 * Pr        (smooth lower clamp)
        I      = min(I_pot, Pr, PET)

    on the post-snow/rain-partition liquid rainfall ``Pr``.  ``alpha`` is the
    mean interception fraction [-] and ``is_time`` the peak interception
    day-of-year [d]; both are learnable.

    PET semantics (corrected, frozen): interception-first shared daily PET
    demand with the exact hard budget limiter,

        I             = min(I_pot, PET)
        PET_after_I   = PET - I
        ET1           = evap_7(S1, Sb1, PET_after_I)
        PET_after_ET1 = PET_after_I - ET1
        ET2           = evap_7(S2, se*s3max, PET_after_ET1)

    so ``I + ET1 + ET2 <= PET`` holds by construction (up to floating point
    tolerance).  ``phase_cos``/``phase_sin`` are accepted for call
    compatibility and unused; ``doy`` is consumed by the seasonal kernel.

    Water path (corrected, frozen): interception is allocated first and
    removed from the liquid rainfall *before* soil entry,

        I       = min(I_pot, PET)
        Pr_net  = Pr - I
        soil    = Pr_net + qn        (net rainfall + snowmelt enters S1)
        q1f     = saturation_1(soil, S1, Sb1)   (event input is net, not gross)

    so interception is sourced exclusively from the current day's liquid
    rainfall (``0 <= I <= Pr``) and is never deducted from snowmelt or from
    pre-existing soil water after ET1.  This is the counterfactual
    "original interception formula + corrected process order".

    离散化策略（顺序显式步进）：
        各通量按顺序从当前状态计算并立即更新，天然保证状态非负。

    通量顺序（S1/土壤）：截留 I（直接从 Pr 扣除）→ 净雨 Pr_net + qn 进入土壤 → 蒸发 et1 → 饱和径流 q1f → 下渗 qw
    通量顺序（S2/地下）：加入 qw   → 地下溢流 q2f → 基流 q2u → 蒸发 et2
    """
    del phase_cos, phase_sin  # unused in the restored forward

    # ── Guards ────────────────────────────────────────────────────
    Sn  = F.relu(Sn)
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Snow Bucket (Sn = MATLAB S1)
    # MATLAB: dS1 = ps - qn
    # ============================================================

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    # 守恒：flux_ps + flux_pr = P

    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    # melt_1 内部保证 flux_qn ≤ Sn

    Sn      = Sn + flux_ps
    Sn_new  = Sn - flux_qn               # flux_qn ≤ Sn，非负保证

    # ============================================================
    # Soil Bucket (S1 = MATLAB S2)
    # MATLAB: dS2 = pr + qn - et1 - i - q1f - qw
    # 顺序：截留 I（直接从 Pr 扣除）→ 净雨 Pr_net + qn 进入土壤 → 蒸发 et1 → 饱和径流 q1f → 下渗 qw
    # ============================================================

    # Step 1: interception-first water path.
    # I is allocated from the exact hard budget min(I_pot, PET) and removed
    # from the liquid rainfall BEFORE it enters the soil bucket:
    #     Pr -> I (evaporative loss) -> Pr_net = Pr - I -> soil input.
    i_pot = interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero)
    flux_i = torch.minimum(i_pot, PET)               # exact hard budget limiter
    flux_pr_net = flux_pr - flux_i
    pet_after_i = PET - flux_i

    soil_input = flux_pr_net + flux_qn
    S1 = S1 + soil_input

    # Step 2: ET1 consumes the remaining shared PET demand after I.
    flux_et1 = evap_7(S1, Sb1, pet_after_i, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S1)
    S1 = S1 - flux_et1
    pet_remaining_after_et1 = pet_after_i - flux_et1

    # Step 3：饱和径流（event input = net rainfall + snowmelt, not gross Pr + qn）
    # MATLAB: flux_q1f = saturation_1(pr+qn, S2, s2max)
    flux_q1f = saturation_1(soil_input, S1, Sb1, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f, S1)
    S1 = S1 - flux_q1f

    # Step 4：下渗
    # MATLAB: flux_qw = recharge_3(tw, S2) → tw * S2
    flux_qw = recharge_3(tw, S1)          # min(tw*S1, S1) ≤ S1
    S1_new  = S1 - flux_qw                # S1_new ≥ 0 保证

    # ============================================================
    # Subsurface Bucket (S2 = MATLAB S3)
    # MATLAB: dS3 = qw - et2 - q2f - q2u
    # 顺序：加入下渗 → 地下溢流 → 基流 → 蒸发
    # ============================================================

    S2 = S2 + flux_qw

    # Step 1：地下溢流
    # MATLAB: flux_q2f = saturation_1(flux_qw, S3, s3max)
    flux_q2f = saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f

    # Step 2：基流
    # MATLAB: flux_q2u = baseflow_1(tu, S3) → tu * S3
    flux_q2u = baseflow_1(tu, S2)         # min(tu*S2, S2) ≤ S2
    S2 = S2 - flux_q2u

    # Step 3：蒸发（frozen path passes the remaining PET demand after I and ET1）
    se_abs = Se * Sb2
    flux_et2 = evap_7(S2, se_abs, pet_remaining_after_et1, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new   = S2 - flux_et2              # S2_new ≥ 0 保证

    # ============================================================
    # Routing Buckets
    # MATLAB: dS4 = q1f + q2f - qf；dS5 = q2u - qs
    # ============================================================

    Sc1      = Sc1 + flux_q1f + flux_q2f
    flux_qf  = baseflow_1(tc, Sc1)
    Sc1_new  = Sc1 - flux_qf              # Sc1_new ≥ 0 保证

    Sc2      = Sc2 + flux_q2u
    flux_qs  = baseflow_1(tc, Sc2)
    Sc2_new  = Sc2 - flux_qs              # Sc2_new ≥ 0 保证

    # ============================================================
    # Output
    # MATLAB: FluxGroups.Ea = [et1, et2]；FluxGroups.Q = [qf, qs]
    # 注：flux_i（截留蒸发）也是实际蒸散发的一部分，
    # 若需要与观测 ET 对比，应加入 ET_total
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new
