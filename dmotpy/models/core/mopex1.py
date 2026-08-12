import torch
import torch.nn.functional as F
from typing import Tuple

from ..flux.mopex import (
    mopex_baseflow_1 as baseflow_1,
    mopex_evap_7 as evap_7,
    mopex_recharge_3 as recharge_3,
    mopex_saturation_1 as saturation_1,
)

# ================================================================
# 1. Parameter Configuration
# 严格对应 MARRMoT MATLAB 原版参数语义
# ================================================================

MOPEX1_PARAMS_BOUNDS = {
    "s1max": [1.0, 2000.0],   # Maximum soil moisture storage [mm]
    "tw":    [0.0, 1.0],      # Groundwater leakage rate [d⁻¹]
    "tu":    [0.0, 1.0],      # Slow flow routing rate [d⁻¹]
    "se":    [1.0, 2000.0],   # Root zone storage capacity [mm]
    "tc":    [0.0, 1.0],      # Mean residence rate [d⁻¹]
}

MOPEX1_PARAMS_DESC = {
    "s1max": "Maximum soil moisture storage [mm]",
    "tw":    "Groundwater leakage rate [d⁻¹], flux = tw * S1",
    "tu":    "Slow flow routing rate [d⁻¹],   flux = tu * S2",
    "se":    "Root zone storage capacity [mm]",
    "tc":    "Mean residence rate [d⁻¹],       flux = tc * S",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (S1, S2, Sc1, Sc2)."""
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
    )


# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex1_step(
    P: torch.Tensor,
    T: torch.Tensor,       # 保留接口，MOPEX-1 原版无融雪，T 未使用
    PET: torch.Tensor,
    Sb1: torch.Tensor,     # 对应 s1max
    tw: torch.Tensor,      # 率参数 [d⁻¹]
    tu: torch.Tensor,      # 率参数 [d⁻¹]
    Se: torch.Tensor,      # 对应 se
    tc: torch.Tensor,      # 率参数 [d⁻¹]
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MOPEX-1 离散单步计算。

    MATLAB ODE 对应关系：
        dS1 = P  - et1 - q1f - qw
        dS2 = qw - et2 - q2u
        dS3 = q1f - qf          (对应 Sc1)
        dS4 = q2u - qs          (对应 Sc2)

    离散化策略：
        MATLAB 用 ODE solver 联立求解（各通量同时作用），
        Python 采用顺序显式步进（Sequential Explicit）：
        每扣除一个通量后立即更新状态，下一个通量从更新后的状态计算。
        这样天然保证所有状态变量非负，无需依赖 clamp 截断来阻止负值
        （避免 clamp 截断破坏水量守恒并导致梯度消失）。

    通量顺序（S1）：溢流 q1f → 蒸发 et1 → 下渗 qw
    通量顺序（S2）：基流 q2u → 蒸发 et2
    以上顺序与 Ye et al. (2012) 描述及其他模型版本保持一致。
    """

    # ── Guards：防止前一步数值误差累积产生的微小负值 ─────────────
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Bucket 1 (Surface Soil / S1)
    # 顺序更新，保证每步后 S1 ≥ 0，且无需水量守恒修正
    # ============================================================

    # Step 1：加雨 + 计算饱和径流（比例型，sigmoid 平滑）
    # MATLAB: flux_q1f = saturation_1(P, S1, s1max)
    # 注意：saturation_1_smooth 基于加雨前的 S1 计算产流比例，
    # 符合 MATLAB 原版语义（S1 越接近 Smax，本次降雨产流比例越大）
    flux_q1f = saturation_1(P, S1, Sb1, nearzero=nearzero)
    S1 = S1 + P - flux_q1f          # q1f ≤ P（frac_runoff ∈ [0,1]），S1 非负

    # Step 2：蒸发（从加雨扣溢流后的 S1 计算）
    # MATLAB: flux_et1 = evap_7(S1, s1max, Ep, dt)
    flux_et1 = evap_7(S1, Sb1, PET, delta_t, nearzero)
    # 顺序步进安全截断：evap_7 内部已有 min(et_pot, S1)，此处再保一道
    flux_et1 = torch.minimum(flux_et1, S1)
    S1 = S1 - flux_et1              # S1 ≥ 0 保证

    # Step 3：下渗（从扣蒸发后的 S1 计算）
    # MATLAB: flux_qw = recharge_3(tw, S1)  →  tw * S1
    flux_qw = recharge_3(tw, S1)    # min(tw*S1, S1) ≤ S1
    S1_new  = S1 - flux_qw          # S1_new ≥ 0 保证，无需 clamp

    # ============================================================
    # Bucket 2 (Subsurface / S2)
    # ============================================================

    S2 = S2 + flux_qw               # 接收下渗，S2 ≥ 0

    # Step 1：基流（先于蒸发，保证 S2 有足够余量供蒸发）
    # MATLAB: flux_q2u = baseflow_1(tu, S2)  →  tu * S2
    flux_q2u = baseflow_1(tu, S2)   # min(tu*S2, S2) ≤ S2
    S2 = S2 - flux_q2u              # S2 ≥ 0 保证

    # Step 2：蒸发（从扣基流后的 S2 计算）
    # MATLAB: flux_et2 = evap_7(S2, se, Ep, dt)
    # 与 MATLAB 的细微偏差：MATLAB ODE 中 et2 和 q2u 同时作用于同一 S2，
    # 此处顺序步进，et2 基于扣除 q2u 后的 S2，离散误差极小但保证非负。
    flux_et2 = evap_7(S2, Se, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)   # 顺序步进安全截断
    S2_new   = S2 - flux_et2        # S2_new ≥ 0 保证，无需 clamp

    # ============================================================
    # Routing (Sc1 = S3, Sc2 = S4)
    # MATLAB: flux_qf = baseflow_1(tc, S3)；flux_qs = baseflow_1(tc, S4)
    # ============================================================

    Sc1     = Sc1 + flux_q1f
    flux_qf = baseflow_1(tc, Sc1)   # min(tc*Sc1, Sc1) ≤ Sc1
    Sc1_new = Sc1 - flux_qf         # Sc1_new ≥ 0 保证，无需 clamp

    Sc2     = Sc2 + flux_q2u
    flux_qs = baseflow_1(tc, Sc2)   # min(tc*Sc2, Sc2) ≤ Sc2
    Sc2_new = Sc2 - flux_qs         # Sc2_new ≥ 0 保证，无需 clamp

    # ============================================================
    # Output
    # MATLAB: FluxGroups.Ea = [et1, et2]；FluxGroups.Q = [qf, qs]
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new
