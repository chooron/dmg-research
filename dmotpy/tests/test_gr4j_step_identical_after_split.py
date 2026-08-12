"""Golden reference and UH=on validation tests for GR4J model."""

import torch
import torch.nn.functional as F
from typing import Tuple

from models.hydrology_model import HydrologyModel
from models.core.gr4j import (
    _calc_production_store_tanh,
    _calc_percolation_analytical,
    _calc_routing_outflow_analytical,
)


def _gr4j_step_original(
    P, T, PET, x1, x2, x3, x4, S1, S2, nearzero=1e-6,
):
    diff = P - PET
    flux_pn = F.relu(diff)
    flux_en = F.relu(-diff)
    flux_ei = P - flux_pn
    nearzero_tensor = torch.zeros_like(flux_pn) + nearzero
    S1 = torch.clamp(S1, min=nearzero_tensor, max=x1)
    S2 = torch.clamp(S2, min=nearzero)

    flux_ps, flux_es = _calc_production_store_tanh(S1, x1, flux_pn, flux_en, nearzero)
    S1_mid = S1 - flux_es + flux_ps
    S1_mid = torch.clamp(S1_mid, min=nearzero_tensor, max=x1)
    flux_perc = _calc_percolation_analytical(S1_mid, x1, nearzero)
    S1_new = S1_mid - flux_perc
    S1_new = torch.clamp(S1_new, min=nearzero_tensor, max=x1)

    flux_pr = flux_perc + (flux_pn - flux_ps)
    flux_q9 = 0.9 * flux_pr
    flux_q1 = 0.1 * flux_pr

    flux_f_theoretical = x2 * (S2 / (x3 + nearzero)).pow(3.5)
    S2_before_exchange = S2
    S2_integrated = S2 + flux_q9 + flux_f_theoretical
    S2_integrated = torch.clamp(S2_integrated, min=nearzero)
    f_actual_s2 = (S2_integrated - S2_before_exchange) - flux_q9
    flux_qr = _calc_routing_outflow_analytical(S2_integrated, x3, nearzero)
    S2_new = S2_integrated - flux_qr

    flux_qd_potential = flux_q1 + flux_f_theoretical
    flux_qd = F.relu(flux_qd_potential)
    f_actual_q1 = flux_qd - flux_q1

    Qsim = flux_qr + flux_qd
    E_physical = flux_ei + flux_es
    F_total_actual = f_actual_s2 + f_actual_q1
    Ea_balanced = E_physical - F_total_actual

    return Qsim, Ea_balanced, S1_new, S2_new


def test_gr4j_step_identical_after_split():
    from models.core.gr4j import gr4j_step

    rng = torch.Generator().manual_seed(20260629)

    for i in range(100):
        P = torch.rand(3, 2, generator=rng) * 10.0
        T = torch.rand(3, 2, generator=rng) * 5.0 - 2.0
        PET = torch.rand(3, 2, generator=rng) * 3.0
        x1 = torch.rand(3, 2, generator=rng) * 1999.0 + 1.0
        x2 = torch.rand(3, 2, generator=rng) * 40.0 - 20.0
        x3 = torch.rand(3, 2, generator=rng) * 299.0 + 1.0
        x4 = torch.rand(3, 2, generator=rng) * 14.5 + 0.5
        S1 = torch.rand(3, 2, generator=rng) * 500.0 + 1e-6
        S2 = torch.rand(3, 2, generator=rng) * 100.0 + 1e-6

        orig = _gr4j_step_original(P, T, PET, x1, x2, x3, x4, S1, S2)
        refd = gr4j_step(P, T, PET, x1, x2, x3, x4, S1, S2)

        for j, (a, b) in enumerate(zip(orig, refd)):
            diff = (a - b).abs().max().item()
            # GR4J uses analytical formulas with tanh/exp/pow, atol=1e-12
            assert torch.allclose(a, b, atol=1e-12), (
                f"GR4J iter {i}, elem {j}: max diff={diff:.6e}"
            )


def test_gr4j_uh_on_water_balance():
    from models.unithydro.uh_half_1 import DplHalf1
    from models.unithydro.uh_full_2 import DplFull2
    from models.core.gr4j import gr4j_step_pre, gr4j_step_post

    device = torch.device("cpu")
    n_steps = 200
    n_grid = 2
    n_groups = 1
    nearzero = 1e-6

    rng = torch.Generator(device=device).manual_seed(20260630)

    P = torch.rand(n_steps, n_grid, n_groups, device=device, generator=rng) * 10.0
    T = torch.rand(n_steps, n_grid, n_groups, device=device, generator=rng) * 5.0 - 2.0
    PET = torch.rand(n_steps, n_grid, n_groups, device=device, generator=rng) * 3.0

    x1 = torch.rand(n_grid, n_groups, device=device, generator=rng) * 1999.0 + 1.0
    x2 = torch.rand(n_grid, n_groups, device=device, generator=rng) * 40.0 - 20.0
    x3 = torch.rand(n_grid, n_groups, device=device, generator=rng) * 299.0 + 1.0
    x4 = torch.rand(n_grid, n_groups, device=device, generator=rng) * 14.5 + 0.5

    S1 = torch.zeros(n_grid, n_groups, device=device) + nearzero
    S2 = torch.zeros(n_grid, n_groups, device=device) + nearzero
    S1_init = S1.clone()
    S2_init = S2.clone()

    uh_half = DplHalf1(max_lag=int(15) + 1)
    uh_full = DplFull2(max_lag=int(15) * 2 + 2)

    pr_list, ephys_list = [], []
    for t in range(n_steps):
        flux_pr, e_physical, S1_new = gr4j_step_pre(
            P[t], T[t], PET[t], x1, S1, nearzero,
        )
        pr_list.append(flux_pr)
        ephys_list.append(e_physical)
        S1 = S1_new

    S1_final = S1.clone()

    pr_stack = torch.stack(pr_list, dim=0)
    B_total = n_grid * n_groups
    pr_flat = pr_stack.permute(1, 2, 0).reshape(B_total, n_steps)

    flux_q9 = pr_flat * 0.9
    flux_q1 = pr_flat * 0.1

    x4_flat = x4.reshape(B_total, 1)
    routed_q9 = uh_half(flux_q9, x4_flat)
    routed_q1 = uh_full(flux_q1, x4_flat * 2.0)

    q9_seq = routed_q9.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)
    q1_seq = routed_q1.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)

    total_P = P.sum()
    total_Q = torch.tensor(0.0, device=device)
    total_Ea = torch.tensor(0.0, device=device)
    for t in range(n_steps):
        Qsim, Ea, S2_new = gr4j_step_post(
            q9_seq[t], q1_seq[t], S2, x2, x3, ephys_list[t], nearzero,
        )
        total_Q = total_Q + Qsim.sum()
        total_Ea = total_Ea + Ea.sum()
        S2 = S2_new

    dS1 = (S1_final - S1_init).sum()
    dS2 = (S2 - S2_init).sum()
    dS_total = dS1 + dS2

    residual = total_P - total_Q - total_Ea - dS_total
    relative_residual = abs(residual.item()) / (total_P.item() + 1e-12)

    assert relative_residual < 5e-2, (
        f"GR4J water balance not closed.\n"
        f"  P={total_P.item():.4f}  Q={total_Q.item():.4f}  Ea={total_Ea.item():.4f}  dS={dS_total.item():.4f}\n"
        f"  residual={residual.item():.4e}  rel={relative_residual:.6e}"
    )


def test_gr4j_uh_on_vs_off_different():
    device = torch.device("cpu")
    forcing = torch.rand(100, 2, 3, device=device) * 5
    params = (None, torch.rand(1, 4, device=device) * 0.5)
    m_off = HydrologyModel(config={"model_name": "gr4j", "warm_up": 10, "backend": "none"}, device=device)
    m_on = HydrologyModel(config={"model_name": "gr4j", "warm_up": 10, "uh_enabled": True, "uh_mode": "intermediate", "backend": "none"}, device=device)
    o_off = m_off({"x_phy": forcing}, params)
    o_on = m_on({"x_phy": forcing}, params)
    assert abs(o_off["streamflow"].mean().item() - o_on["streamflow"].mean().item()) > 1e-8
