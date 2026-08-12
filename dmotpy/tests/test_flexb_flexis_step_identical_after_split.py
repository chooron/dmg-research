"""Golden reference and UH=on validation tests for FlexB and FlexIS models."""

import torch
import torch.nn.functional as F
from typing import Tuple

from models.hydrology_model import HydrologyModel


# ──────────────────────────────────────────────────────────────────────────
# FLEXB golden reference (original pre-split flexb_step)
# ──────────────────────────────────────────────────────────────────────────

def _flexb_step_original(
    P, T, PET, s1max, beta, d_split, percmax, lp, nlagf, nlags, kf, ks,
    S1, S2, S3, nearzero=1e-6,
):
    from models.flux.saturation import saturation_3
    from models.flux.evap import evap_3
    from models.flux.percolation import percolation_2
    from models.flux.split import split_1
    from models.flux.baseflow import baseflow_1

    _ = (nlagf, nlags)

    # --- 1. Unsaturated Zone Processes (S1) ---
    flux_ru = saturation_3(S1, s1max, beta, P, nearzero=nearzero)
    zeros = torch.zeros_like(flux_ru)
    flux_ru = torch.clamp(flux_ru, min=zeros, max=P)
    p_excess = F.relu(P - flux_ru)
    flux_rf = split_1(1.0 - d_split, p_excess, nearzero=nearzero)
    flux_rs = F.relu(p_excess - flux_rf)
    S1_tmp = S1 + flux_ru
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)
    flux_eur = evap_3(lp, S1_tmp, s1max, PET, nearzero=nearzero)
    flux_eur = torch.minimum(flux_eur, S1_tmp - nearzero)
    flux_eur = torch.minimum(flux_eur, PET)
    flux_eur = F.relu(flux_eur)
    S1_tmp2 = S1_tmp - flux_eur
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)
    flux_ps = percolation_2(percmax, S1_tmp2, s1max, nearzero=nearzero)
    flux_ps = torch.minimum(flux_ps, S1_tmp2 - nearzero)
    flux_ps = F.relu(flux_ps)
    S1_new = S1_tmp2 - flux_ps
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Routing Processes (S2 and S3) ---
    flux_rfl = flux_rf
    flux_rsl = flux_ps + flux_rs
    S2_tmp = S2 + flux_rfl
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)
    flux_qf = baseflow_1(kf, S2_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S2_tmp - nearzero)
    flux_qf = F.relu(flux_qf)
    S2_new = S2_tmp - flux_qf
    S2_new = torch.clamp(S2_new, min=nearzero)
    S3_tmp = S3 + flux_rsl
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)
    flux_qs = baseflow_1(ks, S3_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S3_tmp - nearzero)
    flux_qs = F.relu(flux_qs)
    S3_new = S3_tmp - flux_qs
    S3_new = torch.clamp(S3_new, min=nearzero)
    Qsim = flux_qf + flux_qs
    Ea = flux_eur
    return Qsim, Ea, S1_new, S2_new, S3_new


# ──────────────────────────────────────────────────────────────────────────
# FLEXIS golden reference
# ──────────────────────────────────────────────────────────────────────────

def _flexis_step_original(
    P, T, PET, smax, beta, d_split, percmax, lp, nlagf, nlags, kf, ks, imax, tt, ddf,
    S1, S2, S3, S4, S5, nearzero=1e-6,
):
    from models.flux.snowfall import snowfall_1
    from models.flux.rainfall import rainfall_1
    from models.flux.melt import melt_1
    from models.flux.interception import interception_1
    from models.flux.evap import evap_1, evap_3
    from models.flux.saturation import saturation_3
    from models.flux.percolation import percolation_2
    from models.flux.split import split_1
    from models.flux.baseflow import baseflow_1

    _ = (nlagf, nlags)

    # --- 1. Snow Process (S1) ---
    flux_ps = snowfall_1(P, T, tt, nearzero=nearzero)
    flux_pi = rainfall_1(P, T, tt, nearzero=nearzero)
    flux_m = melt_1(ddf, tt, T, S1, nearzero=nearzero)
    flux_m = torch.minimum(flux_m, S1 - nearzero)
    flux_m = F.relu(flux_m)
    S1_new = S1 + flux_ps - flux_m
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Interception Process (S2) ---
    inflow_S2 = flux_m + flux_pi
    flux_peff = interception_1(inflow_S2, S2, imax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_peff)
    flux_peff = torch.clamp(flux_peff, min=zeros, max=inflow_S2)
    S2_tmp = S2 + inflow_S2 - flux_peff
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)
    flux_ei = evap_1(S2_tmp, PET, nearzero=nearzero)
    flux_ei = torch.minimum(flux_ei, S2_tmp - nearzero)
    flux_ei = F.relu(flux_ei)
    S2_new = S2_tmp - flux_ei
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Soil Moisture Process (S3) ---
    flux_ru = saturation_3(S3, smax, beta, flux_peff, nearzero=nearzero)
    flux_ru = torch.clamp(flux_ru, min=zeros, max=flux_peff)
    rem_peff = F.relu(flux_peff - flux_ru)
    flux_rf = split_1(1.0 - d_split, rem_peff, nearzero=nearzero)
    flux_rs = F.relu(rem_peff - flux_rf)
    S3_tmp = S3 + flux_ru
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)
    PET_rem = F.relu(PET - flux_ei)
    flux_eur = evap_3(lp, S3_tmp, smax, PET_rem, nearzero=nearzero)
    flux_eur = torch.minimum(flux_eur, S3_tmp - nearzero)
    flux_eur = F.relu(flux_eur)
    S3_tmp2 = S3_tmp - flux_eur
    S3_tmp2 = torch.clamp(S3_tmp2, min=nearzero)
    flux_rp = percolation_2(percmax, S3_tmp2, smax, nearzero=nearzero)
    flux_rp = torch.minimum(flux_rp, S3_tmp2 - nearzero)
    flux_rp = F.relu(flux_rp)
    S3_new = S3_tmp2 - flux_rp
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. Routing Processes (S4 and S5) ---
    flux_rfl = flux_rf
    flux_rsl = flux_rs + flux_rp
    S4_tmp = S4 + flux_rfl
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)
    flux_qf = baseflow_1(kf, S4_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S4_tmp - nearzero)
    flux_qf = F.relu(flux_qf)
    S4_new = S4_tmp - flux_qf
    S4_new = torch.clamp(S4_new, min=nearzero)
    S5_tmp = S5 + flux_rsl
    S5_tmp = torch.clamp(S5_tmp, min=nearzero)
    flux_qs = baseflow_1(ks, S5_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S5_tmp - nearzero)
    flux_qs = F.relu(flux_qs)
    S5_new = S5_tmp - flux_qs
    S5_new = torch.clamp(S5_new, min=nearzero)
    Qsim = flux_qf + flux_qs
    Ea = flux_ei + flux_eur
    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new, S5_new


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────

def _run_bitwise_test(model_name, original_fn, refactored_fn, n_params, n_states, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)

    for i in range(100):
        P = torch.rand(3, 2, generator=rng) * 10.0
        T = torch.rand(3, 2, generator=rng) * 5.0 - 2.0
        PET = torch.rand(3, 2, generator=rng) * 3.0
        params_list = [torch.rand(3, 2, generator=rng) for _ in range(n_params)]
        states_list = [torch.rand(3, 2, generator=rng) * 10.0 + 1e-6 for _ in range(n_states)]

        orig = original_fn(P, T, PET, *params_list, *states_list, nearzero=1e-6)
        refd = refactored_fn(P, T, PET, *params_list, *states_list, nearzero=1e-6)

        for j, (a, b) in enumerate(zip(orig, refd)):
            diff = (a - b).abs().max().item()
            assert torch.allclose(a, b, atol=1e-15), (
                f"{model_name} iter {i}, elem {j}: max diff={diff:.6e}"
            )


def test_flexb_step_identical_after_split():
    from models.core.flexb import flexb_step
    _run_bitwise_test("flexb", _flexb_step_original, flexb_step, 9, 3, 20260626)


def test_flexis_step_identical_after_split():
    from models.core.flexis import flexis_step
    _run_bitwise_test("flexis", _flexis_step_original, flexis_step, 12, 5, 20260626)


def _test_uh_on_water_balance(model_name, step_pre_fn, step_post_fn, pre_params, post_params, n_pre_states, n_post_states, seed):
    import torch.nn as nn
    from models.unithydro.uh_tri_3 import DplTri3

    device = torch.device("cpu")
    n_steps = 100
    n_grid = 2
    n_groups = 1
    nearzero = 1e-6

    rng = torch.Generator(device=device).manual_seed(seed)

    P = torch.rand(n_steps, n_grid, n_groups, device=device, generator=rng) * 10.0
    T = torch.rand(n_steps, n_grid, n_groups, device=device, generator=rng) * 5.0 - 2.0
    PET = torch.rand(n_steps, n_grid, n_groups, device=device, generator=rng) * 3.0

    pre_param_tensors = [torch.rand(n_grid, n_groups, device=device, generator=rng) for _ in pre_params]
    post_param_tensors = [torch.rand(n_grid, n_groups, device=device, generator=rng) for _ in post_params]

    all_param_tensors = pre_param_tensors + post_param_tensors
    nlagf_val = torch.rand(n_grid, n_groups, device=device, generator=rng) * 4.0 + 1.0
    nlags_val = torch.rand(n_grid, n_groups, device=device, generator=rng) * 14.0 + 1.0

    pre_states = [torch.zeros(n_grid, n_groups, device=device) + nearzero for _ in range(n_pre_states)]
    post_states = [torch.zeros(n_grid, n_groups, device=device) + nearzero for _ in range(n_post_states)]
    pre_states_init = [s.clone() for s in pre_states]
    post_states_init = [s.clone() for s in post_states]

    uh_fast = DplTri3(max_lag=int(5))
    uh_slow = DplTri3(max_lag=int(15))

    # Production loop
    uh_inputs = [[], []]
    passthru_lists = []
    for t in range(n_steps):
        result = step_pre_fn(
            P[t], T[t], PET[t],
            *pre_param_tensors, *pre_states, nearzero,
        )
        uh_inputs[0].append(result[0])
        uh_inputs[1].append(result[1])
        n_passthru = len(result) - 2 - n_pre_states
        if not passthru_lists:
            passthru_lists = [[] for _ in range(n_passthru)]
        for i in range(n_passthru):
            passthru_lists[i].append(result[2 + i])
        pre_states = list(result[2 + n_passthru:])

    pre_states_final = [s.clone() for s in pre_states]

    # UH convolution
    rf_stack = torch.stack(uh_inputs[0], dim=0)
    rsl_stack = torch.stack(uh_inputs[1], dim=0)
    B_total = n_grid * n_groups
    rf_flat = rf_stack.permute(1, 2, 0).reshape(B_total, n_steps)
    rsl_flat = rsl_stack.permute(1, 2, 0).reshape(B_total, n_steps)
    nlagf_flat = nlagf_val.reshape(B_total, 1)
    nlags_flat = nlags_val.reshape(B_total, 1)
    routed_rf = uh_fast(rf_flat, nlagf_flat)
    routed_rsl = uh_slow(rsl_flat, nlags_flat)
    rf_seq = routed_rf.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)
    rsl_seq = routed_rsl.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)

    # Routing loop
    total_P = P.sum()
    total_Q = torch.tensor(0.0, device=device)
    total_Ea = torch.tensor(0.0, device=device)
    for t in range(n_steps):
        passthru_vals = [passthru_lists[i][t] for i in range(len(passthru_lists))]
        Qsim, Ea, *post_new = step_post_fn(
            rf_seq[t], rsl_seq[t], *passthru_vals,
            *post_states, *post_param_tensors, nearzero,
        )
        total_Q = total_Q + Qsim.sum()
        total_Ea = total_Ea + Ea.sum()
        post_states = list(post_new)

    # Water balance
    dS_pre = sum((f - i).sum() for f, i in zip(pre_states_final, pre_states_init))
    dS_post = sum((f - i).sum() for f, i in zip(post_states, post_states_init))
    dS_total = dS_pre + dS_post

    residual = total_P - total_Q - total_Ea - dS_total
    relative_residual = abs(residual.item()) / (total_P.item() + 1e-12)

    assert relative_residual < 5e-2, (
        f"Water balance not closed (UH=on {model_name}).\n"
        f"  P={total_P.item():.4f}  Q={total_Q.item():.4f}  Ea={total_Ea.item():.4f}  dS={dS_total.item():.4f}\n"
        f"  residual={residual.item():.4e}  rel={relative_residual:.6e}"
    )


def test_flexb_uh_on_water_balance():
    from models.core.flexb import flexb_step_pre, flexb_step_post
    _test_uh_on_water_balance(
        "flexb", flexb_step_pre, flexb_step_post,
        pre_params=["s1max", "beta", "d_split", "percmax", "lp"],
        post_params=["kf", "ks"],
        n_pre_states=1, n_post_states=2,
        seed=20260627,
    )


def test_flexis_uh_on_water_balance():
    from models.core.flexis import flexis_step_pre, flexis_step_post
    _test_uh_on_water_balance(
        "flexis", flexis_step_pre, flexis_step_post,
        pre_params=["smax", "beta", "d_split", "percmax", "lp", "imax", "tt", "ddf"],
        post_params=["kf", "ks"],
        n_pre_states=3, n_post_states=2,
        seed=20260628,
    )


def test_flexb_uh_on_vs_off_different():
    device = torch.device("cpu")
    forcing = torch.rand(100, 2, 3, device=device) * 5
    params = (None, torch.rand(1, 9, device=device) * 0.5)
    m_off = HydrologyModel(config={"model_name": "flexb", "warm_up": 10, "backend": "none"}, device=device)
    m_on = HydrologyModel(config={"model_name": "flexb", "warm_up": 10, "uh_enabled": True, "uh_mode": "intermediate", "backend": "none"}, device=device)
    o_off = m_off({"x_phy": forcing}, params)
    o_on = m_on({"x_phy": forcing}, params)
    assert abs(o_off["streamflow"].mean().item() - o_on["streamflow"].mean().item()) > 1e-8


def test_flexis_uh_on_vs_off_different():
    device = torch.device("cpu")
    forcing = torch.rand(100, 2, 3, device=device) * 5
    params = (None, torch.rand(1, 12, device=device) * 0.5)
    m_off = HydrologyModel(config={"model_name": "flexis", "warm_up": 10, "backend": "none"}, device=device)
    m_on = HydrologyModel(config={"model_name": "flexis", "warm_up": 10, "uh_enabled": True, "uh_mode": "intermediate", "backend": "none"}, device=device)
    o_off = m_off({"x_phy": forcing}, params)
    o_on = m_on({"x_phy": forcing}, params)
    assert abs(o_off["streamflow"].mean().item() - o_on["streamflow"].mean().item()) > 1e-8
