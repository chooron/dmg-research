"""UH tail mass balance audit: compute finite-window truncation mass for all 10 UH models."""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from models.hydrology_model import HydrologyModel
from models.core.flexi import flexi_step_pre, flexi_step_post
from models.core.flexb import flexb_step_pre, flexb_step_post
from models.core.flexis import flexis_step_pre, flexis_step_post
from models.core.gr4j import gr4j_step_pre, gr4j_step_post
from models.core.newzealand2 import newzealand2_step
from models.core.hillslope import hillslope_step
from models.core.plateau import plateau_step
from models.core.smar import smar_step
from models.core.ihacres import ihacres_step
from models.core.hbv96 import hbv96_step
from models.unithydro.uh_tri_3 import DplTri3
from models.unithydro.uh_tri_4 import DplTri4
from models.unithydro.uh_exp_5 import DplExp5
from models.unithydro.uh_gamma_6 import DplGamma6
from models.unithydro.uh_delay_8 import DplDelay8
from models.unithydro.uh_half_1 import DplHalf1
from models.unithydro.uh_full_2 import DplFull2
from models.unithydro.uh_uniform_7 import DplUniform7

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "validation_results" / "uh_core_integration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_uh_tail_mass(uh_module, flat_input, uh_params, seq_len):
    """Compute the tail mass that the Conv1d truncates.

    The UH forward() applies symmetric padding then slices the right side.
    The tail mass is the portion of the convolved output beyond the sequence end.
    """
    padd = uh_module.max_lag - 1
    batch_size = flat_input.shape[0]

    # Replicate UH forward logic without truncation
    raw_weights = uh_module.get_weights(uh_params)
    sum_w = raw_weights.sum(dim=-1, keepdim=True)
    denom = torch.where(
        sum_w > uh_module.epsilon,
        sum_w,
        torch.full_like(sum_w, uh_module.epsilon),
    )
    norm_weights = raw_weights / denom
    flipped_weights = torch.flip(norm_weights, dims=[-1])

    x = flat_input.view(1, batch_size, seq_len)
    full_output = F.conv1d(x, flipped_weights, groups=batch_size, padding=padd)

    # Truncated output (same as forward does)
    if padd > 0:
        truncated = full_output[:, :, :-padd]
    else:
        truncated = full_output

    # Tail mass = full_output total - truncated total
    tail = full_output.sum(dim=-1) - truncated.sum(dim=-1)
    return tail.view(batch_size).sum().item()


def compute_intermediate_uh_balance(model_name, n_steps=200, seed=42):
    """Compute water balance for intermediate-UH model including tail mass."""
    device = torch.device("cpu")
    n_grid, n_groups = 2, 1
    nearzero = 1e-6
    rng = torch.Generator(device=device).manual_seed(seed)

    P = torch.rand(n_steps, n_grid, n_groups, generator=rng) * 10.0
    T = torch.rand(n_steps, n_grid, n_groups, generator=rng) * 5.0 - 2.0
    PET = torch.rand(n_steps, n_grid, n_groups, generator=rng) * 3.0

    pre_fn_map = {"flexi": (flexi_step_pre, flexi_step_post), "flexb": (flexb_step_pre, flexb_step_post),
                  "flexis": (flexis_step_pre, flexis_step_post)}
    pre_fn, post_fn = pre_fn_map[model_name]

    if model_name == "flexi":
        pre_params_list = ["smax", "beta", "d_split", "percmax", "lp", "imax"]
        post_params_list = ["kf", "ks"]
        n_pre_s, n_post_s = 2, 2
        n_passthru = 2
    elif model_name == "flexb":
        pre_params_list = ["s1max", "beta", "d_split", "percmax", "lp"]
        post_params_list = ["kf", "ks"]
        n_pre_s, n_post_s = 1, 2
        n_passthru = 1
    else:  # flexis
        pre_params_list = ["smax", "beta", "d_split", "percmax", "lp", "imax", "tt", "ddf"]
        post_params_list = ["kf", "ks"]
        n_pre_s, n_post_s = 3, 2
        n_passthru = 2

    pre_params = [torch.rand(n_grid, n_groups, generator=rng) for _ in pre_params_list]
    post_params = [torch.rand(n_grid, n_groups, generator=rng) for _ in post_params_list]
    nlagf = torch.rand(n_grid, n_groups, generator=rng) * 4.0 + 1.0
    nlags = torch.rand(n_grid, n_groups, generator=rng) * 14.0 + 1.0

    pre_states = [torch.zeros(n_grid, n_groups) + nearzero for _ in range(n_pre_s)]
    post_states = [torch.zeros(n_grid, n_groups) + nearzero for _ in range(n_post_s)]
    pre_states_init = [s.clone() for s in pre_states]
    post_states_init = [s.clone() for s in post_states]

    uh_fast = DplTri3(max_lag=int(5))
    uh_slow = DplTri3(max_lag=int(15))

    rf_list, rsl_list, passthru_lists = [], [], [[] for _ in range(n_passthru)]
    for t in range(n_steps):
        result = pre_fn(P[t], T[t], PET[t], *pre_params, *pre_states, nearzero)
        rf_list.append(result[0])
        rsl_list.append(result[1])
        for i in range(n_passthru):
            passthru_lists[i].append(result[2 + i])
        pre_states = list(result[2 + n_passthru:])

    prev_pre = sum(s.sum() for s in pre_states_init)
    final_pre = sum(s.sum() for s in pre_states)
    dS_pre = final_pre - prev_pre

    rf_stack = torch.stack(rf_list, dim=0)
    rsl_stack = torch.stack(rsl_list, dim=0)
    B = n_grid * n_groups
    rf_f = rf_stack.permute(1, 2, 0).reshape(B, n_steps)
    rsl_f = rsl_stack.permute(1, 2, 0).reshape(B, n_steps)

    nlagf_f = nlagf.reshape(B, 1)
    nlags_f = nlags.reshape(B, 1)

    tail_fast = compute_uh_tail_mass(uh_fast, rf_f, nlagf_f, n_steps)
    tail_slow = compute_uh_tail_mass(uh_slow, rsl_f, nlags_f, n_steps)
    tail_total = tail_fast + tail_slow

    routed_rf = uh_fast(rf_f, nlagf_f)
    routed_rsl = uh_slow(rsl_f, nlags_f)
    rf_seq = routed_rf.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)
    rsl_seq = routed_rsl.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)

    total_P = P.sum()
    total_Q = torch.tensor(0.0)
    total_Ea = torch.tensor(0.0)
    for t in range(n_steps):
        pts = [passthru_lists[i][t] for i in range(n_passthru)]
        Qsim, Ea, *pn = post_fn(rf_seq[t], rsl_seq[t], *pts, *post_states, *post_params, nearzero)
        total_Q += Qsim.sum()
        total_Ea += Ea.sum()
        post_states = list(pn)

    prev_post = sum(s.sum() for s in post_states_init)
    final_post = sum(s.sum() for s in post_states)
    dS_post = final_post - prev_post
    dS = dS_pre + dS_post

    residual_wo = total_P - total_Q - total_Ea - dS
    residual_w = residual_wo - tail_total
    rel_wo = abs(residual_wo.item()) / (total_P.item() + 1e-12)
    rel_w = abs(residual_w.item()) / (total_P.item() + 1e-12)

    return total_P.item(), total_Q.item(), total_Ea.item(), dS.item(), tail_total, residual_wo.item(), residual_w.item(), rel_wo, rel_w


def compute_endpoint_uh_balance(model_name, n_steps=200, seed=42):
    """Compute water balance for endpoint-UH model including tail mass."""
    device = torch.device("cpu")
    n_grid, n_groups = 2, 1
    nearzero = 1e-6
    rng = torch.Generator(device=device).manual_seed(seed)

    P = torch.rand(n_steps, n_grid, n_groups, generator=rng) * 10.0
    T = torch.rand(n_steps, n_grid, n_groups, generator=rng) * 5.0 - 2.0
    PET = torch.rand(n_steps, n_grid, n_groups, generator=rng) * 3.0

    step_fn_map = {"newzealand2": newzealand2_step, "hillslope": hillslope_step,
                   "plateau": plateau_step, "smar": smar_step, "hbv96": hbv96_step}
    step_fn = step_fn_map[model_name]

    n_params_map = {"newzealand2": 8, "hillslope": 7, "plateau": 8, "smar": 8, "hbv96": 15}
    n_st_map = {"newzealand2": 2, "hillslope": 2, "plateau": 2, "smar": 6, "hbv96": 5}
    n_params = n_params_map[model_name]
    n_states = n_st_map[model_name]
    params = [torch.rand(n_grid, n_groups, generator=rng) for _ in range(n_params)]
    states = [torch.zeros(n_grid, n_groups) + nearzero for _ in range(n_states)]
    states_init = [s.clone() for s in states]

    # total-UH case
    qsim_list = []
    ea_list = []
    for t in range(n_steps):
        result = step_fn(P[t], T[t], PET[t], *params, *states, nearzero)
        qsim_list.append(result[0])
        ea_list.append(result[1] if len(result) > 1 else torch.zeros_like(result[0]))
        states = list(result[2:])

    prev_S = sum(s.sum() for s in states_init)
    final_S = sum(s.sum() for s in states)
    dS = final_S - prev_S

    qsim_stack = torch.stack(qsim_list, dim=0)
    B = n_grid * n_groups
    q_flat = qsim_stack.permute(1, 2, 0).reshape(B, n_steps)

    uh_module = DplTri4(max_lag=30) if model_name == "newzealand2" else (
        DplTri3(max_lag=int(120)) if model_name in ("hillslope", "plateau") else (
            DplGamma6(max_lag=int(120)) if model_name == "smar" else DplUniform7(max_lag=int(120))
        )
    )

    uh_param = params[-1] if model_name != "smar" else torch.cat([
        params[6].reshape(B, 1), params[7].reshape(B, 1) / (params[6].reshape(B, 1) + nearzero)
    ], dim=1)

    if model_name in ("hillslope", "plateau", "smar"):
        tail = 0.0
        # Complex split routing - use simplified total-UH for tail estimation
        routed = q_flat
    else:
        uh_param_flat = uh_param.reshape(B, uh_param.shape[-1]) if uh_param.dim() > 1 else uh_param.reshape(B, 1)
        tail = compute_uh_tail_mass(uh_module, q_flat, uh_param_flat, n_steps) if model_name in ("newzealand2", "hbv96") else 0.0
        routed = uh_module(q_flat, uh_param.reshape(B, -1)[:, :1] if model_name not in ("smar",) else uh_param)

    if model_name not in ("hillslope", "plateau", "smar"):
        routed_stack = routed.view(n_grid, n_groups, n_steps).permute(2, 0, 1)
    else:
        routed_stack = qsim_stack

    total_P = P.sum()
    total_Q = routed_stack.sum()
    total_Ea = torch.stack(ea_list, dim=0).sum()

    residual_wo = total_P - total_Q - total_Ea - dS
    residual_w = residual_wo - tail
    rel_wo = abs(residual_wo.item()) / (total_P.item() + 1e-12)
    rel_w = abs(residual_w.item()) / (total_P.item() + 1e-12)

    return total_P.item(), total_Q.item(), total_Ea.item(), dS.item(), tail, residual_wo.item(), residual_w.item(), rel_wo, rel_w


def compute_surface_baseflow_balance(model_name, n_steps=200, seed=42):
    """Detailed balance for surface+baseflow endpoint-UH models."""
    device = torch.device("cpu")
    n_grid, n_groups = 2, 1
    nearzero = 1e-6
    rng = torch.Generator(device=device).manual_seed(seed)

    P = torch.rand(n_steps, n_grid, n_groups, generator=rng) * 10.0
    T = torch.rand(n_steps, n_grid, n_groups, generator=rng) * 5.0 - 2.0
    PET = torch.rand(n_steps, n_grid, n_groups, generator=rng) * 3.0

    step_fn_map = {"hillslope": hillslope_step, "plateau": plateau_step, "smar": smar_step}
    step_fn = step_fn_map[model_name]
    n_params_map = {"hillslope": 7, "plateau": 8, "smar": 8}
    n_st_map = {"hillslope": 2, "plateau": 2, "smar": 6}
    n_params = n_params_map[model_name]
    n_states = n_st_map[model_name]

    params = [torch.rand(n_grid, n_groups, generator=rng) for _ in range(n_params)]
    states = [torch.zeros(n_grid, n_groups) + nearzero for _ in range(n_states)]
    states_init = [s.clone() for s in states]

    surface_list, baseflow_list, ea_list = [], [], []
    for t in range(n_steps):
        result = step_fn(P[t], T[t], PET[t], *params, *states, nearzero, return_routing_fluxes=True)
        fluxes = result[-1]
        surface_list.append(fluxes[0])
        baseflow_list.append(fluxes[1])
        ea_list.append(result[1])
        states = list(result[2:-1])

    prev_S = sum(s.sum() for s in states_init)
    final_S = sum(s.sum() for s in states)
    dS = final_S - prev_S

    surf_stack = torch.stack(surface_list, dim=0)
    base_stack = torch.stack(baseflow_list, dim=0)
    B = n_grid * n_groups

    surf_flat = surf_stack.permute(1, 2, 0).reshape(B, n_steps)

    if model_name == "smar":
        n_res = params[6].reshape(B, 1)
        nk_delay = params[7].reshape(B, 1)
        k_val = nk_delay / (n_res + nearzero)
        uh_params = torch.cat([n_res, k_val], dim=1)
        uh = DplGamma6(max_lag=int(120))
    else:
        uh = DplTri3(max_lag=int(120))
        uh_params = params[-1].reshape(B, 1)

    tail_surface = compute_uh_tail_mass(uh, surf_flat, uh_params, n_steps)
    routed = uh(surf_flat, uh_params[:, :uh_params.shape[-1]] if model_name != "smar" else uh_params)
    routed_surf = routed.view(n_grid, n_groups, n_steps).permute(2, 0, 1)
    q_total = routed_surf + base_stack

    total_P = P.sum()
    total_Q = q_total.sum()
    total_Ea = torch.stack(ea_list, dim=0).sum()

    residual_wo = total_P - total_Q - total_Ea - dS
    residual_w = residual_wo - tail_surface
    rel_wo = abs(residual_wo.item()) / (total_P.item() + 1e-12)
    rel_w = abs(residual_w.item()) / (total_P.item() + 1e-12)

    return total_P.item(), total_Q.item(), total_Ea.item(), dS.item(), tail_surface, residual_wo.item(), residual_w.item(), rel_wo, rel_w


def test_uh_tail_mass_closes_balance_known_models():
    """Verify tail mass explains hillslope/smar large residuals."""
    results = []

    # Intermediate UH models
    for name in ["flexi", "flexb", "flexis"]:
        tp, tq, tea, ds, tail, rwo, rw, rwo_rel, rw_rel = compute_intermediate_uh_balance(name)
        results.append((name, "intermediate", tp, tq, tea, ds, tail, rwo, rw, rwo_rel, rw_rel))

    # Simple endpoint models
    for name in ["newzealand2", "hbv96"]:
        tp, tq, tea, ds, tail, rwo, rw, rwo_rel, rw_rel = compute_endpoint_uh_balance(name)
        results.append((name, "endpoint_total", tp, tq, tea, ds, tail, rwo, rw, rwo_rel, rw_rel))

    # Surface+baseflow models (detailed balance)
    for name in ["hillslope", "plateau", "smar"]:
        tp, tq, tea, ds, tail, rwo, rw, rwo_rel, rw_rel = compute_surface_baseflow_balance(name)
        results.append((name, "endpoint_surface", tp, tq, tea, ds, tail, rwo, rw, rwo_rel, rw_rel))

    # Verify tail mass explanation
    for name, kind, tp, tq, tea, ds, tail, rwo, rw, rwo_rel, rw_rel in results:
        assert rw_rel < 1.0e-3, (
            f"{name}: water-balance residual remains {rw_rel * 100:.4f}% "
            "after accounting for UH water still queued beyond the finite window"
        )
        if name in ("hillslope", "smar"):
            assert rw_rel <= rwo_rel + 1e-9, f"{name}: tail mass ({tail:.1f}) should reduce or maintain residual ({rwo_rel*100:.2f}% -> {rw_rel*100:.2f}%)"

    # Verify no NaN/Inf
    for name, kind, tp, tq, tea, ds, tail, rwo, rw, rwo_rel, rw_rel in results:
        assert not any(np.isnan(x) or np.isinf(x) for x in [tp, tq, tea, ds, tail, rwo, rw])
    print("All tail mass balance checks passed")
