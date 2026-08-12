#!/usr/bin/env python3
"""Water balance validation for ALL 10 UH models in UH=on mode.

Four intermediate (B-type): flexi(10), flexb(9), flexis(12), gr4j(4)
Six endpoint (A-type): newzealand2(8), hillslope(7), plateau(8), smar(8),
                        ihacres(6), hbv96(15)

Computes per model:
  - P_total, Q_total, Ea_total, dS_total
  - residual_abs = P - (Q + Ea + dS)
  - residual_rel (%)
  - uh_mass_rel_diff (%) — pre-UH vs post-UH mass conservation
"""

import os
import sys
# Ensure the local dmotpy package (parent dir of scripts/) is imported,
# NOT the stale site-packages copy.
_LOCAL_DMOTPY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LOCAL_DMOTPY not in sys.path:
    sys.path.insert(0, _LOCAL_DMOTPY)

import torch

from models.hydrology_model import HydrologyModel

DEVICE = torch.device("cpu")
N_STEPS = 200
N_GRID = 2
N_GROUPS = 1
WARM_UP = 10
SEED = 20260627
NEARZERO = 1e-6

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_forcing():
    rng = torch.Generator(device=DEVICE).manual_seed(SEED)
    P = torch.rand(N_STEPS, N_GRID, N_GROUPS, device=DEVICE, generator=rng) * 10.0
    T = torch.rand(N_STEPS, N_GRID, N_GROUPS, device=DEVICE, generator=rng) * 5.0 - 2.0
    PET = torch.rand(N_STEPS, N_GRID, N_GROUPS, device=DEVICE, generator=rng) * 3.0
    return P, T, PET


def _make_raw_params(n_params):
    rng = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    return torch.rand(1, n_params, N_GROUPS, device=DEVICE, generator=rng)


def _uh_mass_rel(mass_in, mass_out):
    denom = abs(mass_in) + 1e-12
    return abs(mass_out - mass_in) / denom * 100.0


# ---------------------------------------------------------------------------
# Intermediate-model validator (flexi / flexb / flexis)
# ---------------------------------------------------------------------------


def validate_intermediate_standard(model_name, n_params):
    config = {
        "model_name": model_name,
        "warm_up": WARM_UP,
        "uh_enabled": True,
        "uh_mode": "intermediate",
        "backend": "none",
    }
    model = HydrologyModel(config=config, device=DEVICE, backend="none")
    P, T, PET = _make_forcing()
    params_dict = model._descale_params(_make_raw_params(n_params))

    pre_params = [params_dict[name] for name in model._pre_param_names]
    post_params = [params_dict[name] for name in model._post_param_names]
    n_pre = model._n_pre_states
    n_post = model._n_post_states
    n_passthru = model._n_pre_passthru

    states = list(model._init_states(N_GRID, N_GROUPS))
    pre_states = states[:n_pre]
    post_states = states[n_pre:]
    pre_init = [s.clone() for s in pre_states]
    post_init = [s.clone() for s in post_states]

    total_P = 0.0
    uh_fast, uh_slow = [], []
    passthru = [[] for _ in range(n_passthru)]

    # --- production loop ---
    for t in range(N_STEPS):
        total_P += P[t].sum().item()
        res = model.step_pre_fn(
            P[t], T[t], PET[t], *pre_params, *pre_states, NEARZERO,
        )
        uh_fast.append(res[0])
        uh_slow.append(res[1])
        for i in range(n_passthru):
            passthru[i].append(res[2 + i])
        pre_states = list(res[2 + n_passthru:])

    dS_pre = sum((sf - si).sum().item() for sf, si in zip(pre_states, pre_init))

    # --- UH convolution ---
    rf_stack = torch.stack(uh_fast, dim=0)
    rs_stack = torch.stack(uh_slow, dim=0)
    B = N_GRID * N_GROUPS
    rf_flat = rf_stack.permute(1, 2, 0).reshape(B, N_STEPS)
    rs_flat = rs_stack.permute(1, 2, 0).reshape(B, N_STEPS)

    nlagf = params_dict["nlagf"].expand(N_GRID, N_GROUPS).reshape(B, 1)
    nlags = params_dict["nlags"].expand(N_GRID, N_GROUPS).reshape(B, 1)

    routed_rf = model.uh_fast(rf_flat, nlagf)
    routed_rs = model.uh_slow(rs_flat, nlags)

    uh_mass_in = (rf_flat.sum() + rs_flat.sum()).item()
    uh_mass_out = (routed_rf.sum() + routed_rs.sum()).item()

    rf_seq = routed_rf.view(N_GRID, N_GROUPS, N_STEPS).permute(2, 0, 1).unbind(0)
    rs_seq = routed_rs.view(N_GRID, N_GROUPS, N_STEPS).permute(2, 0, 1).unbind(0)

    # --- routing loop ---
    total_Q = 0.0
    total_Ea = 0.0
    for t in range(N_STEPS):
        vals = [passthru[i][t] for i in range(n_passthru)]
        Qsim, Ea, *post_new = model.step_post_fn(
            rf_seq[t], rs_seq[t], *vals, *post_states, *post_params, NEARZERO,
        )
        post_states = list(post_new)
        total_Q += Qsim.sum().item()
        total_Ea += Ea.sum().item()

    dS_post = sum((sf - si).sum().item() for sf, si in zip(post_states, post_init))
    dS_total = dS_pre + dS_post

    residual = total_P - total_Q - total_Ea - dS_total
    rel_res = abs(residual) / (abs(total_P) + 1e-12) * 100.0
    uh_rel = _uh_mass_rel(uh_mass_in, uh_mass_out)

    return _build_result(model_name, "intermediate", total_P, total_Q, total_Ea, dS_total, residual, rel_res, uh_rel)


# ---------------------------------------------------------------------------
# Intermediate-model validator (GR4J — special routing with half/full UH)
# ---------------------------------------------------------------------------


def validate_gr4j():
    model_name = "gr4j"
    n_params = 4
    config = {
        "model_name": model_name,
        "warm_up": WARM_UP,
        "uh_enabled": True,
        "uh_mode": "intermediate",
        "backend": "none",
    }
    model = HydrologyModel(config=config, device=DEVICE, backend="none")
    P, T, PET = _make_forcing()
    params_dict = model._descale_params(_make_raw_params(n_params))

    pre_params = [params_dict[name] for name in model._pre_param_names]
    post_params = [params_dict[name] for name in model._post_param_names]
    n_pre = model._n_pre_states
    n_post = model._n_post_states

    states = list(model._init_states(N_GRID, N_GROUPS))
    pre_states = states[:n_pre]
    post_states = states[n_pre:]
    pre_init = [s.clone() for s in pre_states]
    post_init = [s.clone() for s in post_states]

    total_P = 0.0
    pr_list, ephys_list = [], []

    # --- production loop ---
    for t in range(N_STEPS):
        total_P += P[t].sum().item()
        flux_pr, e_physical, *pre_new = model.step_pre_fn(
            P[t], T[t], PET[t], *pre_params, *pre_states, NEARZERO,
        )
        pre_states = list(pre_new)
        pr_list.append(flux_pr)
        ephys_list.append(e_physical)

    dS_pre = sum((sf - si).sum().item() for sf, si in zip(pre_states, pre_init))

    # --- UH convolution ---
    pr_stack = torch.stack(pr_list, dim=0)
    B = N_GRID * N_GROUPS
    pr_flat = pr_stack.permute(1, 2, 0).reshape(B, N_STEPS)

    flux_q9 = pr_flat * 0.9
    flux_q1 = pr_flat * 0.1

    x4 = params_dict["x4"].expand(N_GRID, N_GROUPS).reshape(B, 1)
    x4_2 = x4 * 2.0

    routed_q9 = model.uh_half(flux_q9, x4)
    routed_q1 = model.uh_full(flux_q1, x4_2)

    uh_mass_in = pr_flat.sum().item()
    uh_mass_out = (routed_q9.sum() + routed_q1.sum()).item()

    q9_seq = routed_q9.view(N_GRID, N_GROUPS, N_STEPS).permute(2, 0, 1).unbind(0)
    q1_seq = routed_q1.view(N_GRID, N_GROUPS, N_STEPS).permute(2, 0, 1).unbind(0)

    # --- routing loop ---
    total_Q = 0.0
    total_Ea = 0.0
    for t in range(N_STEPS):
        Qsim, Ea_balanced, *post_new = model.step_post_fn(
            q9_seq[t], q1_seq[t], *post_states, *post_params, ephys_list[t], NEARZERO,
        )
        post_states = list(post_new)
        total_Q += Qsim.sum().item()
        total_Ea += Ea_balanced.sum().item()

    dS_post = sum((sf - si).sum().item() for sf, si in zip(post_states, post_init))
    dS_total = dS_pre + dS_post

    residual = total_P - total_Q - total_Ea - dS_total
    rel_res = abs(residual) / (abs(total_P) + 1e-12) * 100.0
    uh_rel = _uh_mass_rel(uh_mass_in, uh_mass_out)

    return _build_result(model_name, "intermediate", total_P, total_Q, total_Ea, dS_total, residual, rel_res, uh_rel)


# ---------------------------------------------------------------------------
# Endpoint-model validators
# ---------------------------------------------------------------------------


def _validate_endpoint(model_name, n_params, kind):
    """Generic endpoint validator — runs the production loop step-by-step,
    applies UH routing exactly as EndpointUHModel._run_model does, and
    tracks complete water balance."""
    config = {
        "model_name": model_name,
        "warm_up": WARM_UP,
        "uh_enabled": True,
        "uh_mode": "endpoint",
        "backend": "none",
    }
    model = HydrologyModel(config=config, device=DEVICE, backend="none")
    P, T, PET = _make_forcing()
    params_dict = model._descale_params(_make_raw_params(n_params))

    param_vals = [params_dict[name] for name in model.phy_param_names]
    curr_states = list(model._init_states(N_GRID, N_GROUPS))
    init_states = [s.clone() for s in curr_states]
    scheme = model._endpoint_scheme

    need_split = kind in ("surface_baseflow", "exp_delay_chain")

    total_P = 0.0
    total_Ea = 0.0
    qsim_list = []
    surface_list = [] if need_split else None
    baseflow_list = [] if need_split else None

    # --- production loop ---
    for t in range(N_STEPS):
        total_P += P[t].sum().item()
        kwargs = {}
        if need_split:
            kwargs["return_routing_fluxes"] = True

        outputs = model.step_fn(
            P[t], T[t], PET[t], *param_vals, *curr_states, NEARZERO, **kwargs,
        )

        total_Ea += outputs[1].sum().item()

        if need_split:
            qsim_list.append(outputs[0])
            fluxes = outputs[-1]
            surface_list.append(fluxes[0])
            baseflow_list.append(fluxes[1])
            curr_states = list(outputs[2:-1])
        else:
            qsim_list.append(outputs[0])
            curr_states = list(outputs[2:])

    dS_total = sum((sf - si).sum().item() for sf, si in zip(curr_states, init_states))

    # --- UH routing (replicates EndpointUHModel._run_model) ---
    B = N_GRID * N_GROUPS

    if kind == "total":
        qsim_stack = torch.stack(qsim_list, dim=0)
        qsim_flat = qsim_stack.permute(1, 2, 0).reshape(B, N_STEPS)
        uh_param_name = scheme["uhs"][0][1]
        uh_param = params_dict[uh_param_name].expand(N_GRID, N_GROUPS).reshape(B, 1)

        uh_mass_in = qsim_flat.sum().item()
        routed = model.uh_modules[0](qsim_flat, uh_param)
        uh_mass_out = routed.sum().item()

        total_Q = routed.sum().item()

    elif kind == "surface_baseflow":
        surf_stack = torch.stack(surface_list, dim=0)
        base_stack = torch.stack(baseflow_list, dim=0)
        surf_flat = surf_stack.permute(1, 2, 0).reshape(B, N_STEPS)

        if scheme["uhs"][0][0] == "gamma6":
            n_res = params_dict["n_res"].expand(N_GRID, N_GROUPS).reshape(B, 1)
            nk_delay = params_dict["nk_delay"].expand(N_GRID, N_GROUPS).reshape(B, 1)
            k_val = nk_delay / (n_res + NEARZERO)
            uh_params = torch.cat([n_res, k_val], dim=1)
        else:
            uh_param_name = scheme["uhs"][0][1]
            uh_params = params_dict[uh_param_name].expand(N_GRID, N_GROUPS).reshape(B, 1)

        uh_mass_in = surf_flat.sum().item()
        routed = model.uh_modules[0](surf_flat, uh_params)
        uh_mass_out = routed.sum().item()

        routed_surf = routed.view(N_GRID, N_GROUPS, N_STEPS).permute(2, 0, 1)
        total_Q = (routed_surf + base_stack).sum().item()

    elif kind == "exp_delay_chain":
        uq_stack = torch.stack(surface_list, dim=0)
        us_stack = torch.stack(baseflow_list, dim=0)
        uq_flat = uq_stack.permute(1, 2, 0).reshape(B, N_STEPS)
        us_flat = us_stack.permute(1, 2, 0).reshape(B, N_STEPS)

        tau_q = params_dict["tau_q"].expand(N_GRID, N_GROUPS).reshape(B, 1)
        tau_s = params_dict["tau_s"].expand(N_GRID, N_GROUPS).reshape(B, 1)

        routed_uq = model.uh_modules[0](uq_flat, tau_q)
        routed_us = model.uh_modules[1](us_flat, tau_s)

        uh_mass_in = (uq_flat.sum() + us_flat.sum()).item()
        routed_total = routed_uq + routed_us
        uh_mass_out = routed_total.sum().item()
        total_Q = routed_total.sum().item()

    else:
        raise ValueError(f"Unknown endpoint kind: {kind}")

    residual = total_P - total_Q - total_Ea - dS_total
    rel_res = abs(residual) / (abs(total_P) + 1e-12) * 100.0
    uh_rel = _uh_mass_rel(uh_mass_in, uh_mass_out)

    return _build_result(model_name, "endpoint", total_P, total_Q, total_Ea, dS_total, residual, rel_res, uh_rel)


# ---------------------------------------------------------------------------
# Result helper
# ---------------------------------------------------------------------------


def _build_result(name, mtype, p, q, ea, ds, res_abs, res_rel, uh_rel):
    return {
        "model": name,
        "type": mtype,
        "P_total": p,
        "Q_total": q,
        "Ea": ea,
        "dS": ds,
        "residual_abs": res_abs,
        "residual_rel_pct": res_rel,
        "uh_mass_rel_diff_pct": uh_rel,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    results = []

    # Intermediate (B-type)
    results.append(validate_intermediate_standard("flexi", 10))
    results.append(validate_intermediate_standard("flexb", 9))
    results.append(validate_intermediate_standard("flexis", 12))
    results.append(validate_gr4j())

    # Endpoint (A-type)
    results.append(_validate_endpoint("newzealand2", 8, "total"))
    results.append(_validate_endpoint("hillslope", 7, "surface_baseflow"))
    results.append(_validate_endpoint("plateau", 8, "surface_baseflow"))
    results.append(_validate_endpoint("smar", 8, "surface_baseflow"))
    results.append(_validate_endpoint("ihacres", 6, "exp_delay_chain"))
    results.append(_validate_endpoint("hbv96", 15, "total"))

    # --- Print table ---
    header = (
        f"{'Model':<14} {'Type':<14} {'P_total':>10} {'Q_total':>10} "
        f"{'Ea':>10} {'dS':>10} {'res_abs':>12} {'res_rel%':>9} {'UH_mass%':>9}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['model']:<14} {r['type']:<14} "
            f"{r['P_total']:10.3f} {r['Q_total']:10.3f} "
            f"{r['Ea']:10.3f} {r['dS']:10.3f} "
            f"{r['residual_abs']:12.4e} {r['residual_rel_pct']:8.3f} {r['uh_mass_rel_diff_pct']:8.3f}"
        )
    print(sep)

    # --- Summary ---
    high_wb = [r for r in results if r["residual_rel_pct"] > 5.0]
    high_uh = [r for r in results if r["uh_mass_rel_diff_pct"] > 3.0]

    print()
    if high_wb:
        print(f"Models exceeding 5% water balance residual: {[r['model'] for r in high_wb]}")
    else:
        print("All 10 models pass water balance check (residual < 5%).")

    if high_uh:
        print(f"Models with UH mass diff > 3%: {[r['model'] for r in high_uh]}")
    else:
        print("All 10 models pass UH mass conservation check (diff < 3%).")

    return results


if __name__ == "__main__":
    main()
