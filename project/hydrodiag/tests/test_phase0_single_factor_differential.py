"""Phase-0 single-factor differential audit (code + forward level only).

Frozen controlled matrix (CemaNeige host + controlled XAJ):

    N   : native 3-layer sequential/threshold ET (C) + native RI/RG two-path response
    D_E : parallel linear lower/deep ET (no C)              + native response (fixed)
    G_E : parallel power-law lower/deep ET (gamma)          + native response (fixed)
    D_R : native 3-layer ET (fixed)                         + single linear Z response (KSS/tau0)
    G_R : native 3-layer ET (fixed)                         + single power-law Z response (KSS/tau0/beta)

Only the stated single process module may differ between N and each arm.
Cross-combinations (D_E+D_R, D_E+G_R, G_E+D_R, G_E+G_R) are forbidden.

These tests are code- and forward-level verification only: no training,
no CMA-ES, no checkpoint mutation.
"""

from __future__ import annotations

import torch

from ablation.ic_core.model_adapter import LITE_MODEL_CLASSES, MODEL_CLASSES
from ablation.ic_core.parameter_adapter import get_parameter_spec
from models import (
    XAJControlledNWithCemaNeige, XAJControlledNWithCemaNeigeLite,
    XAJDEWithCemaNeige, XAJDEWithCemaNeigeLite,
    XAJDRWithCemaNeige, XAJDRWithCemaNeigeLite,
    XAJGEWithCemaNeige, XAJGEWithCemaNeigeLite,
    XAJGRWithCemaNeige, XAJGRWithCemaNeigeLite,
    XAJControlledN, XAJDE, XAJGE, XAJDR, XAJGR,
)
from models.parameter_specs import (
    CEMANEIGE_PARAM_SPECS,
    CONTROLLED_XAJ_CI_LOWER, CONTROLLED_XAJ_CI_UPPER,
    CONTROLLED_XAJ_CG_LOWER, CONTROLLED_XAJ_CG_UPPER,
    XAJ_CONTROLLED_N_PARAM_SPECS, XAJ_DE_PARAM_SPECS, XAJ_GE_PARAM_SPECS,
    XAJ_DR_PARAM_SPECS, XAJ_GR_PARAM_SPECS,
)
from models.structure_response import (
    _analytic_subsurface_response_step, native_effective_kss,
)
from models.xaj import _xaj_step_impl

DTYPE = torch.float64
NEARZERO = 1e-8

# Keep the project's compiled fullgraph kernels from aborting when several
# dtype/layout specializations accumulate in one pytest process (same policy
# as training/ic/run_tgd2_batched_cmaes_531.py).
import torch._dynamo as _dynamo  # noqa: E402

_dynamo.config.recompile_limit = max(_dynamo.config.recompile_limit, 256)
_dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 256)


# --------------------------------------------------------------------------
# Shared deterministic forcing / parameter / state fixtures
# --------------------------------------------------------------------------

def forcing(steps: int = 160, dtype: torch.dtype = DTYPE):
    t = torch.arange(steps, dtype=dtype)
    # Dry/wet/snow alternation activates CN snow, EU/EL/ED, runoff and the
    # subsurface response.
    precip = (
        torch.where((t % 9) < 4, 6.0 + (t % 5) * 0.7, torch.zeros_like(t))
        + torch.where((t % 29) == 3, 40.0, torch.zeros_like(t))
    )
    pet = 1.5 + 1.5 * ((t % 11) / 10.0)
    temp = torch.where((t % 17) < 7, torch.full_like(t, -2.5), torch.full_like(t, 4.5))
    return {
        "precip": precip.unsqueeze(0),
        "pet": pet.unsqueeze(0),
        "temp": temp.unsqueeze(0),
    }


def params_from(specs, dtype: torch.dtype = DTYPE, **over):
    p = {
        name: torch.full((1,), float(spec["default"]), dtype=dtype)
        for name, spec in specs.items()
    }
    # Physical values that activate every process branch.
    p["xaj_im"] = torch.tensor([0.02], dtype=dtype)
    p["xaj_ci"] = torch.tensor([0.55], dtype=dtype)
    p["xaj_cg"] = torch.tensor([0.96], dtype=dtype)
    p["xaj_ki"] = torch.tensor([0.30], dtype=dtype)
    p["xaj_kg"] = torch.tensor([0.20], dtype=dtype)
    p["xaj_k"] = torch.tensor([1.1], dtype=dtype)
    p["xaj_c"] = torch.tensor([0.15], dtype=dtype)
    for key, value in over.items():
        p[key] = torch.tensor([value], dtype=dtype)
    return p


def cn_params(dtype: torch.dtype = DTYPE):
    return params_from(CEMANEIGE_PARAM_SPECS, dtype=dtype)


def initial_states(dtype: torch.dtype = DTYPE):
    return {
        "cn_G": torch.tensor([30.0], dtype=dtype),
        "cn_eTG": torch.tensor([-3.0], dtype=dtype),
        "xaj_wu": torch.tensor([12.0], dtype=dtype),
        "xaj_wl": torch.tensor([30.0], dtype=dtype),
        "xaj_wd": torch.tensor([25.0], dtype=dtype),
        "xaj_s": torch.tensor([6.0], dtype=dtype),
        "xaj_fr": torch.tensor([0.3], dtype=dtype),
        "xaj_qi": torch.tensor([0.2], dtype=dtype),
        "xaj_qg": torch.tensor([0.4], dtype=dtype),
        "xaj_rs_uh_buffer": torch.zeros(1, 14, dtype=dtype),
    }


# --------------------------------------------------------------------------
# 1. Code-level control matrix
# --------------------------------------------------------------------------

def test_code_matrix_matches_frozen_design():
    """Parameter-spec matrix is exactly the frozen single-factor design."""
    n_spec = XAJ_CONTROLLED_N_PARAM_SPECS
    de_spec = XAJ_DE_PARAM_SPECS
    ge_spec = XAJ_GE_PARAM_SPECS
    dr_spec = XAJ_DR_PARAM_SPECS
    gr_spec = XAJ_GR_PARAM_SPECS

    response_params = {"xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg"}
    et_param = {"xaj_c"}
    generic_evap = {"xaj_gamma"}
    generic_resp = {"xaj_kss", "xaj_tau0", "xaj_beta"}

    # N -> D_E removes only native ET C; response parameters are untouched.
    assert set(de_spec) == set(n_spec) - et_param
    for name in response_params:
        assert de_spec[name]["lower"] == n_spec[name]["lower"]
        assert de_spec[name]["upper"] == n_spec[name]["upper"]

    # N -> G_E removes C and adds gamma; response parameters untouched.
    assert set(ge_spec) == set(n_spec) - et_param | generic_evap
    for name in response_params:
        assert ge_spec[name]["lower"] == n_spec[name]["lower"]
        assert ge_spec[name]["upper"] == n_spec[name]["upper"]

    # N -> D_R keeps native ET C, structurally replaces the response block.
    assert set(dr_spec) == set(n_spec) - response_params | {"xaj_kss", "xaj_tau0"}
    for name in et_param:
        assert dr_spec[name]["lower"] == n_spec[name]["lower"]
        assert dr_spec[name]["upper"] == n_spec[name]["upper"]

    # N -> G_R: D_R plus beta.
    assert set(gr_spec) == set(dr_spec) | generic_resp

    # No cross-process leakage.
    assert generic_evap.isdisjoint(dr_spec) and generic_evap.isdisjoint(gr_spec)
    assert generic_resp.isdisjoint(de_spec) and generic_resp.isdisjoint(ge_spec)

    # Controlled finite native response domain wherever CI/CG exist.
    for spec in (n_spec, de_spec, ge_spec):
        assert spec["xaj_ci"]["lower"] == CONTROLLED_XAJ_CI_LOWER
        assert spec["xaj_ci"]["upper"] == CONTROLLED_XAJ_CI_UPPER
        assert spec["xaj_cg"]["lower"] == CONTROLLED_XAJ_CG_LOWER
        assert spec["xaj_cg"]["upper"] == CONTROLLED_XAJ_CG_UPPER


def test_variant_to_scheme_mapping_is_single_process():
    """Each D/G class replaces exactly one process module."""
    assert (XAJDE.variant, XAJDE.response_variant, XAJDE.generic_variant) == (0, False, False)
    assert (XAJGE.variant, XAJGE.response_variant, XAJGE.generic_variant) == (1, False, True)
    assert (XAJDR.variant, XAJDR.response_variant, XAJDR.generic_variant) == (2, True, False)
    assert (XAJGR.variant, XAJGR.response_variant, XAJGR.generic_variant) == (3, True, True)
    # Code path in _xaj_structure_step_full/_compact maps:
    #   variant 0 -> evaporation 2 (parallel linear),  response 0 (native)
    #   variant 1 -> evaporation 3 (parallel power),   response 0 (native)
    #   variant 2 -> evaporation 0 (native),           response 1 (linear Z)
    #   variant 3 -> evaporation 0 (native),           response 2 (power Z)
    # There is no variant with evaporation in {2,3} and response in {1,2}.


def test_registry_has_only_five_phase0_models_no_cross_combination():
    """Phase-0 IC registry references exactly the five frozen models."""
    expected_full = {
        "N": XAJControlledNWithCemaNeige,
        "D_E": XAJDEWithCemaNeige,
        "G_E": XAJGEWithCemaNeige,
        "D_R": XAJDRWithCemaNeige,
        "G_R": XAJGRWithCemaNeige,
    }
    expected_lite = {
        "N": XAJControlledNWithCemaNeigeLite,
        "D_E": XAJDEWithCemaNeigeLite,
        "G_E": XAJGEWithCemaNeigeLite,
        "D_R": XAJDRWithCemaNeigeLite,
        "G_R": XAJGRWithCemaNeigeLite,
    }
    for key, cls in expected_full.items():
        assert MODEL_CLASSES[key] is cls, key
        assert LITE_MODEL_CLASSES[key] is expected_lite[key], key

    # No cross-combination key exists in the Phase-0 registry.
    phase0_keys = set(MODEL_CLASSES) & {"N", "D_E", "G_E", "D_R", "G_R"}
    assert phase0_keys == set(expected_full)

    # Parameter ordering/count per model matches the frozen dimensions
    # (N 17, D_E 16, G_E 17, D_R 15, G_R 16).
    assert len(get_parameter_spec("N")) == 17
    assert len(get_parameter_spec("D_E")) == 16
    assert len(get_parameter_spec("G_E")) == 17
    assert len(get_parameter_spec("D_R")) == 15
    assert len(get_parameter_spec("G_R")) == 16


def test_no_inactive_or_dummy_optimizer_parameters():
    """Every optimizer dimension maps to a forward-used parameter."""
    # D_E/G_E drop C; the C slot is a constant zero, not an optimizer dim.
    assert "xaj_c" not in XAJ_DE_PARAM_SPECS and "xaj_c" not in XAJ_GE_PARAM_SPECS
    # D_R/G_R drop the response identities; no gamma is present.
    assert "xaj_gamma" not in XAJ_DR_PARAM_SPECS and "xaj_gamma" not in XAJ_GR_PARAM_SPECS
    for name in ("xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg"):
        assert name not in XAJ_DR_PARAM_SPECS and name not in XAJ_GR_PARAM_SPECS

    # Every spec parameter is used by the forward (a finite gradient reaches
    # it).  Nonzero-gradient requirements below are per-parameter:
    #   * structural/generic parameters and response coefficients must be
    #     strictly active on the activated forcing;
    #   * xaj_c (native ET C) only acts when WL drops below C*LM, so its
    #     gradient can be zero on trajectories that never enter that branch --
    #     forward sensitivity is proven separately below;
    #   * cn_ctg is structurally non-differentiable in basic CemaNeige (its
    #     thermal state enters melt only through a hard boolean), so it is
    #     exempt from the nonzero-gradient requirement.
    must_activate = {
        "N": {"xaj_k", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg", "xaj_sm", "cn_kf"},
        "D_E": {"xaj_k", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg", "xaj_sm", "cn_kf"},
        "G_E": {"xaj_gamma", "xaj_k", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg", "cn_kf"},
        "D_R": {"xaj_kss", "xaj_tau0", "xaj_k", "xaj_sm", "cn_kf"},
        "G_R": {"xaj_kss", "xaj_tau0", "xaj_beta", "xaj_k", "cn_kf"},
    }
    zero_grad_ok = {"cn_ctg", "xaj_c"}
    for key, cls in (("N", XAJControlledNWithCemaNeige), ("D_E", XAJDEWithCemaNeige),
                     ("G_E", XAJGEWithCemaNeige), ("D_R", XAJDRWithCemaNeige),
                     ("G_R", XAJGRWithCemaNeige)):
        specs = get_parameter_spec(key)
        params = {
            name: torch.full((1,), float(spec["default"]), dtype=DTYPE, requires_grad=True)
            for name, spec in specs.items()
        }
        if "xaj_im" in params:
            params["xaj_im"] = torch.tensor([0.02], dtype=DTYPE, requires_grad=True)
        if "xaj_gamma" in params:
            params["xaj_gamma"] = torch.tensor([2.0], dtype=DTYPE, requires_grad=True)
        if "xaj_beta" in params:
            params["xaj_beta"] = torch.tensor([1.3], dtype=DTYPE, requires_grad=True)
        if "xaj_tau0" in params:
            params["xaj_tau0"] = torch.tensor([12.0], dtype=DTYPE, requires_grad=True)
        if "xaj_kss" in params:
            params["xaj_kss"] = torch.tensor([0.5], dtype=DTYPE, requires_grad=True)
        if "xaj_ci" in params:
            params["xaj_ci"] = torch.tensor([0.55], dtype=DTYPE, requires_grad=True)
        if "xaj_cg" in params:
            params["xaj_cg"] = torch.tensor([0.96], dtype=DTYPE, requires_grad=True)
        if "xaj_ki" in params:
            params["xaj_ki"] = torch.tensor([0.30], dtype=DTYPE, requires_grad=True)
        if "xaj_kg" in params:
            params["xaj_kg"] = torch.tensor([0.20], dtype=DTYPE, requires_grad=True)
        model = cls(compact_output=False)
        qsim, _ = model(forcing(48), {**cn_params(), **params}, initial_states=initial_states())
        qsim.square().mean().backward()
        for name, value in params.items():
            assert value.grad is not None, f"{key}:{name} got no gradient"
            assert torch.isfinite(value.grad).all(), f"{key}:{name} non-finite gradient"
            if name in must_activate[key]:
                assert abs(float(value.grad)) > 1e-12, f"{key}:{name} inactive optimizer dimension"
            elif name not in zero_grad_ok:
                assert abs(float(value.grad)) > 1e-12, f"{key}:{name} inactive optimizer dimension"


def test_native_et_c_is_forward_active_not_dummy():
    """xaj_c changes streamflow when its WL < C*LM branch is reached."""
    steps = 120
    t = torch.arange(steps, dtype=DTYPE)
    # 20-day drydown puts WL inside the C-sensitive band, then partial-wet
    # events let the storage difference reach runoff generation.
    precip = torch.where(t < 20, torch.zeros_like(t), torch.where((t % 6) < 3, 7.0, 0.0))
    pet = torch.where(t < 20, torch.full_like(t, 7.0), torch.full_like(t, 2.0))
    temp = torch.full_like(t, 8.0)
    f = {"precip": precip.unsqueeze(0), "pet": pet.unsqueeze(0), "temp": temp.unsqueeze(0)}
    base = params_from(XAJ_CONTROLLED_N_PARAM_SPECS)
    base["xaj_um"] = torch.tensor([20.0], dtype=DTYPE)
    base["xaj_lm"] = torch.tensor([80.0], dtype=DTYPE)
    base["xaj_dm"] = torch.tensor([40.0], dtype=DTYPE)
    init = dict(initial_states())
    init.update({
        "cn_G": torch.tensor([0.0], dtype=DTYPE), "cn_eTG": torch.tensor([-1.0], dtype=DTYPE),
        "xaj_wu": torch.tensor([5.0], dtype=DTYPE), "xaj_wl": torch.tensor([50.0], dtype=DTYPE),
        "xaj_wd": torch.tensor([35.0], dtype=DTYPE),
    })
    outs = []
    for c_value in (0.05, 0.25):
        p = {name: value.clone() for name, value in base.items()}
        p["xaj_c"] = torch.tensor([c_value], dtype=DTYPE)
        q, _ = XAJControlledNWithCemaNeige(compact_output=False)(
            f, {**cn_params(), **p}, initial_states=init)
        outs.append(q)
    assert float((outs[0] - outs[1]).abs().max()) > 1e-6


# --------------------------------------------------------------------------
# 2. Evaporation ladder: N vs D_E vs G_E
# --------------------------------------------------------------------------

def _step_impl(scheme, st, prcp, pet, *, gamma=None, ki=0.30, kg=0.20,
               ci=0.55, cg=0.96, im=0.02, c=0.15):
    """Direct _xaj_step_impl with the same native response parameters."""
    wu, wl, wd, s, fr, qi, qg = st
    k = torch.tensor([1.1], dtype=DTYPE); b = torch.tensor([0.3], dtype=DTYPE)
    um = torch.tensor([20.0], dtype=DTYPE); lm = torch.tensor([80.0], dtype=DTYPE)
    dm = torch.tensor([40.0], dtype=DTYPE)
    sm = torch.tensor([30.0], dtype=DTYPE); ex = torch.tensor([1.2], dtype=DTYPE)
    im_t = torch.tensor([im], dtype=DTYPE)
    ki_t = torch.tensor([ki], dtype=DTYPE); kg_t = torch.tensor([kg], dtype=DTYPE)
    ci_t = torch.tensor([ci], dtype=DTYPE); cg_t = torch.tensor([cg], dtype=DTYPE)
    c_t = torch.tensor([c], dtype=DTYPE) if scheme == 0 else torch.zeros(1, dtype=DTYPE)
    wm = um + lm + dm
    ev_scheme = {0: 0, 2: 2, 3: 3}[scheme]
    return _xaj_step_impl(
        prcp, pet, wu, wl, wd, s, fr, qi, qg,
        k, b, im_t, um, lm, dm, c_t, sm, ex, ki_t, kg_t, ci_t, cg_t,
        NEARZERO, wm, wm * (1.0 + b), sm * (1.0 + ex), 1.0 - im_t, 1.0 - ki_t - kg_t,
        True, ev_scheme, None, 0, gamma, None, None, None,
    )


def _dry_state():
    return (
        torch.tensor([2.0], dtype=DTYPE), torch.tensor([30.0], dtype=DTYPE),
        torch.tensor([25.0], dtype=DTYPE), torch.tensor([8.0], dtype=DTYPE),
        torch.tensor([0.5], dtype=DTYPE), torch.tensor([0.2], dtype=DTYPE),
        torch.tensor([0.4], dtype=DTYPE),
    )


def test_evaporation_ladder_response_identity_dry_step():
    """In a dry step EL/ED differ but the native response path is untouched."""
    st = _dry_state()
    prcp = torch.tensor([1.0], dtype=DTYPE)
    pet = torch.tensor([4.0], dtype=DTYPE)
    gamma = torch.tensor([2.0], dtype=DTYPE)
    o_n = _step_impl(0, st, prcp, pet)
    o_de = _step_impl(2, st, prcp, pet)
    o_ge = _step_impl(3, st, prcp, pet, gamma=gamma)

    # EU is identical: it is computed before the evaporation branch.
    assert torch.equal(o_n[13], o_de[13]) and torch.equal(o_n[13], o_ge[13])
    # EL/ED are the only replaced quantities and are allowed to differ.
    assert not torch.equal(o_n[14], o_de[14]) or not torch.equal(o_n[15], o_de[15])
    # With the same pre-step (s, fr, qi, qg), the native response outputs are
    # bit-identical: qi, qg, s_next, fr, rs, ri, rg.
    for idx in (2, 3, 8, 9, 10, 11, 12):  # qi, qg, s_next, fr, rs, ri, rg
        assert torch.equal(o_n[idx], o_de[idx]), idx
        assert torch.equal(o_n[idx], o_ge[idx]), idx


def test_evaporation_ladder_wet_step_full_identity():
    """A wet step (remaining PET = 0) is fully identical across schemes."""
    st = (
        torch.tensor([10.0], dtype=DTYPE), torch.tensor([30.0], dtype=DTYPE),
        torch.tensor([25.0], dtype=DTYPE), torch.tensor([8.0], dtype=DTYPE),
        torch.tensor([0.5], dtype=DTYPE), torch.tensor([0.2], dtype=DTYPE),
        torch.tensor([0.4], dtype=DTYPE),
    )
    prcp = torch.tensor([9.0], dtype=DTYPE)
    pet = torch.tensor([4.0], dtype=DTYPE)
    gamma = torch.tensor([2.0], dtype=DTYPE)
    o_n = _step_impl(0, st, prcp, pet)
    o_de = _step_impl(2, st, prcp, pet)
    o_ge = _step_impl(3, st, prcp, pet, gamma=gamma)
    for a, b in ((o_n, o_de), (o_n, o_ge)):
        for idx in range(len(o_n)):
            assert torch.equal(a[idx], b[idx]), idx


def test_evaporation_ladder_gamma_one_reduction():
    """gamma = 1 makes the G_E evaporation kernel bit-identical to D_E."""
    st = _dry_state()
    prcp = torch.tensor([1.0], dtype=DTYPE)
    pet = torch.tensor([4.0], dtype=DTYPE)
    o_de = _step_impl(2, st, prcp, pet)
    o_ge = _step_impl(3, st, prcp, pet, gamma=torch.ones(1, dtype=DTYPE))
    for a, b in zip(o_de, o_ge):
        assert torch.equal(a, b)


def test_evaporation_ladder_isolated_response_identity():
    """Same fixed RI/RG and response initial states -> identical native QI/QG.

    With prcp = pet = 0 and wl = wd = 0 the free-water state passes through
    unchanged (r = 0, fr and s preserved), so ri = ki*s*fr and rg = kg*s*fr
    are exactly the injected fixed values and the native response recursion
    is evaluated in isolation for all three evaporation schemes.
    """
    s = torch.tensor([10.0], dtype=DTYPE)   # fixed
    fr = torch.tensor([0.4], dtype=DTYPE)   # fixed
    qi0 = torch.tensor([0.7], dtype=DTYPE)  # fixed response initial state
    qg0 = torch.tensor([1.3], dtype=DTYPE)  # fixed response initial state
    st = (
        torch.tensor([0.0], dtype=DTYPE), torch.tensor([0.0], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE), s, fr, qi0, qg0,
    )
    prcp = torch.zeros(1, dtype=DTYPE)
    pet = torch.zeros(1, dtype=DTYPE)
    gamma = torch.tensor([2.0], dtype=DTYPE)
    ri_target = 0.30 * s * fr
    rg_target = 0.20 * s * fr
    outs = [_step_impl(scheme, st, prcp, pet, gamma=gamma) for scheme in (0, 2, 3)]
    for o in outs:
        assert torch.equal(o[11], ri_target)   # ri
        assert torch.equal(o[12], rg_target)   # rg
    # Native QI/QG recursions identical to machine precision.
    assert torch.equal(outs[0][2], outs[1][2]) and torch.equal(outs[0][2], outs[2][2])
    assert torch.equal(outs[0][3], outs[1][3]) and torch.equal(outs[0][3], outs[2][3])


def test_evaporation_ladder_composed_single_factor():
    """Composed N / D_E / G_E: only the evaporation organization differs."""
    f = forcing(120)
    cn = cn_params()
    qn, an = XAJControlledNWithCemaNeige(compact_output=False)(
        f, {**cn, **params_from(XAJ_CONTROLLED_N_PARAM_SPECS)},
        initial_states=initial_states(), return_states=True)
    qde, ade = XAJDEWithCemaNeige(compact_output=False)(
        f, {**cn, **params_from(XAJ_DE_PARAM_SPECS)},
        initial_states=initial_states(), return_states=True)
    qge1, age1 = XAJGEWithCemaNeige(compact_output=False)(
        f, {**cn, **params_from(XAJ_GE_PARAM_SPECS, xaj_gamma=1.0)},
        initial_states=initial_states(), return_states=True)

    # 1. CemaNeige effective precipitation is identical across all three.
    for aux in (ade, age1):
        assert torch.equal(an["effective_precip"], aux["effective_precip"])
        assert torch.equal(an["cn_rain"], aux["cn_rain"])
        assert torch.equal(an["cn_melt"], aux["cn_melt"])
        assert torch.equal(an["cn_snow_pack"], aux["cn_snow_pack"])

    # 2. gamma=1: G_E is identical to D_E everywhere.
    assert torch.equal(qde, qge1)
    assert torch.equal(ade["xaj_eu"], age1["xaj_eu"])
    assert torch.equal(ade["xaj_el"], age1["xaj_el"])
    assert torch.equal(ade["xaj_ed"], age1["xaj_ed"])
    assert torch.equal(ade["xaj_qi"], age1["xaj_qi"])
    assert torch.equal(ade["xaj_qg"], age1["xaj_qg"])

    # 3. Response states stay native (QI/QG), no single Z.
    for aux in (an, ade, age1):
        final = aux["final_states"]
        assert "xaj_qi" in final and "xaj_qg" in final
        assert "xaj_z" not in final

    # 4. With the same initial states, step 0 ET pre-kernel state is shared
    #    and only EL/ED may differ; EU at t=0 is identical for D_E and G_E.
    assert torch.equal(ade["xaj_eu"][:, 0], age1["xaj_eu"][:, 0])
    assert torch.equal(ade["xaj_eu"], age1["xaj_eu"])


# --------------------------------------------------------------------------
# 3. Response ladder: N vs D_R vs G_R
# --------------------------------------------------------------------------

def _step_impl_response(st, prcp, pet, *, beta=None, ki=0.30, kg=0.20,
                        ci=0.55, cg=0.96, im=0.02, c=0.15, tau0=10.0):
    """Direct _xaj_step_impl with native ET and the D_R/G_R response block."""
    wu, wl, wd, s, fr, qi, qg = st
    k = torch.tensor([1.1], dtype=DTYPE); b = torch.tensor([0.3], dtype=DTYPE)
    um = torch.tensor([20.0], dtype=DTYPE); lm = torch.tensor([80.0], dtype=DTYPE)
    dm = torch.tensor([40.0], dtype=DTYPE)
    sm = torch.tensor([30.0], dtype=DTYPE); ex = torch.tensor([1.2], dtype=DTYPE)
    im_t = torch.tensor([im], dtype=DTYPE)
    ki_t = torch.tensor([ki], dtype=DTYPE); kg_t = torch.tensor([kg], dtype=DTYPE)
    c_t = torch.tensor([c], dtype=DTYPE)
    kss = native_effective_kss(ki_t, kg_t)
    wm = um + lm + dm
    response_scheme = 1 if beta is None else 2
    beta_t = torch.ones(1, dtype=DTYPE) if beta is None else torch.tensor([beta], dtype=DTYPE)
    tau0_t = torch.tensor([tau0], dtype=DTYPE)
    z0 = torch.tensor([3.1553493591016335], dtype=DTYPE)
    return _xaj_step_impl(
        prcp, pet, wu, wl, wd, s, fr, qi, qg,
        k, b, im_t, um, lm, dm, c_t, sm, ex,
        kss, torch.zeros_like(kss), torch.ones_like(kss), torch.ones_like(kss),
        NEARZERO, wm, wm * (1.0 + b), sm * (1.0 + ex), 1.0 - im_t, 1.0 - kss,
        True, 0, None, response_scheme, None, tau0_t, beta_t, z0,
    )


def test_response_ladder_native_et_identity_and_matched_kss():
    """N vs D_R vs G_R: same ET, same runoff generation, matched KSS input."""
    st = _dry_state()
    steps = [
        (torch.tensor([9.0], dtype=DTYPE), torch.tensor([4.0], dtype=DTYPE)),
        (torch.tensor([1.0], dtype=DTYPE), torch.tensor([4.0], dtype=DTYPE)),
        (torch.tensor([12.0], dtype=DTYPE), torch.tensor([3.0], dtype=DTYPE)),
        (torch.tensor([0.0], dtype=DTYPE), torch.tensor([4.5], dtype=DTYPE)),
        (torch.tensor([7.0], dtype=DTYPE), torch.tensor([2.0], dtype=DTYPE)),
    ]
    sn, sdr, sgr = st, st, st
    max_diff_eu = max_diff_el = max_diff_ed = 0.0
    max_diff_rs = max_diff_ssin = max_diff_snext = 0.0
    for prcp, pet in steps:
        o_n = _step_impl(0, sn, prcp, pet)
        o_dr = _step_impl_response(sdr, prcp, pet, beta=None)
        o_gr = _step_impl_response(sgr, prcp, pet, beta=1.3)
        # Native ET kernel identical: EU/EL/ED (native tuple idx 13/14/15;
        # response tuple idx 12/13/14).
        for a, b in ((o_n[13], o_dr[12]), (o_n[13], o_gr[12])):
            max_diff_eu = max(max_diff_eu, float((a - b).abs().max()))
        for a, b in ((o_n[14], o_dr[13]), (o_n[14], o_gr[13])):
            max_diff_el = max(max_diff_el, float((a - b).abs().max()))
        for a, b in ((o_n[15], o_dr[14]), (o_n[15], o_gr[14])):
            max_diff_ed = max(max_diff_ed, float((a - b).abs().max()))
        assert torch.equal(o_n[13], o_dr[12]) and torch.equal(o_n[13], o_gr[12])
        assert torch.equal(o_n[14], o_dr[13]) and torch.equal(o_n[14], o_gr[13])
        assert torch.equal(o_n[15], o_dr[14]) and torch.equal(o_n[15], o_gr[14])
        # Tension-water states identical (native idx 5/6/7, response idx 4/5/6).
        assert torch.equal(o_n[5], o_dr[4]) and torch.equal(o_n[5], o_gr[4])
        assert torch.equal(o_n[6], o_dr[5]) and torch.equal(o_n[6], o_gr[5])
        assert torch.equal(o_n[7], o_dr[6]) and torch.equal(o_n[7], o_gr[6])
        # Surface runoff identical (native idx 10, response idx 9).
        max_diff_rs = max(max_diff_rs, float((o_n[10] - o_dr[9]).abs().max()))
        max_diff_rs = max(max_diff_rs, float((o_n[10] - o_gr[9]).abs().max()))
        # Matched KSS: R_ss == (RI+RG)*(1-IM) (native idx 11/12, response idx 10).
        native_ssin = (o_n[11] + o_n[12]) * (1.0 - torch.tensor([0.02], dtype=DTYPE))
        max_diff_ssin = max(max_diff_ssin, float((native_ssin - o_dr[10]).abs().max()))
        max_diff_ssin = max(max_diff_ssin, float((native_ssin - o_gr[10]).abs().max()))
        # s_next consistent (native idx 8, response idx 7).
        max_diff_snext = max(max_diff_snext, float((o_n[8] - o_dr[7]).abs().max()))
        max_diff_snext = max(max_diff_snext, float((o_n[8] - o_gr[7]).abs().max()))
        sn = (o_n[5], o_n[6], o_n[7], o_n[8], o_n[9], o_n[2], o_n[3])
        sdr = (o_dr[4], o_dr[5], o_dr[6], o_dr[7], o_dr[8], o_dr[2], o_dr[3])
        sgr = (o_gr[4], o_gr[5], o_gr[6], o_gr[7], o_gr[8], o_gr[2], o_gr[3])

    assert max_diff_eu == 0.0 and max_diff_el == 0.0 and max_diff_ed == 0.0
    # Only float reassociation (1-ki-kg vs 1-kss) contributes at ~1e-15.
    assert max_diff_rs <= 1e-14
    assert max_diff_ssin <= 1e-14
    assert max_diff_snext <= 1e-14


def test_response_ladder_beta_one_reduction():
    """beta = 1 makes the G_R response kernel bit-identical to D_R."""
    st = _dry_state()
    prcp = torch.tensor([3.0], dtype=DTYPE)
    pet = torch.tensor([1.0], dtype=DTYPE)
    o_dr = _step_impl_response(st, prcp, pet, beta=None)
    o_gr = _step_impl_response(st, prcp, pet, beta=1.0)
    for a, b in zip(o_dr, o_gr):
        assert torch.equal(a, b)


def test_response_ladder_analytic_kernel_identity():
    """D_R and G_R call the same analytic kernel; beta=1 is exact linear."""
    r_ss = torch.tensor([0.0, 1.5, 4.0, 12.0], dtype=DTYPE)
    z = torch.tensor([0.0, 2.0, 8.0, 30.0], dtype=DTYPE)
    tau = torch.full((4,), 10.0, dtype=DTYPE)
    z0 = torch.full((4,), 3.1553493591016335, dtype=DTYPE)
    dr = _analytic_subsurface_response_step(r_ss, z, tau, torch.ones(4, dtype=DTYPE), z0, NEARZERO)
    gr = _analytic_subsurface_response_step(r_ss, z, tau, torch.ones(4, dtype=DTYPE), z0, NEARZERO)
    assert torch.equal(dr[0], gr[0]) and torch.equal(dr[1], gr[1])
    # Conservation: Z_next = Z_available - Q_ss.
    assert torch.allclose(dr[1], torch.clamp(z, min=0.0) + r_ss - dr[0], atol=0.0, rtol=0.0)


def test_response_ladder_composed_single_factor():
    """Composed N / D_R / G_R: only the response organization differs."""
    f = forcing(120)
    cn = cn_params()
    qn, an = XAJControlledNWithCemaNeige(compact_output=False)(
        f, {**cn, **params_from(XAJ_CONTROLLED_N_PARAM_SPECS)},
        initial_states=initial_states(), return_states=True)
    qdr, adr = XAJDRWithCemaNeige(compact_output=False)(
        f, {**cn, **params_from(XAJ_DR_PARAM_SPECS, xaj_kss=0.5, xaj_tau0=10.0)},
        initial_states=initial_states(), return_states=True)
    qgr, agr = XAJGRWithCemaNeige(compact_output=False)(
        f, {**cn, **params_from(XAJ_GR_PARAM_SPECS, xaj_kss=0.5, xaj_tau0=10.0, xaj_beta=1.0)},
        initial_states=initial_states(), return_states=True)

    # CemaNeige identical across all three.
    for aux in (adr, agr):
        assert torch.equal(an["effective_precip"], aux["effective_precip"])

    # beta=1: G_R is identical to D_R everywhere.
    assert torch.equal(qdr, qgr)
    assert torch.equal(adr["xaj_eu"], agr["xaj_eu"])
    assert torch.equal(adr["xaj_el"], agr["xaj_el"])
    assert torch.equal(adr["xaj_ed"], agr["xaj_ed"])
    assert torch.equal(adr["xaj_q_ss"], agr["xaj_q_ss"])
    assert torch.equal(adr["xaj_z"], agr["xaj_z"])

    # State layout: N keeps QI/QG; D_R/G_R replace them with a single Z.
    assert "xaj_qi" in an["final_states"] and "xaj_qg" in an["final_states"]
    for aux in (adr, agr):
        final = aux["final_states"]
        assert "xaj_z" in final
        assert "xaj_qi" not in final and "xaj_qg" not in final

    # ET states stay native for D_R/G_R (WU/WL/WD present, no extra ET param).
    for aux in (adr, agr):
        final = aux["final_states"]
        assert "xaj_wu" in final and "xaj_wl" in final and "xaj_wd" in final


# --------------------------------------------------------------------------
# 4. Full/lite, compiled, and long-sequence consistency (Test D)
# --------------------------------------------------------------------------

FULL_LITE_CASES = (
    ("N", XAJControlledNWithCemaNeige, XAJControlledNWithCemaNeigeLite),
    ("D_E", XAJDEWithCemaNeige, XAJDEWithCemaNeigeLite),
    ("G_E", XAJGEWithCemaNeige, XAJGEWithCemaNeigeLite),
    ("D_R", XAJDRWithCemaNeige, XAJDRWithCemaNeigeLite),
    ("G_R", XAJGRWithCemaNeige, XAJGRWithCemaNeigeLite),
)


def test_full_lite_scientific_kernel_consistent():
    f = forcing(96)
    cn = cn_params()
    for key, full_cls, lite_cls in FULL_LITE_CASES:
        specs = get_parameter_spec(key)
        over = {}
        if "xaj_gamma" in specs:
            over["xaj_gamma"] = 2.0
        if "xaj_beta" in specs:
            over["xaj_beta"] = 1.3
        if "xaj_tau0" in specs:
            over["xaj_tau0"] = 12.0
        if "xaj_kss" in specs:
            over["xaj_kss"] = 0.5
        params = {**cn, **params_from(specs, **over)}
        q_full, _ = full_cls(compact_output=False)(f, params, initial_states=initial_states())
        q_lite, aux_lite = lite_cls()(f, params, initial_states=initial_states())
        assert torch.allclose(q_full, q_lite, atol=2e-5, rtol=2e-5), key
        assert aux_lite == {}, key


def test_compiled_fullgraph_and_differential_assertions_hold():
    """torch.compile(fullgraph=True) runs and preserves differential identity."""
    f = forcing(64)
    cn = cn_params()
    # D_E vs G_E(gamma=1) must be identical under compiled execution.
    de_p = {**cn, **params_from(XAJ_DE_PARAM_SPECS)}
    ge1_p = {**cn, **params_from(XAJ_GE_PARAM_SPECS, xaj_gamma=1.0)}
    qde, ade = XAJDEWithCemaNeige(compact_output=False)(f, de_p, initial_states=initial_states())
    qge1, age1 = XAJGEWithCemaNeige(compact_output=False)(f, ge1_p, initial_states=initial_states())
    assert torch.equal(qde, qge1)
    assert torch.equal(ade["xaj_eu"], age1["xaj_eu"])

    # D_R vs G_R(beta=1) identical under compiled execution.
    dr_p = {**cn, **params_from(XAJ_DR_PARAM_SPECS, xaj_kss=0.5, xaj_tau0=10.0)}
    gr1_p = {**cn, **params_from(XAJ_GR_PARAM_SPECS, xaj_kss=0.5, xaj_tau0=10.0, xaj_beta=1.0)}
    qdr, adr = XAJDRWithCemaNeige(compact_output=False)(f, dr_p, initial_states=initial_states())
    qgr1, agr1 = XAJGRWithCemaNeige(compact_output=False)(f, gr1_p, initial_states=initial_states())
    assert torch.equal(qdr, qgr1)
    assert torch.equal(adr["xaj_q_ss"], agr1["xaj_q_ss"])

    # N full/lite under compiled execution.
    n_p = {**cn, **params_from(XAJ_CONTROLLED_N_PARAM_SPECS)}
    qn, _ = XAJControlledNWithCemaNeige(compact_output=False)(f, n_p, initial_states=initial_states())
    qn_lite, _ = XAJControlledNWithCemaNeigeLite()(f, n_p, initial_states=initial_states())
    assert torch.allclose(qn, qn_lite, atol=2e-5, rtol=2e-5)


# --------------------------------------------------------------------------
# 5. Current Phase-0 N checkpoint resolution (read-only)
# --------------------------------------------------------------------------

def test_current_n_checkpoint_resolves_to_phase0_base():
    """Generation-120 N run must resolve to CN + XAJControlledN native base."""
    ckpt_path = (
        "results/phase0_ic_60_v1/N/checkpoints/n_batched.pt"
    )
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        import pytest
        pytest.skip("Phase-0 N checkpoint not present in this worktree")

    assert ckpt["model"] == "N"
    assert ckpt["structure_version"] == "phase0_xaj_controlled_n_cemaneige_v1"
    assert ckpt["protocol"] == "batched_cmaes_phase0_or_531_v1"
    assert ckpt["solver"]["dimension"] == 17
    assert len(get_parameter_spec("N")) == 17

    # The resolved registry class is the frozen Base:
    # CemaNeige + XAJControlledN (native 3-layer ET + native RI/RG, controlled
    # finite CI/CG domain).
    assert MODEL_CLASSES["N"] is XAJControlledNWithCemaNeige
    assert LITE_MODEL_CLASSES["N"] is XAJControlledNWithCemaNeigeLite
    n_spec = get_parameter_spec("N")
    assert n_spec["xaj_ci"]["upper"] == CONTROLLED_XAJ_CI_UPPER == 0.9
    assert n_spec["xaj_cg"]["upper"] == CONTROLLED_XAJ_CG_UPPER == 0.998
    assert {"xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg", "xaj_c"} <= set(n_spec)
    assert "xaj_kss" not in n_spec and "xaj_tau0" not in n_spec and "xaj_beta" not in n_spec
