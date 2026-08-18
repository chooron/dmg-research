"""Controlled CemaNeige-coupled XAJ structural variants.

The public ``XAJ_2S`` and ``XAJ_RWPE`` registrations intentionally include
CemaNeige.  They reuse the production snow step, XAJ capacity/runoff step and
Gamma-UH routing; only their stated target sub-process differs from XAJ+CN.

The standalone ``XAJDE/XAJGE/XAJDR/XAJGR`` classes below are controlled-XAJ
forward hosts for structural tests.  They are exported but intentionally not
added to training registries or experiment configurations in this phase.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import BaseHydrologicalModel
from .cemaneige import _cemaneige_step, _estimate_psol_annual, _init_basic_states
from .parameter_specs import (
    XAJ_2S_PARAM_SPECS,
    XAJ_CONTROLLED_N_PARAM_SPECS,
    XAJ_DE_PARAM_SPECS,
    XAJ_DR_PARAM_SPECS,
    XAJ_GE_PARAM_SPECS,
    XAJ_GR_PARAM_SPECS,
    XAJ_RWPE_PARAM_SPECS,
)
from .structure_response import DEFAULT_Z0, summarize_response_conditioning
from .utils import validate_forcings, validate_params
from .xaj import (
    XAJ,
    XAJ_UH_MAX_LEN,
    XAJLite,
    _prepare_xaj_parameters,
    _rootzone_moisture_stress_evaporation,
    _route_xaj_surface_runoff,
    _route_xaj_surface_runoff_hydrodl2,
    _xaj_rwpe_step,
    _xaj_rwpe_step_compact,
    _xaj_step_impl,
)


def _prepare_xaj_rwpe_parameters(
    params: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Reuse XAJ's coupled KI/KG boundary mapping with tau_e in C's slot."""
    # The shared extractor enforces the existing KI+KG<1 constraint.  It is
    # deliberately reused here so direct IC parameters and dPL parameters see
    # exactly the same coupled boundary behavior as standard XAJ.
    return _prepare_xaj_parameters({**params, "xaj_c": params["xaj_tau_e"]})


def _prepare_xaj_2s_parameters(
    params: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Prepare shared XAJ parameters plus the two-source KB/CB pair.

    KB and CB arrive already within their physical bounds from the project's
    parameter mapper.  In contrast to KI+KG, no forward-pass renormalisation
    is required because KB's specification is strictly below one.
    """
    return (
        params["xaj_k"],
        params["xaj_b"],
        params["xaj_im"],
        params["xaj_um"],
        params["xaj_lm"],
        params["xaj_dm"],
        params["xaj_c"],
        params["xaj_sm"],
        params["xaj_ex"],
        params["xaj_kb"],
        params["xaj_cb"],
        params["xaj_a"],
        params["xaj_theta"],
    )


def _xaj_2s_step(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    wu: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    s: torch.Tensor,
    fr: torch.Tensor,
    qb: torch.Tensor,
    k: torch.Tensor,
    b: torch.Tensor,
    im: torch.Tensor,
    um: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    c: torch.Tensor,
    sm: torch.Tensor,
    ex: torch.Tensor,
    kb: torch.Tensor,
    cb: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """XAJ step with RI/RG and QI/QG replaced by one merged slow branch."""
    zero = torch.zeros_like(qb)
    wm = um + lm + dm
    out = _xaj_step_impl(
        precip_t,
        pet_t,
        wu,
        wl,
        wd,
        s,
        fr,
        qb,
        zero,
        k,
        b,
        im,
        um,
        lm,
        dm,
        c,
        sm,
        ex,
        kb,
        zero,
        cb,
        zero,
        nearzero,
        wm,
        wm * (1.0 + b),
        sm * (1.0 + ex),
        1.0 - im,
        1.0 - kb,
        True,
    )
    (
        q_out,
        rs_adj,
        qb_next,
        _unused_qg,
        evap_total,
        wu,
        wl,
        wd,
        s_next,
        fr,
        rs,
        rb,
        _unused_rg,
        eu,
        el,
        ed,
    ) = out
    return (
        q_out,
        rs_adj,
        qb_next,
        evap_total,
        wu,
        wl,
        wd,
        s_next,
        fr,
        rs,
        rb,
        eu,
        el,
        ed,
    )


def _cemaneige_xaj_2s_fused_step(
    precip_t: torch.Tensor,
    temp_t: torch.Tensor,
    pet_t: torch.Tensor,
    cn_state: tuple[torch.Tensor, torch.Tensor],
    xaj_state: tuple[torch.Tensor, ...],
    cn_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    xaj_params: tuple[torch.Tensor, ...],
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    effective, G, eTG, sca, rain, melt = _cemaneige_step(
        precip_t,
        temp_t,
        cn_state[0],
        cn_state[1],
        cn_params[0],
        cn_params[1],
        cn_params[2],
        nearzero,
    )
    return (
        effective,
        G,
        eTG,
        sca,
        rain,
        melt,
        *_xaj_2s_step(effective, pet_t, *xaj_state, *xaj_params, nearzero),
    )


def _cemaneige_xaj_rwpe_fused_step(
    precip_t: torch.Tensor,
    temp_t: torch.Tensor,
    pet_t: torch.Tensor,
    cn_state: tuple[torch.Tensor, torch.Tensor],
    xaj_state: tuple[torch.Tensor, ...],
    cn_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    xaj_params: tuple[torch.Tensor, ...],
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    effective, G, eTG, sca, rain, melt = _cemaneige_step(
        precip_t,
        temp_t,
        cn_state[0],
        cn_state[1],
        cn_params[0],
        cn_params[1],
        cn_params[2],
        nearzero,
    )
    return (
        effective,
        G,
        eTG,
        sca,
        rain,
        melt,
        *_xaj_rwpe_step(effective, pet_t, *xaj_state, *xaj_params, nearzero),
    )


class _XAJCNVariant(BaseHydrologicalModel):
    """Small shared utility layer for the two default-CN public variants."""

    routing_method = "gamma"
    uh_max_len = XAJ_UH_MAX_LEN
    compact_output: bool

    def _split_initial(
        self,
        initial_states: Optional[dict[str, torch.Tensor]],
    ) -> tuple[Optional[dict[str, torch.Tensor]], Optional[dict[str, torch.Tensor]]]:
        if initial_states is None:
            return None, None
        return (
            {
                key[3:]: value
                for key, value in initial_states.items()
                if key.startswith("cn_")
            },
            {
                key[4:]: value
                for key, value in initial_states.items()
                if key.startswith("xaj_")
            },
        )

    def _cn_setup(
        self,
        precip: torch.Tensor,
        temp: torch.Tensor,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cn_initial, _ = self._split_initial(initial_states)
        G, eTG = _init_basic_states(batch, device, dtype, cn_initial)
        return (
            params["cn_ctg"],
            params["cn_kf"],
            0.9 * _estimate_psol_annual(precip, temp),
            G,
            eTG,
        )

    @staticmethod
    def _diagnostic_aux(
        *,
        qsim: torch.Tensor,
        effective: list[torch.Tensor],
        snow: list[torch.Tensor],
        thermal: list[torch.Tensor],
        sca: list[torch.Tensor],
        rain: list[torch.Tensor],
        melt: list[torch.Tensor],
        model_name: str,
        evaporation_scheme: str,
    ) -> dict[str, Any]:
        return {
            "q_out": qsim,
            "effective_precip": torch.stack(effective, dim=1),
            "snow_pack": torch.stack(snow, dim=1),
            "thermal_state": torch.stack(thermal, dim=1),
            "sca": torch.stack(sca, dim=1),
            "rain": torch.stack(rain, dim=1),
            "melt": torch.stack(melt, dim=1),
            "model_name": model_name,
            "evaporation_scheme": evaporation_scheme,
        }


class XAJ2SWithCemaNeige(_XAJCNVariant):
    """Two-source XAJ structural baseline, with CemaNeige enabled by default."""

    state_names = ["wu", "wl", "wd", "s", "fr", "qb", "rs_uh_buffer"]
    model_name = "XAJ_2S"
    checkpoint_schema = "xaj_2s_cn_v1"

    def __init__(
        self,
        nearzero: float = 1e-8,
        compact_output: bool = False,
        compile_step: bool = True,
    ):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self._fused_step = (
            torch.compile(_cemaneige_xaj_2s_fused_step, fullgraph=True)
            if compile_step
            else _cemaneige_xaj_2s_fused_step
        )

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return XAJ_2S_PARAM_SPECS

    def forward(self, forcings, params, initial_states=None, return_states=False):
        precip, pet, temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)
        ctg, kf, g_thresh, G, eTG = self._cn_setup(
            precip,
            temp,
            batch,
            device,
            dtype,
            params,
            initial_states,
        )
        _, xaj_initial = self._split_initial(initial_states)
        if xaj_initial is not None and "qb" in xaj_initial:
            # XAJ's common initializer calls its first recursive store QI;
            # map the two-source QB name only at this shared initialization
            # boundary, not in the step equations or public diagnostics.
            xaj_initial = {**xaj_initial, "qi": xaj_initial["qb"]}
        k, b, im, um, lm, dm, c, sm, ex, kb, cb, a_uh, theta_uh = (
            _prepare_xaj_2s_parameters(params)
        )
        # Reuse XAJ's shared physical state initializer, retaining only QB.
        init = XAJ._init_states(self, batch, device, dtype, xaj_initial, um, lm, dm, sm)
        wu, wl, wd, s, fr, qb, _unused_qg, rs_uh_buffer = init
        compact = self.compact_output and not return_states
        rs_values: list[torch.Tensor] = []
        qb_values: list[torch.Tensor] = []
        if not compact:
            effective: list[torch.Tensor] = []
            snow: list[torch.Tensor] = []
            thermal: list[torch.Tensor] = []
            sca: list[torch.Tensor] = []
            rain: list[torch.Tensor] = []
            melt: list[torch.Tensor] = []
            evap: list[torch.Tensor] = []
            wu_values: list[torch.Tensor] = []
            wl_values: list[torch.Tensor] = []
            wd_values: list[torch.Tensor] = []
            s_values: list[torch.Tensor] = []
            fr_values: list[torch.Tensor] = []
            rs_raw: list[torch.Tensor] = []
            rb_values: list[torch.Tensor] = []
            eu_values: list[torch.Tensor] = []
            el_values: list[torch.Tensor] = []
            ed_values: list[torch.Tensor] = []
        for t in range(nsteps):
            out = self._fused_step(
                precip[:, t],
                temp[:, t],
                pet[:, t],
                (G, eTG),
                (wu, wl, wd, s, fr, qb),
                (ctg, kf, g_thresh),
                (k, b, im, um, lm, dm, c, sm, ex, kb, cb),
                self.nearzero,
            )
            (
                effective_t,
                G,
                eTG,
                sca_t,
                rain_t,
                melt_t,
                _instant,
                rs_adj_t,
                qb,
                evap_t,
                wu,
                wl,
                wd,
                s,
                fr,
                rs_t,
                rb_t,
                eu_t,
                el_t,
                ed_t,
            ) = out
            rs_values.append(rs_adj_t)
            qb_values.append(qb)
            if not compact:
                effective.append(effective_t)
                snow.append(G)
                thermal.append(eTG)
                sca.append(sca_t)
                rain.append(rain_t)
                melt.append(melt_t)
                evap.append(evap_t)
                wu_values.append(wu)
                wl_values.append(wl)
                wd_values.append(wd)
                s_values.append(s)
                fr_values.append(fr)
                rs_raw.append(rs_t)
                rb_values.append(rb_t)
                eu_values.append(eu_t)
                el_values.append(el_t)
                ed_values.append(ed_t)
        rs_store = torch.stack(rs_values, dim=1)
        qb_store = torch.stack(qb_values, dim=1)
        rs_routed, rs_uh_buffer = _route_xaj_surface_runoff(
            rs_store, rs_uh_buffer, a_uh, theta_uh, device, dtype
        )
        qsim = rs_routed + qb_store
        if compact:
            return qsim, {}
        aux = self._diagnostic_aux(
            qsim=qsim,
            effective=effective,
            snow=snow,
            thermal=thermal,
            sca=sca,
            rain=rain,
            melt=melt,
            model_name=self.model_name,
            evaporation_scheme="standard_sequential",
        )
        aux.update(
            {
                "rs_adj": rs_store,
                "rs_routed": rs_routed,
                "qb": qb_store,
                "evap_total": torch.stack(evap, dim=1),
                "wu": torch.stack(wu_values, dim=1),
                "wl": torch.stack(wl_values, dim=1),
                "wd": torch.stack(wd_values, dim=1),
                "s_next": torch.stack(s_values, dim=1),
                "fr": torch.stack(fr_values, dim=1),
                "rs": torch.stack(rs_raw, dim=1),
                "rb": torch.stack(rb_values, dim=1),
                "eu": torch.stack(eu_values, dim=1),
                "el": torch.stack(el_values, dim=1),
                "ed": torch.stack(ed_values, dim=1),
            }
        )
        if return_states:
            aux["final_states"] = {
                "cn_G": G,
                "cn_eTG": eTG,
                "xaj_wu": wu,
                "xaj_wl": wl,
                "xaj_wd": wd,
                "xaj_s": s,
                "xaj_fr": fr,
                "xaj_qb": qb,
                "xaj_rs_uh_buffer": rs_uh_buffer,
            }
        return qsim, aux


class XAJRWPEWithCemaNeige(_XAJCNVariant):
    """CN-coupled XAJ with aggregated root-zone moisture-stress evaporation."""

    state_names = ["wu", "wl", "wd", "s", "fr", "qi", "qg", "rs_uh_buffer"]
    model_name = "XAJ_RWPE"
    checkpoint_schema = "xaj_rwpe_rootzone_stress_cn_v2"
    legacy_checkpoint_schemas = frozenset({"xaj_rwpe_cn_v1"})

    @classmethod
    def validate_checkpoint_schema(cls, schema: object) -> None:
        """Reject old parallel-weight RWPE checkpoints without migration."""
        if schema != cls.checkpoint_schema:
            if schema in cls.legacy_checkpoint_schemas:
                raise RuntimeError(
                    "Legacy XAJ_RWPE checkpoint uses parallel layer weights (v1); "
                    "it cannot be loaded as tau_e root-zone stress RWPE v2. "
                    "Use an explicit initialization migration if required."
                )
            raise RuntimeError(
                f"Incompatible XAJ_RWPE checkpoint schema {schema!r}; "
                f"expected {cls.checkpoint_schema!r}."
            )

    def __init__(
        self,
        nearzero: float = 1e-8,
        compact_output: bool = False,
        compile_step: bool = True,
    ):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self._fused_step = (
            torch.compile(_cemaneige_xaj_rwpe_fused_step, fullgraph=True)
            if compile_step
            else _cemaneige_xaj_rwpe_fused_step
        )
        self._compact_step = (
            torch.compile(_xaj_rwpe_step_compact, fullgraph=True)
            if compile_step
            else _xaj_rwpe_step_compact
        )

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return XAJ_RWPE_PARAM_SPECS

    def forward(self, forcings, params, initial_states=None, return_states=False):
        precip, pet, temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)
        ctg, kf, g_thresh, G, eTG = self._cn_setup(
            precip, temp, batch, device, dtype, params, initial_states
        )
        _, xaj_initial = self._split_initial(initial_states)
        xaj_params = _prepare_xaj_rwpe_parameters(params)
        k, b, im, um, lm, dm, tau_e, sm, ex, ki, kg, ci, cg, a_uh, theta_uh = xaj_params
        wu, wl, wd, s, fr, qi, qg, rs_uh_buffer = XAJ._init_states(
            self, batch, device, dtype, xaj_initial, um, lm, dm, sm
        )
        compact = self.compact_output and not return_states
        rs_values: list[torch.Tensor] = []
        qi_values: list[torch.Tensor] = []
        qg_values: list[torch.Tensor] = []
        if not compact:
            effective: list[torch.Tensor] = []
            snow: list[torch.Tensor] = []
            thermal: list[torch.Tensor] = []
            sca: list[torch.Tensor] = []
            rain: list[torch.Tensor] = []
            melt: list[torch.Tensor] = []
            evap: list[torch.Tensor] = []
            wu_values: list[torch.Tensor] = []
            wl_values: list[torch.Tensor] = []
            wd_values: list[torch.Tensor] = []
            s_values: list[torch.Tensor] = []
            fr_values: list[torch.Tensor] = []
            rs_raw: list[torch.Tensor] = []
            ri_values: list[torch.Tensor] = []
            rg_values: list[torch.Tensor] = []
            eu_values: list[torch.Tensor] = []
            el_values: list[torch.Tensor] = []
            ed_values: list[torch.Tensor] = []
            er_values: list[torch.Tensor] = []
            z_root_values: list[torch.Tensor] = []
            stress_values: list[torch.Tensor] = []
        for t in range(nsteps):
            if compact:
                effective_t, G, eTG, _sca_t, _rain_t, _melt_t = _cemaneige_step(
                    precip[:, t],
                    temp[:, t],
                    G,
                    eTG,
                    ctg,
                    kf,
                    g_thresh,
                    self.nearzero,
                )
                wm = um + lm + dm
                rs_adj_t, qi, qg, wu, wl, wd, s, fr = self._compact_step(
                    effective_t,
                    pet[:, t],
                    wu,
                    wl,
                    wd,
                    s,
                    fr,
                    qi,
                    qg,
                    *xaj_params[:-2],
                    self.nearzero,
                    wm,
                    wm * (1.0 + b),
                    sm * (1.0 + ex),
                    1.0 - im,
                    1.0 - ki - kg,
                )
            else:
                wl_before, wd_before = wl, wd
                out = self._fused_step(
                    precip[:, t],
                    temp[:, t],
                    pet[:, t],
                    (G, eTG),
                    (wu, wl, wd, s, fr, qi, qg),
                    (ctg, kf, g_thresh),
                    xaj_params[:-2],
                    self.nearzero,
                )
                (
                    effective_t,
                    G,
                    eTG,
                    sca_t,
                    rain_t,
                    melt_t,
                    _instant,
                    rs_adj_t,
                    qi,
                    qg,
                    evap_t,
                    wu,
                    wl,
                    wd,
                    s,
                    fr,
                    rs_t,
                    ri_t,
                    rg_t,
                    eu_t,
                    el_t,
                    ed_t,
                ) = out
                remaining_pet_t = torch.clamp(
                    torch.clamp(pet[:, t] * k, min=0.0) - eu_t, min=0.0
                )
                _el_check, _ed_check, er_t, z_root_t, stress_t = (
                    _rootzone_moisture_stress_evaporation(
                        remaining_pet_t,
                        wl_before,
                        wd_before,
                        lm,
                        dm,
                        tau_e,
                        self.nearzero,
                    )
                )
            rs_values.append(rs_adj_t)
            qi_values.append(qi)
            qg_values.append(qg)
            if not compact:
                effective.append(effective_t)
                snow.append(G)
                thermal.append(eTG)
                sca.append(sca_t)
                rain.append(rain_t)
                melt.append(melt_t)
                evap.append(evap_t)
                wu_values.append(wu)
                wl_values.append(wl)
                wd_values.append(wd)
                s_values.append(s)
                fr_values.append(fr)
                rs_raw.append(rs_t)
                ri_values.append(ri_t)
                rg_values.append(rg_t)
                eu_values.append(eu_t)
                el_values.append(el_t)
                ed_values.append(ed_t)
                er_values.append(er_t)
                z_root_values.append(z_root_t)
                stress_values.append(stress_t)
        rs_store = torch.stack(rs_values, dim=1)
        qi_store = torch.stack(qi_values, dim=1)
        qg_store = torch.stack(qg_values, dim=1)
        rs_routed, rs_uh_buffer = _route_xaj_surface_runoff(
            rs_store, rs_uh_buffer, a_uh, theta_uh, device, dtype
        )
        qsim = rs_routed + qi_store + qg_store
        if compact:
            return qsim, {}
        aux = self._diagnostic_aux(
            qsim=qsim,
            effective=effective,
            snow=snow,
            thermal=thermal,
            sca=sca,
            rain=rain,
            melt=melt,
            model_name=self.model_name,
            evaporation_scheme="aggregated_rootzone_moisture_stress",
        )
        aux.update(
            {
                "rs_adj": rs_store,
                "rs_routed": rs_routed,
                "qi": qi_store,
                "qg": qg_store,
                "evap_total": torch.stack(evap, dim=1),
                "wu": torch.stack(wu_values, dim=1),
                "wl": torch.stack(wl_values, dim=1),
                "wd": torch.stack(wd_values, dim=1),
                "s_next": torch.stack(s_values, dim=1),
                "fr": torch.stack(fr_values, dim=1),
                "rs": torch.stack(rs_raw, dim=1),
                "ri": torch.stack(ri_values, dim=1),
                "rg": torch.stack(rg_values, dim=1),
                "eu": torch.stack(eu_values, dim=1),
                "el": torch.stack(el_values, dim=1),
                "ed": torch.stack(ed_values, dim=1),
                "er": torch.stack(er_values, dim=1),
                "z_root": torch.stack(z_root_values, dim=1),
                "root_stress": torch.stack(stress_values, dim=1),
                "tau_e": tau_e,
            }
        )
        if return_states:
            aux["final_states"] = {
                "cn_G": G,
                "cn_eTG": eTG,
                "xaj_wu": wu,
                "xaj_wl": wl,
                "xaj_wd": wd,
                "xaj_s": s,
                "xaj_fr": fr,
                "xaj_qi": qi,
                "xaj_qg": qg,
                "xaj_rs_uh_buffer": rs_uh_buffer,
            }
        return qsim, aux


class XAJ2SWithCemaNeigeLite(XAJ2SWithCemaNeige):
    """Streamflow-only training path for :class:`XAJ2SWithCemaNeige`."""

    def __init__(self, nearzero: float = 1e-8, compile_step: bool = True):
        super().__init__(
            nearzero=nearzero, compact_output=True, compile_step=compile_step
        )


class XAJRWPEWithCemaNeigeLite(XAJRWPEWithCemaNeige):
    """Streamflow-only training path for :class:`XAJRWPEWithCemaNeige`."""

    def __init__(self, nearzero: float = 1e-8, compile_step: bool = True):
        super().__init__(
            nearzero=nearzero, compact_output=True, compile_step=compile_step
        )


class XAJControlledN(XAJ):
    """Native XAJ equations under the dissertation finite response domain."""

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return XAJ_CONTROLLED_N_PARAM_SPECS


class XAJControlledNLite(XAJLite):
    """Compact controlled native reference with the same finite domain."""

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return XAJ_CONTROLLED_N_PARAM_SPECS


# Alternative descriptive aliases for the common controlled N reference.
XAJNControlled = XAJControlledN
XAJNControlledLite = XAJControlledNLite


def _xaj_structure_step_full(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    wu: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    s: torch.Tensor,
    fr: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    k: torch.Tensor,
    b: torch.Tensor,
    im: torch.Tensor,
    um: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    c: torch.Tensor,
    sm: torch.Tensor,
    ex: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    ci: torch.Tensor,
    cg: torch.Tensor,
    a_uh: torch.Tensor,
    theta_uh: torch.Tensor,
    nearzero: float,
    wm: torch.Tensor,
    wmm: torch.Tensor,
    ms: torch.Tensor,
    one_minus_im: torch.Tensor,
    one_minus_p1_p2: torch.Tensor,
    variant: int,
    gamma: torch.Tensor,
    tau_0: torch.Tensor,
    beta: torch.Tensor,
    z0: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Compiled full XAJ step for one single-process controlled variant."""
    if variant == 0:  # D_E
        evaporation_scheme, response_scheme = 2, 0
    elif variant == 1:  # G_E
        evaporation_scheme, response_scheme = 3, 0
    elif variant == 2:  # D_R
        evaporation_scheme, response_scheme = 0, 1
    else:  # G_R
        evaporation_scheme, response_scheme = 0, 2
    return _xaj_step_impl(
        precip_t,
        pet_t,
        wu,
        wl,
        wd,
        s,
        fr,
        q1,
        q2,
        k,
        b,
        im,
        um,
        lm,
        dm,
        c,
        sm,
        ex,
        p1,
        p2,
        ci,
        cg,
        nearzero,
        wm,
        wmm,
        ms,
        one_minus_im,
        one_minus_p1_p2,
        True,
        evaporation_scheme,
        None,
        response_scheme,
        gamma,
        tau_0,
        beta,
        z0,
    )


def _xaj_structure_step_compact(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    wu: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    s: torch.Tensor,
    fr: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    k: torch.Tensor,
    b: torch.Tensor,
    im: torch.Tensor,
    um: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    c: torch.Tensor,
    sm: torch.Tensor,
    ex: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    ci: torch.Tensor,
    cg: torch.Tensor,
    a_uh: torch.Tensor,
    theta_uh: torch.Tensor,
    nearzero: float,
    wm: torch.Tensor,
    wmm: torch.Tensor,
    ms: torch.Tensor,
    one_minus_im: torch.Tensor,
    one_minus_p1_p2: torch.Tensor,
    variant: int,
    gamma: torch.Tensor,
    tau_0: torch.Tensor,
    beta: torch.Tensor,
    z0: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Compiled compact XAJ step with the same process equations."""
    if variant == 0:
        evaporation_scheme, response_scheme = 2, 0
    elif variant == 1:
        evaporation_scheme, response_scheme = 3, 0
    elif variant == 2:
        evaporation_scheme, response_scheme = 0, 1
    else:
        evaporation_scheme, response_scheme = 0, 2
    return _xaj_step_impl(
        precip_t,
        pet_t,
        wu,
        wl,
        wd,
        s,
        fr,
        q1,
        q2,
        k,
        b,
        im,
        um,
        lm,
        dm,
        c,
        sm,
        ex,
        p1,
        p2,
        ci,
        cg,
        nearzero,
        wm,
        wmm,
        ms,
        one_minus_im,
        one_minus_p1_p2,
        False,
        evaporation_scheme,
        None,
        response_scheme,
        gamma,
        tau_0,
        beta,
        z0,
    )


class _XAJStructureVariant(BaseHydrologicalModel):
    """Standalone XAJ host for exactly one D/G structure manipulation."""

    routing_method = "gamma"
    uh_max_len = XAJ_UH_MAX_LEN
    variant: int
    model_name: str
    parameter_spec: dict[str, dict[str, Any]]
    response_variant: bool
    generic_variant: bool
    compact_output: bool
    use_hydrodl2_uh: bool

    def __init__(
        self,
        nearzero: float = 1e-8,
        *,
        compact_output: bool = False,
        compile_step: bool = True,
        z0: float | torch.Tensor = DEFAULT_Z0,
    ):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self.use_hydrodl2_uh = compact_output
        if isinstance(z0, torch.Tensor):
            if z0.numel() != 1 or not bool(z0.detach().item() > 0.0):
                raise ValueError("z0 must be one positive scalar")
            self.register_buffer("z0", z0.detach().clone())
        else:
            if z0 <= 0.0:
                raise ValueError("z0 must be positive")
            self.register_buffer("z0", torch.tensor(float(z0), dtype=torch.float64))
        self._step = (
            torch.compile(_xaj_structure_step_full, fullgraph=True)
            if compile_step
            else _xaj_structure_step_full
        )
        self._compact_step = (
            torch.compile(_xaj_structure_step_compact, fullgraph=True)
            if compile_step
            else _xaj_structure_step_compact
        )

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return self.parameter_spec

    def _prepare_parameters(
        self, params: dict[str, torch.Tensor], batch: int
    ) -> tuple[torch.Tensor, ...]:
        if self.response_variant:
            zero = torch.zeros_like(params["xaj_kss"])
            k = params["xaj_k"]
            b = params["xaj_b"]
            im = params["xaj_im"]
            um = params["xaj_um"]
            lm = params["xaj_lm"]
            dm = params["xaj_dm"]
            c = params["xaj_c"]
            sm = params["xaj_sm"]
            ex = params["xaj_ex"]
            kss = params["xaj_kss"]
            tau_0 = params["xaj_tau0"]
            beta = params["xaj_beta"] if self.generic_variant else torch.ones_like(kss)
            return (
                k,
                b,
                im,
                um,
                lm,
                dm,
                c,
                sm,
                ex,
                kss,
                zero,
                torch.ones_like(kss),
                torch.ones_like(kss),
                params["xaj_a"],
                params["xaj_theta"],
                tau_0,
                beta,
            )
        # _prepare_xaj_parameters owns the native KI/KG joint boundary.  D_E
        # and G_E intentionally retain that native response organization.
        with_c = {**params, "xaj_c": torch.zeros_like(params["xaj_k"])}
        k, b, im, um, lm, dm, _c, sm, ex, ki, kg, ci, cg, a, theta = (
            _prepare_xaj_parameters(with_c)
        )
        gamma = params["xaj_gamma"] if self.generic_variant else torch.ones_like(k)
        return (
            k,
            b,
            im,
            um,
            lm,
            dm,
            _c,
            sm,
            ex,
            ki,
            kg,
            ci,
            cg,
            a,
            theta,
            torch.ones_like(k),
            gamma,
        )

    def _init_variant_states(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        initial_states: Optional[dict[str, torch.Tensor]],
        um: torch.Tensor,
        lm: torch.Tensor,
        dm: torch.Tensor,
        sm: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        initial = dict(initial_states or {})
        if self.response_variant:
            if "z" in initial and "qi" not in initial:
                initial["qi"] = initial["z"]
            initial.setdefault("qg", torch.zeros(batch, device=device, dtype=dtype))
        return XAJ._init_states(self, batch, device, dtype, initial, um, lm, dm, sm)

    def forward(self, forcings, params, initial_states=None, return_states=False):
        precip, pet, _temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)
        prepared = self._prepare_parameters(params, batch)
        (
            k,
            b,
            im,
            um,
            lm,
            dm,
            c,
            sm,
            ex,
            p1,
            p2,
            ci,
            cg,
            a_uh,
            theta_uh,
            tau_0,
            beta,
        ) = prepared
        gamma = (
            params["xaj_gamma"]
            if self.generic_variant and not self.response_variant
            else torch.ones_like(k)
        )
        z0 = self.z0.to(device=device, dtype=dtype).expand_as(k)
        wm = um + lm + dm
        wu, wl, wd, s, fr, q1, q2, rs_uh_buffer = self._init_variant_states(
            batch,
            device,
            dtype,
            initial_states,
            um,
            lm,
            dm,
            sm,
        )
        compact = self.compact_output and not return_states
        rs_values: list[torch.Tensor] = []
        baseflow_values: list[torch.Tensor] = []
        if not compact:
            q_values: list[torch.Tensor] = []
            rs_raw_values: list[torch.Tensor] = []
            evap_values: list[torch.Tensor] = []
            eu_values: list[torch.Tensor] = []
            el_values: list[torch.Tensor] = []
            ed_values: list[torch.Tensor] = []
            wu_values: list[torch.Tensor] = []
            wl_values: list[torch.Tensor] = []
            wd_values: list[torch.Tensor] = []
            s_values: list[torch.Tensor] = []
            fr_values: list[torch.Tensor] = []
            response_input_values: list[torch.Tensor] = []
            response_storage_values: list[torch.Tensor] = []
            response_available_values: list[torch.Tensor] = []
            qi_values: list[torch.Tensor] = []
            qg_values: list[torch.Tensor] = []
        for t in range(nsteps):
            out = (self._compact_step if compact else self._step)(
                precip[:, t],
                pet[:, t],
                wu,
                wl,
                wd,
                s,
                fr,
                q1,
                q2,
                k,
                b,
                im,
                um,
                lm,
                dm,
                c,
                sm,
                ex,
                p1,
                p2,
                ci,
                cg,
                a_uh,
                theta_uh,
                self.nearzero,
                wm,
                wm * (1.0 + b),
                sm * (1.0 + ex),
                1.0 - im,
                1.0 - p1 - p2,
                self.variant,
                gamma,
                tau_0,
                beta,
                z0,
            )
            if compact and self.response_variant:
                rs_t, qss_t, z_new, wu, wl, wd, s, fr = out
                rs_values.append(rs_t)
                baseflow_values.append(qss_t)
                q1, q2 = z_new, torch.zeros_like(z_new)
            elif compact:
                rs_t, q1, q2, wu, wl, wd, s, fr = out
                rs_values.append(rs_t)
                baseflow_values.append(q1 + q2)
            elif self.response_variant:
                z_old = q1
                (
                    _q,
                    rs_t,
                    qss_t,
                    evap_t,
                    wu,
                    wl,
                    wd,
                    s,
                    fr,
                    rs_raw_t,
                    r_ss_t,
                    _rg_t,
                    eu_t,
                    el_t,
                    ed_t,
                    z_new,
                    _r_ss_diag,
                ) = out
                rs_values.append(rs_t)
                baseflow_values.append(qss_t)
                q_values.append(_q)
                rs_raw_values.append(rs_raw_t)
                evap_values.append(evap_t)
                eu_values.append(eu_t)
                el_values.append(el_t)
                ed_values.append(ed_t)
                wu_values.append(wu)
                wl_values.append(wl)
                wd_values.append(wd)
                s_values.append(s)
                fr_values.append(fr)
                response_input_values.append(r_ss_t)
                response_storage_values.append(z_new)
                response_available_values.append(
                    torch.clamp(z_old, min=0.0) + torch.clamp(r_ss_t, min=0.0)
                )
                qi_values.append(qss_t)
                qg_values.append(torch.zeros_like(qss_t))
                q1, q2 = z_new, torch.zeros_like(z_new)
            else:
                (
                    _q,
                    rs_t,
                    qi_t,
                    qg_t,
                    evap_t,
                    wu,
                    wl,
                    wd,
                    s,
                    fr,
                    rs_raw_t,
                    ri_t,
                    rg_t,
                    eu_t,
                    el_t,
                    ed_t,
                ) = out
                rs_values.append(rs_t)
                baseflow_values.append(qi_t + qg_t)
                q_values.append(_q)
                rs_raw_values.append(rs_raw_t)
                evap_values.append(evap_t)
                eu_values.append(eu_t)
                el_values.append(el_t)
                ed_values.append(ed_t)
                wu_values.append(wu)
                wl_values.append(wl)
                wd_values.append(wd)
                s_values.append(s)
                fr_values.append(fr)
                qi_values.append(qi_t)
                qg_values.append(qg_t)
                response_input_values.append((ri_t + rg_t) * (1.0 - im))
                response_storage_values.append(torch.zeros_like(qi_t))
                q1, q2 = qi_t, qg_t
        rs_store = torch.stack(rs_values, dim=1)
        baseflow = torch.stack(baseflow_values, dim=1)
        if self.use_hydrodl2_uh:
            rs_routed, rs_uh_buffer = _route_xaj_surface_runoff_hydrodl2(
                rs_store, rs_uh_buffer, a_uh, theta_uh
            )
        else:
            rs_routed, rs_uh_buffer = _route_xaj_surface_runoff(
                rs_store, rs_uh_buffer, a_uh, theta_uh, device, dtype
            )
        qsim = rs_routed + baseflow
        if compact:
            return qsim, {}
        eu_store = torch.stack(eu_values, dim=1)
        el_store = torch.stack(el_values, dim=1)
        ed_store = torch.stack(ed_values, dim=1)
        aux = {
            "q_out": qsim,
            "evap": torch.stack(evap_values, dim=1),
            "evap_total": torch.stack(evap_values, dim=1),
            "rs_instant": rs_store,
            "rs_routed": rs_routed,
            "rs": torch.stack(rs_raw_values, dim=1),
            "wu": torch.stack(wu_values, dim=1),
            "wl": torch.stack(wl_values, dim=1),
            "wd": torch.stack(wd_values, dim=1),
            "s_next": torch.stack(s_values, dim=1),
            "fr": torch.stack(fr_values, dim=1),
            "eu": eu_store,
            "el": el_store,
            "ed": ed_store,
            "er": torch.clamp(pet * k.unsqueeze(1) - eu_store, min=0.0),
            "r_ss": torch.stack(response_input_values, dim=1),
            "response_storage": torch.stack(response_storage_values, dim=1),
            "model_name": self.model_name,
            "evaporation_scheme": "parallel_linear"
            if self.variant == 0
            else "parallel_power"
            if self.variant == 1
            else "native_sequential",
        }
        if self.response_variant:
            aux["q_ss"] = torch.stack(qi_values, dim=1)
            aux["z"] = torch.stack(response_storage_values, dim=1)
            aux["z_available"] = torch.stack(response_available_values, dim=1)
            conditioning = summarize_response_conditioning(
                aux["z_available"],
                aux["z"],
                z0.unsqueeze(1),
            )
            aux["extinction_mask"] = (aux["z_available"] > 0) & (aux["z"] == 0)
            aux.update({f"g_r_{name}": value for name, value in conditioning.items()})
            aux["log_z_ratio"] = torch.where(
                aux["z_available"] > 0,
                torch.log(torch.clamp(aux["z_available"] / z0.unsqueeze(1), min=1e-30)),
                torch.zeros_like(aux["z_available"]),
            )
            aux["kss"] = p1
        else:
            aux["qi"] = torch.stack(qi_values, dim=1)
            aux["qg"] = torch.stack(qg_values, dim=1)
        if return_states:
            final = {
                "wu": wu,
                "wl": wl,
                "wd": wd,
                "s": s,
                "fr": fr,
                "rs_uh_buffer": rs_uh_buffer,
            }
            if self.response_variant:
                final["z"] = q1
            else:
                final["qi"] = q1
                final["qg"] = q2
            aux["final_states"] = final
        return qsim, aux


class XAJDE(_XAJStructureVariant):
    """Standalone XAJ D_E controlled variant."""

    variant, model_name, parameter_spec = 0, "XAJ_D_E", XAJ_DE_PARAM_SPECS
    response_variant, generic_variant = False, False


class XAJGE(_XAJStructureVariant):
    """Standalone XAJ G_E controlled variant."""

    variant, model_name, parameter_spec = 1, "XAJ_G_E", XAJ_GE_PARAM_SPECS
    response_variant, generic_variant = False, True


class XAJDR(_XAJStructureVariant):
    """Standalone XAJ D_R controlled variant."""

    variant, model_name, parameter_spec = 2, "XAJ_D_R", XAJ_DR_PARAM_SPECS
    response_variant, generic_variant = True, False


class XAJGR(_XAJStructureVariant):
    """Standalone XAJ G_R controlled variant."""

    variant, model_name, parameter_spec = 3, "XAJ_G_R", XAJ_GR_PARAM_SPECS
    response_variant, generic_variant = True, True


class XAJDELite(XAJDE):
    def __init__(
        self,
        nearzero: float = 1e-8,
        compile_step: bool = True,
        z0: float | torch.Tensor = DEFAULT_Z0,
    ):
        super().__init__(
            nearzero, compact_output=True, compile_step=compile_step, z0=z0
        )


class XAJGELite(XAJGE):
    def __init__(
        self,
        nearzero: float = 1e-8,
        compile_step: bool = True,
        z0: float | torch.Tensor = DEFAULT_Z0,
    ):
        super().__init__(
            nearzero, compact_output=True, compile_step=compile_step, z0=z0
        )


class XAJDRLite(XAJDR):
    def __init__(
        self,
        nearzero: float = 1e-8,
        compile_step: bool = True,
        z0: float | torch.Tensor = DEFAULT_Z0,
    ):
        super().__init__(
            nearzero, compact_output=True, compile_step=compile_step, z0=z0
        )


class XAJGRLite(XAJGR):
    def __init__(
        self,
        nearzero: float = 1e-8,
        compile_step: bool = True,
        z0: float | torch.Tensor = DEFAULT_Z0,
    ):
        super().__init__(
            nearzero, compact_output=True, compile_step=compile_step, z0=z0
        )
