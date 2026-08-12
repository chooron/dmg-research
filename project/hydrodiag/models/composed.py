from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import BaseHydrologicalModel
from .cemaneige import (
    CemaNeige,
    _cemaneige_step,
    _estimate_psol_annual,
    _init_basic_states,
)
from .gr4j import GR4J, _gr4j_step
from .xaj import (
    XAJ,
    _prepare_xaj_parameters,
    _route_xaj_surface_runoff,
    _xaj_step,
)
from .simhyd import SIMHYD, _route_simhyd_runoff, _simhyd_step
from .unit_hydro import compute_gr4j_uh_ordinates
from .parameter_specs import GR4J_CN_PARAM_SPECS, XAJ_CN_PARAM_SPECS, SIMHYD_CN_PARAM_SPECS
from .utils import validate_forcings, validate_params


def _cemaneige_xaj_fused_step(
    precip_t: torch.Tensor,
    temp_t: torch.Tensor,
    pet_t: torch.Tensor,
    cn_state: tuple[torch.Tensor, torch.Tensor],
    xaj_state: tuple[torch.Tensor, ...],
    cn_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    xaj_params: tuple[torch.Tensor, ...],
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Advance CemaNeige and XAJ for one shared time step.

    This function deliberately contains the two raw step equations in one
    compiled graph.  The composition therefore has one Python time loop and
    one fused kernel graph per day, while preserving the original ordering:
    snow/melt is computed first and its liquid output feeds XAJ on the same
    day.
    """
    effective, G, eTG, sca, rain, melt = _cemaneige_step(
        precip_t, temp_t, cn_state[0], cn_state[1],
        cn_params[0], cn_params[1], cn_params[2], nearzero,
    )
    xaj_out = _xaj_step(
        effective, pet_t, *xaj_state, *xaj_params, nearzero,
    )
    return (effective, G, eTG, sca, rain, melt, *xaj_out)


def _cemaneige_gr4j_fused_step(
    precip_t: torch.Tensor,
    temp_t: torch.Tensor,
    pet_t: torch.Tensor,
    cn_state: tuple[torch.Tensor, torch.Tensor],
    gr4j_state: tuple[torch.Tensor, ...],
    cn_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gr4j_params: tuple[torch.Tensor, ...],
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    effective, G, eTG, sca, rain, melt = _cemaneige_step(
        precip_t, temp_t, cn_state[0], cn_state[1],
        cn_params[0], cn_params[1], cn_params[2], nearzero,
    )
    gr4j_out = _gr4j_step(
        effective, pet_t, *gr4j_state, *gr4j_params, nearzero,
    )
    return (effective, G, eTG, sca, rain, melt, *gr4j_out)


def _cemaneige_simhyd_fused_step(
    precip_t: torch.Tensor,
    temp_t: torch.Tensor,
    pet_t: torch.Tensor,
    cn_state: tuple[torch.Tensor, torch.Tensor],
    simhyd_state: tuple[torch.Tensor, torch.Tensor],
    cn_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    simhyd_params: tuple[torch.Tensor, ...],
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    effective, G, eTG, sca, rain, melt = _cemaneige_step(
        precip_t, temp_t, cn_state[0], cn_state[1],
        cn_params[0], cn_params[1], cn_params[2], nearzero,
    )
    simhyd_out = _simhyd_step(
        effective, pet_t, *simhyd_state, *simhyd_params, nearzero,
    )
    return (effective, G, eTG, sca, rain, melt, *simhyd_out)


class GR4JWithCemaNeige(BaseHydrologicalModel):
    """GR4J model with CemaNeige snow module as preprocessing.

    CemaNeige converts raw precipitation and temperature into effective
    liquid water (rain + snowmelt), which is then fed to GR4J as the
    precipitation input.

    Parameters are prefixed:
        cn_ctg, cn_kf  (two-parameter CemaNeige snow params)
        gr4j_x1, gr4j_x2, gr4j_x3, gr4j_x4  (GR4J runoff params)
    """

    def __init__(self, nearzero: float = 1e-8, compact_output: bool = False):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self._cemaneige = CemaNeige(nearzero=nearzero)
        self._gr4j = GR4J(nearzero=nearzero)
        self._fused_step = torch.compile(_cemaneige_gr4j_fused_step, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return GR4J_CN_PARAM_SPECS

    def _split_params(
        self,
        params: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        cn_params = {}
        gr4j_params = {}
        for key, val in params.items():
            if key.startswith("cn_"):
                cn_params[key] = val
            elif key.startswith("gr4j_"):
                gr4j_params[key[5:]] = val  # strip gr4j_ prefix
        return cn_params, gr4j_params

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        precip, pet, temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)

        cn_params, gr4j_params = self._split_params(params)

        cn_initial = None
        gr4j_initial = None
        if initial_states is not None:
            cn_initial = {k[3:]: v for k, v in initial_states.items() if k.startswith("cn_")}
            gr4j_initial = {k[5:]: v for k, v in initial_states.items() if k.startswith("gr4j_")}

        ctg = cn_params["cn_ctg"]
        kf = cn_params["cn_kf"]
        g_thresh = 0.9 * _estimate_psol_annual(precip, temp)
        G, eTG = _init_basic_states(batch, device, dtype, cn_initial)
        x1 = gr4j_params["x1"]
        x2 = gr4j_params["x2"]
        x3 = gr4j_params["x3"]
        x4 = gr4j_params["x4"]
        uh1_ord, _ = compute_gr4j_uh_ordinates(x4, self._gr4j.UH1_MAX)
        uh2_ord = compute_gr4j_uh_ordinates(x4, self._gr4j.UH2_MAX)[1]
        s_prod, s_route, uh1_buf, uh2_buf = self._gr4j._init_states(
            batch, device, dtype, gr4j_initial, x1, x3,
        )

        compact = self.compact_output and not return_states
        if compact:
            q_values = []
        else:
            effective_precip = torch.zeros_like(precip)
            sca_store = torch.zeros_like(precip)
            rain_store = torch.zeros_like(precip)
            melt_store = torch.zeros_like(precip)
            qsim = torch.zeros_like(precip)
        for t in range(nsteps):
            (
                effective_t, G, eTG, sca_t, rain_t, melt_t,
                q_t, s_prod, s_route, uh1_buf, uh2_buf,
            ) = self._fused_step(
                precip[:, t], temp[:, t], pet[:, t],
                (G, eTG), (s_prod, s_route, uh1_buf, uh2_buf),
                (ctg, kf, g_thresh),
                (uh1_ord, uh2_ord, x1, x2, x3), self.nearzero,
            )
            if compact:
                q_values.append(q_t)
            else:
                effective_precip[:, t] = effective_t
                sca_store[:, t] = sca_t
                rain_store[:, t] = rain_t
                melt_store[:, t] = melt_t
                qsim[:, t] = q_t

        if compact:
            return torch.stack(q_values, dim=1), {}

        cn_aux = {
            "snow_pack": G, "thermal_state": eTG,
            "sca": sca_store, "rain": rain_store, "melt": melt_store,
        }
        gr4j_aux = {"s_prod": s_prod, "s_route": s_route}
        if return_states:
            cn_aux["final_states"] = {"G": G, "eTG": eTG}
            gr4j_aux["final_states"] = {
                "s_prod": s_prod, "s_route": s_route,
                "uh1_buf": uh1_buf, "uh2_buf": uh2_buf,
            }

        aux = {k: v for k, v in cn_aux.items() if k != "final_states"}
        aux.update({k: v for k, v in gr4j_aux.items() if k != "final_states"})
        aux["effective_precip"] = effective_precip
        if return_states:
            aux["final_states"] = {
                **{f"cn_{k}": v for k, v in cn_aux["final_states"].items()},
                **{f"gr4j_{k}": v for k, v in gr4j_aux["final_states"].items()},
            }

        return qsim, aux


class XAJWithCemaNeige(BaseHydrologicalModel):
    """XAJ model with CemaNeige snow module as preprocessing.

    Same pattern as GR4JWithCemaNeige.

    Parameters are prefixed:
        cn_ctg, cn_kf  (two-parameter CemaNeige snow params)
        xaj_k, xaj_b, ... (XAJ runoff params)
    """

    routing_method = "gamma"

    def __init__(self, nearzero: float = 1e-8, compact_output: bool = False):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self._cemaneige = CemaNeige(nearzero=nearzero)
        self._xaj = XAJ(nearzero=nearzero)
        self._fused_step = torch.compile(_cemaneige_xaj_fused_step, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return XAJ_CN_PARAM_SPECS

    def _split_params(
        self,
        params: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        cn_params = {}
        xaj_params = {}
        for key, val in params.items():
            if key.startswith("cn_"):
                cn_params[key] = val
            elif key.startswith("xaj_"):
                xaj_params[key] = val
        return cn_params, xaj_params

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        precip, pet, temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)

        cn_params, xaj_params = self._split_params(params)

        cn_initial = None
        xaj_initial = None
        if initial_states is not None:
            cn_initial = {k[3:]: v for k, v in initial_states.items() if k.startswith("cn_")}
            xaj_initial = {k[4:]: v for k, v in initial_states.items() if k.startswith("xaj_")}

        # Run snow/melt and runoff generation in one shared time loop.  The
        # unit-hydrograph routing remains a sequence-level operation after the
        # loop, exactly as in the standalone XAJ implementation.
        ctg = cn_params["cn_ctg"]
        kf = cn_params["cn_kf"]
        g_thresh = 0.9 * _estimate_psol_annual(precip, temp)
        G, eTG = _init_basic_states(batch, device, dtype, cn_initial)
        xaj_step_params = _prepare_xaj_parameters(xaj_params)
        (
            k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg,
            a_uh, theta_uh,
        ) = xaj_step_params
        (wu, wl, wd, s, fr, qi, qg, rs_uh_buffer) = self._xaj._init_states(
            batch, device, dtype, xaj_initial, um, lm, dm, sm,
        )

        compact = self.compact_output and not return_states
        if compact:
            rs_values = []
            baseflow_values = []
        else:
            effective_precip = torch.zeros_like(precip)
            sca_store = torch.zeros_like(precip)
            rain_store = torch.zeros_like(precip)
            melt_store = torch.zeros_like(precip)
            rs_store = torch.zeros_like(precip)
            qi_store = torch.zeros_like(precip)
            qg_store = torch.zeros_like(precip)
            evap_store = torch.zeros_like(precip)

        for t in range(nsteps):
            (
                effective_t, G, eTG, sca_t, rain_t, melt_t,
                _q_out, rs_adj_t, qi_t, qg_t, evap_t,
                wu, wl, wd, s, fr,
                _rs, _ri, _rg, _eu, _el, _ed,
            ) = self._fused_step(
                precip[:, t], temp[:, t], pet[:, t],
                (G, eTG),
                (wu, wl, wd, s, fr, qi, qg),
                (ctg, kf, g_thresh),
                xaj_step_params[:-2],
                self.nearzero,
            )
            if compact:
                rs_values.append(rs_adj_t)
                baseflow_values.append(qi_t + qg_t)
            else:
                effective_precip[:, t] = effective_t
                sca_store[:, t] = sca_t
                rain_store[:, t] = rain_t
                melt_store[:, t] = melt_t
                rs_store[:, t] = rs_adj_t
                qi_store[:, t] = qi_t
                qg_store[:, t] = qg_t
                evap_store[:, t] = evap_t
            qi, qg = qi_t, qg_t

        if compact:
            rs_store = torch.stack(rs_values, dim=1)
            qi_store = torch.stack(baseflow_values, dim=1)
            qg_store = torch.zeros_like(qi_store)
        rs_routed, rs_uh_buffer = _route_xaj_surface_runoff(
            rs_store, rs_uh_buffer, a_uh, theta_uh, device, dtype,
        )
        qsim = rs_routed + qi_store + qg_store

        if compact:
            return qsim, {}

        cn_aux = {
            "snow_pack": G,
            "thermal_state": eTG,
            "sca": sca_store,
            "rain": rain_store,
            "melt": melt_store,
        }
        xaj_aux = {
            "evap": evap_store,
            "rs_instant": rs_store,
            "rs_routed": rs_routed,
            "qi": qi_store,
            "qg": qg_store,
            "wu": wu, "wl": wl, "wd": wd,
            "s": s, "fr": fr,
        }
        if return_states:
            cn_aux["final_states"] = {"G": G, "eTG": eTG}
            xaj_aux["final_states"] = {
                "wu": wu, "wl": wl, "wd": wd, "s": s, "fr": fr,
                "qi": qi_store[:, -1], "qg": qg_store[:, -1],
                "rs_uh_buffer": rs_uh_buffer,
            }

        aux = {k: v for k, v in cn_aux.items() if k != "final_states"}
        aux.update({k: v for k, v in xaj_aux.items() if k != "final_states"})
        aux["effective_precip"] = effective_precip
        if return_states:
            aux["final_states"] = {
                **{f"cn_{k}": v for k, v in cn_aux["final_states"].items()},
                **{f"xaj_{k}": v for k, v in xaj_aux["final_states"].items()},
            }

        return qsim, aux


class SIMHYDWithCemaNeige(BaseHydrologicalModel):
    """CemaNeige + SIMHYD + gamma unit hydrograph."""

    routing_method = "gamma"

    def __init__(self, nearzero: float = 1e-8, compact_output: bool = False):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self._cemaneige = CemaNeige(nearzero=nearzero)
        self._simhyd = SIMHYD(nearzero=nearzero)
        self._fused_step = torch.compile(_cemaneige_simhyd_fused_step, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return SIMHYD_CN_PARAM_SPECS

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        precip, pet, temp, device = validate_forcings(forcings)
        batch = precip.shape[0]
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)

        cn_params = {key: value for key, value in params.items() if key.startswith("cn_")}
        simhyd_params = {
            key: value for key, value in params.items() if key.startswith("simhyd_")
        }
        cn_initial = None
        simhyd_initial = None
        if initial_states is not None:
            cn_initial = {
                key[3:]: value
                for key, value in initial_states.items()
                if key.startswith("cn_")
            }
            simhyd_initial = {
                key[7:]: value
                for key, value in initial_states.items()
                if key.startswith("simhyd_")
            }

        ctg = cn_params["cn_ctg"]
        kf = cn_params["cn_kf"]
        g_thresh = 0.9 * _estimate_psol_annual(precip, temp)
        G, eTG = _init_basic_states(batch, device, dtype, cn_initial)
        smsc = simhyd_params["simhyd_smsc"]
        soil, groundwater, runoff_uh_buffer = self._simhyd._init_states(
            batch, device, dtype, smsc,
            simhyd_initial,
        )
        simhyd_step_params = (
            simhyd_params["simhyd_insc"], simhyd_params["simhyd_coeff"],
            simhyd_params["simhyd_sq"], simhyd_params["simhyd_smsc"],
            simhyd_params["simhyd_sub"], simhyd_params["simhyd_crak"],
            simhyd_params["simhyd_k"], simhyd_params["simhyd_etmul"],
        )

        compact = self.compact_output and not return_states
        if compact:
            runoff_values = []
        else:
            effective_precip = torch.zeros_like(precip)
            sca_store = torch.zeros_like(precip)
            rain_store = torch.zeros_like(precip)
            melt_store = torch.zeros_like(precip)
            runoff_instant = torch.zeros_like(precip)
            evap = torch.zeros_like(precip)
            interception = torch.zeros_like(precip)
            direct_runoff = torch.zeros_like(precip)
            interflow = torch.zeros_like(precip)
            recharge = torch.zeros_like(precip)
            baseflow = torch.zeros_like(precip)
        for t in range(precip.shape[1]):
            (
                effective_t, G, eTG, sca_t, rain_t, melt_t,
                runoff_t, evap_t, soil, groundwater,
                interception_t, direct_t, interflow_t, recharge_t, baseflow_t,
            ) = self._fused_step(
                precip[:, t], temp[:, t], pet[:, t],
                (G, eTG), (soil, groundwater),
                (ctg, kf, g_thresh), simhyd_step_params, self.nearzero,
            )
            if compact:
                runoff_values.append(runoff_t)
            else:
                effective_precip[:, t] = effective_t
                sca_store[:, t] = sca_t
                rain_store[:, t] = rain_t
                melt_store[:, t] = melt_t
                runoff_instant[:, t] = runoff_t
                evap[:, t] = evap_t
                interception[:, t] = interception_t
                direct_runoff[:, t] = direct_t
                interflow[:, t] = interflow_t
                recharge[:, t] = recharge_t
                baseflow[:, t] = baseflow_t
        if compact:
            runoff_instant = torch.stack(runoff_values, dim=1)

        qsim, runoff_uh_buffer, uh_ordinates, routing_storage = _route_simhyd_runoff(
            runoff_instant, runoff_uh_buffer,
            simhyd_params["simhyd_a"], simhyd_params["simhyd_theta"],
            device, dtype,
        )

        if compact:
            return qsim, {}
        cn_aux = {
            "snow_pack": G, "thermal_state": eTG,
            "sca": sca_store, "rain": rain_store, "melt": melt_store,
        }
        simhyd_aux = {
            "routing_method": self.routing_method,
            "gamma_uh_ordinates": uh_ordinates,
            "evap": evap,
            "runoff_instant": runoff_instant,
            "runoff_routed": qsim,
            "interception": interception,
            "direct_runoff": direct_runoff,
            "interflow": interflow,
            "recharge": recharge,
            "baseflow": baseflow,
            "soil": soil,
            "groundwater": groundwater,
            "routing_storage": routing_storage,
        }
        if return_states:
            cn_aux["final_states"] = {"G": G, "eTG": eTG}
            simhyd_aux["final_states"] = {
                "soil": soil, "groundwater": groundwater,
                "runoff_uh_buffer": runoff_uh_buffer,
            }

        aux = {key: value for key, value in cn_aux.items() if key != "final_states"}
        aux.update(
            {key: value for key, value in simhyd_aux.items() if key != "final_states"}
        )
        aux["effective_precip"] = effective_precip
        if return_states:
            aux["final_states"] = {
                **{
                    f"cn_{key}": value
                    for key, value in cn_aux["final_states"].items()
                },
                **{
                    f"simhyd_{key}": value
                    for key, value in simhyd_aux["final_states"].items()
                },
            }
        return qsim, aux


class GR4JWithCemaNeigeLite(GR4JWithCemaNeige):
    """CemaNeige + GR4J streamflow-only training path."""

    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)


class XAJWithCemaNeigeLite(XAJWithCemaNeige):
    """CemaNeige + XAJ streamflow-only training path."""

    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)


class SIMHYDWithCemaNeigeLite(SIMHYDWithCemaNeige):
    """CemaNeige + SIMHYD streamflow-only training path."""

    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)
