"""HbvStatic — minimal, nmul-aware HBV with torch.compile'd step kernel."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma


def _hbv_step(
    Pt: torch.Tensor,
    Tt: torch.Tensor,
    PETt: torch.Tensor,
    SNOWPACK: torch.Tensor,
    MELTWATER: torch.Tensor,
    SM: torch.Tensor,
    SUZ: torch.Tensor,
    SLZ: torch.Tensor,
    TT: torch.Tensor,
    CFMAX: torch.Tensor,
    CFR: torch.Tensor,
    CWH: torch.Tensor,
    FC: torch.Tensor,
    BETA: torch.Tensor,
    LP: torch.Tensor,
    PERC: torch.Tensor,
    UZL: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    nz: float,
):
    RAIN = Pt * (Tt >= TT).float()
    SNOW = Pt * (Tt < TT).float()
    SNOWPACK = SNOWPACK + SNOW
    melt = torch.clamp(CFMAX * (Tt - TT), min=0.0)
    melt = torch.min(melt, SNOWPACK)
    MELTWATER = MELTWATER + melt
    SNOWPACK = SNOWPACK - melt
    refreezing = torch.clamp(CFR * CFMAX * (TT - Tt), min=0.0)
    refreezing = torch.min(refreezing, MELTWATER)
    SNOWPACK = SNOWPACK + refreezing
    MELTWATER = MELTWATER - refreezing
    tosoil = torch.clamp(MELTWATER - CWH * SNOWPACK, min=0.0)
    MELTWATER = MELTWATER - tosoil

    soil_wetness = torch.clamp((SM / FC) ** BETA, 0.0, 1.0)
    recharge = (RAIN + tosoil) * soil_wetness
    SM = SM + RAIN + tosoil - recharge
    excess = torch.clamp(SM - FC, min=0.0)
    SM = SM - excess
    evapfactor = torch.clamp(SM / (LP * FC), 0.0, 1.0)
    ETact = torch.min(SM, PETt * evapfactor)
    SM = torch.clamp(SM - ETact, min=nz)

    SUZ = SUZ + recharge + excess
    perc = torch.min(SUZ, PERC)
    SUZ = SUZ - perc
    Q0 = K0 * torch.clamp(SUZ - UZL, min=0.0)
    SUZ = SUZ - Q0
    Q1 = K1 * SUZ
    SUZ = SUZ - Q1
    SLZ = SLZ + perc
    Q2 = K2 * SLZ
    SLZ = SLZ - Q2

    return Q0 + Q1 + Q2, SNOWPACK, MELTWATER, SM, SUZ, SLZ


class HbvStatic(nn.Module):
    """Minimal HBV 1.0 with static parameters and routing."""

    parameter_bounds = {
        "parBETA": [1.0, 6.0],
        "parFC": [50, 1000],
        "parK0": [0.05, 0.9],
        "parK1": [0.01, 0.5],
        "parK2": [0.001, 0.2],
        "parLP": [0.2, 1],
        "parPERC": [0, 10],
        "parUZL": [0, 100],
        "parTT": [-2.5, 2.5],
        "parCFMAX": [0.5, 10],
        "parCFR": [0, 0.1],
        "parCWH": [0, 0.2],
    }
    routing_parameter_bounds = {
        "route_a": [0, 2.9],
        "route_b": [0, 6.5],
    }

    N_PHY = len(parameter_bounds)
    N_ROUTE = len(routing_parameter_bounds)

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = "HBV Static"
        cfg = config or {}
        self.warm_up = cfg.get("warm_up", 365)
        self.warm_up_states = cfg.get("warm_up_states", True)
        self.nearzero = cfg.get("nearzero", 1e-5)
        self.nmul = cfg.get("nmul", 1)
        self.variables = cfg.get("forcings", ["prcp", "tmean", "pet"])
        self.learnable_param_count = self.N_PHY * self.nmul + self.N_ROUTE
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            self._step = torch.compile(_hbv_step, fullgraph=True)
        except Exception:
            self._step = _hbv_step

    def _init_states(self, ngrid: int) -> tuple:
        s = torch.full((ngrid, self.nmul), 0.001, dtype=torch.float32, device=self.device)
        return s, s.clone(), s.clone(), s.clone(), s.clone()

    def _unpack(self, parameters: torch.Tensor):
        _, basin_count = parameters.shape[:2]
        p = parameters[-1, :, : self.N_PHY * self.nmul]
        p = p.view(basin_count, self.N_PHY, self.nmul)

        phy = {}
        for idx, (name, bounds) in enumerate(self.parameter_bounds.items()):
            phy[name] = change_param_range(p[:, idx, :], bounds)

        r = parameters[-1, :, self.N_PHY * self.nmul :]
        route = {}
        for idx, (name, bounds) in enumerate(self.routing_parameter_bounds.items()):
            route[name] = change_param_range(r[:, idx], bounds)
        return phy, route

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = x_dict["x_phy"]
        ngrid = x.shape[1]
        phy, route = self._unpack(parameters)

        warm_up = self.warm_up if self.warm_up_states else 0
        pred_cutoff = self.warm_up
        states = self._init_states(ngrid)

        if warm_up > 0:
            with torch.no_grad():
                q_warm, states = self._step_loop(x[:warm_up], states, phy)
        else:
            q_warm = None

        q_pred, _ = self._step_loop(x[warm_up:], states, phy)
        qs = torch.cat([q_warm, q_pred], dim=0).mean(-1) if q_warm is not None else q_pred.mean(-1)

        nsteps_full = qs.shape[0]
        a = route["route_a"].unsqueeze(0).unsqueeze(-1).expand(nsteps_full, -1, 1)
        b = route["route_b"].unsqueeze(0).unsqueeze(-1).expand(nsteps_full, -1, 1)
        uh = uh_gamma(a, b, lenF=15)
        rf = qs.unsqueeze(-1).permute(1, 2, 0)
        streamflow = uh_conv(rf, uh.permute(1, 2, 0)).permute(2, 0, 1)

        if pred_cutoff > 0:
            streamflow = streamflow[pred_cutoff:]
        return {"streamflow": streamflow}

    def _step_loop(
        self,
        forcing: torch.Tensor,
        states: tuple,
        phy: dict,
    ):
        p = forcing[:, :, self.variables.index("prcp")]
        t = forcing[:, :, self.variables.index("tmean")]
        pet = forcing[:, :, self.variables.index("pet")]
        nsteps, ngrid = p.shape
        m = self.nmul

        pm = p.unsqueeze(-1).expand(-1, -1, m)
        tm = t.unsqueeze(-1).expand(-1, -1, m)
        petm = pet.unsqueeze(-1).expand(-1, -1, m)

        tt = phy["parTT"]
        cfmax = phy["parCFMAX"]
        cfr = phy["parCFR"]
        cwh = phy["parCWH"]
        fc = phy["parFC"]
        beta = phy["parBETA"]
        lp = phy["parLP"]
        perc = phy["parPERC"]
        uzl = phy["parUZL"]
        k0 = phy["parK0"]
        k1 = phy["parK1"]
        k2 = phy["parK2"]

        snowpack, meltwater, sm, suz, slz = states
        qs = torch.zeros(nsteps, ngrid, m, device=self.device)
        nz = self.nearzero

        for step in range(nsteps):
            qs[step], snowpack, meltwater, sm, suz, slz = self._step(
                pm[step],
                tm[step],
                petm[step],
                snowpack,
                meltwater,
                sm,
                suz,
                slz,
                tt,
                cfmax,
                cfr,
                cwh,
                fc,
                beta,
                lp,
                perc,
                uzl,
                k0,
                k1,
                k2,
                nz,
            )

        return qs, (snowpack, meltwater, sm, suz, slz)
