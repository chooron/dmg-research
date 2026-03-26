"""HbvStatic — minimal, nmul-aware HBV with torch.compile'd step kernel."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma


# ---------------------------------------------------------------------------
# Compiled single-step HBV kernel (plain function, compiled in __init__)
# ---------------------------------------------------------------------------

def _hbv_step(
    Pt: torch.Tensor,        # [B, M]
    Tt: torch.Tensor,        # [B, M]
    PETt: torch.Tensor,      # [B, M]
    SNOWPACK: torch.Tensor,  # [B, M]
    MELTWATER: torch.Tensor, # [B, M]
    SM: torch.Tensor,        # [B, M]
    SUZ: torch.Tensor,       # [B, M]
    SLZ: torch.Tensor,       # [B, M]
    TT: torch.Tensor,        # [B, M]
    CFMAX: torch.Tensor,     # [B, M]
    CFR: torch.Tensor,       # [B, M]
    CWH: torch.Tensor,       # [B, M]
    FC: torch.Tensor,        # [B, M]
    BETA: torch.Tensor,      # [B, M]
    LP: torch.Tensor,        # [B, M]
    PERC: torch.Tensor,      # [B, M]
    UZL: torch.Tensor,       # [B, M]
    K0: torch.Tensor,        # [B, M]
    K1: torch.Tensor,        # [B, M]
    K2: torch.Tensor,        # [B, M]
    nz: float,
):
    """One HBV timestep. Returns (Q, SNOWPACK, MELTWATER, SM, SUZ, SLZ)."""
    # Snow
    RAIN = Pt * (Tt >= TT).float()
    SNOW = Pt * (Tt <  TT).float()
    SNOWPACK = SNOWPACK + SNOW
    melt = torch.clamp(CFMAX * (Tt - TT), min=0.0)
    melt = torch.min(melt, SNOWPACK)
    MELTWATER = MELTWATER + melt
    SNOWPACK  = SNOWPACK  - melt
    refreezing = torch.clamp(CFR * CFMAX * (TT - Tt), min=0.0)
    refreezing = torch.min(refreezing, MELTWATER)
    SNOWPACK  = SNOWPACK  + refreezing
    MELTWATER = MELTWATER - refreezing
    tosoil = torch.clamp(MELTWATER - CWH * SNOWPACK, min=0.0)
    MELTWATER = MELTWATER - tosoil

    # Soil
    soil_wetness = torch.clamp((SM / FC) ** BETA, 0.0, 1.0)
    recharge = (RAIN + tosoil) * soil_wetness
    SM = SM + RAIN + tosoil - recharge
    excess = torch.clamp(SM - FC, min=0.0)
    SM = SM - excess
    evapfactor = torch.clamp(SM / (LP * FC), 0.0, 1.0)
    ETact = torch.min(SM, PETt * evapfactor)
    SM = torch.clamp(SM - ETact, min=nz)

    # Runoff
    SUZ = SUZ + recharge + excess
    perc = torch.min(SUZ, PERC)
    SUZ  = SUZ - perc
    Q0   = K0 * torch.clamp(SUZ - UZL, min=0.0)
    SUZ  = SUZ - Q0
    Q1   = K1 * SUZ
    SUZ  = SUZ - Q1
    SLZ  = SLZ + perc
    Q2   = K2 * SLZ
    SLZ  = SLZ - Q2

    return Q0 + Q1 + Q2, SNOWPACK, MELTWATER, SM, SUZ, SLZ


# ---------------------------------------------------------------------------
# HbvStatic
# ---------------------------------------------------------------------------

class HbvStatic(nn.Module):
    """Minimal HBV 1.0 with static parameters, nmul support, torch.compile step.

    Parameters
    ----------
    config : dict
        Same phy config dict used by the original Hbv class.
    device : torch.device, optional
    """

    parameter_bounds = {
        'parBETA':  [1.0, 6.0],
        'parFC':    [50,  1000],
        'parK0':    [0.05, 0.9],
        'parK1':    [0.01, 0.5],
        'parK2':    [0.001, 0.2],
        'parLP':    [0.2,  1],
        'parPERC':  [0,    10],
        'parUZL':   [0,    100],
        'parTT':    [-2.5, 2.5],
        'parCFMAX': [0.5,  10],
        'parCFR':   [0,    0.1],
        'parCWH':   [0,    0.2],
    }
    routing_parameter_bounds = {
        'route_a': [0, 2.9],
        'route_b': [0, 6.5],
    }

    N_PHY   = len(parameter_bounds)        # 12
    N_ROUTE = len(routing_parameter_bounds) # 2

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = 'HBV Static'
        cfg = config or {}
        self.warm_up        = cfg.get('warm_up', 365)
        self.warm_up_states = cfg.get('warm_up_states', True)
        self.nearzero       = cfg.get('nearzero', 1e-5)
        self.nmul           = cfg.get('nmul', 1)
        self.variables      = cfg.get('forcings', ['prcp', 'tmean', 'pet'])

        self.learnable_param_count = self.N_PHY * self.nmul + self.N_ROUTE

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Compile the step kernel once at construction time
        try:
            self._step = torch.compile(_hbv_step, fullgraph=True)
        except Exception:
            self._step = _hbv_step

    # ------------------------------------------------------------------
    def _init_states(self, ngrid: int) -> tuple:
        s = torch.full((ngrid, self.nmul), 0.001, dtype=torch.float32, device=self.device)
        return s, s.clone(), s.clone(), s.clone(), s.clone()

    def _unpack(self, parameters: torch.Tensor):
        """parameters: [T, B, N_PHY*nmul + N_ROUTE]"""
        nT, B = parameters.shape[:2]
        # Physical params: use last timestep, reshape to [B, N_PHY, nmul]
        p = parameters[-1, :, :self.N_PHY * self.nmul]
        p = p.view(B, self.N_PHY, self.nmul)  # [B, N_PHY, M]

        phy = {}
        for i, (name, bounds) in enumerate(self.parameter_bounds.items()):
            phy[name] = change_param_range(p[:, i, :], bounds)  # [B, M]

        r = parameters[-1, :, self.N_PHY * self.nmul:]  # [B, 2]
        route = {}
        for i, (name, bounds) in enumerate(self.routing_parameter_bounds.items()):
            route[name] = change_param_range(r[:, i], bounds)  # [B]

        return phy, route

    # ------------------------------------------------------------------
    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = x_dict['x_phy']   # [T, B, 3]
        ngrid = x.shape[1]

        phy, route = self._unpack(parameters)

        warm_up = self.warm_up if self.warm_up_states else 0
        # Warm-up runoff is still needed for routing, but should not be scored
        # or trained against as part of the effective prediction window.
        pred_cutoff = self.warm_up

        states = self._init_states(ngrid)

        # Prediction period
        # Warm-up — store Q so routing sees the full sequence
        if warm_up > 0:
            with torch.no_grad():
                Qw, states = self._step_loop(x[:warm_up], states, phy)
        else:
            Qw = None

        # Prediction period
        Qp, _ = self._step_loop(x[warm_up:], states, phy)

        # Concatenate and average over nmul -> [T_full, B]
        Qs = torch.cat([Qw, Qp], dim=0).mean(-1) if Qw is not None else Qp.mean(-1)

        # Routing
        nsteps_full = Qs.shape[0]
        a = route['route_a'].unsqueeze(0).unsqueeze(-1).expand(nsteps_full, -1, 1)
        b = route['route_b'].unsqueeze(0).unsqueeze(-1).expand(nsteps_full, -1, 1)
        UH = uh_gamma(a, b, lenF=15)
        rf = Qs.unsqueeze(-1).permute(1, 2, 0)       # [B, 1, T]
        streamflow = uh_conv(rf, UH.permute(1, 2, 0)).permute(2, 0, 1)  # [T, B, 1]

        if pred_cutoff > 0:
            streamflow = streamflow[pred_cutoff:]

        return {'streamflow': streamflow}

    # ------------------------------------------------------------------
    def _step_loop(
        self,
        forcing: torch.Tensor,
        states: tuple,
        phy: dict,
    ):
        """Run HBV loop. Returns (Qs, new_states), Qs: [T, B, M]."""
        P   = forcing[:, :, self.variables.index('prcp')]
        T   = forcing[:, :, self.variables.index('tmean')]
        PET = forcing[:, :, self.variables.index('pet')]
        nsteps, ngrid = P.shape
        M = self.nmul

        Pm   = P.unsqueeze(-1).expand(-1, -1, M)
        Tm   = T.unsqueeze(-1).expand(-1, -1, M)
        PETm = PET.unsqueeze(-1).expand(-1, -1, M)

        TT    = phy['parTT'];    CFMAX = phy['parCFMAX']; CFR  = phy['parCFR']
        CWH   = phy['parCWH'];   FC    = phy['parFC'];    BETA = phy['parBETA']
        LP    = phy['parLP'];    PERC  = phy['parPERC'];  UZL  = phy['parUZL']
        K0    = phy['parK0'];    K1    = phy['parK1'];    K2   = phy['parK2']

        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states
        Qs = torch.zeros(nsteps, ngrid, M, device=self.device)

        nz = self.nearzero
        for t in range(nsteps):
            Qs[t], SNOWPACK, MELTWATER, SM, SUZ, SLZ = self._step(
                Pm[t], Tm[t], PETm[t],
                SNOWPACK, MELTWATER, SM, SUZ, SLZ,
                TT, CFMAX, CFR, CWH, FC, BETA, LP, PERC, UZL, K0, K1, K2,
                nz,
            )

        return Qs, (SNOWPACK, MELTWATER, SM, SUZ, SLZ)
