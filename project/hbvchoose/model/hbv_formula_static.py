"""HbvFormulaStatic — mirrors HbvStatic._hbv_step with formula-pool dispatch."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.formula_pool import CandidateFormulaPool

try:
    from hydrodl2.core.calc import uh_conv, uh_gamma
    _HAS_ROUTING = True
except ImportError:
    _HAS_ROUTING = False


class HbvFormulaStatic(nn.Module):
    """HBV with swappable process formulas via CandidateFormulaPool."""

    def __init__(
        self,
        formula_config: dict[str, str] | None = None,
        warm_up: int = 365,
        nearzero: float = 1e-5,
        apply_routing: bool = False,
        param_dicts: dict | None = None,
        compat_mode: bool = False,
    ) -> None:
        super().__init__()
        self.formula_config = formula_config or {
            "snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0",
        }
        self.warm_up = warm_up
        self.nearzero = nearzero
        self.apply_routing = apply_routing
        self.compat_mode = compat_mode
        self.pool = CandidateFormulaPool()
        self._params = param_dicts or {}

    @property
    def _default_combo(self):
        fc = self.formula_config
        return fc["snow"] == "S0" and fc["recharge"] == "R0" and fc["aet"] == "E0" and fc["response"] == "Q0"

    def _t(self, state_ref, *values):
        out = []
        for v in values:
            if torch.is_tensor(v):
                out.append(v.to(dtype=state_ref.dtype, device=state_ref.device))
            else:
                out.append(torch.as_tensor(v, dtype=state_ref.dtype, device=state_ref.device))
        return out[0] if len(out) == 1 else tuple(out)

    # ------------------------------------------------------------------
    # State advancement — mirrors _hbv_step exactly
    # ------------------------------------------------------------------

    def _step(self, Pt, Tt, PETt, SNOWPACK, MELTWATER, SM, SUZ, SLZ,
              TT, CFMAX, CFR, CWH, FC, BETA, LP, PERC, UZL, K0, K1, K2,
              nz, doy, acc):
        """Single HBV step — in compat_mode with default combo, identical to _hbv_step."""

        if self.compat_mode and self._default_combo:
            # --- Exact replica of _hbv_step from hbv_static.py ---
            RAIN = Pt * (Tt >= TT).float()
            SNOW = Pt * (Tt < TT).float()
            acc["rainfall_total"] += RAIN.sum()
            acc["snowfall_total"] += SNOW.sum()

            SNOWPACK2 = SNOWPACK + SNOW
            melt2 = torch.clamp(CFMAX * (Tt - TT), min=0.0)
            melt2 = torch.min(melt2, SNOWPACK2)
            MELTWATER2 = MELTWATER + melt2
            SNOWPACK2 = SNOWPACK2 - melt2
            acc["melt_total"] += melt2.sum()

            refreezing2 = torch.clamp(CFR * CFMAX * (TT - Tt), min=0.0)
            refreezing2 = torch.min(refreezing2, MELTWATER2)
            SNOWPACK2 = SNOWPACK2 + refreezing2
            MELTWATER2 = MELTWATER2 - refreezing2

            tosoil2 = torch.clamp(MELTWATER2 - CWH * SNOWPACK2, min=0.0)
            MELTWATER2 = MELTWATER2 - tosoil2

            soil_wetness2 = torch.clamp((SM / FC) ** BETA, 0.0, 1.0)
            recharge2 = (RAIN + tosoil2) * soil_wetness2
            SM2 = SM + RAIN + tosoil2 - recharge2
            acc["recharge_total"] += recharge2.sum()
            excess2 = torch.clamp(SM2 - FC, min=0.0)
            SM2 = SM2 - excess2

            evapfactor2 = torch.clamp(SM2 / (LP * FC), 0.0, 1.0)
            ETact2 = torch.min(SM2, PETt * evapfactor2)
            SM2 = torch.clamp(SM2 - ETact2, min=nz)
            acc["aet_total"] += ETact2.sum()

            SUZ2 = SUZ + recharge2 + excess2
            perc2 = torch.min(SUZ2, PERC)
            SUZ2 = SUZ2 - perc2
            Q0_2 = K0 * torch.clamp(SUZ2 - UZL, min=0.0)
            SUZ2 = SUZ2 - Q0_2
            Q1_2 = K1 * SUZ2
            SUZ2 = SUZ2 - Q1_2
            SLZ2 = SLZ + perc2
            Q2_2 = K2 * SLZ2
            SLZ2 = SLZ2 - Q2_2

            Q = Q0_2 + Q1_2 + Q2_2
            flux = {"RAIN": RAIN, "SNOW": SNOW, "melt": melt2, "refreezing": refreezing2,
                    "recharge": recharge2, "ETact": ETact2, "Q0": Q0_2, "Q1": Q1_2, "Q2": Q2_2,
                    "tosoil": tosoil2, "excess": excess2, "perc": perc2}
            return Q, SNOWPACK2, MELTWATER2, SM2, SUZ2, SLZ2, flux

        # --- Dispatch mode (non-compat or non-default combo) ---
        RAIN = Pt * (Tt >= TT).float()
        SNOW = Pt * (Tt < TT).float()
        acc["rainfall_total"] += RAIN.sum()
        acc["snowfall_total"] += SNOW.sum()

        SNOWPACK = SNOWPACK + SNOW
        melt = self._dispatch_melt(Tt, SNOWPACK, TT, CFMAX, doy)
        MELTWATER = MELTWATER + melt
        SNOWPACK = SNOWPACK - melt
        acc["melt_total"] += melt.sum()

        refreezing = torch.clamp(CFR * CFMAX * (TT - Tt), min=0.0)
        refreezing = torch.min(refreezing, MELTWATER)
        SNOWPACK = SNOWPACK + refreezing
        MELTWATER = MELTWATER - refreezing

        tosoil = torch.clamp(MELTWATER - CWH * SNOWPACK, min=0.0)
        MELTWATER = MELTWATER - tosoil

        recharge = self._dispatch_recharge(RAIN, tosoil, SM, FC, BETA)
        SM = SM + RAIN + tosoil - recharge
        acc["recharge_total"] += recharge.sum()
        excess = torch.clamp(SM - FC, min=0.0)
        SM = SM - excess

        ETact = self._dispatch_aet(PETt, SM, LP, FC)
        SM = torch.clamp(SM - ETact, min=nz)
        acc["aet_total"] += ETact.sum()

        SUZ = SUZ + recharge + excess
        perc = torch.min(SUZ, PERC)
        SUZ = SUZ - perc
        Q0, Q1, Q2 = self._dispatch_response(SUZ, SLZ, K0, K1, K2, UZL, PERC)
        SUZ = SUZ - Q0 - Q1
        SLZ = SLZ + perc - Q2

        Q = Q0 + Q1 + Q2
        flux = {"RAIN": RAIN, "SNOW": SNOW, "melt": melt, "refreezing": refreezing,
                "recharge": recharge, "ETact": ETact, "Q0": Q0, "Q1": Q1, "Q2": Q2,
                "tosoil": tosoil, "excess": excess, "perc": perc}
        return Q, SNOWPACK, MELTWATER, SM, SUZ, SLZ, flux

    def _dispatch_melt(self, T, SWE, TT, CFMAX, doy):
        fid = self.formula_config["snow"]
        p = self._params.get("snow", {})
        T, SWE = T.to(dtype=torch.float32), SWE.to(dtype=torch.float32)
        if fid == "S0":
            TT_t, CFMAX_t = self._t(T, p.get("TT", TT), p.get("CFMAX", CFMAX))
            return self.pool.call_formula("snow", "S0", T=T, TT=TT_t, CFMAX=CFMAX_t, SWE=SWE)
        elif fid == "S4":
            TT_t, C0, a_s, phi_s = self._t(T, p.get("TT", TT), p.get("CFMAX_0", 3.0),
                                            p.get("a_s", 0.3), p.get("phi_s", 172.0))
            doy_t = torch.as_tensor(doy, dtype=T.dtype, device=T.device)
            return self.pool.call_formula("snow", "S4", T=T, TT=TT_t, CFMAX_0=C0,
                                          a_s=a_s, phi_s=phi_s, doy=doy_t, SWE=SWE)
        elif fid == "S5":
            TT_t, CFMAX_t, c_m = self._t(T, p.get("TT", TT), p.get("CFMAX", CFMAX), p.get("c_m", 0.3))
            return self.pool.call_formula("snow", "S5", T=T, TT=TT_t, CFMAX=CFMAX_t, c_m=c_m, SWE=SWE)
        return torch.clamp(CFMAX * (T - TT), min=0.0)

    def _dispatch_recharge(self, RAIN, tosoil, SM, FC, BETA):
        fid = self.formula_config["recharge"]
        p = self._params.get("recharge", {})
        I = RAIN + tosoil
        if fid == "R0":
            FC_t, beta_t = self._t(I, p.get("FC", FC), p.get("beta", BETA))
            return self.pool.call_formula("recharge", "R0", I=I, SM=SM, FC=FC_t, beta=beta_t)
        elif fid == "R4":
            FC_t, a_r, c_r = self._t(I, p.get("FC", FC), p.get("a_r", 10.0), p.get("c_r", 0.5))
            return self.pool.call_formula("recharge", "R4", I=I, SM=SM, FC=FC_t, a_r=a_r, c_r=c_r)
        elif fid == "R5":
            FC_t, b_v = self._t(I, p.get("FC", FC), p.get("b_v", 1.0))
            return self.pool.call_formula("recharge", "R5", I=I, SM=SM, FC=FC_t, b_v=b_v)
        soil_wetness = torch.clamp((SM / FC) ** BETA, 0.0, 1.0)
        return I * soil_wetness

    def _dispatch_aet(self, PET, SM, LP, FC):
        fid = self.formula_config["aet"]
        p = self._params.get("aet", {})
        if fid == "E0":
            LP_t, FC_t = self._t(PET, p.get("LP", LP), p.get("FC", FC))
            return self.pool.call_formula("aet", "E0", PET=PET, SM=SM, LP=LP_t, FC=FC_t)
        elif fid == "E3":
            FC_t, gamma_E = self._t(PET, p.get("FC", FC), p.get("gamma_E", 1.2))
            return self.pool.call_formula("aet", "E3", PET=PET, SM=SM, FC=FC_t, gamma_E=gamma_E)
        elif fid == "E4":
            FC_t, s_w, s_o = self._t(PET, p.get("FC", FC), p.get("s_w", 0.1), p.get("s_o", 0.6))
            return self.pool.call_formula("aet", "E4", PET=PET, SM=SM, FC=FC_t, s_w=s_w, s_o=s_o)
        evapfactor = torch.clamp(SM / (LP * FC), 0.0, 1.0)
        return torch.min(SM, PET * evapfactor)

    def _dispatch_response(self, SUZ, SLZ, K0, K1, K2, UZL, PERC):
        fid = self.formula_config["response"]
        p = self._params.get("response", {})
        if fid == "Q0":
            K0_t = torch.as_tensor(p.get("K_0", K0), dtype=SUZ.dtype, device=SUZ.device)
            K1_t = torch.as_tensor(p.get("K_1", K1), dtype=SUZ.dtype, device=SUZ.device)
            K2_t = torch.as_tensor(p.get("K_2", K2), dtype=SUZ.dtype, device=SUZ.device)
            UZL_t = torch.as_tensor(p.get("UZL", UZL), dtype=SUZ.dtype, device=SUZ.device)
            Q0 = K0_t * torch.clamp(SUZ - UZL_t, min=0.0)
            SUZ2 = SUZ - Q0
            Q1 = K1_t * SUZ2
            Q2 = K2_t * SLZ
            return Q0, Q1, Q2
        elif fid == "Q2":
            K1_t = torch.as_tensor(p.get("K_1", K1), dtype=SUZ.dtype, device=SUZ.device)
            K2_t = torch.as_tensor(p.get("K_2", K2), dtype=SUZ.dtype, device=SUZ.device)
            alpha_Q = torch.as_tensor(p.get("alpha_Q", 1.2), dtype=SUZ.dtype, device=SUZ.device)
            Q0 = torch.zeros_like(SUZ)
            Quz = K1_t * SUZ ** alpha_Q
            Quz = torch.minimum(Quz, SUZ)
            Q1 = Quz - Q0
            Q2 = K2_t * SLZ
            Q2 = torch.minimum(Q2, SLZ)
            return Q0, Q1, Q2
        elif fid == "Q5":
            PART_t = torch.as_tensor(p.get("PART", 0.7), dtype=SUZ.dtype, device=SUZ.device)
            K1_t = torch.as_tensor(p.get("K_1", K1), dtype=SUZ.dtype, device=SUZ.device)
            K2_t = torch.as_tensor(p.get("K_2", K2), dtype=SUZ.dtype, device=SUZ.device)
            Q0 = torch.zeros_like(SUZ)
            Q1 = K1_t * SUZ
            Q2 = K2_t * SLZ
            return Q0, Q1, Q2
        Q0 = K0 * torch.clamp(SUZ - UZL, min=0.0)
        SUZ2 = SUZ - Q0
        Q1 = K1 * SUZ2
        Q2 = K2 * SLZ
        return Q0, Q1, Q2

    def _storage_total(self, SP, MW, SM, SUZ, SLZ):
        return SP + MW + SM + SUZ + SLZ

    def simulate(self, P, T, PET, doy=None):
        device = P.device
        dtype = P.dtype
        if P.dim() == 1:
            P = P.unsqueeze(-1); T = T.unsqueeze(-1); PET = PET.unsqueeze(-1)
        nsteps = P.shape[0]
        if doy is None:
            doy = torch.arange(1, nsteps + 1, device=device, dtype=dtype).unsqueeze(-1)
        elif doy.dim() == 1:
            doy = doy.unsqueeze(-1)

        nz = self.nearzero
        SP = torch.full_like(P[0:1], 0.001); MW = torch.full_like(P[0:1], 0.001)
        SM = torch.full_like(P[0:1], 0.001); SUZ = torch.full_like(P[0:1], 0.001)
        SLZ = torch.full_like(P[0:1], 0.001)

        params = self._params
        def _p(node, name, default):
            return torch.as_tensor(params.get(node, {}).get(name, default), dtype=dtype, device=device)

        TT=_p("snow","TT",0.5); CFMAX=_p("snow","CFMAX",3.0); CFR=_p("snow","CFR",0.05); CWH=_p("snow","CWH",0.1)
        FC=_p("recharge","FC",200.0); BETA=_p("recharge","beta",2.0); LP=_p("aet","LP",0.8)
        PERC=params.get("_perc",torch.as_tensor(1.5,dtype=dtype,device=device))
        UZL=_p("response","UZL",10.0); K0=_p("response","K_0",0.3); K1=_p("response","K_1",0.1); K2=_p("response","K_2",0.05)

        acc = {k: torch.tensor(0.0, device=device, dtype=dtype) for k in
               ["rainfall_total","snowfall_total","melt_total","refreezing_total","recharge_total","aet_total"]}

        warm_up = min(self.warm_up, nsteps)
        trace = {k: torch.zeros(nsteps, device=device, dtype=dtype) for k in
                  ["SP","MW","SM_before","SM_after","SUZ_before","SUZ_after","SLZ_after",
                   "melt","recharge","ETact","Q0","Q1","Q2","Q_raw","tosoil","excess","perc",
                   "RAIN","SNOW"]}

        for t in range(nsteps):
            phase = "warmup" if t < warm_up else "evaluation"
            trace["SM_before"][t] = SM.squeeze()
            trace["SUZ_before"][t] = SUZ.squeeze()

            q, SP, MW, SM, SUZ, SLZ, flux = self._step(
                P[t], T[t], PET[t], SP, MW, SM, SUZ, SLZ,
                TT, CFMAX, CFR, CWH, FC, BETA, LP, PERC, UZL, K0, K1, K2, nz, doy[t], acc)

            trace["SP"][t] = SP.squeeze()
            trace["MW"][t] = MW.squeeze()
            trace["SM_after"][t] = SM.squeeze()
            trace["SUZ_after"][t] = SUZ.squeeze()
            trace["SLZ_after"][t] = SLZ.squeeze()
            trace["Q_raw"][t] = q.squeeze()
            for k in ["melt","recharge","ETact","Q0","Q1","Q2","tosoil","excess","perc","RAIN","SNOW"]:
                if k in flux:
                    trace[k][t] = flux[k].squeeze()

        # Post-warmup diagnostics using only eval period
        P_eval = P[warm_up:]
        P_total = P_eval.sum()
        Q_eval = trace["Q_raw"][warm_up:]
        Q_total_raw = Q_eval.sum()
        AET_eval = trace["ETact"][warm_up:]
        AET_total = AET_eval.sum()
        # Storage: use states AFTER warm-up (index warm_up-1) as initial
        w_end = max(warm_up - 1, 0)
        init_storage = (trace["SP"][w_end] + trace["MW"][w_end] +
                        trace["SM_after"][w_end] + trace["SUZ_after"][w_end] +
                        trace["SLZ_after"][w_end])
        final_storage = (trace["SP"][-1] + trace["MW"][-1] + trace["SM_after"][-1] +
                         trace["SUZ_after"][-1] + trace["SLZ_after"][-1])
        storage_change = final_storage - init_storage
        residual = P_total - AET_total - Q_total_raw - storage_change
        rel_err = abs(residual) / max(P_total.item(), 1e-6)

        # Routing
        Q_raw_eval = Q_eval
        Qsim = Q_raw_eval
        routing_applied = False
        if self.apply_routing and _HAS_ROUTING and len(Q_raw_eval) > 0:
            a = torch.as_tensor(params.get("_route_a", 2.0), device=device, dtype=dtype)
            b = torch.as_tensor(params.get("_route_b", 2.0), device=device, dtype=dtype)
            # Match HbvStatic: uh_gamma expects (T, B, 1)
            n = len(Q_raw_eval)
            a = a.unsqueeze(0).unsqueeze(-1).expand(n, -1, 1)
            b = b.unsqueeze(0).unsqueeze(-1).expand(n, -1, 1)
            uh = uh_gamma(a, b, lenF=15)
            # Q_raw_eval is (time,) -> need (batch=1, 1, time) for uh_conv
            qs = Q_raw_eval.unsqueeze(0).unsqueeze(-1)  # (1, time, 1)
            rf = qs.permute(1, 2, 0)  # (time, 1, 1) -> (B=1, C=1, T=time) after permute? No...
            # HbvStatic: rf = qs.unsqueeze(-1).permute(1, 2, 0) where qs is (T, B)
            # So qs (T, B=1) -> unsqueeze (T, 1, 1) -> permute (1, 1, T)
            rf = Q_raw_eval.unsqueeze(-1).unsqueeze(-1)  # (T, 1, 1)
            rf = rf.permute(1, 2, 0)  # (1, 1, T)
            routed = uh_conv(rf, uh.permute(1, 2, 0))  # (B, C, T)
            Qsim = routed.permute(2, 0, 1).squeeze()  # (T,)
            routing_applied = True

        diag = {
            "Qsim": Qsim, "Q_raw": Q_raw_eval, "routing_applied": routing_applied,
            "precipitation_total": P_total.item(), "aet_total": AET_total.item(),
            "q_total": Q_total_raw.item(), "storage_change": storage_change.item(),
            "water_balance_residual": residual.item(),
            "relative_water_balance_error": rel_err.item(),
            "trace": trace,
        }
        return diag

    def forward(self, P, T, PET, doy=None):
        return self.simulate(P, T, PET, doy)
