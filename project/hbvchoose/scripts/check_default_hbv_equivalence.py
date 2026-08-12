#!/usr/bin/env python3
"""Equivalence check: HbvStatic eager vs HbvFormulaStatic compat, same params."""

import csv, sys
from pathlib import Path
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_static import HbvStatic, _hbv_step
from model.hbv_formula_static import HbvFormulaStatic
from model.parameter_mapping import ParameterMapper

OUTPUT_DIR = _PROJECT / "validation_results" / "formula_combination_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LENGTH, WARM_UP = 60, 20
DTYPE = torch.float32


def _synth(length=LENGTH):
    P = torch.tensor([0.0,1.0,5.0,20.0,0.0,0.0,10.0,50.0,2.0,0.0]*(length//10+1),dtype=DTYPE)[:length]
    T = torch.tensor([-5.0,-2.0,0.0,1.0,3.0,5.0,10.0,15.0,12.0,8.0]*(length//10+1),dtype=DTYPE)[:length]
    PET= torch.tensor([0.5,1.0,2.0,3.0,4.0,3.0,2.0,1.0,0.5,0.5]*(length//10+1),dtype=DTYPE)[:length]
    return P.unsqueeze(-1), T.unsqueeze(-1), PET.unsqueeze(-1)


def _run_eager_direct(P, T, PET, phy, warm_up):
    """Run HbvStatic step-by-step with SAME physical params as formula model."""
    nsteps = P.shape[0]
    tt = phy["parTT"]; cfmax = phy["parCFMAX"]; cfr = phy["parCFR"]; cwh = phy["parCWH"]
    fc = phy["parFC"]; beta = phy["parBETA"]; lp = phy["parLP"]; perc = phy["parPERC"]
    uzl = phy["parUZL"]; k0 = phy["parK0"]; k1 = phy["parK1"]; k2 = phy["parK2"]
    nz = 1e-5

    SP = torch.full((1, 1), 0.001, dtype=DTYPE)
    MW = torch.full((1, 1), 0.001, dtype=DTYPE)
    SM = torch.full((1, 1), 0.001, dtype=DTYPE)
    SUZ = torch.full((1, 1), 0.001, dtype=DTYPE)
    SLZ = torch.full((1, 1), 0.001, dtype=DTYPE)
    Q_raw = torch.zeros(nsteps, dtype=DTYPE)

    for t in range(nsteps):
        q, SP, MW, SM, SUZ, SLZ = _hbv_step(
            P[t], T[t], PET[t], SP, MW, SM, SUZ, SLZ,
            tt, cfmax, cfr, cwh, fc, beta, lp, perc, uzl, k0, k1, k2, nz)
        Q_raw[t] = q.squeeze()
    return Q_raw


def _run_formula_direct(P, T, PET, fparams, warm_up):
    """Run HbvFormulaStatic compat_mode with same params."""
    fc = {"snow":"S0","recharge":"R0","aet":"E0","response":"Q0"}
    m = HbvFormulaStatic(formula_config=fc, warm_up=warm_up, param_dicts=fparams,
                         apply_routing=False, compat_mode=True)
    diag = m.simulate(P.squeeze(-1), T.squeeze(-1), PET.squeeze(-1))
    return diag["Q_raw"], diag["trace"]


def _full_trace_eager(P, T, PET, phy, warm_up):
    """Run eager step-by-step with warm-up separation and trace recording."""
    nsteps = P.shape[0]
    tt=phy["parTT"];cfmax=phy["parCFMAX"];cfr=phy["parCFR"];cwh=phy["parCWH"]
    fc_p=phy["parFC"];beta_p=phy["parBETA"];lp=phy["parLP"];perc=phy["parPERC"]
    uzl=phy["parUZL"];k0=phy["parK0"];k1=phy["parK1"];k2=phy["parK2"];nz=1e-5

    SP=torch.full((1,1),0.001,dtype=DTYPE);MW=SP.clone();SM=SP.clone();SUZ=SP.clone();SLZ=SP.clone()
    trace = {k: torch.zeros(nsteps, dtype=DTYPE) for k in
             ["SM_before","SM_after","SUZ_before","SUZ_after","SP","MW","SLZ_after",
              "melt","recharge","ETact","Q_raw","RAIN","SNOW","tosoil"]}

    for t in range(nsteps):
        trace["SM_before"][t] = SM.squeeze()
        trace["SUZ_before"][t] = SUZ.squeeze()

        # Inline _hbv_step to capture fluxes
        RAIN = P[t] * (T[t] >= tt).float()
        SNOW = P[t] * (T[t] < tt).float()
        SP2 = SP + SNOW
        melt = torch.clamp(cfmax * (T[t] - tt), min=0.0)
        melt = torch.min(melt, SP2)
        MW2 = MW + melt
        SP2 = SP2 - melt
        refrz = torch.clamp(cfr * cfmax * (tt - T[t]), min=0.0)
        refrz = torch.min(refrz, MW2)
        SP2 = SP2 + refrz
        MW2 = MW2 - refrz
        tosoil = torch.clamp(MW2 - cwh * SP2, min=0.0)
        MW2 = MW2 - tosoil
        sw = torch.clamp((SM / fc_p) ** beta_p, 0.0, 1.0)
        recharge = (RAIN + tosoil) * sw
        SM2 = SM + RAIN + tosoil - recharge
        excess = torch.clamp(SM2 - fc_p, min=0.0)
        SM2 = SM2 - excess
        evapf = torch.clamp(SM2 / (lp * fc_p), 0.0, 1.0)
        ETact = torch.min(SM2, PET[t] * evapf)
        SM2 = torch.clamp(SM2 - ETact, min=nz)
        SUZ2 = SUZ + recharge + excess
        perc_v = torch.min(SUZ2, perc)
        SUZ2 = SUZ2 - perc_v
        Q0 = k0 * torch.clamp(SUZ2 - uzl, min=0.0)
        SUZ2 = SUZ2 - Q0
        Q1 = k1 * SUZ2
        SUZ2 = SUZ2 - Q1
        SLZ2 = SLZ + perc_v
        Q2 = k2 * SLZ2
        SLZ2 = SLZ2 - Q2
        Q = Q0 + Q1 + Q2

        SP, MW, SM, SUZ, SLZ = SP2, MW2, SM2, SUZ2, SLZ2
        trace["SP"][t]=SP.squeeze();trace["MW"][t]=MW.squeeze()
        trace["SM_after"][t]=SM.squeeze();trace["SUZ_after"][t]=SUZ.squeeze()
        trace["SLZ_after"][t]=SLZ.squeeze();trace["Q_raw"][t]=Q.squeeze()
        trace["melt"][t]=melt.squeeze();trace["recharge"][t]=recharge.squeeze()
        trace["ETact"][t]=ETact.squeeze();trace["RAIN"][t]=RAIN.squeeze()
        trace["SNOW"][t]=SNOW.squeeze();trace["tosoil"][t]=tosoil.squeeze()
    return trace


def main():
    print("=== Direct-Parameter Equivalence Check ===\n")
    mapper = ParameterMapper(nmul=1)
    fc = {"snow":"S0","recharge":"R0","aet":"E0","response":"Q0"}
    norm = torch.full((1, 14), 0.5, dtype=torch.float64)
    phy, route = mapper.normalized_to_physical(norm)
    # Convert all phy values to float32 tensors
    for k in phy:
        phy[k] = phy[k].to(dtype=DTYPE)

    fparams = mapper.physical_to_formula_params(fc, phy)
    fparams["response"]["K_0"] = phy["parK0"]
    fparams["response"]["K_1"] = phy["parK1"]
    fparams["response"]["K_2"] = phy["parK2"]

    P, T, PET = _synth()

    Q_eager = _run_eager_direct(P, T, PET, phy, WARM_UP)
    Q_form, trace_form = _run_formula_direct(P, T, PET, fparams, WARM_UP)
    Q_e = Q_eager[WARM_UP:]; Q_f = Q_form

    max_abs = (Q_e - Q_f).abs().max().item()
    def v(x):
        if x < 1e-5: return "PASS"
        if x < 1e-3: return "ACCEPTABLE"
        return "PENDING"
    print(f"eager vs formula (no routing): max={max_abs:.6e} ({v(max_abs)})")

    # Full trace including warm-up
    trace_eager = _full_trace_eager(P, T, PET, phy, WARM_UP)
    n = min(trace_eager["SM_after"].shape[0], trace_form["SM_after"].shape[0])
    first_t, first_var, first_vo, first_vn, first_d = -1, "", 0.0, 0.0, 0.0

    trace_rows = []
    for t in range(n):
        phase = "warmup" if t < WARM_UP else "evaluation"
        row = {"t": t, "phase": phase, "P": P[t,0].item(), "T": T[t,0].item(), "PET": PET[t,0].item()}
        for k in ["SM_before","SM_after","SUZ_before","SUZ_after","SP","MW","SLZ_after",
                   "melt","recharge","ETact","Q_raw","RAIN","SNOW","tosoil"]:
            fk = k
            ov = trace_eager[k][t].item()
            nv = trace_form[fk][t].item()
            row[f"{k}_original"] = ov; row[f"{k}_formula"] = nv; row[f"{k}_diff"] = ov - nv
            if first_t < 0 and abs(ov - nv) > 1e-8:
                first_t, first_var, first_vo, first_vn, first_d = t, k, ov, nv, abs(ov - nv)
        trace_rows.append(row)

    trace_path = OUTPUT_DIR / "default_hbv_warmup_trace_diff.csv"
    with open(trace_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0])); w.writeheader(); w.writerows(trace_rows)

    print(f"First divergence: t={first_t} ({'warmup' if first_t < WARM_UP else 'eval'}), var={first_var}")
    print(f"  original={first_vo:.10f}, formula={first_vn:.10f}, diff={first_d:.6e}")
    csv_path = OUTPUT_DIR / "default_hbv_equivalence.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["metric","value"]);w.writeheader()
        w.writerows([
            {"metric":"eager_vs_formula_max_abs_q_diff","value":round(max_abs,6)},
            {"metric":"verdict","value":v(max_abs)},
            {"metric":"first_divergence_timestep","value":first_t},
            {"metric":"first_divergence_phase","value":"warmup" if first_t < WARM_UP else "evaluation"},
            {"metric":"first_divergence_variable","value":first_var},
            {"metric":"first_div_val_orig","value":round(first_vo,10)},
            {"metric":"first_div_val_form","value":round(first_vn,10)},
            {"metric":"first_div_abs_diff","value":round(first_d,10)},
        ])
    print(f"Trace: {trace_path}")


if __name__ == "__main__":
    main()
