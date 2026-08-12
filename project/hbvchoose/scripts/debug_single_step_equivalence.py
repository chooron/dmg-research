#!/usr/bin/env python3
"""Single-step debug: find exact SM divergence between HbvStatic and compat_mode."""

import sys, csv
from pathlib import Path
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_static import HbvStatic, _hbv_step
from model.hbv_formula_static import HbvFormulaStatic
from model.parameter_mapping import ParameterMapper

OUTPUT_DIR = _PROJECT / "validation_results" / "formula_combination_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_step_by_step(P, T, PET, tt, cfmax, cfr, cwh, fc_val, beta_val, lp, perc, uzl, k0, k1, k2, nz, nsteps):
    """Run both models step-by-step, comparing after each step."""
    dtype, device = P.dtype, P.device

    # Init states (identical)
    SP_o = torch.full((1, 1), 0.001, dtype=dtype, device=device)
    MW_o = torch.full((1, 1), 0.001, dtype=dtype, device=device)
    SM_o = torch.full((1, 1), 0.001, dtype=dtype, device=device)
    SUZ_o = torch.full((1, 1), 0.001, dtype=dtype, device=device)
    SLZ_o = torch.full((1, 1), 0.001, dtype=dtype, device=device)

    SP_f = SP_o.clone(); MW_f = MW_o.clone(); SM_f = SM_o.clone()
    SUZ_f = SUZ_o.clone(); SLZ_f = SLZ_o.clone()

    # Build formula model for compat step
    fc = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
    mapper = ParameterMapper(nmul=1)
    norm = torch.full((1, 14), 0.5, dtype=torch.float64)
    phy, route = mapper.normalized_to_physical(norm)
    fparams = mapper.physical_to_formula_params(fc, phy)
    fparams["response"]["K_0"] = phy["parK0"].float()
    fparams["response"]["K_1"] = phy["parK1"].float()
    fparams["response"]["K_2"] = phy["parK2"].float()
    fm = HbvFormulaStatic(formula_config=fc, warm_up=0, param_dicts=fparams, compat_mode=True)

    rows = []

    for t in range(nsteps):
        acc_f = {k: torch.tensor(0.0, device=device, dtype=dtype) for k in ["rainfall_total","snowfall_total","melt_total","refreezing_total","recharge_total","aet_total"]}
        doy_t = torch.as_tensor(float(t + 1), device=device, dtype=dtype)

        # Original step
        Q_o, SP_o, MW_o, SM_o, SUZ_o, SLZ_o = _hbv_step(
            P[t], T[t], PET[t], SP_o, MW_o, SM_o, SUZ_o, SLZ_o,
            tt, cfmax, cfr, cwh, fc_val, beta_val, lp, perc, uzl, k0, k1, k2, nz)

        # Formula compat step
        Q_f, SP_f, MW_f, SM_f, SUZ_f, SLZ_f, flux_f = fm._step(
            P[t], T[t], PET[t], SP_f, MW_f, SM_f, SUZ_f, SLZ_f,
            tt, cfmax, cfr, cwh, fc_val, beta_val, lp, perc, uzl, k0, k1, k2, nz, doy_t, acc_f)

        row = {"t": t, "P": P[t].item(), "T": T[t].item(), "PET": PET[t].item()}

        # Compare key states
        pairs = [("SM", SM_o, SM_f), ("SUZ", SUZ_o, SUZ_f), ("SLZ", SLZ_o, SLZ_f),
                 ("SP", SP_o, SP_f), ("MW", MW_o, MW_f), ("Q", Q_o, Q_f)]
        div_vars = []
        for name, vo, vf in pairs:
            d = (vo - vf).abs().max().item()
            row[f"{name}_original"] = vo.item()
            row[f"{name}_formula"] = vf.item()
            row[f"{name}_diff"] = d
            if d > 1e-8:
                div_vars.append(name)

        if div_vars:
            row["divergent_vars"] = ",".join(div_vars)
        else:
            row["divergent_vars"] = ""

        rows.append(row)

        if div_vars and t == 0:
            print(f"  DIVERGENCE at step 0! Vars: {div_vars}")
        if div_vars:
            print(f"  t={t}: divergence in {div_vars}")
            for name, vo, vf in pairs:
                d = (vo - vf).abs().max().item()
                if d > 1e-12:
                    print(f"    {name}: orig={vo.item():.10f}, form={vf.item():.10f}, diff={d:.6e}")
            break

    if not any(r["divergent_vars"] for r in rows):
        print(f"  All {nsteps} steps equivalent within 1e-8")

    csv_path = OUTPUT_DIR / "single_step_equivalence.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Results: {csv_path}")


def main():
    print("=== Single-Step Equivalence Debug ===\n")

    dtype = torch.float32
    L = 40
    P = torch.tensor([0.0,1.0,5.0,20.0,0.0,0.0,10.0,50.0,2.0,0.0]*(L//10+1), dtype=dtype)[:L].unsqueeze(-1)
    T = torch.tensor([-5.0,-2.0,0.0,1.0,3.0,5.0,10.0,15.0,12.0,8.0]*(L//10+1), dtype=dtype)[:L].unsqueeze(-1)
    PET= torch.tensor([0.5,1.0,2.0,3.0,4.0,3.0,2.0,1.0,0.5,0.5]*(L//10+1), dtype=dtype)[:L].unsqueeze(-1)

    mapper = ParameterMapper(nmul=1)
    norm = torch.full((1, 14), 0.5, dtype=torch.float64)
    phy, route = mapper.normalized_to_physical(norm)

    tt = phy["parTT"].to(dtype=dtype)
    cfmax = phy["parCFMAX"].to(dtype=dtype)
    cfr = phy["parCFR"].to(dtype=dtype)
    cwh = phy["parCWH"].to(dtype=dtype)
    fc_val = phy["parFC"].to(dtype=dtype)
    beta_val = phy["parBETA"].to(dtype=dtype)
    lp = phy["parLP"].to(dtype=dtype)
    perc = phy["parPERC"].to(dtype=dtype)
    uzl = phy["parUZL"].to(dtype=dtype)
    k0 = phy["parK0"].to(dtype=dtype)
    k1 = phy["parK1"].to(dtype=dtype)
    k2 = phy["parK2"].to(dtype=dtype)
    nz = 1e-5

    print(f"Parameters: TT={tt.item()}, CFMAX={cfmax.item()}, FC={fc_val.item()}, BETA={beta_val.item()}")
    print(f"Initial states: all 0.001")
    print(f"dtype={dtype}, shape=(1,1)\n")

    run_step_by_step(P, T, PET, tt, cfmax, cfr, cwh, fc_val, beta_val, lp, perc, uzl, k0, k1, k2, nz, L)


if __name__ == "__main__":
    main()
