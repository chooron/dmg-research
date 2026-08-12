#!/usr/bin/env python3
"""Fixed-combination small calibration smoke test (30 Adam steps)."""

from __future__ import annotations

import argparse, csv, math, sys
from pathlib import Path
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_formula_static import HbvFormulaStatic
from model.formula_pool import CandidateFormulaPool
from model.parameter_mapping import ParameterMapper

OUTPUT_DIR = _PROJECT / "validation_results" / "formula_calibration_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_CASES = {
    "case_01_dry": {
        "P": [0.0]*10+[2.0,0.0,0.0]*10+[0.0]*50+[1.0]*5+[0.0]*85,
        "T": [15.0]*20+[20.0]*40+[25.0]*40+[18.0]*60,
        "PET":[4.0]*60+[6.0]*40+[5.0]*60,
    },
    "case_02_wet": {
        "P": [5.0,10.0,20.0,15.0,8.0]*32,
        "T": [5.0]*40+[10.0]*40+[15.0]*40+[12.0]*40,
        "PET":[1.0]*40+[2.0]*40+[3.0]*40+[2.0]*40,
    },
    "case_03_snow_dominated": {
        "P": [5.0]*80+[10.0,20.0,30.0,25.0,15.0]*16,
        "T": [-5.0]*60+[-2.0]*20+[0.0]*20+[2.0]*20+[8.0]*40,
        "PET":[0.5]*80+[1.0]*40+[2.0]*40,
    },
    "case_04_rainfall_event": {
        "P": [0.0]*40+[80.0,40.0,20.0,10.0]+[0.0]*116,
        "T": [12.0]*160,
        "PET":[2.0]*80+[1.0]*80,
    },
    "case_05_mixed_seasonal": {
        "P": ([0.0]*20+[2.0,5.0,10.0,8.0,3.0,0.0]*4)*3,
        "T": [0.0]*40+[15.0]*40+[3.0]*40+([8.0]*20+[18.0]*20)*3,
        "PET":[1.0]*40+[4.0]*40+[2.0]*40+[3.0]*40,
    },
}

# Extra params needed per formula (beyond 14 base), with their range and median
EXTRA_PARAMS = {
    "S4": {"a_s": [0.0,0.8], "phi_s": [120.,220.]},
    "S5": {"c_m": [0.01,1.0]},
    "R4": {"a_r": [5.,15.], "c_r": [0.4,0.85]},
    "R5": {"b_v": [0.3,1.5]},
    "E3": {"gamma_E": [0.8,1.8]},
    "E4": {"s_w": [0.05,0.25], "s_o": [0.45,0.85]},
    "Q2": {"alpha_Q": [1.,3.]},
}


def _pad(lst, target):
    if len(lst) >= target: return lst[:target]
    res = []
    while len(res) < target: res.extend(lst)
    return res[:target]


def _make_forcing(case_def, length=120):
    P = torch.tensor(_pad(case_def["P"], length), dtype=torch.float64)
    T = torch.tensor(_pad(case_def["T"], length), dtype=torch.float64)
    PET= torch.tensor(_pad(case_def["PET"], length), dtype=torch.float64)
    return P, T, PET


def _extra_params_for_combo(combo):
    """Return dict of extra param names -> median values for a combo."""
    extra = {}
    for fid, params in EXTRA_PARAMS.items():
        if fid in combo.values():
            for pname, (lo, hi) in params.items():
                extra[pname] = lo + 0.5 * (hi - lo)
    return extra


def _make_fparams(combo, extra, phy):
    fparams = {}
    for node in combo:
        fparams[node] = {}
    # copy base params
    base_map = {
        "snow": ["parTT","parCFMAX","parCFR","parCWH"],
        "recharge": ["parFC","parBETA"],
        "aet": ["parFC","parLP"],
        "response": ["parK0","parK1","parK2","parUZL","parPERC"],
    }
    alias_map = {"parTT":"TT","parCFMAX":"CFMAX","parCFR":"CFR","parCWH":"CWH",
                 "parFC":"FC","parBETA":"beta","parLP":"LP",
                 "parK0":"K_0","parK1":"K_1","parK2":"K_2","parUZL":"UZL","parPERC":"PERC"}
    for node, pnames in base_map.items():
        for pn in pnames:
            if pn in phy and node in fparams:
                fparams[node][alias_map[pn]] = phy[pn].squeeze(-1)
    # add extra params
    node_finder = {"a_s":"snow","phi_s":"snow","c_m":"snow",
                   "a_r":"recharge","c_r":"recharge","b_v":"recharge",
                   "gamma_E":"aet","s_w":"aet","s_o":"aet",
                   "alpha_Q":"response"}
    for ek, ev in extra.items():
        node = node_finder.get(ek)
        if node and node in fparams:
            fparams[node][ek] = ev
    if "parPERC" in phy:
        fparams["_perc"] = phy["parPERC"].squeeze(-1)
    return fparams


def run_calibration(max_combos=None, steps=30, cases=None, output_dir=None):
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    pool = CandidateFormulaPool()
    fids = {n: pool.formulas(n, "main") for n in ["snow","recharge","aet","response"]}
    combos = []
    for sn in fids["snow"]:
        for rc in fids["recharge"]:
            for ae in fids["aet"]:
                for rs in fids["response"]:
                    combos.append({"snow": sn, "recharge": rc, "aet": ae, "response": rs})
    if max_combos:
        combos = combos[:max_combos]

    case_ids = list(SYNTHETIC_CASES) if cases is None else cases
    mapper = ParameterMapper(nmul=1)
    N_BASE = 14

    raw_rows, summary_rows, failure_rows = [], [], []

    for combo in combos:
        combo_id = "_".join(combo[n] for n in ["snow","recharge","aet","response"])
        extra = _extra_params_for_combo(combo)
        print(f"\n{combo_id}", end="", flush=True)

        for case_id in case_ids:
            case_def = SYNTHETIC_CASES[case_id]
            P, T, PET = _make_forcing(case_def, length=60)

            # Target Qobs
            fc_default = {"snow":"S0","recharge":"R0","aet":"E0","response":"Q0"}
            def_model = HbvFormulaStatic(formula_config=fc_default, warm_up=20)
            with torch.no_grad():
                diag_def = def_model.simulate(P, T, PET)
            Qobs = diag_def["Q_raw"] + torch.randn_like(diag_def["Q_raw"]) * 0.01 * diag_def["Q_raw"].std()

            # Raw params -> training
            raw = torch.zeros(N_BASE, dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.Adam([raw], lr=1e-2)

            success, fail_reason = True, ""
            nan_loss, nan_grad, inf_loss, inf_grad = False, False, False, False
            max_grad_norm = 0.0; min_loss, final_loss = float("inf"), float("inf")
            step_records = []

            for s in range(steps):
                optimizer.zero_grad()
                norm = torch.sigmoid(raw)
                phy, route = mapper.normalized_to_physical(norm.unsqueeze(0))
                fparams = _make_fparams(combo, extra, phy)

                model = HbvFormulaStatic(formula_config=combo, warm_up=20, param_dicts=fparams)
                diag = model.simulate(P, T, PET)
                Qsim = diag["Q_raw"]
                loss = torch.nn.functional.mse_loss(Qsim, Qobs)

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_loss = torch.isnan(loss).item(); inf_loss = torch.isinf(loss).item()
                    success = False; fail_reason = "NaN/Inf loss"
                    break

                loss.backward()
                gn_before = raw.grad.norm().item() if raw.grad is not None else 0.0
                if math.isnan(gn_before): nan_grad = True; success = False; fail_reason = "NaN grad"
                if math.isinf(gn_before): inf_grad = True; success = False; fail_reason = "Inf grad"
                max_grad_norm = max(max_grad_norm, gn_before)
                torch.nn.utils.clip_grad_norm_([raw], max_norm=1.0)
                optimizer.step()
                with torch.no_grad():
                    raw.clamp_(-5.0, 5.0)

                lv = loss.item(); 
                if lv < min_loss: min_loss = lv
                final_loss = lv

                step_records.append({
                    "combo_id": combo_id, "case_id": case_id, "step": s,
                    "loss": round(lv, 6), "grad_norm": round(gn_before, 6),
                    "nan_in_loss": int(torch.isnan(loss).item()),
                    "inf_in_loss": int(torch.isinf(loss).item()),
                    "nan_in_grad": int(nan_grad), "inf_in_grad": int(inf_grad),
                    "max_param_value": round(raw.max().item(), 4),
                    "min_param_value": round(raw.min().item(), 4),
                })
                if not success:
                    break

            initial_loss = step_records[0]["loss"] if step_records else float("nan")
            loss_decreased = final_loss < initial_loss - 1e-8 if step_records else False
            loss_ratio = final_loss / max(initial_loss, 1e-8) if step_records else 1.0

            if success:
                with torch.no_grad():
                    norm_final = torch.sigmoid(raw)
                    phy_f, route_f = mapper.normalized_to_physical(norm_final.unsqueeze(0))
                    fp_f = _make_fparams(combo, extra, phy_f)
                    mf = HbvFormulaStatic(formula_config=combo, warm_up=20, param_dicts=fp_f)
                    diag_final = mf.simulate(P, T, PET)
                    Qf = diag_final["Q_raw"]
                    if torch.isnan(Qf).any() or torch.isinf(Qf).any():
                        success = False; fail_reason = "Qsim NaN/Inf after calibration"
                    if diag_final["relative_water_balance_error"] > 0.10:
                        success = False; fail_reason = f"WB error={diag_final['relative_water_balance_error']:.4f}"

            summary_rows.append({
                "combo_id": combo_id, "case_id": case_id,
                "initial_loss": round(initial_loss, 6), "final_loss": round(final_loss, 6),
                "loss_ratio": round(loss_ratio, 4), "loss_decreased": loss_decreased,
                "min_loss": round(min_loss, 6), "max_grad_norm": round(max_grad_norm, 6),
                "nan_in_loss": nan_loss, "nan_in_grad": nan_grad,
                "inf_in_loss": inf_loss, "inf_in_grad": inf_grad,
                "success": success, "failure_reason": fail_reason,
            })
            if not success:
                failure_rows.append({"combo_id": combo_id, "case_id": case_id,
                                     "success": False, "failure_reason": fail_reason})
            raw_rows.extend(step_records)
            print(".", end="", flush=True)

    print()
    _w(raw_rows, out/"calibration_smoke_raw_steps.csv",
       ["combo_id","case_id","step","loss","grad_norm","nan_in_loss","inf_in_loss","nan_in_grad","inf_in_grad","max_param_value","min_param_value"])
    _w(summary_rows, out/"calibration_smoke_summary.csv",
       ["combo_id","case_id","initial_loss","final_loss","loss_ratio","loss_decreased","min_loss","max_grad_norm","nan_in_loss","nan_in_grad","inf_in_loss","inf_in_grad","success","failure_reason"])
    _w(failure_rows, out/"calibration_smoke_failures.csv",
       ["combo_id","case_id","success","failure_reason"])

    n_total = len(summary_rows); n_ok = sum(1 for r in summary_rows if r["success"])
    n_fail = n_total - n_ok
    n_nan = sum(1 for r in summary_rows if r["nan_in_loss"] or r["nan_in_grad"])

    (out/"calibration_smoke_report.md").write_text("\n".join([
        "# Formula Combination Calibration Smoke Report",
        "",
        f"## Summary",
        f"- Total: {n_total}, Success: {n_ok}, Failed: {n_fail}, NaN/Inf: {n_nan}",
    ]))

    print(f"\nTotal: {n_ok}/{n_total} OK, {n_fail} failed")


def _w(rows, path, fields):
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    run_calibration(max_combos=args.max_combos, steps=args.steps,
                    cases=args.cases, output_dir=args.output_dir)
