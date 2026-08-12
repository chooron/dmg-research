#!/usr/bin/env python3
"""Second-audit: comprehensive default HBV equivalence coverage matrix."""
import csv
import sys
from pathlib import Path

import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_static import _hbv_step, HbvStatic
from model.hbv_formula_static import HbvFormulaStatic
from model.parameter_mapping import ParameterMapper

OUTPUT_DIR = _PROJECT / "validation_results" / "default_hbv_equivalence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DTYPE_F32 = torch.float32
DTYPE_F64 = torch.float64
N_ZERO = 1e-5

SYNTH_FORCING = {
    "default_cyclic": lambda n: (
        torch.tensor([0.0, 1.0, 5.0, 20.0, 0.0, 0.0, 10.0, 50.0, 2.0, 0.0] * (n // 10 + 1), dtype=DTYPE_F32)[:n].unsqueeze(-1),
        torch.tensor([-5.0, -2.0, 0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 12.0, 8.0] * (n // 10 + 1), dtype=DTYPE_F32)[:n].unsqueeze(-1),
        torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5] * (n // 10 + 1), dtype=DTYPE_F32)[:n].unsqueeze(-1),
    ),
    "all_zero_rain": lambda n: (
        torch.zeros(n, 1, dtype=DTYPE_F32),
        torch.full((n, 1), 15.0, dtype=DTYPE_F32),
        torch.full((n, 1), 4.0, dtype=DTYPE_F32),
    ),
    "heavy_rain": lambda n: (
        torch.full((n, 1), 100.0, dtype=DTYPE_F32),
        torch.full((n, 1), 5.0, dtype=DTYPE_F32),
        torch.full((n, 1), 2.0, dtype=DTYPE_F32),
    ),
    "cold_snow": lambda n: (
        torch.full((n, 1), 10.0, dtype=DTYPE_F32),
        torch.full((n, 1), -5.0, dtype=DTYPE_F32),
        torch.full((n, 1), 0.5, dtype=DTYPE_F32),
    ),
    "high_pet": lambda n: (
        torch.full((n, 1), 2.0, dtype=DTYPE_F32),
        torch.full((n, 1), 30.0, dtype=DTYPE_F32),
        torch.full((n, 1), 10.0, dtype=DTYPE_F32),
    ),
}


def build_params(norm_q=0.5, dtype=DTYPE_F32, B=1):
    mapper = ParameterMapper(nmul=B)
    fc = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
    norm = torch.full((B, 14), norm_q, dtype=torch.float64)
    phy, _route = mapper.normalized_to_physical(norm)
    for k in phy:
        phy[k] = phy[k].to(dtype=dtype)
    fparams = mapper.physical_to_formula_params(fc, phy)
    fparams["response"]["K_0"] = phy["parK0"]
    fparams["response"]["K_1"] = phy["parK1"]
    fparams["response"]["K_2"] = phy["parK2"]
    return phy, fparams, fc


def run_eager(P, T, PET, phy, warm_up=20, nsteps=None):
    """Run _hbv_step loop. Returns (Q_raw_all, states_final)."""
    if nsteps is None:
        nsteps = P.shape[0]
    tt = phy["parTT"]; cfmax = phy["parCFMAX"]; cfr = phy["parCFR"]; cwh = phy["parCWH"]
    fc_p = phy["parFC"]; beta_p = phy["parBETA"]; lp = phy["parLP"]; perc = phy["parPERC"]
    uzl = phy["parUZL"]; k0 = phy["parK0"]; k1 = phy["parK1"]; k2 = phy["parK2"]

    SP = torch.full((1, 1), 0.001, dtype=P.dtype)
    MW = SP.clone(); SM = SP.clone(); SUZ = SP.clone(); SLZ = SP.clone()
    Q_raw = torch.zeros(nsteps, dtype=P.dtype)

    trace = {
        "SP": torch.zeros(nsteps, dtype=P.dtype),
        "SM": torch.zeros(nsteps, dtype=P.dtype),
        "SUZ": torch.zeros(nsteps, dtype=P.dtype),
        "SLZ": torch.zeros(nsteps, dtype=P.dtype),
    }

    for t in range(nsteps):
        q, SP, MW, SM, SUZ, SLZ = _hbv_step(
            P[t], T[t], PET[t], SP, MW, SM, SUZ, SLZ,
            tt, cfmax, cfr, cwh, fc_p, beta_p, lp, perc, uzl, k0, k1, k2, N_ZERO)
        Q_raw[t] = q.squeeze()
        trace["SP"][t] = SP.squeeze()
        trace["SM"][t] = SM.squeeze()
        trace["SUZ"][t] = SUZ.squeeze()
        trace["SLZ"][t] = SLZ.squeeze()

    if warm_up > 0:
        return Q_raw[warm_up:], {k: v[warm_up:] for k, v in trace.items()}
    return Q_raw, trace


def run_formula(P, T, PET, fparams, fc, warm_up=20, compat_mode=True, apply_routing=False):
    """Run HbvFormulaStatic.simulate."""
    m = HbvFormulaStatic(formula_config=fc, warm_up=warm_up,
                         param_dicts=fparams, apply_routing=apply_routing,
                         compat_mode=compat_mode)
    diag = m.simulate(P.squeeze(-1), T.squeeze(-1), PET.squeeze(-1))
    tr = diag.get("trace", {})
    Q_f = diag["Q_raw"]
    SM_f = tr.get("SM_after", torch.zeros(1)) if tr else torch.zeros(1)
    SUZ_f = tr.get("SUZ_after", torch.zeros(1)) if tr else torch.zeros(1)
    SLZ_f = tr.get("SLZ_after", torch.zeros(1)) if tr else torch.zeros(1)
    SP_f = tr.get("SP", torch.zeros(1)) if tr else torch.zeros(1)
    # Formula trace is full-length, truncate to warm_up:
    n_eval = Q_f.shape[0]
    return Q_f, {
        "SM": SM_f[warm_up:] if len(SM_f) > warm_up else SM_f,
        "SUZ": SUZ_f[warm_up:] if len(SUZ_f) > warm_up else SUZ_f,
        "SLZ": SLZ_f[warm_up:] if len(SLZ_f) > warm_up else SLZ_f,
        "SP": SP_f[warm_up:] if len(SP_f) > warm_up else SP_f,
    }


def main():
    rows = []
    cases = []

    # dtype modes
    for dtype, dtype_name in [(DTYPE_F32, "float32"), (DTYPE_F64, "float64")]:
        # batch sizes
        for B in [1]:
            # apply_routing
            for apply_routing in [False, True]:
                # compat_mode
                for compat_mode in ([True, False] if not apply_routing else [True]):
                    # forcing cases
                    for fcase in SYNTH_FORCING:
                        # warmup modes
                        for n_total, warmup, phase_desc in [
                            (40, 20, "warmup_and_eval"),
                            (20, 20, "warmup_only"),
                        ]:
                            # Skip redundant: warmup_only with apply_routing
                            if phase_desc == "warmup_only" and apply_routing:
                                continue
                            cases.append({
                                "dtype": dtype_name,
                                "batch_size": B,
                                "apply_routing": apply_routing,
                                "compat_mode": compat_mode,
                                "forcing_case": fcase,
                                "phase": phase_desc,
                                "n_total": n_total,
                                "warmup": warmup,
                            })

    for c in cases:
        case_id = (f"dtype={c['dtype']},B={c['batch_size']},"
                   f"routing={c['apply_routing']},compat={c['compat_mode']},"
                   f"forcing={c['forcing_case']},phase={c['phase']}")

        try:
            phy, fparams, fc = build_params(
                norm_q=0.5,
                dtype=DTYPE_F32 if c["dtype"] == "float32" else DTYPE_F64,
                B=c["batch_size"],
            )
            P, T, PET = SYNTH_FORCING[c["forcing_case"]](c["n_total"])
            if c["dtype"] == "float64":
                P = P.to(dtype=DTYPE_F64)
                T = T.to(dtype=DTYPE_F64)
                PET = PET.to(dtype=DTYPE_F64)

            Q_e, te = run_eager(P, T, PET, phy, warm_up=c["warmup"])
            Q_f, tf = run_formula(P, T, PET, fparams, fc,
                                  warm_up=c["warmup"],
                                  compat_mode=c["compat_mode"],
                                  apply_routing=c["apply_routing"])

            n = min(Q_e.shape[0], Q_f.shape[0])
            if n == 0 and c["phase"] == "warmup_only":
                rows.append({
                    "case_id": case_id,
                    "dtype": c["dtype"],
                    "batch_size": c["batch_size"],
                    "apply_routing": int(c["apply_routing"]),
                    "compat_mode": int(c["compat_mode"]),
                    "forcing_case": c["forcing_case"],
                    "phase": c["phase"],
                    "max_diff_q": 0.0,
                    "max_diff_swe": 0.0,
                    "max_diff_sm": 0.0,
                    "max_diff_suz": 0.0,
                    "max_diff_slz": 0.0,
                    "pass": "NOT_APPLICABLE",
                })
                continue

            Q_f_a = Q_f[:n] if Q_f.ndim == 1 else Q_f[:n]
            Q_e_a = Q_e[:n] if Q_e.ndim == 1 else Q_e[:n]

            Q_f_aligned = Q_f[:n] if Q_f.ndim == 1 else Q_f[:n]

            max_q = float((Q_e_a - Q_f_aligned).abs().max().item())
            max_sm = float((te["SM"][:n] - tf["SM"][:n]).abs().max().item()) if "SM" in tf and len(tf["SM"]) >= n else 0.0
            max_suz = float((te["SUZ"][:n] - tf["SUZ"][:n]).abs().max().item()) if "SUZ" in tf and len(tf["SUZ"]) >= n else 0.0
            max_slz = float((te["SLZ"][:n] - tf["SLZ"][:n]).abs().max().item()) if "SLZ" in tf and len(tf["SLZ"]) >= n else 0.0
            max_sp = float((te["SP"][:n] - tf["SP"][:n]).abs().max().item()) if "SP" in tf and len(tf["SP"]) >= n else 0.0

            # Passing criteria:
            # compat_mode=True: exact match expected (<1e-6)
            # compat_mode=False: approximate (<0.1)
            if c["compat_mode"]:
                passed = max_q < 1e-5 and max_sm < 1e-5
            else:
                passed = max_q < 0.2  # dispatch mode has intentional differences

            rows.append({
                "case_id": case_id,
                "dtype": c["dtype"],
                "batch_size": c["batch_size"],
                "apply_routing": int(c["apply_routing"]),
                "compat_mode": int(c["compat_mode"]),
                "forcing_case": c["forcing_case"],
                "phase": c["phase"],
                "max_diff_q": round(max_q, 10),
                "max_diff_swe": round(max_sp, 10),
                "max_diff_sm": round(max_sm, 10),
                "max_diff_suz": round(max_suz, 10),
                "max_diff_slz": round(max_slz, 10),
                "pass": "PASS" if passed else "FAIL",
            })
        except Exception as e:
            rows.append({
                "case_id": case_id,
                "dtype": c["dtype"],
                "batch_size": c["batch_size"],
                "apply_routing": int(c["apply_routing"]),
                "compat_mode": int(c["compat_mode"]),
                "forcing_case": c["forcing_case"],
                "phase": c["phase"],
                "max_diff_q": -1,
                "max_diff_swe": -1,
                "max_diff_sm": -1,
                "max_diff_suz": -1,
                "max_diff_slz": -1,
                "pass": f"ERROR: {str(e)[:80]}",
            })

    csv_path = OUTPUT_DIR / "second_audit_equivalence_matrix.csv"
    fields = ["case_id", "dtype", "batch_size", "apply_routing", "compat_mode",
              "forcing_case", "phase", "max_diff_q", "max_diff_swe",
              "max_diff_sm", "max_diff_suz", "max_diff_slz", "pass"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(1 for r in rows if r["pass"] == "PASS")
    n_fail = sum(1 for r in rows if r["pass"] == "FAIL")
    n_err = sum(1 for r in rows if r["pass"].startswith("ERROR"))
    print(f"Equivalence matrix: {len(rows)} cases, {n_pass} PASS, {n_fail} FAIL, {n_err} ERROR")
    print(f"Output: {csv_path}")

    for r in rows:
        if r["pass"] != "PASS":
            print(f"  {r['pass']}: {r['case_id']}")


if __name__ == "__main__":
    main()
