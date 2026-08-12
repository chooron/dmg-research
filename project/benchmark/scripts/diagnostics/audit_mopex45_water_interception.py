#!/usr/bin/env python3
"""Stages 2-4: water attribution, interception utilization, plausibility.

Uses the diagnostic step (flux-exposing) with IC / baseline-dPL / continuation
parameters across representative basins (aggregates) and a full-531 sweep of
the cheap per-basin interception metrics.

Outputs:
  water_attribution_daily_sample.csv
  water_attribution_summary.csv
  interception_utilization_by_basin.csv
  interception_utilization_group_summary.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

BENCHMARK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCHMARK), str(BENCHMARK.parents[1]), str(BENCHMARK / "src"),
                str(Path(__file__).resolve().parent)]
OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "formulation_degeneracy_audit"
OUT.mkdir(parents=True, exist_ok=True)

import audit_mopex45_sequential_discretization as A  # noqa: E402
from mopex45_discr_steps import mopex4_step_diag, mopex5_step_diag  # noqa: E402
from dpl.nn_parameterizer import CatchmentParameterizer  # noqa: E402
from dpl.attributes import CatchmentAttributeBuilder  # noqa: E402
from dmotpy.models.flux.mopex import mopex_training_context  # noqa: E402

M5_PILOT_CKPT = (BENCHMARK / "results/mopex45_phase_fix/mopex5_nested_continuation_pilot/seed_45"
                 / "checkpoints/final_endpoint.pt")


def mopex5_pilot_theta(ids):
    ck = torch.load(M5_PILOT_CKPT, map_location="cpu", weights_only=False)
    net = CatchmentParameterizer(35, 12, hidden_dims=[256, 256], dropout=0.05)
    net.load_state_dict(ck["network"])
    net.to(A.DEVICE).eval()
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    with torch.no_grad():
        return net(attrs)


def rollout_water(step_fn, x, theta_norm, basins, start, nparam, bounds):
    b = list(basins)
    P = x[start:start + 730, b, 0]; T = x[start:start + 730, b, 1]
    PET = x[start:start + 730, b, 2]; doy = x[start:start + 730, b, 3]
    theta = A.norm_to_phys(theta_norm.clone(), bounds)
    Sn, S1, S2, Sc1, Sc2 = A._init_states(len(b))
    rec = []
    with mopex_training_context(lambda_i=1.0, lambda_p=1.0, beta=50.0):
        for t in range(365):
            with torch.no_grad():
                _, _, S1, S2, Sc1, Sc2, Sn, _ = step_fn(P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn, doy=doy[t], nearzero=1e-6)
                S1, S2, Sc1, Sc2, Sn = [v.detach() for v in (S1, S2, Sc1, Sc2, Sn)]
        for t in range(365, 730):
            Q, ET, S1, S2, Sc1, Sc2, Sn, fx = step_fn(P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn, doy=doy[t], nearzero=1e-6)
            rec.append({"P": float(P[t].mean()), "T": float(T[t].mean()), "PET": float(PET[t].mean()),
                        "Pr": float(fx["flux"] is not None and 0) if False else 0.0})
            # recover Pr, Ps from snowfall split (use theta and T)
            rec[-1]["Pr"] = float((P[t] * (1 - 1 / (1 + torch.exp((T[t] - theta[:, 0]) / 0.01)))).mean())
            rec[-1]["Ps"] = float(rec[-1]["P"] - rec[-1]["Pr"])
            rec[-1]["ET1"] = float(fx["et1"].mean()); rec[-1]["I"] = float(fx["i"].mean())
            rec[-1]["ET2"] = float(fx["et2"].mean()); rec[-1]["Q"] = float(Q.mean())
            rec[-1]["S1"] = float(fx["S1_new"].mean()); rec[-1]["S2"] = float(fx["S2_new"].mean())
    return rec


def main():
    ids, x, y = A.load_data()
    theta_ic4 = A.ic_theta("mopex4", ids)
    theta_cont4 = A.continuation_theta(ids)
    theta_ic5 = A.ic_theta("mopex5", ids)
    theta_p5 = mopex5_pilot_theta(ids)
    picks = A.select_representative_basins(ids, x, y, theta_ic4, theta_cont4, n_each=4, start=1825)
    basins = [p["basin_idx"] for p in picks]
    start = 1825
    torch.set_num_threads(1); torch.set_num_interop_threads(1)

    # ---------------- Stage 2: water attribution (representative basins) ----
    daily = []
    groups = [("M4_IC", mopex4_step_diag, theta_ic4[basins], A.M4_BOUNDS),
              ("M4_continuation", mopex4_step_diag, theta_cont4[basins], A.M4_BOUNDS),
              ("M5_IC", mopex5_step_diag, theta_ic5[basins], A.M5_BOUNDS),
              ("M5_pilot", mopex5_step_diag, theta_p5[basins], A.M5_BOUNDS)]
    for gname, fn, th, bnds in groups:
        rec = rollout_water(fn, x, th, basins, start, len(bnds), bnds)
        for t, r in enumerate(rec):
            daily.append({"group": gname, "day": t, **r})
    with (OUT / "water_attribution_daily_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "day", "P", "Pr", "Ps", "PET", "ET1", "I", "ET2", "Q", "S1", "S2", "T"])
        w.writeheader(); w.writerows(daily)

    # aggregate attribution
    agg = []
    import pandas as pd
    df = pd.DataFrame(daily)
    for g, sub in df.groupby("group"):
        P, Pr, Ps = sub["P"].sum(), sub["Pr"].sum(), sub["Ps"].sum()
        I, ET1, ET2, PET = sub["I"].sum(), sub["ET1"].sum(), sub["ET2"].sum(), sub["PET"].sum()
        rainy = sub[sub["P"] > 0.1]
        snow = sub[sub["T"] < -1.0]
        agg.append({"group": g,
                    "sum_P": round(P, 2), "sum_Pr": round(Pr, 2), "sum_Ps": round(Ps, 2),
                    "sum_I": round(I, 2), "I_over_Pr": round(I / max(Pr, 1e-9), 4),
                    "I_over_P": round(I / max(P, 1e-9), 4),
                    "ET1pET2_over_PET": round((ET1 + ET2) / max(PET, 1e-9), 4),
                    "I_ET1_ET2_over_PET": round((I + ET1 + ET2) / max(PET, 1e-9), 4),
                    "rainy_I_over_Pr": round(float(rainy["I"].sum() / max(rainy["Pr"].sum(), 1e-9)), 4),
                    "snow_day_I": round(float(sub[sub["T"] < -1.0]["I"].sum()), 3),
                    "PET_near0_day_I": round(float(sub[sub["PET"] < 0.1]["I"].sum()), 3)})
    with (OUT / "water_attribution_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)
    print("=== water attribution summary ===")
    for r in agg:
        print(r)

    # ---------------- Stage 3: interception utilization (full 531) ----------
    # cheap per-basin: run one 365+365 window per basin in chunks of 100
    util_rows = []
    for gname, fn, th, bnds, pkeys in (
            ("M4_IC", mopex4_step_diag, theta_ic4, A.M4_BOUNDS,
             ["alpha", "is_time", "s2max", "tw", "tu", "se", "s3max", "tc"]),
            ("M4_cont41", mopex4_step_diag, A.continuation_theta(ids), A.M4_BOUNDS,
             ["alpha", "is_time", "s2max", "tw", "tu", "se", "s3max", "tc"]),
            ("M5_IC", mopex5_step_diag, theta_ic5, A.M5_BOUNDS,
             ["alpha", "is_time", "tmin", "trange", "s2max", "tw", "tu", "se", "s3max", "tc"]),
            ("M5_pilot", mopex5_step_diag, theta_p5, A.M5_BOUNDS,
             ["alpha", "is_time", "tmin", "trange", "s2max", "tw", "tu", "se", "s3max", "tc"])):
        phys = A.norm_to_phys(th, bnds)
        for i0 in range(0, 531, 60):
            bi = list(range(i0, min(i0 + 60, 531)))
            P = x[start:start + 730, bi, 0]; T = x[start:start + 730, bi, 1]
            PET = x[start:start + 730, bi, 2]; doy = x[start:start + 730, bi, 3]
            thm = phys[bi]
            Sn, S1, S2, Sc1, Sc2 = A._init_states(len(bi))
            I_acc = torch.zeros(len(bi), device=A.DEVICE); Pr_acc = torch.zeros(len(bi), device=A.DEVICE)
            P_acc = torch.zeros(len(bi), device=A.DEVICE)
            rainy_I, rainy_Pr, cap_days, act_days = torch.zeros(len(bi), device=A.DEVICE), torch.zeros(len(bi), device=A.DEVICE), torch.zeros(len(bi), device=A.DEVICE), torch.zeros(len(bi), device=A.DEVICE)
            with mopex_training_context(lambda_i=1.0, lambda_p=1.0, beta=50.0):
                for t in range(365):
                    with torch.no_grad():
                        _, _, S1, S2, Sc1, Sc2, Sn, _ = fn(P[t], T[t], PET[t], *thm.t(), S1, S2, Sc1, Sc2, Sn, doy=doy[t], nearzero=1e-6)
                        S1, S2, Sc1, Sc2, Sn = [v.detach() for v in (S1, S2, Sc1, Sc2, Sn)]
                for t in range(365, 730):
                    Q, ET, S1, S2, Sc1, Sc2, Sn, fx = fn(P[t], T[t], PET[t], *thm.t(), S1, S2, Sc1, Sc2, Sn, doy=doy[t], nearzero=1e-6)
                    i_ = fx["i"]; ir = fx["i_raw"]; s1b = fx["s1_before_i"]
                    pr = P[t] * (1 - 1 / (1 + torch.exp((T[t] - thm[:, 0]) / 0.01)))
                    rainy = P[t] > 0.1
                    I_acc += i_.detach(); Pr_acc += pr.detach(); P_acc += P[t].detach()
                    rainy_I += torch.where(rainy, i_.detach(), torch.zeros_like(i_))
                    rainy_Pr += torch.where(rainy, pr.detach(), torch.zeros_like(pr))
                    cap_days += ((ir >= s1b - 1e-9) & (ir > 1e-9)).detach().float()
                    act_days += (i_ > 1e-9).detach().float()
            for j, b in enumerate(bi):
                row = {"group": gname, "basin_idx": int(b), "basin_id": int(ids[b]),
                       "sum_I_over_sum_Pr": float(I_acc[j] / max(float(Pr_acc[j]), 1e-9)),
                       "sum_I_over_sum_P": float(I_acc[j] / max(float(P_acc[j]), 1e-9)),
                       "rainy_mean_I_over_Pr": float(rainy_I[j] / max(float(rainy_Pr[j]), 1e-9)),
                       "interception_active_fraction": float(act_days[j] / 365.0),
                       "cap_active_fraction": float(cap_days[j] / 365.0)}
                for pk in pkeys:
                    row[pk] = float(thm[j, bnds.index([bnds[k][0] for k in range(len(bnds))][0])]) if False else 0.0
                # physical params by name
                names = [k for k in A.M4_BOUNDS] if fn is mopex4_step_diag else [k for k in A.M5_BOUNDS]
                bounds_list = [list(v) for v in names] if False else None
                # simpler: use registry param order
                from dmotpy.models.registry import PARAM_INFO
                pnames = list(PARAM_INFO["mopex4"] if fn is mopex4_step_diag else PARAM_INFO["mopex5"])
                for pi, pk in enumerate(pnames):
                    row[pk] = float(thm[j, pi])
                util_rows.append(row)
    with (OUT / "interception_utilization_by_basin.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(util_rows[0]))
        w.writeheader(); w.writerows(util_rows)
    udf = pd.DataFrame(util_rows)
    gsum = udf.groupby("group")[["sum_I_over_sum_Pr", "sum_I_over_sum_P", "rainy_mean_I_over_Pr",
                                 "interception_active_fraction", "cap_active_fraction", "alpha"]].median()
    gsum.to_csv(OUT / "interception_utilization_group_summary.csv")
    print("=== interception utilization group median ===")
    print(gsum.round(4).to_string())


if __name__ == "__main__":
    main()
