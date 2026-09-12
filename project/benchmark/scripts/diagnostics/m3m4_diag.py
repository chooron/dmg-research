"""MOPEX3 vs MOPEX4 forward-only diagnostics (C1/C2/D1/D3/D4/E1/E2).

Read-only; no training, no production changes. Reuses audit_mopex34_root_cause
infrastructure (4 representative basins, CPU). Outputs to /tmp/m3m4_diag_out/.
"""
import csv
import sys
from pathlib import Path

import torch

REPO = Path("/home/jingxin/code/dmg-research")
BENCHMARK = REPO / "project" / "benchmark"
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"),
                str(BENCHMARK / "scripts" / "diagnostics")]

import audit_mopex34_root_cause as A
from dmotpy.models.core.mopex3 import mopex3_step
from dmotpy.models.core.mopex4 import mopex4_step

OUT = Path("/tmp/m3m4_diag_out")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")
WARMUP, SCORED = 365, 365

ids, x, y, b = A.load_context()   # x: (T, B, 4) incl. doy; y: obs (T, B)
x = x[A.START:A.START + WARMUP + SCORED]
y = y[A.START:A.START + WARMUP + SCORED]
B = len(b)
P, T, PET = x[:, :, 0], x[:, :, 1], x[:, :, 2]
DOY = x[:, :, 3]

# ---- M3-trained baseline mapped to M4 (non-interception params = M3 optimum) ----
net3, net4, src = A.mapped_m3_network()
attrs = A.CatchmentAttributeBuilder().build_normalized_attributes(
    ids, device="cpu", method="zscore")[b]
with torch.no_grad():
    raw4 = net4(attrs)                      # (B,10) normalized
p4 = A.norm_to_phys(raw4, "mopex4")         # (B,10) physical
# M4 param order: tcrit,ddf,s2max,tw,alpha,is_time,tu,se,s3max,tc
P4 = {n: p4[:, i].clone() for i, n in enumerate(
    ["tcrit","ddf","s2max","tw","alpha","is_time","tu","se","s3max","tc"])}

def run_m4(params, warmup=WARMUP, scored=SCORED):
    S1 = torch.full((B,), 1e-6); S2 = torch.full((B,), 1e-6)
    Sc1 = torch.full((B,), 1e-6); Sc2 = torch.full((B,), 1e-6)
    Sn = torch.full((B,), 1e-6)
    qs = []
    for t in range(warmup + scored):
        q, et, S1, S2, Sc1, Sc2, Sn = mopex4_step(
            P[t], T[t], PET[t], params["tcrit"], params["ddf"], params["s2max"],
            params["tw"], params["alpha"], params["is_time"], params["tu"],
            params["se"], params["s3max"], params["tc"], S1, S2, Sc1, Sc2, Sn,
            doy=DOY[t])
        if t >= warmup:
            qs.append(q)
    return torch.stack(qs)                  # (scored, B)

def run_m3(params, warmup=WARMUP, scored=SCORED):
    Sn = torch.full((B,), 1e-6); S2 = torch.full((B,), 1e-6)
    S3 = torch.full((B,), 1e-6); Sc1 = torch.full((B,), 1e-6)
    Sc2 = torch.full((B,), 1e-6)
    qs = []
    for t in range(warmup + scored):
        q, et, Sn, S2, S3, Sc1, Sc2 = mopex3_step(
            P[t], T[t], PET[t], params["tcrit"], params["ddf"], params["s2max"],
            params["tw"], params["tu"], params["se"], params["s3max"],
            params["tc"], Sn, S2, S3, Sc1, Sc2)
        if t >= warmup:
            qs.append(q)
    return torch.stack(qs)

def nse(q, obs):
    obs = obs[WARMUP:]
    return 1.0 - ((q - obs) ** 2).sum(0) / ((obs - obs.mean(0)) ** 2).sum(0)

# ---- baseline: M3 optimal vs M4-with-zero-interception ----
# build m3 dict directly
m3p = {}
m4_to_m3 = {"tcrit":0,"ddf":1,"s2max":2,"tw":3,"tu":6,"se":7,"s3max":8,"tc":9}
for k, i in m4_to_m3.items():
    m3p[k] = p4[:, i].clone()

obs = y[WARMUP:]
q3 = run_m3(m3p)
nse3 = nse(q3, y)
# M4 with interception off (alpha=0 -> I=0 since frac_pos=max(cos,0)... alpha=0 still intercepts!)
# Use lambda-style off: set alpha such that frac<=0 all year -> alpha=0,is_time makes cos phase irrelevant; alpha=0 -> frac=cos(rad) which is >0 half year.
# TRUE off: alpha=0 & is_time chosen so cos<=0? impossible for all doy. Instead compare M3 vs M4(alpha=0,is_time=1) directly.
m4off = dict(P4); m4off["alpha"] = torch.zeros(B); m4off["is_time"] = torch.ones(B)
q4off = run_m4(m4off)
nse4off = nse(q4off, y)

rows = []
for i, bsid in enumerate([ids[j] for j in b]):
    rows.append({"basin": bsid, "NSE_M3": float(nse3[i]), "NSE_M4_alpha0": float(nse4off[i])})
with open(OUT / "baseline_m3_vs_m4off.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("baseline written")

# ============ C1: interception-only ceiling grid (alpha x is_time) ============
import itertools
alphas = [round(a, 3) for a in torch.linspace(0, 1, 21).tolist()]
is_times = [int(v) for v in torch.linspace(1, 365, 13).tolist()]
best = {i: {"nse": float(nse3[i]), "alpha": None, "is_time": None} for i in range(B)}
grid_rows = []
for a, it in itertools.product(alphas, is_times):
    pp = dict(P4); pp["alpha"] = torch.full((B,), a); pp["is_time"] = torch.full((B,), float(it))
    q = run_m4(pp)
    n = nse(q, y)
    for i in range(B):
        if n[i] > best[i]["nse"]:
            best[i] = {"nse": float(n[i]), "alpha": a, "is_time": it}
    grid_rows.append({"alpha": a, "is_time": it} | {f"nse_b{i}": float(n[i]) for i in range(B)})
with open(OUT / "c1_grid.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=grid_rows[0].keys()); w.writeheader(); w.writerows(grid_rows)
c1 = []
for i, bsid in enumerate([ids[j] for j in b]):
    c1.append({"basin": bsid, "NSE_M3": float(nse3[i]), "NSE_M4_dPL": float(nse4off[i]),
               "NSE_M4_ceiling": best[i]["nse"], "alpha_opt": best[i]["alpha"],
               "is_time_opt": best[i]["is_time"]})
with open(OUT / "c1_ceiling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=c1[0].keys()); w.writeheader(); w.writerows(c1)
print("C1 done:", c1)

# ============ C2: joint ceiling (alpha, is_time, s2max, tw, se) random search ============
torch.manual_seed(0)
rng = torch.Generator().manual_seed(0)
bestj = [dict(best[i]) for i in range(B)]
N_SAMP = 800
for s in range(N_SAMP):
    pp = dict(P4)
    pp["alpha"] = torch.rand(B, generator=rng)
    pp["is_time"] = 1 + 364 * torch.rand(B, generator=rng)
    pp["s2max"] = 1 + 1999 * torch.rand(B, generator=rng)
    pp["tw"] = torch.rand(B, generator=rng)
    pp["se"] = 0.05 + 0.9 * torch.rand(B, generator=rng)
    q = run_m4(pp)
    n = nse(q, y)
    for i in range(B):
        if n[i] > bestj[i]["nse"]:
            bestj[i] = {"nse": float(n[i]), "alpha": float(pp["alpha"][i]),
                        "is_time": float(pp["is_time"][i]), "s2max": float(pp["s2max"][i]),
                        "tw": float(pp["tw"][i]), "se": float(pp["se"][i])}
c2 = []
for i, bsid in enumerate([ids[j] for j in b]):
    c2.append({"basin": bsid, "NSE_joint_ceiling": bestj[i]["nse"],
               **{k: bestj[i][k] for k in ("alpha","is_time","s2max","tw","se")}})
with open(OUT / "c2_joint_ceiling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=c2[0].keys()); w.writeheader(); w.writerows(c2)
print("C2 done:", c2)

# ============ E1/E2: softplus dead zone + min-cap frequency at C1 optimum ============
e_rows = []
for i, bsid in enumerate([ids[j] for j in b]):
    pp = dict(P4); pp["alpha"] = torch.full((B,), best[i]["alpha"])
    pp["is_time"] = torch.full((B,), float(best[i]["is_time"]))
    import math
    rad = 2 * math.pi * (DOY[:, i] - pp["is_time"][i]) / 365.25
    frac = pp["alpha"][i] + (1 - pp["alpha"][i]) * torch.cos(rad)
    dead = (frac < 0).float().mean().item()
    rain = P[:, i] > 0.1
    dead_rain = ((frac < 0) & rain).float().sum().item() / max(rain.sum().item(), 1)
    frac_pos = torch.nn.functional.softplus(frac * 50.0) / 50.0
    cap = (frac_pos > 1.0).float().mean().item()
    e_rows.append({"basin": bsid, "dead_frac_days": round(dead, 4),
                   "dead_frac_raindays": round(dead_rain, 4),
                   "min_cap_frac_days": round(cap, 4)})
with open(OUT / "e1e2_dead_cap.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=e_rows[0].keys()); w.writeheader(); w.writerows(e_rows)
print("E1/E2 done:", e_rows)

# ============ D3: flow-quantile decomposition (M3 vs M4@C1opt) ============
pp = dict(P4); pp["alpha"] = torch.full((B,), best[0]["alpha"])
pp["is_time"] = torch.full((B,), float(best[0]["is_time"]))
q4c = run_m4(pp)
d3_rows = []
for i, bsid in enumerate([ids[j] for j in b]):
    o = y[WARMUP:, i]
    qs_ = torch.stack([q3[:, i], q4c[:, i]])
    edges = torch.quantile(o, torch.tensor([0.2, 0.4, 0.6, 0.8]))
    segs = [(0, edges[0]), (edges[0], edges[1]), (edges[1], edges[2]),
            (edges[2], edges[3]), (edges[3], float("inf"))]
    for si, (lo, hi) in enumerate(segs):
        mask = (o >= lo) & (o < hi)
        if mask.sum() < 5:
            continue
        for name, qq in (("M3", q3[:, i]), ("M4c", q4c[:, i])):
            bias = (qq[mask] - o[mask]).mean().item()
            rmse = ((qq[mask] - o[mask]) ** 2).mean().sqrt().item()
            d3_rows.append({"basin": bsid, "seg": f"Q{si*20}-{si*20+20}", "model": name,
                            "bias": round(bias, 3), "rmse": round(rmse, 3),
                            "n": int(mask.sum())})
with open(OUT / "d3_quantile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=d3_rows[0].keys()); w.writeheader(); w.writerows(d3_rows)
print("D3 done")

# ============ D4: event-scale I/P_r by daily P bin ============
# recompute I with C1-opt params per basin
d4_rows = []
for i, bsid in enumerate([ids[j] for j in b]):
    pp = dict(P4); pp["alpha"] = torch.full((B,), best[i]["alpha"])
    pp["is_time"] = torch.full((B,), float(best[i]["is_time"]))
    # accumulate I and Pr along the scored window
    S1 = torch.full((B,), 1e-6); S2 = torch.full((B,), 1e-6)
    Sc1 = torch.full((B,), 1e-6); Sc2 = torch.full((B,), 1e-6); Sn = torch.full((B,), 1e-6)
    Iacc = torch.zeros(B); Pracc = torch.zeros(B)
    bins = [(0,2),(2,5),(5,10),(10,25),(25,1e9)]
    bin_I = {k: 0.0 for k in range(5)}; bin_P = {k: 0.0 for k in range(5)}
    from dmotpy.models.core.mopex4 import interception_4
    for t in range(WARMUP):
        q, et, S1, S2, Sc1, Sc2, Sn = mopex4_step(
            P[t], T[t], PET[t], pp["tcrit"], pp["ddf"], pp["s2max"], pp["tw"],
            pp["alpha"], pp["is_time"], pp["tu"], pp["se"], pp["s3max"], pp["tc"],
            S1, S2, Sc1, Sc2, Sn, doy=DOY[t])
    for t in range(WARMUP, WARMUP + SCORED):
        q, et, S1, S2, Sc1, Sc2, Sn = mopex4_step(
            P[t], T[t], PET[t], pp["tcrit"], pp["ddf"], pp["s2max"], pp["tw"],
            pp["alpha"], pp["is_time"], pp["tu"], pp["se"], pp["s3max"], pp["tc"],
            S1, S2, Sc1, Sc2, Sn, doy=DOY[t])
        from dmotpy.models.flux.mopex import mopex_rainfall_1
        pr = float(mopex_rainfall_1(P[t, i:i+1], T[t, i:i+1], pp["tcrit"][i:i+1]))
        ipot = float(interception_4(pr * torch.ones(1), DOY[t, i:i+1],
                                    pp["alpha"][i:i+1], pp["is_time"][i:i+1]))
        I = min(ipot, float(PET[t, i]))
        pv = float(P[t, i])
        for bi, (lo, hi) in enumerate(bins):
            if lo <= pv < hi:
                bin_I[bi] += I; bin_P[bi] += pv
                break
    d4_rows.append({"basin": bsid} | {f"I_over_Pr_b{bi}": round(bin_I[bi]/max(bin_P[bi],1e-9), 4)
                                      for bi in range(5)} | {
                   f"I_share_b{bi}": round(bin_I[bi]/max(sum(bin_I.values()),1e-9), 4)
                                      for bi in range(5)})
with open(OUT / "d4_event.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=d4_rows[0].keys()); w.writeheader(); w.writerows(d4_rows)
print("D4 done:", d4_rows)

# ============ D1: precipitation-scaling equivalence (M3 with Pr*(1-r)) ============
# r = annual I/P_r from C1-opt; compare M3-scaled vs M4-C1opt Q series
d1_rows = []
for i, bsid in enumerate([ids[j] for j in b]):
    pp = dict(P4); pp["alpha"] = torch.full((B,), best[i]["alpha"])
    pp["is_time"] = torch.full((B,), float(best[i]["is_time"]))
    # annual I/P_r at optimum (approx from grid: reuse bin sums) -> compute quickly
    from dmotpy.models.flux.mopex import mopex_rainfall_1
    S1 = torch.full((B,), 1e-6); S2 = torch.full((B,), 1e-6)
    Sc1 = torch.full((B,), 1e-6); Sc2 = torch.full((B,), 1e-6); Sn = torch.full((B,), 1e-6)
    for t in range(WARMUP):
        q, et, S1, S2, Sc1, Sc2, Sn = mopex4_step(
            P[t], T[t], PET[t], pp["tcrit"], pp["ddf"], pp["s2max"], pp["tw"],
            pp["alpha"], pp["is_time"], pp["tu"], pp["se"], pp["s3max"], pp["tc"],
            S1, S2, Sc1, Sc2, Sn, doy=DOY[t])
    I_sum = 0.0; Pr_sum = 0.0
    for t in range(WARMUP, WARMUP + SCORED):
        q, et, S1, S2, Sc1, Sc2, Sn = mopex4_step(
            P[t], T[t], PET[t], pp["tcrit"], pp["ddf"], pp["s2max"], pp["tw"],
            pp["alpha"], pp["is_time"], pp["tu"], pp["se"], pp["s3max"], pp["tc"],
            S1, S2, Sc1, Sc2, Sn, doy=DOY[t])
        pr = float(mopex_rainfall_1(P[t, i:i+1], T[t, i:i+1], pp["tcrit"][i:i+1]))
        ipot = float(interception_4(pr*torch.ones(1), DOY[t, i:i+1], pp["alpha"][i:i+1], pp["is_time"][i:i+1]))
        I_sum += min(ipot, float(PET[t, i])); Pr_sum += pr
    r = I_sum / max(Pr_sum, 1e-9)
    # M3 with scaled rainfall: P_scaled = Pr*(1-r) + Ps (snow unchanged)
    Sn = torch.full((B,), 1e-6); S2 = torch.full((B,), 1e-6)
    S3 = torch.full((B,), 1e-6); Sc1 = torch.full((B,), 1e-6); Sc2 = torch.full((B,), 1e-6)
    q3s = []
    from dmotpy.models.flux.mopex import mopex_snowfall_1
    for t in range(WARMUP + SCORED):
        ps = float(mopex_snowfall_1(P[t, i:i+1], T[t, i:i+1], m3p["tcrit"][i:i+1]))
        pr = float(P[t, i]) - ps
        Pmod = (ps + pr * (1 - r)) * torch.ones(1)
        q, et, Sn, S2, S3, Sc1, Sc2 = mopex3_step(
            Pmod, T[t, i:i+1], PET[t, i:i+1], m3p["tcrit"][i:i+1], m3p["ddf"][i:i+1],
            m3p["s2max"][i:i+1], m3p["tw"][i:i+1], m3p["tu"][i:i+1], m3p["se"][i:i+1],
            m3p["s3max"][i:i+1], m3p["tc"][i:i+1], Sn, S2, S3, Sc1, Sc2)
        if t >= WARMUP:
            q3s.append(q)
    q3s = torch.stack(q3s).squeeze(-1)
    q4i = q4c[:, i]
    o = y[WARMUP:, i]
    corr = torch.corrcoef(torch.stack([q3s, q4i]))[0, 1].item()
    rmse = ((q3s - q4i) ** 2).mean().sqrt().item()
    nse_s = float(nse(q3s.unsqueeze(-1), y[:, i:i+1])[0])
    nse_4 = float(nse(q4i.unsqueeze(-1), y[:, i:i+1])[0])
    d1_rows.append({"basin": bsid, "r_I_over_Pr": round(r, 4),
                    "NSE_M3_scaled": round(nse_s, 4), "NSE_M4_C1opt": round(nse_4, 4),
                    "Q_corr": round(corr, 4), "Q_RMSE": round(rmse, 4)})
with open(OUT / "d1_scaling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=d1_rows[0].keys()); w.writeheader(); w.writerows(d1_rows)
print("D1 done:", d1_rows)
print("ALL DIAGNOSTICS DONE")
