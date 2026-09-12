import csv, sys, math
from pathlib import Path
import torch
REPO = Path("/home/jingxin/code/dmg-research")
BENCHMARK = REPO / "project" / "benchmark"
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"), str(BENCHMARK / "scripts" / "diagnostics")]
import audit_mopex34_root_cause as A
from dmotpy.models.core.mopex3 import mopex3_step
from dmotpy.models.core.mopex4 import mopex4_step, interception_4
from dmotpy.models.flux.mopex import mopex_rainfall_1, mopex_snowfall_1

OUT = Path("/tmp/m3m4_diag_out")
WARMUP, SCORED = 365, 365
ids, x, y, b = A.load_context()
x = x[A.START:A.START+730]; y = y[A.START:A.START+730]
P, T, PET = x[:,:,0], x[:,:,1], x[:,:,2]; DOY = x[:,:,3]
net3, net4, _ = A.mapped_m3_network()
attrs = A.CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cpu", method="zscore")[b]
with torch.no_grad():
    p4 = A.norm_to_phys(net4(attrs), "mopex4")
P4 = {n: p4[:, i].clone() for i, n in enumerate(["tcrit","ddf","s2max","tw","alpha","is_time","tu","se","s3max","tc"])}
m4_to_m3 = {"tcrit":0,"ddf":1,"s2max":2,"tw":3,"tu":6,"se":7,"s3max":8,"tc":9}
m3p = {k: p4[:, i].clone() for k, i in m4_to_m3.items()}
# C1 最优
c1 = {r["basin"].strip(): r for r in csv.DictReader(open(OUT/"c1_ceiling.csv"))}
bsids = [ids[j] for j in b]
rows = []
for i, bsid in enumerate(bsids):
    pp = dict(P4)
    pp["alpha"] = torch.full((4,), float(c1[str(bsid)]["alpha_opt"]))
    pp["is_time"] = torch.full((4,), float(c1[str(bsid)]["is_time_opt"]))
    # warmup + compute I/Pr
    S1 = torch.full((4,), 1e-6); S2 = torch.full((4,), 1e-6)
    Sc1 = torch.full((4,), 1e-6); Sc2 = torch.full((4,), 1e-6); Sn = torch.full((4,), 1e-6)
    for t in range(WARMUP):
        q, et, S1, S2, Sc1, Sc2, Sn = mopex4_step(P[t], T[t], PET[t], pp["tcrit"], pp["ddf"], pp["s2max"], pp["tw"], pp["alpha"], pp["is_time"], pp["tu"], pp["se"], pp["s3max"], pp["tc"], S1, S2, Sc1, Sc2, Sn, doy=DOY[t])
    I_sum, Pr_sum = 0.0, 0.0
    q4s = []
    for t in range(WARMUP, WARMUP+SCORED):
        q, et, S1, S2, Sc1, Sc2, Sn = mopex4_step(P[t], T[t], PET[t], pp["tcrit"], pp["ddf"], pp["s2max"], pp["tw"], pp["alpha"], pp["is_time"], pp["tu"], pp["se"], pp["s3max"], pp["tc"], S1, S2, Sc1, Sc2, Sn, doy=DOY[t])
        q4s.append(q[i])
        pr = float(mopex_rainfall_1(P[t,i:i+1], T[t,i:i+1], pp["tcrit"][i:i+1]))
        ipot = float(interception_4(pr*torch.ones(1), DOY[t,i:i+1], pp["alpha"][i:i+1], pp["is_time"][i:i+1]))
        I_sum += min(ipot, float(PET[t,i])); Pr_sum += pr
    r = I_sum/max(Pr_sum, 1e-9)
    q4i = torch.stack(q4s).squeeze(-1)
    # M3 scaled
    Sn = torch.full((4,), 1e-6); S2 = torch.full((4,), 1e-6)
    S3 = torch.full((4,), 1e-6); Sc1 = torch.full((4,), 1e-6); Sc2 = torch.full((4,), 1e-6)
    q3s = []
    for t in range(WARMUP+SCORED):
        ps = float(mopex_snowfall_1(P[t,i:i+1], T[t,i:i+1], m3p["tcrit"][i:i+1]))
        pr = float(P[t,i]) - ps
        Pmod = (ps + pr*(1-r))*torch.ones(1)
        q, et, Sn, S2, S3, Sc1, Sc2 = mopex3_step(Pmod, T[t,i:i+1], PET[t,i:i+1], m3p["tcrit"][i:i+1], m3p["ddf"][i:i+1], m3p["s2max"][i:i+1], m3p["tw"][i:i+1], m3p["tu"][i:i+1], m3p["se"][i:i+1], m3p["s3max"][i:i+1], m3p["tc"][i:i+1], Sn, S2, S3, Sc1, Sc2)
        if t >= WARMUP: q3s.append(q)
    q3s = torch.stack(q3s)[:, i]
    o = y[WARMUP:, i]
    def nse(qq): return 1.0 - ((qq-o)**2).sum()/((o-o.mean())**2).sum()
    corr = torch.corrcoef(torch.stack([q3s, q4i]))[0,1].item()
    rmse = ((q3s-q4i)**2).mean().sqrt().item()
    rows.append({"basin": bsid, "r_I_over_Pr": round(r,4), "NSE_M3_scaled": round(float(nse(q3s)),4), "NSE_M4_C1opt": round(float(nse(q4i)),4), "Q_corr": round(corr,4), "Q_RMSE": round(rmse,4)})
with open(OUT/"d1_scaling.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(rows)
