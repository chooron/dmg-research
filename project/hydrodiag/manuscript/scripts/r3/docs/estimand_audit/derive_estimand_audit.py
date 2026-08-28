#!/usr/bin/env python3
"""R3/3.3 estimand audit - deterministic derivation from canonical basin-level
sources ONLY (results/r3_misspec_analysis_v1, manuscript/results/R1).
No training; no modification of canonical results; new derived artifacts are
written to this directory only.

Derives (test-period primary; train also emitted):
  D              = KGE(CN_refit) - KGE(Base_no-refit)      [reference gap]
  G_base         = KGE(Base_refit) - KGE(Base_no-refit)    [existing]
  G_tgd2         = KGE(TGD2) - KGE(Base_refit)             [existing]
  G_TGD_ko       = KGE(TGD2) - KGE(Base_no-refit)          [raw TGD from knockout]
  F_close        = G_base / D                              [existing]
  F_tgd2         = G_tgd2 / (KGE(CN) - KGE(Base_refit))    [existing current F_TGD]
  F_TGD_star     = G_TGD_ko / D                            [candidate common-reference]
  denom rule: D > 1e-6 and denom_tgd2 > 1e-6, unclipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = Path("/home/jingxin/code/dmg-research/project/hydrodiag/results")
R1 = Path("/home/jingxin/code/dmg-research/project/hydrodiag/manuscript/results/R1")
OUT = HERE

DENOM_TOL = 1e-6
SEEDS = (42, 123, 2026)
BOOT_SEED = 20260730
N_BOOT = 2000

bt = pd.read_csv(RES / "r3_misspec_analysis_v1/posthoc_basin_table.csv")
bt["basin_id"] = bt["basin_id"].astype(str).str.zfill(8)
strata = pd.read_csv(R1 / "r1_snow_attributes.csv")
strata["basin_id"] = strata["basin_id"].astype(str).str.zfill(8)
bt = bt.merge(strata[["basin_id", "snow_stratum"]], on="basin_id", how="left")

tc = pd.read_csv(RES / "r3_misspec_analysis_v1/posthoc_theta_cost.csv")
tc["basin_id"] = tc["basin_id"].astype(str).str.zfill(8)
sc = pd.read_csv(RES / "r3_misspec_analysis_v1/posthoc_state_cost.csv")
sc["basin_id"] = sc["basin_id"].astype(str).str.zfill(8)

rows = []
for _, r in bt.iterrows():
    k_ref = r["kge_base_no_refit"]
    k_b = r["kge_base"]
    k_t = r["kge_tgd2"]
    k_c = r["kge_cn"]
    D = k_c - k_ref
    denom_tgd2 = k_c - k_b
    g_ko = k_t - k_ref
    g_b = k_b - k_ref
    g_t2 = k_t - k_b
    f_close = g_b / D if D > DENOM_TOL else np.nan
    f_star = g_ko / D if D > DENOM_TOL else np.nan
    f_tgd2 = g_t2 / denom_tgd2 if denom_tgd2 > DENOM_TOL else np.nan
    rows.append(
        {
            "basin_id": r["basin_id"],
            "paradigm": r["paradigm"],
            "seed": r["seed"],
            "period": r["period"],
            "kge_base_no_refit": k_ref,
            "kge_base": k_b,
            "kge_tgd2": k_t,
            "kge_cn": k_c,
            "D": D,
            "denom_tgd2": denom_tgd2,
            "G_base": g_b,
            "G_tgd2": g_t2,
            "G_TGD_ko": g_ko,
            "F_close_canonical": r["F_close"],
            "F_tgd2_canonical": r["F_tgd2"],
            "F_TGD_star": f_star,
            "F_close": f_close,
            "F_tgd2": f_tgd2,
            "frac_snow": r["frac_snow"],
            "snow_stratum": r["snow_stratum"],
        }
    )
df = pd.DataFrame(rows)
df.to_csv(OUT / "audit_estimands_basin.csv", index=False)


def boot_ci_median(v: np.ndarray, n_boot=N_BOOT, seed=BOOT_SEED):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(v), len(v))
        draws[b] = np.median(v[idx])
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 5 or x[v].std() == 0 or y[v].std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[v]))
    ry = np.argsort(np.argsort(y[v]))
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    v = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if v.sum() < 8:
        return float("nan")

    def rank(u):
        return np.argsort(np.argsort(u[v])).astype(float) + 1.0

    def resid(u, c):
        A = np.vstack([c, np.ones_like(c)]).T
        coef, *_ = np.linalg.lstsq(A, u, rcond=None)
        return u - A @ coef

    rx, ry, rz = rank(x), rank(y), rank(z)
    ex, ey = resid(rx, rz), resid(ry, rz)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def quant(v, q):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if len(v) else np.nan


def rec_stats(sub):
    D = sub["D"].to_numpy()
    rec = {
        "n": int(len(sub)),
        "D": {
            "le0": int((D <= 0).sum()),
            "gt0_le1e-6": int(((D > 0) & (D <= 1e-6)).sum()),
            "gt1e-6_le1e-4": int(((D > 1e-6) & (D <= 1e-4)).sum()),
            "gt1e-4_le1e-3": int(((D > 1e-4) & (D <= 1e-3)).sum()),
            "gt1e-3": int((D > 1e-3).sum()),
            "median": quant(D, 0.5),
            "q25": quant(D, 0.25),
            "q75": quant(D, 0.75),
            "spearman_frac_snow": spearman(sub["frac_snow"].to_numpy(), D),
        },
    }
    den2 = sub["denom_tgd2"].to_numpy()
    rec["denom_tgd2"] = {
        "le0": int((den2 <= 0).sum()),
        "gt0_le1e-6": int(((den2 > 0) & (den2 <= 1e-6)).sum()),
        "gt1e-6_le1e-4": int(((den2 > 1e-6) & (den2 <= 1e-4)).sum()),
        "median": quant(den2, 0.5),
        "q25": quant(den2, 0.25),
        "q75": quant(den2, 0.75),
    }
    for col in ("G_base", "G_tgd2", "G_TGD_ko"):
        v = sub[col].to_numpy()
        rec[col] = {
            "median": quant(v, 0.5),
            "q25": quant(v, 0.25),
            "q75": quant(v, 0.75),
            "ci": list(boot_ci_median(v)),
            "le0": int((v <= 0).sum()),
            "gt0": int((v > 0).sum()),
        }
    for col in ("F_close", "F_tgd2", "F_TGD_star"):
        v = sub[col].to_numpy()
        fv = v[np.isfinite(v)]
        rec[col] = {
            "n_valid": int(len(fv)),
            "n_excluded": int(len(v) - len(fv)),
            "median": quant(fv, 0.5),
            "q25": quant(fv, 0.25),
            "q75": quant(fv, 0.75),
            "ci": list(boot_ci_median(fv)),
            "lt0": int((fv < 0).sum()),
            "ge0_le1": int(((fv >= 0) & (fv <= 1)).sum()),
            "gt1": int((fv > 1).sum()),
            "gt2": int((fv > 2).sum()),
            "abs_gt5": int((np.abs(fv) > 5).sum()),
        }
    kt = sub["kge_tgd2"].to_numpy()
    kb = sub["kge_base"].to_numpy()
    kr = sub["kge_base_no_refit"].to_numpy()
    kc = sub["kge_cn"].to_numpy()
    rec["pairs"] = {
        "TGD_gt_CN": int((kt > kc).sum()),
        "TGD_lt_no_refit": int((kt < kr).sum()),
        "TGD_lt_Base_refit": int((kt < kb).sum()),
        "n_CN_kge_ge_0p99": int((kc >= 0.99).sum()),
        "n_CN_kge_ge_0p995": int((kc >= 0.995).sum()),
        "n_CN_kge_ge_0p999": int((kc >= 0.999).sum()),
        "n_no_refit_kge_ge_0p99": int((kr >= 0.99).sum()),
    }
    fc = sub["F_close"].to_numpy()
    fs = sub["F_TGD_star"].to_numpy()
    both = np.isfinite(fc) & np.isfinite(fs)
    rec["F_star_vs_F_close"] = {
        "n_both_valid": int(both.sum()),
        "star_lt_close": int((fs[both] < fc[both]).sum()),
        "star_gt_close": int((fs[both] > fc[both]).sum()),
    }
    fsnow = sub["frac_snow"].to_numpy()
    mk1 = np.isfinite(fc)
    mk2 = np.isfinite(fs)
    rec["snow"] = {
        "spearman_G_base": spearman(fsnow, sub["G_base"].to_numpy()),
        "spearman_G_TGD_ko": spearman(fsnow, sub["G_TGD_ko"].to_numpy()),
        "spearman_F_close": spearman(fsnow[mk1], fc[mk1]),
        "spearman_F_TGD_star": spearman(fsnow[mk2], fs[mk2]),
        "spearman_F_tgd2": spearman(fsnow[np.isfinite(sub["F_tgd2"].to_numpy())], sub["F_tgd2"].to_numpy()[np.isfinite(sub["F_tgd2"].to_numpy())]),
    }
    strata_rows = {}
    for st in ("S1", "S2", "S3", "S4", "S5"):
        m = sub["snow_stratum"] == st
        if m.sum() == 0:
            continue
        fcv = sub.loc[m, "F_close"].to_numpy()
        fsv = sub.loc[m, "F_TGD_star"].to_numpy()
        gbv = sub.loc[m, "G_base"].to_numpy()
        gk = sub.loc[m, "G_TGD_ko"].to_numpy()
        dv = sub.loc[m, "D"].to_numpy()
        strata_rows[st] = {
            "n": int(m.sum()),
            "n_no_snow_or_low": int((sub.loc[m, "frac_snow"] <= 0.05).sum()),
            "F_close_valid": int(np.isfinite(fcv).sum()),
            "F_close_valid_rate": float(np.isfinite(fcv).mean()),
            "F_star_valid": int(np.isfinite(fsv).sum()),
            "F_star_valid_rate": float(np.isfinite(fsv).mean()),
            "F_close_median": quant(fcv, 0.5),
            "F_star_median": quant(fsv, 0.5),
            "G_base_median": quant(gbv, 0.5),
            "G_TGD_ko_median": quant(gk, 0.5),
            "D_median": quant(dv, 0.5),
        }
    rec["strata"] = strata_rows
    return rec


summary = {}
for reg in ("IC", "dPL"):
    for period in ("train", "test"):
        sub = df[(df["paradigm"] == reg) & (df["period"] == period)]
        if reg == "dPL":
            per_seed = {}
            for s in SEEDS:
                per_seed[str(s)] = rec_stats(sub[sub["seed"] == s])
            g = sub.groupby("basin_id")
            sm = g.agg(
                D=("D", "median"),
                denom_tgd2=("denom_tgd2", "median"),
                G_base=("G_base", "median"),
                G_tgd2=("G_tgd2", "median"),
                G_TGD_ko=("G_TGD_ko", "median"),
                kge_base=("kge_base", "median"),
                kge_tgd2=("kge_tgd2", "median"),
                kge_cn=("kge_cn", "median"),
                kge_base_no_refit=("kge_base_no_refit", "first"),
                F_close=("F_close", "median"),
                F_tgd2=("F_tgd2", "median"),
                F_TGD_star=("F_TGD_star", "median"),
                frac_snow=("frac_snow", "first"),
                snow_stratum=("snow_stratum", "first"),
            ).reset_index()
            summary[f"{reg}_{period}"] = rec_stats(sm)
            summary[f"{reg}_{period}"]["per_seed"] = per_seed
        else:
            summary[f"{reg}_{period}"] = rec_stats(sub)

# ---- association audit (test) ----
def cost_series(paradigm, seed, structure, col="C_theta_primary"):
    m = (tc["paradigm"] == paradigm) & (tc["structure"] == structure)
    if seed is None:
        m &= tc["seed"].isna()
    else:
        m &= tc["seed"] == seed
    return tc.loc[m, ["basin_id", col]].set_index("basin_id")[col]


def state_cost_series(paradigm, seed, structure, col="C_state_primary"):
    m = (sc["paradigm"] == paradigm) & (sc["structure"] == structure)
    if seed is None:
        m &= sc["seed"].isna()
    else:
        m &= sc["seed"] == seed
    return sc.loc[m, ["basin_id", col]].set_index("basin_id")[col]


assoc = {}
for reg in ("IC", "dPL"):
    for seed in ([None] if reg == "IC" else list(SEEDS)):
        tag = f"{reg}" + ("" if seed is None else f"_s{seed}")
        flag = (df["paradigm"] == reg) & (df["period"] == "test")
        flag &= df["seed"].isna() if seed is None else (df["seed"] == seed)
        sub = df[flag].set_index("basin_id")
        ct_b = cost_series(reg, seed, "Base")
        cs_b = state_cost_series(reg, seed, "Base")
        ct_t = cost_series(reg, seed, "TGD2")
        cs_t = state_cost_series(reg, seed, "TGD2")
        idx = ct_b.index
        gb = sub["G_base"].reindex(idx).to_numpy()
        gko = sub["G_TGD_ko"].reindex(idx).to_numpy()
        gt2 = sub["G_tgd2"].reindex(idx).to_numpy()
        ctb = ct_b.to_numpy()
        csb = cs_b.to_numpy()
        ctt = ct_t.reindex(idx).to_numpy()
        cst = cs_t.reindex(idx).to_numpy()
        fs = sub["frac_snow"].reindex(idx).to_numpy()
        assoc[tag] = {
            "G_base_vs_C_theta": [spearman(gb, ctb), partial_spearman(gb, ctb, fs)],
            "G_base_vs_C_state": [spearman(gb, csb), partial_spearman(gb, csb, fs)],
            "G_TGD_ko_vs_C_theta_TGD": [spearman(gko, ctt), partial_spearman(gko, ctt, fs)],
            "G_TGD_ko_vs_C_state_TGD": [spearman(gko, cst), partial_spearman(gko, cst, fs)],
            "G_tgd2_vs_dC_theta": [spearman(gt2, ctb - ctt), partial_spearman(gt2, ctb - ctt, fs)],
            "G_tgd2_vs_dC_state": [spearman(gt2, csb - cst), partial_spearman(gt2, csb - cst, fs)],
            "C_theta_Base_vs_frac_snow": spearman(fs, ctb),
            "C_state_Base_vs_frac_snow": spearman(fs, csb),
            "strata_raw_G_base_C_theta": {},
            "strata_raw_G_base_C_state": {},
            "strata_raw_G_TGD_ko_C_state_TGD": {},
        }
        for st in ("S1", "S2", "S3", "S4", "S5"):
            m = sub["snow_stratum"] == st
            assoc[tag]["strata_raw_G_base_C_theta"][st] = spearman(gb[m], ct_b.reindex(idx)[m].to_numpy())
            assoc[tag]["strata_raw_G_base_C_state"][st] = spearman(gb[m], cs_b.reindex(idx)[m].to_numpy())
            assoc[tag]["strata_raw_G_TGD_ko_C_state_TGD"][st] = spearman(gko[m], cs_t.reindex(idx)[m].to_numpy())

summary["assoc"] = assoc
with open(OUT / "audit_summary.json", "w") as fh:
    json.dump(summary, fh, indent=1, sort_keys=True, default=float)
print(json.dumps({k: v for k, v in summary.items() if k != "assoc"}, indent=1, sort_keys=True, default=float))
print("===== ASSOC =====")
print(json.dumps(summary["assoc"], indent=1, sort_keys=True, default=float))