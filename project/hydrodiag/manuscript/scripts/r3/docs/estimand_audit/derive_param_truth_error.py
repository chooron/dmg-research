#!/usr/bin/env python3
"""R3/3.3 F6 parameter truth-error correction - deterministic derivation.

Read-only with respect to canonical results. Consumes:
  results/r3_misspec_analysis_v1/paired_parameters.csv   (per-param e, e_cn)
  results/r3_misspec_analysis_v1/posthoc_basin_table.csv (outlet KGEs/G)
  results/r3_synthetic_truth_v1/theta_star.npz           (truth, INDEPENDENT
                                                          verification only)
  results/r3_synthetic_truth_v1/gstar_manifest.json      (bounds, INDEPENDENT
                                                          verification only)
  results/r3_gate_*_xaj*/raw|best_parameters_physical   (INDEPENDENT verify)
  manuscript/results/R1/r1_snow_attributes.csv          (S1-S5)

New canonical parameter-side fields (per basin x regime x seed):
  E_param_M      = median_{p in 15 shared} |e_{M,p}|            (truth-relative)
  E_param_excess_M = E_param_M - E_param_CN                     (correct-CN
                   adjusted; negatives retained)
  C15_M          = median_{p in 15 shared} |e_{M,p} - e_{CN,p}| (CN-refit
                   parameter SEPARATION; old C_theta15 semantics, full-15)

Writes (this directory only):
  param_truth_error_basin.csv
  param_truth_error_summary.json
No training / calibration / forward replay; canonical files unmodified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = Path("/home/jingxin/code/dmg-research/project/hydrodiag")
RES = PROJ / "results"
R1 = PROJ / "manuscript/results/R1"
OUT = HERE

COMMON_XAJ = [
    "xaj_k", "xaj_b", "xaj_im", "xaj_um", "xaj_lm", "xaj_dm", "xaj_c",
    "xaj_sm", "xaj_ex", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg",
    "xaj_a", "xaj_theta",
]
SEEDS = (42, 123, 2026)
BOOT_SEED = 20260730
N_BOOT = 2000

# ---------------------------------------------------------------------------
# helpers (methods identical to posthoc_stats.py / posthoc_validation.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
pp = pd.read_csv(RES / "r3_misspec_analysis_v1/paired_parameters.csv")
pp["basin_id"] = pp["basin_id"].astype(str).str.zfill(8)
# groupby-safe seed key (IC rows have NaN seed; groupby would drop them)
pp["seed_key"] = pp["seed"].fillna(-1).astype(int)
bt = pd.read_csv(RES / "r3_misspec_analysis_v1/posthoc_basin_table.csv")
bt["basin_id"] = bt["basin_id"].astype(str).str.zfill(8)
strata = pd.read_csv(R1 / "r1_snow_attributes.csv")
strata["basin_id"] = strata["basin_id"].astype(str).str.zfill(8)

# --- validation 1: exactly the 15 COMMON_XAJ parameters -------------------
params = sorted(pp["parameter"].unique())
assert params == sorted(COMMON_XAJ), f"param set mismatch: {params}"
print(f"[v1] aggregated params = exactly the 15 COMMON_XAJ: {params == sorted(COMMON_XAJ)}")

# --- validation 2 (data): e_cn identical for Base and TGD2 rows -------------
chk = pp.pivot_table(index=["basin_id", "paradigm", "seed"],
                     columns="structure", values="e_cn", aggfunc="first")
maxdiff = float((chk["Base"] - chk["TGD2"]).abs().max())
print(f"[v2-data] e_cn Base-vs-TGD2 rows max abs diff = {maxdiff:.3e}")
assert maxdiff == 0.0

# ---------------------------------------------------------------------------
# per-basin aggregates
# ---------------------------------------------------------------------------
agg = (
    pp.groupby(["basin_id", "paradigm", "seed_key", "structure"])
    .agg(
        E_param=("e", lambda x: np.median(np.abs(x))),
        C15=("delta_e", lambda x: np.median(np.abs(x))),
    )
    .reset_index()
)
cn = (
    pp.groupby(["basin_id", "paradigm", "seed_key"])
    .agg(E_param_cn=("e_cn", lambda x: np.median(np.abs(x))))
    .reset_index()
)
wide = (
    agg.pivot(index=["basin_id", "paradigm", "seed_key"], columns="structure")
    .reset_index()
)
wide.columns = [f"{a}_{b}" if b else a for a, b in wide.columns]
wide = wide.rename(columns={
    "E_param_Base": "E_param_base",
    "E_param_TGD2": "E_param_tgd",
    "E_param_CN": "E_param_cn",
    "C15_Base": "C15_base",
    "C15_TGD2": "C15_tgd",
})
# E_param_cn recomputed directly from the shared e_cn column (CN-refit
# estimates; identical for Base and TGD2 rows - asserted in v2).
out = wide.merge(cn, on=["basin_id", "paradigm", "seed_key"])
out["seed"] = out["seed_key"].map(lambda k: np.nan if k == -1 else float(k))
out = out.drop(columns=["seed_key"])
out["E_param_excess_base"] = out["E_param_base"] - out["E_param_cn"]
out["E_param_excess_tgd"] = out["E_param_tgd"] - out["E_param_cn"]

# outlet recovery (G_base as stored; G_TGD_ko = kge_tgd2 - kge_base_no_refit)
# + strata
rec = bt[bt["period"] == "test"].copy()
rec["G_TGD_ko"] = rec["kge_tgd2"] - rec["kge_base_no_refit"]
rec = rec.merge(
    strata[["basin_id", "snow_stratum"]], on="basin_id", how="left"
)
out = out.merge(
    rec[["basin_id", "paradigm", "seed", "G_base", "G_TGD_ko", "frac_snow",
         "snow_stratum"]],
    on=["basin_id", "paradigm", "seed"], how="left",
)
out.to_csv(OUT / "param_truth_error_basin.csv", index=False)

# ---------------------------------------------------------------------------
# validation 3 (independent): recompute e from raw fitted physical params
# ---------------------------------------------------------------------------
def bounds_from_manifest():
    gm = json.loads((RES / "r3_synthetic_truth_v1/gstar_manifest.json").read_text())
    b = gm["parameter_bounds"]
    lo = np.array([b[p]["lower"] for p in COMMON_XAJ])
    hi = np.array([b[p]["upper"] for p in COMMON_XAJ])
    return lo, hi


def load_fit_theta(run_dir, kind, structure, basin_ids):
    if kind == "ic":
        raw = sorted((run_dir / "raw" / {"XAJ": "xaj", "XAJ_TGD2": "xaj_tgd2",
                                         "XAJ_CN": "xaj_cn"}[structure]).glob("*.json"))
        best = {}
        for f in raw:
            d = json.loads(f.read_text())
            b = str(d["basin_id"]).zfill(8)
            k = d.get("train_metrics", {}).get("kge", np.nan)
            if d.get("status") == "complete" and np.isfinite(k):
                best.setdefault(b, []).append((k, d["start"], d))
        rows = {}
        for b in basin_ids:
            cands = best.get(b, [])
            cands.sort(key=lambda x: (-x[0], x[1]))
            rows[b] = np.array(cands[0][2]["parameters"], dtype=float)
            names = cands[0][2]["parameter_names"]
        return rows, tuple(names)
    else:
        params = np.load(run_dir / "best_parameters_physical.npz")["params"]
        cfg = json.loads((run_dir / "config.json").read_text())
        return {b: params[i].astype(float) for i, b in enumerate(basin_ids)}, \
               tuple(cfg["parameter_names"])


truth = np.load(RES / "r3_synthetic_truth_v1/theta_star.npz")
theta_star = truth["parameters"]
tnames = [str(x) for x in truth["parameter_names"]]
tids = [str(b).zfill(8) for b in truth["basin_ids"]]
lo, hi = bounds_from_manifest()
tidx = {p: tnames.index(p) for p in COMMON_XAJ}


def e_from_fits(theta_map, names):
    idx = [names.index(p) for p in COMMON_XAJ]
    tidx_arr = [tidx[p] for p in COMMON_XAJ]
    e = {}
    for b, th in theta_map.items():
        e[b] = (th[idx] - theta_star[tids.index(b), tidx_arr]) / (hi - lo)
    return e


checks = []
# (a) M-structures: recompute e = (th_hat - theta*)/(U-L) and compare with
#     the 'e' column in paired_parameters.csv (paired_parameters has NO CN rows).
for structure, run, kind, mname in [
    ("XAJ", RES / "r3_misspec_ic_xaj_531_v1", "ic", "Base"),
    ("XAJ_TGD2", RES / "r3_misspec_ic_xaj_tgd2_531_v1", "ic", "TGD2"),
    ("XAJ", RES / "r3_misspec_dpl_xaj_seed_42", "dpl", "Base"),
    ("XAJ_TGD2", RES / "r3_misspec_dpl_xaj_tgd2_seed_42", "dpl", "TGD2"),
]:
    theta_map, names = load_fit_theta(run, kind, structure, tids)
    e_re = e_from_fits(theta_map, names)
    for b in tids[:3]:  # sampled basins (alphabetical head)
        pp_row = pp[(pp["basin_id"] == b) & (pp["structure"] == mname)
                    & (pp["paradigm"] == ("IC" if kind == "ic" else "dPL"))]
        pp_row = pp_row if kind == "ic" else pp_row[pp_row["seed"] == 42]
        assert len(pp_row) == 15, (structure, b, len(pp_row))
        e_col = pp_row.sort_values("parameter")["e"].to_numpy()
        e_man = np.array([e_re[b][COMMON_XAJ.index(p)]
                          for p in sorted(COMMON_XAJ)])
        d = float(np.abs(e_col - e_man).max())
        em = float(np.median(np.abs(e_col)))
        checks.append((f"{mname} (e)", kind, b, d, em))
# (b) CN-refit: recompute e_cn from CN gate fits and compare with the 'e_cn'
#     column (identical for Base and TGD2 rows; asserted above).
for structure, run, kind in [
    ("XAJ_CN", RES / "r3_gate_ic_xaj_cn_531_v1", "ic"),
    ("XAJ_CN", RES / "r3_gate_dpl_xaj_cn_seed_42", "dpl"),
]:
    theta_map, names = load_fit_theta(run, kind, structure, tids)
    e_re = e_from_fits(theta_map, names)
    for b in tids[:3]:
        pp_row = pp[(pp["basin_id"] == b) & (pp["structure"] == "Base")
                    & (pp["paradigm"] == ("IC" if kind == "ic" else "dPL"))]
        pp_row = pp_row if kind == "ic" else pp_row[pp_row["seed"] == 42]
        e_cn_col = pp_row.sort_values("parameter")["e_cn"].to_numpy()
        e_cn_man = np.array([e_re[b][COMMON_XAJ.index(p)]
                             for p in sorted(COMMON_XAJ)])
        d = float(np.abs(e_cn_col - e_cn_man).max())
        checks.append(("CN (e_cn)", kind, b, d, float(np.nan)))
for name, kind, b, d, em in checks:
    s_em = f"  E_param = {em:.6f}" if em == em else ""
    print(f"[v3] {name:9s} {kind:3s} {b}  max|csv - recomputed| = {d:.3e}{s_em}")
assert all(c[3] < 1e-9 for c in checks), "independent recomputation mismatch"
print("[v3] independent recomputation from raw fitted physical params: all matches (max diff < 1e-9)")

# ---------------------------------------------------------------------------
# population summaries
# ---------------------------------------------------------------------------
summary = {}
for reg in ("IC", "dPL"):
    sub = out[out["paradigm"] == reg]
    if reg == "IC":
        groups = {"IC": sub[sub["seed"].isna()]}
        per_seed = None
    else:
        groups = {f"dPL_s{s}": sub[sub["seed"] == s] for s in SEEDS}
        sm = sub.groupby("basin_id").agg(
            E_param_base=("E_param_base", "median"),
            E_param_tgd=("E_param_tgd", "median"),
            E_param_cn=("E_param_cn", "median"),
            E_param_excess_base=("E_param_excess_base", "median"),
            E_param_excess_tgd=("E_param_excess_tgd", "median"),
            C15_base=("C15_base", "median"),
            C15_tgd=("C15_tgd", "median"),
            G_base=("G_base", "median"),
            G_TGD_ko=("G_TGD_ko", "median"),
            frac_snow=("frac_snow", "first"),
            snow_stratum=("snow_stratum", "first"),
        ).reset_index()
        groups = {"dPL_seedmedian": sm}
        per_seed = {f"dPL_s{s}": sub[sub["seed"] == s] for s in SEEDS}
    for gname, g in groups.items():
        entry = {}
        for col in ("E_param_base", "E_param_tgd", "E_param_cn",
                    "E_param_excess_base", "E_param_excess_tgd",
                    "C15_base", "C15_tgd"):
            v = g[col].to_numpy()
            entry[col] = {
                "n": int(np.isfinite(v).sum()),
                "median": quant(v, 0.5),
                "q25": quant(v, 0.25),
                "q75": quant(v, 0.75),
                "ci": list(boot_ci_median(v)),
                "min": float(np.nanmin(v)) if np.isfinite(v).any() else np.nan,
                "max": float(np.nanmax(v)) if np.isfinite(v).any() else np.nan,
            }
        for col in ("E_param_excess_base", "E_param_excess_tgd"):
            v = g[col].to_numpy()
            entry[f"{col}_frac"] = {
                "lt0": float((v < 0).mean()),
                "eq0": float((v == 0).mean()),
                "gt0": float((v > 0).mean()),
            }
        # S1-S5
        entry["strata"] = {}
        for st in ("S1", "S2", "S3", "S4", "S5"):
            m = g["snow_stratum"] == st
            entry["strata"][st] = {c: quant(g.loc[m, c].to_numpy(), 0.5)
                                   for c in ("E_param_base", "E_param_tgd",
                                             "E_param_cn",
                                             "E_param_excess_base",
                                             "E_param_excess_tgd")}
            entry["strata"][st]["n"] = int(m.sum())
        # frac_snow descriptive
        fs = g["frac_snow"].to_numpy()
        entry["frac_snow_spearman"] = {
            c: spearman(fs, g[c].to_numpy())
            for c in ("E_param_base", "E_param_tgd",
                      "E_param_excess_base", "E_param_excess_tgd", "C15_base")
        }
        summary.setdefault(reg, {})[gname] = entry
        if per_seed is not None:
            for sname, sg in per_seed.items():
                e = {c: {"median": quant(sg[c].to_numpy(), 0.5),
                         "n": int(np.isfinite(sg[c].to_numpy()).sum())}
                     for c in ("E_param_base", "E_param_tgd", "E_param_cn",
                               "E_param_excess_base", "E_param_excess_tgd")}
                summary[reg][sname] = e

# ---------------------------------------------------------------------------
# recovery associations (test): raw / partial(frac_snow) / S1-S5 raw
# ---------------------------------------------------------------------------
def assoc_block(g):
    fs = g["frac_snow"].to_numpy()
    gb = g["G_base"].to_numpy()
    gk = g["G_TGD_ko"].to_numpy()
    eb = g["E_param_excess_base"].to_numpy()
    et = g["E_param_excess_tgd"].to_numpy()
    c15b = g["C15_base"].to_numpy()
    c15t = g["C15_tgd"].to_numpy()
    blk = {
        "G_base_vs_E_excess_base": [spearman(gb, eb), partial_spearman(gb, eb, fs)],
        "G_TGD_ko_vs_E_excess_tgd": [spearman(gk, et), partial_spearman(gk, et, fs)],
        "G_base_vs_C15_base_secondary": [spearman(gb, c15b), partial_spearman(gb, c15b, fs)],
        "G_TGD_ko_vs_C15_tgd_secondary": [spearman(gk, c15t), partial_spearman(gk, c15t, fs)],
        "strata_raw": {},
    }
    for st in ("S1", "S2", "S3", "S4", "S5"):
        m = g["snow_stratum"] == st
        blk["strata_raw"][st] = {
            "G_base_vs_E_excess_base": spearman(gb[m], eb[m]),
            "G_TGD_ko_vs_E_excess_tgd": spearman(gk[m], et[m]),
            "n": int(m.sum()),
        }
    return blk


assoc = {}
for reg in ("IC", "dPL"):
    sub = out[out["paradigm"] == reg]
    if reg == "IC":
        assoc["IC"] = assoc_block(sub[sub["seed"].isna()])
    else:
        sm = sub.groupby("basin_id").agg(
            G_base=("G_base", "median"),
            G_TGD_ko=("G_TGD_ko", "median"),
            E_param_excess_base=("E_param_excess_base", "median"),
            E_param_excess_tgd=("E_param_excess_tgd", "median"),
            C15_base=("C15_base", "median"),
            C15_tgd=("C15_tgd", "median"),
            frac_snow=("frac_snow", "first"),
            snow_stratum=("snow_stratum", "first"),
        ).reset_index()
        assoc["dPL_seedmedian"] = assoc_block(sm)
        assoc["dPL_per_seed"] = {
            f"dPL_s{s}": assoc_block(sub[sub["seed"] == s]) for s in SEEDS
        }

summary["assoc"] = assoc

# --- validation 4: new != old (C15), negatives retained ---------------------
s = out[(out["paradigm"] == "IC") & out["seed"].isna()]
neq = int((np.abs(s["E_param_excess_base"] - s["C15_base"]) > 1e-12).sum())
print(f"[v4] basins where E_param_excess_base == C15_base (within 1e-12): "
      f"{len(s) - neq} / {len(s)}")
print(f"[v4] E_param_excess_base < 0: {(s['E_param_excess_base'] < 0).sum()}, "
      f"min = {s['E_param_excess_base'].min():.4f} (negatives retained, not clipped)")
corr = spearman(s["E_param_excess_base"], s["C15_base"])
print(f"[v4] Spearman(E_param_excess_base, C15_base) = {corr:.3f} (related but distinct)")
print("[v4] example basins IC: ")
for b in ["01022500", "01031500", "01047000"]:
    r = s[s["basin_id"] == b].iloc[0]
    print(f"      {b}: E_param_base={r['E_param_base']:.4f} "
          f"E_param_cn={r['E_param_cn']:.4f} "
          f"E_param_excess_base={r['E_param_excess_base']:+.4f} "
          f"C15_base={r['C15_base']:.4f}")

with open(OUT / "param_truth_error_summary.json", "w") as fh:
    json.dump(summary, fh, indent=1, sort_keys=True, default=float)
print("\nwrote:", OUT / "param_truth_error_basin.csv",
      OUT / "param_truth_error_summary.json")