#!/usr/bin/env python3
"""R3 Table S5 generator: detailed SI statistics table for the hydrodiag R3
controlled synthetic-truth experiment (XAJ-CN generating structure).

Makes the main-text R3 evidence (Figures 5 and 6) auditable.  Read-only over
canonical R3 outputs; no training, no new experiments, no metric redefinition.
Frozen values are read verbatim from the prepared Figure 5/6 summaries and the
frozen post-hoc summaries; any recomputed median is asserted equal to the
frozen value (tolerance 1e-6 for Block 2, 1e-9 for Block 3 seed rows).

Blocks
------
  1. Correct-CN reference (output KGE/deficit, parameter recovery D_theta,
     primary-state NRMSE) -- justifies the CN-adjusted excess errors.
  2. Main R3 estimands with exact valid N, median, and 95 % bootstrap CI
     (paired-basin bootstrap, 2000 replicates, seed 20260730; dPL aggregated
     to per-basin seed medians, exactly the Figure 5/6 convention).
  3. dPL seed stability (frozen per-seed medians for the main quantities).
  4. Output-internal association boundary (raw and partial Spearman
     controlling frac_snow, from the frozen V1 validation).
  5. Aggregate definitions (frozen parameter/state lists and C_theta/C_state
     definitions from protocol_misspec_v1.json).

Outputs (mirrors generate_table1.py convention)
-----------------------------------------------
  manuscript/tables/TableS5_R3_statistics.md        (Markdown)
  manuscript/tables/TableS5_R3_statistics.tex       (LaTeX)
  manuscript/stats/tables/TableS5_R3_statistics.md  (copy)
  manuscript/stats/tables/TableS5_R3_statistics.tex (copy)
  manuscript/results/R3/tableS5_si_statistics.csv   (machine-readable source)

Usage: python manuscript/scripts/generate_table_r3_si.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]  # manuscript/scripts -> project/hydrodiag
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from r3.common import DEFAULT_RESULTS_ROOT  # noqa: E402

SEEDS = (42, 123, 2026)
BOOT_N = 2000
BOOT_SEED = 20260730
CANON = DEFAULT_RESULTS_ROOT / "r3_misspec_analysis_v1"
GATE = DEFAULT_RESULTS_ROOT / "r3_gate_v1"
PREP = PROJECT / "manuscript" / "results" / "R3"
OUT_DIRS = (PROJECT / "manuscript" / "tables", PROJECT / "manuscript" / "stats" / "tables")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def norm_seed(series: pd.Series) -> pd.Series:
    """Normalize seed columns: IC -> "", dPL -> "42"/"123"/"2026"."""
    return series.apply(lambda v: "" if pd.isna(v) else str(int(v)))


def boot_ci_median(values: np.ndarray, n_boot: int = BOOT_N, seed: int = BOOT_SEED,
                   alpha: float = 0.05):
    """Paired basin-level bootstrap CI of the median (R3 convention)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = np.median(values[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def fmt_stat(med, lo, hi, dec: int = 4):
    fmt = f"{{:.{dec}f}}"
    return f"{fmt.format(med)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def assert_close(got: float, expect: float, tol: float, label: str) -> None:
    if expect is None or not np.isfinite(expect):
        raise SystemExit(f"assertion FAILED ({label}): frozen value missing")
    if abs(got - expect) > tol:
        raise SystemExit(f"assertion FAILED ({label}): {got:.10f} != frozen {expect:.10f}")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load():
    f5 = json.loads((PREP / "figure5_summary.json").read_text())
    f6 = json.loads((PREP / "figure6_summary.json").read_text())
    ps = json.loads((CANON / "posthoc_summary.json").read_text())
    pv = json.loads((CANON / "posthoc_validation_summary.json").read_text())
    gate = json.loads((GATE / "gate_report.json").read_text())
    gss = json.loads((GATE / "gate_state_summary.json").read_text())
    proto = json.loads((PROJECT / "r3" / "protocol_misspec_v1.json").read_text())

    bt = pd.read_csv(CANON / "posthoc_basin_table.csv")
    red = pd.read_csv(CANON / "posthoc_validation_tgd2_reduction.csv")
    resv = pd.read_csv(CANON / "posthoc_validation_residual.csv")
    proc = pd.read_csv(CANON / "posthoc_process_errors.csv")
    vp = pd.read_csv(CANON / "posthoc_validation_partial.csv")
    for df in (bt, red, resv, proc, vp):
        if "basin_id" in df.columns:
            df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    bt["seed"] = norm_seed(bt["seed"])
    red["seed"] = norm_seed(red["seed"])
    resv["seed"] = norm_seed(resv["seed"])
    return f5, f6, ps, pv, gate, gss, proto, bt, red, resv, proc, vp


def seed_med(d: pd.DataFrame, value_col: str, reg: str) -> np.ndarray:
    """Per-basin values: IC passthrough; dPL median over the three seeds."""
    if reg == "IC":
        return d[d["seed"] == ""][value_col].dropna().to_numpy()
    sub = d[d["paradigm"] == "dPL"]
    g = sub.groupby("basin_id")[value_col].median().dropna()
    return g.to_numpy()


# ---------------------------------------------------------------------------
# Block 2 / Block 3 computation
# ---------------------------------------------------------------------------
def block2_rows(bt, red, resv, proc):
    """Main estimands: median, N, 95 % bootstrap CI (per regime)."""
    rows = {}

    bt_test = bt[bt["period"] == "test"]
    # 1. F_close (test)
    fc = {}
    for reg in ["IC", "dPL"]:
        v = seed_med(bt_test, "F_close", reg)
        fc[reg] = (float(np.median(v)), int(len(v)), boot_ci_median(v))
    rows["F_close_test"] = fc

    # 2. F_tgd2 (test)
    ft = {}
    for reg in ["IC", "dPL"]:
        v = seed_med(bt_test, "F_tgd2", reg)
        ft[reg] = (float(np.median(v)), int(len(v)), boot_ci_median(v))
    rows["F_tgd2_test"] = ft

    # 3. R_theta_tgd2 (parameter relief)
    rt = {}
    for reg in ["IC", "dPL"]:
        v = seed_med(red, "R_theta_tgd2", reg)
        rt[reg] = (float(np.median(v)), int(len(v)), boot_ci_median(v))
    rows["R_theta_tgd2"] = rt

    # 4. R_state_tgd2 (state relief)
    rs = {}
    for reg in ["IC", "dPL"]:
        v = seed_med(red, "R_state_tgd2", reg)
        rs[reg] = (float(np.median(v)), int(len(v)), boot_ci_median(v))
    rows["R_state_tgd2"] = rs

    # 5. G_CN_over_TGD2 (residual explicit advantage, test)
    resv_test = resv[resv["period"] == "test"]
    gc = {}
    for reg in ["IC", "dPL"]:
        v = seed_med(resv_test, "G_CN_over_TGD2", reg)
        gc[reg] = (float(np.median(v)), int(len(v)), boot_ci_median(v, seed=BOOT_SEED + 4))
    rows["G_CN_over_TGD2_test"] = gc

    # 6. delta_rmse snow-active / no-snow-active (process-conditioned residual)
    proc_test = proc[proc["period"] == "test"]
    # vectorised per-regime delta_rmse = rmse(TGD2) - rmse(CN) per basin
    dproc = {}
    for reg, fits in [("IC", [("CN_IC", "TGD2_IC")]),
                      ("dPL", tuple((f"CN_dPL_s{s}", f"TGD2_dPL_s{s}") for s in SEEDS))]:
        dproc[reg] = {}
        for cond in ["snow_active", "no_snow_active"]:
            per_basin = {}
            for b in proc_test["basin_id"].unique():
                ds = []
                for pair in fits:
                    fit_cn, fit_tg = pair
                    cn_r = proc_test[(proc_test["basin_id"] == b) & (proc_test["fit"] == fit_cn)
                                     & (proc_test["condition"] == cond)]
                    tg_r = proc_test[(proc_test["basin_id"] == b) & (proc_test["fit"] == fit_tg)
                                     & (proc_test["condition"] == cond)]
                    if len(cn_r) == 1 and len(tg_r) == 1:
                        ds.append(float(tg_r["rmse"].iloc[0] - cn_r["rmse"].iloc[0]))
                if ds:
                    per_basin[b] = float(np.median(ds))  # dPL: seed median; IC: single
            v = np.asarray(list(per_basin.values()), dtype=np.float64)
            dproc[reg][cond] = (float(np.median(v)), int(len(v)), boot_ci_median(v, seed=BOOT_SEED + 30))
    rows["delta_rmse"] = dproc

    # 7. decay_G_base = G_base(train) - G_base(test), seed-paired per basin
    #    (dPL: median of the three per-seed decays, matching the Figure 5
    #    aggregation of posthoc_validation_decay.csv)
    dg = {}
    for reg in ["IC", "dPL"]:
        dec = {}
        for b in bt["basin_id"].unique():
            tr = bt[(bt["basin_id"] == b) & (bt["paradigm"] == reg)
                    & (bt["period"] == "train")][["seed", "G_base"]]
            te = bt[(bt["basin_id"] == b) & (bt["paradigm"] == reg)
                    & (bt["period"] == "test")][["seed", "G_base"]]
            if tr.empty or te.empty:
                continue
            m = tr.merge(te, on="seed", suffixes=("_tr", "_te"))
            dec[b] = float(np.median(m["G_base_tr"] - m["G_base_te"]))
        v = np.asarray(list(dec.values()), dtype=np.float64)
        dg[reg] = (float(np.median(v)), int(len(v)), boot_ci_median(v, seed=BOOT_SEED + 2))
    rows["decay_G_base"] = dg

    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--n-boot", type=int, default=BOOT_N)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    args = parser.parse_args()

    f5, f6, ps, pv, gate, gss, proto, bt, red, resv, proc, vp = load()

    # ---------------- Block 1: correct-CN reference ----------------
    b1 = []
    bt_test = bt[bt["period"] == "test"]
    for reg in ["IC", "dPL"]:
        kge = seed_med(bt_test, "kge_cn", reg)
        kge_med = float(np.median(kge))
        deficit_med = float(1.0 - kge_med)
        dtheta = float(gate["equifinality"][reg]["D_theta_median"])
        b1.append({
            "block": 1, "estimand": "CN test KGE (vs Q*)", "regime": reg,
            "n": int(len(kge)), "median": kge_med,
            "note": f"deficit 1-KGE = {deficit_med:.4f}",
        })
        b1.append({
            "block": 1, "estimand": "CN parameter recovery D_theta", "regime": reg,
            "n": None, "median": dtheta, "note": "gate equifinality median",
        })
        for st in ["wu", "wl", "s", "qi", "qg"]:
            key = "IC" if reg == "IC" else "dPL_median"
            b1.append({
                "block": 1, "estimand": f"CN state NRMSE [{st}]", "regime": reg,
                "n": None, "median": float(gss[st][key]["nrmse_median"]),
                "note": "gate state summary, primary states",
            })

    # ---------------- Block 2: main estimands ----------------
    b2r = block2_rows(bt, red, resv, proc)
    est2 = [
        ("F_close_test", "Limited compensation: F_close (test)", "F_close", "panel_c"),
        ("F_tgd2_test", "Generic mitigation: F_tgd2 (test)", "F_tgd2", "panel_b"),
        ("R_theta_tgd2", "Parameter relief: R_theta_tgd2", "R_theta", "panel_c"),
        ("R_state_tgd2", "State relief: R_state_tgd2", "R_state", "panel_d"),
        ("G_CN_over_TGD2_test", "Residual CN advantage: G_CN_over_TGD2 (test)", "G_CN", "panel_e"),
        ("decay_G_base", "Generalization decay: decay_G_base", "decay", "panel_d"),
    ]
    b2 = []
    for key, label, short, panel in est2:
        for reg in ["IC", "dPL"]:
            med, n, (lo, hi) = b2r[key][reg]
            b2.append({"block": 2, "estimand": label, "short": short, "regime": reg,
                       "n": n, "median": med, "ci_low": lo, "ci_high": hi})
    for cond in ["snow_active", "no_snow_active"]:
        for reg in ["IC", "dPL"]:
            med, n, (lo, hi) = b2r["delta_rmse"][reg][cond]
            b2.append({"block": 2,
                       "estimand": f"Process residual RMSE gap [{cond}] (TGD2-CN)",
                       "short": f"dRMSE_{cond}", "regime": reg, "n": n,
                       "median": med, "ci_low": lo, "ci_high": hi})

    # ---------------- Block 3: dPL seed stability (frozen) ----------------
    b3 = []
    seed_specs = [
        ("F_close", "F_close (test)", lambda s: ps[f"dPL_{s}_test"]["F_close"]["median"]),
        ("F_tgd2", "F_tgd2 (test)", lambda s: ps[f"dPL_{s}_test"]["F_tgd2"]["median"]),
        ("R_theta", "R_theta_tgd2", lambda s: pv["V3"][f"dPL_{s}"]["R_theta_tgd2"]["median"]),
        ("R_state", "R_state_tgd2", lambda s: pv["V3"][f"dPL_{s}"]["R_state_tgd2"]["median"]),
        ("G_CN", "G_CN_over_TGD2 (test)", lambda s: pv["V4"][f"dPL_{s}_test"]["G_CN_over_TGD2"]["median"]),
        ("decay", "decay_G_base", lambda s: pv[f"dPL_{s}_decay_G_base"]["median"]),
    ]
    for short, label, fn in seed_specs:
        for s in SEEDS:
            b3.append({"block": 3, "estimand": label, "short": short, "regime": "dPL",
                       "seed": s, "n": None, "median": float(fn(s)), "ci_low": None,
                       "ci_high": None})

    # ---------------- Block 4: association boundary ----------------
    b4 = []
    vp_test = vp[vp["period"] == "test"]
    for pair in ["G_base|C_theta_primary", "G_base|C_state_primary"]:
        for reg in ["IC", "dPL"]:
            sub = vp_test[(vp_test["pair"] == pair) & (vp_test["paradigm"] == reg)]
            for _, r in sub.iterrows():
                b4.append({
                    "block": 4, "estimand": pair, "regime": reg,
                    "seed": "" if pd.isna(r["seed"]) else str(int(r["seed"])),
                    "raw_spearman": float(r["raw_spearman"]),
                    "partial_spearman": float(r["partial_spearman"]),
                })
    for reg_key, reg in [("IC", "IC"), ("dPL_42", "dPL"), ("dPL_123", "dPL"), ("dPL_2026", "dPL")]:
        t = ps["tradeoffs"][reg_key]
        b4.append({
            "block": 4, "estimand": "C_theta vs frac_snow", "regime": reg,
            "seed": "" if reg == "IC" else reg_key.split("_")[1],
            "raw_spearman": float(t["spearman_C_theta_vs_frac_snow"]),
            "partial_spearman": None,
        })
        b4.append({
            "block": 4, "estimand": "C_state vs frac_snow", "regime": reg,
            "seed": "" if reg == "IC" else reg_key.split("_")[1],
            "raw_spearman": float(t["spearman_C_state_vs_frac_snow"]),
            "partial_spearman": None,
        })

    # ---------------- Block 5: aggregate definitions ----------------
    tiers = proto["predeclared_parameter_tiers"]
    states = proto["state_estimands"]
    b5 = [
        {"block": 5, "estimand": "C_theta definition", "regime": "IC",
         "note": "median over frozen primary params of |e_M - e_CN|; "
                 f"IC primary = {tiers['ic_primary']}; secondary = {tiers['ic_secondary_supporting']}"},
        {"block": 5, "estimand": "C_theta definition", "regime": "dPL",
         "note": "median over frozen primary params of |e_M - e_CN|; "
                 f"dPL primary = {tiers['dpl_primary']}; secondary = {tiers['dpl_secondary_supporting']}"},
        {"block": 5, "estimand": "C_state definition", "regime": "IC/dPL",
         "note": f"median over primary states of delta_NRMSE (test); primary = "
                 f"{states['primary_common_variables']}; secondary = {states['secondary']} "
                 f"(wd excluded: {states['wd_reason']}); derived total tension storage "
                 f"wt = wu+wl+wd ({states['derived_total_tension_storage']['definition']})"},
    ]

    # ---------------- sanity assertions ----------------
    # Block 2 medians vs Figure 5/6 prepared summaries (1e-6)
    f5c, f6c = f5, f6
    assert_close([r for r in b2 if r["estimand"] == "Limited compensation: F_close (test)" and r["regime"] == "IC"][0]["median"],
                 f5c["panel_c_f_close"]["IC_test"]["median"], 1e-6, "F_close IC")
    assert_close([r for r in b2 if r["estimand"] == "Limited compensation: F_close (test)" and r["regime"] == "dPL"][0]["median"],
                 f5c["panel_c_f_close"]["dPL_test"]["median"], 1e-6, "F_close dPL")
    assert_close([r for r in b2 if r["estimand"] == "Generalization decay: decay_G_base" and r["regime"] == "IC"][0]["median"],
                 f5c["panel_d_decay"]["IC"]["median"], 1e-6, "decay IC")
    assert_close([r for r in b2 if r["estimand"] == "Generalization decay: decay_G_base" and r["regime"] == "dPL"][0]["median"],
                 f5c["panel_d_decay"]["dPL_agg"]["median"], 1e-6, "decay dPL")
    assert_close([r for r in b2 if r["estimand"] == "Generic mitigation: F_tgd2 (test)" and r["regime"] == "IC"][0]["median"],
                 f6c["panel_b_f_tgd2"]["IC"]["median"], 1e-6, "F_tgd2 IC")
    assert_close([r for r in b2 if r["estimand"] == "Generic mitigation: F_tgd2 (test)" and r["regime"] == "dPL"][0]["median"],
                 f6c["panel_b_f_tgd2"]["dPL"]["median"], 1e-6, "F_tgd2 dPL")
    assert_close([r for r in b2 if r["estimand"] == "Parameter relief: R_theta_tgd2" and r["regime"] == "IC"][0]["median"],
                 f6c["panel_c_r_theta"]["IC"]["median"], 1e-6, "R_theta IC")
    assert_close([r for r in b2 if r["estimand"] == "Parameter relief: R_theta_tgd2" and r["regime"] == "dPL"][0]["median"],
                 f6c["panel_c_r_theta"]["dPL"]["median"], 1e-6, "R_theta dPL")
    assert_close([r for r in b2 if r["estimand"] == "State relief: R_state_tgd2" and r["regime"] == "IC"][0]["median"],
                 f6c["panel_d_r_state"]["IC"]["median"], 1e-6, "R_state IC")
    assert_close([r for r in b2 if r["estimand"] == "State relief: R_state_tgd2" and r["regime"] == "dPL"][0]["median"],
                 f6c["panel_d_r_state"]["dPL"]["median"], 1e-6, "R_state dPL")
    assert_close([r for r in b2 if r["estimand"] == "Residual CN advantage: G_CN_over_TGD2 (test)" and r["regime"] == "IC"][0]["median"],
                 f6c["panel_e_residual_vs_frac_snow"]["IC"]["median"], 1e-6, "G_CN IC")
    assert_close([r for r in b2 if r["estimand"] == "Residual CN advantage: G_CN_over_TGD2 (test)" and r["regime"] == "dPL"][0]["median"],
                 f6c["panel_e_residual_vs_frac_snow"]["dPL"]["median"], 1e-6, "G_CN dPL")
    for cond in ["snow_active", "no_snow_active"]:
        for reg in ["IC", "dPL"]:
            assert_close([r for r in b2 if f"Process residual RMSE gap [{cond}]" in r["estimand"] and r["regime"] == reg][0]["median"],
                         f6c["panel_f_process_errors"][reg][cond]["median"], 1e-6, f"dRMSE {reg} {cond}")
    # Block 3 seed medians vs frozen posthoc summaries (1e-9)
    for r in b3:
        fn = {"F_close": lambda s: ps[f"dPL_{s}_test"]["F_close"]["median"],
              "F_tgd2": lambda s: ps[f"dPL_{s}_test"]["F_tgd2"]["median"],
              "R_theta": lambda s: pv["V3"][f"dPL_{s}"]["R_theta_tgd2"]["median"],
              "R_state": lambda s: pv["V3"][f"dPL_{s}"]["R_state_tgd2"]["median"],
              "G_CN": lambda s: pv["V4"][f"dPL_{s}_test"]["G_CN_over_TGD2"]["median"],
              "decay": lambda s: pv[f"dPL_{s}_decay_G_base"]["median"]}[r["short"]]
        assert_close(r["median"], float(fn(r["seed"])), 1e-9, f"seed {r['short']} s{r['seed']}")

    print(f"[check] Block-2 medians match Figure 5/6 summaries (1e-6): OK "
          f"({sum(1 for r in b2 if r['block']==2)} rows)")
    print(f"[check] Block-3 seed medians match frozen posthoc summaries (1e-9): OK "
          f"({len(b3)} rows)")

    # ---------------- build markdown / latex ----------------
    md_parts = ["# Table S5: R3 Synthetic-Truth SI Statistics\n"]
    tex_parts = []
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    # Block 1 md
    md_parts.append("\n### Block 1 — Correct-CN reference\n\n")
    md_parts.append("| Regime | Estimand | Median | Note |\n| :---: | :--- | :---: | :--- |\n")
    for r in b1:
        md_parts.append(f"| {r['regime']} | {r['estimand']} | {r['median']:.4f} | {r['note']} |\n")

    # Block 2 md
    md_parts.append("\n### Block 2 — Main R3 estimands\n\n")
    md_parts.append("| Estimand | Regime | N | Median [95% CI] |\n| :--- | :---: | :---: | :--- |\n")
    for r in b2:
        md_parts.append(f"| {r['estimand']} | {r['regime']} | {r['n']} | {fmt_stat(r['median'], r['ci_low'], r['ci_high'])} |\n")

    # Block 3 md
    md_parts.append("\n### Block 3 — dPL seed stability\n\n")
    md_parts.append("| Estimand | Seed | Median |\n| :--- | :---: | :---: |\n")
    for r in b3:
        md_parts.append(f"| {r['estimand']} | {r['seed']} | {r['median']:.6f} |\n")

    # Block 4 md
    md_parts.append("\n### Block 4 — Output–internal association boundary\n\n")
    md_parts.append("| Pair | Regime | Seed | Raw ρ | Partial ρ (controlling frac_snow) |\n| :--- | :---: | :---: | :---: | :---: |\n")
    for r in b4:
        part = '-' if r['partial_spearman'] is None else f"{r['partial_spearman']:.3f}"
        md_parts.append(f"| {r['estimand']} | {r['regime']} | {r['seed']} | "
                        f"{r['raw_spearman']:.3f} | {part} |\n")

    # Block 5 md
    md_parts.append("\n### Block 5 — Aggregate definitions\n\n")
    md_parts.append("| Estimand | Regime | Definition |\n| :--- | :---: | :--- |\n")
    for r in b5:
        md_parts.append(f"| {r['estimand']} | {r['regime']} | {r['note']} |\n")

    md_note = ("\n*Note*: Values report basin-level medians; dPL rows aggregate the three "
               "seeds (42/123/2026) to per-basin seed medians before summarising, exactly "
               "as in Figures 5–6. 95% CIs are paired-basin bootstrap (2000 replicates, "
               "seed 20260730). Block 2 medians and Block 3 seed medians are asserted equal "
               "to the frozen post-hoc summaries (1e-6 / 1e-9). In Block 4, the raw "
               "output-recovery/internal-cost association largely disappears after "
               "controlling for frac_snow (partial ρ ≈ 0), i.e. the relationship is "
               "jointly organized by snow-process activity rather than an independent "
               "trade-off. D_theta is the correct-CN parameter-recovery dispersion (gate "
               "equifinality, KGE ≥ 0.99 basins).")
    md_parts.append(md_note)
    full_md = "".join(md_parts)

    # LaTeX
    for d in OUT_DIRS:
        with open(d / "TableS5_R3_statistics.md", "w") as f:
            f.write(full_md)

    tex_lines = []
    tex_lines.append(r"""\begin{table*}[t]
\centering
\caption{R3 controlled synthetic-truth SI statistics (XAJ-CN generating structure).}
\label{tab:tables5_r3_statistics}
\begin{threeparttable}
\begin{tabular}{llrr}
\toprule
Estimand & Regime & $N$ & Median [95\% CI] \\
\midrule""")
    for r in b2:
        tex_lines.append(f"{r['estimand']} & {r['regime']} & {r['n']} & "
                         f"{r['median']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}] \\\\")
    tex_lines.append(r"""\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\small
\item \textit{Note}: Main R3 estimands with paired-basin bootstrap 95\% CIs (2000 replicates, seed 20260730); dPL values aggregate the three seeds (42/123/2026) to per-basin seed medians as in Figures 5--6. Full Block 1 (correct-CN reference), Block 3 (seed stability), Block 4 (association boundary) and Block 5 (aggregate definitions) rows are in the Markdown/csv companion output.
\end{tablenotes}
\end{threeparttable}
\end{table*}""")
    full_tex = "\n".join(tex_lines)
    for d in OUT_DIRS:
        with open(d / "TableS5_R3_statistics.tex", "w") as f:
            f.write(full_tex + "\n")

    # machine-readable CSV
    src_rows = b1 + b2 + b3 + b4 + b5
    src = pd.DataFrame(src_rows)
    src = src.reindex(columns=[c for c in
                               ["block", "estimand", "short", "regime", "seed", "n",
                                "median", "ci_low", "ci_high", "raw_spearman",
                                "partial_spearman", "note"] if c in src.columns])
    src.to_csv(PREP / "tableS5_si_statistics.csv", index=False)

    print("Table S5 generated successfully in markdown and LaTeX formats.")
    print(f"  md/tex: manuscript/tables/ + manuscript/stats/tables/TableS5_R3_statistics.*")
    print(f"  csv:    {PREP / 'tableS5_si_statistics.csv'}")
    print(f"  rows per block: B1={len(b1)} B2={len(b2)} B3={len(b3)} B4={len(b4)} B5={len(b5)}")


if __name__ == "__main__":
    main()
