#!/usr/bin/env python3
"""R3 Figure 5 data preparation — canonical outlet recovery evidence.

Consumes canonical R3 post-hoc outputs (results/r3_misspec_analysis_v1/)
and snow attributes (manuscript/results/R1/r1_snow_attributes.csv).
Produces the tidy basin-level table, seed-median aggregated table, and
figure-facing summary JSON for Figure 5 (Outlet Deficit and Recovery).

Scientific logic:
  - Imposed deficit: D = KGE(CN_refit) - KGE(Base_no-refit)
  - Raw Base recovery: G_base = KGE(Base_refit) - KGE(Base_no-refit)
  - Raw TGD recovery from knockout: G_TGD = KGE(TGD_refit) - KGE(Base_no-refit)
  - Normalized Base gap closure: F_close = G_base / D  (D > 1e-6)
  - Normalized TGD common-reference recovery: F_TGD* = G_TGD / D  (D > 1e-6)

Outputs (manuscript/results/R3/):
  figure5_basin_table.csv          tidy long table (531 x 2 regimes x seeds x 2 periods)
  figure5_basin_seedmedian.csv     dPL aggregated to per-basin seed median (IC passthrough)
  figure5_summary.json             panel-level summary statistics and CIs

Usage: python manuscript/scripts/r3/prepare_figure5_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]  # manuscript/scripts -> project/hydrodiag
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.scripts.r3.common import DEFAULT_RESULTS_ROOT, git_commit, write_json  # noqa: E402

SEEDS = (42, 123, 2026)
REGIMES = ("IC", "dPL")
PERIODS = ("train", "test")
MANUSCRIPT_R3_REL = Path("results") / "R3"
DENOM_TOL = 1e-6


def boot_ci(values: np.ndarray, stat_fn, n_boot: int, seed: int, alpha: float = 0.05):
    """Paired basin-level bootstrap CI."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n = len(vals)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn(vals[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 5 or x[v].std() == 0 or y[v].std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[v]))
    ry = np.argsort(np.argsort(y[v]))
    return float(np.corrcoef(rx, ry)[0, 1])


def quant(v, q):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if len(v) else float("nan")


def norm_seed(series: pd.Series) -> pd.Series:
    """Normalize seed columns: IC -> '', dPL -> '42'/'123'/'2026'."""
    return series.apply(lambda v: "" if pd.isna(v) or v == "" else str(int(float(v))))


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"refusing: Figure 5 input missing ({label}): {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default="r3_misspec_analysis_v1")
    parser.add_argument("--manuscript-root", type=Path, default=PROJECT / "manuscript")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    src = args.results_root / args.run_id
    r1_dir = args.manuscript_root / "results" / "R1"
    out_dir = args.manuscript_root / MANUSCRIPT_R3_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Inputs ----------------
    p_basin = src / "posthoc_basin_table.csv"
    p_decay = src / "posthoc_validation_decay.csv"
    p_sum = src / "posthoc_summary.json"
    p_val = src / "posthoc_validation_summary.json"
    p_strata = r1_dir / "r1_snow_attributes.csv"

    for p, label in [
        (p_basin, "basin table"),
        (p_decay, "decay"),
        (p_sum, "posthoc summary"),
        (p_val, "validation summary"),
        (p_strata, "snow attributes"),
    ]:
        require(p, label)

    bt = pd.read_csv(p_basin)
    dc = pd.read_csv(p_decay)
    strata = pd.read_csv(p_strata)
    frozen_sum = json.loads(p_sum.read_text())
    frozen_val = json.loads(p_val.read_text())

    for df in (bt, dc, strata):
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    bt["seed"] = norm_seed(bt["seed"])
    dc["seed"] = norm_seed(dc["seed"])

    n_basins = int(bt["basin_id"].nunique())
    if n_basins != 531:
        raise SystemExit(f"refusing: expected 531 basins, found {n_basins}")

    # Join strata
    bt = bt.merge(strata[["basin_id", "snow_stratum"]], on="basin_id", how="left")

    # Compute canonical estimands
    bt["D"] = bt["kge_cn"] - bt["kge_base_no_refit"]
    bt["G_TGD"] = bt["kge_tgd2"] - bt["kge_base_no_refit"]
    bt["F_TGD"] = np.where(bt["D"] > DENOM_TOL, bt["G_TGD"] / bt["D"], np.nan)
    bt["F_TGD_star"] = bt["F_TGD"]

    # Join decay metrics
    decay_base = dc[dc["metric"] == "decay_G_base"].rename(
        columns={"decay": "decay_G_base"}
    )[["basin_id", "paradigm", "seed", "decay_G_base"]]
    decay_tgd = dc[dc["metric"] == "decay_G_tgd2"].rename(
        columns={"decay": "decay_G_tgd2"}
    )[["basin_id", "paradigm", "seed", "decay_G_tgd2"]]

    tidy = bt.merge(decay_base, on=["basin_id", "paradigm", "seed"], how="left").merge(
        decay_tgd, on=["basin_id", "paradigm", "seed"], how="left"
    )

    tidy = tidy.sort_values(["paradigm", "seed", "period", "basin_id"]).reset_index(drop=True)
    tidy.to_csv(out_dir / "figure5_basin_table.csv", index=False)

    # ---------------- 2. dPL seed-median aggregation ----------------
    agg_rows = []
    for reg in REGIMES:
        sub = tidy[tidy["paradigm"] == reg]
        if reg == "IC":
            agg_rows.append(sub.assign(seed=""))
        else:
            g = sub.groupby(["basin_id", "period"], as_index=False)
            agg = g.agg(
                frac_snow=("frac_snow", "first"),
                snow_stratum=("snow_stratum", "first"),
                kge_base_no_refit=("kge_base_no_refit", "first"),
                kge_base=("kge_base", "median"),
                kge_tgd2=("kge_tgd2", "median"),
                kge_cn=("kge_cn", "median"),
                decay_G_base=("decay_G_base", "median"),
                decay_G_tgd2=("decay_G_tgd2", "median"),
            )
            agg["D"] = agg["kge_cn"] - agg["kge_base_no_refit"]
            agg["G_base"] = agg["kge_base"] - agg["kge_base_no_refit"]
            agg["G_tgd2"] = agg["kge_tgd2"] - agg["kge_base_no_refit"]
            agg["G_TGD"] = agg["G_tgd2"]
            agg["F_close"] = np.where(agg["D"] > DENOM_TOL, agg["G_base"] / agg["D"], np.nan)
            agg["F_tgd2"] = np.where(agg["D"] > DENOM_TOL, agg["G_tgd2"] / agg["D"], np.nan)
            agg["F_TGD"] = agg["F_tgd2"]
            agg["F_TGD_star"] = agg["F_TGD"]
            agg["paradigm"] = "dPL"
            agg["seed"] = ""
            agg_rows.append(agg)
    seedmed = pd.concat(agg_rows, ignore_index=True)
    seedmed.to_csv(out_dir / "figure5_basin_seedmedian.csv", index=False)

    # ---------------- 3. Summary structure for plot_figure5.py ----------------
    summary: dict = {
        "protocol": "figure5_prepare_v2",
        "source_run": args.run_id,
        "code": git_commit(PROJECT),
        "n_basins": n_basins,
        "n_boot": args.n_boot,
        "boot_seed": args.seed,
    }

    # -- panel (a): correct-structure recoverability (deficit = 1 - KGE_CN) --
    pa = {}
    for reg in REGIMES:
        for period in PERIODS:
            sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == period)]
            d = (1.0 - sub["kge_cn"]).to_numpy()
            pa[f"{reg}_{period}"] = {
                "median": float(np.median(d)),
                "q25": float(np.quantile(d, 0.25)),
                "q75": float(np.quantile(d, 0.75)),
                "min": float(np.min(d)),
                "max": float(np.max(d)),
            }
    summary["panel_a_recoverability"] = pa

    # -- panel (b): unified outlet recovery ladder statistics (test period) --
    pb = {}
    for reg in REGIMES:
        sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == "test")]
        pb[reg] = {
            "Base_no_refit": {
                "median": float(sub["kge_base_no_refit"].median()),
                "q25": float(sub["kge_base_no_refit"].quantile(0.25)),
                "q75": float(sub["kge_base_no_refit"].quantile(0.75)),
            },
            "Base_refit": {
                "median": float(sub["kge_base"].median()),
                "q25": float(sub["kge_base"].quantile(0.25)),
                "q75": float(sub["kge_base"].quantile(0.75)),
            },
            "TGD_refit": {
                "median": float(sub["kge_tgd2"].median()),
                "q25": float(sub["kge_tgd2"].quantile(0.25)),
                "q75": float(sub["kge_tgd2"].quantile(0.75)),
            },
            "CN_refit": {
                "median": float(sub["kge_cn"].median()),
                "q25": float(sub["kge_cn"].quantile(0.25)),
                "q75": float(sub["kge_cn"].quantile(0.75)),
            },
        }
    summary["panel_b_ladder"] = pb
    summary["panel_bc_ladders"] = pb  # backward compatibility

    # -- panel (c): raw recovery from imposed knockout (test period) --
    pc_raw = {}
    for reg in REGIMES:
        sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == "test")]
        d_vals = sub["D"].to_numpy()
        gb_vals = sub["G_base"].to_numpy()
        gt_vals = sub["G_TGD"].to_numpy()

        # Deterministic quantile bins along D
        d_pos = d_vals[d_vals > 0]
        q_edges = np.quantile(d_pos, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bins_d = [(-np.inf, 0.0)] + [(q_edges[i], q_edges[i+1]) for i in range(len(q_edges)-1)]

        binned_summary = []
        for lo, hi in bins_d:
            if lo == -np.inf:
                m = d_vals <= hi
                label = "D <= 0"
            else:
                m = (d_vals > lo) & (d_vals <= hi)
                label = f"{lo:.2f}-{hi:.2f}"
            if m.sum() > 0:
                binned_summary.append({
                    "label": label,
                    "n": int(m.sum()),
                    "D_median": float(np.median(d_vals[m])),
                    "G_base_median": float(np.median(gb_vals[m])),
                    "G_TGD_median": float(np.median(gt_vals[m])),
                    "G_base_ci": list(boot_ci(gb_vals[m], np.median, args.n_boot, args.seed + 10)),
                    "G_TGD_ci": list(boot_ci(gt_vals[m], np.median, args.n_boot, args.seed + 11)),
                })

        pc_raw[reg] = {
            "D": {
                "median": float(np.median(d_vals)),
                "q25": float(np.quantile(d_vals, 0.25)),
                "q75": float(np.quantile(d_vals, 0.75)),
            },
            "G_base": {
                "median": float(np.median(gb_vals)),
                "q25": float(np.quantile(gb_vals, 0.25)),
                "q75": float(np.quantile(gb_vals, 0.75)),
                "ci": list(boot_ci(gb_vals, np.median, args.n_boot, args.seed + 1)),
                "frac_gt_0": float((gb_vals > 0).mean()),
            },
            "G_TGD": {
                "median": float(np.median(gt_vals)),
                "q25": float(np.quantile(gt_vals, 0.25)),
                "q75": float(np.quantile(gt_vals, 0.75)),
                "ci": list(boot_ci(gt_vals, np.median, args.n_boot, args.seed + 2)),
                "frac_gt_0": float((gt_vals > 0).mean()),
            },
            "spearman_D_G_base": float(spearman(d_vals, gb_vals)),
            "spearman_D_G_TGD": float(spearman(d_vals, gt_vals)),
            "binned": binned_summary,
        }
    summary["panel_c_raw_recovery"] = pc_raw
    summary["panel_d_raw_recovery"] = pc_raw  # backward compatibility

    # -- panel (d): normalized recovery fractions F_close and F_TGD --
    pd_frac = {}
    for reg in REGIMES:
        for period in PERIODS:
            sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == period)]
            fc = sub["F_close"].to_numpy()
            ft = sub["F_TGD"].to_numpy()

            v_fc = fc[np.isfinite(fc)]
            v_ft = ft[np.isfinite(ft)]

            f_tgd_data = {
                "median": float(np.median(v_ft)),
                "q25": float(np.quantile(v_ft, 0.25)),
                "q75": float(np.quantile(v_ft, 0.75)),
                "ci": list(boot_ci(v_ft, np.median, args.n_boot, args.seed)),
                "frac_lt_0": float((v_ft < 0).mean()),
                "frac_0_to_1": float(((v_ft >= 0) & (v_ft <= 1)).mean()),
                "frac_gt_1": float((v_ft > 1).mean()),
            }

            entry = {
                "n_valid": int(len(v_fc)),
                "n_total": int(len(fc)),
                "F_close": {
                    "median": float(np.median(v_fc)),
                    "q25": float(np.quantile(v_fc, 0.25)),
                    "q75": float(np.quantile(v_fc, 0.75)),
                    "ci": list(boot_ci(v_fc, np.median, args.n_boot, args.seed)),
                    "frac_lt_0": float((v_fc < 0).mean()),
                    "frac_0_to_1": float(((v_fc >= 0) & (v_fc <= 1)).mean()),
                    "frac_gt_1": float((v_fc > 1).mean()),
                },
                "F_TGD": f_tgd_data,
                "F_TGD_star": f_tgd_data,
            }
            if reg == "dPL":
                seed_fc = []
                seed_ft = []
                for s in SEEDS:
                    s_sub = tidy[(tidy["paradigm"] == reg) & (tidy["period"] == period) & (tidy["seed"] == str(s))]
                    s_fc = s_sub["F_close"].dropna().to_numpy()
                    s_ft = s_sub["F_TGD"].dropna().to_numpy()
                    seed_fc.append(float(np.median(s_fc)))
                    seed_ft.append(float(np.median(s_ft)))
                entry["F_close"]["seed_medians"] = seed_fc
                entry["F_TGD"]["seed_medians"] = seed_ft
                entry["F_TGD_star"]["seed_medians"] = seed_ft

            pd_frac[f"{reg}_{period}"] = entry
    summary["panel_d_fractions"] = pd_frac
    summary["panel_e_fractions"] = pd_frac  # backward compatibility

    # -- panel (e): train-to-test recovery attenuation footer strip --
    pe_decay = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg].drop_duplicates("basin_id")
        d_base = sub["decay_G_base"].dropna().to_numpy()
        d_tgd = sub["decay_G_tgd2"].dropna().to_numpy()

        pe_decay[reg] = {
            "decay_G_base": {
                "median": float(np.median(d_base)),
                "q25": float(np.percentile(d_base, 25)),
                "q75": float(np.percentile(d_base, 75)),
                "p10": float(np.percentile(d_base, 10)),
                "p90": float(np.percentile(d_base, 90)),
                "ci": list(boot_ci(d_base, np.median, args.n_boot, args.seed + 5)),
                "frac_gt_0": float((d_base > 0).mean()),
                "p_gt_0": float((d_base > 0).mean()),
                "n_valid": int(len(d_base)),
            },
            "decay_G_tgd": {
                "median": float(np.median(d_tgd)),
                "q25": float(np.percentile(d_tgd, 25)),
                "q75": float(np.percentile(d_tgd, 75)),
                "p10": float(np.percentile(d_tgd, 10)),
                "p90": float(np.percentile(d_tgd, 90)),
                "ci": list(boot_ci(d_tgd, np.median, args.n_boot, args.seed + 6)),
                "frac_gt_0": float((d_tgd > 0).mean()),
                "p_gt_0": float((d_tgd > 0).mean()),
                "n_valid": int(len(d_tgd)),
            },
        }
        if reg == "dPL":
            s_meds_base = []
            s_meds_tgd = []
            for s in SEEDS:
                s_sub = tidy[(tidy["paradigm"] == reg) & (tidy["seed"] == str(s))].drop_duplicates("basin_id")
                s_meds_base.append(float(s_sub["decay_G_base"].dropna().median()))
                s_meds_tgd.append(float(s_sub["decay_G_tgd2"].dropna().median()))
            pe_decay[reg]["decay_G_base"]["seed_medians"] = s_meds_base
            pe_decay[reg]["decay_G_tgd"]["seed_medians"] = s_meds_tgd

    summary["panel_e_decay"] = pe_decay
    summary["panel_f_decay"] = pe_decay  # backward compatibility

    write_json(out_dir / "figure5_summary.json", summary)

    print(f"COMPLETE Figure 5 data -> {out_dir}", flush=True)
    print(f"  figure5_basin_table.csv      (tidy, per seed: {len(tidy)} rows)", flush=True)
    print(f"  figure5_basin_seedmedian.csv (dPL seed-aggregated: {len(seedmed)} rows)", flush=True)
    print(f"  figure5_summary.json         (panel-level numbers)", flush=True)


if __name__ == "__main__":
    main()
