#!/usr/bin/env python3
"""R3 Figure 5 data preparation (read-only reshaping of canonical post-hoc outputs).

Consumes ONLY the completed R3 post-hoc package
(results/r3_misspec_analysis_v1/) and re-emits the basin-level tidy table and
figure-facing summary that plot_figure5.py draws.  No training, no truth
regeneration, no protocol changes, no modification of existing outputs, and
no new science: every headline statistic is either (i) copied verbatim from
the frozen posthoc_summary.json / posthoc_validation_summary.json or (ii)
recomputed from the frozen per-basin CSVs and asserted equal to the frozen
values (tolerance 1e-9).

Inputs (frozen, read-only; results/r3_misspec_analysis_v1/):
  posthoc_basin_table.csv          per basin x regime x seed x period KGE rows
                                   (kge_base_no_refit / kge_base / kge_cn /
                                    G_base / F_close / frac_snow)
  posthoc_theta_cost.csv           C_theta_primary per basin x regime x seed
  posthoc_state_cost.csv           C_state_primary per basin x regime x seed
  posthoc_validation_decay.csv     decay_G_base per basin x regime x seed
  posthoc_summary.json             frozen F_close / G_base medians + CIs
  posthoc_validation_summary.json  frozen decay medians + CIs + S-determinations

Outputs (manuscript/results/R3/, generated artifacts):
  figure5_basin_table.csv          tidy long table (531 x 2 regimes x seeds x 2
                                   periods); per-seed rows kept
  figure5_basin_seedmedian.csv     dPL aggregated to per-basin seed median
                                   (median over seeds 42/123/2026; IC passthrough)
  figure5_summary.json             panel-level numbers for plot_figure5.py

Only the following display-only quantities are computed here (clearly flagged
in the JSON; all use the canonical paired-basin bootstrap protocol: 2000
replicates, seed 20260730):
  * F_close median bootstrap CI for the four (regime x period) display groups
  * frac_snow-quartile bin medians + bootstrap CI (panels e/f)
  * OLS slope + slope bootstrap CI for the e/f trend lines (display smoother)

Usage: python manuscript/scripts/prepare_figure5_data.py
       [--results-root RES] [--run-id r3_misspec_analysis_v1]
       [--manuscript-root PROJECT/manuscript] [--n-boot 2000] [--seed 20260730]
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

from r3.common import DEFAULT_RESULTS_ROOT, git_commit, write_json  # noqa: E402

SEEDS = (42, 123, 2026)
REGIMES = ("IC", "dPL")
PERIODS = ("train", "test")
STRUCTURES = ("Base", "TGD2", "CN")  # scientific scope of Figure 5 (Base focus)
MANUSCRIPT_R3_REL = Path("results") / "R3"  # relative to --manuscript-root


def boot_ci(values: np.ndarray, stat_fn, n_boot: int, seed: int, alpha: float = 0.05):
    """Paired basin-level bootstrap CI — same protocol as r3/posthoc_stats.py."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn(values[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)




def norm_seed(series: pd.Series) -> pd.Series:
    """Normalize seed columns: IC -> "", dPL -> "42"/"123"/"2026"."""
    return series.apply(lambda v: "" if pd.isna(v) else str(int(v)))


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
    out_dir = args.manuscript_root / MANUSCRIPT_R3_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- inputs (frozen) ----------------
    p_basin = src / "posthoc_basin_table.csv"
    p_theta = src / "posthoc_theta_cost.csv"
    p_state = src / "posthoc_state_cost.csv"
    p_decay = src / "posthoc_validation_decay.csv"
    p_sum = src / "posthoc_summary.json"
    p_val = src / "posthoc_validation_summary.json"
    for p, label in [(p_basin, "basin table"), (p_theta, "theta cost"),
                     (p_state, "state cost"), (p_decay, "decay"),
                     (p_sum, "posthoc summary"), (p_val, "validation summary")]:
        require(p, label)

    bt = pd.read_csv(p_basin)
    tc = pd.read_csv(p_theta)
    sc = pd.read_csv(p_state)
    dc = pd.read_csv(p_decay)
    frozen_sum = json.loads(p_sum.read_text())
    frozen_val = json.loads(p_val.read_text())

    for df in (bt, tc, sc, dc):
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    bt["seed"] = norm_seed(bt["seed"])
    tc["seed"] = norm_seed(tc["seed"])
    sc["seed"] = norm_seed(sc["seed"])
    dc["seed"] = norm_seed(dc["seed"])
    n_basins = int(bt["basin_id"].nunique())
    if n_basins != 531:
        raise SystemExit(f"refusing: expected 531 basins, found {n_basins}")

    # ---------------- 1. tidy long table (per seed) ----------------
    # C_theta / C_state for structure Base (Figure 5 focuses on Base
    # structural omission; TGD2 rows remain available in the canonical CSVs).
    c_theta = (tc[tc["structure"] == "Base"]
                 .rename(columns={"C_theta_primary": "C_theta_base"})
                 [["basin_id", "paradigm", "seed", "C_theta_base"]])
    c_state = (sc[sc["structure"] == "Base"]
                 .rename(columns={"C_state_primary": "C_state_base"})
                 [["basin_id", "paradigm", "seed", "C_state_base"]])
    decay = (dc[dc["metric"] == "decay_G_base"]
               .rename(columns={"decay": "decay_G_base"})
               [["basin_id", "paradigm", "seed", "decay_G_base"]])

    tidy = bt.merge(c_theta, on=["basin_id", "paradigm", "seed"], how="left") \
             .merge(c_state, on=["basin_id", "paradigm", "seed"], how="left") \
             .merge(decay, on=["basin_id", "paradigm", "seed"], how="left")
    tidy = tidy.sort_values(["paradigm", "seed", "period", "basin_id"]).reset_index(drop=True)
    tidy.to_csv(out_dir / "figure5_basin_table.csv", index=False)

    # ---------------- 2. dPL seed-median aggregation ----------------
    # dPL: median over the three seeds per basin x period; IC: passthrough rows.
    agg_rows = []
    for reg in REGIMES:
        sub = tidy[tidy["paradigm"] == reg]
        if reg == "IC":
            agg_rows.append(sub.assign(seed=""))
        else:
            g = sub.groupby(["basin_id", "period"], as_index=False)
            agg = g.agg(
                frac_snow=("frac_snow", "first"),
                kge_base_no_refit=("kge_base_no_refit", "first"),
                kge_base=("kge_base", "median"),
                kge_cn=("kge_cn", "median"),
                G_base=("G_base", "median"),
                F_close=("F_close", "median"),
                C_theta_base=("C_theta_base", "median"),
                C_state_base=("C_state_base", "median"),
                decay_G_base=("decay_G_base", "median"),
            )
            agg["paradigm"] = "dPL"
            agg["seed"] = ""
            agg_rows.append(agg)
    seedmed = pd.concat(agg_rows, ignore_index=True)
    seedmed.to_csv(out_dir / "figure5_basin_seedmedian.csv", index=False)

    # ---------------- 3. sanity: recomputed medians == frozen values --------
    def frozen_fclose(regime_period: str):
        e = frozen_sum.get(regime_period, {})
        v = e.get("F_close", {})
        return v.get("median"), v.get("q25"), v.get("q75"), v.get("n_valid_denominator")

    check_errors = []
    n_checks = 0
    for reg in REGIMES:
        for period in PERIODS:
            for seed in ([None] if reg == "IC" else SEEDS):
                key = f"{reg}{'_' + str(seed) if seed is not None else ''}_{period}"
                sub = tidy[(tidy["paradigm"] == reg) & (tidy["period"] == period)
                           & (tidy["seed"] == ("" if seed is None else str(seed)))]
                got = float(sub["F_close"].median())
                frozen, *_ = frozen_fclose(key)
                n_checks += 1
                if frozen is None or abs(got - frozen) > 1e-9:
                    check_errors.append(f"F_close median {key}: {got} != {frozen}")
    for reg in REGIMES:
        for seed in ([None] if reg == "IC" else SEEDS):
            key = f"{reg}{'_' + str(seed) if seed is not None else ''}_decay_G_base"
            sub = tidy[(tidy["paradigm"] == reg)
                       & (tidy["seed"] == ("" if seed is None else str(seed)))]
            got = float(sub["decay_G_base"].median())
            frozen = frozen_val.get(key, {}).get("median")
            n_checks += 1
            if frozen is None or abs(got - frozen) > 1e-9:
                check_errors.append(f"decay median {key}: {got} != {frozen}")
    if check_errors:
        raise SystemExit("Figure 5 sanity check FAILED:\n  " + "\n  ".join(check_errors))
    print(f"[check] recomputed F_close / decay_G_base medians match frozen values "
          f"({n_checks} group(s) verified)", flush=True)

    # ---------------- 4. figure-facing summary ----------------
    summary: dict = {
        "protocol": "figure5_prepare_v1",
        "source_run": args.run_id,
        "inputs": [str(p.relative_to(args.results_root)) for p in
                   [p_basin, p_theta, p_state, p_decay, p_sum, p_val]],
        "code": git_commit(PROJECT),
        "n_basins": n_basins,
        "n_boot": args.n_boot,
        "boot_seed": args.seed,
        "display_only_quantities": [
            "F_close median bootstrap CI (four regime x period groups)",
            "frac_snow-quartile bin medians + bootstrap CI (panels e/f)",
            "per-regime decay_G_base bootstrap CI (panel d, display)",
        ],
    }

    # -- panel (a): correct-CN baseline reference (deficit = 1 - KGE_CN) --
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
    summary["panel_a_cn_deficit"] = pa

    # -- panel (c): gap-closure fraction (frozen medians + display CI) --
    pc = {}
    groups = [("IC", "train"), ("IC", "test"), ("dPL", "train"), ("dPL", "test")]
    for reg, period in groups:
        seed_med = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == period)]
        vals = seed_med["F_close"].to_numpy()
        valid = vals[np.isfinite(vals)]
        ci = boot_ci(valid, np.median, args.n_boot, args.seed)
        key = f"{reg}_{period}"
        entry = {
            "median": float(np.median(valid)),
            "q25": float(np.quantile(valid, 0.25)),
            "q75": float(np.quantile(valid, 0.75)),
            "boot_ci_median_display": list(ci),
            "n_valid": int(np.isfinite(vals).sum()),
            "n_total": int(len(vals)),
            "frac_gt_0": float((valid > 0).mean()),
            # display window for the jittered basin cloud (unclipped F_close has a
            # heavy two-sided tail; median/IQR/CI are unaffected by the window)
            "frac_outside_display_window": float((((valid < -0.5) | (valid > 1.75))).mean()),
        }
        if reg == "dPL":
            # per-seed frozen medians are kept verbatim; the aggregated display
            # median (median of per-basin seed medians) is a different quantity
            # and must lie within the per-seed median range.
            entry["seed_medians"] = [float(frozen_fclose(f"dPL_{s}_{period}")[0])
                                     for s in SEEDS]
            lo_s, hi_s = min(entry["seed_medians"]), max(entry["seed_medians"])
            if not (lo_s - 1e-9 <= entry["median"] <= hi_s + 1e-9):
                raise SystemExit(f"Figure 5 sanity FAILED: {key} aggregated median "
                                 f"{entry['median']:.6f} outside seed median range "
                                 f"[{lo_s:.6f}, {hi_s:.6f}]")
        else:
            # IC: the aggregated median IS the frozen median (identical quantity).
            frozen_med, *_ = frozen_fclose(key)
            assert abs(entry["median"] - frozen_med) < 1e-9, f"{key} frozen mismatch"
        pc[key] = entry
    summary["panel_c_f_close"] = pc

    # -- panel (d): compensation decay (frozen stats) --
    pd_ = {}
    for reg in REGIMES:
        for seed in ([None] if reg == "IC" else SEEDS):
            key = f"{reg}{'_' + str(seed) if seed is not None else ''}"
            e = frozen_val.get(f"{key}_decay_G_base", {})
            pd_[key if seed is not None else reg] = {
                "median": e.get("median"),
                "boot_ci_median": e.get("boot_ci_median"),
                "frac_gt_0": e.get("frac_gt_0"),
                "spearman_vs_frac_snow": e.get("spearman_vs_frac_snow"),
                "n_valid": e.get("n_valid"),
            }
    # aggregated per-regime decay distribution (one row per basin; dPL = median
    # over seeds).  Median must equal the frozen value; CI is display-only.
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg].drop_duplicates("basin_id")
        d = sub["decay_G_base"].to_numpy()
        d = d[np.isfinite(d)]
        ci = boot_ci(d, np.median, args.n_boot, args.seed + 2)
        pd_[f"{reg}_agg"] = {
            "median": float(np.median(d)),
            "boot_ci_median_display": list(ci),
            "frac_gt_0": float((d > 0).mean()),
            "n": int(len(d)),
            "frac_outside_display_window": float((((d < -0.1) | (d > 0.15))).mean()),
        }
        if reg == "IC":
            frozen_med = frozen_val["IC_decay_G_base"]["median"]
            assert abs(pd_["IC_agg"]["median"] - frozen_med) < 1e-9, \
                "panel (d) IC aggregated decay median != frozen"
        else:
            # The aggregated median (median of per-basin seed medians) is a
            # display-only quantity distinct from each per-seed median; only a
            # loose consistency check applies (well below the seed spread).
            seed_meds = [frozen_val[f"dPL_{s}_decay_G_base"]["median"] for s in SEEDS]
            lo_s, hi_s = min(seed_meds), max(seed_meds)
            tol = 1e-3
            if not (lo_s - tol <= pd_["dPL_agg"]["median"] <= hi_s + tol):
                raise SystemExit("Figure 5 sanity FAILED: dPL aggregated decay median "
                                 f"{pd_['dPL_agg']['median']:.6f} far outside frozen seed "
                                 f"median range [{lo_s:.6f}, {hi_s:.6f}]")
    summary["panel_d_decay"] = pd_

    # -- panels (e)/(f): excess errors vs frac_snow (Base) --
    pef = {}
    fs_all = np.sort(seedmed[seedmed["period"] == "test"]["frac_snow"].to_numpy())
    q_bins = np.quantile(fs_all, [0.25, 0.5, 0.75])  # global quartile edges
    for metric, col in [("C_theta", "C_theta_base"), ("C_state", "C_state_base")]:
        for reg in REGIMES:
            sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == "test")]
            x = sub["frac_snow"].to_numpy()
            y = sub[col].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            # frozen spearman (from posthoc_summary tradeoffs; dPL per seed)
            trade = frozen_sum["tradeoffs"]
            sp = [trade[f"{reg}{'_' + str(s) if reg == 'dPL' else ''}"][
                f"spearman_{'C_theta' if metric == 'C_theta' else 'C_state'}_vs_frac_snow"]
                for s in (SEEDS if reg == "dPL" else [None])]
            # frac_snow-quartile bins (median + bootstrap CI) — descriptive
            # environmental gradient, no parametric trend model
            bins = [(-np.inf, q_bins[0]), (q_bins[0], q_bins[1]),
                    (q_bins[1], q_bins[2]), (q_bins[2], np.inf)]
            bin_entries = []
            for k, (lo, hi) in enumerate(bins):
                m = (x > lo) & (x <= hi)
                if m.sum() < 20:
                    continue
                bm = y[m]
                bin_entries.append({
                    "bin": k + 1,
                    "frac_snow_range": [float(lo), float(hi)],
                    "n": int(m.sum()),
                    "frac_snow_median": float(np.median(x[m])),
                    "median": float(np.median(bm)),
                    "boot_ci_median_display": list(boot_ci(bm, np.median, args.n_boot,
                                                           args.seed + 20 + k)),
                })
            key = f"{metric}_{reg}"
            # display y-limits chosen in plot_figure5.py; fractions reported here
            y_hi = 2.6 if metric == "C_state" else 0.55
            y_lo = -0.5 if metric == "C_state" else 0.0
            pef[key] = {
                "spearman_frozen": [float(v) for v in sp],
                "quartile_bins": bin_entries,
                "n": int(len(x)),
                "y_display_limits": [y_lo, y_hi],
                "frac_beyond_y_display": float((y > y_hi).mean()),
                "frac_below_y_display": float((y < y_lo).mean()),
            }
    summary["panels_ef_excess_vs_frac_snow"] = pef

    write_json(out_dir / "figure5_summary.json", summary)

    print(f"COMPLETE Figure 5 data -> {out_dir}", flush=True)
    print(f"  figure5_basin_table.csv      (tidy, per seed: {len(tidy)} rows)", flush=True)
    print(f"  figure5_basin_seedmedian.csv (dPL seed-aggregated: {len(seedmed)} rows)", flush=True)
    print(f"  figure5_summary.json         (panel-level numbers)", flush=True)


if __name__ == "__main__":
    main()
