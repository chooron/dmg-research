#!/usr/bin/env python3
"""R3 reviewer-proofing validation statistics (read-only post-processing).

Consumes ONLY the completed post-hoc package
(results/r3_misspec_analysis_v1/posthoc_*.csv) plus the frozen protocol and
truth basin list.  No training, no truth regeneration, no protocol changes,
no modification of existing outputs.

Validations:

V1. Output-recovery / internal-cost association vs the shared snow gradient:
    raw and partial Spearman (controlling frac_snow) for
    G_base<->C_theta[Base], G_base<->C_state[Base],
    F_close<->C_theta[Base], F_close<->C_state[Base];
    within-frac_snow-quartile Spearman (sign consistency).
    G_base is primary (F_close unstable for small denominators).
V2. Train->test generalization decay:
    decay_G_base, decay_F_close, decay_G_tgd2, decay_F_tgd2
    (train - test, per basin), with bootstrap CI, frac>0, frac_snow
    association and quartiles.
V3. TGD2 internal-distortion reduction:
    R_theta_tgd2 = C_theta[Base]-C_theta[TGD2],
    R_state_tgd2 = C_state[Base]-C_state[TGD2];
    raw + partial Spearman vs G_tgd2 / F_tgd2.
V4. Residual explicit-process advantage:
    G_CN_over_TGD2 = KGE_CN - KGE_TGD2,
    F_explicit_residual = (KGE_CN-KGE_TGD2)/(KGE_CN-KGE_Base)
    (same denominator rule as F_tgd2, unclipped);
    algebraic check vs 1 - F_tgd2; process-conditioned support using the
    existing posthoc_process_errors.csv (CN-TGD2 RMSE gap on snow-active vs
    complementary days).
V5. Sanity / negative controls: frac_snow==0 subgroup (descriptive);
    process-conditioned ordering Base vs CN vs TGD2; KGE-component
    contribution decomposition (linearized dKGE per component) for
    recalibration (fitted Base vs no-refit) and TGD2 mitigation (TGD2 vs Base).

Statistical rules: basin is the unit; fixed-seed paired basin bootstrap,
2000 replicates, seed 20260730; dPL seed-specific with no pooling of
531x3 rows; F-ratios use the documented denominator rule (> 1e-6) and are
not clipped.

Outputs (results/r3_misspec_analysis_v1/):
  posthoc_validation_partial.csv     raw+partial spearman per pair (with CIs)
  posthoc_validation_quartiles.csv   within-quartile spearman
  posthoc_validation_decay.csv       per-basin decay metrics
  posthoc_validation_tgd2_reduction.csv  per-basin R_theta/R_state
  posthoc_validation_residual.csv    per-basin G_CN_over_TGD2/F_explicit_residual
  posthoc_validation_summary.json    all statistics + S1..S6 determination
  R3_VALIDATION_REPORT.md            concise report (generated separately)

Usage: python r3/posthoc_validation.py [--self-test] [--n-boot 2000] [--seed 20260730]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[0]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from r3.common import DEFAULT_RESULTS_ROOT, git_commit, write_json  # noqa: E402

DENOM_TOL = 1e-6
PRIMARY_PAIRS = [
    ("G_base", "C_theta_primary"),
    ("G_base", "C_state_primary"),
    ("F_close", "C_theta_primary"),
    ("F_close", "C_state_primary"),
]


def rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(np.argsort(x))
    return order.astype(np.float64) + 1.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5 or x[valid].std() == 0 or y[valid].std() == 0:
        return float("nan")
    rx = rankdata(x[valid])
    ry = rankdata(y[valid])
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Rank-transform, regress each on rank(z), correlate residuals (Pearson)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 8:
        return float("nan")
    rx = rankdata(x[valid])
    ry = rankdata(y[valid])
    rz = rankdata(z[valid])
    if rz.std() == 0 or rx.std() == 0 or ry.std() == 0:
        return float("nan")

    def resid(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        A = np.vstack([v, np.ones_like(v)]).T
        coef, *_ = np.linalg.lstsq(A, u, rcond=None)
        return u - A @ coef

    ex = resid(rx, rz)
    ey = resid(ry, rz)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def boot_ci(values: np.ndarray, stat_fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn(values[idx])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def boot_corr_ci(x, y, fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = fn(x[idx], y[idx])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def boot_partial_ci(x, y, z, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = partial_spearman(x[idx], y[idx], z[idx])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def self_test() -> None:
    """Synthetic sanity checks for partial_spearman."""
    rng = np.random.default_rng(12345)
    n = 400
    z = rng.uniform(0, 1, n)
    x = 0.8 * z + 0.2 * rng.uniform(0, 1, n)  # x confounded with z
    y1 = 0.5 * x + 0.5 * z + 0.1 * rng.normal(size=n)  # partial > 0
    y2 = 0.0 * x + 1.0 * z + 0.05 * rng.normal(size=n)  # partial ~ 0
    y3 = (
        1.0 * x + 0.0 * z + 0.1 * rng.normal(size=n)
    )  # partial < raw (x itself confounded)
    raw1, part1 = spearman(x, y1), partial_spearman(x, y1, z)
    raw2, part2 = spearman(x, y2), partial_spearman(x, y2, z)
    raw3, part3 = spearman(x, y3), partial_spearman(x, y3, z)
    print(
        f"[selftest] confounded: raw {raw1:+.3f} partial {part1:+.3f} (expect partial<raw, >0)"
    )
    print(
        f"[selftest] z-driven : raw {raw2:+.3f} partial {part2:+.3f} (expect partial ~0)"
    )
    print(
        f"[selftest] x-driven : raw {raw3:+.3f} partial {part3:+.3f} (expect partial ~raw)"
    )
    ok = (0 < part1 < raw1) and (abs(part2) < 0.12) and (0 < part3 < raw3)
    if not ok:
        raise SystemExit(f"SELFTEST FAIL: {part1:.3f} {part2:.3f} {part3:.3f}")
    print("[selftest] PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default="r3_misspec_analysis_v1")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    out_dir = args.results_root / args.run_id
    bt = pd.read_csv(out_dir / "posthoc_basin_table.csv")
    tc = pd.read_csv(out_dir / "posthoc_theta_cost.csv")
    sc = pd.read_csv(out_dir / "posthoc_state_cost.csv")
    comp = pd.read_csv(out_dir / "posthoc_components.csv")
    proc = pd.read_csv(out_dir / "posthoc_process_errors.csv")
    bt["basin_id"] = bt["basin_id"].astype(str).str.zfill(8)
    tc["basin_id"] = tc["basin_id"].astype(str).str.zfill(8)
    sc["basin_id"] = sc["basin_id"].astype(str).str.zfill(8)
    SEEDS = (42, 123, 2026)

    summary: dict = {}
    partial_rows, quart_rows, decay_rows, red_rows, resid_rows = [], [], [], [], []

    def sel(paradigm, seed, period):
        s = bt[(bt["paradigm"] == paradigm) & (bt["period"] == period)]
        if seed is None:
            return s[s["seed"].isna()].set_index("basin_id")
        return s[s["seed"] == seed].set_index("basin_id")

    def cost(paradigm, seed, structure, metric):
        c = tc if metric.startswith("C_theta") else sc
        c = c[(c["paradigm"] == paradigm) & (c["structure"] == structure)]
        if seed is None:
            c = c[c["seed"].isna()]
        else:
            c = c[c["seed"] == seed]
        return c.set_index("basin_id")[metric]

    # ---------------- V1: raw + partial Spearman ----------------
    for paradigm, seeds in [("IC", [None]), ("dPL", list(SEEDS))]:
        for period in ("test", "train"):
            for seed in seeds:
                key = (
                    f"{paradigm}{'_' + str(seed) if seed is not None else ''}_{period}"
                )
                b = sel(paradigm, seed, period)
                ct = cost(paradigm, seed, "Base", "C_theta_primary")
                cs = cost(paradigm, seed, "Base", "C_state_primary")
                fs = b["frac_snow"]
                for xname, cname in PRIMARY_PAIRS:
                    x = b[xname].to_numpy()
                    c = (
                        (ct if cname == "C_theta_primary" else cs)
                        .reindex(b.index)
                        .to_numpy()
                    )
                    ok = np.isfinite(x) & np.isfinite(c) & np.isfinite(fs.to_numpy())
                    if ok.sum() < 20:
                        continue
                    raw = spearman(x[ok], c[ok])
                    part = partial_spearman(x[ok], c[ok], fs.to_numpy()[ok])
                    raw_ci = boot_corr_ci(
                        x[ok], c[ok], spearman, args.n_boot, args.seed
                    )
                    part_ci = boot_partial_ci(
                        x[ok], c[ok], fs.to_numpy()[ok], args.n_boot, args.seed
                    )
                    partial_rows.append(
                        {
                            "paradigm": paradigm,
                            "seed": "" if seed is None else seed,
                            "period": period,
                            "pair": f"{xname}|{cname}",
                            "n": int(ok.sum()),
                            "raw_spearman": raw,
                            "raw_ci_lo": raw_ci[0],
                            "raw_ci_hi": raw_ci[1],
                            "partial_spearman": part,
                            "partial_ci_lo": part_ci[0],
                            "partial_ci_hi": part_ci[1],
                        }
                    )
                # within-quartile Spearman (G_base primary; also C-state)
                x = b["G_base"].to_numpy()
                c = ct.reindex(b.index).to_numpy()
                csn = cs.reindex(b.index).to_numpy()
                fsv = fs.to_numpy()
                qs = np.quantile(fsv, [0.25, 0.5, 0.75])
                bins = [
                    (-np.inf, qs[0]),
                    (qs[0], qs[1]),
                    (qs[1], qs[2]),
                    (qs[2], np.inf),
                ]
                for qi, (lo, hi) in enumerate(bins):
                    m = (fsv > lo) & (fsv <= hi)
                    for cname, cv in [("C_theta", c), ("C_state", csn)]:
                        mm = m & np.isfinite(x) & np.isfinite(cv)
                        if mm.sum() < 20:
                            continue
                        quart_rows.append(
                            {
                                "paradigm": paradigm,
                                "seed": "" if seed is None else seed,
                                "period": period,
                                "quartile": qi + 1,
                                "frac_snow_range": [float(lo), float(hi)],
                                "n": int(mm.sum()),
                                "pair": f"G_base|{cname}",
                                "spearman": spearman(x[mm], cv[mm]),
                            }
                        )
    pd.DataFrame(partial_rows).to_csv(
        out_dir / "posthoc_validation_partial.csv", index=False
    )
    pd.DataFrame(quart_rows).to_csv(
        out_dir / "posthoc_validation_quartiles.csv", index=False
    )

    # ---------------- V2: decay train - test ----------------
    for paradigm, seeds in [("IC", [None]), ("dPL", list(SEEDS))]:
        for seed in seeds:
            tr = sel(paradigm, seed, "train")
            te = sel(paradigm, seed, "test")
            idx = tr.index.intersection(te.index)
            for dname, col in [
                ("decay_G_base", "G_base"),
                ("decay_F_close", "F_close"),
                ("decay_G_tgd2", "G_tgd2"),
                ("decay_F_tgd2", "F_tgd2"),
            ]:
                d = tr.loc[idx, col].to_numpy() - te.loc[idx, col].to_numpy()
                fs = te.loc[idx, "frac_snow"].to_numpy()
                valid = np.isfinite(d)
                dv = d[valid]
                lo, hi = boot_ci(dv, np.median, args.n_boot, args.seed)
                lo_m, hi_m = boot_ci(dv, np.mean, args.n_boot, args.seed + 3)
                for i, b in enumerate(idx):
                    decay_rows.append(
                        {
                            "basin_id": b,
                            "paradigm": paradigm,
                            "seed": "" if seed is None else seed,
                            "metric": dname,
                            "decay": d[i],
                            "frac_snow": fs[i],
                        }
                    )
                summary[
                    f"{paradigm}{'_' + str(seed) if seed is not None else ''}_{dname}"
                ] = {
                    "n": int(len(idx)),
                    "n_valid": int(valid.sum()),
                    "median": float(np.median(dv)),
                    "mean": float(np.mean(dv)),
                    "q25": float(np.quantile(dv, 0.25)),
                    "q75": float(np.quantile(dv, 0.75)),
                    "boot_ci_median": [lo, hi],
                    "boot_ci_mean": [lo_m, hi_m],
                    "frac_gt_0": float((dv > 0).mean()),
                    "spearman_vs_frac_snow": spearman(
                        fs[np.isfinite(d)], d[np.isfinite(d)]
                    ),
                }
    pd.DataFrame(decay_rows).to_csv(
        out_dir / "posthoc_validation_decay.csv", index=False
    )

    # ---------------- V3: TGD2 internal reduction ----------------
    for paradigm, seeds in [("IC", [None]), ("dPL", list(SEEDS))]:
        for seed in seeds:
            ct_b = cost(paradigm, seed, "Base", "C_theta_primary")
            ct_t = cost(paradigm, seed, "TGD2", "C_theta_primary")
            cs_b = cost(paradigm, seed, "Base", "C_state_primary")
            cs_t = cost(paradigm, seed, "TGD2", "C_state_primary")
            b = sel(paradigm, seed, "test")
            idx = b.index.intersection(ct_b.index).intersection(ct_t.index)
            r_theta = (ct_b.reindex(idx) - ct_t.reindex(idx)).to_numpy()
            r_state = (cs_b.reindex(idx) - cs_t.reindex(idx)).to_numpy()
            fs = b.loc[idx, "frac_snow"].to_numpy()
            g_tgd2 = b.loc[idx, "G_tgd2"].to_numpy()
            f_tgd2 = b.loc[idx, "F_tgd2"].to_numpy()
            key = f"{paradigm}{'_' + str(seed) if seed is not None else ''}"
            summary.setdefault("V3", {})[key] = {
                "n": int(len(idx)),
                "R_theta_tgd2": {
                    "median": float(np.median(r_theta)),
                    "mean": float(np.mean(r_theta)),
                    "q25": float(np.quantile(r_theta, 0.25)),
                    "q75": float(np.quantile(r_theta, 0.75)),
                    "boot_ci_median": boot_ci(
                        r_theta, np.median, args.n_boot, args.seed + 5
                    ),
                    "frac_gt_0": float((r_theta > 0).mean()),
                    "spearman_vs_frac_snow": spearman(fs, r_theta),
                },
                "R_state_tgd2": {
                    "median": float(np.median(r_state)),
                    "mean": float(np.mean(r_state)),
                    "q25": float(np.quantile(r_state, 0.25)),
                    "q75": float(np.quantile(r_state, 0.75)),
                    "boot_ci_median": boot_ci(
                        r_state, np.median, args.n_boot, args.seed + 6
                    ),
                    "frac_gt_0": float((r_state > 0).mean()),
                    "spearman_vs_frac_snow": spearman(fs, r_state),
                },
            }
            for gname, gv, fv, rv, rlabel in [
                ("G_tgd2", g_tgd2, f_tgd2, r_theta, "R_theta"),
                ("G_tgd2", g_tgd2, f_tgd2, r_state, "R_state"),
            ]:
                fok = np.isfinite(fv)
                summary["V3"][key][f"spearman_{gname}_vs_{rlabel}"] = spearman(gv, rv)
                summary["V3"][key][f"partial_{gname}_vs_{rlabel}"] = partial_spearman(
                    gv, rv, fs
                )
                summary["V3"][key][f"boot_partial_{gname}_vs_{rlabel}"] = (
                    boot_partial_ci(gv, rv, fs, args.n_boot, args.seed + 7)
                )
                summary["V3"][key][f"spearman_F_tgd2_vs_{rlabel}"] = spearman(
                    fv[fok], rv[fok]
                )
                summary["V3"][key][f"partial_F_tgd2_vs_{rlabel}"] = partial_spearman(
                    fv[fok], rv[fok], fs[fok]
                )
            for i, b in enumerate(idx):
                red_rows.append(
                    {
                        "basin_id": b,
                        "paradigm": paradigm,
                        "seed": "" if seed is None else seed,
                        "R_theta_tgd2": r_theta[i],
                        "R_state_tgd2": r_state[i],
                        "G_tgd2": g_tgd2[i],
                        "F_tgd2": f_tgd2[i],
                        "frac_snow": fs[i],
                    }
                )
    pd.DataFrame(red_rows).to_csv(
        out_dir / "posthoc_validation_tgd2_reduction.csv", index=False
    )

    # ---------------- V4: residual CN-over-TGD2 ----------------
    for paradigm, seeds in [("IC", [None]), ("dPL", list(SEEDS))]:
        for period in ("test", "train"):
            for seed in seeds:
                b = sel(paradigm, seed, period)
                g_cn_tgd2 = b["kge_cn"].to_numpy() - b["kge_tgd2"].to_numpy()
                denom = b["kge_cn"].to_numpy() - b["kge_base"].to_numpy()
                f_res = np.where(
                    denom > DENOM_TOL,
                    g_cn_tgd2 / np.where(denom > DENOM_TOL, denom, np.nan),
                    np.nan,
                )
                # algebraic consistency: F_explicit_residual == 1 - F_tgd2 where both valid
                f_tgd2 = b["F_tgd2"].to_numpy()
                both = np.isfinite(f_res) & np.isfinite(f_tgd2)
                maxdiff = (
                    float(np.max(np.abs(f_res[both] - (1.0 - f_tgd2[both]))))
                    if both.any()
                    else float("nan")
                )
                fs = b["frac_snow"].to_numpy()
                key = (
                    f"{paradigm}{'_' + str(seed) if seed is not None else ''}_{period}"
                )
                summary.setdefault("V4", {})[key] = {
                    "n": int(len(b)),
                    "G_CN_over_TGD2": {
                        "median": float(np.median(g_cn_tgd2)),
                        "q25": float(np.quantile(g_cn_tgd2, 0.25)),
                        "q75": float(np.quantile(g_cn_tgd2, 0.75)),
                        "boot_ci_median": boot_ci(
                            g_cn_tgd2, np.median, args.n_boot, args.seed + 9
                        ),
                        "frac_gt_0": float((g_cn_tgd2 > 0).mean()),
                        "spearman_vs_frac_snow": spearman(fs, g_cn_tgd2),
                    },
                    "F_explicit_residual": {
                        "median": float(np.nanmedian(f_res)),
                        "n_valid_denominator": int(np.isfinite(f_res).sum()),
                        "n_excluded": int((~np.isfinite(f_res)).sum()),
                        "frac_gt_0": float((f_res[np.isfinite(f_res)] > 0).mean())
                        if np.isfinite(f_res).any()
                        else np.nan,
                    },
                    "algebraic_maxdiff_vs_1_minus_F_tgd2": maxdiff,
                }
                for i, b_id in enumerate(b.index):
                    resid_rows.append(
                        {
                            "basin_id": b_id,
                            "paradigm": paradigm,
                            "seed": "" if seed is None else seed,
                            "period": period,
                            "G_CN_over_TGD2": g_cn_tgd2[i],
                            "F_explicit_residual": f_res[i],
                            "frac_snow": fs[i],
                        }
                    )
    pd.DataFrame(resid_rows).to_csv(
        out_dir / "posthoc_validation_residual.csv", index=False
    )

    # quartiles for G_CN_over_TGD2 (test)
    for paradigm, seeds in [("IC", [None]), ("dPL", list(SEEDS))]:
        for seed in seeds:
            b = sel(paradigm, seed, "test")
            fs = b["frac_snow"].to_numpy()
            g = b["kge_cn"].to_numpy() - b["kge_tgd2"].to_numpy()
            qs = np.quantile(fs, [0.25, 0.5, 0.75])
            bins = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]
            summary.setdefault("V4_quartiles", {})[
                f"{paradigm}{'_' + str(seed) if seed is not None else ''}"
            ] = [
                {
                    "frac_snow_range": [float(lo), float(hi)],
                    "n": int(((fs > lo) & (fs <= hi)).sum()),
                    "G_CN_over_TGD2_median": float(
                        np.median(g[(fs > lo) & (fs <= hi)])
                    ),
                }
                for lo, hi in bins
            ]

    # process-conditioned support: CN-TGD2 RMSE gap on snow-active vs complementary
    proc_support = {}
    for paradigm, fits in [
        ("IC", ("CN_IC", "TGD2_IC")),
        ("dPL", ("CN_dPL_s42", "TGD2_dPL_s42")),
    ]:
        for cond in ("snow_active", "melt_active", "no_snow_active"):
            a = proc[
                (proc["fit"] == fits[0])
                & (proc["period"] == "test")
                & (proc["condition"] == cond)
            ]
            b_ = proc[
                (proc["fit"] == fits[1])
                & (proc["period"] == "test")
                & (proc["condition"] == cond)
            ]
            if a.empty or b_.empty:
                continue
            gap = b_["rmse"].median() - a["rmse"].median()
            proc_support[f"{paradigm}|{cond}"] = {
                "CN_rmse_median": float(a["rmse"].median()),
                "TGD2_rmse_median": float(b_["rmse"].median()),
                "TGD2_minus_CN_rmse": float(gap),
            }
    summary["V4_process_support"] = proc_support

    # ---------------- V5: sanity ----------------
    v5 = {}
    # no-snow subgroup (test), from existing basin table
    for paradigm, seeds in [("IC", [None]), ("dPL", list(SEEDS))]:
        for seed in seeds:
            b = sel(paradigm, seed, "test")
            m = b["frac_snow"] <= 1e-6
            v5[f"{paradigm}{'_' + str(seed) if seed is not None else ''}"] = {
                "n": int(m.sum()),
                "kge_base_no_refit_median": float(
                    np.median(b.loc[m, "kge_base_no_refit"])
                ),
                "kge_cn_median": float(np.median(b.loc[m, "kge_cn"])),
                "kge_tgd2_median": float(np.median(b.loc[m, "kge_tgd2"])),
                "kge_base_median": float(np.median(b.loc[m, "kge_base"])),
                "G_base_median": float(np.median(b.loc[m, "G_base"])),
                "G_CN_over_TGD2_median": float(
                    np.median(b.loc[m, "kge_cn"] - b.loc[m, "kge_tgd2"])
                ),
            }
    # component contribution decomposition (test), linearized dKGE per component
    comp_summary = {}
    for pair, (fit_a, fit_b) in [
        ("recalibration", ("Base_no_refit", "Base_IC")),
        ("tgd2_mitigation", ("Base_IC", "TGD2_IC")),
    ]:
        a = comp[(comp["fit"] == fit_a) & (comp["period"] == "test")].set_index(
            "basin_id"
        )
        b_ = comp[(comp["fit"] == fit_b) & (comp["period"] == "test")].set_index(
            "basin_id"
        )
        idx = a.index.intersection(b_.index)
        contrib = {c: np.full(len(idx), np.nan) for c in ("r", "alpha", "beta")}
        dk = np.full(len(idx), np.nan)
        for i, b_id in enumerate(idx):
            ra, aa, ba = a.loc[b_id, ["r", "alpha", "beta"]].to_numpy()
            rb, ab_, bb_ = b_.loc[b_id, ["r", "alpha", "beta"]].to_numpy()
            if not np.isfinite([ra, aa, ba, rb, ab_, bb_]).all():
                continue
            D = np.sqrt((ra - 1) ** 2 + (aa - 1) ** 2 + (ba - 1) ** 2)
            if D < 1e-12:
                continue
            dk[i] = float(b_.loc[b_id, "kge"] - a.loc[b_id, "kge"])
            contrib["r"][i] = -((ra - 1) / D) * (rb - ra)
            contrib["alpha"][i] = -((aa - 1) / D) * (ab_ - aa)
            contrib["beta"][i] = -((ba - 1) / D) * (bb_ - ba)
        ok = np.isfinite(dk)
        comp_summary[pair] = {
            "n": int(ok.sum()),
            "dKGE_median": float(np.median(dk[ok])),
            "contrib_r_median": float(np.median(contrib["r"][ok])),
            "contrib_alpha_median": float(np.median(contrib["alpha"][ok])),
            "contrib_beta_median": float(np.median(contrib["beta"][ok])),
            "contrib_sum_reconstruction_max_abs_diff": float(
                np.max(
                    np.abs(
                        (contrib["r"][ok] + contrib["alpha"][ok] + contrib["beta"][ok])
                        - dk[ok]
                    )
                )
            ),
        }
    v5["component_contributions"] = comp_summary
    summary["V5"] = v5

    # ---------------- S1..S6 determination ----------------
    def get(path, default=None):
        node = summary
        for k in path.split("."):
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    s = {}
    # Established statistics (F_close / G_tgd2 medians) come from the existing
    # posthoc_summary.json (previous pipeline step), not recomputed here.
    prev = json.loads((out_dir / "posthoc_summary.json").read_text())

    def prev_get(regime_period, key):
        e = prev.get(regime_period)
        if e is None:
            return None
        v = e.get(key)
        return v.get("median") if isinstance(v, dict) else v

    s["S1_limited_calibration_compensation"] = {
        "rule": "F_close median (test) < 0.25 in both regimes (from posthoc_summary.json)",
        "value": {
            k: prev_get(f"{k}_test", "F_close")
            for k in ["IC", "dPL_42", "dPL_123", "dPL_2026"]
        },
    }
    s["S1_supported"] = all(
        v is not None and v < 0.25
        for v in s["S1_limited_calibration_compensation"]["value"].values()
    )
    s["S2_limited_generalization"] = {
        "rule": "median decay_G_base > 0 (train > test) in both regimes",
        "value": {
            k: get(f"{k}.median")
            for k in [
                "IC_decay_G_base",
                "dPL_42_decay_G_base",
                "dPL_123_decay_G_base",
                "dPL_2026_decay_G_base",
            ]
        },
    }
    s["S2_supported"] = all(
        v is not None and v > 0
        for v in s["S2_limited_generalization"]["value"].values()
    )
    # S3: Base excess cost relative to CN positive
    # S3: Base excess cost relative to CN positive (verified from cost tables)
    s3_vals = {}
    for paradigm, seeds in [("IC", [None]), ("dPL", list(SEEDS))]:
        for seed in seeds:
            k = f"{paradigm}{'_' + str(seed) if seed is not None else ''}"
            ct = cost(paradigm, seed, "Base", "C_theta_primary")
            cs = cost(paradigm, seed, "Base", "C_state_primary")
            s3_vals[k] = {
                "C_theta_Base_median": float(np.nanmedian(ct.to_numpy())),
                "C_state_Base_median": float(np.nanmedian(cs.to_numpy())),
            }
    s["S3_internal_reorganization"] = {
        "rule": "median C_theta[Base] > 0 and median C_state[Base] > 0 (test)",
        "values": s3_vals,
    }
    s["S3_supported"] = all(
        v["C_theta_Base_median"] > 0 and v["C_state_Base_median"] > 0
        for v in s3_vals.values()
    )
    # S4: partial G_base<->C_theta and <->C_state controlling frac_snow (test)
    part = pd.DataFrame(partial_rows)
    s4 = {}
    for paradigm in ("IC", "dPL"):
        sub = part[(part["paradigm"] == paradigm) & (part["period"] == "test")]
        for pair in ("G_base|C_theta_primary", "G_base|C_state_primary"):
            r = sub[sub["pair"] == pair]
            if r.empty:
                continue
            # aggregate across seeds for dPL: report range of partial rho
            s4[f"{paradigm}|{pair}"] = {
                "partial_rho": [float(x) for x in r["partial_spearman"]],
                "partial_ci_lo": [float(x) for x in r["partial_ci_lo"]],
                "partial_ci_hi": [float(x) for x in r["partial_ci_hi"]],
                "raw_rho": [float(x) for x in r["raw_spearman"]],
            }
    s["S4_output_internal_tradeoff"] = s4
    # S4 supported iff the G_base|C_theta partial CI excludes 0 (positive) in
    # every regime/seed (test).  Primary pair per the task.
    s4_supported = True
    for paradigm in ("IC", "dPL"):
        sub = part[
            (part["paradigm"] == paradigm)
            & (part["period"] == "test")
            & (part["pair"] == "G_base|C_theta_primary")
        ]
        for _, row in sub.iterrows():
            if not (row["partial_ci_lo"] > 0.0):
                s4_supported = False
    s["S4_supported"] = s4_supported
    s["S4_interpretation"] = (
        "raw association driven by the shared snow gradient; partial association "
        "attenuates to ~0 (IC) or weak positive borderline (dPL)"
        if not s4_supported
        else "output recovery remains positively associated with internal cost beyond the snow gradient"
    )
    # within-quartile sign consistency (G_base|C_theta)
    q = pd.DataFrame(quart_rows)
    sq = {}
    for paradigm in ("IC", "dPL"):
        sub = q[
            (q["paradigm"] == paradigm)
            & (q["period"] == "test")
            & (q["pair"] == "G_base|C_theta")
        ]
        by_seed = sub.groupby("seed")
        sq[paradigm] = {}
        for seed, g in by_seed:
            pos = int((g["spearman"] > 0).sum())
            sq[paradigm][str(seed) if seed else "IC"] = {
                "n_quartiles_positive": pos,
                "n_quartiles": int(len(g)),
                "quartile_rhos": [float(x) for x in g["spearman"]],
            }
    s["S4_within_quartile_sign"] = sq
    # S5: TGD2 output improvement + internal reduction
    v3 = summary.get("V3", {})
    s5 = {}
    for paradigm, seeds in [("IC", [""]), ("dPL", ["42", "123", "2026"])]:
        for sd in seeds:
            k = f"{paradigm}_{sd}" if sd else "IC"
            e = v3.get(k, {})
            s5[k] = {
                "G_tgd2_median_test": prev_get(
                    f"{paradigm}{'_' + sd if sd else ''}_test", "G_tgd2"
                ),
                "R_theta_median": e.get("R_theta_tgd2", {}).get("median"),
                "R_state_median": e.get("R_state_tgd2", {}).get("median"),
                "frac_R_theta_gt_0": e.get("R_theta_tgd2", {}).get("frac_gt_0"),
                "frac_R_state_gt_0": e.get("R_state_tgd2", {}).get("frac_gt_0"),
            }
    s["S5_tgd2_mitigation"] = s5
    s["S5_supported"] = all(
        v.get("G_tgd2_median_test") is not None and v["G_tgd2_median_test"] > 0
        for v in s5.values()
    )
    s["S5_internal_reduction_supported"] = all(
        v.get("R_theta_median") is not None
        and v["R_theta_median"] > 0
        and v.get("R_state_median") is not None
        and v["R_state_median"] > 0
        for v in s5.values()
    )
    # S6: residual CN advantage + process support
    v4 = summary.get("V4", {})
    s6 = {}
    for paradigm, seeds in [("IC", [""]), ("dPL", ["42", "123", "2026"])]:
        for sd in seeds:
            k = f"{paradigm}{'_' + sd if sd else ''}_test"
            s6[k if sd else f"{paradigm}_test"] = {
                "G_CN_over_TGD2_median": v4.get(k, {})
                .get("G_CN_over_TGD2", {})
                .get("median"),
                "boot_ci": v4.get(k, {})
                .get("G_CN_over_TGD2", {})
                .get("boot_ci_median"),
                "frac_gt_0": v4.get(k, {}).get("G_CN_over_TGD2", {}).get("frac_gt_0"),
            }
    s["S6_residual_explicit_advantage"] = s6
    ps = summary.get("V4_process_support", {})
    snow_gap = ps.get("IC|snow_active", {}).get("TGD2_minus_CN_rmse")
    nosnow_gap = ps.get("IC|no_snow_active", {}).get("TGD2_minus_CN_rmse")
    s["S6_process_concentration"] = {
        "IC TGD2-CN rmse gap on snow_active": snow_gap,
        "IC TGD2-CN rmse gap on no_snow_active": nosnow_gap,
        "concentrated_on_snow_active": bool(
            snow_gap is not None and nosnow_gap is not None and snow_gap > nosnow_gap
        ),
    }

    summary["S"] = s
    summary["n_boot"] = args.n_boot
    summary["seed"] = args.seed
    summary["denom_tol"] = DENOM_TOL
    summary["protocol"] = "r3_posthoc_validation_v1"
    summary["frozen_protocol"] = "r3/protocol_misspec_v1.json"
    summary["code"] = git_commit(PROJECT)
    write_json(out_dir / "posthoc_validation_summary.json", summary)
    print(f"COMPLETE validation -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
