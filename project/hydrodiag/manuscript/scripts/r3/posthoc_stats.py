#!/usr/bin/env python3
"""R3 post-hoc statistics on the completed synthetic misspecification experiment.

Extends the canonical R3 analysis package (results/r3_misspec_analysis_v1/) with
the final compensation/mitigation statistics.  No training, no truth
regeneration, no protocol changes.  Frozen inputs:

- truth v1 (theta_star, q_star, x_star, snow_star)
- correct-CN gate: r3_gate_ic_xaj_cn_531_v1 (IC), r3_gate_dpl_xaj_cn_seed_<s>
- misspec fits: r3_misspec_ic_xaj_531_v1, r3_misspec_ic_xaj_tgd2_531_v1,
  r3_misspec_dpl_xaj_seed_<s>, r3_misspec_dpl_xaj_tgd2_seed_<s>
- Base-no-refit: r3_base_no_refit_v1
- paired parameter errors: paired_parameters.csv (from misspec_analysis.py)
- paired state excess: state_excess.csv (from misspec_states.py)
- frozen tiers: r3/protocol_misspec_v1.json

Statistics produced:

A. G_base = KGE_Base_fitted - KGE_Base_no_refit (per basin; train/test; IC and
   dPL seeds) + paired basin-bootstrap CI (2000 reps, fixed seed) + fractions.
   F_close = (KGE_Base_fitted - KGE_Base_no_refit)/(KGE_CN - KGE_Base_no_refit)
   with denominator rules (positive and > 1e-6), no clipping.
B. G_tgd2 = KGE_TGD2 - KGE_Base; F_tgd2 (same denominator rule); frac_snow
   association.
C. KGE component decomposition (r, alpha, beta) with the exact repository KGE
   semantics (same mask/floor as r3/common.standard_kge); reconstruction check.
D. C_theta = median over frozen primary params of |e_M - e_CN| (primary set;
   primary+secondary as robustness) -- from paired_parameters.csv.
E. C_state = median over frozen primary states of delta_NRMSE (test; train)
   -- from state_excess.csv.
F. frac_snow continuous Spearman associations + quartile distributions +
   frac_snow==0 descriptive subgroup (IC negative control; dPL note).
G. Process-conditioned output errors: MAE/RMSE/volume bias during truth
   snow-active (G>1e-6) and melt-active (melt>1e-6) days vs complementary days,
   from recorded-forward q vs q_star.
H. Uncertainty: fixed-seed nonparametric basin bootstrap, 2000 replicates.

Synthesis tables (IC, dPL) and an objective Pattern A/B/C determination.

Outputs (results/r3_misspec_analysis_v1/):
  posthoc_basin_table.csv   per basin x regime x seed x period
  posthoc_theta_cost.csv    C_theta per basin x regime x seed
  posthoc_state_cost.csv    C_state per basin x regime x seed
  posthoc_process_errors.csv
  posthoc_summary.json      all statistics + bootstrap + pattern
  R3_POSTHOC_REPORT.md      concise markdown report

Usage: python r3/posthoc_stats.py [--device cuda] [--n-boot 2000] [--seed 20260730]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.scripts.r3.common import (  # noqa: E402
    COMMON_XAJ,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    frac_snow_series,
    git_commit,
    period_indices,
    standard_kge,
    write_json,
)
from manuscript.scripts.r3.gate_analysis import load_dpl_estimates, load_ic_estimates  # noqa: E402
from manuscript.scripts.r3.recorded_forward import (  # noqa: E402
    build_forcing_dict,
    recorded_forward_for_structure,
)

STATE_KEYS = ["wu", "wl", "wd", "s", "fr", "qi", "qg"]
PRIMARY_STATES = ["wu", "wl", "s", "qi", "qg"]
BATCH = 50
DENOM_TOL = 1e-6  # documented numerical tolerance for F_close/F_tgd2 denominators


def load_protocol() -> dict:
    return json.loads((HERE / "protocol_misspec_v1.json").read_text())


def kge_components(sim: np.ndarray, obs: np.ndarray, min_valid: int = 30) -> dict:
    """Exact repository KGE semantics (r3/common.standard_kge) + components."""
    sim = np.asarray(sim, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(sim) & np.isfinite(obs) & (sim >= 0) & (obs >= 0)
    count = int(mask.sum())
    if count < min_valid:
        return {"r": np.nan, "alpha": np.nan, "beta": np.nan, "kge": np.nan}
    s = sim[mask]
    o = obs[mask]
    o_std = o.std()
    if o_std < 1e-10:
        return {"r": np.nan, "alpha": np.nan, "beta": np.nan, "kge": np.nan}
    r = float(np.corrcoef(s, o)[0, 1])
    alpha = float(s.std() / o_std)
    beta = float(s.mean() / o.mean())
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return {"r": r, "alpha": alpha, "beta": beta, "kge": kge}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5 or x[valid].std() == 0 or y[valid].std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[valid]))
    ry = np.argsort(np.argsort(y[valid]))
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_ci(values: np.ndarray, stat_fn, n_boot: int, seed: int, alpha: float = 0.05):
    """Paired basin-level bootstrap CI for a statistic."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn(values[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def boot_spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = spearman(x[idx], y[idx])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def recorded_q_fit(models, structure, theta_hat, bundle, device, dtype) -> np.ndarray:
    """Recorded forward (production kernels, canonical full axis) -> q [n,12418]."""
    from ablation.ic_core.parameter_adapter import get_parameter_spec

    names = tuple(get_parameter_spec(structure))
    n = theta_hat.shape[0]
    q_full = np.empty((n, bundle.forcing.shape[1]), dtype=np.float64)
    model = models[structure]
    for left in range(0, n, BATCH):
        right = min(n, left + BATCH)
        fc = build_forcing_dict(
            bundle.forcing[left:right].astype(np.float32), device, dtype
        )
        params = {
            name: torch.from_numpy(theta_hat[left:right, i]).to(device, dtype=dtype)
            for i, name in enumerate(names)
        }
        qsim, _stores, _fs = recorded_forward_for_structure(
            structure, model, fc, params, device, dtype
        )
        q_full[left:right] = qsim.detach().cpu().numpy().astype(np.float64)
    return q_full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default="r3_misspec_analysis_v1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--skip-forwards",
        action="store_true",
        help="Reuse cached posthoc_q_<fit>.npy if present (debug).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    RES = args.results_root
    truth_dir = RES / "r3_synthetic_truth_v1"
    out_dir = RES / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    proto = load_protocol()
    pri_ic = set(proto["predeclared_parameter_tiers"]["ic_primary"])
    sec_ic = set(proto["predeclared_parameter_tiers"]["ic_secondary_supporting"])
    pri_dpl = set(proto["predeclared_parameter_tiers"]["dpl_primary"])
    sec_dpl = set(proto["predeclared_parameter_tiers"]["dpl_secondary_supporting"])
    pri_states = list(proto["state_estimands"]["primary_common_variables"])
    SEEDS = (42, 123, 2026)

    theta_npz = np.load(truth_dir / "theta_star.npz")
    theta_star = theta_npz["parameters"]
    cn_names = [str(x) for x in theta_npz["parameter_names"]]
    basin_ids = [str(b).zfill(8) for b in theta_npz["basin_ids"]]
    q_star = np.asarray(
        np.load(truth_dir / "q_star.npz")["target_mm_day"], dtype=np.float64
    )
    snow = np.load(truth_dir / "snow_star.npz")

    from manuscript.scripts.r3.common import load_bundle

    bundle, _ = load_bundle(args.project_root, args.data_root)
    pi = period_indices(bundle)
    train_slice = slice(pi["train"][0], pi["train"][1] + 1)
    test_slice = slice(pi["test"][0], pi["test"][1] + 1)
    fs_map = frac_snow_series(bundle)
    frac_snow = np.array(
        [fs_map.set_index("basin_id")["frac_snow"][b] for b in basin_ids]
    )

    # ---------- official KGE sources ----------
    ic_runs = {
        "CN_IC": (RES / "r3_gate_ic_xaj_cn_531_v1", "XAJ_CN"),
        "Base_IC": (RES / "r3_misspec_ic_xaj_531_v1", "XAJ"),
        "TGD2_IC": (RES / "r3_misspec_ic_xaj_tgd2_531_v1", "XAJ_TGD2"),
    }
    dpl_runs = {}
    for struct, key in [("CN", "XAJ_CN"), ("Base", "XAJ"), ("TGD2", "XAJ_TGD2")]:
        for s in SEEDS:
            tag = (
                "cn" if struct == "CN" else ("xaj" if struct == "Base" else "xaj_tgd2")
            )
            d = (
                (RES / f"r3_gate_dpl_xaj_{tag}_seed_{s}")
                if struct == "CN"
                else (RES / f"r3_misspec_dpl_{tag}_seed_{s}")
            )
            dpl_runs[f"{struct}_dPL_s{s}"] = (
                d,
                "XAJ_CN"
                if struct == "CN"
                else ("XAJ" if struct == "Base" else "XAJ_TGD2"),
            )

    # official per-basin KGE: IC from raw JSON, dPL from basin_final_summary
    official: dict[
        str, dict[str, np.ndarray]
    ] = {}  # fit -> {"train": ..., "test": ...}
    for fit, (d, key) in ic_runs.items():
        est = load_ic_estimates(d, basin_ids)
        official[fit] = {
            "train": np.array(
                [est[b]["train_kge"] for b in basin_ids], dtype=np.float64
            ),
            "test": np.array([est[b]["test_kge"] for b in basin_ids], dtype=np.float64),
        }
    for fit, (d, key) in dpl_runs.items():
        summary = pd.read_csv(d / "basin_final_summary.csv")
        kge_by_basin = dict(
            zip(summary["basin_id"].astype(str).str.zfill(8), summary["val_kge"])
        )
        official[fit] = {
            "train": np.full(
                len(basin_ids), np.nan
            ),  # filled by recorded forward below
            "test": np.array(
                [kge_by_basin.get(b, np.nan) for b in basin_ids], dtype=np.float64
            ),
        }

    # base-no-refit official
    ref = pd.read_csv(RES / "r3_base_no_refit_v1" / "base_no_refit_basin_metrics.csv")
    ref["basin_id"] = ref["basin_id"].astype(str).str.zfill(8)
    ref_map = ref.set_index("basin_id")
    ref_train = np.array(
        [ref_map.loc[b, "kge_train"] for b in basin_ids], dtype=np.float64
    )
    ref_test = np.array(
        [ref_map.loc[b, "kge_test"] for b in basin_ids], dtype=np.float64
    )

    # ---------- recorded forwards (q) ----------
    from models import XAJLite, XAJWithCemaNeigeLite, XAJWithTGD2Lite

    models = {
        "XAJ": XAJLite().to(device).eval(),
        "XAJ_CN": XAJWithCemaNeigeLite().to(device).eval(),
        "XAJ_TGD2": XAJWithTGD2Lite().to(device).eval(),
    }

    fits_theta: dict[str, np.ndarray] = {}
    for fit, (d, key) in ic_runs.items():
        est = load_ic_estimates(d, basin_ids)
        fits_theta[fit] = np.stack([est[b]["theta_hat"] for b in basin_ids])
    for fit, (d, key) in dpl_runs.items():
        est = load_dpl_estimates(d, basin_ids)
        fits_theta[fit] = np.stack([est[b]["theta_hat"] for b in basin_ids])
    # Base-no-refit: shared XAJ params from theta* verbatim
    shared_idx = [cn_names.index(p) for p in COMMON_XAJ]
    fits_theta["Base_no_refit"] = theta_star[:, shared_idx]

    fit_structure = {fit: key for (fit, (d, key)) in {**ic_runs, **dpl_runs}.items()}
    fit_structure["Base_no_refit"] = "XAJ"

    q_store: dict[str, np.ndarray] = {}
    kge_rec: dict[str, dict] = {}
    for fit, theta_hat in fits_theta.items():
        cache = out_dir / f"posthoc_q_{fit}.npy"
        if args.skip_forwards and cache.exists():
            q = np.load(cache)
        else:
            print(f"[fwd] {fit} ({fit_structure[fit]}) ...", flush=True)
            q = recorded_q_fit(
                models, fit_structure[fit], theta_hat, bundle, device, dtype
            )
            np.save(cache, q)
        q_store[fit] = q
        kge_rec[fit] = {
            "train": np.array(
                [
                    kge_components(q[i, train_slice], q_star[i, train_slice])["kge"]
                    for i in range(len(basin_ids))
                ]
            ),
            "test": np.array(
                [
                    kge_components(q[i, test_slice], q_star[i, test_slice])["kge"]
                    for i in range(len(basin_ids))
                ]
            ),
        }

    # fill dPL train KGE from recorded forward
    for fit in dpl_runs:
        official[fit]["train"] = kge_rec[fit]["train"]

    # ---------- A/B: G_base, F_close, G_tgd2, F_tgd2 ----------
    rows = []
    regimes = [("IC", [None]), ("dPL", list(SEEDS))]
    for paradigm, seeds in regimes:
        for seed in seeds:
            tag = "" if paradigm == "IC" else f"_s{seed}"
            base_fit = f"Base_{paradigm}{tag}"
            tgd2_fit = f"TGD2_{paradigm}{tag}"
            cn_fit = f"CN_{paradigm}{tag}"
            for period, key in (("train", "train"), ("test", "test")):
                k_base = official[base_fit][key]
                k_tgd2 = official[tgd2_fit][key]
                k_cn = official[cn_fit][key]
                k_ref = ref_train if key == "train" else ref_test
                g_base = k_base - k_ref
                g_tgd2 = k_tgd2 - k_base
                denom_close = k_cn - k_ref
                denom_tgd2 = k_cn - k_base
                f_close = np.where(
                    denom_close > DENOM_TOL,
                    g_base / np.where(denom_close > DENOM_TOL, denom_close, np.nan),
                    np.nan,
                )
                f_tgd2 = np.where(
                    denom_tgd2 > DENOM_TOL,
                    g_tgd2 / np.where(denom_tgd2 > DENOM_TOL, denom_tgd2, np.nan),
                    np.nan,
                )
                for i, b in enumerate(basin_ids):
                    rows.append(
                        {
                            "basin_id": b,
                            "paradigm": paradigm,
                            "seed": seed if seed is not None else "",
                            "period": period,
                            "kge_base_no_refit": k_ref[i],
                            "kge_base": k_base[i],
                            "kge_tgd2": k_tgd2[i],
                            "kge_cn": k_cn[i],
                            "G_base": g_base[i],
                            "F_close": f_close[i],
                            "G_tgd2": g_tgd2[i],
                            "F_tgd2": f_tgd2[i],
                            "frac_snow": frac_snow[i],
                        }
                    )
    basin_tab = pd.DataFrame(rows)
    basin_tab.to_csv(out_dir / "posthoc_basin_table.csv", index=False)

    # ---------- C: components from recorded q ----------
    comp_rows = []
    comp_fits = ["CN_IC", "Base_IC", "TGD2_IC", "Base_no_refit"]
    comp_fits += [f for f in dpl_runs]
    for fit in comp_fits:
        q = q_store[fit]
        for period, sl in (("train", train_slice), ("test", test_slice)):
            comps = np.array(
                [
                    list(kge_components(q[i, sl], q_star[i, sl]).values())
                    for i in range(len(basin_ids))
                ]
            )
            for i, b in enumerate(basin_ids):
                comp_rows.append(
                    {
                        "basin_id": b,
                        "fit": fit,
                        "period": period,
                        "r": comps[i, 0],
                        "alpha": comps[i, 1],
                        "beta": comps[i, 2],
                        "kge": comps[i, 3],
                    }
                )
    comp_tab = pd.DataFrame(comp_rows)
    comp_tab.to_csv(out_dir / "posthoc_components.csv", index=False)

    # reconstruction check: kge from components == standard_kge (same series)
    maxdiff = 0.0
    for fit in comp_fits:
        q = q_store[fit]
        for period, sl in (("train", train_slice), ("test", test_slice)):
            for i in range(0, len(basin_ids), 17):
                sk = standard_kge(q[i, sl], q_star[i, sl])
                comp = kge_components(q[i, sl], q_star[i, sl])["kge"]
                if np.isfinite(sk) and np.isfinite(comp):
                    maxdiff = max(maxdiff, abs(sk - comp))
    print(f"[check] component-KGE vs standard_kge max diff: {maxdiff:.3e}", flush=True)

    # ---------- D: C_theta from paired_parameters.csv ----------
    pp = pd.read_csv(out_dir / "paired_parameters.csv")
    pp["basin_id"] = pp["basin_id"].astype(str).str.zfill(8)
    theta_cost_rows = []
    for paradigm, pri, sec in [("IC", pri_ic, sec_ic), ("dPL", pri_dpl, sec_dpl)]:
        for seed in [None] if paradigm == "IC" else SEEDS:
            if seed is None:
                sub = pp[(pp["paradigm"] == paradigm) & (pp["seed"].isna())]
            else:
                sub = pp[(pp["paradigm"] == paradigm) & (pp["seed"] == seed)]
            for struct in ["Base", "TGD2"]:
                ssub = sub[sub["structure"] == struct]
                for b in basin_ids:
                    bsub = ssub[ssub["basin_id"] == b]
                    if bsub.empty:
                        continue
                    d_abs = bsub["delta_e"].abs()  # |e_M - e_CN|
                    pri_set = set(pri)
                    c_pri = d_abs[bsub["parameter"].isin(pri_set)].median()
                    c_pri_sec = d_abs[
                        bsub["parameter"].isin(pri_set | set(sec))
                    ].median()
                    theta_cost_rows.append(
                        {
                            "basin_id": b,
                            "paradigm": paradigm,
                            "structure": struct,
                            "seed": seed if seed is not None else "",
                            "C_theta_primary": float(c_pri)
                            if np.isfinite(c_pri)
                            else np.nan,
                            "C_theta_primary_secondary": float(c_pri_sec)
                            if np.isfinite(c_pri_sec)
                            else np.nan,
                        }
                    )
    theta_tab = pd.DataFrame(theta_cost_rows)
    theta_tab.to_csv(out_dir / "posthoc_theta_cost.csv", index=False)

    # ---------- E: C_state from state_excess.csv ----------
    ex = pd.read_csv(out_dir / "state_excess.csv")
    ex["basin_id"] = ex["basin_id"].astype(str).str.zfill(8)
    state_cost_rows = []
    for paradigm in ["IC", "dPL"]:
        for seed in [None] if paradigm == "IC" else SEEDS:
            if seed is None:
                sub = ex[
                    (ex["paradigm"] == paradigm)
                    & (ex["metric"] == "nrmse")
                    & (ex["period"] == "test")
                    & (ex["seed"].isna())
                ]
            else:
                sub = ex[
                    (ex["paradigm"] == paradigm)
                    & (ex["metric"] == "nrmse")
                    & (ex["period"] == "test")
                    & (ex["seed"] == seed)
                ]
            for struct in ["Base", "TGD2"]:
                ssub = sub[sub["structure"] == struct]
                for b in basin_ids:
                    bsub = ssub[
                        (ssub["basin_id"] == b) & (ssub["variable"].isin(pri_states))
                    ]
                    c = bsub["delta_E"].median()
                    wd = ssub[(ssub["basin_id"] == b) & (ssub["variable"] == "wd")][
                        "delta_E"
                    ]
                    wt = ssub[(ssub["basin_id"] == b) & (ssub["variable"] == "wt")][
                        "delta_E"
                    ]
                    state_cost_rows.append(
                        {
                            "basin_id": b,
                            "paradigm": paradigm,
                            "structure": struct,
                            "seed": seed if seed is not None else "",
                            "C_state_primary": float(c) if np.isfinite(c) else np.nan,
                            "wd_delta_NRMSE": float(wd.iloc[0]) if len(wd) else np.nan,
                            "wt_delta_NRMSE": float(wt.iloc[0]) if len(wt) else np.nan,
                        }
                    )
    state_tab = pd.DataFrame(state_cost_rows)
    state_tab.to_csv(out_dir / "posthoc_state_cost.csv", index=False)

    # ---------- G: process-conditioned errors ----------
    snow_active = snow["G"] > 1e-6  # truth snow pack present (mm)
    melt_active = snow["melt"] > 1e-6  # truth melt flux (mm/d)
    proc_rows = []
    proc_fits = ["CN_IC", "Base_IC", "TGD2_IC", "Base_no_refit"] + [f for f in dpl_runs]
    for fit in proc_fits:
        q = q_store[fit]
        for period, sl in (("train", train_slice), ("test", test_slice)):
            idx = np.arange(bundle.forcing.shape[1])[sl]
            for cond_name, cond in [
                ("snow_active", snow_active[:, sl]),
                ("melt_active", melt_active[:, sl]),
                ("no_snow_active", ~snow_active[:, sl]),
            ]:
                for i, b in enumerate(basin_ids):
                    sim = q[i, sl][cond[i]]
                    obs = q_star[i, sl][cond[i]]
                    if len(sim) < 30:
                        continue
                    err = sim - obs
                    proc_rows.append(
                        {
                            "basin_id": b,
                            "fit": fit,
                            "period": period,
                            "condition": cond_name,
                            "n_days": int(len(sim)),
                            "mae": float(np.mean(np.abs(err))),
                            "rmse": float(np.sqrt(np.mean(err**2))),
                            "volume_bias_mm": float(err.sum()),
                        }
                    )
    proc_tab = pd.DataFrame(proc_rows)
    proc_tab.to_csv(out_dir / "posthoc_process_errors.csv", index=False)

    # ---------- statistics + bootstrap ----------
    rng_boot = args.seed
    summary: dict = {}
    for paradigm, seeds in regimes:
        for period in ("train", "test"):
            for seed in seeds:
                tag = f"_{seed}" if seed is not None else ""
                sub = basin_tab[
                    (basin_tab["paradigm"] == paradigm)
                    & (basin_tab["period"] == period)
                    & (basin_tab["seed"] == (seed if seed is not None else ""))
                ]
                g = sub["G_base"].to_numpy()
                f = sub["F_close"].to_numpy()
                gt = sub["G_tgd2"].to_numpy()
                ft = sub["F_tgd2"].to_numpy()
                fs = sub["frac_snow"].to_numpy()
                valid_f = f[np.isfinite(f)]
                valid_ft = ft[np.isfinite(ft)]

                def med(x):
                    return float(np.median(x)) if len(x) else float("nan")

                def mn(x):
                    return float(np.mean(x)) if len(x) else float("nan")

                lo_m, hi_m = boot_ci(g, np.median, args.n_boot, rng_boot)
                lo_M, hi_M = boot_ci(g, np.mean, args.n_boot, rng_boot + 1)
                entry = {
                    "n": int(len(sub)),
                    "G_base": {
                        "median": med(g),
                        "mean": mn(g),
                        "q25": float(np.quantile(g, 0.25)),
                        "q75": float(np.quantile(g, 0.75)),
                        "q10": float(np.quantile(g, 0.10)),
                        "q90": float(np.quantile(g, 0.90)),
                        "boot_ci_median": [lo_m, hi_m],
                        "boot_ci_mean": [lo_M, hi_M],
                        "frac_gt_0": float((g > 0).mean()),
                        "frac_le_0": float((g <= 0).mean()),
                    },
                    "F_close": {
                        "median": med(valid_f),
                        "mean": mn(valid_f),
                        "q25": float(np.quantile(valid_f, 0.25))
                        if len(valid_f)
                        else np.nan,
                        "q75": float(np.quantile(valid_f, 0.75))
                        if len(valid_f)
                        else np.nan,
                        "n_valid_denominator": int(len(valid_f)),
                        "n_excluded": int(len(f) - len(valid_f)),
                        "frac_gt_0": float((valid_f > 0).mean())
                        if len(valid_f)
                        else np.nan,
                        "frac_ge_0p5": float((valid_f >= 0.5).mean())
                        if len(valid_f)
                        else np.nan,
                        "frac_ge_1": float((valid_f >= 1.0).mean())
                        if len(valid_f)
                        else np.nan,
                    },
                    "G_tgd2": {
                        "median": med(gt),
                        "mean": mn(gt),
                        "q25": float(np.quantile(gt, 0.25)),
                        "q75": float(np.quantile(gt, 0.75)),
                        "boot_ci_median": boot_ci(
                            gt, np.median, args.n_boot, rng_boot + 2
                        ),
                        "frac_gt_0": float((gt > 0).mean()),
                    },
                    "F_tgd2": {
                        "median": med(valid_ft),
                        "mean": mn(valid_ft),
                        "n_valid_denominator": int(len(valid_ft)),
                        "n_excluded": int(len(ft) - len(valid_ft)),
                        "frac_gt_0": float((valid_ft > 0).mean())
                        if len(valid_ft)
                        else np.nan,
                        "frac_ge_0p5": float((valid_ft >= 0.5).mean())
                        if len(valid_ft)
                        else np.nan,
                    },
                    "frac_snow": {
                        "spearman_G_base": spearman(fs, g),
                        "spearman_F_close": spearman(
                            fs[np.isfinite(f)], f[np.isfinite(f)]
                        ),
                        "spearman_G_tgd2": spearman(fs, gt),
                        "spearman_F_tgd2": spearman(
                            fs[np.isfinite(ft)], ft[np.isfinite(ft)]
                        ),
                    },
                }
                summary[f"{paradigm}{tag}_{period}"] = entry

    # trade-offs D/E vs G_base/F_close/G_tgd2 + frac_snow
    trade = {}
    for paradigm, seeds in regimes:
        for seed in seeds:
            tag = f"_{seed}" if seed is not None else ""
            bt = basin_tab[
                (basin_tab["paradigm"] == paradigm)
                & (basin_tab["period"] == "test")
                & (basin_tab["seed"] == (seed if seed is not None else ""))
            ].set_index("basin_id")
            tc = theta_tab[
                (theta_tab["paradigm"] == paradigm)
                & (theta_tab["seed"] == (seed if seed is not None else ""))
            ]
            sc = state_tab[
                (state_tab["paradigm"] == paradigm)
                & (state_tab["seed"] == (seed if seed is not None else ""))
            ]
            ct_b = (
                tc[tc["structure"] == "Base"]
                .set_index("basin_id")
                .loc[bt.index, "C_theta_primary"]
                .to_numpy()
            )
            ct_t = (
                tc[tc["structure"] == "TGD2"]
                .set_index("basin_id")
                .loc[bt.index, "C_theta_primary"]
                .to_numpy()
            )
            cs_b = (
                sc[sc["structure"] == "Base"]
                .set_index("basin_id")
                .loc[bt.index, "C_state_primary"]
                .to_numpy()
            )
            cs_t = (
                sc[sc["structure"] == "TGD2"]
                .set_index("basin_id")
                .loc[bt.index, "C_state_primary"]
                .to_numpy()
            )
            g = bt["G_base"].to_numpy()
            f = bt["F_close"].to_numpy()
            gt = bt["G_tgd2"].to_numpy()
            fs = bt["frac_snow"].to_numpy()
            fok = np.isfinite(f)
            key = f"{paradigm}{tag}"
            trade[key] = {
                "n": int(len(bt)),
                "spearman_G_base_vs_C_theta": spearman(g, ct_b),
                "spearman_F_close_vs_C_theta": spearman(f[fok], ct_b[fok]),
                "spearman_G_base_vs_C_state": spearman(g, cs_b),
                "spearman_F_close_vs_C_state": spearman(f[fok], cs_b[fok]),
                "spearman_G_tgd2_vs_delta_C_theta": spearman(gt, ct_b - ct_t),
                "spearman_G_tgd2_vs_delta_C_state": spearman(gt, cs_b - cs_t),
                "spearman_C_theta_vs_frac_snow": spearman(fs, ct_b),
                "spearman_C_state_vs_frac_snow": spearman(fs, cs_b),
                "spearman_delta_C_theta_vs_frac_snow": spearman(fs, ct_b - ct_t),
                "spearman_delta_C_state_vs_frac_snow": spearman(fs, cs_b - cs_t),
            }
    summary["tradeoffs"] = trade

    # quartile + no-snow descriptive
    quart = {}
    for paradigm, seeds in regimes:
        for period in ("train", "test"):
            for seed in seeds:
                tag = f"_{seed}" if seed is not None else ""
                sub = basin_tab[
                    (basin_tab["paradigm"] == paradigm)
                    & (basin_tab["period"] == period)
                    & (basin_tab["seed"] == (seed if seed is not None else ""))
                ]
                fs = sub["frac_snow"].to_numpy()
                qs = np.quantile(fs, [0.25, 0.5, 0.75])
                bins = [
                    (-np.inf, qs[0]),
                    (qs[0], qs[1]),
                    (qs[1], qs[2]),
                    (qs[2], np.inf),
                ]
                qd = []
                for lo, hi in bins:
                    m = (fs > lo) & (fs <= hi)
                    qd.append(
                        {
                            "frac_snow_range": [float(lo), float(hi)],
                            "n": int(m.sum()),
                            "G_base_median": float(
                                np.median(sub["G_base"].to_numpy()[m])
                            ),
                            "F_close_median": float(
                                np.nanmedian(sub["F_close"].to_numpy()[m])
                            ),
                            "G_tgd2_median": float(
                                np.median(sub["G_tgd2"].to_numpy()[m])
                            ),
                        }
                    )
                quart[f"{paradigm}{tag}_{period}"] = qd
    summary["frac_snow_quartiles"] = quart

    nosnow = {}
    for paradigm, seeds in regimes:
        for seed in seeds:
            tag = f"_{seed}" if seed is not None else ""
            sub = basin_tab[
                (basin_tab["paradigm"] == paradigm)
                & (basin_tab["period"] == "test")
                & (basin_tab["seed"] == (seed if seed is not None else ""))
            ]
            m = sub["frac_snow"] <= 1e-6
            nosnow[f"{paradigm}{tag}"] = {
                "n": int(m.sum()),
                "G_base_median": float(np.median(sub.loc[m, "G_base"])),
                "F_close_median": float(np.nanmedian(sub.loc[m, "F_close"])),
                "G_tgd2_median": float(np.median(sub.loc[m, "G_tgd2"])),
            }
    summary["no_snow_subgroup"] = nosnow

    summary["n_basins"] = len(basin_ids)
    summary["n_boot"] = args.n_boot
    summary["boot_seed"] = args.seed
    summary["denom_tol"] = DENOM_TOL
    summary["protocol"] = "r3_posthoc_stats_v1"
    summary["frozen_protocol"] = "r3/protocol_misspec_v1.json"
    summary["code"] = git_commit(args.project_root)
    summary["component_reconstruction_max_abs_diff"] = maxdiff
    write_json(out_dir / "posthoc_summary.json", summary)

    print(f"COMPLETE posthoc stats -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
