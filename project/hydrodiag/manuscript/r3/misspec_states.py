#!/usr/bin/env python3
"""Misspecified-structure state/flux truth-error analysis (R3, frozen protocol).

Implements the frozen state estimands (r3/protocol_misspec_v1.json) for the
Base/TGD2 531-basin fits once the correct-CN gate is recovered:

- primary common variables wu, wl, s, qi, qg; secondary wd; derived wt
  (= wu + wl + wd); CN snow diagnostics are CN-only, not shared outcomes
- state metric set: RMSE, NRMSE = RMSE/(std(truth)+1e-8), temporal Pearson
  correlation, mean bias (per basin, per period train/test/full + fixed
  calendar seasons DJF/MAM/JJAS)
- paired excess cost delta_E[structure] = E[structure] - E[CN] within regime
  (IC best-restart; dPL seed-matched); frac_snow association reported on
  delta_E
- recorded-forward replay uses the repository production kernels
  (r3.recorded_forward.recorded_forward_for_structure) over the full axis;
  Base and TGD2 ignore the snow override by construction

Outputs under results/<run-id>/: state_metrics_basin.csv, state_excess.csv
(delta_E), state_summary.json (distributions + frac_snow regressions).

Usage: python manuscript/r3/misspec_states.py [--skip-ic|--skip-dpl] [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.r3.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    frac_snow_series,
    git_commit,
    period_indices,
    write_json,
)
from manuscript.r3.gate_analysis import (  # noqa: E402
    load_dpl_estimates,
    load_ic_estimates,
)
from manuscript.r3.recorded_forward import (  # noqa: E402
    build_forcing_dict,
    recorded_forward_for_structure,
)

STATE_KEYS = ["wu", "wl", "wd", "s", "fr", "qi", "qg"]
PRIMARY_STATES = ["wu", "wl", "s", "qi", "qg"]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJAS": [6, 7, 8, 9]}
BATCH = 50

STRUCTURE_KEY = {"Base": "XAJ", "TGD2": "XAJ_TGD2", "CN": "XAJ_CN"}


def state_metrics(sim: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    sim = np.asarray(sim, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((sim - truth) ** 2)))
    nrmse = float(rmse / (truth.std() + 1e-8))
    corr = float(np.corrcoef(sim, truth)[0, 1]) if truth.std() > 1e-12 else float("nan")
    bias = float(np.mean(sim - truth))
    return {"rmse": rmse, "nrmse": nrmse, "corr": corr, "bias": bias}


def build_models(device: torch.device) -> dict[str, object]:
    from models import XAJLite, XAJWithCemaNeigeLite, XAJWithTGD2Lite

    return {
        "XAJ": XAJLite().to(device).eval(),
        "XAJ_CN": XAJWithCemaNeigeLite().to(device).eval(),
        "XAJ_TGD2": XAJWithTGD2Lite().to(device).eval(),
    }


def recorded_pass(
    models,
    structure,
    theta_hat,
    bundle,
    pi,
    x_star,
    months,
    device,
    dtype,
    primary_only=False,
) -> list[dict]:
    """Recorded-forward over the full axis; return state metric rows."""
    from ablation.ic_core.parameter_adapter import get_parameter_spec

    names = tuple(get_parameter_spec(structure))
    n = theta_hat.shape[0]
    rows: list[dict] = []
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
        _qsim, stores, _fs = recorded_forward_for_structure(
            structure, model, fc, params, device, dtype
        )
        for k in range(left, right):
            b = k - left
            for period, (si, ei) in (("train", pi["train"]), ("test", pi["test"])):
                for var in STATE_KEYS:
                    sim = (
                        stores[var]
                        .detach()
                        .cpu()
                        .numpy()[b, si : ei + 1]
                        .astype(np.float64)
                    )
                    rows.append(
                        {
                            "basin_id": "",
                            "variable": var,
                            "period": period,
                            **state_metrics(sim, x_star[var][k, si : ei + 1]),
                        }
                    )
            if not primary_only:
                for season, ms in SEASONS.items():
                    sel = np.isin(months, list(ms))
                    for var in PRIMARY_STATES:
                        sim = (
                            stores[var]
                            .detach()
                            .cpu()
                            .numpy()[b, sel]
                            .astype(np.float64)
                        )
                        rows.append(
                            {
                                "basin_id": "",
                                "variable": var,
                                "period": season,
                                **state_metrics(sim, x_star[var][k, sel]),
                            }
                        )
            # wt = wu + wl + wd, secondary derived; emitted for the same
            # train/test periods as the correct-CN gate baseline
            # (r3_gate_v1/gate_state_metrics_basin.csv) so delta_E pairs by ID.
            wu_np = stores["wu"].detach().cpu().numpy()[b].astype(np.float64)
            wl_np = stores["wl"].detach().cpu().numpy()[b].astype(np.float64)
            wd_np = stores["wd"].detach().cpu().numpy()[b].astype(np.float64)
            wt_np = wu_np + wl_np + wd_np
            wt_truth_np = x_star["wu"][k] + x_star["wl"][k] + x_star["wd"][k]
            for period, (si, ei) in (("train", pi["train"]), ("test", pi["test"])):
                rows.append(
                    {
                        "basin_id": "",
                        "variable": "wt",
                        "period": period,
                        **state_metrics(wt_np[si : ei + 1], wt_truth_np[si : ei + 1]),
                    }
                )
        # assign basin ids in a second sweep (rows appended in basin-major order)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--base-ic-run-id", default="r3_misspec_ic_xaj_531_v1")
    parser.add_argument("--tgd2-ic-run-id", default="r3_misspec_ic_xaj_tgd2_531_v1")
    parser.add_argument("--base-dpl-prefix", default="r3_misspec_dpl_xaj_seed_")
    parser.add_argument("--tgd2-dpl-prefix", default="r3_misspec_dpl_xaj_tgd2_seed_")
    parser.add_argument("--run-id", default="r3_misspec_analysis_v1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--skip-ic", action="store_true")
    parser.add_argument("--skip-dpl", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    RES = args.results_root
    truth_dir = RES / args.truth_run_id
    theta_star = np.load(truth_dir / "theta_star.npz")
    basin_ids = [str(b).zfill(8) for b in theta_star["basin_ids"]]
    x_npz = np.load(truth_dir / "x_star.npz")
    x_star = {k: x_npz[k] for k in STATE_KEYS}

    from manuscript.r3.common import load_bundle

    bundle, _ = load_bundle(args.project_root, args.data_root)
    pi = period_indices(bundle)
    months = pd.to_datetime(bundle.dates).month.to_numpy()
    n = len(basin_ids)

    fits: dict[str, dict] = {}
    if not args.skip_ic:
        for label, run_id, key in [
            ("Base_IC", args.base_ic_run_id, "XAJ"),
            ("TGD2_IC", args.tgd2_ic_run_id, "XAJ_TGD2"),
        ]:
            d = RES / run_id
            if not (d / "DONE.json").exists():
                raise SystemExit(f"refusing: IC results incomplete for {label} ({d})")
            est = load_ic_estimates(d, basin_ids)
            fits[label] = {
                "theta_hat": np.stack([est[b]["theta_hat"] for b in basin_ids]),
                "structure": key,
                "paradigm": "IC",
            }
    if not args.skip_dpl:
        for label, prefix, key in [
            ("Base_dPL", args.base_dpl_prefix, "XAJ"),
            ("TGD2_dPL", args.tgd2_dpl_prefix, "XAJ_TGD2"),
        ]:
            for s in (42, 123, 2026):
                d = RES / f"{prefix}{s}"
                if not (d / "COMPLETE").exists():
                    raise SystemExit(
                        f"refusing: dPL results incomplete for {label} seed {s} ({d})"
                    )
                est = load_dpl_estimates(d, basin_ids)
                fits[f"{label}_seed{s}"] = {
                    "theta_hat": np.stack([est[b]["theta_hat"] for b in basin_ids]),
                    "structure": key,
                    "paradigm": "dPL",
                    "seed": s,
                }
    if not fits:
        raise SystemExit("no fits selected")

    cn_csv = RES / "r3_gate_v1" / "gate_state_metrics_basin.csv"
    if not cn_csv.exists():
        raise SystemExit(f"refusing: CN gate state metrics missing ({cn_csv})")
    cn = pd.read_csv(cn_csv)
    cn["basin_id"] = cn["basin_id"].astype(str).str.zfill(8)

    models = build_models(device)
    all_rows: list[dict] = []
    for fit_name, fit in fits.items():
        print(
            f"[states] {fit_name} ({fit['structure']}) recorded forward ...", flush=True
        )
        rows = recorded_pass(
            models,
            fit["structure"],
            fit["theta_hat"],
            bundle,
            pi,
            x_star,
            months,
            device,
            dtype,
        )
        # rows were collected without basin ids in per-basin order; tag them
        for idx, row in enumerate(rows):
            row["fit"] = fit_name
            row["paradigm"] = fit["paradigm"]
            row["seed"] = fit.get("seed")
            row["structure"] = "Base" if fit["structure"] == "XAJ" else "TGD2"
        # rows are emitted basin-major: 531 basins x (vars*periods) — tag basin
        per_basin_count = len(rows) // n
        for i, row in enumerate(rows):
            row["basin_id"] = basin_ids[i // per_basin_count]
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_dir = RES / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "state_metrics_basin.csv", index=False)

    # paired excess cost delta_E = E_M - E_CN (seed-matched / IC-matched)
    excess_rows: list[dict] = []
    for fit_name, fit in fits.items():
        cn_run = "IC" if fit["paradigm"] == "IC" else f"dPL_seed{fit['seed']}"
        cn_sub = cn[(cn["run"] == cn_run) & (cn["variable"].isin(STATE_KEYS + ["wt"]))]
        sub = df[df["fit"] == fit_name]
        for _, r in sub.iterrows():
            match = cn_sub[
                (cn_sub["basin_id"] == r["basin_id"])
                & (cn_sub["variable"] == r["variable"])
                & (cn_sub["period"] == r["period"])
            ]
            if match.empty:
                continue
            m = match.iloc[0]
            for metric in ("rmse", "nrmse", "bias"):
                excess_rows.append(
                    {
                        "basin_id": r["basin_id"],
                        "fit": fit_name,
                        "paradigm": fit["paradigm"],
                        "seed": fit.get("seed"),
                        "structure": r["structure"],
                        "variable": r["variable"],
                        "period": r["period"],
                        "metric": metric,
                        "e_M": float(r[metric]),
                        "e_CN": float(m[metric]),
                        "delta_E": float(r[metric]) - float(m[metric]),
                    }
                )
    excess = pd.DataFrame(excess_rows)
    excess.to_csv(out_dir / "state_excess.csv", index=False)

    fs = frac_snow_series(bundle)
    snow = dict(zip(fs["basin_id"].astype(str).str.zfill(8), fs["frac_snow"]))
    summary: dict = {}
    # The correct-CN gate baseline (r3_gate_v1/gate_state_metrics_basin.csv)
    # uses train/test + DJF/MAM/JJAS periods (no "full" axis), so the paired
    # delta_E summary is computed per scored period with test as the headline.
    summary_full: dict = {}
    for period in ("test", "train"):
        sub = excess[(excess["period"] == period) & (excess["metric"] == "nrmse")]
        for structure in ["Base", "TGD2"]:
            for paradigm in ["IC", "dPL"]:
                key = f"{structure}_{paradigm}"
                ssub = sub[
                    (sub["structure"] == structure) & (sub["paradigm"] == paradigm)
                ]
                per_var = {}
                for var in STATE_KEYS + ["wt"]:
                    v = ssub[ssub["variable"] == var]["delta_E"]
                    if v.empty:
                        continue
                    entry = {
                        "median_delta_E": float(v.median()),
                        "q25": float(v.quantile(0.25)),
                        "q75": float(v.quantile(0.75)),
                        "frac_positive": float((v > 0).mean()),
                    }
                    snow_vals = np.array(
                        [snow[b] for b in ssub[ssub["variable"] == var]["basin_id"]]
                    )
                    de = v.to_numpy()
                    if snow_vals.std() > 0 and np.isfinite(de).all():
                        entry["spearman_delta_E_vs_frac_snow"] = float(
                            np.corrcoef(
                                np.argsort(np.argsort(snow_vals)),
                                np.argsort(np.argsort(de)),
                            )[0, 1]
                        )
                    per_var[var] = entry
                summary_full[key] = per_var
    summary["periods"] = {"headline": "test", "available": ["test", "train"]}
    summary["per_period"] = summary_full
    summary["n_basins"] = n
    summary["protocol"] = "r3_misspec_states_v1"
    summary["frozen_protocol"] = "r3/protocol_misspec_v1.json"
    summary["code"] = git_commit(args.project_root)
    write_json(out_dir / "state_summary.json", summary)
    print(f"COMPLETE state analysis -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
