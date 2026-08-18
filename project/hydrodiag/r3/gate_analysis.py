#!/usr/bin/env python3
"""Phase D3: complete correct-CN gate analysis package over all 531 basins.

Consumes (read-only):

- ``r3_synthetic_truth_v1`` (theta*, q_star, x_star, snow_star, final_states);
- ``r3_gate_ic_xaj_cn_531_v1`` (CN + IC-CMA-ES, 10 starts x 300 gens, Q*);
- ``r3_gate_dpl_xaj_cn_seed_<42|123|2026>`` (CN + dPL, Q*);
- ``r3_gate_v1/oracle_dpl_audit_windows_canonical.csv`` (dPL eval-path
  ceiling at theta*).

Produces the truth-recovery statistics needed to decide which parameters
are identifiable from correct-CN results only (no Base/TGD2 runs, no
automatic subset freeze):

1. input validation/alignment (basin IDs by ID, headline KGE recomputation);
2. per-parameter recovery profile (normalized signed/absolute errors,
   quantiles, correlations, OLS slope, boundary fractions, frac_snow
   association) for IC best-restart, dPL median-of-seeds and each seed;
3. IC restart-to-restart dispersion and dPL seed spread per parameter;
4. equifinality analysis: basin KGE vs common-parameter recovery D_theta;
5. state/flux recovery (wu,wl,wd,s,fr,qi,qg + CN snow diagnostics) with
   RMSE / normalized RMSE / temporal correlation / mean bias for
   train/test/full and the fixed a-priori calendar seasons DJF/MAM/JJAS
   (same windows as ``r3/analyze_pilot.py``);
6. snow-dependence diagnostics (frac_snow vs KGE deficit, D_theta,
   per-parameter |e|, state NRMSE);
7. a machine-readable manifest and a concise Markdown report.

Definitions (all recorded in the manifest):

- normalized error ``e[p,i] = (theta_hat - theta_star) / (upper - lower)``;
- ``D_theta[i] = median over the 15 shared XAJ parameters of abs(e)``;
- NRMSE = RMSE / (std(truth) + 1e-8);
- boundary contact: within 1e-9 of lower/upper (truth-generator convention);
- IC restart selection: best train-KGE restart per basin (lowest start
  breaks ties); dPL: ``best_parameters_physical.npz`` exported from the
  runner's ``best_checkpoint.pt`` (repository convention);
- state simulation: repository recorded-forward (production kernels) with
  the canonical ``cn_psol_annual``, full 12418-day axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[0]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from r3.common import (  # noqa: E402
    COMMON_XAJ,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    bundle_with_synthetic_target,
    frac_snow_series,
    git_commit,
    nse,
    pbias,
    period_indices,
    standard_kge,
    write_json,
)
from r3.oracle_identity import canonical_psol  # noqa: E402
from r3.recorded_forward import (  # noqa: E402
    build_forcing_dict,
    recorded_cn_forward,
)

SEEDS = (42, 123, 2026)
STATE_KEYS = ("wu", "wl", "wd", "s", "fr", "qi", "qg")
SNOW_KEYS = ("G", "eTG", "sca", "rain", "melt")
# Fixed a-priori calendar snow-season windows, identical to
# r3/analyze_pilot.py SEASON_MONTHS (DJF accumulation / MAM melt /
# JJAS non-snow).  Used unchanged; not tuned to results.
SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9)}
BOUND_EPS = 1e-9
NRMSE_EPS = 1e-8


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ic_estimates(run_dir: Path, basin_ids: list[str]) -> dict[str, dict]:
    """Best train-KGE restart per basin, plus all per-start theta/z records."""
    # The IC runner writes per-basin-start records under raw/<model_key>/
    # (e.g. raw/xaj_cn for XAJ_CN, raw/xaj for XAJ, raw/xaj_tgd2 for
    # XAJ_TGD2).  Discover the subdirectory instead of assuming the CN key
    # so the same loader serves the correct-CN gate and the Base/TGD2
    # misspecification IC runs (analysis mechanics only; no protocol change).
    raw_candidates = [
        run_dir / "raw" / "xaj_cn",
        run_dir / "raw" / "xaj",
        run_dir / "raw" / "xaj_tgd2",
        run_dir / "raw",
    ]
    raw_dir = next((p for p in raw_candidates if p.is_dir()), raw_candidates[0])
    records: dict[str, list[tuple[float, int, dict]]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text())
        basin = str(data["basin_id"]).zfill(8)
        kge = float(data.get("train_metrics", {}).get("kge", np.nan))
        if data.get("status") == "complete" and np.isfinite(kge):
            records.setdefault(basin, []).append((kge, int(data["start"]), data))
    out: dict[str, dict] = {}
    for basin in basin_ids:
        candidates = records.get(basin, [])
        if not candidates:
            raise ValueError(f"no complete IC restart for basin {basin}")
        candidates.sort(key=lambda item: (-item[0], item[1]))
        _kge, start, data = candidates[0]
        starts = sorted(r[1] for r in records[basin])
        out[basin] = {
            "theta_hat": np.asarray(data["parameters"], dtype=np.float64),
            "restart": start,
            "seed": data.get("seed"),
            "parameter_names": tuple(data.get("parameter_names", ())),
            "train_kge": _kge,
            "test_kge": float(data.get("test_metrics", {}).get("kge", np.nan)),
            "starts": starts,
            "start_kges": [r[0] for r in records[basin]],
            "source_file": str(path),
        }
    return out


def load_ic_all_starts(
    run_dir: Path, basin_ids: list[str]
) -> dict[str, dict[int, np.ndarray]]:
    """All 10 restart parameter vectors per basin (for dispersion analysis)."""
    raw_dir = run_dir / "raw" / "xaj_cn"
    out: dict[str, dict[int, np.ndarray]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text())
        basin = str(data["basin_id"]).zfill(8)
        if data.get("status") != "complete":
            continue
        out.setdefault(basin, {})[int(data["start"])] = np.asarray(
            data["parameters"], dtype=np.float64
        )
    for basin in basin_ids:
        if basin not in out or len(out[basin]) != 10:
            raise ValueError(
                f"basin {basin}: expected 10 IC restarts, got {len(out.get(basin, {}))}"
            )
    return out


def load_dpl_estimates(run_dir: Path, basin_ids: list[str]) -> dict[str, dict]:
    params = np.load(run_dir / "best_parameters_physical.npz")["params"]
    config = json.loads((run_dir / "config.json").read_text())
    names = tuple(config["parameter_names"])
    if params.shape[0] != len(basin_ids):
        raise ValueError(f"dPL rows {params.shape[0]} != basins {len(basin_ids)}")
    summary = pd.read_csv(run_dir / "basin_final_summary.csv")
    kge_by_basin = dict(
        zip(summary["basin_id"].astype(str).str.zfill(8), summary["val_kge"])
    )
    out: dict[str, dict] = {}
    for i, basin in enumerate(basin_ids):
        out[basin] = {
            "theta_hat": params[i].astype(np.float64),
            "restart": None,
            "seed": config.get("training", {}).get("seed"),
            "parameter_names": names,
            "train_kge": np.nan,
            "test_kge": float(kge_by_basin.get(basin, np.nan)),
            "source_file": str(run_dir / "best_parameters_physical.npz"),
        }
    return out


def oracle_kge_cn(bundle, q_star, theta_star, device) -> dict[str, np.ndarray]:
    """theta* through the exact IC objective path (canonical psol)."""
    from ablation.ic_core.parameter_adapter import physical_to_normalized
    from ablation.ic_core.runtime import ICObjectiveRuntime

    syn_bundle = bundle_with_synthetic_target(bundle, q_star)
    config = {
        "device": str(device),
        "model_variant": "lite",
        "batching": {"basin_batch_size": 4, "cache_device_data": False},
        "objective": {"min_samples": 30},
        "canonical_cn_psol_annual": True,
    }
    runtime = ICObjectiveRuntime(syn_bundle, config, "XAJ_CN", model_variant="lite")
    theta_01 = physical_to_normalized("XAJ_CN", theta_star, clip=False)
    result = {}
    for split in ("train", "test"):
        fit, _ = runtime.evaluate_candidates_tensor(
            torch.from_numpy(theta_01).unsqueeze(1).to(device, dtype=torch.float64),
            basin_indices=list(range(len(bundle.basin_ids))),
            split=split,
        )
        result[split] = fit[:, 0].detach().cpu().numpy()
    return result


def load_dpl_eval_ceiling(results_root: Path) -> np.ndarray:
    """Per-basin dPL eval-path oracle KGE (theta* through the dPL evaluation
    path, canonical psol) from the completed 531-basin oracle audit."""
    audit = results_root / "r3_gate_v1" / "oracle_dpl_audit_windows_canonical.csv"
    if not audit.exists():
        raise FileNotFoundError(f"dPL oracle audit missing: {audit}")
    df = pd.read_csv(audit)
    eval_rows = df[df["window_offset"] < 0]
    if len(eval_rows) != 531:
        raise ValueError(f"dPL eval-path oracle rows {len(eval_rows)} != 531")
    # align by basin ID (audit uses the canonical basin order)
    order = eval_rows.sort_values("basin_id")["kge"].to_numpy()
    return order


def validate_and_align(bundle, truth_dir, ic_dir, dpl_dirs, q_star) -> dict:
    """Basin-ID alignment and headline-KGE recomputation checks."""
    basin_ids = list(bundle.basin_ids)
    n = len(basin_ids)
    report: dict = {"n_basins": n, "checks": {}}

    if len(set(basin_ids)) != n:
        raise ValueError("bundle basin IDs are not unique")
    truth_ids = [str(b) for b in np.load(truth_dir / "theta_star.npz")["basin_ids"]]
    if list(truth_ids) != basin_ids:
        raise ValueError("theta_star basin order differs from the bundle")
    for key, arr in (
        ("q", np.load(truth_dir / "q_star.npz")["target_mm_day"]),
        ("x", np.load(truth_dir / "x_star.npz")["wu"]),
        ("snow", np.load(truth_dir / "snow_star.npz")["G"]),
    ):
        if arr.shape[0] != n:
            raise ValueError(f"{key}_star rows {arr.shape[0]} != {n}")

    # IC raw records: complete set, unique basins, 10 starts each
    ic_raw = ic_dir / "raw" / "xaj_cn"
    ic_ids = set()
    ic_starts = {}
    for path in sorted(ic_raw.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "complete":
            continue
        b = str(data["basin_id"]).zfill(8)
        ic_ids.add(b)
        ic_starts.setdefault(b, set()).add(int(data["start"]))
    if sorted(ic_ids) != sorted(basin_ids):
        raise ValueError("IC raw basin set does not match the canonical 531 list")
    if any(len(v) != 10 for v in ic_starts.values()):
        bad = [b for b, v in ic_starts.items() if len(v) != 10]
        raise ValueError(f"IC basins without 10 restarts: {bad[:5]}")
    report["checks"]["ic_raw_records"] = {
        "basins": len(ic_ids),
        "restarts_per_basin": sorted(set(map(len, ic_starts.values()))),
    }

    # dPL: basin_final_summary + config parameter names
    for seed in SEEDS:
        d = dpl_dirs[seed]
        summary = pd.read_csv(d / "basin_final_summary.csv")
        ids = set(summary["basin_id"].astype(str).str.zfill(8))
        if sorted(ids) != sorted(basin_ids):
            raise ValueError(f"dPL seed {seed} basin set mismatch")
        config = json.loads((d / "config.json").read_text())
        if tuple(config["parameter_names"]) != tuple(
            np.load(truth_dir / "theta_star.npz")["parameter_names"]
        ):
            raise ValueError(f"dPL seed {seed} parameter names mismatch")
        if not (d / "COMPLETE").exists():
            raise ValueError(f"dPL seed {seed} missing COMPLETE marker")
    report["checks"]["dpl_seeds"] = {str(s): "ok" for s in SEEDS}

    # headline KGE recomputation from raw artifacts
    ic_best = {}
    for path in sorted(ic_raw.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "complete":
            continue
        b = str(data["basin_id"]).zfill(8)
        k = float(data["train_metrics"]["kge"])
        if b not in ic_best or k > ic_best[b][0]:
            ic_best[b] = (k, int(data["start"]))
    kges = np.array([ic_best[b][0] for b in basin_ids])
    report["headline_kge"] = {
        "ic_best_restart_train_median": float(np.median(kges)),
        "ic_best_restart_train_min": float(kges.min()),
        "dpl_val_kge_median": {
            str(s): float(
                pd.read_csv(dpl_dirs[s] / "basin_final_summary.csv")["val_kge"].median()
            )
            for s in SEEDS
        },
    }
    # oracle-path recomputation sanity: q_star slice finite/non-negative
    if not (np.isfinite(q_star).all() and (q_star >= 0).all()):
        raise ValueError("q_star not finite/non-negative")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--ic-run-id", default="r3_gate_ic_xaj_cn_531_v1")
    parser.add_argument("--dpl-run-prefix", default="r3_gate_dpl_xaj_cn_seed_")
    parser.add_argument("--run-id", default="r3_gate_v1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-basins", type=int, default=64)
    parser.add_argument(
        "--skip-states",
        action="store_true",
        help="Skip recorded-forward state export (analysis only).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    truth_dir = args.results_root / args.truth_run_id
    ic_dir = args.results_root / args.ic_run_id
    dpl_dirs = {s: args.results_root / f"{args.dpl_run_prefix}{s}" for s in SEEDS}
    output_dir = args.results_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, _config = load_bundle(args.project_root, args.data_root)
    pi = period_indices(bundle)
    dates = pd.to_datetime(bundle.dates)
    months = dates.month.to_numpy()
    basin_ids = list(bundle.basin_ids)
    n = len(basin_ids)
    theta_npz = np.load(truth_dir / "theta_star.npz")
    theta_star = theta_npz["parameters"]
    names = [str(x) for x in theta_npz["parameter_names"]]
    shared_idx = [names.index(x) for x in COMMON_XAJ]
    q_star = np.asarray(
        np.load(truth_dir / "q_star.npz")["target_mm_day"], dtype=np.float64
    )
    x_star = {
        k: np.asarray(np.load(truth_dir / "x_star.npz")[k], dtype=np.float64)
        for k in STATE_KEYS
    }
    snow_star = {
        k: np.asarray(np.load(truth_dir / "snow_star.npz")[k], dtype=np.float64)
        for k in SNOW_KEYS
    }
    psol = canonical_psol(bundle)
    snow = frac_snow_series(bundle).set_index("basin_id")["frac_snow"]

    from models import XAJWithCemaNeigeLite
    from models.parameter_specs import XAJ_CN_PARAM_SPECS

    specs = XAJ_CN_PARAM_SPECS
    lower = np.asarray([specs[x]["lower"] for x in names], dtype=np.float64)
    upper = np.asarray([specs[x]["upper"] for x in names], dtype=np.float64)
    p_range = upper - lower

    # ---------- 1. validation + alignment ----------
    print("validating and aligning inputs ...", flush=True)
    validation = validate_and_align(bundle, truth_dir, ic_dir, dpl_dirs, q_star)
    write_json(output_dir / "gate_input_validation.json", validation)
    print("  headline KGE:", json.dumps(validation["headline_kge"]), flush=True)

    # ---------- 2. oracle ceilings ----------
    print("oracle (theta* through IC objective path) ...", flush=True)
    oracle = oracle_kge_cn(bundle, q_star, theta_star, device)
    dpl_ceiling = load_dpl_eval_ceiling(args.results_root)  # canonical basin order

    # ---------- 3. parameter estimates ----------
    print("loading IC/dPL fits ...", flush=True)
    ic_est = load_ic_estimates(ic_dir, basin_ids)
    ic_starts_all = load_ic_all_starts(ic_dir, basin_ids)
    dpl_est = {s: load_dpl_estimates(dpl_dirs[s], basin_ids) for s in SEEDS}

    z_ic = np.stack([(ic_est[b]["theta_hat"] - lower) / p_range for b in basin_ids])
    z_dpl = {
        s: np.stack([(dpl_est[s][b]["theta_hat"] - lower) / p_range for b in basin_ids])
        for s in SEEDS
    }
    z_dpl_median = np.median(np.stack([z_dpl[s] for s in SEEDS]), axis=0)
    z_star_all = (theta_star - lower) / p_range

    # ---------- 4. discharge metrics via recorded forward (batched) ----------
    model = XAJWithCemaNeigeLite().to(device).eval()
    dtype = torch.float32
    batch = args.batch_basins

    fits = {"IC": z_ic, "dPL_median": z_dpl_median}
    for s in SEEDS:
        fits[f"dPL_seed{s}"] = z_dpl[s]

    def theta_hat_physical(z: np.ndarray) -> np.ndarray:
        return lower + z * p_range

    q_hat: dict[str, np.ndarray] = {}
    state_rows: list[dict] = []
    metric_rows: list[dict] = []

    def state_metrics(sim: np.ndarray, truth: np.ndarray) -> dict:
        mask = np.isfinite(sim) & np.isfinite(truth)
        if mask.sum() < 2:
            return {
                "rmse": float("nan"),
                "nrmse": float("nan"),
                "corr": float("nan"),
                "bias": float("nan"),
            }
        s, t = sim[mask], truth[mask]
        rmse = float(np.sqrt(np.mean((s - t) ** 2)))
        std_t = float(t.std())
        nrmse = float(rmse / (std_t + NRMSE_EPS))
        if std_t < 1e-12 or s.std() < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(s, t)[0, 1])
        return {
            "rmse": rmse,
            "nrmse": nrmse,
            "corr": corr,
            "bias": float(np.mean(s - t)),
        }

    # theta* round-trip pass first (recorded forward == frozen truth)
    print("theta* round-trip verification (recorded forward) ...", flush=True)
    rt_max = 0.0
    for left in range(0, n, batch):
        right = min(n, left + batch)
        fc = build_forcing_dict(
            bundle.forcing[left:right].astype(np.float32), device, dtype
        )
        fc["cn_psol_annual"] = torch.from_numpy(psol[left:right]).to(
            device, dtype=dtype
        )
        p = {
            name: torch.from_numpy(theta_star[left:right, i]).to(device, dtype=dtype)
            for i, name in enumerate(names)
        }
        qsim, _stores, _fs = recorded_cn_forward(model, fc, p, device, dtype)
        rt_max = max(
            rt_max,
            float(
                np.abs(
                    qsim.detach().cpu().numpy().astype(np.float64) - q_star[left:right]
                ).max()
            ),
        )
    print(f"  theta* round-trip max abs diff vs q_star: {rt_max:.3e}", flush=True)

    print("recorded-forward passes for all fits ...", flush=True)
    for fit_name, z in fits.items():
        theta_hat = theta_hat_physical(z)
        q_full = np.empty((n, bundle.forcing.shape[1]), dtype=np.float32)
        for left in range(0, n, batch):
            right = min(n, left + batch)
            fc = build_forcing_dict(
                bundle.forcing[left:right].astype(np.float32), device, dtype
            )
            fc["cn_psol_annual"] = torch.from_numpy(psol[left:right]).to(
                device, dtype=dtype
            )
            p = {
                name: torch.from_numpy(theta_hat[left:right, i]).to(device, dtype=dtype)
                for i, name in enumerate(names)
            }
            qsim, stores, _fs = recorded_cn_forward(model, fc, p, device, dtype)
            q_full[left:right] = qsim.detach().cpu().numpy()
            if not args.skip_states:
                for k in range(left, right):
                    b = k - left
                    for period, (si, ei) in (
                        ("train", pi["train"]),
                        ("test", pi["test"]),
                    ):
                        for var in STATE_KEYS + SNOW_KEYS:
                            sim = (
                                stores[var]
                                .detach()
                                .cpu()
                                .numpy()[b, si : ei + 1]
                                .astype(np.float64)
                            )
                            truth = (x_star if var in STATE_KEYS else snow_star)[var][
                                k, si : ei + 1
                            ]
                            row = {
                                "basin_id": basin_ids[k],
                                "paradigm": fit_name.split("_")[0],
                                "run": fit_name,
                                "variable": var,
                                "period": period,
                            }
                            row.update(state_metrics(sim, truth))
                            state_rows.append(row)
                    for season, ms in SEASON_MONTHS.items():
                        sel = np.isin(months, list(ms))
                        for var in STATE_KEYS[:5]:
                            sim = (
                                stores[var]
                                .detach()
                                .cpu()
                                .numpy()[b, sel]
                                .astype(np.float64)
                            )
                            truth = x_star[var][k, sel]
                            row = {
                                "basin_id": basin_ids[k],
                                "paradigm": fit_name.split("_")[0],
                                "run": fit_name,
                                "variable": var,
                                "period": season,
                            }
                            row.update(state_metrics(sim, truth))
                            state_rows.append(row)
                    # derived total tension-water storage wt = wu + wl + wd
                    # (protocol_misspec_v1.json state_estimands; secondary).
                    for period, (si, ei) in (
                        ("train", pi["train"]),
                        ("test", pi["test"]),
                    ):
                        wt = (
                            stores["wu"]
                            .detach()
                            .cpu()
                            .numpy()[b, si : ei + 1]
                            .astype(np.float64)
                            + stores["wl"]
                            .detach()
                            .cpu()
                            .numpy()[b, si : ei + 1]
                            .astype(np.float64)
                            + stores["wd"]
                            .detach()
                            .cpu()
                            .numpy()[b, si : ei + 1]
                            .astype(np.float64)
                        )
                        wt_truth = (
                            x_star["wu"][k, si : ei + 1]
                            + x_star["wl"][k, si : ei + 1]
                            + x_star["wd"][k, si : ei + 1]
                        )
                        row = {
                            "basin_id": basin_ids[k],
                            "paradigm": fit_name.split("_")[0],
                            "run": fit_name,
                            "variable": "wt",
                            "period": period,
                        }
                        row.update(state_metrics(wt, wt_truth))
                        state_rows.append(row)
        q_hat[fit_name] = q_full
        for k, basin in enumerate(basin_ids):
            row = {
                "basin_id": basin,
                "paradigm": fit_name.split("_")[0],
                "run": fit_name,
            }
            for period, (si, ei) in (("train", pi["train"]), ("test", pi["test"])):
                sim = q_full[k, si : ei + 1].astype(np.float64)
                obs = q_star[k, si : ei + 1]
                row[f"kge_{period}"] = standard_kge(sim, obs)
                row[f"nse_{period}"] = nse(sim, obs)
                row[f"pbias_{period}"] = pbias(sim, obs)
            row["kge_test"] = row.get("kge_test", float("nan"))
            metric_rows.append(row)
        print(f"  {fit_name} pass done", flush=True)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "gate_discharge_metrics.csv", index=False
    )
    if not args.skip_states:
        pd.DataFrame(state_rows).to_csv(
            output_dir / "gate_state_metrics_basin.csv", index=False
        )

    # ---------- 5. parameter recovery tables ----------
    print("parameter recovery statistics ...", flush=True)
    # Secondary CN-only diagnostics are included in the same table but never
    # mixed into the shared-parameter aggregate (D_theta uses COMMON_XAJ only).
    analysis_params = list(COMMON_XAJ) + ["cn_ctg", "cn_kf"]
    param_rows = []
    for name in analysis_params:
        j = names.index(name)
        star_j = z_star_all[:, j]
        frac = snow.reindex(basin_ids).to_numpy()
        for paradigm, z_run, run_label in (
            ("IC", z_ic[:, j], "best-restart"),
            ("dPL", z_dpl_median[:, j], "median-seeds"),
        ) + tuple(("dPL", z_dpl[s][:, j], f"seed-{s}") for s in SEEDS):
            e = z_run - star_j
            ae = np.abs(e)
            valid = np.isfinite(e)
            pearson = (
                float(np.corrcoef(z_run[valid], star_j[valid])[0, 1])
                if valid.sum() > 2
                else float("nan")
            )
            spearman = float(
                pd.Series(z_run[valid]).corr(
                    pd.Series(star_j[valid]), method="spearman"
                )
            )
            slope, intercept = (
                np.polyfit(star_j[valid], z_run[valid], 1)
                if valid.sum() > 2
                else (float("nan"), float("nan"))
            )
            phys = lower[j] + z_run * p_range[j]
            lo_b, up_b = lower[j], upper[j]
            param_rows.append(
                {
                    "parameter": name,
                    "parameter_group": "cn_only"
                    if name in ("cn_ctg", "cn_kf")
                    else "shared",
                    "paradigm": paradigm,
                    "run": run_label,
                    "n_basins": int(valid.sum()),
                    "median_signed_e": float(np.median(e[valid])),
                    "median_abs_e": float(np.median(ae[valid])),
                    "q25_abs_e": float(np.quantile(ae[valid], 0.25)),
                    "q75_abs_e": float(np.quantile(ae[valid], 0.75)),
                    "q90_abs_e": float(np.quantile(ae[valid], 0.90)),
                    "mean_abs_e": float(np.mean(ae[valid])),
                    "pearson_theta_hat_vs_star": pearson,
                    "spearman_theta_hat_vs_star": spearman,
                    "ols_slope": float(slope),
                    "ols_intercept": float(intercept),
                    "frac_at_lower": float((phys <= lo_b + BOUND_EPS).mean()),
                    "frac_at_upper": float((phys >= up_b - BOUND_EPS).mean()),
                    "spearman_signed_e_vs_frac_snow": float(
                        pd.Series(e[valid]).corr(
                            pd.Series(frac[valid]), method="spearman"
                        )
                    ),
                    "spearman_abs_e_vs_frac_snow": float(
                        pd.Series(ae[valid]).corr(
                            pd.Series(frac[valid]), method="spearman"
                        )
                    ),
                }
            )
    param_df = pd.DataFrame(param_rows)
    param_df.to_csv(output_dir / "parameter_recovery_summary.csv", index=False)

    # per-basin parameter rows (z_hat/z_star/e) for IC best-restart + dPL median + seeds
    rec_rows = []
    for j, name in enumerate(COMMON_XAJ):
        star_j = z_star_all[:, shared_idx[j]]
        for paradigm, z_run, run_label in (
            ("IC", z_ic[:, shared_idx[j]], "best-restart"),
            ("dPL", z_dpl_median[:, shared_idx[j]], "median-seeds"),
        ) + tuple(("dPL", z_dpl[s][:, shared_idx[j]], f"seed-{s}") for s in SEEDS):
            for k, basin in enumerate(basin_ids):
                rec_rows.append(
                    {
                        "basin_id": basin,
                        "paradigm": paradigm,
                        "run": run_label,
                        "parameter": name,
                        "z_hat": float(z_run[k]),
                        "z_star": float(star_j[k]),
                        "delta_z": float(z_run[k] - star_j[k]),
                        "abs_delta_z": float(abs(z_run[k] - star_j[k])),
                    }
                )
    pd.DataFrame(rec_rows).to_csv(
        output_dir / "parameters_recoverability.csv", index=False
    )

    # IC restart dispersion per parameter (all 10 starts)
    disp_rows = []
    for k, basin in enumerate(basin_ids):
        for j, name in enumerate(COMMON_XAJ):
            z_starts = np.array(
                [
                    (ic_starts_all[basin][s] - lower) / p_range
                    for s in ic_starts_all[basin]
                ]
            )
            zj = z_starts[:, shared_idx[j]]
            disp_rows.append(
                {
                    "basin_id": basin,
                    "parameter": name,
                    "z_std_across_starts": float(zj.std()),
                    "z_range_across_starts": float(zj.max() - zj.min()),
                }
            )
    disp_df = pd.DataFrame(disp_rows)
    disp_df.to_csv(output_dir / "ic_restart_parameter_dispersion.csv", index=False)

    # dPL seed spread per parameter
    seed_rows = []
    for k, basin in enumerate(basin_ids):
        for j, name in enumerate(COMMON_XAJ):
            zj = np.array([z_dpl[s][k, shared_idx[j]] for s in SEEDS])
            seed_rows.append(
                {
                    "basin_id": basin,
                    "parameter": name,
                    "z_std_across_seeds": float(zj.std()),
                    "z_min_across_seeds": float(zj.min()),
                    "z_max_across_seeds": float(zj.max()),
                }
            )
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(output_dir / "dpl_seed_parameter_spread.csv", index=False)

    # ---------- 6. equifinality: KGE vs D_theta ----------
    print("equifinality analysis ...", flush=True)
    # D_theta per basin from the per-basin parameter rows
    dtheta = {}
    for paradigm, run_label in (("IC", "best-restart"), ("dPL", "median-seeds")):
        sub = pd.DataFrame(rec_rows)
        sub = sub[(sub["paradigm"] == paradigm) & (sub["run"] == run_label)]
        vals = sub.groupby("basin_id")["abs_delta_z"].median()
        dtheta[paradigm] = vals.reindex(basin_ids).to_numpy()
    metric_df = pd.DataFrame(metric_rows)
    eq_rows = []
    for paradigm, run_label in (("IC", "IC"), ("dPL", "dPL_median")):
        sub = metric_df[
            (metric_df["paradigm"] == paradigm) & (metric_df["run"] == run_label)
        ]
        for k, basin in enumerate(basin_ids):
            row = sub[sub["basin_id"] == basin]
            if row.empty:
                continue
            r = row.iloc[0]
            eq_rows.append(
                {
                    "basin_id": basin,
                    "paradigm": paradigm,
                    "kge_train": float(r["kge_train"]),
                    "kge_test": float(r["kge_test"]),
                    "D_theta": float(dtheta[paradigm][k]),
                }
            )
    eq_df = pd.DataFrame(eq_rows)
    eq_df.to_csv(output_dir / "kge_vs_parameter_recovery.csv", index=False)
    eq_summary = {}
    for paradigm in ("IC", "dPL"):
        sub = eq_df[eq_df["paradigm"] == paradigm]
        kge = sub["kge_train"].to_numpy()
        dt = sub["D_theta"].to_numpy()
        eq_summary[paradigm] = {
            "D_theta_median": float(np.median(dt)),
            "D_theta_q25": float(np.quantile(dt, 0.25)),
            "D_theta_q75": float(np.quantile(dt, 0.75)),
            "D_theta_q90": float(np.quantile(dt, 0.90)),
            "D_theta_max": float(np.max(dt)),
            "pearson_kge_vs_D_theta": float(np.corrcoef(kge, dt)[0, 1]),
            "spearman_kge_vs_D_theta": float(
                pd.Series(kge).corr(pd.Series(dt), method="spearman")
            ),
            "D_theta_among_kge_ge_0p999": float(np.median(dt[kge >= 0.999])),
            "D_theta_among_kge_ge_0p99": float(np.median(dt[kge >= 0.99])),
            "D_theta_among_kge_ge_q90": float(
                np.median(dt[kge >= np.quantile(kge, 0.90)])
            ),
            "n_kge_ge_0p999": int((kge >= 0.999).sum()),
            "n_kge_ge_0p99": int((kge >= 0.99).sum()),
        }
    # example basins: high KGE + high D_theta
    examples = {}
    for paradigm in ("IC", "dPL"):
        sub = eq_df[eq_df["paradigm"] == paradigm]
        kge = sub["kge_train"].to_numpy()
        dt = sub["D_theta"].to_numpy()
        hi = (kge >= np.quantile(kge, 0.90)) & (dt >= np.quantile(dt, 0.75))
        ex = sub[hi].nlargest(8, "D_theta")[["basin_id", "kge_train", "D_theta"]]
        examples[paradigm] = ex.to_dict("records")
    eq_summary["examples_high_kge_poor_dtheta"] = examples

    # ---------- 7. snow-dependence diagnostics ----------
    print("snow-dependence diagnostics ...", flush=True)
    frac_arr = snow.reindex(basin_ids).to_numpy()
    snow_diag = {"n_basins": n}
    for paradigm in ("IC", "dPL"):
        sub = eq_df[eq_df["paradigm"] == paradigm]
        kge = sub["kge_train"].to_numpy()
        dt = sub["D_theta"].to_numpy()
        snow_diag[paradigm] = {
            "spearman_kge_train_vs_frac_snow": float(
                pd.Series(kge).corr(pd.Series(frac_arr), method="spearman")
            ),
            "spearman_D_theta_vs_frac_snow": float(
                pd.Series(dt).corr(pd.Series(frac_arr), method="spearman")
            ),
        }
    # KGE deficit vs frac_snow (IC: oracle_train - kge_train; dPL: ceiling - kge_test)
    deficit_rows = []
    for k, basin in enumerate(basin_ids):
        ic_row = metric_df[
            (metric_df["basin_id"] == basin) & (metric_df["paradigm"] == "IC")
        ]
        dpl_row = metric_df[
            (metric_df["basin_id"] == basin)
            & (metric_df["paradigm"] == "dPL")
            & (metric_df["run"] == "dPL_median")
        ]
        deficit_rows.append(
            {
                "basin_id": basin,
                "frac_snow": float(frac_arr[k]),
                "ic_deficit_train": float(
                    oracle["train"][k] - ic_row.iloc[0]["kge_train"]
                ),
                "ic_deficit_test": float(
                    oracle["test"][k] - ic_row.iloc[0]["kge_test"]
                ),
                "dpl_deficit_test": float(dpl_ceiling[k] - dpl_row.iloc[0]["kge_test"]),
            }
        )
    deficit_df = pd.DataFrame(deficit_rows)
    deficit_df.to_csv(output_dir / "kge_deficit_vs_frac_snow.csv", index=False)
    for col in ("ic_deficit_train", "ic_deficit_test", "dpl_deficit_test"):
        snow_diag[f"spearman_{col}_vs_frac_snow"] = float(
            deficit_df[col].corr(deficit_df["frac_snow"], method="spearman")
        )
    # per-parameter |e| vs frac_snow (from param summary rows)
    for paradigm, run_label in (("IC", "best-restart"), ("dPL", "median-seeds")):
        sub = param_df[
            (param_df["paradigm"] == paradigm) & (param_df["run"] == run_label)
        ]
        snow_diag[f"param_abs_e_vs_frac_snow_{paradigm}"] = {
            r["parameter"]: float(r["spearman_abs_e_vs_frac_snow"])
            for _, r in sub.iterrows()
        }
    # state NRMSE (primary vars, full period) vs frac_snow
    if not args.skip_states:
        state_summary = {}
        for var in STATE_KEYS + SNOW_KEYS:
            for fit_name in fits:
                sub = pd.DataFrame(state_rows)
                sub = sub[
                    (sub["run"] == fit_name)
                    & (sub["variable"] == var)
                    & (sub["period"] == "train")
                ]
                merged = sub.merge(
                    pd.DataFrame({"basin_id": basin_ids, "frac_snow": frac_arr}),
                    on="basin_id",
                )
                state_summary.setdefault(var, {})[fit_name] = {
                    "rmse_median": float(merged["rmse"].median()),
                    "nrmse_median": float(merged["nrmse"].median()),
                    "corr_median": float(merged["corr"].median()),
                    "bias_median": float(merged["bias"].median()),
                    "spearman_nrmse_vs_frac_snow": float(
                        merged["nrmse"].corr(merged["frac_snow"], method="spearman")
                    ),
                }
        write_json(output_dir / "gate_state_summary.json", state_summary)
        snow_diag["state_nrmse_vs_frac_snow"] = {
            var: {
                fit: state_summary[var][fit]["spearman_nrmse_vs_frac_snow"]
                for fit in fits
            }
            for var in STATE_KEYS
        }

    # ---------- 8. aggregate report + manifest + markdown ----------
    ic_metric = metric_df[metric_df["paradigm"] == "IC"]
    dpl_metric = metric_df[metric_df["paradigm"] == "dPL"]

    report = {
        "protocol": "r3_correct_cn_gate_v1",
        "created_at": _utcnow(),
        "code": git_commit(args.project_root),
        "truth_run_id": args.truth_run_id,
        "ic_run_id": args.ic_run_id,
        "dpl_run_prefix": args.dpl_run_prefix,
        "n_basins": n,
        "round_trip_theta_star_max_abs_diff": rt_max,
        "q_recovery_ic": {
            "median_kge_train": float(ic_metric["kge_train"].median()),
            "median_kge_test": float(ic_metric["kge_test"].median()),
            "median_oracle_kge_train": float(np.median(oracle["train"])),
            "median_oracle_gap_train": float(
                np.median(oracle["train"] - ic_metric["kge_train"])
            ),
            "median_oracle_gap_test": float(
                np.median(oracle["test"] - ic_metric["kge_test"])
            ),
        },
        "q_recovery_dpl": {
            "median_kge_train": float(dpl_metric["kge_train"].median()),
            "median_kge_test": float(
                dpl_metric[dpl_metric["run"] == "dPL_median"]["kge_test"].median()
            ),
            "median_eval_ceiling": float(np.median(dpl_ceiling)),
            "median_gap_test": float(
                np.median(
                    dpl_ceiling
                    - dpl_metric[dpl_metric["run"] == "dPL_median"]["kge_test"]
                )
            ),
            "per_seed_median_test_kge": {
                str(s): float(
                    dpl_metric[dpl_metric["run"] == f"dPL_seed{s}"]["kge_test"].median()
                )
                for s in SEEDS
            },
        },
        "equifinality": eq_summary,
        "snow_diagnostics": snow_diag,
        "files": {
            "gate_input_validation.json": "basin alignment + headline KGE checks",
            "parameters_recoverability.csv": "per basin x param: z_hat/z_star/delta_z",
            "parameter_recovery_summary.csv": "per parameter x regime: error/quantiles/corr/slope/bounds/frac_snow",
            "ic_restart_parameter_dispersion.csv": "per basin x param: std/range over 10 starts",
            "dpl_seed_parameter_spread.csv": "per basin x param: std/range over 3 seeds",
            "gate_discharge_metrics.csv": "KGE/NSE/PBIAS vs Q* per fit",
            "kge_vs_parameter_recovery.csv": "per basin: KGE + D_theta",
            "kge_deficit_vs_frac_snow.csv": "per basin: oracle deficits + frac_snow",
            "gate_state_metrics_basin.csv": "per basin x fit x var x period: rmse/nrmse/corr/bias",
            "gate_state_summary.json": "state recovery distributions + frac_snow association",
            "gate_report.md": "concise analysis report",
            "gate_manifest.json": "inputs, code state, definitions",
        },
    }
    write_json(output_dir / "gate_report.json", report)
    print("writing manifest ...", flush=True)
    attr_names = list(_attribute_names())
    manifest = {
        "protocol": "r3_gate_analysis_v1",
        "created_at": _utcnow(),
        "code": git_commit(args.project_root),
        "inputs": {
            "truth_run_id": args.truth_run_id,
            "ic_run_id": args.ic_run_id,
            "dpl_run_prefix": args.dpl_run_prefix,
            "dpl_eval_ceiling_source": "r3_gate_v1/oracle_dpl_audit_windows_canonical.csv (window_offset=-1 rows)",
        },
        "basin_identity": {
            "n_basins": n,
            "basin_id_list_hash": hashlib.sha256(
                "\n".join(basin_ids).encode()
            ).hexdigest(),
            "first_last": [basin_ids[0], basin_ids[-1]],
        },
        "attribute_identity": {
            "n_attributes": len(attr_names),
            "names_hash": hashlib.sha256("\n".join(attr_names).encode()).hexdigest(),
            "frac_snow_index": attr_names.index("frac_snow"),
        },
        "parameter_contract": {
            "order": names,
            "bounds": {
                x: {"lower": specs[x]["lower"], "upper": specs[x]["upper"]}
                for x in names
            },
            "shared_xaj": COMMON_XAJ,
            "cn_only": ["cn_ctg", "cn_kf"],
        },
        "selection_rules": {
            "ic": "best train-KGE restart per basin (lowest start breaks ties); all 10 restarts retained for dispersion",
            "dpl": "best_parameters_physical.npz exported by the dPL runner from best_checkpoint.pt (repository convention)",
        },
        "definitions": {
            "normalized_error": "e[p,i] = (theta_hat - theta_star)/(upper - lower)",
            "D_theta": "median over the 15 shared XAJ parameters of abs(e)",
            "nrmse": "RMSE / (std(truth) + 1e-8)",
            "boundary_contact": "within 1e-9 of lower/upper",
            "season_windows": {k: sorted(v) for k, v in SEASON_MONTHS.items()},
            "state_simulation": "repository recorded forward (production kernels), canonical cn_psol_annual, full 12418-day axis",
            "kge": "standard KGE (see r3/docs/kge_audit.md)",
        },
        "validation": validation,
        "round_trip_theta_star_max_abs_diff": rt_max,
        "outputs": sorted(p.name for p in output_dir.glob("gate_*")),
    }
    write_json(output_dir / "gate_manifest.json", manifest)
    print(f"COMPLETE gate analysis -> {output_dir}", flush=True)


def _attribute_names():
    from ablation.ic_core.data_adapter import ATTRIBUTE_NAMES

    return ATTRIBUTE_NAMES


def load_bundle(project_root: Path, data_root: Path):
    from r3.common import load_bundle as _load

    return _load(project_root, data_root)


if __name__ == "__main__":
    main()
