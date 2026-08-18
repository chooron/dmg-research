#!/usr/bin/env python3
"""R3 pilot analysis: machine-readable gate outputs (Phases 4-5).

Reads the IC raw JSON records and dPL parameter exports produced by
``pilot.py`` and computes, per pilot basin:

- discharge metrics vs Q* (KGE/NSE/PBIAS, train and test);
- objective-path oracle KGE for the correct-structure CN runs (theta*
  evaluated through the exact IC objective path);
- shared-XAJ parameter estimates and truth-relative normalized deviations
  (``z_hat - z*`` in unit-normalized coordinates; RMS over the 15 shared
  parameters, plus per-parameter rows);
- common internal states and truth-relative errors (RMSE per state vs X*;
  fixed a-priori calendar-season decomposition DJF/MAM/JJAS);
- ``frac_snow`` and run metadata (seed/restart/checkpoint/failure flags).

No hard thresholds are applied; this is the identifiability baseline for
freezing the formal-run parameter subset.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.r3.common import (  # noqa: E402
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
from manuscript.r3.recorded_forward import (  # noqa: E402
    build_forcing_dict,
    recorded_forward_for_structure,
)

# Fixed a-priori calendar snow-season windows (documented, not tuned):
# DJF = accumulation, MAM = melt, JJAS = non-snow.  All comparisons use
# these windows unchanged.
SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9)}
COMMON_STATES = ("wu", "wl", "wd", "s", "fr", "qi", "qg")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ic_estimates(
    run_dir: Path, model: str, basin_ids: list[str]
) -> dict[str, dict]:
    """Best train-KGE restart per basin (R1 canonical rule)."""
    raw_dir = run_dir / "raw" / model.lower()
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
            raise ValueError(f"no complete IC restart for {model} basin {basin}")
        candidates.sort(key=lambda item: (-item[0], item[1]))
        _kge, start, data = candidates[0]
        names = tuple(data["parameter_names"])
        out[basin] = {
            "theta_hat": np.asarray(data["parameters"], dtype=np.float64),
            "names": names,
            "restart": start,
            "seed": data.get("seed"),
            "train_kge": _kge,
            "stored_test_kge": float(data.get("test_metrics", {}).get("kge", np.nan)),
            "source_file": str(path),
            "candidate_evaluations": data.get("candidate_evaluations"),
            "generations": data.get("generations"),
        }
    return out


def load_dpl_estimates(
    run_dir: Path, model: str, basin_ids: list[str]
) -> dict[str, dict]:
    params = np.load(run_dir / "best_parameters_physical.npz")["params"]
    config = json.loads((run_dir / "config.json").read_text())
    names = tuple(config["parameter_names"])
    if params.shape[0] != len(basin_ids):
        raise ValueError(
            f"dPL parameter rows {params.shape[0]} != pilot basins {len(basin_ids)}"
        )
    out: dict[str, dict] = {}
    for i, basin in enumerate(basin_ids):
        out[basin] = {
            "theta_hat": params[i].astype(np.float64),
            "names": names,
            "restart": None,
            "seed": config.get("training", {}).get("seed"),
            "source_file": str(run_dir / "best_parameters_physical.npz"),
            "train_kge": np.nan,
            "stored_test_kge": np.nan,
            "candidate_evaluations": None,
            "generations": None,
        }
    return out


def oracle_kge_cn(
    bundle,
    q_star: np.ndarray,
    theta_star: np.ndarray,
    basin_ids: list[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """KGE of theta* through the exact IC objective path (Q* target)."""
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
    index = {b: i for i, b in enumerate(bundle.basin_ids)}
    basin_indices = [index[b] for b in basin_ids]
    theta_01 = physical_to_normalized(
        "XAJ_CN", theta_star[np.asarray(basin_indices)], clip=False
    )
    result = {}
    for split in ("train", "test"):
        fit, _ = runtime.evaluate_candidates_tensor(
            torch.from_numpy(theta_01).unsqueeze(1).to(device, dtype=torch.float64),
            basin_indices=basin_indices,
            split=split,
        )
        result[split] = fit[:, 0].detach().cpu().numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--pilot-run-id", default="r3_pilot_v1")
    parser.add_argument("--run-id", default="r3_pilot_analysis_v1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    truth_dir = args.results_root / args.truth_run_id
    pilot_dir = args.results_root / args.pilot_run_id
    output_dir = args.results_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pilot_manifest = json.loads((pilot_dir / "pilot_manifest.json").read_text())
    basin_ids = list(pilot_manifest["pilot_basins"])
    bundle, _config = load_bundle(args.project_root, args.data_root)
    pi = period_indices(bundle)
    dates = pd.to_datetime(bundle.dates)
    q_star = np.asarray(
        np.load(truth_dir / "q_star.npz")["target_mm_day"], dtype=np.float64
    )
    x_star = np.load(truth_dir / "x_star.npz")
    theta_npz = np.load(truth_dir / "theta_star.npz")
    theta_star = theta_npz["parameters"]
    star_names = [str(n) for n in theta_npz["parameter_names"]]
    shared_star = theta_star[:, [star_names.index(n) for n in COMMON_XAJ]]

    from models.parameter_specs import (
        XAJ_CN_PARAM_SPECS,
        XAJ_PARAM_SPECS,
        XAJ_TGD2_PARAM_SPECS,
    )

    spec_for = {
        "XAJ_CN": XAJ_CN_PARAM_SPECS,
        "XAJ": XAJ_PARAM_SPECS,
        "XAJ_TGD2": XAJ_TGD2_PARAM_SPECS,
    }

    snow = frac_snow_series(bundle).set_index("basin_id")["frac_snow"]

    def state_rmse(sim: np.ndarray, truth: np.ndarray) -> float:
        mask = np.isfinite(sim) & np.isfinite(truth)
        if mask.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((sim[mask] - truth[mask]) ** 2)))

    param_rows: list[dict] = []
    metric_rows: list[dict] = []
    state_rows: list[dict] = []
    run_flags: dict[str, dict] = {}

    cn_oracle: dict[str, np.ndarray] | None = None

    for stage_key, stage in pilot_manifest["stages"].items():
        kind, *rest = stage_key.split(":")
        if kind == "ic":
            model = rest[0]
            estimates = load_ic_estimates(Path(stage["output"]), model, basin_ids)
            run_flags[stage_key] = {
                "status": "complete",
                "checkpoint": stage.get("checkpoint"),
                "done_marker": (Path(stage["output"]) / "DONE.json").exists(),
                "records": len(estimates),
            }
            run_label = f"IC-{model}"
        else:
            model, seed = rest
            run_dir = Path(stage["output_dir"])
            estimates = load_dpl_estimates(run_dir, model, basin_ids)
            run_flags[stage_key] = {
                "status": "complete"
                if (run_dir / "COMPLETE").exists()
                else "missing_COMPLETE",
                "checkpoint": str(run_dir / "best_checkpoint.pt"),
                "epoch_history": (run_dir / "epoch_history.csv").exists(),
                "records": len(estimates),
            }
            run_label = f"dPL-{model}-seed{seed}"

        if model == "XAJ_CN" and cn_oracle is None and kind == "ic":
            cn_oracle = oracle_kge_cn(bundle, q_star, theta_star, basin_ids, device)
            for i, basin in enumerate(basin_ids):
                param_rows.append(
                    {
                        "basin_id": basin,
                        "paradigm": "oracle",
                        "structure": "CN",
                        "run": "theta*",
                        "parameter": "KGE_oracle",
                        "z_hat": float("nan"),
                        "z_star": float("nan"),
                        "delta_z": float("nan"),
                        "kge_train": float(cn_oracle["train"][i]),
                        "kge_test": float(cn_oracle["test"][i]),
                    }
                )

        for basin in basin_ids:
            est = estimates[basin]
            names = est["names"]
            spec = spec_for[model]
            lower = np.asarray([spec[n]["lower"] for n in names], dtype=np.float64)
            upper = np.asarray([spec[n]["upper"] for n in names], dtype=np.float64)
            z_hat_full = (est["theta_hat"] - lower) / (upper - lower)
            theta_hat = est["theta_hat"]

            # simulate with the fitted parameters on the full axis
            from ablation.ic_core.model_adapter import (
                LITE_MODEL_CLASSES,  # noqa: PLC0415
            )

            model_cls = LITE_MODEL_CLASSES[model]
            m = model_cls().to(device).eval()
            idx = list(bundle.basin_ids).index(basin)
            fc = build_forcing_dict(
                bundle.forcing[idx : idx + 1].astype(np.float32), device, torch.float32
            )
            p = {
                name: torch.tensor([theta_hat[i]], device=device, dtype=torch.float32)
                for i, name in enumerate(names)
            }
            qsim, stores, _fs = recorded_forward_for_structure(
                model, m, fc, p, device, torch.float32
            )
            q_full = qsim.detach().cpu().numpy()[0].astype(np.float64)

            metric_row = {
                "basin_id": basin,
                "paradigm": "IC" if kind == "ic" else "dPL",
                "structure": model,
                "run": run_label,
            }
            for period, (si, ei) in (("train", pi["train"]), ("test", pi["test"])):
                sim = q_full[si : ei + 1]
                obs = q_star[idx, si : ei + 1]
                metric_row[f"kge_{period}"] = standard_kge(sim, obs)
                metric_row[f"nse_{period}"] = nse(sim, obs)
                metric_row[f"pbias_{period}"] = pbias(sim, obs)
            metric_rows.append(metric_row)

            # shared-parameter deviations
            z_star_shared = (
                shared_star[idx]
                - np.asarray([XAJ_PARAM_SPECS[n]["lower"] for n in COMMON_XAJ])
            ) / (
                np.asarray([XAJ_PARAM_SPECS[n]["upper"] for n in COMMON_XAJ])
                - np.asarray([XAJ_PARAM_SPECS[n]["lower"] for n in COMMON_XAJ])
            )
            for n in COMMON_XAJ:
                if n not in names:
                    raise ValueError(f"shared parameter {n} missing from {model} fit")
                j = names.index(n)
                z_hat = z_hat_full[j]
                param_rows.append(
                    {
                        "basin_id": basin,
                        "paradigm": "IC" if kind == "ic" else "dPL",
                        "structure": model,
                        "run": run_label,
                        "parameter": n,
                        "value_physical": float(theta_hat[j]),
                        "z_hat": float(z_hat),
                        "z_star": float(z_star_shared[COMMON_XAJ.index(n)]),
                        "delta_z": float(z_hat - z_star_shared[COMMON_XAJ.index(n)]),
                        "restart_or_seed": est["restart"]
                        if est["restart"] is not None
                        else est["seed"],
                        "kge_train": float(est["train_kge"])
                        if np.isfinite(est["train_kge"])
                        else metric_row["kge_train"],
                    }
                )

            # state errors vs X* (common states only)
            state_row = {
                "basin_id": basin,
                "paradigm": "IC" if kind == "ic" else "dPL",
                "structure": model,
                "run": run_label,
            }
            months = dates.month.to_numpy()
            for period, (si, ei) in (("train", pi["train"]), ("test", pi["test"])):
                for key in COMMON_STATES:
                    sim = (
                        stores[key]
                        .detach()
                        .cpu()
                        .numpy()[0, si : ei + 1]
                        .astype(np.float64)
                    )
                    truth = x_star[key][idx, si : ei + 1].astype(np.float64)
                    state_row[f"rmse_{key}_{period}"] = state_rmse(sim, truth)
            for season, months_tuple in SEASON_MONTHS.items():
                sel = np.isin(months, list(months_tuple))
                for key in ("wu", "wl", "wd", "s", "fr"):
                    sim = stores[key].detach().cpu().numpy()[0, sel].astype(np.float64)
                    truth = x_star[key][idx, sel].astype(np.float64)
                    state_row[f"rmse_{key}_{season}"] = state_rmse(sim, truth)
            state_rows.append(state_row)

    pd.DataFrame(param_rows).to_csv(
        output_dir / "parameter_deviations.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(output_dir / "discharge_metrics.csv", index=False)
    pd.DataFrame(state_rows).to_csv(output_dir / "state_metrics.csv", index=False)
    write_json(output_dir / "run_flags.json", run_flags)

    summary = {
        "protocol": "r3_pilot_analysis_v1",
        "created_at": _utcnow(),
        "code": git_commit(args.project_root),
        "pilot_basins": basin_ids,
        "frac_snow": {b: float(snow[b]) for b in basin_ids},
        "truth_run_id": args.truth_run_id,
        "pilot_run_id": args.pilot_run_id,
        "snow_season_windows": {k: sorted(v) for k, v in SEASON_MONTHS.items()},
        "oracle_definition": "theta* through the IC objective path (Q* target), KGE(Q*) train/test",
        "parameter_baseline": "delta_z = z_hat - z* in unit-normalized coordinates; correct-CN recoverability is the identifiability baseline",
        "files": {
            "parameter_deviations.csv": "per-parameter z_hat/z_star/delta_z",
            "discharge_metrics.csv": "KGE/NSE/PBIAS vs Q* per basin/run",
            "state_metrics.csv": "per-state RMSE vs X* (train/test + DJF/MAM/JJAS)",
            "run_flags.json": "run status/checkpoint/failure flags",
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"COMPLETE pilot analysis -> {output_dir}", flush=True)


def load_bundle(project_root: Path, data_root: Path):
    from manuscript.r3.common import load_bundle as _load

    return _load(project_root, data_root)


if __name__ == "__main__":
    main()
