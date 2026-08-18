#!/usr/bin/env python3
"""Export compact seasonal process trajectories for R3 Figure 6.

This is a deterministic inference replay of the frozen R3 fitted parameters;
it does not train, calibrate, or alter canonical results.  The replay uses the
same recorded-forward kernels as ``manuscript/r3/posthoc_stats.py`` and retains only
per-basin water-year-month means over the frozen test period.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ablation.ic_core.parameter_adapter import get_parameter_spec  # noqa: E402
from models import XAJLite, XAJWithCemaNeigeLite, XAJWithTGD2Lite  # noqa: E402

from manuscript.r3.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_ROOT,
    frac_snow_series,
    git_commit,
    load_bundle,
    period_indices,
)
from manuscript.r3.gate_analysis import (  # noqa: E402
    load_dpl_estimates,
    load_ic_estimates,
)
from manuscript.r3.recorded_forward import (  # noqa: E402
    build_forcing_dict,
    recorded_forward_for_structure,
)

SEEDS = (42, 123, 2026)
BATCH = 50
MONTH_LABELS = (
    "Oct",
    "Nov",
    "Dec",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
)


def fit_catalog(results_root: Path):
    """Return fit -> (structure, run directory, loader kind, seed)."""
    out = {
        "Base_IC": ("XAJ", results_root / "r3_misspec_ic_xaj_531_v1", "ic", None),
        "TGD2_IC": (
            "XAJ_TGD2",
            results_root / "r3_misspec_ic_xaj_tgd2_531_v1",
            "ic",
            None,
        ),
        "CN_IC": ("XAJ_CN", results_root / "r3_gate_ic_xaj_cn_531_v1", "ic", None),
    }
    for seed in SEEDS:
        out[f"Base_dPL_s{seed}"] = (
            "XAJ",
            results_root / f"r3_misspec_dpl_xaj_seed_{seed}",
            "dpl",
            seed,
        )
        out[f"TGD2_dPL_s{seed}"] = (
            "XAJ_TGD2",
            results_root / f"r3_misspec_dpl_xaj_tgd2_seed_{seed}",
            "dpl",
            seed,
        )
        out[f"CN_dPL_s{seed}"] = (
            "XAJ_CN",
            results_root / f"r3_gate_dpl_xaj_cn_seed_{seed}",
            "dpl",
            seed,
        )
    return out


def monthly_index(dates: np.ndarray, test_start: int, test_end: int) -> np.ndarray:
    """Return water-year month indices (Oct=0,...,Sep=11) for test dates."""
    months = np.asarray([int(str(d)[5:7]) for d in dates[test_start : test_end + 1]])
    return ((months - 10) % 12).astype(np.int64)


def month_means(values: np.ndarray, wy_month: np.ndarray) -> np.ndarray:
    """Average [batch,time] daily values into [batch,12] water-year months."""
    result = np.empty((values.shape[0], 12), dtype=np.float32)
    for month in range(12):
        result[:, month] = values[:, wy_month == month].mean(axis=1)
    return result


def theta_matrix(
    run_dir: Path, kind: str, structure: str, basin_ids: list[str]
) -> np.ndarray:
    if kind == "ic":
        estimates = load_ic_estimates(run_dir, basin_ids)
    else:
        estimates = load_dpl_estimates(run_dir, basin_ids)
    return np.stack([estimates[b]["theta_hat"] for b in basin_ids]).astype(np.float64)


def replay_profile(
    model,
    structure: str,
    theta_hat: np.ndarray,
    forcing: np.ndarray,
    parameter_names: tuple[str, ...],
    test_start: int,
    test_end: int,
    wy_month: np.ndarray,
    keep: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay one fit and return [n_selected,12] input and wt profiles."""
    n = theta_hat.shape[0]
    input_profile = np.empty((n, 12), dtype=np.float32)
    state_profile = np.empty((n, 12), dtype=np.float32)
    dtype = torch.float32
    for left in range(0, n, BATCH):
        right = min(n, left + BATCH)
        fc = build_forcing_dict(forcing[left:right].astype(np.float32), device, dtype)
        params = {
            name: torch.from_numpy(theta_hat[left:right, i]).to(
                device=device, dtype=dtype
            )
            for i, name in enumerate(parameter_names)
        }
        with torch.no_grad():
            _qsim, stores, _final = recorded_forward_for_structure(
                structure, model, fc, params, device, dtype
            )
            if structure == "XAJ":
                inp = forcing[left:right, test_start : test_end + 1, 0]
            else:
                inp = (
                    stores["effective_precip"][:, test_start : test_end + 1]
                    .detach()
                    .cpu()
                    .numpy()
                )
            wt = (
                (
                    stores["wu"][:, test_start : test_end + 1]
                    + stores["wl"][:, test_start : test_end + 1]
                    + stores["wd"][:, test_start : test_end + 1]
                )
                .detach()
                .cpu()
                .numpy()
            )
        input_profile[left:right] = month_means(
            np.asarray(inp, dtype=np.float32), wy_month
        )
        state_profile[left:right] = month_means(wt.astype(np.float32), wy_month)
        print(f"    batch {left}:{right}/{n}", flush=True)
    return input_profile[keep], state_profile[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "manuscript/results/R3/fig6_seasonal",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    bundle, _ = load_bundle(args.project_root, args.data_root)
    ids = [str(x).zfill(8) for x in bundle.basin_ids]
    pi = period_indices(bundle)
    test_start, test_end = pi["test"]
    wy_month = monthly_index(bundle.dates, test_start, test_end)

    fs = (
        frac_snow_series(bundle)
        .set_index("basin_id")
        .loc[ids, "frac_snow"]
        .to_numpy(float)
    )
    threshold = float(np.quantile(fs, 0.75))
    keep = np.flatnonzero(fs >= threshold)
    high_ids = [ids[i] for i in keep]
    print(
        f"device={device}; high-snow threshold={threshold:.12g}; n={len(keep)}",
        flush=True,
    )
    print(
        f"test indices={test_start}:{test_end}; days={test_end - test_start + 1}",
        flush=True,
    )

    models = {
        "XAJ": XAJLite().to(device).eval(),
        "XAJ_TGD2": XAJWithTGD2Lite().to(device).eval(),
        "XAJ_CN": XAJWithCemaNeigeLite().to(device).eval(),
    }
    catalog = fit_catalog(args.results_root)
    for fit, (_structure, run_dir, _kind, _seed) in catalog.items():
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)

    # Keep the exact per-fit profiles temporarily; dPL is median across seeds
    # per basin before any across-basin summary in the plotting script.
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for fit, (structure, run_dir, kind, _seed) in catalog.items():
        print(f"[replay] {fit} ({structure})", flush=True)
        theta = theta_matrix(run_dir, kind, structure, ids)
        names = tuple(get_parameter_spec(structure))
        inp, state = replay_profile(
            models[structure],
            structure,
            theta,
            bundle.forcing,
            names,
            test_start,
            test_end,
            wy_month,
            keep,
            device,
        )
        profiles[fit] = {"input": inp, "state": state}
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, np.ndarray]] = {"input": {}, "state": {}}
    for structure_name in ("Base", "TGD2", "CN"):
        for quantity in ("input", "state"):
            ic = profiles[f"{structure_name}_IC"][quantity]
            dpl = np.median(
                np.stack(
                    [profiles[f"{structure_name}_dPL_s{s}"][quantity] for s in SEEDS]
                ),
                axis=0,
            )
            result[quantity][f"{structure_name}_IC"] = ic
            result[quantity][f"{structure_name}_dPL"] = dpl

    np.savez_compressed(
        out / "fig6_seasonal_input.npz",
        basin_ids=np.asarray(high_ids),
        **{key: value for key, value in result["input"].items()},
    )
    np.savez_compressed(
        out / "fig6_seasonal_state.npz",
        basin_ids=np.asarray(high_ids),
        **{key: value for key, value in result["state"].items()},
    )
    meta = {
        "protocol": "figure6_process_reexport_v1",
        "provenance": "deterministic recorded-forward replay of frozen fitted parameters; no training, calibration, or new experiment",
        "git": git_commit(args.project_root),
        "device": str(device),
        "fit_catalog": {
            k: {"structure": v[0], "run_dir": str(v[1]), "kind": v[2], "seed": v[3]}
            for k, v in catalog.items()
        },
        "quantity": "effective liquid-water input entering the shared XAJ core (Base raw precipitation; TGD2 delayed effective precipitation; CN rain plus snowmelt)",
        "state": "wt = wu + wl + wd, total XAJ tension-water storage (shared derived state)",
        "shared_state_components": ["wu", "wl", "wd"],
        "subset": {
            "criterion": "frac_snow >= upper quartile",
            "threshold": threshold,
            "n_basins": len(keep),
            "basin_ids": high_ids,
        },
        "period": {
            "name": "test",
            "start": str(bundle.dates[test_start]),
            "end": str(bundle.dates[test_end]),
            "n_days": int(test_end - test_start + 1),
        },
        "seasonal_axis": {
            "type": "water-year month",
            "start": "October 1",
            "labels": list(MONTH_LABELS),
        },
        "aggregation": "per-basin mean over each water-year month across the test period; dPL median across seeds per basin; plot median and IQR across high-snow basins",
        "output_shapes": {k: list(v.shape) for k, v in result["input"].items()},
        "runtime_seconds": time.time() - t0,
    }
    (out / "fig6_seasonal_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}; runtime={meta['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
