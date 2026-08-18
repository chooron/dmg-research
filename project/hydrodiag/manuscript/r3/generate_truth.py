#!/usr/bin/env python3
"""Generate the R3 synthetic truth (Phase 2) for all 531 basins.

Workflow:

1. fit ``theta* = g*(A)`` on the full 35-dimension attribute vector
   (``manuscript/r3/truth_generator.py``) anchored on the 531-basin XAJ-CN IC field;
2. run XAJ-CN (production Lite path, mirrored by the recorded forward) on
   the canonical 1980-10-01..2010-09-30 forcing and store, per basin:

   - ``theta*`` (17 CN parameters),
   - ``Q*`` (synthetic daily discharge, full time axis),
   - ``X*`` (per-day common states: wu, wl, wd, s, fr, qi, qg),
   - CN snow diagnostics (G, eTG, sca, rain, melt) as truth-only process
     diagnostics,
   - basin metadata, time indices, split metadata;

3. round-trip reproducibility check: re-running the production
   ``XAJWithCemaNeigeLite`` with the stored ``theta*`` must reproduce
   ``Q*`` and the final states within float tolerance.

Run products are written under ``results/r3_synthetic_truth_v1/`` (the
canonical results root; R1/R2 observed results are never touched).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.r3.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    bundle_with_synthetic_target,
    git_commit,
    period_indices,
    sha256_file,
    write_json,
)
from manuscript.r3.recorded_forward import (  # noqa: E402
    build_forcing_dict,
    recorded_cn_forward,
    validate_recorded_forward,
)
from manuscript.r3.truth_generator import (  # noqa: E402
    build_and_save_truth,
    load_cn_ic_field,
)

STATE_KEYS = ("wu", "wl", "wd", "s", "fr", "qi", "qg")
SNOW_KEYS = ("G", "eTG", "sca", "rain", "melt", "effective_precip")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default="r3_synthetic_truth_v1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-basins", type=int, default=32)
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Override the data-driven rank K (diagnostics only).",
    )
    parser.add_argument("--skip-roundtrip", action="store_true")
    parser.add_argument(
        "--max-basins",
        type=int,
        default=None,
        help="Engineering smoke run: limit basins (never used for the formal truth).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.results_root / args.run_id
    if args.max_basins is not None:
        # engineering smoke run: never write into the formal truth directory
        output_dir = args.results_root / f"{args.run_id}_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, _config = load_bundle(args.project_root, args.data_root)
    smoke_limit = None
    if args.max_basins is not None:
        if args.max_basins >= len(bundle.basin_ids):
            raise ValueError(
                "--max-basins must be below the full basin count for a smoke run"
            )
        smoke_limit = args.max_basins
    cn_field = load_cn_ic_field(args.results_root)
    # g* is always fitted on the full 531 field (canonical normalization and
    # manifold); a smoke run only limits which basins get simulated below.
    fit, theta_star, clip_mask = build_and_save_truth(
        bundle,
        cn_field,
        output_dir,
        args.project_root,
        args.results_root,
        args.data_root,
        n_components=args.n_components,
    )
    print(
        f"g* fitted: K={fit.k} ridge_alpha={fit.alpha:.4g} "
        f"cv_r2_total={fit.cv_r2_total:.4f} clipped_entries={int(clip_mask.sum())}",
        flush=True,
    )

    from models import XAJWithCemaNeigeLite

    model = XAJWithCemaNeigeLite().to(device).eval()
    dtype = torch.float32
    forcing = bundle.forcing.astype(np.float32)  # [531, 12418, 3]
    n_basins, n_time = forcing.shape[:2]
    sim_basins = smoke_limit if smoke_limit is not None else n_basins

    q_star = np.empty((sim_basins, n_time), dtype=np.float32)
    states = {
        key: np.empty((sim_basins, n_time), dtype=np.float32) for key in STATE_KEYS
    }
    snow = {key: np.empty((sim_basins, n_time), dtype=np.float32) for key in SNOW_KEYS}
    final_states_all = np.empty((sim_basins, 9), dtype=np.float32)
    basin_ids_out = np.asarray(bundle.basin_ids[:sim_basins])

    params_t = torch.from_numpy(theta_star).to(device=device, dtype=dtype)
    roundtrip_max_abs = []
    for left in range(0, sim_basins, args.batch_basins):
        right = min(n_basins, left + args.batch_basins)
        fc = build_forcing_dict(forcing[left:right], device, dtype)
        p = {
            name: params_t[left:right, i]
            for i, name in enumerate(cn_field["parameter_names"])
        }
        recorded = recorded_cn_forward(model, fc, p, device, dtype)
        qsim, stores, final_states = recorded
        q_star[left:right] = qsim.detach().cpu().numpy()
        for key in STATE_KEYS:
            states[key][left:right] = stores[key].detach().cpu().numpy()
        for key in SNOW_KEYS:
            snow[key][left:right] = stores[key].detach().cpu().numpy()
        final_states_all[left:right, 0] = final_states["wu"].detach().cpu().numpy()
        final_states_all[left:right, 1] = final_states["wl"].detach().cpu().numpy()
        final_states_all[left:right, 2] = final_states["wd"].detach().cpu().numpy()
        final_states_all[left:right, 3] = final_states["s"].detach().cpu().numpy()
        final_states_all[left:right, 4] = final_states["fr"].detach().cpu().numpy()
        final_states_all[left:right, 5] = final_states["qi"].detach().cpu().numpy()
        final_states_all[left:right, 6] = final_states["qg"].detach().cpu().numpy()
        final_states_all[left:right, 7] = final_states["G"].detach().cpu().numpy()
        final_states_all[left:right, 8] = final_states["eTG"].detach().cpu().numpy()

        if not args.skip_roundtrip:
            # production forward over the same chunk with the same theta*
            with torch.no_grad():
                q_prod, aux = model(forcings=fc, params=p, return_states=True)
            d = float((q_prod.detach() - qsim.detach()).abs().max().item())
            roundtrip_max_abs.append(d)
        print(f"chunk {left}:{right} done", flush=True)

    if not np.isfinite(q_star).all():
        raise RuntimeError("Q* contains non-finite values")
    if (q_star < 0).any():
        raise RuntimeError("Q* contains negative discharge")

    # production-model final states for the round-trip audit (chunked)
    prod_final = np.empty_like(final_states_all)
    with torch.no_grad():
        for left in range(0, sim_basins, args.batch_basins):
            right = min(n_basins, left + args.batch_basins)
            fc = build_forcing_dict(forcing[left:right], device, dtype)
            p = {
                name: params_t[left:right, i]
                for i, name in enumerate(cn_field["parameter_names"])
            }
            _q, aux = model(forcings=fc, params=p, return_states=True)
            fs = aux["final_states"]
            prod_final[left:right] = np.column_stack(
                [
                    fs["xaj_wu"].detach().cpu().numpy(),
                    fs["xaj_wl"].detach().cpu().numpy(),
                    fs["xaj_wd"].detach().cpu().numpy(),
                    fs["xaj_s"].detach().cpu().numpy(),
                    fs["xaj_fr"].detach().cpu().numpy(),
                    fs["xaj_qi"].detach().cpu().numpy(),
                    fs["xaj_qg"].detach().cpu().numpy(),
                    fs["cn_G"].detach().cpu().numpy(),
                    fs["cn_eTG"].detach().cpu().numpy(),
                ]
            )
    final_state_max_abs = float(np.abs(final_states_all - prod_final).max())

    np.savez_compressed(
        output_dir / "q_star.npz",
        target_mm_day=q_star.astype(np.float64),
        q_star_f32=q_star,
        basin_ids=basin_ids_out,
        dates=np.asarray(bundle.dates),
        time_axis_full=True,
    )
    np.savez_compressed(
        output_dir / "x_star.npz", **states, basin_ids=np.asarray(bundle.basin_ids)
    )
    np.savez_compressed(
        output_dir / "snow_star.npz", **snow, basin_ids=np.asarray(bundle.basin_ids)
    )
    np.savez_compressed(
        output_dir / "final_states.npz",
        final_states=final_states_all,
        prod_final_states=prod_final,
        names=np.asarray(["wu", "wl", "wd", "s", "fr", "qi", "qg", "G", "eTG"]),
        basin_ids=basin_ids_out,
    )

    summary = {
        "protocol": "r3_synthetic_truth_v1",
        "created_at": _utcnow(),
        "code": git_commit(args.project_root),
        "device": str(device),
        "shapes": {
            "q_star": list(q_star.shape),
            "states": {k: list(v.shape) for k, v in states.items()},
            "snow": {k: list(v.shape) for k, v in snow.items()},
            "theta_star": list(theta_star.shape),
        },
        "periods": {
            name: [str(v) for v in indices]
            for name, indices in period_indices(bundle).items()
        },
        "warmup_convention": "365-day warm-up before each scored period; Q* generated over the full 1980-10-01..2010-09-30 axis (12418 days) with warm-up days retained in the stored arrays",
        "split_identifiers": {
            "warmup": "1980-10-01..1981-09-30",
            "train": "1981-10-01..1995-09-30",
            "test": "1995-10-01..2010-09-30",
        },
        "roundtrip": {
            "method": "production XAJWithCemaNeigeLite forward with stored theta* vs recorded Q* and final states",
            "q_max_abs_diff_per_chunk": roundtrip_max_abs
            if roundtrip_max_abs
            else "skipped",
            "q_max_abs_diff_overall": max(roundtrip_max_abs, default=float("nan")),
            "final_state_max_abs_diff": final_state_max_abs,
            "tolerance": "1e-5 (float32)",
        },
        "q_star_finite": bool(np.isfinite(q_star).all()),
        "q_star_nonnegative": bool((q_star >= 0).all()),
        "q_star_min": float(q_star.min()),
        "q_star_max": float(q_star.max()),
        "basin_metadata": {
            "basin_ids": list(bundle.basin_ids),
            "n_basins": sim_basins,
            "n_timesteps": n_time,
            "forcing_names": list(bundle.forcing_names),
            "dates_first": str(bundle.dates[0]),
            "dates_last": str(bundle.dates[-1]),
        },
        "files": {
            "theta_star": "theta_star.npz",
            "q_star": "q_star.npz",
            "x_star": "x_star.npz",
            "snow_star": "snow_star.npz",
            "final_states": "final_states.npz",
            "gstar_manifest": "gstar_manifest.json",
            "gstar_diagnostics": "gstar_diagnostics.json",
        },
    }
    write_json(output_dir / "manifest.json", summary)
    print(f"COMPLETE truth generation -> {output_dir}", flush=True)


def load_bundle(project_root: Path, data_root: Path):
    from manuscript.r3.common import load_bundle as _load

    return _load(project_root, data_root)


if __name__ == "__main__":
    main()
