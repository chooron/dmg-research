#!/usr/bin/env python3
"""Phase A2: oracle forward-identity report across all 531 basins.

For every basin the generating parameters ``theta_star`` are pushed through
each path that the student pipelines actually use, and the simulated
discharge is compared with ``q_star`` on the same target time indices:

- Path 1 (canonical full-axis): recorded forward vs production forward;
- Path 2 (IC objective path): split forcing through the IC model adapter,
  warm-up 365 days, scored train/test slices;
- Path 3 (dPL training-window path): 730-day windows (365 warm-up + 365
  scored) at fixed deterministic offsets;
- Path 4 (dPL evaluation path): 365-day warm-up + the full test period.

Each path is run twice for CN: with the canonical full-record
``cn_psol_annual`` override (the R3 fixed semantics) and without it (the
historical per-sequence ``g_thresh``), so the report quantifies exactly what
the fix removes.  Reported per path: max/mean abs q-diff and KGE vs q_star.
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
PROJECT = HERE.parents[0]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from r3.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    git_commit,
    standard_kge,
    write_json,
)
from r3.recorded_forward import (  # noqa: E402
    build_forcing_dict,
    recorded_cn_forward,
    validate_recorded_forward,
)

WINDOW_START_OFFSETS = (365, 1825)  # deterministic dPL-window probes


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_psol(bundle) -> np.ndarray:
    from models.cemaneige import _estimate_psol_annual

    forcing = torch.from_numpy(bundle.forcing)
    with torch.no_grad():
        return _estimate_psol_annual(forcing[:, :, 0], forcing[:, :, 1]).numpy().astype(
            np.float32, copy=False
        )


def _model_forward(model, forcing_np, params_np, names, device, dtype,
                   psol_annual_np):
    """Run XAJ_CN through the production adapter-compatible path."""
    fc = build_forcing_dict(forcing_np, device, dtype)
    if psol_annual_np is not None:
        fc["cn_psol_annual"] = torch.from_numpy(psol_annual_np).to(device=device, dtype=dtype)
    p = {name: torch.from_numpy(params_np[:, i]).to(device=device, dtype=dtype)
         for i, name in enumerate(names)}
    with torch.no_grad():
        q, _ = model(forcings=fc, params=p)
    return q.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--run-id", default="r3_gate_v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-basins", type=int, default=64)
    parser.add_argument("--max-basins", type=int, default=None, help="smoke limit")
    args = parser.parse_args()

    device = torch.device(args.device)
    truth_dir = args.results_root / args.truth_run_id
    output_dir = args.results_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, _config = load_bundle(args.project_root, args.data_root)
    theta = np.load(truth_dir / "theta_star.npz")
    theta_star = theta["parameters"]
    names = [str(n) for n in theta["parameter_names"]]
    q_star = np.asarray(np.load(truth_dir / "q_star.npz")["target_mm_day"], dtype=np.float64)

    n_all = len(bundle.basin_ids)
    n = n_all if args.max_basins is None else min(args.max_basins, n_all)
    psol = canonical_psol(bundle)
    p = bundle.periods
    warmup = p.warmup.days

    from models import XAJWithCemaNeigeLite

    model = XAJWithCemaNeigeLite().to(device).eval()
    dtype = torch.float32

    rows: list[dict] = []
    batch = args.batch_basins
    for left in range(0, n, batch):
        right = min(n, left + batch)
        idx = np.arange(left, right)
        theta_b = theta_star[left:right]
        psol_b = psol[left:right]

        # --- Path 1: canonical full-axis (recorded vs production) ---
        fc = build_forcing_dict(bundle.forcing[left:right].astype(np.float32), device, dtype)
        p1 = {name: torch.from_numpy(theta_b[:, i]).to(device=device, dtype=dtype)
              for i, name in enumerate(names)}
        recorded = recorded_cn_forward(model, fc, p1, device, dtype)
        diffs = validate_recorded_forward(model, recorded, fc, p1)

        # one row per basin; all paths fill into it
        chunk_rows = [{"basin_id": str(bundle.basin_ids[b]),
                       "path1_q_abs_max": diffs["q_abs_max"]} for b in idx]

        # --- Path 2: IC objective path (train/test splits) ---
        for split, (f_si, f_ei, si, ei) in (
            ("train", (p.train_forcing_start_index, p.train_forcing_end_index,
                       p.train.start_index, p.train.end_index)),
            ("test", (p.test_forcing_start_index, p.test_forcing_end_index,
                      p.test.start_index, p.test.end_index)),
        ):
            forcing_np = bundle.forcing[left:right, f_si:f_ei].astype(np.float32)
            q_c = _model_forward(model, forcing_np, theta_b, names, device, dtype, psol_b)
            q_s = _model_forward(model, forcing_np, theta_b, names, device, dtype, None)
            for v, q in (("canonical", q_c), ("split_gthresh", q_s)):
                q_eval = q[:, warmup:]
                for k, b in enumerate(idx):
                    obs = q_star[b, si:ei + 1]
                    diff = np.abs(q_eval[k].astype(np.float64) - obs)
                    chunk_rows[k][f"path2_{split}_{v}_abs_max"] = float(diff.max())
                    chunk_rows[k][f"path2_{split}_{v}_abs_mean"] = float(diff.mean())
                    chunk_rows[k][f"path2_{split}_{v}_kge"] = standard_kge(q_eval[k], obs)

        # --- Path 3: dPL training-window path (fixed offsets) ---
        for kwin, offset in enumerate(WINDOW_START_OFFSETS):
            s = p.train.start_index + offset
            f_si = s - warmup
            forcing_np = bundle.forcing[left:right, f_si:s + 365].astype(np.float32)
            q_c = _model_forward(model, forcing_np, theta_b, names, device, dtype, psol_b)
            q_s = _model_forward(model, forcing_np, theta_b, names, device, dtype, None)
            for v, q in (("canonical", q_c), ("split_gthresh", q_s)):
                q_scored = q[:, warmup:]
                for k, b in enumerate(idx):
                    obs = q_star[b, s:s + 365]
                    diff = np.abs(q_scored[k].astype(np.float64) - obs)
                    chunk_rows[k][f"path3_win{kwin}_{v}_abs_max"] = float(diff.max())
                    chunk_rows[k][f"path3_win{kwin}_{v}_abs_mean"] = float(diff.mean())
                    chunk_rows[k][f"path3_win{kwin}_{v}_kge"] = standard_kge(q_scored[k], obs)

        # --- Path 4: dPL evaluation path ---
        f_si = p.test_forcing_start_index
        f_ei = p.test_forcing_end_index
        si, ei = p.test.start_index, p.test.end_index
        forcing_np = bundle.forcing[left:right, f_si:f_ei].astype(np.float32)
        q_c = _model_forward(model, forcing_np, theta_b, names, device, dtype, psol_b)
        q_s = _model_forward(model, forcing_np, theta_b, names, device, dtype, None)
        for v, q in (("canonical", q_c), ("split_gthresh", q_s)):
            q_eval = q[:, warmup:]
            for k, b in enumerate(idx):
                obs = q_star[b, si:ei + 1]
                diff = np.abs(q_eval[k].astype(np.float64) - obs)
                chunk_rows[k][f"path4_{v}_abs_max"] = float(diff.max())
                chunk_rows[k][f"path4_{v}_abs_mean"] = float(diff.mean())
                chunk_rows[k][f"path4_{v}_kge"] = standard_kge(q_eval[k], obs)

        rows.extend(chunk_rows)
        print(f"basins {right}/{n} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "oracle_identity_basin.csv", index=False)

    # aggregate summary: medians across basins of the per-basin stats
    summary: dict[str, float] = {}
    for col in df.columns:
        if col == "basin_id":
            continue
        summary[col + "_median"] = float(df[col].median())
        summary[col + "_p95"] = float(df[col].quantile(0.95))
        summary[col + "_max"] = float(df[col].max())
    report = {
        "protocol": "r3_oracle_identity_v1",
        "created_at": _utcnow(),
        "code": git_commit(args.project_root),
        "truth_run_id": args.truth_run_id,
        "n_basins": n,
        "g_thresh_fix": {
            "canonical_definition": (
                "cn_psol_annual = 365.25 * mean(precip * frac_solid(T)) over the "
                "full 12418-day record; g_thresh = 0.9 * cn_psol_annual; identical "
                "quantity in truth generation, IC objective path and dPL window/"
                "eval paths"
            ),
            "historical_definition": "per-input-sequence estimate inside the model",
        },
        "paths": {
            "path1": "canonical full-axis: recorded forward vs production forward (expect ~0)",
            "path2": "IC objective path: split forcing, 365d warm-up, scored train/test",
            "path3": "dPL training-window path: 730d windows at fixed offsets (365+365)",
            "path4": "dPL evaluation path: 365d warm-up + test period",
        },
        "summary": summary,
        "acceptance": (
            "path2/path4 canonical variants and path1 must be float-level; path3 "
            "canonical residual is the frozen 365-day-window warm-up convention "
            "effect only (no g_thresh redefinition)"
        ),
    }
    write_json(output_dir / "oracle_identity.json", report)
    print(f"COMPLETE oracle identity -> {output_dir / 'oracle_identity.json'}", flush=True)


def load_bundle(project_root: Path, data_root: Path):
    from r3.common import load_bundle as _load

    return _load(project_root, data_root)


if __name__ == "__main__":
    main()
