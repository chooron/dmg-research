#!/usr/bin/env python3
"""
Re-evaluate Full300 (gen-300, best-of-N) IC frozen parameters on the aligned
1995-10-01..2010-09-30 window with a 365-day warmup, using the EXACT protocol
of the all36 aligned diagnosis: streaming_kge(eps=0.1), parameter_mapping=linear.

This replaces the previous aligned IC column that (incorrectly) loaded
pilot-run (gen-30, best-of-5) checkpoints from
remote_runs/20260729_120525.  The correct source is the full300 run
(full300_20260729_160112, incl. the 2026-07-30 continuation).

Provenance guards:
  * only directories with a DONE marker and chunk_*_gen_{generation}.pt are read;
  * checkpoint resolved_config.generation must equal the requested generation;
  * per-model best_fitness (train KGE) median is recorded for sanity checks.

Usage:
  python evaluate_ic_aligned_gen300.py --ckpt-root <dir> --generation 300 \
      --starts 10 --out results/all36_ic_gen300_aligned_20260812
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from src.checkpoint_guard import validate_canonical_checkpoint
from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from src.data_selection import frozen_parameters, load_ids
from src.model_registry import build_model
from src.objective import streaming_kge

EVAL_START, EVAL_END = "1995-10-01", "2010-09-30"
WARM_START = "1994-10-01"
WARMUP_DAYS = 365


def load_window(ids: list[int], device: torch.device):
    """Forcing from WARM_START..EVAL_END (incl. warmup), targets EVAL_START..EVAL_END."""
    with open(REPO_ROOT / "data/camels_dataset", "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        forcings, target, attrs = bundle["forcings"], bundle["streamflow"], bundle["attributes"]
    else:
        forcings, target, attrs = bundle
    reference = np.load(REPO_ROOT / "data/gage_id.npy")
    idx = np.array([np.where(reference == int(b))[0][0] for b in ids])
    dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
    wl = dates.get_loc(pd.Timestamp(WARM_START))
    ys = dates.get_loc(pd.Timestamp(EVAL_START))
    wr = dates.get_loc(pd.Timestamp(EVAL_END)) + 1
    fx = forcings[idx, wl:wr, :3]
    fy = target[idx, ys:wr, 0].copy()
    area = attrs[idx, 11].astype(float)
    fy *= (0.0283168 * 86400.0 * 1e3 / (area * 1e6))[:, None]
    x = torch.as_tensor(np.transpose(fx, (1, 0, 2)), dtype=torch.float64, device=device)
    y = torch.as_tensor(fy.T, dtype=torch.float64, device=device)
    return x, y


def evaluate_model(model: str, ckpt_dir: Path, generation: int, starts: int,
                   device: torch.device) -> dict:
    # Canonical provenance guard: gen-300, 531-basin, DONE-marked final set only.
    validate_canonical_checkpoint(
        ckpt_dir, model_name=model,
        required_generation=generation,
        required_basins=531,
        required_basin_ids=load_ids(REPO_ROOT / "data/531sub_id.txt"),
    )
    basin_ids, latent, ckpt_train = frozen_parameters(ckpt_dir, generation, starts)

    x, y = load_window([int(b) for b in basin_ids], device)
    if model in CALENDAR_MODELS:
        x, _ = add_calendar_forcing(
            x, pd.date_range(WARM_START, EVAL_END, freq="D"), model_name=model
        )
    model_inst = build_model(model, device, warm_up=WARMUP_DAYS, backend="compile", dtype=torch.float64)
    raw = torch.sigmoid(latent.to(device=device, dtype=torch.float64)).unsqueeze(-1)
    with torch.inference_mode():
        q = model_inst({"x_phy": x}, (None, raw))["streamflow"].reshape(-1, len(basin_ids), 1, 1)
        score, invalid = streaming_kge(q, y)
    kge = score[:, 0, 0].detach().cpu().numpy()
    return {
        "model": model,
        "generation": generation,
        "starts": starts,
        "n_basins": len(basin_ids),
        "n_invalid": int(invalid.sum().item()),
        "median_kge": float(np.nanmedian(kge)),
        "mean_kge": float(np.nanmean(kge)),
        "ckpt_train_kge_median": float(np.median(ckpt_train)),
        "ckpt_source": str(ckpt_dir),
        "basin_ids": np.asarray(basin_ids, dtype=np.int64),
        "kge": kge,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--generation", type=int, default=300)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--models", default=None, help="comma list or None=all completed dirs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/all36_ic_gen300_aligned_20260812")
    args = parser.parse_args()

    out = BENCHMARK_ROOT / args.out
    (out / "by_basin").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ckpt_root = Path(args.ckpt_root)

    dirs = sorted(d for d in ckpt_root.iterdir() if d.is_dir())
    if args.models:
        wanted = set(args.models.split(","))
        dirs = [d for d in dirs if d.name in wanted]

    rows = []
    for d in dirs:
        if not (d / "DONE").is_file():
            print(f"[skip] {d.name}: no DONE marker", flush=True)
            continue
        if not list(d.glob(f"chunk_*_gen_{args.generation}.pt")):
            print(f"[skip] {d.name}: no chunk_*_gen_{args.generation}.pt", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            r = evaluate_model(d.name, d, args.generation, args.starts, device)
            pd.DataFrame({"basin_id": [f"{int(b):08d}" for b in r["basin_ids"]],
                          "kge_ic": r["kge"]}).to_csv(
                out / "by_basin" / f"{d.name}.csv", index=False, float_format="%.10f")
            rows.append({k: r[k] for k in
                         ("model", "generation", "starts", "n_basins", "n_invalid",
                          "median_kge", "mean_kge", "ckpt_train_kge_median", "ckpt_source")})
            print(f"[{d.name}] n={r['n_basins']} median={r['median_kge']:.4f} "
                  f"ckpt_train_median={r['ckpt_train_kge_median']:.4f} "
                  f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
        except Exception as exc:
            print(f"[{d.name}] ERROR: {exc}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = pd.DataFrame(rows).sort_values("model")
    summary.to_csv(out / "ic_gen300_aligned_summary.csv", index=False)
    print(f"\nSummary written: {out / 'ic_gen300_aligned_summary.csv'} ({len(rows)} models)")


if __name__ == "__main__":
    main()
