#!/usr/bin/env python3
"""
Reselect dPL checkpoints by TRAINING LOSS (not validation KGE) and re-evaluate
per-basin KGE on the common 1995-10-01..2010-09-30 window.

Rationale
---------
The original all36 diagnosis selected the dPL epoch by maximum validation
median KGE (round13_finalize.py).  Because that validation window is exactly
the 1995-2010 period used for the final IC-vs-dPL comparison, the reported
dPL KGE was in-sample with respect to checkpoint selection, unlike IC whose
selection uses train-period KGE only (selection_period: train_only).

This script changes the selection signal to the per-epoch training loss
(train_loss_1_minus_kge from auto100/epochs.csv, mean 1-KGE over sampled
windows), which never observes the validation period:

    selected_epoch = argmin(train_loss_1_minus_kge) over SAVED checkpoints

Pre-failure caps (epoch at which training loss collapses to 1.0):
    simhyd <= 63 (loss collapses at epoch 64), vic <= 56 (collapses at 57).

Evaluation protocol is byte-identical to the all36 diagnosis:
    * build_model(warm_up=365, backend=compile, parameter_mapping=auto)
    * forcing 1994-10-01..2010-09-30, targets 1995-10-01..2010-09-30
    * per-basin KGE = streaming_kge(eps=0.1) over the 5479-day window
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model
from src.objective import streaming_kge

import importlib.util


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NATIVE = load_module(BENCHMARK_ROOT / "scripts/run_dpl_benchmark_dmg_native.py", "reselect_native")

CKPT_ROOT = BENCHMARK_ROOT / "results/dpl_round13_20260805/auto100/checkpoints"
EPOCHS_CSV = BENCHMARK_ROOT / "results/dpl_round13_20260805/auto100/epochs.csv"
# Epoch at which the training loss collapses; cap = collapse_epoch - 1.
PRE_FAILURE_CAP = {"simhyd": 63, "vic": 56}
COLLAPSED_LOSS = 0.99  # recorded train_loss >= this is a collapsed/broken epoch


def select_epochs() -> pd.DataFrame:
    epochs = pd.read_csv(EPOCHS_CSV)
    rows = []
    for model in sorted(epochs.model.unique()):
        saved = sorted(
            int(fn[6:-3])
            for fn in os.listdir(CKPT_ROOT / model)
            if fn.startswith("epoch_") and fn.endswith(".pt")
        )
        cap = PRE_FAILURE_CAP.get(model, 10**9)
        traj = epochs[epochs.model == model].set_index("epoch")
        candidates = [e for e in saved if e <= cap and traj.loc[e, "train_loss_1_minus_kge"] < COLLAPSED_LOSS]
        if not candidates:
            raise RuntimeError(f"{model}: no valid saved checkpoints after pre-failure cap {cap}")
        best = min(candidates, key=lambda e: traj.loc[e, "train_loss_1_minus_kge"])
        rows.append(
            {
                "model": model,
                "selected_epoch": best,
                "selected_train_loss": float(traj.loc[best, "train_loss_1_minus_kge"]),
                "selected_validation_median_kge": float(traj.loc[best, "validation_median_kge"]),
                "n_saved_checkpoints": len(saved),
                "n_candidates": len(candidates),
                "pre_failure_cap": None if cap == 10**9 else cap,
                "all_saved_epochs": ";".join(map(str, saved)),
            }
        )
    return pd.DataFrame(rows).sort_values("model")


def evaluate_model(model: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    ids = [int(x) for x in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device=device, method="zscore")
    _tx, _ty, val_x_np, val_y_np = NATIVE.load_camels_time_series(ids)
    val_x = torch.as_tensor(val_x_np, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(val_y_np, dtype=torch.float32, device=device)
    if model in CALENDAR_MODELS:
        val_x, _ = add_calendar_forcing(
            val_x, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name=model
        )
    hydro = build_model(
        model, device, warm_up=365, backend="compile",
        parameter_mapping="auto", warmup_grad_mode="detach",
    )
    network = CatchmentParameterizer(
        attrs.shape[1], NPARAM_INFO_36[model], hidden_dims=[256, 256], dropout=0.05
    ).to(device)
    return ids, attrs, val_x, val_y, hydro, network


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="single model or None for all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/all36_dpl_gap_diagnosis_20260812_trainloss")
    args = parser.parse_args()

    out = BENCHMARK_ROOT / args.out
    (out / "by_basin").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sel = select_epochs()
    sel.to_csv(out / "dpl_epoch_selection_trainloss.csv", index=False)
    print(f"Selection table written: {out / 'dpl_epoch_selection_trainloss.csv'}")
    print(sel[["model", "selected_epoch", "selected_train_loss", "selected_validation_median_kge"]].to_string(index=False))

    models = [args.model] if args.model else list(sel.model)
    sel_idx = sel.set_index("model")
    for model in models:
        epoch = int(sel_idx.loc[model, "selected_epoch"])
        t0 = time.perf_counter()
        ids, attrs, val_x, val_y, hydro, network = evaluate_model(model, device)
        ckpt = torch.load(
            CKPT_ROOT / model / f"epoch_{epoch:03d}.pt", map_location="cpu", weights_only=False
        )
        network.load_state_dict(ckpt["network"])
        network.eval()
        with torch.inference_mode():
            theta = network(attrs)
            q = hydro({"x_phy": val_x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            score, invalid = streaming_kge(q.unsqueeze(-1).unsqueeze(-1), val_y, eps=0.1)
        kge = score[:, 0, 0].detach().cpu().numpy()
        if bool(invalid.any()):
            print(f"[{model}] WARNING: {int(invalid.sum())} invalid basins")
        frame = pd.DataFrame({"basin_id": [f"{b:08d}" for b in ids], "kge_dpl": kge})
        frame.to_csv(out / "by_basin" / f"{model}.csv", index=False, float_format="%.10f")
        print(
            f"[{model}] epoch={epoch:3d} median={np.nanmedian(kge):.6f} mean={np.nanmean(kge):.6f} "
            f"elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        del hydro, network, val_x, val_y, q, attrs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
