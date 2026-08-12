#!/usr/bin/env python3
"""Full 531-basin MOPEX4 dPL validation for the opt-in continuation path.

The baseline and continuation arms use identical midpoint initialization,
sampling, optimizer, and seed budgets.  Every validation score is evaluated
with the exact physical endpoint (lambda_i=lambda_p=1, beta=50); continuation
is used only while fitting the network.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]

from dmotpy.data_contract import add_calendar_forcing
from dmotpy.models.hydrology_model import HydrologyModel
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import load_camels_time_series
from project.benchmark.src.data_selection import load_ids
from project.benchmark.src.model_registry import model_config

import importlib.util

_pilot_spec = importlib.util.spec_from_file_location(
    "mopex4_full_h_training_pilot", BENCHMARK / "scripts/diagnostics/h_training_pilot.py"
)
_pilot = importlib.util.module_from_spec(_pilot_spec)
assert _pilot_spec.loader is not None
_pilot_spec.loader.exec_module(_pilot)

DEVICE = torch.device("cuda")
OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "full_continuation"
WARMUP, WINDOW, BATCH, STEPS = 365, 730, 100, 169
SEEDS = (41, 42, 43)


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def latest_checkpoint(arm: str, seed: int) -> Path | None:
    files = sorted((OUT / "checkpoints" / arm / f"seed_{seed}").glob("epoch_*.pt"))
    return files[-1] if files else None


def make_model(lambda_i: float = 1.0) -> HydrologyModel:
    cfg = model_config(
        "mopex4", warm_up=WARMUP, backend="compile", parameter_mapping="auto",
        warmup_grad_mode="detach",
    )
    cfg.update(continuation_lambda_i=float(lambda_i), continuation_lambda_p=1.0,
               continuation_beta=50.0)
    return HydrologyModel(cfg, device=DEVICE, backend="compile").to(DEVICE)


def set_continuation(model: HydrologyModel, lambda_i: float) -> None:
    model.continuation_lambda_i = float(lambda_i)
    model.continuation_lambda_p = 1.0
    model.continuation_beta = 50.0


def lambda_for_epoch(epoch: int, epochs: int) -> float:
    """Use five equal continuation blocks and reach the physical endpoint."""
    if epochs <= 1:
        return 1.0
    fraction = min(max((epoch - 1) / max(epochs - 1, 1), 0.0), 1.0)
    return [0.0, 0.25, 0.5, 0.75, 1.0][min(int(fraction * 5.0), 4)]


def evaluate(model, network, attrs, val_x, val_y) -> tuple[float, float]:
    set_continuation(model, 1.0)
    network.eval()
    with torch.no_grad():
        theta = network(attrs)
        q = model({"x_phy": val_x}, (None, theta.unsqueeze(-1)))["streamflow"]
        q = q.squeeze(-1).squeeze(-1)
        _, kge = _pilot.NATIVE.compute_differentiable_kge(q, val_y, warmup_days=WARMUP)
    return float(kge.median()), float(kge.mean())


def run_arm(arm: str, seed: int, epochs: int, lr: float, data: tuple, resume: bool) -> None:
    ids, attrs, train_x, train_y, val_x, val_y, catalog, lengths = data
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    network = CatchmentParameterizer(
        attrs.shape[1], 10, hidden_dims=[256, 256], dropout=.05,
    ).to(DEVICE)
    with torch.no_grad():
        network.net[-1].weight.zero_()
        network.net[-1].bias.zero_()
    model = make_model(1.0)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    all_rows: list[dict] = []
    checkpoint_dir = OUT / "checkpoints" / arm / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    if resume:
        checkpoint = latest_checkpoint(arm, seed)
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
            network.load_state_dict(payload["network"])
            optimizer.load_state_dict(payload["optimizer"])
            if "cpu_rng" in payload:
                torch.random.set_rng_state(payload["cpu_rng"])
            if "cuda_rng" in payload:
                torch.cuda.set_rng_state(payload["cuda_rng"], device=DEVICE)
            start_epoch = int(payload["epoch"]) + 1
            print(f"{arm} seed={seed} resuming from epoch {payload['epoch']}", flush=True)
    eval_epochs = set(range(10, epochs + 1, 10)) | {epochs}

    for epoch in range(start_epoch, epochs + 1):
        lambda_i = 1.0 if arm == "J0" else lambda_for_epoch(epoch, epochs)
        set_continuation(model, lambda_i)
        network.train()
        loss_total = 0.0
        started = time.perf_counter()
        for _ in range(STEPS):
            basin_indices = torch.randperm(len(ids), device=DEVICE)[:BATCH]
            choices = (torch.rand(BATCH, device=DEVICE) * lengths[basin_indices]).long()
            starts = catalog[basin_indices, choices]
            x_batch = _pilot.gather_window(train_x, starts, basin_indices)
            y_batch = _pilot.gather_window(train_y, starts, basin_indices)
            optimizer.zero_grad(set_to_none=True)
            theta = network(attrs[basin_indices])
            q = model({"x_phy": x_batch}, (None, theta.unsqueeze(-1)))["streamflow"]
            q = q.squeeze(-1).squeeze(-1)
            loss, _ = _pilot.NATIVE.compute_differentiable_kge(q, y_batch[WARMUP:], warmup_days=0)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total += float(loss.detach())

        if epoch not in eval_epochs:
            continue
        median, mean = evaluate(model, network, attrs, val_x, val_y)
        row = {
            "model": "mopex4", "arm": arm, "seed": seed, "epoch": epoch,
            "training_lambda_i": lambda_i, "validation_lambda_i": 1.0,
            "validation_beta": 50.0, "validation_median_kge": median,
            "validation_mean_kge": mean, "train_loss_1_minus_kge": loss_total / STEPS,
            "seconds": time.perf_counter() - started,
        }
        append_csv(OUT / "epochs.csv", [row])
        all_rows.append(row)
        torch.save({"epoch": epoch, "arm": arm, "seed": seed,
                    "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.random.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(DEVICE)},
                   checkpoint_dir / f"epoch_{epoch:03d}.pt")
        print(f"{arm} seed={seed} epoch={epoch} lambda_i={lambda_i:.2f} "
              f"physical_median={median:.4f}", flush=True)

    final_median, final_mean = evaluate(model, network, attrs, val_x, val_y)
    append_csv(OUT / "summary.csv", [{
        "model": "mopex4", "arm": arm, "seed": seed, "epochs": epochs,
        "final_validation_lambda_i": 1.0, "final_validation_beta": 50.0,
        "final_median_kge": final_median, "final_mean_kge": final_mean,
        "nonfinite": not (torch.isfinite(torch.tensor(final_median)) and
                           torch.isfinite(torch.tensor(final_mean))),
    }])
    del model, network, optimizer
    torch.cuda.empty_cache()


def load_data():
    ids = [int(value) for value in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    train_x_np, train_y_np, val_x_np, val_y_np = load_camels_time_series(ids)
    train_x, train_y = (torch.as_tensor(value, dtype=torch.float32, device=DEVICE)
                        for value in (train_x_np, train_y_np))
    val_x, val_y = (torch.as_tensor(value, dtype=torch.float32, device=DEVICE)
                    for value in (val_x_np, val_y_np))
    train_x, _ = add_calendar_forcing(train_x, pd.date_range("1980-10-01", "1995-09-30", freq="D"), model_name="mopex4")
    val_x, _ = add_calendar_forcing(val_x, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name="mopex4")
    catalog, lengths = _pilot.make_catalog(train_y[WARMUP:])
    return ids, attrs, train_x, train_y, val_x, val_y, catalog, lengths


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("J0", "J2", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="isolated output directory for one parallel run")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest checkpoint in output-dir")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir is not None:
        OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    (OUT / "contract.json").write_text(json.dumps({
        "basins": 531, "model": "mopex4", "window": "365 warm-up + 365 scored",
        "batch_size": BATCH, "steps_per_epoch": STEPS, "epochs": args.epochs,
        "arms": {"J0": "physical lambda_i=1 throughout",
                 "J2": "lambda_i blocks 0, .25, .5, .75, 1"},
        "selected_arm": args.arm,
        "resume": args.resume,
        "validation": "always exact lambda_i=1, lambda_p=1, beta=50",
        "seeds": args.seeds,
    }, indent=2) + "\n")
    data = load_data()
    arms = ("J0", "J2") if args.arm == "both" else (args.arm,)
    for arm in arms:
        for seed in args.seeds:
            run_arm(arm, seed, args.epochs, args.lr, data, args.resume)


if __name__ == "__main__":
    main()
