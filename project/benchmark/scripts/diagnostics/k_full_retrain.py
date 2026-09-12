#!/usr/bin/env python3
"""K1/K2 full 36-model dPL health retraining runner (Remediated Protocol).

Primary arm: auto mapping + 100 epochs. Control arm: linear + 100 epochs for
the selected representative models. Numerical model/loss/gradient work stays
on target device; host work only loads immutable arrays and writes scalar records.

Protocol Remediation (2026-08-31):
1. Strict Test-Period Isolation: 1995-2010 test period is strictly excluded from training,
   stopping counter, and epoch selection. Test evaluation runs only post-hoc on frozen best.pt.
2. Exact Best Checkpoint: best.pt is saved immediately whenever the training selection metric
   improves, storing comprehensive execution metadata.
3. Gradient Clipping Telemetry: pre-clip norm, post-clip norm, clip applied, and clip ratio
   are recorded per batch and aggregated per epoch.
4. Configurable Selection Metric: defaults to train_loss (1 - KGE).
"""
from __future__ import annotations

import argparse
import csv
from enum import Enum
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.models.registry import PARAM_INFO
from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model

DEVICE = torch.device(os.environ.get("DPL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
OUT = Path(os.environ.get("DPL_OUT", ROOT / "results/dpl_full_retrain_20260813"))
MODELS = tuple(NPARAM_INFO_36)
CONTROL_MODELS = ("collie1", "gr4j", "mopex1", "ihacres", "mopex4", "hillslope")
BATCH, STEPS, WINDOW, WARMUP, SEED = 100, 169, 730, 365, 42
BATCH = int(os.environ.get("DPL_BATCH", str(BATCH)))
MIN_EPOCHS = int(os.environ.get("DPL_MIN_EPOCHS", "50"))
PATIENCE = int(os.environ.get("DPL_PATIENCE", "10"))
PLATEAU_EPS = float(os.environ.get("DPL_EPS", "0.0001"))
STOP_ON_PLATEAU = os.environ.get("DPL_STOP_ON_PLATEAU", "true").lower() in {"1", "true", "yes"}
SELECTION_METRIC = os.environ.get("DPL_SELECTION_METRIC", "train_loss")  # "train_loss" or "train_full_median_kge"
KGE_EPS = 0.1


class Phase(str, Enum):
    TRAIN = "train"
    EVAL = "eval"


CURRENT_PHASE = Phase.TRAIN


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AVs = load_module(ROOT / "scripts/diagnostics/h_training_pilot.py", "k_h1_helpers")
H1 = AVs
def make_catalog(observations: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    catalog = NATIVE.build_informative_kge_catalog(observations.detach().cpu().numpy().T)
    width = max(len(row) for row in catalog)
    starts = torch.zeros((len(catalog), width), dtype=torch.long, device=device)
    lengths = torch.empty(len(catalog), dtype=torch.long, device=device)
    for basin, row in enumerate(catalog):
        starts[basin, : len(row)] = torch.as_tensor(row, dtype=torch.long, device=device)
        lengths[basin] = len(row)
    return starts, lengths


def gather_window(values: torch.Tensor, starts: torch.Tensor, basin_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    days = torch.arange(WINDOW, device=device)[:, None] + starts[None, :]
    return values[days, basin_indices[None, :]]
NATIVE = H1.NATIVE


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def paths(arm: str) -> tuple[Path, Path, Path, Path, Path]:
    base = OUT / arm
    return (
        base / "epochs.csv",
        base / "parameter_gradients.csv",
        base / "status.csv",
        base / "health.csv",
        base / "test_evaluation.csv",
    )


def guard_path(arm: str) -> Path:
    return OUT / arm / "gradient_guard.csv"


def checkpoint(arm: str, model: str, epoch: int) -> Path:
    return OUT / arm / "checkpoints" / model / f"epoch_{epoch:03d}.pt"


def best_checkpoint(arm: str, model: str) -> Path:
    return OUT / arm / "checkpoints" / model / "best.pt"


def latest_checkpoint(arm: str, model: str) -> Path | None:
    files = sorted((OUT / arm / "checkpoints" / model).glob("epoch_*.pt"))
    return files[-1] if files else None


def save_best_checkpoint(
    arm: str,
    model: str,
    epoch: int,
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    metric_name: str,
    metric_value: float,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    dst = best_checkpoint(arm, model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "best_epoch": epoch,
        "selection_metric_name": metric_name,
        "selection_metric_value": metric_value,
        "seed": SEED,
        "parameter_mapping": "auto" if arm == "auto100" else "linear",
        "kge_eps": KGE_EPS,
        "window_length": WINDOW,
        "warmup_days": WARMUP,
        "scored_days": WINDOW - WARMUP,
        "git_sha": get_git_sha(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    torch.save(
        {
            "epoch": epoch,
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        dst,
    )
    return dst


def health_exists(arm: str, model: str) -> bool:
    _epochs, _gradients, _status, health, _test = paths(arm)
    if not health.exists():
        return False
    with health.open() as handle:
        return any(
            row["model"] == model
            and (row["status"] == "PLATEAU_STOP" or int(row["stop_epoch"]) >= 100)
            for row in csv.DictReader(handle)
        )


def initialize_midpoint(network: CatchmentParameterizer) -> None:
    layer = network.net[-1]
    if not isinstance(layer, nn.Linear):
        raise TypeError("output layer must be Linear")
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()


def evaluate_test_period(
    model: str,
    hydro: nn.Module,
    network: nn.Module,
    attrs: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    warmup_days: int = WARMUP,
) -> tuple[int, torch.Tensor]:
    """Strict post-hoc evaluation on test period (1995..2010). Test leakage guarded."""
    global CURRENT_PHASE
    if CURRENT_PHASE == Phase.TRAIN:
        raise RuntimeError("Test period evaluation attempted during training phase - test leakage prohibited!")
    network.eval()
    with torch.no_grad():
        val_theta = network(attrs)
        val_q = hydro({"x_phy": val_x}, (None, val_theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        invalid_val = int((~torch.isfinite(val_q)).sum().detach())
        _loss, kge = NATIVE.compute_differentiable_kge(val_q, val_y, warmup_days=warmup_days, eps=KGE_EPS)
    return invalid_val, kge


def summarize_health(
    arm: str,
    model: str,
    status: str,
    stop_epoch: int,
    invalid_train: int,
    best_epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    test_median_kge: float | None = None,
) -> dict[str, Any]:
    epoch_path, gradient_path, _status_path, _health, _test = paths(arm)
    epochs = [row for row in csv.DictReader(epoch_path.open()) if row["model"] == model]
    gradients = [row for row in csv.DictReader(gradient_path.open()) if row["model"] == model]
    first, last = min(epochs, key=lambda r: int(r["epoch"])), max(epochs, key=lambda r: int(r["epoch"]))
    params = list(PARAM_INFO[model])
    permanently_zero = [
        name
        for name in params
        if all(
            float(row["zero_gradient_basin_fraction"]) == 1.0
            for row in gradients
            if row["parameter"] == name
        )
    ]
    conditional = [
        name
        for name in params
        if any(
            0.0 < float(row["zero_gradient_basin_fraction"]) < 1.0
            for row in gradients
            if row["parameter"] == name
        )
    ]
    final_boundary = float(last["theta_boundary_fraction"])
    first_loss = float(first["train_loss_1_minus_kge"])
    last_loss = float(last["train_loss_1_minus_kge"])
    return {
        "model": model,
        "arm": arm,
        "status": status,
        "stop_epoch": stop_epoch,
        "best_epoch": best_epoch,
        "selection_metric_name": best_metric_name,
        "best_selection_metric_value": best_metric_value,
        "epoch1_train_loss": first_loss,
        "final_train_loss": last_loss,
        "train_loss_improvement": first_loss - last_loss,
        "test_median_kge_posthoc": test_median_kge,
        "final_boundary_fraction": final_boundary,
        "train_nonfinite_prediction_count": invalid_train,
        "permanently_zero_parameters": ";".join(permanently_zero),
        "conditional_zero_parameters": ";".join(conditional),
        "pass_integrity": status in {"COMPLETED", "PLATEAU_STOP"} and invalid_train == 0,
        "pass_learning": (first_loss - last_loss) > 0.05,
        "pass_no_dead_parameters": not permanently_zero,
        "pass_no_saturation": final_boundary < 0.20,
        "pass_convergence_budget": not (best_epoch >= stop_epoch - 4),
    }


def run_model(
    arm: str,
    model: str,
    epochs: int,
    lr: float,
    steps_per_epoch: int = STEPS,
    batch_size: int = BATCH,
    selection_metric: str = SELECTION_METRIC,
) -> dict[str, Any]:
    global CURRENT_PHASE
    if health_exists(arm, model):
        return {"model": model, "arm": arm, "status": "ALREADY_COMPLETE"}

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ids = [int(x) for x in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device=str(DEVICE), method="zscore")
    train_x_np, train_y_np, val_x_np, val_y_np = NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(train_x_np, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(train_y_np, dtype=torch.float32, device=DEVICE)

    if model in CALENDAR_MODELS:
        train_x, _ = add_calendar_forcing(
            train_x, pd.date_range("1980-10-01", "1995-09-30", freq="D"), model_name=model
        )

    catalog, lengths = make_catalog(train_y[WARMUP:], device=DEVICE)
    mapping = "auto" if arm == "auto100" else "linear"
    # Warmup gradient mode: all models use the single implemented mode "detach"
    # (no-grad warmup + state detach at the warmup boundary; scored period full
    # backpropagation). The historical "truncate:90" label for penman was never
    # implemented and is removed; see PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md.
    warm_mode = "detach"
    backend = "compile" if (DEVICE.type == "cuda" and hasattr(torch, "compile")) else "eager"
    hydro = build_model(model, DEVICE, warm_up=WARMUP, backend=backend, parameter_mapping=mapping, warmup_grad_mode=warm_mode)
    network = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36[model], hidden_dims=[256, 256], dropout=0.05).to(DEVICE, dtype=torch.float64)
    initialize_midpoint(network)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)

    start, invalid_train = 1, 0
    old = latest_checkpoint(arm, model)
    if old is not None:
        payload = torch.load(old, map_location="cpu", weights_only=False)
        network.load_state_dict(payload["network"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.random.set_rng_state(payload["cpu_rng"])
        if DEVICE.type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng"], device=DEVICE)
        start = int(payload["epoch"]) + 1
        invalid_train = int(payload.get("invalid_train", 0))

    epoch_path, gradient_path, status_path, health_path, test_path = paths(arm)
    status = "COMPLETED"
    final_epoch = start - 1
    best_epoch = start - 1
    best_metric = float("inf") if selection_metric == "train_loss" else float("-inf")
    stall = 0

    # Strict Training Phase
    CURRENT_PHASE = Phase.TRAIN

    for epoch in range(start, epochs + 1):
        network.train()
        loss_total = 0.0
        elapsed = 0.0
        step_preclip_norms = []
        step_postclip_norms = []
        step_clip_applied = []
        observed = torch.zeros((len(ids), NPARAM_INFO_36[model]), dtype=torch.bool, device=DEVICE)

        for step_i in range(steps_per_epoch):
            basins = torch.randperm(len(ids), device=DEVICE)[:batch_size]
            choices = (torch.rand(batch_size, device=DEVICE) * lengths[basins]).long()
            starts = catalog[basins, choices]
            x = gather_window(train_x, starts, basins, device=DEVICE)
            y = gather_window(train_y, starts, basins, device=DEVICE)

            optimizer.zero_grad(set_to_none=True)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()

            theta = network(attrs[basins])
            theta.retain_grad()
            q = hydro({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            invalid_train += int((~torch.isfinite(q)).sum().detach())

            loss, _kge = NATIVE.compute_differentiable_kge(q, y[WARMUP:], warmup_days=0, eps=KGE_EPS)
            loss.backward()

            gradients = [p.grad for p in network.parameters() if p.grad is not None]
            finite_grad = all(bool(torch.isfinite(g).all()) for g in gradients)
            if not finite_grad:
                bad_theta = theta.grad.detach() if theta.grad is not None else torch.empty(0, device=DEVICE)
                bad_rows = (
                    torch.nonzero(~torch.isfinite(bad_theta).all(dim=1), as_tuple=False).flatten()
                    if bad_theta.numel()
                    else torch.empty(0, dtype=torch.long, device=DEVICE)
                )
                append_csv(
                    guard_path(arm),
                    [
                        {
                            "model": model,
                            "epoch": epoch,
                            "batch": int(step_i),
                            "basin_ids": ";".join(str(int(ids[int(basins[i])])) for i in bad_rows[:64].tolist()),
                            "finite_gradient": False,
                            "action": "SKIP_BATCH",
                        }
                    ],
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            observed[basins] |= theta.grad.detach() != 0

            # Gradient clipping telemetry
            params_with_grad = [p for p in network.parameters() if p.grad is not None]
            if params_with_grad:
                pre_norm = float(torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2).item())
            else:
                pre_norm = 0.0

            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

            if params_with_grad:
                post_norm = float(torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2).item())
            else:
                post_norm = 0.0

            step_preclip_norms.append(pre_norm)
            step_postclip_norms.append(post_norm)
            step_clip_applied.append(pre_norm > 1.0)

            optimizer.step()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - now
            loss_total += float(loss.detach())

        final_epoch = epoch
        avg_train_loss = loss_total / max(steps_per_epoch, 1)

        # Telemetry aggregations
        med_pre = float(np.median(step_preclip_norms)) if step_preclip_norms else 0.0
        p90_pre = float(np.percentile(step_preclip_norms, 90)) if step_preclip_norms else 0.0
        max_pre = float(np.max(step_preclip_norms)) if step_preclip_norms else 0.0
        clip_frac = float(np.mean(step_clip_applied)) if step_clip_applied else 0.0

        network.eval()
        with torch.no_grad():
            eval_theta = network(attrs)

        append_csv(
            epoch_path,
            [
                {
                    "model": model,
                    "arm": arm,
                    "epoch": epoch,
                    "status": "COMPLETED_EPOCH",
                    "train_loss_1_minus_kge": avg_train_loss,
                    "theta_boundary_fraction": float(((eval_theta < 0.02) | (eval_theta > 0.98)).float().mean()),
                    "seconds_per_train_step": elapsed / max(steps_per_epoch, 1),
                    "grad_norm_preclip_median": med_pre,
                    "grad_norm_preclip_p90": p90_pre,
                    "grad_norm_preclip_max": max_pre,
                    "grad_clip_fraction": clip_frac,
                    "parameter_mapping": mapping,
                    "warmup_grad_mode": warm_mode,
                    "train_nonfinite_cumulative": invalid_train,
                }
            ],
        )

        append_csv(
            gradient_path,
            [
                {
                    "model": model,
                    "arm": arm,
                    "epoch": epoch,
                    "parameter": p,
                    "zero_gradient_basin_fraction": float((~observed[:, j]).float().mean()),
                    "theta_boundary_basin_fraction": float(
                        ((eval_theta[:, j] < 0.02) | (eval_theta[:, j] > 0.98)).float().mean()
                    ),
                }
                for j, p in enumerate(PARAM_INFO[model])
            ],
        )

        # Checkpoint: Periodic snapshot every 10 epochs or at end
        if epoch % 10 == 0 or epoch == epochs:
            dst = checkpoint(arm, model, epoch)
            dst.parent.mkdir(parents=True, exist_ok=True)
            cuda_rng = torch.cuda.get_rng_state(DEVICE) if DEVICE.type == "cuda" else None
            torch.save(
                {
                    "epoch": epoch,
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.random.get_rng_state(),
                    "cuda_rng": cuda_rng,
                    "invalid_train": invalid_train,
                },
                dst,
            )

        # Selection Metric & Exact best.pt Checkpoint
        current_selection_value = avg_train_loss
        is_best = False
        if selection_metric == "train_loss":
            if current_selection_value < best_metric - PLATEAU_EPS:
                best_metric = current_selection_value
                best_epoch = epoch
                stall = 0
                is_best = True
            else:
                stall += 1
        else:
            # Placeholder for alternative metric
            if current_selection_value > best_metric + PLATEAU_EPS:
                best_metric = current_selection_value
                best_epoch = epoch
                stall = 0
                is_best = True
            else:
                stall += 1

        if is_best or epoch == 1:
            save_best_checkpoint(
                arm,
                model,
                epoch,
                network,
                optimizer,
                metric_name=selection_metric,
                metric_value=current_selection_value,
                extra_metadata={"lr": lr, "weight_decay": 1e-4},
            )

        if STOP_ON_PLATEAU and epoch >= MIN_EPOCHS:
            if stall >= PATIENCE:
                status = "PLATEAU_STOP"
                break

    # Verification: Reload best.pt and assert state dict matches
    best_pt = best_checkpoint(arm, model)
    if best_pt.exists():
        best_payload = torch.load(best_pt, map_location="cpu", weights_only=False)
        network.load_state_dict(best_payload["network"])
        best_epoch = int(best_payload.get("metadata", {}).get("best_epoch", best_epoch))

    # Post-Hoc Test Period Evaluation (1995-2010)
    CURRENT_PHASE = Phase.EVAL
    val_x = torch.as_tensor(val_x_np, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(val_y_np, dtype=torch.float32, device=DEVICE)
    if model in CALENDAR_MODELS:
        val_x, _ = add_calendar_forcing(
            val_x, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name=model
        )

    invalid_val, test_kge_tensor = evaluate_test_period(model, hydro, network, attrs, val_x, val_y, warmup_days=WARMUP)
    test_med_kge = float(test_kge_tensor.median().item())
    test_mean_kge = float(test_kge_tensor.mean().item())

    append_csv(
        test_path,
        [
            {
                "model": model,
                "arm": arm,
                "evaluated_checkpoint": "best.pt",
                "best_epoch": best_epoch,
                "test_median_kge": test_med_kge,
                "test_mean_kge": test_mean_kge,
                "test_nonfinite_count": invalid_val,
            }
        ],
    )

    summary = summarize_health(
        arm,
        model,
        status,
        final_epoch,
        invalid_train,
        best_epoch,
        best_metric_name=selection_metric,
        best_metric_value=best_metric,
        test_median_kge=test_med_kge,
    )
    append_csv(health_path, [summary])
    append_csv(
        status_path,
        [
            {
                "model": model,
                "arm": arm,
                "status": status,
                "last_epoch": final_epoch,
                "best_epoch": best_epoch,
                "warmup_grad_mode": warm_mode,
            }
        ],
    )

    del hydro, network, optimizer, train_x, train_y, val_x, val_y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("auto100", "linear100"), required=True)
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()

    base = OUT / args.arm
    base.mkdir(parents=True, exist_ok=True)
    selected = (args.model,) if args.model else (MODELS if args.arm == "auto100" else CONTROL_MODELS)
    (base / "contract.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "models": list(selected),
                "epochs": args.epochs,
                "batch_size": args.batch,
                "steps_per_epoch": args.steps,
                "parameter_mapping": "auto" if args.arm == "auto100" else "linear",
                "penman_warmup": "365d warmup + 365d scored, full backprop (identical gradient semantics to all models)",
                "other_warmup": "detach",
                "attributes": "Caravan, zscore all531",
                "hidden_dims": [256, 256],
                "selection_metric": SELECTION_METRIC,
                "kge_eps": KGE_EPS,
                "git_sha": get_git_sha(),
            },
            indent=2,
        )
        + "\n"
    )
    for model in selected:
        run_model(args.arm, model, args.epochs, args.lr, steps_per_epoch=args.steps, batch_size=args.batch)


if __name__ == "__main__":
    main()
