#!/usr/bin/env python3
"""Canonical dPL v2 Single-Model Training Runner (36-Model Benchmark Baseline).

Frozen Protocol Specification (Canonical v2 - 2026-08-31):
1. Training Period: 1980-10-01..1995-09-30 (15 hydrological years, 531 CAMELS basins).
2. Test Period: 1995-10-01..2010-09-30 (with 1-year warmup from 1994-10-01). STRICT TEST ISOLATION during training.
3. Sequence Horizon:
   - 35 Standard Models: 730d warmup + 365d scored (total window = 1095d).
   - Penman (Pre-registered warmup-length exception, kept from Phase W2):
     365d warmup + 365d scored (total window = 730d).
   All 36 models use identical gradient semantics: warmup runs under no_grad with
   state detach at the warmup boundary, and the scored period runs FULL
   backpropagation. No truncated-BPTT mode exists or is used.
5. Architecture: CatchmentParameterizer MLP [256, 256], LayerNorm, GELU, Dropout 0.05, 35 Caravan attributes (z-score normalized).
6. Optimizer: AdamW (lr=1e-3, weight_decay=1e-4, scheduler=None).
7. Gradient Clipping: max_norm = 1.0 (telemetry recorded per batch).
8. KGE Formulation: Sample-variance KGE with eps = 0.1.
9. Checkpointing: Exact best.pt saved on train_loss improvement + periodic snapshot every 10 epochs.
10. Stopping: 100 max epochs, plateau early stop on train_loss (min_epochs=50, patience=10, plateau_eps=1e-4).
"""
from __future__ import annotations

import argparse
import csv
from enum import Enum
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src"), str(REPO_ROOT / "dmotpy")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.models.registry import PARAM_INFO
from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model

import project.benchmark.scripts.diagnostics.h_training_pilot as H1
NATIVE = H1.NATIVE

KGE_EPS = 0.1
EVAL_WARMUP = 365


class Phase(str, Enum):
    TRAIN = "train"
    EVAL = "eval"


CURRENT_PHASE = Phase.TRAIN


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def get_file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def initialize_midpoint(network: CatchmentParameterizer) -> None:
    layer = network.net[-1]
    if not isinstance(layer, nn.Linear):
        raise TypeError("output layer must be Linear")
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()


def make_catalog(observations: torch.Tensor, horizon_days: int, warmup_days: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    scored_days = horizon_days - warmup_days
    catalog = NATIVE.build_informative_kge_catalog(
        observations.detach().cpu().numpy().T,
        prediction_days=scored_days,
        min_valid_points=30,
        min_observation_std=0.01,
    )
    width = max(len(row) for row in catalog)
    starts = torch.zeros((len(catalog), width), dtype=torch.long, device=device)
    lengths = torch.empty(len(catalog), dtype=torch.long, device=device)
    for basin, row in enumerate(catalog):
        starts[basin, : len(row)] = torch.as_tensor(row, dtype=torch.long, device=device)
        lengths[basin] = len(row)
    return starts, lengths


def gather_window(values: torch.Tensor, starts: torch.Tensor, basin_indices: torch.Tensor, horizon_days: int, device: torch.device) -> torch.Tensor:
    days = torch.arange(horizon_days, device=device)[:, None] + starts[None, :]
    return values[days, basin_indices[None, :]]


def evaluate_test_partition(
    model_name: str,
    hydro: nn.Module,
    network: nn.Module,
    attrs: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    warmup_days: int = EVAL_WARMUP,
) -> tuple[int, dict[str, float], np.ndarray]:
    """Strict post-hoc evaluation on 1995-2010 test period. Test leakage strictly guarded."""
    global CURRENT_PHASE
    if CURRENT_PHASE == Phase.TRAIN:
        raise RuntimeError("Test period evaluation attempted during training phase - test leakage prohibited!")
    network.eval()
    with torch.no_grad():
        val_theta = network(attrs)
        val_q = hydro({"x_phy": val_x}, (None, val_theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        invalid_count = int((~torch.isfinite(val_q)).sum().detach().item())
        _loss, kge_tensor = NATIVE.compute_differentiable_kge(val_q, val_y, warmup_days=warmup_days, eps=KGE_EPS)

    kges = kge_tensor.cpu().numpy()
    valid_kges = kges[np.isfinite(kges)]
    if len(valid_kges) == 0:
        return invalid_count, {"median": -1.0, "mean": -1.0, "q25": -1.0, "q10": -1.0}, kges

    return invalid_count, {
        "median": float(np.median(valid_kges)),
        "mean": float(np.mean(valid_kges)),
        "q25": float(np.percentile(valid_kges, 25)),
        "q10": float(np.percentile(valid_kges, 10)),
    }, kges


def execute_canonical_v2_model(job_config: dict[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    global CURRENT_PHASE
    model_name = job_config["model"]
    horizon = int(job_config.get("horizon_days", 730 if model_name == "penman" else 1095))
    warmup = int(job_config.get("warmup_days", 365 if model_name == "penman" else 730))
    # Penman's 365d training warmup is a W2-evidence-based warmup-LENGTH exception;
    # it changes no gradient semantics (all models: full backprop on the scored period).
    scored = int(job_config.get("scored_days", 365))
    declared_warmup_mode = str(job_config.get("warmup_grad_mode", "detach"))
    if declared_warmup_mode != "detach":
        raise RuntimeError(
            f"Unsupported warmup_grad_mode {declared_warmup_mode!r} declared in job config "
            f"for {model_name}. Only 'detach' is implemented; 'truncate:N'/'full' modes were ",
            "historical dead config (see PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md) ",
            "and are rejected to prevent silent no-ops."
        )
    mapping = job_config.get("mapping", "auto")
    lr = float(job_config.get("lr", 1e-3))
    weight_decay = float(job_config.get("weight_decay", 1e-4))
    clip_norm = float(job_config.get("clip_norm", 1.0))
    seed = int(job_config.get("seed", 42))
    epochs = int(job_config.get("epochs", 100))
    min_epochs = int(job_config.get("min_epochs", 50))
    patience = int(job_config.get("patience", 10))
    plateau_eps = float(job_config.get("plateau_eps", 1e-4))
    selection_metric = job_config.get("selection_metric", "train_loss")

    run_dir = output_dir / "runs" / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_file = run_dir / ".lock"
    done_file = run_dir / "DONE"
    failed_file = run_dir / "FAILED"

    # Save job configuration
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(job_config, f, indent=2)

    (run_dir / "source_commit.txt").write_text(get_git_sha() + "\n")
    (run_dir / "environment.txt").write_text(
        f"DEVICE: {device}\n"
        f"TORCH_VERSION: {torch.__version__}\n"
        f"PID: {os.getpid()}\n"
        f"START_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    lock_file.write_text(f"PID={os.getpid()}\nSTARTED={time.time()}\n")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    start_time = time.time()

    try:
        ids = [int(x) for x in load_ids("data/531sub_id.txt")]
        attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device=str(device), method="zscore")
        train_x_np, train_y_np, val_x_np, val_y_np = NATIVE.load_camels_time_series(ids)

        train_x = torch.as_tensor(train_x_np, dtype=torch.float32, device=device)
        train_y = torch.as_tensor(train_y_np, dtype=torch.float32, device=device)

        if model_name in CALENDAR_MODELS:
            train_x, _ = add_calendar_forcing(
                train_x, pd.date_range("1980-10-01", "1995-09-30", freq="D"), model_name=model_name
            )

        # Build catalog on training period
        catalog, lengths = make_catalog(train_y[warmup:], horizon_days=horizon, warmup_days=warmup, device=device)

        # Single implemented warmup-gradient mode for ALL models (incl. penman):
        # no-grad warmup + state detach at the warmup boundary; scored period full
        # backpropagation. The historical "truncate:90" penman label was never
        # implemented and is removed; see PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md.
        warm_mode = "detach"
        backend = "compile" if (device.type == "cuda" and hasattr(torch, "compile")) else "eager"
        
        hydro_train = build_model(model_name, device, warm_up=warmup, backend=backend, parameter_mapping=mapping, warmup_grad_mode=warm_mode)
        network = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36[model_name], hidden_dims=[256, 256], dropout=0.05).to(device, dtype=torch.float64)
        initialize_midpoint(network)
        optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)

        batch_size = 100
        # 169 steps per epoch for standard 531 basins sampling
        steps_per_epoch = 169

        epoch_csv_path = run_dir / "epoch_metrics.csv"
        grad_csv_path = run_dir / "gradient_telemetry.csv"

        best_metric = float("inf") if selection_metric == "train_loss" else float("-inf")
        best_epoch = 1
        stall = 0
        status = "COMPLETED"
        invalid_train = 0

        CURRENT_PHASE = Phase.TRAIN

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            network.train()
            loss_total = 0.0
            preclip_norms = []
            postclip_norms = []
            clips_applied = []
            observed_grads = torch.zeros((len(ids), NPARAM_INFO_36[model_name]), dtype=torch.bool, device=device)

            for step_i in range(steps_per_epoch):
                basins = torch.randperm(len(ids), device=device)[:batch_size]
                choices = (torch.rand(batch_size, device=device) * lengths[basins]).long()
                starts = catalog[basins, choices]
                x_b = gather_window(train_x, starts, basins, horizon_days=horizon, device=device)
                y_b = gather_window(train_y, starts, basins, horizon_days=horizon, device=device)

                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                theta = network(attrs[basins])
                theta.retain_grad()
                q = hydro_train({"x_phy": x_b}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
                invalid_train += int((~torch.isfinite(q)).sum().detach().item())

                loss, _kge = NATIVE.compute_differentiable_kge(q, y_b[warmup:], warmup_days=0, eps=KGE_EPS)
                loss.backward()

                grads = [p.grad for p in network.parameters() if p.grad is not None]
                if not all(bool(torch.isfinite(g).all()) for g in grads):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                observed_grads[basins] |= theta.grad.detach() != 0

                params_with_grad = [p for p in network.parameters() if p.grad is not None]
                pre_norm = float(torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2).item()) if params_with_grad else 0.0
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_norm)
                post_norm = float(torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2).item()) if params_with_grad else 0.0

                preclip_norms.append(pre_norm)
                postclip_norms.append(post_norm)
                clips_applied.append(pre_norm > clip_norm)

                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                loss_total += float(loss.detach().item())

            avg_train_loss = loss_total / steps_per_epoch
            ep_time = time.time() - ep_start

            # Telemetry evaluation
            network.eval()
            with torch.no_grad():
                u_eval = network(attrs)
                jac_eval = hydro_train.normalized_parameter_mapping_jacobian(u_eval)

            u_np = u_eval.cpu().numpy()
            jac_np = jac_eval.cpu().numpy()
            sat_low = float(np.mean(u_np < 0.02))
            sat_high = float(np.mean(u_np > 0.98))
            jac_p05 = float(np.percentile(jac_np, 5))
            jac_med = float(np.median(jac_np))

            append_csv(
                epoch_csv_path,
                [
                    {
                        "model": model_name,
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "grad_norm_preclip_median": float(np.median(preclip_norms)) if preclip_norms else 0.0,
                        "grad_norm_preclip_p90": float(np.percentile(preclip_norms, 90)) if preclip_norms else 0.0,
                        "grad_norm_preclip_max": float(np.max(preclip_norms)) if preclip_norms else 0.0,
                        "grad_clip_fraction": float(np.mean(clips_applied)) if clips_applied else 0.0,
                        "saturation_lower": sat_low,
                        "saturation_upper": sat_high,
                        "jacobian_p05": jac_p05,
                        "jacobian_median": jac_med,
                        "epoch_time_s": ep_time,
                    }
                ],
            )

            append_csv(
                grad_csv_path,
                [
                    {
                        "model": model_name,
                        "epoch": epoch,
                        "parameter": p,
                        "zero_gradient_basin_fraction": float((~observed_grads[:, j]).float().mean().item()),
                        "theta_boundary_basin_fraction": float(
                            ((u_eval[:, j] < 0.02) | (u_eval[:, j] > 0.98)).float().mean().item()
                        ),
                    }
                    for j, p in enumerate(PARAM_INFO[model_name])
                ],
            )

            # Checkpoint: Periodic snapshot every 10 epochs or at end
            if epoch % 10 == 0 or epoch == epochs:
                dst = run_dir / f"epoch_{epoch:03d}.pt"
                cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
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

            # Selection on train_loss -> Exact best.pt
            is_best = False
            if selection_metric == "train_loss":
                if avg_train_loss < best_metric - plateau_eps:
                    best_metric = avg_train_loss
                    best_epoch = epoch
                    stall = 0
                    is_best = True
                else:
                    stall += 1

            if is_best or epoch == 1:
                best_dst = run_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "network": network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "job_config": job_config,
                        "selection_metric": selection_metric,
                        "best_selection_value": best_metric,
                        "git_sha": get_git_sha(),
                    },
                    best_dst,
                )
                (run_dir / "best_metadata.json").write_text(
                    json.dumps(
                        {
                            "model": model_name,
                            "best_epoch": epoch,
                            "selection_metric": selection_metric,
                            "best_selection_value": best_metric,
                            "git_sha": get_git_sha(),
                            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        indent=2,
                    )
                )

            # Early Stopping Check
            if epoch >= min_epochs and stall >= patience:
                status = "PLATEAU_STOP"
                break

        # Reload best.pt for Post-Hoc Test Evaluation (1995-2010)
        best_pt_path = run_dir / "best.pt"
        if best_pt_path.exists():
            payload = torch.load(best_pt_path, map_location="cpu", weights_only=False)
            network.load_state_dict(payload["network"])
            best_epoch = int(payload.get("epoch", best_epoch))

        CURRENT_PHASE = Phase.EVAL
        val_x = torch.as_tensor(val_x_np, dtype=torch.float32, device=device)
        val_y = torch.as_tensor(val_y_np, dtype=torch.float32, device=device)
        if model_name in CALENDAR_MODELS:
            val_x, _ = add_calendar_forcing(
                val_x, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name=model_name
            )

        hydro_eval = build_model(model_name, device, warm_up=EVAL_WARMUP, backend=backend, parameter_mapping=mapping, warmup_grad_mode=warm_mode)
        invalid_val, test_metrics, kge_basin = evaluate_test_partition(
            model_name, hydro_eval, network, attrs, val_x, val_y, warmup_days=EVAL_WARMUP
        )

        total_runtime = time.time() - start_time

        # Save test evaluation outputs
        (run_dir / "test_evaluation.json").write_text(json.dumps(test_metrics, indent=2))
        pd.DataFrame({"basin_id": ids, "kge": kge_basin}).to_csv(run_dir / "basin_test_kge.csv", index=False)
        (run_dir / "runtime.json").write_text(
            json.dumps(
                {
                    "model": model_name,
                    "total_runtime_s": total_runtime,
                    "final_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_train_loss": best_metric,
                    "test_median_kge": test_metrics["median"],
                    "test_q25_kge": test_metrics["q25"],
                    "test_mean_kge": test_metrics["mean"],
                    "status": status,
                },
                indent=2,
            )
        )
        (run_dir / "exit_status.json").write_text(json.dumps({"exit_code": 0, "status": status}, indent=2))

        if best_pt_path.exists():
            (run_dir / "best_state_sha256.txt").write_text(get_file_sha256(best_pt_path) + "\n")

        done_file.write_text(f"COMPLETED={time.strftime('%Y-%m-%d %H:%M:%S')}\nSTATUS={status}\n")
        if lock_file.exists():
            lock_file.unlink()

        return {
            "model": model_name,
            "status": "DONE",
            "exit_code": 0,
            "best_epoch": best_epoch,
            "best_train_loss": best_metric,
            "test_median_kge": test_metrics["median"],
            "test_q25_kge": test_metrics["q25"],
            "test_mean_kge": test_metrics["mean"],
            "runtime_s": total_runtime,
        }

    except Exception as exc:
        total_runtime = time.time() - start_time
        err_msg = traceback.format_exc()
        failed_file.write_text(f"FAILED_TIME={time.strftime('%Y-%m-%d %H:%M:%S')}\nERROR={str(exc)}\n")
        (run_dir / "failure_reason.md").write_text(f"# Failure Report for `{model_name}`\n\n```text\n{err_msg}\n```\n")
        (run_dir / "exit_status.json").write_text(json.dumps({"exit_code": 1, "error": str(exc)}, indent=2))
        if lock_file.exists():
            lock_file.unlink()
        return {
            "model": model_name,
            "status": "FAILED",
            "exit_code": 1,
            "error": str(exc),
            "runtime_s": total_runtime,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-config", type=str, required=True, help="JSON or YAML string/path of job config")
    parser.add_argument("--out", type=str, required=True, help="Output canonical v2 directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, cuda:0, etc.)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    cfg_raw = args.job_config
    if len(cfg_raw) < 256 and Path(cfg_raw).exists():
        with open(cfg_raw) as f:
            job_cfg = yaml.safe_load(f) if cfg_raw.endswith((".yaml", ".yml")) else json.load(f)
    else:
        try:
            job_cfg = json.loads(cfg_raw)
        except json.JSONDecodeError:
            job_cfg = yaml.safe_load(cfg_raw)

    dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)

    res = execute_canonical_v2_model(job_cfg, out_dir, dev)
    sys.exit(res["exit_code"])


if __name__ == "__main__":
    main()
