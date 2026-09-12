#!/usr/bin/env python3
"""Ablation Job Runner for dPL Training Protocol Study (v2).

Executes a single ablation experiment defined by job configuration.
Strict Protocol Rules:
1. TEST Isolation: ZERO access to 1995-10-01..2010-09-30 test data.
2. FIT partition: 1980-10-01..1990-09-30 (10 yr) used strictly for training.
3. INNER-VAL partition: 1990-10-01..1995-09-30 (5 yr, 365d warmup) used for protocol evaluation.
4. Exact best.pt checkpointing based on training selection metric (train_loss).
5. Comprehensive telemetry: pre/post clip norms, saturation, mapping Jacobians.
6. Support for exact optimizer update budget stopping (Phase U).
"""
from __future__ import annotations

import argparse
import csv
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
FIT_START, FIT_END = "1980-10-01", "1990-09-30"
INNER_VAL_WARM_START = "1989-10-01"
INNER_VAL_START, INNER_VAL_END = "1990-10-01", "1995-09-30"


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


def evaluate_inner_val(
    model_name: str,
    hydro: nn.Module,
    network: nn.Module,
    attrs: torch.Tensor,
    val_x: torch.Tensor,
    scored_y: torch.Tensor,
) -> tuple[int, dict[str, float]]:
    network.eval()
    with torch.no_grad():
        theta = network(attrs)
        q = hydro({"x_phy": val_x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        invalid_count = int((~torch.isfinite(q)).sum().detach().item())
        _loss, kge_tensor = NATIVE.compute_differentiable_kge(q, scored_y, warmup_days=0, eps=KGE_EPS)

    kges = kge_tensor.cpu().numpy()
    valid_kges = kges[np.isfinite(kges)]
    if len(valid_kges) == 0:
        return invalid_count, {"median": -1.0, "mean": -1.0, "q25": -1.0, "q10": -1.0}

    return invalid_count, {
        "median": float(np.median(valid_kges)),
        "mean": float(np.mean(valid_kges)),
        "q25": float(np.percentile(valid_kges, 25)),
        "q10": float(np.percentile(valid_kges, 10)),
    }


def execute_ablation_job(job_config: dict[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    job_id = job_config["job_id"]
    model_name = job_config["model"]
    horizon = int(job_config["horizon_days"])
    warmup = int(job_config["warmup_days"])
    scored = int(job_config["scored_days"])
    mapping = job_config.get("mapping", "auto")
    lr = float(job_config.get("lr", 1e-3))
    scheduler_name = job_config.get("scheduler", "none")
    weight_decay = float(job_config.get("weight_decay", 1e-4))
    clip_norm = float(job_config.get("clip_norm", 1.0))
    seed = int(job_config.get("seed", 42))
    selection_metric = job_config.get("selection_metric", "train_loss")
    inner_val_freq = int(job_config.get("inner_val_freq", 5))

    # Support for Update-Budget Matched Execution (Phase U)
    max_updates = job_config.get("max_optimizer_updates")
    if max_updates is not None:
        max_updates = int(max_updates)

    epochs = int(job_config.get("epochs", 100))
    min_epochs = int(job_config.get("min_epochs", 50))
    patience = int(job_config.get("patience", 10))
    plateau_eps = float(job_config.get("plateau_eps", 1e-4))

    run_dir = output_dir / "runs" / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_file = run_dir / ".lock"
    done_file = run_dir / "DONE"
    failed_file = run_dir / "FAILED"

    # Save job configuration
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(job_config, f, indent=2)

    # Record environment and provenance
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
        train_x_raw, train_y_raw, val_x_raw, val_y_raw = NATIVE.load_camels_time_series(ids)

        # STRICT TEST PERIOD ZERO-ACCESS GUARD
        # Slicing exclusively from 1980-10-01..1995-09-30 (5478 days)
        # FIT: 0..3652 (1980-10-01..1990-09-30)
        # INNER-VAL: 3287..5478 (1989-10-01..1995-09-30, 365d warmup + 1826d scoring)
        fit_len = 3652
        val_start_idx = 3652 - warmup

        fit_x = torch.as_tensor(train_x_raw[:fit_len], dtype=torch.float32, device=device)
        fit_y = torch.as_tensor(train_y_raw[:fit_len], dtype=torch.float32, device=device)

        inner_x = torch.as_tensor(train_x_raw[val_start_idx:], dtype=torch.float32, device=device)
        scored_y = torch.as_tensor(train_y_raw[3652:], dtype=torch.float32, device=device)

        if model_name in CALENDAR_MODELS:
            fit_x, _ = add_calendar_forcing(
                fit_x, pd.date_range("1980-10-01", periods=len(fit_x), freq="D"), model_name=model_name
            )
            cal_dates = pd.date_range("1980-10-01", periods=len(train_x_raw), freq="D")[val_start_idx:]
            inner_x, _ = add_calendar_forcing(
                inner_x, cal_dates, model_name=model_name
            )

        # Build catalog on FIT period
        catalog, lengths = make_catalog(fit_y[warmup:], horizon_days=horizon, warmup_days=warmup, device=device)

        # Single implemented warmup-gradient mode for ALL models: no-grad warmup +
        # state detach at the warmup boundary; scored period full backpropagation.
        # Historical "truncate:90" penman label was never implemented (no-op) and is
        # removed; see PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md.
        warm_mode = "detach"
        backend = "compile" if (device.type == "cuda" and hasattr(torch, "compile")) else "eager"
        hydro = build_model(model_name, device, warm_up=warmup, backend=backend, parameter_mapping=mapping, warmup_grad_mode=warm_mode)
        network = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36[model_name], hidden_dims=[256, 256], dropout=0.05).to(device, dtype=torch.float64)
        initialize_midpoint(network)
        optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)

        batch_size = 100
        steps_per_epoch = max(1, fit_len // horizon)  # e.g. 5 for 730d, 2 for 1825d
        steps_per_epoch = min(steps_per_epoch, 50)

        # If max_updates is specified (Phase U), calculate max epochs to cover it
        if max_updates is not None:
            epochs = int(np.ceil(max_updates / steps_per_epoch)) + 1
            milestone_updates = set(int(np.ceil(p * max_updates)) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        else:
            milestone_updates = set()

        scheduler = None
        if scheduler_name.lower() == "cosineannealinglr":
            t_max = job_config.get("scheduler_params", {}).get("T_max", epochs)
            eta_min = job_config.get("scheduler_params", {}).get("eta_min", 1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

        epoch_csv_path = run_dir / "epoch_metrics.csv"
        grad_csv_path = run_dir / "gradient_telemetry.csv"
        update_csv_path = run_dir / "inner_validation_by_update.csv"

        best_metric = float("inf") if selection_metric == "train_loss" else float("-inf")
        best_epoch = 1
        best_update = 1
        stall = 0
        status = "COMPLETED"
        last_inner_val = {"median": -1.0, "mean": -1.0, "q25": -1.0, "q10": -1.0}
        total_updates_performed = 0
        hit_update_limit = False

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            network.train()
            loss_total = 0.0
            preclip_norms = []
            postclip_norms = []
            clips_applied = []
            steps_in_this_epoch = 0
            observed_grads = torch.zeros((len(ids), NPARAM_INFO_36[model_name]), dtype=torch.bool, device=device)

            for step_i in range(steps_per_epoch):
                if max_updates is not None and total_updates_performed >= max_updates:
                    hit_update_limit = True
                    break

                basins = torch.randperm(len(ids), device=device)[:batch_size]
                choices = (torch.rand(batch_size, device=device) * lengths[basins]).long()
                starts = catalog[basins, choices]
                x_b = gather_window(fit_x, starts, basins, horizon_days=horizon, device=device)
                y_b = gather_window(fit_y, starts, basins, horizon_days=horizon, device=device)

                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                theta = network(attrs[basins])
                theta.retain_grad()
                q = hydro({"x_phy": x_b}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)

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
                total_updates_performed += 1
                steps_in_this_epoch += 1

                if device.type == "cuda":
                    torch.cuda.synchronize()
                loss_total += float(loss.detach().item())

                # Update-budget milestone evaluation for Phase U
                if max_updates is not None and total_updates_performed in milestone_updates:
                    _inv_count, milestone_inner_val = evaluate_inner_val(
                        model_name, hydro, network, attrs, inner_x, scored_y
                    )
                    append_csv(
                        update_csv_path,
                        [
                            {
                                "job_id": job_id,
                                "update": total_updates_performed,
                                "target_updates": max_updates,
                                "progress_fraction": total_updates_performed / max_updates,
                                "step_loss": float(loss.detach().item()),
                                "inner_val_median_kge": milestone_inner_val["median"],
                                "inner_val_q25_kge": milestone_inner_val["q25"],
                                "inner_val_mean_kge": milestone_inner_val["mean"],
                            }
                        ],
                    )

            if steps_in_this_epoch == 0 and hit_update_limit:
                break

            if scheduler is not None:
                scheduler.step()

            avg_train_loss = loss_total / max(steps_in_this_epoch, 1)
            ep_time = time.time() - ep_start

            # Evaluate saturation and Jacobians
            network.eval()
            with torch.no_grad():
                u_eval = network(attrs)
                jac_eval = hydro.normalized_parameter_mapping_jacobian(u_eval)

            u_np = u_eval.cpu().numpy()
            jac_np = jac_eval.cpu().numpy()
            sat_low = float(np.mean(u_np < 0.02))
            sat_high = float(np.mean(u_np > 0.98))
            jac_p05 = float(np.percentile(jac_np, 5))
            jac_med = float(np.median(jac_np))

            # Periodic Inner-Val Evaluation
            if (max_updates is None and (epoch % inner_val_freq == 0 or epoch == epochs or epoch == 1)) or hit_update_limit:
                _inv_count, last_inner_val = evaluate_inner_val(
                    model_name, hydro, network, attrs, inner_x, scored_y
                )

            # Record epoch metrics
            append_csv(
                epoch_csv_path,
                [
                    {
                        "job_id": job_id,
                        "epoch": epoch,
                        "cumulative_updates": total_updates_performed,
                        "train_loss": avg_train_loss,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "inner_val_median_kge": last_inner_val["median"],
                        "inner_val_mean_kge": last_inner_val["mean"],
                        "inner_val_q25_kge": last_inner_val["q25"],
                        "inner_val_q10_kge": last_inner_val["q10"],
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

            # Selection & Exact best.pt
            is_best = False
            if selection_metric == "train_loss":
                if avg_train_loss < best_metric - plateau_eps:
                    best_metric = avg_train_loss
                    best_epoch = epoch
                    best_update = total_updates_performed
                    stall = 0
                    is_best = True
                else:
                    stall += 1

            if is_best or epoch == 1:
                best_dst = run_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "update": total_updates_performed,
                        "network": network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "job_config": job_config,
                        "selection_metric": selection_metric,
                        "best_selection_value": best_metric,
                        "inner_val_metrics": last_inner_val,
                        "git_sha": get_git_sha(),
                    },
                    best_dst,
                )
                (run_dir / "best_metadata.json").write_text(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "model": model_name,
                            "best_epoch": epoch,
                            "best_update": best_update,
                            "target_optimizer_updates": max_updates,
                            "actual_optimizer_updates": total_updates_performed,
                            "steps_per_epoch": steps_per_epoch,
                            "partial_final_epoch": (total_updates_performed % steps_per_epoch != 0),
                            "selection_metric": selection_metric,
                            "best_selection_value": best_metric,
                            "inner_val_median_kge": last_inner_val["median"],
                            "inner_val_q25_kge": last_inner_val["q25"],
                            "git_sha": get_git_sha(),
                            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        indent=2,
                    )
                )

            # Stopping Conditions
            if hit_update_limit:
                status = "UPDATE_BUDGET_REACHED"
                break

            if max_updates is None and epoch >= min_epochs and stall >= patience:
                status = "PLATEAU_STOP"
                break

        # Final reload of best.pt to compute conclusive inner-val metrics
        best_pt_path = run_dir / "best.pt"
        if best_pt_path.exists():
            payload = torch.load(best_pt_path, map_location="cpu", weights_only=False)
            network.load_state_dict(payload["network"])
            _inv_count, final_inner_val = evaluate_inner_val(
                model_name, hydro, network, attrs, inner_x, scored_y
            )
        else:
            final_inner_val = last_inner_val

        total_runtime = time.time() - start_time

        # Save summary JSONs
        (run_dir / "inner_validation_metrics.json").write_text(json.dumps(final_inner_val, indent=2))
        (run_dir / "runtime.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "total_runtime_s": total_runtime,
                    "final_epoch": epoch,
                    "best_epoch": best_epoch,
                    "target_optimizer_updates": max_updates,
                    "actual_optimizer_updates": total_updates_performed,
                    "steps_per_epoch": steps_per_epoch,
                    "partial_final_epoch": (total_updates_performed % steps_per_epoch != 0) if max_updates else False,
                    "status": status,
                },
                indent=2,
            )
        )
        (run_dir / "exit_status.json").write_text(json.dumps({"exit_code": 0, "status": status}, indent=2))

        # Checkpoint sha256
        if best_pt_path.exists():
            (run_dir / "best_state_sha256.txt").write_text(get_file_sha256(best_pt_path) + "\n")

        # Mark DONE
        done_file.write_text(f"COMPLETED={time.strftime('%Y-%m-%d %H:%M:%S')}\nSTATUS={status}\nUPDATES={total_updates_performed}\n")
        if lock_file.exists():
            lock_file.unlink()

        return {
            "job_id": job_id,
            "status": "DONE",
            "exit_code": 0,
            "best_epoch": best_epoch,
            "best_update": best_update,
            "target_optimizer_updates": max_updates,
            "actual_optimizer_updates": total_updates_performed,
            "best_train_loss": best_metric,
            "inner_val_median_kge": final_inner_val["median"],
            "inner_val_q25_kge": final_inner_val["q25"],
            "runtime_s": total_runtime,
        }

    except Exception as exc:
        total_runtime = time.time() - start_time
        err_msg = traceback.format_exc()
        failed_file.write_text(f"FAILED_TIME={time.strftime('%Y-%m-%d %H:%M:%S')}\nERROR={str(exc)}\n")
        (run_dir / "failure_reason.md").write_text(f"# Failure Report for `{job_id}`\n\n```text\n{err_msg}\n```\n")
        (run_dir / "exit_status.json").write_text(json.dumps({"exit_code": 1, "error": str(exc)}, indent=2))
        if lock_file.exists():
            lock_file.unlink()
        return {
            "job_id": job_id,
            "status": "FAILED",
            "exit_code": 1,
            "error": str(exc),
            "runtime_s": total_runtime,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-config", type=str, required=True, help="JSON or YAML string/path of job config")
    parser.add_argument("--out", type=str, required=True, help="Output ablation root directory")
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

    res = execute_ablation_job(job_cfg, out_dir, dev)
    sys.exit(res["exit_code"])


if __name__ == "__main__":
    main()
