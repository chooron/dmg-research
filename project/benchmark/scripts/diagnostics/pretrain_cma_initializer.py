#!/usr/bin/env python3
"""H2: CUDA-only supervised CMA-theta initialization for the dPL MLP.

This module never instantiates a hydrological model. It fits the existing
two-hidden-layer CatchmentParameterizer to archived CMA theta values and saves
the resulting state for the B2b training branch.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.data_selection import load_ids

DEVICE = torch.device("cuda")
OUT = ROOT / "results/dpl_training_pilot_20260801/pretrain"
MODELS = ("collie3", "newzealand1", "penman", "flexi", "flexis")
CMA_ROOT = ROOT / "dmotpy/experiments/cmaes_36models/downloads/full300_20260729_160112_partial_20260730/checkpoints_latest"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def archive_theta(model_name: str) -> tuple[list[int], torch.Tensor, Path]:
    path = CMA_ROOT / model_name / "chunk_0_gen_300.pt"
    if not path.exists():
        raise FileNotFoundError(f"CMA archive unavailable: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    basin_ids = [int(value) for value in payload["basin_ids"]]
    state = payload["solver"]["state"]
    latent = state["best_latent"].to(device=DEVICE, dtype=torch.float32)
    fitness = state["best_fitness"].to(device=DEVICE, dtype=torch.float32)
    starts = latent.shape[0] // len(basin_ids)
    latent = latent.reshape(len(basin_ids), starts, -1)
    fitness = fitness.reshape(len(basin_ids), starts)
    best = fitness.argmax(dim=1)
    theta = torch.sigmoid(latent[torch.arange(len(basin_ids), device=DEVICE), best])
    return basin_ids, theta, path


def r2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = (prediction - target).square().sum(dim=0)
    total = (target - target.mean(dim=0, keepdim=True)).square().sum(dim=0)
    return 1.0 - residual / total.clamp_min(torch.finfo(target.dtype).eps)


def fit(model_name: str, scale: float, max_steps: int, lr: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(20260801)
    torch.cuda.manual_seed_all(20260801)
    all_ids = [int(value) for value in load_ids("data/531sub_id.txt")]
    all_attrs = CatchmentAttributeBuilder().build_normalized_attributes(all_ids, device="cuda", method="zscore")
    archive_ids, theta_star, archive_path = archive_theta(model_name)
    index = {basin: i for i, basin in enumerate(all_ids)}
    missing = [basin for basin in archive_ids if basin not in index]
    if missing:
        raise RuntimeError(f"archive basins absent from 531 list: {missing[:5]}")
    attrs = all_attrs[torch.as_tensor([index[basin] for basin in archive_ids], device=DEVICE)]
    target = 0.5 + scale * (theta_star - 0.5)
    network = CatchmentParameterizer(
        in_features=attrs.shape[1], out_features=target.shape[1], hidden_dims=[256, 256], dropout=0.05,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, stale = float("inf"), None, 0
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        network.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(network(attrs), target)
        loss.backward(); optimizer.step()
        network.eval()
        with torch.no_grad():
            current = float(torch.nn.functional.mse_loss(network(attrs), target))
        if current < best - 1e-10:
            best = current
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 50:
            break
    if best_state is None:
        raise RuntimeError("pretraining never produced a finite state")
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        prediction = network(attrs)
        full_prediction = network(all_attrs)
    elapsed = time.perf_counter() - started
    parameter_rows = []
    names = list(__import__("dmotpy.models.registry", fromlist=["PARAM_INFO"]).PARAM_INFO[model_name])
    for j, name in enumerate(names):
        parameter_rows.append({
            "model": model_name, "scale": scale, "parameter": name,
            "r2_theta_target": float(r2(prediction, target)[j]),
            "mse_theta_target": float(torch.nn.functional.mse_loss(prediction[:, j], target[:, j])),
            "target_boundary_fraction": float(((target[:, j] < .02) | (target[:, j] > .98)).float().mean()),
            "predicted_boundary_fraction_archive": float(((prediction[:, j] < .02) | (prediction[:, j] > .98)).float().mean()),
            "predicted_boundary_fraction_531": float(((full_prediction[:, j] < .02) | (full_prediction[:, j] > .98)).float().mean()),
        })
    output_path = OUT / f"{model_name}_s{scale:.1f}.pt"
    torch.save({
        "model": model_name, "scale": scale, "archive_basin_ids": archive_ids,
        "archive_path": str(archive_path), "state_dict": network.state_dict(),
        "hidden_dims": [256, 256], "dropout": 0.05,
    }, output_path)
    summary = {
        "model": model_name, "scale": scale, "archive_basin_count": len(archive_ids),
        "steps": step, "best_mse": best, "elapsed_seconds": elapsed,
        "prediction_boundary_fraction_archive": float(((prediction < .02) | (prediction > .98)).float().mean()),
        "prediction_boundary_fraction_531": float(((full_prediction < .02) | (full_prediction > .98)).float().mean()),
        "checkpoint": str(output_path), "archive": str(archive_path),
    }
    del network, optimizer, all_attrs, attrs, target, theta_star
    torch.cuda.empty_cache()
    return summary, parameter_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--scale", choices=(0.8, 1.0), type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, params = [], []
    for model in (args.model,) if args.model else MODELS:
        for scale in (args.scale,) if args.scale is not None else (0.8, 1.0):
            summary, rows = fit(model, scale, args.max_steps, args.lr)
            summaries.append(summary); params.extend(rows)
    write_csv(OUT / "pretrain_summary.csv", summaries)
    write_csv(OUT / "pretrain_parameter_r2.csv", params)
    (OUT / "pretrain_contract.json").write_text(json.dumps({
        "architecture": "CatchmentParameterizer hidden_dims=[256,256], dropout=0.05",
        "attribute_source": "caravan_671_attributes.npy via CatchmentAttributeBuilder, zscore over all 531 basins",
        "target": "0.5 + scale * (theta_star - 0.5)",
        "early_stopping": "full archive theta MSE has no improvement greater than 1e-10 for 50 steps",
        "hydrological_forward_calls": 0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
