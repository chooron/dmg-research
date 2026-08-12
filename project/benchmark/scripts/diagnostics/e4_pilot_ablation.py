#!/usr/bin/env python3
"""E4 B0-B2 CUDA pilot, isolated from the production dPL runner.

The pilot intentionally uses the common 32-CMA-basin diagnostic cohort so all
five archived models share the same data contract.  Every hydrology/loss/
gradient operation is a CUDA tensor operation; host work only loads immutable
data and appends scalar CSV rows.  B3 is not implemented because E3 measured
zero strict revivals for state_init.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.models.registry import PARAM_INFO
from src.model_registry import NPARAM_INFO_36, build_model

DEVICE = torch.device("cuda")
OUT = ROOT / "results/dpl_reachability_20260731/e4_pilot"
MODELS = ("collie3", "newzealand1", "penman", "flexi", "flexis", "hbv96")
CONFIGS = ("B0", "B1", "B2")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROUND3 = load_module(ROOT / "scripts/diagnostics/dpl_third_round_diagnostics.py", "e4_round3")
NATIVE = load_module(ROOT / "scripts/run_dpl_benchmark_dmg_native.py", "e4_native_runner")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ids32() -> list[int]:
    return [int(value) for value in ROUND3.ids32().cpu().tolist()]


def cma_median(model_name: str) -> tuple[torch.Tensor | None, str]:
    try:
        theta, source, count = ROUND3.all_archive_theta(model_name, torch.float32)
    except RuntimeError as exc:
        return None, str(exc)
    return theta.median(dim=0).values, f"{source}; median of {count} archive basins"


def fixed_output_initialization(parameterizer: CatchmentParameterizer, theta: torch.Tensor | None) -> None:
    """Make B0/B1 exactly theta=0.5 and B2 exactly its CMA median at epoch 0."""
    output = parameterizer.net[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("CatchmentParameterizer final module must be nn.Linear")
    with torch.no_grad():
        output.weight.zero_()
        output.bias.zero_()
    if theta is not None:
        parameterizer.initialize_output_bias_from_theta(theta)


def gather_window(values: torch.Tensor, starts: torch.Tensor, length: int) -> torch.Tensor:
    days = torch.arange(length, device=DEVICE)[:, None] + starts[None, :]
    basins = torch.arange(values.shape[1], device=DEVICE)[None, :]
    return values[days, basins]


def model_kge(q: torch.Tensor, y: torch.Tensor, *, warmup_days: int) -> tuple[torch.Tensor, torch.Tensor]:
    loss, per_basin = NATIVE.compute_differentiable_kge(q, y, warmup_days=warmup_days)
    return loss, per_basin


def run_one(model_name: str, config_name: str, epochs: int, lr: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    # Every B0/B1/B2 comparison sees the same MLP trunk initialization and
    # the same all-CUDA window sequence; only the named intervention varies.
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    ids = ids32()
    train_x_np, train_y_np, val_x_np, val_y_np = NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(train_x_np, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(train_y_np, dtype=torch.float32, device=DEVICE)
    val_x = torch.as_tensor(val_x_np, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(val_y_np, dtype=torch.float32, device=DEVICE)
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")

    mapping = "linear" if config_name == "B0" else "auto"
    median, median_source = cma_median(model_name)
    if config_name == "B2" and median is None:
        return [], [], {"model": model_name, "config": config_name, "status": "SKIPPED_NO_CMA_ARCHIVE", "detail": median_source}
    hydro = build_model(model_name, DEVICE, warm_up=365, backend="compile", parameter_mapping=mapping, warmup_grad_mode="detach")
    parameterizer = CatchmentParameterizer(
        in_features=attrs.shape[1], out_features=NPARAM_INFO_36[model_name], hidden_dims=[256, 256], dropout=0.05,
    ).to(DEVICE)
    fixed_output_initialization(parameterizer, median if config_name == "B2" else None)
    optimizer = torch.optim.AdamW(parameterizer.parameters(), lr=lr, weight_decay=1e-4)

    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        parameterizer.train()
        starts = torch.randint(0, train_x.shape[0] - 730 + 1, (len(ids),), device=DEVICE)
        x_batch = gather_window(train_x, starts, 730)
        y_batch = gather_window(train_y, starts, 730)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); started = time.perf_counter()
        theta = parameterizer(attrs)
        theta.retain_grad()
        q = hydro({"x_phy": x_batch}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        loss, train_kge = model_kge(q, y_batch[365:], warmup_days=0)
        loss.backward()
        theta_grad = theta.grad.detach()
        nn.utils.clip_grad_norm_(parameterizer.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize(); step_seconds = time.perf_counter() - started

        parameterizer.eval()
        with torch.no_grad():
            val_theta = parameterizer(attrs)
            val_q = hydro({"x_phy": val_x}, (None, val_theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            _val_loss, val_kge = model_kge(val_q, val_y, warmup_days=365)
        rows.append({
            "model": model_name, "config": config_name, "epoch": epoch,
            "train_mean_kge": float(train_kge.mean()), "validation_median_kge": float(val_kge.median()),
            "validation_mean_kge": float(val_kge.mean()),
            "theta_boundary_fraction": float(((val_theta < 0.02) | (val_theta > 0.98)).float().mean()),
            "seconds_per_step": step_seconds,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "parameter_mapping": mapping,
            "cma_bias_source": median_source if config_name == "B2" else "not enabled",
        })
        for index, parameter in enumerate(PARAM_INFO[model_name]):
            parameter_rows.append({
                "model": model_name, "config": config_name, "epoch": epoch, "parameter": parameter,
                "zero_gradient_basin_fraction": float((theta_grad[:, index] == 0).float().mean()),
                "theta_boundary_basin_fraction": float(((val_theta[:, index] < 0.02) | (val_theta[:, index] > 0.98)).float().mean()),
            })
    del hydro, parameterizer, optimizer
    torch.cuda.empty_cache()
    return rows, parameter_rows, {"model": model_name, "config": config_name, "status": "COMPLETED", "detail": median_source if config_name == "B2" else ""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--config", choices=CONFIGS, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E4")
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, all_parameter_rows, status_rows = [], [], []
    for model_name in (args.model,) if args.model else MODELS:
        for config_name in (args.config,) if args.config else CONFIGS:
            rows, parameter_rows, status = run_one(model_name, config_name, args.epochs, args.lr)
            all_rows.extend(rows); all_parameter_rows.extend(parameter_rows); status_rows.append(status)
            if rows:
                write_csv(OUT / f"{model_name}_{config_name}_epochs.csv", rows)
                write_csv(OUT / f"{model_name}_{config_name}_parameter_gradients.csv", parameter_rows)
    write_csv(OUT / "e4_status.csv", status_rows)
    (OUT / "e4_scope.json").write_text(json.dumps({
        "basins": 32, "epochs": args.epochs, "hidden_dims": [256, 256],
        "B0": "detach + linear + exact theta=0.5 output initialization",
        "B1": "B0 + parameter_mapping=auto",
        "B2": "B1 + CMA archive median output bias",
        "B3": "excluded: E3 strict revival = 0 parameters / 0 models",
        "hbv96_B2": "skipped: no local CMA archive",
        "flexi_flexis_CMA_median": "computed from their local 128-basin archives",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
