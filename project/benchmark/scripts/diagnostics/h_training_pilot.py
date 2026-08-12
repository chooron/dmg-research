#!/usr/bin/env python3
"""H1/H3: resumable 531-basin four-branch dPL training pilot.

The runner intentionally preserves the historical native dPL contract: all
531 basins, 1980-10-01..1995-09-30 training, 1995-10-01..2010-09-30
validation, 365+365 windows, batch size 100, 169 steps/epoch, detach warm-up,
and the native differentiable KGE.  It neither changes a model formula nor
uses a new basin split.
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
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model

DEVICE = torch.device("cuda")
# h1 held a smoke run that reported only the final 100-basin minibatch's
# gradients. h1_v2 is the authoritative result directory with full-531
# per-epoch gradient coverage.
OUT = ROOT / "results/dpl_training_pilot_20260801/h1_v2"
PRETRAIN = ROOT / "results/dpl_training_pilot_20260801/pretrain"
MODELS = ("collie3", "newzealand1", "penman", "flexi", "flexis", "hbv96")
CONFIGS = ("B0", "B1", "B2a", "B2b")
BATCH_SIZE = 100
STEPS_PER_EPOCH = 169
WINDOW_DAYS = 730
WARMUP_DAYS = 365
SEED = 42


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NATIVE = load_module(ROOT / "scripts/run_dpl_benchmark_dmg_native.py", "h1_native")
PRETRAIN_MODULE = load_module(ROOT / "scripts/diagnostics/pretrain_cma_initializer.py", "h1_pretrain")


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def archive_median(model_name: str) -> tuple[torch.Tensor | None, str]:
    try:
        _ids, theta, path = PRETRAIN_MODULE.archive_theta(model_name)
    except FileNotFoundError as exc:
        return None, str(exc)
    return theta.median(dim=0).values, f"{path}; median of {theta.shape[0]} archive basins"


def initialize_output_at_theta_half(network: CatchmentParameterizer) -> None:
    output = network.net[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("parameterizer output must be Linear")
    with torch.no_grad():
        output.weight.zero_(); output.bias.zero_()


def configuration(model_name: str, branch: str, attrs: torch.Tensor) -> tuple[CatchmentParameterizer | None, str, str]:
    mapping = "linear" if branch == "B0" else "auto"
    if branch in ("B2a", "B2b") and model_name == "hbv96":
        return None, mapping, "SKIPPED_NO_LOCAL_CMA_ARCHIVE"
    network = CatchmentParameterizer(
        in_features=attrs.shape[1], out_features=NPARAM_INFO_36[model_name], hidden_dims=[256, 256], dropout=.05,
    ).to(DEVICE)
    if branch in ("B0", "B1"):
        initialize_output_at_theta_half(network)
        return network, mapping, "exact_theta_0.5"
    if branch == "B2a":
        theta, source = archive_median(model_name)
        if theta is None:
            return None, mapping, "SKIPPED_NO_LOCAL_CMA_ARCHIVE: " + source
        initialize_output_at_theta_half(network)
        network.initialize_output_bias_from_theta(theta)
        return network, mapping, source
    if branch == "B2b":
        path = PRETRAIN / f"{model_name}_s0.8.pt"
        if not path.exists():
            return None, mapping, "MISSING_PRETRAIN_CHECKPOINT: " + str(path)
        payload = torch.load(path, map_location=DEVICE, weights_only=False)
        if payload.get("hidden_dims") != [256, 256] or payload.get("scale") != .8:
            raise RuntimeError(f"invalid B2b pretrain contract in {path}")
        network.load_state_dict(payload["state_dict"])
        return network, mapping, str(path)
    raise ValueError(branch)


def make_catalog(observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfer immutable valid-window indices once; random selection is CUDA."""
    catalog = NATIVE.build_informative_kge_catalog(observations.detach().cpu().numpy().T)
    width = max(len(row) for row in catalog)
    starts = torch.zeros((len(catalog), width), dtype=torch.long, device=DEVICE)
    lengths = torch.empty(len(catalog), dtype=torch.long, device=DEVICE)
    for basin, row in enumerate(catalog):
        starts[basin, :len(row)] = torch.as_tensor(row, dtype=torch.long, device=DEVICE)
        lengths[basin] = len(row)
    return starts, lengths


def gather_window(values: torch.Tensor, starts: torch.Tensor, basin_indices: torch.Tensor) -> torch.Tensor:
    days = torch.arange(WINDOW_DAYS, device=DEVICE)[:, None] + starts[None, :]
    return values[days, basin_indices[None, :]]


def checkpoint_path(model_name: str, branch: str, epoch: int) -> Path:
    return OUT / "checkpoints" / model_name / f"{branch}_epoch_{epoch:03d}.pt"


def latest_checkpoint(model_name: str, branch: str) -> Path | None:
    files = sorted((OUT / "checkpoints" / model_name).glob(f"{branch}_epoch_*.pt"))
    return files[-1] if files else None


def previously_logged(model_name: str, branch: str) -> set[int]:
    path = OUT / "epochs.csv"
    if not path.exists():
        return set()
    with path.open() as handle:
        return {int(row["epoch"]) for row in csv.DictReader(handle) if row["model"] == model_name and row["branch"] == branch}


def run_branch(model_name: str, branch: str, epochs: int, lr: float, evaluation_every: int) -> dict[str, Any]:
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    ids = [int(value) for value in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    parameterizer, mapping, initialization = configuration(model_name, branch, attrs)
    if parameterizer is None:
        return {"model": model_name, "branch": branch, "status": initialization}
    train_x_np, train_y_np, val_x_np, val_y_np = NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(train_x_np, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(train_y_np, dtype=torch.float32, device=DEVICE)
    val_x = torch.as_tensor(val_x_np, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(val_y_np, dtype=torch.float32, device=DEVICE)
    catalog, catalog_lengths = make_catalog(train_y[WARMUP_DAYS:])
    hydro = build_model(model_name, DEVICE, warm_up=WARMUP_DAYS, backend="compile", parameter_mapping=mapping, warmup_grad_mode="detach")
    optimizer = torch.optim.AdamW(parameterizer.parameters(), lr=lr, weight_decay=1e-4)
    start_epoch = 1
    old = latest_checkpoint(model_name, branch)
    if old is not None:
        # Keep the saved CPU RNG byte tensor on CPU; load_state_dict moves
        # parameter and optimizer buffers to their live CUDA parameters.
        payload = torch.load(old, map_location="cpu", weights_only=False)
        parameterizer.load_state_dict(payload["parameterizer"]); optimizer.load_state_dict(payload["optimizer"])
        torch.random.set_rng_state(payload["cpu_rng"]); torch.cuda.set_rng_state(payload["cuda_rng"], device=DEVICE)
        start_epoch = int(payload["epoch"]) + 1
    logged = previously_logged(model_name, branch)
    all_basins = torch.arange(len(ids), device=DEVICE)
    plateau_values: list[float] = []
    stopped = False
    for epoch in range(start_epoch, epochs + 1):
        parameterizer.train(); epoch_loss, elapsed = 0.0, 0.0
        # A basin is marked nonzero if any of its native 169 sampled windows
        # has a nonzero theta derivative. With 169x100 draws every 531 basin
        # is repeatedly observed, and this preserves the exact-zero audit
        # meaning without adding a diagnostic hydrology backward pass.
        observed_nonzero = torch.zeros((len(ids), NPARAM_INFO_36[model_name]), dtype=torch.bool, device=DEVICE)
        for _ in range(STEPS_PER_EPOCH):
            basin_indices = torch.randperm(len(ids), device=DEVICE)[:BATCH_SIZE]
            choices = (torch.rand(BATCH_SIZE, device=DEVICE) * catalog_lengths[basin_indices]).long()
            starts = catalog[basin_indices, choices]
            x_batch = gather_window(train_x, starts, basin_indices)
            y_batch = gather_window(train_y, starts, basin_indices)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); begun = time.perf_counter()
            theta = parameterizer(attrs[basin_indices]); theta.retain_grad()
            q = hydro({"x_phy": x_batch}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            loss, _ = NATIVE.compute_differentiable_kge(q, y_batch[WARMUP_DAYS:], warmup_days=0)
            loss.backward()
            observed_nonzero[basin_indices] |= theta.grad.detach() != 0
            nn.utils.clip_grad_norm_(parameterizer.parameters(), max_norm=1.0); optimizer.step()
            torch.cuda.synchronize(); elapsed += time.perf_counter() - begun
            epoch_loss += float(loss.detach())
        if epoch % evaluation_every != 0 and epoch != epochs:
            continue
        parameterizer.eval()
        with torch.no_grad():
            val_theta = parameterizer(attrs)
            val_q = hydro({"x_phy": val_x}, (None, val_theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            _validation_loss, validation_kge = NATIVE.compute_differentiable_kge(val_q, val_y, warmup_days=WARMUP_DAYS)
        median = float(validation_kge.median())
        plateau_values.append(median)
        row = {
            "model": model_name, "branch": branch, "epoch": epoch, "status": "COMPLETED_EPOCH",
            "validation_median_kge": median, "validation_mean_kge": float(validation_kge.mean()),
            "train_loss_1_minus_kge": epoch_loss / STEPS_PER_EPOCH,
            "theta_boundary_fraction": float(((val_theta < .02) | (val_theta > .98)).float().mean()),
            "seconds_per_train_step": elapsed / STEPS_PER_EPOCH, "parameter_mapping": mapping,
            "initialization": initialization,
        }
        if epoch not in logged:
            append_csv(OUT / "epochs.csv", [row])
            parameter_rows = [{
                "model": model_name, "branch": branch, "epoch": epoch, "parameter": parameter,
                "zero_gradient_basin_fraction": float((~observed_nonzero[:, j]).float().mean()),
                "theta_boundary_basin_fraction": float(((val_theta[:, j] < .02) | (val_theta[:, j] > .98)).float().mean()),
            } for j, parameter in enumerate(PARAM_INFO[model_name])]
            append_csv(OUT / "parameter_gradients.csv", parameter_rows)
        if epoch % 10 == 0 or epoch == epochs:
            destination = checkpoint_path(model_name, branch, epoch); destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "parameterizer": parameterizer.state_dict(), "optimizer": optimizer.state_dict(),
                        "cpu_rng": torch.random.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(DEVICE)}, destination)
        if len(plateau_values) >= 21 and plateau_values[-1] - plateau_values[-21] < .002:
            stopped = True; break
    del hydro, parameterizer, optimizer, train_x, train_y, val_x, val_y
    torch.cuda.empty_cache()
    return {"model": model_name, "branch": branch, "status": "PLATEAU_STOP" if stopped else "COMPLETED", "last_epoch": epoch,
            "parameter_mapping": mapping, "initialization": initialization}


def summarize() -> None:
    path = OUT / "epochs.csv"
    if not path.exists():
        return
    latest: dict[tuple[str, str], dict[str, str]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = row["model"], row["branch"]
            if key not in latest or int(row["epoch"]) > int(latest[key]["epoch"]): latest[key] = row
    rows = []
    for model in MODELS:
        b0, b1, b2a, b2b = (latest.get((model, branch)) for branch in CONFIGS)
        def value(row): return float(row["validation_median_kge"]) if row else None
        base, auto, bias, pre = map(value, (b0, b1, b2a, b2b))
        rows.append({"model": model, "B0_final_median_kge": base, "B1_final_median_kge": auto,
                     "B2a_final_median_kge": bias, "B2b_final_median_kge": pre,
                     "B0_to_B1_delta": auto - base if auto is not None and base is not None else None,
                     "B1_to_B2a_delta": bias - auto if bias is not None and auto is not None else None,
                     "B1_to_B2b_delta": pre - auto if pre is not None and auto is not None else None})
    write_csv(OUT / "increment_summary.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--branch", choices=CONFIGS, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--evaluation-every", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contract.json").write_text(json.dumps({
        "basins": 531, "train_period": "1980-10-01..1995-09-30", "validation_period": "1995-10-01..2010-09-30",
        "window": "365 warm-up + 365 scored", "batch_size": BATCH_SIZE, "steps_per_epoch": STEPS_PER_EPOCH,
        "warmup_grad_mode": "detach", "branches": {"B0": "linear + exact theta=.5", "B1": "auto + exact theta=.5",
        "B2a": "auto + CMA median bias", "B2b": "auto + s=.8 supervised pretrain weights"},
        "hbv96": "B2a/B2b skipped due missing local CMA archive", "flexi_flexis": "CMA/pretrain labels use 128 archive basins",
    }, indent=2) + "\n")
    if args.summarize:
        summarize(); return
    status_rows = []
    for model in (args.model,) if args.model else MODELS:
        for branch in (args.branch,) if args.branch else CONFIGS:
            status_rows.append(run_branch(model, branch, args.epochs, args.lr, args.evaluation_every))
            write_csv(OUT / "status.csv", status_rows)
            summarize()


if __name__ == "__main__":
    main()
