#!/usr/bin/env python3
"""I4: archive-internal five-fold OOF R2 for supervised theta pretraining."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.models.registry import PARAM_INFO
from src.data_selection import load_ids

DEVICE = torch.device("cuda")
OUT = ROOT / "results/dpl_training_pilot_20260801/i4_pretrain_cv"
MODELS = ("collie3", "newzealand1", "penman", "flexi", "flexis")
SCALE, FOLDS, SEED = .8, 5, 20260802


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


PRETRAIN = load_module(ROOT / "scripts/diagnostics/pretrain_cma_initializer.py", "i4_pretrain")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def r2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = (prediction - target).square().sum(0)
    total = (target - target.mean(0, keepdim=True)).square().sum(0)
    return 1.0 - residual / total.clamp_min(torch.finfo(target.dtype).eps)


def run_model(model_name: str, max_steps: int, lr: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    all_ids = [int(value) for value in load_ids("data/531sub_id.txt")]
    all_attrs = CatchmentAttributeBuilder().build_normalized_attributes(all_ids, device="cuda", method="zscore")
    archive_ids, theta_star, source = PRETRAIN.archive_theta(model_name)
    id_index = {basin: index for index, basin in enumerate(all_ids)}
    attrs = all_attrs[torch.as_tensor([id_index[basin] for basin in archive_ids], device=DEVICE)]
    target = .5 + SCALE * (theta_star - .5)
    order = torch.randperm(len(archive_ids), device=DEVICE)
    fold_ids = torch.arange(len(archive_ids), device=DEVICE) % FOLDS
    fold_ids = fold_ids[torch.argsort(order)]  # fixed shuffled fold assignment in original row order
    oof = torch.empty_like(target)
    summary = []
    for fold in range(FOLDS):
        train, test = fold_ids != fold, fold_ids == fold
        network = CatchmentParameterizer(attrs.shape[1], target.shape[1], hidden_dims=[256, 256], dropout=.05).to(DEVICE)
        optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
        best, state, stale = float("inf"), None, 0
        begun = time.perf_counter()
        for step in range(1, max_steps + 1):
            network.train(); optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(network(attrs[train]), target[train]); loss.backward(); optimizer.step()
            network.eval()
            with torch.no_grad(): current = float(torch.nn.functional.mse_loss(network(attrs[train]), target[train]))
            if current < best - 1e-10:
                best, state, stale = current, {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}, 0
            else: stale += 1
            if stale >= 50: break
        network.load_state_dict(state); network.eval()
        with torch.no_grad(): prediction = network(attrs[test]); oof[test] = prediction
        summary.append({"model": model_name, "fold": fold, "train_count": int(train.sum()), "test_count": int(test.sum()),
                        "steps": step, "best_train_mse": best, "test_mse": float(torch.nn.functional.mse_loss(prediction, target[test])),
                        "elapsed_seconds": time.perf_counter() - begun, "archive": str(source)})
        del network, optimizer
    rows = []
    names = list(PARAM_INFO[model_name])
    scores = r2(oof, target)
    for j, name in enumerate(names):
        rows.append({"model": model_name, "parameter": name, "scale": SCALE, "oof_r2_theta_target": float(scores[j]),
                     "oof_mse_theta_target": float(torch.nn.functional.mse_loss(oof[:, j], target[:, j])),
                     "archive_basin_count": len(archive_ids), "archive": str(source)})
    del all_attrs, attrs, theta_star, target, oof
    torch.cuda.empty_cache()
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--max-steps", type=int, default=5000); parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True, exist_ok=True)
    summary, parameters = [], []
    for model in (args.model,) if args.model else MODELS:
        folds, rows = run_model(model, args.max_steps, args.lr); summary.extend(folds); parameters.extend(rows)
    write_csv(OUT / "i4_fold_summary.csv", summary); write_csv(OUT / "i4_oof_parameter_r2.csv", parameters)


if __name__ == "__main__":
    main()
