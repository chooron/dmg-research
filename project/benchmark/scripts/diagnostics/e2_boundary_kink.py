#!/usr/bin/env python3
"""E2 CUDA audit: CMA-boundary gradient concentration and directional kinks.

This script intentionally evaluates the same CMA-aligned objective used in T6:
fp32 forcing/theta, 1825-day warm-up, 1989-1998 score period, eps=0.1, and
full warm-up reverse mode.  Every model evaluation is CUDA-only and uses the
registry's compiled step backend.  Eight-basin chunks are a memory partition,
not per-basin serial evaluation.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dmotpy.models.registry import PARAM_INFO
from src.model_registry import build_model
from src.objective import streaming_kge

DEVICE = torch.device("cuda")
OUT = ROOT / "results/dpl_reachability_20260731"
MODELS = ("collie3", "penman")
CHUNK_SIZE = 8
HS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROUND3 = load_module(ROOT / "scripts/diagnostics/dpl_third_round_diagnostics.py", "e2_round3")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cma_score(model, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
    score, invalid = streaming_kge(q.unsqueeze(-1).unsqueeze(-1), y, eps=0.1)
    if bool(invalid.any()):
        raise FloatingPointError("non-finite CMA-aligned prediction")
    return score[:, 0, 0]


def slices(n: int):
    return tuple(slice(left, min(left + CHUNK_SIZE, n)) for left in range(0, n, CHUNK_SIZE))


def loss_and_gradient(model_name: str, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Return the global mean loss and its correctly normalized batched gradient."""
    model = build_model(model_name, DEVICE, warm_up=1825, backend="compile", warmup_grad_mode="full")
    parts = slices(theta.shape[0])
    grads, losses = [], []
    for part in parts:
        local = theta[part].detach().clone().requires_grad_(True)
        loss = 1.0 - cma_score(model, x[:, part], y[:, part], local).mean()
        grads.append(torch.autograd.grad(loss, local)[0] / len(parts))
        losses.append(loss.detach())
    del model
    torch.cuda.empty_cache()
    return float(torch.stack(losses).mean()), torch.cat(grads)


def loss_only(model_name: str, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> float:
    model = build_model(model_name, DEVICE, warm_up=1825, backend="compile", warmup_grad_mode="full")
    values = []
    with torch.no_grad():
        for part in slices(theta.shape[0]):
            values.append(1.0 - cma_score(model, x[:, part], y[:, part], theta[part]).mean())
    del model
    torch.cuda.empty_cache()
    return float(torch.stack(values).mean())


def audit(model_name: str) -> None:
    x, y = ROUND3.cma_data()
    theta, source = ROUND3.best_theta(model_name, torch.float32)
    theta = theta.detach()
    loss, gradient = loss_and_gradient(model_name, x, y, theta)
    norm = gradient.norm()
    direction = -gradient / norm.clamp_min(torch.finfo(gradient.dtype).eps)
    total_sq = gradient.square().sum().clamp_min(torch.finfo(gradient.dtype).tiny)

    contribution_rows, boundary_rows = [], []
    lower_all = theta < 1e-3
    upper_all = theta > 1.0 - 1e-3
    for j, parameter in enumerate(PARAM_INFO[model_name]):
        gj = gradient[:, j]
        lower = lower_all[:, j]
        upper = upper_all[:, j]
        # Direction d is the proposed descending direction.  At a lower bound
        # d>0 points inward; at an upper bound d<0 points inward.
        lower_in = (direction[:, j][lower] > 0).sum()
        upper_in = (direction[:, j][upper] < 0).sum()
        contribution_rows.append({
            "model": model_name,
            "parameter": parameter,
            "gradient_sq_contribution": float(gj.square().sum() / total_sq),
            "gradient_l2": float(gj.norm()),
            "source": source,
        })
        boundary_rows.append({
            "model": model_name,
            "parameter": parameter,
            "lower_count": int(lower.sum()),
            "upper_count": int(upper.sum()),
            "boundary_count": int(lower.sum() + upper.sum()),
            "lower_direction_inward_count": int(lower_in),
            "upper_direction_inward_count": int(upper_in),
            "lower_direction_outward_count": int(lower.sum() - lower_in),
            "upper_direction_outward_count": int(upper.sum() - upper_in),
            "source": source,
        })

    derivative_rows = []
    for h in HS:
        plus_raw = theta + h * direction
        minus_raw = theta - h * direction
        plus = plus_raw.clamp(0.0, 1.0)
        minus = minus_raw.clamp(0.0, 1.0)
        loss_plus = loss_only(model_name, x, y, plus)
        loss_minus = loss_only(model_name, x, y, minus)
        derivative_rows.append({
            "model": model_name,
            "h": h,
            "loss_theta_star": loss,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "forward_directional_derivative": (loss_plus - loss) / h,
            "backward_directional_derivative": (loss - loss_minus) / h,
            "derivative_abs_difference": abs((loss_plus - loss) / h - (loss - loss_minus) / h),
            "plus_projected_fraction": float((plus != plus_raw).float().mean()),
            "minus_projected_fraction": float((minus != minus_raw).float().mean()),
            "theta_dtype": str(theta.dtype),
            "source": source,
        })

    write_csv(OUT / f"e2_{model_name}_gradient_contribution.csv", sorted(contribution_rows, key=lambda row: row["gradient_sq_contribution"], reverse=True))
    write_csv(OUT / f"e2_{model_name}_boundary.csv", boundary_rows)
    write_csv(OUT / f"e2_{model_name}_directional_derivatives.csv", derivative_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E2")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (args.model,) if args.model else MODELS:
        audit(name)


if __name__ == "__main__":
    main()
