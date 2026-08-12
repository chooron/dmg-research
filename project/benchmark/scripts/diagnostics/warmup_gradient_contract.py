#!/usr/bin/env python3
"""CUDA diagnostics for the configurable warm-up gradient contract (T6-T8)."""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dmotpy.models.registry import PARAM_INFO
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model
from src.objective import streaming_kge

DEVICE = torch.device("cuda")
OUT = ROOT / "results/dpl_warmup_contract_20260731"
EVIDENCE = ROOT / "results/dpl_gradient_evidence_20260731_epoch"
MODELS = ("collie3", "newzealand1", "penman", "flexi", "flexis")
MODES = ("detach", "truncate:90", "truncate:180", "full")


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OLD = module_from(ROOT / "scripts/diagnostics/dpl_gradient_evidence.py", "warmup_first_audit")
ROUND3 = module_from(ROOT / "scripts/diagnostics/dpl_third_round_diagnostics.py", "warmup_round3")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ids32() -> list[int]:
    return json.loads((EVIDENCE / "run_metadata.json").read_text())["basin_ids"]


def cma_data(ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = ROUND3.load_decision_module().cma_data(__import__("numpy").asarray(ids, dtype="int64"))
    return x.to(device=DEVICE, dtype=torch.float32), y.to(device=DEVICE, dtype=torch.float32)


def dpl_data(ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = OLD._load_forcing(__import__("numpy").asarray(ids, dtype="int64"), "1989-01-01")
    return x.to(device=DEVICE, dtype=torch.float32), y.to(device=DEVICE, dtype=torch.float32)


def chunks(n: int, size: int):
    return [slice(left, min(left + size, n)) for left in range(0, n, size)]


def best_theta(model_name: str, ids: list[int], dtype: torch.dtype) -> torch.Tensor:
    theta, _fitness, source = ROUND3.load_decision_module().archive_theta_and_fitness(model_name, __import__("numpy").asarray(ids, dtype="int64"))
    if theta is None:
        raise RuntimeError(f"missing CMA archive for {model_name}: {source}")
    return theta.to(device=DEVICE, dtype=dtype)


def cma_scores(model, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
    score, invalid = streaming_kge(q.unsqueeze(-1).unsqueeze(-1), y, eps=0.1)
    if bool(invalid.any()):
        raise FloatingPointError("CMA-aligned loss received invalid predictions")
    return score[:, 0, 0]


def aggregate_gradient(
    model_name: str,
    mode: str,
    x: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
    chunk_size: int = 8,
) -> tuple[float, torch.Tensor]:
    grad_chunks, score_chunks = [], []
    model = build_model(model_name, DEVICE, warm_up=1825, backend="compile", warmup_grad_mode=mode)
    for part in chunks(theta.shape[0], chunk_size):
        local = theta[part].detach().clone().requires_grad_(True)
        scores = cma_scores(model, x[:, part], y[:, part], local)
        loss = 1.0 - scores.mean()
        grad_chunks.append(torch.autograd.grad(loss, local)[0])
        score_chunks.append(scores.detach())
    del model
    torch.cuda.empty_cache()
    return float(1.0 - torch.cat(score_chunks).mean()), torch.cat(grad_chunks)


def aggregate_loss(
    model_name: str,
    mode: str,
    x: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
    chunk_size: int = 8,
) -> float:
    model = build_model(model_name, DEVICE, warm_up=1825, backend="compile", warmup_grad_mode=mode)
    values = []
    with torch.no_grad():
        for part in chunks(theta.shape[0], chunk_size):
            values.append(cma_scores(model, x[:, part], y[:, part], theta[part]).detach())
    del model
    torch.cuda.empty_cache()
    return float(1.0 - torch.cat(values).mean())


def t6() -> None:
    ids = ids32()
    x, y = cma_data(ids)
    line_rows, norm_rows, penman_rows = [], [], []
    for model_name in MODELS:
        theta_star = best_theta(model_name, ids, torch.float32)
        midpoint = torch.full_like(theta_star, 0.5)
        for mode in MODES:
            loss_star, grad_star = aggregate_gradient(model_name, mode, x, y, theta_star)
            loss_mid, grad_mid = aggregate_gradient(model_name, mode, x, y, midpoint)
            norm_star, norm_mid = grad_star.norm(), grad_mid.norm()
            direction = grad_star / norm_star.clamp_min(torch.finfo(grad_star.dtype).eps)
            deltas = []
            for h in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
                candidate = (theta_star - h * direction).clamp(0.0, 1.0)
                delta = aggregate_loss(model_name, mode, x, y, candidate) - loss_star
                deltas.append(delta)
                line_rows.append({"model": model_name, "mode": mode, "h": h, "loss_theta_star": loss_star, "loss_delta": delta, "has_descent_at_mode": any(item < 0.0 for item in deltas)})
            norm_rows.append({"model": model_name, "mode": mode, "loss_theta_star": loss_star, "loss_midpoint": loss_mid, "grad_norm_theta_star": float(norm_star), "grad_norm_midpoint": float(norm_mid), "star_to_midpoint_ratio": float(norm_star / norm_mid.clamp_min(torch.finfo(norm_mid.dtype).eps)), "has_descent": any(item < 0.0 for item in deltas)})

    # Penman GAM uses the dPL 365+365 slice, so FD and reverse-mode share the
    # same parameterized warm-up and can be compared mode by mode.
    xd, yd = dpl_data(ids)
    j = list(PARAM_INFO["penman"]).index("gam")
    base = torch.full((len(ids), len(PARAM_INFO["penman"])), 0.5, device=DEVICE, dtype=torch.float32)
    for mode in MODES:
        model = build_model("penman", DEVICE, warm_up=365, backend="compile", warmup_grad_mode=mode)
        for center in (0.1, 0.3, 0.5, 0.7, 0.9):
            h = 1e-4
            lo, hi = base.clone(), base.clone()
            lo[:, j], hi[:, j] = center - h, center + h
            with torch.no_grad():
                qlo = model({"x_phy": xd}, (None, lo.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
                qhi = model({"x_phy": xd}, (None, hi.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
                fd = (OLD._runner_kge(qhi, yd[365:])[0] - OLD._runner_kge(qlo, yd[365:])[0]) / (2.0 * h)
            theta = base.clone(); theta[:, j] = center; theta.requires_grad_(True)
            q = model({"x_phy": xd}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            loss = OLD._runner_kge(q, yd[365:])[0]
            loss.backward()
            autograd = theta.grad[:, j].sum()
            penman_rows.append({"mode": mode, "theta": center, "fd": float(fd), "autograd": float(autograd), "relative_difference": float((fd - autograd).abs() / torch.maximum(fd.abs(), autograd.abs()).clamp_min(1e-12))})
        del model
        torch.cuda.empty_cache()
    write_csv(OUT / "t6_line_search.csv", line_rows)
    write_csv(OUT / "t6_gradient_norm_ratio.csv", norm_rows)
    write_csv(OUT / "t6_penman_gam_fd_autograd.csv", penman_rows)


def t5() -> None:
    ids = ids32()[:4]
    x, _y = dpl_data(ids)
    rows = []
    for model_name in ("penman", "collie3", "flexi", "hbv96"):
        outputs = {}
        for mode in ("detach", "full", "truncate:90"):
            model = build_model(model_name, DEVICE, warm_up=365, backend="compile", warmup_grad_mode=mode)
            theta = torch.full((len(ids), NPARAM_INFO_36[model_name]), 0.5, device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                outputs[mode] = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"]
            del model
        rows.append({"model": model_name, "detach_full_bit_exact": bool(torch.equal(outputs["detach"], outputs["full"])), "detach_truncate90_bit_exact": bool(torch.equal(outputs["detach"], outputs["truncate:90"])), "max_abs_difference": float((outputs["detach"] - outputs["full"]).abs().max())})
    write_csv(OUT / "t5_forward_parity.csv", rows)


def dpl_loss_and_grad(model_name: str, mode: str, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    model = build_model(model_name, DEVICE, warm_up=365, backend="compile", warmup_grad_mode=mode)
    raw = theta.detach().clone().requires_grad_(True)
    inputs = {"x_phy": x}
    if model_name in {"mopex4", "mopex5"}:
        # This audit starts on 1989-01-01; both 365-day years in the slice
        # therefore use the exact 1..365 day-of-year sequence.
        inputs["doy"] = (torch.arange(x.shape[0], device=DEVICE, dtype=x.dtype) % 365 + 1).view(-1, 1).expand(-1, x.shape[1])
    q = model(inputs, (None, raw.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
    loss = OLD._runner_kge(q, y[365:])[0]
    grad = torch.autograd.grad(loss, raw)[0]
    del model
    torch.cuda.empty_cache()
    return grad


def t7() -> None:
    ids = ids32()
    x, y = dpl_data(ids)
    rows = []
    revived_params, revived_models = 0, set()
    for model_name in NPARAM_INFO_36:
        theta = torch.full((len(ids), NPARAM_INFO_36[model_name]), 0.5, device=DEVICE, dtype=torch.float32)
        detach_grad = dpl_loss_and_grad(model_name, "detach", x, y, theta)
        full_grad = dpl_loss_and_grad(model_name, "full", x, y, theta)
        for index, parameter in enumerate(PARAM_INFO[model_name]):
            detach_zero = float((detach_grad[:, index] == 0).float().mean())
            full_zero = float((full_grad[:, index] == 0).float().mean())
            revived = detach_zero == 1.0 and full_zero < 1.0
            if revived:
                revived_params += 1
                revived_models.add(model_name)
            rows.append({"model": model_name, "parameter": parameter, "detach_zero_gradient_basin_fraction": detach_zero, "full_zero_gradient_basin_fraction": full_zero, "revived": revived})
    write_csv(OUT / "t7_gradient_revival.csv", rows)
    (OUT / "t7_summary.json").write_text(json.dumps({"revived_parameters": revived_params, "models_with_revived_parameter": len(revived_models), "models": sorted(revived_models)}, indent=2) + "\n")


def t7_state_init() -> None:
    ids = ids32()
    x, y = dpl_data(ids)
    rows = []
    revived_params, revived_models = 0, set()
    for model_name in NPARAM_INFO_36:
        theta = torch.full((len(ids), NPARAM_INFO_36[model_name]), 0.5, device=DEVICE, dtype=torch.float32)
        gradients = {
            mode: dpl_loss_and_grad(model_name, mode, x, y, theta)
            for mode in ("detach", "state_init", "full")
        }
        for index, parameter in enumerate(PARAM_INFO[model_name]):
            zero = {mode: float((gradients[mode][:, index] == 0).float().mean()) for mode in gradients}
            revived = zero["detach"] == 1.0 and zero["state_init"] < 1.0
            if revived:
                revived_params += 1
                revived_models.add(model_name)
            rows.append({"model": model_name, "parameter": parameter, **{f"{mode}_zero_gradient_basin_fraction": value for mode, value in zero.items()}, "state_init_revived": revived})
    write_csv(OUT / "e3_state_init_revival.csv", rows)
    (OUT / "e3_state_init_summary.json").write_text(json.dumps({"revived_parameters": revived_params, "models_with_revived_parameter": len(revived_models), "models": sorted(revived_models)}, indent=2) + "\n")


def benchmark_one(model_name: str, mode: str, x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    torch.cuda.empty_cache()
    model = build_model(model_name, DEVICE, warm_up=365, backend="compile", warmup_grad_mode=mode)

    def one_step() -> None:
        theta = torch.full((x.shape[1], NPARAM_INFO_36[model_name]), 0.5, device=DEVICE, dtype=torch.float32, requires_grad=True)
        q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        loss = OLD._runner_kge(q, y[365:])[0]
        loss.backward()

    # Populate both forward and backward Inductor graphs before timing.
    one_step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start = time.perf_counter()
    one_step()
    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    del model
    torch.cuda.empty_cache()
    return {"model": model_name, "mode": mode, "seconds_per_step": elapsed, "peak_memory_mib": peak / 2**20}


def t8() -> None:
    ids = load_ids("data/531sub_id.txt")[:100]
    x, y = dpl_data([int(value) for value in ids])
    rows = []
    for model_name in ("hbv96", "collie3", "flexis"):
        for mode in (*MODES, "state_init"):
            try:
                rows.append(benchmark_one(model_name, mode, x, y))
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                rows.append({"model": model_name, "mode": mode, "seconds_per_step": None, "peak_memory_mib": None, "error": str(exc)})
    for model_name in {row["model"] for row in rows}:
        base = next((row for row in rows if row["model"] == model_name and row["mode"] == "detach"), None)
        for row in rows:
            if row["model"] == model_name and base and row.get("seconds_per_step") and base.get("seconds_per_step"):
                row["seconds_vs_detach"] = row["seconds_per_step"] / base["seconds_per_step"]
                row["memory_vs_detach"] = row["peak_memory_mib"] / base["peak_memory_mib"]
    write_csv(OUT / "t8_cost.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("t5", "t6", "t7", "t7_state_init", "t8", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = ("t5", "t6", "t7", "t7_state_init", "t8") if args.task == "all" else (args.task,)
    if "t5" in tasks:
        t5()
    if "t6" in tasks:
        t6()
    if "t7" in tasks:
        t7()
    if "t7_state_init" in tasks:
        t7_state_init()
    if "t8" in tasks:
        t8()


if __name__ == "__main__":
    main()
