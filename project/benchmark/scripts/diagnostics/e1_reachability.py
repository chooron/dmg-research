#!/usr/bin/env python3
"""Resumable CUDA E1 free-theta Adam reachability experiment."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.model_registry import NPARAM_INFO_36, build_model
from src.objective import streaming_kge

OUT = ROOT / "results/dpl_reachability_20260731"
MODELS = ("collie3", "newzealand1", "penman", "flexi", "flexis", "hbv96")
LRS = (3e-3, 1e-2, 3e-2)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None; sys.modules[name] = module; spec.loader.exec_module(module); return module


ROUND3 = load_module(ROOT / "scripts/diagnostics/dpl_third_round_diagnostics.py", "e1_round3")


def score(model, x, y, theta):
    q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
    kge, invalid = streaming_kge(q.unsqueeze(-1).unsqueeze(-1), y, eps=0.1)
    if bool(invalid.any()): raise FloatingPointError("invalid prediction")
    return kge[:, 0, 0]


def initializations(model: str, ids: list[int]):
    shape = (len(ids), NPARAM_INFO_36[model])
    base = torch.full(shape, .5, device="cuda", dtype=torch.float32)
    yield "midpoint", None, base
    try:
        star, _ = ROUND3.best_theta(model, torch.float32)
        all_theta, _source, _n = ROUND3.all_archive_theta(model, torch.float32)
    except RuntimeError:
        # HBV96 has no local CMA archive.  Its two CMA-dependent starts are
        # intentionally omitted rather than replaced with an invented value.
        star = None
    if star is not None:
        yield "cma_median", None, all_theta.median(0).values.expand_as(star).clone()
        yield "cma_perbasin", None, star
    for seed in (20260731, 20260732, 20260733):
        gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
        yield "random", seed, .1 + .8 * torch.rand(shape, device="cuda", generator=gen)


def completed(path: Path, steps: int):
    if not path.exists(): return set()
    with path.open() as f:
        return {(r["model"], r["init"], r["seed"], r["lr"])
                for r in csv.DictReader(f) if r["step"] == str(steps)}


def logged_steps(path: Path, key: tuple[str, str, str, str]) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {int(row["step"]) for row in csv.DictReader(handle)
                if (row["model"], row["init"], row["seed"], row["lr"]) == key}


def append(path: Path, rows):
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","init","seed","lr","step","mean_kge","median_kge","best_mean_kge","best_step"])
        if not exists: w.writeheader()
        w.writerows(rows)


def checkpoint_path(model: str, init: str, seed: int | None, lr: float) -> Path:
    seed_name = "none" if seed is None else str(seed)
    return OUT / "e1_checkpoints" / f"{model}__{init}__{seed_name}__{lr:.6g}.pt"


def run_model(
    model_name: str,
    steps: int,
    init_filter: set[str] | None = None,
    lr_filter: set[float] | None = None,
):
    ids = ROUND3.ids32().cpu().numpy().tolist(); x, y = ROUND3.cma_data(); x=x.float(); y=y.float()
    path=OUT/"e1_trajectories.csv"; done=completed(path, steps)
    for init, seed, theta0 in initializations(model_name, ids):
        if init_filter is not None and init not in init_filter:
            continue
        for lr in LRS:
            if lr_filter is not None and lr not in lr_filter:
                continue
            key=(model_name,init,"" if seed is None else str(seed),str(lr))
            if key in done: continue
            model=build_model(model_name,"cuda",warm_up=1825,backend="compile",warmup_grad_mode="detach")
            checkpoint = checkpoint_path(model_name, init, seed, lr)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint.exists():
                payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
                theta = torch.nn.Parameter(payload["theta"].to(device="cuda", dtype=torch.float32))
                opt = torch.optim.Adam([theta], lr=lr)
                opt.load_state_dict(payload["optimizer"])
                start_step = int(payload["next_step"])
                best, best_step = float(payload["best_mean_kge"]), int(payload["best_step"])
            else:
                theta=torch.nn.Parameter(theta0.detach().clone()); opt=torch.optim.Adam([theta],lr=lr)
                start_step, best, best_step = 0, -float("inf"), 0
            prior_logged = logged_steps(path, key)
            for step in range(start_step, steps+1):
                opt.zero_grad(set_to_none=True); kge=score(model,x,y,theta); loss=1-kge.mean()
                value=float(kge.mean().detach())
                if value>best: best,best_step=value,step
                if step < steps:
                    loss.backward(); opt.step()
                    with torch.no_grad(): theta.clamp_(1e-7,1-1e-7)
                if (step % 10 == 0 or step == steps) and step not in prior_logged:
                    append(path, [{"model":model_name,"init":init,"seed":"" if seed is None else seed,"lr":lr,"step":step,"mean_kge":value,"median_kge":float(kge.median().detach()),"best_mean_kge":best,"best_step":best_step}])
                # State is deliberately saved every step: on interruption only
                # the current CUDA graph is lost, never an optimizer update.
                torch.save({"next_step": step + 1, "theta": theta.detach(), "optimizer": opt.state_dict(), "best_mean_kge": best, "best_step": best_step}, checkpoint)
            checkpoint.unlink(missing_ok=True)
            del model,theta,opt; torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default=None)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--init", action="append", choices=("midpoint", "cma_median", "cma_perbasin", "random"))
    p.add_argument("--lr", action="append", type=float, choices=LRS)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True,exist_ok=True)
    init_filter = set(a.init) if a.init else None
    lr_filter = set(a.lr) if a.lr else None
    for name in (a.model,) if a.model else MODELS:
        run_model(name, a.steps, init_filter, lr_filter)

if __name__ == "__main__": main()
