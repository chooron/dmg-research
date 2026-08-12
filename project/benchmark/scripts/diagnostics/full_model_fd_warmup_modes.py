#!/usr/bin/env python3
"""Verify the full-model FD-vs-autograd discrepancy cause.

Hypothesis: the canonical warmup_grad_mode="detach" makes autograd exclude
the warmup-state trajectory (gradient flows only through scored steps),
while centered FD measures total sensitivity (incl. warmup states).  With
warmup_grad_mode="full", autograd should match FD.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model

DT = torch.float64
torch.manual_seed(0)
ids = [int(x) for x in load_ids("data/531sub_id.txt")]
attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore").to(DT)
b = torch.randperm(len(ids), device="cuda")[:16]
P = torch.clamp(torch.randn(730, 16, device="cuda", dtype=DT) * 4 + 4, min=0.0)
T = (5.0 + 15.0 * torch.sin(torch.arange(730, dtype=DT, device="cuda") / 365.0 * 2 * torch.pi)).view(730, 1).expand(730, 16)
PET = torch.clamp(2.0 + 3.0 * torch.randn(730, 16, device="cuda", dtype=DT), min=0.1)
doy = ((torch.arange(730, device="cuda") % 365) + 1).to(DT).view(730, 1, 1).expand(730, 16, 1)
x = torch.cat([P.unsqueeze(-1), T.unsqueeze(-1), PET.unsqueeze(-1), doy], dim=-1)


def run(mode: str):
    hydro = build_model("mopex4", "cuda", warm_up=365, backend="eager",
                        parameter_mapping="auto", warmup_grad_mode=mode)
    with torch.no_grad():
        theta0 = torch.full((16, 10), 0.5, dtype=DT, device="cuda")
        theta0[:, 4] = 0.55
        theta0[:, 5] = 0.45
        theta0[1, 4] = 0.3; theta0[2, 4] = 0.8; theta0[3, 5] = 0.2; theta0[4, 5] = 0.9

    def loss_at(tn):
        q = hydro({"x_phy": x}, (None, tn.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        return q.square().mean()

    theta = theta0.clone().requires_grad_(True)
    loss = loss_at(theta)
    loss.backward()
    g_auto = theta.grad

    eps = 1e-6
    out = {}
    for slot, name in [(4, "alpha"), (5, "is_time")]:
        th_p = theta0.clone(); th_p[:, slot] += eps
        th_m = theta0.clone(); th_m[:, slot] -= eps
        with torch.no_grad():
            lp = loss_at(th_p).item(); lm = loss_at(th_m).item()
        fd = (lp - lm) / (2 * eps)
        ag = float(g_auto[:, slot].mean())
        out[name] = {"autograd": ag, "fd": fd,
                     "abs_err": abs(ag - fd), "rel_err": abs(ag - fd) / (abs(fd) + 1e-12)}
        print(f"  [{mode}] {name:8s} autograd={ag:.6e} fd={fd:.6e} "
              f"abs_err={out[name]['abs_err']:.3e} rel_err={out[name]['rel_err']:.3e}")
    return out


print("warmup_grad_mode = detach (canonical)")
r_detach = run("detach")
print("warmup_grad_mode = full")
r_full = run("full")

import json
OUT = ROOT / "results/mopex4_formula_decouple_20260811"
(OUT / "full_model_fd_vs_autograd_warmup_modes.json").write_text(
    json.dumps({"detach": r_detach, "full": r_full}, indent=2) + "\n")
print("written:", OUT / "full_model_fd_vs_autograd_warmup_modes.json")
