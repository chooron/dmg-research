#!/usr/bin/env python3
"""Full-model-path finite-difference vs autograd for restored MOPEX4.

Perturbs the normalized parameterizer output slot (alpha slot 4, is_time
slot 5) through the canonical model path (auto parameter mapping + step +
differentiable KGE) and compares centered finite differences to autograd.
float64 for precision; fixed 16-basin x 730d window.
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
DEVICE = "cuda"

torch.manual_seed(0)
ids = [int(x) for x in load_ids("data/531sub_id.txt")]
attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore").to(DT)

hydro = build_model("mopex4", "cuda", warm_up=365, backend="eager",
                    parameter_mapping="auto", warmup_grad_mode="detach")

# fixed 16 basins, 730 days, realistic forcing
b = torch.randperm(len(ids), device="cuda")[:16]
P = torch.clamp(torch.randn(730, 16, device="cuda", dtype=DT) * 4 + 4, min=0.0)
T = (5.0 + 15.0 * torch.sin(torch.arange(730, dtype=DT, device="cuda") / 365.0 * 2 * torch.pi)).view(730, 1).expand(730, 16)
PET = torch.clamp(2.0 + 3.0 * torch.randn(730, 16, device="cuda", dtype=DT), min=0.1)
doy = ((torch.arange(730, device="cuda") % 365) + 1).to(DT).view(730, 1, 1).expand(730, 16, 1)
x = torch.cat([P.unsqueeze(-1), T.unsqueeze(-1), PET.unsqueeze(-1), doy], dim=-1)

net = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36["mopex4"], hidden_dims=[256, 256], dropout=0.05).to("cuda")
net = net.to(DT)
with torch.no_grad():
    net.net[-1].weight.zero_(); net.net[-1].bias.zero_()

# theta at fixed midpoint-ish values (0.55 / 0.45 etc. to avoid exact-kink points)
with torch.no_grad():
    theta0 = torch.full((16, 10), 0.5, dtype=DT, device="cuda")
    theta0[:, 4] = 0.55   # alpha normalized
    theta0[:, 5] = 0.45   # is_time normalized
    # leave a couple basins at different alpha values
    theta0[1, 4] = 0.3; theta0[2, 4] = 0.8; theta0[3, 5] = 0.2; theta0[4, 5] = 0.9

def q_of(theta_norm):
    theta = theta_norm.unsqueeze(-1)
    return hydro({"x_phy": x}, (None, theta))["streamflow"].squeeze(-1).squeeze(-1)

def loss_at(theta_norm):
    return q_of(theta_norm).square().mean()

# autograd
theta = theta0.clone().requires_grad_(True)
loss = loss_at(theta)
loss.backward()
g_auto = theta.grad

# centered FD: perturb ONE basin's slot at a time; take the mean over basins
# (matching the autograd mean) so the two quantities have the same scaling.
eps = 1e-6
report = {}
for slot, name in [(4, "alpha"), (5, "is_time")]:
    fd_per_basin = []
    for k in range(theta0.shape[0]):
        th_p = theta0.clone(); th_p[k, slot] += eps
        th_m = theta0.clone(); th_m[k, slot] -= eps
        with torch.no_grad():
            qp = q_of(th_p); qm = q_of(th_m)
            lp = qp[:, k].square().mean().item()
            lm = qm[:, k].square().mean().item()
        fd_per_basin.append((lp - lm) / (2 * eps))
    # L_full = mean over (days x basins), so a basin-local FD must be scaled by
    # 1/n_basins to match the autograd mean over basins of dL_full/dtheta.
    fd = float(sum(fd_per_basin) / len(fd_per_basin) / theta0.shape[0])
    ag = float(g_auto[:, slot].mean())
    abs_err = abs(ag - fd)
    rel_err = abs_err / (abs(fd) + 1e-12)
    report[name] = {"autograd": ag, "fd": fd, "abs_err": abs_err, "rel_err": rel_err}
    print(f"  slot {slot} ({name}): autograd={ag:.6e} fd={fd:.6e} abs_err={abs_err:.3e} rel_err={rel_err:.3e}")

import json
OUT = ROOT / "results/mopex4_formula_decouple_20260811"
(OUT / "full_model_fd_vs_autograd.json").write_text(json.dumps(report, indent=2) + "\n")
print("written:", OUT / "full_model_fd_vs_autograd.json")
