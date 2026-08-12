#!/usr/bin/env python3
"""Debug S_eff / c gradient flow through the canonical mopex4 path."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from dmotpy.models.flux.mopex import mopex_interception_4_liu2

# 1) direct kernel gradient
S = torch.tensor([2.5], requires_grad=True)
c = torch.tensor([0.54], requires_grad=True)
Pr = torch.tensor([8.0])
I = mopex_interception_4_liu2(Pr, S, c)
I.backward()
print("kernel dI/dS_eff:", S.grad.item(), " dI/dc:", c.grad.item())

# 2) full model path: numeric vs autograd on S_eff / c slots
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.model_registry import build_model, NPARAM_INFO_36
from src.data_selection import load_ids

ids = [int(x) for x in load_ids("data/531sub_id.txt")]
attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
hydro = build_model("mopex4", torch.device("cuda"), warm_up=365, backend="eager",
                    parameter_mapping="auto", warmup_grad_mode="detach")
net = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36["mopex4"], hidden_dims=[256, 256], dropout=0.05).to("cuda")
with torch.no_grad():
    net.net[-1].weight.zero_(); net.net[-1].bias.zero_()

torch.manual_seed(0)
b = torch.arange(8, device="cuda")
P = torch.clamp(torch.randn(400, 8, device="cuda") * 4 + 4, min=0.0)
T = 5.0 + 15.0 * torch.sin(2 * torch.pi * torch.arange(400, device="cuda") / 365.0).view(400, 1)
PET = (2.0 + 4.0 * torch.sin(2 * torch.pi * torch.arange(400, device="cuda") / 365.0 + 0.3)).view(400, 1).expand(400, 8).clamp(min=0.5)
x = torch.stack([P, T.expand(400, 8), PET], dim=-1)
doy = (torch.arange(400, device="cuda") % 365).float().view(400, 1, 1).expand(400, 8, 1)
x4 = torch.cat([x, doy], dim=-1)

def forward(theta):
    return hydro({"x_phy": x4}, (None, theta))["streamflow"].squeeze(-1).squeeze(-1)

theta = net(attrs[b]).unsqueeze(-1).detach().clone().requires_grad_(True)
q = forward(theta)
loss = q.square().mean()
loss.backward()
g = theta.grad.squeeze(-1)
print("autograd theta.grad mean abs per slot:", [f"{v:.3e}" for v in g.abs().mean(0).tolist()])

# numeric: perturb slot 4 (S_eff) and slot 5 (c)
eps = 1e-3
for slot in (4, 5):
    th_p = theta.detach().clone(); th_p[:, slot] += eps
    th_m = theta.detach().clone(); th_m[:, slot] -= eps
    with torch.no_grad():
        lp = forward(th_p).square().mean().item()
        lm = forward(th_m).square().mean().item()
    print(f"slot {slot} numeric dLoss/dtheta = {(lp - lm) / (2*eps):.6e}  autograd = {g[:, slot].mean().item():.6e}")

# 3) descaled param path: check _descale_params keeps grad
pd = hydro._descale_params(theta.squeeze(-1))
print("descaled S_eff requires_grad:", pd["S_eff"].requires_grad, " c:", pd["c"].requires_grad)

# 4) check the actual interception input: flux_pr per day on basin 0
# re-run step by hand with the same theta
import torch.nn.functional as F
from dmotpy.models.core.mopex4 import mopex4_step
params = {name: pd[name][0] for name in hydro.phy_param_names}
Sn = torch.zeros(1, device="cuda") + 1e-6
S1 = torch.zeros(1, device="cuda") + 1e-6
S2 = torch.zeros(1, device="cuda") + 1e-6
Sc1 = torch.zeros(1, device="cuda") + 1e-6
Sc2 = torch.zeros(1, device="cuda") + 1e-6
Is = []
for t in range(400):
    p, tmp, pet = x4[t, 0, 0].item(), x4[t, 0, 1].item(), x4[t, 0, 2].item()
    out = mopex4_step(
        torch.tensor([p], device="cuda"), torch.tensor([tmp], device="cuda"), torch.tensor([pet], device="cuda"),
        **{k: torch.tensor([v], device="cuda") for k, v in params.items()},
        S1=S1, S2=S2, Sc1=Sc1, Sc2=Sc2, Sn=Sn, delta_t=1.0, nearzero=1e-6, doy=torch.tensor([t % 365], device="cuda"),
    )
    S1, S2, Sc1, Sc2, Sn = out[2], out[3], out[4], out[5], out[6]
print("manual step OK; final Q:", out[0].item(), "ET:", out[1].item())
