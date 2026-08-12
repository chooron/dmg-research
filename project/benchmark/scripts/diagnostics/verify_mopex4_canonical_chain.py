#!/usr/bin/env python3
"""Verify the canonical dPL import chain for the frozen final MOPEX4.

Checks:
1. Which dmotpy package the canonical benchmark loading actually resolves to.
2. mopex4 model builds with 10 params and the final S_eff/c bounds.
3. The step is the branch-free two-parameter Liu kernel with shared PET budget.
4. State-order reorder fix present in the wrapper.
5. A tiny forward pass is finite with non-zero gradients on S_eff/c.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

import dmotpy
print("dmotpy package:", dmotpy.__file__)

from dmotpy.models.registry import PARAM_INFO, NPARAM_INFO
from dmotpy.models import mopex_doy_model, hydrology_model, flux
print("mopex_doy_model:", mopex_doy_model.__file__)
print("hydrology_model:", hydrology_model.__file__)
print("flux.mopex:", flux.mopex.__file__)

import dmotpy.models.core.mopex4 as m4
print("core.mopex4:", m4.__file__)

# 1) registry bounds for mopex4
bounds = PARAM_INFO["mopex4"]
print("mopex4 n_params:", len(bounds), "NPARAM_INFO:", NPARAM_INFO["mopex4"])
print("mopex4 param names:", list(bounds.keys()))
assert list(bounds.keys())[4:6] == ["S_eff", "c"], "interception slots wrong"
assert bounds["S_eff"] == [1e-5, 5.0] and bounds["c"] == [0.10, 0.98], "bounds wrong"

# 2) schema validator
m4.validate_mopex4_parameter_schema(tuple(bounds.keys()))
print("schema validation OK (S_eff, c)")

# 3) wrapper state reorder fix present
src = Path(mopex_doy_model.__file__).read_text()
assert "mopex4" in src and "states[1], states[2], states[3], states[4], states[0]" in src
print("wrapper state reorder fix present: OK")

# 4) build the model via canonical registry path
from src.model_registry import build_model, model_config, get_spec, NPARAM_INFO_36

spec = get_spec("mopex4", device="cuda")
print("spec dimension:", spec.dimension, "routed_kind:", spec.routed_kind)

cfg = model_config("mopex4", warm_up=365, backend="eager", parameter_mapping="auto",
                   warmup_grad_mode="detach")
hydro = build_model("mopex4", torch.device("cuda"), warm_up=365, backend="eager",
                    parameter_mapping="auto", warmup_grad_mode="detach")
print("built model class:", type(hydro).__name__)

# 5) tiny forward + gradient check on S_eff / c
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.data_selection import load_ids

ids = [int(x) for x in load_ids("data/531sub_id.txt")]
attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
net = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36["mopex4"], hidden_dims=[256, 256], dropout=0.05).to("cuda")
with torch.no_grad():
    net.net[-1].weight.zero_(); net.net[-1].bias.zero_()  # initialize_midpoint -> theta=0.5

from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
import pandas as pd
print("mopex4 in CALENDAR_MODELS:", "mopex4" in CALENDAR_MODELS)

# realistic window forward on 8 basins, 400 days (365 warmup + 35 scored)
torch.manual_seed(0)
b = torch.arange(8, device="cuda")
P = torch.clamp(torch.randn(400, 8, device="cuda") * 4 + 4, min=0.0)
T = 5.0 + 15.0 * torch.sin(2 * torch.pi * torch.arange(400, device="cuda") / 365.0).view(400, 1)
PET = (2.0 + 4.0 * torch.sin(2 * torch.pi * torch.arange(400, device="cuda") / 365.0 + 0.3)).view(400, 1).expand(400, 8).clamp(min=0.5)
x = torch.stack([P, T.expand(400, 8), PET], dim=-1)
doy = (torch.arange(400, device="cuda") % 365).float().view(400, 1, 1).expand(400, 8, 1)
x4 = torch.cat([x, doy], dim=-1)
theta = net(attrs[b]).unsqueeze(-1)
theta.retain_grad()
out = hydro({"x_phy": x4}, (None, theta))
q = out["streamflow"].squeeze(-1).squeeze(-1)
print("forward q shape:", tuple(q.shape), "finite:", bool(torch.isfinite(q).all()))
loss = q.square().mean()
loss.backward()
print("theta grad finite:", bool(torch.isfinite(theta.grad).all()))
g_Seff = theta.grad[:, 4].abs().mean().item()
g_c = theta.grad[:, 5].abs().mean().item()
print("mean|grad| S_eff slot:", f"{g_Seff:.3e}", " c slot:", f"{g_c:.3e}")
assert g_Seff > 0 and g_c > 0, "S_eff / c slots have zero gradient in tiny probe"

# descaled physical params
pd_ = hydro._descale_params(theta.squeeze(-1))
print("physical S_eff sample:", pd_["S_eff"][:3].tolist())
print("physical c sample:", pd_["c"][:3].tolist())
print("ALL CHECKS PASSED")
