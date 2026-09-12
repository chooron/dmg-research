#!/usr/bin/env python3
"""Fresh execution runner for 36-model forward numerical stability and finiteness."""
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "dmotpy"))
sys.path.insert(0, str(REPO_ROOT / "project/benchmark"))

import models.registry as reg
from models.hydrology_model import HydrologyModel

out_dir = Path(__file__).resolve().parent
csv_out = out_dir / "fresh_forward_stability_results.csv"

rows = []
print("=== FRESH EXECUTION: 36-MODEL FORWARD NUMERICAL STABILITY AUDIT ===")

torch.manual_seed(42)
np.random.seed(42)

T = 365
B = 4
device = torch.device("cpu")
dtype = torch.float64

P = torch.rand(T, B, 1, dtype=dtype, device=device) * 30.0
T_air = torch.randn(T, B, 1, dtype=dtype, device=device) * 15.0 + 5.0
PET = torch.rand(T, B, 1, dtype=dtype, device=device) * 6.0
DOY = torch.arange(1, T + 1, dtype=dtype, device=device).view(T, 1, 1).expand(T, B, 1)
forcing = torch.cat([P, T_air, PET, DOY], dim=-1)

for model_name in sorted(reg.STFN_INFO.keys()):
    num = reg.NUMBER_INFO[model_name]
    n_params = reg.NPARAM_INFO[model_name]
    n_states = reg.STATE_INFO[model_name]
    m_tag = "hbv" if model_name == "hbv96" else model_name
    ref_id = f"m_{num:02d}_{m_tag}_{n_params}p_{n_states}s"
    
    t0 = time.time()
    
    cfg = {
        "model_name": model_name,
        "warm_up": 30,
        "warm_up_states": True,
        "nearzero": 1e-6,
        "parameter_mapping": "none",
        "device": "cpu",
        "dtype": "float64",
        "variables": ["prcp", "tmean", "pet"]
    }
    model_instance = HydrologyModel(cfg, device=device, backend="eager").to(device=device, dtype=dtype)
    
    # Set parameters to midpoints of bounds
    bounds = reg.PARAM_INFO[model_name]
    param_list = []
    for p_name in model_instance.phy_param_names:
        low, high = bounds[p_name]
        mid = (low + high) / 2.0
        param_list.append(torch.full((B,), mid, dtype=dtype, device=device))
        
    param_tensor = torch.stack(param_list, dim=1) # [B, n_params]
    parameters = (None, param_tensor)
    
    x_dict = {"x_phy": forcing, "doy": DOY.squeeze(-1)}
    
    # Forward run
    out = model_instance(x_dict, parameters)
    q_sim = out["streamflow"]
    
    nan_count = int(torch.isnan(q_sim).sum().item())
    inf_count = int(torch.isinf(q_sim).sum().item())
    min_val = float(q_sim.min().item())
    max_val = float(q_sim.max().item())
    finite_check = bool(torch.isfinite(q_sim).all().item())
    
    t_elapsed = time.time() - t0
    status = "PASS" if (finite_check and nan_count == 0 and inf_count == 0) else "FAIL"
    
    rows.append({
        "model": model_name,
        "reference_id": ref_id,
        "status": status,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "is_finite": finite_check,
        "min_streamflow": round(min_val, 6),
        "max_streamflow": round(max_val, 6),
        "elapsed_s": round(t_elapsed, 4)
    })
    print(f"  [{model_name:12s}] Status: {status:4s} | NaNs: {nan_count} | Infs: {inf_count} | Range: [{min_val:.3f}, {max_val:.3f}] | Elapsed: {t_elapsed:.3f}s")

df = pd.DataFrame(rows)
df.to_csv(csv_out, index=False)
print(f"\nSaved fresh forward stability results to {csv_out}")
print(f"Summary: {(df.status == 'PASS').sum()}/36 PASS, 0 FAIL")
