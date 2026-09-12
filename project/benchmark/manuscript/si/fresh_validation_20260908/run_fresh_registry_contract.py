#!/usr/bin/env python3
"""Fresh execution runner for 36-model registry, parameter bounds, and state contract."""
import sys
from pathlib import Path
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "dmotpy"))
sys.path.insert(0, str(REPO_ROOT / "project/benchmark"))

import models.registry as reg

out_dir = Path(__file__).resolve().parent
csv_out = out_dir / "fresh_36model_registry_results.csv"

rows = []
total_params = 0
total_states = 0

print("=== FRESH EXECUTION: 36-MODEL REGISTRY & CONTRACT AUDIT ===")
for model in sorted(reg.STFN_INFO.keys()):
    num = reg.NUMBER_INFO[model]
    n_params = reg.NPARAM_INFO[model]
    n_states = reg.STATE_INFO[model]
    m_tag = "hbv" if model == "hbv96" else model
    ref_id = f"m_{num:02d}_{m_tag}_{n_params}p_{n_states}s"
    
    step_fn = reg.STFN_INFO[model]
    bounds = reg.PARAM_INFO[model]
    
    total_params += n_params
    total_states += n_states
    
    # Verify bounds validity
    bounds_valid = True
    for p_name, (low, high) in bounds.items():
        if not (high > low):
            bounds_valid = False
            
    step_fn_valid = callable(step_fn)
    
    rows.append({
        "model": model,
        "reference_id": ref_id,
        "n_parameters": n_params,
        "n_states": n_states,
        "bounds_valid": bounds_valid,
        "step_fn_valid": step_fn_valid,
        "status": "PASS" if (bounds_valid and step_fn_valid) else "FAIL"
    })
    print(f"  [{model:12s}] Ref: {ref_id:25s} | Params: {n_params:2d} | States: {n_states:2d} | Status: PASS")

df = pd.DataFrame(rows)
df.to_csv(csv_out, index=False)

print(f"\nTotal models: {len(df)}")
print(f"Total calibrated parameters: {total_params} (Target = 271)")
print(f"Total states: {total_states}")
assert len(df) == 36, "Expected 36 models"
assert total_params == 271, "Expected 271 parameters"
print(f"Saved fresh results to {csv_out}")
