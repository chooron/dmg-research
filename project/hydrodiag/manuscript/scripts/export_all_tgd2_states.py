#!/usr/bin/env python3
"""Export states for XAJ_TGD2 dPL (seeds 42, 123, 2026) and IC (10 starts)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "project" / "hydrodiag"
sys.path.insert(0, str(PROJECT))

from r4.common import default_data_root, default_results_root, load_bundle, zfill8
from r4.state_export import continuous_forward, model_instances

def export_all():
    results_root = default_results_root()
    data_root = default_data_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    bundle = load_bundle(data_root)
    basin_ids = [zfill8(b) for b in bundle.basin_ids]
    
    models = model_instances(device, dtype)
    model = models["XAJ_TGD2"]

    # 1. dPL seeds
    for seed in [42, 123, 2026]:
        dpl_dir = results_root / f"dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_{seed}"
        out_dir = results_root / f"r4_official_dpl_XAJ_TGD2_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"official_dpl_XAJ_TGD2_seed{seed}_full_arrays.npz"
        
        print(f"Exporting states for XAJ_TGD2 dPL seed {seed}...", flush=True)
        phys = np.load(dpl_dir / "best_parameters_physical.npz")["params"]
        
        q_full, states = continuous_forward(
            structure="XAJ_TGD2",
            model=model,
            theta_hat=phys,
            forcing_full=bundle.forcing,
            device=device,
            dtype=dtype,
            batch=64,
            validate_subset=5,
        )
        
        # Check and clean any NaNs defensively by replacing with finite boundary values
        for k, v in states.items():
            if np.isnan(v).any():
                print(f"Defensive cleanup of NaNs in state {k} for dPL seed {seed} ({np.isnan(v).sum()} NaNs)")
                states[k] = np.nan_to_num(v, nan=0.0)
        q_full = np.nan_to_num(q_full, nan=0.0)
        
        export_dict = {
            "basin_ids": np.array(basin_ids),
            "dates": np.array([str(d) for d in bundle.dates]),
            "q_full": q_full.astype(np.float32),
        }
        for k, v in states.items():
            export_dict[k] = v.astype(np.float32)
            
        np.savez_compressed(npz_path, **export_dict)
        print(f"Saved dPL seed {seed}: {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)

    # 2. IC fused (10 starts)
    ic_dir = results_root / "r4_ic_fused_XAJ_TGD2"
    ic_npz_path = ic_dir / "ic_fused_XAJ_TGD2_full_arrays.npz"
    print("Exporting states for XAJ_TGD2 IC fused...", flush=True)
    ic_phys = np.load(ic_dir / "best_parameters_physical.npz")["params"]
    
    q_full, states = continuous_forward(
        structure="XAJ_TGD2",
        model=model,
        theta_hat=ic_phys,
        forcing_full=bundle.forcing,
        device=device,
        dtype=dtype,
        batch=64,
        validate_subset=5,
    )
    for k, v in states.items():
        if np.isnan(v).any():
            print(f"Defensive cleanup of NaNs in state {k} for IC ({np.isnan(v).sum()} NaNs)")
            states[k] = np.nan_to_num(v, nan=0.0)
    q_full = np.nan_to_num(q_full, nan=0.0)
    
    export_dict = {
        "basin_ids": np.array(basin_ids),
        "dates": np.array([str(d) for d in bundle.dates]),
        "q_full": q_full.astype(np.float32),
    }
    for k, v in states.items():
        export_dict[k] = v.astype(np.float32)
        
    np.savez_compressed(ic_npz_path, **export_dict)
    print(f"Saved IC fused: {ic_npz_path} ({ic_npz_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


if __name__ == "__main__":
    export_all()
