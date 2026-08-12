"""Unit tests and regression benchmarks for saturation_2 fix."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

OUT_DIR = ROOT / "results/dpl_round13_20260805/vic_saturation_fix"

def old_saturation_2(
    S: torch.Tensor,
    Smax: torch.Tensor,
    p1: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    s_rel = S / (Smax + nearzero)
    term = torch.clamp(1.0 - s_rel, min=0.0, max=1.0)
    out_frac = 1.0 - (term + nearzero).pow(p1)
    return out_frac * incoming_flux

def new_saturation_2(
    S: torch.Tensor,
    Smax: torch.Tensor,
    p1: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    Smax_safe = torch.clamp(Smax, min=nearzero)
    s_rel = S / Smax_safe
    term = torch.clamp(
        1.0 - s_rel,
        min=nearzero,
        max=1.0,
    )
    out_frac = 1.0 - term.pow(p1)
    return out_frac * incoming_flux

def run_sweeps():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    s_ratios = [0.0, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.999999, 1.0, 1.0 + 1e-7, 1.0 + 1e-5]
    p1_vals = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.99, 1.0, 2.0, 5.0, 10.0]
    
    rows = []
    
    for s_ratio in s_ratios:
        for p1_val in p1_vals:
            Smax = torch.tensor([100.0], dtype=torch.float32, requires_grad=True)
            S = torch.tensor([s_ratio * 100.0], dtype=torch.float32, requires_grad=True)
            p1 = torch.tensor([p1_val], dtype=torch.float32, requires_grad=True)
            inc = torch.tensor([10.0], dtype=torch.float32, requires_grad=True)
            
            # Forward test
            out = new_saturation_2(S, Smax, p1, inc)
            out_finite = bool(torch.isfinite(out).all())
            out_non_negative = float(out.detach()) >= -1e-6
            
            # Backward test
            loss = out.sum()
            loss.backward()
            
            s_grad = S.grad
            smax_grad = Smax.grad
            p1_grad = p1.grad
            inc_grad = inc.grad
            
            grads_finite = (
                torch.isfinite(s_grad).all() and
                torch.isfinite(smax_grad).all() and
                torch.isfinite(p1_grad).all() and
                torch.isfinite(inc_grad).all()
            )
            
            max_grad_mag = max([
                float(s_grad.abs().max()),
                float(smax_grad.abs().max()),
                float(p1_grad.abs().max()),
                float(inc_grad.abs().max())
            ])
            
            # Old saturation_2 comparison
            Smax_old = torch.tensor([100.0], dtype=torch.float32)
            S_old = torch.tensor([s_ratio * 100.0], dtype=torch.float32)
            p1_old = torch.tensor([p1_val], dtype=torch.float32)
            inc_old = torch.tensor([10.0], dtype=torch.float32)
            out_old = old_saturation_2(S_old, Smax_old, p1_old, inc_old)
            
            abs_diff = float((out - out_old).abs())
            
            rows.append({
                "s_ratio": s_ratio,
                "p1": p1_val,
                "out_val": float(out.detach()),
                "out_finite": out_finite,
                "out_non_negative": out_non_negative,
                "grads_finite": bool(grads_finite),
                "max_grad_mag": max_grad_mag,
                "s_grad": float(s_grad),
                "smax_grad": float(smax_grad),
                "p1_grad": float(p1_grad),
                "inc_grad": float(inc_grad),
                "old_out_val": float(out_old),
                "abs_diff": abs_diff
            })
            
    df_sweep = pd.DataFrame(rows)
    df_sweep.to_csv(OUT_DIR / "saturation2_gradient_sweep.csv", index=False)
    print(f"Saved saturation2_gradient_sweep.csv with {len(rows)} entries. All forward finite: {df_sweep['out_finite'].all()}, All backward finite: {df_sweep['grads_finite'].all()}")

def run_forward_regression():
    torch.manual_seed(42)
    N = 10000
    
    # 1. Interior Region: S/Smax in [0.01, 0.95], p1 in [0.01, 10.0], Smax in [1.0, 2000.0]
    smax_int = torch.rand(N) * 1999.0 + 1.0
    s_rel_int = torch.rand(N) * 0.94 + 0.01
    s_int = s_rel_int * smax_int
    p1_int = torch.exp(torch.rand(N) * 5.0 - 2.0) # [0.13, 20.0]
    inc_int = torch.rand(N) * 50.0
    
    out_old_int = old_saturation_2(s_int, smax_int, p1_int, inc_int)
    out_new_int = new_saturation_2(s_int, smax_int, p1_int, inc_int)
    
    diff_int = (out_new_int - out_old_int).abs().numpy()
    rel_diff_int = (diff_int / (out_old_int.abs().numpy() + 1e-8))
    
    # 2. Numerical Guard Region: S/Smax in [0.999, 1.05]
    smax_guard = torch.rand(N) * 1999.0 + 1.0
    s_rel_guard = torch.rand(N) * 0.051 + 0.999
    s_guard = s_rel_guard * smax_guard
    p1_guard = torch.exp(torch.rand(N) * 5.0 - 2.0)
    inc_guard = torch.rand(N) * 50.0
    
    out_old_guard = old_saturation_2(s_guard, smax_guard, p1_guard, inc_guard)
    out_new_guard = new_saturation_2(s_guard, smax_guard, p1_guard, inc_guard)
    
    diff_guard = (out_new_guard - out_old_guard).abs().numpy()
    rel_diff_guard = (diff_guard / (out_old_guard.abs().numpy() + 1e-8))
    
    reg_rows = [
        {
            "region": "normal_interior",
            "count": N,
            "max_abs_diff": float(np.max(diff_int)),
            "median_abs_diff": float(np.median(diff_int)),
            "p99_abs_diff": float(np.percentile(diff_int, 99)),
            "max_rel_diff": float(np.max(rel_diff_int))
        },
        {
            "region": "numerical_guard",
            "count": N,
            "max_abs_diff": float(np.max(diff_guard)),
            "median_abs_diff": float(np.median(diff_guard)),
            "p99_abs_diff": float(np.percentile(diff_guard, 99)),
            "max_rel_diff": float(np.max(rel_diff_guard))
        }
    ]
    
    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(OUT_DIR / "saturation2_forward_regression.csv", index=False)
    print("Saved saturation2_forward_regression.csv:")
    print(df_reg.to_string())

def run_gradcheck():
    print("Running Float64 gradcheck on interior points...")
    Smax = torch.tensor([500.0], dtype=torch.float64, requires_grad=True)
    S = torch.tensor([250.0], dtype=torch.float64, requires_grad=True)
    p1 = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    inc = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)
    
    res = torch.autograd.gradcheck(new_saturation_2, (S, Smax, p1, inc), eps=1e-6, atol=1e-4)
    print(f"Float64 gradcheck interior test result: {res}")
    assert res, "Gradcheck failed!"

if __name__ == "__main__":
    run_sweeps()
    run_forward_regression()
    run_gradcheck()
