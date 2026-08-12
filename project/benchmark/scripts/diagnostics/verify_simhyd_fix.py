"""Lightweight verification script for SIMHYD numerical stabilization fix."""
import torch
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import dmotpy.models.core.simhyd as m1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running lightweight SIMHYD verification on {device}...", flush=True)

    # Test cases respecting SIMHYD physical parameter bounds (smsc >= 1.0)
    cases = [
        ("Normal Interior", 1.5, 200.0, 5.0, 500.0, 0.2, 0.1, 0.05),
        ("Zero Coeff Boundary", 1.5, 0.0, 5.0, 500.0, 0.2, 0.1, 0.05),
        ("Zero INSC Boundary", 0.0, 200.0, 5.0, 500.0, 0.2, 0.1, 0.05),
        ("Min SMSC Boundary (1.0mm)", 1.5, 200.0, 5.0, 1.0, 0.2, 0.1, 0.05),
        ("High Infiltration (600mm)", 1.5, 600.0, 15.0, 2000.0, 1.0, 1.0, 1.0),
    ]

    batch = 4
    nsteps = 365
    records = []

    for name, insc_val, coeff_val, sq_val, smsc_val, sub_val, crak_val, k_val in cases:
        P = torch.rand(batch, nsteps, device=device) * 10.0
        P[:, 50:150] = 0.0
        T = torch.rand(batch, nsteps, device=device) * 25.0
        PET = torch.rand(batch, nsteps, device=device) * 5.0

        insc = torch.tensor([insc_val] * batch, device=device, requires_grad=True)
        coeff = torch.tensor([coeff_val] * batch, device=device, requires_grad=True)
        sq = torch.tensor([sq_val] * batch, device=device, requires_grad=True)
        smsc = torch.tensor([smsc_val] * batch, device=device, requires_grad=True)
        sub = torch.tensor([sub_val] * batch, device=device, requires_grad=True)
        crak = torch.tensor([crak_val] * batch, device=device, requires_grad=True)
        k = torch.tensor([k_val] * batch, device=device, requires_grad=True)

        soil, gw = m1.create_initial_state(batch, 1, device)

        soil_t, gw_t = soil, gw
        q_list = []
        for t in range(nsteps):
            q_t, _, soil_t, gw_t = m1.simhyd_step(
                P[:, t:t+1], T[:, t:t+1], PET[:, t:t+1],
                insc.unsqueeze(-1), coeff.unsqueeze(-1), sq.unsqueeze(-1),
                smsc.unsqueeze(-1), sub.unsqueeze(-1), crak.unsqueeze(-1),
                k.unsqueeze(-1), soil_t, gw_t
            )
            q_list.append(q_t)

        q_tensor = torch.stack(q_list, dim=1).squeeze(-1).squeeze(-1)
        loss = q_tensor.sum()
        loss.backward()

        grads = [insc.grad, coeff.grad, sq.grad, smsc.grad, sub.grad, crak.grad, k.grad]
        all_finite = all(g is not None and torch.isfinite(g).all() for g in grads)
        max_grad = max(g.abs().max().item() for g in grads if g is not None)

        records.append({
            "case": name,
            "forward_finite": bool(torch.isfinite(q_tensor).all()),
            "backward_finite": all_finite,
            "max_grad_magnitude": max_grad
        })
        print(f"Case '{name}': Forward Finite={torch.isfinite(q_tensor).all().item()}, Backward Finite={all_finite}, Max Grad={max_grad:.4e}")

    df = pd.DataFrame(records)
    out_dir = ROOT / "results/dpl_round13_20260805/simhyd_fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "simhyd_boundary_verification.csv", index=False)
    print("Verification completed cleanly. Saved simhyd_boundary_verification.csv")

if __name__ == "__main__":
    main()
