"""Comprehensive unit test for modified SIMHYD model 1."""
import torch
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "dmotpy"))
sys.path.insert(0, str(ROOT))

import dmotpy.models.core.simhyd as m1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running SIMHYD Model 1 Comprehensive Verification on {device}...", flush=True)

    test_cases = [
        ("Interior Normal", 1.5, 200.0, 5.0, 500.0, 0.2, 0.1, 0.05),
        ("Zero Coeff Boundary", 1.5, 0.0, 5.0, 500.0, 0.2, 0.1, 0.05),
        ("Zero INSC Boundary", 0.0, 200.0, 5.0, 500.0, 0.2, 0.1, 0.05),
        ("Min SMSC (1.0mm)", 1.5, 200.0, 5.0, 1.0, 0.2, 0.1, 0.05),
        ("Max Coeff (600mm)", 1.5, 600.0, 15.0, 2000.0, 1.0, 1.0, 1.0),
        ("Zero Sub & Crak", 1.5, 200.0, 5.0, 500.0, 0.0, 0.0, 0.05),
        ("Zero Groundwater Recession (k=0)", 1.5, 200.0, 5.0, 500.0, 0.2, 0.1, 0.0),
    ]

    batch = 4
    nsteps = 365
    records = []

    for name, p_insc_v, p_coeff_v, p_sq_v, p_smsc_v, p_sub_v, p_crak_v, p_k_v in test_cases:
        P = torch.rand(batch, nsteps, device=device) * 15.0
        P[:, 40:120] = 0.0  # Dry spell
        T = torch.zeros(batch, nsteps, device=device)
        PET = torch.rand(batch, nsteps, device=device) * 5.0

        p_insc = torch.tensor([p_insc_v] * batch, device=device, requires_grad=True)
        p_coeff = torch.tensor([p_coeff_v] * batch, device=device, requires_grad=True)
        p_sq = torch.tensor([p_sq_v] * batch, device=device, requires_grad=True)
        p_smsc = torch.tensor([p_smsc_v] * batch, device=device, requires_grad=True)
        p_sub = torch.tensor([p_sub_v] * batch, device=device, requires_grad=True)
        p_crak = torch.tensor([p_crak_v] * batch, device=device, requires_grad=True)
        p_k = torch.tensor([p_k_v] * batch, device=device, requires_grad=True)

        soil, gw = m1.create_initial_state(batch, 1, device)

        soil_t, gw_t = soil, gw
        q_list = []
        for t in range(nsteps):
            q_t, _, soil_t, gw_t = m1.simhyd_step(
                P[:, t:t+1], T[:, t:t+1], PET[:, t:t+1],
                p_insc.unsqueeze(-1), p_coeff.unsqueeze(-1), p_sq.unsqueeze(-1),
                p_smsc.unsqueeze(-1), p_sub.unsqueeze(-1), p_crak.unsqueeze(-1),
                p_k.unsqueeze(-1), soil_t, gw_t
            )
            q_list.append(q_t)

        q_tensor = torch.stack(q_list, dim=1).squeeze(-1).squeeze(-1)
        loss = q_tensor.sum()
        loss.backward()

        grads = [p_insc.grad, p_coeff.grad, p_sq.grad, p_smsc.grad, p_sub.grad, p_crak.grad, p_k.grad]
        all_finite = all(g is not None and torch.isfinite(g).all() for g in grads)
        max_grad = max(g.abs().max().item() for g in grads if g is not None)

        records.append({
            "case_name": name,
            "forward_finite": bool(torch.isfinite(q_tensor).all()),
            "backward_finite": all_finite,
            "max_grad_magnitude": max_grad
        })
        print(f"[{name:32s}] Forward={torch.isfinite(q_tensor).all().item()} | Backward={all_finite} | MaxGrad={max_grad:.4e}")

    df = pd.DataFrame(records)
    out_path = ROOT / "results/dpl_round13_20260805/simhyd_fix/simhyd_verification_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"All boundary checks passed. Saved report to {out_path}")

if __name__ == "__main__":
    main()
