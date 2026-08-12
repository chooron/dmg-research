#!/usr/bin/env python3
"""Fidelity-fixed forward experiment: production MOPEX4 vs MARRMoT-faithful-flux
MOPEX4 at fixed parameters (IC and continuation), representative basins.

Isolates the *formula* mismatch effect (same sequential discretization in both).
Outputs fidelity_forward_comparison.csv and feeds the final report."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

BENCHMARK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCHMARK), str(BENCHMARK.parents[1]), str(BENCHMARK / "src"),
                str(Path(__file__).resolve().parent)]
OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "formulation_degeneracy_audit"
OUT.mkdir(parents=True, exist_ok=True)

import audit_mopex45_sequential_discretization as A  # noqa: E402
from mopex45_faithful_steps import mopex4_step_faithful  # noqa: E402
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import compute_differentiable_kge  # noqa: E402

DEVICE = A.DEVICE


def rollout_q(step_fn, x, theta_norm, basins, start, warmup=365, scored=365):
    """Reuse A.rollout_m4's structure but with the given step_fn (faithful
    returns 7-tuple without fluxes dict)."""
    b = list(basins)
    P = x[start:start + warmup + scored, b, 0]
    T = x[start:start + warmup + scored, b, 1]
    PET = x[start:start + warmup + scored, b, 2]
    doy = x[start:start + warmup + scored, b, 3]
    theta = A.norm_to_phys(theta_norm.clone(), A.M4_BOUNDS)
    Sn, S1, S2, Sc1, Sc2 = A._init_states(len(b))
    qs, ets = [], []
    from dmotpy.models.flux.mopex import mopex_training_context
    with mopex_training_context(lambda_i=1.0, lambda_p=1.0, beta=50.0):
        for t in range(warmup):
            with torch.no_grad():
                out = step_fn(P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn, doy=doy[t], nearzero=1e-6)
                S1, S2, Sc1, Sc2, Sn = [v.detach() for v in out[2:7]]
        for t in range(warmup, warmup + scored):
            out = step_fn(P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn, doy=doy[t], nearzero=1e-6)
            qs.append(out[0]); ets.append(out[1])
            S1, S2, Sc1, Sc2, Sn = out[2:7]
    return torch.stack(qs), torch.stack(ets)


def main():
    ids, x, y = A.load_data()
    theta_ic4 = A.ic_theta("mopex4", ids)
    theta_cont4 = A.continuation_theta(ids)
    picks = A.select_representative_basins(ids, x, y, theta_ic4, theta_cont4, n_each=4, start=1825)
    basins = [p["basin_idx"] for p in picks]
    start = 1825

    rows = []
    for pset, theta in (("IC", theta_ic4[basins]), ("continuation", theta_cont4[basins])):
        q_prod, et_prod = rollout_q(A.mopex4_step_diag, x, theta, basins, start)
        q_faith, et_faith = rollout_q(mopex4_step_faithful, x, theta, basins, start)
        for j, b in enumerate(basins):
            yw = torch.stack([y[start + 365 + t, b] for t in range(365)])
            with torch.no_grad():
                _, kge_prod = compute_differentiable_kge(q_prod[:, j:j + 1], yw.unsqueeze(-1), warmup_days=0)
                _, kge_faith = compute_differentiable_kge(q_faith[:, j:j + 1], yw.unsqueeze(-1), warmup_days=0)
            d = q_prod[:, j] - q_faith[:, j]
            rows.append({
                "param_set": pset, "basin_idx": int(b), "basin_id": int(ids[b]),
                "kge_production": float(kge_prod[0]), "kge_faithful": float(kge_faith[0]),
                "kge_delta_faithful_minus_production": float(kge_faith[0] - kge_prod[0]),
                "q_rmse": float(d.square().mean().sqrt()),
                "q_corr": float(torch.corrcoef(torch.stack([q_prod[:, j], q_faith[:, j]]))[0, 1]),
                "q_max_abs": float(d.abs().max()),
                "q_vol_diff": float((q_prod[:, j].sum() - q_faith[:, j].sum()).abs()),
                "et_rmse": float((et_prod[:, j] - et_faith[:, j]).square().mean().sqrt()),
            })
    with (OUT / "fidelity_forward_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
