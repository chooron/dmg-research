"""
validate_gsfb_smooth_variant.py
================================
Validation script for the smooth differentiable GSFB variant.

Outputs (all written to validation_results/gsfb_smooth_variant/):
  gsfb_smooth_forward_equivalence.csv   — forward bias vs original gsfb
  gsfb_smooth_tau_sensitivity.csv       — sensitivity across tau values
  gsfb_smooth_gradient_summary.csv      — gradient flow metrics
  gsfb_smooth_euler_errors.csv          — per-substep Euler errors
  gsfb_smooth_euler_orders.csv          — empirical convergence orders
  gsfb_smooth_euler_summary.csv         — pass/fail summary
"""

import sys
import os
import csv
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from models.core.gsfb import gsfb_step
from models.core.gsfb_smooth import gsfb_smooth_step

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "validation_results", "gsfb_smooth_variant"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
torch.manual_seed(42)
N_GRID = 32
NMUL = 1
NEARZERO = 1e-6
TAUS = [1e-2, 1e-3, 1e-4]
TAU_DEFAULT = 1e-3

SUBSTEP_COUNTS = [1, 2, 4, 8, 16]
REF_SUBSTEPS = 1024
PASS_BAND = (0.85, 1.15)

device = torch.device("cpu")


def make_params(n_grid, nmul):
    """Random params in valid ranges."""
    def u(lo, hi):
        return torch.rand(n_grid, nmul) * (hi - lo) + lo

    return dict(
        c      = u(0.0,   1.0),
        ndc    = u(0.05,  0.95),
        smax   = u(1.0,   2000.0),
        emax   = u(0.0,   20.0),
        frate  = u(0.0,   200.0),
        b      = u(0.0,   1.0),
        dpf    = u(0.0,   1.0),
        sdrmax = u(1.0,   300.0),
    )


def make_forcing(n_grid, nmul):
    P   = torch.rand(n_grid, nmul) * 20.0
    T   = torch.rand(n_grid, nmul) * 30.0 - 5.0
    PET = torch.rand(n_grid, nmul) * 10.0
    return P, T, PET


def make_states(n_grid, nmul, params):
    S1 = torch.rand(n_grid, nmul) * params["smax"] * 0.5 + NEARZERO
    S2 = torch.rand(n_grid, nmul) * 100.0 + NEARZERO
    S3 = torch.rand(n_grid, nmul) * 200.0 + NEARZERO
    return S1, S2, S3


# ---------------------------------------------------------------------------
# 1. Forward equivalence / bias analysis
# ---------------------------------------------------------------------------
def run_forward_equivalence():
    print("[1/4] Forward equivalence analysis ...")
    rows = []
    params = make_params(N_GRID, NMUL)
    P, T, PET = make_forcing(N_GRID, NMUL)
    S1, S2, S3 = make_states(N_GRID, NMUL, params)

    Qsim_orig, Ea_orig, S1o, S2o, S3o = gsfb_step(
        P, T, PET, **params, S1=S1, S2=S2, S3=S3, nearzero=NEARZERO
    )
    Qsim_sm, Ea_sm, S1s, S2s, S3s = gsfb_smooth_step(
        P, T, PET, **params, S1=S1, S2=S2, S3=S3, nearzero=NEARZERO, tau=TAU_DEFAULT
    )

    for var_name, orig, sm in [
        ("Qsim", Qsim_orig, Qsim_sm),
        ("Ea",   Ea_orig,   Ea_sm),
        ("S1",   S1o,       S1s),
        ("S2",   S2o,       S2s),
        ("S3",   S3o,       S3s),
    ]:
        diff = (sm - orig).abs()
        rel  = diff / (orig.abs() + NEARZERO)
        rows.append(dict(
            variable       = var_name,
            tau            = TAU_DEFAULT,
            mean_abs_diff  = diff.mean().item(),
            max_abs_diff   = diff.max().item(),
            mean_rel_diff  = rel.mean().item(),
            max_rel_diff   = rel.max().item(),
        ))

    path = os.path.join(OUT_DIR, "gsfb_smooth_forward_equivalence.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# 2. Tau sensitivity
# ---------------------------------------------------------------------------
def run_tau_sensitivity():
    print("[2/4] Tau sensitivity analysis ...")
    rows = []
    params = make_params(N_GRID, NMUL)
    P, T, PET = make_forcing(N_GRID, NMUL)
    S1, S2, S3 = make_states(N_GRID, NMUL, params)

    Qsim_orig, Ea_orig, *_ = gsfb_step(
        P, T, PET, **params, S1=S1, S2=S2, S3=S3, nearzero=NEARZERO
    )

    for tau in TAUS:
        Qsim_sm, Ea_sm, *_ = gsfb_smooth_step(
            P, T, PET, **params, S1=S1, S2=S2, S3=S3, nearzero=NEARZERO, tau=tau
        )
        for var_name, orig, sm in [("Qsim", Qsim_orig, Qsim_sm), ("Ea", Ea_orig, Ea_sm)]:
            diff = (sm - orig).abs()
            rows.append(dict(
                tau=tau,
                variable=var_name,
                mean_abs_diff=diff.mean().item(),
                max_abs_diff=diff.max().item(),
            ))

    path = os.path.join(OUT_DIR, "gsfb_smooth_tau_sensitivity.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# 3. Gradient validation
# ---------------------------------------------------------------------------
def run_gradient_validation():
    print("[3/4] Gradient validation ...")
    rows = []
    param_names = ["c", "ndc", "smax", "emax", "frate", "b", "dpf", "sdrmax"]

    for tau in TAUS:
        params_raw = make_params(N_GRID, NMUL)
        params = {k: v.requires_grad_(True) for k, v in params_raw.items()}
        P, T, PET = make_forcing(N_GRID, NMUL)
        S1_r = torch.rand(N_GRID, NMUL).requires_grad_(True)
        S2_r = torch.rand(N_GRID, NMUL).requires_grad_(True) * 100
        S3_r = torch.rand(N_GRID, NMUL).requires_grad_(True) * 200

        Qsim, Ea, S1n, S2n, S3n = gsfb_smooth_step(
            P, T, PET, **params, S1=S1_r, S2=S2_r, S3=S3_r,
            nearzero=NEARZERO, tau=tau,
        )
        loss = Qsim.sum() + Ea.sum() + S1n.sum() + S2n.sum() + S3n.sum()
        loss.backward()

        for pname in param_names:
            g = params[pname].grad
            has_grad = g is not None
            nan_frac = float(g.isnan().float().mean().item()) if has_grad else 1.0
            inf_frac = float(g.isinf().float().mean().item()) if has_grad else 1.0
            mean_abs = float(g.abs().mean().item()) if has_grad else float("nan")
            rows.append(dict(
                tau=tau,
                parameter=pname,
                has_gradient=has_grad,
                nan_fraction=nan_frac,
                inf_fraction=inf_frac,
                mean_abs_grad=mean_abs,
                grad_ok=(has_grad and nan_frac == 0.0 and inf_frac == 0.0),
            ))

    path = os.path.join(OUT_DIR, "gsfb_smooth_gradient_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# 4. Euler substep convergence
# ---------------------------------------------------------------------------
def _run_substeps(n_sub, params, P_daily, T_daily, PET_daily, S1_0, S2_0, S3_0, tau):
    """Run gsfb_smooth_step n_sub times with dt=1/n_sub scaling."""
    dt = 1.0 / n_sub
    S1, S2, S3 = S1_0.clone(), S2_0.clone(), S3_0.clone()
    Qsim_acc = torch.zeros_like(S1)
    Ea_acc   = torch.zeros_like(S1)

    # Scale rate parameters by dt; capacity params unchanged
    p = dict(params)
    p["c"]      = params["c"]      * dt
    p["emax"]   = params["emax"]   * dt
    p["frate"]  = params["frate"]  * dt
    p["dpf"]    = params["dpf"]    * dt

    P_sub   = P_daily   * dt
    PET_sub = PET_daily * dt

    # Scale tau by dt so the smooth-cap bias shrinks proportionally with step
    # size, preserving first-order Euler convergence.
    tau_sub = tau * dt

    for _ in range(n_sub):
        Q, Ea, S1, S2, S3 = gsfb_smooth_step(
            P_sub, T_daily, PET_sub, **p, S1=S1, S2=S2, S3=S3,
            nearzero=NEARZERO, tau=tau_sub,
        )
        Qsim_acc = Qsim_acc + Q
        Ea_acc   = Ea_acc   + Ea

    return Qsim_acc, Ea_acc, S1, S2, S3


def run_euler_convergence():
    print("[4/4] Euler substep convergence ...")
    params = make_params(N_GRID, NMUL)
    P, T, PET = make_forcing(N_GRID, NMUL)
    S1_0, S2_0, S3_0 = make_states(N_GRID, NMUL, params)

    error_rows  = []
    order_rows  = []
    summary_rows = []

    for tau in [TAU_DEFAULT]:  # primary tau
        # Reference solution at 1024 substeps
        Q_ref, Ea_ref, S1_ref, S2_ref, S3_ref = _run_substeps(
            REF_SUBSTEPS, params, P, T, PET, S1_0, S2_0, S3_0, tau=tau
        )

        errors = {}
        for n_sub in SUBSTEP_COUNTS:
            Q_n, Ea_n, S1_n, S2_n, S3_n = _run_substeps(
                n_sub, params, P, T, PET, S1_0, S2_0, S3_0, tau=tau
            )
            err_Q  = (Q_n  - Q_ref).abs().mean().item()
            err_Ea = (Ea_n - Ea_ref).abs().mean().item()
            err_S1 = (S1_n - S1_ref).abs().mean().item()
            err    = max(err_Q, err_Ea, err_S1)
            errors[n_sub] = err
            error_rows.append(dict(
                tau=tau, n_substeps=n_sub,
                err_Qsim=err_Q, err_Ea=err_Ea, err_S1=err_S1,
                err_max=err,
            ))

        # Compute convergence orders from consecutive substep pairs
        substep_list = SUBSTEP_COUNTS
        orders = []
        for i in range(1, len(substep_list)):
            n1, n2 = substep_list[i-1], substep_list[i]
            e1, e2 = errors[n1], errors[n2]
            if e2 > 0 and e1 > 0:
                p_emp = math.log(e1 / e2) / math.log(n2 / n1)
            else:
                p_emp = float("nan")
            orders.append(p_emp)
            order_rows.append(dict(
                tau=tau, n_coarse=n1, n_fine=n2,
                err_coarse=e1, err_fine=e2, order=p_emp,
            ))

        valid_orders = [o for o in orders if not math.isnan(o)]
        mean_order = sum(valid_orders) / len(valid_orders) if valid_orders else float("nan")
        in_band = PASS_BAND[0] <= mean_order <= PASS_BAND[1]
        status = "PASS" if in_band else "FAIL_CONVERGENCE"

        summary_rows.append(dict(
            model="gsfb_smooth",
            tau=tau,
            mean_order=mean_order,
            pass_band_lo=PASS_BAND[0],
            pass_band_hi=PASS_BAND[1],
            in_band=in_band,
            euler_status=status,
        ))
        print(f"  tau={tau}: mean_order={mean_order:.3f}  status={status}")

    # Write outputs
    path_err = os.path.join(OUT_DIR, "gsfb_smooth_euler_errors.csv")
    with open(path_err, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(error_rows[0].keys()))
        w.writeheader(); w.writerows(error_rows)
    print(f"  -> {path_err}")

    path_ord = os.path.join(OUT_DIR, "gsfb_smooth_euler_orders.csv")
    with open(path_ord, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(order_rows[0].keys()))
        w.writeheader(); w.writerows(order_rows)
    print(f"  -> {path_ord}")

    path_sum = os.path.join(OUT_DIR, "gsfb_smooth_euler_summary.csv")
    with open(path_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"  -> {path_sum}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_forward_equivalence()
    run_tau_sensitivity()
    run_gradient_validation()
    run_euler_convergence()
    print("\nAll validation outputs written to:", OUT_DIR)
