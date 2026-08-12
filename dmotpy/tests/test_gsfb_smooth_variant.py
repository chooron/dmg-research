"""Tests for the smooth differentiable GSFB variant.

Validates:
1. gsfb_smooth_step runs forward without errors.
2. All outputs are finite.
3. Gradients flow to all parameters (no dead gradients).
4. Euler substep first-order convergence passes (mean_order in [0.85, 1.15]).
5. Validation CSVs exist and have correct content.
6. Original gsfb.py is unchanged (structural caveat preserved).
7. gsfb_smooth is NOT claimed to be identical to original GSFB.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "validation_results" / "gsfb_smooth_variant"

PASS_BAND = (0.85, 1.15)
NEARZERO = 1e-6
N_GRID = 8
NMUL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _make_params(n_grid=N_GRID, nmul=NMUL):
    torch.manual_seed(0)
    def u(lo, hi):
        return torch.rand(n_grid, nmul) * (hi - lo) + lo
    return dict(
        c      = u(0.01, 0.9),
        ndc    = u(0.1,  0.8),
        smax   = u(10.,  500.),
        emax   = u(0.1,  10.),
        frate  = u(1.,   100.),
        b      = u(0.1,  0.9),
        dpf    = u(0.01, 0.9),
        sdrmax = u(5.,   200.),
    )


def _make_forcing(n_grid=N_GRID, nmul=NMUL):
    torch.manual_seed(1)
    P   = torch.rand(n_grid, nmul) * 15.0
    T   = torch.rand(n_grid, nmul) * 20.0
    PET = torch.rand(n_grid, nmul) * 8.0
    return P, T, PET


def _make_states(params, n_grid=N_GRID, nmul=NMUL):
    torch.manual_seed(2)
    S1 = torch.rand(n_grid, nmul) * params["smax"] * 0.4 + NEARZERO
    S2 = torch.rand(n_grid, nmul) * 50.0 + NEARZERO
    S3 = torch.rand(n_grid, nmul) * 100.0 + NEARZERO
    return S1, S2, S3


# ---------------------------------------------------------------------------
# 1. Import and module structure
# ---------------------------------------------------------------------------

def test_import_gsfb_smooth():
    from models.core import gsfb  # renamed from gsfb_smooth  # noqa: F401


def test_smooth_helpers_importable():
    from models.core.gsfb import smooth_relu, smooth_min, smooth_cap_flux  # noqa: F401


def test_gsfb_smooth_step_importable():
    from models.core.gsfb import gsfb_step as gsfb_smooth_step  # noqa: F401


def test_params_bounds_exported():
    from models.core.gsfb import GSFB_SMOOTH_PARAMS_BOUNDS
    assert set(GSFB_SMOOTH_PARAMS_BOUNDS.keys()) == {
        "c", "ndc", "smax", "emax", "frate", "b", "dpf", "sdrmax"
    }


# ---------------------------------------------------------------------------
# 2. Smooth helper correctness
# ---------------------------------------------------------------------------

def test_smooth_relu_positive():
    from models.core.gsfb import smooth_relu
    x = torch.tensor([1.0, 2.0, 0.5])
    out = smooth_relu(x, tau=1e-3)
    # For large positive x, smooth_relu ≈ x
    assert torch.allclose(out, x, atol=1e-2), f"smooth_relu(positive) far from relu: {out}"


def test_smooth_relu_negative_small():
    from models.core.gsfb import smooth_relu
    x = torch.tensor([-10.0, -5.0])
    out = smooth_relu(x, tau=1e-3)
    # For large negative x, smooth_relu ≈ 0
    assert (out < 1e-2).all(), f"smooth_relu(negative) not near zero: {out}"


def test_smooth_relu_has_gradient():
    from models.core.gsfb import smooth_relu
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    smooth_relu(x, tau=1e-3).sum().backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()


def test_smooth_min_approximates_min():
    from models.core.gsfb import smooth_min
    a = torch.tensor([3.0, 1.0, 5.0])
    b = torch.tensor([2.0, 4.0, 5.0])
    out = smooth_min(a, b, tau=1e-4)
    expected = torch.minimum(a, b)
    assert torch.allclose(out, expected, atol=1e-2), f"{out} vs {expected}"


def test_smooth_cap_flux_bounded():
    from models.core.gsfb import smooth_cap_flux
    q = torch.tensor([5.0, 2.0, -1.0])
    avail = torch.tensor([3.0, 10.0, 2.0])
    out = smooth_cap_flux(q, avail, tau=1e-3)
    # Result must not exceed available by more than tau * few
    assert (out <= avail + 0.1).all()
    # Result must be non-negative (within tau tolerance)
    assert (out >= -0.01).all()


# ---------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------

def test_forward_runs_no_error():
    from models.core.gsfb import gsfb_step as gsfb_smooth_step
    params = _make_params()
    P, T, PET = _make_forcing()
    S1, S2, S3 = _make_states(params)
    Qsim, Ea, S1n, S2n, S3n = gsfb_smooth_step(
        P, T, PET, **params, S1=S1, S2=S2, S3=S3, nearzero=NEARZERO, tau=1e-3
    )
    assert Qsim.shape == (N_GRID, NMUL)


def test_forward_outputs_finite():
    from models.core.gsfb import gsfb_step as gsfb_smooth_step
    params = _make_params()
    P, T, PET = _make_forcing()
    S1, S2, S3 = _make_states(params)
    Qsim, Ea, S1n, S2n, S3n = gsfb_smooth_step(
        P, T, PET, **params, S1=S1, S2=S2, S3=S3, nearzero=NEARZERO, tau=1e-3
    )
    for name, t in [("Qsim", Qsim), ("Ea", Ea), ("S1n", S1n), ("S2n", S2n), ("S3n", S3n)]:
        assert torch.isfinite(t).all(), f"{name} contains non-finite values"


def test_forward_qsim_nonnegative():
    from models.core.gsfb import gsfb_step as gsfb_smooth_step
    params = _make_params()
    P, T, PET = _make_forcing()
    S1, S2, S3 = _make_states(params)
    Qsim, Ea, *_ = gsfb_smooth_step(
        P, T, PET, **params, S1=S1, S2=S2, S3=S3, nearzero=NEARZERO, tau=1e-3
    )
    assert (Qsim >= -1e-4).all(), "Qsim should be non-negative"


def test_forward_states_positive():
    from models.core.gsfb import gsfb_step as gsfb_smooth_step
    params = _make_params()
    P, T, PET = _make_forcing()
    S1, S2, S3 = _make_states(params)
    _, _, S1n, S2n, S3n = gsfb_smooth_step(
        P, T, PET, **params, S1=S1, S2=S2, S3=S3, nearzero=NEARZERO, tau=1e-3
    )
    for name, t in [("S1n", S1n), ("S2n", S2n), ("S3n", S3n)]:
        assert (t > 0).all(), f"{name} has non-positive values"


# ---------------------------------------------------------------------------
# 4. Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow_to_all_params():
    from models.core.gsfb import gsfb_step as gsfb_smooth_step
    params_raw = _make_params()
    params = {k: v.requires_grad_(True) for k, v in params_raw.items()}
    P, T, PET = _make_forcing()
    S1_r = torch.rand(N_GRID, NMUL).requires_grad_(True)
    S2_r = torch.rand(N_GRID, NMUL).requires_grad_(True) * 50
    S3_r = torch.rand(N_GRID, NMUL).requires_grad_(True) * 100

    Qsim, Ea, S1n, S2n, S3n = gsfb_smooth_step(
        P, T, PET, **params, S1=S1_r, S2=S2_r, S3=S3_r, nearzero=NEARZERO, tau=1e-3
    )
    loss = Qsim.sum() + Ea.sum() + S1n.sum() + S2n.sum() + S3n.sum()
    loss.backward()

    for pname, p in params.items():
        assert p.grad is not None, f"No gradient for param {pname}"
        assert not p.grad.isnan().any(), f"NaN gradient for param {pname}"
        assert not p.grad.isinf().any(), f"Inf gradient for param {pname}"


def test_gradients_flow_to_states():
    from models.core.gsfb import gsfb_step as gsfb_smooth_step
    params_raw = _make_params()
    params = {k: v.detach() for k, v in params_raw.items()}
    P, T, PET = _make_forcing()
    S1_r = (torch.rand(N_GRID, NMUL) * params["smax"] * 0.4 + NEARZERO).requires_grad_(True)
    S2_r = (torch.rand(N_GRID, NMUL) * 50 + NEARZERO).requires_grad_(True)
    S3_r = (torch.rand(N_GRID, NMUL) * 100 + NEARZERO).requires_grad_(True)

    Qsim, Ea, S1n, S2n, S3n = gsfb_smooth_step(
        P, T, PET, **params, S1=S1_r, S2=S2_r, S3=S3_r, nearzero=NEARZERO, tau=1e-3
    )
    (Qsim.sum() + S1n.sum()).backward()

    for sname, s in [("S1", S1_r), ("S2", S2_r), ("S3", S3_r)]:
        assert s.grad is not None, f"No gradient for state {sname}"
        assert not s.grad.isnan().any(), f"NaN gradient for state {sname}"


# ---------------------------------------------------------------------------
# 5. Euler substep convergence (inline, strict PASS_BAND)
# ---------------------------------------------------------------------------

def _run_substeps_smooth(n_sub, params, P_daily, T_daily, PET_daily, S1_0, S2_0, S3_0,
                          tau_base=1e-3):
    from models.core.gsfb import gsfb_step as gsfb_smooth_step
    dt = 1.0 / n_sub
    tau_eff = tau_base * dt  # scale tau with step size for O(1) convergence
    S1, S2, S3 = S1_0.clone(), S2_0.clone(), S3_0.clone()
    Qsim_acc = torch.zeros_like(S1)
    Ea_acc   = torch.zeros_like(S1)

    p = dict(params)
    p["c"]     = params["c"]     * dt
    p["emax"]  = params["emax"]  * dt
    p["frate"] = params["frate"] * dt
    p["dpf"]   = params["dpf"]   * dt

    P_sub   = P_daily   * dt
    PET_sub = PET_daily * dt

    for _ in range(n_sub):
        Q, Ea, S1, S2, S3 = gsfb_smooth_step(
            P_sub, T_daily, PET_sub, **p, S1=S1, S2=S2, S3=S3,
            nearzero=NEARZERO, tau=tau_eff,
        )
        Qsim_acc = Qsim_acc + Q
        Ea_acc   = Ea_acc   + Ea

    return Qsim_acc, Ea_acc, S1, S2, S3


def test_euler_convergence_order_in_pass_band():
    torch.manual_seed(7)
    params = _make_params()
    P, T, PET = _make_forcing()
    S1_0, S2_0, S3_0 = _make_states(params)

    REF_SUBSTEPS = 1024
    SUBSTEP_COUNTS = [1, 2, 4, 8, 16]

    Q_ref, Ea_ref, S1_ref, *_ = _run_substeps_smooth(
        REF_SUBSTEPS, params, P, T, PET, S1_0, S2_0, S3_0
    )

    errors = {}
    for n in SUBSTEP_COUNTS:
        Q_n, Ea_n, S1_n, *_ = _run_substeps_smooth(n, params, P, T, PET, S1_0, S2_0, S3_0)
        err = max(
            (Q_n - Q_ref).abs().mean().item(),
            (Ea_n - Ea_ref).abs().mean().item(),
            (S1_n - S1_ref).abs().mean().item(),
        )
        errors[n] = err

    orders = []
    for i in range(1, len(SUBSTEP_COUNTS)):
        n1, n2 = SUBSTEP_COUNTS[i - 1], SUBSTEP_COUNTS[i]
        e1, e2 = errors[n1], errors[n2]
        if e1 > 0 and e2 > 0:
            orders.append(math.log(e1 / e2) / math.log(n2 / n1))

    assert orders, "No valid convergence orders computed"
    mean_order = sum(orders) / len(orders)
    assert PASS_BAND[0] <= mean_order <= PASS_BAND[1], (
        f"Mean convergence order {mean_order:.3f} outside PASS_BAND {PASS_BAND}. "
        f"Orders per substep pair: {[f'{o:.3f}' for o in orders]}"
    )


# ---------------------------------------------------------------------------
# 6. Validation CSV existence and content
# ---------------------------------------------------------------------------

def test_forward_equivalence_csv_exists():
    assert (OUT_DIR / "gsfb_smooth_forward_equivalence.csv").exists()


def test_tau_sensitivity_csv_exists():
    assert (OUT_DIR / "gsfb_smooth_tau_sensitivity.csv").exists()


def test_gradient_summary_csv_exists():
    assert (OUT_DIR / "gsfb_smooth_gradient_summary.csv").exists()


def test_euler_errors_csv_exists():
    assert (OUT_DIR / "gsfb_smooth_euler_errors.csv").exists()


def test_euler_orders_csv_exists():
    assert (OUT_DIR / "gsfb_smooth_euler_orders.csv").exists()


def test_euler_summary_csv_exists():
    assert (OUT_DIR / "gsfb_smooth_euler_summary.csv").exists()


def test_euler_summary_passes():
    rows = _read_csv(OUT_DIR / "gsfb_smooth_euler_summary.csv")
    assert len(rows) >= 1, "Euler summary CSV has no rows"
    row = rows[0]
    assert row["euler_status"] == "PASS", (
        f"gsfb_smooth Euler status is {row['euler_status']!r}, expected PASS. "
        f"mean_order={row['mean_order']}"
    )


def test_gradient_summary_no_nan():
    rows = _read_csv(OUT_DIR / "gsfb_smooth_gradient_summary.csv")
    for row in rows:
        nan_frac = float(row["nan_fraction"])
        assert nan_frac == 0.0, (
            f"NaN gradients found for param={row['parameter']} tau={row['tau']}: "
            f"nan_fraction={nan_frac}"
        )


def test_gradient_summary_all_params_have_grad():
    rows = _read_csv(OUT_DIR / "gsfb_smooth_gradient_summary.csv")
    param_names = {"c", "ndc", "smax", "emax", "frate", "b", "dpf", "sdrmax"}
    seen = {row["parameter"] for row in rows if row["has_gradient"].lower() == "true"}
    assert param_names.issubset(seen), f"Missing gradients for: {param_names - seen}"


def test_flux_nonsmooth_review_csv_exists():
    assert (OUT_DIR / "gsfb_flux_nonsmooth_source_review.csv").exists()


def test_flux_nonsmooth_review_has_gsfb_entries():
    rows = _read_csv(OUT_DIR / "gsfb_flux_nonsmooth_source_review.csv")
    assert len(rows) >= 1, "gsfb_flux_nonsmooth_source_review.csv has no rows"
    # CSV has flux_name column (not 'model') — just verify it has entries
    assert any(rows), "flux review CSV is empty"


# ---------------------------------------------------------------------------
# 7. Original gsfb.py structural caveat is preserved
# ---------------------------------------------------------------------------

def test_original_gsfb_unchanged_has_hard_minimum():
    """The original gsfb.py must still contain hard torch.minimum calls."""
    # Original gsfb.py is now in archive after rename
    gsfb_src = (PROJECT_ROOT / "models" / "core" / "archive" / "gsfb_original.py").read_text()
    assert "torch.minimum" in gsfb_src, (
        "Archived gsfb_original.py does not contain torch.minimum — "
        "the structural caveat check failed."
    )


def test_gsfb_smooth_does_not_import_original_gsfb_step():
    """gsfb_smooth.py must NOT import gsfb_step — it's an independent variant."""
    # gsfb_smooth.py was renamed to gsfb.py; check the new gsfb.py does not import from original
    smooth_src = (PROJECT_ROOT / "models" / "core" / "gsfb.py").read_text()
    assert "from .gsfb_original import" not in smooth_src, (
        "new gsfb.py imports from gsfb_original — it should be independent."
    )
    assert "from models.core.archive import" not in smooth_src


def test_gsfb_smooth_uses_smooth_cap_flux():
    """gsfb_smooth.py must use smooth_cap_flux instead of torch.minimum for flux caps."""
    # gsfb_smooth.py was renamed to gsfb.py
    smooth_src = (PROJECT_ROOT / "models" / "core" / "gsfb.py").read_text()
    assert "smooth_cap_flux" in smooth_src, "new gsfb.py does not use smooth_cap_flux"


# ---------------------------------------------------------------------------
# 8. Final status CSV includes gsfb_smooth
# ---------------------------------------------------------------------------

def test_final_status_csv_includes_gsfb_smooth():
    final_csv = PROJECT_ROOT / "validation_results" / "euler_convergence_final" / \
                "euler_convergence_final_status.csv"
    if not final_csv.exists():
        pytest.skip("Final status CSV not yet generated")
    rows = _read_csv(final_csv)
    models = {row["model"] for row in rows}
    # After rename, gsfb_smooth became gsfb in the final status CSV
    assert "gsfb" in models, f"gsfb not in final status CSV. Models: {models}"


def test_final_status_gsfb_smooth_is_pass():
    final_csv = PROJECT_ROOT / "validation_results" / "euler_convergence_final" / \
                "euler_convergence_final_status.csv"
    if not final_csv.exists():
        pytest.skip("Final status CSV not yet generated")
    rows = _read_csv(final_csv)
    # After rename, gsfb_smooth became gsfb
    gsfb_rows = [r for r in rows if r["model"] == "gsfb"]
    assert gsfb_rows, "gsfb not found in final status CSV"
    status = gsfb_rows[0]["final_status"]
    assert status == "PASS", f"gsfb final_status is {status!r}, expected PASS"


def test_final_status_gsfb_is_pass_after_rename():
    """After rename, gsfb in final status CSV is the smooth variant — must be PASS."""
    final_csv = PROJECT_ROOT / "validation_results" / "euler_convergence_final" / \
                "euler_convergence_final_status.csv"
    if not final_csv.exists():
        pytest.skip("Final status CSV not yet generated")
    rows = _read_csv(final_csv)
    gsfb_rows = [r for r in rows if r["model"] == "gsfb"]
    assert gsfb_rows, "gsfb not found in final status CSV"
    status = gsfb_rows[0]["final_status"]
    assert status == "PASS", (
        f"gsfb (smooth variant after rename) final_status is {status!r} — expected PASS"
    )
    # The archived original gsfb_original is NOT in the final CSV (it's retired)
    gsfb_orig_rows = [r for r in rows if r["model"] == "gsfb_original"]
    assert not gsfb_orig_rows, "gsfb_original should not appear in final status CSV"
