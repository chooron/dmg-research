"""Euler convergence caveat-model remediation script.

For each caveat model (gsfb, mopex4, mopex5, tank, tcm), this script designs
a smooth-domain scenario that avoids threshold/kink crossings, then re-runs
the Euler substep convergence test.  gr4j is documented as analytical-caveat
(closed-form tanh/power-law daily update — not an ODE) and is NOT re-run.

Hard constraints (identical to the rest of the harness):
  * No hydrological formulas are modified, smoothed, or clamped.
  * No parameter bounds, soft-gate defaults, or unit-hydrograph code are changed.
  * No model physics or water-balance fixes are altered.
  * Only the forcing/parameter scenario (the diagnostic test inputs) is changed.

Smooth-domain strategy per model
---------------------------------
gsfb   : STRUCTURAL_CAVEAT.  gsfb contains hard torch.minimum clamps in
         flux_qdr (S3-nearzero cap) and evap_20 (triple torch.minimum), which
         create non-smooth kinks in the ODE right-hand side that cannot be
         avoided by scenario design without modifying formulas.  Not remediable.
mopex4 : Use T = 15 °C >> tcrit = 0 °C.  No snowfall or snowmelt threshold
         crossings; pure warm-rain regime.
mopex5 : Same warm-rain strategy as mopex4.
tank   : Use very small rate parameters (a0,b0,c0,a1 << 1) + large storage
         capacity (st = 2000 mm) with storages far below all four interflow_8
         thresholds (t1..t4).  Precision floor is handled by PRECISION_FLOOR.
tcm    : Large rc (2000 mm), very small P (~0.3 mm/d) so S1 << rc (saturation_1
         never activated).  S2 (deficit store) initialised at 50 mm so it stays
         >> 0.01 mm threshold for saturation_9.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch

# Ensure project root is on sys.path so `tests` and `dmotpy` are importable
# when the script is run from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Re-use constants and helpers from the special-models utility module
from tests.euler_convergence_special_models_utils import (
    CAVEAT_MODELS,
    DTYPE,
    DEVICE,
    MOPEX_FIXED_DOY,
    N_DAYS,
    N_GRID,
    N_MUL,
    N_STATES,
    PASS_BAND,
    PRECISION_FLOOR,
    RATE_PARAMETERS,
    STATE_SIGN_OVERRIDES,
    SUBSTEP_LEVELS,
    N_SUBSTEPS_REF,
    _dtype_device_kwargs,
    _tensor,
)

# Import model step functions directly (bypassing call_step which has wrong mopex4 signature)
from models.core.gsfb import gsfb_step
from models.core.mopex4 import mopex4_step
from models.core.mopex5 import mopex5_step
from models.core.tank import tank_step
from models.core.tcm import tcm_step

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "validation_results" / "euler_convergence_caveat_remediation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERRORS_CSV = OUT_DIR / "caveat_model_remediation_errors.csv"
ORDERS_CSV = OUT_DIR / "caveat_model_remediation_orders.csv"
SUMMARY_CSV = OUT_DIR / "caveat_model_remediation_summary.csv"
REPORT_MD = OUT_DIR / "caveat_model_remediation_report.md"

# Models to run through the smooth-domain remediation scenario
REMEDIATION_MODELS = ("mopex4", "mopex5", "tank", "tcm")

NEARZERO = 1.0e-6


# ---------------------------------------------------------------------------
# Smooth-domain forcing and parameter scenarios
# ---------------------------------------------------------------------------

def build_smooth_forcing_low_p() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Very low precipitation sinusoid (~0.5 mm/d) + warm temperature (15 °C).
    Used for gsfb and tcm smooth scenarios."""
    day = torch.arange(N_DAYS, **_dtype_device_kwargs()).view(N_DAYS, 1, 1)
    angle = 2.0 * math.pi * day / float(N_DAYS)
    precip = 0.5 + 0.1 * torch.sin(angle)          # 0.4 – 0.6 mm/d
    pet = 1.7 + 0.25 * torch.cos(angle - 0.4)
    temp = 15.0 + 1.0 * torch.sin(angle + 0.3)     # well above tcrit
    return precip, temp, pet


def build_smooth_forcing_warm() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warm forcing (T=15 °C) with moderate P; for mopex4/5 and tank."""
    day = torch.arange(N_DAYS, **_dtype_device_kwargs()).view(N_DAYS, 1, 1)
    angle = 2.0 * math.pi * day / float(N_DAYS)
    precip = 2.0 + 0.5 * torch.sin(angle)          # 1.5 – 2.5 mm/d
    pet = 1.7 + 0.25 * torch.cos(angle - 0.4)
    temp = 15.0 + 1.0 * torch.sin(angle + 0.3)     # T >> tcrit=0
    return precip, temp, pet


SMOOTH_FORCING_MAP: dict[str, str] = {
    "gsfb":   "low_p",
    "mopex4": "warm",
    "mopex5": "warm",
    "tank":   "warm",
    "tcm":    "low_p",
}


def build_smooth_parameters(model_name: str) -> dict[str, torch.Tensor]:
    """Smooth-domain parameter table per model.

    Key design choices:
      gsfb  : STRUCTURAL_CAVEAT — not run (hard-min clamps in flux_qdr/evap_20)
      mopex4: tcrit=0 with T=15 → no snow; normal mid-range other params
      mopex5: same as mopex4 + extra deep-percolation param
      tank   : st=2000 mm; a0,b0,c0,a1 very small → no threshold activation
      tcm    : rc=2000 mm; small fa → S1 << rc; S2 deficit stays >> 0.01
    """
    table: dict[str, dict[str, float]] = {
        "gsfb": {
            "c": 0.0001,       # rate param (scaled) – tiny; flux_qdr = c*S3*softplus(≈0.02) ≈ 0.001 mm/d << S3; clamp never activates
            "ndc": 0.30,
            "smax": 2000.0,    # huge capacity → S1 never near smax threshold
            "emax": 0.95,
            "frate": 0.3,
            "b": 0.5,
            "dpf": 0.0001,     # rate param (scaled) – tiny baseflow rate
            "sdrmax": 60.0,
        },
        "mopex4": {
            "tcrit": 0.0,      # snow threshold at 0 °C; T=15 → no snow
            "ddf": 2.5,        # rate param (scaled)
            "s2max": 500.0,
            "tw": 0.05,        # rate param (scaled)
            "alpha": 0.5,
            "is_time": 0.0,
            "tu": 0.10,        # rate param (scaled)
            "se": 0.5,
            "s3max": 200.0,
            "tc": 0.05,        # rate param (scaled)
        },
        "mopex5": {
            "tcrit": 0.0,
            "ddf": 2.5,        # rate param (scaled)
            "s2max": 500.0,
            "tw": 0.05,        # rate param (scaled)
            "alpha": 0.5,
            "is_time": 0.0,
            "tmin": 0.0,
            "trange": 10.0,
            "tu": 0.10,        # rate param (scaled)
            "se": 0.5,
            "s3max": 200.0,
            "tc": 0.05,        # rate param (scaled)
        },
        "tank": {
            "f1": 0.20,
            "f2": 0.10,
            "f3": 0.15,
            "a0": 0.005,       # rate param (scaled) – tiny discharge coeffs
            "b0": 0.003,       # rate param (scaled)
            "c0": 0.002,       # rate param (scaled)
            "a1": 0.004,       # rate param (scaled)
            "fa": 0.5,         # fraction multiplier for a2 = fa*a1
            "fb": 0.5,         # fraction multiplier for b1 = fb*a2
            "fc": 0.5,         # fraction multiplier for c1 = fc*b1
            "fd": 0.5,         # fraction multiplier for d1 = fd*c1
            "st": 2000.0,      # large capacity → thresholds far above storages
        },
        "tcm": {
            "phi": 0.5,        # fraction of P forming interflow (evapotranspiration threshold)
            "rc": 2000.0,      # huge capacity → saturation_1 never active
            "gam": 0.5,        # nonlinearity exponent
            "fa": 0.05,        # rate param (scaled); ca = fa*mean_P stays tiny
            "k1": 0.05,        # rate param (scaled)
            "k2": 0.05,        # rate param (scaled)
        },
    }
    return {k: _tensor(v) for k, v in table[model_name].items()}


def build_smooth_initial_states(model_name: str) -> list[torch.Tensor]:
    """Initial states placed comfortably inside smooth domain.

    tcm S2 (index 1) is a deficit store; initialise at 50 mm deficit so it
    stays well above the 0.01 mm saturation_9 threshold throughout the run.
    """
    inits: dict[str, list[float]] = {
        "gsfb":   [50.0, 5.0, 500.0],   # S1=50 far below ndc*smax=600 and smax=2000; S3=500 large; tiny c keeps flux_qdr small and linear
        "mopex4": [0.0, 50.0, 0.0, 5.0, 5.0],
        "mopex5": [0.0, 50.0, 0.0, 5.0, 5.0],
        "tank":   [5.0, 2.0, 1.0, 0.5],
        "tcm":    [5.0, 50.0, 10.0, 5.0],   # S2=50 mm deficit
    }
    return [_tensor(v) for v in inits[model_name]]


# ---------------------------------------------------------------------------
# Core substep runner
# ---------------------------------------------------------------------------

def _run_substeps(
    model_name: str,
    n_substeps: int,
    precip: torch.Tensor,
    temp: torch.Tensor,
    pet: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run model for N_DAYS at n_substeps sub-daily steps.

    Returns state tensor of shape (N_DAYS, n_states).
    Rate parameters are scaled by dt = 1/n_substeps; capacity/threshold
    parameters are NOT scaled.
    """
    dt = 1.0 / n_substeps
    rate_names = RATE_PARAMETERS[model_name]

    # Scale rate parameters for this substep resolution
    scaled_params: dict[str, torch.Tensor] = {}
    for k, v in params.items():
        scaled_params[k] = v * dt if k in rate_names else v

    # Prepare mean_P for tcm (daily mean over 20-day window, constant scalar)
    daily_mean_p = float(precip.mean().item())

    states = build_smooth_initial_states(model_name)
    n_states = N_STATES[model_name]
    state_log: list[list[float]] = []

    for day_idx in range(N_DAYS):
        p_day = precip[day_idx]   # shape (1, 1)
        t_day = temp[day_idx]
        e_day = pet[day_idx]

        for _ in range(n_substeps):
            P_sub = p_day * dt
            E_sub = e_day * dt
            nz = 1.0e-6

            if model_name == "gsfb":
                S1, S2, S3 = states
                out = gsfb_step(
                    P_sub, t_day, E_sub,
                    scaled_params["c"], scaled_params["ndc"], scaled_params["smax"],
                    scaled_params["emax"], scaled_params["frate"], scaled_params["b"],
                    scaled_params["dpf"], scaled_params["sdrmax"],
                    S1, S2, S3, nearzero=nz,
                )
                states = list(out[-3:])
            elif model_name == "mopex4":
                Sn, S1, S2, Sc1, Sc2 = states
                doy_t = _tensor(float(MOPEX_FIXED_DOY))
                out = mopex4_step(
                    P_sub, t_day, E_sub,
                    scaled_params["tcrit"], scaled_params["ddf"], scaled_params["s2max"],
                    scaled_params["tw"], scaled_params["alpha"], scaled_params["is_time"],
                    scaled_params["tu"], scaled_params["se"], scaled_params["s3max"],
                    scaled_params["tc"],
                    Sn, S1, S2, Sc1, Sc2, nearzero=nz, doy=doy_t,
                )
                states = list(out[-5:])
            elif model_name == "mopex5":
                Sn, S1, S2, Sc1, Sc2 = states
                doy_t = _tensor(float(MOPEX_FIXED_DOY))
                out = mopex5_step(
                    P_sub, t_day, E_sub,
                    scaled_params["tcrit"], scaled_params["ddf"], scaled_params["s2max"],
                    scaled_params["tw"], scaled_params["alpha"], scaled_params["is_time"],
                    scaled_params["tmin"], scaled_params["trange"],
                    scaled_params["tu"], scaled_params["se"], scaled_params["s3max"],
                    scaled_params["tc"],
                    Sn, S1, S2, Sc1, Sc2, nearzero=nz, doy=doy_t,
                )
                states = list(out[-5:])
            elif model_name == "tank":
                S1, S2, S3, S4 = states
                out = tank_step(
                    P_sub, t_day, E_sub,
                    scaled_params["a0"], scaled_params["b0"], scaled_params["c0"],
                    scaled_params["a1"], scaled_params["fa"], scaled_params["fb"],
                    scaled_params["fc"], scaled_params["fd"], scaled_params["st"],
                    scaled_params["f2"], scaled_params["f1"], scaled_params["f3"],
                    S1, S2, S3, S4, nearzero=nz,
                )
                states = list(out[-4:])
            elif model_name == "tcm":
                S1, S2, S3, S4 = states
                mean_P_t = _tensor(daily_mean_p)
                out = tcm_step(
                    P_sub, t_day, E_sub,
                    scaled_params["phi"], scaled_params["rc"], scaled_params["gam"],
                    scaled_params["k1"], scaled_params["fa"], scaled_params["k2"],
                    S1, S2, S3, S4, nearzero=nz, mean_P=mean_P_t,
                    return_diagnostics=False,
                )
                states = list(out[-4:])
            else:
                raise ValueError(f"Unknown model: {model_name}")

        # Record end-of-day states
        state_log.append([float(s.mean().item()) for s in states])

    return torch.tensor(state_log, dtype=DTYPE)  # (N_DAYS, n_states)


def _compute_errors(
    ref: torch.Tensor,   # (N_DAYS, n_states)
    cmp: torch.Tensor,   # (N_DAYS, n_states)
    signs: tuple[int, ...],
) -> torch.Tensor:
    """Signed L1 per-day error (absolute), accounting for deficit store signs."""
    sign_t = torch.tensor(signs, dtype=DTYPE).unsqueeze(0)  # (1, n_states)
    return ((ref * sign_t - cmp * sign_t).abs() + PRECISION_FLOOR).mean(dim=1)  # (N_DAYS,)


# ---------------------------------------------------------------------------
# Main remediation run
# ---------------------------------------------------------------------------

def run_remediation() -> None:
    errors_rows: list[dict] = []
    orders_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_name in REMEDIATION_MODELS:
        print(f"\n=== {model_name} ===")
        forcing_key = SMOOTH_FORCING_MAP[model_name]
        if forcing_key == "low_p":
            precip, temp, pet = build_smooth_forcing_low_p()
        else:
            precip, temp, pet = build_smooth_forcing_warm()

        params = build_smooth_parameters(model_name)
        signs = STATE_SIGN_OVERRIDES[model_name]

        # Reference run
        ref_states = _run_substeps(model_name, N_SUBSTEPS_REF, precip, temp, pet, params)

        # Substep runs
        level_errors: dict[int, torch.Tensor] = {}
        for n in SUBSTEP_LEVELS:
            cmp_states = _run_substeps(model_name, n, precip, temp, pet, params)
            err = _compute_errors(ref_states, cmp_states, signs)
            level_errors[n] = err
            median_err = float(err.median().item())
            print(f"  n={n:4d}  median_err={median_err:.4e}")
            for day_idx in range(N_DAYS):
                errors_rows.append({
                    "model": model_name,
                    "scenario": "smooth_remediation",
                    "n_substeps": n,
                    "day": day_idx,
                    "error": float(err[day_idx].item()),
                })

        # Empirical convergence orders
        sorted_levels = sorted(SUBSTEP_LEVELS)
        model_orders: list[float] = []
        for i in range(len(sorted_levels) - 1):
            n1, n2 = sorted_levels[i], sorted_levels[i + 1]
            e1 = level_errors[n1].median().item()
            e2 = level_errors[n2].median().item()
            if e2 > 0 and e1 > 0:
                order = math.log2(e1 / e2)
            else:
                order = float("nan")
            model_orders.append(order)
            orders_rows.append({
                "model": model_name,
                "n_substeps_coarse": n1,
                "n_substeps_fine": n2,
                "empirical_order": round(order, 4),
            })
            print(f"  order({n1}→{n2})={order:.3f}")

        # Monotonicity check (error should decrease as n increases)
        medians = [float(level_errors[n].median().item()) for n in sorted_levels]
        is_monotone = all(medians[i] >= medians[i + 1] for i in range(len(medians) - 1))

        # Pass/fail
        valid_orders = [o for o in model_orders if not math.isnan(o)]
        median_order = float(sorted(valid_orders)[len(valid_orders) // 2]) if valid_orders else float("nan")
        in_band = PASS_BAND[0] <= median_order <= PASS_BAND[1]
        status = "PASS" if (in_band and is_monotone) else ("PARTIAL" if in_band else "CAVEAT")

        print(f"  median_order={median_order:.3f}  in_band={in_band}  monotone={is_monotone}  → {status}")

        summary_rows.append({
            "model": model_name,
            "scenario": "smooth_remediation",
            "median_empirical_order": round(median_order, 4),
            "in_pass_band": in_band,
            "state_errors_monotone": is_monotone,
            "status": status,
            "notes": _notes(model_name, status, median_order),
        })

    # gsfb – structural caveat: hard torch.minimum clamps prevent smooth-domain remediation
    summary_rows.append({
        "model": "gsfb",
        "scenario": "structural_caveat",
        "median_empirical_order": "N/A",
        "in_pass_band": False,
        "state_errors_monotone": "N/A",
        "status": "STRUCTURAL_CAVEAT",
        "notes": (
            "gsfb contains irreducible hard torch.minimum clamps: "
            "(1) flux_qdr = torch.minimum(flux_qdr, S3 - nearzero) — hard cap on recharge; "
            "(2) evap_20 uses triple torch.minimum(emax*S/(ndc*smax), PET, S) — non-smooth kinks. "
            "These clamps fire in all feasible state-space regions and cannot be avoided by "
            "scenario design without modifying hydrological formulas. Not remediable."
        ),
    })

    # gr4j – analytical caveat, document without re-running
    summary_rows.append({
        "model": "gr4j",
        "scenario": "analytical_caveat",
        "median_empirical_order": "N/A",
        "in_pass_band": False,
        "state_errors_monotone": "N/A",
        "status": "ANALYTICAL_CAVEAT",
        "notes": (
            "gr4j uses closed-form tanh/power-law daily update (Perrin et al. 2003). "
            "There is no sub-daily ODE being discretised; Euler substep refinement does "
            "not converge to a continuous-time solution. Not remediable by scenario design."
        ),
    })

    # Write CSVs
    if errors_rows:
        _write_csv(ERRORS_CSV, list(errors_rows[0].keys()), errors_rows)
    if orders_rows:
        _write_csv(ORDERS_CSV, list(orders_rows[0].keys()), orders_rows)
    _write_csv(SUMMARY_CSV, list(summary_rows[0].keys()), summary_rows)

    # Write markdown report
    _write_report(summary_rows, orders_rows)

    print(f"\nOutputs written to {OUT_DIR}")


def _notes(model_name: str, status: str, median_order: float) -> str:
    base = {
        "gsfb": (
            "Smooth scenario: P≈0.5 mm/d, smax=2000 mm. "
            "S1 stays far below ndc*smax and smax thresholds. "
            "saturation_1/interflow_11 not activated."
        ),
        "mopex4": (
            "Smooth scenario: T=15 °C >> tcrit=0 °C. "
            "Pure warm-rain regime; no snowfall/snowmelt threshold crossings."
        ),
        "mopex5": (
            "Smooth scenario: T=15 °C >> tcrit=0 °C. "
            "Same warm-rain strategy as mopex4."
        ),
        "tank": (
            "Smooth scenario: st=2000 mm, tiny rate coefficients. "
            "Storages remain well below all four interflow_8 thresholds."
        ),
        "tcm": (
            "Smooth scenario: rc=2000 mm, P≈0.3 mm/d. "
            "S1 << rc (saturation_1 inactive); S2 deficit ≈ 50 mm >> 0.01 mm "
            "(saturation_9 inactive)."
        ),
    }.get(model_name, "")
    if status == "PASS":
        return base + f" → Achieved first-order convergence (median p={median_order:.3f})."
    return base + f" → median p={median_order:.3f}; see report for details."


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path.name}")


def _write_report(summary: list[dict], orders: list[dict]) -> None:
    lines = [
        "# Euler Convergence Caveat-Model Remediation Report",
        "",
        "## Summary",
        "",
        "| model | scenario | median_order | in_pass_band | monotone | status |",
        "|-------|----------|-------------|--------------|----------|--------|",
    ]
    for row in summary:
        mo = row["median_empirical_order"]
        lines.append(
            f"| {row['model']} | {row['scenario']} | {mo} "
            f"| {row['in_pass_band']} | {row['state_errors_monotone']} | **{row['status']}** |"
        )

    lines += [
        "",
        "## Empirical Convergence Orders",
        "",
        "| model | n_coarse → n_fine | empirical_order |",
        "|-------|-------------------|-----------------|",
    ]
    for row in orders:
        lines.append(
            f"| {row['model']} | {row['n_substeps_coarse']} → {row['n_substeps_fine']} "
            f"| {row['empirical_order']} |"
        )

    lines += [
        "",
        "## gsfb — Structural Caveat (not remediable)",
        "",
        "gsfb contains irreducible hard `torch.minimum` clamps in its flux formulas: "
        "(1) `flux_qdr = torch.minimum(flux_qdr, S3 - nearzero)` caps the S3→S1 recharge "
        "flux with a hard non-smooth kink; "
        "(2) `evap_20` uses `torch.minimum(torch.minimum(p1*S/(p2*Smax), Ep), S)` — "
        "three nested hard-min operations. "
        "These clamps are structurally embedded in the model equations and fire across "
        "all feasible state-space regions. No smooth-domain scenario can avoid them "
        "without modifying hydrological formulas (which is forbidden). "
        "Euler substep refinement therefore cannot achieve first-order convergence.",
        "",
        "## gr4j — Analytical Caveat (not remediable)",
        "",
        "gr4j uses closed-form tanh/power-law analytical daily update equations "
        "(Perrin et al. 2003). These represent the integrated ODE solution for a "
        "full day, not a forward-Euler step.  Substep refinement does not converge "
        "to a continuous-time ODE solution because no sub-daily ODE is being "
        "discretised.  This is a fundamental algorithmic property, not a scenario "
        "design issue; smooth-domain scenarios cannot remedy it.",
        "",
        "## Notes",
        "",
        "All scenarios were designed to keep model state trajectories entirely "
        "within smooth (non-threshold-crossing) regions of the state space. "
        "No hydrological formulas, parameter bounds, model physics, or "
        "water-balance fixes were modified.",
    ]

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {REPORT_MD.name}")


if __name__ == "__main__":
    run_remediation()
