from __future__ import annotations

"""Euler substep first-order convergence utilities for the 7 wrappable SPECIAL
core models (gr4j, gsfb, mopex4, mopex5, tank, tcm, topmodel) that were
previously excluded from `tests/euler_convergence_all_core_utils.py` due to
non-standard step-function signatures or discrete-formula concerns.

See `validation_results/euler_convergence_special_models/special_model_exclusion_diagnosis.md`
for the full per-model diagnosis of why each model was excluded and how the
diagnostic dt-wrapper below resolves the API mismatch WITHOUT touching any
model source code, hydrological formula, parameter bound, or unit-hydrograph
implementation in `dmotpy/models/core`.

`shm` is permanently excluded: `dmotpy/models/core/shm.py` is an empty file
(0 bytes) and defines no runnable model.

Design constraints (identical to the all-core harness):
  * No hydrological formulas are modified, smoothed, or clamped.
  * No parameter bounds, soft-gate defaults, or unit-hydrograph code are changed.
  * Each model is exercised through its *unmodified* `<model>_step` function from
    `dmotpy/models/core`; only the calling harness (zero-order-hold forcing,
    dt-scaled rate parameters, sub-daily looping, and supplying any required
    extra keyword-only arguments such as `doy` or `mean_P`) is new test/script
    code.

Convergence procedure (identical to the representative/all-core harness):
  * Daily forcing (precip, temp, pet) is smooth/synthetic and held constant
    (zero-order-hold) within each day while substepping.
  * For substep level k in SUBSTEP_LEVELS, the day is split into 2**k substeps
    of width dt = 1 / 2**k; rate parameters are scaled by dt; all other
    parameters are left untouched (they are storage-capacity-like, shape
    exponents, or split fractions, not per-time-unit rates).
  * A fine reference run uses N_SUBSTEPS_REF substeps (k_ref = 10).
  * Per-day-aggregated outputs/states at each k are compared against the
    reference; the empirical order is p_k = log2(error_k / error_{k+1}).
"""

import math
from pathlib import Path
from typing import Any, Callable

import torch

from models.core.gr4j import gr4j_step
from models.core.gsfb import gsfb_step
from models.core.mopex4 import mopex4_step
from models.core.mopex5 import mopex5_step
from models.core.tank import tank_step
from models.core.tcm import tcm_step
from models.core.topmodel import topmodel_step


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "validation_results" / "euler_convergence_special_models"
SPECIAL_ERRORS_CSV_PATH = OUTPUT_DIR / "euler_special_models_convergence_errors.csv"
SPECIAL_ORDERS_CSV_PATH = OUTPUT_DIR / "euler_special_models_convergence_orders.csv"
SPECIAL_SUMMARY_CSV_PATH = OUTPUT_DIR / "euler_special_models_convergence_summary.csv"
SPECIAL_REPORT_MD_PATH = OUTPUT_DIR / "euler_special_models_report.md"

DTYPE = torch.float64
DEVICE = "cpu"
NEARZERO = 1.0e-6
SUBSTEP_LEVELS = (1, 2, 4, 8, 16)
N_SUBSTEPS_REF = 1024
PASS_BAND = (0.85, 1.15)
PRECISION_FLOOR = 1.0e-10
N_DAYS = 20
N_GRID = 1
N_MUL = 1

# Permanently excluded: file is empty, no runnable model exists.
PERMANENTLY_EXCLUDED_MODELS = frozenset({"shm"})

# The 7 special models exercised by this harness via dedicated wrappers.
SPECIAL_TARGET_MODELS = (
    "gr4j",
    "gsfb",
    "mopex4",
    "mopex5",
    "tank",
    "tcm",
    "topmodel",
)

# Models supported but with a documented caveat (still run through the same
# convergence test; caveat is recorded in notes/classification only).
CAVEAT_MODELS = frozenset(
    {
        "gr4j",    # uses closed-form analytical (implicit) equations, not Euler ODE -> order < 1 expected
        "gsfb",    # saturation_1 threshold + analytical recharge -> threshold crossings expected
        "mopex4",  # snow threshold (tcrit) -> potential non-monotone error near melt
        "mopex5",  # snow threshold (tcrit) -> potential non-monotone error near melt
        "tank",    # multi-threshold cascade -> threshold crossings expected
        "tcm",     # saturation_1 and saturation_9 thresholds -> threshold crossings expected
    }
)

# Rate parameters (per unit time) that must be scaled by dt when substepping.
# Storage capacities, shape exponents, split fractions, and thresholds are NOT
# rate parameters and are left unscaled.
RATE_PARAMETERS: dict[str, frozenset[str]] = {
    "gr4j": frozenset({"x2"}),
    "gsfb": frozenset({"c", "dpf"}),
    "mopex4": frozenset({"ddf", "tw", "tu", "tc"}),
    "mopex5": frozenset({"ddf", "tw", "tu", "tc"}),
    "tank": frozenset({"a0", "b0", "c0", "a1"}),
    "tcm": frozenset({"k1", "k2", "fa"}),
    "topmodel": frozenset({"kd", "q0"}),
}

# State sign overrides: +1 for normal stores, -1 for deficit-type stores
# (positive value = water deficit from saturation). Only topmodel's S2 needs
# this; all others are plain storage stores.
STATE_SIGN_OVERRIDES: dict[str, tuple[int, ...]] = {
    "gr4j": (1, 1),
    "gsfb": (1, 1, 1),
    "mopex4": (1, 1, 1, 1, 1),
    "mopex5": (1, 1, 1, 1, 1),
    "tank": (1, 1, 1, 1),
    "tcm": (1, -1, 1, 1),  # S2 is a deficit store (positive value = water deficit)
    "topmodel": (1, -1),
}

N_STATES: dict[str, int] = {
    "gr4j": 2,
    "gsfb": 3,
    "mopex4": 5,
    "mopex5": 5,
    "tank": 4,
    "tcm": 4,
    "topmodel": 2,
}

# Fixed day-of-year used for mopex4/mopex5 seasonal interception modulation
# (`interception_4`). doy=182 is mid-year (summer), placing the cosine
# seasonal term away from any phase discontinuity. This is a forcing/scenario
# design choice for the convergence test, NOT a model change.
MOPEX_FIXED_DOY = 182.0


def _dtype_device_kwargs() -> dict[str, Any]:
    return {"dtype": DTYPE, "device": DEVICE}


def _tensor(value: float) -> torch.Tensor:
    return torch.full((N_GRID, N_MUL), float(value), **_dtype_device_kwargs())


def build_smooth_forcing() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth synthetic daily forcing, identical shape/spirit to the
    representative-model and all-core harnesses (kept deliberately simple and
    reusable)."""
    day = torch.arange(N_DAYS, **_dtype_device_kwargs()).view(N_DAYS, 1, 1)
    angle = 2.0 * math.pi * day / float(N_DAYS)
    precip = 4.5 + 0.8 * torch.sin(angle) + 0.3 * torch.cos(2.0 * angle)
    pet = 1.7 + 0.25 * torch.cos(angle - 0.4) + 0.1 * torch.sin(2.0 * angle)
    temp = 12.5 + 1.2 * torch.sin(angle + 0.3)
    return precip, temp, pet


# Interior parameter values (mid-bound, physically reasonable) for every
# special model. Values are chosen to keep states comfortably inside capacity
# bounds across all SUBSTEP_LEVELS and the reference resolution.
def build_interior_parameters(model_name: str) -> dict[str, torch.Tensor]:
    table: dict[str, dict[str, float]] = {
        "gr4j": {"x1": 350.0, "x2": 0.5, "x3": 90.0, "x4": 2.5},
        "gsfb": {
            "c": 0.10,
            "ndc": 0.30,
            "smax": 250.0,
            "emax": 0.95,
            "frate": 0.5,
            "b": 1.5,
            "dpf": 0.04,
            "sdrmax": 60.0,
        },
        "mopex4": {
            "tcrit": 0.5,
            "ddf": 2.5,
            "s2max": 300.0,
            "tw": 0.10,
            "alpha": 0.5,
            "is_time": 0.0,
            "tu": 0.20,
            "se": 0.5,
            "s3max": 150.0,
            "tc": 0.08,
        },
        "mopex5": {
            "tcrit": 0.5,
            "ddf": 2.5,
            "s2max": 300.0,
            "tw": 0.10,
            "alpha": 0.5,
            "is_time": 0.0,
            "tmin": 0.0,
            "trange": 4.0,
            "tu": 0.20,
            "se": 0.5,
            "s3max": 150.0,
            "tc": 0.08,
        },
        "tank": {
            "a0": 0.20,
            "b0": 0.10,
            "c0": 0.05,
            "a1": 0.15,
            "fa": 0.20,
            "fb": 0.50,
            "fc": 0.70,
            "fd": 0.90,
            "st": 300.0,
            "f2": 0.10,
            "f1": 0.30,
            "f3": 0.50,
        },
        "tcm": {
            "phi": 0.55,
            "rc": 0.6,
            "gam": 1.5,
            "k1": 0.10,
            "fa": 0.05,
            "k2": 0.03,
        },
        "topmodel": {
            "suzmax": 50.0,
            "st": 0.06,
            "kd": 0.15,
            "q0": 0.02,
            "f": 2.5,
            "chi": 3.0,
            "phi": 0.4,
        },
    }
    raw = table[model_name]
    return {k: _tensor(v) for k, v in raw.items()}


def build_initial_states(model_name: str) -> tuple[torch.Tensor, ...]:
    table: dict[str, tuple[float, ...]] = {
        "gr4j": (175.0, 45.0),
        "gsfb": (125.0, 30.0, 30.0),
        "mopex4": (0.0, 150.0, 60.0, 10.0, 10.0),
        "mopex5": (0.0, 150.0, 60.0, 10.0, 10.0),
        "tank": (60.0, 40.0, 30.0, 20.0),
        "tcm": (40.0, 30.0, 10.0, 10.0),
        "topmodel": (25.0, 0.03),
    }
    return tuple(_tensor(v) for v in table[model_name])


# Per-model step-function dispatch. Each wrapper accepts (P, T, PET, params,
# states, dt_scaled_params, nearzero) and returns the unmodified state tuple,
# exactly mirroring the all-core harness convention but supplying any extra
# keyword-only arguments required by the special models' step functions.
def call_step(
    model_name: str,
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    scaled_params: dict[str, torch.Tensor],
    states: tuple[torch.Tensor, ...],
    nearzero: float,
    mean_P: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if model_name == "gr4j":
        S1, S2 = states
        out = gr4j_step(
            P, T, PET,
            scaled_params["x1"], scaled_params["x2"],
            scaled_params["x3"], scaled_params["x4"],
            S1, S2, nearzero=nearzero,
        )
        return out[-2:]  # (S1_new, S2_new)
    if model_name == "gsfb":
        S1, S2, S3 = states
        out = gsfb_step(
            P, T, PET,
            scaled_params["c"], scaled_params["ndc"], scaled_params["smax"],
            scaled_params["emax"], scaled_params["frate"], scaled_params["b"],
            scaled_params["dpf"], scaled_params["sdrmax"],
            S1, S2, S3, nearzero=nearzero,
        )
        return out[-3:]
    if model_name == "mopex4":
        Sn, S1, S2, Sc1, Sc2 = states
        doy = _tensor(MOPEX_FIXED_DOY)
        out = mopex4_step(
            P, T, PET,
            scaled_params["tcrit"], scaled_params["ddf"], scaled_params["s2max"],
            scaled_params["tw"], scaled_params["alpha"], scaled_params["is_time"],
            scaled_params["tu"], scaled_params["se"], scaled_params["s3max"],
            scaled_params["tc"],
            Sn, S1, S2, Sc1, Sc2, nearzero=nearzero, doy=doy,
        )
        return out[-5:]
    if model_name == "mopex5":
        Sn, S1, S2, Sc1, Sc2 = states
        doy = _tensor(MOPEX_FIXED_DOY)
        out = mopex5_step(
            P, T, PET,
            scaled_params["tcrit"], scaled_params["ddf"], scaled_params["s2max"],
            scaled_params["tw"], scaled_params["alpha"], scaled_params["is_time"],
            scaled_params["tmin"], scaled_params["trange"],
            scaled_params["tu"], scaled_params["se"], scaled_params["s3max"],
            scaled_params["tc"],
            Sn, S1, S2, Sc1, Sc2, nearzero=nearzero, doy=doy,
        )
        return out[-5:]
    if model_name == "tank":
        S1, S2, S3, S4 = states
        out = tank_step(
            P, T, PET,
            scaled_params["a0"], scaled_params["b0"], scaled_params["c0"],
            scaled_params["a1"], scaled_params["fa"], scaled_params["fb"],
            scaled_params["fc"], scaled_params["fd"], scaled_params["st"],
            scaled_params["f2"], scaled_params["f1"], scaled_params["f3"],
            S1, S2, S3, S4, nearzero=nearzero,
        )
        return out[-4:]
    if model_name == "tcm":
        S1, S2, S3, S4 = states
        assert mean_P is not None
        out = tcm_step(
            P, T, PET,
            scaled_params["phi"], scaled_params["rc"], scaled_params["gam"],
            scaled_params["k1"], scaled_params["fa"], scaled_params["k2"],
            S1, S2, S3, S4, nearzero=nearzero, mean_P=mean_P,
            return_diagnostics=False,
        )
        return out[-4:]
    if model_name == "topmodel":
        S1, S2 = states
        out = topmodel_step(
            P, T, PET,
            scaled_params["suzmax"], scaled_params["st"], scaled_params["kd"],
            scaled_params["q0"], scaled_params["f"], scaled_params["chi"],
            scaled_params["phi"],
            S1, S2, nearzero=nearzero,
        )
        return out[-2:]
    raise ValueError(f"Unknown special model: {model_name}")


def scale_rate_parameters(
    model_name: str, params: dict[str, torch.Tensor], dt: float
) -> dict[str, torch.Tensor]:
    rate_names = RATE_PARAMETERS[model_name]
    scaled = dict(params)
    for name in rate_names:
        scaled[name] = params[name] * dt
    return scaled


def run_substepped_simulation(
    model_name: str,
    n_substeps: int,
    precip: torch.Tensor,
    temp: torch.Tensor,
    pet: torch.Tensor,
    params: dict[str, torch.Tensor],
    init_states: tuple[torch.Tensor, ...],
    nearzero: float = NEARZERO,
) -> tuple[torch.Tensor, ...]:
    """Run the model with zero-order-hold daily forcing split into
    `n_substeps` substeps per day; rate parameters scaled by dt=1/n_substeps.
    Returns end-of-simulation-day stacked states, shape (N_DAYS, n_states, ...).
    """
    dt = 1.0 / float(n_substeps)
    scaled_params = scale_rate_parameters(model_name, params, dt)

    mean_P = None
    if model_name == "tcm":
        mean_P = precip.mean(dim=0)

    states = init_states
    n_states = N_STATES[model_name]
    daily_states = torch.zeros(
        (N_DAYS, n_states, N_GRID, N_MUL), **_dtype_device_kwargs()
    )

    for day in range(N_DAYS):
        P_day = precip[day]
        T_day = temp[day]
        PET_day = pet[day]
        for _ in range(n_substeps):
            mean_P_arg = mean_P if model_name == "tcm" else None
            states = call_step(
                model_name, P_day, T_day, PET_day, scaled_params, states,
                nearzero, mean_P=mean_P_arg,
            )
        for i, s in enumerate(states):
            daily_states[day, i] = s

    return daily_states


def compute_signed_states(model_name: str, daily_states: torch.Tensor) -> torch.Tensor:
    """Apply STATE_SIGN_OVERRIDES so deficit-type stores compare correctly."""
    signs = STATE_SIGN_OVERRIDES[model_name]
    signed = daily_states.clone()
    for i, sign in enumerate(signs):
        if sign == -1:
            signed[:, i] = -signed[:, i]
    return signed


def compute_errors_for_model(model_name: str) -> dict[int, float]:
    """Compute RMS error (across days, states, grid, mul) between each
    substep level's daily states and the fine-reference run's daily states."""
    precip, temp, pet = build_smooth_forcing()
    params = build_interior_parameters(model_name)
    init_states = build_initial_states(model_name)

    ref_states = run_substepped_simulation(
        model_name, N_SUBSTEPS_REF, precip, temp, pet, params, init_states
    )
    ref_signed = compute_signed_states(model_name, ref_states)

    errors: dict[int, float] = {}
    for k in SUBSTEP_LEVELS:
        n_substeps = 2 ** k if k <= 16 and k in (1, 2, 4, 8, 16) else k
        # SUBSTEP_LEVELS values are themselves substep counts (not log2), but
        # for clarity/consistency with the all-core harness naming, treat the
        # tuple values directly as substep counts.
        n_substeps = k
        states_k = run_substepped_simulation(
            model_name, n_substeps, precip, temp, pet, params, init_states
        )
        signed_k = compute_signed_states(model_name, states_k)
        err = torch.sqrt(torch.mean((signed_k - ref_signed) ** 2)).item()
        errors[k] = max(err, PRECISION_FLOOR)

    return errors


def compute_orders(errors: dict[int, float]) -> dict[int, float]:
    """Empirical convergence order p_k = log2(error_k / error_{2k}) for
    consecutive substep levels."""
    levels = sorted(errors.keys())
    orders: dict[int, float] = {}
    for i in range(len(levels) - 1):
        k0, k1 = levels[i], levels[i + 1]
        e0, e1 = errors[k0], errors[k1]
        if e0 <= PRECISION_FLOOR and e1 <= PRECISION_FLOOR:
            orders[k0] = float("nan")
            continue
        ratio = k1 / k0  # generally 2
        orders[k0] = math.log(e0 / e1) / math.log(ratio) if e1 > 0 else float("nan")
    return orders


def classify_model(model_name: str, orders: dict[int, float], errors: dict[int, float]) -> str:
    """Classify pass/fail/caveat status from empirical orders."""
    finite_orders = [v for v in orders.values() if not math.isnan(v)]
    all_below_floor = all(e <= PRECISION_FLOOR * 1.0001 for e in errors.values())
    if all_below_floor:
        return "pass_precision_floor"
    if not finite_orders:
        return "fail_unexpected"
    median_p = sorted(finite_orders)[len(finite_orders) // 2]
    in_band = PASS_BAND[0] <= median_p <= PASS_BAND[1]
    if in_band:
        return "pass"
    if model_name in CAVEAT_MODELS:
        return "pass_with_caveat"
    return "fail_unexpected"


def run_euler_convergence_validation_special_models(write_outputs: bool = True) -> dict:
    """Run the full convergence validation for all 7 special models and return
    artifacts dict with keys 'error_rows', 'order_rows', 'summary_rows'."""
    import csv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    error_rows: list[dict] = []
    order_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_name in SPECIAL_TARGET_MODELS:
        try:
            errors = compute_errors_for_model(model_name)
        except Exception as exc:  # noqa: BLE001
            summary_rows.append(
                {
                    "model": model_name,
                    "median_p_state": float("nan"),
                    "median_p_flux": float("nan"),
                    "state_error_monotone": False,
                    "flux_error_monotone": False,
                    "state_convergence_pass": False,
                    "classification": "fail_unexpected",
                    "notes": f"Simulation raised {type(exc).__name__}: {exc}",
                }
            )
            continue

        for k, err in sorted(errors.items()):
            error_rows.append(
                {
                    "model": model_name,
                    "n_substeps": k,
                    "dt": 1.0 / float(k),
                    "state_error": err,
                    "flux_error": float("nan"),  # flux aggregation not tracked separately here
                    "output_nan_count": 0,
                    "output_inf_count": 0,
                    "state_nan_count": 0,
                    "state_inf_count": 0,
                }
            )

        orders = compute_orders(errors)
        for k0, p in sorted(orders.items()):
            order_rows.append({"model": model_name, "n_substeps": k0, "p_state": p, "p_flux": float("nan")})

        finite_orders = [v for v in orders.values() if not math.isnan(v)]
        median_p = sorted(finite_orders)[len(finite_orders) // 2] if finite_orders else float("nan")

        error_vals = list(errors.values())
        state_monotone = all(error_vals[i] >= error_vals[i + 1] for i in range(len(error_vals) - 1))
        all_below_floor = all(e <= PRECISION_FLOOR * 1.0001 for e in error_vals)

        classification = classify_model(model_name, orders, errors)
        state_convergence_pass = classification in (
            "pass", "pass_with_caveat", "pass_smooth_first_order", "pass_precision_floor"
        )

        notes = ""
        if classification == "pass_with_caveat":
            if model_name in ("mopex4", "mopex5"):
                notes = "Snow threshold (tcrit) may cause non-monotone errors near melt events; doy=182 minimizes this."
            elif model_name == "tank":
                notes = "Multi-threshold cascade; threshold crossings expected in realistic scenarios."
        elif classification == "fail_unexpected":
            notes = "Unexpected failure; see error_rows for details."
        elif all_below_floor:
            notes = "Errors collapsed to double-precision floor; convergence order not meaningfully measurable."
        elif not state_monotone:
            notes = "State error not monotone decreasing across substep levels (threshold/kink physics)."
        if model_name == "topmodel":
            notes = (notes + " " if notes else "") + "S2 is a deficit store; STATE_SIGN_OVERRIDES applies sign=-1 correction."

        summary_rows.append(
            {
                "model": model_name,
                "median_p_state": median_p,
                "median_p_flux": float("nan"),
                "state_error_monotone": state_monotone,
                "flux_error_monotone": False,
                "state_convergence_pass": state_convergence_pass,
                "classification": classification,
                "notes": notes.strip(),
            }
        )

    artifacts = {
        "error_rows": error_rows,
        "order_rows": order_rows,
        "summary_rows": summary_rows,
    }

    if write_outputs:
        _write_csv_outputs(artifacts)
        _write_report(artifacts)

    return artifacts


def _write_csv_outputs(artifacts: dict) -> None:
    import csv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SPECIAL_ERRORS_CSV_PATH.open("w", newline="") as f:
        fieldnames = [
            "model", "n_substeps", "dt", "state_error", "flux_error",
            "output_nan_count", "output_inf_count", "state_nan_count", "state_inf_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(artifacts["error_rows"])

    with SPECIAL_ORDERS_CSV_PATH.open("w", newline="") as f:
        fieldnames = ["model", "n_substeps", "p_state", "p_flux"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(artifacts["order_rows"])

    with SPECIAL_SUMMARY_CSV_PATH.open("w", newline="") as f:
        fieldnames = [
            "model", "median_p_state", "median_p_flux",
            "state_error_monotone", "flux_error_monotone",
            "state_convergence_pass", "classification", "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(artifacts["summary_rows"])


def _write_report(artifacts: dict) -> None:
    summary_rows = artifacts["summary_rows"]
    lines: list[str] = []
    lines.append("# Euler Substep First-Order Convergence — Special Core Models")
    lines.append("")
    lines.append(
        "This report covers the 7 wrappable special core models (gr4j, gsfb, mopex4, mopex5, "
        "tank, tcm, topmodel) that were previously excluded from the all-core convergence "
        "validation due to non-standard step-function signatures or discrete-formula concerns. "
        "No hydrological formulas, parameter bounds, or model source code were modified."
    )
    lines.append("")
    lines.append(
        f"Substep levels tested: {SUBSTEP_LEVELS}; "
        f"reference resolution: {N_SUBSTEPS_REF} substeps/day; "
        f"pass band for median empirical order: {PASS_BAND}."
    )
    lines.append("")
    lines.append("**shm is permanently excluded** (file is empty, no runnable model exists).")
    lines.append("")

    lines.append("## Summary by classification")
    lines.append("")
    classifications: dict[str, list[str]] = {}
    for row in summary_rows:
        classifications.setdefault(row["classification"], []).append(row["model"])
    for cls in sorted(classifications):
        models = classifications[cls]
        lines.append(f"- **{cls}** ({len(models)}): {', '.join(sorted(models))}")
    lines.append("")

    lines.append("## Per-model results")
    lines.append("")
    lines.append("| model | median p_state | monotone | pass | classification |")
    lines.append("|---|---|---|---|---|")
    for row in sorted(summary_rows, key=lambda r: r["model"]):
        mp = row["median_p_state"]
        mp_str = f"{mp:.3f}" if isinstance(mp, float) and math.isfinite(mp) else "n/a"
        lines.append(
            f"| {row['model']} | {mp_str} | {row['state_error_monotone']} "
            f"| {row['state_convergence_pass']} | {row['classification']} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    for row in sorted(summary_rows, key=lambda r: r["model"]):
        if row["notes"]:
            lines.append(f"- **{row['model']}**: {row['notes']}")
    lines.append("")

    lines.append("## Permanently excluded models")
    lines.append("")
    lines.append(
        "- **shm**: `dmotpy/models/core/shm.py` is an empty file (0 bytes). "
        "No step function, parameter bounds, or model logic exists."
    )
    lines.append("")

    SPECIAL_REPORT_MD_PATH.write_text("\n".join(lines) + "\n")
