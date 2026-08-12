from __future__ import annotations

"""Euler substep first-order convergence utilities for ALL runnable core models.

This module extends the representative-model convergence harness
(`tests/euler_convergence_utils.py`, covering hbv96/hymod/flexb/vic) to every
core model classified as substep-feasible by
`scripts/review_euler_substep_feasibility_all_core.py`.

Design constraints (must hold for every model added here):
  * No hydrological formulas are modified, smoothed, or clamped.
  * No parameter bounds, soft-gate defaults, or unit-hydrograph code are changed.
  * Each model is exercised through its *unmodified* `<model>_step` function from
    `dmotpy/models/core`; only the calling harness (zero-order-hold forcing,
    dt-scaled rate parameters, sub-daily looping) is new test/script code.
  * Excluded models (gr4j, gsfb, mopex4, mopex5, shm, tank, tcm, topmodel) are not
    exercised here; see the feasibility CSV/MD for reasons.

Convergence procedure (identical to the representative harness):
  * Daily forcing (precip, temp, pet) is smooth/synthetic and held constant
    (zero-order-hold) within each day while substepping.
  * For substep level k in SUBSTEP_LEVELS, the day is split into 2**k substeps of
    width dt = 1 / 2**k; rate parameters are scaled by dt; all other parameters
    are left untouched (they are storage-capacity-like, not per-time-unit rates).
  * A fine reference run uses N_SUBSTEPS_REF substeps (k_ref = 10).
  * Per-day-aggregated outputs/states at each k are compared against the
    reference; the empirical order is p_k = log2(error_k / error_{k+1}).
"""

import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from tests.core_model_registry import CORE_MODEL_REGISTRY, CoreModelEntry


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "validation_results" / "euler_convergence_all_core"
ALL_CORE_ERRORS_CSV_PATH = OUTPUT_DIR / "euler_all_core_convergence_errors.csv"
ALL_CORE_ORDERS_CSV_PATH = OUTPUT_DIR / "euler_all_core_convergence_orders.csv"
ALL_CORE_SUMMARY_CSV_PATH = OUTPUT_DIR / "euler_all_core_convergence_summary.csv"
ALL_CORE_REPORT_MD_PATH = OUTPUT_DIR / "euler_all_core_report.md"

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

# Models excluded from the dt-substep convergence test (see feasibility review).
EXCLUDED_MODELS = frozenset(
    {"gr4j", "gsfb", "mopex4", "mopex5", "shm", "tank", "tcm"}
)

# Models classified as substep-feasible and exercised by this harness.
# (Representative TARGET_MODELS from euler_convergence_utils.py are included here
# too, so the all-core report is comprehensive on its own; the original
# representative test file/utility module is left untouched.)
ALL_CORE_TARGET_MODELS = tuple(
    sorted(
        name
        for name, entry in CORE_MODEL_REGISTRY.items()
        if entry.enabled and name not in EXCLUDED_MODELS
    )
)

# Models supported but with a documented caveat (still run through the same
# convergence test; caveat is recorded in notes/classification only).
CAVEAT_MODELS = frozenset(
    {
        "alpine1",
        "alpine2",
        "australia",   # threshold crossing (median_p=1.711, outside band)
        "collie1",     # precision floor — errors collapse to float64 noise
        "collie2",     # precision floor — errors collapse to float64 noise
        "collie3",     # threshold crossing (median_p=0.587)
        "flexis",
        "hbv96",       # non-monotone state errors due to snow/threshold physics
        "ihacres",
        "modhydrolog",
        "mopex2",
        "mopex3",
        "smar",
        "susannah1",   # threshold crossing (median_p=1.792)
        "topmodel",    # deficit-store structure; dt-aware wrapper designed for this model
        "us1",         # threshold crossing (median_p=2.051)
        "vic",         # threshold crossing (median_p=4.903)
    }
)

# Rate parameters (per unit time) that must be scaled by dt when substepping.
# Storage capacities, shape exponents, split fractions, and thresholds are NOT
# rate parameters and are left unscaled.
RATE_PARAMETERS: dict[str, frozenset[str]] = {
    "alpine1": frozenset({"ddf", "tc"}),
    "alpine2": frozenset({"ddf", "tcin", "tcbf"}),
    "australia": frozenset({"alpha_ss", "k_deep", "alpha_bf"}),
    "collie1": frozenset(),
    "collie2": frozenset({"a"}),
    "collie3": frozenset({"a"}),
    "flexb": frozenset({"percmax", "kf", "ks"}),
    "flexi": frozenset({"percmax", "kf", "ks"}),
    "flexis": frozenset({"ddf", "percmax", "kf", "ks"}),
    "hbv96": frozenset({"cfmax", "cflux", "k0", "perc", "k1"}),
    "hillslope": frozenset({"a", "kh"}),
    "hymod": frozenset({"kf", "ks"}),
    "ihacres": frozenset({"alpha", "tau_q", "tau_s"}),
    "modhydrolog": frozenset({"crak", "vcond", "k1", "k2", "k3"}),
    "mopex1": frozenset({"tw", "tu", "tc"}),
    "mopex2": frozenset({"ddf", "tw", "tu", "tc"}),
    "mopex3": frozenset({"ddf", "tw", "tu", "tc"}),
    "newzealand1": frozenset({"m", "tcbf"}),
    "newzealand2": frozenset({"m", "tcbf"}),
    "penman": frozenset({"k1"}),
    "plateau": frozenset({"dp", "tp", "kp"}),
    "simhyd": frozenset({"crak", "k"}),
    "smar": frozenset({"kg"}),
    "susannah1": frozenset({"r"}),
    "susannah2": frozenset({"r"}),
    "topmodel": frozenset({"kd", "q0"}),
    "us1": frozenset(),
    "vic": frozenset({"k1", "k2"}),
    "wetland": frozenset({"kw"}),
    "xinanjiang": frozenset({"ki", "kg", "ci", "cg"}),
}


def _dtype_device_kwargs() -> dict[str, Any]:
    return {"dtype": DTYPE, "device": DEVICE}


def _tensor(value: float) -> torch.Tensor:
    return torch.full((N_GRID, N_MUL), float(value), **_dtype_device_kwargs())


def build_smooth_forcing() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth synthetic daily forcing, identical shape/spirit to the
    representative-model harness (kept deliberately simple and reusable)."""
    day = torch.arange(N_DAYS, **_dtype_device_kwargs()).view(N_DAYS, 1, 1)
    angle = 2.0 * math.pi * day / float(N_DAYS)
    precip = 4.5 + 0.8 * torch.sin(angle) + 0.3 * torch.cos(2.0 * angle)
    pet = 1.7 + 0.25 * torch.cos(angle - 0.4) + 0.1 * torch.sin(2.0 * angle)
    temp = 12.5 + 1.2 * torch.sin(angle + 0.3)
    return precip, temp, pet


# Interior parameter values (mid-bound, physically reasonable) for every
# substep-feasible model. Values are chosen to keep states comfortably inside
# capacity bounds across all SUBSTEP_LEVELS and the reference resolution.
def build_interior_parameters(model_name: str) -> dict[str, torch.Tensor]:
    table: dict[str, dict[str, float]] = {
        "alpine1": {"tt": 0.5, "ddf": 2.5, "Smax": 300.0, "tc": 0.08},
        "alpine2": {
            "tt": 0.5,
            "ddf": 2.5,
            "Smax": 300.0,
            "Cfc": 0.6,
            "tcin": 0.15,
            "tcbf": 0.05,
        },
        "australia": {
            "sb": 250.0,
            "phi": 0.55,
            "fc_frac": 0.6,
            "alpha_ss": 0.12,
            "beta_ss": 1.5,
            "k_deep": 0.04,
            "alpha_bf": 0.03,
            "beta_bf": 1.3,
        },
        "collie1": {"Smax": 220.0},
        "collie2": {"Smax": 220.0, "Sfc_frac": 0.55, "a": 0.10, "M": 1.4},
        "collie3": {
            "smax": 220.0,
            "fc": 0.55,
            "a": 0.10,
            "m": 1.4,
            "b": 1.6,
            "lambda_par": 0.3,
        },
        "flexb": {
            "s1max": 280.0,
            "beta": 1.6,
            "d_split": 0.4,
            "percmax": 0.7,
            "lp": 0.6,
            "nlagf": 2.0,
            "nlags": 6.0,
            "kf": 0.10,
            "ks": 0.035,
        },
        "flexi": {
            "smax": 280.0,
            "beta": 1.6,
            "d_split": 0.4,
            "percmax": 0.7,
            "lp": 0.6,
            "nlagf": 2.0,
            "nlags": 6.0,
            "kf": 0.10,
            "ks": 0.035,
            "imax": 2.0,
        },
        "flexis": {
            "smax": 280.0,
            "beta": 1.6,
            "d_split": 0.4,
            "percmax": 0.7,
            "lp": 0.6,
            "nlagf": 2.0,
            "nlags": 6.0,
            "kf": 0.10,
            "ks": 0.035,
            "imax": 2.0,
            "tt": 0.5,
            "ddf": 2.5,
        },
        "hbv96": {
            "tt": 0.5,
            "tti": 4.0,
            "ttm": 0.0,
            "cfr": 0.05,
            "cfmax": 2.0,
            "whc": 0.08,
            "cflux": 0.2,
            "fc": 300.0,
            "lp": 0.6,
            "beta": 2.0,
            "k0": 0.08,
            "alpha": 1.0,
            "perc": 0.6,
            "k1": 0.04,
            "maxbas": 5.0,
        },
        "hillslope": {
            "dw": 0.3,
            "betaw": 1.4,
            "swmax": 260.0,
            "a": 0.08,
            "th": 0.5,
            "c_rad": 0.6,
            "kh": 0.03,
        },
        "hymod": {
            "smax": 250.0,
            "b_exp": 1.5,
            "a_split": 0.45,
            "kf": 0.12,
            "ks": 0.04,
        },
        "ihacres": {
            "lp": 0.5,
            "d": 200.0,
            "p": 1.5,
            "alpha": 0.3,
            "tau_q": 2.0,
            "tau_s": 30.0,
        },
        "modhydrolog": {
            "insc": 1.5,
            "coeff": 200.0,
            "sq": 2.0,
            "smsc": 250.0,
            "sub": 0.4,
            "crak": 0.1,
            "em": 1.5,
            "dsc": 5.0,
            "ads": 0.3,
            "md": 10.0,
            "vcond": 0.05,
            "dlev": 2.0,
            "k1": 0.3,
            "k2": 0.1,
            "k3": 0.05,
        },
        "mopex1": {"s1max": 280.0, "tw": 0.2, "tu": 0.5, "se": 0.3, "tc": 0.1},
        "mopex2": {
            "tcrit": 0.5,
            "ddf": 2.5,
            "s2max": 280.0,
            "tw": 0.2,
            "tu": 0.5,
            "se": 0.3,
            "tc": 0.1,
        },
        "mopex3": {
            "tcrit": 0.5,
            "ddf": 2.5,
            "s2max": 280.0,
            "tw": 0.2,
            "tu": 0.5,
            "se": 0.3,
            "s3max": 150.0,
            "tc": 0.1,
        },
        "newzealand1": {
            "s1max": 240.0,
            "sfc_frac": 0.55,
            "m": 1.4,
            "a": 0.2,
            "b": 1.5,
            "tcbf": 0.04,
        },
        "newzealand2": {
            "s1max": 60.0,
            "s2max": 240.0,
            "sfc_frac": 0.55,
            "m": 1.4,
            "a": 0.2,
            "b": 1.5,
            "tcbf": 0.04,
            "d_delay": 0.5,
        },
        "penman": {"smax": 250.0, "phi": 0.5, "gam": 0.6, "k1": 0.05},
        "plateau": {
            "fmax": 0.6,
            "dp": 0.1,
            "sumax": 260.0,
            "lp": 0.6,
            "p_coeff": 1.5,
            "tp": 0.06,
            "c_rise": 0.4,
            "kp": 0.03,
        },
        "simhyd": {
            "insc": 1.5,
            "coeff": 200.0,
            "sq": 2.0,
            "smsc": 250.0,
            "sub": 0.4,
            "crak": 0.1,
            "k": 0.3,
        },
        "smar": {
            "h_runoff": 0.05,
            "y_inf": 50.0,
            "smax": 250.0,
            "c_evap": 0.3,
            "g_rech": 0.1,
            "kg": 0.04,
            "n_res": 3.0,
            "nk_delay": 2.0,
        },
        "susannah1": {
            "sb": 250.0,
            "sfc_frac": 0.55,
            "m": 1.4,
            "a": 0.2,
            "b": 1.5,
            "r": 0.04,
        },
        "susannah2": {
            "sb": 250.0,
            "phi": 0.55,
            "fc": 0.6,
            "r": 0.04,
            "c": 1.5,
            "d": 0.3,
        },
        "susannah2": {
            "sb": 250.0,
            "phi": 0.55,
            "fc": 0.6,
            "r": 0.04,
            "c": 1.5,
            "d": 0.3,
        },
        "topmodel": {
            "suzmax": 250.0,
            "st": 0.4,
            "kd": 0.08,
            "q0": 5.0,
            "f": 0.15,
            "chi": 3.5,
            "phi": 1.2,
        },
        "us1": {"alpha_ei": 0.2, "m": 1.4, "smax": 250.0, "fc": 0.55, "alpha_ss": 0.1},
        "vic": {
            "ibar": 1.5,
            "idelta": 0.2,
            "ishift": 180.0,
            "stot": 420.0,
            "fsm": 0.55,
            "b": 1.6,
            "k1": 0.07,
            "c1": 1.5,
            "k2": 0.035,
            "c2": 1.4,
        },
        "wetland": {"dw": 0.3, "betaw": 1.4, "swmax": 220.0, "kw": 0.04},
        "xinanjiang": {
            "aim": 0.05,
            "par_a": 0.3,
            "par_b": 1.5,
            "stot": 280.0,
            "fwm": 0.2,
            "flm": 0.6,
            "par_c": 0.15,
            "ex": 1.4,
            "ki": 0.2,
            "kg": 0.06,
            "ci": 0.6,
            "cg": 0.04,
        },
    }
    if model_name not in table:
        raise KeyError(model_name)
    return {name: _tensor(value) for name, value in table[model_name].items()}


def build_initial_states(model_name: str, params: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Mid-capacity initial states, one builder per model, matching each
    model's actual init_fn state count and ordering."""
    p = params
    if model_name == "alpine1":
        return [_tensor(1.0), p["Smax"] * 0.35]
    if model_name == "alpine2":
        return [_tensor(1.0), p["Smax"] * 0.35]
    if model_name == "australia":
        return [p["sb"] * 0.3, p["sb"] * 0.1, _tensor(40.0)]
    if model_name == "collie1":
        return [p["Smax"] * 0.35]
    if model_name == "collie2":
        return [p["Smax"] * 0.35]
    if model_name == "collie3":
        return [p["smax"] * 0.35, _tensor(40.0)]
    if model_name == "flexb":
        return [p["s1max"] * 0.36, _tensor(4.0), _tensor(18.0)]
    if model_name == "flexi":
        return [_tensor(0.5), p["smax"] * 0.36, _tensor(4.0), _tensor(18.0)]
    if model_name == "flexis":
        return [_tensor(1.0), _tensor(0.5), p["smax"] * 0.36, _tensor(4.0), _tensor(18.0)]
    if model_name == "hbv96":
        return [
            _tensor(1.0),
            _tensor(0.4),
            p["fc"] * 0.32,
            _tensor(18.0),
            _tensor(40.0),
        ]
    if model_name == "hillslope":
        return [p["swmax"] * 0.32, _tensor(40.0)]
    if model_name == "hymod":
        return [p["smax"] * 0.35, _tensor(3.0), _tensor(2.4), _tensor(1.8), _tensor(16.0)]
    if model_name == "ihacres":
        return [_tensor(p["d"].item() * 0.4)]
    if model_name == "modhydrolog":
        return [_tensor(0.5), p["smsc"] * 0.32, _tensor(20.0), _tensor(40.0), _tensor(60.0)]
    if model_name == "mopex1":
        return [p["s1max"] * 0.34, _tensor(40.0), _tensor(2.0), _tensor(1.0)]
    if model_name == "mopex2":
        return [_tensor(1.0), p["s2max"] * 0.34, _tensor(40.0), _tensor(2.0), _tensor(1.0)]
    if model_name == "mopex3":
        return [
            _tensor(1.0),
            p["s2max"] * 0.34,
            p["s3max"] * 0.30,
            _tensor(2.0),
            _tensor(1.0),
        ]
    if model_name == "newzealand1":
        return [p["s1max"] * 0.35]
    if model_name == "newzealand2":
        return [p["s1max"] * 0.5, p["s2max"] * 0.35]
    if model_name == "penman":
        return [p["smax"] * 0.35, p["smax"] * 0.30, _tensor(40.0)]
    if model_name == "plateau":
        return [p["sumax"] * 0.32, _tensor(40.0)]
    if model_name == "simhyd":
        return [p["smsc"] * 0.32, _tensor(40.0)]
    if model_name == "smar":
        return [
            p["smax"] * 0.2,
            p["smax"] * 0.2,
            p["smax"] * 0.2,
            p["smax"] * 0.2,
            p["smax"] * 0.2,
            _tensor(20.0),
        ]
    if model_name == "susannah1":
        return [p["sb"] * 0.32, _tensor(40.0)]
    if model_name == "susannah2":
        return [p["sb"] * 0.3, p["sb"] * 0.1]
    if model_name == "topmodel":
        return [p["suzmax"] * 0.35, _tensor(50.0)]
    if model_name == "us1":
        return [p["smax"] * 0.32, _tensor(40.0)]
    if model_name == "vic":
        smmax = p["fsm"] * p["stot"]
        gwmax = (1.0 - p["fsm"]) * p["stot"]
        return [p["ibar"] * 0.35, smmax * 0.32, gwmax * 0.28]
    if model_name == "wetland":
        return [p["swmax"] * 0.32]
    if model_name == "xinanjiang":
        return [p["fwm"] * 0.3, p["flm"] * 0.3, _tensor(40.0), _tensor(20.0)]
    raise KeyError(model_name)


def _step_fn_uses_temp(step_fn: Callable) -> bool:
    """Return True if `step_fn`'s signature includes a temperature argument
    named exactly 'T'. Many core models accept T even when they are not
    classified as snow models (uses_snow=False); this detection is purely
    structural (signature inspection) and does not alter any model code."""
    try:
        sig = inspect.signature(step_fn)
    except (TypeError, ValueError):
        return False
    return "T" in sig.parameters


def _scaled_parameter_list(entry: CoreModelEntry, params: dict[str, torch.Tensor], dt: float) -> list[torch.Tensor]:
    rate_names = RATE_PARAMETERS.get(entry.model_name, frozenset())
    scaled: list[torch.Tensor] = []
    for name in entry.param_bounds:
        value = params[name]
        if name in rate_names:
            value = value * dt
        scaled.append(value)
    return scaled


@dataclass(frozen=True)
class SimulationResult:
    model: str
    n_substeps: int
    dt: float
    state_daily: torch.Tensor
    flux_daily: torch.Tensor
    output_nan_count: int
    output_inf_count: int
    state_nan_count: int
    state_inf_count: int


def simulate_with_substeps(model_name: str, n_substeps: int) -> SimulationResult:
    """Run `model_name` with the daily forcing held constant (zero-order-hold)
    across `n_substeps` Euler substeps per day, using the model's own
    unmodified `<model>_step` function from dmotpy/models/core.

    Returns per-day aggregated state (end-of-day) and flux (sum-over-day,
    i.e. daily total) tensors of shape (N_DAYS, n_states) / (N_DAYS, n_flux).
    """
    entry = CORE_MODEL_REGISTRY[model_name]
    if not entry.enabled or model_name in EXCLUDED_MODELS:
        raise KeyError(model_name)

    dt = 1.0 / float(n_substeps)
    precip, temp, pet = build_smooth_forcing()
    params = build_interior_parameters(model_name)
    states = build_initial_states(model_name, params)
    scaled_params = _scaled_parameter_list(entry, params, dt)

    n_states = len(states)
    state_daily = torch.zeros((N_DAYS, n_states), **_dtype_device_kwargs())
    flux_daily_list: list[torch.Tensor] = []

    output_nan_count = 0
    output_inf_count = 0
    state_nan_count = 0
    state_inf_count = 0

    for day in range(N_DAYS):
        p_day = precip[day]
        t_day = temp[day]
        e_day = pet[day]
        flux_accum: torch.Tensor | None = None
        for _ in range(n_substeps):
            if _step_fn_uses_temp(entry.step_fn):
                step_args = (p_day, t_day, e_day, *scaled_params, *states)
            else:
                step_args = (p_day, e_day, *scaled_params, *states)
            outputs = entry.step_fn(*step_args)
            outputs = list(outputs)
            new_states = outputs[:n_states]
            flux_outputs = outputs[n_states:]

            flux_stack = torch.cat([torch.as_tensor(f, **_dtype_device_kwargs()).reshape(-1) for f in flux_outputs])
            output_nan_count += int(torch.isnan(flux_stack).sum().item())
            output_inf_count += int(torch.isinf(flux_stack).sum().item())

            states_stack = torch.cat([torch.as_tensor(s, **_dtype_device_kwargs()).reshape(-1) for s in new_states])
            state_nan_count += int(torch.isnan(states_stack).sum().item())
            state_inf_count += int(torch.isinf(states_stack).sum().item())

            # Flux outputs are instantaneous rates over the substep; integrate
            # (sum, scaled by dt) to accumulate a daily total, consistent with
            # zero-order-hold forcing and explicit Euler substepping.
            flux_dt = flux_stack * dt
            flux_accum = flux_dt if flux_accum is None else flux_accum + flux_dt

            states = list(new_states)

        state_daily[day] = torch.stack([torch.as_tensor(s, **_dtype_device_kwargs()).reshape(()) for s in states])
        flux_daily_list.append(flux_accum)

    flux_daily = torch.stack(flux_daily_list, dim=0)

    return SimulationResult(
        model=model_name,
        n_substeps=n_substeps,
        dt=dt,
        state_daily=state_daily,
        flux_daily=flux_daily,
        output_nan_count=output_nan_count,
        output_inf_count=output_inf_count,
        state_nan_count=state_nan_count,
        state_inf_count=state_inf_count,
    )


def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.clamp(torch.abs(b), min=NEARZERO)
    return float(torch.max(torch.abs(a - b) / denom).item())


def _classify(
    model_name: str,
    state_errors: list[float],
    median_p_state: float,
    state_monotone: bool,
    has_bad_values: bool,
    min_error: float,
) -> str:
    if has_bad_values:
        return "fail_unexpected"
    if min_error < PRECISION_FLOOR:
        return "fail_due_to_precision_floor"
    if not state_monotone:
        return "fail_due_to_threshold_crossing"
    if PASS_BAND[0] <= median_p_state <= PASS_BAND[1]:
        if model_name in CAVEAT_MODELS:
            return "pass_with_caveat"
        return "pass_smooth_first_order"
    return "fail_due_to_threshold_crossing"


def run_euler_convergence_validation_all_core(write_outputs: bool = True) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    error_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name in ALL_CORE_TARGET_MODELS:
        try:
            reference = simulate_with_substeps(model_name, N_SUBSTEPS_REF)
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
                    "notes": f"Reference simulation raised {type(exc).__name__}: {exc}",
                }
            )
            continue

        results: list[SimulationResult] = []
        has_bad_values = (
            reference.output_nan_count > 0
            or reference.output_inf_count > 0
            or reference.state_nan_count > 0
            or reference.state_inf_count > 0
        )
        for n_substeps in SUBSTEP_LEVELS:
            try:
                res = simulate_with_substeps(model_name, n_substeps)
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
                        "notes": f"Substep k simulation raised {type(exc).__name__}: {exc}",
                    }
                )
                results = []
                break
            has_bad_values = (
                has_bad_values
                or res.output_nan_count > 0
                or res.output_inf_count > 0
                or res.state_nan_count > 0
                or res.state_inf_count > 0
            )
            results.append(res)

            error_rows.append(
                {
                    "model": model_name,
                    "n_substeps": n_substeps,
                    "dt": res.dt,
                    "state_error": _rel_error(res.state_daily, reference.state_daily),
                    "flux_error": _rel_error(res.flux_daily, reference.flux_daily),
                    "output_nan_count": res.output_nan_count,
                    "output_inf_count": res.output_inf_count,
                    "state_nan_count": res.state_nan_count,
                    "state_inf_count": res.state_inf_count,
                }
            )

        if not results:
            continue

        state_errors = [_rel_error(r.state_daily, reference.state_daily) for r in results]
        flux_errors = [_rel_error(r.flux_daily, reference.flux_daily) for r in results]

        state_orders = [
            math.log2(state_errors[i] / state_errors[i + 1])
            if state_errors[i + 1] > 0 and state_errors[i] > 0
            else float("nan")
            for i in range(len(state_errors) - 1)
        ]
        flux_orders = [
            math.log2(flux_errors[i] / flux_errors[i + 1])
            if flux_errors[i + 1] > 0 and flux_errors[i] > 0
            else float("nan")
            for i in range(len(flux_errors) - 1)
        ]
        for level_idx, (n_substeps, p_state, p_flux) in enumerate(
            zip(SUBSTEP_LEVELS[:-1], state_orders, flux_orders)
        ):
            order_rows.append(
                {
                    "model": model_name,
                    "n_substeps": n_substeps,
                    "p_state": p_state,
                    "p_flux": p_flux,
                }
            )

        finite_state_orders = [p for p in state_orders if math.isfinite(p)]
        finite_flux_orders = [p for p in flux_orders if math.isfinite(p)]
        median_p_state = (
            float(torch.median(torch.tensor(finite_state_orders)).item()) if finite_state_orders else float("nan")
        )
        median_p_flux = (
            float(torch.median(torch.tensor(finite_flux_orders)).item()) if finite_flux_orders else float("nan")
        )

        state_monotone = all(
            state_errors[i] >= state_errors[i + 1] for i in range(len(state_errors) - 1)
        )
        flux_monotone = all(
            flux_errors[i] >= flux_errors[i + 1] for i in range(len(flux_errors) - 1)
        )

        min_error = min(state_errors) if state_errors else float("nan")
        state_convergence_pass = (
            not has_bad_values
            and state_monotone
            and math.isfinite(median_p_state)
            and PASS_BAND[0] <= median_p_state <= PASS_BAND[1]
        )

        classification = _classify(
            model_name=model_name,
            state_errors=state_errors,
            median_p_state=median_p_state,
            state_monotone=state_monotone,
            has_bad_values=has_bad_values,
            min_error=min_error,
        )

        notes = ""
        if model_name in CAVEAT_MODELS and classification == "pass_with_caveat":
            notes = "Supported with caveat per feasibility review; convergence still verified at first order."
        elif has_bad_values:
            notes = "NaN/Inf encountered in outputs or states during substepping."
        elif min_error < PRECISION_FLOOR:
            notes = "Errors below double-precision floor; convergence order not meaningfully measurable."
        elif not state_monotone:
            notes = "State error not monotone decreasing across substep levels (possible threshold/kink)."

        summary_rows.append(
            {
                "model": model_name,
                "median_p_state": median_p_state,
                "median_p_flux": median_p_flux,
                "state_error_monotone": state_monotone,
                "flux_error_monotone": flux_monotone,
                "state_convergence_pass": state_convergence_pass,
                "classification": classification,
                "notes": notes,
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


def _write_csv_outputs(artifacts: dict[str, Any]) -> None:
    import csv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with ALL_CORE_ERRORS_CSV_PATH.open("w", newline="") as f:
        fieldnames = [
            "model",
            "n_substeps",
            "dt",
            "state_error",
            "flux_error",
            "output_nan_count",
            "output_inf_count",
            "state_nan_count",
            "state_inf_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(artifacts["error_rows"])

    with ALL_CORE_ORDERS_CSV_PATH.open("w", newline="") as f:
        fieldnames = ["model", "n_substeps", "p_state", "p_flux"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(artifacts["order_rows"])

    with ALL_CORE_SUMMARY_CSV_PATH.open("w", newline="") as f:
        fieldnames = [
            "model",
            "median_p_state",
            "median_p_flux",
            "state_error_monotone",
            "flux_error_monotone",
            "state_convergence_pass",
            "classification",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(artifacts["summary_rows"])


def _write_report(artifacts: dict[str, Any]) -> None:
    summary_rows = artifacts["summary_rows"]
    lines: list[str] = []
    lines.append("# Euler Substep First-Order Convergence — All Core Models")
    lines.append("")
    lines.append(
        "This report extends the representative-model convergence validation "
        "(hbv96, hymod, flexb, vic) to every core model classified as "
        "substep-feasible in `euler_all_core_substep_feasibility.md`. "
        "No hydrological formulas, parameter bounds, soft-gate defaults, or "
        "unit-hydrograph code were modified to produce these results."
    )
    lines.append("")
    lines.append(
        f"Substep levels tested: {SUBSTEP_LEVELS} (k = 0..{len(SUBSTEP_LEVELS) - 1}); "
        f"reference resolution: {N_SUBSTEPS_REF} substeps/day; "
        f"pass band for median empirical order: {PASS_BAND}."
    )
    lines.append("")

    classifications: dict[str, list[str]] = {}
    for row in summary_rows:
        classifications.setdefault(row["classification"], []).append(row["model"])

    lines.append("## Summary by classification")
    lines.append("")
    for classification in sorted(classifications):
        models = classifications[classification]
        lines.append(f"- **{classification}** ({len(models)}): {', '.join(sorted(models))}")
    lines.append("")

    lines.append("## Per-model results")
    lines.append("")
    lines.append("| model | median p_state | median p_flux | monotone | pass | classification |")
    lines.append("|---|---|---|---|---|---|")
    for row in sorted(summary_rows, key=lambda r: r["model"]):
        mp_state = row["median_p_state"]
        mp_flux = row["median_p_flux"]
        mp_state_str = f"{mp_state:.3f}" if isinstance(mp_state, float) and math.isfinite(mp_state) else "n/a"
        mp_flux_str = f"{mp_flux:.3f}" if isinstance(mp_flux, float) and math.isfinite(mp_flux) else "n/a"
        lines.append(
            f"| {row['model']} | {mp_state_str} | {mp_flux_str} | "
            f"{row['state_error_monotone']} | {row['state_convergence_pass']} | {row['classification']} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    for row in sorted(summary_rows, key=lambda r: r["model"]):
        if row["notes"]:
            lines.append(f"- **{row['model']}**: {row['notes']}")
    lines.append("")

    lines.append("## Excluded models")
    lines.append("")
    lines.append(
        "The following models are excluded from this convergence test "
        "(see `euler_all_core_substep_feasibility.md` for full reasons): "
        + ", ".join(sorted(EXCLUDED_MODELS))
        + "."
    )
    lines.append("")

    ALL_CORE_REPORT_MD_PATH.write_text("\n".join(lines) + "\n")
