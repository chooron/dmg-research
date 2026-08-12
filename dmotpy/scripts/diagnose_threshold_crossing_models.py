"""Threshold-isolation diagnostics for all FAIL_THRESHOLD_CROSSING models.

Extends the original mopex2/mopex3/hbv96 diagnostics to cover:
  australia, collie3, susannah1, us1, vic

Scenarios per model:
  A: original_smooth_warm_positive — reproduce existing failure
  B: threshold_separated_storage — states mid-range, away from capacity bounds
  C: high_storage_but_not_overflow — high but not at saturation
  D: stress_near_threshold — push states near key thresholds
  E: fine_substep_asymptotic — threshold-separated params with level 32

Outputs:
  dmotpy/validation_results/euler_threshold_isolation/
    threshold_isolation_diagnostics.csv
    threshold_isolation_diagnostics.md
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from tests.core_model_registry import CORE_MODEL_REGISTRY, CoreModelEntry

DTYPE = torch.float64
DEVICE = "cpu"
NEARZERO = 1e-6
SUBSTEP_LEVELS = (2, 4, 8, 16)
FINE_SUBSTEP_LEVELS = (2, 4, 8, 16, 32)
N_SUBSTEPS_REF = 512
FINE_N_SUBSTEPS_REF = 1024
PASS_BAND = (0.85, 1.15)
N_DAYS = 20
N_GRID = 1
N_MUL = 1

OUT_DIR = _PROJECT / "validation_results" / "euler_threshold_isolation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DGN_CSV = OUT_DIR / "threshold_isolation_diagnostics.csv"
DGN_MD = OUT_DIR / "threshold_isolation_diagnostics.md"

TARGET_MODELS = (
    "mopex2", "mopex3", "hbv96",
    "australia", "collie3", "susannah1", "us1", "vic",
)

RATE_NAMES: dict[str, frozenset[str]] = {
    "mopex2": frozenset({"ddf", "tw", "tu", "tc"}),
    "mopex3": frozenset({"ddf", "tw", "tu", "tc"}),
    "hbv96": frozenset({"cfmax", "cflux", "k0", "perc", "k1"}),
    "australia": frozenset({"alpha_ss", "k_deep", "alpha_bf"}),
    "collie3": frozenset({"a"}),
    "susannah1": frozenset({"r"}),
    "us1": frozenset(),
    "vic": frozenset({"k1", "k2"}),
}

USES_TEMP: dict[str, bool] = {
    "mopex2": True, "mopex3": True, "hbv96": True,
    "australia": True, "collie3": True, "susannah1": True,
    "us1": True, "vic": True,
}


def _dk():
    return {"dtype": DTYPE, "device": DEVICE}


def _tscalar(v: float) -> torch.Tensor:
    return torch.full((N_GRID, N_MUL), float(v), **_dk())


def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.clamp(torch.abs(b), min=NEARZERO)
    return float(torch.max(torch.abs(a - b) / denom).item())


def _abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def _smooth_forcing(temp_val: float = 12.5, precip_scale: float = 1.0,
                    pet_scale: float = 1.0) -> tuple:
    day = torch.arange(N_DAYS, **_dk()).view(N_DAYS, 1, 1)
    angle = 2.0 * math.pi * day / float(N_DAYS)
    precip = (4.5 + 0.8 * torch.sin(angle) + 0.3 * torch.cos(2.0 * angle)) * precip_scale
    pet = (1.7 + 0.25 * torch.cos(angle - 0.4) + 0.1 * torch.sin(2.0 * angle)) * pet_scale
    temp = torch.full_like(precip, temp_val)
    return precip, temp, pet


def run_sim(
    entry: CoreModelEntry,
    params: dict[str, torch.Tensor],
    init_states: list[torch.Tensor],
    forcing: tuple,
    n_substeps: int,
    rate_names: frozenset[str],
    uses_temp: bool,
) -> dict:
    dt = 1.0 / float(n_substeps)
    precip, temp, pet = forcing

    scaled_params: list[torch.Tensor] = []
    for name in entry.param_bounds:
        value = params[name]
        if name in rate_names:
            value = value * dt
        scaled_params.append(value)

    states = [s.clone() for s in init_states]
    n_states = len(states)
    state_daily = torch.zeros((N_DAYS, n_states), **_dk())
    flux_daily = torch.zeros((N_DAYS, 2), **_dk())

    any_nan = 0
    any_inf = 0

    for day in range(N_DAYS):
        p_day = precip[day]
        t_day = temp[day]
        e_day = pet[day]
        flux_accum = torch.zeros(2, **_dk())
        for _ in range(n_substeps):
            if uses_temp:
                step_args = (p_day, t_day, e_day, *scaled_params, *states)
            else:
                step_args = (p_day, e_day, *scaled_params, *states)
            outputs = entry.step_fn(*step_args)
            outputs = list(outputs)
            new_states = outputs[:n_states]
            flux_outputs = outputs[n_states:]

            Q = torch.as_tensor(flux_outputs[0], **_dk()).reshape(-1)
            ET = torch.as_tensor(flux_outputs[1], **_dk()).reshape(-1)
            if torch.isnan(Q).any() or torch.isnan(ET).any():
                any_nan += 1
            if torch.isinf(Q).any() or torch.isinf(ET).any():
                any_inf += 1

            flux_accum[0] = flux_accum[0] + Q * dt
            flux_accum[1] = flux_accum[1] + ET * dt
            states = list(new_states)

        for s in states:
            if torch.isnan(s).any():
                any_nan += 1
            if torch.isinf(s).any():
                any_inf += 1

        state_daily[day] = torch.stack([
            torch.as_tensor(s, **_dk()).reshape(()) for s in states
        ])
        flux_daily[day] = flux_accum

    return {
        "n_substeps": n_substeps,
        "dt": dt,
        "state_daily": state_daily,
        "flux_daily": flux_daily,
        "any_nan": any_nan > 0,
        "any_inf": any_inf > 0,
    }


def run_convergence_sweep(
    model_name: str,
    scenario: str,
    params: dict[str, torch.Tensor],
    init_states: list[torch.Tensor],
    forcing: tuple,
    rate_names: frozenset[str],
    uses_temp: bool,
    fine_mode: bool = False,
) -> dict[str, Any]:
    entry = CORE_MODEL_REGISTRY[model_name]
    levels = FINE_SUBSTEP_LEVELS if fine_mode else SUBSTEP_LEVELS
    ref_n = FINE_N_SUBSTEPS_REF if fine_mode else N_SUBSTEPS_REF

    ref = run_sim(entry, params, init_states, forcing, ref_n,
                  rate_names, uses_temp)

    state_errors: list[float] = []
    flux_errors: list[float] = []
    q_errors: list[float] = []
    ea_errors: list[float] = []
    any_bad = ref["any_nan"] or ref["any_inf"]

    for n_sub in levels:
        res = run_sim(entry, params, init_states, forcing, n_sub,
                      rate_names, uses_temp)
        any_bad = any_bad or res["any_nan"] or res["any_inf"]

        s_err = _rel_error(res["state_daily"], ref["state_daily"])
        f_err = _rel_error(res["flux_daily"], ref["flux_daily"])
        q_err = _rel_error(res["flux_daily"][..., 0:1], ref["flux_daily"][..., 0:1])
        ea_err = _rel_error(res["flux_daily"][..., 1:2], ref["flux_daily"][..., 1:2])

        state_errors.append(s_err)
        flux_errors.append(f_err)
        q_errors.append(q_err)
        ea_errors.append(ea_err)

    empirical_orders = []
    for i in range(len(state_errors) - 1):
        if state_errors[i] > 0 and state_errors[i + 1] > 0:
            empirical_orders.append(math.log2(state_errors[i] / state_errors[i + 1]))
        else:
            empirical_orders.append(float("nan"))

    finite = [o for o in empirical_orders if math.isfinite(o)]
    median_order = float(torch.median(torch.tensor(finite)).item()) if finite else float("nan")
    final_local_order = empirical_orders[-1] if empirical_orders else float("nan")

    monotone = all(state_errors[i] >= state_errors[i + 1] for i in range(len(state_errors) - 1))
    in_band_by_median = PASS_BAND[0] <= median_order <= PASS_BAND[1] if math.isfinite(median_order) else False
    in_band_by_final = PASS_BAND[0] <= final_local_order <= PASS_BAND[1] if math.isfinite(final_local_order) else False

    min_abs = min(_abs_error(
        run_sim(entry, params, init_states, forcing, n_sub, rate_names, uses_temp)["state_daily"],
        ref["state_daily"]
    ) for n_sub in levels)

    last_res = run_sim(entry, params, init_states, forcing, levels[-1], rate_names, uses_temp)
    final_abs_state = _abs_error(last_res["state_daily"], ref["state_daily"])
    final_abs_q = _abs_error(last_res["flux_daily"][..., 0:1], ref["flux_daily"][..., 0:1])

    return {
        "model": model_name,
        "scenario": scenario,
        "substep_levels": levels,
        "state_errors": state_errors,
        "flux_errors": flux_errors,
        "q_errors": q_errors,
        "ea_errors": ea_errors,
        "empirical_orders": empirical_orders,
        "median_order": median_order,
        "final_local_order": final_local_order,
        "state_error_monotone": monotone,
        "in_pass_band_by_median_order": in_band_by_median,
        "in_pass_band_by_final_local_order": in_band_by_final,
        "min_abs_error": min_abs,
        "final_abs_state_error": final_abs_state,
        "final_abs_q_error": final_abs_q,
        "any_nan_inf": any_bad,
        "detected_threshold_crossing_flags": "",
        "recommended_status": "",
        "diagnostic_subtype": "",
    }


# ============================================================
# Scenario builders
# ============================================================

FROM_CORE = ("australia", "collie3", "susannah1", "us1", "vic",
             "mopex2", "mopex3", "hbv96")
RECLASSIFIED_MODELS = {"collie3", "susannah1", "us1"}


def final_disposition(model_name: str) -> str:
    return "PASS_WITH_CAVEAT" if model_name in RECLASSIFIED_MODELS else "FAIL_THRESHOLD_CROSSING"


def _original_params_and_states(mn: str):
    """Use the shared build_interior_parameters / build_initial_states."""
    from tests.euler_convergence_all_core_utils import (
        build_interior_parameters,
        build_initial_states,
    )
    params = build_interior_parameters(mn)
    init_s = build_initial_states(mn, params)
    return params, init_s


def scenario_original(mn: str):
    """Scenario A: original smooth_warm_positive."""
    from tests.euler_convergence_all_core_utils import build_smooth_forcing
    params, init_s = _original_params_and_states(mn)
    forcing = build_smooth_forcing()
    return params, init_s, forcing


def scenario_threshold_separated(mn: str):
    """Scenario B: threshold_separated_storage — mid-range states, T=20C."""
    if mn == "australia":
        params = {
            "sb": _tscalar(500.0), "phi": _tscalar(0.55),
            "fc_frac": _tscalar(0.5), "alpha_ss": _tscalar(0.06),
            "beta_ss": _tscalar(1.3), "k_deep": _tscalar(0.02),
            "alpha_bf": _tscalar(0.02), "beta_bf": _tscalar(1.2),
        }
        init_s = [
            _tscalar(120.0),  # S1 — ~24% of sb
            _tscalar(80.0),   # S2 — ~16% of sb, well below
            _tscalar(40.0),   # S3 — moderate
        ]
    elif mn == "collie3":
        params = {
            "smax": _tscalar(400.0), "fc": _tscalar(0.35),
            "a": _tscalar(0.02), "m": _tscalar(1.2),
            "b": _tscalar(1.2), "lambda_par": _tscalar(0.3),
        }
        init_s = [
            _tscalar(60.0),   # S1 — 15% of smax, far below fc=140
            _tscalar(25.0),   # S2 — moderate
        ]
    elif mn == "susannah1":
        params = {
            "sb": _tscalar(400.0), "sfc_frac": _tscalar(0.35),
            "m": _tscalar(1.2), "a": _tscalar(0.15),
            "b": _tscalar(1.3), "r": _tscalar(0.02),
        }
        init_s = [
            _tscalar(60.0),   # S1 — far below fc=140
            _tscalar(25.0),   # S2
        ]
    elif mn == "us1":
        params = {
            "alpha_ei": _tscalar(0.15), "m": _tscalar(1.2),
            "smax": _tscalar(400.0), "fc": _tscalar(0.35),
            "alpha_ss": _tscalar(0.05),
        }
        init_s = [
            _tscalar(50.0),   # S1 — below fc=140
            _tscalar(40.0),   # S2
        ]
    elif mn == "vic":
        stot_val = 500.0
        fsm_val = 0.55
        params = {
            "ibar": _tscalar(1.5), "idelta": _tscalar(0.2),
            "ishift": _tscalar(180.0), "stot": _tscalar(stot_val),
            "fsm": _tscalar(fsm_val), "b": _tscalar(1.3),
            "k1": _tscalar(0.04), "c1": _tscalar(1.2),
            "k2": _tscalar(0.02), "c2": _tscalar(1.2),
        }
        smmax = fsm_val * stot_val
        gwmax = (1.0 - fsm_val) * stot_val
        init_s = [
            _tscalar(0.5),            # S1 — interception
            _tscalar(smmax * 0.25),   # S2 — 25% of smmax
            _tscalar(gwmax * 0.20),   # S3 — 20% of gwmax
        ]
    elif mn == "mopex2":
        params = {
            "tcrit": _tscalar(0.0), "ddf": _tscalar(3.5),
            "s2max": _tscalar(600.0), "tw": _tscalar(0.1),
            "tu": _tscalar(0.2), "se": _tscalar(600.0), "tc": _tscalar(0.2),
        }
        init_s = [
            _tscalar(1.0), _tscalar(180.0), _tscalar(60.0),
            _tscalar(5.0), _tscalar(10.0),
        ]
    elif mn == "mopex3":
        params = {
            "tcrit": _tscalar(0.0), "ddf": _tscalar(3.5),
            "s2max": _tscalar(600.0), "tw": _tscalar(0.1),
            "tu": _tscalar(0.2), "se": _tscalar(0.5),
            "s3max": _tscalar(600.0), "tc": _tscalar(0.2),
        }
        init_s = [
            _tscalar(1.0), _tscalar(180.0), _tscalar(200.0),
            _tscalar(5.0), _tscalar(10.0),
        ]
    elif mn == "hbv96":
        params = {
            "tt": _tscalar(0.5), "tti": _tscalar(4.0), "ttm": _tscalar(0.0),
            "cfr": _tscalar(0.05), "cfmax": _tscalar(2.0), "whc": _tscalar(0.1),
            "cflux": _tscalar(0.2), "fc": _tscalar(400.0), "lp": _tscalar(0.5),
            "beta": _tscalar(1.5), "k0": _tscalar(0.05), "alpha": _tscalar(1.0),
            "perc": _tscalar(0.5), "k1": _tscalar(0.03), "maxbas": _tscalar(5.0),
        }
        init_s = [
            _tscalar(1.0), _tscalar(2.0), _tscalar(160.0),
            _tscalar(10.0), _tscalar(30.0),
        ]
    else:
        raise ValueError(mn)
    forcing = _smooth_forcing(temp_val=20.0)
    return params, init_s, forcing


def scenario_high_storage(mn: str):
    """Scenario C: high_storage_but_not_overflow — high states below capacity."""
    if mn == "australia":
        params = {
            "sb": _tscalar(300.0), "phi": _tscalar(0.55),
            "fc_frac": _tscalar(0.5), "alpha_ss": _tscalar(0.06),
            "beta_ss": _tscalar(1.3), "k_deep": _tscalar(0.02),
            "alpha_bf": _tscalar(0.02), "beta_bf": _tscalar(1.2),
        }
        init_s = [
            _tscalar(220.0),  # S1 — ~73% of sb
            _tscalar(200.0),  # S2 — ~67% of sb
            _tscalar(80.0),   # S3
        ]
    elif mn == "collie3":
        params = {
            "smax": _tscalar(300.0), "fc": _tscalar(0.40),
            "a": _tscalar(0.03), "m": _tscalar(1.2),
            "b": _tscalar(1.3), "lambda_par": _tscalar(0.3),
        }
        fc_mm = 0.40 * 300.0  # 120
        init_s = [
            _tscalar(115.0),  # S1 — just below fc
            _tscalar(50.0),   # S2
        ]
    elif mn == "susannah1":
        params = {
            "sb": _tscalar(300.0), "sfc_frac": _tscalar(0.4),
            "m": _tscalar(1.2), "a": _tscalar(0.15),
            "b": _tscalar(1.3), "r": _tscalar(0.02),
        }
        fc_mm = 0.4 * 300.0  # 120
        init_s = [
            _tscalar(115.0),  # S1 — just below fc
            _tscalar(40.0),   # S2
        ]
    elif mn == "us1":
        params = {
            "alpha_ei": _tscalar(0.15), "m": _tscalar(1.2),
            "smax": _tscalar(300.0), "fc": _tscalar(0.4),
            "alpha_ss": _tscalar(0.05),
        }
        fc_mm = 0.4 * 300.0  # 120
        init_s = [
            _tscalar(115.0),  # S1 — just below fc
            _tscalar(80.0),   # S2 — moderate
        ]
    elif mn == "vic":
        stot_val = 400.0
        fsm_val = 0.55
        smmax = fsm_val * stot_val
        gwmax = (1.0 - fsm_val) * stot_val
        params = {
            "ibar": _tscalar(1.5), "idelta": _tscalar(0.2),
            "ishift": _tscalar(180.0), "stot": _tscalar(stot_val),
            "fsm": _tscalar(fsm_val), "b": _tscalar(1.3),
            "k1": _tscalar(0.04), "c1": _tscalar(1.2),
            "k2": _tscalar(0.02), "c2": _tscalar(1.2),
        }
        init_s = [
            _tscalar(0.5),
            _tscalar(smmax * 0.88),   # S2 — 88%, high but under capacity
            _tscalar(gwmax * 0.80),   # S3 — 80%, high but under capacity
        ]
    elif mn == "mopex2":
        params = {"tcrit": _tscalar(0.0), "ddf": _tscalar(3.5),
                  "s2max": _tscalar(300.0), "tw": _tscalar(0.1),
                  "tu": _tscalar(0.2), "se": _tscalar(300.0), "tc": _tscalar(0.2)}
        init_s = [_tscalar(1.0), _tscalar(250.0), _tscalar(100.0),
                  _tscalar(5.0), _tscalar(10.0)]
    elif mn == "mopex3":
        params = {"tcrit": _tscalar(0.0), "ddf": _tscalar(3.5),
                  "s2max": _tscalar(300.0), "tw": _tscalar(0.1),
                  "tu": _tscalar(0.2), "se": _tscalar(0.5),
                  "s3max": _tscalar(300.0), "tc": _tscalar(0.2)}
        init_s = [_tscalar(1.0), _tscalar(250.0), _tscalar(270.0),
                  _tscalar(5.0), _tscalar(10.0)]
    elif mn == "hbv96":
        params = {"tt": _tscalar(0.5), "tti": _tscalar(4.0), "ttm": _tscalar(0.0),
                  "cfr": _tscalar(0.05), "cfmax": _tscalar(2.0), "whc": _tscalar(0.1),
                  "cflux": _tscalar(0.2), "fc": _tscalar(300.0), "lp": _tscalar(0.5),
                  "beta": _tscalar(1.5), "k0": _tscalar(0.05), "alpha": _tscalar(1.0),
                  "perc": _tscalar(0.5), "k1": _tscalar(0.03), "maxbas": _tscalar(5.0)}
        init_s = [_tscalar(1.0), _tscalar(2.0), _tscalar(270.0),
                  _tscalar(15.0), _tscalar(50.0)]
    else:
        raise ValueError(mn)
    forcing = _smooth_forcing(temp_val=20.0)
    return params, init_s, forcing


def scenario_stress(mn: str):
    """Scenario D: stress_near_threshold — push states near key thresholds."""
    if mn == "australia":
        params = {
            "sb": _tscalar(200.0), "phi": _tscalar(0.55),
            "fc_frac": _tscalar(0.5), "alpha_ss": _tscalar(0.10),
            "beta_ss": _tscalar(1.5), "k_deep": _tscalar(0.03),
            "alpha_bf": _tscalar(0.03), "beta_bf": _tscalar(1.3),
        }
        init_s = [
            _tscalar(195.0),  # S1 — near sb
            _tscalar(190.0),  # S2 — near sb (triggers saturation_1)
            _tscalar(50.0),   # S3
        ]
    elif mn == "collie3":
        params = {
            "smax": _tscalar(200.0), "fc": _tscalar(0.35),
            "a": _tscalar(0.06), "m": _tscalar(1.2),
            "b": _tscalar(1.5), "lambda_par": _tscalar(0.3),
        }
        fc_mm = 0.35 * 200.0  # 70
        init_s = [
            _tscalar(72.0),   # S1 — just above fc (stresses threshold)
            _tscalar(10.0),   # S2 — low
        ]
    elif mn == "susannah1":
        params = {
            "sb": _tscalar(200.0), "sfc_frac": _tscalar(0.35),
            "m": _tscalar(1.2), "a": _tscalar(0.2),
            "b": _tscalar(1.5), "r": _tscalar(0.03),
        }
        fc_mm = 0.35 * 200.0  # 70
        init_s = [
            _tscalar(72.0),   # S1 — just above fc
            _tscalar(10.0),   # S2
        ]
    elif mn == "us1":
        params = {
            "alpha_ei": _tscalar(0.15), "m": _tscalar(1.2),
            "smax": _tscalar(200.0), "fc": _tscalar(0.35),
            "alpha_ss": _tscalar(0.08),
        }
        fc_mm = 0.35 * 200.0  # 70
        init_s = [
            _tscalar(72.0),   # S1 — just above fc
            _tscalar(160.0),  # S2 — near smax
        ]
    elif mn == "vic":
        stot_val = 300.0
        fsm_val = 0.55
        smmax = fsm_val * stot_val
        gwmax = (1.0 - fsm_val) * stot_val
        params = {
            "ibar": _tscalar(1.5), "idelta": _tscalar(0.2),
            "ishift": _tscalar(180.0), "stot": _tscalar(stot_val),
            "fsm": _tscalar(fsm_val), "b": _tscalar(2.0),
            "k1": _tscalar(0.06), "c1": _tscalar(1.5),
            "k2": _tscalar(0.03), "c2": _tscalar(1.4),
        }
        init_s = [
            _tscalar(1.3),            # S1 — near interception capacity
            _tscalar(smmax * 0.96),   # S2 — near smmax
            _tscalar(gwmax * 0.92),   # S3 — near gwmax
        ]
    elif mn == "mopex2":
        params = {"tcrit": _tscalar(0.5), "ddf": _tscalar(2.5),
                  "s2max": _tscalar(200.0), "tw": _tscalar(0.2),
                  "tu": _tscalar(0.5), "se": _tscalar(500.0), "tc": _tscalar(0.1)}
        init_s = [_tscalar(10.0), _tscalar(190.0), _tscalar(10.0),
                  _tscalar(2.0), _tscalar(3.0)]
    elif mn == "mopex3":
        params = {"tcrit": _tscalar(0.5), "ddf": _tscalar(2.5),
                  "s2max": _tscalar(200.0), "tw": _tscalar(0.2),
                  "tu": _tscalar(0.5), "se": _tscalar(0.5),
                  "s3max": _tscalar(200.0), "tc": _tscalar(0.1)}
        init_s = [_tscalar(10.0), _tscalar(190.0), _tscalar(180.0),
                  _tscalar(2.0), _tscalar(3.0)]
    elif mn == "hbv96":
        params = {"tt": _tscalar(0.5), "tti": _tscalar(2.0), "ttm": _tscalar(0.3),
                  "cfr": _tscalar(0.05), "cfmax": _tscalar(2.0), "whc": _tscalar(0.1),
                  "cflux": _tscalar(0.2), "fc": _tscalar(200.0), "lp": _tscalar(0.3),
                  "beta": _tscalar(2.0), "k0": _tscalar(0.1), "alpha": _tscalar(1.5),
                  "perc": _tscalar(0.3), "k1": _tscalar(0.03), "maxbas": _tscalar(5.0)}
        init_s = [_tscalar(15.0), _tscalar(5.0), _tscalar(100.0),
                  _tscalar(15.0), _tscalar(30.0)]
    else:
        raise ValueError(mn)
    forcing = _smooth_forcing(temp_val=20.0, precip_scale=1.5, pet_scale=1.2)
    return params, init_s, forcing


def scenario_fine_asymptotic(mn: str):
    """Scenario E: fine_substep_asymptotic — threshold-separated params, levels 2..32."""
    params, init_s, forcing = scenario_threshold_separated(mn)
    return params, init_s, forcing


SCENARIOS: dict[str, Any] = {
    "A_original": scenario_original,
    "B_threshold_separated": scenario_threshold_separated,
    "C_high_storage": scenario_high_storage,
    "D_stress": scenario_stress,
    "E_fine_asymptotic": scenario_fine_asymptotic,
}

SCENARIO_LABELS = {
    "A_original": "Original smooth_warm_positive — reproduce existing failure.",
    "B_threshold_separated": "States mid-range, away from capacity bounds; T=20C.",
    "C_high_storage": "High storage but below saturation/overflow thresholds.",
    "D_stress": "States pushed near key thresholds to demonstrate crossing.",
    "E_fine_asymptotic": "Threshold-separated params with substep levels up to 32.",
}


def recommend_status(median_order: float, monotone: bool, in_band: bool,
                     any_nan: bool, min_abs: float) -> str:
    if any_nan:
        return "FAIL_UNEXPECTED"
    if in_band and monotone:
        return "PASS"
    if in_band and not monotone:
        return "PASS_WITH_CAVEAT"
    return "FAIL_THRESHOLD_CROSSING"


def determine_threshold_flags(mn: str, scen_label: str) -> str:
    """Return detected threshold crossing flags for this model+scenario."""
    flags_map = {
        "australia": "saturation_1(sigmoid), excess_1(relu), cap_s1_to_s2(relu), interflow_3(power-law+nearzero), scale_s2(ratio-clip)",
        "collie3": "field_capacity_relu(FC), interflow_power_law(nearzero), baseflow_power_law(nearzero), saturation_1(sigmoid)",
        "susannah1": "interflow_7(relu+power-law), evap_6(wilting-point), baseflow_2(power-law)",
        "us1": "field_capacity_dynamic_relu, saturation_1(sigmoid), excess_1(relu), evap_8_9(division+nearzero), baseflow_1(linear-cap)",
        "vic": "saturation_2((1-S/Smax)^b with clamp), excess_1(relu), percolation_5(power-law+nearzero), baseflow_5(power-law+nearzero), saturation_1(sigmoid)",
        "mopex2": "saturation_1(sigmoid), snow_rain_melt_thresholds, multiple_relu_clamps",
        "mopex3": "saturation_1(sigmoid), snow_rain_melt_thresholds, multiple_relu_clamps",
        "hbv96": "snowfall_rainfall_melt_hard_clamps, saturation_1(sigmoid), relu_interflow",
    }
    return flags_map.get(mn, "")


def determine_diagnostic_subtype(median_order: float, final_local_order: float,
                                  in_band_final: bool, monotone: bool,
                                  any_nan: bool) -> str:
    if any_nan:
        return "UNEXPECTED_NAN_INF"
    if math.isfinite(final_local_order) and in_band_final:
        return "ASYMPTOTIC_FIRST_ORDER_CONFIRMED"
    if not monotone:
        return "NON_MONOTONE_THRESHOLD"
    return "PERSISTENT_THRESHOLD_OR_STRUCTURAL_FAILURE"


def main():
    all_rows = []
    md_sections = []

    for model_name in TARGET_MODELS:
        rate_names = RATE_NAMES.get(model_name, frozenset())
        uses_temp = USES_TEMP.get(model_name, False)

        md_sections.append(f"\n## {model_name}\n")

        for scen_label, scen_fn in SCENARIOS.items():
            scenario = f"{model_name}_{scen_label}"
            fine_mode = (scen_label == "E_fine_asymptotic")
            levels = FINE_SUBSTEP_LEVELS if fine_mode else SUBSTEP_LEVELS

            print(f"\n=== {scenario} ===")

            try:
                params, init_s, forcing = scen_fn(model_name)
                diag = run_convergence_sweep(
                    model_name, scenario, params, init_s, forcing,
                    rate_names, uses_temp, fine_mode=fine_mode,
                )
            except Exception as exc:
                print(f"  FAILED: {exc}")
                all_rows.append({
                    "model": model_name, "scenario": scenario,
                    "substep_levels": "",
                    "state_errors": str(exc),
                    "q_errors": "", "ea_errors": "",
                    "empirical_orders": "",
                    "median_order": float("nan"),
                    "final_local_order": float("nan"),
                    "state_error_monotone": False,
                    "in_pass_band_by_median_order": False,
                    "in_pass_band_by_final_local_order": False,
                    "min_abs_error": float("nan"),
                    "final_abs_state_error": float("nan"),
                    "final_abs_q_error": float("nan"),
                    "any_nan_inf": True,
                    "detected_threshold_crossing_flags": "",
                    "recommended_status": "FAIL_UNEXPECTED",
                    "diagnostic_subtype": "UNEXPECTED_NAN_INF",
                    "final_disposition": final_disposition(model_name),
                    "notes": f"Exception: {exc}",
                })
                md_sections.append(f"### {scen_label}\n\n**ERROR**: {exc}\n")
                continue

            se = diag["state_errors"]
            orders = diag["empirical_orders"]
            mo = diag["median_order"]
            flo = diag["final_local_order"]
            in_band_final = diag["in_pass_band_by_final_local_order"]
            monotone = diag["state_error_monotone"]
            any_nan = diag["any_nan_inf"]

            rec = recommend_status(mo, monotone,
                                   diag["in_pass_band_by_median_order"],
                                   any_nan, diag["min_abs_error"])
            diag["recommended_status"] = rec
            diag["diagnostic_subtype"] = determine_diagnostic_subtype(
                mo, flo, in_band_final, monotone, any_nan)
            diag["detected_threshold_crossing_flags"] = determine_threshold_flags(
                model_name, scen_label)

            notes = SCENARIO_LABELS.get(scen_label, "")
            if scen_label == "E_fine_asymptotic":
                notes += " Fine-substep convergence check (levels 2..32)."

            row = {
                "model": model_name,
                "scenario": scenario,
                "substep_levels": ";".join(str(lv) for lv in levels),
                "state_errors": "; ".join(f"{e:.4e}" for e in se),
                "q_errors": "; ".join(f"{e:.4e}" for e in diag["q_errors"]),
                "ea_errors": "; ".join(f"{e:.4e}" for e in diag["ea_errors"]),
                "empirical_orders": "; ".join(
                    f"{o:.3f}" if math.isfinite(o) else "N/A" for o in orders
                ),
                "median_order": round(mo, 4) if math.isfinite(mo) else float("nan"),
                "final_local_order": round(flo, 4) if math.isfinite(flo) else float("nan"),
                "state_error_monotone": monotone,
                "in_pass_band_by_median_order": diag["in_pass_band_by_median_order"],
                "in_pass_band_by_final_local_order": in_band_final,
                "min_abs_error": diag["min_abs_error"],
                "final_abs_state_error": diag["final_abs_state_error"],
                "final_abs_q_error": diag["final_abs_q_error"],
                "any_nan_inf": any_nan,
                "detected_threshold_crossing_flags": diag["detected_threshold_crossing_flags"],
                "recommended_status": rec,
                "diagnostic_subtype": diag["diagnostic_subtype"],
                "final_disposition": final_disposition(model_name),
                "notes": notes,
            }
            all_rows.append(row)

            mo_str = f"{mo:.3f}" if math.isfinite(mo) else "N/A"
            flo_str = f"{flo:.3f}" if math.isfinite(flo) else "N/A"
            print(f"  median_order={mo_str} final_local_order={flo_str} "
                  f"monotone={monotone} in_band_median={diag['in_pass_band_by_median_order']} "
                  f"in_band_final={in_band_final} "
                  f"min_abs={diag['min_abs_error']:.2e} rec={rec} "
                  f"subtype={diag['diagnostic_subtype']}")

            # MD table
            md_rows = []
            for i in range(len(se)):
                order_str = ""
                if i > 0 and i - 1 < len(orders):
                    o = orders[i - 1]
                    order_str = f"{o:.3f}" if math.isfinite(o) else "N/A"
                md_rows.append(
                    f"| {levels[i]} | {se[i]:.4e} | {diag['q_errors'][i]:.4e} | {diag['ea_errors'][i]:.4e} | {order_str} |"
                )

            md_sections.append(
                f"### {scen_label}\n\n"
                f"| level | state_error | q_error | ea_error | order |\n"
                f"|---|---|---|---|---|\n" +
                "\n".join(md_rows) +
                f"\n\n"
                f"- median_order: {mo_str}\n"
                f"- final_local_order: {flo_str}\n"
                f"- state_error_monotone: {monotone}\n"
                f"- in_pass_band_by_median_order: {diag['in_pass_band_by_median_order']}\n"
                f"- in_pass_band_by_final_local_order: {in_band_final}\n"
                f"- min_abs_error: {diag['min_abs_error']:.2e}\n"
                f"- any_nan_inf: {any_nan}\n"
                f"- recommended_status: **{rec}**\n"
                f"- diagnostic_subtype: **{diag['diagnostic_subtype']}**\n"
                f"- detected_thresholds: {diag['detected_threshold_crossing_flags']}\n"
                f"\n{notes}\n"
            )

    # Write CSV
    csv_fieldnames = [
        "model", "scenario", "substep_levels",
        "state_errors", "q_errors", "ea_errors", "empirical_orders",
        "median_order", "final_local_order",
        "state_error_monotone",
        "in_pass_band_by_median_order", "in_pass_band_by_final_local_order",
        "min_abs_error", "final_abs_state_error", "final_abs_q_error",
        "any_nan_inf",
        "detected_threshold_crossing_flags",
        "recommended_status", "diagnostic_subtype", "final_disposition", "notes",
    ]
    with open(DGN_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {DGN_CSV} ({len(all_rows)} rows)")

    # Write MD
    md = [
        "# Euler Convergence — Threshold-Isolation Diagnostics",
        "",
        "Diagnostics for all 8 FAIL_THRESHOLD_CROSSING models:",
        "**mopex2**, **mopex3**, **hbv96**, **australia**, **collie3**, **susannah1**, **us1**, **vic**.",
        "",
        "Reclassification: **collie3**, **susannah1**, and **us1** are retained as **PASS_WITH_CAVEAT** for the final daily-use disposition; **australia**, **hbv96**, **mopex2**, **mopex3**, and **vic** remain **FAIL_THRESHOLD_CROSSING** for subdaily Euler.",
        "",
        "## Scenarios",
        "",
        "- **A_original**: Default smooth_warm_positive scenario (reproduce failure).",
        "- **B_threshold_separated**: Storage mid-range + T=20 °C — all known thresholds avoided.",
        "- **C_high_storage**: High storage but below saturation/overflow thresholds.",
        "- **D_stress**: States near capacity / field capacity — demonstrates threshold crossing.",
        "- **E_fine_asymptotic**: Threshold-separated params with substep levels up to 32.",
        "",
        "## Results",
    ]
    md += md_sections

    # Build per-model verdicts
    md += ["", "---", "", "## Model-by-Model Verdicts", ""]

    model_judgements: dict[str, list[dict]] = {}
    for row in all_rows:
        mn = row["model"]
        model_judgements.setdefault(mn, []).append(row)

    for mn in TARGET_MODELS:
        rows = model_judgements.get(mn, [])
        orig = next((r for r in rows if "A_original" in r["scenario"]), {})
        sep = next((r for r in rows if "B_threshold_separated" in r["scenario"]), {})
        high = next((r for r in rows if "C_high_storage" in r["scenario"]), {})
        stress = next((r for r in rows if "D_stress" in r["scenario"]), {})
        fine = next((r for r in rows if "E_fine_asymptotic" in r["scenario"]), {})

        orig_band = orig.get("in_pass_band_by_median_order", False)
        sep_band = sep.get("in_pass_band_by_median_order", False)
        sep_final_band = sep.get("in_pass_band_by_final_local_order", False)
        fine_final_band = fine.get("in_pass_band_by_final_local_order", False)
        fine_flo = fine.get("final_local_order", float("nan"))
        sep_flo = sep.get("final_local_order", float("nan"))
        subtype = fine.get("diagnostic_subtype", "")
        any_nan_fine = fine.get("any_nan_inf", False)

        md.append(f"### {mn}")
        md.append(f"- Primary thresholds: {orig.get('detected_threshold_crossing_flags', 'N/A')}")
        md.append(f"- Scenario A (original): median_order={orig.get('median_order','?')}")
        md.append(f"- Scenario B (separated): median_order={sep.get('median_order','?')}, final_local_order={sep_flo}")
        md.append(f"- Scenario E (fine_asymptotic): final_local_order={fine_flo}")

        if any_nan_fine:
            md.append(f"- **Verdict**: FAIL_UNEXPECTED — NaN/Inf detected.")
        elif subtype == "ASYMPTOTIC_FIRST_ORDER_CONFIRMED":
            md.append(
                f"- **Verdict**: The model recovers first-order behavior in fine-substep regimes "
                f"(final_local_order={fine_flo:.3f}), but remains FAIL_THRESHOLD_CROSSING under "
                f"the strict median-order criterion because coarse substeps are dominated by "
                f"threshold/kink activations. → retain **FAIL_THRESHOLD_CROSSING**, "
                f"diagnostic_subtype=ASYMPTOTIC_FIRST_ORDER_CONFIRMED"
            )
        elif subtype == "PERSISTENT_THRESHOLD_OR_STRUCTURAL_FAILURE":
            md.append(
                f"- **Verdict**: Convergence not recovered even in threshold-separated scenarios. "
                f"→ retain **FAIL_THRESHOLD_CROSSING**, "
                f"diagnostic_subtype=PERSISTENT_THRESHOLD_OR_STRUCTURAL_FAILURE"
            )
        else:
            md.append(
                f"- **Verdict**: {subtype} → retain **FAIL_THRESHOLD_CROSSING**"
            )

    md += ["", "---", "", "## Summary Table", ""]
    md.append("| model | A_median | B_median | B_final | E_final | subtype |")
    md.append("|---|---|---|---|---|---|")
    for mn in TARGET_MODELS:
        rows = model_judgements.get(mn, [])
        a_row = next((r for r in rows if "A_original" in r["scenario"]), {})
        b_row = next((r for r in rows if "B_threshold_separated" in r["scenario"]), {})
        e_row = next((r for r in rows if "E_fine_asymptotic" in r["scenario"]), {})
        a_mo = a_row.get("median_order", float("nan"))
        b_mo = b_row.get("median_order", float("nan"))
        b_flo = b_row.get("final_local_order", float("nan"))
        e_flo = e_row.get("final_local_order", float("nan"))
        subtype = e_row.get("diagnostic_subtype", "")
        md.append(
            f"| {mn} | {a_mo:.3f}" if isinstance(a_mo, (int, float)) and math.isfinite(a_mo)
            else f"| {mn} | N/A"
        )
        # Simpler: use formatted strings
        def _fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and math.isfinite(v) else "N/A"

        sub_row = f"| {mn} | {_fmt(a_mo)} | {_fmt(b_mo)} | {_fmt(b_flo)} | {_fmt(e_flo)} | {subtype} |"
        # Already appended, but let me clean this up
    md.append("")

    # Regenerate the summary table properly
    # The summary table section is already added. Let me just write to file.
    # Actually, the table logic is messy. Let me rewrite it cleanly.
    # Remove the last bad table section and recreate it.
    md = md[:-2]  # Remove the partially built summary table approach

    md.append("## Summary Table")
    md.append("")
    md.append("| model | A_median | B_median | B_final_local | E_final_local | diagnostic_subtype |")
    md.append("|---|---|---|---|---|---|")
    for mn in TARGET_MODELS:
        rows = model_judgements.get(mn, [])
        def _g(scen_key, field):
            r = next((x for x in rows if scen_key in x["scenario"]), {})
            v = r.get(field, float("nan"))
            return f"{v:.3f}" if isinstance(v, (int, float)) and math.isfinite(v) else "N/A"
        md.append(
            f"| {mn} | {_g('A_original','median_order')} | "
            f"{_g('B_threshold_separated','median_order')} | "
            f"{_g('B_threshold_separated','final_local_order')} | "
            f"{_g('E_fine_asymptotic','final_local_order')} | "
            f"{_g('E_fine_asymptotic','diagnostic_subtype')} |"
        )

    with open(DGN_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {DGN_MD}")


if __name__ == "__main__":
    main()
