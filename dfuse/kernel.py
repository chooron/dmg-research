"""First-stage differentiable FUSE kernel.

This is a clean-room tensor implementation of the public FUSE flux/state
contract.  It intentionally does not import or copy upstream Fortran.  The
current integration target is the paper's fixed-decision family with one
lumped elevation band; all 78 decision specifications resolve, while the
kernel keeps the remaining paper-specific multi-band snow path separate from this first-stage lumped interface.
reference-facing oracle and fidelity harness live in `project/autofuse`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import time
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from .spec import FLUX_NAMES, PARAMETER_NAMES, STATE_NAMES, get_structure


@dataclass
class SimulationResult:
    """Outputs from one basin/model forward pass."""

    model_id: int
    q: Tensor
    q_instantaneous: Tensor
    states: Tensor
    state_names: tuple[str, ...]
    fluxes: dict[str, Tensor]
    water_balance_residual: Tensor
    snow: Tensor
    snow_balance_residual: Tensor
    sequential_diagnostics: dict[str, Tensor] | None = None

    @property
    def max_abs_water_balance_error(self) -> Tensor:
        return self.water_balance_residual.abs().amax()


def _to_tensor(value: object, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _forcing_tensor(forcing: Tensor | Mapping[str, object]) -> Tensor:
    if isinstance(forcing, Mapping):
        try:
            values = [forcing[name] for name in ("ppt", "pet", "temp")]
        except KeyError as exc:
            raise ValueError("forcing mapping must contain ppt, pet, and temp") from exc
        dtype = next((v.dtype for v in values if isinstance(v, Tensor) and v.is_floating_point()), torch.get_default_dtype())
        device = next((v.device for v in values if isinstance(v, Tensor)), torch.device("cpu"))
        tensor_values = [_to_tensor(value, dtype=dtype, device=device).reshape(-1) for value in values]
        n = tensor_values[0].numel()
        if any(value.numel() != n for value in tensor_values):
            raise ValueError("forcing columns must have equal lengths")
        return torch.stack(tensor_values, dim=-1)
    if not isinstance(forcing, Tensor):
        forcing = torch.as_tensor(forcing, dtype=torch.get_default_dtype())
    if forcing.ndim != 2 or forcing.shape[1] != 3:
        raise ValueError("forcing must have shape [time, 3] with columns ppt, pet, temp")
    return forcing


def _parameter_values(
    params: Mapping[str, object] | Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Tensor]:
    if params is None:
        from .spec import default_parameters

        source: Mapping[str, object] = default_parameters()
    elif isinstance(params, Tensor):
        if params.ndim != 1 or params.numel() not in (len(PARAMETER_NAMES),):
            raise ValueError(f"parameter tensor must have {len(PARAMETER_NAMES)} union coordinates")
        source = dict(zip(PARAMETER_NAMES, params))
    else:
        source = params
    result = {}
    from .spec import default_parameters

    defaults = default_parameters()
    for name in PARAMETER_NAMES:
        result[name] = _to_tensor(source.get(name, defaults[name]), dtype=dtype, device=device)
    return result


def _day_of_year(
    n: int,
    dates: Sequence[date | datetime] | Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if dates is None:
        return (
            torch.remainder(torch.arange(n, device=device, dtype=dtype), 365.0) + 1.0,
            torch.zeros(n, device=device, dtype=torch.bool),
        )
    if isinstance(dates, Tensor):
        result = dates.to(device=device, dtype=dtype).reshape(-1)
        leap = torch.zeros(result.numel(), device=device, dtype=torch.bool)
    else:
        result = torch.as_tensor(
            [item.timetuple().tm_yday for item in dates], device=device, dtype=dtype
        )
        leap = torch.as_tensor(
            [item.year % 4 == 0 for item in dates], device=device, dtype=torch.bool
        )
    if result.numel() != n:
        raise ValueError("dates/day_of_year must have one entry per forcing row")
    return result, leap


def _capacity(params: Mapping[str, Tensor]) -> dict[str, Tensor]:
    max1 = params["MAXWATR_1"]
    max2 = params["MAXWATR_2"]
    fracten = params["FRACTEN"]
    return {
        "MAXWATR_1": max1,
        "MAXWATR_2": max2,
        "MAXTENS_1": fracten * max1,
        "MAXFREE_1": (1.0 - fracten) * max1,
        "MAXTENS_2": fracten * max2,
        "MAXFREE_2": (1.0 - fracten) * max2,
        "MAXTENS_1A": params.get("FRCHZNE", torch.zeros_like(max1)) * fracten * max1,
        "MAXTENS_1B": (1.0 - params.get("FRCHZNE", torch.zeros_like(max1))) * fracten * max1,
        "MAXFREE_2A": params.get("FPRIMQB", torch.zeros_like(max2)) * (1.0 - fracten) * max2,
        "MAXFREE_2B": (1.0 - params.get("FPRIMQB", torch.zeros_like(max2))) * (1.0 - fracten) * max2,
    }


def _state_dict(state: Tensor, names: Sequence[str]) -> dict[str, Tensor]:
    return {name: state[i] for i, name in enumerate(names)}


def _derived_states(state: Mapping[str, Tensor], spec, cap: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Add the upstream derived state aliases without adding state coordinates."""
    result = dict(state)
    if spec.decisions["ARCH1"] == "tension2_1":
        result["TENS_1"] = state["TENS_1A"] + state["TENS_1B"]
        result["WATR_1"] = result["TENS_1"] + state["FREE_1"]
    elif spec.decisions["ARCH1"] == "tension1_1":
        result["WATR_1"] = state["TENS_1"] + state["FREE_1"]
    else:
        result["TENS_1"] = torch.minimum(state["WATR_1"], cap["MAXTENS_1"])
        result["FREE_1"] = torch.clamp_min(state["WATR_1"] - cap["MAXTENS_1"], 0.0)
    if spec.decisions["ARCH2"] == "tens2pll_2":
        result["FREE_2"] = state["FREE_2A"] + state["FREE_2B"]
        result["WATR_2"] = state["TENS_2"] + result["FREE_2"]
    else:
        result["TENS_2"] = torch.minimum(state["WATR_2"], cap["MAXTENS_2"])
        result["FREE_2"] = torch.clamp_min(state["WATR_2"] - cap["MAXTENS_2"], 0.0)
    return result


def _project_state(state: Tensor, spec, cap: Mapping[str, Tensor]) -> Tensor:
    """Keep trial states in the domain required by power/log fluxes."""
    values = []
    state_capacity = {
        "TENS_1A": "MAXTENS_1A", "TENS_1B": "MAXTENS_1B",
        "TENS_1": "MAXTENS_1", "FREE_1": "MAXFREE_1", "WATR_1": "MAXWATR_1",
        "TENS_2": "MAXTENS_2", "FREE_2A": "MAXFREE_2A", "FREE_2B": "MAXFREE_2B",
        "WATR_2": "MAXWATR_2",
    }
    for i, name in enumerate(spec.state_names):
        maximum = cap[state_capacity[name]]
        value = state[i].clamp_min(maximum * 1.0e-8)
        if name != "WATR_2" or spec.decisions["ARCH2"] == "fixedsiz_2":
            value = value.clamp_max(maximum)
        values.append(value)
    return torch.stack(values)


def _logismooth(value: Tensor, maximum: Tensor, fraction: float = 0.01) -> Tensor:
    smooth = fraction * maximum
    return torch.sigmoid((value - (maximum - 5.0 * smooth)) / smooth.clamp_min(1.0e-12))

def _regularized_gamma_p(shape: Tensor, argument: Tensor) -> Tensor:
    """Evaluate regularized gamma P exactly with a vmap/compile-safe shape gradient."""
    shape_value = shape.detach()
    base = torch.special.gammainc(shape_value, argument)
    safe_shape = shape_value.abs().clamp_min(1.0)
    step = 1.0e-4 * safe_shape
    shape_plus = shape_value + step
    shape_minus = (shape_value - step).clamp_min(1.0e-6)
    shape_slope = (
        torch.special.gammainc(shape_plus, argument.detach())
        - torch.special.gammainc(shape_minus, argument.detach())
    ) / (shape_plus - shape_minus)
    # The correction is identically zero in forward mode but carries the
    # finite-difference derivative with respect to the shape parameter.
    return base + shape_slope.detach() * (shape - shape_value)

def _topographic_mean(params: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Match ``MEAN_TIPOW`` in the frozen FUSE Fortran reference.

    The reference uses a 2,000-bin midpoint quadrature over the shifted Gamma
    distribution of log topographic index (offset 3, upper bound 50).  Keeping
    the quadrature tensorized preserves gradients through ``LOGLAMB``,
    ``TISHAPE`` and ``QB_POWR`` without materializing masked invalid branches.
    """
    dtype = params["QB_POWR"].dtype
    device = params["QB_POWR"].device
    power = params["QB_POWR"].clamp_min(1.0e-6)
    shape = params["TISHAPE"].clamp_min(1.0e-6)
    offset = torch.as_tensor(3.0, dtype=dtype, device=device)
    chi = ((params["LOGLAMB"] - offset) / shape).clamp_min(1.0e-6)
    index = torch.arange(1, 2001, dtype=dtype, device=device)
    upper = index * (50.0 / 2000.0)
    lower = torch.cat((upper[:1] * 0.0, upper[:-1]))
    upper_probability = _regularized_gamma_p(shape, (upper - offset).clamp_min(0.0) / chi)
    lower_probability = torch.cat((upper_probability[:1] * 0.0, upper_probability[:-1]))
    probability = upper_probability - lower_probability
    log_value = 0.5 * (lower + upper)
    power_value = torch.exp(log_value / power)
    mean_power = (power_value * probability).sum()
    max_power = power_value[-1]
    return mean_power, max_power

_PARAMETER_INDEX = {name: index for index, name in enumerate(PARAMETER_NAMES)}
_STATE_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}
_CTX_A1_T2 = 0
_CTX_A1_T1 = 1
_CTX_A1_ONE = 2
_CTX_A2_PLL = 3
_CTX_A2_FRC = 4
_CTX_A2_POW = 5
_CTX_A2_FIXED = 6
_CTX_QS_ARNO = 7
_CTX_QS_PRMS = 8
_CTX_QS_TMDL = 9
_CTX_QP_F2 = 10
_CTX_QP_W2 = 11
_CTX_QP_LOWER = 12
_CTX_STATE_MASK = 13
_CTX_PARAMETER_MASK = _CTX_STATE_MASK + len(STATE_NAMES)
_CTX_SIZE = _CTX_PARAMETER_MASK + len(PARAMETER_NAMES)
_COMPILED_RHS_CACHE: dict[tuple[object, ...], Callable[..., tuple[Tensor, Tensor, Tensor]]] = {}
_COMPILED_RESIDUAL_CACHE: dict[tuple[object, ...], object] = {}
_COMPILE_DIAGNOSTICS: dict[str, object] = {
    "level_a": {
        "compile_attempts": 0,
        "compile_successes": 0,
        "fallbacks": 0,
        "calls": 0,
        "cold_compile_seconds": [],
        "failures": [],
        "cache_keys": [],
        "dynamo_counters": {},
        "graph_breaks": 0,
        "recompilations": 0,
        "unique_graphs": 0,
    },
    "level_b": {
        "compile_attempts": 0,
        "compile_successes": 0,
        "fallbacks": 0,
        "calls": 0,
        "cold_compile_seconds": [],
        "failures": [],
        "cache_keys": [],
        "dynamo_counters": {},
        "graph_breaks": 0,
        "recompilations": 0,
        "unique_graphs": 0,
    },
    "sequential": {
        "variants": {},
    }
}


def reset_compile_diagnostics() -> None:
    """Reset compile diagnostics and both sequential step registries."""
    _COMPILED_RHS_CACHE.clear()
    _COMPILED_RESIDUAL_CACHE.clear()
    if "_COMPILED_SEQUENTIAL_CACHE" in globals():
        _COMPILED_SEQUENTIAL_CACHE.clear()
    _reset_dynamo_counters()
    _COMPILE_DIAGNOSTICS["sequential"] = {"variants": {}}
    _COMPILE_DIAGNOSTICS["level_a"] = {
        "compile_attempts": 0,
        "compile_successes": 0,
        "fallbacks": 0,
        "calls": 0,
        "cold_compile_seconds": [],
        "failures": [],
        "cache_keys": [],
        "dynamo_counters": {},
        "graph_breaks": 0,
        "recompilations": 0,
        "unique_graphs": 0,
    }
    _COMPILE_DIAGNOSTICS["level_b"] = {
        "compile_attempts": 0,
        "compile_successes": 0,
        "fallbacks": 0,
        "calls": 0,
        "cold_compile_seconds": [],
        "failures": [],
        "cache_keys": [],
        "dynamo_counters": {},
        "graph_breaks": 0,
        "recompilations": 0,
        "unique_graphs": 0,
    }
    try:
        from .runtime import reset_runtime_registries
    except ImportError:
        pass
    else:
        reset_runtime_registries()

def compile_diagnostics() -> dict[str, object]:
    """Return compile diagnostics, including runtime-generated steps."""
    result = {
        level: {
            key: list(value) if isinstance(value, list) else value
            for key, value in record.items()
        }
        for level, record in _COMPILE_DIAGNOSTICS.items()
    }
    try:
        from .runtime import runtime_compile_diagnostics
    except ImportError:
        pass
    else:
        result["runtime"] = runtime_compile_diagnostics()
    return result

def _parameter_vector(params: Mapping[str, Tensor]) -> Tensor:
    return torch.stack([params[name] for name in PARAMETER_NAMES])


def _structure_context(spec, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    decisions = spec.decisions
    values = [
        float(decisions["ARCH1"] == "tension2_1"),
        float(decisions["ARCH1"] == "tension1_1"),
        float(decisions["ARCH1"] == "onestate_1"),
        float(decisions["ARCH2"] == "tens2pll_2"),
        float(decisions["ARCH2"] == "unlimfrc_2"),
        float(decisions["ARCH2"] == "unlimpow_2"),
        float(decisions["ARCH2"] == "fixedsiz_2"),
        float(decisions["QSURF"] == "arno_x_vic"),
        float(decisions["QSURF"] == "prms_varnt"),
        float(decisions["QSURF"] == "tmdl_param"),
        float(decisions["QPERC"] == "perc_f2sat"),
        float(decisions["QPERC"] == "perc_w2sat"),
        float(decisions["QPERC"] == "perc_lower"),
    ]
    values.extend(float(spec.state_mask[name]) for name in STATE_NAMES)
    values.extend(float(spec.parameter_mask[name]) for name in PARAMETER_NAMES)
    return torch.as_tensor(values, dtype=dtype, device=device)


def _theta(theta: Tensor, name: str) -> Tensor:
    return theta[_PARAMETER_INDEX[name]]


def _compiled_flux_rhs(
    state: Tensor,
    effective: Tensor,
    pet: Tensor,
    theta: Tensor,
    context: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
 ) -> tuple[Tensor, Tensor, Tensor]:
    """Universal fixed-shape tensor RHS used by all 78 structures.

    Every structure decision enters as a tensor flag and every state/parameter
    coordinate uses the frozen union layout.  No Newton, line-search, routing,
    Python convergence check, or structure ID enters this function.
    """
    a1_t2 = context[_CTX_A1_T2] > 0.5
    a1_t1 = context[_CTX_A1_T1] > 0.5
    a1_one = context[_CTX_A1_ONE] > 0.5
    a2_pll = context[_CTX_A2_PLL] > 0.5
    a2_frc = context[_CTX_A2_FRC] > 0.5
    a2_pow = context[_CTX_A2_POW] > 0.5
    a2_fixed = context[_CTX_A2_FIXED] > 0.5
    qs_arno = context[_CTX_QS_ARNO] > 0.5
    qs_prms = context[_CTX_QS_PRMS] > 0.5
    qp_f2 = context[_CTX_QP_F2] > 0.5
    qp_w2 = context[_CTX_QP_W2] > 0.5
    zero = state[0] * 0.0
    t1a, t1b, t1, free1_state, watr1_state = state[:5]
    t2_state, free2a, free2b, watr2_state = state[5:]
    fracten = _theta(theta, "FRACTEN")
    maxwatr1 = _theta(theta, "MAXWATR_1")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    maxtens1 = fracten * maxwatr1
    maxfree1 = (1.0 - fracten) * maxwatr1
    maxtens1a = _theta(theta, "FRCHZNE") * fracten * maxwatr1
    maxtens1b = (1.0 - _theta(theta, "FRCHZNE")) * fracten * maxwatr1
    maxtens2 = fracten * maxwatr2
    maxfree2a = _theta(theta, "FPRIMQB") * (1.0 - fracten) * maxwatr2
    maxfree2b = (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - fracten) * maxwatr2
    t2 = torch.where(a2_pll, t2_state, torch.minimum(watr2_state, maxtens2))
    watr2 = torch.where(a2_pll, t2_state + free2a + free2b, watr2_state)
    watr1_t2 = t1a + t1b + free1_state
    watr1_t1 = t1 + free1_state
    watr1_one = watr1_state
    watr1 = torch.where(a1_t2, watr1_t2, torch.where(a1_t1, watr1_t1, watr1_one))
    tens1 = torch.where(a1_t2, t1a + t1b, torch.where(a1_t1, t1, torch.minimum(watr1_state, maxtens1)))
    free1 = torch.where(a1_t2 | a1_t1, free1_state, torch.clamp_min(watr1_state - maxtens1, 0.0))
    ratio1 = (watr1 / maxwatr1).clamp(0.0, 1.0)
    ratio_tens1 = (tens1 / maxtens1).clamp(0.0, 1.0)
    arno_area = 1.0 - (1.0 - ratio1).clamp_min(1.0e-12) ** _theta(theta, "AXV_BEXP").clamp_min(1.0e-6)
    prms_area = ratio_tens1 * _theta(theta, "SAREAMAX")
    shape = _theta(theta, "TISHAPE").clamp_min(1.0e-6)
    qb_power = _theta(theta, "QB_POWR").clamp_min(1.0e-6)
    ti_sat = topographic_mean / (watr2 / maxwatr2 + 1.0e-8)
    ti_log = torch.log(ti_sat.clamp_min(1.0e-12) ** qb_power)
    chi = ((_theta(theta, "LOGLAMB") - 3.0) / shape).clamp_min(1.0e-8)
    argument = torch.clamp_min(ti_log - 3.0, 0.0) / chi
    tmdl_area = torch.where(
        ti_sat > topographic_max,
        zero,
        1.0 - _regularized_gamma_p(shape, argument),
    )
    saturation_area = torch.where(qs_arno, arno_area, torch.where(qs_prms, prms_area, tmdl_area)).clamp(0.0, 1.0)
    qsurf = effective * saturation_area
    evapa = pet * t1a / maxtens1a
    evapb = (pet - evapa) * t1b / maxtens1b
    evap1_split = evapa + evapb
    evap1_single = pet * tens1 / maxtens1
    evap1 = torch.where(a1_t2, evap1_split, evap1_single)
    rchrt2 = _logismooth(t1a, maxtens1a) * (effective - qsurf)
    trans_t1 = _logismooth(tens1, maxtens1) * (effective - qsurf)
    trans_upper = torch.where(a1_t2, _logismooth(t1b, maxtens1b) * rchrt2, torch.where(a1_t1, trans_t1, zero))
    oflow1_split = _logismooth(free1, maxfree1) * trans_upper
    oflow1_one = _logismooth(watr1, maxwatr1) * (effective - qsurf)
    oflow1 = torch.where(a1_one, oflow1_one, oflow1_split)
    qbsat_pll = _theta(theta, "QBRATE_2A") * maxfree2a + _theta(theta, "QBRATE_2B") * maxfree2b
    qbsat_frc = _theta(theta, "QB_PRMS") * maxwatr2
    topmdm = maxwatr2 / 1000.0 / qb_power
    qbsat_pow = _theta(theta, "BASERTE") * topmdm / topographic_mean.clamp_min(1.0e-12) ** qb_power
    qbsat = torch.where(a2_pll, qbsat_pll, torch.where(a2_frc, qbsat_frc, torch.where(a2_pow, qbsat_pow, _theta(theta, "BASERTE"))))
    free1_ratio = (free1 / maxfree1).clamp_min(0.0)
    qperc_f2 = _theta(theta, "PERCRTE") * free1_ratio ** _theta(theta, "PERCEXP")
    qperc_w2 = _theta(theta, "PERCRTE") * (watr1 / maxwatr1).clamp_min(0.0) ** _theta(theta, "PERCEXP")
    demand = 1.0 + _theta(theta, "SACPMLT") * (1.0 - watr2 / maxwatr2).clamp_min(0.0) ** _theta(theta, "SACPEXP")
    qperc_lower = qbsat * demand * free1_ratio
    qperc = torch.where(qp_f2, qperc_f2, torch.where(qp_w2, qperc_w2, qperc_lower))
    evap2_candidate = (pet - evap1) * t2 / maxtens2
    evap2 = torch.where((a2_pll | a2_fixed) & ~a1_t2, evap2_candidate, zero)
    tens2free2 = _logismooth(t2, maxtens2) * qperc * (1.0 - _theta(theta, "PERCFRAC"))
    incoming = qperc * _theta(theta, "PERCFRAC") / 2.0 + tens2free2 / 2.0
    qbase2a = _theta(theta, "QBRATE_2A") * free2a
    qbase2b = _theta(theta, "QBRATE_2B") * free2b
    qbase_pll = qbase2a + qbase2b
    ratio2 = (watr2 / maxwatr2).clamp_min(0.0)
    qbase_frc = _theta(theta, "QB_PRMS") * watr2
    qbase_pow = qbsat * ratio2 ** _theta(theta, "QB_POWR")
    qbase_fixed = _theta(theta, "BASERTE") * ratio2 ** _theta(theta, "QB_POWR")
    qbase = torch.where(a2_pll, qbase_pll, torch.where(a2_frc, qbase_frc, torch.where(a2_pow, qbase_pow, qbase_fixed)))
    oflow2a = _logismooth(free2a, maxfree2a) * incoming
    oflow2b = _logismooth(free2b, maxfree2b) * incoming
    oflow2 = torch.where(a2_pll, oflow2a + oflow2b, torch.where(a2_fixed, _logismooth(watr2, maxwatr2) * qperc, zero))
    rchr = torch.where(a1_t2, rchrt2, zero)
    tens2free1 = trans_upper
    d_t1a = effective - qsurf - evapa - rchr
    d_t1b = rchr - evapb - tens2free1
    d_free1_split = tens2free1 - qperc - oflow1
    d_t1 = effective - qsurf - evap1 - tens2free1
    d_free1_single = tens2free1 - qperc - oflow1
    d_watr1 = effective - qsurf - evap1 - qperc - oflow1
    d_t2 = qperc * (1.0 - _theta(theta, "PERCFRAC")) - evap2 - tens2free2
    d_free2a = incoming - qbase2a - oflow2a
    d_free2b = incoming - qbase2b - oflow2b
    d_watr2 = qperc - evap2 - qbase - oflow2
    rhs = torch.stack((
        torch.where(a1_t2, d_t1a, zero),
        torch.where(a1_t2, d_t1b, zero),
        torch.where(a1_t1, d_t1, zero),
        torch.where(a1_t2 | a1_t1, d_free1_split, zero),
        torch.where(a1_one, d_watr1, zero),
        torch.where(a2_pll, d_t2, zero),
        torch.where(a2_pll, d_free2a, zero),
        torch.where(a2_pll, d_free2b, zero),
        torch.where(~a2_pll, d_watr2, zero),
    ))
    rhs = rhs * context[_CTX_STATE_MASK : _CTX_STATE_MASK + len(STATE_NAMES)]
    flux = torch.stack((
        effective,
        saturation_area,
        torch.where(a1_t2, evapa, zero),
        torch.where(a1_t2, evapb, zero),
        evap1,
        rchr,
        tens2free1,
        qperc,
        zero,
        oflow1,
        qsurf,
        evap2,
        torch.where(a2_pll, tens2free2, zero),
        torch.where(a2_pll, qbase2a, zero),
        torch.where(a2_pll, qbase2b, zero),
        qbase,
        torch.where(a2_pll, oflow2a, zero),
        torch.where(a2_pll, oflow2b, zero),
        oflow2,
    ))
    diagnostics = torch.stack((saturation_area, qperc, qbase, qsurf, oflow1 + oflow2, evap1 + evap2))
    return rhs, flux, diagnostics

class _CompiledFluxAutograd(torch.autograd.Function):
    """Keep Inductor in forward-only mode while retaining differentiable Newton."""

    @staticmethod
    def forward(ctx, compiled, state, effective, pet, theta, context, topographic_mean, topographic_max):
        ctx.save_for_backward(
            state, effective, pet, theta, context, topographic_mean, topographic_max
        )
        with torch.no_grad():
            return compiled(
                state.detach(),
                effective.detach(),
                pet.detach(),
                theta.detach(),
                context.detach(),
                topographic_mean.detach(),
                topographic_max.detach(),
            )

    @staticmethod
    def backward(ctx, grad_rhs, grad_flux, grad_diagnostics):
        originals = ctx.saved_tensors
        local = tuple(value.detach().requires_grad_(True) for value in originals)
        input_indices = [
            index for index, value in enumerate(originals) if value.requires_grad
        ]
        create_graph = torch.is_grad_enabled()
        gradients = [None] * len(originals)
        if input_indices:
            inputs = tuple(local[index] for index in input_indices)
            with torch.enable_grad():
                proxy = tuple(
                    local_value + (original - original.detach())
                    for local_value, original in zip(local, originals)
                )
                rhs, flux, diagnostics = _compiled_flux_rhs(*proxy)
                outputs = (rhs, flux, diagnostics)
                output_gradients = (
                    torch.zeros_like(rhs) if grad_rhs is None else grad_rhs,
                    torch.zeros_like(flux) if grad_flux is None else grad_flux,
                    torch.zeros_like(diagnostics) if grad_diagnostics is None else grad_diagnostics,
                )
                active_outputs = [
                    (output, gradient)
                    for output, gradient in zip(outputs, output_gradients)
                    if output.requires_grad
                ]
                if active_outputs:
                    input_gradients = torch.autograd.grad(
                        tuple(output for output, _ in active_outputs),
                        inputs,
                        grad_outputs=tuple(gradient for _, gradient in active_outputs),
                        allow_unused=True,
                        create_graph=create_graph,
                    )
                else:
                    input_gradients = (None,) * len(inputs)
            for index, gradient in zip(input_indices, input_gradients):
                gradients[index] = gradient
        return (None, *gradients)

def _dynamo_counter_snapshot() -> dict[str, dict[str, int]]:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    return {
        str(group): {str(name): int(value) for name, value in counter.items()}
        for group, counter in counters.items()
        if group in {"frames", "inductor", "graph_break", "stats", "aot_autograd"} and counter
    }

def _update_dynamo_audit(record: dict[str, object]) -> None:
    snapshot = _dynamo_counter_snapshot()
    record["dynamo_counters"] = snapshot
    stats = snapshot.get("stats", {})
    record["unique_graphs"] = int(stats.get("unique_graphs", 0))
    record["graph_breaks"] = sum(snapshot.get("graph_break", {}).values())
    record["recompilations"] = max(int(record["unique_graphs"]) - int(record["compile_successes"]), 0)


def _reset_dynamo_counters() -> None:
    try:
        import torch._dynamo as dynamo
        from torch._dynamo.utils import counters
        dynamo.reset()
        counters.clear()
    except Exception:
        pass


def _new_compile_record() -> dict[str, object]:
    return {
        "compile_attempts": 0,
        "compile_successes": 0,
        "fallbacks": 0,
        "calls": 0,
        "cold_compile_seconds": [],
        "failures": [],
        "cache_keys": [],
        "dynamo_counters": {},
    }


def _record_compile_failure(level: str, exc: BaseException) -> None:
    record = _COMPILE_DIAGNOSTICS[level]
    record["fallbacks"] = int(record["fallbacks"]) + 1
    failures = record["failures"]
    if len(failures) < 8:
        failures.append(f"{type(exc).__name__}: {str(exc)[:500]}")


def _get_compiled_rhs(
    *,
    device: torch.device,
    dtype: torch.dtype,
    backend: str,
    fullgraph: bool,
 ) -> tuple[Callable[..., tuple[Tensor, Tensor, Tensor]], tuple[object, ...], dict[str, object]]:
    key = ("rhs", backend, bool(fullgraph), device.type, device.index, str(dtype))
    record = _COMPILE_DIAGNOSTICS["level_a"]
    if key not in _COMPILED_RHS_CACHE:
        record["compile_attempts"] = int(record["compile_attempts"]) + 1
        record["cache_keys"].append([str(item) for item in key])
        _COMPILED_RHS_CACHE[key] = torch.compile(
            _compiled_flux_rhs, backend=backend, fullgraph=fullgraph
        )
        _COMPILED_RHS_CACHE[("meta",) + key] = {"warmed": False}
    return _COMPILED_RHS_CACHE[key], key, _COMPILED_RHS_CACHE[("meta",) + key]


class _CompiledInnerKernel:
    """Python adapter around one fixed-shape compiled tensor RHS callable."""

    def __init__(
        self,
        spec,
        params: Mapping[str, Tensor],
        cap: Mapping[str, Tensor],
        topographic: tuple[Tensor, Tensor] | None,
        *,
        backend: str,
        fullgraph: bool,
     ) -> None:
        self.spec = spec
        self.params = params
        self.cap = cap
        self.theta = _parameter_vector(params)
        self.context = _structure_context(
            spec, dtype=self.theta.dtype, device=self.theta.device
        )
        if topographic is None:
            zero_mean = torch.zeros_like(self.theta[0])
            zero_max = torch.zeros_like(self.theta[0])
            self.topographic = (zero_mean, zero_max)
        else:
            self.topographic = topographic
        self.backend = backend
        self.fullgraph = fullgraph
        self.failed = False
        self.failure_recorded = False

    def _union_state(self, state: Tensor) -> Tensor:
        zero = state[0] * 0.0
        active = dict(zip(self.spec.state_names, state))
        return torch.stack([active.get(name, zero) for name in STATE_NAMES])

    def _eager_result(
        self, state: Tensor, effective: Tensor, pet: Tensor
     ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        derivative, flux = _eager_derivatives(
            state, self.spec, self.params, self.cap, effective, pet, self.topographic
        )
        diagnostics = torch.stack(
            (
                flux["SATAREA"],
                flux["QPERC_12"],
                flux["QBASE_2"],
                flux["QSURF"],
                flux["OFLOW_1"] + flux["OFLOW_2"],
                flux["EVAP_1"] + flux["EVAP_2"],
            )
        )
        return derivative, flux, diagnostics

    def evaluate(
        self, state: Tensor, effective: Tensor, pet: Tensor
     ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        record = _COMPILE_DIAGNOSTICS["level_a"]
        record["calls"] = int(record["calls"]) + 1
        if self.failed:
            return self._eager_result(state, effective, pet)
        try:
            compiled, key, meta = _get_compiled_rhs(
                device=self.theta.device,
                dtype=self.theta.dtype,
                backend=self.backend,
                fullgraph=self.fullgraph,
            )
            started = time.perf_counter() if not meta["warmed"] else None
            full_state = self._union_state(state)
            rhs, flux_tensor, diagnostics = _CompiledFluxAutograd.apply(
                compiled, full_state, effective, pet, self.theta, self.context,
                self.topographic[0], self.topographic[1],
            )
            if not meta["warmed"]:
                meta["warmed"] = True
                record["compile_successes"] = int(record["compile_successes"]) + 1
                record["cold_compile_seconds"].append(time.perf_counter() - started)
            _update_dynamo_audit(record)
            flux = {name: flux_tensor[index] for index, name in enumerate(FLUX_NAMES)}
            derivative = torch.stack([rhs[_STATE_INDEX[name]] for name in self.spec.state_names])
            return derivative, flux, diagnostics
        except Exception as exc:
            self.failed = True
            if not self.failure_recorded:
                _record_compile_failure("level_a", exc)
                self.failure_recorded = True
            return self._eager_result(state, effective, pet)

def _compiled_residual(
    candidate: Tensor,
    old_state: Tensor,
    effective: Tensor,
    pet: Tensor,
    dt_days: Tensor,
    theta: Tensor,
    context: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
 ) -> tuple[Tensor, Tensor, Tensor]:
    rhs, flux, diagnostics = _compiled_flux_rhs(
        candidate, effective, pet, theta, context, topographic_mean, topographic_max
    )
    return candidate - old_state - dt_days * rhs, flux, diagnostics

class _CompiledResidualAutograd(torch.autograd.Function):
    """Forward through Inductor; eager differentiable backward for one residual."""

    @staticmethod
    def forward(ctx, compiled, candidate, old_state, effective, pet, dt_days, theta, context, topographic_mean, topographic_max):
        ctx.save_for_backward(
            candidate, old_state, effective, pet, dt_days, theta, context,
            topographic_mean, topographic_max,
        )
        with torch.no_grad():
            return compiled(
                candidate.detach(),
                old_state.detach(),
                effective.detach(),
                pet.detach(),
                dt_days.detach(),
                theta.detach(),
                context.detach(),
                topographic_mean.detach(),
                topographic_max.detach(),
            )

    @staticmethod
    def backward(ctx, grad_residual, grad_flux, grad_diagnostics):
        originals = ctx.saved_tensors
        local = tuple(value.detach().requires_grad_(True) for value in originals)
        input_indices = [
            index for index, value in enumerate(originals) if value.requires_grad
        ]
        gradients = [None] * len(originals)
        create_graph = torch.is_grad_enabled()
        if input_indices:
            inputs = tuple(local[index] for index in input_indices)
            with torch.enable_grad():
                proxy = tuple(
                    local_value + (original - original.detach())
                    for local_value, original in zip(local, originals)
                )
                residual, flux, diagnostics = _compiled_residual(*proxy)
                outputs = (residual, flux, diagnostics)
                output_gradients = (
                    torch.zeros_like(residual) if grad_residual is None else grad_residual,
                    torch.zeros_like(flux) if grad_flux is None else grad_flux,
                    torch.zeros_like(diagnostics) if grad_diagnostics is None else grad_diagnostics,
                )
                active_outputs = [
                    (output, gradient)
                    for output, gradient in zip(outputs, output_gradients)
                    if output.requires_grad
                ]
                if active_outputs:
                    input_gradients = torch.autograd.grad(
                        tuple(output for output, _ in active_outputs),
                        inputs,
                        grad_outputs=tuple(gradient for _, gradient in active_outputs),
                        allow_unused=True,
                        create_graph=create_graph,
                    )
                else:
                    input_gradients = (None,) * len(inputs)
            for index, gradient in zip(input_indices, input_gradients):
                gradients[index] = gradient
        return (None, *gradients)


def _get_compiled_residual(
    *,
    device: torch.device,
    dtype: torch.dtype,
    backend: str,
    fullgraph: bool,
 ) -> tuple[Callable[..., tuple[Tensor, Tensor, Tensor]], tuple[object, ...], dict[str, object]]:
    key = ("residual", backend, bool(fullgraph), device.type, device.index, str(dtype))
    record = _COMPILE_DIAGNOSTICS["level_b"]
    if key not in _COMPILED_RESIDUAL_CACHE:
        record["compile_attempts"] = int(record["compile_attempts"]) + 1
        record["cache_keys"].append([str(item) for item in key])
        _COMPILED_RESIDUAL_CACHE[key] = torch.compile(
            _compiled_residual, backend=backend, fullgraph=fullgraph
        )
        _COMPILED_RESIDUAL_CACHE[("meta",) + key] = {"warmed": False}
    return _COMPILED_RESIDUAL_CACHE[key], key, _COMPILED_RESIDUAL_CACHE[("meta",) + key]


class _CompiledResidualKernel:
    """Adapter for one independently tested residual evaluation."""

    def __init__(
        self,
        spec,
        params: Mapping[str, Tensor],
        cap: Mapping[str, Tensor],
        topographic: tuple[Tensor, Tensor] | None,
        *,
        backend: str,
        fullgraph: bool,
     ) -> None:
        self.spec = spec
        self.params = params
        self.cap = cap
        self.theta = _parameter_vector(params)
        self.context = _structure_context(
            spec, dtype=self.theta.dtype, device=self.theta.device
        )
        if topographic is None:
            zero_mean = torch.zeros_like(self.theta[0])
            zero_max = torch.zeros_like(self.theta[0])
            self.topographic = (zero_mean, zero_max)
        else:
            self.topographic = topographic
        self.backend = backend
        self.fullgraph = fullgraph
        self.failed = False
        self.failure_recorded = False

    def _union_state(self, state: Tensor) -> Tensor:
        zero = state[0] * 0.0
        active = dict(zip(self.spec.state_names, state))
        return torch.stack([active.get(name, zero) for name in STATE_NAMES])

    def evaluate(
        self,
        candidate: Tensor,
        old_state: Tensor,
        effective: Tensor,
        pet: Tensor,
        dt_days: float,
     ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        record = _COMPILE_DIAGNOSTICS["level_b"]
        record["calls"] = int(record["calls"]) + 1
        if self.failed:
            derivative, flux = _eager_derivatives(
                candidate, self.spec, self.params, self.cap, effective, pet, self.topographic
            )
            return candidate - old_state - dt_days * derivative, flux, torch.stack(
                (
                    flux["SATAREA"],
                    flux["QPERC_12"],
                    flux["QBASE_2"],
                    flux["QSURF"],
                    flux["OFLOW_1"] + flux["OFLOW_2"],
                    flux["EVAP_1"] + flux["EVAP_2"],
                )
            )
        try:
            compiled, key, meta = _get_compiled_residual(
                device=self.theta.device,
                dtype=self.theta.dtype,
                backend=self.backend,
                fullgraph=self.fullgraph,
            )
            started = time.perf_counter() if not meta["warmed"] else None
            residual, flux_tensor, diagnostics = _CompiledResidualAutograd.apply(
                compiled,
                self._union_state(candidate),
                self._union_state(old_state),
                effective,
                pet,
                torch.as_tensor(dt_days, dtype=self.theta.dtype, device=self.theta.device),
                self.theta,
                self.context,
                self.topographic[0],
                self.topographic[1],
            )
            if not meta["warmed"]:
                meta["warmed"] = True
                record["compile_successes"] = int(record["compile_successes"]) + 1
                record["cold_compile_seconds"].append(time.perf_counter() - started)
            _update_dynamo_audit(record)
            active_residual = torch.stack([residual[_STATE_INDEX[name]] for name in self.spec.state_names])
            flux = {name: flux_tensor[index] for index, name in enumerate(FLUX_NAMES)}
            return active_residual, flux, diagnostics
        except Exception as exc:
            self.failed = True
            if not self.failure_recorded:
                _record_compile_failure("level_b", exc)
                self.failure_recorded = True
            derivative, flux = _eager_derivatives(
                candidate, self.spec, self.params, self.cap, effective, pet, self.topographic
            )
            return candidate - old_state - dt_days * derivative, flux, torch.stack(
                (
                    flux["SATAREA"],
                    flux["QPERC_12"],
                    flux["QBASE_2"],
                    flux["QSURF"],
                    flux["OFLOW_1"] + flux["OFLOW_2"],
                    flux["EVAP_1"] + flux["EVAP_2"],
                )
            )


def _fluxes(state: Tensor, spec, params: Mapping[str, Tensor], cap: Mapping[str, Tensor], eff_ppt: Tensor, pet: Tensor, topographic: tuple[Tensor, Tensor] | None = None) -> dict[str, Tensor]:
    zero = state[0] * 0.0
    values = {name: zero for name in FLUX_NAMES}
    current = _derived_states(_state_dict(state, spec.state_names), spec, cap)
    decisions = spec.decisions

    values["EFF_PPT"] = eff_ppt
    if decisions["QSURF"] == "arno_x_vic":
        ratio = (current["WATR_1"] / cap["MAXWATR_1"]).clamp(0.0, 1.0)
        saturation_area = 1.0 - (1.0 - ratio).clamp_min(1.0e-12) ** params["AXV_BEXP"].clamp_min(1.0e-6)
    elif decisions["QSURF"] == "prms_varnt":
        saturation_area = (current["TENS_1"] / cap["MAXTENS_1"]).clamp(0.0, 1.0) * params["SAREAMAX"]
    else:
        mean_power, max_power = _topographic_mean(params) if topographic is None else topographic
        ti_sat = mean_power / (current["WATR_2"] / params["MAXWATR_2"] + 1.0e-8)
        qb_power = params["QB_POWR"].clamp_min(1.0e-6)
        ti_log = torch.log(ti_sat.clamp_min(1.0e-12) ** qb_power)
        shape = params["TISHAPE"].clamp_min(1.0e-6)
        chi = ((params["LOGLAMB"] - 3.0) / shape).clamp_min(1.0e-8)
        arg = torch.clamp_min(ti_log - 3.0, 0.0) / chi
        saturation_area = torch.where(
            ti_sat > max_power,
            zero,
            1.0 - _regularized_gamma_p(shape, arg),
        )
    values["SATAREA"] = saturation_area.clamp(0.0, 1.0)
    values["QSURF"] = eff_ppt * values["SATAREA"]

    if decisions["ARCH1"] == "tension2_1":
        values["EVAP_1A"] = pet * current["TENS_1A"] / cap["MAXTENS_1A"]
        values["EVAP_1B"] = (pet - values["EVAP_1A"]) * current["TENS_1B"] / cap["MAXTENS_1B"]
        values["EVAP_1"] = values["EVAP_1A"] + values["EVAP_1B"]
        values["RCHR2EXCS"] = _logismooth(current["TENS_1A"], cap["MAXTENS_1A"]) * (eff_ppt - values["QSURF"])
        values["TENS2FREE_1"] = _logismooth(current["TENS_1B"], cap["MAXTENS_1B"]) * values["RCHR2EXCS"]
        values["OFLOW_1"] = _logismooth(current["FREE_1"], cap["MAXFREE_1"]) * values["TENS2FREE_1"]
    else:
        values["EVAP_1"] = pet * current["TENS_1"] / cap["MAXTENS_1"]
        if decisions["ARCH1"] == "tension1_1":
            values["TENS2FREE_1"] = _logismooth(current["TENS_1"], cap["MAXTENS_1"]) * (eff_ppt - values["QSURF"])
            values["OFLOW_1"] = _logismooth(current["FREE_1"], cap["MAXFREE_1"]) * values["TENS2FREE_1"]
        else:
            values["OFLOW_1"] = _logismooth(current["WATR_1"], cap["MAXWATR_1"]) * (eff_ppt - values["QSURF"])

    if decisions["ARCH2"] in ("tens2pll_2", "fixedsiz_2") and decisions["ARCH1"] != "tension2_1":
        values["EVAP_2"] = (pet - values["EVAP_1"]) * current["TENS_2"] / cap["MAXTENS_2"]

    if decisions["QPERC"] == "perc_f2sat":
        values["QPERC_12"] = params["PERCRTE"] * (current["FREE_1"] / cap["MAXFREE_1"]).clamp_min(0.0) ** params["PERCEXP"]
    elif decisions["QPERC"] == "perc_w2sat":
        values["QPERC_12"] = params["PERCRTE"] * (current["WATR_1"] / params["MAXWATR_1"]).clamp_min(0.0) ** params["PERCEXP"]
    else:
        qbsat = _qbsat(current, spec, params, cap, topographic)
        demand = 1.0 + params["SACPMLT"] * (1.0 - current["WATR_2"] / params["MAXWATR_2"]).clamp_min(0.0) ** params["SACPEXP"]
        values["QPERC_12"] = qbsat * demand * (current["FREE_1"] / cap["MAXFREE_1"]).clamp_min(0.0)

    if decisions["ARCH2"] == "tens2pll_2":
        values["TENS2FREE_2"] = _logismooth(current["TENS_2"], cap["MAXTENS_2"]) * values["QPERC_12"] * (1.0 - params["PERCFRAC"])
        incoming = values["QPERC_12"] * params["PERCFRAC"] / 2.0 + values["TENS2FREE_2"] / 2.0
        values["QBASE_2A"] = params["QBRATE_2A"] * current["FREE_2A"]
        values["QBASE_2B"] = params["QBRATE_2B"] * current["FREE_2B"]
        values["QBASE_2"] = values["QBASE_2A"] + values["QBASE_2B"]
        values["OFLOW_2A"] = _logismooth(current["FREE_2A"], cap["MAXFREE_2A"]) * incoming
        values["OFLOW_2B"] = _logismooth(current["FREE_2B"], cap["MAXFREE_2B"]) * incoming
        values["OFLOW_2"] = values["OFLOW_2A"] + values["OFLOW_2B"]
    else:
        values["QBASE_2"] = _qbase(current, spec, params, cap)
        if decisions["ARCH2"] == "fixedsiz_2":
            values["OFLOW_2"] = _logismooth(current["WATR_2"], cap["MAXWATR_2"]) * values["QPERC_12"]
    return values


def _qbsat(state: Mapping[str, Tensor], spec, params: Mapping[str, Tensor], cap: Mapping[str, Tensor], topographic: tuple[Tensor, Tensor] | None = None) -> Tensor:
    arch2 = spec.decisions["ARCH2"]
    if arch2 == "tens2pll_2":
        return params["QBRATE_2A"] * cap["MAXFREE_2A"] + params["QBRATE_2B"] * cap["MAXFREE_2B"]
    if arch2 == "unlimfrc_2":
        return params["QB_PRMS"] * params["MAXWATR_2"]
    if arch2 == "unlimpow_2":
        topmdm = params["MAXWATR_2"] / 1000.0 / params["QB_POWR"]
        mean_power, _ = _topographic_mean(params) if topographic is None else topographic
        return params["BASERTE"] * topmdm / mean_power.clamp_min(1.0e-12) ** params["QB_POWR"]
    return params["BASERTE"]


def _qbase(state: Mapping[str, Tensor], spec, params: Mapping[str, Tensor], cap: Mapping[str, Tensor]) -> Tensor:
    arch2 = spec.decisions["ARCH2"]
    if arch2 == "unlimfrc_2":
        return params["QB_PRMS"] * state["WATR_2"]
    if arch2 == "unlimpow_2":
        qbsat = _qbsat(state, spec, params, cap)
        return qbsat * (state["WATR_2"] / params["MAXWATR_2"]).clamp_min(0.0) ** params["QB_POWR"]
    return params["BASERTE"] * (state["WATR_2"] / params["MAXWATR_2"]).clamp_min(0.0) ** params["QB_POWR"]


def _eager_derivatives(state: Tensor, spec, params: Mapping[str, Tensor], cap: Mapping[str, Tensor], eff_ppt: Tensor, pet: Tensor, topographic: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    flux = _fluxes(state, spec, params, cap, eff_ppt, pet, topographic)
    current = _derived_states(_state_dict(state, spec.state_names), spec, cap)
    upper = spec.decisions["ARCH1"]
    if upper == "tension2_1":
        deriv = {
            "TENS_1A": flux["EFF_PPT"] - flux["QSURF"] - flux["EVAP_1A"] - flux["RCHR2EXCS"],
            "TENS_1B": flux["RCHR2EXCS"] - flux["EVAP_1B"] - flux["TENS2FREE_1"],
            "FREE_1": flux["TENS2FREE_1"] - flux["QPERC_12"] - flux["QINTF_1"] - flux["OFLOW_1"],
        }
    elif upper == "tension1_1":
        deriv = {
            "TENS_1": flux["EFF_PPT"] - flux["QSURF"] - flux["EVAP_1"] - flux["TENS2FREE_1"],
            "FREE_1": flux["TENS2FREE_1"] - flux["QPERC_12"] - flux["QINTF_1"] - flux["OFLOW_1"],
        }
    else:
        deriv = {"WATR_1": flux["EFF_PPT"] - flux["QSURF"] - flux["EVAP_1"] - flux["QPERC_12"] - flux["QINTF_1"] - flux["OFLOW_1"]}

    if spec.decisions["ARCH2"] == "tens2pll_2":
        deriv.update(
            {
                "TENS_2": flux["QPERC_12"] * (1.0 - params["PERCFRAC"]) - flux["EVAP_2"] - flux["TENS2FREE_2"],
                "FREE_2A": flux["QPERC_12"] * params["PERCFRAC"] / 2.0 + flux["TENS2FREE_2"] / 2.0 - flux["QBASE_2A"] - flux["OFLOW_2A"],
                "FREE_2B": flux["QPERC_12"] * params["PERCFRAC"] / 2.0 + flux["TENS2FREE_2"] / 2.0 - flux["QBASE_2B"] - flux["OFLOW_2B"],
            }
        )
    else:
        deriv["WATR_2"] = flux["QPERC_12"] - flux["EVAP_2"] - flux["QBASE_2"] - flux["OFLOW_2"]
    return torch.stack([deriv[name] for name in spec.state_names]), flux

def _derivatives(
    state: Tensor,
    spec,
    params: Mapping[str, Tensor],
    cap: Mapping[str, Tensor],
    eff_ppt: Tensor,
    pet: Tensor,
    topographic: tuple[Tensor, Tensor] | None = None,
    inner: _CompiledInnerKernel | None = None,
 ) -> tuple[Tensor, dict[str, Tensor]]:
    if inner is None:
        return _eager_derivatives(state, spec, params, cap, eff_ppt, pet, topographic)
    derivative, flux, _ = inner.evaluate(state, eff_ppt, pet)
    return derivative, flux

def evaluate_implicit_residual(
    model_id: int | str,
    state_candidate: Tensor,
    state_old: Tensor,
    effective: Tensor | float,
    pet: Tensor | float,
    params: Mapping[str, object] | Tensor | None = None,
    dt_days: float = 1.0,
    *,
    compile_residual: bool = False,
    compile_backend: str = "inductor",
    compile_fullgraph: bool = True,
 ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    """Evaluate one implicit residual, optionally with a Level-B compiler.

    This function performs exactly one residual evaluation.  It never runs a
    Newton iteration, Jacobian construction, line search, or time loop.
    """
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    if not isinstance(state_candidate, Tensor):
        state_candidate = torch.as_tensor(state_candidate)
    if not state_candidate.is_floating_point():
        state_candidate = state_candidate.to(dtype=torch.get_default_dtype())
    state_candidate = state_candidate.reshape(-1)
    state_old = _to_tensor(
        state_old, dtype=state_candidate.dtype, device=state_candidate.device
    ).reshape(-1)
    spec = get_structure(model_id)
    if state_candidate.numel() != len(spec.state_names) or state_old.numel() != len(spec.state_names):
        raise ValueError(f"residual states must have {len(spec.state_names)} active coordinates")
    effective = _to_tensor(effective, dtype=state_candidate.dtype, device=state_candidate.device)
    pet = _to_tensor(pet, dtype=state_candidate.dtype, device=state_candidate.device)
    params_tensor = _parameter_values(
        params, dtype=state_candidate.dtype, device=state_candidate.device
    )
    cap = _capacity(params_tensor)
    topographic = None
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        topographic = _topographic_mean(params_tensor)
    if compile_residual:
        runner = _CompiledResidualKernel(
            spec,
            params_tensor,
            cap,
            topographic,
            backend=compile_backend,
            fullgraph=compile_fullgraph,
        )
        return runner.evaluate(state_candidate, state_old, effective, pet, dt_days)
    derivative, flux = _eager_derivatives(
        state_candidate, spec, params_tensor, cap, effective, pet, topographic
    )
    diagnostics = torch.stack(
        (
            flux["SATAREA"],
            flux["QPERC_12"],
            flux["QBASE_2"],
            flux["QSURF"],
            flux["OFLOW_1"] + flux["OFLOW_2"],
            flux["EVAP_1"] + flux["EVAP_2"],
        )
    )
    return state_candidate - state_old - dt_days * derivative, flux, diagnostics


def _snow_step(
    ppt: Tensor,
    temp: Tensor,
    params: Mapping[str, Tensor],
    snow: Tensor,
    jday: Tensor,
    leap: Tensor,
    dt: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    denom = torch.where(leap, jday * 0.0 + 366.0, jday * 0.0 + 365.0)
    offset = torch.where(leap, jday * 0.0 + 81.0, jday * 0.0 + 80.0)
    melt_factor = (0.5 * torch.sin((jday - offset) * 2.0 * torch.pi / denom) + 0.5) * (params["MFMAX"] - params["MFMIN"]) + params["MFMIN"]
    melt = torch.where((snow > 0.0) & (temp > params["MBASE"]), melt_factor * (temp - params["MBASE"]), snow * 0.0)
    accumulation = torch.where(temp < params["PXTEMP"], ppt * params["RFERR_MLT"], snow * 0.0)
    next_snow = snow + (accumulation - melt) * dt
    melt = torch.where(next_snow >= 0.0, melt, snow / dt + accumulation)
    next_snow = torch.clamp_min(snow + (accumulation - melt) * dt, 0.0)
    effective = torch.where(temp > params["PXTEMP"], ppt * params["RFERR_MLT"] + melt, melt)
    return effective, next_snow, accumulation - melt

def _snow_balance_input(ppt: Tensor, temp: Tensor, params: Mapping[str, Tensor]) -> Tensor:
    """Return precipitation participating in the source snow partition."""
    precipitation = ppt * params["RFERR_MLT"]
    zero = precipitation * 0.0
    return torch.where(
        temp < params["PXTEMP"],
        precipitation,
        torch.where(temp > params["PXTEMP"], precipitation, zero),
    )


def _implicit_step(
    state: Tensor,
    spec,
    params: Mapping[str, Tensor],
    cap: Mapping[str, Tensor],
    effective: Tensor,
    pet: Tensor,
    dt_days: float,
    iterations: int,
    topographic: tuple[Tensor, Tensor] | None = None,
    inner: _CompiledInnerKernel | None = None,
 ) -> Tensor:
    """Solve the reference-style old-state implicit-Euler residual.

    The frozen numerix selects ``STATE_OLD`` as the Newton initial guess and
    line-searches every candidate.  Jacobians are constructed with PyTorch
    autograd so the solve remains differentiable with respect to parameters.
    """
    tracking = torch.is_grad_enabled()
    with torch.enable_grad():
        trial = state.clone()
        trial.requires_grad_(True)
        for _ in range(max(1, iterations)):
            derivative, _ = _derivatives(trial, spec, params, cap, effective, pet, topographic, inner)
            residual = trial - state - dt_days * derivative
            residual_norm = residual.abs().amax()
            if float(residual_norm.detach()) <= 1.0e-12:
                break
            rows = []
            for component in range(trial.numel()):
                rows.append(torch.autograd.grad(
                    residual[component], trial, retain_graph=True, create_graph=tracking, allow_unused=False
                )[0])
            jacobian = torch.stack(rows)
            delta = torch.linalg.lstsq(jacobian, -residual.unsqueeze(-1)).solution.squeeze(-1)
            candidate = _project_state(trial + delta, spec, cap).clone()
            candidate.requires_grad_(True)
            candidate_residual = candidate - state - dt_days * _derivatives(
                candidate, spec, params, cap, effective, pet, topographic, inner
            )[0]
            for _ in range(12):
                if float(candidate_residual.abs().amax().detach()) <= float(residual_norm.detach()):
                    break
                candidate = _project_state(trial + 0.5 * delta, spec, cap).clone()
                candidate.requires_grad_(True)
                candidate_residual = candidate - state - dt_days * _derivatives(
                    candidate, spec, params, cap, effective, pet, topographic, inner
                )[0]
                delta = 0.5 * delta
            trial = candidate
            if float(delta.abs().amax().detach()) <= 1.0e-12:
                break
    return trial if tracking else trial.detach()


def _enforce_boundary_water_balance(
    state: Tensor,
    state_next: Tensor,
    spec,
    cap: Mapping[str, Tensor],
    flux: dict[str, Tensor],
    effective: Tensor,
    dt_days: float,
) -> dict[str, Tensor]:
    """Apply one shared overflow correction when a bounded state is clipped.

    This is the tensor analogue of the upstream ``FIX_STATES`` conservation
    step.  It changes only shared overflow/runoff flux bookkeeping and is a
    no-op away from a state bound.  The correction is global because the public
    kernel stores basin-aggregated states rather than the upstream trial-state
    structures used for sequential disaggregation.
    """
    state_capacity = {
        "TENS_1A": "MAXTENS_1A", "TENS_1B": "MAXTENS_1B",
        "TENS_1": "MAXTENS_1", "FREE_1": "MAXFREE_1", "WATR_1": "MAXWATR_1",
        "TENS_2": "MAXTENS_2", "FREE_2A": "MAXFREE_2A", "FREE_2B": "MAXFREE_2B",
        "WATR_2": "MAXWATR_2",
    }
    at_bound = False
    for index, name in enumerate(spec.state_names):
        maximum = cap[state_capacity[name]]
        upper_bounded = name != "WATR_2" or spec.decisions["ARCH2"] == "fixedsiz_2"
        if bool((state_next[index] <= maximum * 1.0e-8).detach()) or (upper_bounded and bool((state_next[index] >= maximum * (1.0 - 1.0e-6)).detach())):
            at_bound = True
    if not at_bound:
        return flux
    outflow_names = ("QSURF", "OFLOW_1", "QINTF_1", "OFLOW_2", "QBASE_2")
    outflow = sum((flux[name] for name in outflow_names), flux["EFF_PPT"] * 0.0)
    external = effective - flux["EVAP_1"] - flux["EVAP_2"] - outflow
    correction = (state_next.sum() - state.sum()) / dt_days - external
    corrected = dict(flux)
    adjustment = -correction
    if bool((adjustment.abs() > 1.0e-10).detach()) and bool((outflow + adjustment >= 0.0).detach()):
        corrected["OFLOW_1"] = corrected["OFLOW_1"] + adjustment
    elif bool((adjustment.abs() > 1.0e-10).detach()) and bool((outflow > 0.0).detach()):
        scale = ((outflow + adjustment) / outflow).clamp(0.0, 1.0)
        for name in outflow_names:
            corrected[name] = corrected[name] * scale
    return corrected
def _routing_fractions(time_delay: Tensor, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    bins = torch.arange(1, 501, device=device, dtype=dtype)
    alpha = torch.as_tensor(2.5, device=device, dtype=dtype)
    cumulative = _regularized_gamma_p(alpha, alpha / time_delay.clamp_min(1.0e-6) * bins)
    previous = torch.cat((cumulative[:1] * 0.0, cumulative[:-1]))
    increments = (cumulative - previous).clamp_min(0.0)
    return increments / increments.sum().clamp_min(1.0e-12)


def _initial_state(spec, params: Mapping[str, Tensor], cap: Mapping[str, Tensor], fraction: float) -> Tensor:
    result = []
    for name in spec.state_names:
        if name == "WATR_1":
            value = params["MAXWATR_1"] * fraction
        elif name == "WATR_2":
            value = params["MAXWATR_2"] * fraction
        elif name == "TENS_1":
            value = cap["MAXTENS_1"] * fraction
        elif name == "FREE_1":
            value = cap["MAXFREE_1"] * fraction
        elif name == "TENS_2":
            value = cap["MAXTENS_2"] * fraction
        elif name == "TENS_1A":
            value = cap["MAXTENS_1A"] * fraction
        elif name == "TENS_1B":
            value = cap["MAXTENS_1B"] * fraction
        elif name == "FREE_2A":
            value = cap["MAXFREE_2A"] * fraction
        elif name == "FREE_2B":
            value = cap["MAXFREE_2B"] * fraction
        else:
            raise ValueError(f"cannot initialize state {name}")
        result.append(value)
    return torch.stack(result)


def simulate_explicit(
    model_id: int | str,
    forcing: Tensor | Mapping[str, object],
    params: Mapping[str, object] | Tensor | None = None,
    *,
    initial_state: Tensor | Mapping[str, object] | None = None,
    initial_fraction: float = 0.25,
    dates: Sequence[date | datetime] | Tensor | None = None,
    dt_days: float = 1.0,
    n_substeps: int = 1,
) -> SimulationResult:
    """Run fixed-substep forward Euler without altering the implicit path."""
    if not isinstance(n_substeps, int) or isinstance(n_substeps, bool) or n_substeps < 1:
        raise ValueError("n_substeps must be a positive integer")
    spec = get_structure(model_id)
    forcing_tensor = _forcing_tensor(forcing)
    if forcing_tensor.shape[0] == 0:
        raise ValueError("forcing must contain at least one time step")
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    dtype = forcing_tensor.dtype if forcing_tensor.is_floating_point() else torch.get_default_dtype()
    forcing_tensor = forcing_tensor.to(dtype=dtype)
    params_tensor = _parameter_values(params, dtype=dtype, device=forcing_tensor.device)
    cap = _capacity(params_tensor)
    if initial_state is None:
        state = _initial_state(spec, params_tensor, cap, initial_fraction)
    elif isinstance(initial_state, Mapping):
        missing = [name for name in spec.state_names if name not in initial_state]
        if missing:
            raise ValueError(f"initial_state missing active coordinates: {missing}")
        state = torch.stack([_to_tensor(initial_state[name], dtype=dtype, device=forcing_tensor.device) for name in spec.state_names])
    else:
        state = initial_state.to(device=forcing_tensor.device, dtype=dtype)
        if state.ndim != 1 or state.numel() != len(spec.state_names):
            raise ValueError(f"initial_state must have {len(spec.state_names)} active coordinates")
    state = _project_state(state, spec, cap)
    days, leap_years = _day_of_year(forcing_tensor.shape[0], dates, dtype=dtype, device=forcing_tensor.device)
    snow = torch.zeros((), dtype=dtype, device=forcing_tensor.device)
    future = torch.zeros(500, dtype=dtype, device=forcing_tensor.device)
    fractions = _routing_fractions(params_tensor["TIMEDELAY"], dtype=dtype, device=forcing_tensor.device)
    topographic = None
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        topographic = _topographic_mean(params_tensor)
    dt_sub = torch.as_tensor(dt_days / n_substeps, dtype=dtype, device=forcing_tensor.device)
    states = [state]
    histories = {name: [] for name in FLUX_NAMES}
    q_instantaneous = []
    q_routed = []
    balance = []
    snow_balance = []
    snow_history = [snow]
    for i in range(forcing_tensor.shape[0]):
        ppt, pet, temp = forcing_tensor[i]
        day_balance = state.sum() * 0.0
        day_snow_balance = state.sum() * 0.0
        final_flux = None
        for _ in range(n_substeps):
            effective, next_snow, snow_rate = _snow_step(
                ppt, temp, params_tensor, snow, days[i], leap_years[i], dt_sub
            )
            derivative, raw_flux = _derivatives(state, spec, params_tensor, cap, effective, pet, topographic)
            state_next = _project_state(state + dt_sub * derivative, spec, cap)
            flux = _enforce_boundary_water_balance(
                state, state_next, spec, cap, raw_flux, effective, dt_sub
            )
            instantaneous = flux["QSURF"] + flux["OFLOW_1"] + flux["QINTF_1"] + flux["OFLOW_2"] + flux["QBASE_2"]
            external = effective - flux["EVAP_1"] - flux["EVAP_2"] - instantaneous
            day_balance = day_balance + state_next.sum() - state.sum() - external * dt_sub
            day_snow_balance = day_snow_balance + snow_rate * dt_sub - (next_snow - snow)
            state = state_next
            snow = next_snow
            final_flux = flux
        assert final_flux is not None
        for name in FLUX_NAMES:
            histories[name].append(final_flux[name])
        instantaneous = final_flux["QSURF"] + final_flux["OFLOW_1"] + final_flux["QINTF_1"] + final_flux["OFLOW_2"] + final_flux["QBASE_2"]
        routed = future[0] + instantaneous * fractions[0]
        future = torch.cat((future[1:] + instantaneous * fractions[1:], future[-1:] * 0.0))
        q_instantaneous.append(instantaneous)
        q_routed.append(routed)
        states.append(state)
        balance.append(day_balance)
        snow_balance.append(day_snow_balance / dt_days)
        snow_history.append(snow)
    return SimulationResult(
        model_id=spec.model_id,
        q=torch.stack(q_routed),
        q_instantaneous=torch.stack(q_instantaneous),
        states=torch.stack(states),
        state_names=spec.state_names,
        fluxes={name: torch.stack(values) for name, values in histories.items()},
        water_balance_residual=torch.stack(balance),
        snow=torch.stack(snow_history),
        snow_balance_residual=torch.stack(snow_balance),
    )


def simulate(
    model_id: int | str,
    forcing: Tensor | Mapping[str, object],
    params: Mapping[str, object] | Tensor | None = None,
    *,
    initial_state: Tensor | Mapping[str, object] | None = None,
    initial_fraction: float = 0.25,
    dates: Sequence[date | datetime] | Tensor | None = None,
    dt_days: float = 1.0,
    implicit_iterations: int = 16,
    solver: str = "implicit",
    n_substeps: int = 1,
    compile_inner: bool = False,
    compile_inner_backend: str = "inductor",
    compile_inner_fullgraph: bool = True,
    sequential_order: str = "S1",
    compile_step: bool = True,
    compile_step_backend: str = "inductor",
    compile_step_fullgraph: bool = True,
 ) -> SimulationResult:
    """Run a single lumped FUSE structure with tensor-only operations.

    ``forcing`` columns are ``ppt``, ``pet``, and ``temp`` in mm/day, mm/day,
    and degC.  The returned active state matrix has shape ``[time + 1,
    n_active_states]``.  The fixed-step implicit-Euler solve uses
    differentiable Newton iterations and matches the reference numerix
    interface while retaining tensor-only operations.
    """
    if solver == "coupled_rk2":
        if compile_inner:
            raise ValueError("compile_inner is not used with solver='coupled_rk2'")
        return simulate_coupled_rk2(
            model_id,
            forcing,
            params,
            initial_state=initial_state,
            initial_fraction=initial_fraction,
            dates=dates,
            dt_days=dt_days,
            compile_step=compile_step,
            compile_backend=compile_step_backend,
            compile_fullgraph=compile_step_fullgraph,
        )
    if solver == "sequential":
        if compile_inner:
            raise ValueError("compile_inner is not used with solver='sequential'")
        return simulate_sequential(
            model_id,
            forcing,
            params,
            initial_state=initial_state,
            initial_fraction=initial_fraction,
            dates=dates,
            dt_days=dt_days,
            n_substeps=n_substeps,
            order=sequential_order,
            compile_step=compile_step,
            compile_backend=compile_step_backend,
            compile_fullgraph=compile_step_fullgraph,
        )
    if solver == "explicit" and compile_inner:
        raise ValueError("compile_inner is only supported with solver='implicit'")
    if solver == "explicit":
        return simulate_explicit(
            model_id,
            forcing,
            params,
            initial_state=initial_state,
            initial_fraction=initial_fraction,
            dates=dates,
            dt_days=dt_days,
            n_substeps=n_substeps,
        )
    if solver != "implicit":
        raise ValueError("solver must be 'implicit' or 'explicit'")
    if n_substeps != 1:
        raise ValueError("n_substeps is only used with solver='explicit'")
    spec = get_structure(model_id)
    forcing_tensor = _forcing_tensor(forcing)
    if forcing_tensor.shape[0] == 0:
        raise ValueError("forcing must contain at least one time step")
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    dtype = forcing_tensor.dtype if forcing_tensor.is_floating_point() else torch.get_default_dtype()
    forcing_tensor = forcing_tensor.to(dtype=dtype)
    params_tensor = _parameter_values(params, dtype=dtype, device=forcing_tensor.device)
    cap = _capacity(params_tensor)
    if initial_state is None:
        state = _initial_state(spec, params_tensor, cap, initial_fraction)
    elif isinstance(initial_state, Mapping):
        missing = [name for name in spec.state_names if name not in initial_state]
        if missing:
            raise ValueError(f"initial_state missing active coordinates: {missing}")
        state = torch.stack([_to_tensor(initial_state[name], dtype=dtype, device=forcing_tensor.device) for name in spec.state_names])
    else:
        state = initial_state.to(device=forcing_tensor.device, dtype=dtype)
        if state.ndim != 1 or state.numel() != len(spec.state_names):
            raise ValueError(f"initial_state must have {len(spec.state_names)} active coordinates")
    state = _project_state(state, spec, cap)

    days, leap_years = _day_of_year(forcing_tensor.shape[0], dates, dtype=dtype, device=forcing_tensor.device)
    snow = torch.zeros((), dtype=dtype, device=forcing_tensor.device)
    future = torch.zeros(500, dtype=dtype, device=forcing_tensor.device)
    fractions = _routing_fractions(params_tensor["TIMEDELAY"], dtype=dtype, device=forcing_tensor.device)
    topographic = None
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        topographic = _topographic_mean(params_tensor)
    inner = None
    if compile_inner:
        inner = _CompiledInnerKernel(
            spec,
            params_tensor,
            cap,
            topographic,
            backend=compile_inner_backend,
            fullgraph=compile_inner_fullgraph,
        )
    states = [state]
    histories = {name: [] for name in FLUX_NAMES}
    q_instantaneous = []
    q_routed = []
    balance = []
    snow_balance = []
    snow_history = [snow]

    for i in range(forcing_tensor.shape[0]):
        ppt, pet, temp = forcing_tensor[i]
        effective, next_snow, snow_rate = _snow_step(
            ppt, temp, params_tensor, snow, days[i], leap_years[i], dt_days
        )
        state_next = _implicit_step(
            state, spec, params_tensor, cap, effective, pet, dt_days, implicit_iterations, topographic, inner
        )
        derivative, flux = _derivatives(state_next, spec, params_tensor, cap, effective, pet, topographic, inner)
        flux = _enforce_boundary_water_balance(state, state_next, spec, cap, flux, effective, dt_days)
        # State equation residual uses the same end-point flux convention as
        # the implicit Euler solve and therefore exposes numerical drift.
        total_start = state.sum()
        total_end = state_next.sum()
        external = effective - flux["EVAP_1"] - flux["EVAP_2"] - flux["QSURF"] - flux["QINTF_1"] - flux["OFLOW_1"] - flux["OFLOW_2"] - flux["QBASE_2"]
        balance.append(total_end - total_start - external * dt_days)
        snow_balance.append(snow_rate - (next_snow - snow) / dt_days)
        for name in FLUX_NAMES:
            histories[name].append(flux[name])
        instantaneous = flux["QSURF"] + flux["OFLOW_1"] + flux["QINTF_1"] + flux["OFLOW_2"] + flux["QBASE_2"]
        routed = future[0] + instantaneous * fractions[0]
        future = torch.cat((future[1:] + instantaneous * fractions[1:], future[-1:] * 0.0))
        q_instantaneous.append(instantaneous)
        q_routed.append(routed)
        states.append(state_next)
        state = state_next
        snow = next_snow
        snow_history.append(snow)
    return SimulationResult(
        model_id=spec.model_id,
        q=torch.stack(q_routed),
        q_instantaneous=torch.stack(q_instantaneous),
        states=torch.stack(states),
        state_names=spec.state_names,
        fluxes={name: torch.stack(values) for name, values in histories.items()},
        water_balance_residual=torch.stack(balance),
        snow=torch.stack(snow_history),
        snow_balance_residual=torch.stack(snow_balance),
    )

# Sequential process-update candidates share this fixed macro order vocabulary.
SEQUENTIAL_ORDERS = {
    "S1": ("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow"),
    "S2": ("recharge", "surface_runoff", "interflow", "et", "percolation", "baseflow"),
    "S3": ("recharge", "et", "percolation", "baseflow", "interflow", "surface_runoff"),
    "S4": ("recharge", "percolation", "baseflow", "et", "interflow", "surface_runoff"),
}
SEQUENTIAL_DIAGNOSTIC_NAMES = ("water_balance", "snow_balance", "instantaneous", "spill_total", "floor_additions", "projection_triggers")
COUPLED_RK2_DIAGNOSTIC_NAMES = (
    *SEQUENTIAL_DIAGNOSTIC_NAMES,
    "stage1_qperc",
    "stage2_qperc",
    "stage1_qsurf",
    "stage2_qsurf",
    "stage1_evap_total",
    "stage2_evap_total",
 )
_SEQUENTIAL_STATE_SIZE = len(STATE_NAMES) + 1 + 500
_COMPILED_SEQUENTIAL_CACHE: dict[tuple[object, ...], Callable[..., tuple[Tensor, Tensor, Tensor]]] = {}


def _sequential_project_union(state: Tensor, theta: Tensor, context: Tensor) -> Tensor:
    """Apply the shared state floor/capacity rule to the union state."""
    zero = state[0] * 0.0
    fracten = _theta(theta, "FRACTEN")
    maxwatr1 = _theta(theta, "MAXWATR_1")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    max_values = torch.stack((
        _theta(theta, "FRCHZNE") * fracten * maxwatr1,
        (1.0 - _theta(theta, "FRCHZNE")) * fracten * maxwatr1,
        fracten * maxwatr1,
        (1.0 - fracten) * maxwatr1,
        maxwatr1,
        fracten * maxwatr2,
        _theta(theta, "FPRIMQB") * (1.0 - fracten) * maxwatr2,
        (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - fracten) * maxwatr2,
        maxwatr2,
    ))
    active = context[_CTX_STATE_MASK : _CTX_STATE_MASK + len(STATE_NAMES)] > 0.5
    lower_bounded = torch.maximum(state, max_values * 1.0e-8)
    bounded_upper = torch.minimum(lower_bounded[:8], max_values[:8])
    watr2 = torch.where(
        context[_CTX_A2_FIXED] > 0.5,
        torch.minimum(lower_bounded[8:9], max_values[8:9]),
        lower_bounded[8:9],
    )
    return torch.where(
        active,
        torch.cat((bounded_upper, watr2)),
        zero,
    )

def _sequential_delta(state: Tensor, raw: Tensor, process: str, theta: Tensor, context: Tensor) -> Tensor:
    """Map one shared raw-flux evaluation to one immediate process update."""
    zero = state[0] * 0.0
    a1_t2 = context[_CTX_A1_T2] > 0.5
    a1_t1 = context[_CTX_A1_T1] > 0.5
    a1_one = context[_CTX_A1_ONE] > 0.5
    a2_pll = context[_CTX_A2_PLL] > 0.5
    evap1a, evap1b, evap1, rchr, transfer1 = raw[2], raw[3], raw[4], raw[5], raw[6]
    qperc, qintf = raw[7], raw[8]
    oflow1, qsurf = raw[9], raw[10]
    evap2, transfer2 = raw[11], raw[12]
    qbase2a, qbase2b, qbase2 = raw[13], raw[14], raw[15]
    oflow2a, oflow2b, oflow2 = raw[16], raw[17], raw[18]
    if process == "recharge":
        incoming = raw[0]
        return torch.stack((torch.where(a1_t2, incoming, zero), zero, torch.where(a1_t1, incoming, zero), zero, torch.where(a1_one, incoming, zero), zero, zero, zero, zero))
    if process == "et":
        return torch.stack((torch.where(a1_t2, -evap1a, zero), torch.where(a1_t2, -evap1b, zero), torch.where(a1_t1, -evap1, zero), zero, torch.where(a1_one, -evap1, zero), torch.where(a2_pll, -evap2, zero), zero, zero, torch.where(~a2_pll, -evap2, zero)))
    if process == "surface_runoff":
        return torch.stack((torch.where(a1_t2, -qsurf, zero), zero, torch.where(a1_t1, -qsurf, zero), torch.where(~a1_one, -oflow1, zero), torch.where(a1_one, -qsurf - oflow1, zero), zero, zero, zero, zero))
    if process == "percolation":
        frac = _theta(theta, "PERCFRAC")
        return torch.stack((zero, zero, zero, torch.where(~a1_one, -qperc, zero), torch.where(a1_one, -qperc, zero), torch.where(a2_pll, qperc * (1.0 - frac) - transfer2, zero), torch.where(a2_pll, qperc * frac / 2.0 + transfer2 / 2.0, zero), torch.where(a2_pll, qperc * frac / 2.0 + transfer2 / 2.0, zero), torch.where(~a2_pll, qperc, zero)))
    if process == "interflow":
        return torch.stack((torch.where(a1_t2, -rchr, zero), torch.where(a1_t2, rchr - transfer1, zero), torch.where(a1_t1, -transfer1, zero), torch.where(a1_t2 | a1_t1, transfer1 - qintf, zero), torch.where(a1_one, -qintf, zero), zero, zero, zero, zero))
    if process == "baseflow":
        return torch.stack((zero, zero, zero, zero, zero, zero, torch.where(a2_pll, -qbase2a - oflow2a, zero), torch.where(a2_pll, -qbase2b - oflow2b, zero), torch.where(~a2_pll, -qbase2 - oflow2, zero)))
    raise ValueError(f"unknown sequential process: {process}")


_SEQUENTIAL_FLUX_MASKS = {
    "recharge": (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "et": (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    "surface_runoff": (0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "percolation": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    "interflow": (0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "baseflow": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0),
}


def _sequential_snow_step(ppt: Tensor, temp: Tensor, snow: Tensor, jday: Tensor, leap: Tensor, theta: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
    denom = torch.where(leap > 0.5, jday * 0.0 + 366.0, jday * 0.0 + 365.0)
    offset = torch.where(leap > 0.5, jday * 0.0 + 81.0, jday * 0.0 + 80.0)
    melt_factor = (0.5 * torch.sin((jday - offset) * 2.0 * torch.pi / denom) + 0.5) * (_theta(theta, "MFMAX") - _theta(theta, "MFMIN")) + _theta(theta, "MFMIN")
    melt = torch.where((snow > 0.0) & (temp > _theta(theta, "MBASE")), melt_factor * (temp - _theta(theta, "MBASE")), snow * 0.0)
    accumulation = torch.where(temp < _theta(theta, "PXTEMP"), ppt * _theta(theta, "RFERR_MLT"), snow * 0.0)
    next_snow = snow + (accumulation - melt) * dt
    melt = torch.where(next_snow >= 0.0, melt, snow / dt + accumulation)
    next_snow = torch.clamp_min(snow + (accumulation - melt) * dt, 0.0)
    effective = torch.where(temp > _theta(theta, "PXTEMP"), ppt * _theta(theta, "RFERR_MLT") + melt, melt)
    return effective, next_snow

def _sequential_raw_process(
    state: Tensor,
    effective: Tensor,
    pet: Tensor,
    theta: Tensor,
    context: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
    process: str,
 ) -> Tensor:
    """Evaluate only the fluxes needed by one process using shared equations."""
    zero = state[0] * 0.0
    a1_t2 = context[_CTX_A1_T2] > 0.5
    a1_t1 = context[_CTX_A1_T1] > 0.5
    a1_one = context[_CTX_A1_ONE] > 0.5
    a2_pll = context[_CTX_A2_PLL] > 0.5
    a2_frc = context[_CTX_A2_FRC] > 0.5
    a2_pow = context[_CTX_A2_POW] > 0.5
    a2_fixed = context[_CTX_A2_FIXED] > 0.5
    t1a, t1b, t1, free1_state, watr1_state = state[:5]
    t2_state, free2a, free2b, watr2_state = state[5:]
    fracten = _theta(theta, "FRACTEN")
    maxwatr1 = _theta(theta, "MAXWATR_1")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    maxtens1 = fracten * maxwatr1
    maxfree1 = (1.0 - fracten) * maxwatr1
    maxtens1a = _theta(theta, "FRCHZNE") * fracten * maxwatr1
    maxtens1b = (1.0 - _theta(theta, "FRCHZNE")) * fracten * maxwatr1
    maxtens2 = fracten * maxwatr2
    maxfree2a = _theta(theta, "FPRIMQB") * (1.0 - fracten) * maxwatr2
    maxfree2b = (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - fracten) * maxwatr2
    tens1 = torch.where(a1_t2, t1a + t1b, torch.where(a1_t1, t1, torch.minimum(watr1_state, maxtens1)))
    watr1 = torch.where(a1_t2, t1a + t1b + free1_state, torch.where(a1_t1, t1 + free1_state, watr1_state))
    free1 = torch.where(a1_t2 | a1_t1, free1_state, torch.clamp_min(watr1_state - maxtens1, 0.0))
    t2 = torch.where(a2_pll, t2_state, torch.minimum(watr2_state, maxtens2))
    watr2 = torch.where(a2_pll, t2_state + free2a + free2b, watr2_state)
    if process == "recharge":
        return torch.stack((effective, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero))
    if process == "et":
        evap1a = pet * t1a / maxtens1a
        evap1b = (pet - evap1a) * t1b / maxtens1b
        evap1 = torch.where(a1_t2, evap1a + evap1b, pet * tens1 / maxtens1)
        evap2_candidate = (pet - evap1) * t2 / maxtens2
        evap2 = torch.where((a2_pll | a2_fixed) & ~a1_t2, evap2_candidate, zero)
        return torch.stack((zero, zero, evap1a, evap1b, evap1, zero, zero, zero, zero, zero, zero, evap2, zero, zero, zero, zero, zero, zero, zero))
    if process == "surface_runoff" or process == "interflow":
        ratio1 = (watr1 / maxwatr1).clamp(0.0, 1.0)
        ratio_tens1 = (tens1 / maxtens1).clamp(0.0, 1.0)
        arno_area = 1.0 - (1.0 - ratio1).clamp_min(1.0e-12) ** _theta(theta, "AXV_BEXP").clamp_min(1.0e-6)
        prms_area = ratio_tens1 * _theta(theta, "SAREAMAX")
        shape = _theta(theta, "TISHAPE").clamp_min(1.0e-6)
        qb_power = _theta(theta, "QB_POWR").clamp_min(1.0e-6)
        ti_sat = topographic_mean / (watr2 / maxwatr2 + 1.0e-8)
        ti_log = torch.log(ti_sat.clamp_min(1.0e-12) ** qb_power)
        chi = ((_theta(theta, "LOGLAMB") - 3.0) / shape).clamp_min(1.0e-8)
        argument = torch.clamp_min(ti_log - 3.0, 0.0) / chi
        tmdl_area = torch.where(ti_sat > topographic_max, zero, 1.0 - _regularized_gamma_p(shape, argument))
        saturation_area = torch.where(context[_CTX_QS_ARNO] > 0.5, arno_area, torch.where(context[_CTX_QS_PRMS] > 0.5, prms_area, tmdl_area)).clamp(0.0, 1.0)
        qsurf = effective * saturation_area
        if process == "surface_runoff":
            rchr = torch.where(a1_t2, _logismooth(t1a, maxtens1a) * (effective - qsurf), zero)
            transfer = torch.where(a1_t2, _logismooth(t1b, maxtens1b) * rchr, torch.where(a1_t1, _logismooth(tens1, maxtens1) * (effective - qsurf), zero))
            oflow1 = torch.where(a1_one, _logismooth(watr1, maxwatr1) * (effective - qsurf), _logismooth(free1, maxfree1) * transfer)
            return torch.stack((zero, saturation_area, zero, zero, zero, zero, zero, zero, zero, oflow1, qsurf, zero, zero, zero, zero, zero, zero, zero, zero))
        rchr = torch.where(a1_t2, _logismooth(t1a, maxtens1a) * (effective - qsurf), zero)
        transfer = torch.where(a1_t2, _logismooth(t1b, maxtens1b) * rchr, torch.where(a1_t1, _logismooth(tens1, maxtens1) * (effective - qsurf), zero))
        return torch.stack((zero, zero, zero, zero, zero, rchr, transfer, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero))
    qb_power = _theta(theta, "QB_POWR").clamp_min(1.0e-6)
    qbsat_pll = _theta(theta, "QBRATE_2A") * maxfree2a + _theta(theta, "QBRATE_2B") * maxfree2b
    qbsat_frc = _theta(theta, "QB_PRMS") * maxwatr2
    topmdm = maxwatr2 / 1000.0 / qb_power
    qbsat_pow = _theta(theta, "BASERTE") * topmdm / topographic_mean.clamp_min(1.0e-12) ** qb_power
    qbsat = torch.where(a2_pll, qbsat_pll, torch.where(a2_frc, qbsat_frc, torch.where(a2_pow, qbsat_pow, _theta(theta, "BASERTE"))))
    free1_ratio = (free1 / maxfree1).clamp_min(0.0)
    qperc_f2 = _theta(theta, "PERCRTE") * free1_ratio ** _theta(theta, "PERCEXP")
    qperc_w2 = _theta(theta, "PERCRTE") * (watr1 / maxwatr1).clamp_min(0.0) ** _theta(theta, "PERCEXP")
    demand = 1.0 + _theta(theta, "SACPMLT") * (1.0 - watr2 / maxwatr2).clamp_min(0.0) ** _theta(theta, "SACPEXP")
    qperc_lower = qbsat * demand * free1_ratio
    qperc = torch.where(context[_CTX_QP_F2] > 0.5, qperc_f2, torch.where(context[_CTX_QP_W2] > 0.5, qperc_w2, qperc_lower))
    transfer2 = torch.where(a2_pll, _logismooth(t2, maxtens2) * qperc * (1.0 - _theta(theta, "PERCFRAC")), zero)
    if process == "percolation":
        return torch.stack((zero, zero, zero, zero, zero, zero, zero, qperc, zero, zero, zero, zero, transfer2, zero, zero, zero, zero, zero, zero))
    if process == "baseflow":
        qbase2a = _theta(theta, "QBRATE_2A") * free2a
        qbase2b = _theta(theta, "QBRATE_2B") * free2b
        qbase_pll = qbase2a + qbase2b
        qbase_frc = _theta(theta, "QB_PRMS") * watr2
        qbase_pow = qbsat * (watr2 / maxwatr2).clamp_min(0.0) ** _theta(theta, "QB_POWR")
        qbase_fixed = _theta(theta, "BASERTE") * (watr2 / maxwatr2).clamp_min(0.0) ** _theta(theta, "QB_POWR")
        qbase = torch.where(a2_pll, qbase_pll, torch.where(a2_frc, qbase_frc, torch.where(a2_pow, qbase_pow, qbase_fixed)))
        incoming = qperc * _theta(theta, "PERCFRAC") / 2.0 + transfer2 / 2.0
        oflow2a = _logismooth(free2a, maxfree2a) * incoming
        oflow2b = _logismooth(free2b, maxfree2b) * incoming
        oflow2 = torch.where(a2_pll, oflow2a + oflow2b, torch.where(a2_fixed, _logismooth(watr2, maxwatr2) * qperc, zero))
        return torch.stack((zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, qbase2a, qbase2b, qbase, oflow2a, oflow2b, oflow2))
    raise ValueError(f"unknown sequential process: {process}")


def _sequential_substep(
    hydro: Tensor,
    snow: Tensor,
    ppt: Tensor,
    pet: Tensor,
    temp: Tensor,
    jday: Tensor,
    leap: Tensor,
    theta: Tensor,
    context: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
    dt: Tensor,
    order: tuple[str, ...],
 ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    effective, next_snow = _sequential_snow_step(
        ppt, temp, snow, jday, leap, theta, dt
    )
    sub_flux = torch.zeros((len(FLUX_NAMES),), dtype=hydro.dtype, device=hydro.device)
    spill_upper_sum = hydro[0] * 0.0
    spill_lower_sum = hydro[0] * 0.0
    floor_sum = hydro[0] * 0.0
    projection_trigger_sum = hydro[0] * 0.0
    for process in order:
        raw = _sequential_raw_process(
            hydro, effective, pet, theta, context, topographic_mean, topographic_max, process
        )
        delta = _sequential_delta(hydro, raw, process, theta, context)
        floor_values = _sequential_project_union(hydro * 0.0, theta, context)
        sink = (-dt * delta).clamp_min(1.0e-30)
        scale = torch.where(
            delta < 0.0,
            ((hydro - floor_values) / sink).clamp(0.0, 1.0),
            torch.ones_like(delta),
        ).amin()
        delta = delta * scale
        proposal = hydro + dt * delta
        projected = _sequential_project_union(proposal, theta, context)
        spill = (proposal - projected).clamp_min(0.0)
        floor = (projected - proposal).clamp_min(0.0)
        spill_upper_sum = spill_upper_sum + spill[:5].sum()
        spill_lower_sum = spill_lower_sum + spill[5:].sum()
        floor_sum = floor_sum + floor.sum()
        projection_trigger_sum = projection_trigger_sum + (spill + floor > 0.0).to(hydro.dtype).sum()
        hydro = projected
        sub_flux = sub_flux + raw * raw.new_tensor(_SEQUENTIAL_FLUX_MASKS[process][:len(FLUX_NAMES)]) * scale
    return (
        hydro,
        next_snow,
        sub_flux,
        effective,
        spill_upper_sum,
        spill_lower_sum,
        floor_sum,
        projection_trigger_sum,
    )

def _make_sequential_step(order: tuple[str, ...], n_substeps: int) -> Callable[..., tuple[Tensor, Tensor, Tensor]]:
    def step(packed: Tensor, forcing: Tensor, theta: Tensor, context: Tensor, topographic_mean: Tensor, topographic_max: Tensor, fractions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hydro = packed[: len(STATE_NAMES)].clone()
        snow = packed[len(STATE_NAMES)].clone()
        future = packed[len(STATE_NAMES) + 1 :].clone()
        ppt, pet, temp = forcing[0], forcing[1], forcing[2]
        jday, leap, dt_days = forcing[3], forcing[4], forcing[5]
        dt = dt_days / float(n_substeps)
        start_hydro, start_snow = hydro, snow
        flux_sum = torch.zeros((len(FLUX_NAMES),), dtype=packed.dtype, device=packed.device)
        effective_sum = packed[0] * 0.0
        spill_upper_sum = packed[0] * 0.0
        spill_lower_sum = packed[0] * 0.0
        floor_sum = packed[0] * 0.0
        projection_trigger_sum = packed[0] * 0.0
        for _ in range(n_substeps):
            hydro, snow, sub_flux, effective, upper, lower, floors, triggers = _sequential_substep(
                hydro, snow, ppt, pet, temp, jday, leap, theta, context, topographic_mean, topographic_max, dt, order
            )
            flux_sum = flux_sum + sub_flux
            effective_sum = effective_sum + effective
            spill_upper_sum = spill_upper_sum + upper
            spill_lower_sum = spill_lower_sum + lower
            floor_sum = floor_sum + floors
            projection_trigger_sum = projection_trigger_sum + triggers
        spill_sum = spill_upper_sum + spill_lower_sum
        spill_flux = torch.zeros_like(flux_sum)
        spill_flux = spill_flux + torch.nn.functional.pad(spill_upper_sum.reshape(1), (9, len(FLUX_NAMES) - 10))
        spill_flux = spill_flux + torch.nn.functional.pad(spill_lower_sum.reshape(1), (18, len(FLUX_NAMES) - 19))
        flux = (flux_sum + spill_flux / dt) / float(n_substeps)
        instantaneous = flux[10] + flux[9] + flux[8] + flux[18] + flux[15]
        routed = future[0] + instantaneous * fractions[0]
        future_next = torch.cat((future[1:] + instantaneous * fractions[1:], future[-1:] * 0.0))
        water_balance = hydro.sum() - start_hydro.sum() - (flux[0] - flux[4] - flux[11] - instantaneous) * dt_days
        snow_precipitation = torch.where(temp < _theta(theta, "PXTEMP"), ppt * _theta(theta, "RFERR_MLT"), torch.where(temp > _theta(theta, "PXTEMP"), ppt * _theta(theta, "RFERR_MLT"), ppt * 0.0))
        snow_balance = snow_precipitation * dt_days - effective_sum * dt - (snow - start_snow)
        diagnostics = torch.cat((flux, torch.stack((water_balance, snow_balance / dt_days, instantaneous, spill_sum, floor_sum, projection_trigger_sum))))
        return torch.cat((hydro, snow.reshape(1), future_next)), routed, diagnostics
    return step



class _CompiledSequentialAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, compiled, eager, packed, forcing, theta, context, topographic_mean, topographic_max, fractions):
        ctx.eager = eager
        ctx.save_for_backward(packed, forcing, theta, context, topographic_mean, topographic_max, fractions)
        with torch.no_grad():
            return compiled(packed.detach(), forcing.detach(), theta.detach(), context.detach(), topographic_mean.detach(), topographic_max.detach(), fractions.detach())

    @staticmethod
    def backward(ctx, grad_packed, grad_q, grad_diagnostics):
        originals = ctx.saved_tensors
        local = tuple(value.detach().requires_grad_(True) for value in originals)
        indices = [i for i, value in enumerate(originals) if value.requires_grad]
        gradients = [None] * len(originals)
        if indices:
            inputs = tuple(local[i] for i in indices)
            create_graph = torch.is_grad_enabled()
            with torch.enable_grad():
                proxy = tuple(
                    local_value + (original - original.detach())
                    for local_value, original in zip(local, originals)
                )
                outputs = ctx.eager(*proxy)
                output_grads = (grad_packed, grad_q, grad_diagnostics)
                active = [
                    (output, grad)
                    for output, grad in zip(outputs, output_grads)
                    if output.requires_grad
                ]
                if active:
                    input_gradients = torch.autograd.grad(
                        tuple(output for output, _ in active),
                        inputs,
                        grad_outputs=tuple(grad for _, grad in active),
                        allow_unused=True,
                        create_graph=create_graph,
                    )
                else:
                    input_gradients = (None,) * len(inputs)
            for index, gradient in zip(indices, input_gradients):
                gradients[index] = gradient
        return (None, None, *gradients)


# Runtime-generated structure-specialized steps are provided by dfuse.runtime.


def _sequential_union_state(state: Tensor, spec) -> Tensor:
    zero = state[0] * 0.0
    return torch.stack([state[spec.state_names.index(name)] if name in spec.state_names else zero for name in STATE_NAMES])


def simulate_sequential(model_id: int | str, forcing: Tensor | Mapping[str, object], params: Mapping[str, object] | Tensor | None = None, *, initial_state: Tensor | Mapping[str, object] | None = None, initial_fraction: float = 0.25, dates: Sequence[date | datetime] | Tensor | None = None, dt_days: float = 1.0, n_substeps: int = 1, order: str = "S1", compile_step: bool = True, compile_backend: str = "inductor", compile_fullgraph: bool = True, execution_mode: str = "sequential") -> SimulationResult:
    if order not in SEQUENTIAL_ORDERS:
        raise ValueError(f"order must be one of {tuple(SEQUENTIAL_ORDERS)}")
    if execution_mode not in ("sequential", "storage_block"):
        raise ValueError("execution_mode must be sequential or storage_block")
    if execution_mode == "storage_block" and order != "S4":
        raise ValueError("storage_block execution requires order='S4'")
    if not isinstance(n_substeps, int) or isinstance(n_substeps, bool) or n_substeps < 1:
        raise ValueError("n_substeps must be a positive integer")
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    spec = get_structure(model_id)
    forcing_tensor = _forcing_tensor(forcing)
    if forcing_tensor.shape[0] == 0:
        raise ValueError("forcing must contain at least one time step")
    dtype = forcing_tensor.dtype if forcing_tensor.is_floating_point() else torch.get_default_dtype()
    forcing_tensor = forcing_tensor.to(dtype=dtype)
    params_tensor = _parameter_values(params, dtype=dtype, device=forcing_tensor.device)
    cap = _capacity(params_tensor)
    if initial_state is None:
        active_state = _initial_state(spec, params_tensor, cap, initial_fraction)
    elif isinstance(initial_state, Mapping):
        missing = [name for name in spec.state_names if name not in initial_state]
        if missing:
            raise ValueError(f"initial_state missing active coordinates: {missing}")
        active_state = torch.stack([_to_tensor(initial_state[name], dtype=dtype, device=forcing_tensor.device) for name in spec.state_names])
    else:
        active_state = initial_state.to(device=forcing_tensor.device, dtype=dtype)
        if active_state.ndim != 1 or active_state.numel() != len(spec.state_names):
            raise ValueError(f"initial_state must have {len(spec.state_names)} active coordinates")
    context = _structure_context(spec, dtype=dtype, device=forcing_tensor.device)
    theta = _parameter_vector(params_tensor)
    state = _sequential_project_union(_sequential_union_state(active_state, spec), theta, context)
    days, leap_years = _day_of_year(forcing_tensor.shape[0], dates, dtype=dtype, device=forcing_tensor.device)
    topo = _topographic_mean(params_tensor) if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2" else (theta[0] * 0.0, theta[0] * 0.0)
    fractions = _routing_fractions(params_tensor["TIMEDELAY"], dtype=dtype, device=forcing_tensor.device)
    from .runtime import get_compiled_step, get_generated_step
    order_tuple = SEQUENTIAL_ORDERS[order]
    _signature, generated_step = get_generated_step(spec, order=order_tuple, n_substeps=n_substeps, execution_mode=execution_mode)
    if compile_step:
        _, step_callable = get_compiled_step(
            spec,
            order=order_tuple,
            n_substeps=n_substeps,
            device=forcing_tensor.device,
            dtype=dtype,
            input_shapes=(
                (len(STATE_NAMES) + 1 + 500,),
                (6,),
                (len(PARAMETER_NAMES),),
                (),
                (),
                (500,),
            ),
            backend=compile_backend,
            fullgraph=compile_fullgraph,
            execution_mode=execution_mode,
        )
    else:
        step_callable = generated_step
    packed = torch.cat((state, torch.zeros((501,), dtype=dtype, device=state.device)))
    states = [torch.stack([packed[_STATE_INDEX[name]] for name in spec.state_names])]
    histories = {name: [] for name in FLUX_NAMES}
    q_values, q_instantaneous, balances, snow_balances, snow_history = [], [], [], [], [packed[len(STATE_NAMES)]]
    sequential_diagnostics = {name: [] for name in SEQUENTIAL_DIAGNOSTIC_NAMES}
    step_forcing = torch.cat((forcing_tensor, days.unsqueeze(1), leap_years.to(dtype).unsqueeze(1), torch.full((forcing_tensor.shape[0], 1), dt_days, dtype=dtype, device=forcing_tensor.device)), dim=1)
    for index in range(forcing_tensor.shape[0]):
        packed, routed, diagnostics = step_callable(
            packed, step_forcing[index], theta, topo[0], topo[1], fractions
        )
        flux = diagnostics[: len(FLUX_NAMES)]
        for flux_index, name in enumerate(FLUX_NAMES):
            histories[name].append(flux[flux_index])
        q_values.append(routed)
        q_instantaneous.append(diagnostics[len(FLUX_NAMES) + 2])
        balances.append(diagnostics[len(FLUX_NAMES)])
        snow_balances.append(diagnostics[len(FLUX_NAMES) + 1])
        for diagnostic_index, name in enumerate(SEQUENTIAL_DIAGNOSTIC_NAMES):
            sequential_diagnostics[name].append(diagnostics[len(FLUX_NAMES) + diagnostic_index])
        states.append(torch.stack([packed[_STATE_INDEX[name]] for name in spec.state_names]))
        snow_history.append(packed[len(STATE_NAMES)])
    return SimulationResult(
        model_id=spec.model_id,
        q=torch.stack(q_values),
        q_instantaneous=torch.stack(q_instantaneous),
        states=torch.stack(states),
        state_names=spec.state_names,
        fluxes={name: torch.stack(values) for name, values in histories.items()},
        water_balance_residual=torch.stack(balances),
        snow=torch.stack(snow_history),
        snow_balance_residual=torch.stack(snow_balances),
        sequential_diagnostics={name: torch.stack(values) for name, values in sequential_diagnostics.items()},
    )


def simulate_storage_block(model_id: int | str, forcing: Tensor | Mapping[str, object], params: Mapping[str, object] | Tensor | None = None, *, initial_state: Tensor | Mapping[str, object] | None = None, initial_fraction: float = 0.25, dates: Sequence[date | datetime] | Tensor | None = None, dt_days: float = 1.0, n_substeps: int = 1, compile_step: bool = True, compile_backend: str = "inductor", compile_fullgraph: bool = True) -> SimulationResult:
    """Run the bounded storage-block sequential explicit prototype.

    This diagnostic-only wrapper freezes the S4 macro order and changes only
    how competing outgoing fluxes are evaluated and committed.  It never
    invokes an implicit, Newton, Jacobian, or IFT path.
    """
    return simulate_sequential(
        model_id, forcing, params, initial_state=initial_state, initial_fraction=initial_fraction,
        dates=dates, dt_days=dt_days, n_substeps=n_substeps, order="S4",
        compile_step=compile_step, compile_backend=compile_backend,
        compile_fullgraph=compile_fullgraph, execution_mode="storage_block",
    )

def simulate_coupled_rk2(
    model_id: int | str,
    forcing: Tensor | Mapping[str, object],
    params: Mapping[str, object] | Tensor | None = None,
    *,
    initial_state: Tensor | Mapping[str, object] | None = None,
    initial_fraction: float = 0.25,
    dates: Sequence[date | datetime] | Tensor | None = None,
    dt_days: float = 1.0,
    compile_step: bool = True,
    compile_backend: str = "inductor",
    compile_fullgraph: bool = True,
 ) -> SimulationResult:
    """Run fixed daily RK2/Heun from a structure-specialized coupled RHS."""
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    spec = get_structure(model_id)
    forcing_tensor = _forcing_tensor(forcing)
    if forcing_tensor.shape[0] == 0:
        raise ValueError("forcing must contain at least one time step")
    dtype = forcing_tensor.dtype if forcing_tensor.is_floating_point() else torch.get_default_dtype()
    forcing_tensor = forcing_tensor.to(dtype=dtype)
    params_tensor = _parameter_values(params, dtype=dtype, device=forcing_tensor.device)
    cap = _capacity(params_tensor)
    if initial_state is None:
        active_state = _initial_state(spec, params_tensor, cap, initial_fraction)
    elif isinstance(initial_state, Mapping):
        missing = [name for name in spec.state_names if name not in initial_state]
        if missing:
            raise ValueError(f"initial_state missing active coordinates: {missing}")
        active_state = torch.stack([_to_tensor(initial_state[name], dtype=dtype, device=forcing_tensor.device) for name in spec.state_names])
    else:
        active_state = initial_state.to(device=forcing_tensor.device, dtype=dtype)
        if active_state.ndim != 1 or active_state.numel() != len(spec.state_names):
            raise ValueError(f"initial_state must have {len(spec.state_names)} active coordinates")
    context = _structure_context(spec, dtype=dtype, device=forcing_tensor.device)
    theta = _parameter_vector(params_tensor)
    state = _sequential_project_union(_sequential_union_state(active_state, spec), theta, context)
    days, leap_years = _day_of_year(forcing_tensor.shape[0], dates, dtype=dtype, device=forcing_tensor.device)
    topo = _topographic_mean(params_tensor) if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2" else (theta[0] * 0.0, theta[0] * 0.0)
    fractions = _routing_fractions(params_tensor["TIMEDELAY"], dtype=dtype, device=forcing_tensor.device)
    from .runtime import get_compiled_step, get_generated_step
    coupled_marker = ("coupled_rhs",)
    _signature, generated_step = get_generated_step(spec, order=coupled_marker, n_substeps=1, execution_mode="coupled_rk2")
    if compile_step:
        _, step_callable = get_compiled_step(
            spec,
            order=coupled_marker,
            n_substeps=1,
            device=forcing_tensor.device,
            dtype=dtype,
            input_shapes=(
                (len(STATE_NAMES) + 1 + 500,),
                (6,),
                (len(PARAMETER_NAMES),),
                (),
                (),
                (500,),
            ),
            backend=compile_backend,
            fullgraph=compile_fullgraph,
            execution_mode="coupled_rk2",
        )
    else:
        step_callable = generated_step
    packed = torch.cat((state, torch.zeros((501,), dtype=dtype, device=state.device)))
    states = [torch.stack([packed[_STATE_INDEX[name]] for name in spec.state_names])]
    histories = {name: [] for name in FLUX_NAMES}
    q_values, q_instantaneous, balances, snow_balances, snow_history = [], [], [], [], [packed[len(STATE_NAMES)]]
    coupled_diagnostics = {name: [] for name in COUPLED_RK2_DIAGNOSTIC_NAMES}
    step_forcing = torch.cat((
        forcing_tensor,
        days.unsqueeze(1),
        leap_years.to(dtype).unsqueeze(1),
        torch.full((forcing_tensor.shape[0], 1), dt_days, dtype=dtype, device=forcing_tensor.device),
    ), dim=1)
    for index in range(forcing_tensor.shape[0]):
        packed, routed, diagnostics = step_callable(
            packed, step_forcing[index], theta, topo[0], topo[1], fractions
        )
        flux = diagnostics[: len(FLUX_NAMES)]
        for flux_index, name in enumerate(FLUX_NAMES):
            histories[name].append(flux[flux_index])
        q_values.append(routed)
        q_instantaneous.append(diagnostics[len(FLUX_NAMES) + 2])
        balances.append(diagnostics[len(FLUX_NAMES)])
        snow_balances.append(diagnostics[len(FLUX_NAMES) + 1])
        for diagnostic_index, name in enumerate(COUPLED_RK2_DIAGNOSTIC_NAMES):
            coupled_diagnostics[name].append(diagnostics[len(FLUX_NAMES) + diagnostic_index])
        states.append(torch.stack([packed[_STATE_INDEX[name]] for name in spec.state_names]))
        snow_history.append(packed[len(STATE_NAMES)])
    return SimulationResult(
        model_id=spec.model_id,
        q=torch.stack(q_values),
        q_instantaneous=torch.stack(q_instantaneous),
        states=torch.stack(states),
        state_names=spec.state_names,
        fluxes={name: torch.stack(values) for name, values in histories.items()},
        water_balance_residual=torch.stack(balances),
        snow=torch.stack(snow_history),
        snow_balance_residual=torch.stack(snow_balances),
        sequential_diagnostics={name: torch.stack(values) for name, values in coupled_diagnostics.items()},
    )
