"""Runtime-generated, structure-specialized sequential Euler steps.

The public sequential simulator keeps the packed union state and parameter
interfaces used by :mod:`dfuse.kernel`, but this module moves every discrete
choice in front of ``torch.compile``.  A generated step has tensor-only inputs;
its structure and process order are literals in an independently compiled
Python code object.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from .spec import DECISION_ORDER, FLUX_NAMES, PARAMETER_NAMES, STATE_NAMES, StructureSpec
from .kernel import (
    _SEQUENTIAL_FLUX_MASKS,
    _logismooth,
    _regularized_gamma_p,
    _theta,
)


_PROCESS_NAMES = ("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float):
        return float(value)
    return value


@dataclass(frozen=True)
class GraphSignature:
    """Canonical identity of a generated step's discrete computation path.

    Basin/model IDs, values, forcing, epochs, and seeds are intentionally not
    part of this identity.  ``n_substeps`` is included because the explicit
    Euler substep is unrolled into the generated graph.
    """

    decisions: tuple[tuple[str, str], ...]
    sequential_order: tuple[str, ...]
    n_substeps: int
    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    topology: Any
    execution_mode: str = "sequential"
    @classmethod
    def from_spec(
        cls, spec: StructureSpec, *, order: str | tuple[str, ...], n_substeps: int, execution_mode: str = "sequential"
    ) -> "GraphSignature":
        from .kernel import SEQUENTIAL_ORDERS
        if execution_mode == "coupled_rk2":
            order_tuple = ("coupled_rhs",)
        else:
            order_tuple = SEQUENTIAL_ORDERS[order] if isinstance(order, str) else tuple(order)
        return cls.from_structure(spec, sequential_order=order_tuple, n_substeps=n_substeps, execution_mode=execution_mode)

    @classmethod
    def from_structure(
        cls, spec: StructureSpec, *, sequential_order: tuple[str, ...], n_substeps: int, execution_mode: str = "sequential"
    ) -> "GraphSignature":
        if execution_mode == "coupled_rk2":
            if tuple(sequential_order) not in (("coupled_rhs",), tuple(_PROCESS_NAMES)):
                raise ValueError("coupled_rk2 execution requires the coupled RHS marker")
            order_tuple = ("coupled_rhs",)
        else:
            if tuple(sorted(sequential_order)) != tuple(sorted(_PROCESS_NAMES)):
                raise ValueError(f"sequential_order must contain each process exactly once: {_PROCESS_NAMES}")
            if len(set(sequential_order)) != len(sequential_order):
                raise ValueError("sequential_order must not repeat a process")
            order_tuple = tuple(sequential_order)
        if not isinstance(n_substeps, int) or isinstance(n_substeps, bool) or n_substeps < 1:
            raise ValueError("n_substeps must be a positive integer")
        if execution_mode not in ("sequential", "storage_block", "coupled_rk2"):
            raise ValueError("execution_mode must be sequential, storage_block, or coupled_rk2")
        if execution_mode == "storage_block" and tuple(order_tuple) != ("recharge", "percolation", "baseflow", "et", "interflow", "surface_runoff"):
            raise ValueError("storage_block execution requires the frozen S4 process order")
        if execution_mode == "coupled_rk2" and n_substeps != 1:
            raise ValueError("coupled_rk2 uses one fixed daily step; substeps are not supported")
        return cls(
            decisions=tuple((name, str(spec.decisions[name])) for name in DECISION_ORDER),
            sequential_order=tuple(order_tuple),
            n_substeps=n_substeps,
            state_names=tuple(spec.state_names),
            parameter_names=tuple(spec.parameter_names),
            topology=_freeze(spec.topology),
            execution_mode=execution_mode,
        )

    def decision(self, name: str) -> str:
        return dict(self.decisions)[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": {name: value for name, value in self.decisions},
            "sequential_order": list(self.sequential_order),
            "n_substeps": self.n_substeps,
            "state_names": list(self.state_names),
            "parameter_names": list(self.parameter_names),
            "topology": self.topology,
            "execution_mode": self.execution_mode,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CompiledStepKey:
    signature: GraphSignature
    device: str
    dtype: str
    input_shapes: tuple[tuple[int, ...], ...]
    backend: str
    fullgraph: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_signature": self.signature.to_dict(),
            "device": self.device,
            "dtype": self.dtype,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "backend": self.backend,
            "fullgraph": self.fullgraph,
        }


def _saturation_area(
    state: Tensor,
    theta: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
    *,
    arch1: str,
    arch2: str,
    qsurf: str,
) -> Tensor:
    if arch1 == "tension2_1":
        tens1 = state[0] + state[1]
        watr1 = tens1 + state[3]
    elif arch1 == "tension1_1":
        tens1 = state[2]
        watr1 = tens1 + state[3]
    elif arch1 == "onestate_1":
        watr1 = state[4]
        tens1 = torch.minimum(watr1, _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_1"))
    else:
        raise ValueError(f"unsupported ARCH1: {arch1}")

    maxwatr1 = _theta(theta, "MAXWATR_1")
    if qsurf == "arno_x_vic":
        ratio = (watr1 / maxwatr1).clamp(0.0, 1.0)
        area = torch.where(ratio >= 1.0, torch.ones_like(ratio), 1.0 - (1.0 - ratio) ** _theta(theta, "AXV_BEXP").clamp_min(1.0e-6))
    elif qsurf == "prms_varnt":
        ratio_tens = (tens1 / (_theta(theta, "FRACTEN") * maxwatr1)).clamp(0.0, 1.0)
        area = ratio_tens * _theta(theta, "SAREAMAX")
    elif qsurf == "tmdl_param":
        maxwatr2 = _theta(theta, "MAXWATR_2")
        qb_power = _theta(theta, "QB_POWR").clamp_min(1.0e-6)
        watr2 = state[5] + state[6] + state[7] if arch2 == "tens2pll_2" else state[8]
        ti_sat = topographic_mean / (watr2 / maxwatr2 + 1.0e-8)
        ti_log = torch.log(ti_sat.clamp_min(1.0e-12) ** qb_power)
        shape = _theta(theta, "TISHAPE").clamp_min(1.0e-6)
        chi = ((_theta(theta, "LOGLAMB") - 3.0) / shape).clamp_min(1.0e-8)
        argument = torch.clamp_min(ti_log - 3.0, 0.0) / chi
        area = torch.where(
            ti_sat > topographic_max,
            state[0] * 0.0,
            1.0 - _regularized_gamma_p(shape, argument),
        )
    else:
        raise ValueError(f"unsupported QSURF: {qsurf}")
    return area.clamp(0.0, 1.0)


def _qperc(
    state: Tensor,
    theta: Tensor,
    *,
    arch1: str,
    arch2: str,
    qperc: str,
    topographic_mean: Tensor,
) -> Tensor:
    fracten = _theta(theta, "FRACTEN")
    maxwatr1 = _theta(theta, "MAXWATR_1")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    watr2 = state[5] + state[6] + state[7] if arch2 == "tens2pll_2" else state[8]
    if arch1 == "tension2_1":
        free1 = state[3]
        watr1 = state[0] + state[1] + state[3]
    elif arch1 == "tension1_1":
        free1 = state[3]
        watr1 = state[2] + state[3]
    elif arch1 == "onestate_1":
        watr1 = state[4]
        free1 = torch.clamp_min(watr1 - fracten * maxwatr1, 0.0)
    else:
        raise ValueError(f"unsupported ARCH1: {arch1}")

    if qperc == "perc_f2sat":
        return _theta(theta, "PERCRTE") * (
            free1 / ((1.0 - fracten) * maxwatr1)
        ).clamp_min(0.0) ** _theta(theta, "PERCEXP")
    if qperc == "perc_w2sat":
        return _theta(theta, "PERCRTE") * (watr1 / maxwatr1).clamp_min(0.0) ** _theta(theta, "PERCEXP")

    if arch2 == "tens2pll_2":
        qbsat = _theta(theta, "QBRATE_2A") * _theta(theta, "FPRIMQB") * (1.0 - fracten) * maxwatr2
        qbsat = qbsat + _theta(theta, "QBRATE_2B") * (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - fracten) * maxwatr2
    elif arch2 == "unlimfrc_2":
        qbsat = _theta(theta, "QB_PRMS") * maxwatr2
    elif arch2 == "unlimpow_2":
        qb_power = _theta(theta, "QB_POWR").clamp_min(1.0e-6)
        qbsat = _theta(theta, "BASERTE") * (maxwatr2 / 1000.0 / qb_power) / topographic_mean.clamp_min(1.0e-12) ** qb_power
    elif arch2 == "fixedsiz_2":
        qbsat = _theta(theta, "BASERTE")
    else:
        raise ValueError(f"unsupported ARCH2: {arch2}")
    demand = 1.0 + _theta(theta, "SACPMLT") * (
        1.0 - watr2 / maxwatr2
    ).clamp_min(0.0) ** _theta(theta, "SACPEXP")
    return qbsat * demand * (free1 / ((1.0 - fracten) * maxwatr1)).clamp_min(0.0)


def _qbase(
    state: Tensor,
    theta: Tensor,
    *,
    arch2: str,
    topographic_mean: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    fracten = _theta(theta, "FRACTEN")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    if arch2 == "tens2pll_2":
        qbase2a = _theta(theta, "QBRATE_2A") * state[6]
        qbase2b = _theta(theta, "QBRATE_2B") * state[7]
        qbase = qbase2a + qbase2b
        return qbase2a, qbase2b, qbase
    if arch2 == "unlimfrc_2":
        qbase = _theta(theta, "QB_PRMS") * state[8]
    elif arch2 == "unlimpow_2":
        qb_power = _theta(theta, "QB_POWR").clamp_min(1.0e-6)
        qbsat = _theta(theta, "BASERTE") * (maxwatr2 / 1000.0 / qb_power) / topographic_mean.clamp_min(1.0e-12) ** qb_power
        qbase = qbsat * (state[8] / maxwatr2).clamp_min(0.0) ** qb_power
    elif arch2 == "fixedsiz_2":
        qbase = _theta(theta, "BASERTE") * (state[8] / maxwatr2).clamp_min(0.0) ** _theta(theta, "QB_POWR")
    else:
        raise ValueError(f"unsupported ARCH2: {arch2}")
    return state[6] * 0.0, state[7] * 0.0, qbase


def _fixed_raw_process(
    state: Tensor,
    effective: Tensor,
    pet: Tensor,
    theta: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
    *,
    choices: tuple[str, str, str, str],
    process: str,
) -> Tensor:
    """Evaluate exactly one process for one fixed structure.

    The Python ``if`` statements are reached with literals emitted by the
    generated function.  Dynamo therefore traces only this process and this
    architecture; no tensor flags or candidate-process masks enter the graph.
    """
    arch1, arch2, qsurf, qperc_name = choices
    zero = state[0] * 0.0
    if arch1 == "tension2_1":
        t1a, t1b, t1 = state[0], state[1], state[0] + state[1]
        free1, watr1 = state[3], state[0] + state[1] + state[3]
    elif arch1 == "tension1_1":
        t1a, t1b, t1 = zero, zero, state[2]
        free1, watr1 = state[3], state[2] + state[3]
    elif arch1 == "onestate_1":
        t1a, t1b, t1 = zero, zero, torch.minimum(state[4], _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_1"))
        watr1, free1 = state[4], torch.clamp_min(state[4] - _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_1"), 0.0)
    else:
        raise ValueError(f"unsupported ARCH1: {arch1}")
    if arch2 == "tens2pll_2":
        t2, watr2 = state[5], state[5] + state[6] + state[7]
    else:
        t2, watr2 = torch.minimum(state[8], _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_2")), state[8]

    if process == "recharge":
        return torch.stack((effective, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero))

    if process == "et":
        if arch1 == "tension2_1":
            evap1a = pet * t1a / (_theta(theta, "FRCHZNE") * _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_1"))
            evap1b = (pet - evap1a) * t1b / ((1.0 - _theta(theta, "FRCHZNE")) * _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_1"))
            evap1 = evap1a + evap1b
        else:
            evap1a = zero
            evap1b = zero
            evap1 = pet * t1 / (_theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_1"))
        if arch2 in ("tens2pll_2", "fixedsiz_2") and arch1 != "tension2_1":
            evap2 = (pet - evap1) * t2 / (_theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_2"))
        else:
            evap2 = zero
        return torch.stack((zero, zero, evap1a, evap1b, evap1, zero, zero, zero, zero, zero, zero, evap2, zero, zero, zero, zero, zero, zero, zero))

    if process in ("surface_runoff", "interflow"):
        area = _saturation_area(
            state, theta, topographic_mean, topographic_max, arch1=arch1, arch2=arch2, qsurf=qsurf
        )
        qsurf_value = effective * area
        if arch1 == "tension2_1":
            recharge_excess = _logismooth(t1a, _theta(theta, "FRCHZNE") * _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_1")) * (effective - qsurf_value)
            transfer = _logismooth(t1b, (1.0 - _theta(theta, "FRCHZNE")) * _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_1")) * recharge_excess
        elif arch1 == "tension1_1":
            recharge_excess = zero
            transfer = _logismooth(t1, _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_1")) * (effective - qsurf_value)
        else:
            recharge_excess = zero
            transfer = zero
        if process == "surface_runoff":
            if arch1 == "onestate_1":
                oflow1 = _logismooth(watr1, _theta(theta, "MAXWATR_1")) * (effective - qsurf_value)
            else:
                oflow1 = _logismooth(free1, (1.0 - _theta(theta, "FRACTEN")) * _theta(theta, "MAXWATR_1")) * transfer
            return torch.stack((zero, area, zero, zero, zero, zero, zero, zero, zero, oflow1, qsurf_value, zero, zero, zero, zero, zero, zero, zero, zero))
        return torch.stack((zero, zero, zero, zero, zero, recharge_excess, transfer, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero))

    if process == "percolation":
        qperc_value = _qperc(
            state, theta, arch1=arch1, arch2=arch2, qperc=qperc_name,
            topographic_mean=topographic_mean,
        )
        if arch2 == "tens2pll_2":
            transfer2 = _logismooth(t2, _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_2")) * qperc_value * (1.0 - _theta(theta, "PERCFRAC"))
        else:
            transfer2 = zero
        return torch.stack((zero, zero, zero, zero, zero, zero, zero, qperc_value, zero, zero, zero, zero, transfer2, zero, zero, zero, zero, zero, zero))

    if process == "baseflow":
        qperc_value = _qperc(
            state, theta, arch1=arch1, arch2=arch2, qperc=qperc_name,
            topographic_mean=topographic_mean,
        )
        qbase2a, qbase2b, qbase = _qbase(
            state, theta, arch2=arch2, topographic_mean=topographic_mean
        )
        if arch2 == "tens2pll_2":
            transfer2 = _logismooth(t2, _theta(theta, "FRACTEN") * _theta(theta, "MAXWATR_2")) * qperc_value * (1.0 - _theta(theta, "PERCFRAC"))
            incoming = qperc_value * _theta(theta, "PERCFRAC") / 2.0 + transfer2 / 2.0
            oflow2a = _logismooth(state[6], _theta(theta, "FPRIMQB") * (1.0 - _theta(theta, "FRACTEN")) * _theta(theta, "MAXWATR_2")) * incoming
            oflow2b = _logismooth(state[7], (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - _theta(theta, "FRACTEN")) * _theta(theta, "MAXWATR_2")) * incoming
            oflow2 = oflow2a + oflow2b
        elif arch2 == "fixedsiz_2":
            oflow2a, oflow2b = zero, zero
            oflow2 = _logismooth(state[8], _theta(theta, "MAXWATR_2")) * qperc_value
        else:
            oflow2a, oflow2b, oflow2 = zero, zero, zero
        return torch.stack((zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, qbase2a, qbase2b, qbase, oflow2a, oflow2b, oflow2))

    if process == "interflow":
        raise AssertionError("interflow is handled above")
    raise ValueError(f"unknown sequential process: {process}")



def _fixed_delta_with_theta(
    state: Tensor,
    raw: Tensor,
    theta: Tensor,
    *,
    choices: tuple[str, str, str, str],
    process: str,
 ) -> Tensor:
    """Immediate explicit-Euler process update for a fixed architecture."""
    arch1, arch2, _, _ = choices
    zero = state[0] * 0.0
    evap1a, evap1b, evap1, recharge_excess, transfer1 = raw[2], raw[3], raw[4], raw[5], raw[6]
    qperc_value, qintf = raw[7], raw[8]
    oflow1, qsurf_value = raw[9], raw[10]
    evap2, transfer2 = raw[11], raw[12]
    qbase2a, qbase2b, qbase = raw[13], raw[14], raw[15]
    oflow2a, oflow2b, oflow2 = raw[16], raw[17], raw[18]
    if process == "recharge":
        if arch1 == "tension2_1":
            return torch.stack((raw[0], zero, zero, zero, zero, zero, zero, zero, zero))
        if arch1 == "tension1_1":
            return torch.stack((zero, zero, raw[0], zero, zero, zero, zero, zero, zero))
        return torch.stack((zero, zero, zero, zero, raw[0], zero, zero, zero, zero))
    if process == "et":
        if arch1 == "tension2_1":
            upper = torch.stack((-evap1a, -evap1b, zero, zero, zero))
        elif arch1 == "tension1_1":
            upper = torch.stack((zero, zero, -evap1, zero, zero))
        else:
            upper = torch.stack((zero, zero, zero, zero, -evap1))
        if arch2 == "tens2pll_2" and arch1 != "tension2_1":
            lower = torch.stack((-evap2, zero, zero, zero))
        elif arch2 == "fixedsiz_2" and arch1 != "tension2_1":
            lower = torch.stack((zero, zero, zero, -evap2))
        else:
            lower = torch.stack((zero, zero, zero, zero))
        return torch.cat((upper, lower))
    if process == "surface_runoff":
        if arch1 == "tension2_1":
            return torch.stack((-qsurf_value, zero, zero, -oflow1, zero, zero, zero, zero, zero))
        if arch1 == "tension1_1":
            return torch.stack((zero, zero, -qsurf_value, -oflow1, zero, zero, zero, zero, zero))
        return torch.stack((zero, zero, zero, zero, -qsurf_value - oflow1, zero, zero, zero, zero))
    if process == "percolation":
        if arch1 == "onestate_1":
            upper = torch.stack((zero, zero, zero, zero, -qperc_value))
        else:
            upper = torch.stack((zero, zero, zero, -qperc_value, zero))
        if arch2 == "tens2pll_2":
            frac = _theta(theta, "PERCFRAC")
            lower = torch.stack((qperc_value * (1.0 - frac) - transfer2, qperc_value * frac / 2.0 + transfer2 / 2.0, qperc_value * frac / 2.0 + transfer2 / 2.0, zero))
        else:
            lower = torch.stack((zero, zero, zero, qperc_value))
        return torch.cat((upper, lower))
    if process == "interflow":
        if arch1 == "tension2_1":
            return torch.stack((-recharge_excess, recharge_excess - transfer1, zero, transfer1 - qintf, zero, zero, zero, zero, zero))
        if arch1 == "tension1_1":
            return torch.stack((zero, zero, -transfer1, transfer1 - qintf, zero, zero, zero, zero, zero))
        return torch.stack((zero, zero, zero, zero, -qintf, zero, zero, zero, zero))
    if process == "baseflow":
        if arch2 == "tens2pll_2":
            return torch.stack((zero, zero, zero, zero, zero, zero, -qbase2a - oflow2a, -qbase2b - oflow2b, zero))
        return torch.stack((zero, zero, zero, zero, zero, zero, zero, zero, -qbase - oflow2))
    raise ValueError(f"unknown sequential process: {process}")

def _fixed_project(state: Tensor, theta: Tensor, *, choices: tuple[str, str, str, str]) -> Tensor:
    arch1, arch2, _, _ = choices
    zero = state[0] * 0.0
    fracten = _theta(theta, "FRACTEN")
    maxwatr1 = _theta(theta, "MAXWATR_1")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    max_t1 = fracten * maxwatr1
    max_f1 = (1.0 - fracten) * maxwatr1
    max_t1a = _theta(theta, "FRCHZNE") * max_t1
    max_t1b = (1.0 - _theta(theta, "FRCHZNE")) * max_t1
    max_t2 = fracten * maxwatr2
    max_f2a = _theta(theta, "FPRIMQB") * (1.0 - fracten) * maxwatr2
    max_f2b = (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - fracten) * maxwatr2

    def bounded(index: int, maximum: Tensor, cap: bool = True) -> Tensor:
        value = state[index].clamp_min(maximum * 1.0e-8)
        return value.clamp_max(maximum) if cap else value

    values = [zero] * len(STATE_NAMES)
    if arch1 == "tension2_1":
        values[0], values[1], values[3] = bounded(0, max_t1a), bounded(1, max_t1b), bounded(3, max_f1)
    elif arch1 == "tension1_1":
        values[2], values[3] = bounded(2, max_t1), bounded(3, max_f1)
    elif arch1 == "onestate_1":
        values[4] = bounded(4, maxwatr1)
    else:
        raise ValueError(f"unsupported ARCH1: {arch1}")
    if arch2 == "tens2pll_2":
        values[5], values[6], values[7] = bounded(5, max_t2), bounded(6, max_f2a), bounded(7, max_f2b)
    elif arch2 == "fixedsiz_2":
        values[8] = bounded(8, maxwatr2)
    elif arch2 in ("unlimfrc_2", "unlimpow_2"):
        values[8] = bounded(8, maxwatr2, cap=False)
    else:
        raise ValueError(f"unsupported ARCH2: {arch2}")
    return torch.stack(values)


_FIXSTATE_FRAC_MIN = 1.0e-9


def _state_set_component(state: Tensor, index: int, value: Tensor) -> Tensor:
    mask = state.new_tensor(tuple(1.0 if position == index else 0.0 for position in range(len(STATE_NAMES))))
    return state + mask * (value - state[index])


def _flux_add_component(flux: Tensor, index: int, value: Tensor) -> Tensor:
    mask = flux.new_tensor(tuple(1.0 if position == index else 0.0 for position in range(len(FLUX_NAMES))))
    return flux + mask * value


def _flux_set_component(flux: Tensor, index: int, value: Tensor) -> Tensor:
    return _flux_add_component(flux, index, value - flux[index])


def _flux_proportional_loss(flux: Tensor, indices: tuple[int, ...], error_loss: Tensor) -> Tensor:
    total = sum((flux[index] for index in indices), flux[0] * 0.0)
    safe_total = torch.where(total == total * 0.0, total * 0.0 + 1.0, total)
    weights = flux.new_tensor(tuple(1.0 if position in indices else 0.0 for position in range(len(FLUX_NAMES))))
    return flux + weights * (flux / safe_total) * error_loss


def _fix_lower_proportional(
    state: Tensor,
    flux: Tensor,
    index: int,
    lower: Tensor,
    dt: Tensor,
    flux_indices: tuple[int, ...],
    error_lower: Tensor | None = None,
 ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    old = state[index]
    violated = old < lower
    error_reference = lower if error_lower is None else error_lower
    error_loss = torch.where(violated, (old - error_reference) / dt, old * 0.0)
    corrected_flux = _flux_proportional_loss(flux, flux_indices, error_loss)
    corrected_state = torch.where(violated, lower, old)
    return _state_set_component(state, index, corrected_state), torch.where(violated, corrected_flux, flux), error_loss, violated


def _fix_lower_direct(
    state: Tensor,
    flux: Tensor,
    index: int,
    lower: Tensor,
    dt: Tensor,
    flux_index: int,
 ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    old = state[index]
    violated = old < lower
    error_loss = torch.where(violated, (old - lower) / dt, old * 0.0)
    corrected_flux = _flux_add_component(flux, flux_index, error_loss)
    corrected_state = torch.where(violated, lower, old)
    return _state_set_component(state, index, corrected_state), torch.where(violated, corrected_flux, flux), error_loss, violated


def _fix_upper_direct(
    state: Tensor,
    flux: Tensor,
    index: int,
    upper: Tensor,
    dt: Tensor,
    flux_index: int,
    trigger: Tensor | None = None,
    error_state: Tensor | None = None,
 ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    old = state[index]
    active = old if error_state is None else error_state
    violated = (active > upper) if trigger is None else trigger
    error_loss = torch.where(violated, (old - upper) / dt, old * 0.0)
    corrected_flux = _flux_add_component(flux, flux_index, error_loss)
    corrected_state = torch.where(violated, upper, old)
    return _state_set_component(state, index, corrected_state), torch.where(violated, corrected_flux, flux), error_loss, violated


def _runtime_recompute_lower_from_qperc(
    state: Tensor,
    flux: Tensor,
    previous: Tensor,
    theta: Tensor,
    dt: Tensor,
    *,
    choices: tuple[str, str, str, str],
 ) -> tuple[Tensor, Tensor]:
    _, arch2, _, _ = choices
    zero = state[0] * 0.0
    fracten = _theta(theta, "FRACTEN")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    max_t2 = fracten * maxwatr2
    qperc = flux[7]
    if arch2 == "tens2pll_2":
        percfrac = _theta(theta, "PERCFRAC")
        max_f2a = _theta(theta, "FPRIMQB") * (1.0 - fracten) * maxwatr2
        max_f2b = (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - fracten) * maxwatr2
        transfer2 = torch.maximum(zero, qperc * (1.0 - percfrac) - (max_t2 - previous[5]) / dt)
        flux = _flux_set_component(flux, 12, transfer2)
        incoming = qperc * percfrac / 2.0 + transfer2 / 2.0
        oflow2a = torch.maximum(zero, incoming - (max_f2a - previous[6]) / dt)
        oflow2b = torch.maximum(zero, incoming - (max_f2b - previous[7]) / dt)
        flux = _flux_set_component(flux, 16, oflow2a)
        flux = _flux_set_component(flux, 17, oflow2b)
        flux = _flux_set_component(flux, 18, oflow2a + oflow2b)
        state = _state_set_component(
            state, 5, previous[5] + (qperc * (1.0 - percfrac) - flux[11] - transfer2) * dt
        )
        state = _state_set_component(
            state, 6, previous[6] + (incoming - flux[13] - oflow2a) * dt
        )
        state = _state_set_component(
            state, 7, previous[7] + (incoming - flux[14] - oflow2b) * dt
        )
        return state, flux
    if arch2 == "fixedsiz_2":
        maxwatr2 = _theta(theta, "MAXWATR_2")
        oflow2 = torch.maximum(zero, qperc - (maxwatr2 - previous[8]) / dt)
        flux = _flux_set_component(flux, 18, oflow2)
    state = _state_set_component(
        state, 8, previous[8] + (qperc - flux[11] - flux[15] - flux[18]) * dt
    )
    return state, flux


def _runtime_recompute_parallel_from_transfer(
    state: Tensor,
    flux: Tensor,
    previous: Tensor,
    theta: Tensor,
    dt: Tensor,
 ) -> tuple[Tensor, Tensor]:
    zero = state[0] * 0.0
    fracten = _theta(theta, "FRACTEN")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    max_t2 = fracten * maxwatr2
    max_f2a = _theta(theta, "FPRIMQB") * (1.0 - fracten) * maxwatr2
    max_f2b = (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - fracten) * maxwatr2
    qperc = flux[7]
    transfer2 = flux[12]
    incoming = qperc * _theta(theta, "PERCFRAC") / 2.0 + transfer2 / 2.0
    oflow2a = torch.maximum(zero, incoming - (max_f2a - previous[6]) / dt)
    oflow2b = torch.maximum(zero, incoming - (max_f2b - previous[7]) / dt)
    flux = _flux_set_component(flux, 16, oflow2a)
    flux = _flux_set_component(flux, 17, oflow2b)
    flux = _flux_set_component(flux, 18, oflow2a + oflow2b)
    state = _state_set_component(
        state, 5, previous[5] + (qperc * (1.0 - _theta(theta, "PERCFRAC")) - flux[11] - transfer2) * dt
    )
    state = _state_set_component(
        state, 6, previous[6] + (incoming - flux[13] - oflow2a) * dt
    )
    state = _state_set_component(
        state, 7, previous[7] + (incoming - flux[14] - oflow2b) * dt
    )
    return state, flux


def _runtime_fix_states(
    previous: Tensor,
    candidate: Tensor,
    raw_flux: Tensor,
    theta: Tensor,
    dt: Tensor,
    *,
    choices: tuple[str, str, str, str],
 ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Tensor translation of the pinned FIX_STATES one-pass mutation.

    ``previous`` is Fortran BSTATE and ``candidate`` is ESTATE.  Structural
    choices are literals in the generated caller; all value-dependent decisions
    are tensor operations.  The returned vectors are state correction, ERR_*
    values, lower-violation flags, and upper-violation flags respectively.
    """
    arch1, arch2, _, _ = choices
    state = candidate
    flux = raw_flux
    errors = state * 0.0
    lower_flags = state * 0.0
    upper_flags = state * 0.0
    xmin = state[0] * 0.0 + _FIXSTATE_FRAC_MIN
    fracten = _theta(theta, "FRACTEN")
    maxwatr1 = _theta(theta, "MAXWATR_1")
    maxwatr2 = _theta(theta, "MAXWATR_2")
    max_t1 = fracten * maxwatr1
    max_f1 = (1.0 - fracten) * maxwatr1
    max_t1a = _theta(theta, "FRCHZNE") * max_t1
    max_t1b = (1.0 - _theta(theta, "FRCHZNE")) * max_t1
    max_t2 = fracten * maxwatr2
    max_f2a = _theta(theta, "FPRIMQB") * (1.0 - fracten) * maxwatr2
    max_f2b = (1.0 - _theta(theta, "FPRIMQB")) * (1.0 - fracten) * maxwatr2
    zero = state[0] * 0.0
    free1_index = 4

    if arch1 == "tension2_1":
        state, flux, lower_error, lower = _fix_lower_proportional(state, flux, 0, xmin * max_t1a, dt, (10, 2))
        state, flux, upper_error, upper = _fix_upper_direct(state, flux, 0, max_t1a, dt, 5)
        errors = _state_set_component(errors, 0, lower_error + upper_error)
        lower_flags = _state_set_component(lower_flags, 0, lower.to(dtype=state.dtype))
        upper_flags = _state_set_component(upper_flags, 0, upper.to(dtype=state.dtype))
        corrected_tens1b = previous[1] + (flux[5] - flux[3] - flux[6]) * dt
        state = _state_set_component(state, 1, torch.where(upper, corrected_tens1b, state[1]))
        state, flux, lower_error, lower = _fix_lower_direct(state, flux, 1, xmin * max_t1b, dt, 3)
        state, flux, upper_error, upper = _fix_upper_direct(state, flux, 1, max_t1b, dt, 6)
        errors = _state_set_component(errors, 1, lower_error + upper_error)
        lower_flags = _state_set_component(lower_flags, 1, lower.to(dtype=state.dtype))
        upper_flags = _state_set_component(upper_flags, 1, upper.to(dtype=state.dtype))
        corrected_free1 = previous[3] + (flux[6] - flux[7] - flux[8] - flux[9]) * dt
        state = _state_set_component(state, 3, torch.where(upper, corrected_free1, state[3]))
        free1_index = 3
    elif arch1 == "tension1_1":
        state, flux, lower_error, lower = _fix_lower_proportional(state, flux, 2, xmin * max_t1, dt, (10, 4))
        state, flux, upper_error, upper = _fix_upper_direct(state, flux, 2, max_t1, dt, 6)
        errors = _state_set_component(errors, 2, lower_error + upper_error)
        lower_flags = _state_set_component(lower_flags, 2, lower.to(dtype=state.dtype))
        upper_flags = _state_set_component(upper_flags, 2, upper.to(dtype=state.dtype))
        corrected_free1 = previous[3] + (flux[6] - flux[7] - flux[8] - flux[9]) * dt
        state = _state_set_component(state, 3, torch.where(upper, corrected_free1, state[3]))
        free1_index = 3
    elif arch1 == "onestate_1":
        state, flux, lower_error, lower = _fix_lower_proportional(state, flux, 4, xmin * maxwatr1, dt, (10, 4, 7, 8))
        state, flux, upper_error, upper = _fix_upper_direct(state, flux, 4, maxwatr1, dt, 9)
        errors = _state_set_component(errors, 4, lower_error + upper_error)
        lower_flags = _state_set_component(lower_flags, 4, lower.to(dtype=state.dtype))
        upper_flags = _state_set_component(upper_flags, 4, upper.to(dtype=state.dtype))
        free1_lower = lower
    else:
        raise ValueError(f"unsupported ARCH1: {arch1}")
    if arch1 != "onestate_1":
        state, flux, lower_error, free1_lower = _fix_lower_proportional(state, flux, free1_index, xmin * max_f1, dt, (7, 8))
        state, flux, upper_error, upper = _fix_upper_direct(state, flux, free1_index, max_f1, dt, 9)
        errors = _state_set_component(errors, free1_index, lower_error + upper_error)
        lower_flags = _state_set_component(lower_flags, free1_index, free1_lower.to(dtype=state.dtype))
        upper_flags = _state_set_component(upper_flags, free1_index, upper.to(dtype=state.dtype))
    state_from_qperc, flux_from_qperc = _runtime_recompute_lower_from_qperc(state, flux, previous, theta, dt, choices=choices)
    state = torch.where(free1_lower, state_from_qperc, state)
    flux = torch.where(free1_lower, flux_from_qperc, flux)
    if arch2 == "tens2pll_2":
        state, flux, lower_error, lower = _fix_lower_direct(state, flux, 5, xmin * max_t2, dt, 11)
        state, flux, upper_error, upper = _fix_upper_direct(state, flux, 5, max_t2, dt, 12)
        errors = _state_set_component(errors, 5, lower_error + upper_error)
        lower_flags = _state_set_component(lower_flags, 5, lower.to(dtype=state.dtype))
        upper_flags = _state_set_component(upper_flags, 5, upper.to(dtype=state.dtype))
        state_from_transfer, flux_from_transfer = _runtime_recompute_parallel_from_transfer(state, flux, previous, theta, dt)
        state = torch.where(upper, state_from_transfer, state)
        flux = torch.where(upper, flux_from_transfer, flux)
        state, flux, lower_error, lower = _fix_lower_direct(state, flux, 6, xmin * max_f2a, dt, 13)
        state, flux, upper_error, upper = _fix_upper_direct(state, flux, 6, max_f2a, dt, 16)
        errors = _state_set_component(errors, 6, lower_error + upper_error)
        lower_flags = _state_set_component(lower_flags, 6, lower.to(dtype=state.dtype))
        upper_flags = _state_set_component(upper_flags, 6, upper.to(dtype=state.dtype))
        state, flux, lower_error, lower = _fix_lower_direct(state, flux, 7, xmin * max_f2b, dt, 14)
        state, flux, upper_error, upper = _fix_upper_direct(state, flux, 7, max_f2b, dt, 17)
        errors = _state_set_component(errors, 7, lower_error + upper_error)
        lower_flags = _state_set_component(lower_flags, 7, lower.to(dtype=state.dtype))
        upper_flags = _state_set_component(upper_flags, 7, upper.to(dtype=state.dtype))
    else:
        # Intentional Torch correction: the WATR_2 error reference must match
        # the lower bound used for the corrected state (MAXWATR_2). The pinned
        # Fortran branch uses MAXWATR_1 here and is not mass-conservative.
        state, flux, lower_error, lower = _fix_lower_proportional(state, flux, 8, xmin * maxwatr2, dt, (11, 15))
        errors = _state_set_component(errors, 8, lower_error)
        lower_flags = _state_set_component(lower_flags, 8, lower.to(dtype=state.dtype))
        # Single-state XTRY_2_STR leaves FREE_2B undefined; the pinned diagnostic
        # wrapper observes its zero initialization, so the source upper test is false.
        source_missing_free2b = zero
        source_trigger = source_missing_free2b > max_f2b
        state, flux, upper_error, upper = _fix_upper_direct(state, flux, 8, maxwatr2, dt, 18, trigger=source_trigger)
        errors = _state_set_component(errors, 8, lower_error + upper_error)
        upper_flags = _state_set_component(upper_flags, 8, upper.to(dtype=state.dtype))
    # Component lower-floor corrections alter EVAP_1A/EVAP_1B. Keep the
    # aggregate EVAP_1 accepted flux synchronized with those components so the
    # state transition and water-balance bookkeeping use the same flux vector.
    if arch1 == "tension2_1":
        flux = _flux_set_component(flux, 4, flux[2] + flux[3])
    if arch2 == "tens2pll_2":
        flux = _flux_set_component(flux, 15, flux[13] + flux[14])
        flux = _flux_set_component(flux, 18, flux[16] + flux[17])
    correction = state - candidate
    return state, flux, correction, errors, lower_flags, upper_flags


def _runtime_snow_step(ppt: Tensor, temp: Tensor, snow: Tensor, jday: Tensor, leap: Tensor, theta: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
    denom = torch.where(leap > 0.5, jday * 0.0 + 366.0, jday * 0.0 + 365.0)
    offset = torch.where(leap > 0.5, jday * 0.0 + 81.0, jday * 0.0 + 80.0)
    melt_factor = (0.5 * torch.sin((jday - offset) * 2.0 * torch.pi / denom) + 0.5) * (_theta(theta, "MFMAX") - _theta(theta, "MFMIN")) + _theta(theta, "MFMIN")
    melt = torch.where((snow > 0.0) & (temp > _theta(theta, "MBASE")), melt_factor * (temp - _theta(theta, "MBASE")), snow * 0.0)
    accumulation = torch.where(temp < _theta(theta, "PXTEMP"), ppt * _theta(theta, "RFERR_MLT"), snow * 0.0)
    trial = snow + (accumulation - melt) * dt
    melt = torch.where(trial >= 0.0, melt, snow / dt + accumulation)
    next_snow = torch.clamp_min(snow + (accumulation - melt) * dt, 0.0)
    effective = torch.where(temp > _theta(theta, "PXTEMP"), ppt * _theta(theta, "RFERR_MLT") + melt, melt)
    return effective, next_snow

def _runtime_snow_balance_input(ppt: Tensor, temp: Tensor, theta: Tensor) -> Tensor:
    """Return precipitation that participates in the source snow partition."""
    precipitation = ppt * _theta(theta, "RFERR_MLT")
    zero = precipitation * 0.0
    return torch.where(
        temp < _theta(theta, "PXTEMP"),
        precipitation,
        torch.where(temp > _theta(theta, "PXTEMP"), precipitation, zero),
    )


def _runtime_coupled_rhs_impl(
    hydro: Tensor,
    effective: Tensor,
    pet: Tensor,
    theta: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
    *,
    choices: tuple[str, str, str, str],
 ) -> tuple[Tensor, Tensor]:
    """Evaluate all active flux laws from one common stage state.

    The calls are deliberately explicit and have no process-order loop.  Each
    generated caller supplies literal structural choices, so Dynamo traces one
    structure-specific coupled RHS rather than a union of masked structures.
    """
    raw_recharge = _fixed_raw_process(
        hydro, effective, pet, theta, topographic_mean, topographic_max,
        choices=choices, process="recharge",
    )
    raw_et = _fixed_raw_process(
        hydro, effective, pet, theta, topographic_mean, topographic_max,
        choices=choices, process="et",
    )
    raw_surface = _fixed_raw_process(
        hydro, effective, pet, theta, topographic_mean, topographic_max,
        choices=choices, process="surface_runoff",
    )
    raw_percolation = _fixed_raw_process(
        hydro, effective, pet, theta, topographic_mean, topographic_max,
        choices=choices, process="percolation",
    )
    raw_interflow = _fixed_raw_process(
        hydro, effective, pet, theta, topographic_mean, topographic_max,
        choices=choices, process="interflow",
    )
    raw_baseflow = _fixed_raw_process(
        hydro, effective, pet, theta, topographic_mean, topographic_max,
        choices=choices, process="baseflow",
    )
    raw_flux = raw_recharge * raw_recharge.new_tensor(_SEQUENTIAL_FLUX_MASKS["recharge"][:len(FLUX_NAMES)])
    raw_flux = raw_flux + raw_et * raw_et.new_tensor(_SEQUENTIAL_FLUX_MASKS["et"][:len(FLUX_NAMES)])
    raw_flux = raw_flux + raw_surface * raw_surface.new_tensor(_SEQUENTIAL_FLUX_MASKS["surface_runoff"][:len(FLUX_NAMES)])
    raw_flux = raw_flux + raw_percolation * raw_percolation.new_tensor(_SEQUENTIAL_FLUX_MASKS["percolation"][:len(FLUX_NAMES)])
    raw_flux = raw_flux + raw_interflow * raw_interflow.new_tensor(_SEQUENTIAL_FLUX_MASKS["interflow"][:len(FLUX_NAMES)])
    raw_flux = raw_flux + raw_baseflow * raw_baseflow.new_tensor(_SEQUENTIAL_FLUX_MASKS["baseflow"][:len(FLUX_NAMES)])
    derivative = _fixed_delta_with_theta(hydro, raw_recharge, theta, choices=choices, process="recharge")
    derivative = derivative + _fixed_delta_with_theta(hydro, raw_et, theta, choices=choices, process="et")
    derivative = derivative + _fixed_delta_with_theta(hydro, raw_surface, theta, choices=choices, process="surface_runoff")
    derivative = derivative + _fixed_delta_with_theta(hydro, raw_percolation, theta, choices=choices, process="percolation")
    derivative = derivative + _fixed_delta_with_theta(hydro, raw_interflow, theta, choices=choices, process="interflow")
    derivative = derivative + _fixed_delta_with_theta(hydro, raw_baseflow, theta, choices=choices, process="baseflow")
    return derivative, raw_flux
def _runtime_coupled_rk2_step_impl(
    packed: Tensor,
    forcing: Tensor,
    theta: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
    fractions: Tensor,
    *,
    choices: tuple[str, str, str, str],
    order: tuple[str, ...],
    n_substeps: int,
 ) -> tuple[Tensor, Tensor, Tensor]:
    """Take one fixed daily Heun step from a coupled structure-specific RHS.

    The predictor is kept in the legal structure domain before stage two, as
    required by the upstream explicit-Heun safeguard.  Boundary correction is
    deliberately audited against FIX_STATES rather than treated as a new flux
    law; the one-step gate decides whether this prototype is usable.
    """
    del order
    if n_substeps != 1:
        raise ValueError("coupled_rk2 accepts exactly one fixed daily step")
    hydro = packed[: len(STATE_NAMES)]
    snow = packed[len(STATE_NAMES)]
    future = packed[len(STATE_NAMES) + 1 :]
    ppt, pet, temp = forcing[0], forcing[1], forcing[2]
    jday, leap, dt_days = forcing[3], forcing[4], forcing[5]
    effective, next_snow = _runtime_snow_step(ppt, temp, snow, jday, leap, theta, dt_days)
    start_hydro = hydro
    k1, flux1 = _runtime_coupled_rhs_impl(
        hydro, effective, pet, theta, topographic_mean, topographic_max, choices=choices,
    )
    predictor_raw = hydro + dt_days * k1
    predictor, predictor_flux_fixed, predictor_correction, predictor_errors, predictor_lower, predictor_upper = _runtime_fix_states(
        hydro, predictor_raw, flux1, theta, dt_days, choices=choices
    )
    k2, flux2 = _runtime_coupled_rhs_impl(
        predictor, effective, pet, theta, topographic_mean, topographic_max, choices=choices,
    )
    proposal = hydro + 0.5 * dt_days * (k1 + k2)
    flux_avg_raw = 0.5 * (flux1 + flux2)
    projected, flux, final_correction, final_errors, final_lower, final_upper = _runtime_fix_states(
        hydro, proposal, flux_avg_raw, theta, dt_days, choices=choices
    )
    instantaneous = flux[10] + flux[9] + flux[8] + flux[18] + flux[15]
    routed = future[0] + instantaneous * fractions[0]
    future_next = torch.cat((future[1:] + instantaneous * fractions[1:], future[-1:] * 0.0))
    water_balance = projected.sum() - start_hydro.sum() - (flux[0] - flux[4] - flux[11] - instantaneous) * dt_days
    snow_balance = _runtime_snow_balance_input(ppt, temp, theta) * dt_days - effective * dt_days - (next_snow - snow)
    diagnostics = torch.cat((
        flux,
        torch.stack((
            water_balance,
            snow_balance / dt_days,
            instantaneous,
            (flux[18] - flux_avg_raw[18]).clamp_min(0.0) * dt_days,
            final_correction.clamp_min(0.0).sum(),
            (final_lower + final_upper).sum(),
            flux1[7], flux2[7], flux1[10], flux2[10], flux1[4] + flux1[11], flux2[4] + flux2[11],
        )),
    ))
    return torch.cat((projected, next_snow.reshape(1), future_next)), routed, diagnostics


def _runtime_step_impl(
    packed: Tensor,
    forcing: Tensor,
    theta: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
    fractions: Tensor,
    *,
    choices: tuple[str, str, str, str],
    order: tuple[str, ...],
    n_substeps: int,
) -> tuple[Tensor, Tensor, Tensor]:
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
        effective, next_snow = _runtime_snow_step(ppt, temp, snow, jday, leap, theta, dt)
        sub_flux = torch.zeros((len(FLUX_NAMES),), dtype=hydro.dtype, device=hydro.device)
        for process in order:
            raw = _fixed_raw_process(
                hydro, effective, pet, theta, topographic_mean, topographic_max,
                choices=choices, process=process,
            )
            delta = _fixed_delta_with_theta(hydro, raw, theta, choices=choices, process=process)
            floor_values = _fixed_project(hydro * 0.0, theta, choices=choices)
            sink = (-dt * delta).clamp_min(1.0e-30)
            scale = torch.where(
                delta < 0.0,
                ((hydro - floor_values) / sink).clamp(0.0, 1.0),
                torch.ones_like(delta),
            ).amin()
            delta = delta * scale
            proposal = hydro + dt * delta
            projected = _fixed_project(proposal, theta, choices=choices)
            spill = (proposal - projected).clamp_min(0.0)
            floor = (projected - proposal).clamp_min(0.0)
            spill_upper_sum = spill_upper_sum + spill[:5].sum()
            spill_lower_sum = spill_lower_sum + spill[5:].sum()
            floor_sum = floor_sum + floor.sum()
            projection_trigger_sum = projection_trigger_sum + (spill + floor > 0.0).to(hydro.dtype).sum()
            hydro = projected
            sub_flux = sub_flux + raw * raw.new_tensor(_SEQUENTIAL_FLUX_MASKS[process][:len(FLUX_NAMES)]) * scale
        snow = next_snow
        flux_sum = flux_sum + sub_flux
        effective_sum = effective_sum + effective
    spill_sum = spill_upper_sum + spill_lower_sum
    spill_flux = torch.zeros_like(flux_sum)
    spill_flux = spill_flux + torch.nn.functional.pad(spill_upper_sum.reshape(1), (9, len(FLUX_NAMES) - 10))
    spill_flux = spill_flux + torch.nn.functional.pad(spill_lower_sum.reshape(1), (18, len(FLUX_NAMES) - 19))
    flux = (flux_sum + spill_flux / dt) / float(n_substeps)
    instantaneous = flux[10] + flux[9] + flux[8] + flux[18] + flux[15]
    routed = future[0] + instantaneous * fractions[0]
    future_next = torch.cat((future[1:] + instantaneous * fractions[1:], future[-1:] * 0.0))
    water_balance = hydro.sum() - start_hydro.sum() - (flux[0] - flux[4] - flux[11] - instantaneous) * dt_days
    snow_balance = _runtime_snow_balance_input(ppt, temp, theta) * dt_days - effective_sum * dt - (snow - start_snow)
    diagnostics = torch.cat((flux, torch.stack((water_balance, snow_balance / dt_days, instantaneous, spill_sum, floor_sum, projection_trigger_sum))))
    return torch.cat((hydro, snow.reshape(1), future_next)), routed, diagnostics

def _block_scale(state: Tensor, delta: Tensor, floor_values: Tensor, dt: Tensor) -> Tensor:
    sink = (-dt * delta).clamp_min(1.0e-30)
    candidate = torch.where(delta < 0.0, ((state - floor_values) / sink).clamp(0.0, 1.0), torch.ones_like(delta))
    if state.shape[0] == 9:
        return torch.minimum(torch.minimum(torch.minimum(candidate[0], candidate[1]), torch.minimum(candidate[2], candidate[3])), torch.minimum(torch.minimum(candidate[4], candidate[5]), torch.minimum(torch.minimum(candidate[6], candidate[7]), candidate[8])))
    if state.shape[0] == 5:
        return torch.minimum(torch.minimum(candidate[0], candidate[1]), torch.minimum(torch.minimum(candidate[2], candidate[3]), candidate[4]))
    if state.shape[0] == 4:
        return torch.minimum(torch.minimum(candidate[0], candidate[1]), torch.minimum(candidate[2], candidate[3]))
    raise ValueError("storage-block scale received an unsupported state width")


def _runtime_block_step_impl(
    packed: Tensor,
    forcing: Tensor,
    theta: Tensor,
    topographic_mean: Tensor,
    topographic_max: Tensor,
    fractions: Tensor,
    *,
    choices: tuple[str, str, str, str],
    order: tuple[str, ...],
    n_substeps: int,
 ) -> tuple[Tensor, Tensor, Tensor]:
    """Storage-block explicit prototype using the existing raw process equations.

    Recharge is applied first.  All outgoing fluxes for the upper and lower
    stores are then evaluated from one post-recharge snapshot.  The upper
    storage block is committed before the lower block, but lower-block fluxes
    retain the pre-update lower-store snapshot.  This is explicit operator
    splitting, not an implicit solve.
    """
    del order
    hydro = packed[: len(STATE_NAMES)].clone()
    snow = packed[len(STATE_NAMES)].clone()
    future = packed[len(STATE_NAMES) + 1 :].clone()
    ppt, pet, temp = forcing[0], forcing[1], forcing[2]
    jday, leap, dt_days = forcing[3], forcing[4], forcing[5]
    dt = dt_days / float(n_substeps)
    start_hydro, start_snow = hydro, snow
    flux_sum = torch.zeros((len(FLUX_NAMES),), dtype=packed.dtype, device=packed.device)
    effective_sum = packed[0] * 0.0
    spill_sum = packed[0] * 0.0
    spill_upper_sum = packed[0] * 0.0
    spill_lower_sum = packed[0] * 0.0
    floor_sum = packed[0] * 0.0
    projection_trigger_sum = packed[0] * 0.0
    for _ in range(n_substeps):
        effective, next_snow = _runtime_snow_step(ppt, temp, snow, jday, leap, theta, dt)
        zero = hydro[0] * 0.0
        recharge_raw = _fixed_raw_process(hydro, effective, pet, theta, topographic_mean, topographic_max, choices=choices, process="recharge")
        recharge_delta = _fixed_delta_with_theta(hydro, recharge_raw, theta, choices=choices, process="recharge")
        recharge_floor = _fixed_project(hydro * 0.0, theta, choices=choices)
        recharge_scale = _block_scale(hydro, recharge_delta, recharge_floor, dt)
        recharge_proposal = hydro + dt * recharge_delta * recharge_scale
        recharge_projected = _fixed_project(recharge_proposal, theta, choices=choices)
        recharge_spill = (recharge_proposal - recharge_projected).clamp_min(0.0)
        recharge_floor_add = (recharge_projected - recharge_proposal).clamp_min(0.0)
        hydro = recharge_projected
        block_state = hydro
        block_floor = _fixed_project(block_state * 0.0, theta, choices=choices)
        et_raw = _fixed_raw_process(block_state, effective, pet, theta, topographic_mean, topographic_max, choices=choices, process="et")
        surface_raw = _fixed_raw_process(block_state, effective, pet, theta, topographic_mean, topographic_max, choices=choices, process="surface_runoff")
        percolation_raw = _fixed_raw_process(block_state, effective, pet, theta, topographic_mean, topographic_max, choices=choices, process="percolation")
        interflow_raw = _fixed_raw_process(block_state, effective, pet, theta, topographic_mean, topographic_max, choices=choices, process="interflow")
        baseflow_raw = _fixed_raw_process(block_state, effective, pet, theta, topographic_mean, topographic_max, choices=choices, process="baseflow")
        et_delta = _fixed_delta_with_theta(block_state, et_raw, theta, choices=choices, process="et")
        surface_delta = _fixed_delta_with_theta(block_state, surface_raw, theta, choices=choices, process="surface_runoff")
        percolation_delta = _fixed_delta_with_theta(block_state, percolation_raw, theta, choices=choices, process="percolation")
        interflow_delta = _fixed_delta_with_theta(block_state, interflow_raw, theta, choices=choices, process="interflow")
        baseflow_delta = _fixed_delta_with_theta(block_state, baseflow_raw, theta, choices=choices, process="baseflow")
        upper_delta = torch.stack((et_delta[0] + surface_delta[0] + percolation_delta[0] + interflow_delta[0], et_delta[1] + surface_delta[1] + percolation_delta[1] + interflow_delta[1], et_delta[2] + surface_delta[2] + percolation_delta[2] + interflow_delta[2], et_delta[3] + surface_delta[3] + percolation_delta[3] + interflow_delta[3], et_delta[4] + surface_delta[4] + percolation_delta[4] + interflow_delta[4]))
        lower_outgoing_delta = torch.stack((et_delta[5] + baseflow_delta[5], et_delta[6] + baseflow_delta[6], et_delta[7] + baseflow_delta[7], et_delta[8] + baseflow_delta[8]))
        upper_scale = _block_scale(torch.stack((block_state[0], block_state[1], block_state[2], block_state[3], block_state[4])), upper_delta, torch.stack((block_floor[0], block_floor[1], block_floor[2], block_floor[3], block_floor[4])), dt)
        lower_scale = _block_scale(torch.stack((block_state[5], block_state[6], block_state[7], block_state[8])), lower_outgoing_delta, torch.stack((block_floor[5], block_floor[6], block_floor[7], block_floor[8])), dt)
        upper_proposal = torch.stack((block_state[0] + dt * upper_delta[0] * upper_scale, block_state[1] + dt * upper_delta[1] * upper_scale, block_state[2] + dt * upper_delta[2] * upper_scale, block_state[3] + dt * upper_delta[3] * upper_scale, block_state[4] + dt * upper_delta[4] * upper_scale, block_state[5], block_state[6], block_state[7], block_state[8]))
        upper_projected = _fixed_project(upper_proposal, theta, choices=choices)
        lower_proposal = torch.stack((upper_projected[0], upper_projected[1], upper_projected[2], upper_projected[3], upper_projected[4], upper_projected[5] + dt * (percolation_delta[5] * upper_scale + lower_outgoing_delta[0] * lower_scale), upper_projected[6] + dt * (percolation_delta[6] * upper_scale + lower_outgoing_delta[1] * lower_scale), upper_projected[7] + dt * (percolation_delta[7] * upper_scale + lower_outgoing_delta[2] * lower_scale), upper_projected[8] + dt * (percolation_delta[8] * upper_scale + lower_outgoing_delta[3] * lower_scale)))
        hydro = _fixed_project(lower_proposal, theta, choices=choices)
        upper_spill = (upper_proposal - upper_projected).clamp_min(0.0)
        lower_spill = (lower_proposal - hydro).clamp_min(0.0)
        upper_floor_add = (upper_projected - upper_proposal).clamp_min(0.0)
        lower_floor_add = (hydro - lower_proposal).clamp_min(0.0)
        spill_upper_sum = spill_upper_sum + recharge_spill[0] + recharge_spill[1] + recharge_spill[2] + recharge_spill[3] + recharge_spill[4] + upper_spill[0] + upper_spill[1] + upper_spill[2] + upper_spill[3] + upper_spill[4]
        spill_lower_sum = spill_lower_sum + recharge_spill[5] + recharge_spill[6] + recharge_spill[7] + recharge_spill[8] + lower_spill[5] + lower_spill[6] + lower_spill[7] + lower_spill[8]
        spill_sum = spill_upper_sum + spill_lower_sum
        floor_sum = floor_sum + recharge_floor_add.sum() + upper_floor_add.sum() + lower_floor_add.sum()
        projection_trigger_sum = projection_trigger_sum + ((recharge_spill + recharge_floor_add + upper_spill + upper_floor_add + lower_spill + lower_floor_add) > 0.0).to(hydro.dtype).sum()
        recharge_flux = recharge_raw * recharge_raw.new_tensor(_SEQUENTIAL_FLUX_MASKS["recharge"][:len(FLUX_NAMES)]) * recharge_scale
        upper_flux_scale = torch.stack(tuple(upper_scale for _ in range(len(FLUX_NAMES))))
        lower_flux_scale = torch.stack(tuple(lower_scale for _ in range(len(FLUX_NAMES))))
        et_mask = et_raw.new_tensor(_SEQUENTIAL_FLUX_MASKS["et"][:len(FLUX_NAMES)])
        et_lower_mask = et_raw.new_tensor(tuple(1 if index == 11 else 0 for index in range(len(FLUX_NAMES))))
        et_component_scale = torch.where(et_lower_mask > 0.0, lower_flux_scale, upper_flux_scale)
        block_flux = recharge_flux
        block_flux = block_flux + et_raw * et_mask * et_component_scale
        block_flux = block_flux + surface_raw * surface_raw.new_tensor(_SEQUENTIAL_FLUX_MASKS["surface_runoff"][:len(FLUX_NAMES)]) * upper_scale
        block_flux = block_flux + percolation_raw * percolation_raw.new_tensor(_SEQUENTIAL_FLUX_MASKS["percolation"][:len(FLUX_NAMES)]) * upper_scale
        block_flux = block_flux + interflow_raw * interflow_raw.new_tensor(_SEQUENTIAL_FLUX_MASKS["interflow"][:len(FLUX_NAMES)]) * upper_scale
        block_flux = block_flux + baseflow_raw * baseflow_raw.new_tensor(_SEQUENTIAL_FLUX_MASKS["baseflow"][:len(FLUX_NAMES)]) * lower_scale
        flux_sum = flux_sum + block_flux
        effective_sum = effective_sum + effective
        snow = next_snow
    spill_flux = torch.zeros_like(flux_sum)
    spill_flux = spill_flux + torch.nn.functional.pad(spill_upper_sum.reshape(1), (9, len(FLUX_NAMES) - 10))
    spill_flux = spill_flux + torch.nn.functional.pad(spill_lower_sum.reshape(1), (18, len(FLUX_NAMES) - 19))
    flux = (flux_sum + spill_flux / dt) / float(n_substeps)
    instantaneous = flux[10] + flux[9] + flux[8] + flux[18] + flux[15]
    routed = future[0] + instantaneous * fractions[0]
    future_next = torch.cat((future[1:] + instantaneous * fractions[1:], future[-1:] * 0.0))
    water_balance = hydro.sum() - start_hydro.sum() - (flux[0] - flux[4] - flux[11] - instantaneous) * dt_days
    snow_balance = _runtime_snow_balance_input(ppt, temp, theta) * dt_days - effective_sum * dt - (snow - start_snow)
    diagnostics = torch.cat((flux, torch.stack((water_balance, snow_balance / dt_days, instantaneous, spill_sum, floor_sum, projection_trigger_sum))))
    return torch.cat((hydro, snow.reshape(1), future_next)), routed, diagnostics


class RuntimeStepBuilder:
    """Build one eager function for one canonical graph signature."""

    def __init__(self, spec: StructureSpec, *, order: tuple[str, ...], n_substeps: int, execution_mode: str = "sequential") -> None:
        self.spec = spec
        self.signature = GraphSignature.from_structure(spec, sequential_order=order, n_substeps=n_substeps, execution_mode=execution_mode)

    def build(self) -> Callable[..., tuple[Tensor, Tensor, Tensor]]:
        signature = self.signature
        if signature.execution_mode == "storage_block":
            implementation = "_runtime_block_step_impl"
        elif signature.execution_mode == "coupled_rk2":
            implementation = "_runtime_coupled_rk2_step_impl"
        else:
            implementation = "_runtime_step_impl"
        choices = tuple(signature.decision(name) for name in ("ARCH1", "ARCH2", "QSURF", "QPERC"))
        function_name = f"generated_step_{signature.digest}"
        source = (
            f"def {function_name}(packed, forcing, theta, topographic_mean, topographic_max, fractions):\n"
            f"    return {implementation}(\n"
            "        packed, forcing, theta, topographic_mean, topographic_max, fractions,\n"
            f"        choices={choices!r}, order={signature.sequential_order!r}, n_substeps={signature.n_substeps},\n"
            "    )\n"
        )
        namespace = {"_runtime_step_impl": _runtime_step_impl, "_runtime_block_step_impl": _runtime_block_step_impl, "_runtime_coupled_rk2_step_impl": _runtime_coupled_rk2_step_impl}
        exec(compile(source, f"<dfuse-runtime-{signature.digest}>", "exec"), namespace)
        step = namespace[function_name]
        step.__module__ = "dfuse.runtime"
        step.__qualname__ = function_name
        step.graph_signature = signature  # type: ignore[attr-defined]
        step.generated_source = source  # type: ignore[attr-defined]
        return step


class GeneratedStepRegistry:
    def __init__(self) -> None:
        self._steps: dict[GraphSignature, Callable[..., tuple[Tensor, Tensor, Tensor]]] = {}
        self.builds = 0

    def get(self, spec: StructureSpec, *, order: tuple[str, ...], n_substeps: int, execution_mode: str = "sequential") -> Callable[..., tuple[Tensor, Tensor, Tensor]]:
        signature = GraphSignature.from_structure(spec, sequential_order=order, n_substeps=n_substeps, execution_mode=execution_mode)
        step = self._steps.get(signature)
        if step is None:
            step = RuntimeStepBuilder(spec, order=order, n_substeps=n_substeps, execution_mode=execution_mode).build()
            self._steps[signature] = step
            self.builds += 1
        return step

    def clear(self) -> None:
        self._steps.clear()
        self.builds = 0

    def signatures(self) -> tuple[GraphSignature, ...]:
        return tuple(self._steps)


_ACTIVE_RECORDS: list[dict[str, Any]] = []

class _AuditedCompiledStep:
    def __init__(self, compiled: Callable[..., tuple[Tensor, Tensor, Tensor]], record: dict[str, Any]) -> None:
        self.compiled = compiled
        self.record = record

    def __call__(self, *args: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        baseline = _counter_snapshot()
        for record in _ACTIVE_RECORDS:
            record["counter_baseline"] = baseline
        mode = "autograd" if any(isinstance(arg, Tensor) and arg.requires_grad for arg in args) else "forward"
        self.record["calls"] += 1
        started = time.perf_counter() if not self.record["warmed"] else None
        result = self.compiled(*args)
        if started is not None:
            self.record["warmed"] = True
            self.record["compile_successes"] += 1
            self.record["cold_compile_seconds"].append(time.perf_counter() - started)
        _audit_record(self.record, mode)
        after = _counter_snapshot()
        for record in _ACTIVE_RECORDS:
            record["counter_baseline"] = after
        return result

    @property
    def __name__(self) -> str:
        return getattr(self.compiled, "__name__", "compiled_step")


def _counter_snapshot() -> dict[str, dict[str, int]]:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    return {
        str(group): {str(name): int(value) for name, value in counter.items()}
        for group, counter in counters.items()
        if group in {"frames", "inductor", "graph_break", "stats", "aot_autograd", "guard_failures"} and counter
    }


def _counter_delta(before: Mapping[str, Mapping[str, int]], after: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for group in set(before) | set(after):
        values = {}
        for name in set(before.get(group, {})) | set(after.get(group, {})):
            delta = int(after.get(group, {}).get(name, 0)) - int(before.get(group, {}).get(name, 0))
            if delta:
                values[name] = delta
        if values:
            result[group] = values
    return result

def _audit_record(record: dict[str, Any], mode: str) -> None:
    current = _counter_snapshot()
    counters = _counter_delta(record["counter_baseline"], current)
    record["counter_baseline"] = current
    for group, values in counters.items():
        accumulated = record["dynamo_counters"].setdefault(group, {})
        for name, value in values.items():
            accumulated[name] = int(accumulated.get(name, 0)) + int(value)
    record["graph_breaks"] += sum(counters.get("graph_break", {}).values())
    graph_delta = int(counters.get("stats", {}).get("unique_graphs", 0))
    record[f"{mode}_unique_graphs"] += graph_delta
    record["unique_graphs"] = record["forward_unique_graphs"]
    record["recompilations"] = max(record["forward_unique_graphs"] - 1, 0)
    record["autograd_recompilations"] = max(record["autograd_unique_graphs"] - 1, 0)
    for name, value in counters.get("guard_failures", {}).items():
        record["guard_failure_reasons"][name] = int(record["guard_failure_reasons"].get(name, 0)) + int(value)


def _canonical_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device

def _publish_legacy_sequential_record(record: dict[str, Any], signature: GraphSignature, key: CompiledStepKey) -> None:
    # Keep the historical diagnostics surface while making the actual cache
    # key structure-specific.  No compiled execution uses this dictionary.
    try:
        from .kernel import _COMPILE_DIAGNOSTICS
    except ImportError:
        return
    legacy_key = repr(("runtime", signature.digest, key.device, key.dtype, key.backend, key.fullgraph, key.input_shapes))
    _COMPILE_DIAGNOSTICS.setdefault("sequential", {"variants": {}})["variants"][legacy_key] = record

class CompiledStepRegistry:
    def __init__(self) -> None:
        self._steps: dict[CompiledStepKey, _AuditedCompiledStep] = {}
        self.compile_calls = 0

    def get(
        self,
        signature: GraphSignature,
        generated_step: Callable[..., tuple[Tensor, Tensor, Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_shapes: tuple[tuple[int, ...], ...],
        backend: str = "inductor",
        fullgraph: bool = True,
    ) -> _AuditedCompiledStep:
        device = _canonical_device(device)
        key = CompiledStepKey(
            signature=signature,
            device=f"{device.type}:{device.index}",
            dtype=str(dtype),
            input_shapes=tuple(tuple(int(dim) for dim in shape) for shape in input_shapes),
            backend=backend,
            fullgraph=bool(fullgraph),
        )
        existing = self._steps.get(key)
        if existing is not None:
            return existing
        record: dict[str, Any] = {
            "key": key.to_dict(),
            "graph_signature": signature.to_dict(),
            "code_object_id": id(generated_step.__code__),
            "code_object_name": generated_step.__qualname__,
            "compile_attempts": 1,
            "compile_successes": 0,
            "fallbacks": 0,
            "calls": 0,
            "warmed": False,
            "cold_compile_seconds": [],
            "failures": [],
            "dynamo_counters": {},
            "graph_breaks": 0,
            "recompilations": 0,
            "forward_unique_graphs": 0,
            "autograd_unique_graphs": 0,
            "autograd_recompilations": 0,
            "unique_graphs": 0,
            "guard_failure_reasons": {},
            "counter_baseline": _counter_snapshot(),
        }
        try:
            compiled = torch.compile(generated_step, backend=backend, fullgraph=bool(fullgraph))
        except Exception as exc:
            record["fallbacks"] = 1
            record["failures"].append(f"{type(exc).__name__}: {str(exc)[:500]}")
            raise
        self.compile_calls += 1
        entry = _AuditedCompiledStep(compiled, record)
        self._steps[key] = entry
        _ACTIVE_RECORDS.append(record)
        record_key = f"{signature.digest}|{key.device}|{key.dtype}|{key.backend}|{key.fullgraph}|{key.input_shapes}"
        _RUNTIME_DIAGNOSTICS["compiled"][record_key] = record
        _publish_legacy_sequential_record(record, signature, key)
        return entry

    def clear(self) -> None:
        self._steps.clear()
        self.compile_calls = 0
        _ACTIVE_RECORDS.clear()

    def entries(self) -> tuple[_AuditedCompiledStep, ...]:
        return tuple(self._steps.values())


GENERATED_STEP_REGISTRY = GeneratedStepRegistry()
COMPILED_STEP_REGISTRY = CompiledStepRegistry()
_RUNTIME_DIAGNOSTICS: dict[str, Any] = {"generated_builds": 0, "compiled": {}}


def get_generated_step(spec: StructureSpec, *, order: tuple[str, ...], n_substeps: int, execution_mode: str = "sequential") -> tuple[GraphSignature, Callable[..., tuple[Tensor, Tensor, Tensor]]]:
    step = GENERATED_STEP_REGISTRY.get(spec, order=order, n_substeps=n_substeps, execution_mode=execution_mode)
    return step.graph_signature, step  # type: ignore[attr-defined, no-any-return]


def get_compiled_step(
    spec: StructureSpec,
    *,
    order: tuple[str, ...],
    n_substeps: int,
    device: torch.device,
    dtype: torch.dtype,
    input_shapes: tuple[tuple[int, ...], ...],
    backend: str = "inductor",
    fullgraph: bool = True,
    execution_mode: str = "sequential",
) -> tuple[GraphSignature, _AuditedCompiledStep]:
    signature, generated = get_generated_step(spec, order=order, n_substeps=n_substeps, execution_mode=execution_mode)
    return signature, COMPILED_STEP_REGISTRY.get(
        signature,
        generated,
        device=device,
        dtype=dtype,
        input_shapes=input_shapes,
        backend=backend,
        fullgraph=fullgraph,
    )


def runtime_compile_diagnostics() -> dict[str, Any]:
    records = {}
    for digest, record in _RUNTIME_DIAGNOSTICS["compiled"].items():
        records[digest] = {key: value for key, value in record.items() if key not in {"counter_baseline", "warmed"}}
    return {
        "generated_builds": GENERATED_STEP_REGISTRY.builds,
        "generated_signatures": [signature.to_dict() for signature in GENERATED_STEP_REGISTRY.signatures()],
        "compiled_registry_size": len(COMPILED_STEP_REGISTRY._steps),
        "compile_calls": COMPILED_STEP_REGISTRY.compile_calls,
        "records": records,
    }


def reset_runtime_registries() -> None:
    GENERATED_STEP_REGISTRY.clear()
    COMPILED_STEP_REGISTRY.clear()
    _RUNTIME_DIAGNOSTICS["generated_builds"] = 0
    _RUNTIME_DIAGNOSTICS["compiled"] = {}
