"""Runoff models composed with the temperature-agnostic precipitation delay."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import BaseHydrologicalModel
from .gr4j import GR4J, GR4JLite
from .parameter_specs import (
    GR4J_PD_PARAM_SPECS,
    SIMHYD_PD_PARAM_SPECS,
    XAJ_PD_PARAM_SPECS,
)
from .precip_delay import PrecipitationDelay
from .simhyd import SIMHYD, SIMHYDLite
from .utils import validate_forcings, validate_params
from .xaj import XAJ, XAJLite


class _WithPrecipitationDelay(BaseHydrologicalModel):
    delay_prefix = "pd_"
    runoff_prefix = ""
    strip_runoff_param_prefix = False
    routing_method: Optional[str] = None

    def _split_params(
        self,
        params: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        delay_params = {}
        runoff_params = {}
        for key, value in params.items():
            if key.startswith(self.delay_prefix):
                delay_params[key] = value
            elif key.startswith(self.runoff_prefix):
                runoff_params[
                    key[len(self.runoff_prefix) :]
                    if self.strip_runoff_param_prefix
                    else key
                ] = value
        return delay_params, runoff_params

    def _split_initial_states(
        self,
        initial_states: Optional[dict[str, torch.Tensor]],
    ) -> tuple[Optional[dict[str, torch.Tensor]], Optional[dict[str, torch.Tensor]]]:
        if initial_states is None:
            return None, None
        delay_initial = {
            key[len(self.delay_prefix) :]: value
            for key, value in initial_states.items()
            if key.startswith(self.delay_prefix)
        }
        runoff_initial = {
            key[len(self.runoff_prefix) :]: value
            for key, value in initial_states.items()
            if key.startswith(self.runoff_prefix)
        }
        return delay_initial, runoff_initial

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        precip, pet, temp, device = validate_forcings(forcings)
        batch = precip.shape[0]
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)
        delay_params, runoff_params = self._split_params(params)
        delay_initial, runoff_initial = self._split_initial_states(initial_states)

        effective_precip, delay_aux = self._delay(
            forcings=forcings,
            params=delay_params,
            initial_states=delay_initial,
            return_states=return_states,
        )
        qsim, runoff_aux = self._runoff(
            forcings={"precip": effective_precip, "pet": pet, "temp": temp},
            params=runoff_params,
            initial_states=runoff_initial,
            return_states=return_states,
        )

        if getattr(self, "compact_output", False) and not return_states:
            return qsim, {}

        aux = {key: value for key, value in delay_aux.items() if key != "final_states"}
        aux.update(
            {key: value for key, value in runoff_aux.items() if key != "final_states"}
        )
        aux["effective_precip"] = effective_precip
        if return_states:
            aux["final_states"] = {
                **{
                    f"{self.delay_prefix}{key}": value
                    for key, value in delay_aux["final_states"].items()
                },
                **{
                    f"{self.runoff_prefix}{key}": value
                    for key, value in runoff_aux["final_states"].items()
                },
            }
        return qsim, aux


class GR4JWithPrecipitationDelay(_WithPrecipitationDelay):
    """GR4J preceded by a two-parameter conservative precipitation delay."""

    runoff_prefix = "gr4j_"
    strip_runoff_param_prefix = True

    def __init__(self, nearzero: float = 1e-8):
        super().__init__()
        self.nearzero = nearzero
        self._delay = PrecipitationDelay(nearzero=nearzero)
        self._runoff = GR4J(nearzero=nearzero)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return GR4J_PD_PARAM_SPECS


class XAJWithPrecipitationDelay(_WithPrecipitationDelay):
    """XAJ preceded by a two-parameter conservative precipitation delay."""

    runoff_prefix = "xaj_"
    routing_method = "gamma"

    def __init__(self, nearzero: float = 1e-8):
        super().__init__()
        self.nearzero = nearzero
        self._delay = PrecipitationDelay(nearzero=nearzero)
        self._runoff = XAJ(nearzero=nearzero)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return XAJ_PD_PARAM_SPECS


class SIMHYDWithPrecipitationDelay(_WithPrecipitationDelay):
    """SIMHYD preceded by a two-parameter conservative precipitation delay."""

    runoff_prefix = "simhyd_"
    routing_method = "gamma"

    def __init__(self, nearzero: float = 1e-8):
        super().__init__()
        self.nearzero = nearzero
        self._delay = PrecipitationDelay(nearzero=nearzero)
        self._runoff = SIMHYD(nearzero=nearzero)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return SIMHYD_PD_PARAM_SPECS


class GR4JWithPrecipitationDelayLite(GR4JWithPrecipitationDelay):
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero)
        self.compact_output = True
        self._runoff = GR4JLite(nearzero=nearzero)


class XAJWithPrecipitationDelayLite(XAJWithPrecipitationDelay):
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero)
        self.compact_output = True
        self._runoff = XAJLite(nearzero=nearzero)


class SIMHYDWithPrecipitationDelayLite(SIMHYDWithPrecipitationDelay):
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero)
        self.compact_output = True
        self._runoff = SIMHYDLite(nearzero=nearzero)
