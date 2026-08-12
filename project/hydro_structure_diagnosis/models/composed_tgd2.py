"""TGD2 compositions: generic precipitation memory before unchanged runoff models."""
from __future__ import annotations

from typing import Any, Optional
import torch

from .base import BaseHydrologicalModel
from .gr4j import GR4J, GR4JLite
from .parameter_specs import GR4J_TGD2_PARAM_SPECS, SIMHYD_TGD2_PARAM_SPECS, XAJ_TGD2_PARAM_SPECS
from .simhyd import SIMHYD, SIMHYDLite
from .tgd2 import TemperatureDependentGenericDelay2
from .utils import validate_forcings, validate_params
from .xaj import XAJ, XAJLite


class _WithTGD2(BaseHydrologicalModel):
    delay_prefix = "tgd_"
    runoff_prefix = ""
    strip_runoff_param_prefix = False
    routing_method: Optional[str] = None

    def _split_params(self, params: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        delay, runoff = {}, {}
        for key, value in params.items():
            if key.startswith(self.delay_prefix): delay[key] = value
            elif key.startswith(self.runoff_prefix): runoff[key[len(self.runoff_prefix):] if self.strip_runoff_param_prefix else key] = value
        return delay, runoff

    def forward(self, forcings: dict[str, torch.Tensor], params: dict[str, torch.Tensor],
                initial_states: Optional[dict[str, torch.Tensor]] = None, return_states: bool = False):
        precip, pet, temp, device = validate_forcings(forcings)
        validate_params(params, self.parameter_specs, precip.shape[0], device, precip.dtype)
        delay_params, runoff_params = self._split_params(params)
        delay_initial = None if initial_states is None else {k[4:]: v for k, v in initial_states.items() if k.startswith("tgd_")}
        runoff_initial = None if initial_states is None else {k[len(self.runoff_prefix):]: v for k, v in initial_states.items() if k.startswith(self.runoff_prefix)}
        effective, delay_aux = self._delay(forcings, delay_params, delay_initial, return_states)
        qsim, runoff_aux = self._runoff({"precip": effective, "pet": pet, "temp": temp}, runoff_params, runoff_initial, return_states)
        if self.compact_output and not return_states: return qsim, {}
        aux: dict[str, Any] = {**{k: v for k, v in delay_aux.items() if k != "final_states"},
                               **{k: v for k, v in runoff_aux.items() if k != "final_states"},
                               "effective_precipitation": effective}
        if return_states:
            aux["final_states"] = {**{f"tgd_{k}": v for k, v in delay_aux["final_states"].items()},
                                   **{f"{self.runoff_prefix}{k}": v for k, v in runoff_aux["final_states"].items()}}
        return qsim, aux


class GR4JWithTGD2(_WithTGD2):
    runoff_prefix, strip_runoff_param_prefix = "gr4j_", True
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(); self.compact_output = False; self._delay = TemperatureDependentGenericDelay2(); self._runoff = GR4J(nearzero=nearzero)
    @property
    def parameter_specs(self): return GR4J_TGD2_PARAM_SPECS

class XAJWithTGD2(_WithTGD2):
    runoff_prefix, routing_method = "xaj_", "gamma"
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(); self.compact_output = False; self._delay = TemperatureDependentGenericDelay2(); self._runoff = XAJ(nearzero=nearzero)
    @property
    def parameter_specs(self): return XAJ_TGD2_PARAM_SPECS

class SIMHYDWithTGD2(_WithTGD2):
    runoff_prefix, routing_method = "simhyd_", "gamma"
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(); self.compact_output = False; self._delay = TemperatureDependentGenericDelay2(); self._runoff = SIMHYD(nearzero=nearzero)
    @property
    def parameter_specs(self): return SIMHYD_TGD2_PARAM_SPECS

class GR4JWithTGD2Lite(GR4JWithTGD2):
    def __init__(self, nearzero: float = 1e-8): super().__init__(nearzero); self.compact_output = True; self._delay = TemperatureDependentGenericDelay2(True); self._runoff = GR4JLite(nearzero)
class XAJWithTGD2Lite(XAJWithTGD2):
    def __init__(self, nearzero: float = 1e-8): super().__init__(nearzero); self.compact_output = True; self._delay = TemperatureDependentGenericDelay2(True); self._runoff = XAJLite(nearzero)
class SIMHYDWithTGD2Lite(SIMHYDWithTGD2):
    def __init__(self, nearzero: float = 1e-8): super().__init__(nearzero); self.compact_output = True; self._delay = TemperatureDependentGenericDelay2(True); self._runoff = SIMHYDLite(nearzero)
