"""Controlled dissertation XAJ+CemaNeige compositions."""

from __future__ import annotations

from typing import Any

from .base import BaseHydrologicalModel
from .cemaneige import CemaNeige
from .parameter_specs import (
    CEMANEIGE_PARAM_SPECS, XAJ_CONTROLLED_N_PARAM_SPECS,
    XAJ_DE_PARAM_SPECS, XAJ_GE_PARAM_SPECS, XAJ_DR_PARAM_SPECS,
    XAJ_GR_PARAM_SPECS,
)
from .xaj import XAJLite
from .xaj_variants import (
    XAJControlledN, XAJControlledNLite,
    XAJDE, XAJDELite, XAJGE, XAJGELite, XAJDR, XAJDRLite, XAJGR, XAJGRLite,
)
from .utils import validate_forcings, validate_params


class _ControlledXAJWithCemaNeige(BaseHydrologicalModel):
    """Two-stage CemaNeige -> controlled XAJ composition.

    CemaNeige is run over the complete forcing sequence and its effective
    liquid precipitation is passed to the unchanged controlled XAJ/variant
    equations.  This is the same ordering as the existing fused composition,
    while making the five Phase-0 model contracts explicit and reusable.
    """

    structure_cls: type
    structure_lite_cls: type
    structure_specs: dict[str, dict[str, Any]]
    model_name: str

    def __init__(self, nearzero: float = 1e-8, compact_output: bool = False):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self._snow = CemaNeige(nearzero=nearzero)
        structure_cls = self.structure_lite_cls if compact_output else self.structure_cls
        if compact_output:
            self._structure = structure_cls(nearzero=nearzero)
        else:
            self._structure = structure_cls(nearzero=nearzero, compact_output=False)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return {**CEMANEIGE_PARAM_SPECS, **self.structure_specs}

    def forward(self, forcings, params, initial_states=None, return_states=False):
        precip, pet, temp, device = validate_forcings(forcings)
        batch = precip.shape[0]
        validate_params(params, self.parameter_specs, batch, device, precip.dtype)
        cn_params = {k: v for k, v in params.items() if k.startswith("cn_")}
        structure_params = {k: v for k, v in params.items() if k.startswith("xaj_")}
        cn_initial = None
        structure_initial = None
        if initial_states is not None:
            cn_initial = {k[3:]: v for k, v in initial_states.items() if k.startswith("cn_")}
            structure_initial = {k[4:]: v for k, v in initial_states.items() if k.startswith("xaj_")}
        effective, cn_aux = self._snow(
            {"precip": precip, "pet": pet, "temp": temp}, cn_params,
            initial_states=cn_initial, return_states=return_states,
        )
        qsim, structure_aux = self._structure(
            {"precip": effective, "pet": pet, "temp": temp}, structure_params,
            initial_states=structure_initial, return_states=return_states,
        )
        if self.compact_output and not return_states:
            return qsim, {}
        aux = {f"cn_{k}": v for k, v in cn_aux.items() if k != "final_states"}
        aux.update({f"xaj_{k}": v for k, v in structure_aux.items() if k != "final_states"})
        aux["effective_precip"] = effective
        aux["model_name"] = self.model_name
        if return_states:
            aux["final_states"] = {
                **{f"cn_{k}": v for k, v in cn_aux.get("final_states", {}).items()},
                **{f"xaj_{k}": v for k, v in structure_aux.get("final_states", {}).items()},
            }
        return qsim, aux


class XAJControlledNWithCemaNeige(_ControlledXAJWithCemaNeige):
    structure_cls, structure_lite_cls = XAJControlledN, XAJControlledNLite
    structure_specs = XAJ_CONTROLLED_N_PARAM_SPECS
    model_name = "XAJ_CONTROLLED_N_CN"


class XAJControlledNWithCemaNeigeLite(XAJControlledNWithCemaNeige):
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)


class XAJDEWithCemaNeige(_ControlledXAJWithCemaNeige):
    structure_cls, structure_lite_cls = XAJDE, XAJDELite
    structure_specs = XAJ_DE_PARAM_SPECS
    model_name = "XAJ_D_E_CN"


class XAJDEWithCemaNeigeLite(XAJDEWithCemaNeige):
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)


class XAJGEWithCemaNeige(_ControlledXAJWithCemaNeige):
    structure_cls, structure_lite_cls = XAJGE, XAJGELite
    structure_specs = XAJ_GE_PARAM_SPECS
    model_name = "XAJ_G_E_CN"


class XAJGEWithCemaNeigeLite(XAJGEWithCemaNeige):
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)


class XAJDRWithCemaNeige(_ControlledXAJWithCemaNeige):
    structure_cls, structure_lite_cls = XAJDR, XAJDRLite
    structure_specs = XAJ_DR_PARAM_SPECS
    model_name = "XAJ_D_R_CN"


class XAJDRWithCemaNeigeLite(XAJDRWithCemaNeige):
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)


class XAJGRWithCemaNeige(_ControlledXAJWithCemaNeige):
    structure_cls, structure_lite_cls = XAJGR, XAJGRLite
    structure_specs = XAJ_GR_PARAM_SPECS
    model_name = "XAJ_G_R_CN"


class XAJGRWithCemaNeigeLite(XAJGRWithCemaNeige):
    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)
