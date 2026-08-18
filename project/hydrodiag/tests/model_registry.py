"""Canonical model/test registry for the hydro-structure diagnosis project.

Keep model discovery in one place so adding a model cannot silently leave its
forward, gradient, compilation, or boundary checks behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from models import (
    BaseHydrologicalModel,
    CemaNeige,
    CemaNeigeHyst,
    PrecipitationDelay,
    TemperatureDependentGenericDelay2,
    GR4J,
    GR4JWithCemaNeige,
    GR4JWithPrecipitationDelay,
    GR4JWithTGD2,
    HBV,
    SIMHYD,
    SIMHYDWithCemaNeige,
    SIMHYDWithPrecipitationDelay,
    SIMHYDWithTGD2,
    XAJ,
    XAJWithCemaNeige,
    XAJWithPrecipitationDelay,
    XAJWithTGD2,
)
from models.parameter_specs import (
    CEMANEIGE_PARAM_SPECS,
    CEMANEIGE_HYST_PARAM_SPECS,
    PRECIP_DELAY_PARAM_SPECS,
    TGD2_PARAM_SPECS,
    GR4J_CN_PARAM_SPECS,
    GR4J_PD_PARAM_SPECS,
    GR4J_PARAM_SPECS,
    GR4J_TGD2_PARAM_SPECS,
    HBV_PARAM_SPECS,
    SIMHYD_CN_PARAM_SPECS,
    SIMHYD_PD_PARAM_SPECS,
    SIMHYD_PARAM_SPECS,
    SIMHYD_TGD2_PARAM_SPECS,
    XAJ_CN_PARAM_SPECS,
    XAJ_PD_PARAM_SPECS,
    XAJ_PARAM_SPECS,
    XAJ_TGD2_PARAM_SPECS,
)


@dataclass(frozen=True)
class ModelCase:
    name: str
    model_cls: Type[BaseHydrologicalModel]
    parameter_specs: dict
    category: str
    has_snow: bool
    has_gamma_uh: bool


MODEL_REGISTRY: tuple[ModelCase, ...] = (
    ModelCase("HBV", HBV, HBV_PARAM_SPECS, "rainfall-runoff", True, False),
    ModelCase("GR4J", GR4J, GR4J_PARAM_SPECS, "rainfall-runoff", False, False),
    ModelCase("XAJ", XAJ, XAJ_PARAM_SPECS, "rainfall-runoff", False, True),
    ModelCase("SIMHYD", SIMHYD, SIMHYD_PARAM_SPECS, "rainfall-runoff", False, True),
    ModelCase("CemaNeige", CemaNeige, CEMANEIGE_PARAM_SPECS, "snow", True, False),
    ModelCase("CemaNeigeHyst", CemaNeigeHyst, CEMANEIGE_HYST_PARAM_SPECS, "snow", True, False),
    ModelCase("PrecipitationDelay", PrecipitationDelay, PRECIP_DELAY_PARAM_SPECS, "control", False, False),
    ModelCase("TemperatureDependentGenericDelay2", TemperatureDependentGenericDelay2, TGD2_PARAM_SPECS, "control", False, False),
    ModelCase(
        "GR4JWithCemaNeige",
        GR4JWithCemaNeige,
        GR4J_CN_PARAM_SPECS,
        "composition",
        True,
        False,
    ),
    ModelCase(
        "GR4JWithPrecipitationDelay",
        GR4JWithPrecipitationDelay,
        GR4J_PD_PARAM_SPECS,
        "composition",
        False,
        False,
    ),
    ModelCase(
        "XAJWithCemaNeige",
        XAJWithCemaNeige,
        XAJ_CN_PARAM_SPECS,
        "composition",
        True,
        True,
    ),
    ModelCase(
        "XAJWithPrecipitationDelay",
        XAJWithPrecipitationDelay,
        XAJ_PD_PARAM_SPECS,
        "composition",
        False,
        True,
    ),
    ModelCase(
        "SIMHYDWithCemaNeige",
        SIMHYDWithCemaNeige,
        SIMHYD_CN_PARAM_SPECS,
        "composition",
        True,
        True,
    ),
    ModelCase(
        "SIMHYDWithPrecipitationDelay",
        SIMHYDWithPrecipitationDelay,
        SIMHYD_PD_PARAM_SPECS,
        "composition",
        False,
        True,
    ),
    ModelCase("GR4JWithTGD2", GR4JWithTGD2, GR4J_TGD2_PARAM_SPECS, "composition", False, False),
    ModelCase("SIMHYDWithTGD2", SIMHYDWithTGD2, SIMHYD_TGD2_PARAM_SPECS, "composition", False, True),
    ModelCase("XAJWithTGD2", XAJWithTGD2, XAJ_TGD2_PARAM_SPECS, "composition", False, True),
)

MODEL_BY_NAME = {case.name: case for case in MODEL_REGISTRY}

# The canonical model gate.  The paper-style CMA-ES tests are intentionally
# separate because they validate the optimizer, not hydrological structure.
CANONICAL_MODEL_TEST_FILES: tuple[str, ...] = (
    "tests/test_model_registry.py",
    "tests/test_models_forward.py",
    "tests/test_models_grad.py",
    "tests/test_step_compile.py",
    "tests/test_gr4j_x4_gradient.py",
    "tests/test_xaj_state_carry.py",
    "tests/test_simhyd.py",
    "tests/test_precipitation_delay.py",
    "tests/test_tgd2.py",
    "tests/test_gr4j_xaj_boundaries.py",
    "tests/test_training_registry.py",
)
