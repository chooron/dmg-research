"""Smoke and gradient checks for streamflow-only model variants."""

import pytest
import torch

from models import (
    HBVLite, GR4JLite, SIMHYDLite,
    GR4JWithCemaNeigeLite, XAJWithCemaNeigeLite, SIMHYDWithCemaNeigeLite,
    GR4JWithPrecipitationDelayLite, XAJWithPrecipitationDelayLite,
    SIMHYDWithPrecipitationDelayLite,
    GR4JWithTGD2Lite,
    SIMHYDWithTGD2Lite,
    XAJWithTGD2Lite,
)
from models.parameter_specs import (
    HBV_PARAM_SPECS, GR4J_PARAM_SPECS, SIMHYD_PARAM_SPECS,
    GR4J_CN_PARAM_SPECS, XAJ_CN_PARAM_SPECS, SIMHYD_CN_PARAM_SPECS,
    GR4J_PD_PARAM_SPECS, XAJ_PD_PARAM_SPECS, SIMHYD_PD_PARAM_SPECS,
    GR4J_TGD2_PARAM_SPECS, SIMHYD_TGD2_PARAM_SPECS, XAJ_TGD2_PARAM_SPECS,
)


CASES = (
    (HBVLite, HBV_PARAM_SPECS, False),
    (GR4JLite, GR4J_PARAM_SPECS, False),
    (SIMHYDLite, SIMHYD_PARAM_SPECS, False),
    (GR4JWithCemaNeigeLite, GR4J_CN_PARAM_SPECS, False),
    (XAJWithCemaNeigeLite, XAJ_CN_PARAM_SPECS, False),
    (SIMHYDWithCemaNeigeLite, SIMHYD_CN_PARAM_SPECS, False),
    (GR4JWithPrecipitationDelayLite, GR4J_PD_PARAM_SPECS, False),
    (XAJWithPrecipitationDelayLite, XAJ_PD_PARAM_SPECS, False),
    (SIMHYDWithPrecipitationDelayLite, SIMHYD_PD_PARAM_SPECS, False),
    (GR4JWithTGD2Lite, GR4J_TGD2_PARAM_SPECS, False),
    (SIMHYDWithTGD2Lite, SIMHYD_TGD2_PARAM_SPECS, False),
    (XAJWithTGD2Lite, XAJ_TGD2_PARAM_SPECS, False),
)


def _forcing(batch=2, steps=12):
    torch.manual_seed(31)
    forcing = {
        "precip": torch.rand(batch, steps) * 8.0,
        "pet": torch.rand(batch, steps) * 4.0,
        "temp": torch.rand(batch, steps) * 20.0 - 5.0,
    }
    return forcing


def _params(specs, batch=2):
    return {
        name: torch.full(
            (batch,), float(spec["default"]), requires_grad=True,
        )
        for name, spec in specs.items()
    }


@pytest.mark.parametrize("model_cls,specs,_has_tgd_stats", CASES)
def test_lite_model_returns_only_finite_streamflow_and_gradients(
    model_cls, specs, _has_tgd_stats,
):
    forcing = _forcing()
    params = _params(specs)
    qsim, aux = model_cls()(forcing, params)
    qsim.square().mean().backward()

    assert qsim.shape == (2, 12)
    assert torch.isfinite(qsim).all()
    assert aux == {}
    assert all(value.grad is not None for value in params.values())
    assert all(torch.isfinite(value.grad).all() for value in params.values())
