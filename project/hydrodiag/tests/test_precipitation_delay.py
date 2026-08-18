"""Tests for the temperature-agnostic, parameter-matched placebo module."""

from __future__ import annotations

import pytest
import torch
from models import (
    GR4J,
    SIMHYD,
    XAJ,
    GR4JWithPrecipitationDelay,
    PrecipitationDelay,
    SIMHYDWithPrecipitationDelay,
    XAJWithPrecipitationDelay,
)
from models.parameter_specs import (
    GR4J_PARAM_SPECS,
    GR4J_PD_PARAM_SPECS,
    PRECIP_DELAY_PARAM_SPECS,
    SIMHYD_PARAM_SPECS,
    SIMHYD_PD_PARAM_SPECS,
    XAJ_PARAM_SPECS,
    XAJ_PD_PARAM_SPECS,
)
from models.xaj import XAJ_UH_MAX_LEN, _gamma_uh_ordinates

COMPOSED = (
    (GR4JWithPrecipitationDelay, GR4J, GR4J_PD_PARAM_SPECS, GR4J_PARAM_SPECS),
    (XAJWithPrecipitationDelay, XAJ, XAJ_PD_PARAM_SPECS, XAJ_PARAM_SPECS),
    (SIMHYDWithPrecipitationDelay, SIMHYD, SIMHYD_PD_PARAM_SPECS, SIMHYD_PARAM_SPECS),
)


def _forcing(batch: int = 2, time: int = 96, dtype: torch.dtype = torch.float64):
    phase = torch.arange(time, dtype=dtype)
    precip = torch.clamp(5.0 + 4.0 * torch.sin(phase * 0.17), min=0.0)
    pet = torch.clamp(2.0 + 1.5 * torch.cos(phase * 0.11), min=0.0)
    temp = 8.0 * torch.sin(phase * 0.13) - 1.0
    return {
        "precip": precip.expand(batch, -1).clone(),
        "pet": pet.expand(batch, -1).clone(),
        "temp": temp.expand(batch, -1).clone(),
    }


def _params(specs: dict, batch: int, dtype: torch.dtype = torch.float64):
    return {
        name: torch.full((batch,), float(spec["default"]), dtype=dtype)
        for name, spec in specs.items()
    }


def test_delay_is_temperature_agnostic_and_mass_conservative():
    forcings = _forcing()
    params = _params(PRECIP_DELAY_PARAM_SPECS, forcings["precip"].shape[0])
    model = PrecipitationDelay()

    effective_a, aux_a = model(forcings=forcings, params=params, return_states=True)
    altered = {**forcings, "temp": forcings["temp"] + 30.0}
    effective_b, _ = model(forcings=altered, params=params, return_states=True)

    assert torch.allclose(effective_a, effective_b, atol=1e-12, rtol=0.0)
    residual = (
        forcings["precip"].sum(dim=1)
        - effective_a.sum(dim=1)
        - aux_a["final_states"]["S"]
    )
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-12, rtol=0.0)


def test_alpha_zero_is_exact_no_delay_limit():
    forcings = _forcing(batch=1)
    params = _params(PRECIP_DELAY_PARAM_SPECS, 1)
    params["pd_alpha"][:] = 0.0
    effective, aux = PrecipitationDelay()(
        forcings=forcings, params=params, return_states=True
    )

    assert torch.allclose(effective, forcings["precip"], atol=0.0, rtol=0.0)
    assert torch.equal(aux["final_states"]["S"], torch.zeros(1, dtype=effective.dtype))


def test_precipitation_delay_fixed_value_regression():
    """Lock the historical GD recurrence independently of TGD additions."""
    forcings = {
        "precip": torch.tensor([[0.0, 10.0, 0.0, 3.0, 2.0]], dtype=torch.float64),
        "pet": torch.zeros(1, 5, dtype=torch.float64),
        "temp": torch.tensor([[1.0, -1.0, 4.0, 0.0, 2.0]], dtype=torch.float64),
    }
    params = {
        "pd_alpha": torch.tensor([0.35], dtype=torch.float64),
        "pd_tau": torch.tensor([4.0], dtype=torch.float64),
    }
    effective, aux = PrecipitationDelay()(forcings, params, return_states=True)
    expected_effective = torch.tensor(
        [
            [
                0.0,
                7.274197259250083,
                0.602945431755700,
                2.651833552175690,
                2.001427971870230,
            ]
        ],
        dtype=torch.float64,
    )
    expected_storage = torch.tensor([2.469595784948297], dtype=torch.float64)
    assert torch.allclose(effective, expected_effective, atol=1e-14, rtol=0.0)
    assert torch.allclose(
        aux["final_states"]["S"], expected_storage, atol=1e-14, rtol=0.0
    )


@pytest.mark.parametrize("delay_cls,base_cls,composed_specs,base_specs", COMPOSED)
def test_composed_delay_is_finite_and_has_all_parameter_gradients(
    delay_cls, base_cls, composed_specs, base_specs
):
    del base_cls, base_specs
    forcings = _forcing(batch=2, dtype=torch.float32)
    params = {
        name: torch.nn.Parameter(value.float())
        for name, value in _params(composed_specs, 2, torch.float32).items()
    }
    qsim, aux = delay_cls()(forcings=forcings, params=params, return_states=True)
    loss = qsim.square().mean() + 1e-3 * aux["effective_precip"].square().mean()
    loss.backward()

    assert qsim.shape == forcings["precip"].shape
    assert torch.isfinite(qsim).all()
    assert torch.isfinite(aux["effective_precip"]).all()
    for name, parameter in params.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for name in ("pd_alpha", "pd_tau"):
        assert params[name].grad.abs().max() > 0.0, name


@pytest.mark.parametrize("delay_cls,base_cls,composed_specs,base_specs", COMPOSED)
def test_alpha_zero_composition_matches_base_model(
    delay_cls, base_cls, composed_specs, base_specs
):
    forcings = _forcing(batch=2)
    delay_params = _params(composed_specs, 2)
    delay_params["pd_alpha"][:] = 0.0
    # Use the declared base specs to make the comparison explicit and
    # independent of dictionary order.
    base_params = {
        name: delay_params[
            f"gr4j_{name}" if delay_cls is GR4JWithPrecipitationDelay else name
        ]
        for name in base_specs
    }
    q_delay, aux = delay_cls()(
        forcings=forcings, params=delay_params, return_states=True
    )
    q_base, _ = base_cls()(forcings=forcings, params=base_params)

    assert torch.allclose(q_delay, q_base, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        aux["effective_precip"], forcings["precip"], atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("delay_cls,base_cls,composed_specs,base_specs", COMPOSED)
def test_composed_delay_boundary_values_are_finite(
    delay_cls, base_cls, composed_specs, base_specs
):
    del base_cls, base_specs
    forcings = _forcing(batch=5, dtype=torch.float32)
    names = list(composed_specs)
    defaults = [float(spec["default"]) for spec in composed_specs.values()]
    rows = [defaults.copy()]
    for name in ("pd_alpha", "pd_tau"):
        for edge in ("lower", "upper"):
            row = defaults.copy()
            row[names.index(name)] = float(composed_specs[name][edge])
            rows.append(row)
    values = torch.tensor(rows, dtype=torch.float32)
    params = {
        name: torch.nn.Parameter(values[:, index].clone())
        for index, name in enumerate(names)
    }

    qsim, aux = delay_cls()(forcings=forcings, params=params, return_states=True)
    outputs = [qsim, aux["effective_precip"], aux["delay_storage"]]
    assert all(torch.isfinite(value).all() for value in outputs)
    loss = qsim.square().mean() + aux["effective_precip"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in params.values())
    assert all(torch.isfinite(parameter.grad).all() for parameter in params.values())


@pytest.mark.parametrize(
    "model_cls,specs",
    [
        (GR4JWithPrecipitationDelay, GR4J_PD_PARAM_SPECS),
        (XAJWithPrecipitationDelay, XAJ_PD_PARAM_SPECS),
        (SIMHYDWithPrecipitationDelay, SIMHYD_PD_PARAM_SPECS),
    ],
)
def test_composed_delay_full_system_water_balance(model_cls, specs):
    """Raw P - ET - Q equals the change in delay and runoff stores.

    PET is set to zero for GR4J because its current public auxiliary output
    does not expose the internal production-store evaporation separately.  XAJ
    and SIMHYD are checked through their explicit evaporation outputs.  The
    small tolerance reflects the existing nearzero protections in GR4J/XAJ.
    """
    forcings = _forcing(batch=2, time=180)
    forcings["pet"].zero_()
    params = _params(specs, 2)
    if model_cls is GR4JWithPrecipitationDelay:
        # Groundwater exchange is an external GR4J flux when x2 != 0; setting
        # it to zero makes this a closed-system water-balance audit.
        params["gr4j_x2"].zero_()
    model = model_cls()
    qsim, aux = model(forcings=forcings, params=params, return_states=True)

    states = aux["final_states"]
    if model_cls is GR4JWithPrecipitationDelay:
        initial_storage = 0.5 * params["gr4j_x1"] + 0.5 * params["gr4j_x3"]
        final_storage = (
            states["gr4j_s_prod"]
            + states["gr4j_s_route"]
            + states["gr4j_uh1_buf"][:, 1:].sum(dim=1)
            + states["gr4j_uh2_buf"][:, 1:].sum(dim=1)
        )
        evap = torch.zeros_like(qsim)
    elif model_cls is XAJWithPrecipitationDelay:
        im = params["xaj_im"]
        ci = params["xaj_ci"]
        cg = params["xaj_cg"]
        uh = _gamma_uh_ordinates(
            params["xaj_a"],
            params["xaj_theta"],
            XAJ_UH_MAX_LEN,
            qsim.device,
            qsim.dtype,
        )
        pending_fraction = torch.cumsum(torch.flip(uh[:, 1:], dims=[-1]), dim=-1)
        initial_storage = (
            (1.0 - im)
            * (
                0.6 * params["xaj_um"]
                + 0.6 * params["xaj_lm"]
                + 0.6 * params["xaj_dm"]
                + 0.05 * params["xaj_sm"]
            )
            + ci * 0.1 / (1.0 - ci)
            + cg * 0.1 / (1.0 - cg)
        )
        final_storage = (
            (1.0 - im) * (aux["wu"] + aux["wl"] + aux["wd"] + aux["fr"] * aux["s"])
            + ci * states["xaj_qi"] / (1.0 - ci)
            + cg * states["xaj_qg"] / (1.0 - cg)
            + (states["xaj_rs_uh_buffer"] * pending_fraction).sum(dim=1)
        )
        evap = aux["evap"]
    else:
        initial_storage = 0.5 * params["simhyd_smsc"]
        final_storage = aux["soil"] + aux["groundwater"] + aux["routing_storage"]
        evap = aux["evap"]

    final_storage = final_storage + states["pd_S"]
    residual = (
        forcings["precip"].sum(dim=1)
        - evap.sum(dim=1)
        - qsim.sum(dim=1)
        - (final_storage - initial_storage)
    )
    assert torch.allclose(residual, torch.zeros_like(residual), atol=3e-5, rtol=0.0), (
        residual
    )
