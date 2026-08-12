"""Regression test for XAJ linear-reservoir state carry."""

import torch

from models import GR4J, XAJ, XAJWithCemaNeige
from models.parameter_specs import GR4J_PARAM_SPECS, XAJ_PARAM_SPECS, XAJ_CN_PARAM_SPECS


def test_xaj_updates_linear_reservoir_states_between_steps():
    model = XAJ().to(dtype=torch.float32)
    forcing = {
        "precip": torch.zeros(1, 20),
        "pet": torch.zeros(1, 20),
        "temp": torch.zeros(1, 20),
    }
    params = {
        name: torch.tensor([spec["default"]], dtype=torch.float32)
        for name, spec in XAJ_PARAM_SPECS.items()
    }
    params["xaj_ki"][:] = 0.0
    params["xaj_kg"][:] = 0.0
    params["xaj_ci"][:] = 0.1
    params["xaj_cg"][:] = 0.9

    _, aux = model(forcings=forcing, params=params)

    qi = aux["qi"][0]
    qg = aux["qg"][0]
    assert qi[-1] < qi[0] * 1e-3
    assert qg[-1] < qg[0] * 0.2


def test_xaj_deep_evaporation_cannot_exceed_deep_storage():
    """Dry-period ET must not remove more water than WD contains."""
    model = XAJ().to(dtype=torch.float64)
    forcings = {
        "precip": torch.zeros(1, 1, dtype=torch.float64),
        "pet": torch.full((1, 1), 6.0, dtype=torch.float64),
        "temp": torch.zeros(1, 1, dtype=torch.float64),
    }
    params = {
        name: torch.tensor([spec["default"]], dtype=torch.float64)
        for name, spec in XAJ_PARAM_SPECS.items()
    }
    initial = {
        "wu": torch.zeros(1, dtype=torch.float64),
        "wl": torch.zeros(1, dtype=torch.float64),
        "wd": torch.full((1,), 0.5, dtype=torch.float64),
        "qi": torch.zeros(1, dtype=torch.float64),
        "qg": torch.zeros(1, dtype=torch.float64),
    }
    with torch.no_grad():
        _, aux = model(forcings=forcings, params=params, initial_states=initial, return_states=True)
    assert torch.allclose(aux["evap"], torch.full((1, 1), 0.5, dtype=torch.float64))
    assert torch.allclose(aux["final_states"]["wd"], torch.zeros(1, dtype=torch.float64))


def test_xaj_chunked_run_matches_full_run_with_uh_buffer():
    torch.manual_seed(5)
    model = XAJ().to(dtype=torch.float32)
    forcings = {
        "precip": torch.rand(1, 80) * 8.0,
        "pet": torch.rand(1, 80) * 4.0,
        "temp": torch.rand(1, 80) * 20.0 - 5.0,
    }
    params = {
        name: torch.tensor([spec["default"]], dtype=torch.float32)
        for name, spec in XAJ_PARAM_SPECS.items()
    }
    with torch.no_grad():
        q_full, _ = model(forcings=forcings, params=params)
        _, first_aux = model(
            forcings={k: v[:, :40] for k, v in forcings.items()},
            params=params,
            return_states=True,
        )
        q_second, _ = model(
            forcings={k: v[:, 40:] for k, v in forcings.items()},
            params=params,
            initial_states=first_aux["final_states"],
        )
    assert torch.allclose(q_full[:, 40:], q_second, atol=1e-6, rtol=1e-5)


def test_xaj_cemaneige_preserves_prefixed_states():
    model = XAJWithCemaNeige().to(dtype=torch.float32)
    forcings = {"precip": torch.zeros(1, 4), "pet": torch.zeros(1, 4), "temp": torch.zeros(1, 4)}
    params = {
        name: torch.tensor([spec["default"]], dtype=torch.float32)
        for name, spec in XAJ_CN_PARAM_SPECS.items()
    }
    init = {"xaj_qi": torch.tensor([5.0]), "xaj_qg": torch.tensor([2.0]), "cn_G": torch.tensor([3.0])}
    with torch.no_grad():
        q_default, _ = model(forcings=forcings, params=params)
        q_initial, aux = model(forcings=forcings, params=params, initial_states=init, return_states=True)
    assert not torch.allclose(q_default, q_initial)
    assert {"cn_G", "xaj_qi", "xaj_qg", "xaj_rs_uh_buffer"} <= set(aux["final_states"])


def test_gr4j_chunked_run_matches_full_run_with_uh_buffers():
    """Pending GR4J unit-hydrograph water must be included in final states."""
    torch.manual_seed(11)
    model = GR4J().to(dtype=torch.float32)
    forcings = {
        "precip": torch.rand(1, 80) * 8.0,
        "pet": torch.rand(1, 80) * 4.0,
        "temp": torch.rand(1, 80) * 20.0 - 5.0,
    }
    params = {
        name: torch.tensor([spec["default"]], dtype=torch.float32)
        for name, spec in GR4J_PARAM_SPECS.items()
    }
    with torch.no_grad():
        q_full, _ = model(forcings=forcings, params=params)
        _, first_aux = model(
            forcings={k: v[:, :40] for k, v in forcings.items()},
            params=params,
            return_states=True,
        )
        q_second, _ = model(
            forcings={k: v[:, 40:] for k, v in forcings.items()},
            params=params,
            initial_states=first_aux["final_states"],
        )
    assert {"s_prod", "s_route", "uh1_buf", "uh2_buf"} == set(first_aux["final_states"])
    assert torch.allclose(q_full[:, 40:], q_second, atol=1e-6, rtol=1e-5)
