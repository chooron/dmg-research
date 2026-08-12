"""Regression tests for the MARRMoT m_04 simultaneous-flux Euler step."""
from __future__ import annotations

import torch

from dmotpy.models.core.newzealand1 import newzealand1_step
from dmotpy.models.flux.baseflow import baseflow_1
from dmotpy.models.flux.evap import evap_5, evap_6
from dmotpy.models.flux.interflow import interflow_9
from dmotpy.models.flux.saturation import saturation_1


def test_newzealand1_uses_beginning_of_step_state_for_every_flux() -> None:
    """m_04's RHS evaluates qse, ET, qss and qbf from the same S1."""
    nearzero = 1e-6
    P = torch.tensor([[35.0]])
    T = torch.tensor([[5.0]])
    PET = torch.tensor([[4.0]])
    S1 = torch.tensor([[500.0]])
    s1max = torch.tensor([[1000.0]])
    sfc = torch.tensor([[0.20]])
    m = torch.tensor([[0.30]])
    a = torch.tensor([[0.01]])
    b = torch.tensor([[1.0]])
    tcbf = torch.tensor([[0.10]])

    q, ea, next_s = newzealand1_step(P, T, PET, s1max, sfc, m, a, b, tcbf, S1, nearzero=nearzero)

    qse = torch.minimum(saturation_1(P, S1, s1max, nearzero=nearzero).clamp(min=0.0), P)
    et = evap_6(m, sfc, S1, s1max, PET, nearzero=nearzero) + evap_5(m, S1, s1max, PET, nearzero=nearzero)
    qss = interflow_9(S1, a, sfc * s1max, b, nearzero=nearzero).clamp(min=0.0)
    qbf = baseflow_1(tcbf, S1, nearzero=nearzero).clamp(min=0.0)
    expected_q = qse + qss + qbf
    expected_s = S1 + P - et - expected_q

    # This interior case does not invoke the conservative depletion limiter.
    assert bool((expected_s > nearzero).all())
    torch.testing.assert_close(q, expected_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ea, et, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(next_s, expected_s, rtol=1e-6, atol=1e-6)


def test_newzealand1_conserves_water_when_simultaneous_rates_deplete_store() -> None:
    nearzero = 1e-6
    P = torch.zeros((1, 1))
    T = torch.zeros((1, 1))
    PET = torch.full((1, 1), 100.0)
    S1 = torch.full((1, 1), 1.0)
    params = [torch.full((1, 1), value) for value in (10.0, 0.05, 0.95, 1.0, 1.0, 1.0)]

    q, ea, next_s = newzealand1_step(P, T, PET, *params, S1, nearzero=nearzero)
    torch.testing.assert_close(S1 + P, q + ea + next_s, rtol=0.0, atol=2e-6)
    assert bool((next_s >= nearzero).all())
