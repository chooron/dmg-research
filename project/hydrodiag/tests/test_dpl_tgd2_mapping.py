"""dPL retains Lite-v2 sigmoid followed by its physical inverse mapping."""

import math

import torch
from models.parameter_specs import XAJ_TGD2_PARAM_SPECS
from training.dpl.run_dpl_model import StaticParameterNet, physical_parameters


def test_dpl_tgd2_log_residence_mapping_and_gradients():
    names = list(XAJ_TGD2_PARAM_SPECS)
    lower = torch.tensor(
        [XAJ_TGD2_PARAM_SPECS[name]["lower"] for name in names], dtype=torch.float64
    )
    upper = torch.tensor(
        [XAJ_TGD2_PARAM_SPECS[name]["upper"] for name in names], dtype=torch.float64
    )
    theta = torch.full((2, len(names)), 0.5, dtype=torch.float64, requires_grad=True)
    physical = physical_parameters(theta, names, lower, upper - lower)
    assert torch.allclose(
        physical["tgd_tau_warm"],
        torch.full((2,), math.sqrt(1e-4 * 3.0), dtype=torch.float64),
    )
    assert torch.allclose(
        physical["tgd_delta_tau_cold"],
        torch.full((2,), math.sqrt(0.1 * 180.0), dtype=torch.float64),
    )
    (physical["tgd_tau_warm"].sum() + physical["tgd_delta_tau_cold"].sum()).backward()
    assert theta.grad[:, :2].isfinite().all() and (theta.grad[:, :2].abs() > 0).all()
    net = StaticParameterNet(4, XAJ_TGD2_PARAM_SPECS, [8], 0.0, 1e-4)
    expected = torch.tensor([0.25, 10.0], dtype=torch.float64)
    generated = net(torch.zeros(1, 4))
    generated_physical = physical_parameters(generated, names, lower, upper - lower)
    assert torch.allclose(
        torch.stack(
            (
                generated_physical["tgd_tau_warm"][0],
                generated_physical["tgd_delta_tau_cold"][0],
            )
        ),
        expected,
        rtol=1e-4,
    )
