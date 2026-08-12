"""hbv96 UH=on gradient activation audit across 4 climate scenarios."""

import torch
from models.hydrology_model import HydrologyModel


def _make_forcing(n_steps, n_grid, scenario):
    """Build synthetic forcing for different climate scenarios."""
    device = torch.device("cpu")
    P = torch.zeros(n_steps, n_grid, 3)
    if scenario == "warm_rain":
        P[:, :, 0] = torch.rand(n_steps, n_grid) * 10.0  # precip
        P[:, :, 1] = torch.rand(n_steps, n_grid) * 5.0 + 5.0  # warm T
        P[:, :, 2] = torch.rand(n_steps, n_grid) * 2.0  # moderate PET
    elif scenario == "cold_snow":
        P[:, :, 0] = torch.rand(n_steps, n_grid) * 8.0  # precip
        P[:, :, 1] = torch.rand(n_steps, n_grid) * -5.0 - 5.0  # cold T
        P[:, :, 2] = torch.rand(n_steps, n_grid) * 0.5  # low PET
    elif scenario == "freeze_thaw":
        half = n_steps // 2
        P[:half, :, 0] = torch.rand(half, n_grid) * 8.0
        P[:half, :, 1] = torch.rand(half, n_grid) * -5.0 - 5.0  # cold
        P[half:, :, 0] = torch.rand(half, n_grid) * 6.0
        P[half:, :, 1] = torch.rand(half, n_grid) * 5.0 + 5.0  # warm
        P[:, :, 2] = torch.rand(n_steps, n_grid) * 2.0
    elif scenario == "saturated_heavy_rain":
        P[:, :, 0] = torch.rand(n_steps, n_grid) * 20.0 + 5.0  # heavy rain
        P[:, :, 1] = torch.rand(n_steps, n_grid) * 3.0 + 5.0  # mild T
        P[:, :, 2] = torch.rand(n_steps, n_grid) * 0.5  # low PET
    return P


def test_hbv96_gradient_activation_all_scenarios():
    """Verify hbv96 gradients are activated by different climate forcings."""
    param_names = ["tt", "tti", "ttm", "cfr", "cfmax", "whc", "cflux",
                   "fc", "lp", "beta", "k0", "alpha", "perc", "k1", "maxbas"]
    scenarios = ["warm_rain", "cold_snow", "freeze_thaw", "saturated_heavy_rain"]
    device = torch.device("cpu")

    activation = {name: {"scenarios": set(), "grad_norms": []} for name in param_names}

    for scenario in scenarios:
        forcing = _make_forcing(200, 2, scenario)
        raw = torch.rand(1, 15, requires_grad=True)
        m = HydrologyModel(
            config={"model_name": "hbv96", "warm_up": 5, "uh_enabled": True, "uh_mode": "endpoint", "backend": "none"},
            device=device,
        )
        out = m({"x_phy": forcing}, (None, raw))
        loss = out["streamflow"].mean()
        loss.backward()
        grad = raw.grad

        assert grad is not None, f"grad is None for {scenario}"
        assert not grad.isnan().any(), f"NaN grad in {scenario}"
        assert not grad.isinf().any(), f"Inf grad in {scenario}"

        for i, name in enumerate(param_names):
            grad_val = grad[0, i].abs().item()
            activation[name]["grad_norms"].append(grad_val)
            if grad_val > 1e-12:
                activation[name]["scenarios"].add(scenario)

    # Report activation
    any_activated = False
    for name in param_names:
        n_scenarios = len(activation[name]["scenarios"])
        max_norm = max(activation[name]["grad_norms"]) if activation[name]["grad_norms"] else 0
        if n_scenarios > 0:
            any_activated = True
            print(f"  {name}: activated in {n_scenarios}/4 scenarios, max_grad={max_norm:.2e}")

    assert any_activated, "No parameters have non-zero gradient in any scenario"
    print(f"hbv96 gradient activation: PASS")
