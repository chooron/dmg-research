import math

import pytest
import torch

from project.benchmark.dpl.nn_parameterizer import CatchmentParameterizer


PARAMETERS = ["smax", "beta", "d_split", "percmax", "lp", "nlagf", "nlags", "kf", "ks", "imax"]
GROUPS = {
    "production": PARAMETERS[:5],
    "routing": PARAMETERS[5:9],
    "interception": PARAMETERS[9:],
}


def make_model(*, architecture="legacy", output_transform="sigmoid"):
    return CatchmentParameterizer(
        in_features=35,
        out_features=len(PARAMETERS),
        hidden_dims=[16, 16],
        dropout=0.0,
        architecture=architecture,
        output_transform=output_transform,
        parameter_names=PARAMETERS,
        parameter_groups=GROUPS,
    )


def test_round2_bounded_mappings_are_finite_and_in_range():
    x = torch.linspace(-100.0, 100.0, 401).reshape(-1, 1).repeat(1, len(PARAMETERS))
    for transform in ("sigmoid", "softsign", "arctan"):
        model = make_model(output_transform=transform)
        normalized, diagnostics = model._apply_transform(x)
        assert torch.isfinite(normalized).all()
        assert torch.isfinite(diagnostics).all()
        if transform == "sigmoid":
            assert bool((normalized >= 0).all() and (normalized <= 1).all())
        else:
            assert bool((normalized > 0).all() and (normalized < 1).all())
        assert bool((diagnostics >= 0).all())


def test_round2_mapping_formulas_match_reference():
    z = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]] * len(PARAMETERS))
    for transform in ("softsign", "arctan"):
        model = make_model(output_transform=transform)
        normalized, jacobian = model._apply_transform(z)
        if transform == "softsign":
            expected = 0.5 * (z / (1 + z.abs()) + 1)
            expected_jacobian = 0.5 / (1 + z.abs()).square()
        else:
            expected = 0.5 + torch.atan(z) / torch.pi
            expected_jacobian = 0.5 / (torch.pi * (1 + z.square()))
        assert torch.allclose(normalized, expected)
        assert torch.allclose(jacobian, expected_jacobian)


def test_residual_process_zero_init_matches_legacy():
    torch.manual_seed(7)
    legacy = make_model(architecture="legacy")
    residual = make_model(architecture="residual_process")
    residual.net.load_state_dict(legacy.net.state_dict())
    x = torch.randn(11, 35)
    with torch.no_grad():
        y_legacy = legacy(x)
        y_residual, diagnostics = residual(x, return_diagnostics=True)
    assert torch.equal(y_legacy, y_residual)
    assert torch.equal(diagnostics["raw_latent"], legacy._raw_output(x))


def test_residual_selective_only_routes_routing_and_interception():
    model = make_model(architecture="residual_selective")
    assert set(model.residual_adapters.keys()) == {"routing", "interception"}
    assert "production" not in model.residual_adapters


def test_residual_adapters_receive_finite_gradients_and_round_trip():
    torch.manual_seed(11)
    model = make_model(architecture="residual_process")
    x = torch.randn(13, 35)
    output = model(x)
    loss = output.square().mean()
    loss.backward()
    residual_params = [p for n, p in model.named_parameters() if "residual_adapters" in n]
    assert residual_params
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in residual_params)

    restored = make_model(architecture="residual_process")
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))


def test_legacy_state_dict_strict_load_remains_compatible():
    source = make_model(architecture="legacy")
    restored = make_model(architecture="legacy")
    restored.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        x = torch.randn(5, 35)
        assert torch.equal(source(x), restored(x))
