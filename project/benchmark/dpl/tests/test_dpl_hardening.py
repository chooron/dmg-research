from __future__ import annotations

import pytest
import torch

from dpl.nn_parameterizer import CatchmentParameterizer
from dpl.optimizer_transaction import FiniteOptimizerTransaction, validate_finite_training_state
from src.model_registry import get_spec


def test_finite_transaction_updates_and_clips() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    transaction = FiniteOptimizerTransaction(optimizer, [parameter], clip_norm=0.1, failure_policy="skip")

    result = transaction.step((1000.0 * parameter.square()).sum())

    assert result.success
    assert result.diagnostics["clipped"]
    assert parameter.isfinite()
    assert transaction.successful_steps == 1
    assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))


def test_float32_large_finite_gradient_is_clipped_without_norm_overflow() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    transaction = FiniteOptimizerTransaction(optimizer, [parameter], clip_norm=1.0)

    result = transaction.step((parameter * 1.0e20).sum())

    assert result.success
    assert result.diagnostics["clipped"]
    assert result.diagnostics["pre_clip_grad_norm"] == pytest.approx(1.0e20, rel=1.0e-6)
    assert result.diagnostics["post_clip_grad_norm"] == pytest.approx(1.0, rel=1.0e-6)
    assert torch.isfinite(parameter).all()



def test_grad_scaler_transaction_scales_backward_before_unscale() -> None:
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
        pytest.skip("GradScaler unavailable")
    try:
        scaler = torch.amp.GradScaler("cpu")
    except (TypeError, RuntimeError):
        pytest.skip("CPU GradScaler unavailable")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    transaction = FiniteOptimizerTransaction(optimizer, [parameter], scaler=scaler, clip_norm=1.0)
    result = transaction.step(parameter.square().sum())
    assert result.success
    assert torch.isfinite(parameter).all()


def test_nonfinite_loss_does_not_step_or_create_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    transaction = FiniteOptimizerTransaction(optimizer, [parameter], failure_policy="skip")
    before = parameter.detach().clone()

    result = transaction.step(torch.tensor(float("nan")))

    assert not result.success
    assert result.reason == "nonfinite_loss"
    torch.testing.assert_close(parameter, before)
    assert optimizer.state == {}


def test_nonfinite_gradient_is_rejected_without_step() -> None:
    class InfGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value * 0.0

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output * float("inf")

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    transaction = FiniteOptimizerTransaction(optimizer, [parameter], failure_policy="skip")
    before = parameter.detach().clone()

    result = transaction.step(InfGradient.apply(parameter).sum())

    assert not result.success
    assert result.reason == "nonfinite_gradient"
    torch.testing.assert_close(parameter, before)
    assert optimizer.state == {}


def test_corrupted_optimizer_state_is_rejected() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    transaction = FiniteOptimizerTransaction(optimizer, [parameter], failure_policy="skip")
    assert transaction.step(parameter.square().sum()).success
    for state in optimizer.state.values():
        state["exp_avg"].fill_(float("nan"))
    before = parameter.detach().clone()

    result = transaction.step(parameter.square().sum())

    assert not result.success
    assert result.reason == "nonfinite_pre_step_state"
    torch.testing.assert_close(parameter, before)


def test_post_step_corruption_fails_fast() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    class CorruptingAdam(torch.optim.Adam):
        def step(self, closure=None):
            result = super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("nan"))
            return result

    optimizer = CorruptingAdam([parameter], lr=0.1)
    transaction = FiniteOptimizerTransaction(optimizer, [parameter], failure_policy="skip")
    with pytest.raises(FloatingPointError, match="post_step_corruption"):
        transaction.step(parameter.square().sum())
    assert transaction.aborted_steps == 1


def test_checkpoint_state_validation_rejects_corruption() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.square().sum().backward()
    optimizer.step()
    next(iter(optimizer.state.values()))["exp_avg"].fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="checkpoint_nonfinite_state"):
        validate_finite_training_state([parameter], optimizer, loss=1.0)


def test_sigmoid_saturation_metric_penalty_and_recovery_gradient_are_finite() -> None:
    parameterizer = CatchmentParameterizer(2, 1, hidden_dims=[4], saturation_floor=0.01)
    raw = torch.tensor([[0.0], [20.0], [-20.0]], requires_grad=True)
    diagnostics = parameterizer.mapping_diagnostics(raw)
    jacobian = diagnostics["normalized_jacobian"]
    assert jacobian[0, 0] > jacobian[1, 0]
    assert jacobian[0, 0] > jacobian[2, 0]
    penalty = parameterizer.saturation_regularizer_from_diagnostics(diagnostics)
    gradient = torch.autograd.grad(penalty, raw)[0]
    assert torch.isfinite(penalty)
    assert torch.isfinite(gradient).all()
    assert gradient[1, 0] > 0.0
    assert gradient[2, 0] < 0.0


def test_linear_transform_has_no_saturation_penalty() -> None:
    parameterizer = CatchmentParameterizer(2, 1, hidden_dims=[4], output_transform="linear")
    raw = torch.tensor([[100.0]], requires_grad=True)
    penalty = parameterizer.saturation_regularizer(raw)
    assert penalty.item() == 0.0
    assert torch.autograd.grad(penalty, raw)[0].item() == 0.0


def test_legacy_forward_and_bounds_remain_unchanged() -> None:
    bounds = (torch.tensor([2.0, -1.0]), torch.tensor([4.0, 3.0]))
    parameterizer = CatchmentParameterizer(3, 2, hidden_dims=[5, 4], param_bounds=bounds)
    attributes = torch.randn(7, 3)
    expected = bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(parameterizer.net(attributes))
    actual = parameterizer(attributes)
    torch.testing.assert_close(actual, expected)
    assert bool((actual >= bounds[0]).all()) and bool((actual <= bounds[1]).all())


def test_flex_process_heads_preserve_registry_order_bounds_and_gradients() -> None:
    for model_name in ("flexb", "flexi", "flexis"):
        spec = get_spec(model_name)
        assert spec.parameter_groups is not None
        legacy = CatchmentParameterizer(4, spec.dimension, hidden_dims=[8], parameter_names=list(spec.parameter_names))
        multi = CatchmentParameterizer(
            4,
            spec.dimension,
            hidden_dims=[8],
            parameter_names=list(spec.parameter_names),
            parameter_groups=spec.parameter_groups,
            architecture="process_heads",
            param_bounds=(spec.bounds[:, 0].float(), spec.bounds[:, 1].float()),
        )
        attributes = torch.randn(3, 4)
        output, diagnostics = multi(attributes, return_diagnostics=True)
        assert output.shape == (3, spec.dimension)
        assert tuple(diagnostics["parameter_names"]) == spec.parameter_names
        assert bool((output >= spec.bounds[:, 0].float()).all())
        assert bool((output <= spec.bounds[:, 1].float()).all())
        output.sum().backward()
        for name, parameter in multi.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

        # Both architectures remain independently constructible and the
        # canonical names/order are supplied by the real registry.
        assert legacy.parameter_names == multi.parameter_names == spec.parameter_names


def test_process_head_state_dict_round_trip_and_architecture_mismatch_is_strict() -> None:
    spec = get_spec("flexis")
    kwargs = dict(
        in_features=3,
        out_features=spec.dimension,
        hidden_dims=[6],
        parameter_names=list(spec.parameter_names),
        parameter_groups=spec.parameter_groups,
        architecture="process_heads",
    )
    original = CatchmentParameterizer(**kwargs)
    restored = CatchmentParameterizer(**kwargs)
    restored.load_state_dict(original.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        CatchmentParameterizer(
            3,
            spec.dimension,
            hidden_dims=[6],
            parameter_names=list(spec.parameter_names),
        ).load_state_dict(original.state_dict(), strict=True)


def test_parameterizer_rejects_invalid_bounds_and_duplicate_groups() -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        CatchmentParameterizer(2, 2, param_bounds=(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])))
    with pytest.raises(ValueError, match="duplicate parameter"):
        CatchmentParameterizer(
            2,
            2,
            architecture="process_heads",
            parameter_names=["a", "b"],
            parameter_groups={"bad": ("a", "a"), "other": ("b",)},
        )
