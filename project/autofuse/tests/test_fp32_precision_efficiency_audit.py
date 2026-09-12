import torch

from dfuse import simulate_coupled_rk2_batched
from project.autofuse.dpl import DPLConfig, StructureConditionedParameterizer



def test_explicit_fp32_path_keeps_physics_and_loss_float32():
    dtype = torch.float32
    forcing = torch.tensor([[[5.0, 2.0, 10.0], [4.0, 2.0, 9.0], [3.0, 1.0, 8.0]]], dtype=dtype)
    attributes = torch.randn(1, 35, dtype=dtype)
    target = torch.ones((1, 2), dtype=dtype)
    model = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=16)).to(dtype=dtype)
    theta = model(attributes, 2)
    result = simulate_coupled_rk2_batched(2, forcing, theta, basin_ids=("a",), compile_step=False, output_mode="full")
    tensors = [result.q, result.q_instantaneous, result.states, result.water_balance_residual, result.snow, result.snow_balance_residual, *result.fluxes.values(), *result.sequential_diagnostics.values()]
    assert all(value.dtype == dtype for value in tensors)
    assert theta.dtype == dtype
    assert all(parameter.dtype == dtype for parameter in model.parameters())
    loss = torch.mean((result.q[:, 1:] - target) ** 2)
    assert loss.dtype == dtype
    loss.backward()
    assert all(parameter.grad is None or parameter.grad.dtype == dtype for parameter in model.parameters())


def test_float64_reference_path_remains_float64_without_default_dtype_change():
    dtype = torch.float64
    forcing = torch.tensor([[[5.0, 2.0, 10.0], [4.0, 2.0, 9.0]]], dtype=dtype)
    theta = torch.zeros((1, 37), dtype=dtype)
    result = simulate_coupled_rk2_batched(2, forcing, theta, basin_ids=("a",), compile_step=False, output_mode="lite")
    assert result.q.dtype == dtype
    assert result.states.dtype == dtype
    assert result.water_balance_residual.dtype == dtype
    assert result.snow_balance_residual.dtype == dtype
    assert torch.get_default_dtype() == torch.float32
