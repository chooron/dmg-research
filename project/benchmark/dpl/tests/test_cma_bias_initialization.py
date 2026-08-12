import torch
import pytest

from dpl.nn_parameterizer import CatchmentParameterizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CMA bias contract is verified on CUDA")
def test_cma_theta_initializes_output_zero_logit() -> None:
    theta = torch.tensor([0.12, 0.5, 0.91], device="cuda")
    parameterizer = CatchmentParameterizer(
        in_features=2,
        out_features=3,
        hidden_dims=[4, 4],
        initial_theta=theta,
    ).cuda()

    recovered = torch.sigmoid(parameterizer.net[-1].bias)
    torch.testing.assert_close(recovered, theta, rtol=0.0, atol=1e-6)
