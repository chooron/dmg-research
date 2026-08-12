"""Regression coverage for the MOPEX production wrapper's daily time step."""
from __future__ import annotations

import pytest
import torch

from dmotpy.models.hydrology_model import HydrologyModel


@pytest.mark.parametrize("model_name", ["mopex1", "mopex2", "mopex3", "mopex4", "mopex5"])
def test_mopex_wrapper_matches_explicit_one_day_raw_step(model_name: str) -> None:
    config = {
        "model_name": model_name,
        "warm_up": 0,
        "warm_up_states": True,
        "variables": ["prcp", "tmean", "pet"],
        "nearzero": 1e-6,
        "parameter_mapping": "linear",
        "backend": "eager",
    }
    model = HydrologyModel(config, device=torch.device("cpu"), backend="eager")
    # A cold snowfall followed by warm days catches a delta_t/nearzero swap.
    forcing = torch.tensor([
        [[10.0, -5.0, 2.0, 20.0]],
        [[0.0, 5.0, 2.0, 21.0]],
        [[0.0, 5.0, 2.0, 22.0]],
        [[0.0, 5.0, 2.0, 23.0]],
    ])
    if model_name not in {"mopex4", "mopex5"}:
        forcing = forcing[..., :3]
    raw = torch.full((1, len(model.phy_param_names), 1), 0.5)
    wrapped_q = model({"x_phy": forcing}, (None, raw))["streamflow"].squeeze()

    params = model._descale_params(model.unpack_parameters((None, raw)))
    param_values = [params[name] for name in model.phy_param_names]
    states = model._init_states(n_grid=1, n_groups=1)
    expected = []
    for t in range(forcing.shape[0]):
        args = [forcing[t, :, 0:1], forcing[t, :, 1:2], forcing[t, :, 2:3], *param_values, *states]
        kwargs = {"delta_t": 1.0, "nearzero": model.nearzero}
        if model_name in {"mopex4", "mopex5"}:
            kwargs["doy"] = forcing[t, :, 3:4]
        outputs = model.raw_step_fn(*args, **kwargs)
        expected.append(outputs[0])
        states = tuple(outputs[2:])
    expected_q = torch.stack(expected).squeeze()
    torch.testing.assert_close(wrapped_q, expected_q, rtol=1e-6, atol=1e-6)
    if model_name != "mopex1":
        assert float(wrapped_q[1]) > 1.0
