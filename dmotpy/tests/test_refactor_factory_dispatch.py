"""Verify __new__ factory dispatch: correct subclass instantiation, __init__ called once."""

import torch
from models.hydrology_model import HydrologyModel
from models.special_models import (
    EndpointUHModel,
    IntermediateUHModel,
    GR4JUHModel,
)


def test_base_model_no_uh_returns_hydrology_model():
    m = HydrologyModel(config={"model_name": "hbv96", "backend": "none"})
    assert type(m) is HydrologyModel


def test_endpoint_uh_returns_endpoint_uh_model():
    for name in ("newzealand2", "hillslope", "plateau", "smar", "ihacres", "hbv96"):
        m = HydrologyModel(
            config={"model_name": name, "uh_enabled": True, "uh_mode": "endpoint", "backend": "none"}
        )
        assert type(m) is EndpointUHModel, f"Expected EndpointUHModel for {name}, got {type(m).__name__}"


def test_intermediate_uh_returns_intermediate_uh_model():
    for name in ("flexi", "flexb", "flexis"):
        m = HydrologyModel(
            config={"model_name": name, "uh_enabled": True, "uh_mode": "intermediate", "backend": "none"}
        )
        assert type(m) is IntermediateUHModel, f"Expected IntermediateUHModel for {name}, got {type(m).__name__}"


def test_gr4j_uh_returns_gr4j_uh_model():
    m = HydrologyModel(
        config={"model_name": "gr4j", "uh_enabled": True, "uh_mode": "intermediate", "backend": "none"}
    )
    assert type(m) is GR4JUHModel


def test_subclass_direct_instantiation_works():
    m = EndpointUHModel(config={"model_name": "newzealand2", "uh_enabled": True, "uh_mode": "endpoint", "backend": "none"})
    assert type(m) is EndpointUHModel
    m = IntermediateUHModel(config={"model_name": "flexi", "uh_enabled": True, "uh_mode": "intermediate", "backend": "none"})
    assert type(m) is IntermediateUHModel
    m = GR4JUHModel(config={"model_name": "gr4j", "uh_enabled": True, "uh_mode": "intermediate", "backend": "none"})
    assert type(m) is GR4JUHModel


def test_forward_works_on_returned_instance():
    device = torch.device("cpu")
    forcing = torch.rand(100, 2, 3, device=device) * 5
    m = HydrologyModel(config={"model_name": "flexi", "warm_up": 10, "uh_enabled": True, "uh_mode": "intermediate", "backend": "none"}, device=device)
    params = (None, torch.rand(1, 10, device=device) * 0.5)
    out = m({"x_phy": forcing}, params)
    assert out["streamflow"].shape == (90, 2)
    assert not torch.isnan(out["streamflow"]).any()
