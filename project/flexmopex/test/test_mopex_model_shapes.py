import torch

from project.flexmopex.models.fixed_weight_mopex import FixedWeightMopex
from project.flexmopex.models.learned_weight_mopex import LearnedWeightMopex
from project.flexmopex.models.parameter_nets import LearnedStructureNet, ParamRoutingNet
from project.flexmopex.models.static_mopex import StaticMopex


def _base_config(nmul=2):
    return {
        "nmul": nmul,
        "warm_up": 1,
        "warm_up_states": True,
        "variables": ["prcp", "tmean", "pet"],
        "nearzero": 1e-5,
        "disable_compile": True,
    }


def _x_dict(n_steps=5, batch_size=3):
    x_phy = torch.rand(n_steps, batch_size, 3)
    x_phy[:, :, 1] = x_phy[:, :, 1] * 20.0
    doy = torch.arange(1, n_steps + 1, dtype=torch.float32).view(n_steps, 1, 1)
    return {"x_phy": x_phy, "doy": doy.expand(-1, batch_size, -1)}


def _param_routing(batch_size=3, nmul=2):
    return {
        "params": torch.randn(batch_size, 12 * nmul),
        "gamma_uh": torch.randn(batch_size, 2),
    }


def test_param_routing_net_forward_keys_and_shapes():
    batch_size = 3
    nx2 = 7
    nmul = 2
    net = ParamRoutingNet(
        input_dim=nx2,
        hidden_dim=16,
        dropout=0.0,
        nmul=nmul,
        device="cpu",
    )
    out = net({"c_nn_norm": torch.randn(batch_size, nx2)})
    assert set(out) == {"params", "gamma_uh"}
    assert out["params"].shape == (batch_size, 12 * nmul)
    assert out["gamma_uh"].shape == (batch_size, 2)


def test_learned_structure_net_forward_keys_and_shapes():
    batch_size = 3
    nx2 = 7
    nmul = 2
    net = LearnedStructureNet(
        input_dim=nx2,
        hidden_dim=16,
        dropout=0.0,
        nmul=nmul,
        device="cpu",
    )
    out = net({"c_nn_norm": torch.randn(batch_size, nx2)})
    assert set(out) == {"params", "weights", "gamma_uh"}
    assert out["params"].shape == (batch_size, 12 * nmul)
    assert out["weights"].shape == (batch_size, 8)
    assert out["gamma_uh"].shape == (batch_size, 2)


def test_static_mopex_forward_outputs_streamflow_only():
    batch_size = 3
    nmul = 2
    model = StaticMopex(_base_config(nmul), device="cpu")
    out = model(_x_dict(batch_size=batch_size), _param_routing(batch_size, nmul))
    assert set(out) == {"streamflow"}
    assert out["streamflow"].shape == (4, batch_size, 1)


def test_fixed_weight_mopex_forward_outputs_config_weights():
    batch_size = 3
    nmul = 2
    config = _base_config(nmul)
    config["fixed_weights"] = {
        "w_phen": 0.0,
        "w_int": 1.0,
        "w_snow": 0.25,
        "w_sub": 0.75,
    }
    model = FixedWeightMopex(config, device="cpu")
    out = model(_x_dict(batch_size=batch_size), _param_routing(batch_size, nmul))
    assert set(out) == {"streamflow", "w_phen", "w_int", "w_snow", "w_sub"}
    assert out["streamflow"].shape == (4, batch_size, 1)
    for name, value in config["fixed_weights"].items():
        assert out[name].shape == (4, batch_size, 1)
        assert torch.allclose(out[name], torch.full_like(out[name], value))


def test_learned_weight_mopex_forward_outputs_probabilities():
    batch_size = 3
    nmul = 2
    config = _base_config(nmul)
    config["structure_tau"] = 1.0
    model = LearnedWeightMopex(config, device="cpu")
    model.eval()
    parameters = _param_routing(batch_size, nmul)
    parameters["weights"] = torch.randn(batch_size, 8)
    out = model(_x_dict(batch_size=batch_size), parameters)
    assert set(out) == {"streamflow", "w_phen", "w_int", "w_snow", "w_sub"}
    assert out["streamflow"].shape == (4, batch_size, 1)
    for name in ("w_phen", "w_int", "w_snow", "w_sub"):
        assert out[name].shape == (4, batch_size, 1)
        assert torch.all(out[name] >= 0.0)
        assert torch.all(out[name] <= 1.0)
