from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from project.bettermodel.local_model_handler import PHY_MODELS_DIR, _load_component


class TestHbv2Gradients(unittest.TestCase):
    def test_hbv2_streamflow_retains_gradient(self) -> None:
        config = {
            "warm_up": 0,
            "warm_up_states": False,
            "dynamic_params": {"Hbv_2": ["parBETA", "parK0", "parBETAET"]},
            "variables": ["prcp", "tmean", "pet"],
            "routing": False,
            "comprout": False,
            "nearzero": 1e-5,
            "nmul": 1,
        }

        with patch("torch.compile", lambda fn, *args, **kwargs: fn):
            hbv2_cls = _load_component(PHY_MODELS_DIR, "Hbv_2")
            model = hbv2_cls(config, device=torch.device("cpu"))

        dy_params = torch.randn(5, 2, 3, requires_grad=True)
        static_params = torch.randn(2, 13, requires_grad=True)
        forcings = torch.rand(5, 2, 3)

        output = model({"x_phy": forcings}, (dy_params, static_params))
        loss = output["streamflow"].sum()

        self.assertTrue(loss.requires_grad)

        loss.backward()

        self.assertIsNotNone(dy_params.grad)
        self.assertIsNotNone(static_params.grad)


if __name__ == "__main__":
    unittest.main()
