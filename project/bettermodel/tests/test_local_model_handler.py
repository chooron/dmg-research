from __future__ import annotations

import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from project.bettermodel.local_model_handler import LocalModelHandler
from project.bettermodel.model_builder import build_nn_model, build_phy_model


class TestLocalModelHandlerTrim(unittest.TestCase):
    def test_trim_aligns_target_to_prediction_length_without_warmup_recrop(self) -> None:
        handler = LocalModelHandler.__new__(LocalModelHandler)
        handler.config = {"model": {"warm_up": 365, "warmup": 365}}

        output = torch.randn(365, 4, 1)
        target = torch.randn(365, 4, 1)

        trimmed_output, trimmed_target = handler._trim(output, target)

        self.assertEqual(trimmed_output.shape, (365, 4))
        self.assertEqual(trimmed_target.shape, (365, 4))

    def test_trim_uses_tail_alignment_when_target_is_longer(self) -> None:
        handler = LocalModelHandler.__new__(LocalModelHandler)
        handler.config = {"model": {"warm_up": 365, "warmup": 365}}

        output = torch.randn(365, 4, 1)
        target = torch.randn(730, 4, 1)

        trimmed_output, trimmed_target = handler._trim(output, target)

        self.assertEqual(trimmed_output.shape, (365, 4))
        self.assertEqual(trimmed_target.shape, (365, 4))
        self.assertTrue(torch.equal(trimmed_target, target[-365:, :, 0]))


class TestBettermodelModelBuilder(unittest.TestCase):
    def _make_config(self, model_dir: str) -> dict:
        return {
            "device": "cpu",
            "mode": "train",
            "model_dir": model_dir,
            "train": {
                "target": ["streamflow"],
                "start_epoch": 0,
            },
            "model": {
                "rho": 30,
                "warm_up": 30,
                "phy": {
                    "name": ["Hbv_2"],
                    "dynamic_params": {"Hbv_2": ["parBETA", "parK0", "parCFMAX"]},
                    "nmul": 1,
                    "variables": ["prcp", "tmean", "pet"],
                    "routing": True,
                },
                "nn": {
                    "name": "LstmMlpModel",
                    "forcings": ["prcp", "tmean", "pet"],
                    "attributes": ["area", "elev"],
                    "lstm_hidden_size": 8,
                    "lstm_dropout": 0.0,
                    "mlp_hidden_size": 6,
                    "mlp_dropout": 0.0,
                },
            },
        }

    def test_model_builder_uses_direct_imports_without_hydrodl2_shim(self) -> None:
        sys.modules.pop("dmg.models.hydrodl2", None)

        with TemporaryDirectory() as model_dir:
            config = self._make_config(model_dir)
            with patch("project.bettermodel.implements.phy_models.hbv_2.torch.compile", side_effect=lambda fn, **_: fn):
                phy_model = build_phy_model(config, "Hbv_2", device="cpu")
                nn_model = build_nn_model(config, phy_model, device="cpu")

        self.assertEqual(type(phy_model).__module__, "project.bettermodel.implements.phy_models.hbv_2")
        self.assertEqual(type(nn_model).__module__, "project.bettermodel.implements.neural_networks.lstm_mlp")
        self.assertNotIn("dmg.models.hydrodl2", sys.modules)

    def test_handler_initializes_models_via_builder_module(self) -> None:
        with TemporaryDirectory() as model_dir:
            config = self._make_config(model_dir)
            with patch("project.bettermodel.implements.phy_models.hbv_2.torch.compile", side_effect=lambda fn, **_: fn):
                handler = LocalModelHandler(config, device="cpu")

        self.assertIn("Hbv_2", handler.model_dict)
        self.assertEqual(
            type(handler.model_dict["Hbv_2"].phy_model).__module__,
            "project.bettermodel.implements.phy_models.hbv_2",
        )
        self.assertEqual(
            type(handler.model_dict["Hbv_2"].nn_model).__module__,
            "project.bettermodel.implements.neural_networks.lstm_mlp",
        )


if __name__ == "__main__":
    unittest.main()
