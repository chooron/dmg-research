from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch

from project.bettermodel.implements.neural_networks.ablation import (
    S4DBaseline,
    S4DLN,
    S4DLNSoftsign,
    S4DSoftsign,
    S5DConvOnly,
    S5DFull,
)
from project.bettermodel.implements.neural_networks.ablation.diagnostics import (
    save_dynamic_parameter_diagnostics,
)


class TestS5DAblationModels(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "nx": 5,
            "nx1": 5,
            "nx2": 2,
            "ny1": 6,
            "ny2": 4,
            "nmul": 2,
            "hope_hidden_size": 8,
            "mlp_hidden_size": 6,
            "hope_dropout": 0.0,
            "mlp_dropout": 0.0,
        }

    def _data(self) -> dict:
        return {
            "xc_nn_norm": torch.randn(12, 3, 5),
            "c_nn_norm": torch.randn(3, 2),
        }

    def test_all_variants_return_normalized_parameter_outputs(self) -> None:
        for cls in (
            S4DBaseline,
            S4DLN,
            S4DSoftsign,
            S4DLNSoftsign,
            S5DConvOnly,
            S5DFull,
        ):
            with self.subTest(model=cls.__name__):
                model = cls.build_by_config(self._config(), device="cpu")
                dynamic_out, static_out = model(self._data())

                self.assertEqual(dynamic_out.shape, (12, 3, 6))
                self.assertEqual(static_out.shape, (3, 4))
                self.assertTrue(torch.all(dynamic_out >= 0.0))
                self.assertTrue(torch.all(dynamic_out <= 1.0))
                self.assertTrue(torch.all(static_out >= 0.0))
                self.assertTrue(torch.all(static_out <= 1.0))

                params = model.predict_timevar_parameters(self._data()["xc_nn_norm"])
                self.assertEqual(params.shape, (12, 3, 3, 2))

    def test_diagnostics_save_trajectories_and_stats(self) -> None:
        params = np.linspace(0.0, 1.0, num=24, dtype=np.float32).reshape(4, 1, 3, 2)

        with TemporaryDirectory() as tmpdir:
            stats = save_dynamic_parameter_diagnostics(
                tmpdir,
                params,
                variant="s4d_baseline",
                parameter_names=["parBETA", "parK0", "parBETAET"],
            )

        self.assertEqual(stats["shape"], [4, 1, 3, 2])
        self.assertIn("median_abs_day_to_day_change", stats)
        self.assertIn("either_boundary_ratio", stats)


if __name__ == "__main__":
    unittest.main()
