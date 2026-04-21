from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from project.parameterize.analysis.common import normalize_parameters_to_unit_interval
from project.parameterize.analysis.correlation_analysis import build_correlation_long
from project.parameterize.analysis.parameter_analysis import (
    compute_cross_loss_parameter_variance,
    compute_seed_parameter_variance,
)


class TestAnalysisModules(unittest.TestCase):
    def setUp(self) -> None:
        self.parameter_bounds = {
            "parFC": (50.0, 1000.0),
            "parBETA": (1.0, 6.0),
        }
        self.params_long = pd.DataFrame(
            {
                "model": ["distributional"] * 8,
                "loss": ["NseBatchLoss"] * 4 + ["HybridNseBatchLoss"] * 4,
                "seed": [111, 222, 111, 222, 111, 222, 111, 222],
                "sample_count": [100] * 8,
                "basin_id": [1, 1, 2, 2, 1, 1, 2, 2],
                "parameter": ["parFC", "parFC", "parFC", "parFC", "parBETA", "parBETA", "parBETA", "parBETA"],
                "mean": [100.0, 200.0, 300.0, 500.0, 1.5, 2.5, 3.0, 4.0],
                "std": [0.1] * 8,
            }
        )
        self.attributes = pd.DataFrame(
            {
                "basin_id": [1, 2],
                "aridity": [0.2, 0.8],
                "slope_mean": [1.0, 4.0],
            }
        )

    def test_normalize_parameters_to_unit_interval_uses_bounds(self) -> None:
        normalized = normalize_parameters_to_unit_interval(self.params_long, self.parameter_bounds, value_column="mean")
        fc_values = normalized.loc[normalized["parameter"] == "parFC", "mean_unit"].to_numpy()
        expected = np.asarray([(100 - 50) / 950, (200 - 50) / 950, (300 - 50) / 950, (500 - 50) / 950])
        np.testing.assert_allclose(fc_values, expected)

    def test_compute_seed_parameter_variance_outputs_absdiff_and_variance(self) -> None:
        seed_variance = compute_seed_parameter_variance(self.params_long, self.parameter_bounds)
        basin1_fc = seed_variance.loc[
            (seed_variance["basin_id"] == 1)
            & (seed_variance["parameter"] == "parFC")
            & (seed_variance["loss"] == "NseBatchLoss")
        ].iloc[0]
        normalized = np.asarray([(100 - 50) / 950, (200 - 50) / 950])
        self.assertAlmostEqual(float(basin1_fc["variance_unit"]), float(np.var(normalized, ddof=0)))
        self.assertAlmostEqual(float(basin1_fc["mean_abs_seed_diff"]), float(abs(normalized[0] - normalized[1])))

    def test_compute_cross_loss_parameter_variance_supports_pooled_mode(self) -> None:
        pooled = compute_cross_loss_parameter_variance(self.params_long, self.parameter_bounds, mode="pooled")
        self.assertIn("pooled_loss_variance_unit", pooled.columns)
        self.assertIn("pooled_loss_mean_abs_diff", pooled.columns)

    def test_build_correlation_long_supports_multiple_methods(self) -> None:
        corr_tables = build_correlation_long(self.params_long, self.attributes, methods=("spearman", "pearson"))
        self.assertEqual(set(corr_tables), {"spearman", "pearson"})
        spearman = corr_tables["spearman"]
        self.assertIn("corr", spearman.columns)
        self.assertIn("p_value", spearman.columns)
        self.assertGreater(len(spearman), 0)


if __name__ == "__main__":
    unittest.main()

