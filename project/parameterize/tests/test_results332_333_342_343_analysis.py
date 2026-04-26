from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from project.parameterize.analysis.results332_333_342_343_analysis import (
    _build_correlation_long_value,
    _run_id,
)


class TestResults332333342343Analysis(unittest.TestCase):
    def test_build_correlation_long_value_works_for_std_column(self) -> None:
        params = pd.DataFrame(
            {
                "basin_id": [1, 2, 3, 4],
                "model": ["distributional"] * 4,
                "loss": ["NseBatchLoss"] * 4,
                "seed": [111] * 4,
                "parameter": ["parFC"] * 4,
                "mean": [1.0, 2.0, 3.0, 4.0],
                "std": [0.4, 0.3, 0.2, 0.1],
            }
        )
        attrs = pd.DataFrame({"basin_id": [1, 2, 3, 4], "aridity": [1.0, 2.0, 3.0, 4.0]})
        outputs = _build_correlation_long_value(params, attrs, value_column="std", methods=("spearman",))
        table = outputs["spearman"]
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(float(table.loc[0, "corr"]), -1.0)

    def test_run_id_builds_expected_string(self) -> None:
        frame = pd.DataFrame(
            {
                "model": ["distributional"],
                "loss": ["HybridNseBatchLoss"],
                "seed": [333],
            }
        )
        self.assertEqual(_run_id(frame).iloc[0], "distributional__HybridNseBatchLoss__seed_333")


if __name__ == "__main__":
    unittest.main()
