from __future__ import annotations

import unittest

import numpy as np

from project.parameterize.analysis.final_completeness_check_analysis import _partial_corr


class TestFinalCompletenessCheckAnalysis(unittest.TestCase):
    def test_partial_corr_removes_shared_linear_component(self) -> None:
        rng = np.random.default_rng(42)
        control = rng.normal(size=(100, 1))
        x = control[:, 0] + rng.normal(scale=0.1, size=100)
        y = control[:, 0] + rng.normal(scale=0.1, size=100)
        raw_corr = np.corrcoef(x, y)[0, 1]
        partial = _partial_corr(y, x, control)
        self.assertGreater(raw_corr, 0.8)
        self.assertLess(abs(partial), 0.3)


if __name__ == "__main__":
    unittest.main()
