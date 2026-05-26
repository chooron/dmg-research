from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.parameterize.manuscript.plots import common
from project.parameterize.manuscript.plots.fig01_predictive_performance import draw as draw_fig01
from project.parameterize.manuscript.plots.fig02_cross_seed_parameter_stability import draw as draw_fig02


class TestPublicationPlotsSplit(unittest.TestCase):
    def test_common_paths_point_to_expected_workspace_locations(self) -> None:
        self.assertEqual(common.MANUSCRIPT_ROOT, Path("/workspace/autoresearch/project/parameterize/manuscript"))
        self.assertEqual(common.PLOTS_ROOT, Path("/workspace/autoresearch/project/parameterize/manuscript/plots"))

    def test_fig01_and_fig02_can_render_into_temp_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with mock.patch.object(common, "MAIN_FIG_DIR", tmp_path), mock.patch(
                "project.parameterize.manuscript.plots.fig01_predictive_performance.MAIN_FIG_DIR", tmp_path
            ), mock.patch(
                "project.parameterize.manuscript.plots.fig02_cross_seed_parameter_stability.MAIN_FIG_DIR", tmp_path
            ):
                common.setup_style()
                draw_fig01()
                draw_fig02()
            self.assertTrue((tmp_path / "Fig01_predictive_performance.png").exists())
            self.assertFalse((tmp_path / "Fig01_predictive_performance.pdf").exists())
            self.assertTrue((tmp_path / "Fig02_parameter_stability_boundary_interval.png").exists())


if __name__ == "__main__":
    unittest.main()
