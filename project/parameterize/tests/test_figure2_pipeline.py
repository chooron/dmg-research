from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from project.parameterize.figure2.src.api import build_registry, generate_all_figures, generate_figure


CONFIG_PATH = "/workspace/autoresearch/project/parameterize/conf/config_param_paper.yaml"
ANALYSIS_ROOT = "/workspace/autoresearch/project/parameterize/outputs/analysis/stability_stats"


class TestFigure2Pipeline(unittest.TestCase):
    def test_registry_parses_subset_ids_and_spatial_join(self) -> None:
        registry = build_registry(CONFIG_PATH, ANALYSIS_ROOT)
        self.assertEqual(len(registry.subset_basin_ids), 531)
        self.assertIsNotNone(registry.spatial)
        self.assertEqual(len(registry.spatial), 531)
        self.assertEqual(registry.reference_loss, "HybridNseBatchLoss")

    def test_registry_loads_core_tables(self) -> None:
        registry = build_registry(CONFIG_PATH, ANALYSIS_ROOT)
        for table_name in (
            "metrics_long",
            "params_long",
            "basin_attributes",
            "relationship_classes",
            "results332_matrix_similarity",
            "results341_distributional_mean_relationships",
            "results342_distributional_std_relationships",
        ):
            frame = registry.table(table_name)
            self.assertGreater(len(frame), 0, table_name)

    def test_generate_single_figure_writes_outputs_and_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = generate_figure(
                figure_number=1,
                config_path=CONFIG_PATH,
                analysis_root=ANALYSIS_ROOT,
                output_dir=tmp_dir,
            )
            self.assertTrue(Path(tmp_dir, "Fig01_performance_merged_revised.png").exists())
            self.assertTrue(Path(tmp_dir, "Fig01_performance_merged_revised.pdf").exists())
            self.assertTrue(Path(payload["manifest"]).exists())
            self.assertTrue(Path(payload["qc_report"]).exists())

    def test_generate_selected_figures_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = generate_all_figures(
                config_path=CONFIG_PATH,
                analysis_root=ANALYSIS_ROOT,
                output_dir=tmp_dir,
                figure_numbers=[1, 3, 5],
            )
            expected = {
                "01": "Fig01_performance_merged_revised",
                "03": "Fig03_cross_seed_corr_stability_revised",
                "05": "Fig05_shared_dominant_core_revised",
            }
            for figure_id, stem in expected.items():
                self.assertIn(figure_id, payload["generated"])
                self.assertTrue(Path(tmp_dir, f"{stem}.png").exists())
                self.assertTrue(Path(tmp_dir, f"{stem}.pdf").exists())
            self.assertTrue(Path(tmp_dir, "manifest.json").exists())
            self.assertTrue(Path(tmp_dir, "qc_report.md").exists())


if __name__ == "__main__":
    unittest.main()
