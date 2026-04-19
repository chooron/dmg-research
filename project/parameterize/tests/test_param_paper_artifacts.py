from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project.parameterize.implements.my_trainer import MyTrainer
from project.parameterize.paper_variants import write_run_metadata


class _FakeMetrics:
    def __init__(self) -> None:
        self.kge = np.asarray([0.4, 0.5], dtype=np.float32)
        self.nse = np.asarray([0.3, 0.4], dtype=np.float32)
        self.rmse = np.asarray([0.1, 0.2], dtype=np.float32)
        self.corr = np.asarray([0.8, 0.7], dtype=np.float32)
        self.mae = np.asarray([0.05, 0.06], dtype=np.float32)
        self.pbias_abs = np.asarray([1.0, 2.0], dtype=np.float32)

    def model_dump(self) -> dict:
        return {
            "kge": self.kge,
            "nse": self.nse,
            "rmse": self.rmse,
            "corr": self.corr,
            "mae": self.mae,
            "pbias_abs": self.pbias_abs,
            "pred": None,
            "target": None,
        }

    def calc_stats(self) -> dict:
        return {
            "kge": {"mean": 0.45, "median": 0.45, "std": 0.05},
            "nse": {"mean": 0.35, "median": 0.35, "std": 0.05},
            "rmse": {"mean": 0.15, "median": 0.15, "std": 0.05},
            "corr": {"mean": 0.75, "median": 0.75, "std": 0.05},
            "mae": {"mean": 0.055, "median": 0.055, "std": 0.005},
            "pbias_abs": {"mean": 1.5, "median": 1.5, "std": 0.5},
        }


class TestParamPaperArtifacts(unittest.TestCase):
    def test_artifact_contract_writers_emit_expected_files_and_keys(self) -> None:
        metrics = _FakeMetrics()
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = MyTrainer.__new__(MyTrainer)
            trainer.config = {
                "output_dir": tmp_dir,
                "seed": 123,
                "model": {"nn": {"name": "DistributionalParamModel"}},
            }
            trainer.eval_basin_ids = np.asarray([101, 102], dtype=np.int64)
            trainer.mc_samples = 3
            trainer.mc_selection_metric = "mse"

            trainer._write_metrics_files(metrics, prefix="metrics")
            trainer._write_metrics_files(metrics, prefix="metrics_avg")
            trainer._write_mc_metrics(
                raw_metrics=[{"pass_index": 0, "metrics": trainer._serialize_metrics(metrics)}],
                agg_metrics=[{"pass_index": 0, "agg_stats": trainer._jsonify(metrics.calc_stats())}],
            )
            trainer._write_selection_artifacts(
                best_pass_idx=np.asarray([0, 1]),
                best_scores=np.asarray([0.1, 0.2]),
            )
            trainer._save_eval_results(metrics)

            meta_path = Path(
                write_run_metadata(
                    {
                        "seed": 123,
                        "output_dir": tmp_dir,
                        "paper": {
                            "variant": "distributional",
                            "split": "main",
                            "seeds": [123, 124],
                        },
                        "model": {
                            "nn": {
                                "name": "DistributionalParamModel",
                                "output_activation": "sigmoid",
                                "static_pool": "last",
                            }
                        },
                        "train": {"loss_function": {"name": "KgeBatchLoss"}},
                        "test": {"mc_samples": 1},
                        "data": {"basin_ids_path": "./data/531sub_id.txt"},
                    }
                )
            )

            expected_files = {
                "metrics.json",
                "metrics_agg.json",
                "metrics_avg.json",
                "metrics_avg_agg.json",
                "metrics_mc.json",
                "metrics_mc_agg.json",
                "mc_selection.json",
                "results_seed123.csv",
                "run_meta.json",
            }
            self.assertTrue(expected_files.issubset({path.name for path in Path(tmp_dir).iterdir()}))

            results_frame = Path(tmp_dir, "results_seed123.csv").read_text(encoding="utf-8")
            self.assertIn("basin_id", results_frame)
            self.assertIn("kge", results_frame)
            self.assertIn("seed", results_frame)
            self.assertIn("nn_model", results_frame)

            metrics_agg = json.loads(Path(tmp_dir, "metrics_agg.json").read_text(encoding="utf-8"))
            for key in ("kge", "nse", "rmse", "corr", "mae", "pbias_abs"):
                self.assertIn(key, metrics_agg)

            metrics_avg_agg = json.loads(
                Path(tmp_dir, "metrics_avg_agg.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(metrics_avg_agg["kge"].keys()), {"mean", "median", "std"})

            mc_selection = json.loads(
                Path(tmp_dir, "mc_selection.json").read_text(encoding="utf-8")
            )
            for key in ("mc_samples", "selection_metric", "best_pass_index", "best_reference_score"):
                self.assertIn(key, mc_selection)

            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            for key in (
                "paper_variant",
                "seed",
                "split",
                "nn_name",
                "loss_name",
                "mc_samples",
                "output_activation",
                "static_pool",
                "paper_seeds",
                "data_basin_ids_path",
            ):
                self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
