from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from project.parameterize.paper_variants import write_run_metadata


class TestParamPaperMultiseed(unittest.TestCase):
    def test_write_run_metadata_uses_fixed_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
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
            path = Path(write_run_metadata(config))

            self.assertEqual(path.name, "run_meta.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["paper_variant"], "distributional")
            self.assertEqual(payload["seed"], 123)
            self.assertEqual(payload["split"], "main")
            self.assertEqual(payload["nn_name"], "DistributionalParamModel")
            self.assertEqual(payload["loss_name"], "KgeBatchLoss")
            self.assertEqual(payload["mc_samples"], 1)
            self.assertEqual(payload["output_activation"], "sigmoid")
            self.assertEqual(payload["static_pool"], "last")
            self.assertEqual(payload["paper_seeds"], [123, 124])
            self.assertEqual(payload["data_basin_ids_path"], "./data/531sub_id.txt")


if __name__ == "__main__":
    unittest.main()
