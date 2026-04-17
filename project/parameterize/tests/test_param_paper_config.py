from __future__ import annotations

import copy
import unittest

from project.parameterize.paper_variants import normalize_paper_config


def _base_raw_config() -> dict:
    return {
        "seed": 42,
        "paper": {"variant": "mc_dropout", "split": "main", "seeds": [42, 123]},
        "test": {"mc_dropout": True, "mc_samples": 10},
        "data": {
            "basin_ids_path": "./data/531sub_id.txt",
            "basin_ids_reference_path": "./data/gage_id.npy",
        },
        "model": {
            "phy": {"nmul": 1},
            "nn": {
                "attributes": ["attr_a", "attr_b"],
                "forcings": [],
                "output_activation": "sigmoid",
            },
        },
    }


class TestParamPaperConfig(unittest.TestCase):
    def test_normalize_distributional_variant_targets_main_split_runtime(self) -> None:
        raw_config = copy.deepcopy(_base_raw_config())
        raw_config["paper"]["variant"] = "distributional"

        normalize_paper_config(raw_config)

        self.assertEqual(raw_config["trainer"], "ParamLearnTrainer")
        self.assertEqual(raw_config["model"]["nn"]["name"], "DistributionalParamModel")
        self.assertFalse(raw_config["test"]["mc_dropout"])
        self.assertEqual(raw_config["test"]["mc_samples"], 1)
        self.assertEqual(raw_config["paper"]["seeds"], [42, 123])

    def test_normalize_mc_dropout_variant_preserves_mc_eval(self) -> None:
        raw_config = copy.deepcopy(_base_raw_config())

        normalize_paper_config(raw_config)

        self.assertEqual(raw_config["model"]["nn"]["name"], "McMlpModel")
        self.assertTrue(raw_config["test"]["mc_dropout"])
        self.assertEqual(raw_config["test"]["mc_samples"], 10)


if __name__ == "__main__":
    unittest.main()
