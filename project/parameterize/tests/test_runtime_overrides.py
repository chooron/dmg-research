from __future__ import annotations

import unittest
from types import SimpleNamespace

from project.parameterize.runtime_overrides import apply_runtime_overrides


class TestRuntimeOverrides(unittest.TestCase):
    def test_apply_device_and_gpu_id_override(self) -> None:
        raw_config = {
            "mode": "train_test",
            "device": "cpu",
            "gpu_id": 0,
            "paper": {"variant": "mc_dropout", "split": "main", "seeds": [42, 123]},
            "train": {"epochs": 100},
            "test": {"mc_samples": 10},
        }
        args = SimpleNamespace(
            variant=None,
            split=None,
            mode="train",
            device="cuda",
            gpu_id=2,
            seed=None,
            seeds=None,
            mc_samples=None,
            epochs=None,
        )

        apply_runtime_overrides(raw_config, args)

        self.assertEqual(raw_config["mode"], "train")
        self.assertEqual(raw_config["device"], "cuda")
        self.assertEqual(raw_config["gpu_id"], 2)


if __name__ == "__main__":
    unittest.main()
