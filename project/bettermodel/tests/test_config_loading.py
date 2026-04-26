from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from project.bettermodel import load_config


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent


class TestBettermodelConfigLoading(unittest.TestCase):
    def test_loads_camels_531_and_671_observation_configs(self) -> None:
        configs = {
            "camels_531": PROJECT_DIR / "conf" / "config_dhbv_hopev2_531.yaml",
            "camels_671": PROJECT_DIR / "conf" / "config_dhbv_lstm.yaml",
        }

        with patch("project.bettermodel.initialize_config", side_effect=lambda config: config):
            for observation_name, config_path in configs.items():
                with self.subTest(observations=observation_name):
                    config = load_config(str(config_path))

                    self.assertEqual(config["observations"]["name"], observation_name)
                    self.assertEqual(
                        config["observations"]["data_path"],
                        str((REPO_ROOT / "data" / "camels_dataset").resolve()),
                    )
                    self.assertEqual(
                        config["observations"]["gage_info"],
                        str((REPO_ROOT / "data" / "gage_id.npy").resolve()),
                    )
                    self.assertTrue(Path(config["output_dir"]).is_absolute())
                    self.assertEqual(config["model"]["warmup"], config["model"]["warm_up"])

                    if observation_name == "camels_531":
                        self.assertEqual(
                            config["observations"]["subset_path"],
                            str((REPO_ROOT / "data" / "531sub_id.txt").resolve()),
                        )
                    else:
                        self.assertIsNone(config["observations"]["subset_path"])


if __name__ == "__main__":
    unittest.main()
