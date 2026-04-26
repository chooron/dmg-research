from __future__ import annotations

from argparse import Namespace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from project.bettermodel import load_config
from project.bettermodel.run_experiment import apply_runtime_overrides


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

    def test_runtime_seed_override_updates_resolved_output_paths(self) -> None:
        config_path = PROJECT_DIR / "conf" / "config_dhbv_hopev1.yaml"

        with patch("project.bettermodel.initialize_config", side_effect=lambda config: config):
            config = load_config(str(config_path))

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs" / "dhbv_hopev1" / "camels_671" / "seed_42"
            config["output_dir"] = str(output_dir) + "/"
            config["model_dir"] = str(output_dir / "model") + "/"
            config["plot_dir"] = str(output_dir / "plot") + "/"
            config["sim_dir"] = str(output_dir / "sim") + "/"
            config["save_path"] = config["output_dir"]
            config["model_path"] = config["model_dir"]
            config["out_path"] = config["sim_dir"]

            apply_runtime_overrides(
                config,
                Namespace(
                    mode="train_test",
                    seed=111,
                    test_epoch=100,
                    start_epoch=None,
                    epochs=None,
                    loss=None,
                ),
            )

            self.assertEqual(config["seed"], 111)
            self.assertEqual(config["random_seed"], 111)
            for key in ("output_dir", "model_dir", "plot_dir", "sim_dir", "save_path", "model_path", "out_path"):
                self.assertIn("seed_111", config[key], key)
                self.assertNotIn("seed_42", config[key], key)
                self.assertTrue(Path(config[key]).is_dir(), key)


if __name__ == "__main__":
    unittest.main()
