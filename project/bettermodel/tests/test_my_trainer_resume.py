from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from project.bettermodel.implements.my_trainer import MyTrainer


def _write_train_state(directory: str, epoch: int) -> None:
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": {"epoch": epoch},
            "scheduler_state_dict": {"epoch": epoch},
            "random_state": torch.get_rng_state(),
        },
        Path(directory, f"train_state_epoch_{epoch}.pt"),
    )


class TestMyTrainerResume(unittest.TestCase):
    def _make_trainer(self, model_dir: str, epochs: int) -> MyTrainer:
        trainer = MyTrainer.__new__(MyTrainer)
        trainer.config = {
            "device": "cpu",
            "model_dir": model_dir,
            "model_path": model_dir,
            "train": {"epochs": epochs, "start_epoch": 0},
            "test": {"test_epoch": epochs},
            "delta_model": {"phy_model": {"model": ["Hbv_2"]}},
        }
        trainer.epochs = epochs
        trainer.model = MagicMock()
        trainer.optimizer = MagicMock()
        trainer.scheduler = MagicMock()
        trainer.is_in_train = False
        return trainer

    def test_load_states_uses_latest_checkpoint_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "hbv_2_ep5.pt").touch()
            Path(tmp_dir, "hbv_2_ep10.pt").touch()
            _write_train_state(tmp_dir, 5)
            _write_train_state(tmp_dir, 10)

            trainer = self._make_trainer(tmp_dir, epochs=20)
            trainer.load_states()

            self.assertEqual(trainer.start_epoch, 11)
            self.assertEqual(trainer.config["train"]["start_epoch"], 10)
            trainer.model.load_model.assert_called_once_with(epoch=10)
            trainer.optimizer.load_state_dict.assert_called_once()
            trainer.scheduler.load_state_dict.assert_called_once_with({"epoch": 10})

    def test_load_states_starts_from_scratch_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self._make_trainer(tmp_dir, epochs=20)
            trainer.load_states()

            self.assertEqual(trainer.start_epoch, 1)
            trainer.model.load_model.assert_not_called()
            trainer.optimizer.load_state_dict.assert_not_called()

    def test_train_skips_when_checkpoint_already_reached_target_epoch(self) -> None:
        trainer = self._make_trainer("/tmp/unused", epochs=10)
        trainer.start_epoch = 11

        with patch("project.bettermodel.implements.my_trainer.create_training_grid") as grid:
            trainer.train()

        self.assertTrue(trainer.is_in_train)
        grid.assert_not_called()


if __name__ == "__main__":
    unittest.main()
