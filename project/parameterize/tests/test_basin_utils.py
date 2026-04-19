from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from project.parameterize.implements.basin_utils import (
    basin_subset_indices,
    load_basin_ids,
    subset_dataset_by_basin_ids,
)


class TestBasinUtils(unittest.TestCase):
    def test_load_basin_ids_supports_python_list_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            basin_path = Path(tmp_dir) / "subset.txt"
            basin_path.write_text("[1022500, 1031500, 1047000]\n")
            basin_ids = load_basin_ids(basin_path)

        np.testing.assert_array_equal(basin_ids, np.array([1022500, 1031500, 1047000]))

    def test_repo_531_basin_file_parses(self) -> None:
        basin_ids = load_basin_ids(Path("/workspace/autoresearch/data/531sub_id.txt"))
        self.assertEqual(len(basin_ids), 531)

    def test_subset_indices_follow_explicit_basin_ids_not_prefix(self) -> None:
        reference = np.array([10, 20, 30, 40, 50])
        subset = np.array([50, 10, 30])
        indices = basin_subset_indices(reference, subset)
        np.testing.assert_array_equal(indices, np.array([4, 0, 2]))

    def test_subset_dataset_uses_mapped_basin_indices(self) -> None:
        reference = np.array([10, 20, 30, 40, 50])
        subset = np.array([50, 10, 30])
        dataset = {
            "xc_nn_norm": torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3),
            "c_nn": torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4),
        }

        subset_dataset, basin_ids = subset_dataset_by_basin_ids(dataset, reference, subset)

        np.testing.assert_array_equal(basin_ids, subset)
        self.assertEqual(tuple(subset_dataset["xc_nn_norm"].shape), (2, 3, 3))
        self.assertEqual(tuple(subset_dataset["c_nn"].shape), (3, 4))
        torch.testing.assert_close(subset_dataset["xc_nn_norm"][:, 0], dataset["xc_nn_norm"][:, 4])
        torch.testing.assert_close(subset_dataset["xc_nn_norm"][:, 1], dataset["xc_nn_norm"][:, 0])


if __name__ == "__main__":
    unittest.main()
