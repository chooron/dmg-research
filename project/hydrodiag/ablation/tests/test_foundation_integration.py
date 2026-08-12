from pathlib import Path

import numpy as np

from ablation.ic_core.config import load_resolved_config
from ablation.ic_core.data_adapter import load_531_bundle
from ablation.ic_core.parameter_adapter import get_parameter_spec
from ablation.ic_core.runtime import ICObjectiveRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_resolved_config_records_effective_values() -> None:
    config = load_resolved_config(PROJECT_ROOT / "ablation/configs/ic_foundation_531_v1.json", device_override="cpu")
    config["periods"] = {"warmup": {"start": "1988-01-01", "end": "1988-12-31"}, "train": {"start": "1989-01-01", "end": "1998-12-31"}, "test": {"start": "1999-01-01", "end": "2010-09-30"}}
    assert config["basin_list_path"].endswith("531sub_id.txt")
    assert config["dataset_path"].endswith("camels_dataset")
    assert config["periods"]["train"]["start"] == "1989-01-01"
    assert config["periods"]["test"]["start"] == "1999-01-01"
    assert config["target_model_unit"] == "mm/day"
    assert config["output_root"].endswith("ic_ablation/foundation_v1")


def test_single_basin_candidate_shape_and_precision() -> None:
    config = load_resolved_config(PROJECT_ROOT / "ablation/configs/ic_foundation_531_v1.json", device_override="cpu")
    config["periods"] = {"warmup": {"start": "1988-01-01", "end": "1988-12-31"}, "train": {"start": "1989-01-01", "end": "1998-12-31"}, "test": {"start": "1999-01-01", "end": "2010-09-30"}}
    bundle = load_531_bundle(config)
    runtime = ICObjectiveRuntime(bundle, config, "XAJ")
    theta = np.full(len(get_parameter_spec("XAJ")), 0.5)
    result = runtime.evaluate_candidates(theta, basin_indices=[0], split="train")
    assert result.fitness.shape == (1, 1)
    assert result.q_shape == (1, 1, 3652)
    assert result.candidate_evaluations == 1
    assert result.metric_dtype == "float64"
