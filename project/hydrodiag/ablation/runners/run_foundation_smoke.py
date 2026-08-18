from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation.ic_core.config import environment_snapshot, load_resolved_config
from ablation.ic_core.checkpoint import CheckpointStore
from ablation.ic_core.data_adapter import load_531_bundle, manifest_for_bundle
from ablation.ic_core.parameter_adapter import get_parameter_spec
from ablation.ic_core.result_io import atomic_write_json, atomic_write_text
from ablation.ic_core.runtime import ICObjectiveRuntime
from ablation.ic_core.schemas import RESULT_FIELDS


def _candidate_matrix(model_key: str, population: int) -> np.ndarray:
    dimension = len(get_parameter_spec(model_key))
    offsets = np.linspace(-0.18, 0.18, population, dtype=np.float64)
    theta = np.full((population, dimension), 0.5, dtype=np.float64)
    theta += offsets[:, None] * np.linspace(-1.0, 1.0, dimension, dtype=np.float64)[None, :]
    return np.clip(theta, 0.05, 0.95)


def _record_evaluation(bundle, model_key: str, evaluation, basin_indices, theta_shape, device) -> dict[str, object]:
    return {
        "status": "pass",
        "optimizer": "none_smoke",
        "model_key": model_key,
        "basin_ids": [bundle.basin_ids[int(index)] for index in basin_indices],
        "basin_indices": [int(index) for index in basin_indices],
        "theta_shape": list(theta_shape),
        "fitness_shape": list(evaluation.fitness.shape),
        "fitness": evaluation.fitness.tolist(),
        "valid_shape": list(evaluation.valid.shape),
        "valid_count_min": int(evaluation.valid_count.min()),
        "candidate_evaluations": evaluation.candidate_evaluations,
        "split": evaluation.split,
        "forcing_shape_transferred": list(evaluation.forcing_shape),
        "q_shape_after_warmup": list(evaluation.q_shape),
        "forward_dtype": "float32",
        "metric_dtype": evaluation.metric_dtype,
        "device": str(device),
        "runtime_seconds": evaluation.runtime_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "ablation/configs/ic_foundation_531_v1.json")
    parser.add_argument("--device", default="cpu", help="smoke device; CPU is the default to avoid full GPU residency")
    args = parser.parse_args()
    config = load_resolved_config(args.config, device_override=args.device)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "resolved_config.json", config)
    atomic_write_json(output_root / "environment.json", environment_snapshot(config))
    bundle = load_531_bundle(config)
    manifest = manifest_for_bundle(bundle, config)
    atomic_write_json(output_root / "smoke_data.json", {
        "status": "pass",
        "n_basins": len(bundle.basin_ids),
        "n_unique_basins": len(set(bundle.basin_ids)),
        "no_559_fallback": True,
        "selected_shapes": manifest["selected_shapes"],
        "forcing_names": list(bundle.forcing_names),
        "periods": bundle.periods.as_dict(),
        "target_raw_unit": bundle.target_unit_raw,
        "target_model_unit": bundle.target_unit_ic,
        "area_field": bundle.area_field,
    })
    atomic_write_json(output_root / "smoke_result_schema.json", {
        "status": "pass",
        "fields": list(RESULT_FIELDS),
        "optimizer_for_smoke": "none_smoke",
        "budget_definition": "one basin/parameter vector completing one full train forward and one fitness",
        "test_metrics_used_for_selection": False,
    })

    single_runtime = ICObjectiveRuntime(bundle, config, "XAJ")
    single_theta = np.full(len(get_parameter_spec("XAJ")), 0.5, dtype=np.float64)
    single = single_runtime.evaluate_candidates(single_theta, basin_indices=[0], split="train")
    atomic_write_json(output_root / "smoke_single_basin.json", _record_evaluation(bundle, "XAJ", single, [0], single_theta.shape, args.device))

    multi_theta = _candidate_matrix("XAJ", 8)
    multi_candidate = single_runtime.evaluate_candidates(multi_theta, basin_indices=[0], split="train")
    atomic_write_json(output_root / "smoke_multi_candidate.json", _record_evaluation(bundle, "XAJ", multi_candidate, [0], multi_theta.shape, args.device))

    multi_basin_theta = np.stack([multi_theta for _ in range(4)], axis=0)
    multi_basin = single_runtime.evaluate_candidates(multi_basin_theta, basin_indices=[0, 1, 2, 3], split="train")
    atomic_write_json(output_root / "smoke_multi_basin.json", _record_evaluation(bundle, "XAJ", multi_basin, [0, 1, 2, 3], multi_basin_theta.shape, args.device))

    cross_model_records = []
    cross_models = [
        "GR4J", "GR4J_CN", "GR4J_TGD2",
        "SIMHYD", "SIMHYD_CN", "SIMHYD_TGD2",
        "XAJ", "XAJ_CN", "XAJ_TGD2",
        "HBV",
    ]
    for model_key in cross_models:
        started = time.perf_counter()
        try:
            runtime = ICObjectiveRuntime(bundle, config, model_key)
            theta = _candidate_matrix(model_key, 2)
            evaluation = runtime.evaluate_candidates(theta, basin_indices=[0], split="train")
            record = _record_evaluation(bundle, model_key, evaluation, [0], theta.shape, args.device)
            record["elapsed_wall_seconds"] = time.perf_counter() - started
        except Exception as exc:
            record = {
                "status": "fail",
                "model_key": model_key,
                "error": repr(exc),
                "elapsed_wall_seconds": time.perf_counter() - started,
            }
        cross_model_records.append(record)
    atomic_write_json(output_root / "smoke_cross_model.json", {
        "status": "pass" if all(row["status"] == "pass" for row in cross_model_records) else "fail",
        "models": cross_model_records,
        "purpose": "registry, parameter dimension, mapping, objective and output-shape smoke; no performance comparison",
    })
    checkpoint = CheckpointStore(output_root / "checkpoint_smoke")
    checkpoint.mark_complete({
        "run_id": "ic_foundation_531_v1_smoke",
        "resolved_config_hash": config["config_hash"],
        "basin_model_start_seed": {"basin": 0, "model": "XAJ", "start": "none", "seed": config["seed"]},
        "completed_evaluations": 1 + 8 + 32,
        "optimizer_state": None,
    })
    atomic_write_json(output_root / "smoke_checkpoint.json", {
        "status": "pass",
        "complete_marker": str(checkpoint.complete_path),
        "completed_evaluations": 41,
        "resume_semantics": "completed marker prevents duplicate smoke work; optimizer state is reserved, not required for none_smoke",
    })
    if any(row["status"] != "pass" for row in cross_model_records):
        raise SystemExit("one or more cross-model smoke tests failed")


if __name__ == "__main__":
    main()
