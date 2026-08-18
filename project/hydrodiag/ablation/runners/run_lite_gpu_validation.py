from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation.ic_core.config import environment_snapshot, load_resolved_config
from ablation.ic_core.data_adapter import load_531_bundle
from ablation.ic_core.model_adapter import model_variant_inventory
from ablation.ic_core.parameter_adapter import get_parameter_spec
from ablation.ic_core.result_io import atomic_write_json, atomic_write_text
from ablation.ic_core.runtime import ICObjectiveRuntime


def _candidate_matrix(model_key: str, population: int) -> np.ndarray:
    dimension = len(get_parameter_spec(model_key))
    offsets = np.linspace(-0.12, 0.12, population, dtype=np.float64)
    direction = np.linspace(-1.0, 1.0, dimension, dtype=np.float64)
    theta = np.full((population, dimension), 0.5, dtype=np.float64)
    theta += offsets[:, None] * direction[None, :]
    return np.clip(theta, 0.05, 0.95)


def _gpu_memory() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _record_failure(
    model_key: str, lite_class: str, started: float, exc: Exception
) -> dict[str, object]:
    return {
        "status": "fail",
        "model_key": model_key,
        "model_variant": "lite",
        "lite_class": lite_class,
        "error": repr(exc),
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small GPU-only validation for all IC Lite model variants."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "ablation/configs/ic_foundation_531_v1.json",
    )
    parser.add_argument("--basin-index", type=int, default=0)
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    # Keep the host-side orchestration deliberately small. Model forward and KGE
    # are placed on CUDA by ICObjectiveRuntime; no full dataset copy is made.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable; this validation is intentionally GPU-only"
        )
    if args.population < 1:
        raise SystemExit("--population must be positive")

    config = load_resolved_config(args.config, device_override="cuda")
    base_output = Path(config["output_root"])
    output_root = base_output / "lite_gpu_validation"
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    models = [row["model_key"] for row in model_variant_inventory()]
    if args.models:
        unknown = sorted(set(args.models) - set(models))
        if unknown:
            raise SystemExit(f"unknown model keys: {unknown}")
        models = args.models

    bundle = load_531_bundle(config)
    if not 0 <= args.basin_index < len(bundle.basin_ids):
        raise SystemExit(f"basin index out of range: {args.basin_index}")

    resolved = dict(config)
    resolved["output_root"] = str(output_root)
    resolved["model_variant"] = "lite"
    resolved["validation"] = {
        "purpose": "start validation only; no optimizer generations",
        "basin_index": args.basin_index,
        "basin_id": bundle.basin_ids[args.basin_index],
        "population": args.population,
        "candidate_evaluations_per_model": args.population,
        "cpu_threads": 1,
        "device": "cuda",
    }
    atomic_write_json(output_root / "resolved_config.json", resolved)
    atomic_write_json(output_root / "environment.json", environment_snapshot(config))
    atomic_write_json(
        output_root / "lite_model_inventory.json",
        {
            "model_variant": "lite",
            "models": model_variant_inventory(),
        },
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_memory = _gpu_memory()
    rows: list[dict[str, object]] = []
    for model_key in models:
        started = time.perf_counter()
        lite_class = next(
            row["lite_class"]
            for row in model_variant_inventory()
            if row["model_key"] == model_key
        )
        try:
            theta = _candidate_matrix(model_key, args.population)
            runtime = ICObjectiveRuntime(
                bundle, config, model_key, model_variant="lite"
            )
            torch.cuda.synchronize()
            gpu_started = time.perf_counter()
            evaluation = runtime.evaluate_candidates(
                theta,
                basin_indices=[args.basin_index],
                split="train",
            )
            torch.cuda.synchronize()
            gpu_seconds = time.perf_counter() - gpu_started
            row = {
                "status": "pass" if np.isfinite(evaluation.fitness).all() else "fail",
                "model_key": model_key,
                "model_variant": "lite",
                "lite_class": lite_class,
                "basin_index": args.basin_index,
                "basin_id": bundle.basin_ids[args.basin_index],
                "population": args.population,
                "candidate_evaluations": evaluation.candidate_evaluations,
                "fitness_shape": list(evaluation.fitness.shape),
                "fitness": evaluation.fitness.tolist(),
                "valid_count": evaluation.valid_count.tolist(),
                "forcing_shape_transferred": list(evaluation.forcing_shape),
                "q_shape_after_warmup": list(evaluation.q_shape),
                "forward_dtype": "float32",
                "metric_dtype": evaluation.metric_dtype,
                "device": str(runtime.device),
                "runtime_seconds": time.perf_counter() - started,
                "gpu_timed_seconds": gpu_seconds,
                "gpu_memory": _gpu_memory(),
            }
        except Exception as exc:
            row = _record_failure(model_key, str(lite_class), started, exc)
        rows.append(row)
        atomic_write_json(output_root / f"{model_key}_result.json", row)

    after_memory = _gpu_memory()
    atomic_write_json(
        output_root / "lite_gpu_validation.json",
        {
            "status": "pass"
            if all(row["status"] == "pass" for row in rows)
            else "fail",
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "basin_index": args.basin_index,
            "basin_id": bundle.basin_ids[args.basin_index],
            "population": args.population,
            "models_requested": models,
            "models": rows,
            "gpu_memory_before": before_memory,
            "gpu_memory_after": after_memory,
            "note": "Each model used one 531 basin and a small candidate batch; no optimizer generation was started.",
        },
    )
    fieldnames = [
        "model_key",
        "status",
        "lite_class",
        "candidate_evaluations",
        "runtime_seconds",
        "gpu_timed_seconds",
        "error",
    ]
    with (output_root / "lite_gpu_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# Lite GPU validation report",
        "",
        f"- status: {'PASS' if all(row['status'] == 'pass' for row in rows) else 'FAIL'}",
        f"- device: {torch.cuda.get_device_name(0)}",
        f"- basin: {bundle.basin_ids[args.basin_index]} (index {args.basin_index})",
        f"- population: {args.population}",
        "- split: train only; test metrics were not used",
        "- CPU policy: torch threads=1; only the selected basin/candidates were transferred to GPU",
        "",
        "| Model | Lite class | Status | GPU seconds | Fitness |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_key']} | {row['lite_class']} | {row['status']} | "
            f"{row.get('gpu_timed_seconds', '')} | {row.get('fitness', row.get('error', ''))} |"
        )
    atomic_write_text(
        output_root / "lite_gpu_validation_report.md", "\n".join(lines) + "\n"
    )
    atomic_write_text(
        output_root / "logs" / "validation.log",
        "\n".join(
            [
                "command: python -m ablation.runners.run_lite_gpu_validation",
                f"device: {torch.cuda.get_device_name(0)}",
                "cpu_threads: 1",
                f"basin: {bundle.basin_ids[args.basin_index]}",
                f"population: {args.population}",
                f"models: {','.join(models)}",
                f"status: {'PASS' if all(row['status'] == 'pass' for row in rows) else 'FAIL'}",
            ]
        )
        + "\n",
    )
    if not all(row["status"] == "pass" for row in rows):
        raise SystemExit("one or more Lite GPU validations failed")


if __name__ == "__main__":
    main()
