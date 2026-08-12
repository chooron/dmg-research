#!/usr/bin/env python
"""Evaluate one completed dPL seed on full calibration and test periods."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch._inductor.config as inductor_config


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "project" / "hydrodiag"
sys.path.insert(0, str(PROJECT_DIR))

from training.dpl.run_dpl_model import (  # noqa: E402
    LITE_MODEL_REGISTRY,
    StaticParameterNet,
    evaluate,
    gate_time_index,
    load_data,
    robust_normalize,
)
from models.parameter_specs import TGD_STRUCTURE_VERSION  # noqa: E402


MODELS = (
    "XAJ", "XAJ_CN", "XAJ_TGD", "GR4J", "GR4J_CN", "GR4J_TGD",
    "HBV", "SIMHYD", "SIMHYD_CN", "SIMHYD_TGD",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "results" / "dpl_camels_531_lite_v2",
    )
    parser.add_argument("--compile-threads", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.compile_threads < 1 or args.torch_threads < 1:
        parser.error("thread counts must be positive")

    inductor_config.compile_threads = args.compile_threads
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_paths = {
        "gage_ids_path": str(REPO_ROOT / "data" / "gage_id.npy"),
        "dates_path": str(REPO_ROOT / "data" / "camels_dates.npy"),
        "data_pkl_dataset": str(REPO_ROOT / "data" / "camels_dataset"),
        "data_basin_ids": str(REPO_ROOT / "data" / "531sub_id.txt"),
    }
    rows: list[dict[str, float | int | str]] = []
    print(
        f"Evaluating seed={args.seed} device={device} "
        f"torch_threads={args.torch_threads} compile_threads={args.compile_threads}",
        flush=True,
    )

    for model_name in MODELS:
        output_dir = args.output_root / model_name / f"seed_{args.seed}"
        checkpoint_path = output_dir / "best_checkpoint.pt"
        if not (output_dir / "COMPLETE").exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"Incomplete result: {output_dir}")
        config = json.loads((output_dir / "config.json").read_text())
        if model_name.endswith("_TGD") and config.get("tgd_structure_version") != TGD_STRUCTURE_VERSION:
            raise RuntimeError(
                f"{output_dir} has incompatible TGD structure version "
                f"{config.get('tgd_structure_version')!r}; expected {TGD_STRUCTURE_VERSION!r}"
            )
        config.update(data_paths)
        indices = gate_time_index(config)
        basin_ids, raw_attrs, train_forcing, train_obs, test_forcing, test_obs = load_data(
            config, indices, None
        )
        attrs = torch.from_numpy(robust_normalize(raw_attrs)[0])
        model_cls, specs = LITE_MODEL_REGISTRY[model_name]
        hidden_sizes = [int(value) for value in config["network"]["hidden_sizes"]]
        net = StaticParameterNet(
            attrs.shape[1], specs, hidden_sizes,
            config["network"]["dropout"], config["network"]["output_epsilon"],
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if model_name.endswith("_TGD"):
            checkpoint_version = checkpoint.get("tgd_structure_version")
            if checkpoint_version != TGD_STRUCTURE_VERSION:
                raise RuntimeError(
                    f"{checkpoint_path} has incompatible TGD structure version "
                    f"{checkpoint_version!r}; expected {TGD_STRUCTURE_VERSION!r}"
                )
        expected_parameter_names = list(specs)
        if checkpoint.get("parameter_names") not in (None, expected_parameter_names):
            raise RuntimeError(
                f"{checkpoint_path} parameter names do not match {expected_parameter_names}"
            )
        net.load_state_dict(checkpoint["state_dict"])
        started = time.time()
        evaluate_args = (
            net, model_cls, specs, attrs, int(config["training"]["batch_size"]),
            device, int(config["window"]["warmup_days"]),
        )
        train_kge, _, _ = evaluate(*evaluate_args[:4], train_forcing, train_obs, *evaluate_args[4:])
        test_kge, _, _ = evaluate(*evaluate_args[:4], test_forcing, test_obs, *evaluate_args[4:])
        train_finite = np.isfinite(train_kge)
        test_finite = np.isfinite(test_kge)
        row: dict[str, float | int | str] = {
            "model": model_name,
            "seed": args.seed,
            "best_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "train_kge_mean": float(np.nanmean(train_kge)),
            "train_kge_median": float(np.nanmedian(train_kge)),
            "test_kge_mean": float(np.nanmean(test_kge)),
            "test_kge_median": float(np.nanmedian(test_kge)),
            "train_finite_basins": int(train_finite.sum()),
            "test_finite_basins": int(test_finite.sum()),
            "elapsed_s": time.time() - started,
        }
        rows.append(row)
        with (output_dir / "train_test_kge_by_basin.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["basin_id", "train_kge", "test_kge"])
            writer.writeheader()
            for basin_id, train_value, test_value in zip(basin_ids, train_kge, test_kge):
                writer.writerow({
                    "basin_id": basin_id,
                    "train_kge": f"{train_value:.10f}",
                    "test_kge": f"{test_value:.10f}",
                })
        print(
            f"{model_name}: train={row['train_kge_mean']:.6f} "
            f"test={row['test_kge_mean']:.6f} "
            f"finite={row['train_finite_basins']}/{row['test_finite_basins']}",
            flush=True,
        )
        del net, attrs, train_forcing, train_obs, test_forcing, test_obs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = args.output_root / f"seed_{args.seed}_train_test_kge_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
