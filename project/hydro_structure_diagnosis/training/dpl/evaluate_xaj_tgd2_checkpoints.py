#!/usr/bin/env python3
"""Evaluate paused/completed Lite-v2-protocol XAJ-TGD2 dPL checkpoints.

This is evaluation only: it does not update weights, checkpoints, or training
history.  It reports calibration and held-out evaluation KGE for every basin
and seed, using exactly the forcing windows and 365-day warm-up path of the
dPL trainer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from training.dpl.run_dpl_model import (  # noqa: E402
    LITE_MODEL_REGISTRY,
    StaticParameterNet,
    evaluate,
    gate_time_index,
    latest_checkpoint,
    load_data,
    robust_normalize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Model root containing seed_<seed> directories.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: list[pd.DataFrame] = []
    checkpoint_rows: list[dict] = []
    for seed in args.seeds:
        seed_dir = args.root / f"seed_{seed}"
        config = json.loads((seed_dir / "config.json").read_text())
        data_root = PROJECT.parents[1] / "data"
        config["gage_ids_path"] = str(data_root / "gage_id.npy")
        config["dates_path"] = str(data_root / "camels_dates.npy")
        model_name = config["model_name"]
        model_cls, specs = LITE_MODEL_REGISTRY[model_name]
        indices = gate_time_index(config)
        basin_ids, raw_attrs, train_fc, train_obs, eval_fc, eval_obs = load_data(
            config, indices, max_basins=None
        )
        attrs, _ = robust_normalize(raw_attrs)
        attributes = torch.from_numpy(attrs)
        net_cfg = config["network"]
        hidden_sizes = [int(v) for v in net_cfg.get(
            "hidden_sizes", [net_cfg["hidden_size"]] * net_cfg.get("depth", 2)
        )]
        net = StaticParameterNet(
            attributes.shape[1], specs, hidden_sizes,
            net_cfg["dropout"], net_cfg["output_epsilon"],
        ).to(device)
        checkpoint_path = seed_dir / "best_checkpoint.pt"
        if not checkpoint_path.exists():
            checkpoint_path = latest_checkpoint(seed_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint for seed {seed}: {seed_dir}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        warmup = int(config["window"]["warmup_days"])
        batch_size = int(config["training"]["batch_size"])
        train_kge, _, _ = evaluate(
            net, model_cls, specs, attributes, train_fc, train_obs, batch_size, device, warmup
        )
        eval_kge, _, _ = evaluate(
            net, model_cls, specs, attributes, eval_fc, eval_obs, batch_size, device, warmup
        )
        all_rows.append(pd.DataFrame({
            "basin_id": basin_ids,
            "seed": seed,
            "checkpoint": checkpoint_path.name,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "train_kge": train_kge,
            "validation_kge": eval_kge,
        }))
        checkpoint_rows.append({
            "seed": seed,
            "checkpoint": checkpoint_path.name,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_val_kge_median": float(checkpoint.get("val_kge_median", np.nan)),
            "train_kge_median": float(np.nanmedian(train_kge)),
            "validation_kge_median": float(np.nanmedian(eval_kge)),
            "validation_kge_mean": float(np.nanmean(eval_kge)),
        })
        print(json.dumps(checkpoint_rows[-1]), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.concat(all_rows, ignore_index=True)
    table.to_csv(args.output / "per_seed_kge.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(args.output / "per_seed_summary.csv", index=False)
    median = table.groupby("basin_id", as_index=False)[["train_kge", "validation_kge"]].median()
    median.to_csv(args.output / "median_of_3_per_basin.csv", index=False)
    (args.output / "manifest.json").write_text(json.dumps({
        "purpose": "evaluation_only_paused_xaj_tgd2_dpl_checkpoints",
        "seeds": args.seeds,
        "model": "XAJ_TGD2",
        "selection": "best validation checkpoint available at pause; no optimization performed",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
