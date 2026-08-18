from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")

from ablation.controlled_optimizer_ablation.experiment_matrix import (
    generate_phase1_matrix,
)
from ablation.controlled_optimizer_ablation.runner import run_tasks


def run_large_scale_ablation():
    output_root = "/autodl-fs/data/dmg_hydro_structure_diagnosis/outputs/ic_ablation"
    run_subdir = "large_scale_screening/v1"

    # Check remote data path
    if os.path.exists("/autodl-fs/data/531sub_id.txt"):
        dataset_root = "/autodl-fs/data"
    else:
        dataset_root = "/root/dmg-research/data"

    dataset_path = os.path.join(dataset_root, "camels_dataset")
    gage_ids_path = os.path.join(dataset_root, "gage_id.npy")
    dates_path = os.path.join(dataset_root, "camels_dates.npy")
    basin_list_path = os.path.join(dataset_root, "531sub_id.txt")

    # Models and dimension-coupled population grids
    model_configs = [
        {
            "model_key": "GR4J",
            "populations": [8, 16, 24, 32],  # D=4, P = 2D, 4D, 6D, 8D
        },
        {
            "model_key": "SIMHYD",
            "populations": [20, 40, 60, 80],  # D=10, P = 2D, 4D, 6D, 8D
        },
        {
            "model_key": "XAJ",
            "populations": [30, 60, 90, 120],  # D=15, P = 2D, 4D, 6D, 8D
        },
    ]

    optimizers = ["XNES", "CMAES"]
    seeds = [101, 202, 303]
    starts = 3
    generations = 200
    stdev_init = 0.25
    max_concurrent_groups = (
        6  # BUMPED TO 6 CONCURRENT GPU WORKER GROUPS FOR OPTIMAL CONCURRENCY & SAFETY!
    )

    # Load base foundation config
    foundation_cfg_path = os.path.join(
        "ablation", "configs", "ic_foundation_531_v1.json"
    )
    with open(foundation_cfg_path, "r") as f:
        base_config = json.load(f)

    base_config.update(
        {
            "dataset_path": dataset_path,
            "gage_ids_path": gage_ids_path,
            "dates_path": dates_path,
            "basin_list_path": basin_list_path,
            "split": "A",
            "n_basins": 32,
            "optimizers": optimizers,
            "starts": starts,
            "optimizer_seeds": seeds,
            "generations": generations,
            "stdev_init": stdev_init,
            "compute_test_metric": False,
            "output_root": output_root,
            "run_subdir": run_subdir,
            "device": "cuda",
            "max_concurrent_groups": max_concurrent_groups,
            "periods": {
                "warmup": {"start": "1988-01-01", "end": "1988-12-31"},
                "train": {"start": "1989-01-01", "end": "1998-12-31"},
                "test": None,
            },
        }
    )

    all_tasks = []
    print("=== Generating Large-Scale Multi-Model Task Matrix ===")
    for m_cfg in model_configs:
        m_key = m_cfg["model_key"]
        pops = m_cfg["populations"]
        cfg = dict(base_config)
        cfg["model_key"] = m_key
        cfg["populations"] = pops

        tasks = generate_phase1_matrix(cfg)
        print(
            f"  Model {m_key:<10}: {len(tasks)} tasks generated across populations {pops}"
        )
        all_tasks.extend(tasks)

    print(f"\nTotal tasks across all 3 models: {len(all_tasks)}")
    print(f"Output directory: {os.path.join(output_root, run_subdir)}")
    print(f"GPU Worker Threads: {max_concurrent_groups} concurrent groups on CUDA\n")

    # Save master config
    out_dir = os.path.join(output_root, run_subdir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "master_config.json"), "w") as f:
        json.dump(base_config, f, indent=2)

    # Execute all tasks using GPU concurrent worker pool
    t0 = time.time()
    run_tasks(all_tasks, base_config)
    elapsed = time.time() - t0
    print(f"\n=== Large-Scale Phase 1 Completed in {elapsed / 3600:.2f} hours ===")

    # Write COMPLETION marker for watcher
    with open(os.path.join(out_dir, "LARGE_SCALE_PHASE1_COMPLETED.txt"), "w") as f:
        f.write(f"Completed at {time.ctime()} in {elapsed / 3600:.2f} hours\n")


if __name__ == "__main__":
    run_large_scale_ablation()
