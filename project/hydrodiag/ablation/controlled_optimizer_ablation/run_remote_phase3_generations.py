from __future__ import annotations

import os
import json
import time
from ablation.controlled_optimizer_ablation.protocol import Phase1Task
from ablation.controlled_optimizer_ablation.experiment_matrix import generate_lhs_center_for_basin_model, PARAMETER_DIMENSIONS
from ablation.controlled_optimizer_ablation.runner import run_tasks

# Fixed optimal populations P* from Phase 1 per model
MODEL_OPTIMAL_POPS = {
    "GR4J": 24,
    "SIMHYD": 40,
    "XAJ": 90
}

# Fixed optimal stdev_init* from Phase 2 per model
MODEL_OPTIMAL_STDEV = {
    "GR4J": 0.10,
    "SIMHYD": 0.05,
    "XAJ": 0.05
}

def generate_phase3_tasks(output_root: str, basin_ids: list[str]) -> list[Phase1Task]:
    tasks = []
    models = ["GR4J", "SIMHYD", "XAJ"]
    optimizers = ["XNES", "CMAES"]
    # Single 600-generation run per start! All intermediate convergence checkpoints (50, 100, 150, 200, 300, 400, 500, 600)
    # are recorded naturally in the generation trace, eliminating redundant 1..50 re-runs!
    max_gen = 600
    seeds = [101, 202, 303]
    starts = 3

    for model_key in models:
        pop = MODEL_OPTIMAL_POPS[model_key]
        stdev = MODEL_OPTIMAL_STDEV[model_key]
        dim = PARAMETER_DIMENSIONS[model_key]
        for opt_name in optimizers:
            for b_id in basin_ids:
                for seed in seeds:
                    for s_idx in range(starts):
                        center = generate_lhs_center_for_basin_model(b_id, model_key, s_idx, dim)
                        t_dir = os.path.join(
                            output_root, model_key, opt_name, f"gen_{max_gen}", b_id, f"seed_{seed}", f"start_{s_idx}"
                        )
                        t = Phase1Task(
                            basin_id=b_id,
                            optimizer_name=opt_name,
                            seed=seed,
                            start_idx=s_idx,
                            population=pop,
                            generations=max_gen,
                            stdev_init=stdev,
                            output_dir=t_dir,
                            model_key=model_key,
                            center_init=center,
                            compute_test_metric=False
                        )
                        tasks.append(t)
    return tasks

def run_phase3_ablation():
    t0 = time.perf_counter()
    output_root = os.path.join("outputs", "ic_ablation", "phase3_generations_screening", "v1", "tasks")
    os.makedirs(output_root, exist_ok=True)
    
    # Load 32 basins from Split A
    basin_manifest_path = os.path.join("ablation", "manifests", "ic_ablation_96_basins_v1.json")
    with open(basin_manifest_path) as f:
        bm = json.load(f)
    split_a_basins = [b["basin_id"] for b in bm["basins"] if b.get("split") == "A"]
    
    all_tasks = generate_phase3_tasks(output_root, split_a_basins)
    
    print(f"=== Starting Phase 3 Max-Generation (Gen=600) Convergence Trajectory Ablation ===", flush=True)
    print(f"Total Phase 3 tasks: {len(all_tasks)} (Single Gen=600 run per start, 0 redundant re-runs!)", flush=True)
    print(f"Models & Fixed Hyperparameters:", flush=True)
    print(f"  GR4J: Pop=24, stdev_init=0.10", flush=True)
    print(f"  SIMHYD: Pop=40, stdev_init=0.05", flush=True)
    print(f"  XAJ: Pop=90, stdev_init=0.05", flush=True)
    
    foundation_cfg_path = os.path.join("ablation", "configs", "ic_foundation_531_v1.json")
    with open(foundation_cfg_path) as f:
        base_config = json.load(f)
        
    base_config["device"] = "cuda"
    base_config["max_concurrent_groups"] = 12

    base_config["periods"] = {
        "warmup": {"start": "1988-01-01", "end": "1988-12-31"},
        "train": {"start": "1989-01-01", "end": "1998-12-31"}
    }

    if os.path.exists("/autodl-fs/data/531sub_id.txt"):
        base_config["basin_list_path"] = "/autodl-fs/data/531sub_id.txt"
        base_config["gage_ids_path"] = "/autodl-fs/data/gage_id.npy"
        base_config["dates_path"] = "/autodl-fs/data/camels_dates.npy"
        base_config["camels_dataset_dir"] = "/autodl-fs/data/camels_dataset"

    # Execute single Gen=600 run for all tasks
    run_tasks(all_tasks, base_config)

    elapsed_hours = (time.perf_counter() - t0) / 3600
    print(f"\n=== Phase 3 Generations Ablation Completed in {elapsed_hours:.2f} hours ===", flush=True)
    
    completed_marker = os.path.join("outputs", "ic_ablation", "phase3_generations_screening", "v1", "PHASE3_GENERATIONS_COMPLETED.txt")
    with open(completed_marker, "w") as f:
        f.write(f"Completed at {time.ctime()} in {elapsed_hours:.2f} hours")

if __name__ == "__main__":
    run_phase3_ablation()
