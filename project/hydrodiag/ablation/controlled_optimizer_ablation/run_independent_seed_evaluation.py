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

# Fixed optimal generations G* from Phase 3 per model
MODEL_OPTIMAL_GENS = {
    "GR4J": 200,
    "SIMHYD": 200,
    "XAJ": 300
}

def generate_independent_eval_tasks(output_root: str, basin_ids: list[str]) -> list[Phase1Task]:
    tasks = []
    models = ["GR4J", "SIMHYD", "XAJ"]
    optimizers = ["XNES", "CMAES"]
    # NEW UNSEEN INDEPENDENT SEEDS to eliminate cross-grid selection bias (Winner's Curse)
    independent_seeds = [404, 505, 606]
    starts = 3

    for model_key in models:
        pop = MODEL_OPTIMAL_POPS[model_key]
        stdev = MODEL_OPTIMAL_STDEV[model_key]
        gen = MODEL_OPTIMAL_GENS[model_key]
        dim = PARAMETER_DIMENSIONS[model_key]

        for opt_name in optimizers:
            for b_id in basin_ids:
                for seed in independent_seeds:
                    for s_idx in range(starts):
                        center = generate_lhs_center_for_basin_model(b_id, model_key, s_idx, dim)
                        t_dir = os.path.join(
                            output_root, model_key, opt_name, b_id, f"seed_{seed}", f"start_{s_idx}"
                        )
                        t = Phase1Task(
                            basin_id=b_id,
                            optimizer_name=opt_name,
                            seed=seed,
                            start_idx=s_idx,
                            population=pop,
                            generations=gen,
                            stdev_init=stdev,
                            output_dir=t_dir,
                            model_key=model_key,
                            center_init=center,
                            compute_test_metric=False
                        )
                        tasks.append(t)
    return tasks

def run_independent_eval():
    t0 = time.perf_counter()
    output_root = "/root/outputs/independent_seed_evaluation/v1/tasks"
    os.makedirs(output_root, exist_ok=True)
    
    # Load 32 basins from Split A
    basin_manifest_path = os.path.join("ablation", "manifests", "ic_ablation_96_basins_v1.json")
    with open(basin_manifest_path) as f:
        bm = json.load(f)
    split_a_basins = [b["basin_id"] for b in bm["basins"] if b.get("split") == "A"]
    
    all_tasks = generate_independent_eval_tasks(output_root, split_a_basins)
    
    print(f"=== Starting Independent Seed Rerun Evaluation (Seeds 404, 505, 606) ===", flush=True)
    print(f"Total tasks: {len(all_tasks)} (32 basins x 3 models x 2 optimizers x 3 seeds x 3 starts)", flush=True)
    print(f"Models & Locked Optimal Hyperparameters (P*, stdev*, G*):", flush=True)
    print(f"  GR4J: Pop=24, stdev_init=0.10, Gen=200", flush=True)
    print(f"  SIMHYD: Pop=40, stdev_init=0.05, Gen=200", flush=True)
    print(f"  XAJ: Pop=90, stdev_init=0.05, Gen=300", flush=True)
    
    foundation_cfg_path = os.path.join("ablation", "configs", "ic_foundation_531_v1.json")
    with open(foundation_cfg_path) as f:
        base_config = json.load(f)
        
    base_config["device"] = "cuda"
    base_config["max_concurrent_groups"] = 8  # Safe concurrency to guarantee 5GB VRAM safety buffer & 0 CUDA OOM!

    base_config["periods"] = {
        "warmup": {"start": "1988-01-01", "end": "1988-12-31"},
        "train": {"start": "1989-01-01", "end": "1998-12-31"}
    }

    if os.path.exists("/autodl-fs/data/531sub_id.txt"):
        base_config["basin_list_path"] = "/autodl-fs/data/531sub_id.txt"
        base_config["gage_ids_path"] = "/autodl-fs/data/gage_id.npy"
        base_config["dates_path"] = "/autodl-fs/data/camels_dates.npy"
        base_config["camels_dataset_dir"] = "/autodl-fs/data/camels_dataset"

    # Execute independent seed rerun evaluation tasks
    run_tasks(all_tasks, base_config)

    elapsed_hours = (time.perf_counter() - t0) / 3600
    print(f"\n=== Independent Seed Rerun Evaluation Completed in {elapsed_hours:.2f} hours ===", flush=True)
    
    completed_marker = os.path.join("/root/outputs/independent_seed_evaluation/v1", "INDEPENDENT_EVAL_COMPLETED.txt")
    os.makedirs(os.path.dirname(completed_marker), exist_ok=True)
    with open(completed_marker, "w") as f:
        f.write(f"Completed at {time.ctime()} in {elapsed_hours:.2f} hours")

if __name__ == "__main__":
    run_independent_eval()
