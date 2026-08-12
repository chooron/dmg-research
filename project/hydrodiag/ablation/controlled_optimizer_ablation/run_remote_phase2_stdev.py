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

def generate_phase2_tasks(output_root: str, basin_ids: list[str]) -> list[tuple[Phase1Task, float]]:
    tasks = []
    models = ["GR4J", "SIMHYD", "XAJ"]
    optimizers = ["XNES", "CMAES"]
    stdev_candidates = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
    seeds = [101, 202, 303]
    starts = 3

    for model_key in models:
        pop = MODEL_OPTIMAL_POPS[model_key]
        dim = PARAMETER_DIMENSIONS[model_key]
        for opt_name in optimizers:
            for stdev in stdev_candidates:
                stdev_str = f"stdev_{stdev:.2f}"
                for b_id in basin_ids:
                    for seed in seeds:
                        for s_idx in range(starts):
                            center = generate_lhs_center_for_basin_model(b_id, model_key, s_idx, dim)
                            t_dir = os.path.join(
                                output_root, model_key, opt_name, stdev_str, b_id, f"seed_{seed}", f"start_{s_idx}"
                            )
                            t = Phase1Task(
                                basin_id=b_id,
                                optimizer_name=opt_name,
                                seed=seed,
                                start_idx=s_idx,
                                population=pop,
                                generations=200,
                                stdev_init=stdev,
                                output_dir=t_dir,
                                model_key=model_key,
                                center_init=center,
                                compute_test_metric=False
                            )
                            tasks.append((t, stdev))
    return tasks

def run_phase2_ablation():
    t0 = time.perf_counter()
    output_root = os.path.join("outputs", "ic_ablation", "phase2_stdev_screening", "v1", "tasks")
    os.makedirs(output_root, exist_ok=True)
    
    # Remove old premature completion marker if exists
    completed_marker = os.path.join("outputs", "ic_ablation", "phase2_stdev_screening", "v1", "PHASE2_STDEV_COMPLETED.txt")
    if os.path.exists(completed_marker):
        os.remove(completed_marker)

    # Load 32 basins from Split A
    basin_manifest_path = os.path.join("ablation", "manifests", "ic_ablation_96_basins_v1.json")
    with open(basin_manifest_path) as f:
        bm = json.load(f)
    split_a_basins = [b["basin_id"] for b in bm["basins"] if b.get("split") == "A"]
    
    tasks_with_stdev = generate_phase2_tasks(output_root, split_a_basins)
    all_tasks = [t for t, _ in tasks_with_stdev]
    
    print(f"=== Starting Phase 2 stdev_init Ablation ===", flush=True)
    print(f"Total Phase 2 tasks: {len(all_tasks)}", flush=True)
    print(f"Models: GR4J(Pop=24, dim=4), SIMHYD(Pop=40, dim=10), XAJ(Pop=90, dim=15)", flush=True)
    print(f"stdev_init candidates: [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]", flush=True)
    
    foundation_cfg_path = os.path.join("ablation", "configs", "ic_foundation_531_v1.json")
    with open(foundation_cfg_path) as f:
        base_config = json.load(f)
        
    base_config["generations"] = 200
    base_config["device"] = "cuda"
    base_config["max_concurrent_groups"] = 8  # 8 CONCURRENT GPU WORKERS FOR HIGH PARALLELISM!

    # Align periods protocol with Phase 1
    base_config["periods"] = {
        "warmup": {"start": "1988-01-01", "end": "1988-12-31"},
        "train": {"start": "1989-01-01", "end": "1998-12-31"}
    }

    # Update data paths for remote server if running on remote autodl-fs
    if os.path.exists("/autodl-fs/data/531sub_id.txt"):
        base_config["basin_list_path"] = "/autodl-fs/data/531sub_id.txt"
        base_config["gage_ids_path"] = "/autodl-fs/data/gage_id.npy"
        base_config["dates_path"] = "/autodl-fs/data/camels_dates.npy"
        base_config["camels_dataset_dir"] = "/autodl-fs/data/camels_dataset"

    # Execute via runner
    for stdev_val in [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]:
        stdev_tasks = [t for t, s in tasks_with_stdev if s == stdev_val]
        cfg = dict(base_config)
        cfg["stdev_init"] = stdev_val
        print(f"\n--- Running stdev_init = {stdev_val:.2f} ({len(stdev_tasks)} tasks) ---", flush=True)
        run_tasks(stdev_tasks, cfg)

    elapsed_hours = (time.perf_counter() - t0) / 3600
    print(f"\n=== Phase 2 stdev_init Ablation Completed in {elapsed_hours:.2f} hours ===", flush=True)
    
    with open(completed_marker, "w") as f:
        f.write(f"Completed at {time.ctime()} in {elapsed_hours:.2f} hours")

if __name__ == "__main__":
    run_phase2_ablation()
