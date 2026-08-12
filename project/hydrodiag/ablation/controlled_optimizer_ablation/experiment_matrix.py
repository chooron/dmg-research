import json
import os
import numpy as np
from typing import List
from scipy.stats import qmc
from .protocol import Phase1Task, SEEDS

PARAMETER_DIMENSIONS = {
    "GR4J": 4,
    "SIMHYD": 10,
    "XAJ": 15,
    "GR4J_CN": 6,
    "SIMHYD_CN": 12,
    "XAJ_CN": 17,
}

def generate_lhs_center_for_basin_model(basin_id: str, model_key: str, start_idx: int, dim: int) -> List[float]:
    """Deterministically generate LHS center in [0, 1]^dim for a given (basin_id, model_key, start_idx)."""
    seed_str = f"{basin_id}_{model_key}_{start_idx}"
    seed = int(abs(hash(seed_str)) % (2**31 - 1))
    engine = qmc.LatinHypercube(d=dim, seed=seed)
    sample = engine.random(n=1)[0]
    return sample.tolist()

def generate_phase1_matrix(config: dict) -> List[Phase1Task]:
    tasks = []
    split = config["split"]
    model_key = config["model_key"]
    dim = PARAMETER_DIMENSIONS.get(model_key, 15)
    
    # Load basins
    manifest_path = os.path.join("ablation", "manifests", "ic_ablation_96_basins_v1.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    # Filter basins by split and n_basins
    basins = [b["basin_id"] for b in manifest["basins"] if b["split"] == split]
    if len(basins) < config["n_basins"]:
        raise ValueError(f"Found only {len(basins)} basins for split {split}, but {config['n_basins']} requested.")
    basins = basins[:config["n_basins"]]
    
    # Check if LHS centers exist in npz
    starts_path = os.path.join("ablation", "manifests", "ic_stage1_lhs_centers_v1.npz")
    has_npz_centers = False
    if os.path.exists(starts_path):
        starts_npz = np.load(starts_path)
        lhs_basin_ids = starts_npz["basin_ids"].tolist()
        lhs_model_keys = starts_npz["model_keys"].tolist()
        if model_key in lhs_model_keys:
            has_npz_centers = True
            starts_data = starts_npz["centers"]
            m_idx_lhs = lhs_model_keys.index(model_key)
    
    output_root = config["output_root"]
    run_subdir = config["run_subdir"]
    
    for basin_id in basins:
        for opt in config["optimizers"]:
            for pop in config["populations"]:
                for start_idx in range(config["starts"]):
                    if has_npz_centers and basin_id in lhs_basin_ids:
                        b_idx_lhs = lhs_basin_ids.index(basin_id)
                        center = starts_data[b_idx_lhs, m_idx_lhs, start_idx].tolist()
                    else:
                        center = generate_lhs_center_for_basin_model(basin_id, model_key, start_idx, dim)
                        
                    for seed in config["optimizer_seeds"]:
                        out_dir = os.path.join(output_root, run_subdir, "tasks", model_key, opt, f"pop_{pop}", basin_id, f"seed_{seed}", f"start_{start_idx}")
                        tasks.append(Phase1Task(
                            basin_id=basin_id,
                            optimizer_name=opt,
                            seed=seed,
                            start_idx=start_idx,
                            population=pop,
                            generations=config["generations"],
                            stdev_init=config["stdev_init"],
                            output_dir=out_dir,
                            model_key=model_key,
                            center_init=center,
                            compute_test_metric=config["compute_test_metric"]
                        ))
    
    return tasks
