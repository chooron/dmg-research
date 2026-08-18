import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append("/home/jingxin/code/dmg-research/project/hydrodiag")

from ablation.ic_core.data_adapter import load_531_bundle
from ablation.ic_core.runtime import ICObjectiveRuntime
from ablation.optimizers.xnes import XNESAdapter

base_dir = Path("/home/jingxin/code/dmg-research/project/hydrodiag")
audit_out_dir = base_dir / "outputs/ic_ablation/stage1_screening/v1/xnes_audit"
config_path = (
    base_dir / "outputs/ic_ablation/stage1_screening/v1/xnes/resolved_config.json"
)


def get_config():
    with open(config_path) as f:
        config = json.load(f)
    config["device"] = "cpu"
    config["dataset_path"] = str(base_dir.parent.parent / "data/camels_dataset")
    config["gage_ids_path"] = str(base_dir.parent.parent / "data/gage_id.npy")
    config["dates_path"] = str(base_dir.parent.parent / "data/camels_dates.npy")
    config["basin_list_path"] = str(base_dir.parent.parent / "data/531sub_id.txt")
    return config


def run_step9_fresh_rerun():
    print("Step 9: Fresh Rerun")
    config = get_config()
    bundle = load_531_bundle(config)

    out_dir = audit_out_dir / "fresh_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_basins = bundle.basin_ids[:3]  # just take first 3 basins

    for basin_id in target_basins:
        print(f"  Fresh rerun for {basin_id}")
        b_idx = bundle.basin_ids.index(basin_id)

        runtime = ICObjectiveRuntime(bundle, config, "XAJ")

        opt = XNESAdapter(
            dimension=15,  # XAJ has 15 params
            population_size=48,
            center_init=np.full(15, 0.5, dtype=np.float32),
            stdev_init=0.25,
            seed=42,
            bounds=(0.0, 1.0),
        )

        log_file = out_dir / f"{basin_id}_fresh_run.log"
        with open(log_file, "w") as lf:
            for gen in range(20):
                cands = opt.ask()
                cands_tensor = torch.tensor(cands, device=runtime.device).unsqueeze(0)
                evals = runtime.evaluate_candidates(
                    cands_tensor, basin_indices=[b_idx], split="train"
                )

                fits = evals.fitness[0].cpu().numpy()
                lf.write(f"Gen {gen} fitnesses: {fits.tolist()}\n")

                opt.tell(cands, fits)
                lf.write(f"Gen {gen} best fitness so far: {opt.best_fitness}\n")


if __name__ == "__main__":
    run_step9_fresh_rerun()
    print("Done audit_step9.py")
