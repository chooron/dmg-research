import copy
import json

from ablation.controlled_optimizer_ablation.experiment_matrix import (
    generate_phase1_matrix,
)
from ablation.controlled_optimizer_ablation.runner import run_tasks

with open("ablation/configs/controlled_optimizer_ablation/phase1_optimizer.json") as f:
    config = json.load(f)

short_config = copy.deepcopy(config)
short_config["n_basins"] = 1
short_config["generations"] = 5
short_config["starts"] = 2
short_config["run_subdir"] = "controlled_optimizer_ablation/v1/_validation_short_run"

tasks = generate_phase1_matrix(short_config)
print(f"Short run task count: {len(tasks)}")
run_tasks(tasks, short_config)
print("Gate 4: Short run complete.")
