import json
from ablation.controlled_optimizer_ablation.experiment_matrix import generate_phase1_matrix

with open("ablation/configs/controlled_optimizer_ablation/phase1_optimizer.json") as f:
    config = json.load(f)

tasks = generate_phase1_matrix(config)
assert len(tasks) == 576, f"Expected 576 tasks, got {len(tasks)}"
evals_per_task = config["population"] * config["generations"]
print(f"Tasks: {len(tasks)}")
print(f"Evals/task: {evals_per_task}")
print("Gate 3: Dry-run Matrix Passed")
