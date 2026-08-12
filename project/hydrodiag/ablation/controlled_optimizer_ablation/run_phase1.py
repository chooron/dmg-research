import json
import os
from ablation.controlled_optimizer_ablation.experiment_matrix import generate_phase1_matrix
from ablation.controlled_optimizer_ablation.runner import run_tasks
from ablation.controlled_optimizer_ablation.aggregation import compute_metrics, generate_report

with open("ablation/configs/controlled_optimizer_ablation/phase1_optimizer.json") as f:
    config = json.load(f)

tasks = generate_phase1_matrix(config)
print(f"Full run task count: {len(tasks)}")
run_tasks(tasks, config)

print("Computing metrics and generating reports...")
agg_df = compute_metrics(config)
report_dir = os.path.join(config["output_root"], config["run_subdir"])
generate_report(agg_df, report_dir)

print("Phase 1 execution complete.")
