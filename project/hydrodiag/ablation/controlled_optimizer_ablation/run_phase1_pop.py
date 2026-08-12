import json
from ablation.controlled_optimizer_ablation.experiment_matrix import generate_phase1_matrix
from ablation.controlled_optimizer_ablation.runner import run_tasks

def main():
    config_path = "ablation/configs/controlled_optimizer_ablation/phase1_population.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    tasks = generate_phase1_matrix(config)
    print(f"Generated {len(tasks)} tasks for Phase 1 Population Tuning.")
    
    run_tasks(tasks, config)

if __name__ == "__main__":
    main()
