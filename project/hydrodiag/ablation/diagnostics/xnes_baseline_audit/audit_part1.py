import os
import json
import hashlib
import csv
import glob
import numpy as np

base_dir = '/home/jingxin/code/dmg-research/project/hydrodiag'
tasks_dir = os.path.join(base_dir, 'outputs/ic_ablation/stage1_screening/v1/xnes/tasks')
audit_out_dir = os.path.join(base_dir, 'outputs/ic_ablation/stage1_screening/v1/xnes_audit')
data_dir = os.path.join(base_dir, 'data', 'camels_subset')  # guess based on typical hydro setups, need to find actual data if different

def hash_file(filepath):
    if not os.path.exists(filepath): return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def step1_raw_precision_audit():
    print("Running step 1...")
    result_files = glob.glob(os.path.join(tasks_dir, '*', '*', '*', 'result.json'))
    out_csv = os.path.join(audit_out_dir, 'raw_precision_audit.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'basin_id', 'seed', 'result_json_hash', 'trace_jsonl_hash', 'params_json_hash'])
        for res_f in sorted(result_files):
            task_dir = os.path.dirname(res_f)
            parts = task_dir.split('/')
            seed_dir = parts[-1]
            model_basin = parts[-2]
            basin_id = model_basin.split('_')[1] if '_' in model_basin else model_basin # Assuming format model_basin
            
            trace_f = os.path.join(task_dir, 'trace.jsonl')
            params_f = os.path.join(task_dir, 'result.json')
            
            h_res = hash_file(res_f)
            h_trace = hash_file(trace_f)
            h_param = hash_file(params_f)
            writer.writerow([model_basin, basin_id, seed_dir, h_res, h_trace, h_param])

def step4_param_diversity():
    print("Running step 4...")
    result_files = glob.glob(os.path.join(tasks_dir, '*', '*', '*', 'result.json'))
    out_csv = os.path.join(audit_out_dir, 'parameter_diversity_audit.csv')
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['basin_id', 'seed', 'best_theta_norm_hash', 'best_theta_phys_hash'])
        for res_f in sorted(result_files):
            task_dir = os.path.dirname(res_f)
            parts = task_dir.split('/')
            seed_dir = parts[-1]
            model_basin = parts[-2]
            basin_id = model_basin.split('_')[1] if '_' in model_basin else model_basin
            
            params_f = os.path.join(task_dir, 'result.json')
            try:
                with open(params_f, 'r') as fp:
                    p = json.load(fp)
                    n_hash = hashlib.sha256(json.dumps(p.get('best_theta_normalized', [])).encode()).hexdigest()
                    p_hash = hashlib.sha256(json.dumps(p.get('best_theta_physical', [])).encode()).hexdigest()
                    writer.writerow([basin_id, seed_dir, n_hash, p_hash])
            except Exception:
                pass
                
def step8_convergence_trace():
    print("Running step 8...")
    trace_files = glob.glob(os.path.join(tasks_dir, '*', '*', '*', 'trace.jsonl'))
    out_csv = os.path.join(audit_out_dir, 'convergence_trace_audit.csv')
    gens_to_extract = {1, 10, 25, 50, 100, 200, 300, 400}
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['basin_id', 'seed', 'generation', 'best_fitness'])
        for trace_f in sorted(trace_files):
            task_dir = os.path.dirname(trace_f)
            parts = task_dir.split('/')
            seed_dir = parts[-1]
            model_basin = parts[-2]
            basin_id = model_basin.split('_')[1] if '_' in model_basin else model_basin
            
            with open(trace_f, 'r') as tf:
                for line in tf:
                    try:
                        d = json.loads(line)
                        if d.get('generation') in gens_to_extract:
                            writer.writerow([basin_id, seed_dir, d.get('generation'), d.get('best_fitness_gen')])
                    except Exception:
                        pass

if __name__ == '__main__':
    step1_raw_precision_audit()
    step4_param_diversity()
    step8_convergence_trace()
    print("Done scripts")
