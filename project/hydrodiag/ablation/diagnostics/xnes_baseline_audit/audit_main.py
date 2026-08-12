import os
import json
import hashlib
import csv
import glob
import numpy as np
import torch

from pathlib import Path

import sys
sys.path.append('/home/jingxin/code/dmg-research/project/hydrodiag')

from ablation.ic_core.data_adapter import load_531_bundle
from ablation.ic_core.runtime import ICObjectiveRuntime
from ablation.ic_core.parameter_adapter import normalized_to_physical

base_dir = Path('/home/jingxin/code/dmg-research/project/hydrodiag')
tasks_dir = base_dir / 'outputs/ic_ablation/stage1_screening/v1/xnes/tasks'
audit_out_dir = base_dir / 'outputs/ic_ablation/stage1_screening/v1/xnes_audit'
config_path = base_dir / 'outputs/ic_ablation/stage1_screening/v1/xnes/resolved_config.json'

def calc_kge_cpu(sim: np.ndarray, obs: np.ndarray) -> float:
    valid = np.isfinite(obs)
    if not np.any(valid): return -999.0
    sim = sim[valid]
    obs = obs[valid]
    if len(obs) == 0: return -999.0
    
    mean_sim = np.mean(sim)
    mean_obs = np.mean(obs)
    std_sim = np.std(sim, ddof=0)
    std_obs = np.std(obs, ddof=0)
    if std_obs == 0 or np.isnan(std_obs): return -999.0
    
    if std_sim == 0 or np.isnan(std_sim):
        r = 0.0
    else:
        cc = np.corrcoef(sim, obs)
        r = cc[0, 1] if np.size(cc) > 1 else 0.0
        if np.isnan(r): r = 0.0
        
    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs if mean_obs != 0 else float('inf')
    
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    return float(kge)


def get_task_info():
    result_files = sorted(glob.glob(str(tasks_dir / '*' / '*' / '*' / 'result.json')))
    tasks = []
    for res_f in result_files:
        task_dir = Path(res_f).parent
        seed_dir = task_dir.parent.name # seed_000
        start_dir = task_dir.name # start_00
        basin_id = task_dir.parent.parent.name
        tasks.append((str(task_dir), basin_id, seed_dir, start_dir))
    return tasks


def get_config():
    with open(config_path) as f:
        config = json.load(f)
    config["device"] = "cpu"
    config["dataset_path"] = str(base_dir.parent.parent / 'data/camels_dataset')
    config["gage_ids_path"] = str(base_dir.parent.parent / 'data/gage_id.npy')
    config["dates_path"] = str(base_dir.parent.parent / 'data/camels_dates.npy')
    config["basin_list_path"] = str(base_dir.parent.parent / 'data/531sub_id.txt')
    return config


def run_step3_data_isolation():
    print("Step 3: Basin Data Isolation")
    config = get_config()
    bundle = load_531_bundle(config)
    
    out_csv = audit_out_dir / 'basin_data_identity.csv'
    
    def hash_array(arr):
        return hashlib.sha256(arr.tobytes()).hexdigest()
        
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['basin_id', 'forcing_hash', 'target_cfs_hash', 'target_mm_day_hash', 'valid_mask_hash', 'area'])
        
        count = 0
        for i, bid in enumerate(bundle.basin_ids):
            f_hash = hash_array(bundle.forcing[i])
            tcfs_hash = hash_array(bundle.target_cfs[i])
            tmm_hash = hash_array(bundle.target_mm_day[i])
            vm_hash = hash_array(bundle.valid_target_mask[i])
            area = bundle.raw_area_km2[i]
            writer.writerow([bid, f_hash, tcfs_hash, tmm_hash, vm_hash, float(area)])
            
            if count < 4:
                tmm = bundle.target_mm_day[i]
                v_mask = bundle.valid_target_mask[i]
                valid_tmm = tmm[v_mask]
                print(f"Basin {bid} Area: {area}")
                if len(valid_tmm) > 0:
                    print(f"  first 5 valid target_mm_day: {valid_tmm[:5]}")
                    print(f"  last 5 valid target_mm_day: {valid_tmm[-5:]}")
                count += 1
                
def run_step5_and_6_and_7():
    print("Step 5, 6, 7: Independent KGE, Hydrograph, Candidate Discrimination")
    config = get_config()
    bundle = load_531_bundle(config)
    
    all_tasks = get_task_info()
    basins_found = {}
    for task_dir, basin_id, seed, start_id in all_tasks:
        if basin_id not in basins_found:
            basins_found[basin_id] = []
        basins_found[basin_id].append((task_dir, seed))
        
    target_basins = list(basins_found.keys())[:8]
    
    runtime = ICObjectiveRuntime(bundle, config, "XAJ")
    out_kge_csv = audit_out_dir / 'independent_kge_recalculation.csv'
    out_discrim_csv = audit_out_dir / 'candidate_discrimination.csv'
    
    discrim_basins = target_basins[:6]
    
    with open(out_kge_csv, 'w', newline='') as fkge, \
         open(out_discrim_csv, 'w', newline='') as fdiscrim:
         
        kge_writer = csv.writer(fkge)
        kge_writer.writerow(['basin_id', 'seed', 'saved_best_train_kge', 'gpu_recomputed', 'cpu_independent_fp64'])
        
        discrim_writer = csv.writer(fdiscrim)
        discrim_writer.writerow(['basin_id', 'candidate_type', 'kge'])
        
        for basin_id in target_basins:
            b_idx = bundle.basin_ids.index(basin_id)
            
            task_dir, seed = basins_found[basin_id][0]
            with open(Path(task_dir) / 'result.json') as f:
                res_data = json.load(f)
                best_norm = np.array(res_data['best_theta_normalized'], dtype=np.float32)
                saved_kge = res_data['best_train_kge']
            
            cand_tensor = torch.tensor(best_norm, device=runtime.device).unsqueeze(0).unsqueeze(0)
            
            # evaluate_candidates takes (theta_01, basin_indices, split)
            evals = runtime.evaluate_candidates(cand_tensor, basin_indices=[b_idx], split="train")
            gpu_kge = evals.fitness[0, 0].item()
            
            physical = normalized_to_physical(runtime.model_key, cand_tensor, clip=True).squeeze(0).squeeze(0) # (D)
            forcing, target_mm, warmup_days = runtime._split_arrays("train")
            forcing_sub = forcing[[b_idx]]
            
            forcing_tensor = torch.from_numpy(forcing_sub).to(device=runtime.device, dtype=torch.float32)
            physical_tensor = torch.from_numpy(physical.cpu().numpy()).unsqueeze(0).to(device=runtime.device, dtype=torch.float32)
            
            sim_tensor, _ = runtime.model_adapter.run_model(
                forcing_tensor,
                physical_tensor,
                forcing_names=bundle.forcing_names,
                temp_mean_train=torch.from_numpy(bundle.temp_mean_train[[b_idx]]).to(device=runtime.device, dtype=torch.float32),
                temp_std_train=torch.from_numpy(bundle.temp_std_train[[b_idx]]).to(device=runtime.device, dtype=torch.float32)
            )
            sim_cpu = sim_tensor[0].detach().cpu().numpy().astype(np.float64)
            sim_cpu = sim_cpu[warmup_days:]
            
            
            obs = target_mm[b_idx].astype(np.float64)
            # find valid by looking at non-negative values or finite values? 
            # In KGE calculation calc_kge_cpu we already filter by np.isfinite(obs)!
            # We can just replace valid mask logic. Let's see if there are invalid fitnesses -999.0
            invalid_val = -999.0
            obs[obs < -100] = np.nan

            
            cpu_kge = calc_kge_cpu(sim_cpu, obs)
            
            kge_writer.writerow([basin_id, seed, saved_kge, gpu_kge, cpu_kge])
            
            hydro_csv = audit_out_dir / 'hydrographs' / f'{basin_id}.csv'
            with open(hydro_csv, 'w', newline='') as hf:
                hw = csv.writer(hf)
                hw.writerow(['timestep', 'obs_mm_day', 'sim_mm_day'])
                for t in range(len(obs)):
                    hw.writerow([t, obs[t], sim_cpu[t]])
            
            if basin_id in discrim_basins:
                c_mid = np.ones_like(best_norm) * 0.5
                c_rand1 = np.random.uniform(0, 1, size=best_norm.shape).astype(np.float32)
                c_pert1 = np.clip(best_norm + np.random.normal(0, 0.01, size=best_norm.shape), 0, 1).astype(np.float32)
                
                cands_to_test = {
                    'saved_best': best_norm,
                    'midpoint_0.5': c_mid,
                    'random_1': c_rand1,
                    'perturbed_small': c_pert1,
                }
                
                for c_name, c_val in cands_to_test.items():
                    c_tensor = torch.tensor(c_val, device=runtime.device).unsqueeze(0).unsqueeze(0)
                    ev = runtime.evaluate_candidates(c_tensor, basin_indices=[b_idx], split="train")
                    discrim_writer.writerow([basin_id, c_name, ev.fitness[0, 0].item()])

if __name__ == '__main__':
    run_step3_data_isolation()
    run_step5_and_6_and_7()
    print("Done audit_main.py")
