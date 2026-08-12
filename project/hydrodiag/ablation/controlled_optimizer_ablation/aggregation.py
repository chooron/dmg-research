import json
import os
import glob
import numpy as np
import pandas as pd
from typing import List, Dict

def load_traces(output_dir: str) -> List[Dict]:
    path = os.path.join(output_dir, "trace.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

def compute_metrics(config: dict) -> pd.DataFrame:
    output_root = config["output_root"]
    run_subdir = config["run_subdir"]
    
    tasks_dir = os.path.join(output_root, run_subdir, "tasks")
    
    rows = []
    
    optimizers = config["optimizers"]
    
    for opt in optimizers:
        opt_dir = os.path.join(tasks_dir, opt)
        if not os.path.exists(opt_dir):
            continue
        
        for basin_id in os.listdir(opt_dir):
            basin_dir = os.path.join(opt_dir, basin_id)
            if not os.path.isdir(basin_dir):
                continue
                
            for seed in config["optimizer_seeds"]:
                seed_dir = os.path.join(basin_dir, f"seed_{seed}")
                if not os.path.exists(seed_dir):
                    continue
                    
                # Get best over all 3 starts for this replicate
                best_kge = -np.inf
                
                for start_idx in range(config["starts"]):
                    start_dir = os.path.join(seed_dir, f"start_{start_idx}")
                    traces = load_traces(start_dir)
                    if traces:
                        start_best = max(t["best_fitness"] for t in traces)
                        if start_best > best_kge:
                            best_kge = start_best
                
                if best_kge > -np.inf:
                    rows.append({
                        "basin_id": basin_id,
                        "optimizer": opt,
                        "seed": seed,
                        "best_kge": best_kge
                    })
                    
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
        
    # Aggregate across the 3 seeds
    agg_df = df.groupby(["basin_id", "optimizer"])["best_kge"].agg(
        median="median",
        min="min",
        max="max",
        std="std"
    ).reset_index()
    
    return agg_df

def generate_report(agg_df: pd.DataFrame, report_dir: str):
    os.makedirs(report_dir, exist_ok=True)
    
    agg_df.to_csv(os.path.join(report_dir, "per_basin_optimizer.csv"), index=False)
    
    # Paired comparison
    basins = agg_df["basin_id"].unique()
    paired_rows = []
    
    xnes_wins = 0
    cmaes_wins = 0
    ties = 0
    
    for basin in basins:
        basin_df = agg_df[agg_df["basin_id"] == basin]
        xnes_row = basin_df[basin_df["optimizer"] == "XNES"]
        cmaes_row = basin_df[basin_df["optimizer"] == "CMAES"]
        
        if len(xnes_row) == 1 and len(cmaes_row) == 1:
            xnes_kge = xnes_row["median"].values[0]
            cmaes_kge = cmaes_row["median"].values[0]
            delta = xnes_kge - cmaes_kge
            
            if abs(delta) < 0.003:
                ties += 1
                winner = "TIE"
            elif delta >= 0.003:
                xnes_wins += 1
                winner = "XNES"
            else:
                cmaes_wins += 1
                winner = "CMAES"
                
            paired_rows.append({
                "basin_id": basin,
                "xnes_median": xnes_kge,
                "cmaes_median": cmaes_kge,
                "delta": delta,
                "winner": winner
            })
            
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(report_dir, "paired_optimizer_comparison.csv"), index=False)
    
    n_basins = len(paired_df)
    
    # Advancing logic
    # median advantage >= 0.003 and >=70% basins better -> single winner
    threshold = 0.7 * n_basins
    
    decision = "BOTH_RETAINED"
    if xnes_wins >= threshold:
        decision = "XNES_ADVANCES"
    elif cmaes_wins >= threshold:
        decision = "CMAES_ADVANCES"
        
    with open(os.path.join(report_dir, "PHASE1_OPTIMIZER_REPORT.md"), "w") as f:
        f.write("# Phase 1 Optimizer Report\n\n")
        f.write(f"Total Basins Compared: {n_basins}\n")
        f.write(f"XNES Wins: {xnes_wins}\n")
        f.write(f"CMAES Wins: {cmaes_wins}\n")
        f.write(f"Ties: {ties}\n\n")
        f.write(f"Decision: **{decision}**\n\n")
        f.write("## Phase 2 Plan\n")
        if decision == "BOTH_RETAINED":
            f.write("- Evaluate XNES on population grid: 24, 48, 72, 96\n")
            f.write("- Evaluate CMAES on population grid: 24, 48, 72, 96\n")
        elif decision == "XNES_ADVANCES":
            f.write("- Evaluate XNES on population grid: 24, 48, 72, 96\n")
        else:
            f.write("- Evaluate CMAES on population grid: 24, 48, 72, 96\n")
