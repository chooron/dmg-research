#!/usr/bin/env python3
"""Stage 1: Audit all R5 train-best cases from 20-basin experiment.

Identifies cases where R5 is selected as train-window oracle best,
analyzes margin, eval generalization, and seed concentration.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_baseline_data(base_dir: Path):
    """Load all baseline data files."""
    # Load oracle labels
    oracle_labels = pd.read_csv(base_dir / "oracle_labels_train.csv")
    
    # Load oracle eval audit
    oracle_eval = pd.read_csv(base_dir / "oracle_eval_audit.csv")
    
    # Load train metrics for each seed
    train_metrics = {}
    for seed in range(3):
        seed_dir = base_dir / f"fixed_formula_seed{seed}"
        if seed_dir.exists():
            metrics_file = seed_dir / "formula_metrics_train.csv"
            if metrics_file.exists():
                train_metrics[seed] = pd.read_csv(metrics_file)
    
    return oracle_labels, oracle_eval, train_metrics


def compute_margins(train_df: pd.DataFrame, basin_id: int, seed: int):
    """Compute MSE margins for a specific basin and seed."""
    subset = train_df[(train_df['basin_id'] == basin_id) & (train_df['seed'] == seed)]
    
    if len(subset) < 3:
        return None
    
    mse_values = {}
    nse_values = {}
    for _, row in subset.iterrows():
        formula = row['formula_id']
        mse_values[formula] = row['train_mse']
        nse_values[formula] = row['train_nse']
    
    # Find R5 values
    r5_mse = mse_values.get('R5', np.nan)
    r5_nse = nse_values.get('R5', np.nan)
    
    # Find second best MSE
    sorted_mse = sorted(mse_values.items(), key=lambda x: x[1])
    if sorted_mse[0][0] == 'R5':
        second_best_mse = sorted_mse[1][1] if len(sorted_mse) > 1 else np.nan
        second_best_formula = sorted_mse[1][0] if len(sorted_mse) > 1 else 'N/A'
    else:
        # R5 is not the best, find its rank
        r5_rank = next(i for i, (f, _) in enumerate(sorted_mse) if f == 'R5') + 1
        second_best_mse = sorted_mse[0][1]
        second_best_formula = sorted_mse[0][0]
    
    absolute_mse_margin = second_best_mse - r5_mse
    relative_mse_margin = absolute_mse_margin / (abs(second_best_mse) + 1e-8)
    
    return {
        'train_mse_R5': r5_mse,
        'train_nse_R5': r5_nse,
        'second_best_formula': second_best_formula,
        'second_best_mse': second_best_mse,
        'absolute_mse_margin': absolute_mse_margin,
        'relative_mse_margin': relative_mse_margin,
        'R5_rank': 1 if sorted_mse[0][0] == 'R5' else r5_rank
    }


def audit_r5_cases(base_dir: Path, output_dir: Path):
    """Main audit function."""
    print("Loading baseline data...")
    oracle_labels, oracle_eval, train_metrics = load_baseline_data(base_dir)
    
    # Filter R5 train-best cases
    r5_cases = oracle_labels[oracle_labels['best_train_formula'] == 'R5'].copy()
    print(f"Found {len(r5_cases)} R5 train-best cases")
    
    # Prepare detailed results
    results = []
    for _, case in r5_cases.iterrows():
        basin_id = case['basin_id']
        seed = case['seed']
        
        # Get train metrics
        train_df = train_metrics.get(seed)
        if train_df is None:
            print(f"Warning: No train metrics for seed {seed}")
            continue
        
        # Compute margins
        margin_info = compute_margins(train_df, basin_id, seed)
        if margin_info is None:
            print(f"Warning: Could not compute margins for basin {basin_id}, seed {seed}")
            continue
        
        # Get eval metrics from oracle_eval
        eval_row = oracle_eval[
            (oracle_eval['basin_id'] == basin_id) & 
            (oracle_eval['seed'] == seed)
        ]
        
        if len(eval_row) == 0:
            print(f"Warning: No eval data for basin {basin_id}, seed {seed}")
            continue
        
        eval_row = eval_row.iloc[0]
        
        # Get train NSE for all formulas
        train_subset = train_df[(train_df['basin_id'] == basin_id) & (train_df['seed'] == seed)]
        train_nse = {row['formula_id']: row['train_nse'] for _, row in train_subset.iterrows()}
        train_mse = {row['formula_id']: row['train_mse'] for _, row in train_subset.iterrows()}
        
        # Determine eval rank of R5
        eval_nse_values = {
            'R0': eval_row['eval_nse_R0'],
            'R4': eval_row['eval_nse_R4'],
            'R5': eval_row['eval_nse_R5']
        }
        sorted_eval = sorted(eval_nse_values.items(), key=lambda x: x[1], reverse=True)
        eval_rank_R5 = next(i for i, (f, _) in enumerate(sorted_eval) if f == 'R5') + 1
        
        result = {
            'basin_id': basin_id,
            'seed': seed,
            'best_train_formula': 'R5',
            'train_nse_R0': train_nse.get('R0', np.nan),
            'train_nse_R4': train_nse.get('R4', np.nan),
            'train_nse_R5': train_nse.get('R5', np.nan),
            'train_mse_R0': train_mse.get('R0', np.nan),
            'train_mse_R4': train_mse.get('R4', np.nan),
            'train_mse_R5': train_mse.get('R5', np.nan),
            'second_best_formula': margin_info['second_best_formula'],
            'absolute_mse_margin': margin_info['absolute_mse_margin'],
            'relative_mse_margin': margin_info['relative_mse_margin'],
            'absolute_nse_margin': train_nse.get('R5', 0) - max(train_nse.get('R0', 0), train_nse.get('R4', 0)),
            'eval_nse_R0': eval_row['eval_nse_R0'],
            'eval_nse_R4': eval_row['eval_nse_R4'],
            'eval_nse_R5': eval_row['eval_nse_R5'],
            'eval_rank_R5': eval_rank_R5,
            'R5_generalizes_to_eval': eval_row.get('generalizes_to_eval', False)
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "01_r5_train_best_cases.csv", index=False)
    
    # Generate summary
    summary_lines = []
    summary_lines.append("# Stage 1: R5 Train-Best Cases Audit Report\n")
    summary_lines.append(f"> Total R5 train-best cases: {len(results_df)}\n")
    
    # Margin analysis
    summary_lines.append("## 1. Margin Analysis\n")
    clear_winners = results_df[results_df['relative_mse_margin'] >= 0.05]
    weak_winners = results_df[
        (results_df['relative_mse_margin'] >= 0.02) & 
        (results_df['relative_mse_margin'] < 0.05)
    ]
    marginal_winners = results_df[results_df['relative_mse_margin'] < 0.02]
    
    summary_lines.append(f"- Clear winners (margin >= 5%): {len(clear_winners)}")
    summary_lines.append(f"- Weak winners (2% <= margin < 5%): {len(weak_winners)}")
    summary_lines.append(f"- Marginal winners (margin < 2%): {len(marginal_winners)}\n")
    
    # Eval generalization
    summary_lines.append("## 2. Eval Generalization\n")
    generalizes = results_df[results_df['R5_generalizes_to_eval'] == True]
    summary_lines.append(f"- R5 generalizes to eval: {len(generalizes)}/{len(results_df)} ({100*len(generalizes)/len(results_df):.1f}%)\n")
    
    # Eval rank distribution
    summary_lines.append("## 3. Eval Rank Distribution\n")
    rank_counts = results_df['eval_rank_R5'].value_counts().sort_index()
    for rank, count in rank_counts.items():
        summary_lines.append(f"- Rank {rank}: {count} cases")
    summary_lines.append("")
    
    # Seed concentration
    summary_lines.append("## 4. Seed Concentration\n")
    seed_counts = results_df['seed'].value_counts().sort_index()
    for seed, count in seed_counts.items():
        summary_lines.append(f"- Seed {seed}: {count} R5 train-best cases")
    summary_lines.append("")
    
    # Basin concentration
    summary_lines.append("## 5. Basin Concentration\n")
    basin_counts = results_df['basin_id'].value_counts()
    multi_seed_basins = basin_counts[basin_counts >= 2]
    summary_lines.append(f"- Basins with R5 best in 2+ seeds: {len(multi_seed_basins)}")
    for basin, count in multi_seed_basins.items():
        summary_lines.append(f"  - Basin {basin}: {count} seeds")
    summary_lines.append("")
    
    # Key findings
    summary_lines.append("## 6. Key Findings\n")
    
    # Check if R5 is consistently bad in eval
    r5_eval_worse = results_df[results_df['eval_nse_R5'] < results_df['eval_nse_R0']]
    summary_lines.append(f"- Cases where R5 eval NSE < R0 eval NSE: {len(r5_eval_worse)}/{len(results_df)}")
    
    # Check R5 eval NSE distribution
    summary_lines.append(f"- Mean R5 eval NSE: {results_df['eval_nse_R5'].mean():.3f}")
    summary_lines.append(f"- Median R5 eval NSE: {results_df['eval_nse_R5'].median():.3f}")
    summary_lines.append(f"- R5 eval NSE < 0: {(results_df['eval_nse_R5'] < 0).sum()} cases")
    
    # Check margin vs generalization
    if len(clear_winners) > 0:
        clear_gen = clear_winners[clear_winners['R5_generalizes_to_eval'] == True]
        summary_lines.append(f"\n- Clear winners that generalize: {len(clear_gen)}/{len(clear_winners)}")
    
    # Conclusion
    summary_lines.append("\n## 7. Conclusion\n")
    
    if len(generalizes) / len(results_df) < 0.3:
        summary_lines.append("**R5 train-best labels show POOR eval generalization.**")
        summary_lines.append("Most R5 train-best cases do not translate to eval performance.")
    elif len(generalizes) / len(results_df) < 0.5:
        summary_lines.append("**R5 train-best labels show MODERATE eval generalization.**")
        summary_lines.append("Some R5 labels may be reliable, but many are questionable.")
    else:
        summary_lines.append("**R5 train-best labels show GOOD eval generalization.**")
    
    if len(marginal_winners) > len(clear_winners):
        summary_lines.append("\n**Many R5 wins are marginal, suggesting calibration noise.**")
    
    with open(output_dir / "01_r5_train_best_summary.md", "w") as f:
        f.write("\n".join(summary_lines))
    
    print(f"\nResults saved to {output_dir}")
    print(f"Total R5 train-best cases: {len(results_df)}")
    print(f"Clear winners: {len(clear_winners)}")
    print(f"Weak winners: {len(weak_winners)}")
    print(f"Marginal winners: {len(marginal_winners)}")
    print(f"Generalizes to eval: {len(generalizes)}/{len(results_df)}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Audit R5 train-best cases")
    parser.add_argument("--base-dir", type=str, 
                        default="validation_results/static_router_20basin_calibrated",
                        help="Baseline results directory")
    parser.add_argument("--output-dir", type=str,
                        default="validation_results/r5_oracle_reliability_audit_20basin",
                        help="Output directory")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    audit_r5_cases(base_dir, output_dir)


if __name__ == "__main__":
    main()
