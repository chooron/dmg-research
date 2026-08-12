#!/usr/bin/env python3
"""Stage 2: Calibration convergence audit.

Since the original calibration used random search (100 samples),
this script analyzes:
1. Whether 100 samples is sufficient
2. R5 calibration quality vs R0/R4
3. Train-eval gap analysis
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import math


def load_all_metrics(base_dir: Path):
    """Load train and eval metrics for all seeds."""
    all_train = []
    all_eval = []
    
    for seed in range(3):
        seed_dir = base_dir / f"fixed_formula_seed{seed}"
        if not seed_dir.exists():
            continue
        
        train_file = seed_dir / "formula_metrics_train.csv"
        eval_file = seed_dir / "formula_metrics_eval.csv"
        
        if train_file.exists():
            train_df = pd.read_csv(train_file)
            all_train.append(train_df)
        
        if eval_file.exists():
            eval_df = pd.read_csv(eval_file)
            all_eval.append(eval_df)
    
    return pd.concat(all_train, ignore_index=True) if all_train else pd.DataFrame(), \
           pd.concat(all_eval, ignore_index=True) if all_eval else pd.DataFrame()


def analyze_convergence(base_dir: Path, output_dir: Path):
    """Analyze calibration convergence and quality."""
    print("Loading metrics...")
    train_df, eval_df = load_all_metrics(base_dir)
    
    if train_df.empty:
        print("No train metrics found!")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge train and eval
    merged = train_df.merge(eval_df, on=['basin_id', 'seed', 'formula_id'], suffixes=('_train', '_eval'))
    
    # Compute train-eval gap
    merged['train_eval_gap'] = merged['train_nse'] - merged['eval_nse']
    merged['train_eval_gap_abs'] = abs(merged['train_eval_gap'])
    
    # Analyze by formula
    summary_records = []
    for formula_id in ['R0', 'R4', 'R5']:
        formula_data = merged[merged['formula_id'] == formula_id]
        if formula_data.empty:
            continue
        
        # Train NSE statistics
        train_nse = formula_data['train_nse'].dropna()
        eval_nse = formula_data['eval_nse'].dropna()
        train_eval_gap = formula_data['train_eval_gap'].dropna()
        
        # Count negative train NSE
        negative_train_nse = (train_nse < 0).sum()
        very_negative_train_nse = (train_nse < -1).sum()
        
        # Count cases where train NSE > 0 but eval NSE < 0
        overfit_cases = ((formula_data['train_nse'] > 0) & (formula_data['eval_nse'] < 0)).sum()
        
        # Count cases where R5 is significantly worse than R0/R4 in eval
        if formula_id == 'R5':
            r0_eval = merged[merged['formula_id'] == 'R0'][['basin_id', 'seed', 'eval_nse']].rename(columns={'eval_nse': 'eval_nse_R0'})
            r5_eval = formula_data[['basin_id', 'seed', 'eval_nse']].rename(columns={'eval_nse': 'eval_nse_R5'})
            comparison = r5_eval.merge(r0_eval, on=['basin_id', 'seed'])
            r5_worse_than_r0 = (comparison['eval_nse_R5'] < comparison['eval_nse_R0']).sum()
            r5_much_worse = (comparison['eval_nse_R5'] < comparison['eval_nse_R0'] - 0.5).sum()
        else:
            r5_worse_than_r0 = None
            r5_much_worse = None
        
        summary_records.append({
            'formula_id': formula_id,
            'n_cases': len(formula_data),
            'mean_train_nse': train_nse.mean(),
            'median_train_nse': train_nse.median(),
            'std_train_nse': train_nse.std(),
            'negative_train_nse_count': negative_train_nse,
            'very_negative_train_nse_count': very_negative_train_nse,
            'mean_eval_nse': eval_nse.mean(),
            'median_eval_nse': eval_nse.median(),
            'std_eval_nse': eval_nse.std(),
            'mean_train_eval_gap': train_eval_gap.mean(),
            'median_train_eval_gap': train_eval_gap.median(),
            'overfit_cases': overfit_cases,
            'r5_worse_than_r0': r5_worse_than_r0,
            'r5_much_worse_than_r0': r5_much_worse
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_dir / "02_calibration_convergence_summary.csv", index=False)
    
    # Generate report
    report_lines = []
    report_lines.append("# Stage 2: Calibration Convergence Audit Report\n")
    
    report_lines.append("## 1. Calibration Method\n")
    report_lines.append("- **Method**: Random search (100 samples per basin per formula)")
    report_lines.append("- **Optimization**: Find parameter set minimizing train MSE")
    report_lines.append("- **No gradient information available**")
    report_lines.append("- **Budget**: 100 random samples\n")
    
    report_lines.append("## 2. Train NSE Statistics\n")
    report_lines.append("| Formula | Mean | Median | Std | Negative (<0) | Very Negative (<-1) |")
    report_lines.append("|---------|------|--------|-----|---------------|---------------------|")
    for _, row in summary_df.iterrows():
        report_lines.append(f"| {row['formula_id']} | {row['mean_train_nse']:.3f} | {row['median_train_nse']:.3f} | {row['std_train_nse']:.3f} | {row['negative_train_nse_count']} | {row['very_negative_train_nse_count']} |")
    
    report_lines.append("\n## 3. Eval NSE Statistics\n")
    report_lines.append("| Formula | Mean | Median | Std |")
    report_lines.append("|---------|------|--------|-----|")
    for _, row in summary_df.iterrows():
        report_lines.append(f"| {row['formula_id']} | {row['mean_eval_nse']:.3f} | {row['median_eval_nse']:.3f} | {row['std_eval_nse']:.3f} |")
    
    report_lines.append("\n## 4. Train-Eval Gap Analysis\n")
    report_lines.append("| Formula | Mean Gap | Median Gap | Overfit Cases |")
    report_lines.append("|---------|----------|------------|---------------|")
    for _, row in summary_df.iterrows():
        report_lines.append(f"| {row['formula_id']} | {row['mean_train_eval_gap']:.3f} | {row['median_train_eval_gap']:.3f} | {row['overfit_cases']} |")
    
    # R5 specific analysis
    report_lines.append("\n## 5. R5 vs R0 Comparison\n")
    r5_row = summary_df[summary_df['formula_id'] == 'R5']
    if not r5_row.empty:
        r5_row = r5_row.iloc[0]
        report_lines.append(f"- R5 cases worse than R0 in eval: {r5_row['r5_worse_than_r0']}/{r5_row['n_cases']}")
        report_lines.append(f"- R5 cases much worse than R0 (ΔNSE < -0.5): {r5_row['r5_much_worse_than_r0']}/{r5_row['n_cases']}")
    
    # Key findings
    report_lines.append("\n## 6. Key Findings\n")
    
    # Check R5 calibration quality
    r5_train_mean = r5_row['mean_train_nse'] if not r5_row.empty else None
    r0_train_mean = summary_df[summary_df['formula_id'] == 'R0']['mean_train_nse'].values
    
    if r5_train_mean is not None and len(r0_train_mean) > 0:
        if r5_train_mean < -0.5:
            report_lines.append("**R5 calibration is POOR**: Mean train NSE is significantly negative.")
            report_lines.append("This suggests 100 random samples may be insufficient for R5.")
        elif r5_train_mean < 0:
            report_lines.append("**R5 calibration is WEAK**: Mean train NSE is negative.")
            report_lines.append("R5 may need more calibration budget or different optimization.")
        else:
            report_lines.append("**R5 calibration is ACCEPTABLE**: Mean train NSE is non-negative.")
    
    # Check overfitting
    if not r5_row.empty and r5_row['overfit_cases'] > r5_row['n_cases'] * 0.3:
        report_lines.append("\n**R5 shows significant overfitting**: Many cases have positive train NSE but negative eval NSE.")
    
    # Conclusion
    report_lines.append("\n## 7. Conclusion\n")
    
    if r5_train_mean is not None:
        if r5_train_mean < -0.5:
            report_lines.append("**FAIL**: R5 calibration quality is too low for reliable oracle labels.")
            report_lines.append("Recommendation: Increase calibration budget or disable R5 for router labels.")
        elif r5_train_mean < 0:
            report_lines.append("**PARTIAL**: R5 calibration quality is marginal.")
            report_lines.append("Recommendation: Use R5 labels with caution, consider filtering.")
        else:
            report_lines.append("**PASS**: R5 calibration quality is acceptable.")
    
    with open(output_dir / "02_calibration_convergence_report.md", "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nResults saved to {output_dir}")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Audit calibration convergence")
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
    
    analyze_convergence(base_dir, output_dir)


if __name__ == "__main__":
    main()
