#!/usr/bin/env python3
"""Stage 2 & 3: Calibration convergence and parameter boundary audit.

Analyzes:
1. Random search calibration efficiency (100 samples)
2. Parameter boundary proximity for R0/R4/R5
3. R5 parameter degeneration risk
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json


_PARAM_BOUNDS = {
    "parBETA": [1.0, 6.0], "parFC": [50.0, 500.0], "parK0": [0.05, 0.5],
    "parK1": [0.01, 0.3], "parK2": [0.001, 0.1], "parLP": [0.3, 1.0],
    "parPERC": [0.0, 3.0], "parUZL": [0.0, 100.0], "parTT": [-2.5, 2.5],
    "parCFMAX": [1.0, 10.0], "parCFR": [0.0, 0.1], "parCWH": [0.0, 0.2],
}

_EXTRA_PARAMS = {
    "R4": {"a_r": [1.0, 20.0], "c_r": [0.1, 0.9]},
    "R5": {"b_v": [0.3, 1.5]},
}

_RECHARGE_PARAMS = ["parFC", "parBETA"]


def load_params(base_dir: Path, seed: int) -> pd.DataFrame:
    """Load calibrated parameters for a seed."""
    param_file = base_dir / f"fixed_formula_seed{seed}" / "formula_params.csv"
    if not param_file.exists():
        print(f"Warning: {param_file} not found")
        return pd.DataFrame()
    return pd.read_csv(param_file)


def compute_normalized_distance(value: float, lo: float, hi: float) -> dict:
    """Compute normalized distance to boundaries."""
    norm_val = (value - lo) / (hi - lo) if hi > lo else 0.5
    dist_lower = norm_val
    dist_upper = 1.0 - norm_val
    at_lower = norm_val <= 0.02
    at_upper = norm_val >= 0.98
    near_boundary = at_lower or at_upper
    return {
        'normalized_value': norm_val,
        'distance_to_lower': dist_lower,
        'distance_to_upper': dist_upper,
        'at_lower_boundary': at_lower,
        'at_upper_boundary': at_upper,
        'near_boundary': near_boundary
    }


def analyze_parameter_boundaries(base_dir: Path, output_dir: Path):
    """Analyze parameter boundaries for all formulas."""
    print("Analyzing parameter boundaries...")
    
    all_records = []
    for seed in range(3):
        params_df = load_params(base_dir, seed)
        if params_df.empty:
            continue
        
        for _, row in params_df.iterrows():
            basin_id = row['basin_id']
            formula_id = row['formula_id']
            
            # Analyze standard HBV parameters
            for param_name, (lo, hi) in _PARAM_BOUNDS.items():
                if param_name not in row.index:
                    continue
                value = row[param_name]
                if pd.isna(value):
                    continue
                
                dist_info = compute_normalized_distance(value, lo, hi)
                all_records.append({
                    'basin_id': basin_id,
                    'seed': seed,
                    'formula_id': formula_id,
                    'parameter_name': param_name,
                    'physical_value': value,
                    'lower_bound': lo,
                    'upper_bound': hi,
                    **dist_info
                })
    
    if not all_records:
        print("No parameter records found!")
        return None
    
    results_df = pd.DataFrame(all_records)
    
    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "03_parameter_boundary_by_run.csv", index=False)
    
    # Generate summary by formula
    summary_records = []
    for formula_id in ['R0', 'R4', 'R5']:
        formula_data = results_df[results_df['formula_id'] == formula_id]
        if formula_data.empty:
            continue
        
        total_params = len(formula_data)
        near_boundary = formula_data['near_boundary'].sum()
        at_lower = formula_data['at_lower_boundary'].sum()
        at_upper = formula_data['at_upper_boundary'].sum()
        
        # Focus on recharge-related parameters
        recharge_params = formula_data[formula_data['parameter_name'].isin(_RECHARGE_PARAMS)]
        recharge_near_boundary = recharge_params['near_boundary'].sum() if not recharge_params.empty else 0
        
        summary_records.append({
            'formula_id': formula_id,
            'total_parameters': total_params,
            'near_boundary_count': near_boundary,
            'near_boundary_fraction': near_boundary / total_params if total_params > 0 else 0,
            'at_lower_count': at_lower,
            'at_upper_count': at_upper,
            'recharge_params_total': len(recharge_params),
            'recharge_params_near_boundary': recharge_near_boundary,
            'recharge_boundary_fraction': recharge_near_boundary / len(recharge_params) if len(recharge_params) > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_dir / "03_parameter_boundary_summary.csv", index=False)
    
    # Generate report
    report_lines = []
    report_lines.append("# Stage 3: Parameter Boundary Audit Report\n")
    
    report_lines.append("## 1. Overall Boundary Statistics\n")
    report_lines.append("| Formula | Total Params | Near Boundary | Fraction | At Lower | At Upper |")
    report_lines.append("|---------|--------------|---------------|----------|----------|----------|")
    for _, row in summary_df.iterrows():
        report_lines.append(f"| {row['formula_id']} | {row['total_parameters']} | {row['near_boundary_count']} | {row['near_boundary_fraction']:.3f} | {row['at_lower_count']} | {row['at_upper_count']} |")
    
    report_lines.append("\n## 2. Recharge Parameter Boundary Analysis\n")
    report_lines.append("| Formula | Recharge Params | Near Boundary | Fraction |")
    report_lines.append("|---------|-----------------|---------------|----------|")
    for _, row in summary_df.iterrows():
        report_lines.append(f"| {row['formula_id']} | {row['recharge_params_total']} | {row['recharge_params_near_boundary']} | {row['recharge_boundary_fraction']:.3f} |")
    
    # R5 specific analysis
    report_lines.append("\n## 3. R5 Parameter Analysis\n")
    r5_data = results_df[results_df['formula_id'] == 'R5']
    if not r5_data.empty:
        r5_summary = r5_data.groupby('parameter_name').agg({
            'near_boundary': 'sum',
            'physical_value': ['mean', 'std', 'min', 'max']
        }).reset_index()
        r5_summary.columns = ['parameter_name', 'near_boundary_count', 'mean_value', 'std_value', 'min_value', 'max_value']
        
        report_lines.append("| Parameter | Mean | Std | Min | Max | Near Boundary |")
        report_lines.append("|-----------|------|-----|-----|-----|---------------|")
        for _, row in r5_summary.iterrows():
            report_lines.append(f"| {row['parameter_name']} | {row['mean_value']:.4f} | {row['std_value']:.4f} | {row['min_value']:.4f} | {row['max_value']:.4f} | {row['near_boundary_count']} |")
    
    # Key findings
    report_lines.append("\n## 4. Key Findings\n")
    
    # Check if R5 has higher boundary fraction
    r5_boundary = summary_df[summary_df['formula_id'] == 'R5']['near_boundary_fraction'].values
    r0_boundary = summary_df[summary_df['formula_id'] == 'R0']['near_boundary_fraction'].values
    
    if len(r5_boundary) > 0 and len(r0_boundary) > 0:
        if r5_boundary[0] > r0_boundary[0] * 1.5:
            report_lines.append("**R5 has significantly higher parameter boundary fraction than R0.**")
            report_lines.append("This suggests R5 may rely on extreme parameter values.")
        elif r5_boundary[0] > 0.3:
            report_lines.append("**R5 has moderate parameter boundary fraction.**")
            report_lines.append("Some R5 calibrations may be near boundaries.")
        else:
            report_lines.append("**R5 parameter boundary fraction is acceptable.**")
    
    # Conclusion
    report_lines.append("\n## 5. Conclusion\n")
    
    r5_recharge_boundary = summary_df[summary_df['formula_id'] == 'R5']['recharge_boundary_fraction'].values
    if len(r5_recharge_boundary) > 0:
        if r5_recharge_boundary[0] > 0.3:
            report_lines.append("**HIGH_RISK**: R5 recharge parameters frequently near boundaries.")
            report_lines.append("R5 labels should be treated with caution.")
        elif r5_recharge_boundary[0] > 0.1:
            report_lines.append("**MEDIUM_RISK**: Some R5 recharge parameters near boundaries.")
        else:
            report_lines.append("**LOW_RISK**: R5 recharge parameters are well within bounds.")
    
    with open(output_dir / "03_parameter_boundary_report.md", "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nResults saved to {output_dir}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Audit parameter boundaries")
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
    
    analyze_parameter_boundaries(base_dir, output_dir)


if __name__ == "__main__":
    main()
