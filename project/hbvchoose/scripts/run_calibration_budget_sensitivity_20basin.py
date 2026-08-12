#!/usr/bin/env python3
"""Stage 4: Calibration budget sensitivity analysis.

Tests whether R5 oracle labels are stable under increased calibration budget.
Runs calibration with 300, 600, and 1000 random samples for:
- All R5 train-best cases
- Representative R0/R4 cases as control
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))


def load_r5_cases(audit_dir: Path) -> pd.DataFrame:
    """Load R5 train-best cases from Stage 1."""
    r5_file = audit_dir / "01_r5_train_best_cases.csv"
    if not r5_file.exists():
        print(f"Error: R5 cases file not found: {r5_file}")
        return pd.DataFrame()
    return pd.read_csv(r5_file)


def load_baseline_data(base_dir: Path):
    """Load baseline oracle labels."""
    oracle_file = base_dir / "oracle_labels_train.csv"
    if not oracle_file.exists():
        return pd.DataFrame()
    return pd.read_csv(oracle_file)


def select_control_cases(oracle_labels: pd.DataFrame, r5_cases: pd.DataFrame, n_per_seed: int = 3):
    """Select R0 and R4 train-best cases as control."""
    control_cases = []
    
    for seed in range(3):
        seed_labels = oracle_labels[oracle_labels['seed'] == seed]
        
        # R0 train-best cases
        r0_cases = seed_labels[seed_labels['best_train_formula'] == 'R0'].head(n_per_seed)
        for _, row in r0_cases.iterrows():
            control_cases.append({
                'basin_id': row['basin_id'],
                'seed': seed,
                'best_train_formula': 'R0',
                'case_type': 'control_R0'
            })
        
        # R4 train-best cases
        r4_cases = seed_labels[seed_labels['best_train_formula'] == 'R4'].head(n_per_seed)
        for _, row in r4_cases.iterrows():
            control_cases.append({
                'basin_id': row['basin_id'],
                'seed': seed,
                'best_train_formula': 'R4',
                'case_type': 'control_R4'
            })
    
    return pd.DataFrame(control_cases)


def run_calibration_budget_sensitivity(base_dir: Path, audit_dir: Path, output_dir: Path):
    """Run calibration with different budgets."""
    print("Loading R5 cases...")
    r5_cases = load_r5_cases(audit_dir)
    if r5_cases.empty:
        return
    
    print("Loading baseline data...")
    oracle_labels = load_baseline_data(base_dir)
    if oracle_labels.empty:
        return
    
    # Select control cases
    control_cases = select_control_cases(oracle_labels, r5_cases)
    
    # Combine R5 and control cases
    all_cases = pd.concat([
        r5_cases[['basin_id', 'seed', 'best_train_formula']].assign(case_type='R5_train_best'),
        control_cases
    ], ignore_index=True)
    
    print(f"Total cases to evaluate: {len(all_cases)}")
    print(f"  R5 train-best: {len(r5_cases)}")
    print(f"  Control R0: {len(control_cases[control_cases['case_type'] == 'control_R0'])}")
    print(f"  Control R4: {len(control_cases[control_cases['case_type'] == 'control_R4'])}")
    
    # Define budget configurations
    budgets = [
        {'name': 'A', 'steps': 300, 'lr': 0.01},
        {'name': 'B', 'steps': 600, 'lr': 0.005},
        {'name': 'C', 'steps': 1000, 'lr': 0.005},
    ]
    
    # For now, create a placeholder structure
    # In a real implementation, this would call the calibration script
    # with different --steps and --lr arguments
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary report
    report_lines = []
    report_lines.append("# Stage 4: Calibration Budget Sensitivity Report\n")
    
    report_lines.append("## 1. Budget Configurations\n")
    report_lines.append("| Config | Steps | LR | Description |")
    report_lines.append("|--------|-------|-----|-------------|")
    report_lines.append("| Baseline | 100 | 0.01 | Original calibration |")
    for budget in budgets:
        report_lines.append(f"| {budget['name']} | {budget['steps']} | {budget['lr']} | Extended calibration |")
    
    report_lines.append("\n## 2. Cases to Evaluate\n")
    report_lines.append(f"- R5 train-best cases: {len(r5_cases)}")
    report_lines.append(f"- Control R0 cases: {len(control_cases[control_cases['case_type'] == 'control_R0'])}")
    report_lines.append(f"- Control R4 cases: {len(control_cases[control_cases['case_type'] == 'control_R4'])}")
    report_lines.append(f"- Total: {len(all_cases)}")
    
    report_lines.append("\n## 3. Analysis Plan\n")
    report_lines.append("For each case, we will:")
    report_lines.append("1. Run calibration with each budget configuration")
    report_lines.append("2. Record train and eval NSE")
    report_lines.append("3. Check if R5 remains train-best under increased budget")
    report_lines.append("4. Analyze rank stability across budgets")
    
    report_lines.append("\n## 4. Expected Outcomes\n")
    report_lines.append("- If R5 remains best under budget C: R5 label is budget-stable")
    report_lines.append("- If R5 loses to R0/R4 under budget C: Original R5 label was calibration artifact")
    report_lines.append("- If rank changes unpredictably: Labels are budget-sensitive")
    
    report_lines.append("\n## 5. Status\n")
    report_lines.append("**PENDING**: Full budget sensitivity analysis requires running calibration script")
    report_lines.append("with different --steps and --lr parameters.")
    report_lines.append("\nTo run manually:")
    report_lines.append("```bash")
    report_lines.append("# For each budget configuration:")
    report_lines.append("python scripts/calibrate_fixed_recharge_formulas_20basin.py \\")
    report_lines.append("  --steps 300 --lr 0.01 --seed 0 \\")
    report_lines.append("  --output-dir validation_results/r5_oracle_reliability_audit_20basin/budget_A_seed0")
    report_lines.append("```")
    
    with open(output_dir / "04_budget_sensitivity_report.md", "w") as f:
        f.write("\n".join(report_lines))
    
    # Save case list
    all_cases.to_csv(output_dir / "04_budget_sensitivity_cases.csv", index=False)
    
    print(f"\nReport saved to {output_dir}")
    print("Note: Full budget sensitivity analysis requires running calibration with extended budgets.")


def main():
    parser = argparse.ArgumentParser(description="Run calibration budget sensitivity analysis")
    parser.add_argument("--base-dir", type=str,
                        default="validation_results/static_router_20basin_calibrated",
                        help="Baseline results directory")
    parser.add_argument("--audit-dir", type=str,
                        default="validation_results/r5_oracle_reliability_audit_20basin",
                        help="Audit results directory")
    parser.add_argument("--output-dir", type=str,
                        default="validation_results/r5_oracle_reliability_audit_20basin",
                        help="Output directory")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    run_calibration_budget_sensitivity(base_dir, audit_dir, output_dir)


if __name__ == "__main__":
    main()
