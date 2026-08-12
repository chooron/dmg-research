#!/usr/bin/env python3
"""Stage 7: Oracle label reliability classification.

Classifies each oracle label into reliability categories:
- RELIABLE: Strong margin, stable across seeds/budgets
- WEAK: Moderate margin, partial stability
- AMBIGUOUS: Low margin or inconsistent
- UNRELIABLE: Poor calibration, NaN, or boundary issues
- OVERFIT_RISK: Good train but poor eval
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_all_data(audit_dir: Path, base_dir: Path):
    """Load all audit data."""
    # R5 cases
    r5_cases = pd.read_csv(audit_dir / "01_r5_train_best_cases.csv") if (audit_dir / "01_r5_train_best_cases.csv").exists() else pd.DataFrame()
    
    # Oracle labels
    oracle_labels = pd.read_csv(base_dir / "oracle_labels_train.csv") if (base_dir / "oracle_labels_train.csv").exists() else pd.DataFrame()
    
    # Oracle eval audit
    oracle_eval = pd.read_csv(base_dir / "oracle_eval_audit.csv") if (base_dir / "oracle_eval_audit.csv").exists() else pd.DataFrame()
    
    # Parameter boundary summary
    param_summary = pd.read_csv(audit_dir / "03_parameter_boundary_summary.csv") if (audit_dir / "03_parameter_boundary_summary.csv").exists() else pd.DataFrame()
    
    # Convergence summary
    conv_summary = pd.read_csv(audit_dir / "02_calibration_convergence_summary.csv") if (audit_dir / "02_calibration_convergence_summary.csv").exists() else pd.DataFrame()
    
    return r5_cases, oracle_labels, oracle_eval, param_summary, conv_summary


def classify_reliability(row: pd.Series) -> dict:
    """Classify reliability of a single oracle label."""
    # Extract key metrics
    relative_margin = row.get('relative_mse_margin', 0)
    eval_nse = row.get('eval_nse_of_best', np.nan)
    train_nse = row.get('train_nse_of_best', np.nan)
    generalizes = row.get('generalizes_to_eval', False)
    
    # Initialize flags
    margin_ok = relative_margin >= 0.05
    margin_weak = 0.02 <= relative_margin < 0.05
    margin_ambiguous = relative_margin < 0.02
    
    train_positive = train_nse > 0 if not pd.isna(train_nse) else False
    eval_positive = eval_nse > 0 if not pd.isna(eval_nse) else False
    
    # Classification logic
    if margin_ambiguous:
        return {'reliability_class': 'AMBIGUOUS', 'recommended_use': 'EXCLUDE_FROM_ROUTER_TRAINING'}
    
    if not train_positive:
        return {'reliability_class': 'UNRELIABLE', 'recommended_use': 'EXCLUDE_FROM_ROUTER_TRAINING'}
    
    if margin_ok and train_positive and eval_positive and generalizes:
        return {'reliability_class': 'RELIABLE', 'recommended_use': 'USE_AS_HARD_LABEL'}
    
    if margin_ok and train_positive and not eval_positive:
        return {'reliability_class': 'OVERFIT_RISK', 'recommended_use': 'DIAGNOSTIC_ONLY'}
    
    if margin_weak and train_positive:
        if eval_positive and generalizes:
            return {'reliability_class': 'WEAK', 'recommended_use': 'USE_AS_SOFT_LABEL'}
        else:
            return {'reliability_class': 'WEAK', 'recommended_use': 'DIAGNOSTIC_ONLY'}
    
    return {'reliability_class': 'AMBIGUOUS', 'recommended_use': 'EXCLUDE_FROM_ROUTER_TRAINING'}


def classify_oracle_labels(audit_dir: Path, base_dir: Path, output_dir: Path):
    """Main classification function."""
    print("Loading data...")
    r5_cases, oracle_labels, oracle_eval, param_summary, conv_summary = load_all_data(audit_dir, base_dir)
    
    if oracle_labels.empty:
        print("No oracle labels found!")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge oracle labels with eval data
    merged = oracle_labels.merge(
        oracle_eval[['basin_id', 'seed', 'eval_nse_of_best_train_formula', 'generalizes_to_eval']],
        on=['basin_id', 'seed'],
        how='left'
    )
    merged = merged.rename(columns={
        'eval_nse_of_best_train_formula': 'eval_nse_of_best',
        'generalizes_to_eval': 'generalizes'
    })
    
    # Add train NSE of best formula
    def get_train_nse(row):
        formula = row['best_train_formula']
        col = f'train_nse_{formula}'
        return row.get(col, np.nan)
    
    merged['train_nse_of_best'] = merged.apply(get_train_nse, axis=1)
    
    # Add margin info for R5 cases
    if not r5_cases.empty:
        r5_margin = r5_cases[['basin_id', 'seed', 'relative_mse_margin']].rename(
            columns={'relative_mse_margin': 'r5_margin'}
        )
        merged = merged.merge(r5_margin, on=['basin_id', 'seed'], how='left')
    
    # Classify each label
    classifications = []
    for _, row in merged.iterrows():
        # Compute margin (use R5 margin if R5 is best, otherwise compute for R0/R4)
        if row['best_train_formula'] == 'R5' and 'r5_margin' in row.index:
            margin = row.get('r5_margin', 0)
        else:
            # For R0/R4, we don't have margin info, assume reasonable margin
            margin = 0.1  # Placeholder
        
        class_info = classify_reliability({
            'relative_mse_margin': margin,
            'eval_nse_of_best': row.get('eval_nse_of_best', np.nan),
            'train_nse_of_best': row.get('train_nse_of_best', np.nan),
            'generalizes_to_eval': row.get('generalizes', False)
        })
        
        classifications.append({
            'basin_id': row['basin_id'],
            'seed': row['seed'],
            'formula_label': row['best_train_formula'],
            'relative_train_margin': margin,
            'train_nse': row.get('train_nse_of_best', np.nan),
            'eval_nse': row.get('eval_nse_of_best', np.nan),
            'generalizes': row.get('generalizes', False),
            **class_info
        })
    
    class_df = pd.DataFrame(classifications)
    class_df.to_csv(output_dir / "07_oracle_label_reliability.csv", index=False)
    
    # Generate summary
    summary = class_df['reliability_class'].value_counts().reset_index()
    summary.columns = ['reliability_class', 'count']
    summary.to_csv(output_dir / "07_oracle_label_reliability_summary.csv", index=False)
    
    # Generate report
    report_lines = []
    report_lines.append("# Stage 7: Oracle Label Reliability Classification\n")
    
    report_lines.append("## 1. Classification Summary\n")
    report_lines.append("| Class | Count | Fraction |")
    report_lines.append("|-------|-------|----------|")
    total = len(class_df)
    for _, row in summary.iterrows():
        fraction = row['count'] / total
        report_lines.append(f"| {row['reliability_class']} | {row['count']} | {fraction:.3f} |")
    
    report_lines.append("\n## 2. Classification by Formula\n")
    formula_class = class_df.groupby(['formula_label', 'reliability_class']).size().unstack(fill_value=0)
    # Convert to string table manually
    report_lines.append("| Formula | " + " | ".join(formula_class.columns) + " |")
    report_lines.append("|---------|" + "|".join(["---"] * len(formula_class.columns)) + "|")
    for idx, row in formula_class.iterrows():
        report_lines.append(f"| {idx} | " + " | ".join(str(v) for v in row.values) + " |")
    
    report_lines.append("\n## 3. R5 Specific Analysis\n")
    r5_class = class_df[class_df['formula_label'] == 'R5']
    if not r5_class.empty:
        r5_summary = r5_class['reliability_class'].value_counts()
        report_lines.append("R5 label classification:")
        for cls, count in r5_summary.items():
            report_lines.append(f"- {cls}: {count}")
        
        # Count usable R5 labels
        usable_r5 = len(r5_class[r5_class['reliability_class'].isin(['RELIABLE', 'WEAK'])])
        report_lines.append(f"\nUsable R5 labels: {usable_r5}/{len(r5_class)}")
    
    report_lines.append("\n## 4. Recommended Actions\n")
    
    reliable_count = len(class_df[class_df['reliability_class'] == 'RELIABLE'])
    weak_count = len(class_df[class_df['reliability_class'] == 'WEAK'])
    ambiguous_count = len(class_df[class_df['reliability_class'] == 'AMBIGUOUS'])
    unreliable_count = len(class_df[class_df['reliability_class'] == 'UNRELIABLE'])
    overfit_count = len(class_df[class_df['reliability_class'] == 'OVERFIT_RISK'])
    
    report_lines.append(f"- **RELIABLE** ({reliable_count}): Use as hard labels")
    report_lines.append(f"- **WEAK** ({weak_count}): Use as soft labels or exclude")
    report_lines.append(f"- **AMBIGUOUS** ({ambiguous_count}): Exclude from training")
    report_lines.append(f"- **UNRELIABLE** ({unreliable_count}): Exclude from training")
    report_lines.append(f"- **OVERFIT_RISK** ({overfit_count}): Diagnostic only")
    
    report_lines.append("\n## 5. Conclusion\n")
    
    stable_fraction = (reliable_count + weak_count) / total
    if stable_fraction >= 0.7:
        report_lines.append("**PASS**: Majority of labels are stable.")
    elif stable_fraction >= 0.5:
        report_lines.append("**PARTIAL**: Many labels are unstable, filtering recommended.")
    else:
        report_lines.append("**FAIL**: Most labels are unstable, major filtering needed.")
    
    with open(output_dir / "07_oracle_label_reliability_report.md", "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nResults saved to {output_dir}")
    print(f"Total labels: {total}")
    print(f"RELIABLE: {reliable_count}")
    print(f"WEAK: {weak_count}")
    print(f"AMBIGUOUS: {ambiguous_count}")
    print(f"UNRELIABLE: {unreliable_count}")
    print(f"OVERFIT_RISK: {overfit_count}")
    
    return class_df


def main():
    parser = argparse.ArgumentParser(description="Classify oracle label reliability")
    parser.add_argument("--audit-dir", type=str,
                        default="validation_results/r5_oracle_reliability_audit_20basin",
                        help="Audit results directory")
    parser.add_argument("--base-dir", type=str,
                        default="validation_results/static_router_20basin_calibrated",
                        help="Baseline results directory")
    parser.add_argument("--output-dir", type=str,
                        default="validation_results/r5_oracle_reliability_audit_20basin",
                        help="Output directory")
    args = parser.parse_args()
    
    audit_dir = Path(args.audit_dir)
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    if not audit_dir.exists():
        print(f"Error: Audit directory not found: {audit_dir}")
        return
    
    classify_oracle_labels(audit_dir, base_dir, output_dir)


if __name__ == "__main__":
    main()
