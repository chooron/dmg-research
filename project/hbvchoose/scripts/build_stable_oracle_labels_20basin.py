#!/usr/bin/env python3
"""Stage 8: Build stable oracle label set.

Creates filtered oracle labels based on reliability classification:
- Hard labels: RELIABLE only
- Soft labels: WEAK (optional)
- Excluded: AMBIGUOUS, UNRELIABLE, OVERFIT_RISK
"""

import argparse
import pandas as pd
from pathlib import Path


def build_stable_labels(audit_dir: Path, base_dir: Path, output_dir: Path):
    """Build stable oracle label set."""
    print("Loading reliability classification...")
    class_file = audit_dir / "07_oracle_label_reliability.csv"
    if not class_file.exists():
        print(f"Error: Classification file not found: {class_file}")
        return
    
    class_df = pd.read_csv(class_file)
    
    # Load original oracle labels
    oracle_file = base_dir / "oracle_labels_train.csv"
    oracle_df = pd.read_csv(oracle_file)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge classification with oracle labels
    merged = oracle_df.merge(
        class_df[['basin_id', 'seed', 'reliability_class', 'recommended_use']],
        on=['basin_id', 'seed'],
        how='left'
    )
    
    # Hard labels: RELIABLE only
    hard_labels = merged[merged['reliability_class'] == 'RELIABLE'].copy()
    hard_labels.to_csv(output_dir / "08_stable_oracle_labels_hard.csv", index=False)
    
    # Soft labels: WEAK
    soft_labels = merged[merged['reliability_class'] == 'WEAK'].copy()
    soft_labels.to_csv(output_dir / "08_stable_oracle_labels_soft.csv", index=False)
    
    # Excluded labels
    excluded = merged[merged['reliability_class'].isin(['AMBIGUOUS', 'UNRELIABLE', 'OVERFIT_RISK'])].copy()
    excluded.to_csv(output_dir / "08_excluded_ambiguous_labels.csv", index=False)
    
    # Generate summary
    report_lines = []
    report_lines.append("# Stage 8: Stable Oracle Labels Summary\n")
    
    report_lines.append("## 1. Label Counts\n")
    report_lines.append(f"- Original labels: {len(oracle_df)}")
    report_lines.append(f"- Hard labels (RELIABLE): {len(hard_labels)}")
    report_lines.append(f"- Soft labels (WEAK): {len(soft_labels)}")
    report_lines.append(f"- Excluded: {len(excluded)}")
    report_lines.append(f"- Hard label fraction: {len(hard_labels)/len(oracle_df):.3f}")
    
    report_lines.append("\n## 2. Hard Label Distribution by Formula\n")
    if not hard_labels.empty:
        hard_formula = hard_labels['best_train_formula'].value_counts()
        for formula, count in hard_formula.items():
            report_lines.append(f"- {formula}: {count}")
    
    report_lines.append("\n## 3. R5 in Stable Labels\n")
    r5_hard = len(hard_labels[hard_labels['best_train_formula'] == 'R5'])
    r5_soft = len(soft_labels[soft_labels['best_train_formula'] == 'R5'])
    r5_excluded = len(excluded[excluded['best_train_formula'] == 'R5'])
    report_lines.append(f"- R5 hard labels: {r5_hard}")
    report_lines.append(f"- R5 soft labels: {r5_soft}")
    report_lines.append(f"- R5 excluded: {r5_excluded}")
    
    report_lines.append("\n## 4. Exclusion Reasons\n")
    if not excluded.empty:
        exclusion_reasons = excluded['reliability_class'].value_counts()
        for reason, count in exclusion_reasons.items():
            report_lines.append(f"- {reason}: {count}")
    
    report_lines.append("\n## 5. Recommendation\n")
    
    if len(hard_labels) >= 30:
        report_lines.append("**Sufficient hard labels for router training.**")
        report_lines.append(f"Use {len(hard_labels)} hard labels for stable training.")
    elif len(hard_labels) + len(soft_labels) >= 30:
        report_lines.append("**Use hard + soft labels for router training.**")
        report_lines.append(f"Combined: {len(hard_labels) + len(soft_labels)} labels.")
    else:
        report_lines.append("**Insufficient stable labels.**")
        report_lines.append("Consider recalibrating with higher budget or different method.")
    
    with open(output_dir / "08_stable_oracle_summary.md", "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nResults saved to {output_dir}")
    print(f"Hard labels: {len(hard_labels)}")
    print(f"Soft labels: {len(soft_labels)}")
    print(f"Excluded: {len(excluded)}")
    print(f"R5 hard labels: {r5_hard}")
    
    return hard_labels, soft_labels, excluded


def main():
    parser = argparse.ArgumentParser(description="Build stable oracle labels")
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
    
    build_stable_labels(audit_dir, base_dir, output_dir)


if __name__ == "__main__":
    main()
