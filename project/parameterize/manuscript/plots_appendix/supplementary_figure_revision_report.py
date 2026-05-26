"""Generate the revision report for supplementary Figures S14 and S15."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figS14_leave_group_out_sensitivity import (  # noqa: E402
    GROUP_ORDER,
    INSUFFICIENT_N,
    LEAVE_GROUP_OUT_FILE,
    RELATIONSHIP_ORDER,
    WITHIN_GROUP_FILE,
    load_leave_group_out_data,
    load_within_group_data,
)
from figS15_uncertainty_coupling_boundary import (  # noqa: E402
    DIAGNOSTIC_FILE,
    label_parameter,
    load_uncertainty_diagnostics,
)
from common_appendix import APP_FIG_DIR  # noqa: E402

REPORT_FILE = APP_FIG_DIR / "supplementary_figure_revision_report.md"


def format_rows(rows: pd.DataFrame, columns: list[str]) -> str:
    """Return a compact markdown table or a none message."""
    if rows.empty:
        return "None.\n"
    table = rows[columns].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else str(value))
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    leave_out = load_leave_group_out_data()
    within_group = load_within_group_data()
    diagnostics = load_uncertainty_diagnostics()

    leave_large_delta = leave_out.loc[leave_out["delta_vs_all"].abs() >= 0.15].copy()
    within_sign_changes = within_group.loc[~within_group["sign_matches_all"].astype(bool)].copy()
    within_na = within_group.loc[within_group["within_group_rho"].isna()].copy()
    within_insufficient = within_group.loc[within_group["within_group_n"] < INSUFFICIENT_N].copy()

    less_confounded = diagnostics.loc[diagnostics["diagnostic_class"].astype(str) == "less-confounded"].copy()
    cautionary = diagnostics.loc[diagnostics["diagnostic_class"].astype(str) != "less-confounded"].copy()

    diagnostics["parameter_label_report"] = diagnostics["parameter"].map(label_parameter)
    less_confounded["parameter_label_report"] = less_confounded["parameter"].map(label_parameter)
    cautionary["parameter_label_report"] = cautionary["parameter"].map(label_parameter)

    generated_paths = [
        APP_FIG_DIR / "figS14_leave_group_out_and_within_group_sensitivity.png",
        APP_FIG_DIR / "figS14_leave_group_out_and_within_group_sensitivity.pdf",
        APP_FIG_DIR / "figS14_data_used.csv",
        APP_FIG_DIR / "figS14_caption_note.md",
        APP_FIG_DIR / "figS15_uncertainty_coupling_boundary_abs.png",
        APP_FIG_DIR / "figS15_uncertainty_coupling_boundary_abs.pdf",
        APP_FIG_DIR / "figS15_data_used.csv",
        APP_FIG_DIR / "figS15_caption_note.md",
        REPORT_FILE,
    ]

    report = f"""# Supplementary Figure Revision Report

## Input Files Used

- `{LEAVE_GROUP_OUT_FILE}`
- `{WITHIN_GROUP_FILE}`
- `{DIAGNOSTIC_FILE}`

No model retraining was performed. Existing CSV / processed output files were used only.

## Figure S14 Coverage

- Selected relationships: {len(RELATIONSHIP_ORDER)}
- Hydroclimatic groups: {len(GROUP_ORDER)}
- Leave-one-group-out rows: {len(leave_out)}
- Within-group rows: {len(within_group)}
- NA within-group cells: {int(within_na.shape[0])}
- Within-group cells with n < {INSUFFICIENT_N}: {int(within_insufficient.shape[0])}

## S14 Sign and Magnitude Diagnostics

Within-group relationships that changed sign relative to the full-sample relationship:

{format_rows(within_sign_changes, ["relationship_label", "group_id", "group_name_figure4", "all_basins_rho", "within_group_rho", "within_group_n"])}

Leave-one-group-out rho values with |delta rho| >= 0.15 relative to the full-sample relationship:

{format_rows(leave_large_delta, ["relationship_label", "group_id", "group_name_figure4", "all_basins_rho", "leave_group_out_rho", "delta_vs_all"])}

Interpretation constraint: Figure S14 is a hydroclimatic-group sensitivity diagnostic, not ungauged validation.

## Figure S15 Coverage

- Parameters included: {diagnostics["parameter"].nunique()}
- Diagnostic rows: {len(diagnostics)}
- Missing diagnostic values: {int(diagnostics[["spearman_rho", "mean_std_spearman", "boundary_distance_std_spearman", "near_boundary_share"]].isna().sum().sum())}

Less-confounded uncertainty parameters:

{format_rows(less_confounded, ["parameter_label_report", "attribute", "spearman_rho", "abs_mean_std_spearman", "abs_boundary_distance_std_spearman", "near_boundary_share", "diagnostic_class"])}

Mean-coupled or boundary-sensitive uncertainty parameters:

{format_rows(cautionary, ["parameter_label_report", "attribute", "spearman_rho", "abs_mean_std_spearman", "abs_boundary_distance_std_spearman", "near_boundary_share", "diagnostic_class"])}

## Generated Outputs

""" + "".join(f"- `{path}`\n" for path in generated_paths)

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
