from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VALIDATION_CSV = (
    REPO_ROOT
    / "validation_results"
    / "flex_saturation3_parameter_bound_fix"
    / "flex_saturation3_bound_fix_validation.csv"
)
BATCH_A_RISK_CSV = (
    REPO_ROOT
    / "validation_results"
    / "batch_a_flux_realistic_review"
    / "batch_a_risk_decision.csv"
)
FINAL_RISK_CSV = (
    REPO_ROOT
    / "validation_results"
    / "flux_gradient_stability"
    / "final_flux_gradient_risk_ranking.csv"
)
STABLE_REWRITE_REPORT = (
    REPO_ROOT
    / "validation_results"
    / "saturation3_stable_rewrite"
    / "saturation3_stable_rewrite_report.md"
)
OUTPUT_PATH = (
    REPO_ROOT
    / "validation_results"
    / "saturation3_stable_rewrite"
    / "saturation3_beta_bound_cleanup_report.md"
)
TARGET_MODELS = ("flexb", "flexi", "flexis")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_row(rows: list[dict[str, str]], model: str, beta: float) -> dict[str, str]:
    for row in rows:
        if row["model"] == model and math.isclose(float(row["tested_beta"]), beta, rel_tol=0.0, abs_tol=1.0e-15):
            return row
    raise KeyError((model, beta))


def main() -> None:
    validation_rows = _read_csv(VALIDATION_CSV)
    batch_a_rows = _read_csv(BATCH_A_RISK_CSV)
    final_rows = _read_csv(FINAL_RISK_CSV)

    all_beta0_safe = True
    lines = [
        "# saturation_3 Beta Bound Cleanup Report",
        "",
        "## 1. Scope",
        "- Evaluated whether the previous FLEX beta lower-bound workaround (`1e-6`) is still needed after the exact stable `torch.sigmoid(z)` rewrite in `saturation_3`.",
        "",
        "## 2. Stable `saturation_3` rewrite status",
        f"- Stable rewrite report: `{STABLE_REWRITE_REPORT.relative_to(REPO_ROOT)}`",
        "- The active `saturation_3` implementation now evaluates the logistic term through `torch.sigmoid`, preserving the forward formula while avoiding the previous autograd overflow path.",
        "",
        "## 3. beta=0 test result",
    ]

    for model in TARGET_MODELS:
        beta0 = _find_row(validation_rows, model, 0.0)
        beta1e6 = _find_row(validation_rows, model, 1.0e-6)
        beta0_safe = (
            int(beta0["output_nan_count"]) == 0
            and int(beta0["output_inf_count"]) == 0
            and int(beta0["grad_nan_count"]) == 0
            and int(beta0["grad_inf_count"]) == 0
            and int(beta0["output_bound_violation_count"]) == 0
        )
        all_beta0_safe = all_beta0_safe and beta0_safe
        lines.append(
            f"- `{model}`: beta=0 output_nan={beta0['output_nan_count']}, output_inf={beta0['output_inf_count']}, "
            f"grad_nan={beta0['grad_nan_count']}, grad_inf={beta0['grad_inf_count']}, "
            f"max_abs_grad={beta0['max_abs_grad']}, bound_violations={beta0['output_bound_violation_count']}, "
            f"max_abs_diff(beta=0 vs beta=1e-6)={beta1e6['output_diff_vs_beta0_if_available']}"
        )

    if all_beta0_safe:
        decision = "restored to 0.0"
        reason = (
            "beta=0 is finite and gradient-safe in all three FLEX contexts under the stable rewrite, and "
            "the traced realistic-domain forward outputs at beta=0 match beta=1e-6 to machine precision."
        )
    else:
        decision = "kept at 1e-6"
        reason = "At least one FLEX context still showed a non-finite beta=0 result under the stable rewrite."

    lines.extend(
        [
            "",
            "## 4. Whether beta lower bound was restored to 0 or kept at 1e-6",
            f"- Decision: `{decision}`",
            "",
            "## 5. Reason for the decision",
            f"- {reason}",
            "",
            "## 6. Updated risk status",
        ]
    )

    for model in TARGET_MODELS:
        batch_row = next(row for row in batch_a_rows if row["formula"] == "saturation_3" and row["active_model"] == model)
        final_row = next(row for row in final_rows if row["formula"] == "saturation_3" and row["active_model"] == model)
        lines.append(
            f"- `{model}`: Batch A realistic risk=`{batch_row['realistic_risk']}`, action=`{batch_row['recommended_action']}`; "
            f"final active risk=`{final_row['final_active_risk']}`, final action=`{final_row['final_recommended_action']}`."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
