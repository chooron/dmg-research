from __future__ import annotations

import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.training_regression_utils import (  # noqa: E402
    ALL_MODEL_FIELDNAMES,
    ALL_MODEL_SMOKE_SUMMARY_CSV,
    FLEX_FIELDNAMES,
    FLEX_SATURATION3_TRAINING_CSV,
    MEDIUM_CONTEXT_MONITORING_CSV,
    MEDIUM_FIELDNAMES,
    OUTPUT_DIR,
    REPORT_MD_PATH,
    TRAINING_SMOKE_FIELDNAMES,
    TRAINING_SMOKE_SUMMARY_CSV,
    build_all_model_smoke_rows,
    build_flex_regression_rows,
    build_medium_context_rows,
    build_training_smoke_rows,
    write_csv,
)


def _count(rows: list[dict], key: str, expected) -> int:
    return sum(1 for row in rows if row.get(key) == expected)


def _max_float(rows: list[dict], key: str) -> float:
    return max(float(row.get(key, 0.0)) for row in rows) if rows else 0.0


def _report_text(
    training_rows: list[dict],
    flex_rows: list[dict],
    medium_rows: list[dict],
    all_model_rows: list[dict],
) -> str:
    training_failed = [row for row in training_rows if row["status"] != "passed"]
    flex_failed = [row for row in flex_rows if row["status"] != "passed"]
    medium_failed = [row for row in medium_rows if row["status"] != "passed"]
    all_model_failed = [row for row in all_model_rows if row["status"] != "passed"]

    total_loss_nan = sum(int(row["loss_nan_count"]) for row in training_rows + flex_rows)
    total_loss_inf = sum(int(row["loss_inf_count"]) for row in training_rows + flex_rows)
    total_grad_nan = sum(int(row["grad_nan_count"]) for row in training_rows + flex_rows + medium_rows + all_model_rows)
    total_grad_inf = sum(int(row["grad_inf_count"]) for row in training_rows + flex_rows + medium_rows + all_model_rows)
    total_optimizer_failures = sum(not bool(row["optimizer_step_success"]) for row in training_rows + medium_rows)

    can_proceed = not (training_failed or flex_failed or medium_failed or all_model_failed)
    formula_change_line = (
        "- No formula modification is needed."
        if can_proceed
        else "- No formula modification is recommended automatically; investigate the failed training cases first."
    )
    next_step_line = (
        "- Proceed to the heavier benchmark/calibration runs, keeping the three documented medium contexts in monitoring-only status."
        if can_proceed
        else "- Resolve the failed or suspicious training cases before running the full benchmark/calibration campaign."
    )

    lines = [
        "# Training Regression After Validation Report",
        "",
        "## 1. Scope",
        "- This regression checks short deterministic calibration loops after the formula-level, gradient-level, and water-balance validations were completed.",
        "- No hydrological formulas, bounds, clamps, smoothing rules, or model physics were changed in this workflow.",
        "",
        "## 2. Validation status before training",
        "- End-to-end gradient finite test: 36/36 models passed.",
        "- Representative `torch.autograd.gradcheck`: 9/9 models passed.",
        "- Water-balance regression: 36 models, 188 cases, 0 failures, max residual 1.114e-03.",
        "- Active unresolved high-risk flux contexts: 0.",
        "- Remaining documented medium contexts only: `baseflow_6 / tcm`, `baseflow_9 / gsfb`, `interflow_10 / topmodel`.",
        "",
        "## 3. Models tested",
        f"- Training smoke target set: {', '.join(row['model'] for row in training_rows)}.",
        f"- FLEX saturation_3 regression set: {', '.join(row['model'] for row in flex_rows)}.",
        f"- Medium-context monitoring set: {', '.join(row['model'] for row in medium_rows)}.",
        f"- Optional all-model smoke set: {len(all_model_rows)} runnable models.",
        "",
        "## 4. Training smoke test result",
        f"- Passed models: {_count(training_rows, 'status', 'passed')} / {len(training_rows)}",
        f"- Loss NaN count: {sum(int(row['loss_nan_count']) for row in training_rows)}",
        f"- Loss Inf count: {sum(int(row['loss_inf_count']) for row in training_rows)}",
        f"- Gradient NaN count: {sum(int(row['grad_nan_count']) for row in training_rows)}",
        f"- Gradient Inf count: {sum(int(row['grad_inf_count']) for row in training_rows)}",
        f"- Optimizer step failures: {sum(not bool(row['optimizer_step_success']) for row in training_rows)}",
        f"- Worst max|grad|: {_max_float(training_rows, 'max_abs_grad'):.3e}",
        "",
        "## 5. FLEX saturation_3 training regression",
        f"- Passed models: {_count(flex_rows, 'status', 'passed')} / {len(flex_rows)}",
        f"- Any beta reached exact 0.0: {any(bool(row['beta_reaches_zero']) for row in flex_rows)}",
        f"- Near-zero-beta steps with nonfinite gradients: {sum((not bool(row['beta_near_zero_grad_finite'])) for row in flex_rows)}",
        f"- Mean synthetic NSE after training: {sum(float(row['synthetic_nse']) for row in flex_rows) / len(flex_rows):.3f}",
        f"- Mean synthetic KGE after training: {sum(float(row['synthetic_kge']) for row in flex_rows) / len(flex_rows):.3f}",
        "",
        "## 6. Medium-context monitoring result",
        f"- Passed contexts: {_count(medium_rows, 'status', 'passed')} / {len(medium_rows)}",
        f"- Output NaN count: {sum(int(row['output_nan_count']) for row in medium_rows)}",
        f"- Output Inf count: {sum(int(row['output_inf_count']) for row in medium_rows)}",
        f"- Gradient NaN count: {sum(int(row['grad_nan_count']) for row in medium_rows)}",
        f"- Gradient Inf count: {sum(int(row['grad_inf_count']) for row in medium_rows)}",
        f"- Worst monitored max|grad|: {_max_float(medium_rows, 'max_abs_grad_monitored'):.3e}",
        "",
        "## 7. All-model smoke result if performed",
        f"- Passed models: {_count(all_model_rows, 'status', 'passed')} / {len(all_model_rows)}",
        f"- Output NaN count: {sum(int(row['output_nan_count']) for row in all_model_rows)}",
        f"- Output Inf count: {sum(int(row['output_inf_count']) for row in all_model_rows)}",
        f"- Gradient NaN count: {sum(int(row['grad_nan_count']) for row in all_model_rows)}",
        f"- Gradient Inf count: {sum(int(row['grad_inf_count']) for row in all_model_rows)}",
        f"- Failed basin count total: {sum(int(row['failed_basin_count']) for row in all_model_rows)}",
        f"- Total runtime: {sum(float(row['runtime_seconds']) for row in all_model_rows):.2f} seconds",
        "",
        "## 8. Failed or suspicious cases",
    ]

    failed_rows = training_failed + flex_failed + medium_failed + all_model_failed
    if not failed_rows:
        lines.append("- None.")
    else:
        for row in failed_rows:
            lines.append(
                f"- `{row['model']}`: status={row['status']} failed_stage={row.get('failed_stage', '') or 'n/a'} notes={row['notes']}"
            )

    lines.extend(
        [
            "",
            "## 9. Whether any formula change is suggested",
            formula_change_line,
            "",
            "## 10. Recommended next step for full benchmark training",
            next_step_line,
            "",
            "## Direct Answers",
            f"- Can training proceed? {'yes' if can_proceed else 'no'}",
            f"- Did any model produce NaN/Inf losses? {'no' if (total_loss_nan == 0 and total_loss_inf == 0) else 'yes'}",
            f"- Did any model produce NaN/Inf gradients? {'no' if (total_grad_nan == 0 and total_grad_inf == 0) else 'yes'}",
            f"- Did any optimizer step fail? {'no' if total_optimizer_failures == 0 else 'yes'}",
            f"- Did FLEX models behave normally after the stable `saturation_3` rewrite? {'yes' if not flex_failed else 'no'}",
            f"- Do the documented medium contexts remain safe during training? {'yes' if not medium_failed else 'no'}",
            f"- Is any formula modification needed? {'no' if can_proceed else 'not concluded'}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_rows = build_training_smoke_rows()
    flex_rows = build_flex_regression_rows()
    medium_rows = build_medium_context_rows()
    all_model_rows = build_all_model_smoke_rows()

    write_csv(training_rows, TRAINING_SMOKE_SUMMARY_CSV, TRAINING_SMOKE_FIELDNAMES)
    write_csv(flex_rows, FLEX_SATURATION3_TRAINING_CSV, FLEX_FIELDNAMES)
    write_csv(medium_rows, MEDIUM_CONTEXT_MONITORING_CSV, MEDIUM_FIELDNAMES)
    write_csv(all_model_rows, ALL_MODEL_SMOKE_SUMMARY_CSV, ALL_MODEL_FIELDNAMES)
    REPORT_MD_PATH.write_text(_report_text(training_rows, flex_rows, medium_rows, all_model_rows), encoding="utf-8")

    print(f"Wrote CSV summary to {TRAINING_SMOKE_SUMMARY_CSV}")
    print(f"Wrote FLEX regression CSV to {FLEX_SATURATION3_TRAINING_CSV}")
    print(f"Wrote medium-context CSV to {MEDIUM_CONTEXT_MONITORING_CSV}")
    print(f"Wrote all-model CSV to {ALL_MODEL_SMOKE_SUMMARY_CSV}")
    print(f"Wrote markdown report to {REPORT_MD_PATH}")

    any_failure = any(row["status"] != "passed" for row in training_rows + flex_rows + medium_rows + all_model_rows)
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

