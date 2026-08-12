from __future__ import annotations

import csv
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "validation_results" / "model_gradcheck_water_balance_tests"
END_TO_END_CSV_PATH = OUTPUT_DIR / "model_gradient_end_to_end_summary.csv"
GRADCHECK_CSV_PATH = OUTPUT_DIR / "model_gradcheck_representative_summary.csv"
WATER_BALANCE_CSV_PATH = OUTPUT_DIR / "water_balance_pytest_summary.csv"
REPORT_PATH = OUTPUT_DIR / "model_gradcheck_water_balance_report.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def _int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in {"", None} else 0


def _count(rows: list[dict[str, str]], key: str, expected: str) -> int:
    return sum(1 for row in rows if row.get(key) == expected)


def _max_float(rows: list[dict[str, str]], key: str) -> float:
    values = [_float(row, key) for row in rows]
    return max(values) if values else 0.0


def _models(rows: list[dict[str, str]], key: str = "model") -> list[str]:
    return sorted({row[key] for row in rows})


def build_report() -> str:
    end_to_end_rows = _read_csv(END_TO_END_CSV_PATH)
    gradcheck_rows = _read_csv(GRADCHECK_CSV_PATH)
    water_balance_rows = _read_csv(WATER_BALANCE_CSV_PATH)

    end_to_end_pass = _count(end_to_end_rows, "status", "passed")
    end_to_end_skip = _count(end_to_end_rows, "status", "expected_skip")
    end_to_end_fail = _count(end_to_end_rows, "status", "failed")
    end_to_end_output_nan = sum(_int(row, "output_nan_count") for row in end_to_end_rows)
    end_to_end_output_inf = sum(_int(row, "output_inf_count") for row in end_to_end_rows)
    end_to_end_grad_nan = sum(_int(row, "grad_nan_count") for row in end_to_end_rows)
    end_to_end_grad_inf = sum(_int(row, "grad_inf_count") for row in end_to_end_rows)

    gradcheck_pass = _count(gradcheck_rows, "gradcheck_status", "gradcheck_pass")
    gradcheck_expected_nondiff = _count(
        gradcheck_rows, "gradcheck_status", "gradcheck_expected_nondifferentiable_point"
    )
    gradcheck_api_not_suitable = _count(gradcheck_rows, "gradcheck_status", "gradcheck_api_not_suitable")
    gradcheck_failed = _count(gradcheck_rows, "gradcheck_status", "gradcheck_failed_unexpectedly")
    unexpected_gradcheck_rows = [
        row
        for row in gradcheck_rows
        if row["failure_type"] in {"unexpected_nan_or_inf", "unexpected_exception", "unexpected_gradient_mismatch"}
    ]

    water_balance_pass = _count(water_balance_rows, "status", "passed")
    water_balance_fail = _count(water_balance_rows, "status", "failed")
    water_balance_max_residual = _max_float(water_balance_rows, "water_balance_residual")
    water_balance_max_negative_storage = _max_float(water_balance_rows, "max_negative_storage")
    water_balance_nan = sum(_int(row, "output_nan_count") for row in water_balance_rows)
    water_balance_inf = sum(_int(row, "output_inf_count") for row in water_balance_rows)

    all_end_to_end_finite = (
        end_to_end_fail == 0
        and end_to_end_output_nan == 0
        and end_to_end_output_inf == 0
        and end_to_end_grad_nan == 0
        and end_to_end_grad_inf == 0
    )
    water_balance_ok = water_balance_fail == 0 and water_balance_nan == 0 and water_balance_inf == 0
    only_expected_gradcheck_limitations = gradcheck_failed == 0

    if all_end_to_end_finite and water_balance_ok and only_expected_gradcheck_limitations:
        formula_change_line = "- No formula change is suggested."
        next_step_line = "- Keep these tests in the routine regression path and extend representative gradcheck coverage only when new models or routing logic are added."
    else:
        flagged_models = ", ".join(row["model"] for row in unexpected_gradcheck_rows) or "none"
        formula_change_line = (
            "- No hydrological formula change is recommended automatically here, but the unexpected gradcheck findings below need follow-up."
        )
        next_step_line = f"- Investigate the unexpected gradcheck follow-up set: {flagged_models}."

    lines = [
        "# Model Gradcheck And Water-Balance Report",
        "",
        "## 1. Scope",
        "- This report aggregates the pytest-based full-model gradient finite check, representative `torch.autograd.gradcheck`, and core water-balance regression.",
        "- All outputs are CPU `torch.float64` validations built from the active core registry and written under `validation_results/model_gradcheck_water_balance_tests`.",
        "",
        "## 2. Difference between end-to-end gradient finite test and torch.gradcheck",
        "- The end-to-end gradient test checks that a complete model rollout supports autograd, returns finite discharge, and produces finite parameter gradients for a scalar loss.",
        "- `torch.autograd.gradcheck` is stricter: it compares autograd against finite differences at a specific local operating point and can fail at piecewise thresholds even when routine training gradients remain usable.",
        "",
        "## 3. Models tested",
        f"- End-to-end gradient finite test: {len(end_to_end_rows)} models.",
        f"- Representative gradcheck subset: {len(gradcheck_rows)} models.",
        f"- Water-balance regression: {len(_models(water_balance_rows))} models across {len(water_balance_rows)} pytest cases.",
        "",
        "## 4. End-to-end gradient results",
        f"- Passed: {end_to_end_pass}",
        f"- Expected skips: {end_to_end_skip}",
        f"- Failed: {end_to_end_fail}",
        f"- Total discharge NaN count: {end_to_end_output_nan}",
        f"- Total discharge Inf count: {end_to_end_output_inf}",
        f"- Total gradient NaN count: {end_to_end_grad_nan}",
        f"- Total gradient Inf count: {end_to_end_grad_inf}",
        f"- Maximum absolute gradient magnitude observed: {_max_float(end_to_end_rows, 'max_abs_grad'):.3e}",
        "",
        "## 5. Representative gradcheck results",
        f"- Passed: {gradcheck_pass}",
        f"- Expected nondifferentiable-point classifications: {gradcheck_expected_nondiff}",
        f"- API-not-suitable classifications: {gradcheck_api_not_suitable}",
        f"- Unexpected failures: {gradcheck_failed}",
        f"- Maximum absolute base-point gradient magnitude observed: {_max_float(gradcheck_rows, 'max_abs_grad_if_available'):.3e}",
        "",
        "## 6. Water-balance regression results",
        f"- Passed cases: {water_balance_pass}",
        f"- Failed cases: {water_balance_fail}",
        f"- Maximum absolute water-balance residual: {water_balance_max_residual:.3e}",
        f"- Maximum negative storage violation: {water_balance_max_negative_storage:.3e}",
        f"- Output NaN count: {water_balance_nan}",
        f"- Output Inf count: {water_balance_inf}",
        "",
        "## 7. Expected limitations of gradcheck for threshold-based hydrological models",
        "- Thresholds, ReLU branches, and state updates can create locally non-smooth points where finite-difference checks are not representative of normal rollout behavior.",
        "- That limitation should be documented as a gradcheck boundary condition, not treated automatically as a formula bug, when failures occur only at those expected threshold points.",
        "",
        "## 8. Any failures or expected skips",
    ]

    if not end_to_end_fail and not gradcheck_expected_nondiff and not gradcheck_api_not_suitable and not gradcheck_failed and not water_balance_fail:
        lines.append("- None.")
    else:
        for row in end_to_end_rows:
            if row["status"] != "passed":
                lines.append(
                    f"- End-to-end `{row['model']}`: status={row['status']} stage={row['failed_stage']} notes={row['notes']}"
                )
        for row in gradcheck_rows:
            if row["gradcheck_status"] != "gradcheck_pass":
                lines.append(
                    f"- Gradcheck `{row['model']}`: status={row['gradcheck_status']} type={row['failure_type']} notes={row['notes'] or row['failure_message']}"
                )
        for row in water_balance_rows:
            if row["status"] != "passed":
                lines.append(
                    f"- Water balance `{row['case_id']}`: residual={float(row['water_balance_residual']):.3e} notes={row['notes']}"
                )

    lines.extend(
        [
            "",
            "## 9. Whether any formula change is suggested",
            formula_change_line,
            "",
            "## 10. Recommended next step",
            next_step_line,
            "",
            "## Models Tested",
            f"- End-to-end gradient set: {', '.join(_models(end_to_end_rows))}",
            f"- Representative gradcheck set: {', '.join(_models(gradcheck_rows))}",
            f"- Water-balance model set: {', '.join(_models(water_balance_rows))}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    required_paths = [END_TO_END_CSV_PATH, GRADCHECK_CSV_PATH, WATER_BALANCE_CSV_PATH]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing required input CSV: {path}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Wrote markdown report to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
