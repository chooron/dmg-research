from __future__ import annotations

import csv
import inspect
import math
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.flux.saturation import saturation_3  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "validation_results" / "saturation3_stable_rewrite"
TRACE_CSV = (
    REPO_ROOT
    / "validation_results"
    / "batch_a_flux_realistic_review"
    / "batch_a_realistic_domain_trace.csv"
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
DEFAULT_DTYPE = torch.float64
DEFAULT_NEARZERO = 1.0e-6
RELATIVE_STORAGE_VALUES = (
    0.0,
    1.0e-12,
    1.0e-9,
    1.0e-6,
    1.0e-4,
    1.0e-2,
    0.1,
    0.5,
    1.0,
    1.1,
)
TEST_BETAS = (
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    5.0000005,
)
REALISTIC_QUANTILES = ("min", "p01", "p05", "median", "mean", "p95", "p99", "max")
TARGET_MODELS = ("flexb", "flexi", "flexis")


def old_reference_expression(
    S: torch.Tensor,
    Smax: torch.Tensor,
    beta: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = DEFAULT_NEARZERO,
) -> torch.Tensor:
    ratio = S / (Smax + nearzero)
    z = (ratio + 0.5) / (beta + nearzero)
    return incoming_flux * (1.0 - 1.0 / (1.0 + torch.exp(z)))


def new_expression(
    S: torch.Tensor,
    Smax: torch.Tensor,
    beta: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = DEFAULT_NEARZERO,
) -> torch.Tensor:
    return saturation_3(S, Smax, beta, incoming_flux, nearzero=nearzero)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.12g}"


def _synthetic_points() -> list[dict[str, Any]]:
    Smax = 100.0
    points: list[dict[str, Any]] = []
    for x in RELATIVE_STORAGE_VALUES:
        S = x * (Smax + DEFAULT_NEARZERO)
        for beta in TEST_BETAS:
            points.append(
                {
                    "dataset": "synthetic_grid",
                    "model": "synthetic",
                    "forcing_regime": "synthetic",
                    "anchor": f"x={x:.12g}|beta={beta:.12g}",
                    "relative_storage": x,
                    "S": S,
                    "Smax": Smax,
                    "beta": beta,
                    "incoming_flux": 1.0,
                }
            )
    return points


def _realistic_points() -> list[dict[str, Any]]:
    if not TRACE_CSV.exists():
        return []
    rows = _read_csv(TRACE_CSV)
    points: list[dict[str, Any]] = []
    for model in TARGET_MODELS:
        model_rows = [row for row in rows if row["formula"] == "saturation_3" and row["active_model"] == model]
        for regime in sorted({row["forcing_regime"] for row in model_rows}):
            grouped = {row["argument_name"]: row for row in model_rows if row["forcing_regime"] == regime}
            if set(grouped) != {"S", "Smax", "p1", "incoming_flux"}:
                continue
            for quantile in REALISTIC_QUANTILES:
                S = float(grouped["S"][quantile])
                Smax = float(grouped["Smax"][quantile])
                points.append(
                    {
                        "dataset": "realistic_trace_anchor",
                        "model": model,
                        "forcing_regime": regime,
                        "anchor": quantile,
                        "relative_storage": S / (Smax + DEFAULT_NEARZERO),
                        "S": S,
                        "Smax": Smax,
                        "beta": float(grouped["p1"][quantile]),
                        "incoming_flux": float(grouped["incoming_flux"][quantile]),
                    }
                )
    return points


def _gradient_stats(
    expr,
    S_value: float,
    Smax_value: float,
    beta_value: float,
    incoming_flux_value: float,
) -> tuple[float, float, bool, bool]:
    S = torch.tensor([S_value], dtype=DEFAULT_DTYPE, requires_grad=True)
    Smax = torch.tensor([Smax_value], dtype=DEFAULT_DTYPE)
    beta = torch.tensor([beta_value], dtype=DEFAULT_DTYPE, requires_grad=True)
    incoming_flux = torch.tensor([incoming_flux_value], dtype=DEFAULT_DTYPE)
    output = expr(S, Smax, beta, incoming_flux)
    grad_S, grad_beta = torch.autograd.grad(output.sum(), (S, beta), allow_unused=True)
    grad_S_value = float("nan") if grad_S is None else float(grad_S.detach().item())
    grad_beta_value = float("nan") if grad_beta is None else float(grad_beta.detach().item())
    return (
        grad_S_value,
        grad_beta_value,
        math.isfinite(grad_S_value),
        math.isfinite(grad_beta_value),
    )


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forward_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    for point in _synthetic_points() + _realistic_points():
        S_value = float(point["S"])
        Smax_value = float(point["Smax"])
        beta_value = float(point["beta"])
        incoming_flux_value = float(point["incoming_flux"])

        S = torch.tensor([S_value], dtype=DEFAULT_DTYPE)
        Smax = torch.tensor([Smax_value], dtype=DEFAULT_DTYPE)
        beta = torch.tensor([beta_value], dtype=DEFAULT_DTYPE)
        incoming_flux = torch.tensor([incoming_flux_value], dtype=DEFAULT_DTYPE)

        old_output = float(old_reference_expression(S, Smax, beta, incoming_flux).item())
        new_output = float(new_expression(S, Smax, beta, incoming_flux).item())
        abs_diff = abs(new_output - old_output)
        rel_diff = abs_diff / max(abs(old_output), 1.0e-18)
        bounded = -1.0e-12 <= new_output <= incoming_flux_value + 1.0e-12

        old_grad_S, old_grad_beta, old_grad_S_finite, old_grad_beta_finite = _gradient_stats(
            old_reference_expression,
            S_value,
            Smax_value,
            beta_value,
            incoming_flux_value,
        )
        new_grad_S, new_grad_beta, new_grad_S_finite, new_grad_beta_finite = _gradient_stats(
            new_expression,
            S_value,
            Smax_value,
            beta_value,
            incoming_flux_value,
        )

        base = {
            "dataset": point["dataset"],
            "model": point["model"],
            "forcing_regime": point["forcing_regime"],
            "anchor": point["anchor"],
            "relative_storage": _format_float(float(point["relative_storage"])),
            "S": _format_float(S_value),
            "Smax": _format_float(Smax_value),
            "beta": _format_float(beta_value),
            "incoming_flux": _format_float(incoming_flux_value),
        }
        forward_rows.append(
            {
                **base,
                "old_output": _format_float(old_output),
                "new_output": _format_float(new_output),
                "abs_diff": _format_float(abs_diff),
                "rel_diff": _format_float(rel_diff),
                "old_output_finite": math.isfinite(old_output),
                "new_output_finite": math.isfinite(new_output),
                "new_output_bounded": bounded,
            }
        )
        gradient_rows.append(
            {
                **base,
                "old_grad_S": _format_float(old_grad_S),
                "old_grad_beta": _format_float(old_grad_beta),
                "old_grad_S_finite": old_grad_S_finite,
                "old_grad_beta_finite": old_grad_beta_finite,
                "new_grad_S": _format_float(new_grad_S),
                "new_grad_beta": _format_float(new_grad_beta),
                "new_grad_S_finite": new_grad_S_finite,
                "new_grad_beta_finite": new_grad_beta_finite,
                "new_output_finite": math.isfinite(new_output),
                "new_output_bounded": bounded,
            }
        )
    return forward_rows, gradient_rows


def _forward_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    finite_rows = [row for row in rows if row["old_output_finite"] and row["new_output_finite"]]
    abs_diffs = [float(row["abs_diff"]) for row in finite_rows]
    old_sq = sum(float(row["old_output"]) ** 2 for row in finite_rows)
    diff_sq = sum(float(row["abs_diff"]) ** 2 for row in finite_rows)
    return {
        "max_abs_diff": max(abs_diffs) if abs_diffs else float("nan"),
        "relative_L2_diff": math.sqrt(diff_sq / old_sq) if old_sq > 0.0 else 0.0,
        "count": float(len(finite_rows)),
    }


def _synthetic_beta_summary(rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    synthetic_rows = [row for row in rows if row["dataset"] == "synthetic_grid"]
    summary: dict[float, dict[str, Any]] = {}
    for beta in TEST_BETAS:
        subset = [row for row in synthetic_rows if math.isclose(float(row["beta"]), beta, rel_tol=0.0, abs_tol=1.0e-15)]
        summary[beta] = {
            "old_nonfinite_grad_count": sum(
                (not row["old_grad_S_finite"]) or (not row["old_grad_beta_finite"])
                for row in subset
            ),
            "new_nonfinite_grad_count": sum(
                (not row["new_grad_S_finite"]) or (not row["new_grad_beta_finite"])
                for row in subset
            ),
            "new_max_abs_grad_beta": max(abs(float(row["new_grad_beta"])) for row in subset) if subset else float("nan"),
            "new_max_abs_grad_S": max(abs(float(row["new_grad_S"])) for row in subset) if subset else float("nan"),
        }
    return summary


def _batch_a_risk_lines() -> list[str]:
    lines: list[str] = []
    if BATCH_A_RISK_CSV.exists():
        rows = _read_csv(BATCH_A_RISK_CSV)
        lines.append("## 8. FLEX Batch A risk status after rewrite")
        for model in TARGET_MODELS:
            row = next(row for row in rows if row["formula"] == "saturation_3" and row["active_model"] == model)
            lines.append(
                f"- `{model}`: realistic_risk=`{row['realistic_risk']}`, action=`{row['recommended_action']}`, reason={row['short_reason']}"
            )
        lines.append("")
    return lines


def _remaining_risk_lines() -> list[str]:
    lines: list[str] = []
    if FINAL_RISK_CSV.exists():
        rows = _read_csv(FINAL_RISK_CSV)
        lines.append("## 11. Remaining risks, if any")
        saturation_rows = [row for row in rows if row["formula"] == "saturation_3" and row["active_model"] in TARGET_MODELS]
        for row in saturation_rows:
            lines.append(
                f"- `{row['active_model']}` final active risk: `{row['final_active_risk']}` with action `{row['final_recommended_action']}`."
            )
        if not saturation_rows:
            lines.append("- No finalized saturation_3 risk rows were available when this report was generated.")
        lines.append("")
    return lines


def write_report(forward_rows: list[dict[str, Any]], gradient_rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "saturation3_forward_equivalence.csv", forward_rows)
    _write_csv(OUTPUT_DIR / "saturation3_gradient_stability.csv", gradient_rows)

    overall_forward = _forward_summary(forward_rows)
    synthetic_forward = _forward_summary([row for row in forward_rows if row["dataset"] == "synthetic_grid"])
    realistic_forward = _forward_summary([row for row in forward_rows if row["dataset"] == "realistic_trace_anchor"])
    beta_summary = _synthetic_beta_summary(gradient_rows)
    source = inspect.getsource(saturation_3).strip()

    lines = [
        "# saturation_3 Stable Rewrite Report",
        "",
        "## 1. Scope",
        "- Implemented the exact numerically stable rewrite for `models/flux/saturation.py::saturation_3`.",
        "- This is an algebraically equivalent numerical-stability refactor only; no parameter bounds or hydrological semantics were changed.",
        "",
        "## 2. Old expression",
        "- `incoming_flux * (1 - 1 / (1 + exp(z)))`",
        "- `z = (S / (Smax + eps) + 0.5) / (beta + eps)`",
        "",
        "## 3. New expression",
        "- `incoming_flux * sigmoid(z)`",
        "- Active implementation:",
        "",
        "```python",
        source,
        "```",
        "",
        "## 4. Why the rewrite is algebraically equivalent",
        "- `1 - 1 / (1 + exp(z)) = exp(z) / (1 + exp(z)) = 1 / (1 + exp(-z)) = sigmoid(z)`.",
        "- The refactor changes only the numerical evaluation path, not the mathematical function.",
        "",
        "## 5. Why the old expression caused non-finite gradients",
        "- For large positive `z`, the forward value saturates to `incoming_flux` and stays finite.",
        "- The old backward path differentiates through `exp(z)`, so overflow can create `inf * 0` cancellation and `NaN` gradients.",
        "- `torch.sigmoid(z)` uses a numerically stable implementation and avoids that backward overflow path.",
        "",
        "## 6. Forward equivalence test results",
        f"- Overall finite-region max_abs_diff: `{_format_float(overall_forward['max_abs_diff'])}`",
        f"- Overall finite-region relative_L2_diff: `{_format_float(overall_forward['relative_L2_diff'])}`",
        f"- Synthetic grid max_abs_diff: `{_format_float(synthetic_forward['max_abs_diff'])}`",
        f"- Synthetic grid relative_L2_diff: `{_format_float(synthetic_forward['relative_L2_diff'])}`",
        f"- Realistic trace anchors max_abs_diff: `{_format_float(realistic_forward['max_abs_diff'])}`",
        f"- Realistic trace anchors relative_L2_diff: `{_format_float(realistic_forward['relative_L2_diff'])}`",
        "",
        "## 7. Gradient-stability test results",
    ]
    for beta in (1.0e-6, 1.0e-5, 1.0e-4, 5.0000005):
        summary = beta_summary[beta]
        lines.append(
            f"- beta=`{_format_float(beta)}`: old non-finite grid gradients=`{summary['old_nonfinite_grad_count']}`, "
            f"new non-finite grid gradients=`{summary['new_nonfinite_grad_count']}`, "
            f"new max|dS|=`{_format_float(summary['new_max_abs_grad_S'])}`, "
            f"new max|dbeta|=`{_format_float(summary['new_max_abs_grad_beta'])}`"
        )
    lines.extend([""] + _batch_a_risk_lines())
    lines.extend(
        [
            "## 9. Confirmation that no parameter bounds were changed",
            "- The rewrite did not touch any model parameter bounds or parameter transforms.",
            "",
            "## 10. Confirmation that no hydrological formula semantics were changed",
            "- `nearzero` handling, argument order, device/dtype behavior, and output meaning are unchanged.",
            "- No internal beta clamp was added.",
            "- No smoothing or default changes were introduced.",
            "",
        ]
    )
    lines.extend(_remaining_risk_lines())
    report_path = OUTPUT_DIR / "saturation3_stable_rewrite_report.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    forward_rows, gradient_rows = build_rows()
    write_report(forward_rows, gradient_rows)


if __name__ == "__main__":
    main()
