from __future__ import annotations

import ast
import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.core_model_registry import CORE_MODEL_REGISTRY
from tests.core_water_balance_utils import evaluate_model


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "validation_results" / "model_gradcheck_water_balance_tests"
SUMMARY_CSV_PATH = OUTPUT_DIR / "water_balance_pytest_summary.csv"


def _enabled_models() -> list[str]:
    return [name for name, entry in CORE_MODEL_REGISTRY.items() if entry.enabled]


def _supports_external_sink_diagnostics(model_name: str) -> bool:
    return model_name in {"tcm", "susannah2"}


def _case_id(row: dict[str, Any]) -> str:
    return f"{row['model_name']}::{row['test_case']}::{row['parameter_case']}::{row['initial_state_case']}"


def _n_basins(batch_shape_text: str) -> int:
    batch_shape = ast.literal_eval(batch_shape_text)
    return int(batch_shape[0])


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    notes = [
        f"tolerance={float(row['tolerance']):.3e}",
        f"step_tolerance={float(row['step_tolerance']):.3e}",
    ]
    if _supports_external_sink_diagnostics(str(row["model_name"])):
        notes.append("external sink diagnostics included")
    if row["suspected_cause_if_failed"]:
        notes.append(f"cause={row['suspected_cause_if_failed']}")

    return {
        "model": row["model_name"],
        "case_id": _case_id(row),
        "dtype": row["dtype"],
        "device": row["device"],
        "n_timesteps": row["sequence_length"],
        "n_basins": _n_basins(row["batch_shape"]),
        "water_balance_residual": row["max_absolute_full_period_residual"],
        "max_negative_storage": row["max_negative_storage"],
        "output_nan_count": row["nan_count"],
        "output_inf_count": row["inf_count"],
        "status": "passed" if row["pass_fail"] else "failed",
        "notes": "; ".join(notes),
    }


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "case_id",
        "dtype",
        "device",
        "n_timesteps",
        "n_basins",
        "water_balance_residual",
        "max_negative_storage",
        "output_nan_count",
        "output_inf_count",
        "status",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


@lru_cache(maxsize=1)
def _artifacts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    for model_name in _enabled_models():
        raw_rows.extend(evaluate_model(CORE_MODEL_REGISTRY[model_name], torch.float64, "cpu", "pytest"))
    summary_rows = [_summary_row(row) for row in raw_rows]
    _write_summary_csv(summary_rows, SUMMARY_CSV_PATH)
    return raw_rows, summary_rows


def test_water_balance_regression_pytest_has_full_model_coverage() -> None:
    raw_rows, _ = _artifacts()
    assert {row["model_name"] for row in raw_rows} == set(_enabled_models())


def test_water_balance_regression_pytest_has_zero_failures() -> None:
    raw_rows, _ = _artifacts()
    failures = [row for row in raw_rows if not row["pass_fail"]]
    if failures:
        details = "\n".join(
            f"{row['model_name']} {row['test_case']} residual={row['max_absolute_full_period_residual']:.3e} "
            f"tol={row['tolerance']:.3e} cause={row['suspected_cause_if_failed'] or 'n/a'}"
            for row in failures[:20]
        )
        pytest.fail(f"{len(failures)} water-balance regression cases failed:\n{details}")


def test_water_balance_regression_pytest_max_residual_within_established_tolerance() -> None:
    raw_rows, _ = _artifacts()
    max_excess = max(float(row["max_absolute_full_period_residual"]) - float(row["tolerance"]) for row in raw_rows)
    assert max_excess <= 0.0


def test_water_balance_regression_pytest_summary_csv_is_written() -> None:
    _artifacts()
    assert SUMMARY_CSV_PATH.exists()
    assert SUMMARY_CSV_PATH.stat().st_size > 0
