from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import pytest

from scripts.finalize_complete_flux_gradient_review import (
    COMPLETE_RANKING_PATH,
    COMPLETE_REPORT_PATH,
    COMPLETE_SUMMARY_CSV_PATH,
    COMPLETE_SUMMARY_MD_PATH,
    run_complete_review,
)


@lru_cache(maxsize=1)
def _artifacts():
    return run_complete_review()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_complete_review_files_exist():
    _artifacts()
    assert COMPLETE_RANKING_PATH.exists()
    assert COMPLETE_SUMMARY_CSV_PATH.exists()
    assert COMPLETE_SUMMARY_MD_PATH.exists()
    assert COMPLETE_REPORT_PATH.exists()


def test_complete_review_has_zero_final_active_high_risk_and_zero_unresolved():
    _artifacts()
    summary_row = _read_csv(COMPLETE_SUMMARY_CSV_PATH)[0]
    assert int(summary_row["final_active_high_risk_contexts"]) == 0
    assert int(summary_row["contexts_remaining_unresolved"]) == 0


@pytest.mark.parametrize(
    "active_model",
    ["flexb", "flexi", "flexis"],
)
def test_saturation3_marked_as_stable_numerical_rewrite(active_model):
    _artifacts()
    rows = _read_csv(COMPLETE_RANKING_PATH)
    match = next(row for row in rows if row["formula"] == "saturation_3" and row["active_model"] == active_model)
    assert match["final_decision"] == "stable_numerical_rewrite_applied"
    assert match["formula_changed"] == "yes"
    assert match["change_type"] == "stable_numerical_rewrite"


def test_batch_a_b_c_artifact_decisions_are_present():
    _artifacts()
    rows = {(row["formula"], row["active_model"]): row for row in _read_csv(COMPLETE_RANKING_PATH)}
    assert rows[("saturation_2", "hymod")]["final_decision"] == "broad_domain_artifact"
    assert rows[("baseflow_4", "topmodel")]["final_decision"] == "bound_heuristic_artifact"
    assert rows[("recharge_2", "hbv96")]["final_decision"] == "model_level_cap_resolves"
    assert rows[("evap_16", "penman")]["final_decision"] == "bound_heuristic_artifact"
    assert rows[("split_1", "flexb")]["final_decision"] == "broad_domain_artifact"


def test_no_original_active_high_risk_context_remains_unresolved():
    artifacts = _artifacts()
    unresolved = [
        row
        for row in artifacts["ranking_rows"]
        if row["original_broad_risk"] == "high"
        and (row["final_realistic_risk"] == "high" or row["final_decision"] == "manual_review_required")
    ]
    assert unresolved == []
