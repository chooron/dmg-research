from __future__ import annotations

import inspect

from scripts.audit_core_flux_architecture import (
    build_core_inline_inventory,
    build_flux_inventory,
)
from trainers.common_trainer import CommonTrainer


def test_core_contains_no_standalone_formula_migration_candidates() -> None:
    _, flux_functions = build_flux_inventory()
    rows = build_core_inline_inventory(flux_functions)

    assert {row["should_move_to_flux"] for row in rows} <= {"no"}
    assert len(rows) == 3
    assert {row["model_name"] for row in rows} == {"gr4j"}
    assert all("tightly coupled" in row["reason"] for row in rows)


def test_migrated_aliased_flux_functions_are_detected_as_active() -> None:
    rows, _ = build_flux_inventory()
    status = {row["function_name"]: row["active_usage_status"] for row in rows}

    for name in (
        "baseflow_tcm",
        "evap_ihacres_deficit",
        "interception_modhydrolog",
        "mopex_interception_4",
    ):
        assert status[name] == "active"


def test_trainer_rejects_nonfinite_values_instead_of_sanitizing_gradients() -> None:
    source = inspect.getsource(CommonTrainer._train_one_epoch_core)

    assert "nan_to_num" not in source
    assert "FloatingPointError" in source
    assert source.index("not torch.isfinite(loss)") < source.index("loss.backward()")
    assert source.index("bad_gradient_indices") < source.index("clip_grad_norm_")
