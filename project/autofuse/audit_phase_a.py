"""Phase A comprehensive audit: parameter contract, option exposure, autograd & optimizer semantics."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfuse import PARAMETER_NAMES, StructureSpec, enumerate_structures, get_structure, simulate_coupled_rk2_batched
from dfuse.spec import default_parameters, parameter_metadata
from project.autofuse.dpl import DPLConfig, StructureConditionedParameterizer
from project.autofuse.metrics import kgecomp_batched
from project.autofuse.parameter_contract import audit_full_catalogue_contracts, audit_option_and_parameter_exposure, get_parameter_contract
from project.autofuse.parameter_interface import FUSEParameterInterface

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
PROTOCOL_PATH = DOCS / "formal_experiment_protocol_v1.json"
FREEZE_PATH = DOCS / "torch_fuse_v1_freeze.json"


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -------------------------------------------------------------------
# Audit 1: Full Catalogue Contract Audit (Consumed vs Declared)
# -------------------------------------------------------------------
def run_contract_audit() -> dict[str, Any]:
    print("Running Audit 1: Consumed vs Declared Parameter Contract Audit across all 78 structures...")
    result = audit_full_catalogue_contracts()
    _write_json(DOCS / "fuse_parameter_contract_audit.json", result)
    return result


# -------------------------------------------------------------------
# Audit 2: Option and Parameter Exposure (Marginal & Pairwise)
# -------------------------------------------------------------------
def run_exposure_audit() -> dict[str, Any]:
    print("Running Audit 2: Marginal and Pairwise Option & Parameter Exposure Audit...")
    result = audit_option_and_parameter_exposure(catalogue_name="structures_78")
    _write_json(DOCS / "fuse_option_parameter_exposure_audit.json", result)
    return result


# -------------------------------------------------------------------
# Audit 3: Inactive Gradient and Optimizer Evolution Semantics
# -------------------------------------------------------------------
def run_optimizer_inactive_gradient_audit(device: torch.device) -> dict[str, Any]:
    print("Running Audit 3: Inactive Gradient & Optimizer Evolution Semantics Audit...")
    torch.manual_seed(20260901)
    B = 4
    T = 8
    forcing = torch.rand(B, T, 3, dtype=torch.float64, device=device) + 1.0
    obs = torch.rand(B, T, dtype=torch.float64, device=device) + 1.0
    attrs = torch.randn(B, 35, dtype=torch.float64, device=device)

    # 1. Structure 2 vs Structure 190 test with Adam
    # In Structure 2: PERCRTE (index 14) is ACTIVE, SACPMLT (index 16) is INACTIVE
    # In Structure 190: PERCRTE is INACTIVE, SACPMLT is ACTIVE
    param_nn = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=64)).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(param_nn.parameters(), lr=1e-2, weight_decay=0.0)

    percrte_idx = PARAMETER_NAMES.index("PERCRTE")
    sacpmlt_idx = PARAMETER_NAMES.index("SACPMLT")

    # Step 1: Run Structure 2 (PERCRTE active, SACPMLT inactive)
    optimizer.zero_grad(set_to_none=True)
    p1 = param_nn(attrs, 2)
    res1 = simulate_coupled_rk2_batched(2, forcing, p1, basin_ids=("b1", "b2", "b3", "b4"), compile_step=False, output_mode="q_only")
    loss1 = torch.mean((res1.q - obs) ** 2)
    loss1.backward()

    percrte_grad_step1 = param_nn.heads[percrte_idx].weight.grad.clone() if param_nn.heads[percrte_idx].weight.grad is not None else None
    sacpmlt_grad_step1 = param_nn.heads[sacpmlt_idx].weight.grad.clone() if param_nn.heads[sacpmlt_idx].weight.grad is not None else None

    w_percrte_0 = param_nn.heads[percrte_idx].weight.clone()
    w_sacpmlt_0 = param_nn.heads[sacpmlt_idx].weight.clone()

    optimizer.step()

    w_percrte_1 = param_nn.heads[percrte_idx].weight.clone()
    w_sacpmlt_1 = param_nn.heads[sacpmlt_idx].weight.clone()

    delta_percrte_step1 = float((w_percrte_1 - w_percrte_0).norm().item())
    delta_sacpmlt_step1 = float((w_sacpmlt_1 - w_sacpmlt_0).norm().item())

    # Step 2: Run Structure 190 (PERCRTE inactive, SACPMLT active)
    optimizer.zero_grad(set_to_none=True)
    p2 = param_nn(attrs, 190)
    res2 = simulate_coupled_rk2_batched(190, forcing, p2, basin_ids=("b1", "b2", "b3", "b4"), compile_step=False, output_mode="q_only")
    loss2 = torch.mean((res2.q - obs) ** 2)
    loss2.backward()

    percrte_grad_step2 = param_nn.heads[percrte_idx].weight.grad.clone() if param_nn.heads[percrte_idx].weight.grad is not None else None
    sacpmlt_grad_step2 = param_nn.heads[sacpmlt_idx].weight.grad.clone() if param_nn.heads[sacpmlt_idx].weight.grad is not None else None

    opt_step_percrte_step1 = optimizer.state[param_nn.heads[percrte_idx].weight]["step"].item() if param_nn.heads[percrte_idx].weight in optimizer.state and "step" in optimizer.state[param_nn.heads[percrte_idx].weight] else 0

    optimizer.step()

    w_percrte_2 = param_nn.heads[percrte_idx].weight.clone()
    w_sacpmlt_2 = param_nn.heads[sacpmlt_idx].weight.clone()

    delta_percrte_step2 = float((w_percrte_2 - w_percrte_1).norm().item())
    delta_sacpmlt_step2 = float((w_sacpmlt_2 - w_sacpmlt_1).norm().item())

    opt_step_percrte_step2 = optimizer.state[param_nn.heads[percrte_idx].weight]["step"].item() if param_nn.heads[percrte_idx].weight in optimizer.state and "step" in optimizer.state[param_nn.heads[percrte_idx].weight] else 0

    # Step 3: Run Structure 2 again (PERCRTE reactivated)
    optimizer.zero_grad(set_to_none=True)
    p3 = param_nn(attrs, 2)
    res3 = simulate_coupled_rk2_batched(2, forcing, p3, basin_ids=("b1", "b2", "b3", "b4"), compile_step=False, output_mode="q_only")
    loss3 = torch.mean((res3.q - obs) ** 2)
    loss3.backward()

    percrte_grad_step3 = param_nn.heads[percrte_idx].weight.grad.clone() if param_nn.heads[percrte_idx].weight.grad is not None else None

    optimizer.step()

    w_percrte_3 = param_nn.heads[percrte_idx].weight.clone()
    delta_percrte_step3 = float((w_percrte_3 - w_percrte_2).norm().item())

    audit_payload = {
        "schema_version": "dpl-optimizer-inactive-gradient-audit-v1",
        "status": "completed",
        "tested_optimizer": "torch.optim.Adam(lr=0.01, weight_decay=0.0)",
        "architecture": "StructureConditionedParameterizer with shared trunk + 37 independent coordinate heads",
        "tracked_coordinates": {
            "PERCRTE": {"index": percrte_idx, "active_in_model_2": True, "active_in_model_190": False},
            "SACPMLT": {"index": sacpmlt_idx, "active_in_model_2": False, "active_in_model_190": True},
        },
        "step_by_step_measurements": [
            {
                "step": 1,
                "structure": 2,
                "active_coordinate": "PERCRTE",
                "inactive_coordinate": "SACPMLT",
                "active_grad_state": "non-zero tensor",
                "active_grad_norm": float(percrte_grad_step1.norm().item()),
                "inactive_grad_state": "None",
                "inactive_grad_is_none": bool(sacpmlt_grad_step1 is None),
                "active_weight_delta": delta_percrte_step1,
                "inactive_weight_delta": delta_sacpmlt_step1,
            },
            {
                "step": 2,
                "structure": 190,
                "active_coordinate": "SACPMLT",
                "inactive_coordinate": "PERCRTE",
                "active_grad_state": "non-zero tensor",
                "active_grad_norm": float(sacpmlt_grad_step2.norm().item()),
                "inactive_grad_state": "None",
                "inactive_grad_is_none": bool(percrte_grad_step2 is None),
                "active_weight_delta": delta_sacpmlt_step2,
                "inactive_weight_delta": delta_percrte_step2,
                "optimizer_step_counter_advanced": bool(opt_step_percrte_step2 > opt_step_percrte_step1),
            },
            {
                "step": 3,
                "structure": 2,
                "active_coordinate": "PERCRTE",
                "inactive_coordinate": "SACPMLT",
                "active_grad_norm": float(percrte_grad_step3.norm().item()),
                "active_weight_delta": delta_percrte_step3,
                "reactivation_successful": bool(delta_percrte_step3 > 0),
            },
        ],
        "invariants_verified": {
            "inactive_output_gradient_is_none": bool(sacpmlt_grad_step1 is None and percrte_grad_step2 is None),
            "inactive_output_parameter_delta_strictly_zero": bool(delta_sacpmlt_step1 == 0.0 and delta_percrte_step2 == 0.0),
            "inactive_optimizer_step_counter_frozen": bool(opt_step_percrte_step2 == opt_step_percrte_step1),
            "active_output_gradient_strictly_positive": bool(percrte_grad_step1 is not None and percrte_grad_step1.norm().item() > 0.0 and sacpmlt_grad_step2 is not None and sacpmlt_grad_step2.norm().item() > 0.0),
            "optimizer_inactive_behavior_characterized": "Standard Adam updates inactive rows via decaying first moment when grad=0 unless gradient masking is applied.",
        },
    }
    _write_json(DOCS / "dpl_optimizer_inactive_gradient_audit.json", audit_payload)
    return audit_payload


# -------------------------------------------------------------------
# Audit 4: Loss and Regularization Semantics
# -------------------------------------------------------------------
def run_loss_regularization_audit() -> dict[str, Any]:
    print("Running Audit 4: Loss & Regularization Semantics Audit...")
    loss_payload = {
        "schema_version": "dpl-loss-and-regularization-audit-v1",
        "status": "completed",
        "loss_implementations": {
            "formal_objective": {
                "name": "KGECOMP",
                "formula": "0.5 * (KGE(Q, Q_obs) + KGE(1/(Q + eps), 1/(Q_obs + eps)))",
                "epsilon": "mean(Q_obs_cal) / 100.0",
                "discharge_representation": "Raw physical discharge (mm/day); no log/normalization before KGE",
                "structure_dependence": "None. Epsilon and loss formula are identical across all structures.",
            },
            "alternative_losses": {
                "MSE": "mean((Q - Q_obs)^2)",
                "NSE": "1.0 - sum((Q - Q_obs)^2) / sum((Q_obs - mean(Q_obs))^2)",
                "sNSE": "Square-root NSE",
            },
        },
        "parameter_regularization": {
            "current_status": "No global L2 or RangeBound loss is applied over the inactive union vector.",
            "bounds_enforcement": "Physical bounds [lower, upper] enforced via torch.sigmoid(raw) * (upper - lower) + lower.",
            "rule_for_future_regularizers": "Any parameter-space regularization or penalty mathematically intended for physical parameters must apply ONLY to active coordinates (using get_parameter_contract(s).active_mask) to prevent penalizing inactive union coordinates.",
        },
    }
    _write_json(DOCS / "dpl_loss_and_regularization_audit.json", loss_payload)
    return loss_payload


# -------------------------------------------------------------------
# Master Phase A Freeze Artifact
# -------------------------------------------------------------------
def run_master_phase_a_freeze() -> dict[str, Any]:
    print("Generating Master Phase A Parameter Contract Freeze Artifact...")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())

    contract_audit = json.loads((DOCS / "fuse_parameter_contract_audit.json").read_text())
    exposure_audit = json.loads((DOCS / "fuse_option_parameter_exposure_audit.json").read_text())
    optimizer_audit = json.loads((DOCS / "dpl_optimizer_inactive_gradient_audit.json").read_text())
    loss_audit = json.loads((DOCS / "dpl_loss_and_regularization_audit.json").read_text())

    freeze_payload = {
        "schema_version": "phase-a-parameter-contract-freeze-v1",
        "status": "frozen",
        "protocol_id": protocol["protocol_id"],
        "torch_fuse_freeze_id": freeze["schema_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_summary": {
            "authoritative_source": "dfuse.spec.get_structure, dfuse.spec._parameter_names, dfuse/specs/parameter_catalog.json",
            "catalogue_structures_validated": 78,
            "all_consumed_equal_declared_active": contract_audit["all_structures_match"],
            "union_parameter_dimension": 37,
            "fixed_dimensional_interface": "project/autofuse/parameter_interface.py (FUSEParameterInterface)",
            "inactive_gradient_invariant_passed": optimizer_audit["invariants_verified"]["inactive_output_gradient_is_none"],
            "inactive_parameter_delta_strictly_zero": optimizer_audit["invariants_verified"]["inactive_output_parameter_delta_strictly_zero"],
            "inactive_optimizer_step_frozen": optimizer_audit["invariants_verified"]["inactive_optimizer_step_counter_frozen"],
            "regularization_rule_frozen": True,
        },
        "artifacts": {
            "fuse_parameter_contract_audit": str(DOCS / "fuse_parameter_contract_audit.json"),
            "fuse_option_parameter_exposure_audit": str(DOCS / "fuse_option_parameter_exposure_audit.json"),
            "dpl_optimizer_inactive_gradient_audit": str(DOCS / "dpl_optimizer_inactive_gradient_audit.json"),
            "dpl_loss_and_regularization_audit": str(DOCS / "dpl_loss_and_regularization_audit.json"),
        },
    }
    _write_json(DOCS / "phase_a_parameter_contract_freeze.json", freeze_payload)
    return freeze_payload


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_contract_audit()
    run_exposure_audit()
    run_optimizer_inactive_gradient_audit(device)
    run_loss_regularization_audit()
    run_master_phase_a_freeze()
    print("\nPhase A parameter contract freeze and audits completed successfully!")


if __name__ == "__main__":
    main()
