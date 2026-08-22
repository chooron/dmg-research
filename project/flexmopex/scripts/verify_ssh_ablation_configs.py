#!/usr/bin/env python3
"""Fail-closed invariant checks for the Pure-X35 SSH 2x2 experiment."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIGS = {
    "E1": PROJECT_DIR / "conf/ssh_2x2/config_E1_pure_x35_531_lambda0007.yaml",
    "E2": PROJECT_DIR / "conf/ssh_2x2/config_E2_pure_x35_531_lambda0007.yaml",
    "E3": PROJECT_DIR / "conf/ssh_2x2/config_E3_pure_x35_531_lambda0007.yaml",
    "E4": PROJECT_DIR / "conf/ssh_2x2/config_E4_pure_x35_531_lambda0007.yaml",
}
EXPECTED_PHY = {
    "E1": "LearnedWeightMopex",
    "E2": "LearnedWeightMopexE",
    "E3": "LearnedWeightMopex",
    "E4": "LearnedWeightMopexE",
}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required: torch.cuda.is_available() is false")
    print(f"GPU check: {torch.cuda.device_count()} CUDA device(s)")

    manifest = REPO_ROOT / "data/531sub_id.txt"
    values = ast.literal_eval(manifest.read_text())
    if len(values) != 531:
        raise AssertionError(f"531 manifest has {len(values)} IDs")
    print(f"Basin manifest: {manifest} ({len(values)} IDs)")

    for exp_id, path in CONFIGS.items():
        cfg = yaml.safe_load(path.read_text())
        phy = cfg["delta_model"]["phy_model"]
        nn = cfg["delta_model"]["nn_model"]
        train = cfg["train"]
        is_cf = exp_id in {"E3", "E4"}
        expected_formula = "historical_production_mopex4" if exp_id in {"E1", "E3"} else "corrected_candidate_E_S0"

        assert cfg["experiment_id"] == exp_id
        assert cfg["formula_id"] == expected_formula
        assert cfg["basin_count"] == 531
        assert "531sub_id.txt" in str(cfg["basin_manifest"])
        assert cfg["preserve_phy_model"] is True
        assert nn["model"] == "LearnedStructureNetPureAttrEncoder"
        assert phy["model"][0] == EXPECTED_PHY[exp_id]
        assert cfg["loss_function"]["aic_alpha"] == 0.007
        assert train["epochs"] == 10 and cfg["test"]["test_epoch"] == 10
        assert str(cfg["device"]).startswith("cuda")
        assert phy["disable_compile"] is False and phy["require_torch_compile"] is True
        assert bool(cfg["counterfactual_supervision"]) is is_cf
        assert bool(phy["counterfactual_supervision"]) is is_cf
        assert bool(cfg["confidence_weighted_cf_loss"]) is is_cf
        assert cfg["structure_training"] == (
            "confidence_weighted_counterfactual_BCE" if is_cf else "end_to_end_LQ_plus_lambda_Omega"
        )
        if is_cf:
            assert cfg["trainer"] == "CFTrainer"
            assert cfg["cf_loss_weight"] == 1.0
            assert cfg["structure_optimizer"] == "none"
        else:
            assert cfg["trainer"] == "MyTrainer"
        assert "canonical_freeze" not in str(cfg["save_path"])
        print(
            f"{exp_id}: OK | phy={phy['model'][0]} | nn={nn['model']} | "
            f"cf={is_cf} | lambda={cfg['loss_function']['aic_alpha']} | epochs={train['epochs']}"
        )

    core = (PROJECT_DIR / "models/mopex_core.py").read_text()
    candidates = (PROJECT_DIR / "models/mopex_core_candidates.py").read_text()
    assert "flux_potential = alpha * P * season_factor" in core
    assert "pet_for_soil = F.relu(PET_effective - flux_i)" in core
    assert "0.5 * (1.0 + kappa * torch.cos" in candidates
    assert "pet_independent: bool = False" in candidates
    print("Formula source check: historical mopex_core.py and corrected Candidate E/S0 found")
    print("All 2x2 invariants passed; no training was launched by this checker.")


if __name__ == "__main__":
    main()
