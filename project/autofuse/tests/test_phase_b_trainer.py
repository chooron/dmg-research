"""Comprehensive Phase B unit and integration tests for SharedDPLTrainer and Samplers."""
from __future__ import annotations

import copy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch

from dfuse import PARAMETER_NAMES, enumerate_structures, get_structure
from project.autofuse.dpl import DPLConfig, StructureConditionedParameterizer
from project.autofuse.loader import StochasticTimeWindowLoader, TimeWindowConfig
from project.autofuse.parameter_contract import audit_full_catalogue_contracts, get_parameter_contract
from project.autofuse.samplers import GlobalBasinSampler, ShuffledStructureSampler
from project.autofuse.trainer import SharedDPLTrainer


def test_1_structure_cycle_visits_each_once_no_duplicates():
    sampler = ShuffledStructureSampler("structures_78", seed=100)
    seen = [sampler.next_structure() for _ in range(78)]
    assert len(seen) == 78
    assert len(set(seen)) == 78
    assert set(seen) == {s.model_id for s in enumerate_structures()}


def test_2_consecutive_structure_cycles_differ():
    sampler = ShuffledStructureSampler("structures_78", seed=100)
    c0 = [sampler.next_structure() for _ in range(78)]
    c1 = [sampler.next_structure() for _ in range(78)]
    assert c0 != c1
    assert set(c0) == set(c1)


def test_3_basin_cycle_balanced_marginal_coverage():
    basins = [f"basin_{i:03d}" for i in range(544)]
    sampler = GlobalBasinSampler(basins, batch_size=100, seed=200)
    # Run 544 * 2 / 100 ~ 11 batches (1100 basin samples)
    for _ in range(11):
        _ = sampler.next_batch()
    # Check that min exposure and max exposure differ by at most 1 across all 544 basins
    counts = list(sampler.exposure_counts.values())
    assert max(counts) - min(counts) <= 1


def test_4_basin_cursor_does_not_reset_on_structure_boundary():
    basins = [f"basin_{i:03d}" for i in range(544)]
    basin_sampler = GlobalBasinSampler(basins, batch_size=100, seed=300)
    struct_sampler = ShuffledStructureSampler("structures_78", seed=400)

    # Step through entire structure cycle (78 steps)
    for _ in range(78):
        _ = struct_sampler.next_structure()
        _ = basin_sampler.next_batch()

    # Total basins drawn: 78 * 100 = 7800 basins
    # 7800 % 544 = 184. The cursor must be exactly 184, NOT 0!
    assert basin_sampler.cursor == 7800 % 544
    assert basin_sampler.basin_cycle_index == 7800 // 544


def test_5_adversarial_phase_locking_prevented():
    # Adversarial case: N_basins % batch_size == 0
    basins_adv = [f"basin_{i:02d}" for i in range(100)]  # 100 % 100 == 0
    basin_sampler = GlobalBasinSampler(basins_adv, batch_size=100, seed=500)
    struct_sampler = ShuffledStructureSampler([1, 2, 3], seed=600)  # cycle length 3

    struct_pos_basin_map = {0: [], 1: [], 2: []}
    for _ in range(30):
        pos = struct_sampler.cursor % 3
        _ = struct_sampler.next_structure()
        b = basin_sampler.next_batch()
        struct_pos_basin_map[pos].append(b[0])

    assert len(set(struct_pos_basin_map[0])) > 1, "Adversarial phase locking occurred!"


def test_6_structure_and_basin_rng_independence():
    s_sampler_a = ShuffledStructureSampler("structures_78", seed=100)
    s_sampler_b = ShuffledStructureSampler("structures_78", seed=100)

    b_sampler = GlobalBasinSampler([f"b_{i}" for i in range(100)], batch_size=10, seed=999)

    # Interleaving calls to b_sampler should NOT affect s_sampler_a vs s_sampler_b
    s_a = []
    for _ in range(20):
        s_a.append(s_sampler_a.next_structure())
        _ = b_sampler.next_batch()

    s_b = [s_sampler_b.next_structure() for _ in range(20)]
    assert s_a == s_b, "Basin sampler calls leaked into structure RNG sequence!"


def test_7_8_fixed_dim_interface_and_declared_active_consumption():
    audit = audit_full_catalogue_contracts()
    assert audit["all_structures_match"] is True
    assert audit["structure_count"] == 78


def test_9_10_11_time_window_split_and_autograd_warmup():
    # Fast unit test with 8-day window (4 warmup + 4 scored)
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(20)]
    mock_data = {
        "b1": {
            "ppt": np.ones(20, dtype=np.float64) * 5.0,
            "pet": np.ones(20, dtype=np.float64) * 2.0,
            "temp": np.ones(20, dtype=np.float64) * 10.0,
            "q_obs": (np.sin(np.arange(20)) * 2.0 + 3.0).astype(np.float64),
            "attributes": np.zeros(35, dtype=np.float64),
            "epsilon": 0.05,
        }
    }
    cfg = TimeWindowConfig(total_days=8, warmup_days=4, scored_days=4, calibration_end=date(1987, 1, 20))
    loader = StochasticTimeWindowLoader(mock_data, dates, cfg, seed=42)
    batch = loader.sample_batch(["b1"])

    assert batch.forcing.shape == (1, 8, 3)
    assert batch.target_full.shape == (1, 8)
    assert batch.target_scored.shape == (1, 4)

    param_nn = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=64))
    params = param_nn(batch.attributes, 2)

    from dfuse import simulate_coupled_rk2_batched
    res = simulate_coupled_rk2_batched(2, batch.forcing, params, basin_ids=("b1",), compile_step=False, output_mode="q_only")

    # Scored loss on last 4 days (excludes warmup first 4 days)
    q_scored = res.q[:, 4:]
    loss = torch.mean((q_scored - batch.target_scored) ** 2)
    loss.backward()

    # Warmup parameters must have non-zero gradients via autograd continuity through warmup
    assert param_nn.heads[PARAMETER_NAMES.index("MAXWATR_1")].weight.grad is not None
    assert param_nn.heads[PARAMETER_NAMES.index("MAXWATR_1")].weight.grad.norm().item() > 0


def test_12_13_14_global_step_exposure_and_clipping_diagnostics():
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(20)]
    mock_data = {
        "b1": {
            "ppt": np.ones(20, dtype=np.float64) * 5.0,
            "pet": np.ones(20, dtype=np.float64) * 2.0,
            "temp": np.ones(20, dtype=np.float64) * 10.0,
            "q_obs": (np.sin(np.arange(20)) * 2.0 + 3.0).astype(np.float64),
            "attributes": np.zeros(35, dtype=np.float64),
            "epsilon": 0.05,
        }
    }
    cfg = TimeWindowConfig(total_days=8, warmup_days=4, scored_days=4, calibration_end=date(1987, 1, 20))
    loader = StochasticTimeWindowLoader(mock_data, dates, cfg, seed=42)
    config = {
        "batch_size": 1,
        "basin_ids": ["b1"],
        "structures": [2, 8],
        "max_grad_norm": 1.0,
        "lr": 1e-3,
        "device": "cpu",
    }
    trainer = SharedDPLTrainer(config, loader=loader, compile_step=False)

    step0 = trainer.train_step()
    assert step0["global_step"] == 0
    assert trainer.global_step == 1

    step1 = trainer.train_step()
    assert step1["global_step"] == 1
    assert trainer.global_step == 2

    # Check exposure diagnostics
    assert trainer.diagnostics.structure_exposure[2] + trainer.diagnostics.structure_exposure[8] == 2
    assert trainer.diagnostics.basin_exposure["b1"] == 2
    assert len(trainer.diagnostics.gradient_clipping_history) == 2


def test_15_checkpoint_save_and_deterministic_resume():
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(20)]
    mock_data = {
        f"b{i}": {
            "ppt": np.ones(20, dtype=np.float64) * 5.0,
            "pet": np.ones(20, dtype=np.float64) * 2.0,
            "temp": np.ones(20, dtype=np.float64) * 10.0,
            "q_obs": (np.sin(np.arange(20) + i) * 2.0 + 3.0).astype(np.float64),
            "attributes": np.zeros(35, dtype=np.float64),
            "epsilon": 0.05,
        }
        for i in range(4)
    }
    cfg = TimeWindowConfig(total_days=8, warmup_days=4, scored_days=4, calibration_end=date(1987, 1, 20))
    config = {
        "batch_size": 2,
        "basin_ids": ["b0", "b1", "b2", "b3"],
        "structures": [2, 8, 190, 214],
        "lr": 1e-3,
        "device": "cpu",
    }

    # Run 5 uninterrupted steps
    torch.manual_seed(999)
    loader_a = StochasticTimeWindowLoader(mock_data, dates, cfg, seed=123)
    trainer_a = SharedDPLTrainer(config, loader=loader_a, compile_step=False)
    history_a = [trainer_a.train_step() for _ in range(5)]

    # Run 2 steps, save checkpoint, reload into new trainer, run 3 steps
    torch.manual_seed(999)
    loader_b = StochasticTimeWindowLoader(mock_data, dates, cfg, seed=123)
    trainer_b = SharedDPLTrainer(config, loader=loader_b, compile_step=False)
    history_b = [trainer_b.train_step() for _ in range(2)]

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        trainer_b.save_checkpoint(f.name)
        trainer_resumed = SharedDPLTrainer(config, loader=loader_b, compile_step=False)
        trainer_resumed.load_checkpoint(f.name)

        for _ in range(3):
            history_b.append(trainer_resumed.train_step())

    # Assert exact trajectory match across all 5 steps
    for s_idx in range(5):
        assert history_a[s_idx]["global_step"] == history_b[s_idx]["global_step"]
        assert history_a[s_idx]["model_id"] == history_b[s_idx]["model_id"]
        assert abs(history_a[s_idx]["loss"] - history_b[s_idx]["loss"]) < 1e-12
        assert abs(history_a[s_idx]["mean_kge"] - history_b[s_idx]["mean_kge"]) < 1e-12


def test_16_17_compile_off_smoke_and_tail_evaluation():
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(20)]
    mock_data = {
        "b1": {
            "ppt": np.ones(20, dtype=np.float64) * 5.0,
            "pet": np.ones(20, dtype=np.float64) * 2.0,
            "temp": np.ones(20, dtype=np.float64) * 10.0,
            "q_obs": (np.sin(np.arange(20)) * 2.0 + 3.0).astype(np.float64),
            "attributes": np.zeros(35, dtype=np.float64),
            "epsilon": 0.05,
        }
    }
    cfg = TimeWindowConfig(total_days=8, warmup_days=4, scored_days=4, calibration_end=date(1987, 1, 20))
    loader = StochasticTimeWindowLoader(mock_data, dates, cfg, seed=42)
    config = {
        "batch_size": 1,
        "basin_ids": ["b1"],
        "structures": [2, 8, 190, 214],
        "lr": 1e-3,
        "device": "cpu",
    }
    trainer = SharedDPLTrainer(config, loader=loader, compile_step=False)
    trainer.train(total_steps=4)

    eval_res = trainer.evaluate(structure_subset=[2, 8], tail_metric="worst_decile")
    assert eval_res["total_evaluations"] == 2
    assert "worst_decile_p10_kge" in eval_res
    assert "mean_kge" in eval_res
    assert np.isfinite(eval_res["mean_kge"])


def test_phase_b1_end_to_end_inactive_head_optimizer_invariants_through_trainer():
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(20)]
    mock_data = {
        "b1": {
            "ppt": np.ones(20, dtype=np.float64) * 5.0,
            "pet": np.ones(20, dtype=np.float64) * 2.0,
            "temp": np.ones(20, dtype=np.float64) * 10.0,
            "q_obs": (np.sin(np.arange(20)) * 2.0 + 3.0).astype(np.float64),
            "attributes": np.zeros(35, dtype=np.float64),
            "epsilon": 0.05,
        }
    }
    cfg = TimeWindowConfig(total_days=8, warmup_days=4, scored_days=4, calibration_end=date(1987, 1, 20))
    loader = StochasticTimeWindowLoader(mock_data, dates, cfg, seed=42)
    # Controlled sequence: Model 2 -> Model 190 -> Model 2
    struct_sampler = ShuffledStructureSampler([2, 190], seed=100)
    # Force initial permutation to be exactly [2, 190]
    struct_sampler.current_permutation = [2, 190]
    struct_sampler.cursor = 0

    config = {
        "batch_size": 1,
        "basin_ids": ["b1"],
        "structures": [2, 190],
        "lr": 1e-2,
        "device": "cpu",
    }
    trainer = SharedDPLTrainer(config, structure_sampler=struct_sampler, loader=loader, compile_step=False)

    percrte_idx = PARAMETER_NAMES.index("PERCRTE")  # Active in Model 2, Inactive in Model 190
    sacpmlt_idx = PARAMETER_NAMES.index("SACPMLT")  # Inactive in Model 2, Active in Model 190

    # --- Step 1: Model 2 (PERCRTE active, SACPMLT inactive) ---
    w_percrte_0 = trainer.model.heads[percrte_idx].weight.clone()
    w_sacpmlt_0 = trainer.model.heads[sacpmlt_idx].weight.clone()
    w_trunk_0 = next(trainer.model.trunk.parameters()).clone()

    step1 = trainer.train_step()
    assert step1["model_id"] == 2

    w_percrte_1 = trainer.model.heads[percrte_idx].weight.clone()
    w_sacpmlt_1 = trainer.model.heads[sacpmlt_idx].weight.clone()
    w_trunk_1 = next(trainer.model.trunk.parameters()).clone()

    # PERCRTE and trunk updated; SACPMLT unchanged
    assert (w_percrte_1 - w_percrte_0).norm().item() > 0
    assert (w_sacpmlt_1 - w_sacpmlt_0).norm().item() == 0.0
    assert (w_trunk_1 - w_trunk_0).norm().item() > 0
    assert len(trainer.optimizer.state[trainer.model.heads[sacpmlt_idx].weight]) == 0
    assert len(trainer.optimizer.state[trainer.model.heads[percrte_idx].weight]) > 0
    opt_step_percrte_before_step2 = trainer.optimizer.state[trainer.model.heads[percrte_idx].weight]["step"].item()
    opt_exp_avg_percrte_before_step2 = trainer.optimizer.state[trainer.model.heads[percrte_idx].weight]["exp_avg"].clone()
    opt_exp_avg_sq_percrte_before_step2 = trainer.optimizer.state[trainer.model.heads[percrte_idx].weight]["exp_avg_sq"].clone()

    # --- Step 2: Model 190 (PERCRTE inactive, SACPMLT active) ---
    step2 = trainer.train_step()
    assert step2["model_id"] == 190

    w_percrte_2 = trainer.model.heads[percrte_idx].weight.clone()
    w_sacpmlt_2 = trainer.model.heads[sacpmlt_idx].weight.clone()
    w_trunk_2 = next(trainer.model.trunk.parameters()).clone()

    # Inactive PERCRTE invariants on Step 2:
    assert trainer.model.heads[percrte_idx].weight.grad is None
    assert trainer.model.heads[percrte_idx].bias.grad is None
    assert (w_percrte_2 - w_percrte_1).norm().item() == 0.0
    assert (w_sacpmlt_2 - w_sacpmlt_1).norm().item() > 0
    assert (w_trunk_2 - w_trunk_1).norm().item() > 0

    # Optimizer-local step and moments for PERCRTE must NOT have changed
    opt_step_percrte_after_step2 = trainer.optimizer.state[trainer.model.heads[percrte_idx].weight]["step"].item()
    opt_exp_avg_percrte_after_step2 = trainer.optimizer.state[trainer.model.heads[percrte_idx].weight]["exp_avg"]
    opt_exp_avg_sq_percrte_after_step2 = trainer.optimizer.state[trainer.model.heads[percrte_idx].weight]["exp_avg_sq"]
    assert opt_step_percrte_after_step2 == opt_step_percrte_before_step2
    assert torch.equal(opt_exp_avg_percrte_after_step2, opt_exp_avg_percrte_before_step2)
    assert torch.equal(opt_exp_avg_sq_percrte_after_step2, opt_exp_avg_sq_percrte_before_step2)

    # --- Step 3: Model 2 (PERCRTE reactivated) ---
    # Set permutation to ensure next structure is 2
    trainer.structure_sampler.current_permutation = [2, 190]
    trainer.structure_sampler.cursor = 0

    step3 = trainer.train_step()
    assert step3["model_id"] == 2

    w_percrte_3 = trainer.model.heads[percrte_idx].weight.clone()
    assert (w_percrte_3 - w_percrte_2).norm().item() > 0
    assert trainer.optimizer.state[trainer.model.heads[percrte_idx].weight]["step"].item() == opt_step_percrte_before_step2 + 1


def test_phase_b1_global_basin_sampler_cross_boundary_uniqueness_and_balance():
    # Test 1: 544 basins, batch size 100 across 100 batches (10,000 samples)
    sampler_544 = GlobalBasinSampler([f"b_{i:03d}" for i in range(544)], batch_size=100, seed=42)
    for batch_idx in range(100):
        b = sampler_544.next_batch()
        assert len(b) == 100
        assert len(set(b)) == 100, f"Duplicate basin found in batch {batch_idx}"

    counts_544 = list(sampler_544.exposure_counts.values())
    assert max(counts_544) - min(counts_544) <= 1, "Marginal exposure imbalance in 544-basin test"

    # Test 2: Prime case crossing permutation boundaries: 7 basins, batch size 4 across 50 batches
    sampler_7 = GlobalBasinSampler([f"b_{i}" for i in range(7)], batch_size=4, seed=123)
    for batch_idx in range(50):
        b = sampler_7.next_batch()
        assert len(b) == 4
        assert len(set(b)) == 4, f"Duplicate basin found in batch {batch_idx}: {b}"

    counts_7 = list(sampler_7.exposure_counts.values())
    assert max(counts_7) - min(counts_7) <= 1, "Marginal exposure imbalance in 7-basin test"

    # Test 3: Checkpoint resume across boundary-crossing batch
    sampler_orig = GlobalBasinSampler([f"b_{i}" for i in range(7)], batch_size=4, seed=777)
    for _ in range(3):
        sampler_orig.next_batch()
    st = sampler_orig.state_dict()
    ref_batches = [sampler_orig.next_batch() for _ in range(10)]

    sampler_resumed = GlobalBasinSampler([f"b_{i}" for i in range(7)], batch_size=4, seed=999)
    sampler_resumed.load_state_dict(st)
    res_batches = [sampler_resumed.next_batch() for _ in range(10)]
    assert ref_batches == res_batches, "Resumed sampler output mismatch across boundary!"


def test_phase_b1_phase_locking_actual_basin_sequence():
    # Test phase-locking with actual basin ID sequences across 30 structure cycles
    basins_adv = [f"basin_{i:02d}" for i in range(100)]
    basin_sampler = GlobalBasinSampler(basins_adv, batch_size=100, seed=300)
    struct_sampler = ShuffledStructureSampler([1, 2, 3], seed=400)

    struct_0_basin_batches = []
    for _ in range(90):  # 30 full structure cycles
        s = struct_sampler.next_structure()
        b = basin_sampler.next_batch()
        if s == 1:
            struct_0_basin_batches.append(b)

    assert len(struct_0_basin_batches) == 30
    # Verify that the sequence of basin batches assigned to Structure 1 changes across cycles
    unique_batches_for_struct_0 = len(set(tuple(b) for b in struct_0_basin_batches))
    assert unique_batches_for_struct_0 > 1, "Structure 1 received identical basin batches across all cycles!"


def test_phase_b1_trainer_fail_fast_on_non_finite_loss():
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(20)]
    mock_data = {
        "b1": {
            "ppt": np.ones(20, dtype=np.float64) * 5.0,
            "pet": np.ones(20, dtype=np.float64) * 2.0,
            "temp": np.ones(20, dtype=np.float64) * 10.0,
            "q_obs": np.full(20, np.nan, dtype=np.float64),  # All NaNs -> non-finite loss
            "attributes": np.zeros(35, dtype=np.float64),
            "epsilon": 0.05,
        }
    }
    cfg = TimeWindowConfig(total_days=8, warmup_days=4, scored_days=4, calibration_end=date(1987, 1, 20))
    loader = StochasticTimeWindowLoader(mock_data, dates, cfg, seed=42)
    config = {
        "batch_size": 1,
        "basin_ids": ["b1"],
        "structures": [2],
        "lr": 1e-3,
        "device": "cpu",
    }
    trainer = SharedDPLTrainer(config, loader=loader, compile_step=False)
    import pytest
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        trainer.train_step()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="strict resume gate requires CUDA")
def test_strict_gpu_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(20)]
    mock_data = {
        f"b{i}": {
            "ppt": np.full(20, 5.0 + i, dtype=np.float64),
            "pet": np.full(20, 2.0, dtype=np.float64),
            "temp": np.full(20, 10.0 + i, dtype=np.float64),
            "q_obs": (np.sin(np.arange(20) + i) * 2.0 + 3.0).astype(np.float64),
            "attributes": np.full(35, i, dtype=np.float64),
            "epsilon": 0.05,
        }
        for i in range(3)
    }
    time_config = TimeWindowConfig(total_days=8, warmup_days=4, scored_days=4, calibration_end=date(1987, 1, 20))
    config = {
        "batch_size": 2,
        "basin_ids": ["b0", "b1", "b2"],
        "structures": [2, 8, 190],
        "lr": 1e-3,
        "device": "cuda",
        "dtype": "float64",
    }
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    seed_model = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=16)).to(device="cuda", dtype=torch.float64)

    def make_trainer():
        loader = StochasticTimeWindowLoader(mock_data, dates, time_config, seed=20260902, device="cuda", dtype=torch.float64)
        structures = ShuffledStructureSampler(config["structures"], seed=20260903)
        basins = GlobalBasinSampler(config["basin_ids"], batch_size=2, seed=20260904)
        return SharedDPLTrainer(
            config,
            model=copy.deepcopy(seed_model),
            structure_sampler=structures,
            basin_sampler=basins,
            loader=loader,
            compile_step=False,
        )

    def assert_tree_equal(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left.detach().cpu(), right.detach().cpu(), rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                assert_tree_equal(left_item, right_item)
        else:
            assert left == right

    def run_one(trainer):
        structure_probe = ShuffledStructureSampler(config["structures"], seed=0)
        structure_probe.load_state_dict(trainer.structure_sampler.state_dict())
        expected_structure = structure_probe.next_structure()
        basin_probe = GlobalBasinSampler(config["basin_ids"], batch_size=2, seed=0)
        basin_probe.load_state_dict(trainer.basin_sampler.state_dict())
        expected_basins = basin_probe.next_batch()
        loader_probe = StochasticTimeWindowLoader(mock_data, dates, time_config, seed=0, device="cuda", dtype=torch.float64)
        loader_probe.load_state_dict(trainer.loader.state_dict())
        expected_windows = loader_probe.sample_batch(expected_basins).start_indices
        output = trainer.train_step()
        assert output["model_id"] == expected_structure
        return {
            "model_id": output["model_id"],
            "loss": output["loss"],
            "basins": expected_basins,
            "windows": expected_windows,
            "global_step": trainer.global_step,
            "structure_state": copy.deepcopy(trainer.structure_sampler.state_dict()),
            "basin_state": copy.deepcopy(trainer.basin_sampler.state_dict()),
            "loader_state": copy.deepcopy(trainer.loader.state_dict()),
            "model_state": copy.deepcopy(trainer.model.state_dict()),
            "optimizer_state": copy.deepcopy(trainer.optimizer.state_dict()),
        }

    reference = make_trainer()
    reference_records = [run_one(reference) for _ in range(4)]
    split = make_trainer()
    split_records = [run_one(split) for _ in range(2)]
    checkpoint = tmp_path / "strict_resume.pt"
    split.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["checkpoint_metadata"]
    assert {
        "checkpoint_metadata",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "structure_sampler_state",
        "basin_sampler_state",
        "loader_state",
        "torch_rng_state",
        "cuda_rng_state",
        "diagnostics",
    }.issubset(payload)
    assert metadata["solver"] == "coupled_rk2"
    assert metadata["optimizer"] == "Adam"
    assert metadata["structure_ids"] == config["structures"]
    assert metadata["parameter_names"] == list(PARAMETER_NAMES)
    del split
    resumed = make_trainer()
    resumed.load_checkpoint(checkpoint)
    split_records.extend(run_one(resumed) for _ in range(2))

    assert [record["model_id"] for record in reference_records] == [record["model_id"] for record in split_records]
    assert [record["basins"] for record in reference_records] == [record["basins"] for record in split_records]
    assert [record["windows"] for record in reference_records] == [record["windows"] for record in split_records]
    for reference_record, resumed_record in zip(reference_records, split_records):
        assert reference_record["global_step"] == resumed_record["global_step"]
        assert abs(reference_record["loss"] - resumed_record["loss"]) <= 1e-12
        for key in ("structure_state", "basin_state", "loader_state", "model_state", "optimizer_state"):
            assert_tree_equal(reference_record[key], resumed_record[key])
        assert np.isfinite(reference_record["loss"]) and np.isfinite(resumed_record["loss"])
    torch.cuda.empty_cache()
