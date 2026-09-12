from dfuse import PARAMETER_NAMES
from dfuse.spec import default_parameters, get_structure
import torch

from project.autofuse.evaluator import UnifiedEvaluator
from project.autofuse.metrics import kgecomp, kgecomp_batched
from project.autofuse.protocol import ExperimentProtocol
from project.autofuse.sce import SCEBaseline, SCEConfig
from project.autofuse.dpl import StructureConditionedParameterizer


def test_protocol_is_locked_to_requested_skeleton_and_paper_dates():
    protocol = ExperimentProtocol()
    assert protocol.catchment_count == 544
    assert protocol.paper_source_catchment_count == 559
    assert protocol.structure_count == 78
    assert protocol.periods["warmup"] == ("1987-01-01", "1988-12-31")
    assert protocol.periods["calibration"] == ("1989-01-01", "1998-12-31")
    assert protocol.periods["evaluation"] == ("1999-01-01", "2009-12-31")


def test_kgecomp_perfect_flow_is_one():
    flow = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert torch.allclose(kgecomp(flow, flow), torch.tensor(1.0, dtype=torch.float64))


def test_batched_kgecomp_scores_each_basin_independently():
    flow = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], dtype=torch.float64)
    observed = flow.clone()
    batched = kgecomp_batched(flow, observed)
    expected = torch.stack([kgecomp(flow[index], observed[index]) for index in range(2)])
    torch.testing.assert_close(batched, expected)

def test_sce_and_evaluator_share_dfuse_forward():
    forcing = torch.tensor([[5.0, 2.0, 10.0], [4.0, 2.0, 9.0]], dtype=torch.float64)
    evaluator = UnifiedEvaluator()
    result = evaluator.forward(84, forcing)
    observed = result.q.detach()
    objective = SCEBaseline().objective(84, forcing, observed, {})
    assert torch.isfinite(objective)
    assert result.model_id == 84


def test_formal_sce_runner_uses_batched_coupled_kernel():
    forcing = torch.tensor([[[5.0, 2.0, 10.0], [4.0, 2.0, 9.0], [3.0, 1.0, 8.0], [2.0, 1.0, 7.0]], [[4.0, 2.0, 11.0], [3.0, 2.0, 8.0], [2.0, 1.0, 7.0], [1.0, 1.0, 6.0]]], dtype=torch.float64)
    observed = torch.ones((2, 4), dtype=torch.float64)
    result = SCEBaseline(SCEConfig(max_evaluations=2, pcento=0.0, seed=17)).run(2, forcing, observed, basin_ids=("a", "b"), compile_step=False)
    assert result["execution"]["kernel"] == "simulate_coupled_rk2_batched"
    assert result["execution"]["evaluator"] == "UnifiedEvaluator.score_batched"
    assert result["evaluation_count"] == 2


def test_formal_sce_seed_mask_and_best_theta_recompute():
    forcing = torch.tensor([[[5.0, 2.0, 10.0], [4.0, 2.0, 9.0], [3.0, 1.0, 8.0], [2.0, 1.0, 7.0]]], dtype=torch.float64)
    evaluator = UnifiedEvaluator()
    observed = evaluator.forward_batched(2, forcing, {}, basin_ids=("a",), compile_step=False).q.detach()
    def run(seed):
        return SCEBaseline(SCEConfig(max_evaluations=2, pcento=0.0, seed=seed)).run(2, forcing, observed, basin_ids=("a",), compile_step=False)
    first = run(17)
    repeat = run(17)
    assert first["trajectory"] == repeat["trajectory"]
    theta = torch.as_tensor([first["best_parameters"][name] for name in PARAMETER_NAMES], dtype=torch.float64)
    recomputed = evaluator.score_batched(2, forcing, observed, theta, basin_ids=("a",), compile_step=False)
    assert torch.allclose(1.0 - recomputed.kge_comp, torch.tensor(first["best_score"], dtype=torch.float64))
    defaults = default_parameters()
    active = set(get_structure(2).parameter_names)
    assert all(first["best_parameters"][name] == defaults[name] for name in PARAMETER_NAMES if name not in active)


def test_q_only_and_truncated_calibration_parity():
    from dfuse import simulate_coupled_rk2_batched
    forcing = torch.tensor([[[5.0, 2.0, 10.0], [4.0, 2.0, 9.0], [3.0, 1.0, 8.0], [2.0, 1.0, 7.0]]], dtype=torch.float64)
    theta = torch.zeros((1, 37), dtype=torch.float64)
    defaults = default_parameters()
    for pos, name in enumerate(PARAMETER_NAMES):
        theta[0, pos] = defaults[name]
    res_full = simulate_coupled_rk2_batched(2, forcing, theta, basin_ids=("a",), compile_step=False, output_mode="full")
    res_q = simulate_coupled_rk2_batched(2, forcing, theta, basin_ids=("a",), compile_step=False, output_mode="q_only")
    assert torch.allclose(res_full.q, res_q.q)
    assert (res_full.q - res_q.q).abs().max().item() == 0.0
    res_trunc = simulate_coupled_rk2_batched(2, forcing[:, :2, :], theta, basin_ids=("a",), compile_step=False, output_mode="q_only")
    assert torch.allclose(res_full.q[:, :2], res_trunc.q)
    assert (res_full.q[:, :2] - res_trunc.q).abs().max().item() == 0.0

def test_candidate_batching_exact_parity():
    forcing = torch.tensor([[[5.0, 2.0, 10.0], [4.0, 2.0, 9.0], [3.0, 1.0, 8.0], [2.0, 1.0, 7.0]]], dtype=torch.float64)
    observed = torch.ones((1, 4), dtype=torch.float64)
    config = SCEConfig(max_evaluations=2, pcento=0.0, seed=42)
    res_c1 = SCEBaseline(config).run(2, forcing, observed, basin_ids=("a",), compile_step=False, candidate_batch_size=1)
    res_c2 = SCEBaseline(config).run(2, forcing, observed, basin_ids=("a",), compile_step=False, candidate_batch_size=2)
    assert res_c1["best_score"] == res_c2["best_score"]
    assert res_c1["best_parameters"] == res_c2["best_parameters"]
    assert res_c1["trajectory"] == res_c2["trajectory"]
def test_dpl_parameterizer_is_bounded_and_structure_conditioned():
    parameterizer = StructureConditionedParameterizer()
    values = parameterizer(torch.zeros(2, 35, dtype=torch.float64), 84)
    assert values.shape == (2, 37)
    assert torch.isfinite(values).all()

def test_fuse_parameter_contract_all_78_structures():
    from project.autofuse.parameter_contract import get_parameter_contract, audit_full_catalogue_contracts, audit_option_and_parameter_exposure
    from project.autofuse.parameter_interface import FUSEParameterInterface
    from dfuse import enumerate_structures

    audit = audit_full_catalogue_contracts()
    assert audit["status"] == "passed"
    assert audit["all_structures_match"] is True
    assert audit["structure_count"] == 78

    interface = FUSEParameterInterface(dtype=torch.float64)
    raw_logits = torch.randn(2, 37, dtype=torch.float64, requires_grad=True)
    bounded = interface.bounds_transform(raw_logits)
    assert bounded.shape == (2, 37)
    assert (bounded >= interface.lower).all() and (bounded <= interface.upper).all()

    # Check gradient flow for Structure 2 (active coordinates receive grad, inactive strictly zero)
    active_slice = interface.extract_active_tensor(bounded, 2)
    loss = torch.sum(active_slice ** 2)
    loss.backward()
    active_mask = interface.active_mask_tensor(2)
    assert (raw_logits.grad[:, active_mask].abs() > 0).all()
    assert (raw_logits.grad[:, ~active_mask].abs() == 0).all()

    # Check exposure audit
    exposure = audit_option_and_parameter_exposure()
    assert exposure["structure_count"] == 78
    assert exposure["pairwise_option_coverage"]["totals"]["total_observed_legal_pairs"] == 60
    assert exposure["pairwise_option_coverage"]["totals"]["total_legal_pairs"] == 69
    assert exposure["pairwise_option_coverage"]["totals"]["total_unobserved_legal_pairs"] == 9
    assert exposure["pairwise_option_coverage"]["totals"]["cartesian_identity_satisfied"] is True
    assert exposure["pairwise_option_coverage"]["totals"]["legal_identity_satisfied"] is True

def test_dpl_parameterizer_independent_heads_and_inactive_optimizer_semantics():
    from project.autofuse.dpl import StructureConditionedParameterizer, DPLConfig
    from dfuse import PARAMETER_NAMES, get_structure, simulate_coupled_rk2_batched

    param_nn = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=64)).to(dtype=torch.float64)
    optimizer = torch.optim.Adam(param_nn.parameters(), lr=1e-2, weight_decay=0.0)

    percrte_idx = PARAMETER_NAMES.index("PERCRTE") # Active in Model 2, Inactive in Model 190
    baserte_idx = PARAMETER_NAMES.index("BASERTE") # Inactive in Model 2, Active in Model 190
    attrs = torch.randn(2, 35, dtype=torch.float64)
    forcing = torch.rand(2, 4, 3, dtype=torch.float64) + 1.0
    obs = torch.rand(2, 4, dtype=torch.float64) + 1.0

    # Step 1: Model 2 (PERCRTE active, BASERTE inactive)
    optimizer.zero_grad(set_to_none=True)
    p1 = param_nn(attrs, 2)
    res1 = simulate_coupled_rk2_batched(2, forcing, p1, basin_ids=("b1", "b2"), compile_step=False, output_mode="q_only")
    loss1 = torch.mean((res1.q - obs) ** 2)
    loss1.backward()

    assert param_nn.heads[percrte_idx].weight.grad is not None
    assert param_nn.heads[baserte_idx].weight.grad is None
    assert param_nn.heads[baserte_idx].bias.grad is None

    w_percrte_0 = param_nn.heads[percrte_idx].weight.clone()
    w_baserte_0 = param_nn.heads[baserte_idx].weight.clone()
    optimizer.step()
    w_percrte_1 = param_nn.heads[percrte_idx].weight.clone()
    w_baserte_1 = param_nn.heads[baserte_idx].weight.clone()

    assert (w_percrte_1 - w_percrte_0).norm().item() > 0
    assert (w_baserte_1 - w_baserte_0).norm().item() == 0.0
    assert len(optimizer.state[param_nn.heads[baserte_idx].weight]) == 0

    # Step 2: Model 190 (PERCRTE inactive, BASERTE active)
    optimizer.zero_grad(set_to_none=True)
    p2 = param_nn(attrs, 190)
    res2 = simulate_coupled_rk2_batched(190, forcing, p2, basin_ids=("b1", "b2"), compile_step=False, output_mode="q_only")
    loss2 = torch.mean((res2.q - obs) ** 2)
    loss2.backward()

    assert param_nn.heads[baserte_idx].weight.grad is not None
    assert param_nn.heads[percrte_idx].weight.grad is None

    opt_step_percrte_1 = optimizer.state[param_nn.heads[percrte_idx].weight]["step"].item()
    optimizer.step()
    w_percrte_2 = param_nn.heads[percrte_idx].weight.clone()
    w_baserte_2 = param_nn.heads[baserte_idx].weight.clone()

    assert (w_percrte_2 - w_percrte_1).norm().item() == 0.0
    assert (w_baserte_2 - w_baserte_1).norm().item() > 0
    # Step counter for inactive PERCRTE must not advance
    assert optimizer.state[param_nn.heads[percrte_idx].weight]["step"].item() == opt_step_percrte_1

def test_kgecomp_batched_basin_reduction_semantics():
    from project.autofuse.metrics import kgecomp, kgecomp_batched
    # 3 basins x 10 timesteps
    sim = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                        [2.0, 4.0, 5.0, 3.0, 8.0, 7.0, 6.0, 9.0, 10.0, 11.0],
                        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]], dtype=torch.float64)
    obs = torch.tensor([[1.1, 1.9, 3.2, 3.8, 5.1, 6.2, 6.9, 8.1, 9.0, 9.8],
                        [1.8, 4.2, 4.8, 3.2, 7.9, 7.1, 6.2, 8.8, 10.2, 10.9],
                        [0.6, 1.4, 2.6, 3.4, 4.6, 5.4, 6.6, 7.4, 8.6, 9.4]], dtype=torch.float64)
    eps = 0.05
    batched_kge = kgecomp_batched(sim, obs, epsilon=eps)
    assert batched_kge.shape == (3,)
    # Independent per-basin computation: each element equals scalar kgecomp for that basin alone
    for b in range(3):
        scalar = kgecomp(sim[b], obs[b], epsilon=eps)
        assert torch.allclose(batched_kge[b], scalar)
    # Equal-weight mean across basins
    mean_obj = 1.0 - batched_kge.mean()
    expected_mean = torch.mean(torch.stack([1.0 - kgecomp(sim[b], obs[b], epsilon=eps) for b in range(3)]))
    assert torch.allclose(mean_obj, expected_mean)
