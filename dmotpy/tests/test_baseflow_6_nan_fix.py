from __future__ import annotations

import pytest
import torch

from models.flux.baseflow import baseflow_6
from models.flux.smooth import soft_gate_storage_above


# Reference implementation matching the original baseflow_6 (for comparison)
def _baseflow_6_original(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    q_quadratic = torch.minimum(S, p1 * S.pow(2))
    sf = soft_gate_storage_above(S, p2, nearzero=nearzero)
    return q_quadratic * sf


class TestBaseflow6NaNEdgeCases:
    """Verify that NaN-producing edge cases no longer produce NaN."""

    def test_p1_zero_S_inf(self):
        p1 = torch.tensor(0.0)
        p2 = torch.tensor(0.0)
        S = torch.tensor(float("inf"))
        result = baseflow_6(p1, p2, S)
        assert not torch.isnan(result).any(), "p1=0, S=inf produced NaN"
        assert torch.isfinite(result).all(), "p1=0, S=inf produced non-finite output"

    def test_p1_zero_S_neg_inf(self):
        p1 = torch.tensor(0.0)
        p2 = torch.tensor(0.0)
        S = torch.tensor(float("-inf"))
        result = baseflow_6(p1, p2, S)
        assert not torch.isnan(result).any(), "p1=0, S=-inf produced NaN"
        assert torch.isfinite(result).all(), "p1=0, S=-inf produced non-finite output"

    def test_p1_zero_S_very_large_f32(self):
        p1 = torch.tensor(0.0, dtype=torch.float32)
        p2 = torch.tensor(0.0, dtype=torch.float32)
        S = torch.tensor(2e19, dtype=torch.float32)
        result = baseflow_6(p1, p2, S)
        assert not torch.isnan(result).any(), "p1=0, S=2e19 (float32) produced NaN"
        assert torch.isfinite(result).all()

    def test_p1_zero_S_very_large_f64(self):
        p1 = torch.tensor(0.0, dtype=torch.float64)
        p2 = torch.tensor(0.0, dtype=torch.float64)
        S = torch.tensor(1e200, dtype=torch.float64)
        result = baseflow_6(p1, p2, S)
        assert not torch.isnan(result).any(), "p1=0, S=1e200 (float64) produced NaN"
        assert torch.isfinite(result).all()

    def test_p1_nonzero_S_inf_finite_output(self):
        p1 = torch.tensor(0.1)
        p2 = torch.tensor(0.0)
        S = torch.tensor(float("inf"))
        result = baseflow_6(p1, p2, S)
        assert torch.isfinite(result).all(), "p1=0.1, S=inf should produce finite output"

    def test_batch_mixed_edge_cases(self):
        p1 = torch.tensor([0.0, 0.5, 0.0, 0.1])
        p2 = torch.tensor([0.0, 0.0, 0.0, 0.0])
        S = torch.tensor([float("inf"), 10.0, 2e19, float("-inf")])
        result = baseflow_6(p1, p2, S)
        assert not torch.isnan(result).any(), "Batch mixed edge cases produced NaN"
        assert torch.isfinite(result).all(), "Batch mixed edge cases produced non-finite output"
        assert result[1].item() == pytest.approx(10.0), "Normal element in batch was altered"

    def test_gradient_finite_for_normal_case(self):
        p1 = torch.tensor(0.5, requires_grad=True)
        p2 = torch.tensor(0.0)
        S = torch.tensor(5.0, requires_grad=True)
        result = baseflow_6(p1, p2, S)
        result.backward()
        assert torch.isfinite(p1.grad).all(), "p1 gradient is non-finite"
        assert torch.isfinite(S.grad).all(), "S gradient is non-finite"

    def test_gradient_finite_for_zero_p1(self):
        p1 = torch.tensor(0.0, requires_grad=True)
        p2 = torch.tensor(0.0)
        S = torch.tensor(10.0, requires_grad=True)
        result = baseflow_6(p1, p2, S)
        result.backward()
        assert torch.isfinite(p1.grad).all(), "p1 gradient is non-finite when p1=0"
        assert torch.isfinite(S.grad).all(), "S gradient is non-finite when p1=0"


class TestBaseflow6NormalEquivalence:
    """Verify normal-case outputs match original implementation exactly."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_grid_equivalence(self, dtype):
        torch.manual_seed(20260629)
        n = 2000
        p1_vals = torch.rand(n, dtype=dtype)
        p2_vals = torch.rand(n, dtype=dtype) * 0.1
        S_vals = torch.rand(n, dtype=dtype) * 100.0

        for i in range(n):
            p1 = p1_vals[i].clone()
            p2 = p2_vals[i].clone()
            S = S_vals[i].clone()
            old = _baseflow_6_original(p1, p2, S)
            new = baseflow_6(p1, p2, S)
            assert torch.allclose(new, old, atol=0.0, rtol=0.0), (
                f"Divergence at normal input: p1={p1}, p2={p2}, S={S}"
            )

    def test_identity_at_zero_storage(self):
        p1 = torch.tensor(0.5)
        p2 = torch.tensor(0.0)
        S = torch.tensor(0.0)
        old = _baseflow_6_original(p1, p2, S)
        new = baseflow_6(p1, p2, S)
        assert torch.allclose(new, old, atol=0.0, rtol=0.0)

    def test_identity_below_threshold(self):
        p1 = torch.tensor(0.5)
        p2 = torch.tensor(10.0)
        S = torch.tensor(1.0)  # S < p2, gate nearly closed
        old = _baseflow_6_original(p1, p2, S)
        new = baseflow_6(p1, p2, S)
        assert torch.allclose(new, old, atol=1e-15, rtol=1e-15)

    def test_identity_above_threshold(self):
        p1 = torch.tensor(0.5)
        p2 = torch.tensor(1.0)
        S = torch.tensor(50.0)  # S >> p2, gate open
        old = _baseflow_6_original(p1, p2, S)
        new = baseflow_6(p1, p2, S)
        assert torch.allclose(new, old, atol=1e-15, rtol=1e-15)


class TestTCMForwardFinite:
    """Verify TCM forward pass remains finite with k2=0 edge case."""

    def test_tcm_step_finite_with_k2_zero(self):
        from models.core.tcm import create_initial_state, tcm_step

        n_grid, n_groups = 2, 1
        device = torch.device("cpu")
        nearzero = 1e-5

        states = create_initial_state(n_grid, n_groups, device, nearzero)

        P = torch.tensor([[1.0], [2.0], [0.0]])
        T = torch.tensor([[20.0], [22.0], [18.0]])
        PET = torch.tensor([[2.0], [3.0], [1.0]])

        phi = torch.tensor([[0.5]])
        rc = torch.tensor([[100.0]])
        gam = torch.tensor([[0.3]])
        k1 = torch.tensor([[0.1]])
        fa = torch.tensor([[0.0]])
        k2 = torch.tensor([[0.0]])  # k2=0 is a valid parameter value

        mean_P = torch.tensor([[1.0]])

        for t in range(P.shape[0]):
            Qsim, Ea, *next_states = tcm_step(
                P[t], T[t], PET[t],
                phi, rc, gam, k1, fa, k2,
                *states,
                nearzero,
                mean_P=mean_P,
            )
            assert torch.isfinite(Qsim).all(), f"t={t}: Qsim is non-finite"
            assert torch.isfinite(Ea).all(), f"t={t}: Ea is non-finite"
            for s_idx, s in enumerate(next_states):
                assert torch.isfinite(s).all(), f"t={t}: state S{s_idx+1} is non-finite"
            states = tuple(next_states)

    def test_tcm_step_finite_with_k2_zero_large_inputs(self):
        from models.core.tcm import create_initial_state, tcm_step

        n_grid, n_groups = 2, 1
        device = torch.device("cpu")
        nearzero = 1e-5

        states = create_initial_state(n_grid, n_groups, device, nearzero)

        P = torch.tensor([[100.0], [200.0], [500.0]])
        T = torch.tensor([[30.0], [32.0], [28.0]])
        PET = torch.tensor([[5.0], [6.0], [4.0]])

        phi = torch.tensor([[0.9]])
        rc = torch.tensor([[10.0]])  # small rc → fast saturation
        gam = torch.tensor([[0.8]])
        k1 = torch.tensor([[0.9]])
        fa = torch.tensor([[0.5]])
        k2 = torch.tensor([[0.0]])

        mean_P = torch.tensor([[1.0]])

        for _ in range(50):
            t = 1
            Qsim, Ea, *next_states = tcm_step(
                P[t], T[t], PET[t],
                phi, rc, gam, k1, fa, k2,
                *states,
                nearzero,
                mean_P=mean_P,
            )
            assert torch.isfinite(Qsim).all(), "Qsim is non-finite"
            assert torch.isfinite(Ea).all(), "Ea is non-finite"
            for s_idx, s in enumerate(next_states):
                assert torch.isfinite(s).all(), f"state S{s_idx+1} is non-finite"
            states = tuple(next_states)

    def test_tcm_backward_finite_with_k2_zero(self):
        from models.core.tcm import create_initial_state

        n_grid, n_groups = 1, 1
        device = torch.device("cpu")
        nearzero = 1e-5

        S1, S2, S3, S4 = create_initial_state(n_grid, n_groups, device, nearzero)

        P = torch.tensor([[1.0], [2.0], [0.0]])
        T = torch.tensor([[20.0], [22.0], [18.0]])
        PET = torch.tensor([[2.0], [3.0], [1.0]])

        phi = torch.tensor([[0.5]], requires_grad=True)
        rc = torch.tensor([[100.0]], requires_grad=True)
        gam = torch.tensor([[0.3]], requires_grad=True)
        k1 = torch.tensor([[0.1]], requires_grad=True)
        fa = torch.tensor([[0.0]], requires_grad=True)
        k2 = torch.tensor([[0.0]], requires_grad=True)

        mean_P = torch.tensor([[1.0]])

        from models.core.tcm import tcm_step

        Qsim_sum = torch.tensor(0.0)
        states = (S1, S2, S3, S4)
        for t in range(P.shape[0]):
            Qsim, Ea, *next_states = tcm_step(
                P[t], T[t], PET[t],
                phi, rc, gam, k1, fa, k2,
                *states,
                nearzero,
                mean_P=mean_P,
            )
            Qsim_sum = Qsim_sum + Qsim.sum()
            states = tuple(s.detach() for s in next_states)

        Qsim_sum.backward()

        for name, param in [("phi", phi), ("rc", rc), ("gam", gam), ("k1", k1), ("fa", fa), ("k2", k2)]:
            assert torch.isfinite(param.grad).all(), f"{name} gradient is non-finite"
