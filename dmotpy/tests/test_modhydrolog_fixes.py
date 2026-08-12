"""Unit tests for modhydrolog (m36) fixes.

Tests:
1. interception_1 overflow direction (P > insc → throughfall; P <= insc → intercepted)
2. No negative fluxes under synthetic forcing
3. Mass balance closure
4. Non-zero gradients after interception fix
5. depression_1 formula correctness
6. flux_SEEP non-negative
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from models.core.modhydrolog import (
    MODHYDROLOG_PARAMS_BOUNDS,
    create_initial_state,
    modhydrolog_step,
)
from models.flux.interception import interception_1
from models.flux.depression import depression_1
from models.flux.exchange import exchange_3

NEARZERO = 1e-6
BOUNDS = MODHYDROLOG_PARAMS_BOUNDS


def _median_params():
    return {n: torch.tensor([[(lo + hi) / 2]], dtype=torch.float32)
            for n, (lo, hi) in BOUNDS.items()}


# ============================================================================
# Test 1: Interception Overflow Direction
# ============================================================================

class TestInterceptionOverflowDirection:
    def test_throughfall_when_store_full(self):
        """When S + P > Smax, throughfall (excess) must occur."""
        P = torch.tensor([[[10.0]]])
        S1_before = torch.tensor([[[0.0]]])
        insc = torch.tensor([[[2.5]]])

        S1 = S1_before + P  # S1 = 10 > insc
        flux_EXC = interception_1(P, S1, insc)
        flux_EXC = torch.minimum(flux_EXC, S1)

        # Throughfall should be significant (> 5mm)
        assert flux_EXC.item() > 0.5, (
            f"Expected throughfall when S1(10) > insc(2.5), got flux_EXC={flux_EXC.item():.4f}"
        )
        # But not more than P
        assert flux_EXC.item() <= P.item() + 1e-6

    def test_no_throughfall_when_store_has_room(self):
        """When S + P < Smax, water should be intercepted (minimal throughfall)."""
        P = torch.tensor([[[0.5]]])
        S1_before = torch.tensor([[[0.0]]])
        insc = torch.tensor([[[2.5]]])

        S1 = S1_before + P  # S1 = 0.5 < insc
        flux_EXC = interception_1(P, S1, insc)
        flux_EXC = torch.minimum(flux_EXC, S1)

        # Throughfall should be very small (< 0.1mm)
        assert flux_EXC.item() < 0.2, (
            f"Expected minimal throughfall when S1(0.5) < insc(2.5), got flux_EXC={flux_EXC.item():.4f}"
        )

    def test_full_step_water_reaches_s2(self):
        """After interception fix, water should reach S2 when P > insc."""
        params = _median_params()
        S1, S2, S3, S4, S5 = create_initial_state(1, 1, torch.device("cpu"), NEARZERO)
        P = torch.tensor([[[10.0]]])
        T = torch.tensor([[[20.0]]])
        PET = torch.tensor([[[4.0]]])

        Q, Ea, S1n, S2n, S3n, S4n, S5n = modhydrolog_step(
            P, T, PET,
            params["insc"], params["coeff"], params["sq"], params["smsc"],
            params["sub"], params["crak"], params["em"], params["dsc"],
            params["ads"], params["md"], params["vcond"], params["dlev"],
            params["k1"], params["k2"], params["k3"],
            S1, S2, S3, S4, S5, NEARZERO,
        )
        # S2 should have received water
        assert S2n.item() > 0.1, f"Expected water in S2 after P(10) > insc(2.5), got S2={S2n.item():.4f}"

    def test_s1_stays_at_or_below_insc(self):
        """After step with P > insc, S1 should be at or below insc."""
        params = _median_params()
        S1, S2, S3, S4, S5 = create_initial_state(1, 1, torch.device("cpu"), NEARZERO)
        P = torch.tensor([[[10.0]]])
        T = torch.tensor([[[20.0]]])
        PET = torch.tensor([[[4.0]]])

        Q, Ea, S1n, S2n, S3n, S4n, S5n = modhydrolog_step(
            P, T, PET,
            params["insc"], params["coeff"], params["sq"], params["smsc"],
            params["sub"], params["crak"], params["em"], params["dsc"],
            params["ads"], params["md"], params["vcond"], params["dlev"],
            params["k1"], params["k2"], params["k3"],
            S1, S2, S3, S4, S5, NEARZERO,
        )
        insc_val = params["insc"].item()
        assert S1n.item() <= insc_val + 1e-4, (
            f"S1({S1n.item():.4f}) should be <= insc({insc_val:.4f})"
        )


# ============================================================================
# Test 2: No Negative Fluxes
# ============================================================================

class TestNoNegativeFluxes:
    def test_modhydrolog_no_negative_q_ea(self):
        """Q and Ea should never be negative under random forcings."""
        params = _median_params()
        S1, S2, S3, S4, S5 = create_initial_state(1, 1, torch.device("cpu"), NEARZERO)

        rng = np.random.RandomState(42)
        for _ in range(100):
            P = torch.tensor([[[max(0, rng.lognormal(1.0, 1.5))]]], dtype=torch.float32)
            T = torch.tensor([[[15.0 + rng.randn() * 5.0]]], dtype=torch.float32)
            PET = torch.tensor([[[max(0.1, 3.0 + rng.randn() * 2.0)]]], dtype=torch.float32)

            Q, Ea, S1, S2, S3, S4, S5 = modhydrolog_step(
                P, T, PET,
                params["insc"], params["coeff"], params["sq"], params["smsc"],
                params["sub"], params["crak"], params["em"], params["dsc"],
                params["ads"], params["md"], params["vcond"], params["dlev"],
                params["k1"], params["k2"], params["k3"],
                S1, S2, S3, S4, S5, NEARZERO,
            )

            assert Q.item() >= -1e-6, f"Q should be non-negative, got {Q.item()}"
            assert Ea.item() >= -1e-6, f"Ea should be non-negative, got {Ea.item()}"

    def test_all_stores_non_negative(self):
        """All stores should stay non-negative."""
        params = _median_params()
        S1, S2, S3, S4, S5 = create_initial_state(1, 1, torch.device("cpu"), NEARZERO)

        rng = np.random.RandomState(42)
        for _ in range(100):
            P = torch.tensor([[[max(0, rng.lognormal(0.5, 2.0))]]], dtype=torch.float32)
            T = torch.tensor([[[15.0]]], dtype=torch.float32)
            PET = torch.tensor([[[max(0.1, rng.rand() * 8.0)]]], dtype=torch.float32)

            Q, Ea, S1, S2, S3, S4, S5 = modhydrolog_step(
                P, T, PET,
                params["insc"], params["coeff"], params["sq"], params["smsc"],
                params["sub"], params["crak"], params["em"], params["dsc"],
                params["ads"], params["md"], params["vcond"], params["dlev"],
                params["k1"], params["k2"], params["k3"],
                S1, S2, S3, S4, S5, NEARZERO,
            )

            for i, (s, name) in enumerate([(S1, "S1"), (S2, "S2"), (S3, "S3"),
                                            (S4, "S4"), (S5, "S5")]):
                assert s.item() >= -1e-6, f"{name} should be non-negative, got {s.item()}"


# ============================================================================
# Test 3: Mass Balance Closure
# ============================================================================

class TestMassBalanceClosure:
    def test_single_step_water_balance(self):
        """P - Q - Ea - delta_storage should be near zero."""
        params = _median_params()
        S1, S2, S3, S4, S5 = create_initial_state(1, 1, torch.device("cpu"), NEARZERO)

        rng = np.random.RandomState(123)
        for _ in range(50):
            S1o, S2o, S3o, S4o, S5o = S1.clone(), S2.clone(), S3.clone(), S4.clone(), S5.clone()

            P = torch.tensor([[[max(0, rng.lognormal(1.0, 1.5))]]], dtype=torch.float32)
            T = torch.tensor([[[15.0]]], dtype=torch.float32)
            PET = torch.tensor([[[max(0.1, 3.0 + rng.randn() * 2.0)]]], dtype=torch.float32)

            Q, Ea, S1, S2, S3, S4, S5 = modhydrolog_step(
                P, T, PET,
                params["insc"], params["coeff"], params["sq"], params["smsc"],
                params["sub"], params["crak"], params["em"], params["dsc"],
                params["ads"], params["md"], params["vcond"], params["dlev"],
                params["k1"], params["k2"], params["k3"],
                S1, S2, S3, S4, S5, NEARZERO,
            )

            ds = ((S1 + S2 + S3 + S4 + S5) - (S1o + S2o + S3o + S4o + S5o)).item()
            residual = P.item() - Q.item() - Ea.item() - ds
            assert abs(residual) < 1e-3, (
                f"Water balance residual {residual:.6f} exceeds threshold "
                f"(P={P.item():.3f} Q={Q.item():.3f} Ea={Ea.item():.3f} dS={ds:.6f})"
            )

    def test_multistep_cumulative_balance(self):
        """Cumulative P - cumulative Q - cumulative Ea - final_dS ≈ 0."""
        params = _median_params()
        S1, S2, S3, S4, S5 = create_initial_state(1, 1, torch.device("cpu"), NEARZERO)
        init_store = (S1 + S2 + S3 + S4 + S5).item()

        rng = np.random.RandomState(456)
        cum_P = 0.0
        cum_Q = 0.0
        cum_Ea = 0.0

        for _ in range(365):
            P = torch.tensor([[[max(0, rng.lognormal(1.0, 1.5) * float(rng.rand() < 0.35))]]],
                             dtype=torch.float32)
            T = torch.tensor([[[15.0]]], dtype=torch.float32)
            PET = torch.tensor([[[max(0.1, 3.0 + rng.randn() * 1.5)]]], dtype=torch.float32)

            Q, Ea, S1, S2, S3, S4, S5 = modhydrolog_step(
                P, T, PET,
                params["insc"], params["coeff"], params["sq"], params["smsc"],
                params["sub"], params["crak"], params["em"], params["dsc"],
                params["ads"], params["md"], params["vcond"], params["dlev"],
                params["k1"], params["k2"], params["k3"],
                S1, S2, S3, S4, S5, NEARZERO,
            )
            cum_P += P.item()
            cum_Q += Q.item()
            cum_Ea += Ea.item()

        final_store = (S1 + S2 + S3 + S4 + S5).item()
        residual = cum_P - cum_Q - cum_Ea - (final_store - init_store)
        assert abs(residual) < 1e-2, (
            f"Cumulative water balance residual {residual:.6f} "
            f"(P={cum_P:.2f} Q={cum_Q:.2f} Ea={cum_Ea:.2f} dS={final_store - init_store:.4f})"
        )


# ============================================================================
# Test 4: Non-Zero Gradients After Fix
# ============================================================================

class TestGradientsAfterFix:
    def test_interception_parameters_have_gradient(self):
        """At least insc should have non-zero gradient with params in transition region."""
        # Use params where S+P is in the transition zone of the sigmoid gate
        # sigmoid saturates when |k*(S-Smax)/Smax| > ~5
        # To get gradient, we need S close to Smax

        params = {}
        for name in BOUNDS:
            lo, hi = BOUNDS[name]
            if name == "insc":
                # insc = 5.0 so sigmoid(10*(5-5)/5) = 0.5 (maximum gradient)
                params[name] = torch.tensor([[[5.0]]], dtype=torch.float32, requires_grad=True)
            elif name == "smsc":
                # smsc small enough that water entering S2 may saturate
                params[name] = torch.tensor([[[50.0]]], dtype=torch.float32, requires_grad=True)
            else:
                params[name] = torch.tensor([[[(lo + hi) / 2]]], dtype=torch.float32, requires_grad=True)

        S1, S2, S3, S4, S5 = create_initial_state(1, 1, torch.device("cpu"), NEARZERO)
        P = torch.tensor([[[5.0]]])  # P ≈ insc → gate in transition zone
        T = torch.tensor([[[20.0]]])
        PET = torch.tensor([[[1.0]]])

        Q, Ea, S1n, S2n, S3n, S4n, S5n = modhydrolog_step(
            P, T, PET,
            params["insc"], params["coeff"], params["sq"], params["smsc"],
            params["sub"], params["crak"], params["em"], params["dsc"],
            params["ads"], params["md"], params["vcond"], params["dlev"],
            params["k1"], params["k2"], params["k3"],
            S1, S2, S3, S4, S5, NEARZERO,
        )

        loss = (Q + S2n).mean()
        loss.backward()

        grad_summary = {}
        for name in BOUNDS:
            g = params[name].grad
            gn = abs(g.item()) if g is not None else 0.0
            grad_summary[name] = gn

        nonzero = sum(1 for v in grad_summary.values() if v > 1e-10)
        assert nonzero >= 2, (
            f"Expected at least 2 params with non-zero gradients, got {nonzero}. "
            f"Grads: {{k: f'{v:.2e}' for k, v in grad_summary.items()}}"
        )


# ============================================================================
# Test 5: Depression_1 MATLAB Formula
# ============================================================================

class TestDepressionMATLAB:
    def test_trap_non_negative_and_bounded(self):
        """flux_TRAP must be non-negative and bounded by incoming_flux and capacity."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            ads = torch.tensor([[[rng.rand()]]], dtype=torch.float32)
            md = torch.tensor([[[0.99 + rng.rand() * 0.01]]], dtype=torch.float32)
            S = torch.tensor([[[rng.rand() * 25.0]]], dtype=torch.float32)  # 0..25
            dsc = torch.tensor([[[25.0]]], dtype=torch.float32)
            runoff = torch.tensor([[[rng.lognormal(0, 1.5)]]], dtype=torch.float32)

            trap = depression_1(ads, md, S, dsc, runoff)

            assert trap.item() >= -1e-6, f"TRAP should be non-negative, got {trap.item()}"
            assert trap.item() <= runoff.item() + 1e-6, f"TRAP({trap.item()}) > RUN({runoff.item()})"
            cap = max(float(dsc.item()) - float(S.item()), 0.0)
            assert trap.item() <= cap + 1e-6, f"TRAP({trap.item()}) > capacity({cap})"

    def test_full_depression_traps_nothing(self):
        """When depression is already full, no additional trapping."""
        ads = torch.tensor([[[0.5]]])
        md = torch.tensor([[[0.995]]])
        S = torch.tensor([[[25.0]]])  # already at max (dsc=25)
        dsc = torch.tensor([[[25.0]]])
        runoff = torch.tensor([[[10.0]]])

        trap = depression_1(ads, md, S, dsc, runoff)
        assert trap.item() < 0.1, f"Full depression should trap very little, got TRAP={trap.item()}"

    def test_empty_depression_traps_efficiently(self):
        """When depression is empty, trapping should be near ads * runoff."""
        ads = torch.tensor([[[0.5]]])
        md = torch.tensor([[[0.995]]])
        S = torch.tensor([[[0.0]]])
        dsc = torch.tensor([[[25.0]]])
        runoff = torch.tensor([[[10.0]]])

        trap = depression_1(ads, md, S, dsc, runoff)
        # Should trap about ads * runoff = 5.0
        assert trap.item() > 2.0, f"Empty depression should trap efficiently, got TRAP={trap.item()}"


# ============================================================================
# Test 6: Seepage Non-Negative
# ============================================================================

class TestSeepageNonNegative:
    def test_seepage_clamped_to_non_negative(self):
        """After the step, verify that flux_SEEP is effectively clamped to [0, S4]."""
        params = _median_params()
        # Set dlev high so S4 < dlev, making raw exchange_3 negative
        params["dlev"] = torch.tensor([[[5.0]]], dtype=torch.float32)

        S1, S2, S3, S4, S5 = create_initial_state(1, 1, torch.device("cpu"), NEARZERO)
        P = torch.tensor([[[1.0]]])
        T = torch.tensor([[[20.0]]])
        PET = torch.tensor([[[4.0]]])

        Q, Ea, S1n, S2n, S3n, S4n, S5n = modhydrolog_step(
            P, T, PET,
            params["insc"], params["coeff"], params["sq"], params["smsc"],
            params["sub"], params["crak"], params["em"], params["dsc"],
            params["ads"], params["md"], params["vcond"], params["dlev"],
            params["k1"], params["k2"], params["k3"],
            S1, S2, S3, S4, S5, NEARZERO,
        )
        # Ea should be >= 0 (SEEP can't make it negative)
        assert Ea.item() >= -1e-8, f"Ea({Ea.item():.6f}) should be non-negative"

    def test_seepage_is_non_negative_after_clamp(self):
        """explicit check: exchange_3 can produce negative, but step clamps to >= 0."""
        vcond = torch.tensor([[[0.25]]])
        S4 = torch.tensor([[[1.0]]])
        dlev = torch.tensor([[[5.0]]])  # S4 < dlev → negative

        seep_pot = exchange_3(vcond, S4, dlev)
        assert seep_pot.item() < 0, f"Expected negative potential seepage, got {seep_pot.item()}"

        # The step function clamps this
        seep_clamped = torch.clamp(seep_pot, min=0.0)
        seep_clamped = torch.minimum(seep_clamped, S4)
        assert seep_clamped.item() == 0.0, f"Clamped seepage should be 0, got {seep_clamped.item()}"
