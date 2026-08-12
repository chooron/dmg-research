"""Tests for formula combination enumeration and HbvFormulaStatic."""

import sys
from pathlib import Path

import pytest
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_formula_static import HbvFormulaStatic
from model.formula_pool import CandidateFormulaPool

NODES = ["snow", "recharge", "aet", "response"]


class TestCombinationEnumeration:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pool = CandidateFormulaPool()

    def test_node_formulas_exist(self):
        for n in NODES:
            assert len(self.pool.formulas(n, "main")) >= 1

    def test_main_combination_count_after_q5_downgrade(self):
        """Q5 is extension_only → response main = Q0, Q2 → 3×3×3×2 = 54."""
        total = 1
        for n in NODES:
            total *= len(self.pool.formulas(n, "main"))
        assert total == 54, f"Expected 54 main combos, got {total}"

    def test_default_combo_exists(self):
        fids = {n: self.pool.formulas(n, "main") for n in NODES}
        assert "S0" in fids["snow"]
        assert "R0" in fids["recharge"]
        assert "E0" in fids["aet"]
        assert "Q0" in fids["response"]

    def test_q5_not_in_main(self):
        """Q5 must NOT be in main pool after downgrade."""
        main_response = self.pool.formulas("response", "main")
        assert "Q5" not in main_response

    def test_all_main_in_some_combo(self):
        seen = {n: set() for n in NODES}
        fids = {n: self.pool.formulas(n, "main") for n in NODES}
        for sn in fids["snow"]:
            for rc in fids["recharge"]:
                for ae in fids["aet"]:
                    for rs in fids["response"]:
                        seen["snow"].add(sn)
                        seen["recharge"].add(rc)
                        seen["aet"].add(ae)
                        seen["response"].add(rs)
        for n in NODES:
            assert seen[n] == set(fids[n])


class TestHbvFormulaStatic:
    def _synth_forcing(self, length=40):
        P = torch.tensor([0.0, 1.0, 5.0, 20.0, 0.0, 0.0, 10.0, 50.0, 2.0, 0.0] * (length // 10),
                         dtype=torch.float64)[:length]
        T = torch.tensor([-5.0, -2.0, 0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 12.0, 8.0] * (length // 10),
                         dtype=torch.float64)[:length]
        PET = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5] * (length // 10),
                           dtype=torch.float64)[:length]
        return P, T, PET

    def test_default_forward(self):
        model = HbvFormulaStatic(warm_up=10)
        P, T, PET = self._synth_forcing(40)
        diag = model.simulate(P, T, PET)
        assert not torch.any(torch.isnan(diag["Qsim"]))

    def test_diagnostics_has_water_balance_fields(self):
        model = HbvFormulaStatic(warm_up=10)
        P, T, PET = self._synth_forcing(30)
        diag = model.simulate(P, T, PET)
        required = ["precipitation_total", "aet_total", "q_total", "storage_change",
                    "water_balance_residual", "relative_water_balance_error",
                    "trace"]
        for key in required:
            assert key in diag, f"Missing: {key}"

    def test_water_balance_finite(self):
        model = HbvFormulaStatic(warm_up=10)
        P, T, PET = self._synth_forcing(30)
        diag = model.simulate(P, T, PET)
        wb_err = diag["relative_water_balance_error"]
        import math
        assert math.isfinite(wb_err)

    def test_routing_output_both_q(self):
        model = HbvFormulaStatic(warm_up=10, apply_routing=False)
        P, T, PET = self._synth_forcing(30)
        diag = model.simulate(P, T, PET)
        assert "Q_raw" in diag
        assert "Qsim" in diag
        assert diag["routing_applied"] is False

    def test_all_main_combos_no_nan(self):
        pool = CandidateFormulaPool()
        fids = {n: pool.formulas(n, "main") for n in NODES}
        P, T, PET = self._synth_forcing(10)
        for sn in fids["snow"]:
            for rc in fids["recharge"]:
                for ae in fids["aet"]:
                    for rs in fids["response"]:
                        model = HbvFormulaStatic(
                            formula_config={"snow": sn, "recharge": rc, "aet": ae, "response": rs},
                            warm_up=0,
                        )
                        diag = model.simulate(P, T, PET)
                        assert not torch.any(torch.isnan(diag["Qsim"])), f"NaN {sn}_{rc}_{ae}_{rs}"
