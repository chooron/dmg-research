"""Tests for formula_registry accessors and CandidateFormulaPool."""

import sys

import pytest
import torch

sys.path.insert(0, "/home/jingxin/code/dmg-research/project/hbvchoose")

from model.flux.formula_registry import (
    get_node_formulas,
    get_all_main_formulas,
    get_routing_policy,
    list_formula_nodes,
)
from model.formula_pool import CandidateFormulaPool


def _t(*vals):
    return torch.tensor(vals, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Registry accessor tests
# ---------------------------------------------------------------------------

class TestRegistryAccessors:
    def test_list_formula_nodes(self):
        nodes = list_formula_nodes()
        assert set(nodes) == {"aet", "recharge", "response", "snow"}

    def test_get_node_formulas_main(self):
        expected = {"snow": 3, "recharge": 3, "aet": 3, "response": 2}  # Q5 -> extension_only
        for node, count in expected.items():
            entries = get_node_formulas(node, "main")
            assert len(entries) == count, f"{node}: expected {count} main formulas, got {len(entries)}"

    def test_get_node_formulas_unknown_node(self):
        with pytest.raises(ValueError, match="Unknown registry node"):
            get_node_formulas("nonexistent")

    def test_get_node_formulas_unknown_status(self):
        entries = get_node_formulas("snow", "no_such_status")
        assert entries == []

    def test_get_all_main_formulas(self):
        all_main = get_all_main_formulas()
        assert set(all_main) == {"snow", "recharge", "aet", "response"}
        expected = {"snow": 3, "recharge": 3, "aet": 3, "response": 2}
        for node, count in expected.items():
            assert len(all_main[node]) == count, f"{node}: expected {count}, got {len(all_main[node])}"

    def test_routing_policy_recharge(self):
        assert get_routing_policy("recharge") == "hard_only"

    def test_routing_policy_sparse(self):
        for node in ["snow", "aet", "response"]:
            assert get_routing_policy(node) == "sparse_or_top1"

    def test_routing_policy_unknown_node(self):
        with pytest.raises(ValueError, match="Unknown registry node"):
            get_routing_policy("nonexistent")


# ---------------------------------------------------------------------------
# CandidateFormulaPool tests
# ---------------------------------------------------------------------------

class TestCandidateFormulaPool:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pool = CandidateFormulaPool()

    def test_nodes(self):
        assert set(self.pool.nodes()) == {"aet", "recharge", "response", "snow"}

    def test_formulas_main_count(self):
        expected = {"snow": 3, "recharge": 3, "aet": 3, "response": 2}
        for node, count in expected.items():
            fids = self.pool.formulas(node, "main")
            assert len(fids) == count, f"{node}: {fids}"

    def test_formulas_main_ids(self):
        assert self.pool.formulas("snow") == ["S0", "S4", "S5"]
        assert self.pool.formulas("recharge") == ["R0", "R4", "R5"]
        assert self.pool.formulas("aet") == ["E0", "E3", "E4"]
        assert self.pool.formulas("response") == ["Q0", "Q2"]  # Q5 -> extension_only

    def test_routing_policies(self):
        assert self.pool.routing_policy("snow") == "sparse_or_top1"
        assert self.pool.routing_policy("recharge") == "hard_only"
        assert self.pool.routing_policy("aet") == "sparse_or_top1"
        assert self.pool.routing_policy("response") == "sparse_or_top1"

    def test_get_formula_returns_callable(self):
        for node, fids in [
            ("snow", ["S0", "S4", "S5"]),
            ("recharge", ["R0", "R4", "R5"]),
            ("aet", ["E0", "E3", "E4"]),
            ("response", ["Q0", "Q2", "Q5"]),
        ]:
            for fid in fids:
                fn = self.pool.get_formula(node, fid)
                assert callable(fn), f"{node}/{fid} not callable"

    def test_get_formula_unknown(self):
        with pytest.raises(ValueError):
            self.pool.get_formula("snow", "ZZ9")

    # -- call tests --------------------------------------------------------

    def test_call_S0(self):
        r = self.pool.call_formula("snow", "S0", T=_t(5.0), TT=_t(0.5), CFMAX=_t(3.0), SWE=_t(20.0))
        assert torch.isfinite(r).all()

    def test_call_S4(self):
        r = self.pool.call_formula("snow", "S4",
                                   T=_t(5.0), TT=_t(0.5), CFMAX_0=_t(3.0),
                                   a_s=_t(0.3), phi_s=_t(172.0), doy=_t(172.0), SWE=_t(20.0))
        assert torch.isfinite(r).all()

    def test_call_S5(self):
        r = self.pool.call_formula("snow", "S5", T=_t(5.0), TT=_t(0.5), CFMAX=_t(3.0), c_m=_t(0.3), SWE=_t(20.0))
        assert torch.isfinite(r).all()

    def test_call_R0(self):
        r = self.pool.call_formula("recharge", "R0", I=_t(5.0), SM=_t(100.0), FC=_t(200.0), beta=_t(2.0))
        assert torch.isfinite(r).all()

    def test_call_R4(self):
        r = self.pool.call_formula("recharge", "R4", I=_t(5.0), SM=_t(100.0), FC=_t(200.0), a_r=_t(10.0), c_r=_t(0.5))
        assert torch.isfinite(r).all()

    def test_call_R5(self):
        r = self.pool.call_formula("recharge", "R5", I=_t(5.0), SM=_t(100.0), FC=_t(200.0), b_v=_t(1.0))
        assert torch.isfinite(r).all()

    def test_call_E0(self):
        r = self.pool.call_formula("aet", "E0", PET=_t(3.0), SM=_t(100.0), LP=_t(0.8), FC=_t(200.0))
        assert torch.isfinite(r).all()

    def test_call_E3(self):
        r = self.pool.call_formula("aet", "E3", PET=_t(3.0), SM=_t(100.0), FC=_t(200.0), gamma_E=_t(1.2))
        assert torch.isfinite(r).all()

    def test_call_E4(self):
        r = self.pool.call_formula("aet", "E4", PET=_t(3.0), SM=_t(100.0), FC=_t(200.0), s_w=_t(0.1), s_o=_t(0.6))
        assert torch.isfinite(r).all()

    def test_call_Q0(self):
        Q0, Q1, Q2, Q = self.pool.call_formula(
            "response", "Q0", SUZ=_t(10.0), SLZ=_t(50.0),
            K_0=_t(0.3), K_1=_t(0.1), K_2=_t(0.05), UZL=_t(10.0))
        for t in (Q0, Q1, Q2, Q):
            assert torch.isfinite(t).all()

    def test_call_Q2(self):
        Quz, Qlz, Q = self.pool.call_formula(
            "response", "Q2", SUZ=_t(10.0), SLZ=_t(50.0),
            K_1=_t(0.1), K_2=_t(0.05), alpha_Q=_t(1.2))
        for t in (Quz, Qlz, Q):
            assert torch.isfinite(t).all()

    def test_call_Q5(self):
        Rim, Rdel, Q1, Q2, Q = self.pool.call_formula(
            "response", "Q5", R_in=_t(5.0), S_1=_t(10.0), S_2=_t(30.0),
            PART=_t(0.7), K_1=_t(0.1), K_2=_t(0.05))
        for t in (Rim, Rdel, Q1, Q2, Q):
            assert torch.isfinite(t).all()

    # -- no NaN / Inf across all formulas ----------------------------------

    def test_all_main_formulas_finite(self):
        """Sweep all main formulas with typical inputs, assert no NaN/Inf."""
        pool = self.pool
        calls = [
            ("snow", "S0", dict(T=_t(5.0), TT=_t(0.5), CFMAX=_t(3.0), SWE=_t(20.0))),
            ("snow", "S4", dict(T=_t(5.0), TT=_t(0.5), CFMAX_0=_t(3.0),
                                a_s=_t(0.3), phi_s=_t(172.0), doy=_t(172.0), SWE=_t(20.0))),
            ("snow", "S5", dict(T=_t(5.0), TT=_t(0.5), CFMAX=_t(3.0), c_m=_t(0.3), SWE=_t(20.0))),
            ("recharge", "R0", dict(I=_t(5.0), SM=_t(100.0), FC=_t(200.0), beta=_t(2.0))),
            ("recharge", "R4", dict(I=_t(5.0), SM=_t(100.0), FC=_t(200.0), a_r=_t(10.0), c_r=_t(0.5))),
            ("recharge", "R5", dict(I=_t(5.0), SM=_t(100.0), FC=_t(200.0), b_v=_t(1.0))),
            ("aet", "E0", dict(PET=_t(3.0), SM=_t(100.0), LP=_t(0.8), FC=_t(200.0))),
            ("aet", "E3", dict(PET=_t(3.0), SM=_t(100.0), FC=_t(200.0), gamma_E=_t(1.2))),
            ("aet", "E4", dict(PET=_t(3.0), SM=_t(100.0), FC=_t(200.0), s_w=_t(0.1), s_o=_t(0.6))),
            ("response", "Q0", dict(SUZ=_t(10.0), SLZ=_t(50.0), K_0=_t(0.3), K_1=_t(0.1), K_2=_t(0.05), UZL=_t(10.0))),
            ("response", "Q2", dict(SUZ=_t(10.0), SLZ=_t(50.0), K_1=_t(0.1), K_2=_t(0.05), alpha_Q=_t(1.2))),
            ("response", "Q5", dict(R_in=_t(5.0), S_1=_t(10.0), S_2=_t(30.0), PART=_t(0.7), K_1=_t(0.1), K_2=_t(0.05))),
        ]
        for node, fid, kw in calls:
            result = pool.call_formula(node, fid, **kw)
            tensors = result if isinstance(result, tuple) else (result,)
            for t in tensors:
                assert not torch.any(torch.isnan(t)), f"{node}/{fid} has NaN"
                assert not torch.any(torch.isinf(t)), f"{node}/{fid} has Inf"
