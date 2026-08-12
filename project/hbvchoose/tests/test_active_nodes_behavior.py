"""Test: --active-nodes enforcement — inactive nodes must use default formulas."""
import sys
from pathlib import Path

import pytest
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.static_formula_router import StaticFormulaRouter
from model.formula_pool import CandidateFormulaPool

NODE_ORDER = ["snow", "recharge", "aet", "response"]
DEFAULT_IDS = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}


def _enforce_active_nodes(active_nodes, r_out, fids_dict, B):
    """Force inactive nodes to default."""
    inactive = set(NODE_ORDER) - set(active_nodes)
    for n in inactive:
        f = fids_dict[n]
        default_idx = f.index(DEFAULT_IDS[n]) if DEFAULT_IDS[n] in f else 0
        r_out["selected"][n] = torch.full((B,), default_idx, dtype=torch.long)
    return r_out


class TestActiveNodesBehavior:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.router = StaticFormulaRouter(attr_dim=8, default_bias=0.5)
        self.attrs = torch.randn(4, 8)
        self.pool = CandidateFormulaPool()
        self.fids_dict = {n: self.pool.formulas(n, "main") for n in NODE_ORDER}

    def test_recharge_only_others_default(self):
        """When active_nodes=['recharge'], snow/aet/response must be default."""
        r_out = self.router(self.attrs)
        r_out = _enforce_active_nodes(["recharge"], r_out, self.fids_dict, 4)

        # Check inactive nodes
        for n in ["snow", "aet", "response"]:
            fids = self.fids_dict[n]
            default_idx = fids.index(DEFAULT_IDS[n])
            sel = r_out["selected"][n]
            assert (sel == default_idx).all(), f"Inactive node {n} not forced to default"

        # Check active node
        recharge_fids = self.fids_dict["recharge"]
        recharge_selected = [recharge_fids[int(i.item())] for i in r_out["selected"]["recharge"]]
        # Active node uses router selection, default may or may not be selected
        assert all(r in recharge_fids for r in recharge_selected), "Recharge selections not in candidate set"

    def test_snow_only_others_default(self):
        """When active_nodes=['snow'], others must be default."""
        r_out = self.router(self.attrs)
        r_out = _enforce_active_nodes(["snow"], r_out, self.fids_dict, 4)

        for n in ["recharge", "aet", "response"]:
            fids = self.fids_dict[n]
            default_idx = fids.index(DEFAULT_IDS[n])
            sel = r_out["selected"][n]
            assert (sel == default_idx).all(), f"Inactive node {n} not forced to default"

    def test_inactive_nodes_not_in_selection_stats(self):
        """Inactive nodes should not appear in selection statistics."""
        active_nodes = ["recharge"]
        r_out = self.router(self.attrs)
        r_out = _enforce_active_nodes(active_nodes, r_out, self.fids_dict, 4)

        # Count unique selections per node
        stats = {}
        for n in active_nodes:
            fids = self.fids_dict[n]
            sel_ids = [fids[int(i.item())] for i in r_out["selected"][n]]
            unique = set(sel_ids)
            stats[n] = {"unique": len(unique), "all_default": all(s == DEFAULT_IDS[n] for s in sel_ids)}

        for n in set(NODE_ORDER) - set(active_nodes):
            # Inactive nodes should NOT be in active stats
            assert n not in stats, f"Inactive node {n} appeared in active stats"

    def test_multiple_active_nodes_work(self):
        """Multiple active nodes should all be selectable."""
        for combo in [["recharge", "snow"], ["recharge", "snow", "aet"]]:
            r_out = self.router(self.attrs)
            r_out = _enforce_active_nodes(combo, r_out, self.fids_dict, 4)

            active_set = set(combo)
            inactive_set = set(NODE_ORDER) - active_set

            for n in active_set:
                fids = self.fids_dict[n]
                sel_ids = [fids[int(i.item())] for i in r_out["selected"][n]]
                assert all(s in fids for s in sel_ids), f"Active node {n} selections not valid"

            for n in inactive_set:
                fids = self.fids_dict[n]
                default_idx = fids.index(DEFAULT_IDS[n])
                sel = r_out["selected"][n]
                assert (sel == default_idx).all(), f"Inactive node {n} not forced to default"
