"""Tests for StaticFormulaRouter and HbvStaticFormulaRouter."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.static_formula_router import StaticFormulaRouter
from model.hbv_static_router import HbvStaticFormulaRouter
from model.hbv_formula_static import HbvFormulaStatic
from model.formula_pool import CandidateFormulaPool


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_attrs(B, attr_dim=8):
    return torch.randn(B, attr_dim)


def _make_forcing(T=100, B=4):
    P = (torch.rand(T, B) * 5.0).float()
    Tt = (torch.rand(T, B) * 20.0 - 5.0).float()
    PET = (torch.rand(T, B) * 5.0).float()
    return torch.stack([P, Tt, PET], dim=-1)


# ---------------------------------------------------------------------------
# StaticFormulaRouter tests
# ---------------------------------------------------------------------------

class TestStaticFormulaRouter:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.attr_dim = 8
        self.router = StaticFormulaRouter(attr_dim=self.attr_dim, default_bias=2.0, hard_eval=True)

    def test_reads_from_registry(self):
        pool = CandidateFormulaPool()
        for node in ["snow", "recharge", "aet", "response"]:
            reg_fids = pool.formulas(node, "main")
            router_fids = self.router.formula_ids[node]
            assert router_fids == reg_fids, f"{node}: router has {router_fids}, registry has {reg_fids}"

    def test_shape_snow(self):
        B = 4
        attrs = _make_attrs(B, self.attr_dim)
        out = self.router(attrs)
        assert out["weights"]["snow"].shape == (B, 3)

    def test_shape_recharge(self):
        B = 4
        attrs = _make_attrs(B, self.attr_dim)
        out = self.router(attrs)
        assert out["weights"]["recharge"].shape == (B, 3)

    def test_shape_aet(self):
        B = 4
        attrs = _make_attrs(B, self.attr_dim)
        out = self.router(attrs)
        assert out["weights"]["aet"].shape == (B, 3)

    def test_shape_response(self):
        B = 4
        attrs = _make_attrs(B, self.attr_dim)
        out = self.router(attrs)
        assert out["weights"]["response"].shape == (B, 2)

    def test_weights_sum_to_one(self):
        B = 8
        attrs = _make_attrs(B, self.attr_dim)
        self.router.eval()
        out = self.router(attrs)
        for node in ["snow", "recharge", "aet", "response"]:
            w = out["weights"][node]
            sums = w.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                f"weights[{node}] sum != 1: {sums}"

    def test_default_bias_initialized(self):
        default_fids = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
        for node, fid in default_fids.items():
            fids = self.router.formula_ids[node]
            idx = fids.index(fid)
            bias = self.router.heads[node].bias.data
            assert bias.argmax().item() == idx, \
                f"{node}: expected bias max at '{fid}' (idx={idx}), got idx={bias.argmax().item()}"

    def test_default_bias_produces_default_selection(self):
        B = 8
        attrs = torch.zeros(B, self.attr_dim)
        self.router.eval()
        out = self.router(attrs)
        default_map = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
        for node, default_fid in default_map.items():
            fids = out["formula_ids"][node]
            sel_idx = out["selected"][node]
            for b in range(B):
                selected_fid = fids[int(sel_idx[b].item())]
                assert selected_fid == default_fid, \
                    f"{node}[{b}]: expected {default_fid}, got {selected_fid}"

    def test_recharge_hard_eval(self):
        B = 4
        attrs = _make_attrs(B, self.attr_dim)
        self.router.eval()
        out = self.router(attrs)
        w = out["weights"]["recharge"]
        for b in range(B):
            row = w[b]
            ones = (row > 0.99).sum().item()
            zeros = (row < 0.01).sum().item()
            assert ones == 1 and zeros == len(row) - 1, \
                f"recharge weights[{b}] not one-hot: {row}"

    def test_recharge_training_straight_through(self):
        B = 4
        attrs = _make_attrs(B, self.attr_dim)
        attrs.requires_grad = True
        self.router.train()
        out = self.router(attrs)
        w = out["weights"]["recharge"]
        loss = w.sum()
        loss.backward()
        assert attrs.grad is not None, "gradient should flow through recharge weights"

    def test_all_outputs_finite(self):
        B = 4
        attrs = _make_attrs(B, self.attr_dim)
        for mode in [True, False]:
            self.router.train(mode)
            out = self.router(attrs)
            for node in ["snow", "recharge", "aet", "response"]:
                w = out["weights"][node]
                assert torch.isfinite(w).all(), f"weights[{node}] not finite"
            for node in ["snow", "recharge", "aet", "response"]:
                l = out["logits"][node]
                assert torch.isfinite(l).all(), f"logits[{node}] not finite"
            for key in out:
                if key.startswith("entropy_") or key.startswith("max_weight_"):
                    v = out[key]
                    if isinstance(v, torch.Tensor):
                        assert torch.isfinite(v).all(), f"{key} not finite"

    def test_formula_ids_ordered(self):
        expected = {"snow": ["S0", "S4", "S5"], "recharge": ["R0", "R4", "R5"],
                    "aet": ["E0", "E3", "E4"], "response": ["Q0", "Q2"]}
        for node, fids in expected.items():
            assert self.router.formula_ids[node] == fids, \
                f"{node}: expected {fids}, got {self.router.formula_ids[node]}"

    def test_num_formulas(self):
        expected = {"snow": 3, "recharge": 3, "aet": 3, "response": 2}
        nf = self.router.num_formulas
        assert nf == expected, f"unexpected num_formulas: {nf}"


# ---------------------------------------------------------------------------
# HbvStaticFormulaRouter tests
# ---------------------------------------------------------------------------

class TestHbvStaticFormulaRouter:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.attr_dim = 8
        self.B = 2
        self.T = 80
        self.router_model = HbvStaticFormulaRouter(
            attr_dim=self.attr_dim, warm_up=20, hard_eval=True
        )
        self.forcing = _make_forcing(T=self.T, B=self.B)
        self.attrs = torch.randn(self.B, self.attr_dim)

    def test_forward_runs(self):
        self.router_model.eval()
        out = self.router_model(self.forcing, self.attrs)
        assert "Qsim" in out
        assert "Q_raw" in out
        assert "router" in out
        assert "diagnostics" in out

    def test_Qsim_finite(self):
        self.router_model.eval()
        out = self.router_model(self.forcing, self.attrs)
        assert torch.isfinite(out["Qsim"]).all(), "Qsim has NaN/Inf"

    def test_Q_raw_finite(self):
        self.router_model.eval()
        out = self.router_model(self.forcing, self.attrs)
        assert torch.isfinite(out["Q_raw"]).all(), "Q_raw has NaN/Inf"

    def test_router_output_included(self):
        self.router_model.eval()
        out = self.router_model(self.forcing, self.attrs)
        router = out["router"]
        for key in ["logits", "weights", "selected", "formula_ids"]:
            assert key in router, f"router missing key: {key}"

    def test_diagnostics_per_basin(self):
        self.router_model.eval()
        out = self.router_model(self.forcing, self.attrs)
        diags = out["diagnostics"]
        assert len(diags) == self.B
        for d in diags:
            assert isinstance(d, dict)
            assert "Q_raw" in d

    def test_water_balance(self):
        self.router_model.eval()
        out = self.router_model(self.forcing, self.attrs)
        wb = out["water_balance"]
        assert "residual" in wb
        assert "relative_error" in wb
        assert len(wb["residual"]) == self.B

    def test_training_forward_runs(self):
        self.router_model.train()
        attrs = self.attrs.detach().clone().requires_grad_(True)
        out = self.router_model(self.forcing, attrs)
        assert out["Qsim"] is not None
        assert torch.isfinite(out["Qsim"]).all()

    def test_router_gradient_flows_with_pg(self):
        self.router_model.train()
        attrs = self.attrs.detach().clone()
        out_dict = self.router_model(self.forcing, attrs)
        router_out = out_dict["router"]

        pg_loss = torch.tensor(0.0)
        for node in ["snow", "recharge", "aet", "response"]:
            logits = router_out["logits"][node]
            selected_idx = router_out["selected"][node]
            log_probs = F.log_softmax(logits, dim=-1)
            log_p = log_probs.gather(1, selected_idx.unsqueeze(-1)).squeeze(-1)
            pg_loss = pg_loss + log_p.mean()

        pg_loss.backward()
        for name, p in self.router_model.router.named_parameters():
            assert p.grad is not None, f"no gradient in {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"

    def test_does_not_change_hbv_static(self):
        from model.hbv_static import HbvStatic
        hbv = HbvStatic()
        assert hasattr(hbv, "forward")
        assert hasattr(hbv, "parameter_bounds")
        # Check that original HbvStatic.forward signature is intact
        import inspect
        sig = inspect.signature(HbvStatic.forward)
        params = list(sig.parameters)
        assert "x_dict" in params
        assert "parameters" in params

    def test_quick_smoke_script(self):
        from scripts.train_static_router_smoke import run_smoke
        import argparse
        out = _PROJECT / "validation_results" / "static_router_smoke" / "test_quick"
        args = argparse.Namespace(
            steps=2, num_basins=2, attr_dim=4, seq_len=60,
            lr=1e-3, output_dir=str(out), warmup=20,
            anchor_bias=0.5, temperature=2.0, grad_clip=1.0,
            hard_eval=False, active_nodes="recharge",
        )
        run_smoke(args)
        assert (out / "static_router_smoke_steps.csv").exists()
        assert (out / "static_router_smoke_report.md").exists()
