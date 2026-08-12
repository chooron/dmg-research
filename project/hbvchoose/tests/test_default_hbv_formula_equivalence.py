"""Test: Default HBV (S0_R0_E0_Q0) equivalence between HbvStatic._hbv_step and HbvFormulaStatic compat_mode.

Verifies that the formula-based HBV with compat_mode=True produces IDENTICAL results
to the reference _hbv_step kernel.  The formula-dispatch path (compat_mode=False) is also
tested but with relaxed tolerance since formula functions have additional safety clamping.
"""
import copy
import csv
from pathlib import Path

import pytest
import torch

from model.hbv_static import _hbv_step
from model.hbv_formula_static import HbvFormulaStatic
from model.parameter_mapping import ParameterMapper

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "validation_results" / "default_hbv_equivalence"
DTYPE = torch.float32
N_ZERO = 1e-5


def _build_fparams(norm_q=0.5):
    mapper = ParameterMapper(nmul=1)
    fc = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
    norm = torch.full((1, 14), norm_q, dtype=torch.float64)
    phy, _route = mapper.normalized_to_physical(norm)
    for k in phy:
        phy[k] = phy[k].to(dtype=DTYPE)
    fparams = mapper.physical_to_formula_params(fc, phy)
    fparams["response"]["K_0"] = phy["parK0"]
    fparams["response"]["K_1"] = phy["parK1"]
    fparams["response"]["K_2"] = phy["parK2"]
    return phy, fparams, fc


def _deepcopy_params(fparams):
    out = {}
    for k, v in fparams.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.clone() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def _synth_forcing(length=60):
    P = torch.tensor([0.0, 1.0, 5.0, 20.0, 0.0, 0.0, 10.0, 50.0, 2.0, 0.0] * (length // 10 + 1), dtype=DTYPE)[:length].unsqueeze(-1)
    T = torch.tensor([-5.0, -2.0, 0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 12.0, 8.0] * (length // 10 + 1), dtype=DTYPE)[:length].unsqueeze(-1)
    PET = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5] * (length // 10 + 1), dtype=DTYPE)[:length].unsqueeze(-1)
    return P, T, PET


def _run_eager_step_full(P, T, PET, phy, warm_up=0):
    """Run eager _hbv_step, return full traces (pre-truncation)."""
    nsteps = P.shape[0]
    tt = phy["parTT"]; cfmax = phy["parCFMAX"]; cfr = phy["parCFR"]; cwh = phy["parCWH"]
    fc_p = phy["parFC"]; beta_p = phy["parBETA"]; lp = phy["parLP"]; perc = phy["parPERC"]
    uzl = phy["parUZL"]; k0 = phy["parK0"]; k1 = phy["parK1"]; k2 = phy["parK2"]

    SP = torch.full((1, 1), 0.001, dtype=DTYPE)
    MW = torch.full((1, 1), 0.001, dtype=DTYPE)
    SM = torch.full((1, 1), 0.001, dtype=DTYPE)
    SUZ = torch.full((1, 1), 0.001, dtype=DTYPE)
    SLZ = torch.full((1, 1), 0.001, dtype=DTYPE)
    Q_raw = torch.zeros(nsteps, dtype=DTYPE)

    trace = {
        "SM": torch.zeros(nsteps, dtype=DTYPE),
        "SUZ": torch.zeros(nsteps, dtype=DTYPE),
        "SLZ": torch.zeros(nsteps, dtype=DTYPE),
        "SP": torch.zeros(nsteps, dtype=DTYPE),
        "MW": torch.zeros(nsteps, dtype=DTYPE),
    }

    for t in range(nsteps):
        q, SP, MW, SM, SUZ, SLZ = _hbv_step(
            P[t], T[t], PET[t], SP, MW, SM, SUZ, SLZ,
            tt, cfmax, cfr, cwh, fc_p, beta_p, lp, perc, uzl, k0, k1, k2, N_ZERO)
        Q_raw[t] = q.squeeze()
        for name, val in [("SM", SM), ("SUZ", SUZ), ("SLZ", SLZ), ("SP", SP), ("MW", MW)]:
            trace[name][t] = val.squeeze()

    # Return BOTH full and truncated
    if warm_up > 0:
        return Q_raw[warm_up:], {k: v[warm_up:] for k, v in trace.items()}
    return Q_raw, trace


def _run_formula_step(P, T, PET, fparams, fc, warm_up=20, compat_mode=True):
    """Run HbvFormulaStatic.simulate, return Q_raw (eval only) and full-state trace."""
    m = HbvFormulaStatic(formula_config=fc, warm_up=warm_up,
                         param_dicts=fparams, apply_routing=False, compat_mode=compat_mode)
    diag = m.simulate(P.squeeze(-1), T.squeeze(-1), PET.squeeze(-1))
    tr = diag.get("trace", {})
    # diag["Q_raw"] is already truncated to eval period
    # diag["trace"] is FULL length (includes warmup)
    return diag["Q_raw"], {
        "SM": tr.get("SM_after", torch.zeros(1)),
        "SUZ": tr.get("SUZ_after", torch.zeros(1)),
        "SLZ": tr.get("SLZ_after", torch.zeros(1)),
        "SP": tr.get("SP", torch.zeros(1)),
        "MW": tr.get("MW", torch.zeros(1)),
    }


class TestDefaultHbvFormulaEquivalence:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.phy, self.fparams, self.fc = _build_fparams(norm_q=0.5)
        self.P, self.T, self.PET = _synth_forcing(60)

    # ------------------------------------------------------------------
    # Compat-mode tests — EXACT equivalence expected
    # ------------------------------------------------------------------

    def test_compat_discharge_exact_match(self):
        Q_e, _ = _run_eager_step_full(self.P, self.T, self.PET, self.phy, warm_up=20)
        Q_f, _ = _run_formula_step(self.P, self.T, self.PET,
                                   _deepcopy_params(self.fparams), self.fc, warm_up=20, compat_mode=True)
        max_abs = (Q_e - Q_f).abs().max().item()
        assert max_abs < 1e-6, f"compat_mode Q diff = {max_abs:.2e}"

    def test_compat_per_step_all_variables(self):
        tt = self.phy["parTT"]; cfmax = self.phy["parCFMAX"]; cfr = self.phy["parCFR"]; cwh = self.phy["parCWH"]
        fc_p = self.phy["parFC"]; beta_p = self.phy["parBETA"]; lp = self.phy["parLP"]; perc = self.phy["parPERC"]
        uzl = self.phy["parUZL"]; k0 = self.phy["parK0"]; k1 = self.phy["parK1"]; k2 = self.phy["parK2"]

        SP_o = torch.full((1, 1), 0.001, dtype=DTYPE)
        MW_o = torch.full((1, 1), 0.001, dtype=DTYPE)
        SM_o = torch.full((1, 1), 0.001, dtype=DTYPE)
        SUZ_o = torch.full((1, 1), 0.001, dtype=DTYPE)
        SLZ_o = torch.full((1, 1), 0.001, dtype=DTYPE)

        SP_f = SP_o.clone(); MW_f = MW_o.clone(); SM_f = SM_o.clone()
        SUZ_f = SUZ_o.clone(); SLZ_f = SLZ_o.clone()

        fm = HbvFormulaStatic(formula_config=self.fc, warm_up=0,
                              param_dicts=_deepcopy_params(self.fparams), compat_mode=True)

        nsteps = min(self.P.shape[0], 60)
        divergences = []
        rows = []

        for t in range(nsteps):
            acc = {k: torch.tensor(0.0) for k in ["rainfall_total", "snowfall_total", "melt_total", "refreezing_total", "recharge_total", "aet_total"]}
            doy_t = torch.as_tensor(float(t + 1))

            Q_o, SP_o2, MW_o2, SM_o2, SUZ_o2, SLZ_o2 = _hbv_step(
                self.P[t], self.T[t], self.PET[t], SP_o, MW_o, SM_o, SUZ_o, SLZ_o,
                tt, cfmax, cfr, cwh, fc_p, beta_p, lp, perc, uzl, k0, k1, k2, N_ZERO)

            Q_f, SP_f2, MW_f2, SM_f2, SUZ_f2, SLZ_f2, flux_f = fm._step(
                self.P[t], self.T[t], self.PET[t], SP_f, MW_f, SM_f, SUZ_f, SLZ_f,
                tt, cfmax, cfr, cwh, fc_p, beta_p, lp, perc, uzl, k0, k1, k2, N_ZERO, doy_t, acc)

            row = {"t": t, "P": self.P[t].item(), "T": self.T[t].item(), "PET": self.PET[t].item()}
            for name, vo, vf in [
                ("Q", Q_o, Q_f), ("SM", SM_o2, SM_f2), ("SUZ", SUZ_o2, SUZ_f2),
                ("SLZ", SLZ_o2, SLZ_f2), ("SP", SP_o2, SP_f2), ("MW", MW_o2, MW_f2),
            ]:
                d = (vo - vf).abs().max().item()
                row[f"{name}_diff"] = d
                if d > 1e-7:
                    divergences.append(f"t={t} {name}: orig={vo.item():.6f} form={vf.item():.6f} diff={d:.2e}")

            rows.append(row)
            SP_o, MW_o, SM_o, SUZ_o, SLZ_o = SP_o2, MW_o2, SM_o2, SUZ_o2, SLZ_o2
            SP_f, MW_f, SM_f, SUZ_f, SLZ_f = SP_f2, MW_f2, SM_f2, SUZ_f2, SLZ_f2

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = OUTPUT_DIR / "per_step_trace.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        assert len(divergences) == 0, f"Divergences in compat_mode:\n" + "\n".join(divergences[:10])

    def test_compat_state_traces_equivalent(self):
        _, et = _run_eager_step_full(self.P, self.T, self.PET, self.phy, warm_up=20)
        _, ft = _run_formula_step(self.P, self.T, self.PET,
                                  _deepcopy_params(self.fparams), self.fc, warm_up=20, compat_mode=True)
        for var in ["SM", "SUZ", "SLZ", "SP", "MW"]:
            if var in et and var in ft and len(et[var]) > 0 and len(ft[var]) > 0:
                # simulates trace is full-length (incl warmup), eager is truncated
                # Compare eval period only
                ev = et[var]  # already warm_up:[]
                fv = ft[var][20:]  # truncate to eval only
                n = min(len(ev), len(fv))
                if n == 0:
                    continue
                diff = (ev[:n] - fv[:n]).abs().max().item()
                assert diff < 1e-6, f"{var}: max diff (eval) = {diff:.2e}"

    def test_compat_various_forcing(self):
        scenarios = {
            "all_warm_wet": (
                torch.full((40, 1), 20.0, dtype=DTYPE),
                torch.full((40, 1), 15.0, dtype=DTYPE),
                torch.full((40, 1), 4.0, dtype=DTYPE),
            ),
            "all_cold_snow": (
                torch.full((40, 1), 10.0, dtype=DTYPE),
                torch.full((40, 1), -5.0, dtype=DTYPE),
                torch.full((40, 1), 0.5, dtype=DTYPE),
            ),
            "all_dry_hot": (
                torch.full((40, 1), 0.0, dtype=DTYPE),
                torch.full((40, 1), 30.0, dtype=DTYPE),
                torch.full((40, 1), 8.0, dtype=DTYPE),
            ),
            "extreme_rain": (
                torch.full((40, 1), 100.0, dtype=DTYPE),
                torch.full((40, 1), 5.0, dtype=DTYPE),
                torch.full((40, 1), 2.0, dtype=DTYPE),
            ),
        }
        for name, (P, T, PET) in scenarios.items():
            Q_e, _ = _run_eager_step_full(P, T, PET, self.phy, warm_up=0)
            Q_f, _ = _run_formula_step(P, T, PET,
                                       _deepcopy_params(self.fparams), self.fc, warm_up=0, compat_mode=True)
            n = min(len(Q_e), len(Q_f))
            max_abs = (Q_e[:n] - Q_f[:n]).abs().max().item()
            assert max_abs < 1e-5, f"Scenario '{name}': max Q diff = {max_abs:.2e}"

    def test_compat_different_params(self):
        for q_val in [0.1, 0.3, 0.7, 0.9]:
            phy, fparams, fc = _build_fparams(norm_q=q_val)
            P, T, PET = _synth_forcing(40)
            Q_e, _ = _run_eager_step_full(P, T, PET, phy, warm_up=0)
            Q_f, _ = _run_formula_step(P, T, PET,
                                       _deepcopy_params(fparams), fc, warm_up=0, compat_mode=True)
            n = min(len(Q_e), len(Q_f))
            max_abs = (Q_e[:n] - Q_f[:n]).abs().max().item()
            assert max_abs < 1e-6, f"q={q_val}: max Q diff = {max_abs:.2e}"

    def test_compat_routing_disabled_equivalent_to_raw(self):
        m = HbvFormulaStatic(formula_config=self.fc, warm_up=20,
                             param_dicts=_deepcopy_params(self.fparams),
                             apply_routing=False, compat_mode=True)
        diag = m.simulate(self.P.squeeze(-1), self.T.squeeze(-1), self.PET.squeeze(-1))
        assert not diag.get("routing_applied", True), "Routing should NOT be applied"
        Qsim = diag["Qsim"]
        Qraw = diag["Q_raw"]
        assert (Qsim - Qraw).abs().max().item() < 1e-8, "Qsim != Q_raw with routing disabled"

    # ------------------------------------------------------------------
    # Dispatch-mode tests — formula functions have extra clamping, accept epsilon-level diffs
    # ------------------------------------------------------------------

    def test_dispatch_approx_equivalent(self):
        """Dispatch mode (compat_mode=False) should be approximately equivalent (<0.1 diff)."""
        Q_e, _ = _run_eager_step_full(self.P, self.T, self.PET, self.phy, warm_up=20)
        Q_f, _ = _run_formula_step(self.P, self.T, self.PET,
                                   _deepcopy_params(self.fparams), self.fc, warm_up=20, compat_mode=False)
        max_abs = (Q_e - Q_f).abs().max().item()
        assert max_abs < 0.1, f"Dispatch Q diff too large: {max_abs:.4f}"

    def test_dispatch_respects_same_default_combo(self):
        """Dispatch mode with default formula config still produces non-NaN output."""
        P, T, PET = _synth_forcing(40)
        m = HbvFormulaStatic(formula_config=self.fc, warm_up=0,
                             param_dicts=_deepcopy_params(self.fparams),
                             apply_routing=False, compat_mode=False)
        diag = m.simulate(P.squeeze(-1), T.squeeze(-1), PET.squeeze(-1))
        assert not torch.isnan(diag["Q_raw"]).any(), "NaN in dispatch-mode discharge"
        assert not torch.isinf(diag["Q_raw"]).any(), "Inf in dispatch-mode discharge"

    # ------------------------------------------------------------------
    # Comprehensive per-variable trace test
    # ------------------------------------------------------------------

    def test_full_per_variable_trace(self):
        """Generate detailed per-step, per-variable comparison CSV."""
        Q_e, et = _run_eager_step_full(self.P, self.T, self.PET, self.phy, warm_up=20)
        Q_f, ft = _run_formula_step(self.P, self.T, self.PET,
                                    _deepcopy_params(self.fparams), self.fc, warm_up=20, compat_mode=True)

        n = min(len(Q_e), len(Q_f))
        rows = []
        for t in range(n):
            row = {"t": t, "phase": "evaluation"}
            row["Q_eager"] = Q_e[t].item()
            row["Q_formula"] = Q_f[t].item()
            row["Q_diff"] = (Q_e[t] - Q_f[t]).item()

            for var in ["SM", "SUZ", "SLZ", "SP", "MW"]:
                ev = et[var]  # truncated to warm_up:
                fv = ft[var][20:] if len(ft[var]) > 20 else ft[var]
                if t < len(ev) and t < len(fv):
                    row[f"{var}_eager"] = ev[t].item()
                    row[f"{var}_formula"] = fv[t].item()
                    row[f"{var}_diff"] = (ev[t] - fv[t]).item()
            rows.append(row)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = OUTPUT_DIR / "equivalence_trace.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        max_q_diff = max(abs(r["Q_diff"]) for r in rows)
        assert max_q_diff < 1e-6, f"Max Q diff in trace = {max_q_diff:.2e}"
        for var in ["SM", "SUZ", "SLZ", "SP", "MW"]:
            diffs = [abs(r[f"{var}_diff"]) for r in rows if f"{var}_diff" in r]
            if diffs:
                max_d = max(diffs)
                assert max_d < 1e-6, f"Max {var} diff = {max_d:.2e}"
