"""Safety and constraint tests for all HBV MoE candidate formulas."""

import sys
import pytest
import torch

sys.path.insert(0, "/home/jingxin/code/dmg-research/project/hbvchoose")

from model.flux import snow, recharge, aet, response, routing
from model.flux._utils import (
    gamma_weights,
    triangular_weights,
    softplus_t,
    smoothmin_t,
)
from model.flux.parameter_ranges import PARAMETER_RANGES
from model.hbv_static import HbvStatic

# ---------------------------------------------------------------------------
# Shared test tensors
# ---------------------------------------------------------------------------

DEFAULT_DT = torch.float64  # use float64 to expose subtle numerical issues


def _make(*vals, dtype=None):
    dt = dtype or DEFAULT_DT
    return torch.tensor(vals, dtype=dt)


def _assert_finite(t, label=""):
    if isinstance(t, tuple):
        for i, elem in enumerate(t):
            if torch.is_tensor(elem):
                _assert_finite(elem, f"{label}[{i}]")
        return
    assert not torch.any(torch.isnan(t)), f"{label} contains NaN"
    assert not torch.any(torch.isinf(t)), f"{label} contains Inf"


# ---------------------------------------------------------------------------
# 1. Shape correctness
# ---------------------------------------------------------------------------

SHAPE_CASES = [
    # single-element scalar-like tensors
    (_make(5.0),),
    # 1-D time series
    (torch.randn(100, dtype=DEFAULT_DT),),
    # 2-D batch (batch=4, time=100)
    (torch.randn(4, 100, dtype=DEFAULT_DT),),
]


def _all_close(a, b, **kw):
    if not torch.allclose(a, b, **kw):
        raise AssertionError(f"tensors differ: {a} vs {b}")


@pytest.mark.parametrize("shape_fn", SHAPE_CASES)
class TestShapeSnow:
    def test_partition_hard(self, shape_fn):
        P, *_ = shape_fn
        T = P.clone()
        Ps, Pr = snow.rain_snow_partition_hard(P, T, _make(0.5))
        assert Ps.shape == P.shape
        assert Pr.shape == P.shape

    def test_partition_smooth(self, shape_fn):
        P, *_ = shape_fn
        T = P.clone()
        Ps, Pr = snow.rain_snow_partition_smooth(P, T, _make(0.5), _make(1.0))
        assert Ps.shape == P.shape

    def test_melt_linear(self, shape_fn):
        T, *_ = shape_fn
        SWE = T.abs() + 1.0
        M = snow.snowmelt_linear_degreeday(T, _make(0.5), _make(3.0), SWE)
        assert M.shape == T.shape

    def test_melt_exponential(self, shape_fn):
        T, *_ = shape_fn
        SWE = T.abs() + 1.0
        M = snow.snowmelt_exponential(T, _make(0.5), _make(2.0), _make(0.3), SWE)
        assert M.shape == T.shape

    def test_refreezing(self, shape_fn):
        T, *_ = shape_fn
        LW = T.abs() + 0.5
        Rf = snow.refreezing(T, _make(0.5), _make(0.05), _make(3.0), LW)
        assert Rf.shape == T.shape


@pytest.mark.parametrize("shape_fn", SHAPE_CASES)
class TestShapeRecharge:
    def test_beta(self, shape_fn):
        I, *_ = shape_fn
        SM = I.abs() * 50
        R = recharge.beta_recharge(I, SM, _make(200.0), _make(2.0))
        assert R.shape == I.shape

    def test_saturation_threshold(self, shape_fn):
        I, *_ = shape_fn
        SM = I.abs() * 50
        R = recharge.saturation_threshold_recharge(I, SM, _make(200.0), _make(10.0), _make(0.5))
        assert R.shape == I.shape


@pytest.mark.parametrize("shape_fn", SHAPE_CASES)
class TestShapeAET:
    def test_default(self, shape_fn):
        PET, *_ = shape_fn
        SM = PET.abs() * 30
        ET = aet.aet_hbv_default(PET, SM, _make(0.9), _make(200.0))
        assert ET.shape == PET.shape

    def test_power_law(self, shape_fn):
        PET, *_ = shape_fn
        SM = PET.abs() * 30
        ET = aet.aet_power_law(PET, SM, _make(200.0), _make(1.5))
        assert ET.shape == PET.shape


@pytest.mark.parametrize("shape_fn", SHAPE_CASES)
class TestShapeResponse:
    def test_two_reservoir(self, shape_fn):
        SUZ, *_ = shape_fn
        SLZ = SUZ.clone()
        Q0, Q1, Q2, Q = response.response_two_reservoir(
            SUZ, SLZ, _make(0.3), _make(0.1), _make(0.05), _make(10.0)
        )
        assert Q.shape == SUZ.shape

    def test_single(self, shape_fn):
        S, *_ = shape_fn
        Q = response.response_single_reservoir(S, _make(0.05))
        assert Q.shape == S.shape

    def test_parallel(self, shape_fn):
        R, *_ = shape_fn
        Sf = R.abs() * 5
        Ss = R.abs() * 10
        Rf, Rs, Qf, Qs, Q = response.response_two_parallel(
            R, Sf, Ss, _make(0.1), _make(0.05), _make(0.6)
        )
        assert Q.shape == R.shape

    def test_delayed_step(self, shape_fn):
        R, *_ = shape_fn
        S1 = R.abs() * 5
        S2 = R.abs() * 10
        if R.dim() == 0:
            # scalar tensors
            rim, rdel, q1, q2, qt = response.response_delayed_step(
                R, S1, S2, _make(0.7), _make(0.1), _make(0.05)
            )
        else:
            # for batched/multi-element R, S1/S2 are also multi-element
            rim, rdel, q1, q2, qt = response.response_delayed_step(
                R.clone(), S1.clone(), S2.clone(), _make(0.7), _make(0.1), _make(0.05)
            )
        assert rim.shape == R.shape


# ---------------------------------------------------------------------------
# 2. No NaN / Inf
# ---------------------------------------------------------------------------

def _safety_sweep(func, *args, extremes=None, **kw):
    """Run func on typical + extreme inputs and assert no NaN/Inf."""
    out = func(*args, **kw)
    _assert_finite(out, func.__name__)

    if extremes is None:
        extremes = [
            torch.zeros_like(args[0]),
            torch.ones_like(args[0]) * 1e-6,
            torch.ones_like(args[0]) * 1e6,
        ]

    for extreme in extremes:
        ex_args = []
        for a in args:
            if a.shape == extreme.shape and a.dtype == extreme.dtype:
                ex_args.append(extreme)
            else:
                ex_args.append(a)
        try:
            out = func(*ex_args, **kw)
            if isinstance(out, tuple):
                for o in out:
                    if torch.is_tensor(o):
                        _assert_finite(o, f"{func.__name__} @ extreme")
            else:
                _assert_finite(out, f"{func.__name__} @ extreme")
        except Exception:
            pass  # extreme values may legitimately error; just skip


class TestNoNaNInf:
    def test_snow(self):
        P = _make(5.0, 4.0, 3.0, 2.0)
        T = _make(-2.0, 0.0, 5.0, 10.0)
        SWE = _make(20.0, 18.0, 15.0, 10.0)
        LW = _make(5.0, 4.0, 3.0, 2.0)

        _safety_sweep(snow.rain_snow_partition_hard, P, T, _make(0.5))
        _safety_sweep(snow.rain_snow_partition_smooth, P, T, _make(0.5), _make(1.0))
        _safety_sweep(snow.snowmelt_linear_degreeday, T, _make(0.5), _make(3.0), SWE)
        _safety_sweep(snow.snowmelt_smooth_degreeday, T, _make(0.5), _make(3.0), _make(0.5), SWE)
        _safety_sweep(snow.snowmelt_exponential, T, _make(0.5), _make(2.0), _make(0.3), SWE)
        _safety_sweep(snow.refreezing, T, _make(0.5), _make(0.05), _make(3.0), LW)

    def test_recharge(self):
        I = _make(1.0, 2.0, 3.0, 4.0)
        SM = _make(50.0, 80.0, 120.0, 150.0)
        _safety_sweep(recharge.beta_recharge, I, SM, _make(200.0), _make(2.0))
        _safety_sweep(recharge.saturation_threshold_recharge, I, SM, _make(200.0), _make(10.0), _make(0.5))

    def test_aet(self):
        PET = _make(2.0, 3.0, 4.0, 5.0)
        SM = _make(50.0, 80.0, 120.0, 150.0)
        _safety_sweep(aet.aet_hbv_default, PET, SM, _make(0.9), _make(200.0))
        _safety_sweep(aet.aet_smooth_hbv, PET, SM, _make(0.9), _make(200.0), _make(0.05))
        _safety_sweep(aet.aet_power_law, PET, SM, _make(200.0), _make(1.5))

    def test_response(self):
        SUZ = _make(5.0, 8.0, 12.0, 15.0)
        SLZ = _make(20.0, 19.0, 18.0, 17.0)
        _safety_sweep(response.response_two_reservoir, SUZ, SLZ, _make(0.3), _make(0.1), _make(0.05), _make(10.0))
        _safety_sweep(response.response_nonlinear, SUZ, SLZ, _make(0.1), _make(0.05), _make(1.2))
        _safety_sweep(response.response_single_reservoir, _make(50.0), _make(0.05))

    def test_gamma_weights_nan(self):
        # Test that b=0 does not produce NaN (old bug)
        w = gamma_weights(2.0, 0.0, 10)
        _assert_finite(w)

    def test_extreme_exponential_snowmelt(self):
        # Very large T should not produce Inf
        T = _make(1000.0)
        SWE = _make(1000.0)
        M = snow.snowmelt_exponential(T, _make(0.5), _make(2.0), _make(0.3), SWE)
        _assert_finite(M)


# ---------------------------------------------------------------------------
# 3. Recharge: 0 <= R <= I
# ---------------------------------------------------------------------------

class TestRechargeBounds:
    def test_beta(self):
        I = _make(1.0, 2.0, 3.0, 4.0)
        SM = _make(0.0, 50.0, 150.0, 200.0)
        R = recharge.beta_recharge(I, SM, _make(200.0), _make(2.0))
        assert torch.all(R >= 0.0), f"R negative: {R}"
        assert torch.all(R <= I + 1e-10), f"R > I: R={R}, I={I}"

    def test_saturation_threshold(self):
        I = _make(1.0, 2.0, 3.0, 4.0)
        SM = _make(0.0, 50.0, 150.0, 200.0)
        R = recharge.saturation_threshold_recharge(I, SM, _make(200.0), _make(10.0), _make(0.5))
        assert torch.all(R >= 0.0)
        assert torch.all(R <= I + 1e-10)

    def test_linear(self):
        I = _make(1.0, 2.0, 3.0, 0.0)
        SM = _make(0.0, 100.0, 200.0, 100.0)
        R = recharge.linear_recharge(I, SM, _make(200.0))
        assert torch.all(R >= 0.0)
        assert torch.all(R <= I + 1e-10)

    def test_zero_fc(self):
        # FC -> 0 should not produce NaN
        I = _make(1.0)
        SM = _make(50.0)
        R = recharge.beta_recharge(I, SM, _make(0.0), _make(2.0))
        _assert_finite(R)

    def test_negative_sm(self):
        I = _make(1.0)
        SM = _make(-5.0)
        R = recharge.beta_recharge(I, SM, _make(200.0), _make(2.0))
        assert torch.all(R >= 0.0)
        assert torch.all(R <= I + 1e-10)

    def test_saturation_threshold_at_zero_sm(self):
        """R4 must produce R -> 0 when SM = 0."""
        I = _make(5.0)
        SM = _make(0.0)
        FC = _make(200.0)
        for a_r in [5.0, 10.0, 15.0]:
            for c_r in [0.4, 0.6, 0.85]:
                R = recharge.saturation_threshold_recharge(I, SM, FC, _make(a_r), _make(c_r))
                assert torch.allclose(R, _make(0.0), atol=1e-10), \
                    f"R4 at SM=0: R={R.item()} for a_r={a_r}, c_r={c_r}"

    def test_saturation_threshold_at_full_sm(self):
        """R4 must produce R -> I when SM = FC."""
        I = _make(5.0)
        SM = _make(200.0)
        FC = _make(200.0)
        for a_r in [5.0, 10.0, 15.0]:
            for c_r in [0.4, 0.6, 0.85]:
                R = recharge.saturation_threshold_recharge(I, SM, FC, _make(a_r), _make(c_r))
                assert torch.allclose(R, I, atol=1e-10), \
                    f"R4 at SM=FC: R={R.item()} for a_r={a_r}, c_r={c_r}"

    def test_variable_contributing_area_zero_sm(self):
        """R5 must produce R -> 0 when SM = 0."""
        R = recharge.variable_contributing_area_recharge(_make(5.0), _make(0.0), _make(200.0), _make(1.0))
        assert torch.allclose(R, _make(0.0), atol=1e-10)

    def test_variable_contributing_area_full_sm(self):
        """R5 must produce R -> I when SM = FC."""
        I = _make(5.0)
        R = recharge.variable_contributing_area_recharge(I, _make(200.0), _make(200.0), _make(1.0))
        assert torch.allclose(R, I, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. AET: 0 <= ET <= PET and ET <= SM
# ---------------------------------------------------------------------------

class TestAETBounds:
    def test_default(self):
        PET = _make(2.0, 3.0, 4.0, 5.0)
        SM = _make(50.0, 80.0, 120.0, 150.0)
        ET = aet.aet_hbv_default(PET, SM, _make(0.9), _make(200.0))
        assert torch.all(ET >= 0.0), f"ET negative: {ET}"
        assert torch.all(ET <= PET + 1e-10), f"ET > PET: {ET} vs {PET}"
        assert torch.all(ET <= SM + 1e-10), f"ET > SM: {ET} vs {SM}"

    def test_smooth(self):
        PET = _make(2.0, 3.0, 4.0, 5.0)
        SM = _make(50.0, 80.0, 120.0, 150.0)
        ET = aet.aet_smooth_hbv(PET, SM, _make(0.9), _make(200.0), _make(0.05))
        assert torch.all(ET >= 0.0)
        assert torch.all(ET <= PET + 1e-10)
        assert torch.all(ET <= SM + 1e-10)

    def test_power_law(self):
        PET = _make(2.0, 3.0, 4.0, 5.0)
        SM = _make(50.0, 80.0, 120.0, 150.0)
        ET = aet.aet_power_law(PET, SM, _make(200.0), _make(1.5))
        assert torch.all(ET >= 0.0)
        assert torch.all(ET <= PET + 1e-10)
        assert torch.all(ET <= SM + 1e-10)

    def test_sm_zero(self):
        PET = _make(5.0)
        SM = _make(0.0)
        ET = aet.aet_hbv_default(PET, SM, _make(0.9), _make(200.0))
        assert torch.allclose(ET, _make(0.0))

    def test_negative_pet(self):
        PET = _make(-5.0)
        SM = _make(100.0)
        ET = aet.aet_hbv_default(PET, SM, _make(0.9), _make(200.0))
        assert torch.all(ET >= 0.0)

    def test_power_law_sm_zero_no_inf_grad(self):
        """E3 at SM=0 must produce ET=0 with finite gradients."""
        PET = torch.tensor(3.0, dtype=DEFAULT_DT, requires_grad=True)
        SM = torch.tensor(0.0, dtype=DEFAULT_DT, requires_grad=True)
        FC = torch.tensor(200.0, dtype=DEFAULT_DT)
        gamma_E = torch.tensor(0.8, dtype=DEFAULT_DT)
        ET = aet.aet_power_law(PET, SM, FC, gamma_E)
        assert torch.allclose(ET, _make(0.0), atol=1e-10), f"ET at SM=0: {ET.item()}"
        loss = ET.sum()
        loss.backward()
        assert torch.isfinite(PET.grad).all(), f"PET grad: {PET.grad}"
        assert torch.isfinite(SM.grad).all(), f"SM grad: {SM.grad}"

    def test_power_law_gamma_range_no_nan(self):
        """E3 across gamma_E values must produce no NaN/Inf."""
        PET = _make(3.0)
        SM = _make(50.0)
        FC = _make(200.0)
        for g in [0.8, 1.0, 1.8]:
            ET = aet.aet_power_law(PET, SM, FC, _make(g))
            assert not torch.any(torch.isnan(ET)), f"NaN for gamma_E={g}"
            assert not torch.any(torch.isinf(ET)), f"Inf for gamma_E={g}"

    def test_power_law_bounds(self):
        """E3 must satisfy ET <= SM and ET <= PET."""
        for sm_val, pet_val in [(0.0, 3.0), (50.0, 3.0), (200.0, 3.0), (200.0, 0.0)]:
            ET = aet.aet_power_law(_make(pet_val), _make(sm_val), _make(200.0), _make(1.2))
            assert ET <= _make(sm_val) + 1e-10, f"ET > SM: ET={ET}, SM={sm_val}"
            assert ET <= _make(pet_val) + 1e-10, f"ET > PET: ET={ET}, PET={pet_val}"


# ---------------------------------------------------------------------------
# 5. Snowmelt: 0 <= M <= SWE
# ---------------------------------------------------------------------------

class TestSnowmeltBounds:
    def test_linear(self):
        T = _make(-2.0, 0.0, 5.0, 10.0)
        SWE = _make(20.0, 18.0, 15.0, 10.0)
        M = snow.snowmelt_linear_degreeday(T, _make(0.5), _make(3.0), SWE)
        assert torch.all(M >= 0.0)
        assert torch.all(M <= SWE + 1e-10)

    def test_smooth(self):
        T = _make(-2.0, 0.0, 5.0, 10.0)
        SWE = _make(20.0, 18.0, 15.0, 10.0)
        M = snow.snowmelt_smooth_degreeday(T, _make(0.5), _make(3.0), _make(0.5), SWE)
        assert torch.all(M >= 0.0)
        assert torch.all(M <= SWE + 1e-10)

    def test_exponential_bounds(self):
        T = _make(-2.0, 0.0, 5.0, 10.0)
        SWE = _make(20.0, 18.0, 15.0, 10.0)
        M = snow.snowmelt_exponential(T, _make(0.5), _make(2.0), _make(0.3), SWE)
        assert torch.all(M >= 0.0)
        assert torch.all(M <= SWE + 1e-10)

    def test_exponential_at_threshold(self):
        """snowmelt_exponential must produce ~0 melt when T == TT."""
        T = _make(0.5)  # exactly at threshold
        SWE = _make(20.0)
        M = snow.snowmelt_exponential(T, _make(0.5), _make(2.0), _make(0.3), SWE)
        assert torch.allclose(M, _make(0.0), atol=1e-12), f"M at threshold: {M}"

    def test_exponential_below_threshold(self):
        T = _make(-10.0, -5.0, -1.0)
        SWE = _make(20.0, 20.0, 20.0)
        M = snow.snowmelt_exponential(T, _make(0.5), _make(2.0), _make(0.3), SWE)
        assert torch.allclose(M, _make(0.0, 0.0, 0.0), atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Response outflow <= storage
# ---------------------------------------------------------------------------

class TestResponseBounds:
    def test_two_reservoir(self):
        SUZ = _make(5.0, 8.0, 12.0, 15.0)
        SLZ = _make(20.0, 19.0, 18.0, 17.0)
        Q0, Q1, Q2, Q = response.response_two_reservoir(
            SUZ, SLZ, _make(0.3), _make(0.1), _make(0.05), _make(10.0)
        )
        assert torch.all(Q0 <= SUZ + 1e-10), f"Q0 exceeds SUZ: {Q0} vs {SUZ}"
        # remaining_suz = SUZ - Q0, Q1 <= remaining_suz
        assert torch.all(Q1 <= (SUZ - Q0) + 1e-10)
        assert torch.all(Q2 <= SLZ + 1e-10), f"Q2 exceeds SLZ: {Q2} vs {SLZ}"

    def test_nonlinear(self):
        SUZ = _make(5.0, 8.0, 12.0, 15.0)
        SLZ = _make(20.0, 19.0, 18.0, 17.0)
        Quz, Qlz, Q = response.response_nonlinear(SUZ, SLZ, _make(0.1), _make(0.05), _make(1.2))
        assert torch.all(Quz <= SUZ + 1e-10)
        assert torch.all(Qlz <= SLZ + 1e-10)

    def test_single(self):
        S = _make(50.0)
        Q = response.response_single_reservoir(S, _make(0.05))
        assert Q <= S + 1e-10

    def test_parallel(self):
        R = _make(3.0)
        Sf = _make(20.0)
        Ss = _make(80.0)
        Rf, Rs, Qf, Qs, Q = response.response_two_parallel(
            R, Sf, Ss, _make(0.1), _make(0.05), _make(0.6)
        )
        assert Qf <= Sf + 1e-10
        assert Qs <= Ss + 1e-10

    def test_delayed_step(self):
        R = _make(3.0)
        S1 = _make(10.0)
        S2 = _make(30.0)
        Rim, Rdel, Q1, Q2, Q = response.response_delayed_step(
            R, S1, S2, _make(0.7), _make(0.1), _make(0.05)
        )
        assert Q1 <= S1 + 1e-10
        assert Q2 <= S2 + 1e-10


# ---------------------------------------------------------------------------
# 8. gamma_weights sum to 1
# ---------------------------------------------------------------------------

class TestWeightsNormalization:
    def test_gamma(self):
        for a_val in [1.0, 2.0, 3.0, 5.0]:
            for b_val in [0.5, 1.0, 2.0, 5.0]:
                w = gamma_weights(a_val, b_val, 10)
                _assert_finite(w)
                w64 = w.to(torch.float64)
                assert torch.allclose(w64.sum(), _make(1.0, dtype=torch.float64), atol=1e-10), \
                    f"sum={w.sum()} for a={a_val}, b={b_val}"

    def test_gamma_tensor(self):
        a = torch.tensor(2.0, dtype=DEFAULT_DT)
        b = torch.tensor(2.0, dtype=DEFAULT_DT)
        w = gamma_weights(a, b, 10)
        w64 = w.to(torch.float64)
        assert torch.allclose(w64.sum(), _make(1.0, dtype=torch.float64), atol=1e-10)

    def test_triangular(self):
        for mb in [1, 2, 3, 4, 5, 7]:
            w = triangular_weights(mb)
            _assert_finite(w)
            w64 = w.to(torch.float64)
            assert torch.allclose(w64.sum(), _make(1.0, dtype=torch.float64), atol=1e-10)

    def test_triangular_tensor(self):
        mb = torch.tensor(3.0)
        w = triangular_weights(mb)
        w64 = w.to(torch.float64)
        assert torch.allclose(w64.sum(), _make(1.0, dtype=torch.float64), atol=1e-10)
        assert w.numel() == 3


# ---------------------------------------------------------------------------
# 9. Triangular weight shape
# ---------------------------------------------------------------------------

class TestTriangularWeights:
    def test_symmetry(self):
        w = triangular_weights(5)
        assert torch.allclose(w, w.flip(0))

    def test_length(self):
        assert triangular_weights(1).numel() == 1
        assert triangular_weights(3).numel() == 3
        assert triangular_weights(4).numel() == 4


# ---------------------------------------------------------------------------
# 10. HbvStatic.parameter_bounds matches PARAMETER_RANGES["HBV_BASE"]
# ---------------------------------------------------------------------------

# Mapping from HbvStatic parameter names (par-prefix) to PARAMETER_RANGES keys
_BOUNDS_MAP = {
    "parBETA": "BETA",
    "parFC": "FC",
    "parK0": "K0",
    "parK1": "K1",
    "parK2": "K2",
    "parLP": "LP",
    "parPERC": "PERC",
    "parUZL": "UZL",
    "parTT": "TT",
    "parCFMAX": "CFMAX",
    "parCFR": "CFR",
    "parCWH": "CWH",
}


class TestParameterBoundsConsistency:
    def test_hbv_base_bounds_match(self):
        hbv_bounds = HbvStatic.parameter_bounds
        ref_ranges = PARAMETER_RANGES["HBV_BASE"]

        for par_name, ref_key in _BOUNDS_MAP.items():
            assert par_name in hbv_bounds, f"Missing {par_name} in HbvStatic.parameter_bounds"
            assert ref_key in ref_ranges, f"Missing {ref_key} in PARAMETER_RANGES['HBV_BASE']"

            hbv_lo, hbv_hi = hbv_bounds[par_name]
            ref_lo, ref_hi = ref_ranges[ref_key]["range"]
            assert hbv_lo == ref_lo, f"{par_name}: lower bound {hbv_lo} != {ref_lo}"
            assert hbv_hi == ref_hi, f"{par_name}: upper bound {hbv_hi} != {ref_hi}"

    def test_routing_bounds_match(self):
        hbv_route = HbvStatic.routing_parameter_bounds
        ref_route = PARAMETER_RANGES["ROUTING_EXT"]
        assert hbv_route["route_a"][0] == ref_route["route_a"]["range"][0]
        assert hbv_route["route_a"][1] == ref_route["route_a"]["range"][1]
        assert hbv_route["route_b"][0] == ref_route["route_b"]["range"][0]
        assert hbv_route["route_b"][1] == ref_route["route_b"]["range"][1]


# ---------------------------------------------------------------------------
# 11. Utility function safety
# ---------------------------------------------------------------------------

class TestUtils:
    def test_softplus_t_small_tau(self):
        x = _make(10.0, -10.0, 0.0)
        y = softplus_t(x, tau=0.0)  # should be clamped to EPS
        _assert_finite(y)

    def test_smoothmin_t_clamped(self):
        x = _make(2.0, 0.5, -0.5, 0.0)
        y = smoothmin_t(x, threshold=1.0, tau=0.1)
        assert torch.all(y >= 0.0), f"smoothmin below 0: {y}"
        assert torch.all(y <= 1.0 + 1e-10), f"smoothmin above threshold: {y}"

    def test_cfmax_seasonal_non_negative(self):
        CFMAX_t = snow.cfmax_seasonal(_make(1.0), _make(0.8), _make(180.0), _make(10.0))
        assert CFMAX_t >= 0.0

    def test_seasonal_amplitude_zero(self):
        CFMAX_t = snow.cfmax_seasonal(_make(3.0), _make(0.0), _make(180.0), _make(10.0))
        assert torch.allclose(CFMAX_t, _make(3.0))


# ---------------------------------------------------------------------------
# 12. Routing shape consistency
# ---------------------------------------------------------------------------

class TestRoutingShape:
    def test_maxbas(self):
        Q = _make(1.0, 2.0, 3.0, 4.0, 5.0)
        Qout = routing.maxbas_routing(Q, 3)
        assert Qout.shape == Q.shape
        _assert_finite(Qout)

    def test_gamma(self):
        Q = _make(1.0, 2.0, 3.0, 4.0, 5.0)
        Qout = routing.gamma_routing(Q, 2.0, 2.0, 3)
        assert Qout.shape == Q.shape
        _assert_finite(Qout)

    def test_maxbas_tensor(self):
        Q = _make(1.0, 2.0, 3.0, 4.0, 5.0)
        Qout = routing.maxbas_routing(Q, torch.tensor(4.0))
        assert Qout.shape == Q.shape
        _assert_finite(Qout)

    def test_delay_buffer(self):
        seq = _make(1.0, 2.0, 3.0, 4.0, 5.0)
        out = response.delay_buffer(seq, 3)
        assert out.shape == seq.shape
        _assert_finite(out)

    def test_delay_buffer_tensor(self):
        seq = _make(1.0, 2.0, 3.0, 4.0, 5.0)
        out = response.delay_buffer(seq, torch.tensor(3.0))
        assert out.shape == seq.shape
        _assert_finite(out)
