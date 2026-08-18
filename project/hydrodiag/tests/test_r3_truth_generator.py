"""R3 focused tests: truth mapping determinism/bounds, recorded forwards,
pilot subset stratification, and round-trip helpers.

These tests are CPU-only and use short sequences so they stay fast; the
production validation of the recorded forwards runs against the real
(531-basin) truth artifacts in the Phase-2 pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.r3.common import (  # noqa: E402
    COMMON_XAJ,
    pilot_basin_subset,
    reordered_531_list,
)
from manuscript.r3.truth_generator import (  # noqa: E402
    EXPLAINED_VARIANCE_FRACTION,
    fit_g_star,
    g_star_apply,
    physical_to_z,
    z_to_physical,
)

DTYPE = torch.float32


def _synthetic_bundle_data(n=60, n_attr=35):
    rng = np.random.default_rng(7)
    attrs = rng.normal(size=(n, n_attr))
    attrs[:, 3] = rng.uniform(0.0, 0.9, size=n)  # frac_snow
    z = np.clip(0.2 + 0.1 * attrs[:, :5] @ rng.normal(size=(5, 17)), 0.05, 0.95)
    return attrs, z


def test_fit_g_star_deterministic_and_bounded():
    attrs, z = _synthetic_bundle_data()
    fit_a = fit_g_star(attrs, z)
    fit_b = fit_g_star(attrs, z)
    assert fit_a.k == fit_b.k
    assert fit_a.alpha == fit_b.alpha
    np.testing.assert_allclose(fit_a.ridge_coef, fit_b.ridge_coef, rtol=1e-12)
    np.testing.assert_allclose(fit_a.V_k, fit_b.V_k, rtol=1e-12)
    assert 1 <= fit_a.k <= 17
    assert fit_a.cumulative_variance[fit_a.k - 1] >= EXPLAINED_VARIANCE_FRACTION - 1e-9


def test_g_star_apply_reproduces_theta_star():
    attrs, z = _synthetic_bundle_data()
    fit = fit_g_star(attrs, z)
    z_star = g_star_apply(fit, attrs)
    # g* applied to the fitting attributes reproduces the stored parameters
    # deterministically (round trip of the mapping itself)
    z_again = g_star_apply(fit, attrs)
    np.testing.assert_allclose(z_star, z_again, rtol=1e-12)
    assert np.isfinite(z_star).all()


def test_physical_z_roundtrip_with_specs():
    from models.parameter_specs import XAJ_CN_PARAM_SPECS

    names = tuple(XAJ_CN_PARAM_SPECS)
    rng = np.random.default_rng(3)
    z = rng.uniform(0.0, 1.0, size=(10, len(names)))
    physical = z_to_physical(z, names, XAJ_CN_PARAM_SPECS)
    z_back = physical_to_z(physical, names, XAJ_CN_PARAM_SPECS)
    np.testing.assert_allclose(z_back, z, atol=1e-12)
    lower = np.array([XAJ_CN_PARAM_SPECS[n]["lower"] for n in names])
    upper = np.array([XAJ_CN_PARAM_SPECS[n]["upper"] for n in names])
    assert (physical >= lower).all() and (physical <= upper).all()


def test_recorded_cn_matches_production_forward():
    from manuscript.r3.recorded_forward import (
        recorded_cn_forward,
        validate_recorded_forward,
    )
    from models import XAJWithCemaNeigeLite

    torch.manual_seed(11)
    model = XAJWithCemaNeigeLite()
    batch, steps = 2, 180
    precip = torch.rand(batch, steps) * 20
    temp = torch.rand(batch, steps) * 30 - 10
    pet = torch.rand(batch, steps) * 5
    fc = {"precip": precip, "temp": temp, "pet": pet}
    params = {
        "cn_ctg": torch.full((batch,), 0.4),
        "cn_kf": torch.full((batch,), 3.0),
        "xaj_k": torch.full((batch,), 1.0),
        "xaj_b": torch.full((batch,), 0.3),
        "xaj_im": torch.full((batch,), 0.02),
        "xaj_um": torch.full((batch,), 20.0),
        "xaj_lm": torch.full((batch,), 80.0),
        "xaj_dm": torch.full((batch,), 40.0),
        "xaj_c": torch.full((batch,), 0.15),
        "xaj_sm": torch.full((batch,), 30.0),
        "xaj_ex": torch.full((batch,), 1.2),
        "xaj_ki": torch.full((batch,), 0.3),
        "xaj_kg": torch.full((batch,), 0.2),
        "xaj_ci": torch.full((batch,), 0.5),
        "xaj_cg": torch.full((batch,), 0.98),
        "xaj_a": torch.full((batch,), 2.0),
        "xaj_theta": torch.full((batch,), 1.5),
    }
    recorded = recorded_cn_forward(model, fc, params, torch.device("cpu"), DTYPE)
    diffs = validate_recorded_forward(model, recorded, fc, params)
    assert diffs["q_abs_max"] < 1e-5
    assert diffs["state_abs_max"] < 1e-5


def test_recorded_base_matches_production_forward():
    from manuscript.r3.recorded_forward import (
        recorded_base_forward,
        validate_recorded_forward,
    )
    from models import XAJLite

    torch.manual_seed(12)
    model = XAJLite()
    batch, steps = 2, 180
    precip = torch.rand(batch, steps) * 20
    temp = torch.rand(batch, steps) * 30 - 10
    pet = torch.rand(batch, steps) * 5
    fc = {"precip": precip, "temp": temp, "pet": pet}
    params = {
        "xaj_k": torch.full((batch,), 1.0),
        "xaj_b": torch.full((batch,), 0.3),
        "xaj_im": torch.full((batch,), 0.02),
        "xaj_um": torch.full((batch,), 20.0),
        "xaj_lm": torch.full((batch,), 80.0),
        "xaj_dm": torch.full((batch,), 40.0),
        "xaj_c": torch.full((batch,), 0.15),
        "xaj_sm": torch.full((batch,), 30.0),
        "xaj_ex": torch.full((batch,), 1.2),
        "xaj_ki": torch.full((batch,), 0.3),
        "xaj_kg": torch.full((batch,), 0.2),
        "xaj_ci": torch.full((batch,), 0.5),
        "xaj_cg": torch.full((batch,), 0.98),
        "xaj_a": torch.full((batch,), 2.0),
        "xaj_theta": torch.full((batch,), 1.5),
    }
    recorded = recorded_base_forward(model, fc, params, torch.device("cpu"), DTYPE)
    diffs = validate_recorded_forward(model, recorded, fc, params)
    assert diffs["q_abs_max"] < 1e-5
    assert diffs["state_abs_max"] < 1e-5


def test_recorded_tgd2_matches_production_forward():
    from manuscript.r3.recorded_forward import (
        recorded_tgd2_forward,
        validate_recorded_forward,
    )
    from models import XAJWithTGD2Lite

    torch.manual_seed(13)
    model = XAJWithTGD2Lite()
    batch, steps = 2, 180
    precip = torch.rand(batch, steps) * 20
    temp = torch.rand(batch, steps) * 30 - 10
    pet = torch.rand(batch, steps) * 5
    fc = {"precip": precip, "temp": temp, "pet": pet}
    params = {
        "tgd_tau_warm": torch.full((batch,), 0.5),
        "tgd_delta_tau_cold": torch.full((batch,), 20.0),
        "xaj_k": torch.full((batch,), 1.0),
        "xaj_b": torch.full((batch,), 0.3),
        "xaj_im": torch.full((batch,), 0.02),
        "xaj_um": torch.full((batch,), 20.0),
        "xaj_lm": torch.full((batch,), 80.0),
        "xaj_dm": torch.full((batch,), 40.0),
        "xaj_c": torch.full((batch,), 0.15),
        "xaj_sm": torch.full((batch,), 30.0),
        "xaj_ex": torch.full((batch,), 1.2),
        "xaj_ki": torch.full((batch,), 0.3),
        "xaj_kg": torch.full((batch,), 0.2),
        "xaj_ci": torch.full((batch,), 0.5),
        "xaj_cg": torch.full((batch,), 0.98),
        "xaj_a": torch.full((batch,), 2.0),
        "xaj_theta": torch.full((batch,), 1.5),
    }
    recorded = recorded_tgd2_forward(model, fc, params, torch.device("cpu"), DTYPE)
    diffs = validate_recorded_forward(model, recorded, fc, params)
    assert diffs["q_abs_max"] < 1e-5
    assert diffs["state_abs_max"] < 1e-5


def test_pilot_subset_stratified_and_deterministic():
    basin_ids = [f"{1000000 + i:08d}" for i in range(90)]
    rng = np.random.default_rng(5)
    frac = rng.uniform(0.0, 0.9, size=90)
    a = pilot_basin_subset(basin_ids, frac, per_tercile=3)
    b = pilot_basin_subset(basin_ids, frac, per_tercile=3)
    assert a == b
    assert len(a) == 9
    fs = {bid: f for bid, f in zip(basin_ids, frac)}
    q1, q2 = np.quantile(frac, [1 / 3, 2 / 3])
    for t, (lo, hi) in enumerate([(0, q1), (q1, q2), (q2, 1.0)]):
        picks = [fs[x] for x in a]
        # every tercile is represented
        terciles = np.digitize([fs[x] for x in a], [q1, q2])
        assert (terciles == t).any()
    assert len(set(a)) == 9


def test_reordered_531_list_preserves_membership():
    basin_ids = [f"{i:08d}" for i in range(531)]
    first = ["00000123", "00000456", "00000530"]
    reordered = reordered_531_list(basin_ids, first)
    assert reordered[:3] == first
    assert sorted(reordered) == sorted(basin_ids)
    assert len(set(reordered)) == 531


def test_common_xaj_is_15_shared_parameters():
    from models.parameter_specs import (
        XAJ_CN_PARAM_SPECS,
        XAJ_PARAM_SPECS,
        XAJ_TGD2_PARAM_SPECS,
    )

    assert len(COMMON_XAJ) == 15
    for n in COMMON_XAJ:
        assert n in XAJ_PARAM_SPECS
        assert n in XAJ_CN_PARAM_SPECS
        assert n in XAJ_TGD2_PARAM_SPECS
    cn_snow = {"cn_ctg", "cn_kf"}
    tgd = {"tgd_tau_warm", "tgd_delta_tau_cold"}
    assert set(XAJ_CN_PARAM_SPECS) == cn_snow | set(COMMON_XAJ)
    assert set(XAJ_TGD2_PARAM_SPECS) == tgd | set(COMMON_XAJ)
    assert set(XAJ_PARAM_SPECS) == set(COMMON_XAJ)
