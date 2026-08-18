"""Phase A/B regression tests: CN forward identity (g_thresh), oracle paths,
and attribute/target-override identity.

Tests that need the frozen truth artifacts or the CAMELS data skip when those
files are absent, so the suite stays runnable in a bare checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# The production runners raise the Dynamo limits for many-model test
# processes; mirror that here so small shape variations across tests do not
# trip the default recompile cap.
try:
    import torch._dynamo as _dynamo

    _dynamo.config.recompile_limit = max(_dynamo.config.recompile_limit, 256)
    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 256)
except (ImportError, AttributeError):
    pass

from manuscript.r3.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_ROOT,
    load_bundle,
)

TRUTH_DIR = DEFAULT_RESULTS_ROOT / "r3_synthetic_truth_v1"
HAS_TRUTH = (TRUTH_DIR / "q_star.npz").exists() and (
    TRUTH_DIR / "theta_star.npz"
).exists()
HAS_DATA = (DEFAULT_DATA_ROOT / "camels_dataset").exists()

DTYPE = torch.float32


def _cn_params(batch: int) -> dict[str, torch.Tensor]:
    return {
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


def test_cn_model_override_identity_and_effect():
    """cn_psol_annual equal to the input-derived value is a no-op; a different
    value changes the output (override actually reaches the kernel)."""
    from models import XAJWithCemaNeigeLite
    from models.cemaneige import _estimate_psol_annual

    torch.manual_seed(21)
    model = XAJWithCemaNeigeLite()
    batch, steps = 2, 400
    precip = torch.rand(batch, steps) * 20
    temp = torch.rand(batch, steps) * 30 - 10
    pet = torch.rand(batch, steps) * 5
    params = _cn_params(batch)
    psol_derived = _estimate_psol_annual(precip, temp)

    fc_plain = {"precip": precip, "temp": temp, "pet": pet}
    fc_same = {**fc_plain, "cn_psol_annual": psol_derived}
    fc_zero = {**fc_plain, "cn_psol_annual": torch.zeros(batch)}
    with torch.no_grad():
        q_plain, _ = model(forcings=fc_plain, params=params)
        q_same, _ = model(forcings=fc_same, params=params)
        q_zero, _ = model(forcings=fc_zero, params=params)
    assert torch.allclose(q_plain, q_same, atol=1e-6)
    assert not torch.allclose(q_plain, q_zero, atol=1e-3)


def test_cn_override_preserves_recorded_forward_identity():
    """With the canonical override present, the recorded forward still
    matches the production forward exactly."""
    from manuscript.r3.recorded_forward import (
        recorded_cn_forward,
        validate_recorded_forward,
    )
    from models import XAJWithCemaNeigeLite
    from models.cemaneige import _estimate_psol_annual

    torch.manual_seed(22)
    model = XAJWithCemaNeigeLite()
    batch, steps = 2, 300
    precip = torch.rand(batch, steps) * 20
    temp = torch.rand(batch, steps) * 30 - 10
    pet = torch.rand(batch, steps) * 5
    params = _cn_params(batch)
    psol = _estimate_psol_annual(precip, temp)
    fc = {"precip": precip, "temp": temp, "pet": pet, "cn_psol_annual": psol}
    recorded = recorded_cn_forward(model, fc, params, torch.device("cpu"), DTYPE)
    diffs = validate_recorded_forward(model, recorded, fc, params)
    assert diffs["q_abs_max"] < 1e-5
    assert diffs["state_abs_max"] < 1e-5


@pytest.mark.skipif(not HAS_TRUTH, reason="frozen R3 truth artifacts absent")
def test_theta_star_ic_path_reproduces_q_star():
    """theta* through the IC objective path (canonical psol) reproduces q_star
    at float level on the train split for a subset of basins."""
    import torch as _torch
    from ablation.ic_core.parameter_adapter import physical_to_normalized
    from ablation.ic_core.runtime import ICObjectiveRuntime
    from manuscript.r3.common import bundle_with_synthetic_target

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    bundle, _ = load_bundle()
    theta = np.load(TRUTH_DIR / "theta_star.npz")
    theta_star = theta["parameters"]
    q_star = np.load(TRUTH_DIR / "q_star.npz")["target_mm_day"]
    syn_bundle = bundle_with_synthetic_target(bundle, q_star)
    config = {
        "device": str(device),
        "model_variant": "lite",
        "batching": {"basin_batch_size": 4, "cache_device_data": False},
        "objective": {"min_samples": 30},
        "canonical_cn_psol_annual": True,
    }
    runtime = ICObjectiveRuntime(syn_bundle, config, "XAJ_CN", model_variant="lite")
    # first 8 basins (no-snow) + 4 snowy basins
    index = {b: i for i, b in enumerate(bundle.basin_ids)}
    snowy = [b for b in ["08377900", "11522500", "04027000", "12451000"] if b in index]
    basin_ids = list(bundle.basin_ids[:8]) + snowy
    basin_indices = [index[b] for b in basin_ids]
    theta_01 = physical_to_normalized("XAJ_CN", theta_star[basin_indices], clip=False)
    fit, _ = runtime.evaluate_candidates_tensor(
        _torch.from_numpy(theta_01).unsqueeze(1).to(device, dtype=_torch.float64),
        basin_indices=basin_indices,
        split="train",
    )
    # fitness is KGE vs Q*; float-level identity implies KGE ~ 1
    kges = fit[:, 0].detach().cpu().numpy()
    assert np.isfinite(kges).all()
    assert float(kges.min()) > 0.9995, f"IC-path oracle KGE too low: {kges}"


@pytest.mark.skipif(not HAS_TRUTH, reason="frozen R3 truth artifacts absent")
def test_theta_star_dpl_window_path_reproduces_q_star_window():
    """theta* through a 730-day dPL window (canonical psol) reproduces the
    corresponding q_star window; the residual is bounded by the frozen
    365-day warm-up convention (measured <= ~2 mm/day even for snowy basins;
    we assert a conservative 5 mm/day bound)."""
    from manuscript.r3.recorded_forward import build_forcing_dict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle, _ = load_bundle()
    theta = np.load(TRUTH_DIR / "theta_star.npz")
    theta_star = theta["parameters"]
    names = [str(n) for n in theta["parameter_names"]]
    q_star = np.load(TRUTH_DIR / "q_star.npz")["target_mm_day"]

    from models import XAJWithCemaNeigeLite
    from models.cemaneige import _estimate_psol_annual

    model = XAJWithCemaNeigeLite().to(device).eval()
    psol_full = (
        _estimate_psol_annual(
            torch.from_numpy(bundle.forcing[:, :, 0]),
            torch.from_numpy(bundle.forcing[:, :, 1]),
        )
        .numpy()
        .astype(np.float32)
    )
    p = bundle.periods
    warmup = p.warmup.days
    # one snowy and one no-snow basin, one window each
    index = {b: i for i, b in enumerate(bundle.basin_ids)}
    for basin in ("08377900", "01022500"):
        b = index[basin]
        s = p.train.start_index + 730
        forcing_np = bundle.forcing[b : b + 1, s - warmup : s + 365].astype(np.float32)
        fc = build_forcing_dict(forcing_np, device, DTYPE)
        fc["cn_psol_annual"] = torch.from_numpy(psol_full[b : b + 1]).to(
            device, dtype=DTYPE
        )
        params = {
            name: torch.from_numpy(theta_star[b : b + 1, i]).to(device, dtype=DTYPE)
            for i, name in enumerate(names)
        }
        with torch.no_grad():
            q, _ = model(forcings=fc, params=params)
        q_scored = q[0, warmup:].detach().cpu().numpy().astype(np.float64)
        diff = np.abs(q_scored - q_star[b, s : s + 365])
        assert float(diff.max()) < 5.0, f"{basin} window oracle diff {diff.max():.3f}"


@pytest.mark.skipif(not HAS_DATA, reason="CAMELS data absent")
def test_target_override_preserves_basin_order_and_normalization():
    """The dPL loader with target_override_npz must keep basin order,
    attribute rows and normalization statistics identical to the observed
    path."""
    from training.dpl.run_dpl_model import gate_time_index, load_data, robust_normalize

    config = json.loads(
        (PROJECT / "training/dpl/base_config_camels_531.json").read_text()
    )
    config["output_dir"] = str(PROJECT / "tmp_r3_test_out")
    config["data_basin_ids"] = str(DEFAULT_DATA_ROOT / "531sub_id.txt")
    indices = gate_time_index(config)

    base = load_data(config, indices, max_basins=8)
    config["target_override_npz"] = str(TRUTH_DIR / "q_star.npz")
    over = load_data(config, indices, max_basins=8)

    assert base[0] == over[0]  # basin ids identical
    np.testing.assert_array_equal(base[1], over[1])  # raw attribute rows
    norm_base, stats_base = robust_normalize(base[1])
    norm_over, stats_over = robust_normalize(over[1])
    np.testing.assert_allclose(norm_base, norm_over)
    np.testing.assert_allclose(stats_base["median"], stats_over["median"])
    # the override target is exactly the q_star slice (and differs from obs)
    assert over[3].shape == base[3].shape
    # the override target is exactly the q_star slice
    q_star_npz = np.load(TRUTH_DIR / "q_star.npz")
    q_star = q_star_npz["target_mm_day"]
    star_ids = [str(b) for b in q_star_npz["basin_ids"]]
    sel = np.array([star_ids.index(b) for b in over[0]], dtype=np.int64)
    ci_s = indices["calibration"][0]
    ci_e = indices["calibration"][1]
    # loader converts Q* to float32; compare in float32 space
    np.testing.assert_allclose(
        over[3], q_star[sel][:, ci_s : ci_e + 1].astype(np.float32), atol=1e-5
    )


@pytest.mark.skipif(not HAS_TRUTH, reason="frozen R3 truth artifacts absent")
def test_target_override_row_alignment_with_reordered_basin_list():
    """Regression: with a reordered basin list (pilot-first), the loader must
    still pair every basin with its own Q* row (basin-ID based selection)."""
    import tempfile

    from manuscript.r3.common import pilot_basin_subset, reordered_531_list
    from training.dpl.run_dpl_model import gate_time_index, load_data

    bundle, _ = load_bundle()
    snow = frac_snow(bundle)
    frac_map = dict(zip(bundle.basin_ids, snow))
    pilot = pilot_basin_subset(
        bundle.basin_ids, [frac_map[b] for b in bundle.basin_ids], per_tercile=4
    )
    reordered = reordered_531_list(bundle.basin_ids, pilot)
    with tempfile.TemporaryDirectory() as tmp:
        order_file = Path(tmp) / "pilot_basin_order_531.json"
        order_file.write_text(json.dumps(reordered))
        config = json.loads(
            (PROJECT / "training/dpl/base_config_camels_531.json").read_text()
        )
        config["output_dir"] = str(PROJECT / "tmp_r3_test_out")
        config["data_basin_ids"] = str(order_file)
        config["target_override_npz"] = str(TRUTH_DIR / "q_star.npz")
        indices = gate_time_index(config)
        basin_ids, _, _, cal_obs, _, _ = load_data(config, indices, max_basins=12)
        q_star_npz = np.load(TRUTH_DIR / "q_star.npz")
        q_star = q_star_npz["target_mm_day"]
        star_ids = [str(b) for b in q_star_npz["basin_ids"]]
        ci_s = indices["calibration"][0]
        ci_e = indices["calibration"][1]
        for k, basin in enumerate(basin_ids):
            pb = star_ids.index(basin)
            np.testing.assert_allclose(
                cal_obs[k],
                q_star[pb, ci_s : ci_e + 1].astype(np.float32),
                atol=1e-5,
                err_msg=f"basin {basin} override row misaligned",
            )


def frac_snow(bundle):
    from manuscript.r3.common import frac_snow_series

    return list(frac_snow_series(bundle)["frac_snow"])


def load_gage_ids(path):
    from training.data_contract import load_gage_ids as _load

    return _load(path)


def test_attribute_identity_contract():
    """35 attributes, frac_snow at index 3, shared-XAJ names consistent."""
    from ablation.ic_core.data_adapter import ATTRIBUTE_NAMES
    from models.parameter_specs import XAJ_CN_PARAM_SPECS

    assert len(ATTRIBUTE_NAMES) == 35
    assert ATTRIBUTE_NAMES[3] == "frac_snow"
    assert "frac_snow" in ATTRIBUTE_NAMES
    assert {"cn_ctg", "cn_kf"} | set(
        [
            "xaj_k",
            "xaj_b",
            "xaj_im",
            "xaj_um",
            "xaj_lm",
            "xaj_dm",
            "xaj_c",
            "xaj_sm",
            "xaj_ex",
            "xaj_ki",
            "xaj_kg",
            "xaj_ci",
            "xaj_cg",
            "xaj_a",
            "xaj_theta",
        ]
    ) == set(XAJ_CN_PARAM_SPECS)
