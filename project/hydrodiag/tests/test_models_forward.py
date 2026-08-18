"""Forward sanity tests for all full models.

Verifies:
- qsim.shape == [batch, time]
- qsim is finite
- qsim is non-negative or close to non-negative
- aux is a dict
"""

import torch
import pytest

try:
    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 256)
    _dynamo.config.recompile_limit = max(_dynamo.config.recompile_limit, 256)
except (ImportError, AttributeError):
    pass
from models import (
    HBV, GR4J, XAJ, SIMHYD, CemaNeige, CemaNeigeHyst,
    GR4JWithCemaNeige, XAJWithCemaNeige, SIMHYDWithCemaNeige,
    GR4JWithTGD2, XAJWithTGD2, SIMHYDWithTGD2,
)
from models.parameter_specs import (
    HBV_PARAM_SPECS,
    GR4J_PARAM_SPECS,
    XAJ_PARAM_SPECS,
    SIMHYD_PARAM_SPECS,
    CEMANEIGE_PARAM_SPECS,
    CEMANEIGE_HYST_PARAM_SPECS,
    GR4J_CN_PARAM_SPECS,
    XAJ_CN_PARAM_SPECS,
    SIMHYD_CN_PARAM_SPECS,
    GR4J_TGD2_PARAM_SPECS,
    XAJ_TGD2_PARAM_SPECS,
    SIMHYD_TGD2_PARAM_SPECS,
)


BATCH = 3
TIME = 20


def make_synthetic_forcings(batch, time, device, dtype):
    """Create random but physically plausible forcings."""
    torch.manual_seed(42)
    precip = torch.rand(batch, time, device=device, dtype=dtype) * 10.0
    pet = torch.rand(batch, time, device=device, dtype=dtype) * 5.0
    temp = torch.randn(batch, time, device=device, dtype=dtype) * 10.0
    return {"precip": precip, "pet": pet, "temp": temp}


def make_params(param_specs, batch, device, dtype):
    """Create random parameters within valid ranges."""
    torch.manual_seed(123)
    params = {}
    for name, spec in param_specs.items():
        lo = spec["lower"]
        hi = spec["upper"]
        val = lo + torch.rand(batch, device=device, dtype=dtype) * (hi - lo)
        # Ensure requested dtype
        val = val.to(dtype=dtype)
        params[name] = val

    # Special handling: ensure ki+kg < 1 for XAJ
    if "xaj_ki" in params and "xaj_kg" in params:
        s = params["xaj_ki"] + params["xaj_kg"]
        mask = s >= 1.0
        if mask.any():
            params["xaj_ki"] = torch.where(mask, params["xaj_ki"] * 0.95 / s.clamp(min=1e-6), params["xaj_ki"])
            params["xaj_kg"] = torch.where(mask, params["xaj_kg"] * 0.95 / s.clamp(min=1e-6), params["xaj_kg"])
    return params


def make_params_composed(param_specs, batch, device, dtype):
    """Create params for composed models with prefixed names."""
    return make_params(param_specs, batch, device, dtype)


def validate_forward_output(qsim, aux, batch, time, model_name):
    """Check forward pass outputs are valid."""
    assert qsim.shape == (batch, time), (
        f"[{model_name}] qsim shape {qsim.shape} != ({batch}, {time})"
    )
    assert torch.isfinite(qsim).all(), (
        f"[{model_name}] qsim contains NaN or Inf"
    )
    # Allow small negative due to numerical precision
    assert (qsim >= -1e-3).all(), (
        f"[{model_name}] qsim contains large negative values: min={qsim.min().item():.4f}"
    )
    assert isinstance(aux, dict), (
        f"[{model_name}] aux is not a dict: {type(aux)}"
    )


FORWARD_DEVICES_AND_DTYPES = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
] + ([("cuda", torch.float32), ("cuda", torch.float64)] if torch.cuda.is_available() else [])


class TestModelForward:
    """Forward sanity tests for all models."""

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_hbv_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = HBV().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params(HBV_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "HBV")

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_gr4j_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = GR4J().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params(GR4J_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "GR4J")

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_xaj_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = XAJ().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params(XAJ_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "XAJ")

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_cemaneige_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = CemaNeige().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params(CEMANEIGE_PARAM_SPECS, BATCH, device, dtype)
        outflow, aux = model(forcings=forcings, params=params)
        assert outflow.shape == (BATCH, TIME), (
            f"[CemaNeige] outflow shape {outflow.shape} != ({BATCH}, {TIME})"
        )
        assert torch.isfinite(outflow).all(), "[CemaNeige] outflow contains NaN/Inf"
        assert (outflow >= -1e-3).all(), f"[CemaNeige] outflow negative: min={outflow.min().item():.4f}"
        assert isinstance(aux, dict), f"[CemaNeige] aux is not dict: {type(aux)}"

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_cemaneige_hyst_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = CemaNeigeHyst().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params(CEMANEIGE_HYST_PARAM_SPECS, BATCH, device, dtype)
        outflow, aux = model(forcings=forcings, params=params)
        assert outflow.shape == (BATCH, TIME)
        assert torch.isfinite(outflow).all()
        assert (outflow >= -1e-3).all()
        assert isinstance(aux, dict)

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_gr4j_cn_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = GR4JWithCemaNeige().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params_composed(GR4J_CN_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "GR4J+CemaNeige")
        assert "effective_precip" in aux, "GR4J+CemaNeige missing effective_precip in aux"

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_xaj_cn_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = XAJWithCemaNeige().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params_composed(XAJ_CN_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "XAJ+CemaNeige")
        assert "effective_precip" in aux, "XAJ+CemaNeige missing effective_precip in aux"

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_simhyd_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = SIMHYD().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params(SIMHYD_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "SIMHYD")

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_simhyd_cn_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = SIMHYDWithCemaNeige().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params_composed(SIMHYD_CN_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "SIMHYD+CemaNeige")
        assert "effective_precip" in aux, "SIMHYD+CemaNeige missing effective_precip in aux"

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_gr4j_tgd2_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = GR4JWithTGD2().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params_composed(GR4J_TGD2_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "GR4J+TGD2")
        assert "effective_precipitation" in aux, "GR4J+TGD2 missing effective_precipitation in aux"

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_simhyd_tgd2_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = SIMHYDWithTGD2().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params_composed(SIMHYD_TGD2_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "SIMHYD+TGD2")
        assert "effective_precipitation" in aux, "SIMHYD+TGD2 missing effective_precipitation in aux"

    @pytest.mark.parametrize("device_str,dtype", FORWARD_DEVICES_AND_DTYPES)
    def test_xaj_tgd2_forward(self, device_str, dtype):
        device = torch.device(device_str)
        model = XAJWithTGD2().to(device=device, dtype=dtype)
        forcings = make_synthetic_forcings(BATCH, TIME, device, dtype)
        params = make_params_composed(XAJ_TGD2_PARAM_SPECS, BATCH, device, dtype)
        qsim, aux = model(forcings=forcings, params=params)
        validate_forward_output(qsim, aux, BATCH, TIME, "XAJ+TGD2")
        assert "effective_precipitation" in aux, "XAJ+TGD2 missing effective_precipitation in aux"
