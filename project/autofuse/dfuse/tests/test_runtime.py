import torch

from dfuse import (
    GENERATED_STEP_REGISTRY,
    GraphSignature,
    get_generated_step,
    get_structure,
    reset_runtime_registries,
)
from dfuse.kernel import _make_sequential_step
from dfuse.kernel import _capacity, _parameter_values, _parameter_vector
from dfuse.runtime import _fix_lower_proportional, _runtime_snow_balance_input, _runtime_snow_step, _saturation_area
from dfuse.spec import FLUX_NAMES, STATE_NAMES, default_parameters
from project.autofuse.runtime_validation import _default_forcing, _prepare, _run


ORDER = ("recharge", "et", "surface_runoff", "percolation", "interflow", "baseflow")


def test_graph_signature_excludes_model_id_and_generated_code_is_independent():
    reset_runtime_registries()
    rows = [get_generated_step(get_structure(model_id), order=ORDER, n_substeps=1) for model_id in (2, 108, 178, 210)]
    signatures = [signature for signature, _ in rows]
    functions = [function for _, function in rows]
    assert len(set(signatures)) == 4
    assert len({id(function.__code__) for function in functions}) == 4
    assert all("model_id" not in signature.to_dict() for signature in signatures)
    assert GENERATED_STEP_REGISTRY.builds == 4
    same_signature, same_function = get_generated_step(get_structure(2), order=ORDER, n_substeps=1)
    assert same_signature == signatures[0]
    assert same_function is functions[0]
    assert GraphSignature.from_spec(get_structure(2), order="S1", n_substeps=1) == signatures[0]


def test_generated_eager_matches_current_sequential_step_for_mothers():
    reset_runtime_registries()
    forcing = _default_forcing(3, torch.device("cpu"))
    for model_id in (2, 108, 178, 210):
        prepared = _prepare(model_id, forcing)
        _, generated = get_generated_step(prepared["spec"], order=ORDER, n_substeps=1)
        current = _run(_make_sequential_step(ORDER, 1), prepared, current=True)
        result = _run(generated, prepared)
        torch.testing.assert_close(result["q"], current["q"], rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(result["packed"], current["packed"], rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(result["diagnostics"], current["diagnostics"], rtol=1.0e-12, atol=1.0e-12)


def test_arno_surface_area_reaches_one_at_fortran_capacity_boundary():
    params = _parameter_values(default_parameters(), dtype=torch.float64, device=torch.device("cpu"))
    theta = _parameter_vector(params)
    state = torch.zeros(len(STATE_NAMES), dtype=torch.float64)
    state[STATE_NAMES.index("TENS_1")] = params["FRACTEN"] * params["MAXWATR_1"]
    state[STATE_NAMES.index("FREE_1")] = (1.0 - params["FRACTEN"]) * params["MAXWATR_1"]
    area = _saturation_area(
        state, theta, state[0] * 0.0, state[0] * 0.0,
        arch1="tension1_1", arch2="unlimfrc_2", qsurf="arno_x_vic",
    )
    torch.testing.assert_close(area, torch.ones_like(area), rtol=0.0, atol=0.0)

def test_watr2_lower_floor_uses_matching_capacity():
    state = torch.zeros(len(STATE_NAMES), dtype=torch.float64)
    state[STATE_NAMES.index("WATR_2")] = 1.6978680419921876e-6
    lower = torch.tensor(1.7827275868632043e-6, dtype=torch.float64)
    flux = torch.zeros(len(FLUX_NAMES), dtype=torch.float64)
    flux[11] = 0.4
    flux[15] = 0.2
    dt = torch.tensor(1.0, dtype=torch.float64)
    corrected_state, corrected_flux, error_loss, violated = _fix_lower_proportional(state, flux, STATE_NAMES.index("WATR_2"), lower, dt, (11, 15))
    torch.testing.assert_close(corrected_state[8], lower, rtol=0.0, atol=0.0)
    torch.testing.assert_close(error_loss, state[8] - lower, rtol=0.0, atol=0.0)
    torch.testing.assert_close((corrected_state[8] - state[8]) + (corrected_flux[11] + corrected_flux[15] - flux[11] - flux[15]) * dt, torch.zeros((), dtype=torch.float64), rtol=0.0, atol=1.0e-15)
    assert bool(violated)
def test_snow_balance_input_matches_strict_source_temperature_partition():
    params = _parameter_values(default_parameters(), dtype=torch.float64, device=torch.device("cpu"))
    theta = _parameter_vector(params)
    snow = torch.zeros((), dtype=torch.float64)
    jday = torch.tensor(21.0, dtype=torch.float64)
    leap = torch.tensor(0.0, dtype=torch.float64)
    dt = torch.tensor(1.0, dtype=torch.float64)
    ppt = torch.tensor(15.45, dtype=torch.float64)
    threshold_temp = params["PXTEMP"]
    effective, next_snow = _runtime_snow_step(ppt, threshold_temp, snow, jday, leap, theta, dt)
    participating = _runtime_snow_balance_input(ppt, threshold_temp, theta)
    torch.testing.assert_close(participating - effective - (next_snow - snow), torch.zeros_like(snow), rtol=0.0, atol=0.0)
    torch.testing.assert_close(participating, torch.zeros_like(snow), rtol=0.0, atol=0.0)
    warm_effective, warm_next_snow = _runtime_snow_step(ppt, threshold_temp + 1.0, snow, jday, leap, theta, dt)
    warm_participating = _runtime_snow_balance_input(ppt, threshold_temp + 1.0, theta)
    torch.testing.assert_close(warm_participating - warm_effective - (warm_next_snow - snow), torch.zeros_like(snow), rtol=0.0, atol=0.0)
def test_coupled_rk2_has_structure_specialized_rhs_and_stage_diagnostics():
    reset_runtime_registries()
    signature, generated = get_generated_step(
        get_structure(2), order=("coupled_rhs",), n_substeps=1, execution_mode="coupled_rk2"
    )
    assert signature.execution_mode == "coupled_rk2"
    assert signature.sequential_order == ("coupled_rhs",)
    assert "for process in" not in generated.generated_source
    assert "_runtime_coupled_rk2_step_impl" in generated.generated_source
    from dfuse import simulate_coupled_rk2
    result = simulate_coupled_rk2(
        2, torch.tensor([[5.0, 2.0, 10.0]], dtype=torch.float64), compile_step=False
    )
    assert result.states.shape == (2, 5)
    assert result.sequential_diagnostics is not None
    assert "stage1_qperc" in result.sequential_diagnostics
    assert torch.isfinite(result.states).all()
