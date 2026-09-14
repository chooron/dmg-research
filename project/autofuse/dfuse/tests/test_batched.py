import torch

from dfuse import simulate_coupled_rk2, simulate_coupled_rk2_batched
from dfuse.batched import _monitor_code
from dfuse.spec import FLUX_NAMES, STATE_NAMES


def test_batched_eager_matches_serial_for_each_basin():
    forcing = torch.tensor(
        [
            [[5.0, 2.0, 10.0], [4.0, 2.0, 9.0], [0.0, 1.0, -2.0]],
            [[2.0, 1.0, 0.0], [3.0, 1.0, 2.0], [1.0, 1.0, 4.0]],
        ],
        dtype=torch.float64,
    )
    batched = simulate_coupled_rk2_batched(2, forcing, [{}, {}], basin_ids=("a", "b"), compile_step=False)
    assert batched.q.shape == (2, 3)
    assert batched.states.shape == (2, 4, 5)
    assert batched.monitoring["completed_full_period"]
    for basin in range(2):
        serial = simulate_coupled_rk2(2, forcing[basin], {}, compile_step=False)
        torch.testing.assert_close(batched.q[basin], serial.q, rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(batched.states[basin], serial.states, rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(batched.water_balance_residual[basin], serial.water_balance_residual, rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(batched.snow_balance_residual[basin], serial.snow_balance_residual, rtol=1.0e-12, atol=1.0e-12)
        for name in FLUX_NAMES:
            torch.testing.assert_close(batched.fluxes[name][basin], serial.fluxes[name], rtol=1.0e-12, atol=1.0e-12)



def test_lite_and_full_are_explicit_modes_with_final_state_view():
    forcing = torch.tensor(
        [[[5.0, 2.0, 10.0], [4.0, 2.0, 9.0], [0.0, 1.0, -2.0]]],
        dtype=torch.float64,
    )
    full = simulate_coupled_rk2_batched(2, forcing, [{}], compile_step=False, output_mode="full")
    lite = simulate_coupled_rk2_batched(2, forcing, [{}], compile_step=False, output_mode="lite")
    legacy = simulate_coupled_rk2_batched(2, forcing, [{}], compile_step=False, output_mode="q_only")
    assert full.output_mode == "full"
    assert lite.output_mode == "lite"
    assert legacy.output_mode == "lite"
    assert lite.fluxes == {}
    assert lite.q_instantaneous is None
    assert lite.snow is None
    assert legacy.q_instantaneous is None
    assert legacy.snow is None
    assert lite.states.shape == (1, 1, 5)
    assert full.states.shape == (1, 4, 5)
    torch.testing.assert_close(lite.q, full.q)
    torch.testing.assert_close(lite.final_states, full.final_states)
    torch.testing.assert_close(legacy.q, lite.q)

def test_batched_monitor_codes_report_per_basin_water_failure():
    batch = 2
    packed = torch.zeros(batch, len(STATE_NAMES) + 1 + 500, dtype=torch.float64)
    routed = torch.zeros(batch, dtype=torch.float64)
    diagnostics = torch.zeros(batch, len(FLUX_NAMES) + 12, dtype=torch.float64)
    diagnostics[1, len(FLUX_NAMES)] = 1.0e-6
    capacities = torch.ones(batch, len(STATE_NAMES), dtype=torch.float64)
    bounded = torch.ones(batch, len(STATE_NAMES), dtype=torch.bool)
    code = _monitor_code(packed, routed, diagnostics, capacities, bounded)
    assert code.tolist() == [0, 8]
