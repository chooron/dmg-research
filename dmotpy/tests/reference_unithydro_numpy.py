"""Independent NumPy reference for MARRMoT unit-hydrograph routing.

This module reproduces the MATLAB logic in:
  - MARRMoT/Unit Hydrograph files/route.m
  - MARRMoT/Unit Hydrograph files/update_uh.m
  - MARRMoT/Unit Hydrograph files/uh_1_half.m
  - MARRMoT/Unit Hydrograph files/uh_2_full.m
  - MARRMoT/Unit Hydrograph files/uh_3_half.m
  - MARRMoT/Unit Hydrograph files/uh_4_full.m
  - MARRMoT/Unit Hydrograph files/uh_5_half.m
  - MARRMoT/Unit Hydrograph files/uh_6_gamma.m
  - MARRMoT/Unit Hydrograph files/uh_7_uniform.m
  - MARRMoT/Unit Hydrograph files/uh_8_delay.m

The implementation is intentionally independent of the dMoT PyTorch code.
"""

from __future__ import annotations

from math import ceil, exp, floor, gamma, lgamma, log

import numpy as np


DEFAULT_DELTA_T = 1.0
GAMMA_EPS = 1.0e-14
GAMMA_FPMIN = 1.0e-300
GAMMA_ITMAX = 2000


def _as_float64_1d(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape {array.shape}.")
    return array


def _normalize_delay(delay: float) -> float:
    return 1.0 if delay == 0.0 else float(delay)


def _matlab_mod(a: float, b: float) -> float:
    if b == 0.0:
        return float(a)
    return float(np.mod(a, b))


def _gamma_pdf(x_value: float, shape_n: float, scale_k: float) -> float:
    if x_value <= 0.0:
        return 0.0
    return (
        1.0
        / (scale_k * gamma(shape_n))
        * (x_value / scale_k) ** (shape_n - 1.0)
        * exp(-x_value / scale_k)
    )


def _regularized_gamma_p_scalar(shape_a: float, x_value: float) -> float:
    if shape_a <= 0.0:
        raise ValueError("Gamma shape parameter must be positive.")
    if x_value <= 0.0:
        return 0.0

    gln = lgamma(shape_a)
    if x_value < shape_a + 1.0:
        ap = shape_a
        delta = 1.0 / shape_a
        total = delta
        for _ in range(GAMMA_ITMAX):
            ap += 1.0
            delta *= x_value / ap
            total += delta
            if abs(delta) <= abs(total) * GAMMA_EPS:
                break
        return total * exp(-x_value + shape_a * log(x_value) - gln)

    b_term = x_value + 1.0 - shape_a
    c_term = 1.0 / GAMMA_FPMIN
    d_term = 1.0 / b_term
    h_term = d_term
    for i_value in range(1, GAMMA_ITMAX + 1):
        an_term = -i_value * (i_value - shape_a)
        b_term += 2.0
        d_term = an_term * d_term + b_term
        if abs(d_term) < GAMMA_FPMIN:
            d_term = GAMMA_FPMIN
        c_term = b_term + an_term / c_term
        if abs(c_term) < GAMMA_FPMIN:
            c_term = GAMMA_FPMIN
        d_term = 1.0 / d_term
        delta = d_term * c_term
        h_term *= delta
        if abs(delta - 1.0) <= GAMMA_EPS:
            break
    return 1.0 - h_term * exp(-x_value + shape_a * log(x_value) - gln)


def build_uh_1_half_numpy(d_base: float, delta_t: float = DEFAULT_DELTA_T) -> np.ndarray:
    """Reference for MATLAB `uh_1_half.m`."""
    delay = _normalize_delay(float(d_base) / float(delta_t))
    num_steps = ceil(delay)
    sh_curve = np.zeros(num_steps + 1, dtype=np.float64)
    weights = np.zeros(num_steps, dtype=np.float64)
    for t_value in range(1, num_steps + 1):
        if t_value < delay:
            sh_curve[t_value] = (t_value / delay) ** (5.0 / 2.0)
        else:
            sh_curve[t_value] = 1.0
        weights[t_value - 1] = sh_curve[t_value] - sh_curve[t_value - 1]
    return weights


def build_uh_2_full_numpy(d_base: float, delta_t: float = DEFAULT_DELTA_T) -> np.ndarray:
    """Reference for MATLAB `uh_2_full.m`."""
    delay = float(d_base) / float(delta_t)
    num_steps = 2 * ceil(delay)
    sh_curve = np.zeros(num_steps + 1, dtype=np.float64)
    weights = np.zeros(num_steps, dtype=np.float64)
    for t_value in range(1, num_steps + 1):
        if t_value <= delay:
            sh_curve[t_value] = 0.5 * (t_value / delay) ** (5.0 / 2.0)
        elif t_value < 2.0 * delay:
            sh_curve[t_value] = 1.0 - 0.5 * (2.0 - t_value / delay) ** (5.0 / 2.0)
        else:
            sh_curve[t_value] = 1.0
        weights[t_value - 1] = sh_curve[t_value] - sh_curve[t_value - 1]
    return weights


def build_uh_3_half_numpy(d_base: float, delta_t: float = DEFAULT_DELTA_T) -> np.ndarray:
    """Reference for MATLAB `uh_3_half.m`."""
    delay = _normalize_delay(float(d_base) / float(delta_t))
    num_steps = ceil(delay)
    fraction_flow = 1.0 / (0.5 * delay**2)
    weights = np.zeros(num_steps, dtype=np.float64)
    for t_value in range(1, num_steps + 1):
        if t_value <= delay:
            weights[t_value - 1] = fraction_flow * (
                0.5 * t_value**2 - 0.5 * (t_value - 1.0) ** 2
            )
        else:
            weights[t_value - 1] = fraction_flow * (
                0.5 * delay**2 - 0.5 * (t_value - 1.0) ** 2
            )
    return weights


def _tri4_s_curve(time_value: float, delay: float) -> float:
    if time_value <= 0.0:
        return 0.0
    midpoint = 0.5 * delay
    if time_value <= midpoint:
        return 2.0 * (time_value / delay) ** 2
    if time_value < delay:
        return 1.0 - 2.0 * (1.0 - time_value / delay) ** 2
    return 1.0


def build_uh_4_full_numpy(d_base: float, delta_t: float = DEFAULT_DELTA_T) -> np.ndarray:
    """Reference for MATLAB `uh_4_full.m`."""
    delay = _normalize_delay(float(d_base) / float(delta_t))
    num_steps = ceil(delay)
    weights = np.zeros(num_steps, dtype=np.float64)
    for t_value in range(1, num_steps + 1):
        upper = _tri4_s_curve(float(t_value), delay)
        lower = _tri4_s_curve(float(t_value - 1), delay)
        weights[t_value - 1] = upper - lower

    diff = 1.0 - weights.sum()
    if weights.sum() > 0.0:
        weights = weights + weights / weights.sum() * diff
    return weights


def build_uh_5_half_numpy(d_base: float, delta_t: float = DEFAULT_DELTA_T) -> np.ndarray:
    """Reference for MATLAB `uh_5_half.m`."""
    delay = _normalize_delay(float(d_base) / float(delta_t))
    num_steps = ceil(delay)
    step_size = 7.0 / delay
    weights = np.zeros(num_steps, dtype=np.float64)
    for t_value in range(1, num_steps + 1):
        lower = (t_value - 1.0) * step_size
        if t_value < num_steps:
            upper = t_value * step_size
            weights[t_value - 1] = exp(-lower) - exp(-upper)
        else:
            # MATLAB appends the missing <7, inf> tail to the last active bin.
            weights[t_value - 1] = exp(-lower)
    return weights


def build_uh_6_gamma_numpy(
    shape_n: float,
    scale_k: float,
    delta_t: float = DEFAULT_DELTA_T,
) -> np.ndarray:
    """Reference for MATLAB `uh_6_gamma.m`."""
    weights: list[float] = []
    t_value = 1
    while True:
        lower = (t_value - 1.0) * float(delta_t)
        upper = t_value * float(delta_t)
        integral = _regularized_gamma_p_scalar(shape_n, upper / scale_k) - _regularized_gamma_p_scalar(
            shape_n, lower / scale_k
        )
        weights.append(float(integral))
        if integral < max(weights) * 0.001:
            break
        t_value += 1

    array = np.asarray(weights, dtype=np.float64)
    excess = 1.0 - array.sum()
    if array.sum() > 0.0:
        array = array + array / array.sum() * excess
    return array


def build_uh_7_uniform_numpy(d_base: float, delta_t: float = DEFAULT_DELTA_T) -> np.ndarray:
    """Reference for MATLAB `uh_7_uniform.m`."""
    delay = float(d_base) / float(delta_t)
    num_steps = ceil(delay)
    fraction_flow = 1.0 / delay
    weights = np.zeros(num_steps, dtype=np.float64)
    for t_value in range(1, num_steps + 1):
        if t_value < delay:
            weights[t_value - 1] = fraction_flow
        else:
            weights[t_value - 1] = _matlab_mod(delay, t_value - 1.0) * fraction_flow
    return weights


def build_uh_8_delay_numpy(t_delay: float, delta_t: float = DEFAULT_DELTA_T) -> np.ndarray:
    """Reference for MATLAB `uh_8_delay.m`."""
    delay = float(t_delay) / float(delta_t)
    ord1 = 1.0 - float(t_delay) + floor(float(t_delay))
    ord2 = float(t_delay) - floor(float(t_delay))
    t_start = floor(delay)
    weights = np.zeros(t_start + 2, dtype=np.float64)
    weights[t_start] = ord1
    weights[t_start + 1] = ord2
    return weights


REFERENCE_BUILDERS = {
    "half1": build_uh_1_half_numpy,
    "full2": build_uh_2_full_numpy,
    "tri3": build_uh_3_half_numpy,
    "tri4": build_uh_4_full_numpy,
    "exp5": build_uh_5_half_numpy,
    "gamma6": build_uh_6_gamma_numpy,
    "uniform7": build_uh_7_uniform_numpy,
    "delay8": build_uh_8_delay_numpy,
}


MATLAB_FUNCTION_NAMES = {
    "half1": "uh_1_half",
    "full2": "uh_2_full",
    "tri3": "uh_3_half",
    "tri4": "uh_4_full",
    "exp5": "uh_5_half",
    "gamma6": "uh_6_gamma",
    "uniform7": "uh_7_uniform",
    "delay8": "uh_8_delay",
}


def build_unit_hydrograph_numpy(
    kind: str,
    params: float | tuple[float, float] | list[float] | np.ndarray,
    delta_t: float = DEFAULT_DELTA_T,
) -> np.ndarray:
    if kind not in REFERENCE_BUILDERS:
        raise KeyError(f"Unsupported unit hydrograph kind: {kind}")

    builder = REFERENCE_BUILDERS[kind]
    if kind == "gamma6":
        shape_n, scale_k = tuple(np.asarray(params, dtype=np.float64).tolist())
        return builder(float(shape_n), float(scale_k), delta_t)

    scalar = float(np.asarray(params, dtype=np.float64).reshape(-1)[0])
    return builder(scalar, delta_t)


def route_step_numpy(flux_in: float, uh_state: np.ndarray) -> float:
    """Reference for MATLAB `route.m`."""
    return float(uh_state[0, 0] * flux_in + uh_state[1, 0])


def update_uh_numpy(uh_state: np.ndarray, flux_in: float) -> np.ndarray:
    """Reference for MATLAB `update_uh.m`."""
    next_state = uh_state.copy()
    next_state[1, :] = next_state[0, :] * flux_in + next_state[1, :]
    next_state[1, :] = np.roll(next_state[1, :], -1)
    next_state[1, -1] = 0.0
    return next_state


def route_with_unit_hydrograph_numpy(
    flux_in: np.ndarray | list[float] | tuple[float, ...],
    weights: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Route a 1D input series with the exact MATLAB state-update logic."""
    series = _as_float64_1d(flux_in)
    kernel = _as_float64_1d(weights)
    uh_state = np.vstack([kernel, np.zeros_like(kernel)])
    flux_out = np.zeros_like(series)
    for time_index, flux_value in enumerate(series):
        flux_out[time_index] = route_step_numpy(float(flux_value), uh_state)
        uh_state = update_uh_numpy(uh_state, float(flux_value))
    return flux_out


def route_dual_unit_hydrograph_numpy(
    flux_in: np.ndarray | list[float] | tuple[float, ...],
    first_weights: np.ndarray | list[float] | tuple[float, ...],
    second_weights: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = route_with_unit_hydrograph_numpy(flux_in, first_weights)
    second = route_with_unit_hydrograph_numpy(flux_in, second_weights)
    return first, second, first + second


def impulse_response_numpy(
    kind: str,
    params: float | tuple[float, float] | list[float] | np.ndarray,
    sequence_length: int | None = None,
    delta_t: float = DEFAULT_DELTA_T,
) -> np.ndarray:
    weights = build_unit_hydrograph_numpy(kind, params, delta_t)
    length = len(weights) if sequence_length is None else int(sequence_length)
    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = 1.0
    return route_with_unit_hydrograph_numpy(impulse, weights)
