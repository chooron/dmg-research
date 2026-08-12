from __future__ import annotations

import numpy as np


def as_array(x):
    return np.asarray(x, dtype=np.float64)


def hard_gate_above(x, threshold):
    return (as_array(x) > as_array(threshold)).astype(np.float64)


def hard_gate_below_or_equal(x, threshold):
    return (as_array(x) <= as_array(threshold)).astype(np.float64)


def smooth_storage_gate_above_hard(S, threshold):
    # Hard interpretation of the dMoT storage-above soft gate.
    return hard_gate_above(S, threshold)


def smooth_temperature_snow_hard(T, threshold):
    # MARRMoT snowfall_1.m/rainfall_1.m use smoothThreshold_temperature_logistic.
    return hard_gate_below_or_equal(T, threshold)


def snowfall_1_hard(T, threshold, incoming_flux=10.0):
    # MARRMoT Flux files/snowfall_1.m lines 19-22, hard-threshold equivalent.
    return incoming_flux * smooth_temperature_snow_hard(T, threshold)


def rainfall_1_hard(T, threshold, incoming_flux=10.0):
    # MARRMoT Flux files/rainfall_1.m lines 46-49, hard-threshold equivalent.
    return incoming_flux * (1.0 - smooth_temperature_snow_hard(T, threshold))


def melt_3_hard(S2, threshold, p1=3.0, p2=0.0, T=3.0, S1=20.0):
    # MARRMoT Flux files/melt_3.m lines 25-29, hard low-snow equivalent.
    melt_actual = np.minimum(np.maximum(p1 * (T - p2), 0.0), S1)
    return melt_actual * hard_gate_below_or_equal(S2, threshold)


def saturation_1_hard(S, Smax, incoming_flux=10.0):
    # MARRMoT Flux files/saturation_1.m lines 73-78, hard overflow equivalent.
    return incoming_flux * hard_gate_above(S, Smax)


def saturation_9_hard(S, threshold, incoming_flux=10.0):
    # MARRMoT Flux files/saturation_9.m lines 104-109, hard deficit-store equivalent.
    return incoming_flux * hard_gate_below_or_equal(S, threshold)


def saturation_11_hard(S, Smin, Smax=100.0, p1=1.5, p2=2.0, incoming_flux=10.0):
    # MARRMoT Flux files/saturation_11.m lines 26-32, hard above-Smin equivalent.
    ratio = np.maximum(S - Smin, 0.0) / (Smax - Smin)
    term = np.minimum(1.0, p1 * np.power(np.maximum(ratio, 0.0), p2))
    return incoming_flux * term * hard_gate_above(S, Smin)


def evap_14_hard(S2, S2min, p1=0.7, p2=2.0, Ep=8.0, S1=5.0):
    # MARRMoT Flux files/evap_14.m line 136, hard below-threshold equivalent.
    evap = np.minimum((p1**p2) * Ep, S1)
    return evap * hard_gate_below_or_equal(S2, S2min)


def evap_16_hard(S2, S2min, p1=0.7, Ep=8.0, S1=1.0e6):
    # MARRMoT Flux files/evap_16.m line 160, hard below-threshold equivalent.
    evap = p1 * Ep
    return np.minimum(evap * hard_gate_below_or_equal(S2, S2min), S1)


def interflow_11_hard(S, threshold, p1=5.0):
    # MARRMoT Flux files/interflow_11.m lines 226-231, hard above-threshold equivalent.
    return np.minimum(p1, np.maximum(S - threshold, 0.0)) * hard_gate_above(S, threshold)


def interflow_12_hard(S, fc_fraction=0.4, p1=0.3, p3=1.5, Smax=100.0):
    # MARRMoT Flux files/interflow_12.m line 259.
    fc = fc_fraction * Smax
    excess = np.maximum(S - fc, 0.0)
    return hard_gate_above(S, fc) * np.minimum(p1 * np.power(excess, p3), np.maximum(S, 0.0))


def baseflow_6_hard(S, threshold, p1=0.01):
    # MARRMoT Flux files/baseflow_6.m line 181, hard above-threshold equivalent.
    return np.minimum(S, p1 * np.power(S, 2.0)) * hard_gate_above(S, threshold)


def baseflow_9_hard(S, threshold, p1=0.2):
    # MARRMoT Flux files/baseflow_9.m line 202.
    return p1 * np.maximum(0.0, S - threshold)


def phenology_1_hard(T, p1=-5.0, p2=5.0, Ep=8.0):
    # MARRMoT Flux files/phenology_1.m line 306.
    return np.minimum(1.0, np.maximum(0.0, (T - p1) / (p2 - p1))) * Ep


def interception_4_hard(t, p1=0.4, p2=183.0, tmax=365.25, incoming_flux=10.0):
    # MARRMoT Flux files/interception_4.m line 283.
    fraction = p1 + (1.0 - p1) * np.cos(2.0 * np.pi * (t - p2) / tmax)
    return np.maximum(0.0, fraction) * incoming_flux
