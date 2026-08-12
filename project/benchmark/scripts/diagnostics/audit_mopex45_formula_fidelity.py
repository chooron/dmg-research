#!/usr/bin/env python3
"""Stage 1 — MOPEX4/5 formula fidelity audit: numeric comparison of every
flux helper against the MARRMoT MATLAB reference (MARRMoT/Models/Flux files).

Produces formulation_fidelity_matrix.csv and feeds the report."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]
OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "formulation_degeneracy_audit"
OUT.mkdir(parents=True, exist_ok=True)

from dmotpy.models.flux.mopex import (  # noqa: E402
    mopex_evap_7, mopex_saturation_1, mopex_baseflow_1, mopex_recharge_3,
    mopex_snowfall_1, mopex_rainfall_1, mopex_melt_1, mopex_interception_4,
    mopex_phenology_1,
)
from dmotpy.models.flux.mopex import mopex_training_context  # noqa: E402


def matlab_saturation_1(In, S, Smax, r=0.01, e=5.0):
    Smax = torch.clamp(Smax, min=0.0)
    out = 1.0 / (1.0 + torch.exp((S - Smax + r * e * Smax) / (r * Smax)))
    return In * (1.0 - out)


def matlab_evap_7(S, Smax, Ep, dt=1.0):
    return torch.minimum(S / Smax * Ep, S / dt)


def matlab_melt_1(p1, p2, T, S, dt=1.0):
    return torch.clamp(torch.minimum(p1 * (T - p2), S / dt), min=0.0)


def matlab_snowfall_1(In, T, p1):
    return In / (1.0 + torch.exp((T - p1) / 0.01))


def matlab_rainfall_1(In, T, p1):
    return In * (1.0 - 1.0 / (1.0 + torch.exp((T - p1) / 0.01)))


def matlab_interception_4(p1, p2, t, tmax, In):
    return torch.clamp(p1 + (1 - p1) * torch.cos(2 * torch.pi * (t - p2) / tmax), min=0.0) * In


def matlab_phenology_1(T, p1, p2, Ep):
    return torch.clamp((T - p1) / (p2 - p1), 0.0, 1.0) * Ep


def matlab_recharge_3(p1, S):
    return p1 * S


def matlab_baseflow_1(p1, S):
    return p1 * S


def compare(name, py_fn, ml_fn, args, sweep_idx, sweep_min, sweep_max,
            bounds_desc="", semantics="", n=40):
    """Sweep the tensor at args[sweep_idx] over [min,max] and compare py vs ml."""
    grid = torch.linspace(sweep_min, sweep_max, n)
    max_diff, mean_diff, rel_max = 0.0, 0.0, 0.0
    with mopex_training_context(lambda_i=1.0, lambda_p=1.0, beta=50.0):
        for g in grid:
            a = [v.clone() if torch.is_tensor(v) else v for v in args]
            a[sweep_idx] = g
            py = py_fn(*a)
            ml = ml_fn(*a)
            d = (py - ml).abs()
            max_diff = max(max_diff, float(d.max()))
            mean_diff += float(d.mean())
            denom = float(ml.abs().max()) + 1e-9
            rel_max = max(rel_max, float(d.max()) / denom)
    mean_diff /= n
    return {"flux": name, "max_abs_diff": round(max_diff, 6), "mean_abs_diff": round(mean_diff, 6),
            "max_rel_diff": round(rel_max, 4), "bounds": bounds_desc, "semantics": semantics,
            "swept_variable": bounds_desc}


def main():
    rows = []
    Smax = torch.tensor([50.0]); Ep = torch.tensor([3.0]); T = torch.tensor([5.0])
    P = torch.tensor([5.0]); S = torch.tensor([25.0])

    rows.append(compare("saturation_1", mopex_saturation_1, matlab_saturation_1,
                        [P, S, Smax], 1, 0.0, 100.0, "S in [0,2*Smax]",
                        "excess-runoff fraction g(S)"))
    rows.append(compare("evap_7", mopex_evap_7, matlab_evap_7,
                        [S, Smax, Ep], 0, 0.0, 100.0, "S in [0,2*Smax]",
                        "soil evap, PET-limited"))
    rows.append(compare("melt_1", mopex_melt_1, matlab_melt_1,
                        [torch.tensor([2.0]), torch.tensor([0.0]), T, torch.tensor([30.0])],
                        2, -5.0, 10.0, "T in [-5,10]", "degree-day melt"))
    rows.append(compare("snowfall_1", mopex_snowfall_1, matlab_snowfall_1,
                        [P, T, torch.tensor([0.0])], 1, -5.0, 10.0, "T in [-5,10]",
                        "snow partition"))
    rows.append(compare("rainfall_1", mopex_rainfall_1, matlab_rainfall_1,
                        [P, T, torch.tensor([0.0])], 1, -5.0, 10.0, "T in [-5,10]",
                        "rain partition"))
    # interception_4: py(mopex) = (flux_pr, doy, alpha, is_time); ml(marrmot) = (alpha, is_time, t, tmax, In)
    def ic_ml(pr, doy, alpha, is_time):
        return matlab_interception_4(alpha, is_time, doy, 365.25, pr)
    rows.append(compare("interception_4", mopex_interception_4, ic_ml,
                        [P, torch.tensor([180.0]), torch.tensor([0.5]), torch.tensor([365.25])],
                        1, 1.0, 365.0, "doy in [1,365], alpha=0.5",
                        "seasonal interception of Pr"))
    rows.append(compare("phenology_1", mopex_phenology_1, matlab_phenology_1,
                        [T, torch.tensor([0.0]), torch.tensor([10.0]), Ep],
                        0, -5.0, 15.0, "T in [-5,15], tmin=0, trange=10",
                        "GSI ramp on PET"))
    rows.append(compare("recharge_3", mopex_recharge_3, matlab_recharge_3,
                        [torch.tensor([0.3]), S], 1, 0.0, 100.0, "S in [0,100]",
                        "linear leakage tw*S"))
    rows.append(compare("baseflow_1", mopex_baseflow_1, matlab_baseflow_1,
                        [torch.tensor([0.3]), S], 1, 0.0, 100.0, "S in [0,100]",
                        "linear baseflow k*S"))

    with (OUT / "formulation_fidelity_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["flux", "max_abs_diff", "mean_abs_diff", "max_rel_diff",
                                          "swept_variable", "semantics"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
