#!/usr/bin/env python3
"""Stage 0 (metric provenance) + Stage 1 (formula fidelity) artifacts.

Writes metric_provenance_table.csv, formulation_fidelity_matrix.csv
(detailed), and feeds the final report."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[2]
OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "formulation_degeneracy_audit"
OUT.mkdir(parents=True, exist_ok=True)

PROVENANCE = [
    {"metric": "0.7828", "source": "project/benchmark/results/mopex45_phase_fix/circular_stage/00_protocol_and_oracle.csv",
     "basin": "11532500 (single basin)", "basin_count": 1, "parameter_source": "IC/CMA-ES archive (best of 5 starts)",
     "period": "representative-window protocol of circular_stage", "aggregation": "single-basin KGE",
     "objective": "KGE (compute_differentiable_kge)", "model": "mopex4",
     "continuation_context": "production endpoint lambda_i=1, lambda_p=1, beta=50"},
    {"metric": "0.6510", "source": "project/benchmark/results/dpl_round13_20260805/final/dpl_vs_ic_comparison_summary.md",
     "basin": "531 CAMELS basins", "basin_count": 531, "parameter_source": "IC/CMA-ES archive (full-library evaluation)",
     "period": "official IC evaluation protocol (full300)", "aggregation": "median over 531 basins",
     "objective": "KGE", "model": "mopex4", "continuation_context": "n/a (IC evaluation)"},
    {"metric": "0.5306", "source": "MOPEX4 continuation 3-seed mean (full_continuation seeds 41/42/43)",
     "basin": "531 basins", "basin_count": 531, "parameter_source": "dPL continuation network (seed41/42/43)",
     "period": "train 1980-1995 / valid 1995-2010", "aggregation": "median over 531 basins (per-seed), then mean",
     "objective": "KGE", "model": "mopex4", "continuation_context": "endpoint lambda_i=1, beta=50"},
]

FIDELITY = [
    {"flux": "snowfall_1", "python": "P*sigmoid((tcrit-T)/(|tcrit|*0.01+0.01))",
     "marrmot": "P/(1+exp((T-tcrit)/0.01))", "max_abs_diff": 0.3552,
     "note": "smoothing width depends on |tcrit| in Python; MARRMoT fixed 0.01 degC", "verdict": "MISMATCH"},
    {"flux": "rainfall_1", "python": "P*sigmoid((T-tcrit)/(|tcrit|*0.01+0.01))",
     "marrmot": "P*(1-1/(1+exp((T-tcrit)/0.01)))", "max_abs_diff": 0.3552,
     "note": "same width issue as snowfall_1", "verdict": "MISMATCH"},
    {"flux": "melt_1", "python": "min(ddf*sigmoid(T-tcrit)*softplus(T-tcrit), Sn)",
     "marrmot": "max(min(ddf*(T-tcrit), S/dt), 0)", "max_abs_diff": 0.6931,
     "note": "Python: smooth drive; MARRMoT: linear above tcrit, hard zero below", "verdict": "SMOOTHING_DIFF"},
    {"flux": "evap_7", "python": "min(Ep*clamp(S/(Smax+eps),max=1)*dt, S)",
     "marrmot": "min(S/Smax*Ep, S/dt)", "max_abs_diff": 3.0,
     "note": "Python caps at PET when S>Smax; MARRMoT allows ET>S/Smax*Ep to exceed PET", "verdict": "MISMATCH"},
    {"flux": "interception_4", "python": "min(softplus(fraction*beta)/beta,1)*Pr*lambda_i",
     "marrmot": "max(0, alpha+(1-alpha)*cos(2pi(t-is_time)/365.25))*Pr", "max_abs_diff": 0.066,
     "note": "Python softplus(beta=50) smooths the max(0,.) kink; acts on Pr (liquid rain) in both", "verdict": "SMOOTHING_DIFF"},
    {"flux": "saturation_1", "python": "P*sigmoid((S - Smax*(1-r)) / (Smax*r*e + eps))",
     "marrmot": "P*(1 - 1/(1+exp((S - Smax + r*e*Smax)/(r*Smax))))", "max_abs_diff": 2.4855,
     "note": "different centre (0.99 vs 0.95 Smax) and width (0.05 vs 0.01 Smax); large effect on q1f/q2f",
     "verdict": "MISMATCH"},
    {"flux": "recharge_3", "python": "min(tw*S, S)", "marrmot": "tw*S", "max_abs_diff": 0.0,
     "note": "safety cap redundant for tw in [0,1]", "verdict": "MATCH"},
    {"flux": "baseflow_1", "python": "min(k*S, S)", "marrmot": "k*S", "max_abs_diff": 0.0,
     "note": "safety cap redundant for k in [0,1]", "verdict": "MATCH"},
    {"flux": "phenology_1", "python": "clamp((T-tmin)/clamp(trange,min=eps),0,1)*PET",
     "marrmot": "min(1,max(0,(T-tmin)/(trange)))*Ep", "max_abs_diff": 0.0,
     "note": "identical (hard clamp)", "verdict": "MATCH"},
    {"flux": "soil ODE ordering", "python": "sequential explicit within-step updates (ET1->I->q1f->qw)",
     "marrmot": "simultaneous ODE (all fluxes from current state)", "max_abs_diff": None,
     "note": "discretization difference; audited separately (sequential_discretization_audit)", "verdict": "DISCRETIZATION_DIFF"},
    {"flux": "MOPEX5 delta vs MOPEX4", "python": "ET1/ET2 use PET_epc=phenology_1(T,tmin,trange,PET)",
     "marrmot": "flux_epc=phenology_1(T,tmin,tmin+trange,Ep); ET1/ET2 use it", "max_abs_diff": 0.0,
     "note": "MOPEX5 adds only phenology relative to MOPEX4 in both implementations", "verdict": "MATCH"},
]


def main():
    with (OUT / "metric_provenance_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PROVENANCE[0]))
        w.writeheader(); w.writerows(PROVENANCE)
    with (OUT / "formulation_fidelity_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["flux", "python", "marrmot", "max_abs_diff", "note", "verdict"])
        w.writeheader(); w.writerows(FIDELITY)
    summary = {
        "metric_provenance": {
            "0.7828": "single-basin (11532500) IC KGE from circular_stage oracle protocol",
            "0.6510": "531-basin median IC KGE (official full-library summary)",
            "comparable": "NO",
            "reason": ("0.7828 is one basin (11532500) from the 5-basin circular_stage oracle set; "
                       "0.6510 is the median over all 531 basins. Single-basin vs full-library median "
                       "are different aggregation levels; 0.7828 > median 0.6510 is expected for a "
                       "well-calibrated basin."),
        },
        "formula_fidelity": {
            "critical_mismatches": [
                "saturation_1: different sigmoid centre (0.99 vs 0.95 Smax) and width (0.05 vs 0.01 Smax)",
                "evap_7: Python caps ET at PET for S>Smax; MARRMoT allows ET to exceed PET",
                "snowfall_1/rainfall_1: Python smoothing width scales with |tcrit|; MARRMoT fixed 0.01 degC",
            ],
            "smoothing_differences": [
                "melt_1: sigmoid*softplus drive vs linear+hard-zero",
                "interception_4: softplus(beta=50) vs max(0,.)",
            ],
            "matching": ["recharge_3", "baseflow_1", "phenology_1", "MOPEX5-only-phenology delta"],
            "note": ("saturation_1/evap_7/snowfall differences apply to all MOPEX models 1-5 "
                     "(they share dmotpy/models/flux/mopex.py helpers); the IC and dPL results were "
                     "both produced on this Python model, so the differences are systematic w.r.t. the "
                     "dPL-vs-IC comparison, but they do change the reachable parameter space relative "
                     "to the MARRMoT m_32/m_35 reference."),
        },
    }
    (OUT / "stage01_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
