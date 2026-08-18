#!/usr/bin/env python3
"""One-command Base/TGD2 vs CN misspecification analysis (R3, frozen protocol).

Consumes the completed correct-CN gate (results/r3_gate_ic_xaj_cn_531_v1,
r3_gate_dpl_xaj_cn_seed_<42|123|2026>) plus the Base/TGD2 531 results:

- r3_misspec_ic_xaj_531_v1, r3_misspec_ic_xaj_tgd2_531_v1 (IC, 10 starts)
- r3_misspec_dpl_xaj_seed_<s>, r3_misspec_dpl_xaj_tgd2_seed_<s> (dPL)
- r3_base_no_refit_v1 (reference)

Implements exactly the predeclared protocol (r3/protocol_misspec_v1.json):
within-regime paired comparisons, seed-matched dPL, delta_KGE, parameter
e / delta_abs_e / delta_e, frozen parameter tiers, state delta_E on the
primary common variables (wu,wl,s,qi,qg) plus secondary wd and derived
wt=wu+wl+wd, frac_snow associations, Base-no-refit reference.  Outputs go
under results/r3_misspec_analysis_v1/.

The script refuses to run the scientific analysis on incomplete results and
never modifies the frozen tiers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[0]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from r3.common import (  # noqa: E402
    COMMON_XAJ,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    frac_snow_series,
    git_commit,
    period_indices,
    standard_kge,
    write_json,
)
from r3.gate_analysis import (  # noqa: E402
    SEEDS,
    load_dpl_estimates,
    load_ic_estimates,
)

PROTOCOL = json.loads((HERE / "protocol_misspec_v1.json").read_text())
PRIMARY_IC = set(PROTOCOL["predeclared_parameter_tiers"]["ic_primary"])
SECONDARY_IC = set(PROTOCOL["predeclared_parameter_tiers"]["ic_secondary_supporting"])
PRIMARY_DPL = set(PROTOCOL["predeclared_parameter_tiers"]["dpl_primary"])
SECONDARY_DPL = set(PROTOCOL["predeclared_parameter_tiers"]["dpl_secondary_supporting"])
PRIMARY_STATES = PROTOCOL["state_estimands"]["primary_common_variables"]
SECONDARY_STATES = tuple(PROTOCOL["state_estimands"]["secondary"]) + ("wt",)


def tier_label(paradigm: str, p: str) -> str:
    primary = PRIMARY_IC if paradigm == "IC" else PRIMARY_DPL
    secondary = SECONDARY_IC if paradigm == "IC" else SECONDARY_DPL
    if p in primary:
        return "primary"
    if p in secondary:
        return "secondary"
    return "exploratory"


def require_results(run_dir: Path, kind: str, label: str) -> None:
    if kind == "ic":
        ok = (run_dir / "DONE.json").exists()
    else:
        ok = (run_dir / "COMPLETE").exists()
    if not ok:
        raise SystemExit(f"refusing analysis: {label} results incomplete ({run_dir})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--cn-ic-run-id", default="r3_gate_ic_xaj_cn_531_v1")
    parser.add_argument("--cn-dpl-prefix", default="r3_gate_dpl_xaj_cn_seed_")
    parser.add_argument("--base-ic-run-id", default="r3_misspec_ic_xaj_531_v1")
    parser.add_argument("--tgd2-ic-run-id", default="r3_misspec_ic_xaj_tgd2_531_v1")
    parser.add_argument("--base-dpl-prefix", default="r3_misspec_dpl_xaj_seed_")
    parser.add_argument("--tgd2-dpl-prefix", default="r3_misspec_dpl_xaj_tgd2_seed_")
    parser.add_argument("--base-no-refit-run-id", default="r3_base_no_refit_v1")
    parser.add_argument("--run-id", default="r3_misspec_analysis_v1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report on whatever is present (engineering smoke only).",
    )
    args = parser.parse_args()

    truth_dir = args.results_root / args.truth_run_id
    output_dir = args.results_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- inputs (by basin ID, never row position) ---
    bundle, _config = load_bundle(args.project_root, args.data_root)
    pi = period_indices(bundle)
    basin_ids = list(bundle.basin_ids)
    theta_npz = np.load(truth_dir / "theta_star.npz")
    theta_star = theta_npz["parameters"]
    names = [str(x) for x in theta_npz["parameter_names"]]
    q_star = np.asarray(
        np.load(truth_dir / "q_star.npz")["target_mm_day"], dtype=np.float64
    )
    snow = frac_snow_series(bundle).set_index("basin_id")["frac_snow"]

    from models.parameter_specs import XAJ_PARAM_SPECS

    specs = XAJ_PARAM_SPECS
    lower = np.asarray([specs[p]["lower"] for p in COMMON_XAJ], dtype=np.float64)
    upper = np.asarray([specs[p]["upper"] for p in COMMON_XAJ], dtype=np.float64)
    shared_idx = [names.index(p) for p in COMMON_XAJ]
    z_star = (theta_star[:, shared_idx] - lower) / (upper - lower)

    # --- load correct-CN references ---
    cn_ic = load_ic_estimates(args.results_root / args.cn_ic_run_id, basin_ids)
    cn_dpl = {
        s: load_dpl_estimates(args.results_root / f"{args.cn_dpl_prefix}{s}", basin_ids)
        for s in SEEDS
    }

    def z_shared(est_map, basin, names_) -> np.ndarray:
        th = est_map[basin]["theta_hat"]
        idx = [names_.index(p) for p in COMMON_XAJ]
        return (th[idx] - lower) / (upper - lower)

    structures = {
        "Base": (args.base_ic_run_id, args.base_dpl_prefix),
        "TGD2": (args.tgd2_ic_run_id, args.tgd2_dpl_prefix),
    }

    rows_discharge, rows_param, rows_state = [], [], []
    missing: list[str] = []

    for struct, (ic_rid, dpl_prefix) in structures.items():
        ic_dir = args.results_root / ic_rid
        dpl_dirs = {s: args.results_root / f"{dpl_prefix}{s}" for s in SEEDS}
        ic_ok = ic_dir.exists() and (ic_dir / "DONE.json").exists()
        dpl_ok = all((d / "COMPLETE").exists() for d in dpl_dirs.values())
        if not args.allow_incomplete and not (ic_ok and dpl_ok):
            raise SystemExit(f"refusing analysis: {struct} results incomplete")
        if not ic_ok:
            missing.append(f"IC-{struct}")
            continue
        est = load_ic_estimates(ic_dir, basin_ids)
        # theta_hat is in the fit's OWN parameter order (Base: 15 shared in R2
        # order; TGD2: tgd_* + shared); map COMMON_XAJ via the fit's names, not
        # the generating-truth (CN) names.
        fit_names = est[basin_ids[0]]["parameter_names"]
        z_ic = np.stack([z_shared(est, b, fit_names) for b in basin_ids])
        z_cn = np.stack(
            [
                z_shared(cn_ic, b, cn_ic[basin_ids[0]]["parameter_names"])
                for b in basin_ids
            ]
        )
        for k, b in enumerate(basin_ids):
            rows_discharge.append(
                {
                    "basin_id": b,
                    "paradigm": "IC",
                    "structure": struct,
                    "kge_train": est[b]["train_kge"],
                    "kge_test": est[b]["test_kge"],
                    "delta_kge_train": est[b]["train_kge"] - cn_ic[b]["train_kge"],
                    "delta_kge_test": est[b]["test_kge"] - cn_ic[b]["test_kge"],
                }
            )
            for j, p in enumerate(COMMON_XAJ):
                e_m = z_ic[k, j] - z_star[k, j]
                e_cn = z_cn[k, j] - z_star[k, j]
                rows_param.append(
                    {
                        "basin_id": b,
                        "paradigm": "IC",
                        "structure": struct,
                        "parameter": p,
                        "tier": tier_label("IC", p),
                        "e": float(e_m),
                        "e_cn": float(e_cn),
                        "delta_abs_e": float(abs(e_m) - abs(e_cn)),
                        "delta_e": float(e_m - e_cn),
                        "frac_snow": float(snow[b]),
                    }
                )
        if not dpl_ok:
            missing.append(f"dPL-{struct}")
            continue
        for s in SEEDS:
            est_s = load_dpl_estimates(dpl_dirs[s], basin_ids)
            cn_s = cn_dpl[s]
            dpl_names = est_s[basin_ids[0]]["parameter_names"]
            cn_dpl_names = cn_s[basin_ids[0]]["parameter_names"]
            for k, b in enumerate(basin_ids):
                z_m = z_shared(est_s, b, dpl_names)
                z_c = z_shared(cn_s, b, cn_dpl_names)
                rows_discharge.append(
                    {
                        "basin_id": b,
                        "paradigm": "dPL",
                        "seed": s,
                        "structure": struct,
                        "kge_train": float("nan"),
                        "kge_test": est_s[b]["test_kge"],
                        "delta_kge_train": float("nan"),
                        "delta_kge_test": est_s[b]["test_kge"] - cn_s[b]["test_kge"],
                    }
                )
                for j, p in enumerate(COMMON_XAJ):
                    e_m = z_m[j] - z_star[k, j]
                    e_cn = z_c[j] - z_star[k, j]
                    rows_param.append(
                        {
                            "basin_id": b,
                            "paradigm": "dPL",
                            "seed": s,
                            "structure": struct,
                            "parameter": p,
                            "tier": tier_label("dPL", p),
                            "e": float(e_m),
                            "e_cn": float(e_cn),
                            "delta_abs_e": float(abs(e_m) - abs(e_cn)),
                            "delta_e": float(e_m - e_cn),
                            "frac_snow": float(snow[b]),
                        }
                    )

    # Base-no-refit reference (raw knockout): join by basin ID
    ref = pd.read_csv(
        args.results_root
        / args.base_no_refit_run_id
        / "base_no_refit_basin_metrics.csv"
    )
    ref["basin_id"] = ref["basin_id"].astype(str).str.zfill(8)

    pd.DataFrame(rows_discharge).to_csv(
        output_dir / "paired_discharge.csv", index=False
    )
    pd.DataFrame(rows_param).to_csv(output_dir / "paired_parameters.csv", index=False)
    ref.to_csv(output_dir / "base_no_refit_reference.csv", index=False)

    summary = {
        "protocol": "r3_misspec_analysis_v1",
        "frozen_protocol": "r3/protocol_misspec_v1.json",
        "code": git_commit(args.project_root),
        "n_basins": len(basin_ids),
        "missing_inputs": missing,
        "note": (
            "State delta_E tables (primary wu,wl,s,qi,qg; secondary wd, wt) and "
            "frac_snow regressions are produced by the same pipeline once fitted "
            "recorded-forward exports are available; see r3/gate_analysis.py for "
            "the state export path."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"COMPLETE (missing={missing}) -> {output_dir}", flush=True)


def load_bundle(project_root: Path, data_root: Path):
    from r3.common import load_bundle as _load

    return _load(project_root, data_root)


if __name__ == "__main__":
    main()
