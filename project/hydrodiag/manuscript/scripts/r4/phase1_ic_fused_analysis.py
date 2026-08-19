"""R4 IC fused sensitivity analysis (observation-trained, 5 starts x 200 gen).

Repeats the minimal R4 phase-1 analysis with the fused IC Base/CN runs
(``ic_cmaes_recalibration_p20_p25_20260728_fused``).  These runs are
observation-trained but follow a DIFFERENT protocol than the R1 canonical IC
(10 starts x 300 generations, paired_v2); every output is tagged
``IC_FUSED_5x200_SENSITIVITY`` and must never be mixed with canonical IC or
canonical dPL tables.

Analysis A (CN snow-state consistency) and Analysis B (Delta(CN-Base) vs
Snow-17 SWE burden) reuse the exact metric helpers from the canonical dPL
phase-1 script.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/home/jingxin/orca/workspaces/dmg-research/hydrodiag-R4-exp/project/hydrodiag")
sys.path.insert(0, str(PROJECT))

from manuscript.scripts.r4.common import default_data_root, default_results_root, load_bundle, zfill8  # noqa: E402
from manuscript.scripts.r4.forward_export import export_run  # noqa: E402
from manuscript.scripts.r4.input_adapters import read_ic_fused  # noqa: E402
from manuscript.scripts.r4.phase1_dpl_analysis import (  # noqa: E402
    MIN_ACTIVE_DAYS, consistency_for_basin, timing_metrics_for_basin, theil_sen,
)

IC_FUSED_TAG = "IC_FUSED_5x200_SENSITIVITY"
IC_FUSED_RUN = "ic_cmaes_recalibration_p20_p25_20260728_fused"
SWE_REF_DIR = default_results_root() / "r4_swe_reference_v1"
OUT_DIR = default_results_root() / "r4_phase1_ic_fused_sensitivity"


def export_fused_models(results_root: Path, data_root: Path, device: str) -> None:
    from ablation.ic_core.parameter_adapter import get_parameter_spec

    bundle = load_bundle(data_root)
    basin_ids = [zfill8(b) for b in bundle.basin_ids]
    for model in ("XAJ", "XAJ_CN"):
        run_id = f"ic_fused_{model}"
        out_dir = results_root / f"r4_{run_id}"
        if (out_dir / f"{run_id}_full_arrays.npz").exists():
            print(f"exists, skip: {run_id}")
            continue
        csv = results_root / IC_FUSED_RUN / model / "per_start.csv"
        names = tuple(get_parameter_spec(model))
        params, meta = read_ic_fused(csv, model, names, basin_ids)
        manifest = export_run(
            structure=model, parameters=params, parameter_meta=meta,
            basin_ids=basin_ids, data_root=data_root, results_root=results_root,
            run_id=run_id, tag=IC_FUSED_TAG,
            provenance={"source_run": f"{IC_FUSED_RUN}/{model}",
                        "protocol": "fused IC CMA-ES, 5 starts x 200 generations, observed discharge",
                        "note": "NOT canonical IC (10x300 paired_v2); sensitivity only"},
            device=device, batch=64, validate_subset=8, save_npz=True,
        )
        print(f"exported {run_id}: q_finite={manifest['q_finite']}")


def main() -> None:
    import argparse

    import torch

    from training.dpl.run_dpl_model import compute_kge_fp64

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()

    results_root = args.results_root or default_results_root()
    data_root = args.data_root or default_data_root()
    bundle = load_bundle(data_root)
    basin_ids = [zfill8(b) for b in bundle.basin_ids]
    test_sl = slice(5478, 10957)
    test_dates = bundle.dates[test_sl].astype("datetime64[D]")

    if not args.skip_export:
        export_fused_models(results_root, data_root, args.device)

    ref = np.load(SWE_REF_DIR / "swe_ensemble.npz")
    if list(ref["basin_ids"]) != basin_ids:
        raise ValueError("SWE reference basin order differs from bundle order")
    swe_test = ref["swe_median"][:, test_sl]
    burden = pd.read_csv(SWE_REF_DIR / "swe_basin_burden_test.csv", dtype={"basin_id": str}).set_index("basin_id")
    obs_test = bundle.target_mm_day[:, test_sl]

    def arrays(model: str) -> np.ndarray:
        run_id = f"ic_fused_{model}"
        return np.load(results_root / f"r4_{run_id}" / f"{run_id}_full_arrays.npz")

    q = {model: arrays(model)["q_full"][:, test_sl] for model in ("XAJ", "XAJ_CN")}
    g_cn = arrays("XAJ_CN")["G"][:, test_sl]

    kge = {model: np.array([compute_kge_fp64(q[model][i], obs_test[i])
                            for i in range(len(basin_ids))]) for model in ("XAJ", "XAJ_CN")}

    # ---- analysis A --------------------------------------------------------
    rows_a = []
    for i, b in enumerate(basin_ids):
        cons = consistency_for_basin(test_dates, g_cn[i], swe_test[i])
        tim = timing_metrics_for_basin(test_dates, g_cn[i], swe_test[i])
        rows_a.append({"basin_id": b, **cons, **tim})
    df_a = pd.DataFrame(rows_a)

    # ---- analysis B ---------------------------------------------------------
    delta = kge["XAJ_CN"] - kge["XAJ"]
    rows_b = []
    for i, b in enumerate(basin_ids):
        rows_b.append({
            "basin_id": b,
            "kge_cn_test": kge["XAJ_CN"][i], "kge_base_test": kge["XAJ"][i],
            "delta_cn_minus_base": delta[i],
            "median_annual_max_swe_mm": burden.loc[b, "median_annual_max_swe_mm"],
            "median_swe_positive_days": burden.loc[b, "median_swe_positive_days"],
            "frac_snow": bundle.raw_attributes[i, 3],
        })
    df_b = pd.DataFrame(rows_b)

    summary_a = {
        "n_basins": int(len(df_a)),
        "n_with_consistency": int((df_a["n_active_days"] >= MIN_ACTIVE_DAYS).sum()),
        "median_seasonal_corr": float(df_a["seasonal_corr"].median()),
        "median_anomaly_corr": float(df_a["anomaly_corr"].median()),
        "median_nrmse": float(df_a["nrmse"].median()),
        "median_peak_timing_error_days": float(df_a["peak_timing_error_days"].median()),
        "median_depletion_timing_error_days": float(df_a["depletion_timing_error_days"].median()),
    }
    active_b = burden[burden["median_annual_max_swe_mm"] >= 20].index
    sub = df_a[df_a["basin_id"].isin(active_b)]
    summary_a["snow_active_only"] = {
        "n_basins": int(len(sub)),
        "median_seasonal_corr": float(sub["seasonal_corr"].median()),
        "median_anomaly_corr": float(sub["anomaly_corr"].median()),
        "median_peak_timing_error_days": float(sub["peak_timing_error_days"].median()),
    }

    summary_b = {"n_basins": int(len(df_b))}
    for axis in ("median_annual_max_swe_mm", "median_swe_positive_days", "frac_snow"):
        summary_b[axis] = theil_sen(df_b[axis].to_numpy(), df_b["delta_cn_minus_base"].to_numpy())
    suba = df_b[df_b["median_annual_max_swe_mm"] >= 20]
    summary_b["median_annual_max_swe_mm_snow_active_only"] = theil_sen(
        suba["median_annual_max_swe_mm"].to_numpy(), suba["delta_cn_minus_base"].to_numpy())

    report = {
        "tag": IC_FUSED_TAG,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "fused IC Base/CN (5 starts x 200 gen, observed discharge) — SENSITIVITY ONLY, not canonical IC",
        "test_period": "1995-10-01..2010-09-30",
        "kge_test_levels": {"base_median": float(np.nanmedian(kge["XAJ"])),
                            "cn_median": float(np.nanmedian(kge["XAJ_CN"]))},
        "analysis_A_consistency": summary_a,
        "analysis_B_burden_gradient": summary_b,
        "direction_consistency_with_dpl": "compare with r4_phase1_dpl_official report",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "r4_phase1_ic_fused_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    df_a.to_csv(OUT_DIR / "snow_consistency_basin.csv", index=False)
    df_b.to_csv(OUT_DIR / "burden_gradient_basin.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
