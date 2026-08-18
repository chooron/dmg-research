"""R4 phase-1 official analysis: canonical dPL Base/CN vs CAMELS Snow-17 SWE.

Trained parameters are read ONLY from observation-trained R1/R2 artifacts
(dpl_camels_531_lite_v2 seeds 42/123); R3 synthetic-q* checkpoints are never
used here.  Every output is tagged OFFICIAL_DPL_OBSERVATION_TRAINED.

Analysis A — CN snow-state consistency (test period, continuous forward):
  per-basin daily G vs SWE reference: seasonal correlation, monthly-anomaly
  correlation, NRMSE = RMSE/(std(SWE)+eps), peak-timing error, and
  peak->50%-depletion timing error (water-year metrics, validity rules below).

Analysis B — snow burden x structural benefit:
  Delta(CN-Base) = KGE_CN - KGE_Base (test period, continuous-forward KGE vs
  observed discharge) per seed; Spearman + Theil-Sen association with
  Snow-17-derived basin burden (median annual max SWE, SWE-positive days);
  frac_snow retained as the R1/R2 reference axis for comparison.

Validity rules (basin-year level):
  - a water year is snow-active for timing metrics iff SWE annual max >= 5 mm
    AND the CN G annual max >= 0.1 mm;
  - depletion timing requires post-peak SWE to drop below 50% of the annual
    peak (permanent-snowpack years are excluded from depletion metrics);
  - basin-level timing summaries require >= 5 valid years in the test period.

Reference semantics: Snow-17 SWE is an external process-state consistency
reference (same Daymet forcing family), NOT observed snow truth; SWE
depletion is not treated as observed snowmelt.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from manuscript.r4 import OFFICIAL_OBSERVATION_TRAINED  # noqa: E402
from manuscript.r4.common import (  # noqa: E402
    default_data_root,
    default_results_root,
    load_bundle,
    zfill8,
)
from manuscript.r4.forward_export import export_run  # noqa: E402
from manuscript.r4.input_adapters import read_dpl_seed  # noqa: E402

DPL_RUN = "dpl_camels_531_lite_v2"
DPL_SEEDS = (42, 123)
SWE_REF_DIR = default_results_root() / "r4_swe_reference_v1"
OUT_DIR = default_results_root() / "r4_phase1_dpl_official"
TAG = OFFICIAL_OBSERVATION_TRAINED

# metric thresholds
MIN_SWE_PEAK_MM = 5.0
MIN_G_PEAK_MM = 0.1
MIN_VALID_YEARS = 5
MIN_ACTIVE_DAYS = 60


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def export_models(
    results_root: Path, data_root: Path, device: str, *, force: bool = False
) -> None:
    """Continuous full-axis export for dPL Base/CN, seeds 42/123."""
    bundle = load_bundle(data_root)
    basin_ids = [zfill8(b) for b in bundle.basin_ids]
    for model in ("XAJ", "XAJ_CN"):
        for seed in DPL_SEEDS:
            run_id = f"official_dpl_{model}_seed{seed}"
            out_dir = results_root / f"r4_{run_id}"
            if (out_dir / f"{run_id}_full_arrays.npz").exists() and not force:
                print(f"exists, skip: {run_id}")
                continue
            seed_dir = results_root / DPL_RUN / model / f"seed_{seed}"
            params, meta = read_dpl_seed(seed_dir, model, data_root, basin_ids)
            manifest = export_run(
                structure=model,
                parameters=params,
                parameter_meta=meta,
                basin_ids=basin_ids,
                data_root=data_root,
                results_root=results_root,
                run_id=run_id,
                tag=TAG,
                provenance={
                    "source_run": f"{DPL_RUN}/{model}/seed_{seed}",
                    "seed": seed,
                    "trained_on": "observed discharge (R1/R2)",
                },
                device=device,
                batch=64,
                validate_subset=8,
                save_npz=True,
            )
            print(
                f"exported {run_id}: q_finite={manifest['q_finite']} arrays={manifest['arrays']}"
            )


# ---------------------------------------------------------------------------
# analysis A helpers
# ---------------------------------------------------------------------------


def load_run_arrays(results_root: Path, model: str, seed: int) -> dict[str, np.ndarray]:
    run_id = f"official_dpl_{model}_seed{seed}"
    out_dir = results_root / f"r4_{run_id}"
    npz = np.load(out_dir / f"{run_id}_full_arrays.npz")
    return {k: npz[k] for k in npz.files}


def water_year_index(dates: np.ndarray) -> np.ndarray:
    d = pd.to_datetime(dates)
    return np.where(d.month >= 10, d.year + 1, d.year).astype(int)


def wy_doy(dates: np.ndarray, wy: np.ndarray) -> np.ndarray:
    d = pd.to_datetime(dates)
    starts = np.array([np.datetime64(f"{int(w) - 1}-10-01", "D") for w in wy])
    return ((d.values - starts) / np.timedelta64(1, "D")).astype(float) + 1


def timing_metrics_for_basin(
    dates: np.ndarray,
    g: np.ndarray,
    swe: np.ndarray,
) -> dict[str, float]:
    """Per-basin test-period timing metrics (median over valid water years)."""
    wy = water_year_index(dates)
    doy = wy_doy(dates, wy)
    rows = []
    for w in np.unique(wy):
        m = wy == w
        gw, sw, dw = g[m], swe[m], doy[m]
        if len(gw) < 300:
            continue
        s_peak_i = int(np.nanargmax(sw))
        s_peak = sw[s_peak_i]
        if s_peak < MIN_SWE_PEAK_MM:
            continue  # weak-snow year: timing undefined
        g_peak_i = int(np.nanargmax(gw))
        g_peak = gw[g_peak_i]
        if g_peak < MIN_G_PEAK_MM:
            continue  # model sees no snow this year: timing undefined
        peak_err = float(doy[wy == w][g_peak_i] - doy[wy == w][s_peak_i])

        # depletion: last day >= 50% of annual peak after the peak day
        def depletion(x, peak_i, thr_frac):
            thr = thr_frac * x[peak_i]
            post = np.where(x >= thr)[0]
            return float(doy[wy == w][post[-1]]) if len(post) else np.nan

        s_dep = depletion(sw, s_peak_i, 0.5)
        g_dep = depletion(gw, g_peak_i, 0.5)
        dep_err = (
            (g_dep - s_dep) if (np.isfinite(s_dep) and np.isfinite(g_dep)) else np.nan
        )
        rows.append(
            {"peak_timing_error_days": peak_err, "depletion_timing_error_days": dep_err}
        )
    if len(rows) < MIN_VALID_YEARS:
        return {
            "n_valid_years": len(rows),
            "peak_timing_error_days": np.nan,
            "depletion_timing_error_days": np.nan,
        }
    return {
        "n_valid_years": len(rows),
        "peak_timing_error_days": float(
            np.nanmedian([r["peak_timing_error_days"] for r in rows])
        ),
        "depletion_timing_error_days": float(
            np.nanmedian([r["depletion_timing_error_days"] for r in rows])
        ),
    }


def consistency_for_basin(
    dates: np.ndarray, g: np.ndarray, swe: np.ndarray
) -> dict[str, float]:
    """Daily-scale consistency stats on the test period."""
    ok = np.isfinite(g) & np.isfinite(swe)
    if int(ok.sum()) < MIN_ACTIVE_DAYS:
        return {
            "n_active_days": 0,
            "seasonal_corr": np.nan,
            "anomaly_corr": np.nan,
            "nrmse": np.nan,
            "bias_mm": np.nan,
        }
    gg, ss = g[ok].astype(np.float64), swe[ok].astype(np.float64)
    active = (gg > 1e-6) | (ss > 1e-6)
    n_active = int(active.sum())
    if n_active < MIN_ACTIVE_DAYS:
        return {
            "n_active_days": n_active,
            "seasonal_corr": np.nan,
            "anomaly_corr": np.nan,
            "nrmse": np.nan,
            "bias_mm": np.nan,
        }
    ga, sa = gg[active], ss[active]
    seasonal_corr = (
        float(stats.pearsonr(ga, sa)[0]) if ga.std() > 0 and sa.std() > 0 else np.nan
    )
    months = pd.to_datetime(dates[ok])[active].month.values
    g_anom = ga - np.array([ga[months == mm].mean() for mm in months])
    s_anom = sa - np.array([sa[months == mm].mean() for mm in months])
    if g_anom.std() > 0 and s_anom.std() > 0:
        anomaly_corr = float(stats.pearsonr(g_anom, s_anom)[0])
    else:
        anomaly_corr = np.nan
    rmse = float(np.sqrt(np.nanmean((ga - sa) ** 2)))
    nrmse = float(rmse / (sa.std() + 1e-8))
    return {
        "n_active_days": n_active,
        "seasonal_corr": seasonal_corr,
        "anomaly_corr": anomaly_corr,
        "nrmse": nrmse,
        "bias_mm": float(np.mean(ga - sa)),
    }


# ---------------------------------------------------------------------------
# analysis B helpers
# ---------------------------------------------------------------------------


def theil_sen(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.unique(x[ok]).size < 5:
        return {
            "n": int(ok.sum()),
            "slope": np.nan,
            "intercept": np.nan,
            "rho": np.nan,
            "p": np.nan,
        }
    rho, p = stats.spearmanr(x[ok], y[ok])
    slope, intercept, _, _ = stats.theilslopes(y[ok], x[ok])
    return {
        "n": int(ok.sum()),
        "slope": float(slope),
        "intercept": float(intercept),
        "rho": float(rho),
        "p": float(p),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
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
        export_models(results_root, data_root, args.device)

    # ---- load SWE reference (test period) --------------------------------
    ref = np.load(SWE_REF_DIR / "swe_ensemble.npz")
    if list(ref["basin_ids"]) != basin_ids:
        raise ValueError("SWE reference basin order differs from bundle order")
    swe_test = ref["swe_median"][:, test_sl]  # [531, 5479]
    swe_ens_test = ref["swe_ensemble"][:, test_sl, :]  # [531, 5479, 10]
    burden = pd.read_csv(
        SWE_REF_DIR / "swe_basin_burden_test.csv", dtype={"basin_id": str}
    ).set_index("basin_id")
    obs_test = bundle.target_mm_day[:, test_sl]  # [531, 5479] (mm/day, NaN gaps)

    # ---- load runs ---------------------------------------------------------
    q = {
        model: {
            s: load_run_arrays(results_root, model, s)["q_full"][:, test_sl]
            for s in DPL_SEEDS
        }
        for model in ("XAJ", "XAJ_CN")
    }
    g = {
        s: load_run_arrays(results_root, "XAJ_CN", s)["G"][:, test_sl]
        for s in DPL_SEEDS
    }
    from training.dpl.run_dpl_model import compute_kge_fp64

    kge = {
        model: {
            s: np.array(
                [
                    compute_kge_fp64(q[model][s][i], obs_test[i])
                    for i in range(len(basin_ids))
                ]
            )
            for s in DPL_SEEDS
        }
        for model in ("XAJ", "XAJ_CN")
    }

    # ---- analysis A: CN snow-state consistency -----------------------------
    rows_a = []
    for s in DPL_SEEDS:
        for i, b in enumerate(basin_ids):
            cons = consistency_for_basin(test_dates, g[s][i], swe_test[i])
            tim = timing_metrics_for_basin(test_dates, g[s][i], swe_test[i])
            rows_a.append({"basin_id": b, "seed": s, **cons, **tim})
    df_a = pd.DataFrame(rows_a)

    # ---- analysis B: Delta(CN-Base) vs burden ------------------------------
    rows_b = []
    for s in DPL_SEEDS:
        delta = kge["XAJ_CN"][s] - kge["XAJ"][s]
        for i, b in enumerate(basin_ids):
            rows_b.append(
                {
                    "basin_id": b,
                    "seed": s,
                    "kge_cn_test": kge["XAJ_CN"][s][i],
                    "kge_base_test": kge["XAJ"][s][i],
                    "delta_cn_minus_base": delta[i],
                    "median_annual_max_swe_mm": burden.loc[
                        b, "median_annual_max_swe_mm"
                    ],
                    "median_swe_positive_days": burden.loc[
                        b, "median_swe_positive_days"
                    ],
                    "n_valid_swe_years": burden.loc[b, "n_valid_years"],
                    "frac_snow": bundle.raw_attributes[i, 3],
                }
            )
    df_b = pd.DataFrame(rows_b)

    # ---- summary stats ------------------------------------------------------
    # A: medians over basins (seed 42 primary; seed 123 robustness)
    summary_a = {}
    for s in DPL_SEEDS:
        sub = df_a[df_a["seed"] == s]
        summary_a[f"seed_{s}"] = {
            "n_basins": int(len(sub)),
            "n_with_consistency": int((sub["n_active_days"] >= MIN_ACTIVE_DAYS).sum()),
            "median_seasonal_corr": float(sub["seasonal_corr"].median()),
            "median_anomaly_corr": float(sub["anomaly_corr"].median()),
            "median_nrmse": float(sub["nrmse"].median()),
            "median_bias_mm": float(sub["bias_mm"].median()),
            "n_with_peak_timing": int(sub["peak_timing_error_days"].notna().sum()),
            "median_peak_timing_error_days": float(
                sub["peak_timing_error_days"].median()
            ),
            "n_with_depletion_timing": int(
                sub["depletion_timing_error_days"].notna().sum()
            ),
            "median_depletion_timing_error_days": float(
                sub["depletion_timing_error_days"].median()
            ),
            "frac_basins_peak_err_lt_15d": float(
                (sub["peak_timing_error_days"].abs() <= 15).mean()
            ),
        }
    # snow-active subset (median annual max SWE >= 20 mm)
    active_b = burden[burden["median_annual_max_swe_mm"] >= 20].index
    for s in DPL_SEEDS:
        sub = df_a[(df_a["seed"] == s) & (df_a["basin_id"].isin(active_b))]
        summary_a[f"seed_{s}_snow_active_only"] = {
            "n_basins": int(len(sub)),
            "median_seasonal_corr": float(sub["seasonal_corr"].median()),
            "median_anomaly_corr": float(sub["anomaly_corr"].median()),
            "median_peak_timing_error_days": float(
                sub["peak_timing_error_days"].median()
            ),
            "median_depletion_timing_error_days": float(
                sub["depletion_timing_error_days"].median()
            ),
        }

    # B: associations
    summary_b = {}
    for s in DPL_SEEDS:
        sub = df_b[df_b["seed"] == s]
        entry = {"n_basins": int(len(sub))}
        for axis in (
            "median_annual_max_swe_mm",
            "median_swe_positive_days",
            "frac_snow",
        ):
            entry[axis] = theil_sen(
                sub[axis].to_numpy(), sub["delta_cn_minus_base"].to_numpy()
            )
        # snow-active-only association
        suba = sub[sub["median_annual_max_swe_mm"] >= 20]
        entry["median_annual_max_swe_mm_snow_active_only"] = theil_sen(
            suba["median_annual_max_swe_mm"].to_numpy(),
            suba["delta_cn_minus_base"].to_numpy(),
        )
        # quartile means
        q = pd.qcut(sub["median_annual_max_swe_mm"], 4, labels=False, duplicates="drop")
        entry["delta_by_swe_quartile"] = {
            f"q{qv}": float(sub.loc[q == qv, "delta_cn_minus_base"].mean())
            for qv in sorted(q.dropna().unique())
        }
        summary_b[f"seed_{s}"] = entry

    # KGE levels (for context)
    kge_levels = {}
    for s in DPL_SEEDS:
        kge_levels[f"seed_{s}"] = {
            "base_median": float(np.nanmedian(kge["XAJ"][s])),
            "cn_median": float(np.nanmedian(kge["XAJ_CN"][s])),
        }

    report = {
        "tag": TAG,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "canonical dPL Base/CN, seeds 42/123, observation-trained (R1/R2 dpl_camels_531_lite_v2)",
        "test_period": "1995-10-01..2010-09-30",
        "reference": "CAMELS-US Snow-17/SAC-SMA SWE ensemble median (10 seeds), "
        "external process-state consistency reference (not snow truth)",
        "validity_rules": {
            "snow_active_year": f"SWE annual max >= {MIN_SWE_PEAK_MM} mm and CN G annual max >= {MIN_G_PEAK_MM} mm",
            "min_valid_years_per_basin": MIN_VALID_YEARS,
            "depletion_exclusion": "permanent-snowpack years (SWE never < 50% of peak) excluded",
            "min_active_days": MIN_ACTIVE_DAYS,
        },
        "kge_test_levels": kge_levels,
        "analysis_A_consistency": summary_a,
        "analysis_B_burden_gradient": summary_b,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "r4_phase1_dpl_official_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    df_a.to_csv(OUT_DIR / "snow_consistency_basin_seed.csv", index=False)
    df_b.to_csv(OUT_DIR / "burden_gradient_basin_seed.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
