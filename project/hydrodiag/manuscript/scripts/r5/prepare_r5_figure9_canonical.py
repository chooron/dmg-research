#!/usr/bin/env python3
"""Build the R1-aligned statistical assets used by the redesigned R5 Figure 9.

This is a statistics-only rebuild from existing selected IC records and existing
trained dPL parameter files. It does not train, calibrate, or alter model results.
The primary estimand is the paired outlet timing contrast

    delta_abs_CT_Base_CN = abs(signed_CT_Base) - abs(signed_CT_CN)

where signed CT is the R1 water-year median of CT_sim - CT_obs. For dPL, the
basin-level signed CT for each structure is the median across the available
completed seed parameter files before the paired contrast is formed. The manifest
records any incomplete seed set; no missing seed is silently imputed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.scripts.r4.common import (  # noqa: E402
    default_data_root,
    default_results_root,
    load_bundle,
    period_slices,
    zfill8,
)
from models import (  # noqa: E402
    GR4J,
    GR4JWithCemaNeige,
    GR4JWithTGD2,
    GR4J_CN_PARAM_SPECS,
    GR4J_PARAM_SPECS,
    GR4J_TGD2_PARAM_SPECS,
    SIMHYD,
    SIMHYDWithCemaNeige,
    SIMHYDWithTGD2,
    SIMHYD_CN_PARAM_SPECS,
    SIMHYD_PARAM_SPECS,
    SIMHYD_TGD2_PARAM_SPECS,
    XAJ,
    XAJWithCemaNeige,
    XAJWithTGD2,
    XAJ_CN_PARAM_SPECS,
    XAJ_PARAM_SPECS,
    XAJ_TGD2_PARAM_SPECS,
)
from models.cemaneige import _estimate_psol_annual  # noqa: E402

HOSTS = ("XAJ", "GR4J", "SIMHYD")
STRUCTURES = ("Base", "TGD", "CN")
PRIMARY_STRUCTURES = ("Base", "CN")
REGIMES = ("IC", "dPL")
DPL_SEEDS = (42, 123, 2026)
TGD_DPL_SEEDS = (42,)
SNOW_STRATA = (
    ("S1", "[0, 0.05)", 0.00, 0.05, False),
    ("S2", "[0.05, 0.15)", 0.05, 0.15, False),
    ("S3", "[0.15, 0.30)", 0.15, 0.30, False),
    ("S4", "[0.30, 0.50)", 0.30, 0.50, False),
    ("S5", "[0.50, 1.00]", 0.50, 1.00, True),
)
STRATUM_COUNTS = {name: count for name, count in zip((s[0] for s in SNOW_STRATA), (165, 156, 121, 34, 55))}

MODEL_INFO = {
    "XAJ": {
        "Base": (XAJ, XAJ_PARAM_SPECS),
        "TGD": (XAJWithTGD2, XAJ_TGD2_PARAM_SPECS),
        "CN": (XAJWithCemaNeige, XAJ_CN_PARAM_SPECS),
    },
    "GR4J": {
        "Base": (GR4J, GR4J_PARAM_SPECS),
        "TGD": (GR4JWithTGD2, GR4J_TGD2_PARAM_SPECS),
        "CN": (GR4JWithCemaNeige, GR4J_CN_PARAM_SPECS),
    },
    "SIMHYD": {
        "Base": (SIMHYD, SIMHYD_PARAM_SPECS),
        "TGD": (SIMHYDWithTGD2, SIMHYD_TGD2_PARAM_SPECS),
        "CN": (SIMHYDWithCemaNeige, SIMHYD_CN_PARAM_SPECS),
    },
}


def source_structure_key(structure: str) -> str:
    return "TGD2" if structure == "TGD" else structure


def assign_stratum(frac_snow: float) -> str:
    for name, _label, low, high, inclusive in SNOW_STRATA:
        if low <= frac_snow <= high if inclusive else low <= frac_snow < high:
            return name
    raise ValueError(f"Snow fraction outside canonical bounds: {frac_snow}")


def select_ic_params(results_root: Path, host: str, structure: str, basin_ids: list[str]) -> np.ndarray:
    """Select the maximum train-KGE IC restart with minimum-start tie break."""
    source_key = source_structure_key(structure)
    if host == "XAJ":
        run_dir = {
            "Base": "xaj_base_cmaes_531_batched_paired_v2",
            "TGD": "xaj_tgd2_cmaes_531_batched_v1",
            "CN": "xaj_cn_cmaes_531_batched_paired_v2",
        }[structure]
        raw_subdir = {"Base": "xaj", "TGD": "xaj_tgd2", "CN": "xaj_cn"}[structure]
    else:
        raw_subdir = host.lower() if structure == "Base" else f"{host.lower()}_{source_key.lower()}"
        run_dir = f"{raw_subdir}_cmaes_531_batched_v1"
    raw_dir = results_root / run_dir / "raw" / raw_subdir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing IC raw result directory: {raw_dir}")

    records: dict[str, list[tuple[float, int, list[float]]]] = {}
    for path in raw_dir.glob("*.json"):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        basin_id = zfill8(record["basin_id"])
        train_kge = float(record.get("train_metrics", {}).get("kge", np.nan))
        start = int(record["start"])
        params = record.get("parameters")
        if params is None:
            continue
        records.setdefault(basin_id, []).append((train_kge, start, params))

    selected: list[list[float]] = []
    for basin_id in basin_ids:
        candidates = [row for row in records.get(basin_id, []) if np.isfinite(row[0])]
        if not candidates:
            raise RuntimeError(f"No valid IC restart for {host}/{structure}/{basin_id}")
        selected.append(sorted(candidates, key=lambda row: (-row[0], row[1]))[0][2])
    return np.asarray(selected, dtype=np.float32)


def dpl_param_path(results_root: Path, host: str, structure: str, seed: int) -> Path:
    source_key = source_structure_key(structure)
    if structure == "TGD":
        if host == "XAJ":
            return results_root / "dpl_camels_531_lite_v3_tgd2_dpl_audited" / "XAJ_TGD2" / f"seed_{seed}" / "best_parameters_physical.npz"
        return results_root / "dpl_camels_531_lite_v3" / f"{host}_TGD2" / f"seed_{seed}" / "best_parameters_physical.npz"
    model_dir = host if structure == "Base" else f"{host}_{source_key}"
    return results_root / "dpl_camels_531_lite_v2" / model_dir / f"seed_{seed}" / "best_parameters_physical.npz"


def canonical_signed_ct(sim_test: np.ndarray, obs_test: np.ndarray, dates_test: pd.DatetimeIndex) -> np.ndarray:
    """R1 CT: median valid water-year CT_sim - CT_obs for each basin."""
    date_days = dates_test.to_numpy(dtype="datetime64[D]")
    years = dates_test.year.to_numpy() + (dates_test.month.to_numpy() >= 10).astype(int)
    unique_years = sorted(set(years.tolist()))
    out = np.full(sim_test.shape[0], np.nan, dtype=np.float64)

    for basin_idx in range(sim_test.shape[0]):
        basin_values: list[float] = []
        for water_year in unique_years:
            mask = years == water_year
            start = np.datetime64(f"{water_year - 1:04d}-10-01")
            end = np.datetime64(f"{water_year:04d}-10-01")
            expected_dates = np.arange(start, end, dtype="datetime64[D]")
            dates_y = date_days[mask]
            obs_y = np.asarray(obs_test[basin_idx, mask], dtype=np.float64)
            sim_y = np.asarray(sim_test[basin_idx, mask], dtype=np.float64)
            if not np.array_equal(dates_y, expected_dates):
                continue
            if not (np.isfinite(obs_y).all() and np.isfinite(sim_y).all()):
                continue
            if not ((obs_y >= 0).all() and (sim_y >= 0).all()):
                continue
            total_obs = float(obs_y.sum())
            total_sim = float(sim_y.sum())
            if total_obs <= 0 or total_sim <= 0:
                continue
            ct_obs = int(np.searchsorted(np.cumsum(obs_y), 0.5 * total_obs, side="left") + 1)
            ct_sim = int(np.searchsorted(np.cumsum(sim_y), 0.5 * total_sim, side="left") + 1)
            basin_values.append(float(ct_sim - ct_obs))
        if len(basin_values) >= 5:
            out[basin_idx] = float(np.median(basin_values))
    return out


def run_model_test(
    model_name: str,
    structure: str,
    params: np.ndarray,
    forcing: dict[str, torch.Tensor],
    test_slice: slice,
    device: torch.device,
) -> np.ndarray:
    model_class, parameter_specs = MODEL_INFO[model_name][structure]
    model = model_class().to(device)
    model.eval()
    params_dict = {
        name: torch.as_tensor(params[:, idx], dtype=torch.float32, device=device)
        for idx, name in enumerate(parameter_specs)
    }
    with torch.no_grad():
        q_sim, _ = model(forcings=forcing, params=params_dict)
    result = q_sim[:, test_slice].detach().cpu().numpy().astype(np.float32, copy=True)
    del q_sim, model, params_dict
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_ic_test(
    results_root: Path,
    host: str,
    structure: str,
    basin_ids: list[str],
    forcing: dict[str, torch.Tensor],
    test_slice: slice,
    device: torch.device,
) -> np.ndarray:
    params = select_ic_params(results_root, host, structure, basin_ids)
    return run_model_test(host, structure, params, forcing, test_slice, device)


def load_dpl_test(
    results_root: Path,
    host: str,
    structure: str,
    seed: int,
    forcing: dict[str, torch.Tensor],
    test_slice: slice,
    device: torch.device,
) -> np.ndarray:
    path = dpl_param_path(results_root, host, structure, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    params = np.load(path)["params"]
    return run_model_test(host, structure, params, forcing, test_slice, device)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int, n_resamples: int = 1000) -> tuple[float, float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan"), int(x.size)
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boot = np.full(n_resamples, np.nan, dtype=np.float64)
    for idx in range(n_resamples):
        sample = rng.integers(0, x.size, size=x.size)
        boot[idx] = stats.spearmanr(x[sample], y[sample]).statistic
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return rho, float(low), float(high), int(x.size)


def bootstrap_fraction(values: np.ndarray, seed: int, n_resamples: int = 1000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_resamples, values.size), replace=True).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def median_across_seeds(seed_arrays: dict[int, np.ndarray]) -> np.ndarray:
    """Median across seeds while preserving all-seed missing basins as NaN."""
    stacked = np.vstack(list(seed_arrays.values()))
    result = np.full(stacked.shape[1], np.nan, dtype=np.float64)
    for basin_idx in range(stacked.shape[1]):
        finite = stacked[:, basin_idx][np.isfinite(stacked[:, basin_idx])]
        if finite.size:
            result[basin_idx] = float(np.median(finite))
    return result


def load_r1_xaj_canonical(basin_ids: list[str]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Use the promoted R1 XAJ Base/TGD/CN CT contrasts where available."""
    r1_dir = PROJECT_ROOT / "manuscript" / "analysis" / "R1" / "results"
    paired = pd.read_csv(r1_dir / "canonical_paired_contrasts.csv", dtype={"basin_id": str})
    basin = pd.read_csv(r1_dir / "canonical_basin_level.csv", dtype={"basin_id": str})
    paired["basin_id"] = paired["basin_id"].map(zfill8)
    basin["basin_id"] = basin["basin_id"].map(zfill8)
    regime_labels = {"IC": "IC-CMA-ES", "dPL": "dPL-MLP"}
    signed: dict[str, dict[str, np.ndarray]] = {regime: {} for regime in REGIMES}
    effects: dict[str, np.ndarray] = {}
    for regime, label in regime_labels.items():
        p_rows = paired[(paired["regime"] == label) & (paired["period"] == "test")].set_index("basin_id")
        if len(p_rows) != len(basin_ids):
            raise ValueError(f"R1 canonical paired contrast is incomplete for {regime}")
        effects[regime] = p_rows.loc[basin_ids, "delta_absCT_Base_CN"].to_numpy(float)
        for structure in STRUCTURES:
            b_rows = basin[(basin["regime"] == label) & (basin["period"] == "test") & (basin["structure"] == structure)].set_index("basin_id")
            if len(b_rows) != len(basin_ids):
                raise ValueError(f"R1 canonical signed CT is incomplete for {regime}/{structure}")
            signed[regime].setdefault(structure, b_rows.loc[basin_ids, "signed_CT_error"].to_numpy(float))
    return signed, effects


def build_assets(output_dir: Path, device_name: str | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = default_data_root()
    results_root = default_results_root()
    bundle = load_bundle(data_root)
    slices = period_slices(bundle)
    basin_ids = [zfill8(value) for value in bundle.basin_ids]
    frac_snow = bundle.raw_attributes[:, 3].astype(float)
    snow_stratum = np.asarray([assign_stratum(value) for value in frac_snow])
    dates_test = pd.to_datetime(bundle.dates[slices["test"]])
    obs_test = np.asarray(bundle.target_mm_day[:, slices["test"]], dtype=np.float32)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    P = torch.from_numpy(bundle.forcing[:, :, 0]).float().to(device)
    T = torch.from_numpy(bundle.forcing[:, :, 1]).float().to(device)
    PET = torch.from_numpy(bundle.forcing[:, :, 2]).float().to(device)
    with torch.no_grad():
        cn_psol = _estimate_psol_annual(P, T)
    forcing = {
        "precip": P,
        "temp": T,
        "pet": PET,
        "cn_psol_annual": cn_psol,
        "temp_mean_train": T[:, slices["train"]].mean(dim=1),
        "temp_std_train": T[:, slices["train"]].std(dim=1),
    }

    signed_by_regime: dict[str, dict[str, dict[str, np.ndarray]]] = {regime: {} for regime in REGIMES}
    seed_counts: dict[str, dict[str, dict[str, int]]] = {regime: {} for regime in REGIMES}
    provenance: list[dict[str, Any]] = []

    for host in HOSTS:
        for structure in STRUCTURES:
            signed_by_regime["IC"].setdefault(host, {})
            seed_counts["IC"].setdefault(host, {})
            q_test = load_ic_test(results_root, host, structure, basin_ids, forcing, slices["test"], device)
            signed = canonical_signed_ct(q_test, obs_test, dates_test)
            signed_by_regime["IC"][host][structure] = signed
            seed_counts["IC"][host][structure] = 1
            provenance.append({"regime": "IC", "host": host, "structure": structure, "seeds": ["selected_restart"], "n_seeds": 1})

            seed_arrays: dict[int, np.ndarray] = {}
            for seed in DPL_SEEDS if structure in PRIMARY_STRUCTURES else TGD_DPL_SEEDS:
                path = dpl_param_path(results_root, host, structure, seed)
                if not path.exists():
                    continue
                q_seed = load_dpl_test(results_root, host, structure, seed, forcing, slices["test"], device)
                seed_arrays[seed] = canonical_signed_ct(q_seed, obs_test, dates_test)
            if not seed_arrays:
                raise RuntimeError(f"No complete dPL parameter file for {host}/{structure}")
            signed_by_regime["dPL"].setdefault(host, {})[structure] = median_across_seeds(seed_arrays)
            seed_counts["dPL"].setdefault(host, {})[structure] = len(seed_arrays)
            provenance.append({
                "regime": "dPL",
                "host": host,
                "structure": structure,
                "seeds": sorted(seed_arrays),
                "n_seeds": len(seed_arrays),
                "missing_requested_seeds": sorted(set(DPL_SEEDS if structure in PRIMARY_STRUCTURES else TGD_DPL_SEEDS) - set(seed_arrays)),
            })

    # The manuscript-facing R1 package provides the canonical XAJ paired CT rows.
    # Use those rows verbatim so the cross-host extension does not silently diverge.
    r1_xaj_signed, r1_xaj_effects = load_r1_xaj_canonical(basin_ids)
    signed_by_regime["IC"]["XAJ"].update(r1_xaj_signed["IC"])
    signed_by_regime["dPL"]["XAJ"].update(r1_xaj_signed["dPL"])
    seed_counts["IC"]["XAJ"] = {structure: 1 for structure in STRUCTURES}
    seed_counts["dPL"]["XAJ"] = {structure: 3 for structure in STRUCTURES}
    provenance.append({"regime": "IC", "host": "XAJ", "source": "analysis/R1/results/canonical_basin_level.csv", "aggregation": "R1 canonical"})
    provenance.append({"regime": "dPL", "host": "XAJ", "source": "analysis/R1/results/canonical_basin_level.csv", "aggregation": "R1 canonical median_across_seeds"})
    primary = pd.DataFrame({"basin_id": basin_ids, "frac_snow": frac_snow, "snow_stratum": snow_stratum})
    for host in HOSTS:
        for regime in REGIMES:
            base = signed_by_regime[regime][host]["Base"]
            cn = signed_by_regime[regime][host]["CN"]
            primary[f"delta_abs_CT_Base_CN_{host}_{regime}"] = (
                r1_xaj_effects[regime] if host == "XAJ" else np.abs(base) - np.abs(cn)
            )
            primary[f"n_seeds_{host}_{regime}"] = seed_counts[regime][host]["CN"] if regime == "dPL" else 1
    primary_path = output_dir / "r5_figure9_primary_effects.csv"
    primary.to_csv(primary_path, index=False)

    timing_rows: list[dict[str, Any]] = []
    for host in HOSTS:
        for regime in REGIMES:
            for structure in STRUCTURES:
                values = signed_by_regime[regime][host][structure]
                for idx, basin_id in enumerate(basin_ids):
                    timing_rows.append({
                        "basin_id": basin_id,
                        "frac_snow": frac_snow[idx],
                        "snow_stratum": snow_stratum[idx],
                        "host": host,
                        "regime": regime,
                        "structure": structure,
                        "signed_CT_error": values[idx],
                        "n_seeds": seed_counts[regime][host][structure],
                    })
    timing = pd.DataFrame(timing_rows)
    timing_path = output_dir / "r5_figure9_timing_distributions.csv"
    timing.to_csv(timing_path, index=False)

    matrix_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for host in HOSTS:
        for regime in REGIMES:
            effect = primary[f"delta_abs_CT_Base_CN_{host}_{regime}"].to_numpy(float)
            for stratum, _label, _low, _high, _inclusive in SNOW_STRATA:
                vals = effect[snow_stratum == stratum]
                finite = vals[np.isfinite(vals)]
                matrix_rows.append({
                    "host": host,
                    "regime": regime,
                    "stratum": stratum,
                    "N": int(finite.size),
                    "median_delta_abs_CT_Base_CN": float(np.median(finite)) if finite.size else np.nan,
                    "q25": float(np.percentile(finite, 25)) if finite.size else np.nan,
                    "q75": float(np.percentile(finite, 75)) if finite.size else np.nan,
                })
            rho, low, high, n = bootstrap_spearman(
                frac_snow, effect, seed=20260730 + HOSTS.index(host) * 100 + REGIMES.index(regime)
            )
            continuous_rows.append({
                "host": host,
                "regime": regime,
                "N": n,
                "spearman_rho": rho,
                "ci_low": low,
                "ci_high": high,
                "n_seeds": seed_counts[regime][host]["CN"] if regime == "dPL" else 1,
            })
            for endpoint in ("S1", "S5"):
                vals = effect[snow_stratum == endpoint]
                finite = vals[np.isfinite(vals)]
                endpoint_rows.append({
                    "host": host,
                    "regime": regime,
                    "endpoint": endpoint,
                    "N": int(finite.size),
                    "median_delta_abs_CT_Base_CN": float(np.median(finite)) if finite.size else np.nan,
                })

    matrix_path = output_dir / "r5_figure9_primary_effect_matrix.csv"
    pd.DataFrame(matrix_rows).to_csv(matrix_path, index=False)
    continuous_path = output_dir / "r5_figure9_continuous_summary.csv"
    pd.DataFrame(continuous_rows).to_csv(continuous_path, index=False)
    endpoint_path = output_dir / "r5_figure9_endpoint_summary.csv"
    pd.DataFrame(endpoint_rows).to_csv(endpoint_path, index=False)

    agreement_rows: list[dict[str, Any]] = []
    for regime in REGIMES:
        effect_columns = [f"delta_abs_CT_Base_CN_{host}_{regime}" for host in HOSTS]
        effect_matrix = primary[effect_columns].to_numpy(float)
        for stratum, _label, _low, _high, _inclusive in SNOW_STRATA:
            mask = snow_stratum == stratum
            common = np.isfinite(effect_matrix[mask]).all(axis=1)
            values = effect_matrix[mask][common]
            positive_count = (values > 0).sum(axis=1) if values.size else np.array([], dtype=int)
            exactly_two = (positive_count == 2).astype(float)
            all_three = (positive_count == 3).astype(float)
            total = (positive_count >= 2).astype(float)
            ci_low, ci_high = bootstrap_fraction(total, seed=20300000 + REGIMES.index(regime) * 100 + len(agreement_rows)) if total.size else (np.nan, np.nan)
            agreement_rows.append({
                "regime": regime,
                "stratum": stratum,
                "N": int(values.shape[0]),
                "P_exactly_2_of_3": float(exactly_two.mean()) if total.size else np.nan,
                "P_3_of_3": float(all_three.mean()) if total.size else np.nan,
                "P_at_least_2": float(total.mean()) if total.size else np.nan,
                "P_at_least_2_ci_low": ci_low,
                "P_at_least_2_ci_high": ci_high,
            })
    agreement_path = output_dir / "r5_figure9_primary_agreement.csv"
    pd.DataFrame(agreement_rows).to_csv(agreement_path, index=False)

    manifest = {
        "status": "PASS_WITH_DATA_SCOPE",
        "estimand": "abs(signed_CT_Base) - abs(signed_CT_CN)",
        "ct_operator": "water-year (Oct 1-Sep 30) 50 percent cumulative-flow day; signed CT = CT_sim - CT_obs",
        "water_year_rule": "exact contiguous complete water year; finite nonnegative paired series; at least five valid water years",
        "basin_aggregation": "median signed CT across valid water years, then absolute value for the paired contrast",
        "ic_aggregation": "maximum stored train-period KGE with minimum-start tie break",
        "xaj_primary_source": "R1 canonical paired contrasts and basin-level signed CT rows are used verbatim",
        "dpl_aggregation": "median signed CT across available completed seed parameter files per basin and structure before Base-CN contrast",
        "dpl_requested_seeds": list(DPL_SEEDS),
        "snow_strata": {name: {"range": label, "N": STRATUM_COUNTS[name]} for name, label, *_ in SNOW_STRATA},
        "source_results_root": str(results_root),
        "source_data_root": str(data_root),
        "device": str(device),
        "primary_source": str(primary_path),
        "timing_source": str(timing_path),
        "continuous_source": str(continuous_path),
        "agreement_source": str(agreement_path),
        "provenance": provenance,
        "tgd_secondary_scope": "TGD dPL uses the available audited seed-42 TGD2 parameter files; no incomplete seed is imputed.",
        "training_or_calibration": "not performed; existing parameter files and selected IC records only",
    }
    manifest_path = output_dir / "r5_figure9_canonical_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built R1-aligned Figure 9 assets in {output_dir}")
    print(f"device={device}; primary={primary_path}; timing={timing_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "manuscript" / "results" / "R5")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    args = parser.parse_args()
    build_assets(args.output_dir, args.device)


if __name__ == "__main__":
    main()
