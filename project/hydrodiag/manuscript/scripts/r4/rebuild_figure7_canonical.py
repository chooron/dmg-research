#!/usr/bin/env python3
"""Reconstruct the canonical F7 TGD dPL seed-42 source and rebuild F7 data.

The legacy seed-42 physical/state assets are never overwritten.  This script
replays the saved static dPL checkpoints, validates the replay against the
known-good seed-123/2026 artifacts, writes a new seed-42 state asset, rebuilds
F7 source tables, and then delegates rendering to the canonical F7 renderer.
Only the static parameter network and the requested state reconstruction are
executed; no training or parameter optimization occurs.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
HYDRODIAG_ROOT = HERE.parents[2]
REPO_ROOT = HERE.parents[4]
if str(HYDRODIAG_ROOT) not in sys.path:
    sys.path.insert(0, str(HYDRODIAG_ROOT))

from ablation.ic_core.data_adapter import load_531_bundle  # noqa: E402
from manuscript.scripts.r4.audit_tgd_dpl_seed_failures import (  # noqa: E402
    checkpoint_replay,
    load_ids,
    sha256,
)
from manuscript.scripts.r4.common import (  # noqa: E402
    bundle_config,
    default_results_root,
)
from manuscript.scripts.r4.soil_analysis import (  # noqa: E402
    calendar_month_anomaly,
    monthly_aggregate,
    smooth_7d,
    zscore_nrmse,
)
from manuscript.scripts.r4.state_export import (  # noqa: E402
    continuous_forward,
    model_instances,
)

SEEDS = (42, 123, 2026)
MODEL_KEY = "XAJ_TGD2"
PARAM_NAMES = [
    "tgd_tau_warm",
    "tgd_delta_tau_cold",
    "xaj_k",
    "xaj_b",
    "xaj_im",
    "xaj_um",
    "xaj_lm",
    "xaj_dm",
    "xaj_c",
    "xaj_sm",
    "xaj_ex",
    "xaj_ki",
    "xaj_kg",
    "xaj_ci",
    "xaj_cg",
    "xaj_a",
    "xaj_theta",
]
STATE_KEYS = (
    "fr",
    "qg",
    "qi",
    "rs_instant",
    "s",
    "tgd_retention",
    "tgd_storage",
    "tgd_tau",
    "wd",
    "wl",
    "wu",
)
PHASE_NAMES = {
    1: "Phase_1_Snow_Accumulation",
    2: "Phase_2_Active_Melt_Recharge",
    3: "Phase_3_Post_Melt_Transition",
    4: "Phase_4_Summer_Dry_Down",
}
PHASE_LABELS = {
    1: "Accumulation",
    2: "Active melt",
    3: "Post-melt",
    4: "Dry-down",
}
REPLAY_RTOL = 1e-6
REPLAY_ATOL = 1e-3
STATE_SAMPLE_INDICES = np.array([0, 127, 171, 351, 430, 526], dtype=np.int64)


def external_phase_codes(dates: pd.DatetimeIndex, swe: np.ndarray) -> np.ndarray:
    """Use only external SWE and dates to define phase codes."""
    water_year = np.where(dates.month >= 10, dates.year + 1, dates.year).astype(int)
    phase = np.zeros(len(dates), dtype=np.int8)
    for water_year_id in np.unique(water_year):
        indices = np.flatnonzero(water_year == water_year_id)
        sw = swe[indices]
        finite = np.isfinite(sw)
        if not finite.any() or np.nanmax(sw) < 5.0:
            continue
        rel = np.arange(len(indices))
        peak_rel = int(np.nanargmax(sw))
        acc_rel = np.flatnonzero(finite & (sw >= 5.0))
        acc_start_rel = int(acc_rel[0]) if len(acc_rel) else 0
        post_peak = np.flatnonzero((rel > peak_rel) & (sw < 5.0))
        melt_end_rel = int(post_peak[0]) if len(post_peak) else len(indices) - 1
        months = dates[indices].month.to_numpy()
        phase[indices[(rel >= acc_start_rel) & (rel <= peak_rel)]] = 1
        phase[indices[(rel > peak_rel) & (rel <= melt_end_rel)]] = 2
        phase[indices[(rel > melt_end_rel) & (months <= 6)]] = 3
        phase[indices[np.isin(months, [7, 8, 9])]] = 4
    return phase


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    av = a[valid]
    bv = b[valid]
    if np.std(av) == 0.0 or np.std(bv) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def load_w_total(path: Path, test_slice: slice) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        values = (
            archive["wu"][:, test_slice]
            + archive["wl"][:, test_slice]
            + archive["wd"][:, test_slice]
        )
    return values.astype(np.float64)


def state_path(results_root: Path, seed: int) -> Path:
    return results_root / f"r4_official_dpl_XAJ_TGD2_seed{seed}/official_dpl_XAJ_TGD2_seed{seed}_full_arrays.npz"


def legacy_state_path(results_root: Path, structure: str, regime: str) -> Path:
    if regime == "IC_fused":
        return results_root / f"r4_ic_fused_XAJ{'_TGD2' if structure == 'TGD2' else ''}/ic_fused_XAJ{'_TGD2' if structure == 'TGD2' else ''}_full_arrays.npz"
    seed = regime.rsplit("seed", 1)[1]
    return results_root / f"r4_official_dpl_XAJ{'_TGD2' if structure == 'TGD2' else ('_CN' if structure == 'CN' else '')}_seed{seed}/official_dpl_XAJ{'_TGD2' if structure == 'TGD2' else ('_CN' if structure == 'CN' else '')}_seed{seed}_full_arrays.npz"


def write_streaming_npz(
    path: Path,
    basin_ids: list[str],
    dates: np.ndarray,
    forcing: np.ndarray,
    physical: np.ndarray,
    model: Any,
    device: torch.device,
    staging_root: Path,
) -> dict[str, Any]:
    """Run the full forward in small batches and stream float32 arrays to NPZ."""
    n_basins, n_days = forcing.shape[:2]
    staging = staging_root / "arrays"
    staging.mkdir(parents=True, exist_ok=True)
    memmaps: dict[str, np.memmap] = {}
    for key in ("q_full", *STATE_KEYS):
        memmaps[key] = np.lib.format.open_memmap(
            staging / f"{key}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_basins, n_days),
        )
    max_abs = 0.0
    nonfinite_batches: list[str] = []
    for left in range(0, n_basins, 16):
        right = min(left + 16, n_basins)
        q_full, states = continuous_forward(
            structure=MODEL_KEY,
            model=model,
            theta_hat=physical[left:right],
            forcing_full=forcing[left:right],
            device=device,
            dtype=torch.float32,
            batch=16,
            validate_subset=None,
        )
        memmaps["q_full"][left:right] = q_full.astype(np.float32)
        for key in STATE_KEYS:
            values = states[key].astype(np.float32)
            memmaps[key][left:right] = values
            if not np.isfinite(values).all():
                nonfinite_batches.append(f"{key}:{left}:{right}")
        max_abs = max(max_abs, float(np.nanmax(np.abs(q_full))))
        del q_full, states
    for mmap in memmaps.values():
        mmap.flush()
    out_path = path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        out_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for key in ("basin_ids", "dates"):
            temp = staging / f"{key}.npy"
            values = np.asarray(basin_ids if key == "basin_ids" else dates, dtype="<U10")
            np.save(temp, values)
            archive.write(temp, arcname=f"{key}.npy")
        for key in ("q_full", *STATE_KEYS):
            archive.write(staging / f"{key}.npy", arcname=f"{key}.npy")
    return {
        "shape": [n_basins, n_days],
        "state_keys": list(STATE_KEYS),
        "q_max_abs": max_abs,
        "nonfinite_batches": nonfinite_batches,
        "sha256": sha256(out_path),
    }


def compare_parameters(
    saved_path: Path,
    replay: dict[str, Any],
    replay_key: str = "physical",
    kind: str = "physical_parameters",
) -> dict[str, Any]:
    with np.load(saved_path, allow_pickle=False) as archive:
        saved = np.asarray(archive["params"], dtype=np.float64)
    replay_values = np.asarray(replay[replay_key], dtype=np.float64)
    finite = np.isfinite(saved).all(axis=1) & np.isfinite(replay_values).all(axis=1)
    diffs = np.abs(saved[finite] - replay_values[finite])
    close = np.zeros(len(saved), dtype=bool)
    close[finite] = np.isclose(
        saved[finite],
        replay_values[finite],
        rtol=REPLAY_RTOL,
        atol=REPLAY_ATOL,
    ).all(axis=1)
    return {
        "kind": kind,
        "saved_rows": int(len(saved)),
        "saved_finite_rows": int(np.isfinite(saved).all(axis=1).sum()),
        "replay_finite_rows": int(np.isfinite(replay_values).all(axis=1).sum()),
        "finite_pair_rows": int(finite.sum()),
        "within_tolerance_rows": int(close.sum()),
        "mismatch_rows": int((finite & ~close).sum()),
        "max_abs_diff": float(diffs.max()) if diffs.size else None,
        "median_abs_diff": float(np.median(diffs)) if diffs.size else None,
    }


F7_STATE_ATOL = 5e-3

def compare_state_sample(
    state_path_: Path,
    physical: np.ndarray,
    forcing: np.ndarray,
    model: Any,
    device: torch.device,
    test_slice: slice,
) -> dict[str, Any]:
    indices = STATE_SAMPLE_INDICES
    q, states = continuous_forward(
        structure=MODEL_KEY,
        model=model,
        theta_hat=physical[indices],
        forcing_full=forcing[indices],
        device=device,
        dtype=torch.float32,
        batch=16,
        validate_subset=None,
    )
    max_abs: dict[str, float] = {}
    with np.load(state_path_, allow_pickle=False) as archive:
        expected = {"q_full": archive["q_full"][indices]}
        expected.update({key: archive[key][indices] for key in STATE_KEYS})
    # The official arrays are full-axis; the replay is also full-axis.
    got = {"q_full": q}
    got.update(states)
    for key, value in got.items():
        max_abs[key] = float(np.nanmax(np.abs(value - expected[key])))
    expected_w = expected["wu"] + expected["wl"] + expected["wd"]
    got_w = got["wu"] + got["wl"] + got["wd"]
    q_max = max_abs["q_full"]
    w_max = float(np.nanmax(np.abs(got_w - expected_w)))
    return {
        "sample_indices": indices.tolist(),
        "max_abs_diff_by_key": max_abs,
        "max_abs_diff": max(max_abs.values()),
        "q_max_abs_diff": q_max,
        "w_total_max_abs_diff": w_max,
        "all_within_tolerance": q_max <= F7_STATE_ATOL and w_max <= F7_STATE_ATOL,
    }


def rebuild_source_tables(
    results_root: Path,
    out_dir: Path,
    basin_ids: list[str],
    dates_full: pd.DatetimeIndex,
    test_slice: slice,
    sm100_full: np.ndarray,
    swe_full: np.ndarray,
    burden: pd.DataFrame,
    regime_paths: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    test_dates = dates_full[test_slice]
    months = test_dates.month.to_numpy()
    sm100 = sm100_full[:, test_slice].astype(np.float64)
    swe = swe_full[:, test_slice].astype(np.float64)
    phase_full = [external_phase_codes(dates_full, swe_full[i]) for i in range(len(basin_ids))]
    phase_test = np.stack([codes[test_slice] for codes in phase_full])
    burden = burden.set_index("basin_id").loc[basin_ids].reset_index()
    burden_map = burden.set_index("basin_id").to_dict("index")
    source_basin_rows: list[dict[str, Any]] = []
    source_pair_rows: list[dict[str, Any]] = []
    source_phase_rows: list[dict[str, Any]] = []
    for regime, paths in regime_paths.items():
        w = {structure: load_w_total(path, test_slice) for structure, path in paths.items()}
        for structure, values in w.items():
            for i, basin_id in enumerate(basin_ids):
                ref = sm100[i]
                model_values = values[i]
                ref_anom = calendar_month_anomaly(ref, months)
                model_anom = calendar_month_anomaly(model_values, months)
                ref_7d = smooth_7d(ref)
                model_7d = smooth_7d(model_values)
                monthly_ref, monthly_dates = monthly_aggregate(ref, test_dates)
                monthly_model, _ = monthly_aggregate(model_values, test_dates)
                monthly_corr = finite_corr(monthly_ref, monthly_model)
                source_basin_rows.append(
                    {
                        "regime": regime,
                        "structure": structure,
                        "basin_id": basin_id,
                        "snow_burden_swe_mm": burden_map[basin_id]["snow_burden_swe_mm"],
                        "swe_positive_days": burden_map[basin_id].get("median_swe_positive_days", np.nan),
                        "daily_corr": finite_corr(ref, model_values),
                        "smoothed_7d_corr": finite_corr(ref_7d, model_7d),
                        "monthly_anomaly_corr": monthly_corr,
                        "nrmse": zscore_nrmse(model_values, ref),
                    }
                )
        for i, basin_id in enumerate(basin_ids):
            base_anom = calendar_month_anomaly(w["Base"][i], months)
            cn_anom = calendar_month_anomaly(w["CN"][i], months)
            tgd_anom = calendar_month_anomaly(w["TGD2"][i], months)
            base_corr = finite_corr(sm100[i], base_anom)
            cn_corr = finite_corr(sm100[i], cn_anom)
            tgd_corr = finite_corr(sm100[i], tgd_anom)
            source_pair_rows.append(
                {
                    "regime": regime,
                    "basin_id": basin_id,
                    "snow_burden_swe_mm": burden_map[basin_id]["snow_burden_swe_mm"],
                    "base_anomaly_corr": base_corr,
                    "cn_anomaly_corr": cn_corr,
                    "tgd2_anomaly_corr": tgd_corr,
                    "delta_cn_base_anomaly": cn_corr - base_corr if np.isfinite(cn_corr) and np.isfinite(base_corr) else np.nan,
                    "delta_tgd2_base_anomaly": tgd_corr - base_corr if np.isfinite(tgd_corr) and np.isfinite(base_corr) else np.nan,
                    "delta_cn_tgd2_anomaly": cn_corr - tgd_corr if np.isfinite(cn_corr) and np.isfinite(tgd_corr) else np.nan,
                }
            )
            for code, phase_name in PHASE_NAMES.items():
                mask = phase_test[i] == code
                if int(mask.sum()) < 30:
                    continue
                ref = sm100[i][mask]
                base = w["Base"][i][mask]
                cn = w["CN"][i][mask]
                tgd = w["TGD2"][i][mask]
                ref_anom_p = calendar_month_anomaly(sm100[i], months)[mask]
                base_anom_p = calendar_month_anomaly(w["Base"][i], months)[mask]
                cn_anom_p = calendar_month_anomaly(w["CN"][i], months)[mask]
                tgd_anom_p = calendar_month_anomaly(w["TGD2"][i], months)[mask]
                base_daily = finite_corr(ref, base)
                cn_daily = finite_corr(ref, cn)
                tgd_daily = finite_corr(ref, tgd)
                base_phase = finite_corr(ref_anom_p, base_anom_p)
                cn_phase = finite_corr(ref_anom_p, cn_anom_p)
                tgd_phase = finite_corr(ref_anom_p, tgd_anom_p)
                source_phase_rows.append(
                    {
                        "regime": regime,
                        "basin_id": basin_id,
                        "phase_code": code,
                        "phase_name": phase_name,
                        "n_days": int(mask.sum()),
                        "base_daily_corr": base_daily,
                        "cn_daily_corr": cn_daily,
                        "delta_daily_corr": cn_daily - base_daily if np.isfinite(cn_daily) and np.isfinite(base_daily) else np.nan,
                        "base_anomaly_corr": base_phase,
                        "cn_anomaly_corr": cn_phase,
                        "delta_anomaly_corr": cn_phase - base_phase if np.isfinite(cn_phase) and np.isfinite(base_phase) else np.nan,
                        "tgd2_anomaly_corr": tgd_phase,
                        "delta_tgd2_base_anomaly": tgd_phase - base_phase if np.isfinite(tgd_phase) and np.isfinite(base_phase) else np.nan,
                        "tgd2_daily_corr": tgd_daily,
                        "snow_burden_swe_mm": burden_map[basin_id]["snow_burden_swe_mm"],
                    }
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    basin_df = pd.DataFrame(source_basin_rows)
    pair_df = pd.DataFrame(source_pair_rows)
    phase_df = pd.DataFrame(source_phase_rows)
    basin_df.to_csv(out_dir / "three_structure_basin_state_consistency.csv", index=False)
    pair_df.to_csv(out_dir / "three_structure_paired_structural_effects.csv", index=False)
    phase_df.to_csv(out_dir / "robustness_process_phase_consistency.csv", index=False)
    return {
        "basin_rows": int(len(basin_df)),
        "paired_rows": int(len(pair_df)),
        "phase_rows": int(len(phase_df)),
        "paired_finite_tgd_dpl": int(
            np.isfinite(pair_df.loc[pair_df["regime"] == "dPL_seed42", "tgd2_anomaly_corr"]).sum()
        ),
        "phase_finite_tgd_dpl": int(
            np.isfinite(phase_df.loc[phase_df["regime"] == "dPL_seed42", "tgd2_anomaly_corr"]).sum()
        ),
    }


def write_seed_sensitivity_summary(
    source_dir: Path,
    seed: int,
    burden_path: Path,
    output_path: Path,
    status: str = "AVAILABLE",
 ) -> None:
    if status != "AVAILABLE":
        pd.DataFrame([{ "seed": seed, "status": status }]).to_csv(output_path, index=False)
        return
    pair = pd.read_csv(source_dir / "three_structure_paired_structural_effects.csv")
    phase = pd.read_csv(source_dir / "robustness_process_phase_consistency.csv")
    burden = pd.read_csv(burden_path, dtype={"basin_id": str})
    burden["basin_id"] = burden["basin_id"].astype(str).str.zfill(8)
    burden["burden_group"] = burden["burden_group"].astype(str)
    mapping = burden.set_index("basin_id")["burden_group"].to_dict()
    pair["burden_group"] = pair["basin_id"].astype(str).str.zfill(8).map(mapping)
    phase["burden_group"] = phase["basin_id"].astype(str).str.zfill(8).map(mapping)
    rows = []
    for structure, pair_column, phase_column in [("TGD", "delta_tgd2_base_anomaly", "delta_tgd2_base_anomaly"), ("CN", "delta_cn_base_anomaly", "delta_anomaly_corr")]:
        for group in ["No/trace", "Low", "Middle", "High", "Very high"]:
            values = pair[(pair["burden_group"] == group) & pair["regime"].astype(str).str.contains(f"seed{seed}")][pair_column].to_numpy(float)
            rows.append({"seed": seed, "status": status, "quantity": "all_days_burden_gradient", "structure": structure, "burden_group": group, "phase_name": "All days", "median": float(np.nanmedian(values)), "n": int(np.isfinite(values).sum())})
            for phase_name in PHASE_NAMES.values():
                p = phase[(phase["burden_group"] == group) & (phase["phase_name"] == phase_name) & phase["regime"].astype(str).str.contains(f"seed{seed}")]
                values_p = p[phase_column].to_numpy(float)
                rows.append({"seed": seed, "status": status, "quantity": "phase_resolved", "structure": structure, "burden_group": group, "phase_name": phase_name, "median": float(np.nanmedian(values_p)) if np.isfinite(values_p).any() else np.nan, "n": int(np.isfinite(values_p).sum())})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    

def main() -> None:
    results_root = default_results_root()
    data_root = REPO_ROOT / "data"
    training_root = results_root / "dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2"
    out_dir = results_root / "r4_replay_dpl_XAJ_TGD2_seed42"
    f7_source_dir = results_root / "r4_phase1_soil_official/figure7_rebuilt_canonical"
    out_dir.mkdir(parents=True, exist_ok=True)
    f7_source_dir.mkdir(parents=True, exist_ok=True)

    basin_ids = load_ids(data_root)
    bundle = load_531_bundle(bundle_config(data_root))
    forcing = bundle.forcing.astype(np.float32)
    attrs = bundle.raw_attributes.astype(np.float32)
    from training.dpl.run_dpl_model import robust_normalize  # noqa: PLC0415

    attrs_norm, _ = robust_normalize(attrs)
    _, specs = __import__(
        "training.dpl.run_dpl_model", fromlist=["LITE_MODEL_REGISTRY"]
    ).LITE_MODEL_REGISTRY[MODEL_KEY]
    replay: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        replay[seed] = checkpoint_replay(
            training_root / f"seed_{seed}", attrs_norm, PARAM_NAMES, specs
        )
        if not replay[seed]["ok"]:
            raise RuntimeError(f"checkpoint replay failed for seed {seed}: {replay[seed]['error']}")
        if not np.isfinite(replay[seed]["physical"]).all():
            raise RuntimeError(f"checkpoint replay produced non-finite physical parameters for seed {seed}")

    validation_rows: list[dict[str, Any]] = []
    for seed in (123, 2026):
        seed_dir = training_root / f"seed_{seed}"
        physical_comparison = compare_parameters(
            seed_dir / "best_parameters_physical.npz",
            replay[seed],
            replay_key="physical",
            kind="physical_parameters",
        )
        physical_comparison["seed"] = seed
        normalized_comparison = compare_parameters(
            seed_dir / "best_parameters_normalized.npz",
            replay[seed],
            replay_key="theta",
            kind="normalized_network_outputs",
        )
        normalized_comparison["seed"] = seed
        validation_rows.extend([physical_comparison, normalized_comparison])
        if (
            physical_comparison["saved_finite_rows"] != 531
            or physical_comparison["within_tolerance_rows"] != 531
            or physical_comparison["mismatch_rows"] != 0
            or normalized_comparison["saved_finite_rows"] != 531
            or normalized_comparison["within_tolerance_rows"] != 531
            or normalized_comparison["mismatch_rows"] != 0
        ):
            raise RuntimeError(
                f"checkpoint replay does not reproduce seed {seed} saved parameters "
                "within the configured tolerance; F7 rebuild stopped"
            )
    # Seed-42 saved physical parameters are historical evidence only.
    seed42_saved = training_root / "seed_42/best_parameters_physical.npz"
    seed42_comparison = compare_parameters(
        seed42_saved,
        replay[42],
        replay_key="physical",
        kind="historical_physical_parameters",
    )
    seed42_comparison["seed"] = 42
    validation_rows.append(seed42_comparison)

    with np.load(results_root / "r4_caravan_soil_reference_v1/caravan_soil_ensemble.npz", allow_pickle=False) as caravan:
        dates_full = pd.DatetimeIndex(pd.to_datetime(caravan["dates"].astype(str)))
        test_slice = slice(int(caravan["test_slice_start"]), int(caravan["test_slice_stop"]))
        sm100_full = caravan["SM100"].astype(np.float64)
        swe_full = caravan["caravan_swe"].astype(np.float64)
    burden = pd.read_csv(
        results_root / "r4_swe_reference_v1/swe_basin_burden_test.csv",
        dtype={"basin_id": str},
    )
    burden["basin_id"] = burden["basin_id"].astype(str).str.zfill(8)
    burden = burden.rename(columns={"median_annual_max_swe_mm": "snow_burden_swe_mm"})

    # Validate replay against saved state trajectories on a fixed, mixed sample.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_instances(device, torch.float32)[MODEL_KEY]
    for seed in (123, 2026):
        sample = compare_state_sample(
            state_path(results_root, seed),
            np.asarray(replay[seed]["physical"], dtype=np.float32),
            forcing,
            model,
            device,
            test_slice,
        )
        validation_rows.append(
            {
                "seed": seed,
                "kind": "state_sample_replay",
                "saved_rows": len(STATE_SAMPLE_INDICES),
                "saved_finite_rows": len(STATE_SAMPLE_INDICES),
                "replay_finite_rows": len(STATE_SAMPLE_INDICES),
                "finite_pair_rows": len(STATE_SAMPLE_INDICES),
                "within_tolerance_rows": len(STATE_SAMPLE_INDICES) if sample["all_within_tolerance"] else 0,
                "mismatch_rows": 0 if sample["all_within_tolerance"] else 1,
                "max_abs_diff": sample["max_abs_diff"],
                "q_max_abs_diff": sample["q_max_abs_diff"],
                "w_total_max_abs_diff": sample["w_total_max_abs_diff"],
                "median_abs_diff": float(np.median(list(sample["max_abs_diff_by_key"].values()))),
                "sample_indices": ";".join(map(str, sample["sample_indices"])),
            }
        )
        if not sample["all_within_tolerance"]:
            raise RuntimeError(
                f"checkpoint replay does not reproduce seed {seed} state sample; "
                "F7 rebuild stopped"
            )

    # Reconstruct seed 42 from its finite checkpoint replay.  This writes only
    # the new replay source; the legacy seed-42 NPZ remains untouched.
    reconstructed_norm = out_dir / "reconstructed_best_parameters_normalized.npz"
    reconstructed_phys = out_dir / "reconstructed_best_parameters_physical.npz"
    np.savez_compressed(reconstructed_norm, params=np.asarray(replay[42]["theta"], dtype=np.float64))
    np.savez_compressed(reconstructed_phys, params=np.asarray(replay[42]["physical"], dtype=np.float64))
    staging_root = Path(tempfile.mkdtemp(prefix="f7_replay_stage_", dir=out_dir))
    try:
        reconstructed_state = out_dir / "reconstructed_dpl_XAJ_TGD2_seed42_full_arrays.npz"
        state_meta = write_streaming_npz(
            reconstructed_state,
            basin_ids,
            dates_full.astype(str).to_numpy(),
            forcing,
            np.asarray(replay[42]["physical"], dtype=np.float32),
            model,
            device,
            staging_root,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    with np.load(reconstructed_state, allow_pickle=False) as archive:
        w_reconstructed = archive["wu"] + archive["wl"] + archive["wd"]
        reconstructed_finite = np.isfinite(w_reconstructed).all(axis=1)
        reconstructed_nonconstant = np.ptp(w_reconstructed, axis=1) > 0.0
        state_valid = reconstructed_finite & reconstructed_nonconstant
        state_meta.update(
            {
                "state_finite_rows": int(reconstructed_finite.sum()),
                "state_nonconstant_rows": int(reconstructed_nonconstant.sum()),
                "state_valid_rows": int(state_valid.sum()),
                "zero_variance_rows": int((~reconstructed_nonconstant).sum()),
            }
        )

    regime_paths = {
        "dPL_seed42": {
            "Base": results_root / "r4_official_dpl_XAJ_seed42/official_dpl_XAJ_seed42_full_arrays.npz",
            "CN": results_root / "r4_official_dpl_XAJ_CN_seed42/official_dpl_XAJ_CN_seed42_full_arrays.npz",
            "TGD2": reconstructed_state,
        },
        "IC_fused": {
            "Base": results_root / "r4_ic_fused_XAJ/ic_fused_XAJ_full_arrays.npz",
            "CN": results_root / "r4_ic_fused_XAJ_CN/ic_fused_XAJ_CN_full_arrays.npz",
            "TGD2": results_root / "r4_ic_fused_XAJ_TGD2/ic_fused_XAJ_TGD2_full_arrays.npz",
        },
    }
    source_meta = rebuild_source_tables(
        results_root,
        f7_source_dir,
        basin_ids,
        dates_full,
        test_slice,
        sm100_full,
        swe_full,
        burden,
        regime_paths,
    )
    seed123_paths = {
        "dPL_seed123": {
            "Base": results_root / "r4_official_dpl_XAJ_seed123/official_dpl_XAJ_seed123_full_arrays.npz",
            "CN": results_root / "r4_official_dpl_XAJ_CN_seed123/official_dpl_XAJ_CN_seed123_full_arrays.npz",
            "TGD2": results_root / "r4_official_dpl_XAJ_TGD2_seed123/official_dpl_XAJ_TGD2_seed123_full_arrays.npz",
        }
    }
    seed123_dir = f7_source_dir / "seed123_sensitivity"
    seed123_meta = rebuild_source_tables(
        results_root, seed123_dir, basin_ids, dates_full, test_slice, sm100_full, swe_full, burden, seed123_paths
    )
    sensitivity_burden = results_root / "r4_phase1_soil_official/figure7_full_basin_assignments.csv"
    write_seed_sensitivity_summary(
        seed123_dir, 123, sensitivity_burden, f7_source_dir / "seed123_f7_sensitivity_summary.csv"
    )
    seed2026_base = results_root / "r4_official_dpl_XAJ_seed2026/official_dpl_XAJ_seed2026_full_arrays.npz"
    seed2026_cn = results_root / "r4_official_dpl_XAJ_CN_seed2026/official_dpl_XAJ_CN_seed2026_full_arrays.npz"
    write_seed_sensitivity_summary(
        f7_source_dir, 2026, sensitivity_burden, f7_source_dir / "seed2026_f7_sensitivity_summary.csv",
        status="UNAVAILABLE_MISSING_BASE_CN_STATE_ASSETS",
    )
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(f7_source_dir / "replay_validation.csv", index=False)

    manifest = {
        "status": "accepted_for_canonical_f7_seed42",
        "legacy_seed42_assets_superseded_for_f7": True,
        "model": MODEL_KEY,
        "seed42_checkpoint": str(training_root / "seed_42/best_checkpoint.pt"),
        "seed42_checkpoint_sha256": sha256(training_root / "seed_42/best_checkpoint.pt"),
        "preprocessing": "training robust_normalize(raw_attributes), median/IQR, NaN fill, clip ±5",
        "transform": "StaticParameterNet sigmoid output + current physical_parameters inverse-log TGD2 residence-time transform",
        "canonical_seed": 42,
        "n_basins": len(basin_ids),
        "n_full_days": len(dates_full),
        "test_period": "1995-10-01..2010-09-30",
        "reconstructed_parameters": {
            "normalized": str(reconstructed_norm),
            "physical": str(reconstructed_phys),
            "physical_finite_rows": int(np.isfinite(replay[42]["physical"]).all(axis=1).sum()),
        },
        "reconstructed_state": {"path": str(reconstructed_state), **state_meta},
        "validation": validation_rows,
        "f7_source_dir": str(f7_source_dir),
        "f7_source": source_meta,
        "seed123_sensitivity_source": seed123_meta,
        "seed2026_sensitivity_status": "UNAVAILABLE_MISSING_BASE_CN_STATE_ASSETS",
        "no_nan_to_num": True,
        "state_replay_validation": "q_full and W_total compared against known-good persisted states; diagnostic s/fr may show larger FP32 path sensitivity and are not F7 estimands",
        "no_training": True,
        "no_optimization": True,
        "legacy_parameter_asset": str(seed42_saved),
        "legacy_state_asset": str(state_path(results_root, 42)),
    }
    (out_dir / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (f7_source_dir / "rebuild_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"reconstructed_state": str(reconstructed_state), "f7_source": str(f7_source_dir), "state_meta": state_meta, "source_meta": source_meta}, indent=2, default=str))


if __name__ == "__main__":
    main()
