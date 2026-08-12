"""Freeze pre-Phase-0 native-XAJ latent-response scale diagnostics.

This is a pure-forward audit over an already completed native XAJ calibration.
It never calibrates parameters and is intentionally not registered as a
training or experiment entry point.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ablation.ic_core.data_adapter import read_basin_ids
from models.parameter_specs import XAJ_PARAM_SPECS
from models.xaj import XAJ, _prepare_xaj_parameters, _xaj_step_compact

MIN_POSITIVE_STORAGE_SAMPLES = 100
NATIVE_DT_DAYS = 1.0


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
        "median": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
        "max": float(q[6]),
    }


def select_existing_native_parameters(
    raw_directory: str | Path,
    basin_ids: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the best already-recorded native XAJ start per basin.

    Selection observes the stored calibration objective only.  No optimizer or
    model fitting is called here.
    """
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for path in sorted(Path(raw_directory).glob("*.json")):
        record = json.loads(path.read_text())
        basin_id = str(record["basin_id"]).zfill(8)
        score = float(record["train_objective"])
        if basin_id not in best or score > best[basin_id][0] or (
            score == best[basin_id][0]
            and int(record["start"]) < int(best[basin_id][1]["start"])
        ):
            best[basin_id] = (score, record)
    missing = [basin_id for basin_id in basin_ids if basin_id not in best]
    if missing:
        raise RuntimeError(f"Existing native XAJ results missing {len(missing)} basins")
    parameters = np.asarray(
        [best[basin_id][1]["parameters"] for basin_id in basin_ids],
        dtype=np.float64,
    )
    if parameters.shape != (len(basin_ids), len(XAJ_PARAM_SPECS)):
        raise RuntimeError(f"Unexpected selected native parameter shape {parameters.shape}")
    metadata = {
        "source_directory": str(Path(raw_directory).resolve()),
        "selection": "highest stored train_objective; lowest start on exact tie",
        "n_records": len(list(Path(raw_directory).glob("*.json"))),
        "n_selected_basins": len(basin_ids),
    }
    return parameters, metadata


def _load_selected_forcing(
    dataset_path: str | Path,
    gage_ids_path: str | Path,
    basin_list_path: str | Path,
) -> tuple[list[str], np.ndarray]:
    basin_ids = read_basin_ids(basin_list_path)
    full_ids = [str(int(value)).zfill(8) for value in np.load(gage_ids_path)]
    source_index = {basin_id: index for index, basin_id in enumerate(full_ids)}
    indices = [source_index[basin_id] for basin_id in basin_ids]
    with Path(dataset_path).open("rb") as handle:
        forcing, _target, _attributes = pickle.load(handle)
    selected = np.asarray(forcing, dtype=np.float64)[indices]
    if selected.ndim != 3 or selected.shape[2] != 3:
        raise RuntimeError(f"Expected [basin,time,3] forcing, got {selected.shape}")
    return basin_ids, selected


def reconstruct_native_latent_storage(
    forcing: np.ndarray,
    parameters: np.ndarray,
    *,
    nearzero: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct native ``Z_N = CI/(1-CI) QI + CG/(1-CG) QG``.

    The native XAJ recursion is run with the same initial-state defaults as
    ``XAJ._init_states``.  The returned arrays are ``Z_N``, finite-basin mask,
    and positive-sample counts.  C==1 is marked invalid because its latent
    storage is mathematically infinite rather than silently epsilon-clamped.
    """
    if forcing.shape[0] != parameters.shape[0]:
        raise ValueError("forcing and parameters have different basin counts")
    params = {
        name: torch.as_tensor(parameters[:, index], dtype=torch.float64)
        for index, name in enumerate(XAJ_PARAM_SPECS)
    }
    precip = torch.as_tensor(forcing[:, :, 0], dtype=torch.float64)
    temp = torch.as_tensor(forcing[:, :, 1], dtype=torch.float64)
    pet = torch.as_tensor(forcing[:, :, 2], dtype=torch.float64)
    batch, nsteps = precip.shape
    k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, a, theta = _prepare_xaj_parameters(params)
    # XAJ initialization is deterministic and does not require running a
    # second model forward.  The model object is used only for that existing
    # state-layout helper; the loop below calls the existing compact kernel.
    model = XAJ()
    wu, wl, wd, s, fr, qi, qg, _buffer = model._init_states(
        batch, torch.device("cpu"), torch.float64, None, um, lm, dm, sm,
    )
    wm = um + lm + dm
    wmm = wm * (1.0 + b)
    ms = sm * (1.0 + ex)
    one_minus_im = 1.0 - im
    one_minus_ki_kg = 1.0 - ki - kg
    latent = torch.empty((batch, nsteps), dtype=torch.float64)
    with torch.no_grad():
        for timestep in range(nsteps):
            _rs, qi, qg, wu, wl, wd, s, fr = _xaj_step_compact(
                precip[:, timestep], pet[:, timestep], wu, wl, wd, s, fr,
                qi, qg, k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg,
                nearzero, wm, wmm, ms, one_minus_im, one_minus_ki_kg,
            )
            latent[:, timestep] = (
                ci * qi / (1.0 - ci) + cg * qg / (1.0 - cg)
            )
    finite_basin = np.isfinite(parameters[:, 11]) & np.isfinite(parameters[:, 12])
    finite_basin &= (parameters[:, 11] < 1.0) & (parameters[:, 12] < 1.0)
    latent_np = latent.numpy()
    finite_basin &= np.isfinite(latent_np).all(axis=1)
    positive_count = (latent_np > 0.0).sum(axis=1)
    return latent_np, finite_basin, positive_count


def compute_global_z0(
    latent_storage: np.ndarray,
    basin_ids: list[str],
    finite_basin: np.ndarray,
    positive_count: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Apply the two-level equal-basin robust log-center definition."""
    eligible = finite_basin & (positive_count >= MIN_POSITIVE_STORAGE_SAMPLES)
    if not eligible.any():
        raise RuntimeError("No basin has enough finite positive latent-storage samples")
    basin_medians: list[float] = []
    eligible_ids: list[str] = []
    for index, basin_id in enumerate(basin_ids):
        if not eligible[index]:
            continue
        values = latent_storage[index]
        positive = values > 0.0
        basin_medians.append(float(np.median(np.log(values[positive]))))
        eligible_ids.append(basin_id)
    medians = np.asarray(basin_medians, dtype=np.float64)
    log_z0 = float(np.median(medians))
    z0 = float(np.exp(log_z0))
    log_ratio = np.log(latent_storage[eligible][:, :][latent_storage[eligible] > 0.0]) - log_z0
    q25, q75 = np.quantile(medians, [0.25, 0.75])
    iqr = q75 - q25
    anomaly = (medians < q25 - 6.0 * iqr) | (medians > q75 + 6.0 * iqr)
    summary = {
        "eligible_basin_count": int(eligible.sum()),
        "excluded_basin_count": int((~eligible).sum()),
        "minimum_positive_samples": MIN_POSITIVE_STORAGE_SAMPLES,
        "positive_sample_count_distribution": _quantiles(positive_count[eligible]),
        "basin_log_median_distribution": {
            **_quantiles(medians), "iqr": float(iqr),
        },
        "global_log_z0": log_z0,
        "global_z0": z0,
        "all_log_ratio_distribution": {
            **_quantiles(log_ratio),
            "mean": float(log_ratio.mean()),
            "std": float(log_ratio.std()),
            "iqr": float(np.quantile(log_ratio, 0.75) - np.quantile(log_ratio, 0.25)),
            "n_values": int(log_ratio.size),
        },
        "scale_anomaly_count_6_iqr": int(anomaly.sum()),
        "scale_anomaly_basin_ids_6_iqr": [
            basin_id for basin_id, flag in zip(eligible_ids, anomaly) if flag
        ],
    }
    return z0, summary


def audit_native_tau_envelope(parameters: np.ndarray, finite_basin: np.ndarray) -> dict[str, Any]:
    """Audit physical CI/CG bounds and finite selected native timescales."""
    ci_spec = XAJ_PARAM_SPECS["xaj_ci"]
    cg_spec = XAJ_PARAM_SPECS["xaj_cg"]
    ci_lower_tau = -NATIVE_DT_DAYS / np.log(ci_spec["lower"])
    cg_lower_tau = -NATIVE_DT_DAYS / np.log(cg_spec["lower"])
    finite_ci = -NATIVE_DT_DAYS / np.log(parameters[finite_basin, 11])
    finite_cg = -NATIVE_DT_DAYS / np.log(parameters[finite_basin, 12])
    return {
        "dt_days": NATIVE_DT_DAYS,
        "ci_physical_range": [ci_spec["lower"], ci_spec["upper"]],
        "cg_physical_range": [cg_spec["lower"], cg_spec["upper"]],
        "ci_theoretical_tau_range": [float(ci_lower_tau), "infinity_at_C_equal_1"],
        "cg_theoretical_tau_range": [float(cg_lower_tau), "infinity_at_C_equal_1"],
        "finite_selected_ci_tau_distribution": _quantiles(finite_ci),
        "finite_selected_cg_tau_distribution": _quantiles(finite_cg),
        "finite_selected_combined_tau_min": float(min(finite_ci.min(), finite_cg.min())),
        "finite_selected_combined_tau_max": float(max(finite_ci.max(), finite_cg.max())),
        "singular_C_equal_1_basin_count": int((~finite_basin).sum()),
        "decision": "finite selected envelope plus 0.1 log-scale upper safety margin; C=1 remains singular",
    }


def run_freeze(
    *,
    dataset_path: str | Path,
    gage_ids_path: str | Path,
    basin_list_path: str | Path,
    native_raw_directory: str | Path,
) -> dict[str, Any]:
    basin_ids, forcing = _load_selected_forcing(dataset_path, gage_ids_path, basin_list_path)
    parameters, source = select_existing_native_parameters(native_raw_directory, basin_ids)
    latent, finite_basin, positive_count = reconstruct_native_latent_storage(forcing, parameters)
    z0, summary = compute_global_z0(latent, basin_ids, finite_basin, positive_count)
    summary.update({
        "basin_count": len(basin_ids),
        "native_tau_audit": audit_native_tau_envelope(parameters, finite_basin),
        "invalid_latent_storage_basin_count": int((~finite_basin).sum()),
        "invalid_latent_storage_basin_ids": [
            basin_id for basin_id, valid in zip(basin_ids, finite_basin) if not valid
        ],
        "native_dt_days": NATIVE_DT_DAYS,
        "parameter_names": list(XAJ_PARAM_SPECS),
        "source": source,
        "z0_definition": "median over basins of median_t(log(Z_N,t)) for Z_N,t > 0",
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    _repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--dataset", type=Path, default=_repo_root / "data/camels_dataset")
    parser.add_argument("--gage-ids", type=Path, default=_repo_root / "data/gage_id.npy")
    parser.add_argument("--basin-list", type=Path, default=_repo_root / "data/531sub_id.txt")
    parser.add_argument("--native-raw", type=Path, default=Path("results/xaj_base_cmaes_531_batched_paired_v2/raw/xaj"))
    args = parser.parse_args()
    summary = run_freeze(
        dataset_path=args.dataset,
        gage_ids_path=args.gage_ids,
        basin_list_path=args.basin_list,
        native_raw_directory=args.native_raw,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
