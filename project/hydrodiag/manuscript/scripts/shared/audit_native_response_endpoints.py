"""Read-only audit of native XAJ CI/CG endpoint calibration records."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from models.parameter_specs import XAJ_PARAM_SPECS
from models.structure_response import native_effective_kss
from models.xaj import XAJ, _prepare_xaj_parameters

from manuscript.scripts.shared.freeze_native_subsurface_scale import (
    _load_selected_forcing,
    select_existing_native_parameters,
)

TAU0_LOWER = 0.43429448190325187
TAU0_UPPER = 15478.143902262878
OLD_Z0 = 3.1553493591016335


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "max": float(values.max()),
        "iqr": float(q[3] - q[1]),
    }


def _bin_counts(values: np.ndarray) -> dict[str, int]:
    return {
        "<1%": int((values < 0.01).sum()),
        "1-5%": int(((values >= 0.01) & (values < 0.05)).sum()),
        "5-20%": int(((values >= 0.05) & (values < 0.20)).sum()),
        ">=20%": int((values >= 0.20).sum()),
    }


def _bin_ids(
    values: np.ndarray, basin_ids: list[str], mask: np.ndarray
) -> dict[str, list[str]]:
    selected_ids = np.asarray(basin_ids, dtype=object)[mask]
    selected_values = values[mask]
    return {
        "<1%": [str(b) for b, v in zip(selected_ids, selected_values) if v < 0.01],
        "1-5%": [
            str(b) for b, v in zip(selected_ids, selected_values) if 0.01 <= v < 0.05
        ],
        "5-20%": [
            str(b) for b, v in zip(selected_ids, selected_values) if 0.05 <= v < 0.20
        ],
        ">=20%": [str(b) for b, v in zip(selected_ids, selected_values) if v >= 0.20],
    }


def _endpoint_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    ci = np.asarray([float(r["parameters"][11]) for r in records])
    cg = np.asarray([float(r["parameters"][12]) for r in records])
    both = (ci == 1.0) & (cg == 1.0)
    return {
        "n_starts": len(records),
        "ci": {
            "exact_eq_1": int((ci == 1.0).sum()),
            "ge_0_9999": int((ci >= 0.9999).sum()),
            "ge_0_999": int((ci >= 0.999).sum()),
            "ge_0_99": int((ci >= 0.99).sum()),
        },
        "cg": {
            "exact_eq_1": int((cg == 1.0).sum()),
            "ge_0_9999": int((cg >= 0.9999).sum()),
            "ge_0_999": int((cg >= 0.999).sum()),
            "ge_0_99": int((cg >= 0.99).sum()),
        },
        "both_exact_eq_1": int(both.sum()),
        "either_exact_eq_1": int(((ci == 1.0) | (cg == 1.0)).sum()),
        "both_ge_0_999": int(((ci >= 0.999) & (cg >= 0.999)).sum()),
        "either_ge_0_999": int(((ci >= 0.999) | (cg >= 0.999)).sum()),
    }


def _select_records(
    raw_directory: str | Path, basin_ids: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    by_basin: dict[str, list[dict[str, Any]]] = {basin_id: [] for basin_id in basin_ids}
    for path in sorted(Path(raw_directory).glob("*.json")):
        record = json.loads(path.read_text())
        basin_id = str(record["basin_id"]).zfill(8)
        if basin_id in by_basin:
            by_basin[basin_id].append(record)
    if any(len(records) != 10 for records in by_basin.values()):
        raise RuntimeError(
            "Endpoint audit requires exactly 10 existing starts per basin"
        )
    best: dict[str, dict[str, Any]] = {}
    for basin_id, records in by_basin.items():
        best[basin_id] = max(
            records, key=lambda r: (float(r["train_objective"]), -int(r["start"]))
        )
    return by_basin, best


def _objective_gap_audit(
    by_basin: dict[str, list[dict[str, Any]]], best: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    gaps: list[float] = []
    endpoint_fraction: list[float] = []
    no_finite: list[str] = []
    endpoint_best: list[str] = []
    for basin_id, records in by_basin.items():
        endpoint = [
            r
            for r in records
            if float(r["parameters"][11]) == 1.0 or float(r["parameters"][12]) == 1.0
        ]
        endpoint_fraction.append(len(endpoint) / len(records))
        if (
            float(best[basin_id]["parameters"][11]) == 1.0
            or float(best[basin_id]["parameters"][12]) == 1.0
        ):
            endpoint_best.append(basin_id)
        finite = [
            r
            for r in records
            if float(r["parameters"][11]) < 1.0 and float(r["parameters"][12]) < 1.0
        ]
        if not finite:
            no_finite.append(basin_id)
        else:
            best_finite = max(
                finite, key=lambda r: (float(r["train_objective"]), -int(r["start"]))
            )
            gaps.append(
                float(best[basin_id]["train_objective"])
                - float(best_finite["train_objective"])
            )
    gaps_np = np.asarray(gaps, dtype=np.float64)
    return {
        "basins_with_finite_alternative": len(gaps),
        "all_ten_starts_endpoint_no_finite_alternative": len(no_finite),
        "no_finite_alternative_basin_ids": no_finite,
        "endpoint_best_basin_ids": endpoint_best,
        "delta_objective_best_minus_finite": _summary(gaps_np),
        "delta_le_0_001_fraction": float((gaps_np <= 0.001).mean())
        if gaps_np.size
        else None,
        "delta_le_0_005_fraction": float((gaps_np <= 0.005).mean())
        if gaps_np.size
        else None,
        "delta_le_0_01_fraction": float((gaps_np <= 0.01).mean())
        if gaps_np.size
        else None,
        "endpoint_start_fraction": _summary(np.asarray(endpoint_fraction)),
    }


def _best_forward_activity(
    parameters: np.ndarray, forcing: np.ndarray
) -> dict[str, np.ndarray]:
    """Run existing native XAJ once per best basin and aggregate actual fluxes."""
    torch.set_num_threads(1)
    params = {
        name: torch.as_tensor(parameters[:, index], dtype=torch.float64)
        for index, name in enumerate(XAJ_PARAM_SPECS)
    }
    precip = torch.as_tensor(forcing[:, :, 0], dtype=torch.float64)
    pet = torch.as_tensor(forcing[:, :, 2], dtype=torch.float64)
    batch, nsteps = precip.shape
    k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, a, theta = _prepare_xaj_parameters(
        params
    )
    model = XAJ()
    wu, wl, wd, s, fr, qi, qg, _buffer = model._init_states(
        batch,
        torch.device("cpu"),
        torch.float64,
        None,
        um,
        lm,
        dm,
        sm,
    )
    wm = um + lm + dm
    wmm = wm * (1.0 + b)
    ms = sm * (1.0 + ex)
    one_minus_im = 1.0 - im
    one_minus_ki_kg = 1.0 - ki - kg
    vi = torch.zeros(batch, dtype=torch.float64)
    vg = torch.zeros(batch, dtype=torch.float64)
    vs = torch.zeros(batch, dtype=torch.float64)
    with torch.no_grad():
        for timestep in range(nsteps):
            out = model._step(
                precip[:, timestep],
                pet[:, timestep],
                wu,
                wl,
                wd,
                s,
                fr,
                qi,
                qg,
                k,
                b,
                im,
                um,
                lm,
                dm,
                c,
                sm,
                ex,
                ki,
                kg,
                ci,
                cg,
                1e-8,
            )
            (
                _q,
                rs_adj,
                qi,
                qg,
                _evap,
                wu,
                wl,
                wd,
                s,
                fr,
                _rs,
                ri,
                rg,
                _eu,
                _el,
                _ed,
            ) = out
            vs += rs_adj
            vi += ri * one_minus_im
            vg += rg * one_minus_im
    vss = vi + vg
    vtotal = vs + vss
    safe_ss = torch.clamp(vss, min=1e-30)
    safe_total = torch.clamp(vtotal, min=1e-30)
    return {
        "vi": vi.numpy(),
        "vg": vg.numpy(),
        "vss": vss.numpy(),
        "vtotal": vtotal.numpy(),
        "share_i_subsurface": (vi / safe_ss).numpy(),
        "share_g_subsurface": (vg / safe_ss).numpy(),
        "share_i_total": (vi / safe_total).numpy(),
        "share_g_total": (vg / safe_total).numpy(),
        "share_ss_total": (vss / safe_total).numpy(),
    }


def _activity_audit(
    best_parameters: np.ndarray, basin_ids: list[str], activity: dict[str, np.ndarray]
) -> dict[str, Any]:
    ci_endpoint = best_parameters[:, 11] == 1.0
    cg_endpoint = best_parameters[:, 12] == 1.0
    result: dict[str, Any] = {}
    for label, mask, sub_share, total_share in (
        (
            "CI_endpoint_interflow",
            ci_endpoint,
            activity["share_i_subsurface"],
            activity["share_i_total"],
        ),
        (
            "CG_endpoint_groundwater",
            cg_endpoint,
            activity["share_g_subsurface"],
            activity["share_g_total"],
        ),
    ):
        result[label] = {
            "basin_count": int(mask.sum()),
            "basin_ids": [basin_id for basin_id, flag in zip(basin_ids, mask) if flag],
            "subsurface_share_summary": _summary(sub_share[mask]),
            "total_share_summary": _summary(total_share[mask]),
            "subsurface_share_bins": _bin_counts(sub_share[mask]),
            "total_share_bins": _bin_counts(total_share[mask]),
            "subsurface_share_bin_basin_ids": _bin_ids(sub_share, basin_ids, mask),
            "total_share_bin_basin_ids": _bin_ids(total_share, basin_ids, mask),
        }
    result["all_best_activity"] = {
        "vi": _summary(activity["vi"]),
        "vg": _summary(activity["vg"]),
        "vss": _summary(activity["vss"]),
        "vtotal": _summary(activity["vtotal"]),
        "share_i_subsurface": _summary(activity["share_i_subsurface"]),
        "share_g_subsurface": _summary(activity["share_g_subsurface"]),
        "share_ss_total": _summary(activity["share_ss_total"]),
    }
    return result


def _group_comparison(
    parameters: np.ndarray,
    best: dict[str, dict[str, Any]],
    basin_ids: list[str],
    activity: dict[str, np.ndarray],
) -> dict[str, Any]:
    endpoint = (parameters[:, 11] == 1.0) | (parameters[:, 12] == 1.0)
    finite = ~endpoint
    kss = native_effective_kss(
        torch.as_tensor(parameters[:, 9], dtype=torch.float64),
        torch.as_tensor(parameters[:, 10], dtype=torch.float64),
    ).numpy()
    objective = np.asarray(
        [float(best[basin_id]["train_objective"]) for basin_id in basin_ids],
        dtype=np.float64,
    )
    data = {
        "native_train_objective": objective,
        "kss": kss,
        "share_i_subsurface": activity["share_i_subsurface"],
        "share_g_subsurface": activity["share_g_subsurface"],
        "share_ss_total": activity["share_ss_total"],
    }
    for index, name in ((11, "ci"), (12, "cg")):
        data[f"{name}_finite_only"] = parameters[:, index]
    # Existing dataset attribute index 4 is aridity; no regression or
    # significance test is performed.
    groups: dict[str, Any] = {
        "endpoint_basin_count": int(endpoint.sum()),
        "finite_basin_count": int(finite.sum()),
    }
    for name, values in data.items():
        if name.endswith("_finite_only"):
            valid = (
                parameters[:, 11] < 1.0
                if name == "ci_finite_only"
                else parameters[:, 12] < 1.0
            )
            values = values[valid]
            endpoint_mask = endpoint[valid]
            finite_mask = finite[valid]
        else:
            endpoint_mask = endpoint
            finite_mask = finite
        groups[name] = {
            "endpoint": _summary(values[endpoint_mask]),
            "finite": _summary(values[finite_mask]),
        }
        ep_med = np.median(values[endpoint_mask])
        fi_med = np.median(values[finite_mask])
        pooled_iqr = max(
            groups[name]["endpoint"].get("iqr", 0.0),
            groups[name]["finite"].get("iqr", 0.0),
            1e-30,
        )
        groups[name]["standardized_median_difference"] = float(
            (ep_med - fi_med) / pooled_iqr
        )
    groups["basin_ids_endpoint"] = [
        basin_id for basin_id, flag in zip(basin_ids, endpoint) if flag
    ]
    groups["basin_ids_finite"] = [
        basin_id for basin_id, flag in zip(basin_ids, finite) if flag
    ]
    return groups


def _tau_audit(nsteps: int) -> dict[str, float]:
    cmax = math.exp(-1.0 / TAU0_UPPER)
    retention = math.exp(-float(nsteps) / TAU0_UPPER)
    return {
        "tau0_upper_day": TAU0_UPPER,
        "c_equiv_max": cmax,
        "record_length_days": float(nsteps),
        "retention_over_record": retention,
        "released_fraction_over_record": 1.0 - retention,
    }


def run_audit(
    *,
    dataset_path: str | Path,
    gage_ids_path: str | Path,
    basin_list_path: str | Path,
    native_raw_directory: str | Path,
) -> dict[str, Any]:
    basin_ids, forcing = _load_selected_forcing(
        dataset_path, gage_ids_path, basin_list_path
    )
    by_basin, best = _select_records(native_raw_directory, basin_ids)
    best_parameters = np.asarray(
        [best[basin_id]["parameters"] for basin_id in basin_ids], dtype=np.float64
    )
    all_counts = _endpoint_counts(
        [record for records in by_basin.values() for record in records]
    )
    best_counts = _endpoint_counts(list(best.values()))
    activity = _best_forward_activity(best_parameters, forcing)
    endpoint = (best_parameters[:, 11] == 1.0) | (best_parameters[:, 12] == 1.0)
    finite = ~endpoint
    finite_ci = best_parameters[finite, 11]
    finite_cg = best_parameters[finite, 12]
    return {
        "semantics": {
            "native_qi0": 0.1,
            "native_qg0": 0.1,
            "native_state_names": [
                "wu",
                "wl",
                "wd",
                "s",
                "fr",
                "qi",
                "qg",
                "rs_uh_buffer",
            ],
            "c_equal_1": "Q_to_Z inversion singular; forward augmented storage is defined after a finite latent Z initial condition is supplied",
            "native_default_q0_implication": "QI0/QG0 are nonzero, so C=1 does not determine a unique finite initial latent storage",
        },
        "all_start_endpoint_counts": all_counts,
        "best_start_endpoint_counts": best_counts,
        "all_start_near_boundary": all_counts,
        "best_basin_ids": basin_ids,
        "best_endpoint_basin_ids": [
            basin_id for basin_id, flag in zip(basin_ids, endpoint) if flag
        ],
        "best_ci_endpoint_basin_ids": [
            basin_id
            for basin_id, value in zip(basin_ids, best_parameters[:, 11])
            if value == 1.0
        ],
        "best_cg_endpoint_basin_ids": [
            basin_id
            for basin_id, value in zip(basin_ids, best_parameters[:, 12])
            if value == 1.0
        ],
        "objective_gap_audit": _objective_gap_audit(by_basin, best),
        "activity_audit": _activity_audit(best_parameters, basin_ids, activity),
        "tau_upper_finite_window_audit": _tau_audit(forcing.shape[1]),
        "finite_vs_endpoint_descriptive": _group_comparison(
            best_parameters, best, basin_ids, activity
        ),
        "z0_candidate": {
            "status": "not_recomputed_for_C_equal_1",
            "old_432_basin_z0": OLD_Z0,
            "reason": "nonzero native QI0/QG0 does not identify a unique finite C=1 latent initial storage",
            "finite_best_basin_count": int(finite.sum()),
            "endpoint_best_basin_count": int(endpoint.sum()),
        },
        "source": {
            "native_raw_directory": str(Path(native_raw_directory).resolve()),
            "selection": "existing best train_objective per basin; no recalibration",
            "n_basins": len(basin_ids),
            "n_timesteps": int(forcing.shape[1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/jingxin/code/dmg-research/data/camels_dataset"),
    )
    parser.add_argument(
        "--gage-ids",
        type=Path,
        default=Path("/home/jingxin/code/dmg-research/data/gage_id.npy"),
    )
    parser.add_argument(
        "--basin-list",
        type=Path,
        default=Path("/home/jingxin/code/dmg-research/data/531sub_id.txt"),
    )
    parser.add_argument(
        "--native-raw",
        type=Path,
        default=Path("results/xaj_base_cmaes_531_batched_paired_v2/raw/xaj"),
    )
    args = parser.parse_args()
    summary = run_audit(
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
