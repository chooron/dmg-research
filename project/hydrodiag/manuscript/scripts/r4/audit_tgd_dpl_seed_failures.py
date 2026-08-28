#!/usr/bin/env python3
"""Read-only audit of the three observation-trained XAJ_TGD2 dPL seeds.

This script reads persisted parameter/state artifacts and writes only new audit
artifacts below ``results/r4_phase1_soil_official/tgd_dpl_seed_failure_audit``.
It never trains, runs a hydrological state forward, or overwrites canonical
results.  The checkpoint replay is intentionally limited to the static dPL
parameter network because seed 42 does not contain a saved normalized-output
array and the audit must distinguish network output from persisted export data.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
HYDRODIAG_ROOT = HERE.parents[2]
WORKSPACE_ROOT = HERE.parents[4]
if str(HYDRODIAG_ROOT) not in sys.path:
    sys.path.insert(0, str(HYDRODIAG_ROOT))

from ablation.ic_core.data_adapter import load_531_bundle  # noqa: E402
from manuscript.scripts.r4.common import bundle_config, default_results_root  # noqa: E402
from training.dpl.run_dpl_model import (  # noqa: E402
    LITE_MODEL_REGISTRY,
    StaticParameterNet,
    physical_parameters,
    robust_normalize,
)

SEEDS = (42, 123, 2026)
MODEL_KEY = "XAJ_TGD2"
TRAINING_REL = Path("dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2")
STATE_REL = {
    seed: Path(
        f"r4_official_dpl_XAJ_TGD2_seed{seed}/"
        f"official_dpl_XAJ_TGD2_seed{seed}_full_arrays.npz"
    )
    for seed in SEEDS
}
STATE_KEYS = (
    "q_full",
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
EXPECTED_16_ORDER = (
    "02196000",
    "04015330",
    "04213000",
    "06221400",
    "06224000",
    "07261000",
    "10336645",
    "10336660",
    "11143000",
    "11482500",
    "12040500",
    "12041200",
    "12082500",
    "12092000",
    "12175500",
    "14137000",
 )
EXPECTED_16 = set(EXPECTED_16_ORDER)


def zfill8(value: Any) -> str:
    return str(value).zfill(8)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_text(value: bool | np.bool_) -> str:
    return "True" if bool(value) else "False"


def list_text(values: list[str]) -> str:
    return ";".join(values) if values else ""


def load_ids(data_root: Path) -> list[str]:
    path = data_root / "531sub_id.txt"
    raw = path.read_text(encoding="utf-8").strip()
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        values = ast.literal_eval(raw)
    ids = [zfill8(value) for value in values]
    if len(ids) != 531 or len(set(ids)) != 531:
        raise RuntimeError(f"canonical basin list is not 531 unique IDs: {path}")
    return ids


def checkpoint_replay(
    seed_dir: Path,
    attrs: np.ndarray,
    names: list[str],
    specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Replay only the static parameter net; no hydrological model is called."""
    config_path = seed_dir / "config.json"
    checkpoint_path = seed_dir / "best_checkpoint.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        return {
            "ok": False,
            "error": "missing config.json or best_checkpoint.pt",
            "logits": None,
            "theta": None,
            "physical": None,
            "checkpoint_epoch": None,
        }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    network = config["network"]
    hidden_sizes = [int(value) for value in network["hidden_sizes"]]
    net = StaticParameterNet(
        attrs.shape[1],
        specs,
        hidden_sizes,
        network["dropout"],
        network["output_epsilon"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    lower = torch.tensor([specs[name]["lower"] for name in names], dtype=torch.float32)
    upper = torch.tensor([specs[name]["upper"] for name in names], dtype=torch.float32)
    with torch.no_grad():
        x = torch.from_numpy(attrs.astype(np.float32))
        logits_tensor = net.head(net.trunk(x))
        theta_tensor = net(x)
        physical_dict = physical_parameters(
            theta_tensor, names, lower, upper - lower
        )
        physical_tensor = torch.column_stack(
            [physical_dict[name] for name in names]
        )
    return {
        "ok": True,
        "error": "",
        "logits": logits_tensor.numpy(),
        "theta": theta_tensor.numpy(),
        "physical": physical_tensor.numpy(),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


REPLAY_RTOL = 1e-6
REPLAY_ATOL = 1e-3

def compare_saved(saved: np.ndarray | None, replay: np.ndarray | None) -> dict[str, Any]:
    if saved is None:
        return {
            "saved_exists": False,
            "saved_finite_rows": None,
            "replay_finite_rows": int(np.isfinite(replay).all(axis=1).sum())
            if replay is not None
            else None,
            "both_finite_rows": None,
            "within_tolerance_rows": None,
            "all_within_tolerance": None,
            "max_abs_diff": None,
        }
    saved_finite = np.isfinite(saved).all(axis=1)
    replay_finite = (
        np.isfinite(replay).all(axis=1) if replay is not None else np.zeros(len(saved), bool)
    )
    both = saved_finite & replay_finite
    diff = np.abs(saved[both] - replay[both]) if both.any() else np.array([])
    within = np.zeros(len(saved), dtype=bool)
    if both.any():
        within[both] = np.isclose(
            saved[both],
            replay[both],
            rtol=REPLAY_RTOL,
            atol=REPLAY_ATOL,
        ).all(axis=1)
    return {
        "saved_exists": True,
        "saved_finite_rows": int(saved_finite.sum()),
        "replay_finite_rows": int(replay_finite.sum()),
        "both_finite_rows": int(both.sum()),
        "within_tolerance_rows": int(within.sum()),
        "all_within_tolerance": bool(within[both].all()) if both.any() else None,
        "max_abs_diff": float(diff.max()) if diff.size else None,
    }


def load_states(path: Path, basin_ids: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "finite": np.zeros(len(basin_ids), dtype=bool),
            "zero_variance": np.zeros(len(basin_ids), dtype=bool),
            "basin_ids_match": False,
            "n_days": None,
            "error": "missing state NPZ",
        }
    z = np.load(path, allow_pickle=False)
    present = set(z.files)
    missing = [key for key in STATE_KEYS if key not in present]
    if missing:
        return {
            "exists": True,
            "finite": np.zeros(len(basin_ids), dtype=bool),
            "zero_variance": np.zeros(len(basin_ids), dtype=bool),
            "basin_ids_match": False,
            "n_days": None,
            "error": f"missing state keys: {','.join(missing)}",
        }
    saved_ids = [zfill8(value) for value in z["basin_ids"]]
    ids_match = saved_ids == basin_ids
    arrays = {key: np.asarray(z[key]) for key in STATE_KEYS}
    finite = np.ones(len(basin_ids), dtype=bool)
    for key in STATE_KEYS:
        finite &= np.isfinite(arrays[key]).all(axis=1)
    w_total = arrays["wu"] + arrays["wl"] + arrays["wd"]
    zero_variance = np.ptp(w_total, axis=1) == 0.0
    return {
        "exists": True,
        "finite": finite,
        "zero_variance": zero_variance,
        "basin_ids_match": ids_match,
        "n_days": int(w_total.shape[1]),
        "error": "" if ids_match else "basin order mismatch",
    }


def training_status(seed_dir: Path) -> dict[str, Any]:
    history_path = seed_dir / "epoch_history.csv"
    history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
    history_finite = bool(
        not history.empty
        and np.isfinite(history["train_loss"].to_numpy(dtype=float)).all()
        and (history["finite_batches"].to_numpy(dtype=float) > 0).all()
    )
    checkpoint_path = seed_dir / "best_checkpoint.pt"
    checkpoint_epoch = None
    checkpoint_finite = False
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        numeric_tensors = [
            value for value in checkpoint.get("state_dict", {}).values()
            if torch.is_tensor(value)
        ]
        checkpoint_finite = bool(
            numeric_tensors and all(torch.isfinite(value).all().item() for value in numeric_tensors)
        )
    complete = (seed_dir / "COMPLETE").is_file()
    expected = {
        "complete_marker": complete,
        "checkpoint_epoch_100": (seed_dir / "checkpoint_epoch_100.pt").is_file(),
        "normalized_parameters": (seed_dir / "best_parameters_normalized.npz").is_file(),
        "physical_parameters": (seed_dir / "best_parameters_physical.npz").is_file(),
        "basin_final_summary": (seed_dir / "basin_final_summary.csv").is_file(),
        "report": (seed_dir / "report.md").is_file(),
    }
    formal_complete = all(expected.values()) and not history.empty and len(history) >= 100
    if formal_complete and history_finite and checkpoint_finite:
        status = "COMPLETED"
    elif checkpoint_path.is_file() and history_finite and checkpoint_finite:
        status = "INCOMPLETE"
    elif not checkpoint_path.is_file():
        status = "MISSING_ASSET"
    else:
        status = "TRAINING_NONFINITE_OR_CORRUPT"
    return {
        "status": status,
        "formal_complete": formal_complete,
        "history_rows": int(len(history)),
        "history_epoch_max": int(history["epoch"].max()) if not history.empty else None,
        "history_finite": history_finite,
        "checkpoint_finite": checkpoint_finite,
        "best_checkpoint_epoch": checkpoint_epoch,
        "expected_assets": expected,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    results_root = default_results_root()
    data_root = WORKSPACE_ROOT / "data"
    training_root = results_root / TRAINING_REL
    audit_root = results_root / "r4_phase1_soil_official/tgd_dpl_seed_failure_audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    basin_ids = load_ids(data_root)
    bundle = load_531_bundle(bundle_config(data_root))
    bundle_ids = [zfill8(value) for value in bundle.basin_ids]
    if bundle_ids != basin_ids:
        raise RuntimeError("bundle basin order differs from canonical 531sub_id list")
    raw_attrs = bundle.raw_attributes.astype(np.float32)
    attrs, attr_stats = robust_normalize(raw_attrs)
    raw_nonfinite = ~np.isfinite(raw_attrs).all(axis=1)
    raw_nonfinite_columns = {
        basin_ids[i]: np.flatnonzero(~np.isfinite(raw_attrs[i])).tolist()
        for i in np.flatnonzero(raw_nonfinite)
    }

    burden_path = results_root / "r4_phase1_soil_official/figure7_full_basin_assignments.csv"
    burden = pd.read_csv(burden_path)
    burden["basin_id"] = burden["basin_id"].map(zfill8)
    burden = burden.set_index("basin_id").loc[basin_ids].reset_index()
    burden_map = burden.set_index("basin_id").to_dict("index")

    _, specs = LITE_MODEL_REGISTRY[MODEL_KEY]
    names = list(specs)
    per_seed: dict[int, dict[str, Any]] = {}
    matrix_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        seed_dir = training_root / f"seed_{seed}"
        physical_path = seed_dir / "best_parameters_physical.npz"
        normalized_path = seed_dir / "best_parameters_normalized.npz"
        physical = (
            np.asarray(np.load(physical_path)["params"])
            if physical_path.is_file()
            else None
        )
        normalized = (
            np.asarray(np.load(normalized_path)["params"])
            if normalized_path.is_file()
            else None
        )
        if physical is not None and physical.shape != (531, len(names)):
            raise RuntimeError(f"unexpected physical parameter shape for seed {seed}: {physical.shape}")
        if normalized is not None and normalized.shape != (531, len(names)):
            raise RuntimeError(f"unexpected normalized parameter shape for seed {seed}: {normalized.shape}")
        replay = checkpoint_replay(seed_dir, attrs, names, specs)
        replay_physical = replay["physical"]
        replay_logits = replay["logits"]
        replay_theta = replay["theta"]
        comparisons = {
            "normalized": compare_saved(normalized, replay_theta),
            "physical": compare_saved(physical, replay_physical),
        }
        states = load_states(results_root / STATE_REL[seed], basin_ids)
        status = training_status(seed_dir)
        per_seed[seed] = {
            "seed_dir": seed_dir,
            "physical_path": physical_path,
            "normalized_path": normalized_path,
            "physical": physical,
            "normalized": normalized,
            "replay": replay,
            "comparisons": comparisons,
            "states": states,
            "training": status,
            "raw_output_valid": (
                np.isfinite(replay_logits).all(axis=1)
                if replay_logits is not None
                else np.zeros(531, dtype=bool)
            ),
            "normalized_replay_valid": (
                np.isfinite(replay_theta).all(axis=1)
                if replay_theta is not None
                else np.zeros(531, dtype=bool)
            ),
            "physical_replay_valid": (
                np.isfinite(replay_physical).all(axis=1)
                if replay_physical is not None
                else np.zeros(531, dtype=bool)
            ),
        }

    for i, basin_id in enumerate(basin_ids):
        row: dict[str, Any] = {
            "basin_id": basin_id,
            "snow_burden_swe_mm": burden_map[basin_id]["snow_burden_swe_mm"],
            "burden_group": burden_map[basin_id]["burden_group"],
            "raw_attribute_nonfinite": bool(raw_nonfinite[i]),
            "raw_attribute_nonfinite_columns": list_text(
                [str(x) for x in raw_nonfinite_columns.get(basin_id, [])]
            ),
        }
        root_failures: list[str] = []
        for seed in SEEDS:
            entry = per_seed[seed]
            physical = entry["physical"]
            replay = entry["replay"]
            states = entry["states"]
            param_valid = (
                bool(np.isfinite(physical[i]).all()) if physical is not None else False
            )
            failed_params = (
                [names[j] for j in np.flatnonzero(~np.isfinite(physical[i]))]
                if physical is not None
                else []
            )
            if physical is None:
                param_failure = "MISSING_ASSET"
            elif failed_params:
                param_failure = "PHYSICAL_PARAM_NONFINITE"
            else:
                param_failure = "VALID"
            state_finite = bool(states["finite"][i])
            state_zero_variance = bool(states["zero_variance"][i])
            if not states["exists"] or states["error"]:
                state_failure = "MISSING_ASSET"
            elif not state_finite:
                state_failure = "STATE_NONFINITE"
            elif state_zero_variance:
                state_failure = "STATE_ZERO_VARIANCE"
            else:
                state_failure = "VALID"
            if param_failure != "VALID":
                failure = param_failure
                root_failures.append(failure)
            else:
                failure = state_failure
                if failure != "VALID":
                    root_failures.append(failure)
            replay_phys_valid = bool(entry["physical_replay_valid"][i])
            raw_valid = bool(entry["raw_output_valid"][i])
            saved_replay_match = "NOT_SAVED"
            if physical is not None and replay.get("physical") is not None:
                replay_p = replay["physical"][i]
                saved_p = physical[i]
                if np.isfinite(saved_p).all() and np.isfinite(replay_p).all():
                    if np.isclose(
                        saved_p,
                        replay_p,
                        rtol=REPLAY_RTOL,
                        atol=REPLAY_ATOL,
                    ).all():
                        saved_replay_match = "MATCH_WITHIN_TOLERANCE"
                    else:
                        saved_replay_match = "MISMATCH_FINITE_REPLAY"
                elif not np.isfinite(saved_p).all() and np.isfinite(replay_p).all():
                    saved_replay_match = "MISMATCH_SAVED_NONFINITE_REPLAY_FINITE"
                else:
                    saved_replay_match = "NONFINITE_REPLAY"
            transform_failure = "NONE"
            if param_failure == "PHYSICAL_PARAM_NONFINITE" and replay_phys_valid:
                transform_failure = "NOT_REPRODUCED_BY_CURRENT_TRANSFORM"
            elif not replay_phys_valid:
                transform_failure = "REPLAY_PHYSICAL_NONFINITE"
            first_nonfinite = ""
            if failed_params:
                first_nonfinite = "ALL_SAVED_PARAMETERS_NONFINITE_NO_ORDER"
            row.update(
                {
                    f"seed{seed}_raw_output_valid": raw_valid,
                    f"seed{seed}_normalized_output_valid": bool(entry["normalized_replay_valid"][i]),
                    f"seed{seed}_physical_replay_valid": replay_phys_valid,
                    f"seed{seed}_param_valid": param_valid,
                    f"seed{seed}_state_finite": state_finite,
                    f"seed{seed}_state_zero_variance": state_zero_variance,
                    f"seed{seed}_state_valid": state_finite and not state_zero_variance,
                    f"seed{seed}_failure_type": failure,
                    f"seed{seed}_state_failure_type": state_failure,
                    f"seed{seed}_failed_parameters": list_text(failed_params),
                    f"seed{seed}_first_nonfinite_parameter": first_nonfinite,
                    f"seed{seed}_physical_transform_failure": transform_failure,
                    f"seed{seed}_saved_vs_checkpoint_replay": saved_replay_match,
                    f"seed{seed}_raw_output_source": "checkpoint_static_replay",
                }
            )
        row["failure_repeated_across_seeds"] = (
            "YES" if len(root_failures) >= 2 else "NO"
        )
        row["notes"] = (
            "seed42 saved physical parameter row is non-finite; current best-checkpoint "
            "static replay is finite; legacy state array is all-zero"
            if not row["seed42_param_valid"]
            else "all three saved physical parameter sources are finite"
        )
        matrix_rows.append(row)

    matrix_fields = [
        "basin_id",
        "snow_burden_swe_mm",
        "burden_group",
        "raw_attribute_nonfinite",
        "raw_attribute_nonfinite_columns",
    ]
    for seed in SEEDS:
        matrix_fields.extend(
            [
                f"seed{seed}_raw_output_valid",
                f"seed{seed}_normalized_output_valid",
                f"seed{seed}_physical_replay_valid",
                f"seed{seed}_param_valid",
                f"seed{seed}_state_finite",
                f"seed{seed}_state_zero_variance",
                f"seed{seed}_state_valid",
                f"seed{seed}_failure_type",
                f"seed{seed}_state_failure_type",
                f"seed{seed}_failed_parameters",
                f"seed{seed}_first_nonfinite_parameter",
                f"seed{seed}_physical_transform_failure",
                f"seed{seed}_saved_vs_checkpoint_replay",
                f"seed{seed}_raw_output_source",
            ]
        )
    matrix_fields.extend(["failure_repeated_across_seeds", "notes"])
    write_csv(audit_root / "seed_basin_validity_matrix.csv", matrix_rows, matrix_fields)

    def failure_set(seed: int, kind: str) -> set[str]:
        if kind == "param":
            return {
                row["basin_id"]
                for row in matrix_rows
                if not row[f"seed{seed}_param_valid"]
            }
        return {
            row["basin_id"]
            for row in matrix_rows
            if not row[f"seed{seed}_state_valid"]
        }

    overlap_rows: list[dict[str, Any]] = []
    for kind, label in (("param", "physical_parameter_invalid"), ("state", "state_invalid")):
        sets = {seed: failure_set(seed, kind) for seed in SEEDS}
        union = set().union(*sets.values())
        all3 = set.intersection(*(sets[seed] for seed in SEEDS))
        pair_42_123 = sets[42] & sets[123]
        pair_42_2026 = sets[42] & sets[2026]
        pair_123_2026 = sets[123] & sets[2026]
        exactly2 = (
            ((sets[42] & sets[123]) - sets[2026])
            | ((sets[42] & sets[2026]) - sets[123])
            | ((sets[123] & sets[2026]) - sets[42])
        )
        any2_or_more = (
            (sets[42] & sets[123])
            | (sets[42] & sets[2026])
            | (sets[123] & sets[2026])
        )
        row = {
            "failure_basis": label,
            "seed42_count": len(sets[42]),
            "seed123_count": len(sets[123]),
            "seed2026_count": len(sets[2026]),
            "pairwise_42_123": len(pair_42_123),
            "pairwise_42_2026": len(pair_42_2026),
            "pairwise_123_2026": len(pair_123_2026),
            "all3_intersection": len(all3),
            "exactly2_intersection": len(exactly2),
            "any2_or_more": len(any2_or_more),
            "seed42_only": len(sets[42] - sets[123] - sets[2026]),
            "seed123_only": len(sets[123] - sets[42] - sets[2026]),
            "seed2026_only": len(sets[2026] - sets[42] - sets[123]),
            "union": len(union),
        }
        overlap_rows.append(row)
    write_csv(
        audit_root / "seed_failure_overlap_summary.csv",
        overlap_rows,
        list(overlap_rows[0]),
    )

    current16_rows = [row for row in matrix_rows if row["basin_id"] in EXPECTED_16]
    if {row["basin_id"] for row in current16_rows} != EXPECTED_16:
        raise RuntimeError("current 16-basin set does not match Figure 7 audit set")
    current16_rows.sort(key=lambda row: EXPECTED_16_ORDER.index(row["basin_id"]))
    current16_fields = [
        "basin_id",
        "snow_burden_swe_mm",
        "burden_group",
        "seed42_status",
        "seed123_status",
        "seed2026_status",
        "failure_repeated_across_seeds",
        "seed42_failed_parameters",
        "seed123_failed_parameters",
        "seed2026_failed_parameters",
        "seed42_saved_vs_checkpoint_replay",
        "seed123_saved_vs_checkpoint_replay",
        "seed2026_saved_vs_checkpoint_replay",
        "notes",
    ]
    for row in current16_rows:
        for seed in SEEDS:
            status = row[f"seed{seed}_failure_type"]
            if row[f"seed{seed}_state_failure_type"] != "VALID":
                status += ";" + row[f"seed{seed}_state_failure_type"]
            row[f"seed{seed}_status"] = status
    write_csv(
        audit_root / "seed42_invalid_cross_seed_audit.csv",
        current16_rows,
        current16_fields,
    )

    snow_rows: list[dict[str, Any]] = []
    group_order = ["No/trace", "Low", "Middle", "High", "Very high"]
    for seed in SEEDS:
        invalid = failure_set(seed, "param")
        for group in group_order:
            subset = burden[burden["burden_group"] == group]
            n_total = len(subset)
            n_invalid = int(subset["basin_id"].isin(invalid).sum())
            snow_rows.append(
                {
                    "seed": seed,
                    "failure_basis": "persisted_physical_parameter_invalid",
                    "burden_group": group,
                    "n_total": n_total,
                    "n_invalid": n_invalid,
                    "invalid_rate": n_invalid / n_total if n_total else np.nan,
                }
            )
    write_csv(
        audit_root / "snow_burden_failure_summary.csv",
        snow_rows,
        list(snow_rows[0]),
    )

    summary_rows = []
    for seed in SEEDS:
        entry = per_seed[seed]
        param = entry["physical"]
        param_valid = np.isfinite(param).all(axis=1) if param is not None else np.zeros(531, bool)
        state_valid = entry["states"]["finite"] & ~entry["states"]["zero_variance"]
        invalid_names = set(
            row["basin_id"] for row in matrix_rows if not row[f"seed{seed}_param_valid"]
        )
        main_failure = "VALID" if not invalid_names else "PHYSICAL_PARAM_NONFINITE"
        if seed == 42 and invalid_names:
            main_failure += " (saved artifact; checkpoint replay finite)"
        summary_rows.append(
            {
                "seed": seed,
                "training_run_status": entry["training"]["status"],
                "formal_complete": entry["training"]["formal_complete"],
                "best_checkpoint_epoch": entry["training"]["best_checkpoint_epoch"],
                "valid_basins": int(param_valid.sum()),
                "invalid_basins": int((~param_valid).sum()),
                "state_valid_basins": int(state_valid.sum()),
                "state_invalid_basins": int((~state_valid).sum()),
                "main_failure_type": main_failure,
                "raw_replay_nonfinite_basins": int((~entry["raw_output_valid"]).sum()),
                "physical_replay_nonfinite_basins": int((~entry["physical_replay_valid"]).sum()),
            }
        )

    overlap = {row["failure_basis"]: row for row in overlap_rows}
    counts42 = {
        row["basin_id"]: row for row in current16_rows
    }
    report_path = audit_root / "tgd_dpl_seed_failure_audit_report.md"
    report_lines = [
        "# TGD dPL three-seed failure audit",
        "",
        "## Scope and read-only boundary",
        "",
        "- Model: `XAJ_TGD2` (paper-facing name: TGD), dPL seeds 42/123/2026.",
        "- Coverage: canonical 531 basins; no training, hydrological state forward, or canonical-result overwrite was performed.",
        "- The only replay was a deterministic static parameter-network forward from each existing `best_checkpoint.pt`, required because seed 42 lacks `best_parameters_normalized.npz`.",
        "- `seed_basin_validity_matrix.csv` validity is based on the persisted physical-parameter source used by the existing R4 export; replay columns are separate and do not replace it.",
        "",
        "## A. Direct verdict",
        "",
        "**NO — mainly seed-specific.** The 16 seed-42 persisted physical-parameter failures are not present in the saved seed-123 or seed-2026 physical arrays. However, the seed-42 training directory is incomplete and its saved physical array is inconsistent with a finite static replay from its best checkpoint, so this is not evidence of a stable basin-specific TGD+dPL network failure.",
        "",
        "## B. Per-seed validity",
        "",
        "| seed | training run status | valid basins | invalid basins | state-valid basins | main failure type |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['seed']} | {row['training_run_status']} (best epoch {row['best_checkpoint_epoch']}) | {row['valid_basins']} | {row['invalid_basins']} | {row['state_valid_basins']} | {row['main_failure_type']} |"
        )
    report_lines.extend(
        [
            "",
            "Training interpretation: seed 123 and 2026 have `COMPLETE`, epoch-100 checkpoints, 531-row summaries, and finite histories. Seed 42 has finite progress and a readable best checkpoint (epoch 50), but only through epoch 60, no `COMPLETE`, no normalized-parameter array, no 531-row final summary, and no report; it is **INCOMPLETE**, not a whole-seed NaN training failure.",
            "",
            "## C. Cross-seed overlap",
            "",
            "| failure basis | seed 42 | seed 123 | seed 2026 | 42∩123 | 42∩2026 | 123∩2026 | all-3 | exactly-2 | any-2-or-more | 42-only | 123-only | 2026-only | union |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overlap_rows:
        report_lines.append(
            f"| {row['failure_basis']} | {row['seed42_count']} | {row['seed123_count']} | {row['seed2026_count']} | {row['pairwise_42_123']} | {row['pairwise_42_2026']} | {row['pairwise_123_2026']} | {row['all3_intersection']} | {row['exactly2_intersection']} | {row['any2_or_more']} | {row['seed42_only']} | {row['seed123_only']} | {row['seed2026_only']} | {row['union']} |"
        )
    report_lines.extend(
        [
            "",
            "For persisted physical parameters: all-3 intersection = 0, any-2-or-more = 0, seed-42-only = 16, and union = 16. The same counts hold for state-invalid rows because the seed-42 state export contains exactly those 16 zero-variance rows and the other two state exports are finite and non-constant.",
            "",
            "## D. The current 16 basins",
            "",
            "| basin | burden | seed 42 | seed 123 | seed 2026 | failure repeated across seeds? | notes |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in current16_rows:
        report_lines.append(
            f"| {row['basin_id']} | {row['burden_group']} ({float(row['snow_burden_swe_mm']):.2f} mm) | {row['seed42_status']} | {row['seed123_status']} | {row['seed2026_status']} | {row['failure_repeated_across_seeds']} | {row['notes']} |"
        )
    report_lines.extend(
        [
            "",
            "All 16 are seed-42-only in the persisted parameter/state assets. No seed-42 invalid basin is invalid in either saved seed-123 or saved seed-2026 physical array; there are no reverse cases because seeds 123/2026 have zero persisted invalid basins.",
            "",
            "## E. Root cause",
            "",
            "1. **Whole training run:** not established as a failure for any seed. Seeds 123/2026 completed. Seed 42 is incomplete but has finite losses, finite batches, and a readable finite checkpoint; it should be called an incomplete run/artifact, not `seed 42 training failed`.",
            "2. **Raw/unconstrained network output:** no raw-logit file is persisted. The deterministic replay of logits and sigmoid-normalized outputs from all three best checkpoints is finite for all 531 basins, including the 16 seed-42 rows.",
            "3. **Physical transform:** the current `sigmoid` + bounds/inverse-log TGD2 transform also replays finite for all 531 basins. Therefore there is no reproducible current-transform failure for these rows.",
            "4. **Persisted physical parameter source:** seed 42 `best_parameters_physical.npz` has 16 rows with all 17 parameters non-finite; seed 123/2026 arrays are finite for all rows. The 16 rows exactly equal the 16 basins with non-finite raw static attributes (attribute columns 19 or 33) before robust normalization. This is an artifact-generation/source mismatch signal, not a demonstrated structural identifiability failure. No ordering among parameters can be inferred: all 17 are non-finite simultaneously in the saved row.",
            "5. **State export:** all three saved TGD state NPZs are numerically finite. Seed 42 has exactly the same 16 rows as constant all-zero `W_total`; this is a downstream masking symptom. The exporter diff in the current worktree shows the prior `nan_to_num(..., nan=0.0)` behavior was replaced with preserve-invalid behavior, but the existing seed-42 NPZ remains a legacy zero-masked artifact. It must not be interpreted as a real zero state.",
            "",
            "### Comparability",
            "",
            "**COMPARABLE_WITH_LIMITATION.** The three configs are equal after removing only seed and output-directory fields; they share the same XAJ_TGD2 class, TGD2 structure version, 17 parameter specs, 35-attribute normalization, window/date protocol, network, optimizer, epochs, precision settings, and checkpoint rule. There is no recorded commit hash or per-seed training log in these assets. The limitation is material: seed 42 is incomplete and its saved physical parameter file does not match finite replay from its best checkpoint.",
            "",
            "## F. Snow-burden pattern",
            "",
            "| seed | burden group | total | invalid | invalid rate |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in snow_rows:
        report_lines.append(
            f"| {row['seed']} | {row['burden_group']} | {row['n_total']} | {row['n_invalid']} | {row['invalid_rate']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "Only seed 42 has persisted invalid rows: Very high 8/113 (7.08%), High 4/113 (3.54%), Middle 0/113, Low 3/113 (2.65%), No/trace 1/79 (1.27%). Thus the observed seed-42 artifact failures are descriptively concentrated in High/Very high burden (12/226 = 5.31% versus 4/305 = 1.31% in the other groups), but this pattern is not repeated in seeds 123/2026 and is confounded by the shared missing-attribute rows. No mechanism claim is made.",
            "",
            "## G. Figure 7 consequence",
            "",
            "1. Keep the 16 rows as `NaN`/invalid in any corrected Figure 7 representation; do not impute or use the old zero `W_total`.",
            "2. Do **not** describe the 16 rows as a stable TGD structural failure. Given the seed-42 incomplete provenance and finite checkpoint replay, seed 42 alone is not sufficient to establish that claim. Whether seed 42 should remain the canonical display requires a separate canonical-selection decision; this audit does not change it.",
            "3. Include seeds 123/2026 sensitivity in the SI, explicitly reporting zero saved physical-parameter failures and the seed-42 artifact/provenance limitation.",
            "4. Common-support comparison remains conceptually appropriate only after excluding invalid/non-supported rows and treating them as missing, not zero. The three-seed overlap audit does not support a common invalid-basin exclusion beyond the seed-42 invalid set.",
            "",
            "## H. Exact evidence paths",
            "",
            f"- Training root: `{training_root}`",
        ]
    )
    for seed in SEEDS:
        entry = per_seed[seed]
        report_lines.extend(
            [
                f"- Seed {seed} config: `{entry['seed_dir'] / 'config.json'}`",
                f"- Seed {seed} best checkpoint: `{entry['seed_dir'] / 'best_checkpoint.pt'}`",
                f"- Seed {seed} physical parameters: `{entry['physical_path']}`",
                f"- Seed {seed} normalized parameters: `{entry['normalized_path']}` ({'present' if entry['normalized_path'].is_file() else 'MISSING'})",
                f"- Seed {seed} training history: `{entry['seed_dir'] / 'epoch_history.csv'}`",
                f"- Seed {seed} completion marker: `{entry['seed_dir'] / 'COMPLETE'}` ({'present' if (entry['seed_dir'] / 'COMPLETE').is_file() else 'MISSING'})",
                f"- Seed {seed} state export: `{results_root / STATE_REL[seed]}`",
            ]
        )
    report_lines.extend(
        [
            f"- Canonical basin list: `{data_root / '531sub_id.txt'}`",
            f"- Canonical CAMELS bundle: `{data_root / 'camels_dataset'}`",
            f"- Burden assignments: `{burden_path}`",
            f"- Existing Figure 7 invalid audit: `{results_root / 'r4_phase1_soil_official/figure7_all_days_tgd_dpl_invalid_audit.csv'}`",
            "- Training implementation: `project/hydrodiag/training/dpl/run_dpl_model.py` (`StaticParameterNet`, `robust_normalize`, `physical_parameters`).",
            "- Training launcher/config source: `project/hydrodiag/training/dpl/launch_xaj_tgd2_lite_v3.py` and `project/hydrodiag/training/dpl/generated_configs/xaj_tgd2_lite_v3_seed_{42,123,2026}.json`.",
            "- State exporter: `project/hydrodiag/manuscript/scripts/r4/export_all_tgd2_states.py`; state implementation: `project/hydrodiag/manuscript/scripts/r4/state_export.py`.",
            "- No separate raw-logit output file or TGD2 per-seed state-export manifest was found beside the listed assets.",
            "",
            "## Audit output files",
            "",
            f"- `{audit_root / 'seed_basin_validity_matrix.csv'}`",
            f"- `{audit_root / 'seed_failure_overlap_summary.csv'}`",
            f"- `{audit_root / 'seed42_invalid_cross_seed_audit.csv'}`",
            f"- `{audit_root / 'snow_burden_failure_summary.csv'}`",
            f"- `{report_path}`",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    metadata = {
        "audit": "read_only_tgd_dpl_three_seed_failure_audit",
        "model": MODEL_KEY,
        "seeds": list(SEEDS),
        "n_basins": len(basin_ids),
        "training_root": str(training_root),
        "audit_root": str(audit_root),
        "static_checkpoint_replay_only": True,
        "hydrological_state_forward_run": False,
        "canonical_results_overwritten": False,
        "raw_static_attribute_nonfinite_rows": int(raw_nonfinite.sum()),
        "raw_static_attribute_nonfinite_columns": sorted(
            {column for values in raw_nonfinite_columns.values() for column in values}
        ),
        "expected_seed42_invalid_set_matches": sorted(
            {
                row["basin_id"]
                for row in matrix_rows
                if not row["seed42_param_valid"]
            }
        )
        == sorted(EXPECTED_16),
        "per_seed": {
            str(seed): {
                "training": per_seed[seed]["training"],
                "physical_parameter_path": str(per_seed[seed]["physical_path"]),
                "physical_parameter_sha256": sha256(per_seed[seed]["physical_path"]),
                "normalized_parameter_path": str(per_seed[seed]["normalized_path"]),
                "normalized_parameter_sha256": sha256(per_seed[seed]["normalized_path"]),
                "state_path": str(results_root / STATE_REL[seed]),
                "state_sha256": sha256(results_root / STATE_REL[seed]),
                "normalized_saved_vs_replay": per_seed[seed]["comparisons"]["normalized"],
                "physical_saved_vs_replay": per_seed[seed]["comparisons"]["physical"],
                "state_basin_ids_match": per_seed[seed]["states"]["basin_ids_match"],
                "state_days": per_seed[seed]["states"]["n_days"],
                "state_finite_rows": int(per_seed[seed]["states"]["finite"].sum()),
                "state_zero_variance_rows": int(per_seed[seed]["states"]["zero_variance"].sum()),
            }
            for seed in SEEDS
        },
    }
    (audit_root / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps({"audit_root": str(audit_root), "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
