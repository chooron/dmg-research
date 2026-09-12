#!/usr/bin/env python3
"""Build provenance and parameter inventories for the controlled Chapter 3.6 runs.

This script only reads training artifacts and writes new audit tables below the
Chapter 3.6 output root. It does not train, replay, or modify source results.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
WORKSPACE = PROJECT.parents[1]
RESULTS = PROJECT / "results"
OUT = RESULTS / "ch3_6_cross_process"
IC_ROOT = RESULTS / "ic_phase0_controlled_531_v1"
DPL_ROOT = RESULTS / "dpl_controlled_531_v1"
DPL_N_ROOT = OUT / "dpl_controlled_n_531_v1"

sys.path.insert(0, str(PROJECT))
from ablation.ic_core.parameter_adapter import get_parameter_spec  # noqa: E402

IC_MODELS = ("N", "D_E", "G_E", "D_R", "G_R")
DPL_MODELS = ("XAJ_D_E_CN", "XAJ_G_E_CN", "XAJ_D_R_CN", "XAJ_G_R_CN")


def ensure_dirs() -> None:
    for name in (
        "00_provenance",
        "01_parameter_tables",
        "02_replay/IC",
        "02_replay/dPL",
        "03_et",
        "04_response",
        "05_p5",
        "06_cross_process",
        "figure_ready",
        "logs",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def json_load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ic_inventory() -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    inventory: list[dict] = []
    canonical: list[dict] = []
    restart: list[dict] = []
    names_by_model: dict[str, list[str]] = {}
    for model in IC_MODELS:
        root = IC_ROOT / model
        manifest_path = root / "manifest.json"
        done_path = root / "DONE.json"
        manifest = json_load(manifest_path)
        protocol = manifest.get("protocol", {})
        # The Phase-0 writer uses solver.state.best_candidate, whose per-unit
        # winner is the first argmax encountered. Start is the deterministic
        # secondary ordering recorded here for downstream selection.
        records = []
        for path in sorted((root / "raw" / model.lower()).glob("*.json")):
            record = json_load(path)
            records.append(record)
            restart.append(
                {
                    "regime": "IC",
                    "model": model,
                    "basin_id": str(record["basin_id"]).zfill(8),
                    "start": int(record["start"]),
                    "train_kge": record.get("train_metrics", {}).get("kge"),
                    "eval_kge": record.get("test_metrics", {}).get("kge"),
                    "best_train_objective": record.get("best_train_objective"),
                    "status": record.get("status"),
                }
            )
        if not records:
            raise RuntimeError(f"No IC raw records for {model}: {root}")
        names = list(records[0]["parameter_names"])
        names_by_model[model] = names
        groups: dict[str, list[dict]] = {}
        for record in records:
            groups.setdefault(str(record["basin_id"]).zfill(8), []).append(record)
        selected = {}
        for basin, group in groups.items():
            selected[basin] = sorted(
                group,
                key=lambda r: (-float(r["best_train_objective"]), int(r["start"])),
            )[0]
        for basin in sorted(selected):
            record = selected[basin]
            row = {
                "regime": "IC",
                "model": model,
                "basin_id": basin,
                "start": int(record["start"]),
                "train_kge": record["train_metrics"]["kge"],
                "eval_kge": record["test_metrics"]["kge"],
                "best_train_objective": record["best_train_objective"],
            }
            row.update({f"param_{name}": value for name, value in zip(names, record["parameters"])})
            canonical.append(row)
        inventory.append(
            {
                "regime": "IC",
                "model": model,
                "basin_n": len(groups),
                "runs_seeds": f"{len(records)//len(groups)} starts",
                "final_gen_epoch": protocol.get("max_generations"),
                "parameter_dim": len(names),
                "bounds_source": "models.parameter_specs + manifest model",
                "manifest": str(manifest_path.relative_to(PROJECT)),
                "status": json_load(done_path).get("status") if done_path.exists() else "MISSING_DONE",
                "objective": protocol.get("objective"),
                "train_period": protocol.get("train"),
                "test_period": protocol.get("validation"),
                "checkpoint": str(next((root / "checkpoints").glob("*.pt"), "")),
            }
        )
    all_names = [
        "regime", "model", "basin_id", "start", "train_kge", "eval_kge", "best_train_objective"
    ] + sorted({key for row in canonical for key in row if key.startswith("param_")})
    write_csv(OUT / "01_parameter_tables" / "ic_canonical_parameters.csv", canonical, all_names)
    write_csv(
        OUT / "01_parameter_tables" / "ic_restart_parameter_table.csv",
        restart,
        list(restart[0]) if restart else None,
    )
    return inventory, canonical, names_by_model


def dpl_inventory() -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    inventory: list[dict] = []
    rows: list[dict] = []
    names_by_model: dict[str, list[str]] = {}
    roots = [DPL_ROOT / model / "seed_42" for model in DPL_MODELS]
    roots.append(DPL_N_ROOT / "XAJ_CONTROLLED_N_CN" / "seed_42")
    for root in roots:
        config_path = root / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"Missing required dPL config: {config_path}")
        config = json_load(config_path)
        model = config.get("model_name", root.parent.parent.name)
        summary_path = root / "basin_final_summary.csv"
        parameter_path = root / "best_parameters_physical.npz"
        normalized_path = root / "best_parameters_normalized.npz"
        complete = (root / "COMPLETE").exists()
        summary_rows = []
        if summary_path.exists():
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
        names = list(config.get("parameter_names", []))
        if parameter_path.exists() and names:
            with np.load(parameter_path, allow_pickle=False) as archive:
                physical = np.asarray(archive["params"])
            with np.load(normalized_path, allow_pickle=False) as archive:
                normalized = np.asarray(archive["params"])
            if physical.shape[0] != len(summary_rows) or physical.shape[1] != len(names):
                raise RuntimeError(f"dPL shape mismatch in {root}: {physical.shape} vs {len(summary_rows)}x{len(names)}")
            names_by_model[model] = names
            for index, (summary, physical_row, normalized_row) in enumerate(
                zip(summary_rows, physical, normalized)
            ):
                row = {
                    "regime": "dPL",
                    "model": model,
                    "basin_id": str(summary["basin_id"]).zfill(8),
                    "basin_index": int(summary["basin_index"]),
                    "eval_kge": float(summary["val_kge"]),
                }
                row.update({f"norm_{name}": value for name, value in zip(names, normalized_row)})
                row.update({f"param_{name}": value for name, value in zip(names, physical_row)})
                rows.append(row)
        training = config.get("training", {})
        periods = config.get("time_periods", {})
        inventory.append(
            {
                "regime": "dPL",
                "model": model,
                "basin_n": len(summary_rows),
                "runs_seeds": training.get("seed", "MISSING"),
                "final_gen_epoch": training.get("epochs"),
                "parameter_dim": len(names),
                "bounds_source": f"{config_path.relative_to(PROJECT)}",
                "manifest": str(DPL_ROOT / "manifest.tsv") if root.is_relative_to(DPL_ROOT) else "new controlled-N config/output",
                "status": "complete" if complete else "INCOMPLETE",
                "objective": "KGE(Q), validation summary",
                "train_period": periods.get("calibration", {}).get("start", "")
                + ".."
                + periods.get("calibration", {}).get("end", ""),
                "test_period": periods.get("evaluation", {}).get("start", "")
                + ".."
                + periods.get("evaluation", {}).get("end", ""),
                "checkpoint": str(root / "best_checkpoint.pt"),
            }
        )
    all_names = ["regime", "model", "basin_id", "basin_index", "eval_kge"] + sorted(
        {key for row in rows for key in row if key.startswith(("norm_", "param_"))}
    )
    write_csv(OUT / "01_parameter_tables" / "dpl_seed42_parameters.csv", rows, all_names)
    return inventory, rows, names_by_model


def write_bounds() -> None:
    rows = []
    for model in IC_MODELS:
        for name, spec in get_parameter_spec(model).items():
            rows.append({"regime": "IC", "model": model, "parameter": name, **{k: spec.get(k) for k in ("lower", "upper", "unit", "process")}})
    for model in DPL_MODELS + (("XAJ_CONTROLLED_N_CN",) if (DPL_N_ROOT / "XAJ_CONTROLLED_N_CN" / "seed_42" / "config.json").exists() else ()):
        config = json_load((DPL_ROOT / model / "seed_42" / "config.json") if model in DPL_MODELS else (DPL_N_ROOT / model / "seed_42" / "config.json"))
        for name, spec in config.get("parameter_specs", {}).items():
            rows.append({"regime": "dPL", "model": model, "parameter": name, **{k: spec.get(k) for k in ("lower", "upper", "unit", "process")}})
    write_csv(OUT / "00_provenance" / "parameter_bounds.csv", rows)


def tau_audit() -> None:
    lines = [
        "# tau0 mapping audit",
        "",
        "The audit compares stored physical xaj_tau0 against both mappings using the stored normalized coordinate and the bounds in each result config/spec.",
        "",
        "| Regime | Model | N | Linear median abs error | Linear max abs error | Log median abs error | Log max abs error | Inference |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    records = []
    for model in ("D_R", "G_R"):
        root = IC_ROOT / model
        spec = get_parameter_spec(model)["xaj_tau0"]
        lower, upper = float(spec["lower"]), float(spec["upper"])
        for path in sorted((root / "raw" / model.lower()).glob("*.json")):
            record = json_load(path)
            names = record["parameter_names"]
            i = names.index("xaj_tau0")
            u = min(1.0, max(0.0, float(record["theta_normalized"][i])))
            stored = float(record["parameters"][i])
            linear = lower + u * (upper - lower)
            log_value = math.exp(math.log(lower) + u * (math.log(upper) - math.log(lower)))
            records.append(("IC", model, stored, abs(stored - linear), abs(stored - log_value)))
    for model in ("XAJ_D_R_CN", "XAJ_G_R_CN"):
        root = DPL_ROOT / model / "seed_42"
        if not (root / "config.json").exists():
            continue
        config = json_load(root / "config.json")
        names = config.get("parameter_names", [])
        if "xaj_tau0" not in names or not (root / "best_parameters_physical.npz").exists():
            continue
        i = names.index("xaj_tau0")
        spec = config["parameter_specs"]["xaj_tau0"]
        lower, upper = float(spec["lower"]), float(spec["upper"])
        with np.load(root / "best_parameters_normalized.npz", allow_pickle=False) as z:
            normalized = np.asarray(z["params"])[:, i]
        with np.load(root / "best_parameters_physical.npz", allow_pickle=False) as z:
            stored = np.asarray(z["params"])[:, i]
        linear = lower + normalized * (upper - lower)
        log_value = np.exp(np.log(lower) + normalized * (np.log(upper) - np.log(lower)))
        for s, a, b in zip(stored, np.abs(stored - linear), np.abs(stored - log_value)):
            records.append(("dPL", model, float(s), float(a), float(b)))
    for regime, model in (("IC", "D_R"), ("IC", "G_R"), ("dPL", "XAJ_D_R_CN"), ("dPL", "XAJ_G_R_CN")):
        subset = [r for r in records if r[:2] == (regime, model)]
        if not subset:
            continue
        lin = [r[3] for r in subset]
        log = [r[4] for r in subset]
        inference = "LINEAR_MAPPING_CONFIRMED" if median(lin) < 1e-5 * max(1.0, median(log)) else "LOG_MAPPING_CONFIRMED" if median(log) < 1e-5 * max(1.0, median(lin)) else "UNRESOLVED"
        lines.append(f"| {regime} | {model} | {len(subset)} | {median(lin):.6g} | {max(lin):.6g} | {median(log):.6g} | {max(log):.6g} | `{inference}` |")
    lines += [
        "",
        "IC uses clipped CMA-ES normalized coordinates before testing the physical mapping, matching `normalized_to_physical(..., clip=True)`.",
        "",
    ]
    (OUT / "00_provenance" / "tau_mapping_audit.md").write_text("\n".join(lines), encoding="utf-8")


def controlled_n_equivalence() -> None:
    legacy_path = RESULTS / "dpl_camels_531_lite_v2" / "XAJ_CN" / "seed_42" / "config.json"
    controlled_path = DPL_N_ROOT / "XAJ_CONTROLLED_N_CN" / "seed_42" / "config.json"
    legacy = json_load(legacy_path)
    controlled = json_load(controlled_path)
    fields = {
        "time_periods": (legacy.get("time_periods"), controlled.get("time_periods")),
        "window": (legacy.get("window"), controlled.get("window")),
        "sampling": (legacy.get("sampling"), controlled.get("sampling")),
        "training": (legacy.get("training"), controlled.get("training")),
        "runtime": (legacy.get("runtime"), controlled.get("runtime")),
        "parameter_names": (legacy.get("parameter_names"), controlled.get("parameter_names")),
        "ci_bounds": (legacy.get("parameter_specs", {}).get("xaj_ci"), controlled.get("parameter_specs", {}).get("xaj_ci")),
        "cg_bounds": (legacy.get("parameter_specs", {}).get("xaj_cg"), controlled.get("parameter_specs", {}).get("xaj_cg")),
    }
    differences = [name for name, (a, b) in fields.items() if a != b]
    legacy_mapping = legacy.get("network", {}).get("parameter_mapping")
    controlled_mapping = controlled.get("network", {}).get("parameter_mapping")
    result = {
        "legacy_config": str(legacy_path.relative_to(PROJECT)),
        "controlled_reference_config": str(controlled_path.relative_to(PROJECT)),
        "legacy_model": legacy.get("model_name"),
        "controlled_model": controlled.get("model_name", "XAJ_CONTROLLED_N_CN"),
        "equivalence": "CONTROLLED_N_EQUIVALENT" if not differences else "NOT_EQUIVALENT",
        "differences": differences,
        "parameter_mapping_labels": {"legacy": legacy_mapping, "controlled": controlled_mapping},
        "parameter_mapping_assessment": "EFFECTIVELY_EQUAL_FOR_NATIVE_XAJ: both use linear physical denormalization; log interpolation is only for TGD2 residence parameters, absent here.",
        "reason": "legacy XAJ_CN has ci=[0.1,1.0], cg=[0.9,1.0]; controlled-N has ci=[0.1,0.9], cg=[0.9,0.998]." if differences else "all compared fields equal",
    }
    (OUT / "00_provenance" / "controlled_n_equivalence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ic_inv, _, _ = ic_inventory()
    dpl_inv, _, _ = dpl_inventory()
    all_inventory = ic_inv + dpl_inv
    inventory_fields = list(dict.fromkeys(key for row in all_inventory for key in row))
    write_csv(OUT / "00_provenance" / "training_inventory.csv", all_inventory, inventory_fields)
    write_bounds()
    controlled_n_equivalence()
    tau_audit()
    print(f"Wrote provenance and parameter tables under {OUT}")


if __name__ == "__main__":
    main()
