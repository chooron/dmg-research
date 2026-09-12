#!/usr/bin/env python3
"""Stage A R4 audit and seen-basin case freeze.

This stage deliberately never reads an OOB relationship coefficient table.  It audits
completed OOB job evidence, reads only the frozen seen-basin IC/dPL atlas and its
secondary attribute-redundancy definitions, then writes an immutable case manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# R4 is intentionally single-threaded and must not inherit a BLAS thread pool.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[5]
BENCHMARK = REPO / "project/benchmark"
R4 = BENCHMARK / "manuscript/r4"
FORMAL = BENCHMARK / "results/oob_primary8_5fold_20260902"
SEEN = BENCHMARK / "results/ic_dpl_seenbasin_formal_20260901"
CLUSTERS = BENCHMARK / "results/parameter_attribute_atlas_followup_20260829/attribute_redundancy_groups.csv"
ATLAS_SCRIPT = BENCHMARK / "scripts/diagnostics/parameter_attribute_atlas.py"
PRIMARY8 = ("alpine2", "hbv96", "xinanjiang", "newzealand2", "ihacres", "us1", "mopex4", "hillslope")
N_FOLDS = 5
EXPECTED_BASINS = 531
CLASS_ORDER = ("persistent/reproduced", "attenuated", "dPL-emergent", "sign-changing")

EXPECTED_PROTOCOL = {
    "training_warmup_days": 730,
    "scored_horizon_days": 365,
    "evaluation_warmup_days": 365,
    "parameter_mapping": "auto",
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "scheduler": "None",
    "clip_norm": 1.0,
    "kge_eps": 0.1,
    "seed": 42,
    "min_epochs": 50,
    "patience": 10,
    "plateau_eps": 0.0001,
    "max_epochs": 100,
    "selection_metric": "train_loss",
    "exact_best_checkpoint": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value {value!r}")
    return result


def check_equal(actual: Any, expected: Any, label: str, failures: list[str]) -> None:
    if isinstance(expected, float):
        ok = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        ok = actual == expected
    if not ok:
        failures.append(f"{label}: expected {expected!r}, got {actual!r}")


def audit_oob() -> dict[str, Any]:
    failures: list[str] = []
    notes: list[str] = []
    assignment_path = FORMAL / "OOB_FOLD_ASSIGNMENT.csv"
    assignment = load_csv(assignment_path)
    assignment_by_id = {int(row["basin_id"]): int(row["fold"]) for row in assignment}
    if len(assignment_by_id) != EXPECTED_BASINS or len(assignment) != EXPECTED_BASINS:
        failures.append(f"fold assignment rows/IDs are not exactly {EXPECTED_BASINS}")
    fold_sizes = tuple(sum(fold == k for fold in assignment_by_id.values()) for k in range(N_FOLDS))
    if fold_sizes != (107, 106, 106, 106, 106):
        failures.append(f"unexpected fold sizes: {fold_sizes}")

    queue_status_path = FORMAL / "remote_queue_status.json"
    if queue_status_path.exists():
        queue_status = read_json(queue_status_path)
        counts = queue_status.get("counts") or {
            "DONE": int(queue_status.get("completed", 0)),
            "RUNNING": len(queue_status.get("running", [])),
            "QUEUED": int(queue_status.get("queued", 0)),
            "FAILED": int(queue_status.get("failed", 0)),
        }
        if counts != {"DONE": 40, "RUNNING": 0, "QUEUED": 0, "FAILED": 0}:
            failures.append(f"queue status is not complete: {counts}")
    else:
        notes.append("remote_queue_status.json is absent; per-run DONE markers are used")

    held_by_fold: dict[int, set[int]] = {}
    job_count = 0
    retry_lines = 0
    checkpoint_nonfinite = 0
    kge_nonfinite = 0
    expected_files = (
        "config.json", "train_basin_ids.txt", "heldout_basin_ids.txt",
        "attribute_mean.npy", "attribute_std.npy", "normalization_metadata.json",
        "epoch_metrics.csv", "best.pt", "best_metadata.json", "training_health.json",
        "heldout_test_evaluation.json", "heldout_basin_kge.csv", "stdout.log", "stderr.log",
        "source_provenance.json", "result.json", "DONE",
    )
    for model in PRIMARY8:
        model_held: set[int] = set()
        model_folds: set[int] = set()
        for fold in range(N_FOLDS):
            job_count += 1
            run = FORMAL / "runs" / model / f"fold_{fold}"
            missing = [name for name in expected_files if not (run / name).is_file()]
            if missing:
                failures.append(f"{model}_fold{fold} missing files: {','.join(missing)}")
                continue
            config = read_json(run / "config.json")
            if config.get("smoke") is not False:
                failures.append(f"{model}_fold{fold} is not a formal non-smoke run")
            if config.get("models") != list(PRIMARY8):
                failures.append(f"{model}_fold{fold} model list mismatch")
            for key, expected in EXPECTED_PROTOCOL.items():
                check_equal(config["protocol"].get(key), expected, f"{model}_fold{fold} config.protocol.{key}", failures)
            train_ids = {int(line) for line in (run / "train_basin_ids.txt").read_text().split()}
            held_ids = {int(line) for line in (run / "heldout_basin_ids.txt").read_text().split()}
            expected_held = {basin_id for basin_id, assigned_fold in assignment_by_id.items() if assigned_fold == fold}
            expected_train = set(assignment_by_id).difference(expected_held)
            if train_ids != expected_train or held_ids != expected_held or train_ids.intersection(held_ids):
                failures.append(f"{model}_fold{fold} train/heldout partition mismatch")
            model_held.update(held_ids)
            model_folds.add(fold)
            previous_held = held_by_fold.get(fold)
            if previous_held is None:
                held_by_fold[fold] = set(held_ids)
            elif previous_held != held_ids:
                failures.append(f"cross-model fold mismatch at {model}_fold{fold}")

            health = read_json(run / "training_health.json")
            metadata = read_json(run / "best_metadata.json")
            evaluation = read_json(run / "heldout_test_evaluation.json")
            result = read_json(run / "result.json")
            for key in ("min_epochs", "patience", "selection_metric"):
                check_equal(health.get(key), EXPECTED_PROTOCOL[key], f"{model}_fold{fold} health.{key}", failures)
            metadata_keys = {
                "training_warmup_days": "training_warmup_days", "scored_horizon_days": "scored_horizon_days",
                "parameter_mapping": "parameter_mapping",
                "optimizer": "optimizer", "lr": "lr", "weight_decay": "weight_decay", "scheduler": "scheduler",
                "clip_norm": "clip_norm", "kge_eps": "kge_eps", "seed": "seed",
                "selection_metric": "selection_metric_name",
            }
            for key, metadata_key in metadata_keys.items():
                check_equal(metadata.get(metadata_key), EXPECTED_PROTOCOL[key], f"{model}_fold{fold} metadata.{metadata_key}", failures)
            if health.get("status") == "PLATEAU_STOP" and int(health.get("stop_epoch", 0)) < EXPECTED_PROTOCOL["min_epochs"]:
                failures.append(f"{model}_fold{fold} plateau stopped before min_epochs")
            if metadata.get("test_informed_selection") is not False or metadata.get("heldout_target_access_during_training") != 0:
                failures.append(f"{model}_fold{fold} selection/access provenance flags are unsafe")
            if evaluation.get("test_informed_selection") is not False or evaluation.get("heldout_target_access_during_training") != 0:
                failures.append(f"{model}_fold{fold} evaluation provenance flags are unsafe")
            if result.get("test_informed_selection") is not False or result.get("heldout_target_access_during_training") != 0:
                failures.append(f"{model}_fold{fold} result provenance flags are unsafe")
            norm = read_json(run / "normalization_metadata.json")
            if norm.get("scope") != "train_basins_only":
                failures.append(f"{model}_fold{fold} normalization scope is not train_basins_only")
            source = read_json(run / "source_provenance.json")
            if source.get("normalization_scope") != "train_basins_only" or not source.get("monolithic_source_not_opened_by_training_worker", False):
                failures.append(f"{model}_fold{fold} source provenance leakage contract failed")
            if source.get("target_access_policy", "").find("heldout_y.npy opens only after Phase.EVAL") < 0:
                failures.append(f"{model}_fold{fold} target access policy evidence missing")
            if source["files"]["assignment"]["sha256"] != sha256_file(assignment_path):
                failures.append(f"{model}_fold{fold} assignment checksum mismatch")
            for name in ("attribute_mean.npy", "attribute_std.npy"):
                values = np.load(run / name)
                if values.shape != (35,) or not np.isfinite(values).all():
                    failures.append(f"{model}_fold{fold} invalid {name}")
            kge_rows = load_csv(run / "heldout_basin_kge.csv")
            if len(kge_rows) != len(held_ids):
                failures.append(f"{model}_fold{fold} heldout KGE row count mismatch")
            for row in kge_rows:
                try:
                    finite_float(row["kge"])
                    int(row["basin_id"])
                except (ValueError, KeyError):
                    kge_nonfinite += 1
            payload = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            for value in payload.get("network", {}).values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    checkpoint_nonfinite += 1
            del payload
        if len(model_held) != EXPECTED_BASINS:
            failures.append(f"{model} OOF coverage is {len(model_held)}/{EXPECTED_BASINS}")
        if len(model_folds) != N_FOLDS:
            failures.append(f"{model} completed folds are {sorted(model_folds)}")

    queue_log = FORMAL.parent.parent / "../../oob_deploy_20260902/unused"  # never read as evidence
    del queue_log
    remote_queue_log = FORMAL / "remote_queue.log"
    if remote_queue_log.exists():
        retry_lines = sum(1 for line in remote_queue_log.read_text(errors="replace").splitlines() if " RETRY " in line)
    if retry_lines:
        notes.append(f"queue log contains {retry_lines} retry line(s); all run configs were re-audited against the frozen protocol")
    if checkpoint_nonfinite:
        failures.append(f"checkpoint non-finite tensor count={checkpoint_nonfinite}")
    if kge_nonfinite:
        failures.append(f"heldout KGE non-finite/invalid row count={kge_nonfinite}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "notes": notes,
        "jobs": job_count,
        "models": len(PRIMARY8),
        "folds_per_model": N_FOLDS,
        "fold_sizes": list(fold_sizes),
        "oof_basins_per_model": EXPECTED_BASINS if not failures else None,
        "retry_lines": retry_lines,
        "checkpoint_nonfinite_tensors": checkpoint_nonfinite,
        "kge_invalid_rows": kge_nonfinite,
        "assignment_sha256": sha256_file(assignment_path),
        "assignment_path": str(assignment_path),
    }


def rank_min_desc(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: (-abs(item[1]), item[0]))
    result: dict[str, int] = {}
    last_abs: float | None = None
    current_rank = 0
    for index, (key, value) in enumerate(ordered, start=1):
        if last_abs is None or not math.isclose(abs(value), last_abs, rel_tol=0.0, abs_tol=1e-15):
            current_rank = index
            last_abs = abs(value)
        result[key] = current_rank
    return result


def load_seen_pairs() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    atlas = load_csv(SEEN / "09_PARAMETER_ATTRIBUTE_ATLAS_LONG.csv")
    rows = [row for row in atlas if row["attribute_type"] == "CONTINUOUS"]
    by_method: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_method[(row["model"], row["method"] + "\0" + row["parameter_index"] + "\0" + row["parameter"])][row["attribute"]] = finite_float(row["rho"])
    ranks: dict[tuple[str, str], dict[str, int]] = {key: rank_min_desc(values) for key, values in by_method.items()}
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["parameter_index"], row["parameter"], row["attribute"])
        method_key = (row["model"], row["method"] + "\0" + row["parameter_index"] + "\0" + row["parameter"])
        by_key.setdefault(key, {"model": row["model"], "parameter_index": int(row["parameter_index"]), "parameter": row["parameter"], "attribute": row["attribute"], "attribute_type": row["attribute_type"], "n": int(row["n"]), "strict_full300": row["strict_full300"] == "True"})
        by_key[key]["rho_ic" if row["method"] == "IC" else "rho_dpl_seen"] = finite_float(row["rho"])
        by_key[key]["abs_rho_ic" if row["method"] == "IC" else "abs_rho_dpl"] = abs(finite_float(row["rho"]))
        by_key[key]["rank_ic" if row["method"] == "IC" else "rank_dpl"] = ranks[method_key][row["attribute"]]
    pairs = list(by_key.values())
    for pair in pairs:
        pair["delta_abs_rho"] = pair["abs_rho_dpl"] - pair["abs_rho_ic"]
        pair["same_sign"] = pair["rho_ic"] != 0.0 and pair["rho_dpl_seen"] != 0.0 and (pair["rho_ic"] > 0) == (pair["rho_dpl_seen"] > 0)
        pair["both_nontrivial"] = pair["abs_rho_ic"] >= 0.10 and pair["abs_rho_dpl"] >= 0.10
        if pair["both_nontrivial"] and not pair["same_sign"]:
            pair["relationship_class"] = "sign-changing"
        elif pair["abs_rho_ic"] >= 0.20 and pair["abs_rho_dpl"] < 0.10:
            pair["relationship_class"] = "attenuated"
        elif pair["abs_rho_dpl"] >= 0.20 and pair["abs_rho_ic"] < 0.10:
            pair["relationship_class"] = "dPL-emergent"
        elif pair["abs_rho_ic"] >= 0.20 and pair["abs_rho_dpl"] >= 0.20 and pair["same_sign"] and pair["rank_ic"] <= 10 and pair["rank_dpl"] <= 10:
            pair["relationship_class"] = "persistent/reproduced"
        else:
            pair["relationship_class"] = "weak/unresolved"
    return pairs, by_key


def load_boundary_flags() -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = load_csv(SEEN / "05_PARAMETER_ESTIMATES_LONG.csv")
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["model"] not in PRIMARY8 or row["method"] not in {"IC", "dPL"}:
            continue
        try:
            groups[(row["model"], row["method"], row["parameter"])].append(finite_float(row["normalized_u"]))
        except (KeyError, ValueError):
            pass
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, values in groups.items():
        a = np.asarray(values, dtype=float)
        result[key] = {
            "n": int(a.size),
            "boundary_low_fraction": float(np.mean(a <= 0.01)),
            "boundary_high_fraction": float(np.mean(a >= 0.99)),
            "near_constant": bool(np.std(a) <= 0.01),
            "boundary_concentrated": bool(np.mean(a <= 0.01) >= 0.50 or np.mean(a >= 0.99) >= 0.50),
        }
    return result


def load_clusters() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in load_csv(CLUSTERS):
        if not math.isclose(float(row["threshold"]), 0.70, rel_tol=0.0, abs_tol=1.0e-12) or row["attribute_type"] != "CONTINUOUS":
            continue
        result[row["attribute"]] = {"cluster_id": row["cluster_id"], "representative_attribute": row["representative_attribute"], "members": row["member_attributes"]}
    return result


def freeze_cases() -> dict[str, Any]:
    pairs, _ = load_seen_pairs()
    primary = [pair for pair in pairs if pair["model"] in PRIMARY8 and pair["strict_full300"] and pair["n"] == EXPECTED_BASINS]
    boundary = load_boundary_flags()
    clusters = load_clusters()
    all_persistent_models: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        if pair["strict_full300"] and pair["relationship_class"] == "persistent/reproduced":
            all_persistent_models[pair["attribute"]].add(pair["model"])
    for pair in primary:
        pair["persistent_attribute_model_count"] = len(all_persistent_models[pair["attribute"]])
        ic_flag = boundary.get((pair["model"], "IC", pair["parameter"]), {})
        dpl_flag = boundary.get((pair["model"], "dPL", pair["parameter"]), {})
        pair["seen_boundary_concentrated"] = bool(ic_flag.get("boundary_concentrated", False) or dpl_flag.get("boundary_concentrated", False))
        pair["seen_near_constant"] = bool(ic_flag.get("near_constant", False) or dpl_flag.get("near_constant", False))
        cluster = clusters.get(pair["attribute"], {"cluster_id": f"SINGLETON_{pair['attribute']}", "representative_attribute": pair["attribute"], "members": pair["attribute"]})
        pair.update({"cluster_0.70": cluster["cluster_id"], "cluster_representative_0.70": cluster["representative_attribute"], "cluster_members_0.70": cluster["members"]})
    eligible = [pair for pair in primary if not pair["seen_boundary_concentrated"] and not pair["seen_near_constant"]]
    chosen: list[dict[str, Any]] = []
    for cls in CLASS_ORDER:
        candidates = [pair for pair in eligible if pair["relationship_class"] == cls]
        if cls == "persistent/reproduced":
            candidates.sort(key=lambda p: (-p["persistent_attribute_model_count"], -p["abs_rho_ic"], -p["abs_rho_dpl"], p["model"], p["parameter"], p["attribute"]))
        elif cls == "attenuated":
            candidates.sort(key=lambda p: (-p["abs_rho_ic"], p["delta_abs_rho"], p["model"], p["parameter"], p["attribute"]))
        elif cls == "dPL-emergent":
            candidates.sort(key=lambda p: (-p["abs_rho_dpl"], -p["delta_abs_rho"], p["model"], p["parameter"], p["attribute"]))
        else:
            candidates.sort(key=lambda p: (-p["abs_rho_ic"], -p["abs_rho_dpl"], p["model"], p["parameter"], p["attribute"]))
        if not candidates:
            raise RuntimeError(f"no eligible primary-8 seen-evidence case for {cls}")
        selected = dict(candidates[0])
        selected["selection_order"] = len(chosen) + 1
        selected["selection_category"] = cls
        chosen.append(selected)

    fields = [
        "selection_order", "selection_category", "model", "parameter_index", "parameter", "attribute", "attribute_type",
        "rho_ic", "rho_dpl_seen", "abs_rho_ic", "abs_rho_dpl", "rank_ic", "rank_dpl", "delta_abs_rho",
        "relationship_class", "persistent_attribute_model_count", "seen_boundary_concentrated", "seen_near_constant",
        "cluster_0.70", "cluster_representative_0.70", "cluster_members_0.70", "n", "strict_full300",
    ]
    path = R4 / "R4_RELATIONSHIP_CASES_FROZEN.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in chosen:
            writer.writerow({field: row.get(field, "") for field in fields})
    digest = sha256_file(path)
    (R4 / "R4_RELATIONSHIP_CASES_FROZEN.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return {"path": str(path), "sha256": digest, "cases": chosen, "source": str(SEEN / "09_PARAMETER_ATTRIBUTE_ATLAS_LONG.csv")}


def write_stage_a_documents(audit: dict[str, Any], frozen: dict[str, Any]) -> None:
    R4.mkdir(parents=True, exist_ok=True)
    (R4 / "logs/R4_STAGE_A_AUDIT.json").write_text(json.dumps({"audit": audit, "frozen": {"path": frozen["path"], "sha256": frozen["sha256"]}}, indent=2) + "\n", encoding="utf-8")
    status = "PASS" if audit["status"] == "PASS" else "FAIL"
    failures = "\n".join(f"- {item}" for item in audit["failures"]) or "- none"
    notes = "\n".join(f"- {item}" for item in audit["notes"]) or "- none"
    text = f"""# R4 OOB Provenance and Completeness Audit

- **Audit status:** **{status}**
- **Formal OOB root:** `{FORMAL}`
- **Jobs:** {audit['jobs']}/40 discovered; required status is 40/40 DONE.
- **Models/folds:** {audit['models']} models × {audit['folds_per_model']} folds.
- **OOF basin contract:** expected 531 unique held-out basins per model; fold sizes `{audit['fold_sizes']}`.
- **Fold assignment SHA256:** `{audit['assignment_sha256']}`
- **Configuration:** seed 42; canonical dPL v2 values were checked per run; selection metric is train loss.
- **Leakage evidence:** per-run phase/access flags, train-only normalization metadata, partition IDs, and source provenance were checked.
- **Numerical integrity:** checkpoint tensors and held-out KGE rows were checked one run at a time.
- **Retry evidence:** `{audit['retry_lines']}` retry line(s) in the downloaded queue log.

## Failures
{failures}

## Notes
{notes}

## Stage boundary
This audit opened only job metadata, partition files, checkpoint finiteness, KGE evidence, and provenance. It did not calculate or inspect any OOB attribute–parameter relationship coefficient. Population relationship analysis remains locked until the frozen case manifest below.

## Frozen seen-evidence case manifest
- Path: `{frozen['path']}`
- SHA256: `{frozen['sha256']}`
- Cases: {len(frozen['cases'])}
"""
    (R4 / "R4_OOB_PROVENANCE_AND_COMPLETENESS_AUDIT.md").write_text(text, encoding="utf-8")

    rule = """# R4 Relationship Case Selection Rule

## Scope and provenance
No prior R4 relationship-case preregistration or frozen manifest was found in the repository. The deterministic rule below therefore freezes four cases from the final seen-basin IC–dPL atlas before any OOB relationship coefficient is read.

Seen evidence source:
- `project/benchmark/results/ic_dpl_seenbasin_formal_20260901/09_PARAMETER_ATTRIBUTE_ATLAS_LONG.csv`
- Relationship implementation: `project/benchmark/scripts/diagnostics/parameter_attribute_atlas.py`
- Attribute-cluster sensitivity source: `project/benchmark/results/parameter_attribute_atlas_followup_20260829/attribute_redundancy_groups.csv`, threshold `|rho| >= 0.70`

The selected population is the PRIMARY 8 models, all 531 basins, continuous attributes, and `strict_full300=True`. Parameters are excluded if either seen IC or seen dPL normalized coordinates are near-constant (SD ≤ 0.01) or boundary-concentrated (≥50% within 0.01 of either bound).

## Relationship classes inherited from R2/R3 seen evidence
Using basin-wise Spearman correlation with average ranks and the existing descriptive rules, applied in this precedence order:

1. `sign-changing`: both `|rho| >= 0.10`, opposite signs;
2. `attenuated`: IC `|rho| >= 0.20`, dPL `|rho| < 0.10`;
3. `dPL-emergent`: dPL `|rho| >= 0.20`, IC `|rho| < 0.10`;
4. `persistent/reproduced`: same sign, both `|rho| >= 0.20`, and both within-parameter absolute-rho ranks ≤ 10;
5. otherwise `weak/unresolved`.

## Deterministic case selection
One case is selected from each class in this order: persistent/reproduced, attenuated, dPL-emergent, sign-changing.

- Persistent: descending persistent-attribute model recurrence, descending `|rho_IC|`, descending `|rho_dPL|`, then ascending model/parameter/attribute.
- Attenuated: descending `|rho_IC|`, ascending `delta_abs_rho = |rho_dPL|-|rho_IC|`, then ascending model/parameter/attribute.
- dPL-emergent: descending `|rho_dPL|`, descending `delta_abs_rho`, then ascending model/parameter/attribute.
- Sign-changing: descending `|rho_IC|`, descending `|rho_dPL|`, then ascending model/parameter/attribute.

## Freeze declaration
**OOB relationship coefficients had not been inspected or used for case selection before `R4_RELATIONSHIP_CASES_FROZEN.csv` was written and hashed.** The manifest is now frozen; later OOB analysis may evaluate these cases but may not alter them.

The manifest includes the seen rho values only as pre-OOB selection evidence. It is not a truth label, causal claim, or model-suitability selection.
"""
    (R4 / "R4_RELATIONSHIP_CASE_SELECTION_RULE.md").write_text(rule, encoding="utf-8")

    dictionary = """# R4 Estimand Dictionary

- **Question:** whether R2–R3 seen-basin structure-dependent catchment–parameter relationships persist when parameters are generated for held-out basins.
- **Primary models:** alpine2, hbv96, xinanjiang, newzealand2, ihacres, us1, mopex4, hillslope.
- **Relationship metric:** basin-wise Spearman rho with average ranks, inherited from the final seen-basin atlas; no Pearson substitution.
- **Attribute source:** canonical Caravan 35-dimensional matrix, 531 canonical basin IDs. Continuous attributes are primary; categorical-code attributes remain descriptive and are excluded from the primary atlas.
- **Parameter coordinate:** normalized `u` / sigmoid coordinate for IC, seen dPL, and OOB dPL. Physical parameter values are retained in master tables; monotone mapping preserves finite within-cell ranks.
- **Seen sources:** final formal IC–dPL seen-basin atlas (`ic_dpl_seenbasin_formal_20260901`) and its paired parameter table.
- **OOB source:** exact `best.pt` restored from each completed OOB model/fold, evaluated on its held-out basins only; no OOB coefficient was used before case freeze.
- **Classes:** existing descriptive thresholds high `|rho| >= 0.20`, low `|rho| < 0.10`, sign-changing when both sides are nontrivial and signs differ, persistent only when both sides are high, same-sign, and top-rank ≤ 10.
- **Boundary QC:** near-constant SD ≤ 0.01; boundary-concentrated means ≥50% at normalized ≤0.01 or ≥0.99; sensitivity flags these rows rather than silently deleting them from the population.
- **Tie rule:** average ranks for Spearman; existing atlas top-rank rule uses minimum rank for tied absolute-rho values. Effective/tie counts are reported in the boundary/tie sensitivity table.
- **Information clusters:** secondary descriptive graph from the follow-up atlas, continuous attributes connected when absolute Spearman ≥0.70; primary cluster retention uses the 0.70 graph and reports raw-proxy substitution separately.
- **Bootstrap:** basin-level resampling, 5,000 replicates, fixed seed 20260902, percentile 95% CI; generated in CUDA batches when available and only final statistics retained.
- **Aggregation:** pooled OOF rho is computed after concatenating the five held-out folds (each basin appears exactly once per model); fold-wise rho is reported separately; no complex fold significance test.
- **IC/KGE:** `KGE_IC` and `KGE_dPL_seen` are imported from canonical seen-basin paired KGE evidence. IC is not rerun.
- **Interpretation boundary:** descriptive retention/attenuation/reversal only; no physical truth, causality, universal transfer, or model-suitability conclusion.
"""
    (R4 / "R4_ESTIMAND_DICTIONARY.md").write_text(dictionary, encoding="utf-8")


def main() -> None:
    audit = audit_oob()
    if audit["status"] != "PASS":
        # Still write the audit, but do not freeze or run scientific interpretation.
        (R4 / "logs").mkdir(parents=True, exist_ok=True)
        (R4 / "logs/R4_STAGE_A_AUDIT.json").write_text(json.dumps({"audit": audit}, indent=2) + "\n", encoding="utf-8")
        (R4 / "R4_OOB_PROVENANCE_AND_COMPLETENESS_AUDIT.md").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("R4 audit failed; case freeze and scientific analysis are locked")
    frozen = freeze_cases()
    write_stage_a_documents(audit, frozen)
    print(json.dumps({"audit": audit, "frozen_manifest": frozen["path"], "frozen_sha256": frozen["sha256"], "cases": [(x["selection_category"], x["model"], x["parameter"], x["attribute"]) for x in frozen["cases"]]}, indent=2))


if __name__ == "__main__":
    main()
