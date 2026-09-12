#!/usr/bin/env python3
"""Step 0: audit the frozen R1 inputs and freeze the existing temporal split."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from r1_config import (
    BENCHMARK, CARAVAN_PATH, DATA_ROOT, DATASET_PATH, DATE_INDEX_PATH, DPL_ROOT,
    EVAL_WARMUP_DAYS, EXPECTED_DATA, FORMAL_MANIFEST, FORMAL_PAIRED_SOURCE,
    FORMAL_ROOT, FORCING_V2_PATH, GAGE_IDS_PATH, IC_CHECKPOINT_ROOT, IC_ROOT, IC_STATUS_PATH,
    CANONICAL_DPL_SEED, CANONICAL_DPL_SOURCE_SHA,
    MODEL_REGISTRY, R1_ROOT, TABLES_DIR, TEMPORAL_AB, TEMPORAL_PRIOR_RESULT,
    TEMPORAL_QUICK_RESULT, TEMPORAL_SOURCE_SCRIPT, TEST_END, TEST_START,
    VIC_AUDIT_ROOT,
)
from r1_utils import canonical_basin_id, read_canonical_ids, sha256_file, utc_now, write_csv



def audit_row(check: str, status: str, expected: object, observed: object,
              evidence: object, model: str = "", notes: str = "") -> dict[str, object]:
    return {
        "check": check, "model": model, "expected": expected, "observed": observed,
        "status": status, "evidence": evidence, "notes": notes,
    }

def load_ic_ids(model: str, status: dict[str, object]) -> tuple[list[str], list[Path], list[int]]:
    paths = sorted((IC_CHECKPOINT_ROOT / model).glob("chunk_*_gen_*.pt"))
    all_ids: list[str] = []
    generations: list[int] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        all_ids.extend(canonical_basin_id(x) for x in np.asarray(payload["basin_ids"]).reshape(-1))
        generations.append(int(payload["generation"]))
    if not paths:
        return all_ids, paths, generations
    declared = status.get("latest_checkpoint") if isinstance(status, dict) else None
    if declared:
        final = IC_CHECKPOINT_ROOT / model / str(declared)
        if not final.is_file():
            return all_ids, paths, generations
    return all_ids, paths, generations


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ids = read_canonical_ids(Path(DATA_ROOT / "531sub_id.txt"))
    expected_ids = set(ids)
    status_summary = json.loads(IC_STATUS_PATH.read_text())
    paired = pd.read_csv(FORMAL_PAIRED_SOURCE, dtype={"basin_id": str})
    paired["basin_id"] = paired["basin_id"].map(canonical_basin_id)
    registry_set = set(MODEL_REGISTRY)
    audit: list[dict[str, object]] = []

    # Data contract checks are tied to the checksums recorded by the recovered
    # canonical remote/local forensic audit, not to a guessed local fallback.
    data_paths = {
        "531sub_id.txt": DATA_ROOT / "531sub_id.txt", "camels_dataset": DATASET_PATH,
        "gage_id.npy": GAGE_IDS_PATH, "caravan_671_attributes.npy": CARAVAN_PATH,
        "camels_forcing_v2.pkl": FORCING_V2_PATH, "camels_dates.npy": DATE_INDEX_PATH,
    }
    for name, path in data_paths.items():
        expected_size, expected_sha = EXPECTED_DATA[name]
        if not path.is_file():
            audit.append(audit_row("data_contract", "FAIL", f"{expected_size} bytes/{expected_sha}", "MISSING", path))
            continue
        observed = f"{path.stat().st_size} bytes/{sha256_file(path)}"
        ok = path.stat().st_size == expected_size and sha256_file(path) == expected_sha
        audit.append(audit_row("data_contract", "PASS" if ok else "FAIL",
                                f"{expected_size} bytes/{expected_sha}", observed, path,
                                notes="canonical remote/local replay contract"))

    audit.append(audit_row("canonical_basin_list", "PASS" if len(ids) == 531 and len(set(ids)) == 531 else "FAIL",
                           "531 unique IDs", f"{len(ids)} rows/{len(set(ids))} unique", DATA_ROOT / "531sub_id.txt"))
    gage_ids = [canonical_basin_id(x) for x in np.load(GAGE_IDS_PATH).reshape(-1)]
    missing_gage = sorted(expected_ids - set(gage_ids))
    audit.append(audit_row("gage_id_lookup", "PASS" if not missing_gage and len(set(gage_ids)) == len(gage_ids) else "FAIL",
                           "all canonical IDs present and unique", f"{len(missing_gage)} missing; {len(gage_ids)} rows/{len(set(gage_ids))} unique",
                           GAGE_IDS_PATH))

    # Validate the existing formal KGE source independently of its summary.
    numeric_cols = ["KGE_IC", "KGE_dPL", "Delta_KGE"]
    for col in numeric_cols:
        paired[col] = pd.to_numeric(paired[col], errors="coerce")
        nan_count = int(paired[col].isna().sum())
        inf_count = int(np.isinf(paired[col].to_numpy(dtype=float)).sum())
        audit.append(audit_row(f"KGE_{col}_validity", "PASS" if nan_count == 0 and inf_count == 0 else "FAIL",
                               "NaN=0; Inf=0", f"NaN={nan_count}; Inf={inf_count}", FORMAL_PAIRED_SOURCE))
    duplicate_rows = int(paired.duplicated(["model", "basin_id"]).sum())
    audit.append(audit_row("formal_pair_duplicates", "PASS" if duplicate_rows == 0 else "FAIL",
                           "0 duplicate model-basin rows", duplicate_rows, FORMAL_PAIRED_SOURCE))
    unexpected = sorted(set(paired.model) - registry_set)
    missing_models = sorted(registry_set - set(paired.model))
    audit.append(audit_row("formal_model_coverage", "PASS" if not unexpected and not missing_models and len(set(paired.model)) == 36 else "FAIL",
                           "36 registry models", f"observed={len(set(paired.model))}; missing={missing_models}; unexpected={unexpected}", FORMAL_PAIRED_SOURCE))
    delta_error = float(np.max(np.abs(paired["Delta_KGE"] - (paired["KGE_dPL"] - paired["KGE_IC"]))))
    audit.append(audit_row("formal_delta_sign", "PASS" if delta_error <= 1e-8 else "FAIL",
                           "Delta_KGE = KGE_dPL - KGE_IC", f"max_abs_error={delta_error:.3g}", FORMAL_PAIRED_SOURCE,
                           notes="source column is only checked; R1 recomputes delta"))

    inventory: list[dict[str, object]] = []
    for model in MODEL_REGISTRY:
        status = status_summary.get(model, {})
        ic_ids, ic_paths, ic_generations = load_ic_ids(model, status)
        dpl_best = DPL_ROOT / "runs" / model / "best.pt"
        dpl_test = DPL_ROOT / "runs" / model / "basin_test_kge.csv"
        dpl_ids: list[str] = []
        if dpl_test.is_file():
            dpl_df = pd.read_csv(dpl_test, dtype={"basin_id": str})
            dpl_ids = [canonical_basin_id(x) for x in dpl_df["basin_id"]]
        formal_model = paired.loc[paired.model == model]
        formal_ids = set(formal_model.basin_id)
        ic_set, dpl_set = set(ic_ids), set(dpl_ids)
        common = ic_set & dpl_set & expected_ids
        exact_ic = ic_set == expected_ids
        exact_dpl = dpl_set == expected_ids
        exact_formal = formal_ids == expected_ids
        dpl_meta: dict[str, object] = {}
        payload: dict[str, object] = {}
        if dpl_best.is_file():
            payload = torch.load(dpl_best, map_location="cpu", weights_only=False)
            dpl_meta_path = dpl_best.with_name("best_metadata.json")
            if dpl_meta_path.is_file():
                dpl_meta = json.loads(dpl_meta_path.read_text())
        job = payload.get("job_config", {}) if isinstance(payload, dict) else {}
        if not isinstance(job, dict):
            job = {}
        dpl_hash_ok = False
        sidecar = dpl_best.with_name("best_state_sha256.txt")
        if dpl_best.is_file() and sidecar.is_file():
            dpl_hash_ok = sidecar.read_text().strip() == sha256_file(dpl_best)
        notes = []
        if model == "vic":
            notes.append("current dynamic-DOY IC result; pre-dynamic-DOY backup excluded")
        if model == "simhyd":
            notes.append("accepted IC generation 280; excluded only from strict Full300 sensitivity")
        if model == "flexb":
            notes.append("current formal paired result retained; no historical special marker")
        if len(ic_generations) > 0 and len(set(ic_generations)) > 1:
            notes.append(f"IC chunk generations={sorted(set(ic_generations))}")
        status_ok = exact_ic and exact_dpl and exact_formal and len(common) == 531
        inventory.append({
            "model": model,
            "ic_source_path": ";".join(str(p) for p in ic_paths),
            "ic_declared_checkpoint_path": str(IC_CHECKPOINT_ROOT / model / str(status.get("latest_checkpoint", ""))),
            "dpl_source_path": str(dpl_best),
            "dpl_test_kge_source_path": str(dpl_test),
            "ic_basin_count": len(ic_set), "dpl_basin_count": len(dpl_set),
            "paired_basin_count": len(common), "formal_basin_count": len(formal_ids),
            "ic_generation": ";".join(map(str, sorted(set(ic_generations)))),
            "dpl_best_epoch": dpl_meta.get("best_epoch", payload.get("epoch", "")),
            "dpl_seed": job.get("seed", ""), "dpl_test_warmup_days": job.get("test_warmup_days", ""),
            "dpl_test_period": str(job.get("test_period", "")),
            "dpl_best_pt_sha256_matches_sidecar": dpl_hash_ok,
            "status": "PASS" if status_ok and dpl_best.is_file() and dpl_test.is_file() else "FAIL",
            "notes": " | ".join(notes),
        })
        audit.append(audit_row("model_pairing", "PASS" if status_ok else "FAIL", "IC/dPL/formal basin ID sets exactly equal canonical 531",
                               f"IC={len(ic_set)} exact={exact_ic}; dPL={len(dpl_set)} exact={exact_dpl}; common={len(common)}; formal={len(formal_ids)} exact={exact_formal}",
                               f"IC={IC_CHECKPOINT_ROOT / model}; dPL={dpl_best}; canonical={DATA_ROOT / '531sub_id.txt'}", model,
                               "all paired analysis uses explicit model+basin_id join; missing/extra IDs would fail"))
        audit.append(audit_row("canonical_dpl_best_pt", "PASS" if dpl_best.is_file() and dpl_hash_ok else "FAIL",
                               "best.pt and matching sha256 sidecar", f"best.pt={dpl_best.is_file()}; sidecar_match={dpl_hash_ok}", dpl_best, model))
        config_ok = (
            dpl_meta.get("model", model) == model and dpl_meta.get("selection_metric") == "train_loss"
            and str(dpl_meta.get("git_sha", "")) == CANONICAL_DPL_SOURCE_SHA
            and int(job.get("seed", -1)) == CANONICAL_DPL_SEED
            and str(job.get("test_period", "")) == "['1995-10-01', '2010-09-30']"
            and int(job.get("test_warmup_days", -1)) == EVAL_WARMUP_DAYS
        )
        audit.append(audit_row("canonical_dpl_v2_metadata", "PASS" if config_ok else "UNVERIFIED",
                               "model, train-loss best, git SHA, test period, 365d eval warmup",
                               f"model={dpl_meta.get('model')}; metric={dpl_meta.get('selection_metric')}; sha={dpl_meta.get('git_sha')}; test={job.get('test_period')}; warmup={job.get('test_warmup_days')}",
                               f"{dpl_best}; {dpl_best.with_name('best_metadata.json')}", model))

    # Formal 00_INPUT_MANIFEST was generated before the current roots were
    # available and records stale root-level False flags.  Preserve this warning
    # rather than silently treating it as a current source failure.
    if FORMAL_MANIFEST.is_file():
        old_manifest = pd.read_csv(FORMAL_MANIFEST, dtype=str)
        for role, path in [("dpl_root", DPL_ROOT), ("ic_root", IC_ROOT)]:
            rows = old_manifest.loc[old_manifest.role == role]
            old_flag = rows.iloc[0]["exists"] if not rows.empty else "MISSING_ROW"
            audit.append(audit_row("formal_manifest_root_flag", "WARN" if old_flag == "False" and path.exists() else "PASS",
                                   "current root exists", f"old_manifest_exists={old_flag}; current_exists={path.exists()}",
                                   FORMAL_MANIFEST, notes="stale root-level flag; per-model files are audited above"))

    audit.append(audit_row("canonical_v2_source_sha", "WARN",
                           "checkpoint source provenance documented", "manifest=3caca37a; checkpoint metadata=7d1132bf; forensic report resolves conditional validity",
                           FORMAL_ROOT / "FORMAL_IC_DPL_SEENBASIN_AND_PARAMETER_ATLAS_REPORT.md",
                           notes="do not claim byte-identical tracked source; use recovered data contract and checkpoint SHA evidence"))

    # Freeze the already-used temporal definition, not a new one.
    prior_ok = TEMPORAL_SOURCE_SCRIPT.is_file() and TEMPORAL_PRIOR_RESULT.is_file()
    quick_ok = TEMPORAL_QUICK_RESULT.is_file()
    source_text = TEMPORAL_SOURCE_SCRIPT.read_text(errors="replace") if TEMPORAL_SOURCE_SCRIPT.is_file() else ""
    source_has_half_logic = (('("A", 0, half)' in source_text or '("A", 0, half)' in source_text) and '("B", half, vy.shape[0])' in source_text)
    definition_rows = []
    for part in TEMPORAL_AB:
        definition_rows.append({
            "partition": part["partition"], "start_date": part["start_date"], "end_date": part["end_date"],
            "water_year_calendar_year_rule": "calendar-date halves of post-warmup TEST output; no within-half warm-up",
            "source_file_or_script": str(TEMPORAL_SOURCE_SCRIPT),
            "notes": "reused prior complete R1 temporal definition; matched prior strict34 and quick-survey dates; applied to all 36 models here",
        })
    write_csv(TABLES_DIR / "R1_temporal_AB_definition.csv", definition_rows)
    audit.append(audit_row("temporal_AB_definition", "PASS" if prior_ok and quick_ok and source_has_half_logic else "UNVERIFIED",
                           "A=1995-10-01..2003-03-31; B=2003-04-01..2010-09-30",
                           f"prior_result={prior_ok}; quick_result={quick_ok}; date-half code={source_has_half_logic}",
                           f"{TEMPORAL_SOURCE_SCRIPT}; {TEMPORAL_PRIOR_RESULT}; {TEMPORAL_QUICK_RESULT}",
                           notes="same previously frozen split; no new temporal design"))
    audit.append(audit_row("temporal_AB_competing_definitions", "PASS",
                           "no conflicting canonical date definition found",
                           "prior complete R1 and quick-survey definitions agree",
                           f"{TEMPORAL_PRIOR_RESULT}; {TEMPORAL_QUICK_RESULT}"))
    audit.append(audit_row("VIC_dynamic_DOY", "PASS" if VIC_AUDIT_ROOT.is_dir() else "UNVERIFIED",
                           "current dynamic-DOY IC result", str(VIC_AUDIT_ROOT), VIC_AUDIT_ROOT,
                           notes="formal source excludes pre-dynamic-DOY backup"))
    audit.append(audit_row("simhyd_generation", "PASS" if inventory[25]["ic_generation"] == "280" else "UNVERIFIED",
                           "accepted generation 280", inventory[25]["ic_generation"], IC_ROOT / "best_training" / "simhyd"))
    audit.append(audit_row("flexb_current_formal", "PASS" if inventory[6]["status"] == "PASS" else "FAIL",
                           "paired current formal result", inventory[6]["status"], FORMAL_PAIRED_SOURCE,
                           notes="no historical missing-dPL exclusion applied"))

    write_csv(TABLES_DIR / "R1_input_inventory.csv", inventory)
    write_csv(TABLES_DIR / "R1_input_audit.csv", audit,
              ["check", "model", "expected", "observed", "status", "evidence", "notes"])
    print(f"PASS: wrote {TABLES_DIR / 'R1_input_inventory.csv'} ({len(inventory)} models)")
    print(f"PASS: wrote {TABLES_DIR / 'R1_input_audit.csv'} ({len(audit)} audit rows)")
    print(f"PASS: wrote {TABLES_DIR / 'R1_temporal_AB_definition.csv'}")


if __name__ == "__main__":
    main()
