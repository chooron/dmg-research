"""Minimal read-only helpers for the canonical R2 reproduction pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def find_repo() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").is_dir():
            return parent
    raise RuntimeError("repository root not found")


REPO = find_repo()
R2 = REPO / "project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905"
MANUSCRIPT = REPO / "project/benchmark/manuscript/r2"
CACHE = MANUSCRIPT / "cache"
SEED_DPL = 42
IC_TOLERANCE = 0.01
N_MODELS = 36
N_BASINS = 531
N_COORDINATES = 271
ELIGIBLE_MODELS = [
    "alpine1", "alpine2", "collie1", "collie2", "collie3", "flexi", "flexis",
    "gr4j", "hillslope", "hymod", "ihacres", "modhydrolog", "mopex1", "mopex2",
    "mopex3", "newzealand1", "simhyd", "susannah1", "tank", "us1", "vic", "wetland",
    "xinanjiang",
]
INSUFFICIENT_MODELS = [
    "australia", "flexb", "gsfb", "hbv96", "mopex4", "mopex5", "newzealand2",
    "penman", "plateau", "smar", "susannah2", "tcm", "topmodel",
]
ALL_MODELS = sorted(ELIGIBLE_MODELS + INSUFFICIENT_MODELS)

SOURCE = {
    "separation_models": R2 / "claim_audit_multiaudit_20260905/agent_B/tables/MODEL_IC_SELF_SUMMARY.csv",
    "separation_bootstrap": R2 / "claim_audit_multiaudit_20260905/agent_B/tables/BOOTSTRAP_SUMMARY.csv",
    "displacement_model": R2 / "r2/tables/11_MODEL_LEVEL_PARAMETER_DISPLACEMENT.csv",
    "rank_all36": R2 / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/tables/R_rank_model_equal_summaries.csv",
    "rank_models": R2 / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/tables/R_rank_model_summaries.csv",
    "rank_self": R2 / "r2_final_robustness_20260906/agent_A_rank_self_reference/tables/rank_self_sensitivity_summary.csv",
    "rank_coverage": R2 / "r2_final_robustness_20260906/agent_A_rank_self_reference/tables/reference_coverage.csv",
    "raw_localization_models": R2 / "r2_final_robustness_20260906/agent_C_model_equal_claim_audit/tables/frozen_C_eff_model_summaries.csv",
    "raw_localization_summary": R2 / "r2_final_robustness_20260906/agent_C_model_equal_claim_audit/tables/model_equal_reaggregation_summary.csv",
    "coord_excess": R2 / "r2_coordinate_icself_final_audit_20260906/agent_A_coordinate_excess/model_equal_summary.csv",
    "coord_overlap": R2 / "r2_coordinate_icself_final_audit_20260906/agent_A_coordinate_excess/raw_self_overlap.csv",
    "adjusted_localization": R2 / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/model_equal_adjusted_summary.csv",
    "adjusted_raw": R2 / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/model_equal_raw_vs_adjusted_summary.csv",
    "adjusted_l1": R2 / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/sensitivity_L1.csv",
    "adjusted_all36": R2 / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/sensitivity_all36.csv",
    "distribution": R2 / "r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/corrected_model_equal_summary.csv",
    "distribution_matched": R2 / "r2_final_robustness_20260906/agent_B_contraction_robustness/corrected_primary_23models/matched_model_comparison_23.csv",
    "performance": R2 / "claim_audit_multiaudit_20260905/agent_C/tables/model_equal_summary.csv",
    "performance_models": R2 / "r2_coupling_deepdive_20260906/agent_A_excess_displacement/tables/model_association_all.csv",
    "linkage_corr": R2 / "r2_parameter_axis_audit_20260906/agent_D_r2_r3_rank_linkage/tables/correlation_summaries.csv",
    "linkage_uncertainty": R2 / "r2_parameter_axis_audit_20260906/agent_D_r2_r3_rank_linkage/tables/uncertainty_summary.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(key_or_path: str | Path, **kwargs) -> pd.DataFrame:
    path = SOURCE[key_or_path] if isinstance(key_or_path, str) and key_or_path in SOURCE else Path(key_or_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def source_record(path: Path) -> dict:
    return {"path": str(path.relative_to(REPO)), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def ensure_inputs() -> None:
    """Create frozen input attestations once, then verify them on every run.

    Rewriting a source hash from the current tree would silently convert source
    drift into a new claim of frozen provenance. Existing attestations are
    therefore immutable: every listed digest and contract field is checked
    before any downstream module is allowed to run.
    """
    inp = CACHE / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    contract = {
        "rule": "archived non-canonical IC restart training KGE within 0.01 of basin best restart",
        "primary_coverage_threshold": 0.90,
        "eligible_models_primary": ELIGIBLE_MODELS,
        "insufficient_models_sensitivity_only": INSUFFICIENT_MODELS,
        "models_total": N_MODELS,
        "basins_per_model": N_BASINS,
        "parameter_coordinates": N_COORDINATES,
        "canonical_restart_excluded": True,
        "canonical_ic_definition": "best archived IC restart per model-basin",
        "dpl_seed": SEED_DPL,
        "dpl_multiseed_available": False,
        "same_restart_across_parameters": True,
        "no_canonical_fallback": True,
        "normalization": "frozen physical-bound normalized [0,1] coordinates",
        "source_tree_read_only": True,
    }
    alignment = R2 / "r2/cache/parameter_order_bounds_qc.csv"
    if not alignment.exists():
        raise FileNotFoundError(alignment)
    frozen_path = inp / "frozen_contract.json"
    eligible_path = inp / "eligible_models_primary.csv"
    insufficient_path = inp / "insufficient_models_sensitivity.csv"
    alignment_path = inp / "model_parameter_alignment.csv"
    manifest_path = inp / "input_manifest.json"
    checksums_path = inp / "source_checksums.sha256"
    config_path = inp / "config.json"
    expected_config = {"dpl_seed": SEED_DPL, "ic_tolerance": IC_TOLERANCE, "model_count": N_MODELS, "eligible_count": len(ELIGIBLE_MODELS)}
    required = [frozen_path, eligible_path, insufficient_path, alignment_path, manifest_path, checksums_path, config_path]
    if any(p.exists() for p in required):
        if not all(p.exists() for p in required):
            raise RuntimeError("incomplete frozen input attestations; refusing to regenerate")
        if json.loads(frozen_path.read_text()) != contract:
            raise RuntimeError("frozen contract mismatch; source contract drift or tampering detected")
        eligible = pd.read_csv(eligible_path)
        insufficient = pd.read_csv(insufficient_path)
        if eligible.model.astype(str).tolist() != ELIGIBLE_MODELS or set(eligible.primary_status) != {"PASS"}:
            raise RuntimeError("eligible model attestation mismatch")
        if insufficient.model.astype(str).tolist() != INSUFFICIENT_MODELS or set(insufficient.primary_status) != {"INSUFFICIENT_REFERENCE"}:
            raise RuntimeError("insufficient model attestation mismatch")
        if sha256(alignment_path) != sha256(alignment):
            raise RuntimeError("parameter alignment attestation mismatch")
        expected = {p.relative_to(REPO).as_posix(): sha256(p) for p in list(SOURCE.values()) + [alignment]}
        recorded = {}
        for line in checksums_path.read_text().splitlines():
            if line.strip():
                digest, rel = line.strip().split(maxsplit=1)
                recorded[rel.strip()] = digest
        if recorded != expected:
            raise RuntimeError("frozen source checksum mismatch; refusing to re-attest drift")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != contract:
            raise RuntimeError("input manifest contract mismatch")
        manifest_records = {entry["path"]: entry["sha256"] for entry in manifest.get("inputs", [])}
        if manifest_records != expected:
            raise RuntimeError("input manifest source hashes mismatch")
        if json.loads(config_path.read_text()) != expected_config:
            raise RuntimeError("input config attestation mismatch")
        return
    write_json(frozen_path, contract)
    pd.DataFrame({"model": ELIGIBLE_MODELS, "primary_status": "PASS", "coverage_threshold": 0.90}).to_csv(eligible_path, index=False)
    pd.DataFrame({"model": INSUFFICIENT_MODELS, "primary_status": "INSUFFICIENT_REFERENCE", "coverage_threshold": 0.90}).to_csv(insufficient_path, index=False)
    shutil.copy2(alignment, alignment_path)
    manifest = {"contract": contract, "inputs": [source_record(p) for p in SOURCE.values()] + [source_record(alignment)]}
    write_json(manifest_path, manifest)
    with checksums_path.open("w") as f:
        for p in list(SOURCE.values()) + [alignment]:
            f.write(f"{sha256(p)}  {p.relative_to(REPO).as_posix()}\n")
    write_json(config_path, expected_config)


def finalize_group(group: str, source_keys: Iterable[str], outputs: Iterable[Path]) -> None:
    out = CACHE / group
    rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in {"outputs_manifest.json", "CHECKSUMS.txt", "run.log"}:
            rows.append({"path": str(p.relative_to(MANUSCRIPT)), "sha256": sha256(p), "size_bytes": p.stat().st_size})
    write_json(out / "outputs_manifest.json", {"group": group, "source_inputs": [source_record(SOURCE[k]) for k in source_keys], "files": rows})
    with (out / "CHECKSUMS.txt").open("w") as f:
        for row in rows:
            rel = (MANUSCRIPT / row["path"]).relative_to(CACHE / group)
            f.write(f"{row['sha256']}  {rel}\n")


def require_close(observed: float, expected: float, tol: float, label: str) -> None:
    if not np.isfinite(observed) or abs(float(observed) - expected) > tol:
        raise AssertionError(f"{label}: observed={observed} expected={expected} tol={tol}")


def clean_derived() -> None:
    for group in ["separation", "rank", "localization", "distribution", "performance_bridge", "r2_r3_linkage", "final"]:
        path = CACHE / group
        if path.exists():
            for child in path.iterdir():
                if child.is_dir():
                    import shutil as _shutil
                    _shutil.rmtree(child)
                else:
                    child.unlink()
        path.mkdir(parents=True, exist_ok=True)
