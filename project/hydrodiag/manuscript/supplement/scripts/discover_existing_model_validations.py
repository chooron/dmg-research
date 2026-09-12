#!/usr/bin/env python3
"""Inventory existing S2 validation evidence without changing source code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from s2_validation_common import ACTIVE_CASES, PROJECT_ROOT, SUPPLEMENT_ROOT, ensure_dirs, environment, write_csv, write_json


KEYWORDS = (
    "mass balance", "water balance", "balance residual", "conservation", "gradcheck",
    "gradient check", "finite difference", "central difference", "directional derivative",
    "autograd", "backward", "jacobian", "jvp", "vjp", "reference", "equivalence",
    "numerical fidelity", "one step", "step equivalence", "torch.autograd.gradcheck",
    "float64", "double", "unit hydrograph", "routing buffer", "uh tail", "exchange",
)
MODEL_WORDS = ("xaj", "gr4j", "simhyd", "hbv", "cemaneige", "tgd", "temperature", "cn")
TEXT_SUFFIXES = {".py", ".md", ".sh", ".json", ".yaml", ".yml", ".csv", ".log", ".txt", ".toml"}


def evidence_level(path: Path, text: str) -> str:
    value = str(path).replace("\\", "/")
    active_markers = ("project/hydrodiag/models/", "training/dpl/run_dpl_model.py", "ablation/ic_core/model_adapter.py")
    if "project/hydrodiag/tests/" in value and "/archive/" not in value:
        return "EXACT_ACTIVE_CODE"
    if any(marker in value for marker in active_markers) and "/tests/" in value:
        return "EXACT_ACTIVE_CODE"
    if "/manuscript/supplement/" in value and "audit_s2_" in path.name:
        return "SAME_KERNEL_DIFFERENT_WRAPPER"
    if "/tests/" in value and ("dmotpy" in value or "project/flexmopex" in value):
        return "SAME_KERNEL_DIFFERENT_WRAPPER"
    if "/archive/" in value or "legacy" in text.lower():
        return "LEGACY_BUT_RELEVANT"
    return "NOT_APPLICABLE"


def infer_objects(path: Path, text: str) -> tuple[str, str, str]:
    low = (str(path) + "\n" + text).lower()
    models = [name for name in ("XAJ", "GR4J", "SIMHYD", "HBV") if name.lower() in low]
    structures = []
    if re.search(r"\bbase\b|standalone|full", low): structures.append("Base")
    if "tgd" in low or "temperature-conditioned" in low: structures.append("TGD")
    if "cema" in low or re.search(r"\bcn\b", low): structures.append("CN")
    if "hbv" in low and not models: models = ["HBV"]
    if not models: models = ["unknown"]
    if not structures: structures = ["unknown"]
    coverage = ";".join(f"{m}-{s}" for m in models for s in structures)
    return ";".join(models), ";".join(structures), coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SUPPLEMENT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve(); out = args.output_dir.resolve(); ensure_dirs()
    include_roots = [root / "project" / "hydrodiag", root / "project", root / "tests", root]
    seen: set[Path] = set(); rows = []
    for base in include_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path in seen or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            seen.add(path)
            if "/manuscript/supplement/results/" in str(path).replace("\\", "/") and path.name.startswith("s2_existing"):
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            hits = [kw for kw in KEYWORDS if kw.lower() in text.lower()]
            if not hits:
                continue
            models, structures, coverage = infer_objects(path, text)
            level = evidence_level(path, text)
            digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
            executable = path.suffix in {".py", ".sh"}
            rows.append({
                "path": str(path), "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "validation_object": ";".join(hits), "models": models, "structures": structures,
                "active_case_coverage": coverage, "full_lite_legacy": "lite" if "lite" in text.lower() else ("legacy" if level == "LEGACY_BUT_RELEVANT" else "full_or_unspecified"),
                "input_evidence": "synthetic/fixture" if any(token in text.lower() for token in ("synthetic", "forcing", "fixture")) else "not stated",
                "dtype_evidence": ";".join(dtype for dtype in ("float64", "float32", "double") if dtype in text.lower()) or "not stated",
                "tolerance_evidence": ";".join(sorted(set(re.findall(r"(?:atol|rtol|eps|tolerance|tol)\s*[=:]\s*[^,;\n]+", text, re.I))))[:500],
                "run_command": "python3 " + str(path) if path.suffix == ".py" else ("bash " + str(path) if path.suffix == ".sh" else "read-only artifact"),
                "original_result": "result artifact/log present" if path.suffix in {".csv", ".json", ".log", ".md"} else "source only",
                "log_path": str(path) if path.suffix in {".log", ".md", ".csv", ".json"} else "",
                "evidence_level": level, "active_code_match": level == "EXACT_ACTIVE_CODE",
                "directly_reproducible": executable, "needs_adaptation": level in {"LEGACY_BUT_RELEVANT", "SAME_KERNEL_DIFFERENT_WRAPPER"},
                "sha256_prefix": digest,
            })
    rows.sort(key=lambda row: (row["evidence_level"], row["relative_path"]))
    fields = list(rows[0]) if rows else []
    write_csv(out / "results" / "s2_existing_validation_inventory.csv", rows, fields)
    coverage = []
    for model, structure in ACTIVE_CASES:
        target = f"{model}-{structure}"
        matches = [row for row in rows if target in row["active_case_coverage"] or model in row["models"] and (structure in row["structures"] or structure == "reference" and model == "HBV")]
        levels = sorted({row["evidence_level"] for row in matches})
        coverage.append({
            "model": model, "structure": structure, "active_full_class_required": True,
            "matching_evidence_count": len(matches), "evidence_levels": ";".join(levels),
            "exact_active_code": "EXACT_ACTIVE_CODE" in levels, "same_kernel": "SAME_KERNEL_DIFFERENT_WRAPPER" in levels,
            "legacy_relevant": "LEGACY_BUT_RELEVANT" in levels, "direct_coverage_verdict": "COVERED" if "EXACT_ACTIVE_CODE" in levels else ("PARTIAL" if levels else "MISSING"),
            "notes": "Inventory match; exact whole-system mass balance and current full combination still require the closure run." if matches else "No matching existing validation found.",
        })
    write_csv(out / "results" / "s2_existing_validation_coverage.csv", coverage)
    write_json(out / "results" / "s2_existing_validation_logs.json", {"environment": environment(root), "searched_roots": [str(p) for p in include_roots], "keyword_count": len(KEYWORDS), "inventory_rows": len(rows), "coverage_rows": coverage})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
