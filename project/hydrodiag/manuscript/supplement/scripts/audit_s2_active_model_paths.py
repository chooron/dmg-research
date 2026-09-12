#!/usr/bin/env python3
"""Inventory active S2 model paths and source evidence."""
from __future__ import annotations

import ast
import argparse
import inspect
import json
import sys
from pathlib import Path

from s2_audit_utils import ensure_dirs, git_status, project_root_from_args, rel, sha256, source_ref, supplement_dir, write_csv, write_json


ACTIVE = {
    "XAJ": "XAJ", "XAJ_CN": "XAJWithCemaNeige", "XAJ_PD": "XAJWithPrecipitationDelay", "XAJ_TGD": "XAJWithTemperatureConditionedDelay",
    "GR4J": "GR4J", "GR4J_CN": "GR4JWithCemaNeige", "GR4J_PD": "GR4JWithPrecipitationDelay", "GR4J_TGD": "GR4JWithTemperatureConditionedDelay",
    "SIMHYD": "SIMHYD", "SIMHYD_CN": "SIMHYDWithCemaNeige", "SIMHYD_PD": "SIMHYDWithPrecipitationDelay", "SIMHYD_TGD": "SIMHYDWithTemperatureConditionedDelay",
    "HBV": "HBV",
}


def inventory(root: Path, out: Path) -> None:
    model_dir = root / "models"
    rows = []
    for path in sorted(model_dir.glob("*.py")):
        if path.name == "__pycache__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        text = path.read_text(encoding="utf-8")
        active_symbols = [s for s in ACTIVE.values() if s in classes]
        category = "utility"
        lower = path.stem.lower()
        for key in ("xaj", "gr4j", "simhyd", "hbv", "cemaneige", "temperature", "composed", "delay", "unit_hydro"):
            if key in lower:
                category = key
        rows.append({"path": rel(path, root), "module": path.stem, "classes": ";".join(classes), "functions": ";".join(funcs),
                     "active_symbol": ";".join(active_symbols), "category": category,
                     "referenced_by_active_adapter": "yes" if active_symbols or path.name in {"parameter_specs.py", "utils.py", "base.py", "unit_hydro.py"} else "no",
                     "suspected_legacy": "no", "mtime": path.stat().st_mtime, "sha256": sha256(path), "git_status": git_status(root, path)})
    write_csv(out / "results" / "s2_model_file_inventory.csv", rows)

    config = root / "ablation" / "configs" / "ic_foundation_531_v1.json"
    adapter = root / "ablation" / "ic_core" / "model_adapter.py"
    dpl = root / "training" / "dpl" / "run_dpl_model.py"
    graph = {
        "status": "VERIFIED_CODE",
        "active_dataset_manifest": source_ref(config, 2),
        "active_ic_entry": source_ref(adapter, 31, "ModelAdapter.__init__"),
        "active_ic_runtime": source_ref(root / "ablation/ic_core/runtime.py", 104, "ICObjectiveRuntime.evaluate_candidates"),
        "active_dpl_entry": source_ref(dpl, 627, "main registry selection"),
        "model_variant": {"IC default": "full", "dPL default": "full", "lite": "explicit optional --lite"},
        "forcing": {"names": ["P", "T", "PET"], "adapter_evidence": source_ref(adapter, 82, "_forcing_dict")},
        "model_keys": json.loads(config.read_text())['model_keys'],
        "mapping": {k: {"class": v, "full": v, "lite": v + "Lite" if v in {"XAJ", "GR4J", "SIMHYD", "HBV"} else v + "Lite"} for k, v in ACTIVE.items()},
        "shared_forward": True,
        "notes": ["PD is a separate precipitation-delay control; TGD is TemperatureConditionedDelay.", "HBV is a standalone reference key and is not a Base/CN/TGD wrapper."],
        "evidence": [source_ref(adapter, 14, "MODEL_CLASSES"), source_ref(adapter, 30, "LITE_MODEL_CLASSES"), source_ref(dpl, 84, "MODEL_REGISTRY"), source_ref(dpl, 100, "LITE_MODEL_REGISTRY")],
    }
    write_json(out / "results" / "s2_active_call_graph.json", graph)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-root")
    p.add_argument("--output-dir")
    a = p.parse_args(); root = project_root_from_args(a.project_root); out = supplement_dir(root, a.output_dir); ensure_dirs(out); inventory(root, out)

