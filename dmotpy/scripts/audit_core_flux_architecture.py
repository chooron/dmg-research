from __future__ import annotations

import ast
import csv
import inspect
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.core_model_registry import CORE_MODEL_REGISTRY


OUTPUT_DIR = PROJECT_ROOT / "validation_results" / "architecture_audit"
FILE_INVENTORY_CSV = OUTPUT_DIR / "file_inventory.csv"
CORE_INLINE_CSV = OUTPUT_DIR / "core_inline_formula_inventory.csv"
FLUX_USAGE_CSV = OUTPUT_DIR / "flux_function_usage.csv"
CALL_GRAPH_CSV = OUTPUT_DIR / "model_flux_call_graph.csv"
CALL_GRAPH_MD = OUTPUT_DIR / "model_flux_call_graph.md"
ISSUE_LIST_CSV = OUTPUT_DIR / "architecture_issue_list.csv"
REFACTOR_PLAN_MD = OUTPUT_DIR / "refactor_plan.md"
VALIDATION_REQUIREMENTS_MD = OUTPUT_DIR / "refactor_validation_requirements.md"
REPORT_MD = OUTPUT_DIR / "architecture_audit_report.md"


CORE_DIR = PROJECT_ROOT / "models" / "core"
FLUX_DIR = PROJECT_ROOT / "models" / "flux"
SPECIAL_DIR = PROJECT_ROOT / "models" / "special"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


INLINE_FUNC_TYPES = {
    "baseflow": ["baseflow"],
    "evap": ["evap", "et", "phenology"],
    "interception": ["interception"],
    "infiltration": ["infiltration", "inf"],
    "recharge": ["recharge", "perc", "percolation"],
    "saturation": ["saturation", "overflow", "excess"],
    "snow": ["snowfall", "rainfall", "melt", "refreeze"],
    "interflow": ["interflow"],
    "exchange": ["exchange", "leakage", "seep"],
    "soilmoisture": ["soilmoisture", "distribution", "rebalance"],
    "depression": ["depression"],
}

HYDRO_TOKENS = (
    "torch.minimum",
    "torch.maximum",
    "torch.clamp",
    "torch.sigmoid",
    "torch.exp",
    "torch.where",
    "torch.pow",
    "torch.tanh",
    "F.relu",
    "F.softplus",
    "soft_gate_",
    "flux_",
)

DANGEROUS_DUPLICATE_NOTES = {
    ("models/core/modhydrolog.py", "infiltration_1"): "Core-local safe variant diverges from shared generic infiltration semantics and was explicitly kept local.",
    ("models/core/modhydrolog.py", "infiltration_2"): "Core-local safe variant diverges from shared generic infiltration semantics and was explicitly kept local.",
    ("models/core/modhydrolog.py", "evap_2"): "Core-local safe variant diverges from shared generic evaporation semantics and was explicitly kept local.",
    ("models/core/modhydrolog.py", "interception_1"): "Core-local safe variant diverges from shared generic interception semantics and was explicitly kept local.",
    ("models/core/mopex1.py", "saturation_1"): "Core-local MOPEX smoother differs from shared dMoT storage gate implementation.",
    ("models/core/mopex2.py", "snowfall_1"): "Core-local MOPEX snow partition uses a different smoothing formulation than shared temperature gates.",
    ("models/core/mopex2.py", "rainfall_1"): "Core-local MOPEX rain partition uses a different smoothing formulation than shared temperature gates.",
    ("models/core/mopex4.py", "interception_4"): "Core-local MOPEX interception uses softplus smoothing instead of shared relu version.",
    ("models/core/mopex5.py", "phenology_1"): "Core-local MOPEX phenology mirrors shared logic closely and is likely migratable.",
    ("models/core/tcm.py", "baseflow_6"): "Local helper duplicates shared baseflow_6 logic and is a strong migration candidate.",
}


@dataclass
class ModuleInfo:
    path: Path
    module_type: str
    functions: list[str]
    classes: list[str]
    imports: list[str]
    imports_from_flux: list[str]
    imports_from_core: list[str]
    imported_by: list[str]
    registered_model: str
    active_or_inactive: str
    notes: str


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _module_name(path: Path) -> str:
    return _relative(path).replace("/", ".").removesuffix(".py")


def _parse_tree(path: Path) -> ast.AST:
    return ast.parse(_read(path), filename=str(path))


def _function_nodes(tree: ast.AST) -> list[ast.FunctionDef]:
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]


def _class_nodes(tree: ast.AST) -> list[ast.ClassDef]:
    return [node for node in tree.body if isinstance(node, ast.ClassDef)]


def _imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            for alias in node.names:
                names.append(f"{base}:{alias.name}")
    return names


def _imports_from_prefix(tree: ast.AST, prefix: str) -> list[str]:
    results: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if prefix in module:
                results.extend(alias.name for alias in node.names)
            elif node.level and prefix == "flux" and module.startswith("flux"):
                results.extend(alias.name for alias in node.names)
    return sorted(set(results))


def _flux_import_bindings(tree: ast.AST) -> dict[str, str]:
    """Map each local call name to the function defined in ``models.flux``."""
    bindings: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if "flux" not in module:
            continue
        for alias in node.names:
            bindings[alias.asname or alias.name] = alias.name
    return bindings


def _find_imported_by(all_paths: list[Path]) -> dict[str, list[str]]:
    by_module: dict[str, list[str]] = defaultdict(list)
    module_names = {_module_name(path): path for path in all_paths}
    for path in all_paths:
        text = _read(path)
        rel = _relative(path)
        for module_name in module_names:
            tail = module_name.split(".", 1)[-1]
            if module_name == _module_name(path):
                continue
            if module_name in text or tail in text:
                by_module[module_name].append(rel)
    return by_module


def _registered_models() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for model_name, entry in CORE_MODEL_REGISTRY.items():
        if entry.model_file:
            mapping[entry.model_file] = model_name
    return mapping


def _active_status(path: Path, registered_lookup: dict[str, str]) -> tuple[str, str]:
    name = path.name
    if path.parent == CORE_DIR:
        if name in registered_lookup:
            model_name = registered_lookup[name]
            entry = CORE_MODEL_REGISTRY[model_name]
            return ("active" if entry.enabled else "inactive", model_name)
        if name == "__init__.py":
            return ("active", "")
        return ("inactive", "")
    if path.parent == FLUX_DIR:
        return ("active", "")
    if path.parent == SPECIAL_DIR:
        return ("active", "")
    return ("active", "")


def build_file_inventory(all_paths: list[Path]) -> list[dict[str, object]]:
    imported_by = _find_imported_by(all_paths)
    registered_lookup = _registered_models()
    rows: list[dict[str, object]] = []
    for path in all_paths:
        if path.name == "__pycache__":
            continue
        tree = _parse_tree(path)
        module_name = _module_name(path)
        if path.is_relative_to(CORE_DIR):
            module_type = "core"
        elif path.is_relative_to(FLUX_DIR):
            module_type = "flux"
        elif path.is_relative_to(SPECIAL_DIR):
            module_type = "special"
        elif path == PROJECT_ROOT / "models" / "registry.py" or path == CORE_DIR / "__init__.py":
            module_type = "registry"
        elif path.is_relative_to(TESTS_DIR):
            module_type = "test"
        else:
            module_type = "script"

        active_or_inactive, registered_model = _active_status(path, registered_lookup)
        rows.append(
            {
                "file_path": _relative(path),
                "module_type": module_type,
                "class_or_function_names": "; ".join(
                    [node.name for node in _class_nodes(tree)] + [node.name for node in _function_nodes(tree)]
                ),
                "imported_by": "; ".join(sorted(set(imported_by.get(module_name, [])))),
                "imports_from_flux": "; ".join(_imports_from_prefix(tree, "flux")),
                "imports_from_core": "; ".join(_imports_from_prefix(tree, "core")),
                "registered_model": registered_model,
                "active_or_inactive": active_or_inactive,
                "notes": "registry entrypoint" if path == CORE_DIR / "__init__.py" else "",
            }
        )
    return rows


def _infer_formula_type(name: str, text: str) -> str:
    lower_name = name.lower()
    lower_text = text.lower()
    for formula_type, markers in INLINE_FUNC_TYPES.items():
        if any(marker in lower_name for marker in markers):
            return formula_type
        if any(marker in lower_text for marker in markers):
            return formula_type
    return "other"


def _snippet(lines: list[str], start: int, end: int) -> str:
    return "\n".join(line.rstrip() for line in lines[start - 1 : end]).strip()


def _body_contains_hydro_logic(text: str) -> bool:
    if any(token in text for token in HYDRO_TOKENS):
        return True
    return bool(re.search(r"\b(min|max|relu|sigmoid|exp|tanh|pow|clamp)\b", text))


def _line_is_inline_formula_candidate(line: str) -> bool:
    stripped = line.split("#", 1)[0].strip()
    if not stripped.startswith("flux_"):
        return False
    if any(token in stripped for token in ("torch.sigmoid", "soft_gate_", "F.softplus", "torch.exp", "torch.tanh", "torch.where")):
        return True
    if re.search(r"=\s*[^#\n]*\bpow\(", stripped):
        return True
    return False


def _called_flux_names(path: Path) -> set[str]:
    text = _read(path)
    tree = _parse_tree(path)
    called = set()
    for local_name, source_name in _flux_import_bindings(tree).items():
        if re.search(rf"\b{local_name}\s*\(", text):
            called.add(source_name)
    return called


def _collect_flux_usage(flux_functions: dict[str, dict[str, object]], model_paths: list[Path], test_script_paths: list[Path]) -> None:
    for path in model_paths:
        tree = _parse_tree(path)
        text = _read(path)
        caller_name = path.stem
        if path.is_relative_to(CORE_DIR):
            caller_group = "core"
        else:
            caller_group = "special"
        for local_name, name in _flux_import_bindings(tree).items():
            if re.search(rf"\b{local_name}\s*\(", text) and name in flux_functions:
                key = "called_by_core_models" if caller_group == "core" else "called_by_special_models"
                flux_functions[name][key].add(caller_name)
    for path in test_script_paths:
        text = _read(path)
        for name in flux_functions:
            if re.search(rf"\b{name}\b", text):
                flux_functions[name]["called_by_tests"].add(_relative(path))


def _flux_formula_type(path: Path, fn: ast.FunctionDef) -> str:
    return _infer_formula_type(fn.name, _snippet(_read(path).splitlines(), fn.lineno, fn.end_lineno or fn.lineno))


def build_flux_inventory() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    flux_functions: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for path in sorted(FLUX_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = _parse_tree(path)
        for fn in _function_nodes(tree):
            flux_functions[fn.name] = {
                "flux_file": _relative(path),
                "function_name": fn.name,
                "line_start": fn.lineno,
                "line_end": fn.end_lineno or fn.lineno,
                "formula_type": _flux_formula_type(path, fn),
                "called_by_core_models": set(),
                "called_by_special_models": set(),
                "called_by_tests": set(),
                "active_usage_status": "",
                "duplicates_core_formula": "",
                "matlab_counterpart_if_known": "likely" if re.search(r"_\d+$", fn.name) else "",
                "validation_status_if_known": "",
                "keep_or_deprecate": "",
                "notes": "",
            }

    model_paths = list(CORE_DIR.glob("*.py")) + list(SPECIAL_DIR.glob("*.py"))
    test_script_paths = list(TESTS_DIR.glob("*.py")) + list(SCRIPTS_DIR.glob("*.py"))
    _collect_flux_usage(flux_functions, model_paths, test_script_paths)

    for name, data in flux_functions.items():
        core_calls = sorted(data["called_by_core_models"])
        special_calls = sorted(data["called_by_special_models"])
        test_calls = sorted(data["called_by_tests"])
        active = bool(core_calls or special_calls)
        keep_or_deprecate = "keep"
        notes = []
        if not active and test_calls:
            keep_or_deprecate = "deprecate_candidate"
            notes.append("only referenced by tests/scripts")
        elif not active:
            keep_or_deprecate = "inactive_candidate"
            notes.append("no active core/special callers")
        if name in {"baseflow_6", "interception_4", "phenology_1", "infiltration_1", "infiltration_2", "evap_2", "snowfall_1", "rainfall_1"}:
            notes.append("has competing or model-local core variant")
        data["active_usage_status"] = "active" if active else "unused"
        data["validation_status_if_known"] = "tested" if test_calls else ""
        data["keep_or_deprecate"] = keep_or_deprecate
        data["notes"] = "; ".join(notes)
        rows.append(
            {
                "flux_file": data["flux_file"],
                "function_name": name,
                "line_start": data["line_start"],
                "line_end": data["line_end"],
                "formula_type": data["formula_type"],
                "called_by_core_models": "; ".join(core_calls),
                "called_by_special_models": "; ".join(special_calls),
                "called_by_tests": "; ".join(test_calls),
                "active_usage_status": data["active_usage_status"],
                "duplicates_core_formula": "",
                "matlab_counterpart_if_known": data["matlab_counterpart_if_known"],
                "validation_status_if_known": data["validation_status_if_known"],
                "keep_or_deprecate": data["keep_or_deprecate"],
                "notes": data["notes"],
            }
        )
    return rows, flux_functions


def _extract_variables_and_params(args: ast.arguments, model_name: str) -> tuple[list[str], list[str]]:
    names = [arg.arg for arg in args.args]
    states = []
    params = []
    for name in names:
        if re.fullmatch(r"S\d+|Sn|Sc\d+", name):
            states.append(name)
        elif name not in {"P", "T", "PET", "doy", "delta_t", "nearzero", "mean_P", "return_diagnostics"}:
            params.append(name)
    return states, params


def _duplicate_flux_equivalent(name: str) -> str:
    mapping = {
        "evap_linear_deficit": "evap_12",
        "exchange_1": "exchange_3",
        "infiltration_1": "infiltration_1",
        "infiltration_2": "infiltration_2",
        "evap_2": "evap_2",
        "interception_1": "interception_1",
        "depression_1": "depression_1",
        "evap_7": "evap_7",
        "saturation_1": "saturation_1",
        "baseflow_1": "baseflow_1",
        "recharge_3": "recharge_3",
        "snowfall_1": "snowfall_1",
        "rainfall_1": "rainfall_1",
        "melt_1": "melt_1",
        "interception_4": "interception_4",
        "phenology_1": "phenology_1",
        "baseflow_6": "baseflow_6",
    }
    return mapping.get(name, "")


def _recommended_flux_name(model_name: str, helper_name: str, formula_type: str) -> str:
    if model_name == "tcm" and helper_name == "baseflow_6":
        return "baseflow_tcm_quadratic_threshold"
    if model_name == "mopex5" and helper_name == "phenology_1":
        return "phenology_mopex_temperature_ramp"
    if model_name == "mopex4" and helper_name == "interception_4":
        return "interception_mopex_seasonal_softplus"
    if model_name == "mopex2" and helper_name in {"snowfall_1", "rainfall_1", "melt_1"}:
        return f"{helper_name.removesuffix('_1')}_mopex_temperature_gate"
    if model_name == "mopex1" and helper_name in {"evap_7", "saturation_1", "baseflow_1", "recharge_3"}:
        return f"{formula_type}_{model_name}_variant"
    if model_name == "modhydrolog":
        return f"{formula_type}_{model_name}_safe"
    if model_name == "ihacres" and helper_name == "evap_linear_deficit":
        return "evap_ihacres_linear_deficit"
    if model_name == "gr4j":
        return f"{formula_type}_gr4j_analytical"
    return f"{formula_type}_{model_name}_{helper_name}"


def build_core_inline_inventory(flux_functions: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate_index = 1
    for path in sorted(CORE_DIR.glob("*.py")):
        if path.name in {"__init__.py", "shm.py"}:
            continue
        lines = _read(path).splitlines()
        tree = _parse_tree(path)
        model_name = path.stem
        for fn in _function_nodes(tree):
            fn_text = _snippet(lines, fn.lineno, fn.end_lineno or fn.lineno)
            is_step = fn.name.endswith(("_step", "_step_pre", "_step_post"))
            helper_candidate = not is_step and fn.name != "create_initial_state"
            direct_flux_lines = []
            if is_step:
                for index, line in enumerate(fn_text.splitlines(), start=fn.lineno):
                    stripped = line.strip()
                    if _line_is_inline_formula_candidate(stripped):
                        direct_flux_lines.append((index, index, stripped))
            if helper_candidate and _body_contains_hydro_logic(fn_text):
                states, params = _extract_variables_and_params(fn.args, model_name)
                likely = _duplicate_flux_equivalent(fn.name)
                duplicate = "yes" if likely in flux_functions else "no"
                note_key = (_relative(path), fn.name)
                reason = DANGEROUS_DUPLICATE_NOTES.get(note_key, "")
                should_move = "yes"
                risk = "medium"
                if model_name in {"modhydrolog", "mopex1", "mopex2", "mopex4"}:
                    should_move = "conditional"
                    risk = "high"
                    if not reason:
                        reason = "Local helper appears intentionally model-specific or behavior-divergent."
                elif model_name in {"gr4j"}:
                    should_move = "no"
                    risk = "medium"
                    reason = reason or "Analytical/helper logic is tightly coupled to model execution and naming."
                elif model_name in {"tcm", "mopex5", "ihacres"}:
                    should_move = "yes"
                    risk = "low" if model_name == "tcm" else "medium"
                    reason = reason or "Reusable process helper with clear standalone process identity."
                rows.append(
                    {
                        "candidate_id": f"C{candidate_index:03d}",
                        "core_file": _relative(path),
                        "line_start": fn.lineno,
                        "line_end": fn.end_lineno or fn.lineno,
                        "model_name": model_name,
                        "formula_type": _infer_formula_type(fn.name, fn_text),
                        "code_snippet": fn_text,
                        "variables": "; ".join(states),
                        "parameters": "; ".join(params),
                        "uses_soft_gate": "yes" if "soft_gate_" in fn_text or "sigmoid" in fn_text else "no",
                        "likely_flux_equivalent": likely,
                        "duplicate_of_flux_function": duplicate,
                        "should_move_to_flux": should_move,
                        "recommended_flux_function_name": _recommended_flux_name(model_name, fn.name, _infer_formula_type(fn.name, fn_text)),
                        "migration_risk": risk,
                        "reason": reason,
                    }
                )
                candidate_index += 1
            for start, end, snippet in direct_flux_lines:
                states, params = _extract_variables_and_params(fn.args, model_name)
                formula_type = _infer_formula_type("flux_line", snippet)
                likely = ""
                should_move = "no"
                risk = "low"
                reason = "Inline flux assignment inside step function; often sequencing glue unless repeated or reusable."
                if "torch.sigmoid" in snippet or "soft_gate_" in snippet or "softplus" in snippet:
                    should_move = "conditional"
                    risk = "medium"
                    reason = "Inline process expression may merit dedicated model-specific flux extraction if reused."
                rows.append(
                    {
                        "candidate_id": f"C{candidate_index:03d}",
                        "core_file": _relative(path),
                        "line_start": start,
                        "line_end": end,
                        "model_name": model_name,
                        "formula_type": formula_type,
                        "code_snippet": snippet,
                        "variables": "; ".join(states),
                        "parameters": "; ".join(params),
                        "uses_soft_gate": "yes" if "soft_gate_" in snippet or "sigmoid" in snippet else "no",
                        "likely_flux_equivalent": likely,
                        "duplicate_of_flux_function": "no",
                        "should_move_to_flux": should_move,
                        "recommended_flux_function_name": _recommended_flux_name(model_name, f"inline_{candidate_index}", formula_type),
                        "migration_risk": risk,
                        "reason": reason,
                    }
                )
                candidate_index += 1
    return rows


def _update_flux_duplicates(flux_rows: list[dict[str, object]], core_inline_rows: list[dict[str, object]]) -> None:
    duplicates: dict[str, list[str]] = defaultdict(list)
    for row in core_inline_rows:
        equivalent = str(row["likely_flux_equivalent"])
        if equivalent:
            duplicates[equivalent].append(f"{row['model_name']}:{Path(str(row['core_file'])).name}:{row['candidate_id']}")
    for row in flux_rows:
        duplicates_list = duplicates.get(str(row["function_name"]), [])
        row["duplicates_core_formula"] = "; ".join(duplicates_list)
        if duplicates_list and row["keep_or_deprecate"] == "keep":
            row["notes"] = "; ".join(filter(None, [row["notes"], "has core-local duplicate"]))


def build_call_graph(core_inline_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    inline_by_model: dict[str, list[str]] = defaultdict(list)
    for row in core_inline_rows:
        inline_by_model[str(row["model_name"])].append(str(row["candidate_id"]))

    rows: list[dict[str, object]] = []
    for model_name, entry in sorted(CORE_MODEL_REGISTRY.items()):
        if model_name in {"shm"}:
            continue
        path = CORE_DIR / entry.model_file
        flux_calls = sorted(_called_flux_names(path))
        tree = _parse_tree(path)
        special_calls = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("."):
                if any(alias.name.endswith("_step") for alias in node.names):
                    special_calls.extend(alias.name for alias in node.names if alias.name.endswith("_step"))
        rows.append(
            {
                "model_name": model_name,
                "core_file": _relative(path),
                "registered": "yes" if entry.enabled else "no",
                "flux_functions_called": "; ".join(flux_calls),
                "special_functions_called": "; ".join(sorted(set(special_calls))),
                "inline_formula_candidates": "; ".join(sorted(inline_by_model.get(model_name, []))),
                "unregistered_dependencies": "",
                "notes": "uses deficit-store sign override" if model_name in {"ihacres", "penman", "tcm", "topmodel"} else "",
            }
        )
    return rows


def build_issue_list(core_inline_rows: list[dict[str, object]], flux_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    issue_index = 1
    for row in core_inline_rows:
        move = str(row["should_move_to_flux"])
        category = "B"
        severity = "low"
        summary = "Inline core formula should stay in core."
        action = "Keep inline and document model-specific rationale."
        priority = "low"
        risk_if_not_fixed = "Architecture remains harder to explain."
        risk_if_migrated = "Potential behavior drift if extracted incorrectly."
        if move == "yes":
            category = "A"
            severity = "high" if row["migration_risk"] == "low" else "medium"
            summary = "Inline hydrological formula in core appears migratable to flux."
            action = f"Extract to flux as {row['recommended_flux_function_name']} with behavior-preservation tests."
            priority = "high"
            risk_if_not_fixed = "Core/flux separation stays blurry for paper-level architecture description."
            risk_if_migrated = "Extraction may alter ordering or clamps if not wrapped carefully."
        elif move == "conditional":
            category = "D" if row["duplicate_of_flux_function"] == "yes" else "B"
            severity = "high" if row["migration_risk"] == "high" else "medium"
            summary = "Core-local formula competes with or intentionally diverges from shared flux naming."
            action = "If extracted, create model-specific flux function rather than changing shared generic behavior."
            priority = "medium"
            risk_if_not_fixed = "Future contributors may reuse the wrong shared formula."
            risk_if_migrated = "Generic extraction could break existing models relying on corrected local behavior."
        issues.append(
            {
                "issue_id": f"I{issue_index:03d}",
                "category": category,
                "severity": severity,
                "file_path": row["core_file"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "related_model": row["model_name"],
                "related_flux_function": row["likely_flux_equivalent"],
                "issue_summary": summary,
                "recommended_action": action,
                "migration_priority": priority,
                "risk_if_not_fixed": risk_if_not_fixed,
                "risk_if_migrated": risk_if_migrated,
                "notes": row["reason"],
            }
        )
        issue_index += 1

    for row in flux_rows:
        if row["active_usage_status"] == "unused":
            severity = "medium" if row["duplicates_core_formula"] else "low"
            category = "C"
            action = "Mark inactive/deprecate after confirming no hidden runtime dependency."
            notes = row["notes"]
            if row["duplicates_core_formula"]:
                category = "D"
                action = "Do not reuse generically without validating against active core-local variants."
                notes = "; ".join(filter(None, [notes, "unused shared function overlaps with local corrected formula"]))
            issues.append(
                {
                    "issue_id": f"I{issue_index:03d}",
                    "category": category,
                    "severity": severity,
                    "file_path": row["flux_file"],
                    "line_start": row["line_start"],
                    "line_end": row["line_end"],
                    "related_model": "",
                    "related_flux_function": row["function_name"],
                    "issue_summary": "Flux function has no active core/special callers.",
                    "recommended_action": action,
                    "migration_priority": "low",
                    "risk_if_not_fixed": "Dead or misleading API surface remains in flux layer.",
                    "risk_if_migrated": "Deprecation could break hidden external imports if any exist.",
                    "notes": notes,
                }
            )
            issue_index += 1
        if row["duplicates_core_formula"]:
            issues.append(
                {
                    "issue_id": f"I{issue_index:03d}",
                    "category": "E",
                    "severity": "medium",
                    "file_path": row["flux_file"],
                    "line_start": row["line_start"],
                    "line_end": row["line_end"],
                    "related_model": "",
                    "related_flux_function": row["function_name"],
                    "issue_summary": "Shared flux function duplicates or overlaps with one or more core-local formulas.",
                    "recommended_action": "Document divergence and decide between deprecating generic reuse or adding model-specific flux wrappers.",
                    "migration_priority": "medium",
                    "risk_if_not_fixed": "Architecture remains ambiguous; contributors may call the wrong implementation.",
                    "risk_if_migrated": "Unifying too aggressively can regress validated models.",
                    "notes": row["duplicates_core_formula"],
                }
            )
            issue_index += 1
    return issues


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _high_priority_candidates(core_inline_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in core_inline_rows if row["should_move_to_flux"] == "yes"]


def _write_call_graph_markdown(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Model / Flux Call Graph",
        "",
        "| model | core file | registered | flux functions | inline candidates | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['core_file']} | {row['registered']} | "
            f"{row['flux_functions_called'] or '-'} | {row['inline_formula_candidates'] or '-'} | {row['notes'] or '-'} |"
        )
    CALL_GRAPH_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_refactor_plan(core_inline_rows: list[dict[str, object]]) -> None:
    high_priority = _high_priority_candidates(core_inline_rows)
    conditional = [row for row in core_inline_rows if row["should_move_to_flux"] == "conditional"]
    lines = [
        "# Refactor Plan",
        "",
        "## Principles",
        "- Move only standalone hydrological process formulas from `core` into `flux`.",
        "- Prefer model-specific flux names when local behavior intentionally diverges from existing generic functions.",
        "- Keep state sequencing, routing order, and water-balance fixes in `core`.",
        "",
        "## High-Priority Migrations",
    ]
    for row in high_priority:
        lines.extend(
            [
                f"### {row['candidate_id']} {row['model_name']} {row['formula_type']}",
                f"- Current location: `{row['core_file']}:{row['line_start']}`",
                f"- Proposed flux function: `{row['recommended_flux_function_name']}`",
                f"- Current snippet:",
                "```python",
                str(row["code_snippet"]),
                "```",
                f"- Proposed signature: `{row['recommended_flux_function_name']}({', '.join(filter(None, [str(row['parameters']).replace('; ', ', '), str(row['variables']).replace('; ', ', ')]))}, nearzero=1e-6)`",
                f"- Expected inputs: parameters `{row['parameters']}`; states/variables `{row['variables']}`",
                "- Expected outputs: single flux tensor or helper tuple matching current local helper behavior.",
                f"- Soft-gate dependency: {row['uses_soft_gate']}",
                "- Core replacement concept: import the new flux function and replace the local helper call with an equivalent flux-layer call while preserving order and clipping.",
                "- Tests required after migration: behavior-preservation diff test, affected model water-balance regression, independent finite-value/gradient diagnostic, smoke simulation.",
                "",
            ]
        )
    lines.extend(
        [
            "## Conditional / Model-Specific Extractions",
        ]
    )
    for row in conditional:
        lines.extend(
            [
                f"- `{row['candidate_id']}` `{row['core_file']}:{row['line_start']}` -> `{row['recommended_flux_function_name']}` if and only if extracted as model-specific behavior, not by modifying a shared generic flux.",
            ]
        )
    REFACTOR_PLAN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_requirements(core_inline_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Refactor Validation Requirements",
        "",
        "For every future migration from `core` to `flux`, require all of the following:",
        "",
        "1. Behavior-preservation test",
        "- Compare pre-refactor and post-refactor outputs for the moved helper over representative grids and synthetic states within floating-point tolerance.",
        "",
        "2. Core water-balance regression",
        "- Run the affected registered model through the existing core water-balance checks and require no new residual failures.",
        "",
        "3. Formula diagnostic test",
        "- Add or update an isolated formula wrapper test confirming finite outputs and finite autograd gradients for the moved flux function.",
        "",
        "4. Smoke simulation",
        "- Run deterministic short simulations for the affected model and compare Q/Ea totals, peaks, and sign constraints.",
        "",
        "5. Full test suite",
        "- `python -m pytest tests/test_formula_smoothing_diagnostics.py -v`",
        "- `python -m pytest tests/test_core_water_balance.py -v`",
        "- `python -m pytest tests/test_unithydro_consistency.py -v`",
        "",
        "## Candidate-Specific Notes",
    ]
    for row in _high_priority_candidates(core_inline_rows):
        lines.append(
            f"- `{row['candidate_id']}` `{row['recommended_flux_function_name']}`: validate behavior of `{row['model_name']}` before/after extraction, including any soft-gate behavior noted as `{row['uses_soft_gate']}`."
        )
    VALIDATION_REQUIREMENTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    file_inventory: list[dict[str, object]],
    flux_rows: list[dict[str, object]],
    core_inline_rows: list[dict[str, object]],
    issue_rows: list[dict[str, object]],
    call_graph_rows: list[dict[str, object]],
) -> None:
    registered_models = [name for name, entry in CORE_MODEL_REGISTRY.items() if entry.enabled]
    runnable_count = sum(1 for entry in CORE_MODEL_REGISTRY.values() if entry.enabled)
    unused_flux = [row for row in flux_rows if row["active_usage_status"] == "unused"]
    duplicates = [row for row in flux_rows if row["duplicates_core_formula"]]
    high_priority = _high_priority_candidates(core_inline_rows)
    keep_inline = [row for row in core_inline_rows if row["should_move_to_flux"] == "no"]
    conditional = [row for row in core_inline_rows if row["should_move_to_flux"] == "conditional"]

    lines = [
        "# Architecture Audit Report",
        "",
        "## Scope",
        "Audit of `models/core` and `models/flux` separation for paper-quality architecture clarity.",
        "",
        "## Files Inspected",
        f"- Core files: {sum(1 for row in file_inventory if row['module_type'] == 'core')}",
        f"- Flux files: {sum(1 for row in file_inventory if row['module_type'] == 'flux')}",
        f"- Special files: {sum(1 for row in file_inventory if row['module_type'] == 'special')}",
        f"- Registry/test/script files: {sum(1 for row in file_inventory if row['module_type'] in {'registry', 'test', 'script'})}",
        "",
        "## Current Architecture Summary",
        "- `core` is the model execution layer: parameter/basic metadata, state sequencing, routing assembly, and water-balance accounting.",
        "- `flux` is the shared formula layer, including explicitly named model-specific formulas when their semantics differ from generic equations.",
        "- Core-local analytical helpers are retained only when tightly coupled to a model's state transform and not meaningful as standalone flux catalogue entries.",
        "",
        "## Registered Core Model List",
        f"- Runnable registered core models: {runnable_count}",
        f"- Active registered names: {', '.join(sorted(name for name, entry in CORE_MODEL_REGISTRY.items() if entry.enabled))}",
        "",
        "## Flux Function Usage Summary",
        f"- Flux functions audited: {len(flux_rows)}",
        f"- Active flux functions: {sum(1 for row in flux_rows if row['active_usage_status'] == 'active')}",
        f"- Unused flux functions: {len(unused_flux)}",
        "",
        "## Inline Formulas Found In Core",
        f"- Inline/core-local formula candidates: {len(core_inline_rows)}",
        f"- High-priority migration candidates: {len(high_priority)}",
        f"- Conditional/model-specific candidates: {len(conditional)}",
        "",
        "## Unused Flux Functions",
        *(f"- `{row['function_name']}` in `{row['flux_file']}`: {row['notes'] or 'no active core/special callers'}" for row in unused_flux[:20]),
        "",
        "## Duplicate Formulas",
        f"- Flux functions with overlapping core-local duplicates: {len(duplicates)}",
        *(f"- `{row['function_name']}` duplicated by {row['duplicates_core_formula']}" for row in duplicates[:20]),
        "",
        "## High-Priority Architecture Issues",
        *(f"- `{row['candidate_id']}` `{row['core_file']}:{row['line_start']}` -> `{row['recommended_flux_function_name']}` ({row['reason']})" for row in high_priority),
        "",
        "## Recommended Migrations",
        *(f"- Extract `{row['candidate_id']}` as `{row['recommended_flux_function_name']}` in `flux`, keeping behavior byte-for-byte equivalent." for row in high_priority),
        "",
        "## What Should Stay In Core",
        "- State update order, store coupling, routing assembly, and water-balance accounting fixes.",
        "- Analytical/store-coupled helpers that are not meaningful as generic standalone process functions, such as GR4J analytical storage transforms.",
        "",
        "## What Should Remain In Flux",
        "- Shared reusable hydrological process equations: evap, baseflow, interflow, percolation, recharge, interception, snow/rain partition, melt, soft gates, area, capillary, soil moisture, and related independent formulas.",
        "- Model-specific extracted helpers, when needed, should also live in `flux` but with explicit names such as `baseflow_tcm_quadratic_threshold` rather than by mutating existing generic functions.",
        "",
        "## What Should Be Deprecated Or Marked Inactive",
        *(f"- `{row['function_name']}` in `{row['flux_file']}`" for row in unused_flux[:20]),
        "",
        "## Refactor Risk Assessment",
        "- Highest risk is replacing validated model-specific formulas with generic shared functions that have different clipping or smoothing semantics.",
        "- Completed model-specific migrations retain explicit names so their semantics remain visible and independently testable.",
        "",
        "## Suggested Order Of Implementation",
        "1. Migrate exact duplicates with low migration risk and high readability benefit.",
        "2. Add model-specific flux functions for corrected local helpers that should leave `core` but must not overwrite generic behavior.",
        "3. Mark clearly unused or dangerous generic flux functions as inactive/deprecate candidates after confirming no external dependency.",
        "4. Only then consider broader unification of MOPEX or MODHYDROLOG local variants.",
        "",
        "## Required Validation After Refactor",
        "- Behavior-preservation before/after comparisons for each moved helper.",
        "- Core water-balance regressions for each affected model.",
        "- Formula finite-value and finite-gradient diagnostics.",
        "- Smoke simulations for Q/Ea consistency.",
        "- Full standard suite: smoothing diagnostics, core water balance, and unithydro consistency.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = (
        sorted(CORE_DIR.glob("*.py"))
        + sorted(FLUX_DIR.glob("*.py"))
        + sorted(SPECIAL_DIR.glob("*.py"))
        + [PROJECT_ROOT / "models" / "registry.py", PROJECT_ROOT / "models" / "hydrology_model.py"]
        + sorted(TESTS_DIR.glob("*.py"))
        + sorted(SCRIPTS_DIR.glob("*.py"))
    )
    file_inventory = build_file_inventory(all_paths)
    flux_rows, flux_functions = build_flux_inventory()
    core_inline_rows = build_core_inline_inventory(flux_functions)
    _update_flux_duplicates(flux_rows, core_inline_rows)
    call_graph_rows = build_call_graph(core_inline_rows)
    issue_rows = build_issue_list(core_inline_rows, flux_rows)

    _write_csv(
        FILE_INVENTORY_CSV,
        file_inventory,
        [
            "file_path",
            "module_type",
            "class_or_function_names",
            "imported_by",
            "imports_from_flux",
            "imports_from_core",
            "registered_model",
            "active_or_inactive",
            "notes",
        ],
    )
    _write_csv(
        CORE_INLINE_CSV,
        core_inline_rows,
        [
            "candidate_id",
            "core_file",
            "line_start",
            "line_end",
            "model_name",
            "formula_type",
            "code_snippet",
            "variables",
            "parameters",
            "uses_soft_gate",
            "likely_flux_equivalent",
            "duplicate_of_flux_function",
            "should_move_to_flux",
            "recommended_flux_function_name",
            "migration_risk",
            "reason",
        ],
    )
    _write_csv(
        FLUX_USAGE_CSV,
        flux_rows,
        [
            "flux_file",
            "function_name",
            "line_start",
            "line_end",
            "formula_type",
            "called_by_core_models",
            "called_by_special_models",
            "called_by_tests",
            "active_usage_status",
            "duplicates_core_formula",
            "matlab_counterpart_if_known",
            "validation_status_if_known",
            "keep_or_deprecate",
            "notes",
        ],
    )
    _write_csv(
        CALL_GRAPH_CSV,
        call_graph_rows,
        [
            "model_name",
            "core_file",
            "registered",
            "flux_functions_called",
            "special_functions_called",
            "inline_formula_candidates",
            "unregistered_dependencies",
            "notes",
        ],
    )
    _write_call_graph_markdown(call_graph_rows)
    _write_csv(
        ISSUE_LIST_CSV,
        issue_rows,
        [
            "issue_id",
            "category",
            "severity",
            "file_path",
            "line_start",
            "line_end",
            "related_model",
            "related_flux_function",
            "issue_summary",
            "recommended_action",
            "migration_priority",
            "risk_if_not_fixed",
            "risk_if_migrated",
            "notes",
        ],
    )
    _write_refactor_plan(core_inline_rows)
    _write_validation_requirements(core_inline_rows)
    _write_report(file_inventory, flux_rows, core_inline_rows, issue_rows, call_graph_rows)

    print(f"Wrote {FILE_INVENTORY_CSV}")
    print(f"Wrote {CORE_INLINE_CSV}")
    print(f"Wrote {FLUX_USAGE_CSV}")
    print(f"Wrote {CALL_GRAPH_CSV}")
    print(f"Wrote {CALL_GRAPH_MD}")
    print(f"Wrote {ISSUE_LIST_CSV}")
    print(f"Wrote {REFACTOR_PLAN_MD}")
    print(f"Wrote {VALIDATION_REQUIREMENTS_MD}")
    print(f"Wrote {REPORT_MD}")
    print(f"Registered runnable core models: {sum(1 for entry in CORE_MODEL_REGISTRY.values() if entry.enabled)}")
    print(f"Flux functions audited: {len(flux_rows)}")
    print(f"Inline core formula candidates: {len(core_inline_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
