#!/usr/bin/env python3
"""Conservative static scan for PyTorch autograd risks in a repository.

The scanner reports review candidates; it does not prove a differentiability failure.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "__pycache__",
}

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    severity: str
    rule_id: str
    function: str
    message: str
    snippet: str


def dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


class RiskVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, root: Path, lines: list[str]) -> None:
        self.path = path
        self.root = root
        self.lines = lines
        self.function_stack: list[str] = []
        self.findings: list[Finding] = []
        self._seen: set[tuple[int, int, str]] = set()

    @property
    def function(self) -> str:
        return ".".join(self.function_stack) if self.function_stack else "<module>"

    def add(self, node: ast.AST, severity: str, rule_id: str, message: str) -> None:
        line = int(getattr(node, "lineno", 0) or 0)
        col = int(getattr(node, "col_offset", 0) or 0)
        key = (line, col, rule_id)
        if key in self._seen:
            return
        self._seen.add(key)
        snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        try:
            rel = str(self.path.relative_to(self.root))
        except ValueError:
            rel = str(self.path)
        self.findings.append(
            Finding(
                path=rel,
                line=line,
                column=col,
                severity=severity,
                rule_id=rule_id,
                function=self.function,
                message=message,
                snippet=snippet[:240],
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "data":
            self.add(
                node,
                "high",
                "AUTOGRAD_DATA_ACCESS",
                "`.data` can bypass autograd version tracking or detach updates from the graph.",
            )
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            name = dotted_name(item.context_expr.func) if isinstance(item.context_expr, ast.Call) else dotted_name(item.context_expr)
            if name.endswith("no_grad"):
                self.add(
                    item.context_expr,
                    "high",
                    "NO_GRAD_CONTEXT",
                    "`torch.no_grad()` in an active training path disables graph construction.",
                )
            if name.endswith("inference_mode"):
                self.add(
                    item.context_expr,
                    "high",
                    "INFERENCE_MODE_CONTEXT",
                    "`torch.inference_mode()` in an active training path disables autograd.",
                )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.add(
            node,
            "medium",
            "INPLACE_AUGMENTED_ASSIGN",
            "In-place recurrent/state mutation can invalidate values required by backward; verify runtime behavior.",
        )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div):
            self.add(
                node,
                "medium",
                "DIVISION_STABILITY",
                "Division may create unstable or non-finite gradients near a zero denominator.",
            )
        elif isinstance(node.op, ast.Pow):
            self.add(
                node,
                "medium",
                "POWER_DOMAIN",
                "Power operations require domain and boundary checks, especially for learned fractional exponents.",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        base = name.rsplit(".", 1)[-1]
        line_text = self.lines[node.lineno - 1] if 0 < getattr(node, "lineno", 0) <= len(self.lines) else ""
        context_text = f"{self.function} {line_text}".lower()

        if base in {"detach", "detach_"}:
            self.add(
                node,
                "high",
                "DETACH_IN_GRAPH",
                "Detach truncates gradients; verify that it is an intentional boundary rather than a model-step operation.",
            )
        elif base == "item":
            self.add(
                node,
                "high",
                "TENSOR_ITEM_CONVERSION",
                "`.item()` converts a tensor to a Python scalar and can sever parameter dependence.",
            )
        elif base == "numpy":
            self.add(
                node,
                "high",
                "NUMPY_CONVERSION",
                "NumPy conversion leaves the autograd graph; keep it outside the loss path.",
            )
        elif name in {"float", "int", "bool"}:
            self.add(
                node,
                "high",
                "PYTHON_SCALAR_CONVERSION",
                "Python scalar conversion may sever a tensor-dependent computation or create discrete control flow.",
            )
        elif name in {"torch.tensor", "Tensor"}:
            self.add(
                node,
                "medium",
                "TENSOR_REWRAP",
                "Constructing a new tensor from an existing tensor can detach history; inspect the argument source.",
            )
        elif "nan_to_num" in name and "grad" in context_text:
            self.add(
                node,
                "high",
                "GRADIENT_SANITIZATION",
                "Replacing NaN/Inf gradients hides invalid training; convert this to a detector during auditing.",
            )
        elif base in {"argmax", "argmin", "round", "floor", "ceil", "sign", "bucketize", "searchsorted"}:
            self.add(
                node,
                "high",
                "DISCRETE_OPERATION",
                "Discrete selection is non-differentiable with respect to the selected structure.",
            )
        elif base in {"long", "int", "bool", "to"} and any(token in line_text for token in ("long", "int", "bool")):
            self.add(
                node,
                "medium",
                "INTEGER_OR_BOOL_CAST",
                "Integer/bool casting can create a non-differentiable parameter-dependent path.",
            )
        elif base in {"clamp", "clamp_", "clamp_min", "clamp_max", "minimum", "maximum", "relu", "hardtanh"}:
            self.add(
                node,
                "medium",
                "PIECEWISE_ZERO_GRADIENT",
                "Piecewise bounds can create zero-gradient regions; measure hit and occupancy rates.",
            )
        elif base == "where":
            self.add(
                node,
                "medium",
                "WHERE_BRANCH_DOMAIN",
                "Both `where` branches are evaluated; ensure inactive branches remain numerically valid.",
            )
        elif base in {"sqrt", "rsqrt", "log", "log1p", "reciprocal"}:
            self.add(
                node,
                "medium",
                "SINGULAR_DOMAIN_OPERATION",
                "Check the operation domain and derivative near zero or invalid inputs.",
            )
        elif base in {"pow", "float_power"}:
            self.add(
                node,
                "medium",
                "POWER_DOMAIN",
                "Power operations require domain and boundary checks, especially for learned fractional exponents.",
            )
        elif base in {"exp", "expm1", "softplus"}:
            self.add(
                node,
                "low",
                "EXP_RANGE",
                "Check float32 overflow/underflow across the full parameter and forcing range.",
            )
        elif base.endswith("_") and base not in {"requires_grad_", "retain_grad"}:
            self.add(
                node,
                "low",
                "INPLACE_METHOD",
                "In-place tensor operations require review in recurrent graphs and on values saved for backward.",
            )

        self.generic_visit(node)


def iter_python_files(root: Path, excludes: set[str], max_bytes: int) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in excludes for part in path.parts):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        yield path


def scan(root: Path, excludes: set[str], max_bytes: int) -> tuple[list[Finding], list[dict[str, str]]]:
    findings: list[Finding] = []
    parse_errors: list[dict[str, str]] = []
    for path in iter_python_files(root, excludes, max_bytes):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                parse_errors.append({"path": str(path), "error": str(exc)})
                continue
        except OSError as exc:
            parse_errors.append({"path": str(path), "error": str(exc)})
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            parse_errors.append({"path": str(path), "error": f"SyntaxError: {exc}"})
            continue
        visitor = RiskVisitor(path, root, text.splitlines())
        visitor.visit(tree)
        findings.extend(visitor.findings)
    findings.sort(key=lambda x: (SEVERITY_ORDER.get(x.severity, 9), x.path, x.line, x.rule_id))
    return findings, parse_errors


def write_markdown(path: Path, root: Path, findings: list[Finding], parse_errors: list[dict[str, str]]) -> None:
    sev = Counter(f.severity for f in findings)
    rules = Counter(f.rule_id for f in findings)
    lines = [
        "# Static PyTorch gradient-risk scan",
        "",
        f"Root: `{root}`",
        "",
        "> Static findings are review candidates, not proof of a differentiability failure.",
        "",
        "## Summary",
        "",
        f"- Total findings: {len(findings)}",
        f"- High: {sev.get('high', 0)}; medium: {sev.get('medium', 0)}; low: {sev.get('low', 0)}; info: {sev.get('info', 0)}",
        f"- Parse/read errors: {len(parse_errors)}",
        "",
        "## Findings by rule",
        "",
    ]
    for rule, count in sorted(rules.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{rule}`: {count}")
    lines.extend([
        "",
        "## Detailed findings",
        "",
        "| Severity | Rule | Location | Function | Evidence |",
        "|---|---|---|---|---|",
    ])
    for f in findings:
        snippet = f.snippet.replace("|", "\\|").replace("`", "'")
        location = f"`{f.path}:{f.line}`"
        evidence = f"{f.message} `{snippet}`" if snippet else f.message
        lines.append(f"| {f.severity} | `{f.rule_id}` | {location} | `{f.function}` | {evidence} |")
    if parse_errors:
        lines.extend(["", "## Parse/read errors", ""])
        for err in parse_errors:
            lines.append(f"- `{err['path']}`: {err['error']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[], help="Additional directory name to exclude")
    parser.add_argument("--max-file-mb", type=float, default=5.0)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"root is not a directory: {root}")
    excludes = set(DEFAULT_EXCLUDES) | set(args.exclude)
    findings, parse_errors = scan(root, excludes, int(args.max_file_mb * 1024 * 1024))

    payload = {
        "root": str(root),
        "disclaimer": "Static findings are review candidates, not proof of a differentiability failure.",
        "summary": {
            "total": len(findings),
            "by_severity": dict(Counter(f.severity for f in findings)),
            "by_rule": dict(Counter(f.rule_id for f in findings)),
            "parse_errors": len(parse_errors),
        },
        "findings": [asdict(f) for f in findings],
        "errors": parse_errors,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.md_out, root, findings, parse_errors)

    print(f"Scanned {root}")
    print(f"Findings: {len(findings)}; parse/read errors: {len(parse_errors)}")
    print(f"JSON: {args.json_out}")
    print(f"Markdown: {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
