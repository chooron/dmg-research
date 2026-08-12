#!/usr/bin/env python3
"""Run per-target gradient audits through a repository-specific adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import torch


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def load_module(path: Path) -> ModuleType:
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(Path.cwd()))
    spec = importlib.util.spec_from_file_location("hydro_gradient_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def iter_tensors(value: Any, prefix: str = "prediction"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            yield from iter_tensors(child, f"{prefix}[{idx}]")


def tensor_values(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_sparse:
        return tensor.coalesce().values()
    return tensor


def validate_case(case: Mapping[str, Any]) -> None:
    if "loss" not in case or "targets" not in case:
        raise KeyError("build_case must return 'loss' and 'targets'")
    loss = case["loss"]
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise TypeError("case['loss'] must be a scalar torch.Tensor")
    if not isinstance(case["targets"], Mapping) or not case["targets"]:
        raise TypeError("case['targets'] must be a non-empty mapping")
    for name, target in case["targets"].items():
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"target {name!r} is not a torch.Tensor")


def run_one(module: ModuleType, run: Mapping[str, Any], anomaly: bool) -> dict[str, Any]:
    name = str(run.get("name", "unnamed"))
    device = str(run.get("device", "cpu"))
    dtype_name = str(run.get("dtype", "float32"))
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    dtype = DTYPES[dtype_name]
    seed = int(run.get("seed", 0))
    config = run.get("config", {})
    if not isinstance(config, Mapping):
        raise TypeError("run config must be a mapping")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    case = module.build_case(device=device, dtype=dtype, seed=seed, config=config)
    if not isinstance(case, Mapping):
        raise TypeError("build_case must return a mapping")
    validate_case(case)

    loss: torch.Tensor = case["loss"]
    targets: Mapping[str, torch.Tensor] = case["targets"]
    applicability = dict(case.get("applicability", {}))
    parameter_ranges = dict(case.get("parameter_ranges", {}))
    target_groups = dict(case.get("target_groups", {}))

    for target in targets.values():
        if target.requires_grad and not target.is_leaf:
            target.retain_grad()
        target.grad = None

    forward_issues: list[str] = []
    if not torch.isfinite(loss.detach()).all().item():
        forward_issues.append("loss is non-finite before backward")
    for pred_name, pred in iter_tensors(case.get("predictions")):
        if pred.numel() and not torch.isfinite(pred.detach()).all().item():
            forward_issues.append(f"{pred_name} contains non-finite values")

    result: dict[str, Any] = {
        "name": name,
        "device": device,
        "dtype": dtype_name,
        "seed": seed,
        "config": jsonable(config),
        "metadata": jsonable(case.get("metadata", {})),
        "forward_checks": jsonable(case.get("forward_checks", {})),
        "loss": jsonable(loss.detach()),
        "forward_issues": forward_issues,
        "targets": [],
    }

    if forward_issues:
        result["run_status"] = "FORWARD_NONFINITE"
        return result

    try:
        if anomaly:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()
    except Exception as exc:  # report the exact adapter/model failure
        result["run_status"] = "BACKWARD_EXCEPTION"
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return result

    loss_scale = max(abs(float(loss.detach().cpu().item())), 1e-30)
    statuses: list[str] = []
    for target_name, target in targets.items():
        applicable = bool(applicability.get(target_name, True))
        group = str(target_groups.get(target_name, "unspecified"))
        record: dict[str, Any] = {
            "name": target_name,
            "group": group,
            "applicable": applicable,
            "shape": list(target.shape),
            "requires_grad": bool(target.requires_grad),
        }
        if not applicable:
            record["status"] = "NOT_APPLICABLE"
            result["targets"].append(record)
            continue
        grad = target.grad
        if grad is None:
            record["status"] = "NO_GRAD"
            statuses.append(record["status"])
            result["targets"].append(record)
            continue

        values = tensor_values(grad.detach())
        total = int(values.numel())
        nan_count = int(torch.isnan(values).sum().item())
        posinf_count = int(torch.isposinf(values).sum().item())
        neginf_count = int(torch.isneginf(values).sum().item())
        inf_count = posinf_count + neginf_count
        finite_mask = torch.isfinite(values)
        finite_values = values[finite_mask]
        finite_count = int(finite_values.numel())
        zero_count = int((finite_values == 0).sum().item()) if finite_count else 0
        nonzero_count = finite_count - zero_count

        record.update(
            {
                "elements": total,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "finite_fraction": finite_count / total if total else math.nan,
                "zero_fraction_of_finite": zero_count / finite_count if finite_count else math.nan,
                "nonzero_fraction_of_finite": nonzero_count / finite_count if finite_count else math.nan,
            }
        )

        if finite_count:
            abs_values = finite_values.abs().to(torch.float64)
            abs_mean = float(abs_values.mean().item())
            abs_max = float(abs_values.max().item())
            l2 = float(torch.linalg.vector_norm(finite_values.to(torch.float64)).item())
            record.update({"abs_mean": abs_mean, "abs_max": abs_max, "l2": l2})
            range_value = parameter_ranges.get(target_name)
            if isinstance(range_value, (list, tuple)) and len(range_value) == 2:
                width = float(range_value[1]) - float(range_value[0])
                record["parameter_range"] = [float(range_value[0]), float(range_value[1])]
                record["normalized_sensitivity"] = abs_mean * abs(width) / loss_scale

        if nan_count or inf_count:
            record["status"] = "NONFINITE"
        elif nonzero_count == 0:
            record["status"] = "ZERO"
        else:
            record["status"] = "FINITE_NONZERO"
        statuses.append(record["status"])
        result["targets"].append(record)

    if "NONFINITE" in statuses:
        result["run_status"] = "FAIL_NUMERICAL"
    elif "NO_GRAD" in statuses:
        result["run_status"] = "FAIL_AUTOGRAD"
    elif statuses and all(s == "ZERO" for s in statuses):
        result["run_status"] = "ALL_APPLICABLE_ZERO"
    elif "ZERO" in statuses:
        result["run_status"] = "PASS_WITH_ZERO_TARGETS"
    else:
        result["run_status"] = "PASS"
    return result


def aggregate(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_failures = [r for r in runs if r.get("run_status") in {"FORWARD_NONFINITE", "BACKWARD_EXCEPTION"}]
    for run in runs:
        for target in run.get("targets", []):
            if target.get("applicable"):
                by_target[str(target["name"])].append(
                    {
                        "run": run.get("name"),
                        "status": target.get("status"),
                        "dtype": run.get("dtype"),
                        "device": run.get("device"),
                    }
                )

    aggregates: list[dict[str, Any]] = []
    for name in sorted(by_target):
        evidence = by_target[name]
        statuses = [str(x.get("status")) for x in evidence]
        if "NONFINITE" in statuses:
            verdict = "FAIL_NUMERICAL"
        elif "NO_GRAD" in statuses:
            verdict = "FAIL_AUTOGRAD"
        elif statuses and all(status == "ZERO" for status in statuses):
            verdict = "FAIL_TRAINABILITY"
        elif "ZERO" in statuses:
            verdict = "PASS_WITH_CAVEAT"
        elif statuses:
            verdict = "PASS"
        else:
            verdict = "NOT_EVALUATED"
        aggregates.append({"name": name, "verdict": verdict, "evidence": evidence})

    verdicts = [item["verdict"] for item in aggregates]
    if run_failures or "FAIL_NUMERICAL" in verdicts:
        overall = "FAIL_NUMERICAL"
    elif "FAIL_AUTOGRAD" in verdicts:
        overall = "FAIL_AUTOGRAD"
    elif "FAIL_TRAINABILITY" in verdicts:
        overall = "FAIL_TRAINABILITY"
    elif not aggregates:
        overall = "NOT_EVALUATED"
    elif "PASS_WITH_CAVEAT" in verdicts:
        overall = "PASS_WITH_CAVEAT"
    else:
        overall = "PASS"
    return aggregates, overall


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Gradient audit",
        "",
        f"**Overall verdict: {payload['overall_verdict']}**",
        "",
        "## Run summary",
        "",
        "| Run | Device | Dtype | Seed | Loss | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for run in payload["runs"]:
        lines.append(
            f"| `{run.get('name')}` | `{run.get('device')}` | `{run.get('dtype')}` | "
            f"{run.get('seed')} | {run.get('loss', '—')} | **{run.get('run_status')}** |"
        )
    lines.extend([
        "",
        "## Per-target aggregate",
        "",
        "| Target | Verdict | Applicable run evidence |",
        "|---|---|---|",
    ])
    for item in payload["target_aggregate"]:
        evidence = "; ".join(f"{e['run']}={e['status']}" for e in item["evidence"])
        lines.append(f"| `{item['name']}` | **{item['verdict']}** | {evidence} |")

    for run in payload["runs"]:
        lines.extend([
            "",
            f"## Run: {run.get('name')}",
            "",
        ])
        if run.get("forward_issues"):
            for issue in run["forward_issues"]:
                lines.append(f"- Forward issue: {issue}")
        if run.get("exception"):
            lines.append(f"- Backward exception: `{run['exception']}`")
        if run.get("targets"):
            lines.extend([
                "",
                "| Target | Group | Applicable | Status | NaN | Inf | Zero fraction | Mean | Max | Normalized sensitivity |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ])
            for target in run["targets"]:
                def fmt(key: str) -> str:
                    value = target.get(key)
                    if value is None:
                        return "—"
                    if isinstance(value, float):
                        return f"{value:.3e}"
                    return str(value)

                lines.append(
                    f"| `{target['name']}` | `{target.get('group')}` | {target.get('applicable')} | "
                    f"**{target.get('status')}** | {fmt('nan_count')} | {fmt('inf_count')} | "
                    f"{fmt('zero_fraction_of_finite')} | {fmt('abs_mean')} | {fmt('abs_max')} | "
                    f"{fmt('normalized_sensitivity')} |"
                )
        metadata = run.get("metadata")
        if metadata:
            lines.extend(["", "Metadata:", "", "```json", json.dumps(metadata, indent=2, ensure_ascii=False), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anomaly", action="store_true", help="Enable anomaly detection; use only on small failing cases")
    args = parser.parse_args()

    module = load_module(args.adapter)
    if not hasattr(module, "build_case"):
        parser.error("adapter must define build_case")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    runs_spec = matrix.get("runs") if isinstance(matrix, Mapping) else None
    if not isinstance(runs_spec, list) or not runs_spec:
        parser.error("matrix must contain a non-empty 'runs' list")

    runs: list[dict[str, Any]] = []
    for run_spec in runs_spec:
        try:
            runs.append(run_one(module, run_spec, args.anomaly))
        except Exception as exc:
            runs.append(
                {
                    "name": str(run_spec.get("name", "unnamed")) if isinstance(run_spec, Mapping) else "unnamed",
                    "run_status": "ADAPTER_EXCEPTION",
                    "exception": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "targets": [],
                }
            )

    target_aggregate, overall = aggregate(runs)
    payload = {
        "adapter": str(args.adapter.resolve()),
        "matrix": str(args.matrix.resolve()),
        "overall_verdict": overall,
        "target_aggregate": target_aggregate,
        "runs": runs,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "gradient_audit.json"
    md_path = args.output_dir / "gradient_audit.md"
    json_path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)

    print(f"Overall verdict: {overall}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 1 if overall.startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
