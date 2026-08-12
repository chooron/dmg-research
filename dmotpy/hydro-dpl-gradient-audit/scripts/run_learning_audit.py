#!/usr/bin/env python3
"""Audit short-run parameter movement through a repository-specific adapter."""

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
    spec = importlib.util.spec_from_file_location("hydro_learning_adapter", path)
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


def as_flat_double(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)


def run_one(module: ModuleType, run: Mapping[str, Any], boundary_fraction: float,
            min_norm_disp: float) -> dict[str, Any]:
    name = str(run.get("name", "unnamed"))
    device = str(run.get("device", "cpu"))
    dtype_name = str(run.get("dtype", "float32"))
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    dtype = DTYPES[dtype_name]
    seed = int(run.get("seed", 0))
    steps = int(run.get("steps", 20))
    config = run.get("config", {})
    if not isinstance(config, Mapping):
        raise TypeError("run config must be a mapping")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    outcome = module.run_short_training(
        device=device, dtype=dtype, seed=seed, steps=steps, config=config
    )
    if not isinstance(outcome, Mapping):
        raise TypeError("run_short_training must return a mapping")
    initial = outcome.get("initial_parameters")
    final = outcome.get("final_parameters")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        raise TypeError("outcome must contain initial_parameters and final_parameters mappings")

    ranges = dict(outcome.get("parameter_ranges", {}))
    applicability = dict(outcome.get("applicability", {}))
    required_to_move = dict(outcome.get("required_to_move", {}))
    records: list[dict[str, Any]] = []

    all_names = sorted(set(initial) | set(final))
    for param_name in all_names:
        applicable = bool(applicability.get(param_name, True))
        required = bool(required_to_move.get(param_name, False))
        record: dict[str, Any] = {
            "name": param_name,
            "applicable": applicable,
            "required_to_move": required,
        }
        if not applicable:
            record["status"] = "NOT_APPLICABLE"
            records.append(record)
            continue
        if param_name not in initial or param_name not in final:
            record["status"] = "MISSING_SNAPSHOT"
            records.append(record)
            continue
        init = as_flat_double(initial[param_name])
        fin = as_flat_double(final[param_name])
        if init.shape != fin.shape:
            record["status"] = "SHAPE_MISMATCH"
            record["initial_shape"] = list(initial[param_name].shape)
            record["final_shape"] = list(final[param_name].shape)
            records.append(record)
            continue
        if not torch.isfinite(init).all().item() or not torch.isfinite(fin).all().item():
            record["status"] = "NONFINITE_PARAMETER"
            records.append(record)
            continue

        diff = fin - init
        abs_diff = diff.abs()
        record.update(
            {
                "elements": int(diff.numel()),
                "unchanged_fraction_exact": float((diff == 0).to(torch.float64).mean().item()) if diff.numel() else math.nan,
                "abs_mean_displacement": float(abs_diff.mean().item()) if diff.numel() else math.nan,
                "abs_max_displacement": float(abs_diff.max().item()) if diff.numel() else math.nan,
                "l2_displacement": float(torch.linalg.vector_norm(diff).item()),
                "final_std": float(fin.std(unbiased=False).item()) if fin.numel() else math.nan,
            }
        )

        range_value = ranges.get(param_name)
        normalized = None
        boundary_occupancy = None
        if isinstance(range_value, (list, tuple)) and len(range_value) == 2:
            lo, hi = float(range_value[0]), float(range_value[1])
            width = hi - lo
            record["parameter_range"] = [lo, hi]
            if width > 0:
                normalized = float(abs_diff.mean().item() / width)
                tol = boundary_fraction * width
                boundary = (fin <= lo + tol) | (fin >= hi - tol)
                boundary_occupancy = float(boundary.to(torch.float64).mean().item())
                record["normalized_mean_displacement"] = normalized
                record["boundary_occupancy"] = boundary_occupancy

        if diff.numel() and torch.all(diff == 0).item():
            record["status"] = "UNCHANGED_REQUIRED" if required else "UNCHANGED"
        elif normalized is not None and normalized < min_norm_disp:
            record["status"] = "WEAK_MOVEMENT_REQUIRED" if required else "WEAK_MOVEMENT"
        elif boundary_occupancy is not None and boundary_occupancy >= 0.95:
            record["status"] = "BOUNDARY_SATURATED"
        else:
            record["status"] = "MOVED"
        records.append(record)

    loss_history = outcome.get("loss_history")
    loss_summary: dict[str, Any] = {}
    if isinstance(loss_history, (list, tuple)) and loss_history:
        values = [float(x) for x in loss_history]
        finite = all(math.isfinite(x) for x in values)
        loss_summary = {
            "count": len(values),
            "finite": finite,
            "start": values[0],
            "end": values[-1],
            "absolute_change": values[-1] - values[0],
            "relative_reduction": (values[0] - values[-1]) / max(abs(values[0]), 1e-30),
        }

    statuses = [r["status"] for r in records if r.get("applicable")]
    if any(s in {"NONFINITE_PARAMETER"} for s in statuses) or not bool(outcome.get("optimizer_state_finite", True)):
        run_status = "FAIL_NUMERICAL"
    elif any(s in {"MISSING_SNAPSHOT", "SHAPE_MISMATCH"} for s in statuses):
        run_status = "NOT_EVALUATED"
    elif any(s in {"UNCHANGED_REQUIRED", "WEAK_MOVEMENT_REQUIRED"} for s in statuses):
        run_status = "FAIL_TRAINABILITY"
    elif any(s in {"UNCHANGED", "WEAK_MOVEMENT", "BOUNDARY_SATURATED"} for s in statuses):
        run_status = "PASS_WITH_CAVEAT"
    else:
        run_status = "PASS"

    return {
        "name": name,
        "device": device,
        "dtype": dtype_name,
        "seed": seed,
        "steps": steps,
        "config": jsonable(config),
        "metadata": jsonable(outcome.get("metadata", {})),
        "optimizer_state_finite": bool(outcome.get("optimizer_state_finite", True)),
        "loss": loss_summary,
        "parameters": records,
        "run_status": run_status,
    }


def aggregate(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    by_param: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for rec in run.get("parameters", []):
            if rec.get("applicable"):
                by_param[str(rec["name"])].append(
                    {"run": run.get("name"), "status": rec.get("status"), "required_to_move": rec.get("required_to_move", False)}
                )
    aggregate_rows = []
    for name in sorted(by_param):
        evidence = by_param[name]
        statuses = [e["status"] for e in evidence]
        if any(s == "NONFINITE_PARAMETER" for s in statuses):
            verdict = "FAIL_NUMERICAL"
        elif any(s in {"UNCHANGED_REQUIRED", "WEAK_MOVEMENT_REQUIRED"} for s in statuses):
            verdict = "FAIL_TRAINABILITY"
        elif all(s in {"UNCHANGED", "UNCHANGED_REQUIRED"} for s in statuses):
            verdict = "PASS_WITH_CAVEAT"
        elif any(s in {"UNCHANGED", "WEAK_MOVEMENT", "BOUNDARY_SATURATED"} for s in statuses):
            verdict = "PASS_WITH_CAVEAT"
        else:
            verdict = "PASS"
        aggregate_rows.append({"name": name, "verdict": verdict, "evidence": evidence})

    run_statuses = [r.get("run_status") for r in runs]
    verdicts = [r["verdict"] for r in aggregate_rows]
    if "FAIL_NUMERICAL" in run_statuses or "FAIL_NUMERICAL" in verdicts:
        overall = "FAIL_NUMERICAL"
    elif "FAIL_TRAINABILITY" in run_statuses or "FAIL_TRAINABILITY" in verdicts:
        overall = "FAIL_TRAINABILITY"
    elif "NOT_EVALUATED" in run_statuses or not aggregate_rows:
        overall = "NOT_EVALUATED"
    elif "PASS_WITH_CAVEAT" in run_statuses or "PASS_WITH_CAVEAT" in verdicts:
        overall = "PASS_WITH_CAVEAT"
    else:
        overall = "PASS"
    return aggregate_rows, overall


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Short-learning audit",
        "",
        f"**Overall verdict: {payload['overall_verdict']}**",
        "",
        "## Run summary",
        "",
        "| Run | Dtype | Device | Steps | Loss start | Loss end | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for run in payload["runs"]:
        loss = run.get("loss", {})
        lines.append(
            f"| `{run.get('name')}` | `{run.get('dtype')}` | `{run.get('device')}` | {run.get('steps')} | "
            f"{loss.get('start', '—')} | {loss.get('end', '—')} | **{run.get('run_status')}** |"
        )
    lines.extend([
        "",
        "## Per-parameter aggregate",
        "",
        "| Parameter | Verdict | Evidence |",
        "|---|---|---|",
    ])
    for row in payload["parameter_aggregate"]:
        evidence = "; ".join(f"{e['run']}={e['status']}" for e in row["evidence"])
        lines.append(f"| `{row['name']}` | **{row['verdict']}** | {evidence} |")

    for run in payload["runs"]:
        lines.extend([
            "",
            f"## Run: {run.get('name')}",
            "",
            "| Parameter | Required to move | Status | Mean displacement | Normalized displacement | Boundary occupancy | Final std |",
            "|---|---:|---|---:|---:|---:|---:|",
        ])
        for rec in run.get("parameters", []):
            def fmt(key: str) -> str:
                value = rec.get(key)
                if value is None:
                    return "—"
                if isinstance(value, float):
                    return f"{value:.3e}"
                return str(value)
            lines.append(
                f"| `{rec['name']}` | {rec.get('required_to_move')} | **{rec.get('status')}** | "
                f"{fmt('abs_mean_displacement')} | {fmt('normalized_mean_displacement')} | "
                f"{fmt('boundary_occupancy')} | {fmt('final_std')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-fraction", type=float, default=0.02)
    parser.add_argument("--min-normalized-displacement", type=float, default=1e-7)
    args = parser.parse_args()

    module = load_module(args.adapter)
    if not hasattr(module, "run_short_training"):
        parser.error("adapter must define run_short_training")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    runs_spec = matrix.get("runs") if isinstance(matrix, Mapping) else None
    if not isinstance(runs_spec, list) or not runs_spec:
        parser.error("matrix must contain a non-empty 'runs' list")

    runs = []
    for run_spec in runs_spec:
        try:
            runs.append(run_one(module, run_spec, args.boundary_fraction, args.min_normalized_displacement))
        except Exception as exc:
            runs.append(
                {
                    "name": str(run_spec.get("name", "unnamed")) if isinstance(run_spec, Mapping) else "unnamed",
                    "run_status": "ADAPTER_EXCEPTION",
                    "exception": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "parameters": [],
                }
            )

    parameter_aggregate, overall = aggregate(runs)
    payload = {
        "adapter": str(args.adapter.resolve()),
        "matrix": str(args.matrix.resolve()),
        "overall_verdict": overall,
        "parameter_aggregate": parameter_aggregate,
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "learning_audit.json"
    md_path = args.output_dir / "learning_audit.md"
    json_path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)

    print(f"Overall verdict: {overall}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 1 if overall.startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
