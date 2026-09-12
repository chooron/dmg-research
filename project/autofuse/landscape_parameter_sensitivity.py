"""Bounded finite-difference direction check after a topology/ranking bias.

This is a conditional diagnostic only.  It does not alter the frozen S3
solver, calibration vectors, or the formal 12x78 landscape result.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfuse.spec import PARAMETERS, get_structure
from project.autofuse.landscape_gate import _cal_slice, _case_inputs, _eval_slice, _metrics, _parameter_values, sha256_file
from project.autofuse.reference_oracle import run_reference
from dfuse import simulate_sequential

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
MODELS = (164, 166, 188, 190, 212, 214)
CATCHMENTS = ("USA_09447800", "USA_14138900")
FORCING_START = date(1987, 1, 1)
SIMULATION_END = date(2009, 12, 31)
EPSILON_FRACTION = 0.01
SENSITIVITY_PARAMETERS = ("MAXWATR_1", "MAXWATR_2", "BASERTE", "SACPMLT", "SACPEXP", "FRCHZNE", "FRACLOWZ", "LOGLAMB", "TISHAPE")


def _dates() -> list[date]:
    return [FORCING_START + timedelta(days=i) for i in range((SIMULATION_END - FORCING_START).days + 1)]


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _delta(executable: str, model_id: int, values: dict[str, np.ndarray], params: dict[str, float], dates: list[date]) -> dict[str, float]:
    forcing = {name: values[name] for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}
    reference = run_reference(executable, model_id, forcing, params=params, initial_fraction=0.25, dates=dates, dt_days=1.0, timeout_seconds=1800.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1), dtype=torch.float64, device=device)
    result = simulate_sequential(model_id, tensor_forcing, params, initial_fraction=0.25, dates=dates, dt_days=1.0, n_substeps=1, order="S3", compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    obs = values["q_obs"]
    epsilon = float(np.mean(obs[_cal_slice()]) / 100.0)
    ref_metrics = _metrics(np.asarray(reference.q_routed, dtype=np.float64)[_eval_slice()], obs[_eval_slice()], epsilon)
    s3_metrics = _metrics(result.q.detach().cpu().numpy()[_eval_slice()], obs[_eval_slice()], epsilon)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    del result, tensor_forcing, reference
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return {"kgecomp": s3_metrics["kgecomp"] - ref_metrics["kgecomp"], "kge_q": s3_metrics["kge_q"] - ref_metrics["kge_q"], "kge_inv_q": s3_metrics["kge_inv_q"] - ref_metrics["kge_inv_q"]}


def _stats(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if not finite.size: return {"n": 0}
    return {"n": int(finite.size), "median": float(np.median(finite)), "q1": float(np.quantile(finite, .25)), "q3": float(np.quantile(finite, .75)), "positive_fraction": float(np.mean(finite > 0)), "negative_fraction": float(np.mean(finite < 0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DOCS / "landscape_12catchment_manifest.json"))
    parser.add_argument("--input-root", default=str(DOCS / "landscape_inputs"))
    parser.add_argument("--calibration", default=str(DOCS / "reference_calibration_12x78.json"))
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", default=str(DOCS / "landscape_parameter_sensitivity.json"))
    parser.add_argument("--partial", default=str(DOCS / "landscape_parameter_sensitivity.partial.json"))
    parser.add_argument("--perturb-fraction", type=float, default=EPSILON_FRACTION)
    parser.add_argument("--parameters", default=",".join(SENSITIVITY_PARAMETERS), help="comma-separated bounded subset; empty selects all active parameters")
    args = parser.parse_args()
    parameter_filter = {value.strip() for value in args.parameters.split(",") if value.strip()} if args.parameters else None
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    calibration = json.loads(Path(args.calibration).read_text())
    if calibration.get("status") != "complete": raise RuntimeError("calibration archive must be complete")
    manifest = json.loads(Path(args.manifest).read_text())
    input_index = json.loads((Path(args.input_root) / "index.json").read_text())
    input_by_id = {int(row["hru_id"]): row["path"] for row in input_index["rows"]}
    selected = {row["basin_id"]: row for row in manifest["catchments"] if row["basin_id"] in CATCHMENTS}
    cal_rows = {(row["basin_id"], int(row["model_id"])): row for row in calibration["case_results"]}
    dates = _dates()
    prior_partial = json.loads(Path(args.partial).read_text()) if Path(args.partial).is_file() else {}
    rows: list[dict[str, Any]] = list(prior_partial.get("rows", []))
    completed_keys = {(row["basin_id"], int(row["model_id"]), row["parameter"]) for row in rows}
    failure = None
    for basin_id in CATCHMENTS:
        basin = selected[basin_id]
        with np.load(input_by_id[int(basin["hru_id"])], allow_pickle=False) as z:
            values = {name: np.asarray(z[name], dtype=np.float64) for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}
        for model_id in MODELS:
            base_row = cal_rows[(basin_id, model_id)]
            base_params = {name: float(value) for name, value in base_row["parameter_vector"].items()}
            spec = get_structure(model_id)
            for name in spec.parameter_names:
                if parameter_filter is not None and name not in parameter_filter:
                    continue
                if (basin_id, model_id, name) in completed_keys:
                    continue
                bound = PARAMETERS[name]
                center = base_params[name]
                delta = float(args.perturb_fraction * (bound["upper"] - bound["lower"]))
                lower = max(float(bound["lower"]), center - delta)
                upper = min(float(bound["upper"]), center + delta)
                if not (upper > lower):
                    continue
                plus = dict(base_params); minus = dict(base_params); plus[name] = upper; minus[name] = lower
                print(f"{basin_id} model={model_id} parameter={name}", file=sys.stderr, flush=True)
                try:
                    dplus = _delta(args.executable, model_id, values, plus, dates)
                    dminus = _delta(args.executable, model_id, values, minus, dates)
                except Exception as exc:
                    failure = {"basin_id": basin_id, "model_id": model_id, "parameter": name, "error_type": type(exc).__name__, "error": str(exc)[:2000]}
                    break
                rows.append({"basin_id": basin_id, "hru_id": int(basin["hru_id"]), "model_id": model_id, "topology": spec.decisions, "parameter": name, "base_value": center, "minus_value": lower, "plus_value": upper, "step": upper - lower, "delta_minus": dminus, "delta_plus": dplus, "derivative_kgecomp": (dplus["kgecomp"] - dminus["kgecomp"]) / (upper - lower), "derivative_kge_q": (dplus["kge_q"] - dminus["kge_q"]) / (upper - lower), "derivative_kge_inv_q": (dplus["kge_inv_q"] - dminus["kge_inv_q"]) / (upper - lower)})
                completed_keys.add((basin_id, model_id, name))
                Path(args.partial).write_text(json.dumps({"schema_version": "landscape-parameter-sensitivity-v1", "status": "partial", "configuration": {"perturb_fraction_of_bound_width": args.perturb_fraction, "threads": 1}, "rows": rows, "failure": failure}, indent=2, sort_keys=True) + "\n")
            if failure: break
        if failure: break
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows: by_group.setdefault(f"{row['model_id']}:{row['parameter']}", []).append(row)
    summary = {key: {"n": len(vals), "derivative_kgecomp": _stats([v["derivative_kgecomp"] for v in vals]), "derivative_kge_q": _stats([v["derivative_kge_q"] for v in vals]), "derivative_kge_inv_q": _stats([v["derivative_kge_inv_q"] for v in vals])} for key, vals in sorted(by_group.items())}
    result = {"schema_version": "landscape-parameter-sensitivity-v1", "status": "complete" if failure is None and len(rows) > 0 else "blocked", "purpose": "conditional finite-difference direction check after clear topology/ranking bias; diagnostic only, no formal theta replacement", "configuration": {"catchments": list(CATCHMENTS), "models": list(MODELS), "requested_parameters": sorted(parameter_filter) if parameter_filter is not None else "all_active", "parameters_present_in_rows": sorted({row["parameter"] for row in rows}), "perturb_fraction_of_bound_width": args.perturb_fraction, "threads": 1, "same_forcing_initialization_evaluation_and_fixed_reference_theta": True}, "calibration_sha256": sha256_file(Path(args.calibration)), "executable": str(Path(args.executable).resolve()), "executable_sha256": sha256_file(Path(args.executable).resolve()), "rows": rows, "summary_by_model_parameter": summary, "failure": failure, "parameter_direction_summary": {"kgecomp_positive_fraction": float(np.mean([v["derivative_kgecomp"] > 0 for v in rows])) if rows else None, "kgecomp_negative_fraction": float(np.mean([v["derivative_kgecomp"] < 0 for v in rows])) if rows else None, "kgecomp_zero_fraction": float(np.mean([v["derivative_kgecomp"] == 0 for v in rows])) if rows else None}, "scope": {"formal_landscape_unchanged": True, "scientific_equations_modified": False, "conditional_only": True}}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "rows": len(rows), "failure": failure}, sort_keys=True))


if __name__ == "__main__": main()
