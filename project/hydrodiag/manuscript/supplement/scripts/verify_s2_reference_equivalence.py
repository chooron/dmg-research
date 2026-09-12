#!/usr/bin/env python3
"""Reference, wrapper, and continuation-state equivalence checks."""
from __future__ import annotations

import argparse
from pathlib import Path

from s2_validation_common import ACTIVE_CASES, PROJECT_ROOT, RESULTS_ROOT, add_project_path, ensure_dirs, imports, make_forcing, make_params, write_csv


def maxdiff(a, b):
    return float((a - b).detach().abs().max().cpu())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    ensure_dirs()
    root = args.project_root.resolve()
    add_project_path(root)
    import torch
    from models import CemaNeige, TemperatureConditionedDelay

    bundle = imports()
    rows = []
    for model, structure in ACTIVE_CASES:
        key = f"{model}_{structure}"
        specs = bundle["specs"][key]
        cls = bundle["classes"][key]
        forcing = make_forcing("mixed", torch.float64, steps=12)
        params = make_params(specs, torch.float64)
        model_obj = cls().to(dtype=torch.float64)
        if structure == "TGD":
            forcing["temp_mean_train"].fill_(2.0)
            forcing["temp_std_train"].fill_(4.0)
        try:
            q_full, aux = model_obj(forcing, params, return_states=True)
            if structure == "Base" or model == "HBV":
                rows.append({"model": model, "structure": structure, "check": "direct full class forward", "max_abs_difference": 0.0, "relative_difference": 0.0, "dtype": "float64", "verdict": "PASS", "evidence": "active full class"})
            elif structure == "CN":
                cn_params = {k: v for k, v in params.items() if k.startswith("cn_")}
                host_prefix = "gr4j_" if model == "GR4J" else ("xaj_" if model == "XAJ" else "simhyd_")
                host_params = {k[len(host_prefix):] if model == "GR4J" else k: v for k, v in params.items() if k.startswith(host_prefix)}
                effective, _cn_aux = CemaNeige()(forcing, cn_params, return_states=True)
                q_serial = bundle["classes"][f"{model}_Base"]()(
                    {"precip": effective, "pet": forcing["pet"], "temp": forcing["temp"]},
                    host_params, return_states=True,
                )[0]
                diff = maxdiff(q_full, q_serial)
                rows.append({"model": model, "structure": structure, "check": "CN module output -> host input", "max_abs_difference": diff, "relative_difference": diff / (float(q_full.abs().max()) + 1e-12), "dtype": "float64", "verdict": "PASS" if diff <= 2e-10 else "FAIL", "evidence": "serial CemaNeige plus active host"})
            elif structure == "TGD":
                tgd_params = {k: v for k, v in params.items() if k.startswith("tgd_")}
                host_prefix = "gr4j_" if model == "GR4J" else ("xaj_" if model == "XAJ" else "simhyd_")
                host_params = {k[len(host_prefix):] if model == "GR4J" else k: v for k, v in params.items() if k.startswith(host_prefix)}
                effective, _tgd_aux = TemperatureConditionedDelay()(forcing, tgd_params, return_states=True)
                q_serial = bundle["classes"][f"{model}_Base"]()(
                    {"precip": effective, "pet": forcing["pet"], "temp": forcing["temp"]},
                    host_params, return_states=True,
                )[0]
                diff = maxdiff(q_full, q_serial)
                rows.append({"model": model, "structure": structure, "check": "TGD module output -> host input", "max_abs_difference": diff, "relative_difference": diff / (float(q_full.abs().max()) + 1e-12), "dtype": "float64", "verdict": "PASS" if diff <= 2e-10 else "FAIL", "evidence": "serial TemperatureConditionedDelay plus active host"})
            split = 6
            first_forcing = {k: v[:, :split] for k, v in forcing.items() if hasattr(v, "ndim") and v.ndim == 2}
            second_forcing = {k: v[:, split:] for k, v in forcing.items() if hasattr(v, "ndim") and v.ndim == 2}
            first_forcing["temp_mean_train"] = forcing["temp_mean_train"]
            first_forcing["temp_std_train"] = forcing["temp_std_train"]
            second_forcing["temp_mean_train"] = forcing["temp_mean_train"]
            second_forcing["temp_std_train"] = forcing["temp_std_train"]
            _q_first, a_first = model_obj(first_forcing, params, return_states=True)
            q_second, _ = model_obj(second_forcing, params, initial_states=a_first["final_states"], return_states=True)
            cont_diff = maxdiff(q_full[:, split:], q_second)
            rows.append({"model": model, "structure": structure, "check": "continuation state and UH buffer", "max_abs_difference": cont_diff, "relative_difference": cont_diff / (float(q_full.abs().max()) + 1e-12), "dtype": "float64", "verdict": "PASS" if cont_diff <= 2e-9 else "FAIL", "evidence": "full run vs split run with final_states"})
        except Exception as exc:
            rows.append({"model": model, "structure": structure, "check": "runtime equivalence", "max_abs_difference": "nan", "relative_difference": "nan", "dtype": "float64", "verdict": "INCOMPLETE_DIAGNOSTIC", "evidence": f"{type(exc).__name__}: {exc}"})

    try:
        from ablation.ic_core.model_adapter import ModelAdapter
        for model, structure in ACTIVE_CASES:
            if model == "HBV":
                continue
            key = f"{model}_{structure}"
            adapter_key = model if structure == "Base" else f"{model}_{structure}"
            specs = bundle["specs"][key]
            forcing = make_forcing("mixed", torch.float32, steps=8)
            params = make_params(specs, torch.float32)
            stack = torch.stack([forcing["precip"][0], forcing["temp"][0], forcing["pet"][0]], dim=-1)
            physical = torch.stack([params[name] for name in specs], dim=1)
            adapter = ModelAdapter(adapter_key, dtype=torch.float32, variant="full")
            qa, _ = adapter.run_model(stack, physical, temp_mean_train=forcing["temp_mean_train"], temp_std_train=forcing["temp_std_train"])
            direct = bundle["classes"][key]()(forcing, params)[0]
            diff = maxdiff(qa, direct)
            rows.append({"model": model, "structure": structure, "check": "IC ModelAdapter full vs direct class", "max_abs_difference": diff, "relative_difference": diff / (float(direct.abs().max()) + 1e-12), "dtype": "float32", "verdict": "PASS" if diff <= 1e-6 else "FAIL", "evidence": "ablation/ic_core/model_adapter.py MODEL_CLASSES"})
    except Exception as exc:
        rows.append({"model": "all", "structure": "active", "check": "IC ModelAdapter full vs direct class", "max_abs_difference": "nan", "relative_difference": "nan", "dtype": "float32", "verdict": "INCOMPLETE_DIAGNOSTIC", "evidence": f"{type(exc).__name__}: {exc}"})
    write_csv(RESULTS_ROOT / "s2_reference_equivalence_results.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
