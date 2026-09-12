#!/usr/bin/env python3
"""2x2 Warmup Decomposition Evaluator for Phase W2.

Evaluates:
  Cell (Train365, Eval365) = H1 best.pt + 365d eval warmup
  Cell (Train365, Eval730) = H1 best.pt + 730d eval warmup
  Cell (Train730, Eval365) = W2 best.pt + 365d eval warmup
  Cell (Train730, Eval730) = W2 best.pt + 730d eval warmup

Strict Isolation:
  Evaluated strictly on INNER-VAL (1990-10-01..1995-09-30 scored). TEST period is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src"), str(REPO_ROOT / "dmotpy")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model

import project.benchmark.scripts.diagnostics.h_training_pilot as H1
NATIVE = H1.NATIVE

KGE_EPS = 0.1


def evaluate_inner_val_kge(
    model_name: str,
    hydro: torch.nn.Module,
    network: torch.nn.Module,
    attrs: torch.Tensor,
    forcing_x: torch.Tensor,
    target_y: torch.Tensor,
    warmup_days: int,
) -> tuple[dict[str, float], np.ndarray]:
    network.eval()
    with torch.no_grad():
        theta = network(attrs)
        q = hydro({"x_phy": forcing_x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        invalid_count = int((~torch.isfinite(q)).sum().detach().item())
        _loss, kge_tensor = NATIVE.compute_differentiable_kge(q, target_y, warmup_days=warmup_days, eps=KGE_EPS)

    kges = kge_tensor.cpu().numpy()
    valid_kges = kges[np.isfinite(kges)]
    if len(valid_kges) == 0:
        return {"median": -1.0, "mean": -1.0, "q25": -1.0, "q10": -1.0, "invalid_basins": invalid_count}, kges

    return {
        "median": float(np.median(valid_kges)),
        "mean": float(np.mean(valid_kges)),
        "q25": float(np.percentile(valid_kges, 25)),
        "q10": float(np.percentile(valid_kges, 10)),
        "invalid_basins": invalid_count,
    }, kges


def run_2x2_evaluation(ablation_root: Path, device_str: str = "cpu") -> None:
    device = torch.device(device_str if (torch.cuda.is_available() and "cuda" in device_str) else "cpu")
    manifest_path = ablation_root / "experiment_manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    # Load Audit Table
    audit_path = ablation_root / "PHASE_W2_UPDATE_BUDGET_AUDIT.csv"
    df_audit = pd.read_csv(audit_path)
    models = list(df_audit["model"])

    ids = [int(x) for x in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device=str(device), method="zscore")
    train_x_raw, train_y_raw, val_x_raw, val_y_raw = NATIVE.load_camels_time_series(ids)

    # Scored period: 1990-10-01..1995-09-30 (1826 days) -> index 3652..5478
    # EvalWarmup=365: 1989-10-01..1995-09-30 (2191 days) -> index 3287..5478, warmup=365
    # EvalWarmup=730: 1988-10-01..1995-09-30 (2556 days) -> index 2922..5478, warmup=730
    x_w365_raw = torch.as_tensor(train_x_raw[3287:], dtype=torch.float32, device=device)
    y_w365_raw = torch.as_tensor(train_y_raw[3287:], dtype=torch.float32, device=device)

    x_w730_raw = torch.as_tensor(train_x_raw[2922:], dtype=torch.float32, device=device)
    y_w730_raw = torch.as_tensor(train_y_raw[2922:], dtype=torch.float32, device=device)

    decomp_rows = []
    summary_rows = []

    print(f"=== Running 2x2 Warmup Decomposition on {len(models)} models ({device}) ===")

    for _, row_info in df_audit.iterrows():
        m = row_info["model"]
        category = row_info["category"]
        h1_jid = row_info["H1_job_id"]
        w2_jid = row_info["W2_job_id"]

        h1_ckpt_path = ablation_root / "runs" / h1_jid / "best.pt"
        w2_ckpt_path = ablation_root / "runs" / w2_jid / "best.pt"

        if not h1_ckpt_path.exists() or not w2_ckpt_path.exists():
            print(f"  Warning: Checkpoints missing for {m} (H1: {h1_ckpt_path.exists()}, W2: {w2_ckpt_path.exists()}). Skipping.")
            continue

        # Prepare forcing with calendar channels if applicable
        x_w365 = x_w365_raw
        x_w730 = x_w730_raw
        if m in CALENDAR_MODELS:
            x_w365, _ = add_calendar_forcing(
                x_w365, pd.date_range("1989-10-01", periods=len(x_w365), freq="D"), model_name=m
            )
            x_w730, _ = add_calendar_forcing(
                x_w730, pd.date_range("1988-10-01", periods=len(x_w730), freq="D"), model_name=m
            )

        # Single implemented warmup-gradient mode for all models (incl. penman):
        # no-grad warmup + state detach; scored period full backpropagation.
        # Historical "truncate:90" penman label was never implemented and is removed.
        warm_mode = "detach"
        backend = "compile" if (device.type == "cuda" and hasattr(torch, "compile")) else "eager"

        # Model wrappers
        hydro_eval365 = build_model(m, device, warm_up=365, backend=backend, parameter_mapping="auto", warmup_grad_mode=warm_mode)
        hydro_eval730 = build_model(m, device, warm_up=730, backend=backend, parameter_mapping="auto", warmup_grad_mode=warm_mode)

        # 1. Evaluate Checkpoint A (H1: Train365)
        h1_payload = torch.load(h1_ckpt_path, map_location="cpu", weights_only=False)
        net_h1 = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36[m], hidden_dims=[256, 256], dropout=0.05).to(device, dtype=torch.float64)
        net_h1.load_state_dict(h1_payload["network"])

        h1_eval365_metrics, kge_h1_e365 = evaluate_inner_val_kge(m, hydro_eval365, net_h1, attrs, x_w365, y_w365_raw, warmup_days=365)
        h1_eval730_metrics, kge_h1_e730 = evaluate_inner_val_kge(m, hydro_eval730, net_h1, attrs, x_w730, y_w730_raw, warmup_days=730)

        # 2. Evaluate Checkpoint B (W2: Train730)
        w2_payload = torch.load(w2_ckpt_path, map_location="cpu", weights_only=False)
        net_w2 = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36[m], hidden_dims=[256, 256], dropout=0.05).to(device, dtype=torch.float64)
        net_w2.load_state_dict(w2_payload["network"])

        w2_eval365_metrics, kge_w2_e365 = evaluate_inner_val_kge(m, hydro_eval365, net_w2, attrs, x_w365, y_w365_raw, warmup_days=365)
        w2_eval730_metrics, kge_w2_e730 = evaluate_inner_val_kge(m, hydro_eval730, net_w2, attrs, x_w730, y_w730_raw, warmup_days=730)

        # Save per-job evaluation JSONs and basin CSVs
        w2_run_dir = ablation_root / "runs" / w2_jid
        (w2_run_dir / "inner_val_eval_w365.json").write_text(json.dumps(w2_eval365_metrics, indent=2))
        (w2_run_dir / "inner_val_eval_w730.json").write_text(json.dumps(w2_eval730_metrics, indent=2))

        pd.DataFrame({"basin_id": ids, "kge": kge_w2_e365}).to_csv(w2_run_dir / "basin_metrics_eval_w365.csv", index=False)
        pd.DataFrame({"basin_id": ids, "kge": kge_w2_e730}).to_csv(w2_run_dir / "basin_metrics_eval_w730.csv", index=False)

        # Compute Primary Deltas under common EvalWarmup=730
        t365_e730_med = h1_eval730_metrics["median"]
        t730_e730_med = w2_eval730_metrics["median"]
        t365_e730_q25 = h1_eval730_metrics["q25"]
        t730_e730_q25 = w2_eval730_metrics["q25"]

        delta_train_warmup_med = t730_e730_med - t365_e730_med
        delta_train_warmup_q25 = t730_e730_q25 - t365_e730_q25

        # Compute Eval Warmup Effect for H1 and W2
        delta_eval_warmup_h1_med = h1_eval730_metrics["median"] - h1_eval365_metrics["median"]
        delta_eval_warmup_w2_med = w2_eval730_metrics["median"] - w2_eval365_metrics["median"]

        # Basin-level improvements
        valid_mask = np.isfinite(kge_h1_e730) & np.isfinite(kge_w2_e730)
        paired_basin_diff = kge_w2_e730[valid_mask] - kge_h1_e730[valid_mask]
        frac_basins_improved = float(np.mean(paired_basin_diff > 0)) if len(paired_basin_diff) > 0 else 0.0
        med_basin_diff = float(np.median(paired_basin_diff)) if len(paired_basin_diff) > 0 else 0.0

        # Gate Verdict
        if delta_train_warmup_med >= 0.005 or delta_train_warmup_q25 >= 0.010:
            verdict = "WARMUP_BENEFIT"
        elif abs(delta_train_warmup_med) < 0.003 and abs(delta_train_warmup_q25) < 0.005:
            verdict = "WARMUP_NEUTRAL"
        elif delta_train_warmup_med <= -0.005 and delta_train_warmup_q25 <= -0.005:
            verdict = "WARMUP_HARM"
        else:
            verdict = "MIXED"

        decomp_rows.append({
            "model": m,
            "category": category,
            "H1_job": h1_jid,
            "W2_job": w2_jid,
            "target_updates": row_info["W2_target_updates"],
            "t365_e365_med_kge": h1_eval365_metrics["median"],
            "t365_e730_med_kge": h1_eval730_metrics["median"],
            "t730_e365_med_kge": w2_eval365_metrics["median"],
            "t730_e730_med_kge": w2_eval730_metrics["median"],
            "delta_train_warmup_med (T730-T365 @ E730)": delta_train_warmup_med,
            "t365_e365_q25_kge": h1_eval365_metrics["q25"],
            "t365_e730_q25_kge": h1_eval730_metrics["q25"],
            "t730_e365_q25_kge": w2_eval365_metrics["q25"],
            "t730_e730_q25_kge": w2_eval730_metrics["q25"],
            "delta_train_warmup_q25 (T730-T365 @ E730)": delta_train_warmup_q25,
            "delta_eval_warmup_H1_med (E730-E365)": delta_eval_warmup_h1_med,
            "delta_eval_warmup_W2_med (E730-E365)": delta_eval_warmup_w2_med,
            "frac_basins_improved": frac_basins_improved,
            "med_basin_diff": med_basin_diff,
            "verdict": verdict,
        })

        summary_rows.append({
            "model": m,
            "category": category,
            "H1_t365_e365_med": h1_eval365_metrics["median"],
            "W2_t730_e730_med": w2_eval730_metrics["median"],
            "primary_delta_med": delta_train_warmup_med,
            "primary_delta_q25": delta_train_warmup_q25,
            "verdict": verdict,
        })

        print(f"  {m:12s} ({category:17s}) | T365->T730 @E730: {t365_e730_med:.4f} -> {t730_e730_med:.4f} (Δ={delta_train_warmup_med:+.4f}) | Verdict: {verdict}")

    df_decomp = pd.DataFrame(decomp_rows)
    df_sum = pd.DataFrame(summary_rows)

    df_decomp.to_csv(ablation_root / "PHASE_W2_2X2_WARMUP_DECOMPOSITION.csv", index=False)
    df_sum.to_csv(ablation_root / "PHASE_W2_SUMMARY.csv", index=False)

    # Generate Markdown Report
    md_text = """# Phase W2: Warmup-Length Matched-Update Ablation Report (2026-08-31)

**Study Scope**: 8 Representative Hydrological Models evaluating Training Warmup (365d vs 730d) with fixed 365d scored horizon and strictly matched optimizer update budget.  
**Evaluation Partition**: Evaluated strictly on INNER-VAL (`1990-10-01..1995-09-30`) on restored `best.pt`. TEST partition strictly untouched.

---

## 1. 2×2 Warmup Decomposition Table

| Model | Category | Updates | T365/E365 | T365/E730 | T730/E365 | T730/E730 | Primary Δ Med (T730-T365@E730) | Primary Δ Q25 | Eval Effect H1 (E730-E365) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
"""
    for _, r in df_decomp.iterrows():
        md_text += (
            f"| `{r['model']}` | `{r['category']}` | {r['target_updates']} | "
            f"{r['t365_e365_med_kge']:.4f} | {r['t365_e730_med_kge']:.4f} | {r['t730_e365_med_kge']:.4f} | **{r['t730_e730_med_kge']:.4f}** | "
            f"**{r['delta_train_warmup_med (T730-T365 @ E730)']:+.4f}** | {r['delta_train_warmup_q25 (T730-T365 @ E730)']:+.4f} | "
            f"{r['delta_eval_warmup_H1_med (E730-E365)']:+.4f} | `{r['verdict']}` |\n"
        )

    md_text += """
---

## 2. Decision Rules & Criteria
- **`WARMUP_BENEFIT`**: $\\Delta \\text{Median KGE} \\ge +0.005$ OR $\\Delta \\text{Q25 KGE} \\ge +0.010$.
- **`WARMUP_NEUTRAL`**: $|\\Delta \\text{Median KGE}| < 0.003$ AND $|\\Delta \\text{Q25 KGE}| < 0.005$.
- **`WARMUP_HARM`**: $\\Delta \\text{Median KGE} \\le -0.005$ AND $\\Delta \\text{Q25 KGE} \\le -0.005$.
- **`MIXED`**: Intermediate response.

---

## 3. Findings & Protocol Implications
1. **Training Warmup vs Evaluation Warmup Disentanglement**:
   - Compares whether gains come from better neural parameter learning during training (with 730d initial state decay) versus merely providing longer antecedent memory during inference.
2. **Canonical dPL Protocol Recommendation**:
   - If $\\ge 2/3$ warmup-sensitive models benefit without harming control models, 730d warmup becomes the primary candidate for canonical dPL v2.
"""
    (ablation_root / "PHASE_W2_REPORT.md").write_text(md_text)
    print(f"Saved Phase W2 2x2 decomposition and report to {ablation_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="project/benchmark/results/dpl_protocol_ablation_v2_20260831")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_2x2_evaluation(Path(args.root).resolve(), args.device)


if __name__ == "__main__":
    main()
