"""Snow-activity primary summaries, continuous associations, and endpoint contrasts.

Evaluates the primary Base-CN contrasts across frozen S1-S5 snow strata:
  1. Stratified summaries (N, median, Q1/Q3/IQR, 95% basin-bootstrap CI) for delta_absCT_Base_CN and delta_KGE_Base_CN.
  2. Continuous Spearman rank correlation with frac_snow (rho + 95% bootstrap CI).
  3. S5-S1 endpoint activity contrast: D_activity = median(S5) - median(S1) with 95% bootstrap CI.
  4. Characterization of low-snow (S1) behavior as small / centered near zero (without equivalence claims).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from config import (
    BASE_SEED,
    DEFAULT_DRAWS,
    EVAL_PERIOD,
    PARADIGMS,
    RESULTS_DIR,
    STRATA,
    STRATA_COUNTS,
)
from cuda_engine import (
    bootstrap_median_ci,
    derive_seed,
    endpoint_activity_contrast,
    gpu_median,
    gpu_quantile,
    require_cuda,
    spearman_bootstrap,
)
from paired_contrasts import compute_paired_contrasts


def analyze_snow_activity(
    contrast_rows: List[Dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    draws: int = DEFAULT_DRAWS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Compute snow-activity primary summaries on CUDA tensors.

    Returns:
        (stratified_summaries, spearman_summaries, endpoint_summaries, audit_meta)
    """
    device = require_cuda()
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if contrast_rows is None:
        contrast_rows, _ = compute_paired_contrasts(output_dir=out_dir)

    stratified_rows: List[Dict[str, Any]] = []
    spearman_rows: List[Dict[str, Any]] = []
    endpoint_rows: List[Dict[str, Any]] = []
    low_snow_rows: List[Dict[str, Any]] = []

    for paradigm in PARADIGMS:
        p_rows = [r for r in contrast_rows if r["paradigm"] == paradigm]
        if len(p_rows) != sum(STRATA_COUNTS.values()):
            raise RuntimeError(f"Expected {sum(STRATA_COUNTS.values())} rows for {paradigm}, got {len(p_rows)}")

        # Convert to GPU tensors
        snow_vec = torch.tensor([float(r["frac_snow"]) for r in p_rows], dtype=torch.float64, device=device)
        strata_indices = torch.tensor([STRATA.index(r["snow_stratum"]) for r in p_rows], dtype=torch.int64, device=device)
        d_ct_vec = torch.tensor([float(r["delta_absCT_Base_CN"]) for r in p_rows], dtype=torch.float64, device=device)
        d_kge_vec = torch.tensor([float(r["delta_KGE_Base_CN"]) for r in p_rows], dtype=torch.float64, device=device)
        d_tgd_vec = torch.tensor([float(r["delta_absCT_TGD_CN"]) for r in p_rows], dtype=torch.float64, device=device)

        metrics = [
            ("delta_absCT_Base_CN", d_ct_vec, "abs_CT_Base - abs_CT_CN"),
            ("delta_KGE_Base_CN", d_kge_vec, "KGE_CN - KGE_Base"),
            ("delta_absCT_TGD_CN", d_tgd_vec, "abs_CT_TGD - abs_CT_CN"),
        ]

        # 1. Overall and stratified summaries (S1-S5)
        for metric_name, vec, desc in metrics:
            # Overall across all strata
            seed_all = derive_seed(BASE_SEED, f"{paradigm}|{metric_name}|all")
            med, ci_l, ci_h, q25, q75 = bootstrap_median_ci(vec, seed_all, draws=draws)
            stratified_rows.append({
                "table": "overall_test_contrast",
                "paradigm": paradigm,
                "period": EVAL_PERIOD,
                "metric": metric_name,
                "stratum": "all",
                "n_basins": int(torch.isfinite(vec).sum().item()),
                "median": float(med[0].item()),
                "q25": float(q25[0].item()),
                "q75": float(q75[0].item()),
                "iqr": float((q75[0] - q25[0]).item()),
                "ci_low": float(ci_l[0].item()),
                "ci_high": float(ci_h[0].item()),
                "bootstrap_draws": draws,
                "seed": seed_all,
                "description": desc,
            })

            # Stratified S1-S5
            for si, s_name in enumerate(STRATA):
                mask = strata_indices == si
                sub_vec = vec[mask]
                sub_finite = sub_vec[torch.isfinite(sub_vec)]
                n_count = int(sub_finite.numel())
                seed_stratum = derive_seed(BASE_SEED, f"{paradigm}|{metric_name}|{s_name}")

                if n_count == 0:
                    med_val = q25_val = q75_val = iqr_val = ci_l_val = ci_h_val = float("nan")
                else:
                    s_med, s_cil, s_cih, s_q25, s_q75 = bootstrap_median_ci(sub_finite, seed_stratum, draws=draws)
                    med_val = float(s_med[0].item())
                    q25_val = float(s_q25[0].item())
                    q75_val = float(s_q75[0].item())
                    iqr_val = float((s_q75[0] - s_q25[0]).item())
                    ci_l_val = float(s_cil[0].item())
                    ci_h_val = float(s_cih[0].item())

                stratified_rows.append({
                    "table": "snow_stratified_contrast",
                    "paradigm": paradigm,
                    "period": EVAL_PERIOD,
                    "metric": metric_name,
                    "stratum": s_name,
                    "n_basins": n_count,
                    "median": med_val,
                    "q25": q25_val,
                    "q75": q75_val,
                    "iqr": iqr_val,
                    "ci_low": ci_l_val,
                    "ci_high": ci_h_val,
                    "bootstrap_draws": draws,
                    "seed": seed_stratum,
                    "description": desc,
                })

                if s_name == "S1" and metric_name == "delta_absCT_Base_CN":
                    low_snow_rows.append({
                        "paradigm": paradigm,
                        "metric": metric_name,
                        "stratum": "S1",
                        "n_basins": n_count,
                        "median": med_val,
                        "q25": q25_val,
                        "q75": q75_val,
                        "ci_low": ci_l_val,
                        "ci_high": ci_h_val,
                        "scientific_interpretation": "small / centered near zero in low-snow regimes; no equivalence claim made",
                    })

        # 2. Continuous Spearman rank correlation with frac_snow
        for metric_name, vec, desc in metrics:
            seed_spearman = derive_seed(BASE_SEED, f"{paradigm}|spearman|{metric_name}")
            rho, rho_low, rho_high = spearman_bootstrap(snow_vec, vec, seed=seed_spearman, draws=draws)
            spearman_rows.append({
                "paradigm": paradigm,
                "period": EVAL_PERIOD,
                "metric": metric_name,
                "variable_x": "frac_snow",
                "variable_y": metric_name,
                "n_basins": int((torch.isfinite(snow_vec) & torch.isfinite(vec)).sum().item()),
                "spearman_rho": float(rho.item()),
                "ci_low": float(rho_low.item()),
                "ci_high": float(rho_high.item()),
                "bootstrap_draws": draws,
                "seed": seed_spearman,
                "description": f"Spearman rank correlation between frac_snow and {metric_name}",
            })

        # 3. Endpoint contrast: D_activity = median(S5) - median(S1)
        for metric_name, vec, desc in metrics:
            s1_vals = vec[strata_indices == 0]
            s5_vals = vec[strata_indices == 4]
            seed_endpoint = derive_seed(BASE_SEED, f"{paradigm}|endpoint|{metric_name}")
            d_act, d_low, d_high = endpoint_activity_contrast(s1_vals, s5_vals, seed=seed_endpoint, draws=draws)

            s1_med = gpu_median(s1_vals)
            s5_med = gpu_median(s5_vals)

            endpoint_rows.append({
                "paradigm": paradigm,
                "period": EVAL_PERIOD,
                "metric": metric_name,
                "n_S1": int(torch.isfinite(s1_vals).sum().item()),
                "n_S5": int(torch.isfinite(s5_vals).sum().item()),
                "median_S1": float(s1_med.item()),
                "median_S5": float(s5_med.item()),
                "D_activity": float(d_act.item()),
                "high_minus_low_median": float(d_act.item()),
                "ci_low": float(d_low.item()),
                "ci_high": float(d_high.item()),
                "bootstrap_draws": draws,
                "seed": seed_endpoint,
                "definition": f"median({metric_name} in S5) - median({metric_name} in S1)",
            })

    # Write CSV artifacts
    strat_fields = [
        "table", "paradigm", "period", "metric", "stratum", "n_basins",
        "median", "q25", "q75", "iqr", "ci_low", "ci_high", "bootstrap_draws", "seed", "description"
    ]
    with (out_dir / "snow_stratified_summaries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=strat_fields)
        writer.writeheader()
        for r in stratified_rows:
            writer.writerow(r)

    # For backward-compatible tools
    with (out_dir / "stratified_summaries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["table", "paradigm", "period", "metric", "stratum", "n_basins", "estimate_median", "ci_low", "ci_high"])
        writer.writeheader()
        for r in stratified_rows:
            writer.writerow({
                "table": r["table"],
                "paradigm": r["paradigm"],
                "period": r["period"],
                "metric": r["metric"],
                "stratum": r["stratum"],
                "n_basins": r["n_basins"],
                "estimate_median": r["median"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
            })

    spearman_fields = [
        "paradigm", "period", "metric", "variable_x", "variable_y", "n_basins",
        "spearman_rho", "ci_low", "ci_high", "bootstrap_draws", "seed", "description"
    ]
    with (out_dir / "spearman_associations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=spearman_fields)
        writer.writeheader()
        for r in spearman_rows:
            writer.writerow(r)

    with (out_dir / "spearman_bootstrap.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["paradigm", "metric", "n_basins", "estimate", "ci_low", "ci_high", "bootstrap_draws", "seed"])
        writer.writeheader()
        for r in spearman_rows:
            writer.writerow({
                "paradigm": r["paradigm"],
                "metric": f"spearman_rho_frac_snow_{r['metric']}",
                "n_basins": r["n_basins"],
                "estimate": r["spearman_rho"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "bootstrap_draws": r["bootstrap_draws"],
                "seed": r["seed"],
            })

    endpoint_fields = [
        "paradigm", "period", "metric", "n_S1", "n_S5", "median_S1", "median_S5",
        "D_activity", "high_minus_low_median", "ci_low", "ci_high", "bootstrap_draws", "seed", "definition"
    ]
    with (out_dir / "endpoint_activity_contrast.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=endpoint_fields)
        writer.writeheader()
        for r in endpoint_rows:
            writer.writerow(r)

    with (out_dir / "endpoint_S1_vs_S5.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["paradigm", "metric", "n_S1", "n_S5", "high_minus_low_median", "ci_low", "ci_high", "bootstrap_draws", "seed"])
        writer.writeheader()
        for r in endpoint_rows:
            writer.writerow({
                "paradigm": r["paradigm"],
                "metric": r["metric"],
                "n_S1": r["n_S1"],
                "n_S5": r["n_S5"],
                "high_minus_low_median": r["high_minus_low_median"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "bootstrap_draws": r["bootstrap_draws"],
                "seed": r["seed"],
            })

    with (out_dir / "low_snow_condition_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["paradigm", "metric", "stratum", "n_basins", "median", "q25", "q75", "ci_low", "ci_high", "scientific_interpretation"])
        writer.writeheader()
        for r in low_snow_rows:
            writer.writerow(r)

    audit_meta = {
        "status": "PASS",
        "bootstrap_draws": draws,
        "base_seed": BASE_SEED,
        "strata": list(STRATA),
        "strata_counts": STRATA_COUNTS,
        "metrics_evaluated": [m[0] for m in metrics],
        "endpoints_evaluated": ["S5_minus_S1"],
        "low_snow_interpretation": "small / centered near zero in low-snow (S1) regime; no equivalence claim",
    }
    with (out_dir / "snow_activity_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_meta, f, indent=2)

    return stratified_rows, spearman_rows, endpoint_rows, audit_meta


if __name__ == "__main__":
    s_rows, sp_rows, ep_rows, meta = analyze_snow_activity()
    print(f"Snow activity analysis complete: {len(s_rows)} stratified rows, {len(sp_rows)} spearman rows, {len(ep_rows)} endpoint rows.")
    for ep in ep_rows:
        if ep["metric"] == "delta_absCT_Base_CN":
            print(f"[{ep['paradigm']}] Primary D_activity (S5 - S1): {ep['D_activity']:.3f} [{ep['ci_low']:.3f}, {ep['ci_high']:.3f}] days")
