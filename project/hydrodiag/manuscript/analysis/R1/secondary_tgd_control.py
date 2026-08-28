"""Secondary TGD structural control analysis.

Evaluates outlet-level TGD-CN paired contrasts (delta_absCT_TGD_CN = abs_CT_TGD - abs_CT_CN):
  - S1-S5 strata medians, IQR, and 95% basin-bootstrap CIs.
  - Overall median, IQR, and 95% bootstrap CI.
  - Continuous Spearman rho association with frac_snow.
  - S5-S1 endpoint contrast.

Scientific boundary:
  TGD serves strictly as a secondary output-level structural control to verify whether
  a calibrated degree-day parameterization reduces timing error. No claims of
  irreducible snow contribution, F_TGD, parameter compensation, or internal state
  correctness are inferred.
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
    require_cuda,
    spearman_bootstrap,
)
from paired_contrasts import compute_paired_contrasts


def analyze_secondary_tgd_control(
    contrast_rows: List[Dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    draws: int = DEFAULT_DRAWS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute secondary TGD structural control summaries.

    Returns:
        (summary_rows, audit_meta)
    """
    device = require_cuda()
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if contrast_rows is None:
        contrast_rows, _ = compute_paired_contrasts(output_dir=out_dir)

    summary_rows: List[Dict[str, Any]] = []

    for paradigm in PARADIGMS:
        p_rows = [r for r in contrast_rows if r["paradigm"] == paradigm]
        snow_vec = torch.tensor([float(r["frac_snow"]) for r in p_rows], dtype=torch.float64, device=device)
        strata_indices = torch.tensor([STRATA.index(r["snow_stratum"]) for r in p_rows], dtype=torch.int64, device=device)
        d_tgd_ct = torch.tensor([float(r["delta_absCT_TGD_CN"]) for r in p_rows], dtype=torch.float64, device=device)
        d_tgd_kge = torch.tensor([float(r["delta_KGE_TGD_CN"]) for r in p_rows], dtype=torch.float64, device=device)

        tgd_metrics = [
            ("delta_absCT_TGD_CN", d_tgd_ct, "abs_CT_TGD - abs_CT_CN"),
            ("delta_KGE_TGD_CN", d_tgd_kge, "KGE_CN - KGE_TGD"),
        ]

        for metric_name, vec, desc in tgd_metrics:
            # 1. Overall
            seed_all = derive_seed(BASE_SEED, f"{paradigm}|tgd_control|{metric_name}|all")
            med, ci_l, ci_h, q25, q75 = bootstrap_median_ci(vec, seed_all, draws=draws)
            summary_rows.append({
                "table": "tgd_control_overall",
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
                "spearman_rho": float("nan"),
                "spearman_ci_low": float("nan"),
                "spearman_ci_high": float("nan"),
                "bootstrap_draws": draws,
                "seed": seed_all,
                "role": "secondary_output_structural_control",
                "description": desc,
            })

            # 2. Stratified S1-S5
            for si, s_name in enumerate(STRATA):
                mask = strata_indices == si
                sub_vec = vec[mask]
                sub_finite = sub_vec[torch.isfinite(sub_vec)]
                n_count = int(sub_finite.numel())
                seed_stratum = derive_seed(BASE_SEED, f"{paradigm}|tgd_control|{metric_name}|{s_name}")

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

                summary_rows.append({
                    "table": "tgd_control_stratified",
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
                    "spearman_rho": float("nan"),
                    "spearman_ci_low": float("nan"),
                    "spearman_ci_high": float("nan"),
                    "bootstrap_draws": draws,
                    "seed": seed_stratum,
                    "role": "secondary_output_structural_control",
                    "description": desc,
                })

            # 3. Continuous Spearman correlation with frac_snow
            seed_spearman = derive_seed(BASE_SEED, f"{paradigm}|tgd_control|spearman|{metric_name}")
            rho, rho_l, rho_h = spearman_bootstrap(snow_vec, vec, seed=seed_spearman, draws=draws)
            summary_rows.append({
                "table": "tgd_control_continuous_association",
                "paradigm": paradigm,
                "period": EVAL_PERIOD,
                "metric": metric_name,
                "stratum": "continuous_frac_snow",
                "n_basins": int((torch.isfinite(snow_vec) & torch.isfinite(vec)).sum().item()),
                "median": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
                "iqr": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "spearman_rho": float(rho.item()),
                "spearman_ci_low": float(rho_l.item()),
                "spearman_ci_high": float(rho_h.item()),
                "bootstrap_draws": draws,
                "seed": seed_spearman,
                "role": "secondary_output_structural_control",
                "description": f"Spearman rho between frac_snow and {metric_name}",
            })

            # 4. Endpoint contrast: S5 - S1
            s1_vals = vec[strata_indices == 0]
            s5_vals = vec[strata_indices == 4]
            seed_endpoint = derive_seed(BASE_SEED, f"{paradigm}|tgd_control|endpoint|{metric_name}")
            d_act, d_low, d_high = endpoint_activity_contrast(s1_vals, s5_vals, seed=seed_endpoint, draws=draws)
            summary_rows.append({
                "table": "tgd_control_endpoint_S5_minus_S1",
                "paradigm": paradigm,
                "period": EVAL_PERIOD,
                "metric": metric_name,
                "stratum": "S5_minus_S1",
                "n_basins": int((torch.isfinite(s1_vals).sum() + torch.isfinite(s5_vals).sum()).item()),
                "median": float(d_act.item()),
                "q25": float("nan"),
                "q75": float("nan"),
                "iqr": float("nan"),
                "ci_low": float(d_low.item()),
                "ci_high": float(d_high.item()),
                "spearman_rho": float("nan"),
                "spearman_ci_low": float("nan"),
                "spearman_ci_high": float("nan"),
                "bootstrap_draws": draws,
                "seed": seed_endpoint,
                "role": "secondary_output_structural_control",
                "description": f"median({metric_name} in S5) - median({metric_name} in S1)",
            })

    # Write output
    fields = [
        "table", "paradigm", "period", "metric", "stratum", "n_basins",
        "median", "q25", "q75", "iqr", "ci_low", "ci_high",
        "spearman_rho", "spearman_ci_low", "spearman_ci_high",
        "bootstrap_draws", "seed", "role", "description"
    ]
    out_path = out_dir / "secondary_tgd_control_summaries.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    audit_meta = {
        "status": "PASS",
        "role": "secondary output-level structural control",
        "exclusions": [
            "No F_TGD calculation",
            "No irreducible snow contribution claim",
            "No parameter compensation inference",
            "No internal state verification claim",
        ],
        "output_path": str(out_path),
    }

    with (out_dir / "secondary_tgd_control_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_meta, f, indent=2)

    return summary_rows, audit_meta


if __name__ == "__main__":
    rows, meta = analyze_secondary_tgd_control()
    print(f"Secondary TGD control analysis complete: {len(rows)} summary rows.")
    for r in rows:
        if r["table"] == "tgd_control_overall" and r["metric"] == "delta_absCT_TGD_CN":
            print(f"[{r['paradigm']}] Overall delta_absCT_TGD_CN: {r['median']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] days")
