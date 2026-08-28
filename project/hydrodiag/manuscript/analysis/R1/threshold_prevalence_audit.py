"""Threshold prevalence and denominator audit.

Audits metric cutoffs across structure-specific versus common-pass denominators:
  - KGE cutoffs: 0.40, 0.50, 0.60, 0.70, 0.80 (and fine grid 0.40..0.80 by 0.01).
  - CT error cutoffs: 10, 15, 20 days.

Prevalence Definitions:
  1. Conditional prevalence (primary for timing inconsistency under acceptable KGE):
       P(|CT| >= c | KGE >= k) = N(KGE >= k, |CT| >= c) / N_condition
       - Structure-specific: N_condition = N_s(KGE_s >= k)
       - Common-pass: N_condition = N_common(KGE_Base >= k & KGE_TGD >= k & KGE_CN >= k)
  2. Joint prevalence (descriptive cross-sample prevalence):
       P(KGE >= k, |CT| >= c) = N(KGE >= k, |CT| >= c) / N_all_valid (N=531)

Uncertainty:
  - 95% basin-bootstrap confidence intervals computed via GPU resampling (unit = basin).
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
    STRUCTURES,
    TOTAL_BASINS,
)
from cuda_engine import derive_seed, require_cuda
from paired_contrasts import compute_paired_contrasts


def audit_threshold_prevalence(
    contrast_rows: List[Dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    draws: int = DEFAULT_DRAWS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Compute threshold prevalence audit across conditional and joint definitions on CUDA.

    Returns:
        (audit_rows, key_cutoff_rows, meta)
    """
    device = require_cuda()
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if contrast_rows is None:
        contrast_rows, _ = compute_paired_contrasts(output_dir=out_dir)

    audit_rows: List[Dict[str, Any]] = []
    key_cutoff_rows: List[Dict[str, Any]] = []

    kge_cutoffs_grid = [round(i / 100.0, 2) for i in range(40, 81)]
    key_kge_cutoffs = [0.40, 0.50, 0.60, 0.70, 0.80]
    ct_cutoffs = [10, 15, 20]

    for paradigm in PARADIGMS:
        p_rows = [r for r in contrast_rows if r["paradigm"] == paradigm]
        if len(p_rows) != TOTAL_BASINS:
            raise RuntimeError(f"Expected {TOTAL_BASINS} basins for {paradigm}, got {len(p_rows)}")

        # KGE matrix: shape (531, 3) for Base, TGD, CN
        kge_mat = torch.tensor(
            [[float(r["KGE_Base"]), float(r["KGE_TGD"]), float(r["KGE_CN"])] for r in p_rows],
            dtype=torch.float64,
            device=device,
        )
        # Timing error matrix (|signed_e|): shape (531, 3)
        ct_mat = torch.tensor(
            [[float(r["abs_e_Base"]), float(r["abs_e_TGD"]), float(r["abs_e_CN"])] for r in p_rows],
            dtype=torch.float64,
            device=device,
        )

        valid_kge = torch.isfinite(kge_mat)
        valid_ct = torch.isfinite(ct_mat)
        valid_both = valid_kge & valid_ct
        all_valid = valid_both.all(dim=1)
        n_all_valid = int(all_valid.sum().item())

        # Preallocate bootstrap indices for uncertainty on key combinations
        boot_gen = torch.Generator(device=device)
        boot_gen.manual_seed(derive_seed(BASE_SEED, f"{paradigm}|threshold_bootstrap"))
        boot_indices = torch.randint(TOTAL_BASINS, (draws, TOTAL_BASINS), generator=boot_gen, device=device)

        for s_idx, structure in enumerate(STRUCTURES):
            struct_valid = valid_both[:, s_idx]

            for kge_tau in kge_cutoffs_grid:
                # Structure-specific condition: structure itself has KGE >= kge_tau
                struct_kge_pass = struct_valid & (kge_mat[:, s_idx] >= kge_tau)
                # Common pass condition: all structures valid and all achieve KGE >= kge_tau
                common_kge_pass = all_valid & (kge_mat >= kge_tau).all(dim=1)

                for ct_tau in ct_cutoffs:
                    # Timing large error condition: |CT| >= ct_tau
                    timing_large = struct_valid & (ct_mat[:, s_idx] >= ct_tau)
                    timing_acceptable = struct_valid & (ct_mat[:, s_idx] < ct_tau)

                    # 1. Structure-specific denominator
                    num_struct = int((struct_kge_pass & timing_large).sum().item())
                    den_struct_cond = int(struct_kge_pass.sum().item())
                    den_struct_all = int(struct_valid.sum().item())

                    cond_prev_struct = num_struct / den_struct_cond if den_struct_cond > 0 else float("nan")
                    joint_prev_struct = num_struct / den_struct_all if den_struct_all > 0 else float("nan")

                    # Bootstrap CI for key cutoffs
                    cond_ci_l_struct = cond_ci_h_struct = joint_ci_l_struct = joint_ci_h_struct = float("nan")
                    if kge_tau in key_kge_cutoffs and den_struct_cond > 0:
                        joint_mask_b = (struct_kge_pass & timing_large)[boot_indices]
                        cond_mask_b = struct_kge_pass[boot_indices]
                        valid_mask_b = struct_valid[boot_indices]

                        b_num = joint_mask_b.sum(dim=1).to(torch.float64)
                        b_den_cond = cond_mask_b.sum(dim=1).to(torch.float64)
                        b_den_all = valid_mask_b.sum(dim=1).to(torch.float64)

                        b_cond = torch.where(b_den_cond > 0, b_num / b_den_cond, torch.full_like(b_den_cond, float("nan")))
                        b_joint = torch.where(b_den_all > 0, b_num / b_den_all, torch.full_like(b_den_all, float("nan")))

                        b_cond_fin = b_cond[torch.isfinite(b_cond)]
                        if b_cond_fin.numel() > 0:
                            cond_ci_l_struct = float(torch.quantile(b_cond_fin, 0.025).item())
                            cond_ci_h_struct = float(torch.quantile(b_cond_fin, 0.975).item())

                        b_joint_fin = b_joint[torch.isfinite(b_joint)]
                        if b_joint_fin.numel() > 0:
                            joint_ci_l_struct = float(torch.quantile(b_joint_fin, 0.025).item())
                            joint_ci_h_struct = float(torch.quantile(b_joint_fin, 0.975).item())

                    rec_struct = {
                        "paradigm": paradigm,
                        "structure": structure,
                        "kge_threshold": kge_tau,
                        "ct_threshold": ct_tau,
                        "denominator_type": "structure_specific",
                        "numerator": num_struct,
                        "conditional_denominator": den_struct_cond,
                        "conditional_prevalence": cond_prev_struct,
                        "conditional_ci_low": cond_ci_l_struct,
                        "conditional_ci_high": cond_ci_h_struct,
                        "all_valid_denominator": den_struct_all,
                        "joint_prevalence": joint_prev_struct,
                        "joint_ci_low": joint_ci_l_struct,
                        "joint_ci_high": joint_ci_h_struct,
                        "n_kge_pass": den_struct_cond,
                        "fraction_kge_pass": den_struct_cond / den_struct_all if den_struct_all > 0 else float("nan"),
                        "n_ct_acceptable": int(timing_acceptable.sum().item()),
                        "fraction_ct_acceptable": int(timing_acceptable.sum().item()) / den_struct_all if den_struct_all > 0 else float("nan"),
                        "n_timing_large": int(timing_large.sum().item()),
                        "fraction_timing_large": int(timing_large.sum().item()) / den_struct_all if den_struct_all > 0 else float("nan"),
                    }
                    audit_rows.append(rec_struct)
                    if kge_tau in key_kge_cutoffs:
                        key_cutoff_rows.append(rec_struct)

                    # 2. Common-pass denominator
                    num_common = int((common_kge_pass & timing_large).sum().item())
                    den_common = int(common_kge_pass.sum().item())

                    cond_prev_common = num_common / den_common if den_common > 0 else float("nan")
                    joint_prev_common = num_common / n_all_valid if n_all_valid > 0 else float("nan")

                    cond_ci_l_com = cond_ci_h_com = float("nan")
                    if kge_tau in key_kge_cutoffs and den_common > 0:
                        joint_mask_cb = (common_kge_pass & timing_large)[boot_indices]
                        cond_mask_cb = common_kge_pass[boot_indices]

                        b_num_c = joint_mask_cb.sum(dim=1).to(torch.float64)
                        b_den_c = cond_mask_cb.sum(dim=1).to(torch.float64)

                        b_cond_c = torch.where(b_den_c > 0, b_num_c / b_den_c, torch.full_like(b_den_c, float("nan")))
                        b_cond_c_fin = b_cond_c[torch.isfinite(b_cond_c)]
                        if b_cond_c_fin.numel() > 0:
                            cond_ci_l_com = float(torch.quantile(b_cond_c_fin, 0.025).item())
                            cond_ci_h_com = float(torch.quantile(b_cond_c_fin, 0.975).item())

                    rec_common = {
                        "paradigm": paradigm,
                        "structure": structure,
                        "kge_threshold": kge_tau,
                        "ct_threshold": ct_tau,
                        "denominator_type": "common_all_structures_pass",
                        "numerator": num_common,
                        "conditional_denominator": den_common,
                        "conditional_prevalence": cond_prev_common,
                        "conditional_ci_low": cond_ci_l_com,
                        "conditional_ci_high": cond_ci_h_com,
                        "all_valid_denominator": n_all_valid,
                        "joint_prevalence": joint_prev_common,
                        "joint_ci_low": float("nan"),
                        "joint_ci_high": float("nan"),
                        "n_kge_pass": den_common,
                        "fraction_kge_pass": den_common / n_all_valid if n_all_valid > 0 else float("nan"),
                        "n_ct_acceptable": int((common_kge_pass & timing_acceptable).sum().item()),
                        "fraction_ct_acceptable": int((common_kge_pass & timing_acceptable).sum().item()) / den_common if den_common > 0 else float("nan"),
                        "n_timing_large": num_common,
                        "fraction_timing_large": cond_prev_common,
                    }
                    audit_rows.append(rec_common)
                    if kge_tau in key_kge_cutoffs:
                        key_cutoff_rows.append(rec_common)

    # Write output CSV files
    fields = [
        "paradigm", "structure", "kge_threshold", "ct_threshold", "denominator_type",
        "numerator", "conditional_denominator", "conditional_prevalence",
        "conditional_ci_low", "conditional_ci_high",
        "all_valid_denominator", "joint_prevalence", "joint_ci_low", "joint_ci_high",
        "n_kge_pass", "fraction_kge_pass", "n_ct_acceptable", "fraction_ct_acceptable",
        "n_timing_large", "fraction_timing_large",
    ]

    full_path = out_dir / "threshold_denominator_audit.csv"
    with full_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in audit_rows:
            writer.writerow(r)

    summary_path = out_dir / "threshold_prevalence_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in key_cutoff_rows:
            writer.writerow(r)

    # Extract primary conditional prevalence at KGE >= 0.60 & |CT| >= 15 d
    key_combinations = [
        r for r in key_cutoff_rows
        if r["kge_threshold"] == 0.60 and r["ct_threshold"] == 15
    ]

    meta = {
        "status": "PASS",
        "definitions": {
            "conditional_prevalence": "P(|CT| >= c | KGE >= k) = numerator / conditional_denominator",
            "joint_prevalence": "P(KGE >= k & |CT| >= c) = numerator / all_valid_denominator",
            "structure_specific_denominator": "N_s(KGE_s >= k)",
            "common_pass_denominator": "N_common(KGE_Base >= k & KGE_TGD >= k & KGE_CN >= k)",
        },
        "key_findings_kge060_ct15d": key_combinations,
        "total_audit_rows": len(audit_rows),
        "key_summary_rows": len(key_cutoff_rows),
        "bootstrap_draws": draws,
        "resampling_unit": "basin",
    }

    with (out_dir / "threshold_prevalence_audit.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return audit_rows, key_cutoff_rows, meta


if __name__ == "__main__":
    audit_r, sum_r, meta = audit_threshold_prevalence()
    print(f"Threshold prevalence audit complete: {len(audit_r)} grid rows, {len(sum_r)} key summary rows.")
    print("\n--- Key Finding: KGE >= 0.60 & |CT| >= 15 d ---")
    for r in meta["key_findings_kge060_ct15d"]:
        dtype = r["denominator_type"]
        p = r["paradigm"]
        s = r["structure"]
        num = r["numerator"]
        c_den = r["conditional_denominator"]
        c_prev = r["conditional_prevalence"] * 100
        c_l = r["conditional_ci_low"] * 100
        c_h = r["conditional_ci_high"] * 100
        j_prev = r["joint_prevalence"] * 100
        print(f"[{p}] {s:4s} ({dtype:28s}): Cond = {num}/{c_den} = {c_prev:5.2f}% [{c_l:5.2f}%, {c_h:5.2f}%] | Joint = {num}/531 = {j_prev:5.2f}%")
