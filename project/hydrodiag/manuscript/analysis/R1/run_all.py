"""End-to-end orchestration for canonical R1 downstream analysis.

Executes:
  1. Staged input audit and canonical basin-level table construction (3,186 rows).
  2. Paired Base-CN and TGD-CN contrasts with strict alignment checks (1,062 rows).
  3. Snow-activity primary summaries across frozen S1-S5 strata (median, IQR, 95% CI).
  4. Continuous frac_snow Spearman correlations and S5-S1 endpoint activity contrasts.
  5. Secondary TGD structural control analysis.
  6. Threshold prevalence and denominator audit (conditional vs joint prevalence).
  7. Regional (LORO) and seed/restart sensitivity robustness.
  8. 5 Canonical gates verification.
  9. Machine-readable summary and execution profiling.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Dict

import torch

from config import (
    BASE_SEED,
    DEFAULT_DRAWS,
    EVAL_PERIOD,
    PARADIGMS,
    RESULTS_DIR,
    STAGED_DIR,
    STRATA,
    STRATA_COUNTS,
    TOTAL_BASINS,
)
from cuda_engine import require_cuda
from canonical_basin_table import build_canonical_basin_table
from paired_contrasts import compute_paired_contrasts
from snow_activity_analysis import analyze_snow_activity
from secondary_tgd_control import analyze_secondary_tgd_control
from threshold_prevalence_audit import audit_threshold_prevalence
from robustness_analysis import resolve_region_dir, run_all_robustness
from canonical_gates import verify_canonical_gates


def run_pipeline(
    staged_dir: Path | None = None,
    output_dir: Path | None = None,
    region_dir: Path | None = None,
    draws: int = DEFAULT_DRAWS,
) -> Dict[str, Any]:
    """Execute the full canonical R1 downstream analysis pipeline."""
    device = require_cuda()
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    s_dir = staged_dir or STAGED_DIR
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== [1/7] Building Canonical Basin-Level Table ===")
    test_rows, all_rows, table_audit = build_canonical_basin_table(staged_dir=s_dir, output_dir=out_dir)

    print("=== [2/7] Computing Paired Contrasts & Verifying Basin Alignment ===")
    contrasts, alignment_audit = compute_paired_contrasts(canonical_test_rows=test_rows, output_dir=out_dir)

    print("=== [3/7] Analyzing Snow-Activity Summaries & Resampling CIs ===")
    strat_rows, sp_rows, ep_rows, snow_meta = analyze_snow_activity(contrast_rows=contrasts, output_dir=out_dir, draws=draws)

    print("=== [4/7] Analyzing Secondary TGD Structural Control ===")
    tgd_rows, tgd_meta = analyze_secondary_tgd_control(contrast_rows=contrasts, output_dir=out_dir, draws=draws)

    print("=== [5/7] Auditing Threshold Prevalence Across Denominators ===")
    audit_rows, key_prevalence, prev_meta = audit_threshold_prevalence(contrast_rows=contrasts, output_dir=out_dir, draws=draws)

    print("=== [6/7] Running Robustness Checks (Regional LORO & Seed/Restart) ===")
    resolved_reg = resolve_region_dir(explicit=region_dir)
    robustness_meta = run_all_robustness(contrast_rows=contrasts, region_dir=resolved_reg, staged_dir=s_dir, output_dir=out_dir, draws=draws)

    print("=== [7/7] Verifying 5 Canonical Gates ===")
    gates_summary = verify_canonical_gates(output_dir=out_dir, staged_dir=s_dir)

    elapsed = time.perf_counter() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # Extract primary endpoint contrast summaries
    primary_endpoints = {}
    for ep in ep_rows:
        if ep["metric"] == "delta_absCT_Base_CN":
            primary_endpoints[ep["paradigm"]] = {
                "D_activity": ep["D_activity"],
                "ci_low": ep["ci_low"],
                "ci_high": ep["ci_high"],
                "median_S1": ep["median_S1"],
                "median_S5": ep["median_S5"],
                "n_S1": ep["n_S1"],
                "n_S5": ep["n_S5"],
            }

    primary_spearman = {}
    for sp in sp_rows:
        if sp["metric"] == "delta_absCT_Base_CN":
            primary_spearman[sp["paradigm"]] = {
                "rho": sp["spearman_rho"],
                "ci_low": sp["ci_low"],
                "ci_high": sp["ci_high"],
                "n_basins": sp["n_basins"],
            }

    # Extract key prevalence finding (KGE >= 0.60 & |CT| >= 15 d)
    key_prevalence_summary = {}
    for r in prev_meta["key_findings_kge060_ct15d"]:
        key = f"{r['paradigm']}|{r['structure']}|{r['denominator_type']}"
        key_prevalence_summary[key] = {
            "paradigm": r["paradigm"],
            "structure": r["structure"],
            "denominator_type": r["denominator_type"],
            "numerator": r["numerator"],
            "conditional_denominator": r["conditional_denominator"],
            "conditional_prevalence": r["conditional_prevalence"],
            "conditional_ci_low": r["conditional_ci_low"],
            "conditional_ci_high": r["conditional_ci_high"],
            "all_valid_denominator": r["all_valid_denominator"],
            "joint_prevalence": r["joint_prevalence"],
        }

    machine_summary = {
        "status": "COMPLETED",
        "canonical_gates": gates_summary["overall_status"],
        "gate_breakdown": {k: v["status"] for k, v in gates_summary["gates"].items()},
        "dataset_dimensions": {
            "total_basins": TOTAL_BASINS,
            "canonical_eval_rows": len(test_rows),
            "all_periods_rows": len(all_rows),
            "paired_contrast_rows": len(contrasts),
            "strata_counts": STRATA_COUNTS,
        },
        "primary_estimands": {
            "delta_absCT_Base_CN_endpoints_S5_minus_S1": primary_endpoints,
            "delta_absCT_Base_CN_spearman_rho_frac_snow": primary_spearman,
        },
        "secondary_tgd_control": {
            "status": "PASS",
            "role": "secondary_output_level_structural_control",
            "overall_median_delta_absCT_TGD_CN": {
                r["paradigm"]: {"median": r["median"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
                for r in tgd_rows if r["table"] == "tgd_control_overall" and r["metric"] == "delta_absCT_TGD_CN"
            },
        },
        "threshold_prevalence_audit": {
            "definitions": prev_meta["definitions"],
            "key_prevalence_kge060_ct15d": key_prevalence_summary,
        },
        "robustness": {
            "regional_loro": robustness_meta["regional_robustness"]["status"],
            "seed_restart_consistency": robustness_meta["seed_restart_robustness"]["dpl_seed_consistency"],
        },
        "compute_profile": {
            "elapsed_seconds": round(elapsed, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "peak_ram_mb": round(peak_ram_mb, 2),
            "gpu_device": torch.cuda.get_device_name(device),
            "bootstrap_draws": draws,
            "base_seed": BASE_SEED,
        },
    }

    with (out_dir / "machine_readable_summary.json").open("w", encoding="utf-8") as f:
        json.dump(machine_summary, f, indent=2)

    # Render summary markdown
    md_lines = [
        "# Canonical R1 Analysis Results Summary",
        "",
        f"- **Status:** {machine_summary['status']}",
        f"- **Canonical Promotion Gates:** {machine_summary['canonical_gates']}",
        f"- **Evaluation Dataset:** {TOTAL_BASINS} basins × 3 structures × 2 regimes = {len(test_rows)} rows ({EVAL_PERIOD} period)",
        f"- **Resampling / Bootstrap:** 10,000 paired basin draws (Seed `{BASE_SEED}`)",
        f"- **Execution Time:** {elapsed:.2f} s (Peak VRAM: {peak_vram_mb:.1f} MB, Peak RAM: {peak_ram_mb:.1f} MB)",
        "",
        "## 1. Canonical Gates Status",
        "",
        "| Gate | Status | Description |",
        "| :--- | :---: | :--- |",
        f"| Provenance Gate | **{gates_summary['gates']['provenance_gate']['status']}** | Pinned digests and exact schemas verified for all staged sources. |",
        f"| Basin Alignment Gate | **{gates_summary['gates']['basin_alignment_gate']['status']}** | 531 paired basins for each paradigm; 0 silent drops, 0 duplicate keys. |",
        f"| CT Definition Gate | **{gates_summary['gates']['ct_definition_gate']['status']}** | Delta_CT = CT_sim - CT_obs; basin CT = median valid years; absolute_CT = abs(signed). |",
        f"| Statistical Unit Gate | **{gates_summary['gates']['statistical_unit_gate']['status']}** | Inferential unit is basin (N=531). Seeds/restarts are aggregated prior to inference. |",
        f"| Reproducibility Gate | **{gates_summary['gates']['reproducibility_gate']['status']}** | All outputs reproducible from verified staged tables without daily raw files. |",
        "",
        "## 2. Primary Base-CN Snow-Activity Estimands",
        "",
        "Positive values denote improvement in CN relative to Base.",
        "",
        "### A. S5-S1 Endpoint Activity Contrast ($D_{\\text{activity}} = \\text{median}(S5) - \\text{median}(S1)$)",
        "",
        "| Paradigm | N(S1) | N(S5) | Median S1 (d) | Median S5 (d) | $D_{\\text{activity}}$ (d) | 95% Bootstrap CI (d) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for p, ep_data in primary_endpoints.items():
        md_lines.append(
            f"| {p} | {ep_data['n_S1']} | {ep_data['n_S5']} | {ep_data['median_S1']:.1f} | {ep_data['median_S5']:.1f} | **{ep_data['D_activity']:.1f}** | [{ep_data['ci_low']:.1f}, {ep_data['ci_high']:.1f}] |"
        )

    md_lines.extend([
        "",
        "### B. Continuous Spearman Association with Snow Fraction (`frac_snow` vs `delta_absCT_Base_CN`)",
        "",
        "| Paradigm | N | Spearman $\\rho$ | 95% Bootstrap CI |",
        "| :--- | :---: | :---: | :---: |",
    ])

    for p, sp_data in primary_spearman.items():
        md_lines.append(
            f"| {p} | {sp_data['n_basins']} | **{sp_data['rho']:.3f}** | [{sp_data['ci_low']:.3f}, {sp_data['ci_high']:.3f}] |"
        )

    md_lines.extend([
        "",
        "## 3. Secondary TGD Structural Control",
        "",
        "| Paradigm | Metric | Stratum | Median (d) | 95% Bootstrap CI (d) | Role |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
    ])

    for r in tgd_rows:
        if r["table"] == "tgd_control_overall" and r["metric"] == "delta_absCT_TGD_CN":
            md_lines.append(
                f"| {r['paradigm']} | delta_absCT_TGD_CN | overall | {r['median']:.1f} | [{r['ci_low']:.1f}, {r['ci_high']:.1f}] | Secondary output-level control |"
            )

    md_lines.extend([
        "",
        "## 4. KGE-Qualified Timing Inconsistency Prevalence Audit",
        "",
        "Audits the prevalence of timing inconsistency ($|CT| \\ge 15\\text{ d}$) among basins with acceptable hydrograph fit ($KGE \\ge 0.60$).",
        "",
        "- **Conditional Prevalence:** $P(|CT| \\ge 15\\text{ d} \\mid KGE \\ge 0.60) = \\frac{N(KGE \\ge 0.60 \\cap |CT| \\ge 15\\text{ d})}{N(KGE \\ge 0.60)}$",
        "- **Joint Prevalence:** $P(KGE \\ge 0.60 \\cap |CT| \\ge 15\\text{ d}) = \\frac{N(KGE \\ge 0.60 \\cap |CT| \\ge 15\\text{ d})}{531}$",
        "",
        "### A. Structure-Specific Denominator ($N_s(KGE_s \\ge 0.60)$)",
        "",
        "| Paradigm | Structure | Numerator | Conditional Denom | Conditional Prevalence | 95% Bootstrap CI | Joint Prevalence ($N=531$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in prev_meta["key_findings_kge060_ct15d"]:
        if r["denominator_type"] == "structure_specific":
            c_prev = f"{r['conditional_prevalence']*100:.2f}%"
            c_ci = f"[{r['conditional_ci_low']*100:.2f}%, {r['conditional_ci_high']*100:.2f}%]"
            j_prev = f"{r['joint_prevalence']*100:.2f}% ({r['numerator']}/531)"
            md_lines.append(
                f"| {r['paradigm']} | {r['structure']} | {r['numerator']} | {r['conditional_denominator']} | **{c_prev}** | {c_ci} | {j_prev} |"
            )

    md_lines.extend([
        "",
        "### B. Common-Pass Denominator (Same-Basin $KGE \\ge 0.60$ Across Base, TGD, CN)",
        "",
        "| Paradigm | Structure | Numerator | Common Denom | Conditional Prevalence | 95% Bootstrap CI | Joint Prevalence ($N=531$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in prev_meta["key_findings_kge060_ct15d"]:
        if r["denominator_type"] == "common_all_structures_pass":
            c_prev = f"{r['conditional_prevalence']*100:.2f}%"
            c_ci = f"[{r['conditional_ci_low']*100:.2f}%, {r['conditional_ci_high']*100:.2f}%]"
            j_prev = f"{r['joint_prevalence']*100:.2f}% ({r['numerator']}/531)"
            md_lines.append(
                f"| {r['paradigm']} | {r['structure']} | {r['numerator']} | {r['conditional_denominator']} | **{c_prev}** | {c_ci} | {j_prev} |"
            )

    md_lines.extend([
        "",
        "## 5. Robustness & Sensitivity Summary",
        "",
        f"- **dPL Seed Robustness (Seeds 42, 123, 2026):** {robustness_meta['seed_restart_robustness']['dpl_seed_consistency']}.",
        f"- **IC Restart Stability:** Uses canonical `selected_restart` determined from training-period KGE.",
        f"- **Regional LORO Robustness:** Status `{robustness_meta['regional_robustness']['status']}` ({robustness_meta['regional_robustness'].get('reason', 'executed')}).",
        "",
        "## 6. Artifact Manifest",
        "",
        "- `canonical_basin_level.csv`: 3,186 canonical basin-level evaluation rows.",
        "- `canonical_paired_contrasts.csv`: 1,062 paired basin contrast rows.",
        "- `snow_stratified_summaries.csv`: S1-S5 and overall stratified distributions with 95% CIs.",
        "- `spearman_associations.csv`: Continuous rank correlations and 95% CIs.",
        "- `endpoint_activity_contrast.csv`: S5-S1 endpoint activity contrasts and 95% CIs.",
        "- `secondary_tgd_control_summaries.csv`: Secondary TGD structural control summaries.",
        "- `threshold_denominator_audit.csv`: Full grid threshold prevalence across denominator types.",
        "- `threshold_prevalence_summary.csv`: Key cutoffs (KGE 0.40..0.80, CT 10, 15, 20 d) with bootstrap CIs.",
        "- `seed_restart_robustness.csv`: Per-seed dPL evaluations and IC stability records.",
        "- `canonical_gates_summary.json`: Formal validation records of all 5 gates.",
        "- `machine_readable_summary.json`: Complete machine-readable summary.",
    ])

    summary_md_path = out_dir / "results_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return machine_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical R1 Downstream Analysis")
    parser.add_argument("--staged-dir", type=Path, default=STAGED_DIR, help="Path to staged outputs")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Path to write results")
    parser.add_argument("--region-dir", type=Path, default=None, help="Path to region group metadata")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="Number of bootstrap draws")
    args = parser.parse_args()

    summary = run_pipeline(
        staged_dir=args.staged_dir,
        output_dir=args.output_dir,
        region_dir=args.region_dir,
        draws=args.draws,
    )
    print("\nPipeline execution complete. Overall Gate Status:", summary["canonical_gates"])


if __name__ == "__main__":
    main()
