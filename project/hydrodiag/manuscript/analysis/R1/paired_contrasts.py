"""Primary and secondary paired contrasts with strict same-basin alignment checks.

Estimand sign conventions (positive = CN improves relative to baseline):
  - delta_kge_base_cn = KGE_CN - KGE_Base
  - delta_abs_ct_base_cn = abs(signed_e_Base) - abs(signed_e_CN)
  - delta_abs_ct_tgd_cn = abs(signed_e_TGD) - abs(signed_e_CN)
  - delta_kge_tgd_cn = KGE_CN - KGE_TGD
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import EVAL_PERIOD, PARADIGMS, RESULTS_DIR, TOTAL_BASINS
from canonical_basin_table import build_canonical_basin_table


def compute_paired_contrasts(
    canonical_test_rows: List[Dict[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute paired contrasts per basin across paradigms with strict alignment verification.

    Returns:
        (contrast_rows, alignment_audit)
    """
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if canonical_test_rows is None:
        canonical_test_rows, _, _ = build_canonical_basin_table(output_dir=out_dir)

    # Index by (basin_id, paradigm, structure)
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in canonical_test_rows:
        if r["period"] != EVAL_PERIOD:
            continue
        b_id = str(r["basin_id"]).zfill(8)
        key = (b_id, r["paradigm"], r["structure"])
        if key in by_key:
            raise RuntimeError(f"Duplicate basin x paradigm x structure key: {key}")
        by_key[key] = r

    unique_basins = sorted({r["basin_id"] for r in canonical_test_rows})
    if len(unique_basins) != TOTAL_BASINS:
        raise RuntimeError(f"Expected {TOTAL_BASINS} unique basins, got {len(unique_basins)}")

    contrast_rows: List[Dict[str, Any]] = []
    alignment_issues: List[str] = []

    for paradigm in PARADIGMS:
        paradigm_basins = 0
        for b_id in unique_basins:
            k_base = (b_id, paradigm, "Base")
            k_tgd = (b_id, paradigm, "TGD")
            k_cn = (b_id, paradigm, "CN")

            if k_base not in by_key or k_tgd not in by_key or k_cn not in by_key:
                missing = [s for s, k in [("Base", k_base), ("TGD", k_tgd), ("CN", k_cn)] if k not in by_key]
                alignment_issues.append(f"Basin {b_id} in paradigm {paradigm} missing structures: {missing}")
                continue

            r_base = by_key[k_base]
            r_tgd = by_key[k_tgd]
            r_cn = by_key[k_cn]

            # Verify consistent snow metadata across structures
            snow_b = (r_base["frac_snow"], r_base["snow_stratum"])
            snow_t = (r_tgd["frac_snow"], r_tgd["snow_stratum"])
            snow_c = (r_cn["frac_snow"], r_cn["snow_stratum"])
            if not (snow_b == snow_t == snow_c):
                alignment_issues.append(f"Inconsistent snow metadata for basin {b_id} in {paradigm}: {snow_b} vs {snow_t} vs {snow_c}")

            kge_base = float(r_base["KGE"])
            kge_tgd = float(r_tgd["KGE"])
            kge_cn = float(r_cn["KGE"])

            signed_base = float(r_base["signed_CT_error"])
            signed_tgd = float(r_tgd["signed_CT_error"])
            signed_cn = float(r_cn["signed_CT_error"])

            abs_base = abs(signed_base) if signed_base == signed_base else float("nan")
            abs_tgd = abs(signed_tgd) if signed_tgd == signed_tgd else float("nan")
            abs_cn = abs(signed_cn) if signed_cn == signed_cn else float("nan")

            # Estimands: positive = CN improves
            delta_kge_base_cn = kge_cn - kge_base
            delta_abs_ct_base_cn = abs_base - abs_cn
            delta_abs_ct_tgd_cn = abs_tgd - abs_cn
            delta_kge_tgd_cn = kge_cn - kge_tgd

            contrast_rows.append({
                "basin_id": b_id,
                "regime": paradigm,
                "paradigm": paradigm,
                "period": EVAL_PERIOD,
                "frac_snow": r_base["frac_snow"],
                "snow_stratum": r_base["snow_stratum"],
                "delta_KGE_Base_CN": delta_kge_base_cn,
                "delta_absCT_Base_CN": delta_abs_ct_base_cn,
                "delta_absCT_TGD_CN": delta_abs_ct_tgd_cn,
                "delta_KGE_TGD_CN": delta_kge_tgd_cn,
                "KGE_Base": kge_base,
                "KGE_TGD": kge_tgd,
                "KGE_CN": kge_cn,
                "signed_e_Base": signed_base,
                "signed_e_TGD": signed_tgd,
                "signed_e_CN": signed_cn,
                "abs_e_Base": abs_base,
                "abs_e_TGD": abs_tgd,
                "abs_e_CN": abs_cn,
                "valid_year_count_Base": r_base["valid_year_count"],
                "valid_year_count_TGD": r_tgd["valid_year_count"],
                "valid_year_count_CN": r_cn["valid_year_count"],
            })
            paradigm_basins += 1

        if paradigm_basins != TOTAL_BASINS:
            alignment_issues.append(f"Paradigm {paradigm} has {paradigm_basins} paired basins != expected {TOTAL_BASINS}")

    if alignment_issues:
        raise RuntimeError("Basin alignment check FAILED:\n" + "\n".join(alignment_issues))

    if len(contrast_rows) != TOTAL_BASINS * len(PARADIGMS):
        raise RuntimeError(f"Expected {TOTAL_BASINS * len(PARADIGMS)} paired rows, got {len(contrast_rows)}")

    # Write output files
    fields = [
        "basin_id", "regime", "paradigm", "period", "frac_snow", "snow_stratum",
        "delta_KGE_Base_CN", "delta_absCT_Base_CN", "delta_absCT_TGD_CN", "delta_KGE_TGD_CN",
        "KGE_Base", "KGE_CN", "KGE_TGD",
        "signed_e_Base", "signed_e_TGD", "signed_e_CN",
        "abs_e_Base", "abs_e_TGD", "abs_e_CN",
        "valid_year_count_Base", "valid_year_count_TGD", "valid_year_count_CN",
    ]

    paired_csv_path = out_dir / "canonical_paired_contrasts.csv"
    with paired_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in contrast_rows:
            writer.writerow(r)

    # Also write complete_basin_distributions.csv for backward-compatible consumers
    compat_csv_path = out_dir / "complete_basin_distributions.csv"
    with compat_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in contrast_rows:
            writer.writerow(r)

    audit_summary = {
        "status": "PASS",
        "total_paired_basins_per_paradigm": {p: sum(1 for r in contrast_rows if r["paradigm"] == p) for p in PARADIGMS},
        "total_contrast_rows": len(contrast_rows),
        "silent_drop_check": "PASS (0 basins dropped, 0 duplicate keys)",
        "sign_convention": {
            "delta_KGE_Base_CN": "KGE_CN - KGE_Base (positive = CN improves)",
            "delta_absCT_Base_CN": "abs(signed_e_Base) - abs(signed_e_CN) (positive = CN improves)",
            "delta_absCT_TGD_CN": "abs(signed_e_TGD) - abs(signed_e_CN) (positive = CN improves)",
        },
        "output_path": str(paired_csv_path),
    }

    with (out_dir / "basin_alignment_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return contrast_rows, audit_summary


if __name__ == "__main__":
    contrasts, audit = compute_paired_contrasts()
    print(f"Paired contrasts computed successfully: {len(contrasts)} rows.")
    print("Alignment audit:", audit["total_paired_basins_per_paradigm"])
