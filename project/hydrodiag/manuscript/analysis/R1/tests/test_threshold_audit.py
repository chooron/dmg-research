"""Tests for threshold prevalence across conditional and joint definitions and denominators."""
import sys
from pathlib import Path

import pytest
import torch

R1_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R1_DIR))

from threshold_prevalence_audit import audit_threshold_prevalence
from config import TOTAL_BASINS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_threshold_prevalence_conditional_and_joint():
    audit_rows, key_rows, meta = audit_threshold_prevalence(draws=500)

    assert meta["status"] == "PASS"
    assert len(audit_rows) > 0
    assert len(key_rows) > 0

    for r in audit_rows:
        num = r["numerator"]
        c_den = r["conditional_denominator"]
        a_den = r["all_valid_denominator"]

        # Invariant 1: numerator <= conditional_denominator <= all_valid_denominator
        assert num <= c_den <= a_den
        assert a_den == TOTAL_BASINS

        # Invariant 2: prevalence ranges
        if c_den > 0:
            assert 0.0 <= r["conditional_prevalence"] <= 1.0
            assert abs(r["conditional_prevalence"] - num / c_den) < 1e-12
        if a_den > 0:
            assert 0.0 <= r["joint_prevalence"] <= 1.0
            assert abs(r["joint_prevalence"] - num / a_den) < 1e-12

    # Invariant 3: Key combinations at KGE >= 0.60 & |CT| >= 15 d
    kge60_ct15 = {
        (r["paradigm"], r["structure"], r["denominator_type"]): r
        for r in meta["key_findings_kge060_ct15d"]
    }

    # Structure-specific IC Base: 56/331 (cond) vs 56/531 (joint)
    ic_base_struct = kge60_ct15[("IC-CMA-ES", "Base", "structure_specific")]
    assert ic_base_struct["numerator"] == 56
    assert ic_base_struct["conditional_denominator"] == 331
    assert abs(ic_base_struct["conditional_prevalence"] - 56 / 331) < 1e-6
    assert abs(ic_base_struct["joint_prevalence"] - 56 / 531) < 1e-6
    assert ic_base_struct["conditional_ci_low"] <= ic_base_struct["conditional_prevalence"] <= ic_base_struct["conditional_ci_high"]

    # Structure-specific dPL Base: 46/344 (cond) vs 46/531 (joint)
    dpl_base_struct = kge60_ct15[("dPL-MLP", "Base", "structure_specific")]
    assert dpl_base_struct["numerator"] == 46
    assert dpl_base_struct["conditional_denominator"] == 344
    assert abs(dpl_base_struct["conditional_prevalence"] - 46 / 344) < 1e-6
    assert abs(dpl_base_struct["joint_prevalence"] - 46 / 531) < 1e-6
    assert dpl_base_struct["conditional_ci_low"] <= dpl_base_struct["conditional_prevalence"] <= dpl_base_struct["conditional_ci_high"]

    # Common-pass IC (N_common = 321)
    for s in ["Base", "TGD", "CN"]:
        ic_com = kge60_ct15[("IC-CMA-ES", s, "common_all_structures_pass")]
        assert ic_com["conditional_denominator"] == 321
        assert ic_com["conditional_ci_low"] <= ic_com["conditional_prevalence"] <= ic_com["conditional_ci_high"]

    # Common-pass dPL (N_common = 331)
    for s in ["Base", "TGD", "CN"]:
        dpl_com = kge60_ct15[("dPL-MLP", s, "common_all_structures_pass")]
        assert dpl_com["conditional_denominator"] == 331
        assert dpl_com["conditional_ci_low"] <= dpl_com["conditional_prevalence"] <= dpl_com["conditional_ci_high"]
