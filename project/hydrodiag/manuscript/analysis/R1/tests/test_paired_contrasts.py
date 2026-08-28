"""Tests for same-basin paired contrasts and alignment checks."""
import sys
from pathlib import Path

import pytest

R1_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R1_DIR))

from paired_contrasts import compute_paired_contrasts
from config import PARADIGMS, TOTAL_BASINS


def test_paired_alignment_and_sign_conventions():
    contrasts, audit = compute_paired_contrasts()

    assert len(contrasts) == TOTAL_BASINS * len(PARADIGMS)
    assert audit["status"] == "PASS"

    for p in PARADIGMS:
        p_rows = [r for r in contrasts if r["paradigm"] == p]
        assert len(p_rows) == TOTAL_BASINS

    # Check sign convention for first 20 rows:
    # delta_kge_base_cn = KGE_CN - KGE_Base
    # delta_abs_ct_base_cn = abs(signed_e_Base) - abs(signed_e_CN)
    for r in contrasts[:20]:
        kge_diff = r["KGE_CN"] - r["KGE_Base"]
        assert abs(r["delta_KGE_Base_CN"] - kge_diff) < 1e-12

        abs_ct_diff = abs(r["signed_e_Base"]) - abs(r["signed_e_CN"])
        assert abs(r["delta_absCT_Base_CN"] - abs_ct_diff) < 1e-12
