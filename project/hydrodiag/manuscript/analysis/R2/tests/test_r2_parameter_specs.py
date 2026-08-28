"""Tests for authoritative 15 shared parameter specifications."""
import sys
from pathlib import Path

import numpy as np
import pytest

R2_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R2_DIR))

from shared_parameter_specs import (
    PARAMETER_METADATA,
    SHARED_15_PARAMETERS,
    STRUCTURE_PARAM_LAYOUTS,
    get_lowers_and_uppers,
    normalize_parameters,
    physical_from_normalized,
)


def test_15_shared_parameters_count_and_bounds():
    assert len(SHARED_15_PARAMETERS) == 15
    lowers, uppers = get_lowers_and_uppers()
    assert len(lowers) == 15
    assert len(uppers) == 15
    assert (uppers > lowers).all()
    assert np.isfinite(lowers).all()
    assert np.isfinite(uppers).all()


def test_normalization_round_trip():
    lowers, uppers = get_lowers_and_uppers()
    midpoints = (lowers + uppers) / 2.0
    norm_mid = normalize_parameters(midpoints)
    assert np.allclose(norm_mid, 0.5)

    phys_recon = physical_from_normalized(norm_mid)
    assert np.allclose(phys_recon, midpoints)


def test_structure_layouts():
    assert STRUCTURE_PARAM_LAYOUTS["Base"]["total_params"] == 15
    assert STRUCTURE_PARAM_LAYOUTS["CN"]["total_params"] == 17
    assert STRUCTURE_PARAM_LAYOUTS["TGD"]["total_params"] == 17

    for p in SHARED_15_PARAMETERS:
        assert p in STRUCTURE_PARAM_LAYOUTS["Base"]["shared_indices"]
        assert p in STRUCTURE_PARAM_LAYOUTS["CN"]["shared_indices"]
        assert p in STRUCTURE_PARAM_LAYOUTS["TGD"]["shared_indices"]
        assert STRUCTURE_PARAM_LAYOUTS["CN"]["shared_indices"][p] == STRUCTURE_PARAM_LAYOUTS["Base"]["shared_indices"][p] + 2
        assert STRUCTURE_PARAM_LAYOUTS["TGD"]["shared_indices"][p] == STRUCTURE_PARAM_LAYOUTS["Base"]["shared_indices"][p] + 2
