import os

import numpy as np
import pytest

from project.autofuse.fidelity import synthetic_forcing
from project.autofuse.reference_oracle import run_reference


@pytest.mark.skipif(not os.environ.get("FUSE_REFERENCE_EXE"), reason="reference executable not configured")
def test_reference_oracle_runs_without_embedded_reference_values():
    result = run_reference(os.environ["FUSE_REFERENCE_EXE"], 210, synthetic_forcing(8))
    assert result.model_id == 210
    assert result.q_routed.shape == (8,)
    assert np.isfinite(result.q_routed).all()
    assert result.metadata["source_commit"] == "e6e23a4fc4ff4019bcab55f14537ea43b9525967"
    assert result.initial_state["WATR_2"] > 0.0


def test_reference_oracle_rejects_unrepresentable_initial_fraction():
    with pytest.raises(ValueError, match="fracState0=0.25"):
        run_reference("/does/not/run", 210, synthetic_forcing(4), initial_fraction=0.5)
