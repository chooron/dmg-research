"""Regression tests for the Penman ``truncate:90`` dead-config cleanup (2026-08-31).

Contract enforced here:
1. ``warmup_grad_mode`` values other than ``"detach"`` MUST fail fast (they were
   declared historically but never implemented; silent no-ops are prohibited).
2. Penman keeps its W2-evidence-based warmup-LENGTH exception (365d warmup /
   365d scored) but uses the SAME gradient semantics as all other models.
3. The cleanup does not change numerical forward/backward behavior: the post-cleanup
   ``"detach"`` forward + gradient must be bit-identical to the pre-cleanup baseline
   captured while the dead ``"truncate:90"`` label was still accepted.

See project/benchmark/PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md and
project/benchmark/PENMAN_TRUNCATE90_CORRECTION_NOTE_20260831.md.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only tests

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
CLEANUP_RESULTS = BENCHMARK_ROOT / "results" / "penman_truncate_cleanup_20260831"
BASELINE_NPZ = CLEANUP_RESULTS / "pre_cleanup_detach.npz"
BASELINE_TRUNCATE90_NPZ = CLEANUP_RESULTS / "pre_cleanup_truncate90.npz"

from src.model_registry import NPARAM_INFO_36, build_model, model_config  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Fail-fast on declared-but-never-implemented modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_mode", ["truncate:90", "truncate:180", "truncate:30", "full", "state_init"])
def test_model_config_rejects_unimplemented_warmup_modes(bad_mode: str):
    with pytest.raises(ValueError, match="Unsupported warmup_grad_mode"):
        model_config("penman", warm_up=365, parameter_mapping="auto", warmup_grad_mode=bad_mode)


@pytest.mark.parametrize("bad_mode", ["truncate:90", "full"])
def test_build_model_rejects_unimplemented_warmup_modes(bad_mode: str):
    with pytest.raises(ValueError, match="Unsupported warmup_grad_mode"):
        build_model("penman", torch.device("cpu"), warm_up=365,
                    backend="eager", parameter_mapping="auto", warmup_grad_mode=bad_mode)


def test_detach_is_the_only_accepted_mode():
    cfg = model_config("penman", warm_up=365, parameter_mapping="auto", warmup_grad_mode="detach")
    assert cfg["warmup_grad_mode"] == "detach"
    # default must also be detach
    assert model_config("penman")["warmup_grad_mode"] == "detach"


# ---------------------------------------------------------------------------
# 2. Penman canonical contract: warmup-LENGTH exception only
# ---------------------------------------------------------------------------

def test_penman_exception_is_warmup_length_only():
    """Penman: 365d warmup + 365d scored; identical gradient semantics (detach/full backprop)."""
    cfg = model_config("penman", warm_up=365, parameter_mapping="auto", warmup_grad_mode="detach")
    assert cfg["warm_up"] == 365
    assert cfg["warmup_grad_mode"] == "detach"
    # 36-model registry sanity: penman is a 4-parameter / 3-state model
    assert NPARAM_INFO_36["penman"] == 4


# ---------------------------------------------------------------------------
# 3. Numerical equivalence: pre-cleanup baseline vs post-cleanup behavior
# ---------------------------------------------------------------------------

def _run_penman_forward_grad(mode: str, baseline_npz: Path) -> dict:
    """Re-run the exact baseline computation (4 basins, FP64, CPU, 730d window)."""
    from src.data_selection import load_ids
    from run_dpl_benchmark_dmg_native import load_camels_time_series, compute_differentiable_kge

    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    WARMUP, WINDOW = 365, 730
    ids = [int(i) for i in load_ids("data/531sub_id.txt")][:4]
    train_x_np, train_y_np, *_ = load_camels_time_series(ids)
    x = torch.as_tensor(train_x_np, dtype=torch.float64, device=device)
    y = torch.as_tensor(train_y_np, dtype=torch.float64, device=device)
    start = 500
    xw, yw = x[start:start + WINDOW], y[start:start + WINDOW]
    theta = torch.full((len(ids), NPARAM_INFO_36["penman"]), 0.5, device=device, requires_grad=True)

    model = build_model("penman", device, warm_up=WARMUP, backend="eager",
                        parameter_mapping="auto", warmup_grad_mode=mode)
    q = model({"x_phy": xw}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
    loss, _ = compute_differentiable_kge(q, yw[WARMUP:], warmup_days=0, eps=0.1)
    loss.backward()
    g = theta.grad.detach().clone()
    return {
        "loss": float(loss.item()),
        "q_sha256": hashlib.sha256(q.detach().numpy().tobytes()).hexdigest(),
        "grad_sha256": hashlib.sha256(g.numpy().tobytes()).hexdigest(),
    }


def _baseline_digests():
    """Extract the stored pre-cleanup digests (loss, q, grad)."""
    data = np.load(BASELINE_NPZ)
    loss = float(data["loss"])
    q_sha = hashlib.sha256(data["q"].tobytes()).hexdigest()
    g_sha = hashlib.sha256(data["grad"].tobytes()).hexdigest()
    return loss, q_sha, g_sha


def _pre_cleanup_labels_were_bit_identical():
    """Sanity: the pre-cleanup 'truncate:90' and 'detach' runs were bit-identical."""
    a = np.load(BASELINE_TRUNCATE90_NPZ)
    b = np.load(BASELINE_NPZ)
    return (a["q"].tobytes() == b["q"].tobytes()
            and a["grad"].tobytes() == b["grad"].tobytes()
            and float(a["loss"]) == float(b["loss"]))


@pytest.mark.skipif(not BASELINE_NPZ.exists() or not BASELINE_TRUNCATE90_NPZ.exists(),
                    reason="cleanup baseline artifacts not present")
def test_pre_cleanup_truncate90_label_was_bit_identical_to_detach():
    assert _pre_cleanup_labels_were_bit_identical()


@pytest.mark.skipif(not BASELINE_NPZ.exists(), reason="cleanup baseline artifact not present")
def test_post_cleanup_detach_is_bit_identical_to_pre_cleanup_baseline():
    """Cleanup must NOT change numerical behavior: forward + gradient bit-identical."""
    loss, q_sha, g_sha = _baseline_digests()
    result = _run_penman_forward_grad("detach", BASELINE_NPZ)
    assert result["loss"] == loss
    assert result["q_sha256"] == q_sha
    assert result["grad_sha256"] == g_sha