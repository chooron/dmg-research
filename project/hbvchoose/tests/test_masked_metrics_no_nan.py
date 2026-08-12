"""Test: Masked metrics produce no NaN when target has NaN values."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F


def nse_safe(qsim, qobs):
    qsim = np.asarray(qsim, dtype=np.float64)
    qobs = np.asarray(qobs, dtype=np.float64)
    mask = ~np.isnan(qobs) & ~np.isnan(qsim)
    qs, qo = qsim[mask], qobs[mask]
    if len(qo) < 2:
        return float("nan")
    num = ((qs - qo) ** 2).sum()
    den = ((qo - qo.mean()) ** 2).sum()
    if den < 1e-12:
        return float("nan")
    return float(1.0 - num / den)


def masked_mse_loss(qsim, qobs):
    mask = ~(torch.isnan(qsim) | torch.isnan(qobs) | torch.isinf(qsim) | torch.isinf(qobs))
    if mask.sum() < 2:
        return torch.tensor(float("nan"), device=qsim.device)
    return F.mse_loss(qsim[mask], qobs[mask])


class TestMaskedMetrics:

    def test_nse_returns_nan_for_all_nan_target(self):
        qsim = np.array([1.0, 2.0, 3.0])
        qobs = np.array([np.nan, np.nan, np.nan])
        result = nse_safe(qsim, qobs)
        assert np.isnan(result), f"NSE should be NaN for all-NaN target, got {result}"

    def test_nse_works_with_partial_nan(self):
        qsim = np.array([1.0, 2.0, 3.0, 4.0])
        qobs = np.array([1.1, np.nan, 3.2, np.nan])
        result = nse_safe(qsim, qobs)
        assert not np.isnan(result), f"NSE should be finite with partial NaN, got {result}"

    def test_nse_handles_nan_in_qsim_too(self):
        qsim = np.array([1.0, np.nan, 3.0, 4.0])
        qobs = np.array([1.1, 2.0, np.nan, 3.8])
        result = nse_safe(qsim, qobs)
        assert not np.isnan(result), f"NSE should be finite, got {result}"

    def test_masked_mse_returns_nan_when_insufficient_valid(self):
        qsim = torch.tensor([np.nan, np.nan])
        qobs = torch.tensor([1.0, np.nan])
        result = masked_mse_loss(qsim, qobs)
        # With < 2 valid pairwise elements, returns NaN
        assert torch.isnan(result), f"Expected NaN when mask sum < 2"

    def test_masked_mse_ignores_nan_positions(self):
        qsim = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        qobs = torch.tensor([1.1, np.nan, 3.2, np.nan, 5.3])
        result = masked_mse_loss(qsim, qobs)
        assert not torch.isnan(result), f"Expected finite loss, got {result}"
        # Only positions 0, 2, 4 contribute
        expected = F.mse_loss(
            torch.tensor([1.0, 3.0, 5.0]),
            torch.tensor([1.1, 3.2, 5.3])
        )
        assert torch.allclose(result, expected, atol=1e-6)

    def test_mask_rule_consistent_loss_vs_metrics(self):
        """Loss and metrics must use same NaN mask rule."""
        qsim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        qobs = np.array([1.1, np.nan, 3.2, np.nan, 5.3])

        # Loss mask
        qsim_t = torch.from_numpy(qsim)
        qobs_t = torch.from_numpy(qobs)
        loss_mask = ~(torch.isnan(qsim_t) | torch.isnan(qobs_t) | torch.isinf(qsim_t) | torch.isinf(qobs_t))
        n_loss = int(loss_mask.sum().item())

        # Metric mask
        metric_mask = ~np.isnan(qobs) & ~np.isnan(qsim)
        n_metric = int(metric_mask.sum())

        assert n_loss == n_metric, f"Inconsistent masks: loss={n_loss}, metric={n_metric}"
