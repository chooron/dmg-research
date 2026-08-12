"""
Unit Hydrograph 6: Gamma Distribution (Nash Cascade).

Corresponds to MATLAB uh_6_gamma(n, k, delta_t).

Uses gammainc for the forward (numerically identical to the MATLAB reference)
and a log-PDF-based backward pass for differentiability (torch.special.gammainc
lacks a gradient implementation).
"""

import torch

from .base import DplUHBase


def _normalize_incremental(weights: torch.Tensor, epsilon: float) -> torch.Tensor:
    weight_sum = weights.sum(dim=-1, keepdim=True)
    denom = torch.where(
        weight_sum > epsilon,
        weight_sum,
        torch.full_like(weight_sum, epsilon),
    )
    return weights / denom


def _tail_mass_redistribution_reference(incremental: torch.Tensor) -> torch.Tensor:
    """Reference MATLAB-aligned redistribution with Python loops."""
    adjusted = torch.zeros_like(incremental)
    max_lag = incremental.shape[-1]
    for batch_index in range(incremental.shape[0]):
        weights = incremental[batch_index, 0]
        peak = weights.new_tensor(0.0)
        cutoff = max_lag
        for lag_index in range(max_lag):
            peak = torch.maximum(peak, weights[lag_index])
            if weights[lag_index] < peak * 0.001:
                cutoff = lag_index + 1
                break
        kept = weights[:cutoff]
        kept_sum = kept.sum()
        if kept_sum > 0:
            kept = kept + kept / kept_sum * (1.0 - kept_sum)
        adjusted[batch_index, 0, :cutoff] = kept
    return adjusted


def _tail_mass_redistribution_vectorized(incremental: torch.Tensor) -> torch.Tensor:
    """Vectorized equivalent of the reference tail-mass redistribution."""
    weights = incremental[:, 0, :]
    batch_size, max_lag = weights.shape

    peak = torch.cummax(weights, dim=-1).values
    trigger = weights < (peak * 0.001)
    any_trigger = trigger.any(dim=-1)
    first_trigger = trigger.to(torch.int64).argmax(dim=-1)
    cutoff = torch.where(
        any_trigger,
        first_trigger + 1,
        torch.full_like(first_trigger, max_lag),
    )

    lag_idx = torch.arange(max_lag, device=weights.device, dtype=cutoff.dtype).view(
        1, -1
    )
    keep_mask = lag_idx < cutoff.unsqueeze(-1)
    kept = weights * keep_mask.to(weights.dtype)
    kept_sum = kept.sum(dim=-1, keepdim=True)
    adjusted_kept = torch.where(
        kept_sum > 0,
        kept + kept / kept_sum * (1.0 - kept_sum),
        kept,
    )

    adjusted = torch.zeros_like(incremental)
    adjusted[:, 0, :] = adjusted_kept
    return adjusted


def _forward_reference(n_raw, k_raw, t_idx, epsilon):
    n = torch.clamp(n_raw, min=0.1, max=20.0)
    k = torch.clamp(k_raw, min=1e-3)

    t_int = torch.arange(
        1, t_idx.shape[-1] + 1, device=n.device, dtype=n.dtype
    ).view(1, 1, -1)
    x_val = t_int / k
    s_curve = torch.special.gammainc(n, x_val)
    zeros = torch.zeros_like(s_curve[..., :1])
    padded = torch.cat([zeros, s_curve], dim=-1)
    incremental = padded[..., 1:] - padded[..., :-1]
    return _tail_mass_redistribution_reference(incremental)


def _forward_vectorized(n_raw, k_raw, t_idx, epsilon):
    n = torch.clamp(n_raw, min=0.1, max=20.0)
    k = torch.clamp(k_raw, min=1e-3)

    t_int = torch.arange(
        1, t_idx.shape[-1] + 1, device=n.device, dtype=n.dtype
    ).view(1, 1, -1)
    x_val = t_int / k
    s_curve = torch.special.gammainc(n, x_val)
    zeros = torch.zeros_like(s_curve[..., :1])
    padded = torch.cat([zeros, s_curve], dim=-1)
    incremental = padded[..., 1:] - padded[..., :-1]
    return _tail_mass_redistribution_vectorized(incremental)


def _forward_pdf_half_step(n_raw, k_raw, t_idx, epsilon):
    n = torch.clamp(n_raw, min=0.1, max=20.0)
    k = torch.clamp(k_raw, min=1e-3)

    t = t_idx.to(device=n.device, dtype=n.dtype)
    log_denom = torch.lgamma(n) + n * torch.log(k)
    log_num = (n - 1.0) * torch.log(t + 1e-10) - (t / k)
    log_w = log_num - log_denom
    weights = torch.exp(log_w)
    return _normalize_incremental(weights, epsilon)


class _GammaWeightsFunction(torch.autograd.Function):
    """Custom autograd for gamma UH weights: gammainc forward, log-PDF backward."""

    @staticmethod
    def forward(ctx, n_raw, k_raw, t_idx, epsilon):
        adjusted = _forward_vectorized(n_raw, k_raw, t_idx, epsilon)
        ctx.save_for_backward(n_raw, k_raw, t_idx)
        return adjusted

    @staticmethod
    def backward(ctx, grad_output):
        n_raw, k_raw, t_idx = ctx.saved_tensors

        # Use log-PDF gradient as an approximate gradient for the gammainc forward
        # This is close numerically (max diff ~2%) and provides useful gradient signals
        n = torch.clamp(n_raw, min=0.1, max=20.0).detach().requires_grad_(True)
        k = torch.clamp(k_raw, min=1e-3).detach().requires_grad_(True)

        t = t_idx
        with torch.enable_grad():
            log_denom = torch.lgamma(n) + n * torch.log(k)
            log_num = (n - 1.0) * torch.log(t.to(n.device) + 1e-10) - (
                t.to(n.device) / k
            )
            log_w = log_num - log_denom
            w = torch.exp(log_w)
            w_sum = w.sum(dim=-1, keepdim=True)
            w_norm = w / (w_sum + 1e-10)
            surrogate = (w_norm * grad_output).sum()
            surrogate.backward()

        return n.grad, k.grad, None, None


class DplGamma6(DplUHBase):
    """Gamma Unit Hydrograph (Nash Cascade).

    Parameters:
        params: (Batch, 2)
            - params[:, 0] -> n (shape / number of Nash reservoirs)
            - params[:, 1] -> k (scale / lag time per reservoir)

    Forward uses torch.special.gammainc (MATLAB-equivalent).
    Backward uses a differentiable log-PDF surrogate for gradient propagation.
    """

    def __init__(self, max_lag, epsilon=1e-6, kernel_mode="cdf_diff"):
        super().__init__(max_lag=max_lag, epsilon=epsilon)
        if kernel_mode not in ("cdf_diff", "pdf_half_step"):
            raise ValueError(f"Unsupported gamma6 kernel_mode: {kernel_mode}")
        self.kernel_mode = kernel_mode

    def get_weights(self, params):
        if params.shape[-1] != 2:
            raise ValueError(
                "DplGamma6 needs 2 params (n, k), got shape {}".format(params.shape)
            )

        n = params[:, 0:1].unsqueeze(-1)
        k = params[:, 1:2].unsqueeze(-1)

        if self.kernel_mode == "pdf_half_step":
            return _forward_pdf_half_step(n, k, self.t_idx, self.epsilon)

        return _GammaWeightsFunction.apply(n, k, self.t_idx, self.epsilon)
