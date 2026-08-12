import torch
import torch.nn.functional as F

EPS = 1e-6


def _positive_tau(tau):
    """Ensure tau is strictly positive to avoid division by zero."""
    if torch.is_tensor(tau):
        return torch.clamp(tau, min=EPS)
    return max(float(tau), EPS)


def softplus_t(x, tau=1.0):
    """Numerically stable smooth ReLU: tau * softplus(x / tau)."""
    tau = _positive_tau(tau)
    return tau * F.softplus(x / tau)


def smoothmin_t(x, threshold=1.0, tau=1.0):
    """Numerically stable smooth min(x, threshold).

    Returns: threshold - tau * softplus((threshold - x) / tau),
    clamped to [0, threshold].
    """
    tau = _positive_tau(tau)
    y = threshold - tau * F.softplus((threshold - x) / tau)
    return torch.clamp(y, min=0.0, max=float(threshold))


def triangular_weights(maxbas):
    """Generate triangular weights of length maxbas for MAXBAS routing.

    NOTE: maxbas must be a Python int. It is NOT a continuous differentiable
    MAXBAS parameter. If a tensor is passed it will be converted to int.

    Example: maxbas=3 -> [1,2,1]/4, maxbas=4 -> [1,2,2,1]/6.

    Raises:
        ValueError: if maxbas is a tensor with numel() != 1.
    """
    if torch.is_tensor(maxbas):
        if maxbas.numel() != 1:
            raise ValueError("maxbas must be a scalar integer for triangular_weights.")
        maxbas = int(maxbas.detach().cpu().item())
    maxbas = max(int(round(maxbas)), 1)

    half = (maxbas + 1) // 2
    w = torch.arange(1, half + 1, dtype=torch.float32)
    if maxbas % 2 == 0:
        w = torch.cat([w, w.flip(0)])
    else:
        w = torch.cat([w, w[:-1].flip(0)])
    return w / w.sum()


def gamma_weights(a, b, length):
    """Generate gamma-distribution weights, safe for tensors and scalars.

    w_i = i^(a-1) * exp(-i/b), normalised to sum to 1.

    Args:
        a: shape parameter (>0), scalar or 0-d tensor.
        b: scale parameter (>0), scalar or 0-d tensor.
        length: number of weights (int >= 1).

    Returns:
        1-D tensor of normalised weights on the same device as a (or CPU).
    """
    device = a.device if torch.is_tensor(a) else None
    dtype = a.dtype if torch.is_tensor(a) else torch.float32
    a = torch.as_tensor(a, device=device, dtype=dtype).clamp(min=EPS)
    b = torch.as_tensor(b, device=a.device, dtype=dtype).clamp(min=EPS)
    i = torch.arange(1, length + 1, device=a.device, dtype=dtype)

    w = i ** (a - 1.0) * torch.exp(-i / b)
    denom = w.sum().clamp(min=EPS)
    return w / denom


def causal_conv1d(x, weight):
    """Apply causal 1D convolution with given weight kernel.

    Args:
        x: (..., time) tensor.
        weight: 1D kernel tensor, length = kernel_size.

    Returns:
        Convolved tensor with same shape as x.
    """
    weight = weight.to(device=x.device, dtype=x.dtype)
    original_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(1)

    kernel_size = weight.numel()
    w = weight.flip(0).unsqueeze(0).unsqueeze(0)
    x_padded = F.pad(x, (kernel_size - 1, 0))
    y = F.conv1d(x_padded, w)

    if len(original_shape) == 1:
        y = y.squeeze(0).squeeze(0)
    elif len(original_shape) == 2:
        y = y.squeeze(1)
    return y
