"""Subsurface runoff response-path organization ladder for XAJ.

The kernels here start *after* XAJ runoff generation and free-water
separation.  The host must pass one total non-surface input ``R_ss``; no
interflow/groundwater identities or split parameter is present in these
kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .parameter_specs import (
    CONTROLLED_XAJ_CG_LOWER,
    CONTROLLED_XAJ_CG_UPPER,
    CONTROLLED_XAJ_CI_LOWER,
    CONTROLLED_XAJ_CI_UPPER,
    NATIVE_XAJ_LATENT_Z0,
    SUBSURFACE_BETA_PARAM_SPECS,
    SUBSURFACE_TAU0_PARAM_SPECS,
    XAJ_KSS_PARAM_SPEC,
)
from .structure_utils import log_map_normalized, stable_positive_power

RESPONSE_POWER_FLOOR = 1e-6
# Derived from pre-Phase-0 native-XAJ latent-storage scaling; fixed, not fitted.
DEFAULT_Z0 = NATIVE_XAJ_LATENT_Z0
RESPONSE_DT = 1.0
BETA_ONE_TOL = 1e-3
LOG_Y_FLOOR = 1e-30


def normalized_to_controlled_ci(normalized: torch.Tensor) -> torch.Tensor:
    return normalized * normalized.new_tensor(
        CONTROLLED_XAJ_CI_UPPER - CONTROLLED_XAJ_CI_LOWER
    ) + normalized.new_tensor(CONTROLLED_XAJ_CI_LOWER)


def normalized_to_controlled_cg(normalized: torch.Tensor) -> torch.Tensor:
    return normalized * normalized.new_tensor(
        CONTROLLED_XAJ_CG_UPPER - CONTROLLED_XAJ_CG_LOWER
    ) + normalized.new_tensor(CONTROLLED_XAJ_CG_LOWER)


def normalized_to_tau0(normalized: torch.Tensor) -> torch.Tensor:
    """Map normalized coordinates to the D_R/G_R response time scale."""
    spec = SUBSURFACE_TAU0_PARAM_SPECS["tau_0"]
    return log_map_normalized(normalized, spec["lower"], spec["upper"])


def normalized_to_kss(normalized: torch.Tensor) -> torch.Tensor:
    """Map normalized coordinates to native-effective total KSS linearly."""
    spec = XAJ_KSS_PARAM_SPEC["xaj_kss"]
    return normalized * normalized.new_tensor(
        spec["upper"] - spec["lower"]
    ) + normalized.new_tensor(spec["lower"])


def normalized_to_beta(normalized: torch.Tensor) -> torch.Tensor:
    """Map normalized coordinates to beta in log space.

    The interval is an implementation choice for numerical/structural smoke
    tests and remains explicitly provisional until Phase 0 freezes it.
    """
    spec = SUBSURFACE_BETA_PARAM_SPECS["beta"]
    return log_map_normalized(normalized, spec["lower"], spec["upper"])


def native_linear_tau(c: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Return the exact discrete-step tau corresponding to native C."""
    return c.new_tensor(dt) / (-torch.log(c))


def native_linear_storage(q_previous: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Latent storage equivalent to native ``Q_t=C Q_prev+(1-C)R_t``."""
    return c / (1.0 - c) * q_previous


def native_linear_step_from_storage(
    z_available: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Q_t, Z_t)`` for the native-equivalent latent update."""
    z_new = c * z_available
    q_new = (1.0 - c) * z_available
    return q_new, z_new


def native_effective_kss(ki: torch.Tensor, kg: torch.Tensor) -> torch.Tensor:
    """Reproduce XAJ's effective ``KI+KG`` boundary mapping exactly."""
    total = ki + kg
    safe_total = torch.clamp(total, min=1e-6)
    scale = torch.where(
        total < 1.0,
        torch.ones_like(total),
        (1.0 - 1e-5) / safe_total,
    )
    return total * scale


def response_conditioning_tensors(
    z_available: torch.Tensor,
    z_new: torch.Tensor,
    z0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return extinction mask, positive-storage mask and log(Za/Z0)."""
    positive = z_available > 0.0
    extinct = positive & (z_new == 0.0)
    ratio = torch.clamp(z_available / torch.clamp(z0, min=1e-30), min=1e-30)
    log_ratio = torch.where(positive, torch.log(ratio), torch.zeros_like(ratio))
    return extinct, positive, log_ratio


def summarize_response_conditioning(
    z_available: torch.Tensor,
    z_new: torch.Tensor,
    z0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Summarize G_R extinction and log-storage conditioning diagnostics."""
    extinct, positive, log_ratio = response_conditioning_tensors(z_available, z_new, z0)
    values = log_ratio[positive]
    if values.numel() == 0:
        zero = z_available.new_zeros(())
        return {
            "extinction_count": zero,
            "positive_available_count": zero,
            "f_extinct": zero,
            "log_z_ratio_mean": zero,
            "log_z_ratio_std": zero,
            "log_z_ratio_median": zero,
            "log_z_ratio_iqr": zero,
            "log_z_ratio_p05": zero,
            "log_z_ratio_p95": zero,
        }
    extinction_count = extinct.sum()
    positive_count = positive.sum()
    return {
        "extinction_count": extinction_count,
        "positive_available_count": positive_count,
        "f_extinct": extinction_count.to(z_available.dtype)
        / positive_count.to(z_available.dtype),
        "log_z_ratio_mean": values.mean(),
        "log_z_ratio_std": values.std(unbiased=False),
        "log_z_ratio_median": values.median(),
        "log_z_ratio_iqr": torch.quantile(values, 0.75) - torch.quantile(values, 0.25),
        "log_z_ratio_p05": torch.quantile(values, 0.05),
        "log_z_ratio_p95": torch.quantile(values, 0.95),
    }


def xaj_subsurface_input(
    ri: torch.Tensor,
    rg: torch.Tensor,
    one_minus_im: torch.Tensor,
) -> torch.Tensor:
    """Convert native XAJ's two generated slow inputs to its response input.

    In the active XAJ implementation ``RI`` and ``RG`` are generated before
    the impervious-area adjustment and each is multiplied by ``1-IM`` before
    entering its native response recursion.  This helper preserves that
    convention while producing the single input required by D_R/G_R.
    """
    return (ri + rg) * one_minus_im


def _analytic_subsurface_response_step(
    r_ss: torch.Tensor,
    z: torch.Tensor,
    tau_0: torch.Tensor,
    beta: torch.Tensor,
    z0: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Analytic store-first daily D_R/G_R update.

    With ``y = Z/Z0`` and ``c = dt/tau_0`` the beta != 1 solution is

        log(y_new) = log(y_a**(1-beta) - (1-beta)c) / (1-beta).

    The implementation evaluates the bracket with ``expm1``.  For
    ``abs(1-beta) <= BETA_ONE_TOL`` it uses the second-order expansion

        A + d/2 * (L**2 - A**2)
          + d**2 * (L**3/6 - A*L**2/2 + A**3/3),

    where ``d=1-beta``, ``L=log(y_a)`` and ``A=L-c``.  This tends to
    ``L-c`` continuously at beta=1 and avoids division by a small d.
    ``LOG_Y_FLOOR`` is only for the logarithm; the explicit zero mask makes
    zero storage produce exactly zero outflow and state.
    """
    input_available = torch.clamp(r_ss, min=0.0)
    z_available = torch.clamp(z, min=0.0) + input_available
    tau_safe = torch.clamp(tau_0, min=nearzero)
    z0_safe = torch.clamp(z0, min=nearzero)
    y_available = z_available / z0_safe
    y_safe = torch.clamp(y_available, min=LOG_Y_FLOOR)
    log_y = torch.log(y_safe)

    dt = z_available.new_tensor(RESPONSE_DT)
    c = dt / tau_safe
    delta = 1.0 - beta
    delta_near_zero = torch.abs(delta) <= BETA_ONE_TOL
    delta_safe = torch.where(delta_near_zero, torch.ones_like(delta), delta)

    # expm1(delta*log_y) is stable when beta is close to one.  Clamp the
    # bracket before log so an extinct branch never evaluates log(nonpositive)
    # in an unselected tensor branch.
    delta_log_y = delta * log_y
    bracket = 1.0 + torch.expm1(delta_log_y) - delta * c
    bracket_safe = torch.clamp(bracket, min=LOG_Y_FLOOR)
    general_log_y_new = torch.log(bracket_safe) / delta_safe

    a = log_y - c
    series_log_y_new = (
        a
        + 0.5 * delta * (log_y.square() - a.square())
        + delta.square()
        * (log_y.pow(3) / 6.0 - a * log_y.square() / 2.0 + a.pow(3) / 3.0)
    )
    log_y_new = torch.where(
        delta_near_zero,
        series_log_y_new,
        general_log_y_new,
    )
    y_new_positive = torch.exp(log_y_new)
    y_new = torch.where(
        (y_available > 0.0) & (bracket > 0.0),
        y_new_positive,
        torch.zeros_like(y_new_positive),
    )
    z_new = z0_safe * y_new
    q_ss = z_available - z_new
    q_raw = q_ss
    return q_ss, z_new, z_available, q_raw, y_available


def _subsurface_response_step(
    r_ss: torch.Tensor,
    z: torch.Tensor,
    tau_0: torch.Tensor,
    beta: torch.Tensor,
    z0: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Shared full/lite-facing kernel for the analytic response update."""
    return _analytic_subsurface_response_step(
        r_ss,
        z,
        tau_0,
        beta,
        z0,
        nearzero,
    )


def _dr_response_step(
    r_ss: torch.Tensor,
    z: torch.Tensor,
    tau_0: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Full D_R kernel."""
    return _subsurface_response_step(
        r_ss,
        z,
        tau_0,
        torch.ones_like(z),
        z.new_tensor(DEFAULT_Z0),
        nearzero,
    )


def _gr_response_step(
    r_ss: torch.Tensor,
    z: torch.Tensor,
    tau_0: torch.Tensor,
    beta: torch.Tensor,
    z0: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Full G_R kernel with one exponent and fixed/pass-through Z0."""
    return _subsurface_response_step(r_ss, z, tau_0, beta, z0, nearzero)


def _dr_response_step_lite(*args) -> tuple[torch.Tensor, ...]:
    """Compact D_R output: Q_ss and updated reservoir state."""
    out = _dr_response_step(*args)
    return out[0], out[1]


def _gr_response_step_lite(*args) -> tuple[torch.Tensor, ...]:
    """Compact G_R output: Q_ss and updated reservoir state."""
    out = _gr_response_step(*args)
    return out[0], out[1]


class _ResponseModule(nn.Module):
    """Common compile wrapper matching the XAJ full/lite step convention."""

    uses_beta = False

    def __init__(
        self,
        nearzero: float = 1e-8,
        *,
        z0: float | torch.Tensor = DEFAULT_Z0,
        lite: bool = False,
        compile_step: bool = True,
    ):
        super().__init__()
        self.nearzero = nearzero
        self.lite = lite
        # Z0 is a fixed reference scale, not a basin parameter.  Registering a
        # scalar tensor keeps dtype/device movement consistent with nn.Module
        # without adding it to the trainable parameter set.
        if isinstance(z0, torch.Tensor):
            if z0.numel() != 1:
                raise ValueError("z0 must be a scalar tensor")
            self.register_buffer("z0", z0.detach().clone())
        else:
            if z0 <= 0.0:
                raise ValueError("z0 must be positive")
            self.register_buffer("z0", torch.tensor(float(z0), dtype=torch.float64))
        step = _subsurface_response_step
        self._step = torch.compile(step, fullgraph=True) if compile_step else step

    def _beta(self, z: torch.Tensor, beta: torch.Tensor | None) -> torch.Tensor:
        if self.uses_beta:
            if beta is None:
                raise ValueError("G_R requires a basin-specific beta tensor")
            return beta
        return torch.ones_like(z)

    def forward(
        self,
        r_ss: torch.Tensor,
        z: torch.Tensor,
        tau_0: torch.Tensor,
        beta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        z0 = self.z0.to(device=z.device, dtype=z.dtype)
        out = self._step(
            r_ss,
            z,
            tau_0,
            self._beta(z, beta),
            z0,
            self.nearzero,
        )
        if self.lite:
            return out[0], out[1]
        return out


class SubsurfaceResponseDR(_ResponseModule):
    """Full D_R single linear subsurface reservoir."""


class SubsurfaceResponseDRLite(SubsurfaceResponseDR):
    """Lite D_R process module with the same conservative kernel."""

    def __init__(
        self,
        nearzero: float = 1e-8,
        *,
        z0: float | torch.Tensor = DEFAULT_Z0,
        compile_step: bool = True,
    ):
        super().__init__(nearzero, z0=z0, lite=True, compile_step=compile_step)


class SubsurfaceResponseGR(_ResponseModule):
    """Full G_R single nonlinear power-law subsurface reservoir."""

    uses_beta = True


class SubsurfaceResponseGRLite(SubsurfaceResponseGR):
    """Lite G_R process module with the same conservative kernel."""

    def __init__(
        self,
        nearzero: float = 1e-8,
        *,
        z0: float | torch.Tensor = DEFAULT_Z0,
        compile_step: bool = True,
    ):
        super().__init__(nearzero, z0=z0, lite=True, compile_step=compile_step)


DR = SubsurfaceResponseDR
DRLite = SubsurfaceResponseDRLite
GR = SubsurfaceResponseGR
GRLite = SubsurfaceResponseGRLite

__all__ = [
    "DR",
    "DRLite",
    "GR",
    "GRLite",
    "SubsurfaceResponseDR",
    "SubsurfaceResponseDRLite",
    "SubsurfaceResponseGR",
    "SubsurfaceResponseGRLite",
    "DEFAULT_Z0",
    "RESPONSE_POWER_FLOOR",
    "RESPONSE_DT",
    "BETA_ONE_TOL",
    "_analytic_subsurface_response_step",
    "normalized_to_controlled_ci",
    "normalized_to_controlled_cg",
    "normalized_to_tau0",
    "normalized_to_kss",
    "normalized_to_beta",
    "native_linear_tau",
    "native_linear_storage",
    "native_linear_step_from_storage",
    "native_effective_kss",
    "xaj_subsurface_input",
    "response_conditioning_tensors",
    "summarize_response_conditioning",
    "_dr_response_step",
    "_dr_response_step_lite",
    "_gr_response_step",
    "_gr_response_step_lite",
]
