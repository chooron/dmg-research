"""
gsfb_smooth.py — Smooth differentiable variant of the GSFB model.

This module implements a smooth approximation of the GSFB (Griffiths et al.)
hydrological model using soft-capping helpers to replace all hard
`torch.minimum` / `torch.clamp` flux constraints. The smooth variant is
differentiable everywhere, which makes it compatible with gradient-based
optimisation and Euler-substep first-order convergence analysis.

IMPORTANT CAVEATS
-----------------
1. gsfb_smooth is NOT identical to the original MARRMoT GSFB formula.
   Hard constraints become smooth approximations parameterised by ``tau``.
2. The original GSFB structural caveat (non-differentiable flux caps) is
   resolved here; the original ``models/core/gsfb.py`` is left **unchanged**.
3. Do not compare gsfb_smooth against MARRMoT step-by-step daily outputs.

Mathematical background
-----------------------
smooth_relu(x, τ)          = τ · softplus(x / τ)
smooth_min(a, b, τ)        = −τ · logsumexp(−[a, b] / τ,  dim=0)
smooth_cap_flux(q, avail, τ) = smooth_min(smooth_relu(q, τ),
                                           smooth_relu(avail, τ), τ)

As τ → 0 the functions converge to relu / minimum pointwise, but keep
non-zero gradients everywhere.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from ..flux.smooth import smooth_cap_flux, smooth_min, smooth_relu

from ..flux.evap import evap_20
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_11
from ..flux.baseflow import baseflow_1, baseflow_9
from ..flux.recharge import recharge_5


# ---------------------------------------------------------------------------
# Re-export parameter metadata identical to gsfb.py so registry code can
# import from either location.
# ---------------------------------------------------------------------------
GSFB_PARAMS_BOUNDS = {
    "c":      [0.0,   1.0],
    "ndc":    [0.05,  0.95],
    "smax":   [1.0,   2000.0],
    "emax":   [0.0,   20.0],
    "frate":  [0.0,   200.0],
    "b":      [0.0,   1.0],
    "dpf":    [0.0,   1.0],
    "sdrmax": [1.0,   300.0],
}

GSFB_PARAMS_DESC = {
    "c":      "Recharge time coefficient [d-1]",
    "ndc":    "Threshold fraction of Smax [-]",
    "smax":   "Maximum soil moisture storage [mm]",
    "emax":   "Maximum evaporation flux [mm/d]",
    "frate":  "Maximum infiltration rate [mm/d]",
    "b":      "Fraction of subsurface flow that is baseflow [-]",
    "dpf":    "Baseflow time coefficient [d-1]",
    "sdrmax": "Threshold before baseflow can occur [mm]",
}


# ---------------------------------------------------------------------------
# Smooth cap helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# State initialiser (identical to gsfb.py)
# ---------------------------------------------------------------------------

def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create initial states S1, S2, S3 (soil moisture, intermediate, saturated zone)."""
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3


# ---------------------------------------------------------------------------
# Main step function
# ---------------------------------------------------------------------------

def gsfb_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters — identical names/bounds to gsfb_step
    c: torch.Tensor,
    ndc: torch.Tensor,
    smax: torch.Tensor,
    emax: torch.Tensor,
    frate: torch.Tensor,
    b: torch.Tensor,
    dpf: torch.Tensor,
    sdrmax: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    nearzero: float = 1e-6,
    tau: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Smooth differentiable GSFB single-step calculation.

    All hard ``torch.minimum`` / ``torch.clamp`` flux caps from the original
    ``gsfb_step`` are replaced by ``smooth_cap_flux(q_pot, available, tau)``.
    The flux functions themselves (recharge_5, saturation_1, evap_20,
    interflow_11, baseflow_9, baseflow_1) are called with the same arguments.

    Parameters
    ----------
    P, T, PET : forcing tensors (batch, nmul)
    c … sdrmax : parameter tensors (batch, nmul) — same semantics as gsfb_step
    S1, S2, S3 : state tensors (batch, nmul)
    nearzero   : small positive constant for numerical safety
    tau        : smoothness parameter (default 1e-3 mm/d).
                 Smaller τ → closer to original hard caps.
                 Larger τ → smoother but more biased.

    Returns
    -------
    Qsim, Ea, S1_new, S2_new, S3_new
    """

    # ------------------------------------------------------------------
    # 1. Saturated zone → Soil moisture recharge  (S3 → S1)
    # ------------------------------------------------------------------
    threshold_s1 = ndc * smax

    # Potential recharge from saturated zone
    flux_qdr_pot = recharge_5(c, threshold_s1, S3, S1, nearzero=nearzero)
    # Cap: flux_qdr ≤ S3 − ε  (keeps S3 > 0)
    flux_qdr = smooth_cap_flux(flux_qdr_pot, S3 - nearzero, tau=tau)

    # Interim S3 update
    S3_tmp = torch.clamp(S3 - flux_qdr, min=nearzero)

    # ------------------------------------------------------------------
    # 2. Soil Moisture Store  (S1)
    # ------------------------------------------------------------------
    S1_in = P + flux_qdr

    # Saturation excess runoff
    flux_qs_pot = saturation_1(S1_in, S1, smax, nearzero=nearzero)
    # Cap: flux_qs ∈ [0, S1_in]
    flux_qs = smooth_cap_flux(flux_qs_pot, S1_in, tau=tau)

    # S1 after runoff, before evaporation
    S1_tmp = torch.clamp(S1 + S1_in - flux_qs, min=nearzero)

    # Evaporation from S1
    flux_ea_pot = evap_20(emax, ndc, S1_tmp, smax, PET, nearzero=nearzero)
    # Cap 1: flux_ea ≤ S1_tmp − ε
    flux_ea = smooth_cap_flux(flux_ea_pot, S1_tmp - nearzero, tau=tau)
    # Cap 2: flux_ea ≤ PET
    flux_ea = smooth_cap_flux(flux_ea, PET, tau=tau)

    # S1 after evaporation, before infiltration
    S1_tmp2 = torch.clamp(S1_tmp - flux_ea, min=nearzero)

    # Infiltration from S1 to S2
    flux_f_pot = interflow_11(frate, threshold_s1, S1_tmp2, nearzero=nearzero)
    # Cap: flux_f ≤ S1_tmp2 − ε
    flux_f = smooth_cap_flux(flux_f_pot, S1_tmp2 - nearzero, tau=tau)

    S1_new = torch.clamp(S1_tmp2 - flux_f, min=nearzero)

    # ------------------------------------------------------------------
    # 3. Intermediate Store  (S2)
    # ------------------------------------------------------------------
    S2_tmp_in = torch.clamp(S2 + flux_f, min=nearzero)

    # Baseflow from S2 (slow process 1)
    flux_qb_pot = baseflow_9(b * dpf, sdrmax, S2_tmp_in, nearzero=nearzero)
    # Cap: flux_qb ≤ S2_tmp_in − ε
    flux_qb = smooth_cap_flux(flux_qb_pot, S2_tmp_in - nearzero, tau=tau)

    S2_tmp_perc = torch.clamp(S2_tmp_in - flux_qb, min=nearzero)

    # Percolation from S2 to S3 (slow process 2)
    flux_dp_pot = baseflow_1((1.0 - b) * dpf, S2_tmp_perc, nearzero=nearzero)
    # Cap: flux_dp ≤ S2_tmp_perc − ε
    flux_dp = smooth_cap_flux(flux_dp_pot, S2_tmp_perc - nearzero, tau=tau)

    S2_new = torch.clamp(S2_tmp_perc - flux_dp, min=nearzero)

    # ------------------------------------------------------------------
    # 4. Saturated Zone Store  (S3)
    # ------------------------------------------------------------------
    S3_new = torch.clamp(S3_tmp + flux_dp, min=nearzero)

    # ------------------------------------------------------------------
    # 5. Output
    # ------------------------------------------------------------------
    Qsim = flux_qs + flux_qb
    Ea   = flux_ea

    return Qsim, Ea, S1_new, S2_new, S3_new

# Backward-compatibility aliases
GSFB_PARAMS_BOUNDS = GSFB_PARAMS_BOUNDS
gsfb_step = gsfb_step  # alias for tests that still reference old name


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (pre-rename names). gsfb_smooth was renamed
# to gsfb on 2026-06-25; these aliases keep older imports/tests working.
# ---------------------------------------------------------------------------
GSFB_SMOOTH_PARAMS_BOUNDS = GSFB_PARAMS_BOUNDS
GSFB_SMOOTH_PARAMS_DESC = GSFB_PARAMS_DESC
gsfb_smooth_step = gsfb_step
