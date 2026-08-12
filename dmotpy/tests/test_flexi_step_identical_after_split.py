"""Golden reference test: verify the refactored flexi_step is bitwise-identical
to the original pre-split implementation.

The original flexi_step (pre-split) is embedded here as _flexi_step_original so the
comparison reference cannot be altered by the refactoring.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from models.flux.interception import interception_1
from models.flux.evap import evap_1, evap_3
from models.flux.saturation import saturation_3
from models.flux.percolation import percolation_2
from models.flux.split import split_1
from models.flux.baseflow import baseflow_1


def _flexi_step_original(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    smax: torch.Tensor,
    beta: torch.Tensor,
    d_split: torch.Tensor,
    percmax: torch.Tensor,
    lp: torch.Tensor,
    nlagf: torch.Tensor,
    nlags: torch.Tensor,
    kf: torch.Tensor,
    ks: torch.Tensor,
    imax: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # UH parameters are unused (identity routing)
    _ = (nlagf, nlags)

    # --- 1. Interception Process (S1) ---
    # flux_peff: Throughfall (Saturation excess from S1)
    flux_peff = interception_1(P, S1, imax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_peff)
    flux_peff = torch.clamp(flux_peff, min=zeros, max=P)

    # Update S1 for evaporation
    S1_tmp = S1 + P - flux_peff
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_ei: Evaporation from interception
    flux_ei = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_ei = torch.minimum(flux_ei, S1_tmp - nearzero)
    flux_ei = F.relu(flux_ei)

    # Final S1 update
    S1_new = S1_tmp - flux_ei
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Soil Moisture Process (S2) ---
    # flux_ru: Infiltration into S2 soil store
    flux_ru = saturation_3(S2, smax, beta, flux_peff, nearzero=nearzero)
    flux_ru = torch.clamp(flux_ru, min=zeros, max=flux_peff)

    # Surface excess after infiltration
    rem_peff = F.relu(flux_peff - flux_ru)

    # Split excess into fast (rf) and slow (rs) components
    flux_rf = split_1(1.0 - d_split, rem_peff, nearzero=nearzero)
    flux_rs = F.relu(rem_peff - flux_rf)

    # Update S2 for actual ET and percolation
    S2_tmp = S2 + flux_ru
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Remaining PET after interception ET
    PET_rem = F.relu(PET - flux_ei)

    # flux_eur: Evapotranspiration from soil
    flux_eur = evap_3(lp, S2_tmp, smax, PET_rem, nearzero=nearzero)
    flux_eur = torch.minimum(flux_eur, S2_tmp - nearzero)
    flux_eur = F.relu(flux_eur)

    S2_tmp2 = S2_tmp - flux_eur
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)

    # flux_ps: Percolation to slow reservoir
    flux_ps = percolation_2(percmax, S2_tmp2, smax, nearzero=nearzero)
    flux_ps = torch.minimum(flux_ps, S2_tmp2 - nearzero)
    flux_ps = F.relu(flux_ps)

    # Final S2 update
    S2_new = S2_tmp2 - flux_ps
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Routing Processes (S3 and S4) ---

    # TODO: Inner Routing using DplTri3, using nlagf and nlags as delay parameters
    # Instantaneous routing for flux_rf (fast) and (flux_ps + flux_rs) (slow)
    flux_rfl = flux_rf
    flux_rsl = flux_ps + flux_rs

    # S3: Fast Routing Store
    S3_tmp = S3 + flux_rfl
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)

    flux_qf = baseflow_1(kf, S3_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S3_tmp - nearzero)
    flux_qf = F.relu(flux_qf)

    S3_new = S3_tmp - flux_qf
    S3_new = torch.clamp(S3_new, min=nearzero)

    # S4: Slow Routing Store
    S4_tmp = S4 + flux_rsl
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)

    flux_qs = baseflow_1(ks, S4_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S4_tmp - nearzero)
    flux_qs = F.relu(flux_qs)

    S4_new = S4_tmp - flux_qs
    S4_new = torch.clamp(S4_new, min=nearzero)

    # --- 4. Output Aggregation ---
    # Qsim = qf + qs
    # Ea = ei + eur
    Qsim = flux_qf + flux_qs
    Ea = flux_ei + flux_eur

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new


def test_flexi_step_identical_after_split():
    """Verify the refactored flexi_step is bitwise-identical to the original."""
    from models.core.flexi import flexi_step

    rng = torch.Generator()
    rng.manual_seed(20260626)

    for i in range(100):
        P = torch.rand(3, 2, generator=rng) * 10.0
        T = torch.rand(3, 2, generator=rng) * 5.0 - 2.0
        PET = torch.rand(3, 2, generator=rng) * 3.0

        smax = torch.rand(3, 2, generator=rng) * 1999.0 + 1.0
        beta = torch.rand(3, 2, generator=rng) * 10.0
        d_split = torch.rand(3, 2, generator=rng)
        percmax = torch.rand(3, 2, generator=rng) * 20.0
        lp = torch.rand(3, 2, generator=rng) * 0.9 + 0.05
        nlagf = torch.rand(3, 2, generator=rng) * 4.0 + 1.0
        nlags = torch.rand(3, 2, generator=rng) * 14.0 + 1.0
        kf = torch.rand(3, 2, generator=rng)
        ks = torch.rand(3, 2, generator=rng)
        imax = torch.rand(3, 2, generator=rng) * 5.0

        S1 = torch.rand(3, 2, generator=rng) * 3.0 + 1e-6
        S2 = torch.rand(3, 2, generator=rng) * 1000.0 + 1e-6
        S3 = torch.rand(3, 2, generator=rng) * 100.0 + 1e-6
        S4 = torch.rand(3, 2, generator=rng) * 100.0 + 1e-6

        nearzero = 1e-6

        original = _flexi_step_original(
            P, T, PET, smax, beta, d_split, percmax, lp, nlagf, nlags, kf, ks, imax,
            S1, S2, S3, S4, nearzero,
        )
        refactored = flexi_step(
            P, T, PET, smax, beta, d_split, percmax, lp, nlagf, nlags, kf, ks, imax,
            S1, S2, S3, S4, nearzero,
        )

        for j, (a, b) in enumerate(zip(original, refactored)):
            diff = (a - b).abs().max().item()
            assert torch.allclose(a, b, atol=1e-15), (
                f"Iteration {i}, output element {j}: max diff = {diff:.6e}\n"
                f"Original: {a.flatten()[:5]}\n"
                f"Refactored: {b.flatten()[:5]}"
            )
