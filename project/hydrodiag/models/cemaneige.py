from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import BaseHydrologicalModel
from .parameter_specs import CEMANEIGE_HYST_PARAM_SPECS, CEMANEIGE_PARAM_SPECS
from .utils import validate_forcings

CEMANEIGE_MIN_THACC = 1e-6


def _solid_liquid_partition(
    precip_t: torch.Tensor,
    temp_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split precipitation using the mean-temperature branch used by RRMPG."""
    frac_solid = torch.where(
        temp_t <= 0.0,
        torch.ones_like(temp_t),
        torch.where(
            temp_t >= 3.0,
            torch.zeros_like(temp_t),
            1.0 - (temp_t + 1.0) / 4.0,
        ),
    )
    frac_solid = torch.clamp(frac_solid, 0.0, 1.0)
    snow = precip_t * frac_solid
    rain = precip_t - snow
    return snow, rain, frac_solid


def _cemaneige_step(
    precip_t: torch.Tensor,
    temp_t: torch.Tensor,
    G: torch.Tensor,
    eTG: torch.Tensor,
    ctg: torch.Tensor,
    kf: torch.Tensor,
    g_thresh: torch.Tensor,
    nearzero: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """One step of the two-parameter CemaNeige snow routine.

    This is the PyTorch/batched counterpart of RRMPG's ``run_cemaneige``:
    CTG and Kf are calibrated, while the snow-cover threshold is fixed at
    90 percent of mean annual solid precipitation.
    """
    snow, rain, _frac_solid = _solid_liquid_partition(precip_t, temp_t)
    G = G + snow

    eTG = ctg * eTG + (1.0 - ctg) * temp_t
    eTG = torch.clamp(eTG, max=0.0)

    is_melting = (eTG == 0.0) & (temp_t > 0.0)
    pot_melt = torch.where(is_melting, kf * temp_t, torch.zeros_like(temp_t))
    pot_melt = torch.min(pot_melt, G)

    # Basic CemaNeige uses an instantaneous SCA ratio rather than a dynamic
    # hysteresis state.
    g_thresh_safe = torch.clamp(g_thresh, min=nearzero)
    sca = torch.where(
        g_thresh > nearzero,
        torch.clamp(G / (g_thresh_safe + nearzero), 0.0, 1.0),
        torch.zeros_like(G),
    )
    melt = (0.9 * sca + 0.1) * pot_melt
    melt = torch.min(melt, G)
    G = G - melt

    return rain + melt, G, eTG, sca, rain, melt


def _cemaneige_hyst_step(
    precip_t: torch.Tensor,
    temp_t: torch.Tensor,
    G: torch.Tensor,
    eTG: torch.Tensor,
    sca: torch.Tensor,
    swe_max: torch.Tensor,
    ctg: torch.Tensor,
    kf: torch.Tensor,
    thacc: torch.Tensor,
    rsp: torch.Tensor,
    psol_annual: torch.Tensor,
    nearzero: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    snow, rain, _frac_solid = _solid_liquid_partition(precip_t, temp_t)

    G = G + snow

    # Thermal state update
    eTG = ctg * eTG + (1.0 - ctg) * temp_t
    eTG = torch.clamp(eTG, max=0.0)

    # Potential melt
    is_melting = (eTG == 0.0) & (temp_t > 0.0)
    pot_melt = torch.where(is_melting, kf * temp_t, torch.zeros_like(temp_t))
    pot_melt = torch.min(pot_melt, G)

    # Snow balance
    snow_balance = snow - pot_melt

    # Update SCA with hysteresis
    # Accumulation phase (snow_balance >= 0)
    # thacc is a physical scale in a denominator.  Clamp direct callers as
    # well as declaring a positive parameter bound so an exact zero cannot
    # create an excessively sensitive accumulation update.
    thacc_safe = torch.clamp(thacc, min=CEMANEIGE_MIN_THACC)
    sca = torch.where(
        snow_balance >= 0.0,
        sca + snow_balance / (thacc_safe + nearzero),
        sca,  # placeholder, updated below
    )
    swe_max = torch.where(snow_balance >= 0.0, torch.max(swe_max, G), swe_max)

    # Ablation phase (snow_balance < 0)
    Thmelt = psol_annual * rsp
    Thmax = torch.where(swe_max > Thmelt, Thmelt, swe_max)
    sca = torch.where(
        snow_balance < 0.0,
        torch.where(Thmax > 0.0, G / (Thmax + nearzero), torch.zeros_like(G)),
        sca,
    )
    sca = torch.clamp(sca, 0.0, 1.0)

    # Actual melt
    melt = (0.9 * sca + 0.1) * pot_melt
    melt = torch.min(melt, G)
    G = G - melt

    # Reset swe_max if snow pack is empty
    swe_max = torch.where(G <= nearzero, torch.zeros_like(swe_max), swe_max)

    # Liquid water outflow
    outflow_t = rain + melt

    return outflow_t, G, eTG, sca, swe_max, rain, melt


def _estimate_psol_annual(
    precip: torch.Tensor,
    temp: torch.Tensor,
) -> torch.Tensor:
    """Estimate annual solid precipitation per basin."""
    _snow, _rain, frac_solid = _solid_liquid_partition(precip, temp)
    solid_precip = precip * frac_solid
    return 365.25 * solid_precip.mean(dim=1)


def _init_basic_states(
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    initial_states: Optional[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_states is not None:
        return (
            initial_states.get("G", torch.zeros(batch, device=device, dtype=dtype)),
            initial_states.get("eTG", torch.zeros(batch, device=device, dtype=dtype)),
        )
    return (
        torch.zeros(batch, device=device, dtype=dtype),
        torch.zeros(batch, device=device, dtype=dtype),
    )


class CemaNeige(BaseHydrologicalModel):
    """Two-parameter CemaNeige snow module.

    This is the basic CemaNeige formulation represented by RRMPG's
    ``cemaneige_model.py``.  Its calibrated parameters are CTG and Kf; the
    snow-cover threshold is computed as 0.9 times mean annual solid
    precipitation.
    """

    def __init__(self, nearzero: float = 1e-8):
        super().__init__()
        self.nearzero = nearzero
        self._step = torch.compile(_cemaneige_step, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return CEMANEIGE_PARAM_SPECS

    def _estimate_psol_annual(
        self,
        precip: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        return _estimate_psol_annual(precip, temp)

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from .utils import validate_params

        precip, _pet, temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)

        ctg = params["cn_ctg"]
        kf = params["cn_kf"]
        # Optional canonical override: when the caller precomputes the mean
        # annual solid precipitation from a fixed full record
        # (``forcings["cn_psol_annual"]``, [batch]), use it instead of
        # re-estimating it from this call's own input sequence.  Absent the
        # key the historical behavior (estimate from the input sequence) is
        # unchanged.
        psol_annual = forcings.get("cn_psol_annual")
        if psol_annual is None:
            psol_annual = _estimate_psol_annual(precip, temp)
        g_thresh = 0.9 * psol_annual
        G, eTG = _init_basic_states(batch, device, dtype, initial_states)

        outflow = torch.zeros(batch, nsteps, device=device, dtype=dtype)
        sca_store = torch.zeros(batch, nsteps, device=device, dtype=dtype)
        rain_store = torch.zeros(batch, nsteps, device=device, dtype=dtype)
        melt_store = torch.zeros(batch, nsteps, device=device, dtype=dtype)
        for t in range(nsteps):
            (outflow[:, t], G, eTG, sca_t, rain_t, melt_t) = self._step(
                precip[:, t], temp[:, t], G, eTG, ctg, kf, g_thresh, self.nearzero
            )
            sca_store[:, t] = sca_t
            rain_store[:, t] = rain_t
            melt_store[:, t] = melt_t

        aux = {
            "snow_pack": G,
            "thermal_state": eTG,
            "sca": sca_store,
            "rain": rain_store,
            "melt": melt_store,
        }
        if return_states:
            aux["final_states"] = {"G": G, "eTG": eTG}
        return outflow, aux


class CemaNeigeHyst(BaseHydrologicalModel):
    """CemaNeige snow module with hysteresis.

    Based on:
    Valery, A. (2010). PhD thesis, Cemagref.
    Riboust, P. et al. (2019). J. Hydrol. Hydromech., 67, 70-81.

    Reference implementation:
    https://github.com/kratzert/RRMPG/blob/master/rrmpg/models/cemaneigehyst_model.py

    This is a standalone snow module. It outputs liquid water (rain + snowmelt),
    which can be used as precipitation input for a rainfall-runoff model.
    """

    def __init__(self, nearzero: float = 1e-8):
        super().__init__()
        self.nearzero = nearzero
        self._step = torch.compile(_cemaneige_hyst_step, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return CEMANEIGE_HYST_PARAM_SPECS

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from .utils import validate_params

        precip, _pet, temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)

        ctg = params["cn_ctg"]
        kf = params["cn_kf"]
        thacc = params["cn_thacc"]
        rsp = params["cn_rsp"]

        # Optional canonical override (same semantics as CemaNeige.forward).
        psol_annual = forcings.get("cn_psol_annual")
        if psol_annual is None:
            psol_annual = self._estimate_psol_annual(precip, temp)

        G, eTG, sca, swe_max = self._init_states(batch, device, dtype, initial_states)

        outflow, (G, eTG, sca, swe_max, rain_out, melt_out) = self._step_loop(
            precip,
            temp,
            nsteps,
            batch,
            device,
            dtype,
            G,
            eTG,
            sca,
            swe_max,
            ctg,
            kf,
            thacc,
            rsp,
            psol_annual,
        )

        aux = {
            "snow_pack": G,
            "thermal_state": eTG,
            "sca": sca,
            "rain": rain_out,
            "melt": melt_out,
        }
        if return_states:
            aux["final_states"] = {
                "G": G,
                "eTG": eTG,
                "sca": sca,
                "swe_max": swe_max,
            }

        return outflow, aux

    def _estimate_psol_annual(
        self,
        precip: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate annual solid precipitation per basin."""
        frac_solid = torch.where(
            temp <= 0.0,
            torch.ones_like(temp),
            torch.where(
                temp >= 3.0,
                torch.zeros_like(temp),
                1.0 - (temp + 1.0) / 4.0,
            ),
        )
        frac_solid = torch.clamp(frac_solid, 0.0, 1.0)
        solid_precip = precip * frac_solid
        return 365.25 * solid_precip.mean(dim=1)

    def _init_states(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        initial_states: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, ...]:
        if initial_states is not None:
            return (
                initial_states.get("G", torch.zeros(batch, device=device, dtype=dtype)),
                initial_states.get(
                    "eTG", torch.zeros(batch, device=device, dtype=dtype)
                ),
                initial_states.get(
                    "sca", torch.zeros(batch, device=device, dtype=dtype)
                ),
                initial_states.get(
                    "swe_max", torch.zeros(batch, device=device, dtype=dtype)
                ),
            )
        return (
            torch.zeros(batch, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=dtype),
        )

    def _step_loop(
        self,
        precip: torch.Tensor,
        temp: torch.Tensor,
        nsteps: int,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        G: torch.Tensor,
        eTG: torch.Tensor,
        sca: torch.Tensor,
        swe_max: torch.Tensor,
        ctg: torch.Tensor,
        kf: torch.Tensor,
        thacc: torch.Tensor,
        rsp: torch.Tensor,
        psol_annual: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        outflow = torch.zeros(batch, nsteps, device=device, dtype=dtype)
        rain_store = torch.zeros(batch, nsteps, device=device, dtype=dtype)
        melt_store = torch.zeros(batch, nsteps, device=device, dtype=dtype)
        nz = self.nearzero

        for t in range(nsteps):
            outflow[:, t], G, eTG, sca, swe_max, rain_t, melt_t = self._step(
                precip[:, t],
                temp[:, t],
                G,
                eTG,
                sca,
                swe_max,
                ctg,
                kf,
                thacc,
                rsp,
                psol_annual,
                nz,
            )
            rain_store[:, t] = rain_t
            melt_store[:, t] = melt_t

        return outflow, (G, eTG, sca, swe_max, rain_store, melt_store)
