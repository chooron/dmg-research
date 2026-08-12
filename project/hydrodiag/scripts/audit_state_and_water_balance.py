#!/usr/bin/env python
"""Independent state-evolution and mass-balance audit on real CAMELS forcings.

This deliberately calls each uncompiled one-day process kernel, records every
state after every day, and includes routing/UH stores in the mass balance.  It
is a pre-calibration gate, not an optimisation diagnostic.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.cemaneige import (
    CemaNeige,
    CemaNeigeHyst,
    _cemaneige_hyst_step,
    _cemaneige_step,
)
from models.gr4j import GR4J, _gr4j_step
from models.hbv import HBV, _hbv_step
from models.parameter_specs import (
    CEMANEIGE_HYST_PARAM_SPECS,
    CEMANEIGE_PARAM_SPECS,
    GR4J_PARAM_SPECS,
    GR4J_CN_PARAM_SPECS,
    HBV_PARAM_SPECS,
    XAJ_CN_PARAM_SPECS,
    XAJ_PARAM_SPECS,
)
from models.unit_hydro import compute_gr4j_uh_ordinates
from models.xaj import XAJ, XAJ_UH_MAX_LEN, _gamma_uh_ordinates, _xaj_step

DTYPE = torch.float64
EPS = 1e-9
SELECTED_BASINS = ("07148400", "06477500", "09484600")


def scalar_params(specs: dict) -> dict[str, torch.Tensor]:
    return {name: torch.tensor([spec["default"]], dtype=DTYPE) for name, spec in specs.items()}


def tensor_sum(*values: torch.Tensor) -> float:
    return float(sum(v.item() for v in values))


def status(trace: dict[str, list[float]]) -> dict[str, dict[str, float | int | bool]]:
    answer = {}
    for name, values in trace.items():
        a = np.asarray(values, dtype=np.float64)
        changes = int(np.count_nonzero(np.abs(np.diff(a)) > 1e-10))
        answer[name] = {
            "min": float(a.min()), "max": float(a.max()), "final": float(a[-1]),
            "changes": changes, "frozen": bool(changes == 0),
        }
    return answer


def balance_summary(name: str, residuals: list[float], trace: dict[str, list[float]], precipitation: np.ndarray) -> dict:
    residuals_np = np.asarray(residuals, dtype=np.float64)
    total_p = float(precipitation.sum())
    total_abs = float(np.abs(residuals_np).sum())
    return {
        "model": name,
        "days": int(len(precipitation)),
        "total_precip_mm": total_p,
        "cumulative_residual_mm": float(residuals_np.sum()),
        "absolute_residual_mm": total_abs,
        "relative_absolute_residual": total_abs / max(total_p, EPS),
        "max_daily_abs_residual_mm": float(np.abs(residuals_np).max()),
        "passes_1pct": bool(total_abs / max(total_p, EPS) < 0.01),
        "state_evolution": status(trace),
    }


def hbv_et(precip, temp, pet, snowpack, meltwater, sm, p):
    """ETact calculated independently from the pre-step HBV state."""
    rain = precip * (temp >= p["parTT"]).to(DTYPE)
    snow = precip * (temp < p["parTT"]).to(DTYPE)
    snowpack = snowpack + snow
    melt = torch.minimum(torch.clamp(p["parCFMAX"] * (temp - p["parTT"]), min=0.0), snowpack)
    meltwater = meltwater + melt
    snowpack = snowpack - melt
    refreeze = torch.minimum(torch.clamp(p["parCFR"] * p["parCFMAX"] * (p["parTT"] - temp), min=0.0), meltwater)
    snowpack = snowpack + refreeze
    meltwater = meltwater - refreeze
    tosoil = torch.clamp(meltwater - p["parCWH"] * snowpack, min=0.0)
    soil_wetness = torch.clamp((sm / p["parFC"]) ** p["parBETA"], 0.0, 1.0)
    recharge = (rain + tosoil) * soil_wetness
    sm = sm + rain + tosoil - recharge
    sm = sm - torch.clamp(sm - p["parFC"], min=0.0)
    evapfactor = torch.clamp(sm / (p["parLP"] * p["parFC"]), 0.0, 1.0)
    return torch.minimum(sm, pet * evapfactor)


def gr4j_et(precip, pet, s_prod, p):
    mask = precip >= pet
    p_n = torch.where(mask, precip - pet, torch.zeros_like(precip))
    pe_n = torch.where(mask, torch.zeros_like(precip), pet - precip)
    ratio = torch.clamp(s_prod / (p["x1"] + EPS), min=0.0, max=1.0)
    tanh_pe = torch.tanh(pe_n / (p["x1"] + EPS))
    e_s_calc = s_prod * (2.0 - ratio) * tanh_pe / (1.0 + (1.0 - ratio) * tanh_pe + EPS)
    # GR4J uses net forcing.  On wet days Pn=P-PET, so PET is withdrawn;
    # on dry days all rainfall is first consumed and e_s_calc is withdrawn
    # from the production store.
    return torch.where(mask, pet, precip + e_s_calc)


def uh_pending(history: list[float], ordinates: np.ndarray) -> float:
    """Water not yet emitted by a causal finite unit hydrograph."""
    pending = 0.0
    for idx, value in enumerate(history):
        age = len(history) - 1 - idx
        pending += value * float(ordinates[age + 1:].sum())
    return pending


def run_hbv(forcing: dict[str, np.ndarray]) -> tuple[list[float], dict[str, list[float]]]:
    p = scalar_params(HBV_PARAM_SPECS)
    model = HBV()
    snow, melt, sm, suz, slz = model._init_states(1, torch.device("cpu"), DTYPE)
    residuals, trace = [], {k: [] for k in ("SNOWPACK", "MELTWATER", "SM", "SUZ", "SLZ", "Q", "ET")}
    for pr, pe, te in zip(forcing["precip"], forcing["pet"], forcing["temp"]):
        pr, pe, te = (torch.tensor([x], dtype=DTYPE) for x in (pr, pe, te))
        old = tensor_sum(snow, melt, sm, suz, slz)
        et = hbv_et(pr, te, pe, snow, melt, sm, p)
        q, snow, melt, sm, suz, slz = _hbv_step(pr, te, pe, snow, melt, sm, suz, slz,
            p["parTT"], p["parCFMAX"], p["parCFR"], p["parCWH"], p["parFC"], p["parBETA"],
            p["parLP"], p["parPERC"], p["parUZL"], p["parK0"], p["parK1"], p["parK2"], EPS)
        new = tensor_sum(snow, melt, sm, suz, slz)
        residuals.append(float(pr - et - q) - (new - old))
        for name, value in zip(("SNOWPACK", "MELTWATER", "SM", "SUZ", "SLZ", "Q", "ET"), (snow, melt, sm, suz, slz, q, et)):
            trace[name].append(float(value))
    return residuals, trace


def run_cemaneige(forcing: dict[str, np.ndarray]) -> tuple[list[float], dict[str, list[float]], np.ndarray]:
    """Audit the two-parameter CemaNeige used by composed models."""
    p = scalar_params(CEMANEIGE_PARAM_SPECS)
    precip = torch.tensor(forcing["precip"][None, :], dtype=DTYPE)
    temp = torch.tensor(forcing["temp"][None, :], dtype=DTYPE)
    psol = CemaNeige()._estimate_psol_annual(precip, temp)
    g_thresh = 0.9 * psol
    G = eTG = torch.zeros(1, dtype=DTYPE)
    residuals, trace, liquid = [], {k: [] for k in ("G", "eTG", "sca", "Q_liquid")}, []
    for pr, te in zip(precip[0], temp[0]):
        old = float(G)
        q, G, eTG, sca, _rain, _melt = _cemaneige_step(
            pr[None], te[None], G, eTG,
            p["cn_ctg"], p["cn_kf"], g_thresh, EPS,
        )
        residuals.append(float(pr - q) - (float(G) - old))
        for name, value in zip(("G", "eTG", "sca", "Q_liquid"), (G, eTG, sca, q)):
            trace[name].append(float(value))
        liquid.append(float(q))
    return residuals, trace, np.asarray(liquid)


def run_cemaneige_hyst(forcing: dict[str, np.ndarray]) -> tuple[list[float], dict[str, list[float]], np.ndarray]:
    """Audit the retained four-parameter hysteresis interface."""
    p = scalar_params(CEMANEIGE_HYST_PARAM_SPECS)
    precip = torch.tensor(forcing["precip"][None, :], dtype=DTYPE)
    temp = torch.tensor(forcing["temp"][None, :], dtype=DTYPE)
    psol = CemaNeigeHyst()._estimate_psol_annual(precip, temp)
    G = eTG = sca = swe_max = torch.zeros(1, dtype=DTYPE)
    residuals, trace, liquid = [], {k: [] for k in ("G", "eTG", "sca", "swe_max", "Q_liquid")}, []
    for pr, te in zip(precip[0], temp[0]):
        old = float(G)
        q, G, eTG, sca, swe_max, _rain, _melt = _cemaneige_hyst_step(
            pr[None], te[None], G, eTG, sca, swe_max,
            p["cn_ctg"], p["cn_kf"], p["cn_thacc"], p["cn_rsp"], psol, EPS,
        )
        residuals.append(float(pr - q) - (float(G) - old))
        for name, value in zip(("G", "eTG", "sca", "swe_max", "Q_liquid"), (G, eTG, sca, swe_max, q)):
            trace[name].append(float(value))
        liquid.append(float(q))
    return residuals, trace, np.asarray(liquid)


def run_gr4j(forcing: dict[str, np.ndarray]) -> tuple[list[float], dict[str, list[float]]]:
    p = scalar_params(GR4J_PARAM_SPECS)
    model = GR4J()
    s_prod, s_route, uh1, uh2 = model._init_states(1, torch.device("cpu"), DTYPE, x1=p["x1"], x3=p["x3"])
    uh1_ord = compute_gr4j_uh_ordinates(p["x4"], model.UH1_MAX)[0]
    uh2_ord = compute_gr4j_uh_ordinates(p["x4"], model.UH2_MAX)[1]
    residuals, trace = [], {k: [] for k in ("s_prod", "s_route", "uh_pending", "Q", "ET")}
    def storage(): return tensor_sum(s_prod, s_route, uh1[:, 1:].sum(), uh2[:, 1:].sum())
    for pr, pe in zip(forcing["precip"], forcing["pet"]):
        pr, pe = (torch.tensor([x], dtype=DTYPE) for x in (pr, pe))
        old = storage(); et = gr4j_et(pr, pe, s_prod, p)
        q, s_prod, s_route, uh1, uh2 = _gr4j_step(pr, pe, s_prod, s_route, uh1, uh2, uh1_ord, uh2_ord,
            p["x1"], p["x2"], p["x3"], EPS)
        new = storage()
        residuals.append(float(pr - et - q) - (new - old))
        for name, value in (("s_prod", s_prod), ("s_route", s_route), ("uh_pending", uh1[:, 1:].sum() + uh2[:, 1:].sum()), ("Q", q), ("ET", et)):
            trace[name].append(float(value))
    return residuals, trace


def run_xaj(forcing: dict[str, np.ndarray]) -> tuple[list[float], dict[str, list[float]]]:
    p = scalar_params(XAJ_PARAM_SPECS)
    model = XAJ()
    wu, wl, wd, s, fr, qi, qg, _buf = model._init_states(1, torch.device("cpu"), DTYPE, um=p["xaj_um"], lm=p["xaj_lm"], dm=p["xaj_dm"], sm=p["xaj_sm"])
    ki, kg = p["xaj_ki"], p["xaj_kg"]
    if float(ki + kg) >= 1.0: ki, kg = ki * .99 / (ki + kg), kg * .99 / (ki + kg)
    ords = _gamma_uh_ordinates(p["xaj_a"], p["xaj_theta"], XAJ_UH_MAX_LEN, torch.device("cpu"), DTYPE)[0].numpy()
    history: list[float] = []
    residuals, trace = [], {k: [] for k in ("wu", "wl", "wd", "free_water", "qi", "qg", "uh_pending", "Q", "ET")}
    def storage():
        # Tension/free-water stores are expressed per permeable area.  The
        # linear reservoirs and UH already receive their IM-adjusted inflow.
        permeable = 1.0 - p["xaj_im"]
        tension_and_free = permeable * (wu + wl + wd + fr * s)
        # qi/qg are outputs of q_t=c*q_(t-1)+(1-c)*input; their associated
        # reservoir storage is c*q/(1-c), not q/(1-c).
        linear = p["xaj_ci"] * qi / (1.0 - p["xaj_ci"]) + p["xaj_cg"] * qg / (1.0 - p["xaj_cg"])
        return float(tension_and_free + linear) + uh_pending(history, ords)
    for pr, pe in zip(forcing["precip"], forcing["pet"]):
        pr, pe = (torch.tensor([x], dtype=DTYPE) for x in (pr, pe))
        old = storage()
        (_unused, rs_adj, qi, qg, evap, wu, wl, wd, s, fr, _rs, _ri, _rg, _eu, _el, _ed) = _xaj_step(
            pr, pe, wu, wl, wd, s, fr, qi, qg, p["xaj_k"], p["xaj_b"], p["xaj_im"], p["xaj_um"], p["xaj_lm"], p["xaj_dm"],
            p["xaj_c"], p["xaj_sm"], p["xaj_ex"], ki, kg, p["xaj_ci"], p["xaj_cg"], EPS)
        history.append(float(rs_adj)); history = history[-XAJ_UH_MAX_LEN:]
        routed = sum(value * float(ords[len(history) - 1 - idx]) for idx, value in enumerate(history))
        q = routed + float(qi + qg)
        new = storage()
        # On dry-net days the impervious fraction cannot draw from tension
        # water; this is implicit in XAJ's pe=max(P-E,0) formulation.
        evap_basin = evap + p["xaj_im"] * torch.minimum(pr - evap, torch.zeros_like(pr))
        residuals.append(float(pr - evap_basin) - q - (new - old))
        values = (wu, wl, wd, fr * s, p["xaj_ci"] * qi / (1.0 - p["xaj_ci"]), p["xaj_cg"] * qg / (1.0 - p["xaj_cg"]), torch.tensor([uh_pending(history, ords)]), torch.tensor([q]), evap_basin)
        for name, value in zip(trace, values): trace[name].append(float(value))
    return residuals, trace


def plot_trace(output: Path, model: str, basin: str, trace: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, values in trace.items():
        if name not in {"ET", "Q", "Q_liquid"}:
            ax.plot(values, linewidth=.8, label=name)
    ax.set(title=f"{model}: state evolution, basin {basin}", xlabel="day", ylabel="mm or dimensionless")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "precalibration_integrity_audit")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(ROOT_DIR / "data" / "camels_dataset_petv2.npz", allow_pickle=True)
    dates = np.asarray(np.load(ROOT_DIR / "data" / "camels_dates.npy"), dtype="datetime64[D]")
    basin_ids = [str(int(x)).zfill(8) for x in np.load(ROOT_DIR / "data" / "gage_id.npy")]
    indices = [basin_ids.index(basin) for basin in SELECTED_BASINS]
    mask = (dates >= np.datetime64("1988-01-01")) & (dates <= np.datetime64("1998-12-31"))
    forcing_all = data["forcing"][mask]
    reports = {}
    for basin, idx in zip(SELECTED_BASINS, indices):
        forcing = {"precip": forcing_all[:, idx, 0].astype(np.float64), "temp": forcing_all[:, idx, 1].astype(np.float64), "pet": forcing_all[:, idx, 2].astype(np.float64)}
        cn_res, cn_trace, effective = run_cemaneige(forcing)
        hyst_res, hyst_trace, hyst_effective = run_cemaneige_hyst(forcing)
        gr_res, gr_trace = run_gr4j(forcing)
        hbv_res, hbv_trace = run_hbv(forcing)
        xaj_res, xaj_trace = run_xaj(forcing)
        gr_cn_res, gr_cn_trace = run_gr4j({**forcing, "precip": effective})
        xaj_cn_res, xaj_cn_trace = run_xaj({**forcing, "precip": effective})
        # Add CemaNeige storage into composed-system residuals; the CemaNeige
        # residual is numerical round-off, so this yields P - ET - Q - dS_all.
        gr_cn_res = (np.asarray(gr_cn_res) + np.asarray(cn_res)).tolist()
        xaj_cn_res = (np.asarray(xaj_cn_res) + np.asarray(cn_res)).tolist()
        gr_cn_trace = {**{f"cn_{k}": v for k, v in cn_trace.items()}, **{f"gr4j_{k}": v for k, v in gr_cn_trace.items()}}
        xaj_cn_trace = {**{f"cn_{k}": v for k, v in cn_trace.items()}, **{f"xaj_{k}": v for k, v in xaj_cn_trace.items()}}
        reports[basin] = {
            "HBV": balance_summary("HBV", hbv_res, hbv_trace, forcing["precip"]),
            "GR4J": balance_summary("GR4J", gr_res, gr_trace, forcing["precip"]),
            "XAJ": balance_summary("XAJ", xaj_res, xaj_trace, forcing["precip"]),
            "CemaNeige": balance_summary("CemaNeige", cn_res, cn_trace, forcing["precip"]),
            "CemaNeigeHyst": balance_summary("CemaNeigeHyst", hyst_res, hyst_trace, forcing["precip"]),
            "GR4J_CN": balance_summary("GR4J+CemaNeige", gr_cn_res, gr_cn_trace, forcing["precip"]),
            "XAJ_CN": balance_summary("XAJ+CemaNeige", xaj_cn_res, xaj_cn_trace, forcing["precip"]),
        }
        for model, trace in (("HBV", hbv_trace), ("GR4J", gr_trace), ("XAJ", xaj_trace), ("CemaNeige", cn_trace), ("CemaNeigeHyst", hyst_trace), ("GR4J_CN", gr_cn_trace), ("XAJ_CN", xaj_cn_trace)):
            plot_trace(args.output_dir / f"{basin}_{model}_states.png", model, basin, trace)
    (args.output_dir / "report.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
