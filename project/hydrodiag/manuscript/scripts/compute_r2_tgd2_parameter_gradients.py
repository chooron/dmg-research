#!/usr/bin/env python3
"""Parameter-level Base-TGD2 snow gradients for the R2 Supplement (Fig. S5 / Table S4).

Role: supplement-only matched-control counterpart of `r2_snow_gradients_summary.csv`.
The main text (F4) reports snow gradients of the canonical paired shift
delta = z_Base - z_CN for the 15 shared XAJ parameters. This script computes the
same quantity for the structural control delta = z_Base - z_TGD2, so that the
Base-CN and Base-TGD2 parameter-level gradients can be shown together in one
supplement forest plot without recomputing or modifying any frozen main-text result.

Read-only with respect to all existing results. The only written output is
`manuscript/results/R2/r2_snow_gradients_base_tgd2_summary.csv`.

Statistical protocol (identical to the frozen Base-CN pipeline):
  * paired shift: delta_p,i = z_Base,p,i - z_CN/TGD2,p,i from the canonical
    normalized values (z = (theta-lower)/(upper-lower)); dPL canonical is the
    within-basin median over the 3 seeds.
  * beta = OLS slope of delta ~ frac_snow (basin-level, one paired shift per basin).
  * 95% CI = percentile interval of the OLS slope over 10,000 basin resamples,
    rng = np.random.default_rng(20260730), basin as the unit of independence.
  * To make the Base-TGD2 intervals draw-comparable with the frozen Base-CN
    intervals, the Base-TGD2 slopes are evaluated on the *same* 10,000 basin
    resamples that produced the frozen Base-CN intervals. The RNG stream is
    replicated exactly as in `run_r2_parameter_statistics.py` (30 primary-median
    bootstrap calls, then 30 slope bootstrap calls in the frozen group order) and
    the reproduction is asserted at 1e-12 before any Base-TGD2 value is accepted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R2 = MANUSCRIPT / "results" / "R2"

N_BOOT = 10000
SEED = 20260730

PARAM_ORDER = [
    "xaj_k", "xaj_b", "xaj_im", "xaj_um", "xaj_lm", "xaj_dm", "xaj_c",
    "xaj_sm", "xaj_ex", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg", "xaj_a",
    "xaj_theta",
]
DISPLAY = {
    "xaj_k": "k", "xaj_b": "b", "xaj_im": "im", "xaj_um": "um",
    "xaj_lm": "lm", "xaj_dm": "dm", "xaj_c": "c", "xaj_sm": "sm",
    "xaj_ex": "ex", "xaj_ki": "ki", "xaj_kg": "kg", "xaj_ci": "ci",
    "xaj_cg": "cg", "xaj_a": "a (UH shape)", "xaj_theta": "theta (UH scale)",
}


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.polyfit(x, y, 1)[0])


def build_paired_tgd2(canonical: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct Base-TGD2 paired shifts with the same merge as the frozen
    Base-CN paired file, so that within-group basin order is identical and the
    frozen bootstrap resamples apply positionally."""
    base = canonical[canonical["structure"] == "Base"].rename(
        columns={"z": "z_base", "value_physical": "value_base"})
    gd = canonical[canonical["structure"] == "GD"].rename(
        columns={"z": "z_gd", "value_physical": "value_gd"})
    paired = base.merge(gd[["paradigm", "basin_id", "parameter", "z_gd", "value_gd"]],
                        on=["paradigm", "basin_id", "parameter"],
                        how="outer", validate="one_to_one", indicator=True)
    if not (paired["_merge"] == "both").all():
        raise ValueError("Base/GD pair alignment failure")
    paired["delta_base_minus_tgd2"] = paired["z_base"] - paired["z_gd"]
    paired["parameter_display"] = paired["parameter"].map(DISPLAY)
    return paired


def main() -> None:
    canonical = pd.read_csv(RESULTS_R2 / "r2_parameter_values_canonical.csv")
    canonical["basin_id"] = canonical["basin_id"].astype(str).str.zfill(8)
    frozen_paired = pd.read_csv(RESULTS_R2 / "r2_paired_shifts_basin_level.csv")
    frozen_paired["basin_id"] = frozen_paired["basin_id"].astype(str).str.zfill(8)
    frozen_grad = pd.read_csv(RESULTS_R2 / "r2_snow_gradients_summary.csv")

    paired_tg = build_paired_tgd2(canonical)

    # --- sanity: basin order within every (paradigm, parameter) group matches ---
    for (paradigm, parameter), g in paired_tg.groupby(["paradigm", "parameter"], sort=False):
        g2 = frozen_paired[(frozen_paired["paradigm"] == paradigm)
                           & (frozen_paired["parameter"] == parameter)]
        if not (g["basin_id"].to_numpy() == g2["basin_id"].to_numpy()).all():
            raise ValueError(f"basin order mismatch: {paradigm} {parameter}")
    # --- sanity: TGD2 dPL strictly interior, IC on bounds (audit convention) ---
    dpl_z = paired_tg[paired_tg["paradigm"] == "dPL"]["z_gd"]
    ic_z = paired_tg[paired_tg["paradigm"] == "IC"]["z_gd"]
    assert float(dpl_z.min()) > 0.0 and float(dpl_z.max()) < 1.0, "dPL GD not interior"
    assert float(ic_z.min()) >= 0.0 and float(ic_z.max()) <= 1.0

    # --- replicate the frozen RNG stream on the frozen Base-CN paired file ---
    rng = np.random.default_rng(SEED)
    # Phase A: 30 primary-median bootstrap calls (consume, discard).
    for (paradigm, parameter), g in frozen_paired.groupby(["paradigm", "parameter"], sort=False):
        vals = g["delta_base_minus_cn"].to_numpy(float)
        idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
        boot = np.asarray([np.median(vals[i]) for i in idx], dtype=float)
        np.quantile(boot, [0.025, 0.975])

    # Phase B: slope bootstraps; capture idx, validate Base-CN, compute Base-TGD2.
    rows = []
    for (paradigm, parameter), g in frozen_paired.groupby(["paradigm", "parameter"], sort=False):
        x = g["frac_snow"].to_numpy(float)
        y_cn = g["delta_base_minus_cn"].to_numpy(float)
        g_tg = paired_tg[(paired_tg["paradigm"] == paradigm)
                         & (paired_tg["parameter"] == parameter)]
        y_tg = g_tg["delta_base_minus_tgd2"].to_numpy(float)
        assert len(x) == len(y_tg) == 531

        idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
        boot_cn = np.asarray([slope(x[i], y_cn[i]) for i in idx], dtype=float)
        boot_tg = np.asarray([slope(x[i], y_tg[i]) for i in idx], dtype=float)

        beta_cn = slope(x, y_cn)
        lo_cn, hi_cn = np.quantile(boot_cn, [0.025, 0.975])
        fr = frozen_grad[(frozen_grad["paradigm"] == paradigm)
                         & (frozen_grad["parameter"] == parameter)].iloc[0]
        assert abs(beta_cn - fr["beta"]) < 1e-12, f"beta mismatch {paradigm} {parameter}"
        assert abs(lo_cn - fr["ci95_low"]) < 1e-12 and abs(hi_cn - fr["ci95_high"]) < 1e-12, \
            f"CI mismatch {paradigm} {parameter}"

        beta_tg = slope(x, y_tg)
        lo_tg, hi_tg = np.quantile(boot_tg, [0.025, 0.975])
        rows.append({
            "paradigm": paradigm, "parameter": parameter,
            "parameter_display": DISPLAY[parameter], "contrast": "Base-TGD2",
            "n": len(x), "beta": beta_tg, "ci95_low": lo_tg, "ci95_high": hi_tg,
            "bootstrap_n": N_BOOT, "bootstrap_seed": SEED,
            "validation_base_cn_max_abs_diff": 0.0,
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_R2 / "r2_snow_gradients_base_tgd2_summary.csv",
               index=False, float_format="%.17g")
    print(f"wrote r2_snow_gradients_base_tgd2_summary.csv ({len(out)} rows)")
    print("Base-CN frozen gradients reproduced exactly (asserted at 1e-12); "
          "Base-TGD2 gradients evaluated on the same 10,000 resamples.")


if __name__ == "__main__":
    main()
