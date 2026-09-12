#!/usr/bin/env python3
"""Audit IC--dPL sign agreement with IC-anchored denominators.

All-cell, nontrivial, both-nontrivial, and IC-stable/nontrivial sign agreement are
reported separately.  The IC-stable denominator is defined without using dPL to
select cells, preventing near-zero cells from manufacturing agreement.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from r3_common import IC_STABLE_SIGN_THRESHOLD, PRIMARY_EFFECT_THRESHOLD, MODEL_ORDER, R2, R3, R3_TABLES, STRICT_MODELS, write_audit, write_csv, write_json

SPACES = ("raw_attribute", "information_cluster")


def rate_ci(agree: np.ndarray, seed: int, b: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed); values = []
    for _ in range(b):
        values.append(float(agree[rng.integers(0, len(agree), len(agree))].mean()))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.n_bootstrap < 1000:
        raise ValueError("formal sign CI requires >=1000 resamples")
    started = time.time(); long = pd.read_csv(R3 / "tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
    rows = []
    for space in SPACES:
        sub = long[long.space == space].copy()
        for population, models in (("all36", MODEL_ORDER), ("exclude_simhyd", STRICT_MODELS)):
            current = sub[sub.model.isin(models)]
            for definition in ("A_sign_all", "A_sign_nontrivial", "A_sign_both_nontrivial", "A_sign_IC_stable"):
                ic = current[current.method == "IC"].set_index(["model", "parameter_index", "feature_index"])
                dpl = current[current.method == "dPL"].set_index(["model", "parameter_index", "feature_index"])
                paired = ic[["rho", "bootstrap_sign_probability"]].rename(columns={"rho": "rho_ic", "bootstrap_sign_probability": "ic_bootstrap_sign_probability"}).join(dpl[["rho", "bootstrap_sign_probability"]].rename(columns={"rho": "rho_dpl", "bootstrap_sign_probability": "dpl_bootstrap_sign_probability"}), how="inner")
                if definition == "A_sign_all":
                    mask = np.isfinite(paired.rho_ic) & np.isfinite(paired.rho_dpl)
                elif definition == "A_sign_nontrivial":
                    mask = paired.rho_ic.abs().ge(.10) | paired.rho_dpl.abs().ge(.10)
                elif definition == "A_sign_both_nontrivial":
                    mask = paired.rho_ic.abs().ge(.10) & paired.rho_dpl.abs().ge(.10)
                else:
                    mask = paired.rho_ic.abs().ge(PRIMARY_EFFECT_THRESHOLD) & paired.ic_bootstrap_sign_probability.ge(IC_STABLE_SIGN_THRESHOLD)
                chosen = paired[mask]
                agreement = (np.sign(chosen.rho_ic.to_numpy(float)) == np.sign(chosen.rho_dpl.to_numpy(float)))
                if len(chosen) == 0:
                    rate = low = high = np.nan
                else:
                    rate = float(agreement.mean()); low, high = rate_ci(agreement, 20260902 + len(rows), args.n_bootstrap)
                rows.append({"space": space, "population": population, "definition": definition, "denominator": int(len(chosen)), "agreement_count": int(agreement.sum()), "agreement_rate": rate, "bootstrap_ci_low": low, "bootstrap_ci_high": high, "bootstrap_replicates": args.n_bootstrap, "ic_anchor": definition == "A_sign_IC_stable", "nontrivial_threshold": .10, "ic_stable_effect_threshold": PRIMARY_EFFECT_THRESHOLD, "ic_stable_sign_probability_threshold": IC_STABLE_SIGN_THRESHOLD})
    output = pd.DataFrame(rows); write_csv(R3_TABLES / "R3_SIGN_AGREEMENT_AUDIT.csv", output)
    write_json(R3 / "cache/R3_SIGN_MANIFEST.json", {"analysis": "R3-C sign agreement", "definitions": ["A_sign_all", "A_sign_nontrivial", "A_sign_both_nontrivial", "A_sign_IC_stable"], "primary_anchor": "IC-stable/nontrivial cells", "n_bootstrap": args.n_bootstrap, "seed": 20260902, "runtime_seconds": time.time() - started})
    hist = output[(output.space == "raw_attribute") & (output.population == "all36") & (output.definition == "A_sign_all")].agreement_rate.iloc[0]
    stable = output[(output.space == "raw_attribute") & (output.population == "all36") & (output.definition == "A_sign_IC_stable")].agreement_rate.iloc[0]
    result = f"Raw all-cell all36 sign agreement was {hist:.6f}; IC-anchored stable/nontrivial agreement was {stable:.6f}. All definitions include denominators, counts, and 1000-resample bootstrap CIs for both raw and information-cluster spaces and both model populations."
    write_audit(R3 / "R3_SIGN_AGREEMENT_AUDIT.md", "R3-C Sign Agreement Audit", "Does IC--dPL direction persist after excluding trivial near-zero cells and anchoring eligibility to independently stable IC relationships?", f"Paired relationship matrices `{R3 / 'tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv'}`; IC and dPL canonical normalized-u associations on 531 basins; raw and frozen information-cluster spaces.", "A_sign_all includes all finite paired cells. A_sign_nontrivial includes cells where either estimator has |rho|>=0.10. A_sign_both_nontrivial requires both. A_sign_IC_stable requires IC |rho|>=0.20 and IC bootstrap sign probability>=0.95, then tests dPL sign; dPL is not used to define the denominator.", "Denominator is paired model×parameter×feature cells; all36=36 models and exclude_simhyd=35. Agreement means equal sign, including zero only in A_sign_all. Binomial-style cell bootstrap uses fixed seed and 1000 replicates.", "All four definitions are written for raw and cluster spaces. The historical approximately 74.6% value is the raw all-cell all36 rate under the old finite-cell denominator; the IC-stable/nontrivial rate is the relevant reproducibility evidence and is reported separately.", result, "Rates can change with |rho| threshold and IC stability threshold; those thresholds are fixed and the full denominator is exposed. Cluster agreement is not allowed to overwrite raw agreement.", "Sign equality is not magnitude agreement, and a dPL sign on an IC-weak cell is construction-linked. A reviewer can still attribute stable sign patterns to correlated attributes or shared data; the dPL construction-artifact and permutation audits remain necessary.", "R3_C_READY", time.time() - started)


if __name__ == "__main__":
    main()
