#!/usr/bin/env python3
"""Agent B: frozen model-place associations of G_seen and basin susceptibility."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    ALL_MODELS,
    ATTR_TYPES,
    CAMELS_35_ATTRIBUTES,
    RESULTS,
    bh_adjust,
    canonical_attributes,
    corr,
    load_inputs,
    write_csv,
)


def main() -> None:
    ids, paired, _, _, raw, _, _ = load_inputs()
    paired = paired.copy()
    paired["G_seen"] = paired.KGE_IC - paired.KGE_dPL
    attr = pd.DataFrame(raw, columns=CAMELS_35_ATTRIBUTES)
    attr["basin_id"] = [str(int(x)).zfill(8) for x in ids]
    basin = pd.read_csv(RESULTS / "agent_A/A03_BASIN_SHARED_MAPPING_SUSCEPTIBILITY.csv", dtype={"basin_id": str})
    basin["basin_id"] = basin.basin_id.map(lambda x: str(x).zfill(8))
    merged = basin.merge(attr, on="basin_id", validate="one_to_one")

    rows = []
    for model in ALL_MODELS:
        frame = paired[paired.model == model].merge(attr, on="basin_id", validate="one_to_one")
        for attribute in CAMELS_35_ATTRIBUTES:
            rho, p, n = corr(frame[attribute], frame.G_seen)
            rows.append({"model": model, "attribute": attribute, "attribute_type": ATTR_TYPES[attribute], "rho": rho, "p_value": p, "n": n, "label": "MODEL_ATTRIBUTE_G_SEEN_ASSOCIATION"})
    b01 = pd.DataFrame(rows)
    b01["q_value"] = bh_adjust(b01, "p_value")
    b01["fdr_family"] = "all 36 models × 35 attributes"
    write_csv(RESULTS / "agent_B/B01_MODEL_ATTRIBUTE_GAP_ASSOCIATION.csv", b01)

    rows = []
    for metric in ["median_G_seen", "fraction_models_G_positive"]:
        for attribute in CAMELS_35_ATTRIBUTES:
            rho, p, n = corr(merged[attribute], merged[metric])
            rows.append({"metric": metric, "attribute": attribute, "attribute_type": ATTR_TYPES[attribute], "rho": rho, "p_value": p, "n": n, "label": "BASIN_SUSCEPTIBILITY_ATTRIBUTE_ASSOCIATION"})
    b02 = pd.DataFrame(rows)
    b02["q_value"] = bh_adjust(b02, "p_value")
    b02["fdr_family"] = "basin susceptibility metrics × 35 attributes"
    write_csv(RESULTS / "agent_B/B02_BASIN_SUSCEPTIBILITY_ATTRIBUTE_ASSOCIATION.csv", b02)

    consistency = []
    for attribute in CAMELS_35_ATTRIBUTES:
        x = b01[b01.attribute == attribute]
        signs = np.sign(x.rho.dropna().to_numpy())
        significant = x[x.q_value < .05]
        same_sign_fraction = float(max(np.mean(signs > 0), np.mean(signs < 0))) if len(signs) else np.nan
        if len(significant) >= 18 and same_sign_fraction >= .75:
            classification = "cross_model_candidate"
        elif 1 <= len(significant) <= 5:
            classification = "model_specific_candidate"
        else:
            classification = "mixed_or_weak"
        consistency.append({
            "attribute": attribute,
            "attribute_type": ATTR_TYPES[attribute],
            "model_count": int(len(x)),
            "fdr_significant_model_count_q_lt_0_05": int(len(significant)),
            "positive_rho_model_count": int(np.sum(signs > 0)),
            "negative_rho_model_count": int(np.sum(signs < 0)),
            "same_sign_fraction": same_sign_fraction,
            "rho_median": float(x.rho.median()),
            "rho_q25": float(x.rho.quantile(.25)),
            "rho_q75": float(x.rho.quantile(.75)),
            "q_min": float(x.q_value.min()),
            "signal_class": classification,
            "classification_rule": "cross_model_candidate: >=18 FDR-significant models and same-sign fraction >=0.75; model_specific_candidate: 1-5 significant models; else mixed_or_weak",
            "label": "ATTRIBUTE_CROSS_MODEL_CONSISTENCY",
        })
    write_csv(RESULTS / "agent_B/B03_ATTRIBUTE_CROSS_MODEL_CONSISTENCY.csv", pd.DataFrame(consistency))
    print("Agent B complete", len(b01), len(b02), len(consistency))


if __name__ == "__main__":
    main()
