#!/usr/bin/env python3
"""Agent C — hydrologic organization of the interception-beneficial basin subset.

Joins Agent A's basin-level benefit table (results/intercept_candidates/E_S0/
basin_benefit.csv, epoch 10 primary) with raw CAMELS attributes from the
project data bundle, and characterizes the beneficial subset:

  * Spearman(benefit, attribute) for the prioritized attribute set
    (forest fraction, LAI max/diff, aridity, precipitation seasonality,
    mean P/PET, snow fraction as control);
  * top-decile-benefit vs remaining basins (median attribute + Mann-Whitney U);
  * above-threshold (dNSE > 0.01) vs remaining basins;
  * statement on hydrologic organization.

Attribute columns: the raw bundle attributes array is joined to gage_id.npy
basin order; column order is verified against the config all_attributes list by
correlating with the loader-normalized columns (xc_nn_norm tail), falling back
to the normalized columns if the raw mapping cannot be verified.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader,
)

PRIORITY_ATTRS = [
    "frac_forest", "lai_max", "lai_diff", "aridity", "p_seasonality",
    "p_mean", "pet_mean", "frac_snow", "gvf_max", "gvf_diff",
]


def load_raw_attributes(cfg: dict, td) -> tuple[np.ndarray, list[str]]:
    """Raw (671, 35) attributes; verify column order against config names."""
    names = cfg["model"]["nn"]["attributes"]
    bundle = Path(cfg["observations"]["data_path"])
    if not bundle.is_absolute():
        bundle = PROJECT_DIR.parent.parent / bundle
    with bundle.open("rb") as f:
        _, _, attrs_raw = pickle.load(f)          # (671, 35)
    # verify: raw column j vs normalized column j (loader z-scores per column)
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    norm = td["xc_nn_norm"][0, :, -n_attr:].cpu().numpy()
    corr = np.array([np.corrcoef(attrs_raw[:, j], norm[:, j])[0, 1]
                     if np.std(attrs_raw[:, j]) > 0 else 0.0 for j in range(len(names))])
    verified = int(np.sum(np.abs(corr) > 0.99))
    print(f"[attrs] raw-vs-normalized column agreement: {verified}/{len(names)}")
    if verified < len(names):
        print(f"[attrs] WARNING: raw column order NOT verified; using normalized columns")
        return norm, names
    return attrs_raw, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0.yaml")
    ap.add_argument("--output-root", default="results/intercept_candidates")
    ap.add_argument("--run-name", default="E_S0")
    ap.add_argument("--epoch", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.output_root)
    arm_dir = root / args.run_name

    cli = parse_args(["--config", args.config, "--gpu-id", "0",
                      "--output-root", str(args.output_root), "--run-name", args.run_name])
    cfg = load_config(args.config)
    apply_runtime_overrides(cfg, cli, config_path=args.config)
    cfg["mode"] = "train"
    dl = _build_data_loader(cfg)
    td = dl.train_dataset

    attrs_raw, names = load_raw_attributes(cfg, td)
    attr_df = pd.DataFrame(attrs_raw, columns=names)

    benefit = pd.read_csv(arm_dir / "basin_benefit.csv")
    b = benefit[benefit["epoch"] == args.epoch].set_index("basin_idx").sort_index()
    if len(b) == 0:
        b = benefit[benefit["epoch"] == max(benefit["epoch"])].set_index("basin_idx").sort_index()
        print(f"[attrs] epoch {args.epoch} absent; using epoch {max(benefit['epoch'])}")
    df = attr_df.join(b, how="inner")
    n = len(df)
    dn = df["delta_NSE_max"].to_numpy()

    rows = []
    for a in PRIORITY_ATTRS:
        if a not in df.columns:
            continue
        v = df[a].to_numpy()
        mask = ~(np.isnan(dn) | np.isnan(v))
        rho, p = spearmanr(dn[mask], v[mask])
        top = dn >= np.nanquantile(dn, 0.9)
        rest = ~top
        if np.sum(top) > 5 and np.sum(rest) > 5:
            u, pu = mannwhitneyu(v[top & mask], v[rest & mask], alternative="two-sided")
        else:
            u, pu = np.nan, np.nan
        thr = dn > 0.01
        rows.append({
            "attr": a,
            "spearman_dNSE": float(rho), "p": float(p),
            "median_all": float(np.nanmedian(v)),
            "median_top10": float(np.nanmedian(v[top])),
            "median_rest": float(np.nanmedian(v[rest])),
            "median_gt001": float(np.nanmedian(v[thr])),
            "n_gt001": int(np.sum(thr)),
            "mannwhitney_p_top_vs_rest": float(pu),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(arm_dir / "basin_attributes_summary.csv", index=False)

    # profile of the beneficial subgroup
    sub = df[dn > 0.01]
    prof = {
        "n_total": n,
        "n_dNSE_gt001": int(np.sum(dn > 0.01)),
        "frac_dNSE_gt001": float(np.mean(dn > 0.01)),
        "n_top10": int(np.sum(dn >= np.nanquantile(dn, 0.9))),
        "median_dNSE": float(np.nanmedian(dn)),
        "median_learned_w_int": float(df["learned_w_int"].median()),
        "median_learned_w_int_gt001": float(sub["learned_w_int"].median()) if len(sub) else None,
        "frac_learned_w_int_lt001_in_gt001": float(np.mean(sub["learned_w_int"] < 0.01)) if len(sub) else None,
        "top10_median_attrs": {a: float(np.nanmedian(df.loc[df["delta_NSE_max"] >= np.nanquantile(dn, 0.9), a]))
                               for a in PRIORITY_ATTRS if a in df.columns},
    }
    (arm_dir / "basin_attributes_profile.json").write_text(json.dumps(prof, indent=2))
    print(json.dumps(prof, indent=2))
    print(f"[attrs] summary -> {arm_dir / 'basin_attributes_summary.csv'}")


if __name__ == "__main__":
    main()
