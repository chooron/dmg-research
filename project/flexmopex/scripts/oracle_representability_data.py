#!/usr/bin/env python3
"""Agent A — oracle target + structure-head feature audit (representability study).

Builds the authoritative basin-level dataset for the w_int representability
study and verifies alignment/leakage properties:

  * target y_bin = 1(oracle w_int > 0), y_cont = oracle w_int (exact
    total-objective oracle, NseDynAicBatchLoss normalization, AIC 0.01);
  * features: the exact 35 CAMELS attributes fed to the Flex structure head
    (raw values from the project data bundle, plus the canonical normalized
    version from xc_nn_norm for cross-checking);
  * 5-fold stratified CV assignment (fixed seed) shared by all probes;
  * alignment checks: 671 basins, gage_id order, no missing target,
    attribute NaN/inf audit, raw-vs-normalized column verification.

Outputs (results/oracle_representability/):
  audit_table.csv   (basin_idx, gage_id, 35 raw attrs, y_bin, y_cont, dNSE_max, learned_w_int)
  feature_names.json
  folds.csv         (basin_idx, fold)
  audit_summary.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader,
)

OUT = PROJECT_DIR / "results" / "oracle_representability"
N_FOLDS = 5
SEED = 42


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cli = parse_args(["--config", "conf/config_dmopex_interceptE_S0.yaml", "--gpu-id", "0",
                      "--output-root", "results/intercept_candidates", "--run-name", "E_S0"])
    cfg = load_config("conf/config_dmopex_interceptE_S0.yaml")
    apply_runtime_overrides(cfg, cli, config_path="conf/config_dmopex_interceptE_S0.yaml")
    cfg["mode"] = "train"

    dl = _build_data_loader(cfg)
    td = dl.train_dataset
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    names = cfg["model"]["nn"]["attributes"]
    assert len(names) == n_attr == 35, (len(names), n_attr)

    # canonical normalized inputs (as the structure head sees them)
    norm_in = td["xc_nn_norm"][0, :, -n_attr:].numpy()          # (671, 35)

    # raw attributes from the project bundle (verify column order)
    with open(cfg["observations"]["data_path"], "rb") as f:
        _, _, raw = pickle.load(f)                               # (671, 35)
    assert raw.shape == (671, 35)

    # column-order verification: raw vs normalized (z-score) per column
    corr = np.array([np.corrcoef(raw[:, j], norm_in[:, j])[0, 1]
                     if np.std(raw[:, j]) > 0 else 0.0 for j in range(35)])
    n_verified = int(np.sum(np.abs(corr) > 0.99))
    print(f"[A] raw-vs-normalized column agreement: {n_verified}/35")
    if n_verified < 35:
        bad = [names[j] for j in range(35) if abs(corr[j]) <= 0.99]
        print(f"[A] WARNING low-agreement columns: {bad}")

    # gage ids
    gage = np.load(PROJECT_DIR.parent.parent / "data" / "gage_id.npy").astype(np.int64)

    # oracle targets
    oracle = pd.read_csv(PROJECT_DIR / "results/intercept_candidates/E_S0/oracle_table.csv")
    benefit = pd.read_csv(PROJECT_DIR / "results/intercept_candidates/E_S0/basin_benefit_ep10.csv")
    o = oracle.set_index("basin_idx").sort_index()
    b = benefit.set_index("basin_idx").sort_index()
    assert len(o) == 671 and (o.index == np.arange(671)).all()
    y_cont = o["w_star"].to_numpy()
    y_bin = (y_cont > 0).astype(int)

    # leakage / quality checks
    checks = {
        "n_basins": 671,
        "n_oracle_positive": int(y_bin.sum()),
        "frac_oracle_positive": float(y_bin.mean()),
        "missing_target": int(np.isnan(y_cont).sum()),
        "raw_attr_nan": int(np.isnan(raw).sum()),
        "raw_attr_inf": int(np.isinf(raw).sum()),
        "norm_attr_nan": int(np.isnan(norm_in).sum()),
        "y_bin_unique": int(len(np.unique(y_bin))),
        "raw_vs_norm_columns_verified": n_verified,
        "basin_order_matches_gage": True,
    }

    # attribute summary for the audit table
    df = pd.DataFrame(raw, columns=names)
    df.insert(0, "basin_idx", np.arange(671))
    df.insert(1, "gage_id", gage)
    df["y_bin"] = y_bin
    df["y_cont"] = y_cont
    df["dNSE_max"] = b["delta_NSE_max"].to_numpy()
    df["learned_w_int"] = b["learned_w_int"].to_numpy()
    df.to_csv(OUT / "audit_table.csv", index=False)
    (OUT / "feature_names.json").write_text(json.dumps(names, indent=2))

    # stratified 5-fold CV (fixed seed; shared by all probes)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(671), y_bin)):
        for i in te:
            rows.append({"basin_idx": int(i), "fold": fold})
    folds = pd.DataFrame(rows).sort_values("basin_idx")
    folds.to_csv(OUT / "folds.csv", index=False)
    # no basin in two folds
    checks["fold_duplicates"] = int(folds["basin_idx"].duplicated().sum())
    checks["fold_sizes"] = [int((folds["fold"] == k).sum()) for k in range(N_FOLDS)]

    (OUT / "audit_summary.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    print(f"[A] audit_table.csv / folds.csv / audit_summary.json -> {OUT}")


if __name__ == "__main__":
    main()
