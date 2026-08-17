#!/usr/bin/env python3
"""Agent C — KGE-conditioned interception diagnosis (offline, no loss change).

Uses the TRAINING-window per-basin KGE (train1980-1995_Ep10 metrics.json —
legitimate training-time information; test KGE is NOT used to motivate any
training-time rule).  Tests whether oracle-positive interception basins are
systematically concentrated in a predictive-adequacy regime, and whether the
same holds for the other three processes.

Predeclared KGE bins: quintiles (20/40/60/80) -> 5 bins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scipy.stats import spearmanr  # noqa: E402

ARM = PROJECT_DIR / "results/intercept_candidates/E_S0"
PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
BINS = [0.20, 0.40, 0.60, 0.80]           # predeclared quintile cuts


def main() -> None:
    # training-window KGE (per-basin)
    raw = (ARM / "train1980-1995_Ep10/metrics.json").read_text()
    canon = json.loads(raw)
    if isinstance(canon, str):
        canon = json.loads(canon)
    kge_tr = np.asarray(canon["kge"], dtype=float)
    nse_tr = np.asarray(canon.get("nse", np.full(len(kge_tr), np.nan)), dtype=float)
    assert len(kge_tr) == 671
    print(f"[C] train-window KGE: median {np.nanmedian(kge_tr):.3f}, "
          f"NaN {int(np.isnan(kge_tr).sum())}")

    oracle = pd.read_csv(ARM / "four_process/process_oracle_table.csv")
    oracle = oracle[oracle["epoch"] == 10]

    res = {}
    for proc in PROCESSES:
        sub = oracle[oracle["process"] == proc].set_index("basin_idx").sort_index()
        y_act = (sub["w_star"] > 0).to_numpy().astype(float)
        y_w = sub["w_star"].to_numpy()
        dn = sub["dNSE_max"].to_numpy()
        fi = sub["fit_improvement"].to_numpy()
        valid = ~np.isnan(kge_tr)
        assoc = {
            "kge_vs_activation_spearman": float(spearmanr(kge_tr[valid], y_act[valid]).statistic),
            "kge_vs_wstar_spearman": float(spearmanr(kge_tr[valid], y_w[valid]).statistic),
            "kge_vs_dNSE_spearman": float(spearmanr(kge_tr[valid], dn[valid]).statistic),
            "kge_vs_fitbenefit_spearman": float(spearmanr(kge_tr[valid], fi[valid]).statistic),
        }
        # predeclared quintile bins
        cuts = np.nanquantile(kge_tr, BINS)
        bins = np.concatenate([[-np.inf], cuts, [np.inf]])
        labels = [f"q{i}" for i in range(5)]
        bin_idx = np.digitize(kge_tr, cuts) - 1
        cond = {}
        for i in range(5):
            m = (bin_idx == i) & valid
            if m.sum() > 0:
                cond[labels[i]] = {
                    "n": int(m.sum()),
                    "kge_med": float(np.nanmedian(kge_tr[m])),
                    "P_oracle_pos": float(np.mean(y_act[m])),
                }
        assoc["P_oracle_pos_by_KGE_bin"] = cond
        assoc["median_kge_oracle_pos"] = float(np.nanmedian(kge_tr[y_act > 0])) if (y_act > 0).sum() else float("nan")
        assoc["median_kge_oracle_zero"] = float(np.nanmedian(kge_tr[y_act == 0])) if (y_act == 0).sum() else float("nan")
        res[proc] = assoc
        print(f"[C] {proc}: spearman(kge, activation)={assoc['kge_vs_activation_spearman']:+.3f} "
              f"| median kge pos {assoc['median_kge_oracle_pos']:.3f} vs zero {assoc['median_kge_oracle_zero']:.3f}")
        print(f"     P(pos|bin): " + "  ".join(
            f"{k}={v['P_oracle_pos']:.2f}" for k, v in assoc["P_oracle_pos_by_KGE_bin"].items()))

    # verdict
    int_assoc = res["w_int"]["kge_vs_activation_spearman"]
    others = [res[p]["kge_vs_activation_spearman"] for p in PROCESSES if p != "w_int"]
    res["_verdict"] = {
        "w_int_kge_assoc": int_assoc,
        "other_process_kge_assoc": others,
        "note": ("SUPPORTED" if abs(int_assoc) > 0.25 and abs(int_assoc) > max(abs(o) for o in others) + 0.1
                 else "MIXED" if abs(int_assoc) > 0.15
                 else "NOT SUPPORTED"),
    }
    (ARM / "four_process/kge_conditioning.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[C] verdict: {res['_verdict']['note']}")
    print(f"[C] -> {ARM}/four_process/kge_conditioning.json")


if __name__ == "__main__":
    main()
