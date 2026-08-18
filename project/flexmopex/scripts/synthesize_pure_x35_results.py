#!/usr/bin/env python3
"""3-seed synthesis and head-to-head comparison:
Pure-Attribute Structure Encoder (x35) vs Frozen R19 Hybrid Structure Encoder.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SEEDS = [42, 43, 44]
R19_ROOT = Path("results/intercept_r19/E_S0_r19_unified_adadelta")
PURE_ROOT = Path("results/pure_x35_r19")
PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]


def _norm(d: dict) -> dict:
    return {
        "n_oracle_pos": d.get("n_oracle_pos", d.get("n_oracle_positive", 0)),
        "mean": d.get("mean", d.get("learned_w_mean", float("nan"))),
        "std": d.get("std", d.get("learned_w_std", float("nan"))),
        "min": d.get("min", d.get("learned_w_min", float("nan"))),
        "max": d.get("max", d.get("learned_w_max", float("nan"))),
        "pos_mean": d.get("pos_mean", d.get("learned_w_pos_mean", float("nan"))),
        "zero_mean": d.get("zero_mean", d.get("learned_w_zero_mean", float("nan"))),
        "Delta": d.get("Delta", d.get("group_separation_Delta", float("nan"))),
        "spearman": d.get("spearman", d.get("spearman_r_with_oracle", float("nan"))),
    }


def load_summaries(root: Path):
    res = {}
    for s in SEEDS:
        p = root / f"seed_{s}" / f"eval_summary_seed{s}.json"
        raw = json.loads(p.read_text())
        raw["processes"] = {k: _norm(v) for k, v in raw["processes"].items()}
        res[s] = raw
    return res


def main():
    r19 = load_summaries(R19_ROOT)
    pure = load_summaries(PURE_ROOT)

    print("=" * 80)
    print("FLEX-MOPEX VALIDATION: PURE-ATTRIBUTE (x35) vs HYBRID (R19) ENCODER")
    print("Epoch 10 Evaluation on Canonical 5114-Day Out-of-Sample Window (671 CAMELS Basins)")
    print("=" * 80)

    # 1. Predictive Performance Table
    print("\n" + "─" * 80)
    print("1. PREDICTIVE PERFORMANCE (STREAMFLOW ACCURACY)")
    print("─" * 80)
    print(f"{'Metric / Seed':<20} {'Hybrid (R19)':<25} {'Pure x35':<25} {'Difference':<10}")
    print("─" * 80)

    r19_meds = [r19[s]["median_nse"] for s in SEEDS]
    pure_meds = [pure[s]["median_nse"] for s in SEEDS]
    r19_means = [r19[s]["mean_nse"] for s in SEEDS]
    pure_means = [pure[s]["mean_nse"] for s in SEEDS]

    for s in SEEDS:
        print(f"Seed {s} Median NSE:    {r19[s]['median_nse']:<25.4f} {pure[s]['median_nse']:<25.4f} {pure[s]['median_nse']-r19[s]['median_nse']:+.4f}")
    print(f"{'3-Seed Mean Median:':<20} {np.mean(r19_meds):.4f} ± {np.std(r19_meds):.4f}           {np.mean(pure_meds):.4f} ± {np.std(pure_meds):.4f}           {np.mean(pure_meds)-np.mean(r19_meds):+.4f}")

    print()
    for s in SEEDS:
        print(f"Seed {s} Mean NSE:      {r19[s]['mean_nse']:<25.4f} {pure[s]['mean_nse']:<25.4f} {pure[s]['mean_nse']-r19[s]['mean_nse']:+.4f}")
    print(f"{'3-Seed Mean Mean:':<20} {np.mean(r19_means):.4f} ± {np.std(r19_means):.4f}           {np.mean(pure_means):.4f} ± {np.std(pure_means):.4f}           {np.mean(pure_means)-np.mean(r19_means):+.4f}")

    # 2. Four-Process Structural Separation & Correlation Table
    print("\n" + "─" * 80)
    print("2. FOUR-PROCESS STRUCTURAL SEPARATION (Delta = pos_mean - zero_mean)")
    print("─" * 80)
    print(f"{'Process':<10} {'Seed':<6} {'Hybrid Delta':<14} {'Pure Delta':<14} {'Hybrid rho':<14} {'Pure rho':<14} {'Delta diff':<10}")
    print("─" * 80)

    for proc in PROCESSES:
        h_d = [r19[s]["processes"][proc]["Delta"] for s in SEEDS]
        p_d = [pure[s]["processes"][proc]["Delta"] for s in SEEDS]
        h_r = [r19[s]["processes"][proc]["spearman"] for s in SEEDS]
        p_r = [pure[s]["processes"][proc]["spearman"] for s in SEEDS]
        for s in SEEDS:
            d_h = r19[s]["processes"][proc]["Delta"]
            d_p = pure[s]["processes"][proc]["Delta"]
            r_h = r19[s]["processes"][proc]["spearman"]
            r_p = pure[s]["processes"][proc]["spearman"]
            print(f"{proc:<10} {s:<6} {d_h:>+14.4f} {d_p:>+14.4f} {r_h:>+14.4f} {r_p:>+14.4f} {d_p-d_h:>+10.4f}")
        print(f"{proc+' (μ±σ)':<16} {np.mean(h_d):+.4f}±{np.std(h_d):.4f}   {np.mean(p_d):+.4f}±{np.std(p_d):.4f}   {np.mean(h_r):+.4f}±{np.std(h_r):.4f}   {np.mean(p_r):+.4f}±{np.std(p_r):.4f}   {np.mean(p_d)-np.mean(h_d):+.4f}")
        print()

    # 3. Interception Gate Deep-Dive
    print("─" * 80)
    print("3. CANOPY INTERCEPTION (w_int) DEEP-DIVE")
    print("─" * 80)
    print(f"{'Seed':<6} {'Hybrid std':<12} {'Pure std':<12} {'Hybrid range':<20} {'Pure range':<20}")
    for s in SEEDS:
        h_st = r19[s]["processes"]["w_int"]["std"]
        p_st = pure[s]["processes"]["w_int"]["std"]
        h_rg = f"[{r19[s]['processes']['w_int']['min']:.3f}, {r19[s]['processes']['w_int']['max']:.3f}]"
        p_rg = f"[{pure[s]['processes']['w_int']['min']:.3f}, {pure[s]['processes']['w_int']['max']:.3f}]"
        print(f"{s:<6} {h_st:<12.4f} {p_st:<12.4f} {h_rg:<20} {p_rg:<20}")
    h_st_all = [r19[s]["processes"]["w_int"]["std"] for s in SEEDS]
    p_st_all = [pure[s]["processes"]["w_int"]["std"] for s in SEEDS]
    print(f"{'Mean':<6} {np.mean(h_st_all):<12.4f} {np.mean(p_st_all):<12.4f}")

    # 4. Invariant & Architecture Summary
    print("\n" + "─" * 80)
    print("4. ARCHITECTURE & COMPLEXITY SUMMARY")
    print("─" * 80)
    print(f"  Hybrid Encoder Input:  [x35_norm, stopgrad(h128)] (163-D) -> 128 -> 64 -> 8 (29,768 struct params, 76,946 total)")
    print(f"  Pure x35 Encoder Input: x35_norm (35-D)            -> 128 -> 64 -> 8 (13,384 struct params, 60,562 total)")
    print(f"  Parameter Savings:     16,384 parameter weights removed (-21.3% total NN model size)")
    print(f"  Interface Semantics:   Identical [B, 8] logits, identical gate ordering, identical physics")

    # 5. Formal Criteria Evaluation
    print("\n" + "─" * 80)
    print("5. FORMAL RETENTION CRITERIA EVALUATION")
    print("─" * 80)
    crit1 = np.mean(pure_meds) >= np.mean(r19_meds) - 0.002
    crit2 = all(pure[s]["processes"][p]["Delta"] > 0 for s in SEEDS for p in PROCESSES)
    crit3 = all(pure[s]["processes"]["w_int"]["std"] > 0.10 for s in SEEDS)
    crit4 = np.mean([pure[s]["processes"]["w_int"]["Delta"] for s in SEEDS]) > 0.12
    crit5 = np.mean([pure[s]["processes"]["w_int"]["spearman"] for s in SEEDS]) > 0.30

    print(f"  1. Predictive Performance Preservation (Median NSE ≥ R19 - 0.002): {crit1} ({np.mean(pure_meds):.4f} vs {np.mean(r19_meds):.4f}, Δ={np.mean(pure_meds)-np.mean(r19_meds):+.4f})")
    print(f"  2. 100% Sign-Consistent Positive Separation Across All Seeds/Gates: {crit2}")
    print(f"  3. Interception Gate Variance Preserved (std > 0.10, no collapse): {crit3} (mean std = {np.mean(p_st_all):.4f})")
    print(f"  4. Interception Separation Delta > +0.12:                          {crit4} (mean Delta = {np.mean([pure[s]['processes']['w_int']['Delta'] for s in SEEDS]):+.4f} vs R19 {np.mean([r19[s]['processes']['w_int']['Delta'] for s in SEEDS]):+.4f})")
    print(f"  5. Interception Rank Correlation rho > +0.30:                      {crit5} (mean rho = {np.mean([pure[s]['processes']['w_int']['spearman'] for s in SEEDS]):+.4f} vs R19 {np.mean([r19[s]['processes']['w_int']['spearman'] for s in SEEDS]):+.4f})")

    decision = "ADOPT_PURE_X35" if (crit1 and crit2 and crit3 and crit4 and crit5) else "KEEP_HYBRID_R19"
    print("\n" + "=" * 80)
    print(f"FINAL DECISION: {decision}")
    print("=" * 80)


if __name__ == "__main__":
    main()
