#!/usr/bin/env python3
"""R2 TGD2 structure-specificity: within-structure-adjusted parameter-space contrasts.

Computes Base–CN (frozen reference, recomputed only for validation), Base–TGD2, and
TGD2–CN within-structure-adjusted RMS contrasts for the 15 shared XAJ parameters under
both parameter-estimation regimes (IC: 10 restarts; dPL: 3 seeds).

Every definition is reused verbatim from the frozen R2 within-structure baseline
(run_r2_within_structure_baseline.py): same COMMON_XAJ, same bounds, same
within/between/excess construction, same 10,000-resample bootstrap with seed 20260730,
same basin-as-unit. No re-training, no model forward runs, no modification of frozen
Base–CN outputs. Canonical dPL medians are NOT used here (seed-level values only).

New estimand: delta_beta = beta(excess ~ frac_snow; Base–CN) - beta(excess ~ frac_snow;
Base–TGD2), estimated with a PAIRED basin bootstrap (same resample fits both slopes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
MANUSCRIPT = PROJECT / "manuscript"
DATA = PROJECT.parents[1] / "data"
RESULTS_R2 = MANUSCRIPT / "results" / "R2"

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse frozen R2 helpers to guarantee identical statistical definitions.
from run_r2_within_structure_baseline import (  # noqa: E402
    COMMON_XAJ, BASIN_FILE, BOUNDS_FILE, SNOW_FILE,
    assign_regime, rms_dist, bootstrap_stat, bootstrap_regression,
)

N_BOOT = 10000
SEED = 20260730

IC_RAW_DIRS = {
    "Base": PROJECT / "results" / "xaj_base_cmaes_531_batched_paired_v2" / "raw" / "xaj",
    "CN": PROJECT / "results" / "xaj_cn_cmaes_531_batched_paired_v2" / "raw" / "xaj_cn",
    "TGD2": PROJECT / "results" / "xaj_tgd2_cmaes_531_batched_v1" / "raw" / "xaj_tgd2",
}
# In r2_parameter_values_seed_level.csv the TGD2 structure is labelled "GD".
DPL_STRUCT_LABEL = {"Base": "Base", "CN": "CN", "TGD2": "GD"}

CONTRASTS = [("Base", "CN"), ("Base", "TGD2"), ("TGD2", "CN")]

STRATA = [
    ("Full531", lambda d: d),
    ("S1", lambda d: d[d["snow_regime"] == "S1"]),
    ("S2", lambda d: d[d["snow_regime"] == "S2"]),
    ("S3", lambda d: d[d["snow_regime"] == "S3"]),
    ("S4", lambda d: d[d["snow_regime"] == "S4"]),
    ("S5", lambda d: d[d["snow_regime"] == "S5"]),
    ("ExcludeS5", lambda d: d[d["snow_regime"] != "S5"]),
]

SUMMARY_METRICS = ["within_a", "within_b", "within_pooled", "between_all",
                   "excess", "ratio", "prop_excess_gt_0", "prop_between_gt_within"]


def load_ic_z(basins):
    """Return {basin: {structure: {start: (z_vec15, train_kge)}}} from raw CMA-ES JSON."""
    b_df = pd.read_csv(BOUNDS_FILE)
    bounds = b_df[b_df["active_model_key"] == "XAJ"].drop_duplicates("code_name").set_index("code_name")
    lowers = np.array([bounds.loc[n, "lower_bound"] for n in COMMON_XAJ], dtype=float)
    uppers = np.array([bounds.loc[n, "upper_bound"] for n in COMMON_XAJ], dtype=float)

    out = {b: {s: {} for s in IC_RAW_DIRS} for b in basins}
    for struct, raw_dir in IC_RAW_DIRS.items():
        for p in raw_dir.glob("*.json"):
            d = json.loads(p.read_text())
            b = str(d.get("basin_id", "")).zfill(8)
            if b not in out:
                continue
            s = int(d.get("start"))
            train_kge = float(d.get("train_metrics", {}).get("kge", np.nan))
            p_dict = dict(zip(d["parameter_names"], d["parameters"]))
            p_vals = np.array([p_dict[n] for n in COMMON_XAJ], dtype=float)
            z = (p_vals - lowers) / (uppers - lowers)
            out[b][struct][s] = (z, train_kge)
    return out


def load_dpl_z(basins):
    """Return {basin: {structure: {seed: z_vec15}}} from the frozen seed-level CSV."""
    df_seed = pd.read_csv(RESULTS_R2 / "r2_parameter_values_seed_level.csv")
    df_seed["basin_id"] = df_seed["basin_id"].astype(str).str.zfill(8)
    dpl = df_seed[df_seed["paradigm"] == "dPL"].copy()

    out = {b: {} for b in basins}
    for struct, label in DPL_STRUCT_LABEL.items():
        sub = dpl[dpl["structure"] == label]
        for b, bsub in sub.groupby("basin_id"):
            for seed, ssub in bsub.groupby("seed"):
                vec = ssub.set_index("parameter").loc[COMMON_XAJ]["z"].to_numpy(dtype=float)
                out[b].setdefault(struct, {})[str(seed)] = vec
    return out


def _vec(v):
    """IC stores (z_vec, train_kge); dPL stores plain z_vec. Return the z vector."""
    return v[0] if isinstance(v, tuple) else v


def canonical_d(a_dict, b_dict, a_keys, b_keys, paradigm):
    """Canonical distance following the frozen R2 rule per paradigm.

    IC: RMS between the best train-KGE restart of each structure (frozen definition).
    dPL: RMS between the elementwise-median seed vectors (frozen definition).
    """
    if paradigm == "IC":
        best_a = max(a_keys, key=lambda s: a_dict[s][1])
        best_b = max(b_keys, key=lambda s: b_dict[s][1])
        return rms_dist(_vec(a_dict[best_a]), _vec(b_dict[best_b]))
    za = np.median(np.array([_vec(a_dict[k]) for k in a_keys]), axis=0)
    zb = np.median(np.array([_vec(b_dict[k]) for k in b_keys]), axis=0)
    return rms_dist(za, zb)


def contrast_rows(z_map, basins, paradigm):
    rows = []
    for b in basins:
        for (a_name, b_name) in CONTRASTS:
            a_dict = z_map[b][a_name]
            b_dict = z_map[b][b_name]
            a_keys = sorted(a_dict.keys())
            b_keys = sorted(b_dict.keys())
            w_a = float(np.median([rms_dist(_vec(a_dict[s1]), _vec(a_dict[s2]))
                                   for s1, s2 in combinations(a_keys, 2)]))
            w_b = float(np.median([rms_dist(_vec(b_dict[s1]), _vec(b_dict[s2]))
                                   for s1, s2 in combinations(b_keys, 2)]))
            w_pooled = (w_a + w_b) / 2.0
            b_all = float(np.median([rms_dist(_vec(a_dict[s1]), _vec(b_dict[s2]))
                                     for s1 in a_keys for s2 in b_keys]))
            excess = b_all - w_pooled
            ratio = b_all / w_pooled if w_pooled > 1e-12 else np.nan
            matched = [rms_dist(_vec(a_dict[s]), _vec(b_dict[s])) for s in a_keys if s in b_dict]
            matched_median = float(np.median(matched)) if matched else np.nan
            canonical_d_val = canonical_d(a_dict, b_dict, a_keys, b_keys, paradigm)
            rows.append({
                "basin_id": b, "paradigm": paradigm,
                "contrast": f"{a_name}-{b_name}",
                "within_a": w_a, "within_b": w_b, "within_pooled": w_pooled,
                "between_all": b_all, "excess": excess, "ratio": ratio,
                "matched_d_rms": matched_median, "canonical_best_d_rms": canonical_d_val,
            })
    return rows


def paired_slope_difference_bootstrap(x, y_cn, y_tg, n_boot=N_BOOT, seed=SEED):
    """delta_beta = beta(excess~frac_snow; Base-CN) - beta(excess~frac_snow; Base-TGD2).

    One basin resample fits BOTH slopes; the difference is taken within each resample
    so the two slopes are paired by basin draw.
    """
    delta_point = float(np.polyfit(x, y_cn, 1)[0] - np.polyfit(x, y_tg, 1)[0])
    n = len(x)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    deltas = np.zeros(n_boot)
    for i in range(n_boot):
        ix = boot_idx[i]
        deltas[i] = np.polyfit(x[ix], y_cn[ix], 1)[0] - np.polyfit(x[ix], y_tg[ix], 1)[0]
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return delta_point, lo, hi


def main() -> None:
    RESULTS_R2.mkdir(parents=True, exist_ok=True)

    basins = [str(x).zfill(8) for x in json.loads(BASIN_FILE.read_text().strip())]
    assert len(basins) == 531, f"Expected 531 basins, found {len(basins)}"

    snow = pd.read_csv(SNOW_FILE)
    snow["basin_id"] = snow["basin_id"].astype(str).str.zfill(8)
    snow["snow_regime"] = snow["frac_snow"].apply(assign_regime)

    print("Loading IC raw JSON (Base / CN / TGD2)...")
    ic_z = load_ic_z(basins)
    for struct in IC_RAW_DIRS:
        bad = [b for b in basins if len(ic_z[b][struct]) != 10]
        assert not bad, f"IC {struct}: {len(bad)} basins without 10 restarts"

    print("Loading dPL seed-level z (Base / CN / TGD2)...")
    dpl_z = load_dpl_z(basins)
    for struct in DPL_STRUCT_LABEL:
        bad = [b for b in basins if len(dpl_z[b].get(struct, {})) != 3]
        assert not bad, f"dPL {struct}: {len(bad)} basins without 3 seeds"

    # ------------------------------------------------------------------ basin level
    ic_rows = contrast_rows(ic_z, basins, "IC")
    dpl_rows = contrast_rows(dpl_z, basins, "dPL")
    df_all = pd.DataFrame(ic_rows + dpl_rows).merge(snow, on="basin_id")

    # ------------------------------------------------------------------ validation 1
    # Recomputed Base-CN must reproduce the frozen basin-level CSV exactly.
    frozen = pd.read_csv(RESULTS_R2 / "r2_within_structure_basin_level.csv")
    frozen["basin_id"] = frozen["basin_id"].astype(str).str.zfill(8)
    mine = df_all[df_all["contrast"] == "Base-CN"][
        ["basin_id", "paradigm", "within_a", "within_b", "within_pooled",
         "between_all", "excess", "ratio"]].rename(columns={
             "within_a": "within_base", "within_b": "within_cn"})
    key = ["basin_id", "paradigm"]
    m = mine.merge(frozen[key + ["within_base", "within_cn", "within_pooled",
                                 "between_all", "excess", "ratio"]], on=key, suffixes=("", "_f"))
    for col in ["within_base", "within_cn", "within_pooled", "between_all", "excess"]:
        assert np.allclose(m[col], m[col + "_f"], rtol=0, atol=1e-12), \
            f"Base-CN basin-level mismatch on {col}"
    print(f"VALIDATION 1 OK: recomputed Base-CN basin-level matches frozen CSV "
          f"({len(m)} rows, 5 numeric cols atol=1e-12).")

    df_all.to_csv(RESULTS_R2 / "r2_tgd2_specificity_basin_level.csv",
                  index=False, float_format="%.17g")
    print("wrote r2_tgd2_specificity_basin_level.csv")

    # ------------------------------------------------------------------ summaries
    summary_rows = []
    for paradigm in ["IC", "dPL"]:
        for contrast in [f"{a}-{b}" for a, b in CONTRASTS]:
            cdf = df_all[(df_all["paradigm"] == paradigm) & (df_all["contrast"] == contrast)]
            for st_name, filt in STRATA:
                sub = filt(cdf)
                n_b = len(sub)
                for m in ["within_a", "within_b", "within_pooled", "between_all", "excess", "ratio"]:
                    vals = sub[m].dropna().to_numpy()
                    pt, lo, hi = bootstrap_stat(vals, np.median, n_boot=N_BOOT, seed=SEED)
                    summary_rows.append({
                        "paradigm": paradigm, "contrast": contrast, "stratum": st_name,
                        "n_basins": n_b, "metric": m, "median": pt,
                        "ci_lower": lo, "ci_upper": hi,
                    })
                pt, lo, hi = bootstrap_stat((sub["excess"] > 0).astype(float).to_numpy(),
                                            np.mean, n_boot=N_BOOT, seed=SEED)
                summary_rows.append({
                    "paradigm": paradigm, "contrast": contrast, "stratum": st_name,
                    "n_basins": n_b, "metric": "prop_excess_gt_0", "median": pt,
                    "ci_lower": lo, "ci_upper": hi,
                })
                pt, lo, hi = bootstrap_stat(
                    (sub["between_all"] > sub["within_pooled"]).astype(float).to_numpy(),
                    np.mean, n_boot=N_BOOT, seed=SEED)
                summary_rows.append({
                    "paradigm": paradigm, "contrast": contrast, "stratum": st_name,
                    "n_basins": n_b, "metric": "prop_between_gt_within", "median": pt,
                    "ci_lower": lo, "ci_upper": hi,
                })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(RESULTS_R2 / "r2_tgd2_specificity_summary.csv", index=False, float_format="%.17g")

    # ------------------------------------------------------------- regressions (full)
    reg_rows = []
    for paradigm in ["IC", "dPL"]:
        for contrast in [f"{a}-{b}" for a, b in CONTRASTS]:
            cdf = df_all[(df_all["paradigm"] == paradigm) & (df_all["contrast"] == contrast)]
            for st_name, filt in [("Full531", lambda d: d),
                                  ("ExcludeS5", lambda d: d[d["snow_regime"] != "S5"])]:
                sub = filt(cdf)
                x = sub["frac_snow"].to_numpy()
                for dep in ["within_pooled", "between_all", "excess"]:
                    y = sub[dep].to_numpy()
                    res = bootstrap_regression(x, y, n_boot=N_BOOT, seed=SEED)
                    reg_rows.append({
                        "paradigm": paradigm, "contrast": contrast, "stratum": st_name,
                        "dependent_var": dep, **res,
                    })

    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(RESULTS_R2 / "r2_tgd2_specificity_regressions.csv", index=False, float_format="%.17g")

    # -------------------------------------------------- validation 2: frozen regressions
    frozen_reg = pd.read_csv(RESULTS_R2 / "r2_within_structure_regressions.csv")
    mine_reg = df_reg[df_reg["contrast"] == "Base-CN"][
        ["paradigm", "stratum", "dependent_var", "slope", "slope_ci_lower",
         "slope_ci_upper", "spearman_rho"]]
    mr = mine_reg.merge(frozen_reg[["paradigm", "stratum", "dependent_var", "slope",
                                    "slope_ci_lower", "slope_ci_upper", "spearman_rho"]],
                        on=["paradigm", "stratum", "dependent_var"], suffixes=("", "_f"))
    for col in ["slope", "slope_ci_lower", "slope_ci_upper", "spearman_rho"]:
        assert np.allclose(mr[col], mr[col + "_f"], rtol=0, atol=1e-12), \
            f"Base-CN regression mismatch on {col}"
    print(f"VALIDATION 2 OK: recomputed Base-CN regressions match frozen CSV "
          f"({len(mr)} rows, atol=1e-12).")

    # ------------------------------------------- paired slope-difference (delta_beta)
    diff_rows = []
    for paradigm in ["IC", "dPL"]:
        for st_name, filt in [("Full531", lambda d: d),
                              ("ExcludeS5", lambda d: d[d["snow_regime"] != "S5"])]:
            cn = df_all[(df_all["paradigm"] == paradigm) & (df_all["contrast"] == "Base-CN")]
            tg = df_all[(df_all["paradigm"] == paradigm) & (df_all["contrast"] == "Base-TGD2")]
            cn = filt(cn).set_index("basin_id")
            tg = filt(tg).set_index("basin_id")
            common = cn.index.intersection(tg.index)
            assert len(common) == len(cn) == len(tg), "basin pairing mismatch"
            x = cn.loc[common, "frac_snow"].to_numpy()
            y_cn = cn.loc[common, "excess"].to_numpy()
            y_tg = tg.loc[common, "excess"].to_numpy()
            delta, lo, hi = paired_slope_difference_bootstrap(x, y_cn, y_tg)
            # also report the two individual slopes from the same paired pipeline
            diff_rows.append({
                "paradigm": paradigm, "stratum": st_name, "n_basins": len(common),
                "beta_base_cn": float(np.polyfit(x, y_cn, 1)[0]),
                "beta_base_tgd2": float(np.polyfit(x, y_tg, 1)[0]),
                "delta_beta": delta, "delta_beta_ci_lower": lo, "delta_beta_ci_upper": hi,
            })

    df_diff = pd.DataFrame(diff_rows)
    df_diff.to_csv(RESULTS_R2 / "r2_tgd2_slope_difference_summary.csv", index=False, float_format="%.17g")

    # ------------------------------------------------------------------- final checks
    assert df_all[["within_a", "within_b", "within_pooled", "between_all", "excess"]].isna().sum().sum() == 0
    for p in ["IC", "dPL"]:
        for reg, n in [("S1", 165), ("S2", 156), ("S3", 121), ("S4", 34), ("S5", 55)]:
            c = int(((df_all["paradigm"] == p) & (df_all["contrast"] == "Base-CN")
                     & (df_all["snow_regime"] == reg)).sum())
            assert c == n, f"{p} {reg} count {c} != {n}"
    print("VALIDATION 3 OK: no NaN in distance columns; S1-S5 counts 165/156/121/34/55 "
          "per paradigm; Exclude-S5 = 476.")
    print("All TGD2 specificity outputs written.")


if __name__ == "__main__":
    main()
